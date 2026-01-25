"""
Script để train VideoMAE model 10 cho action recognition.
Model 10: Sequential Layer Training Strategy
- Phase 1: Freeze backbone, train head (giống Model 9)
- Phase 2: Sequential layer training - 8 phần, mỗi phần 1 layer train trên subset labels
- Phase 3: Freeze all, train head một lần nữa trên tất cả labels
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
import logging
import argparse

# Fix OpenMP duplicate library error on Windows
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# GPU optimization settings
cuda_launch_blocking = os.environ.get('CUDA_LAUNCH_BLOCKING', '0')
os.environ['CUDA_LAUNCH_BLOCKING'] = cuda_launch_blocking
torch.backends.cudnn.benchmark = True

from sota_training.config import get_default_config
from sota_training.utils import setup_logging, log_system_info, load_checkpoint, save_checkpoint, plot_training_history
from sota_training.dataset import VideoDataset, TestDataset
from sota_training.models import create_model
from sota_training.training import train_model, get_lr_scheduler, train_one_epoch, evaluate
from sota_training.adaptive_lr_scheduler import get_adaptive_lr_scheduler
from sota_training.sequential_layer_training import (
    sequential_layer_training_loop,
    final_head_training,
    freeze_all_backbone_layers
)
from sota_training.inference import run_inference, generate_submission

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_videomae_model10_config():
    """
    Tạo config cho VideoMAE Model 10: Sequential Layer Training Strategy.
    
    Key differences from Model 9:
    - Sequential layer training: 8 phases, mỗi phase train 1 layer trên subset labels
    - Final head training: Freeze all, train head một lần nữa
    """
    default_config = get_default_config()
    config = default_config.copy()
    
    # Architecture
    config['architecture'] = 'videomae'
    config['pretrained_name'] = None
    config['model_name'] = 'MCG-NJU/videomae-large-finetuned-kinetics'
    config['pretrained_ckpt'] = None
    
    # VideoMAE parameters
    config['tubelet_size'] = 2
    config['image_size'] = 224
    config['patch_size'] = 16
    config['num_frames'] = 16
    config['frame_stride'] = 2
    config['img_size'] = 224
    
    # Initialization parameters - Giống Model 9
    config['init_output_gain'] = 2.0   # → std=0.02 (best practice)
    config['init_hidden_gain'] = 1.0   # → std=0.01 (conservative)
    config['use_normal_init'] = True
    
    # Training parameters
    config['batch_size'] = 8
    config['grad_accum_steps'] = 6
    config['epochs'] = 150  # Total epochs (Phase 1)
    config['warmup_epochs'] = 10
    config['cosine_start_epoch'] = 10
    
    # Learning rates - Phase 1: Freeze backbone, train head
    config['base_lr'] = 0.0  # Backbone frozen trong Phase 1
    config['head_lr'] = 1e-3
    
    # Learning rates - Phase 2: Sequential layer training
    config['base_lr_unfreeze'] = 2e-6  # LR cho từng layer khi unfreeze
    config['head_lr_unfreeze'] = 2e-4  # LR cho head trong Phase 2
    
    # Learning rates - Phase 3: Final head training
    config['final_head_lr'] = 1e-4  # LR cho head trong Phase 3
    
    # Freeze strategy - Phase 1
    config['freeze_backbone_epochs'] = 15  # Phase 1: 15 epochs freeze backbone
    
    # Sequential layer training - Phase 2
    config['sequential_layer_training'] = True
    config['num_layer_phases'] = 8  # 8 phases, mỗi phase 1 layer
    config['layer_phase_patience'] = 10  # Patience cho mỗi layer phase
    config['layer_phase_max_epochs'] = 30  # Max epochs mỗi layer phase
    
    # Final head training - Phase 3
    config['final_head_epochs'] = 50  # Epochs cho Phase 3
    
    # Regularization - Phase 1
    config['weight_decay'] = 0.08
    config['dropout'] = 0.2
    config['drop_path_rate'] = 0.0
    config['label_smoothing'] = 0.1
    
    # Regularization - Phase 2 & 3
    config['weight_decay_unfreeze'] = 0.05
    config['dropout_unfreeze'] = 0.15
    config['drop_path_rate_unfreeze'] = 0.1
    config['label_smoothing_unfreeze'] = 0.08
    
    # Augmentation
    config['mixup_alpha'] = 0.4
    config['mixup_alpha_unfreeze'] = 0.3
    config['cutmix_alpha_unfreeze'] = 1.0
    config['use_cutmix_unfreeze'] = True
    config['cutmix_alpha'] = 1.0
    config['use_cutmix'] = True
    config['use_temporal_aug'] = True
    config['use_advanced_spatial'] = True
    config['use_advanced_color'] = True
    
    # Loss
    config['use_focal_loss'] = False
    
    # Features
    config['use_ema'] = False
    config['ema_decay'] = 0.9999
    config['use_adapters'] = False
    config['use_progressive_resize'] = False
    config['use_synthetic_data'] = False
    config['use_distillation'] = False
    config['use_clean_train_loss'] = False  # Tắt để tiết kiệm thời gian
    
    # Adaptive LR - Phase 1
    config['use_adaptive_lr'] = True
    config['lr_plateau_patience'] = 8
    config['lr_min_delta'] = 0.005
    config['lr_threshold_mode'] = 'rel'
    config['lr_cooldown'] = 3
    config['use_val_loss_for_lr'] = False
    config['min_lr_ratio'] = 0.005
    
    # Early stopping - Phase 1
    config['early_stop_patience'] = 20
    
    # Data
    config['val_ratio'] = 0.20
    config['num_workers'] = 0
    
    # Output
    config['output_dir'] = Path('./checkpoints')
    config['logging_dir'] = Path('./logging')
    config['submissions_dir'] = Path('./submissions')
    
    return config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train VideoMAE model 10 (Sequential Layer Training Strategy)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model-id', type=int, default=10, help='Model ID')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--data-dir', type=str, default='./kaggle_data/data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--logging-dir', type=str, default='./logging', help='Logging directory')
    parser.add_argument('--submissions-dir', type=str, default='./submissions', help='Submissions directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint for inference')
    parser.add_argument('--inference-only', action='store_true', help='Inference only mode')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Get config
    config = get_videomae_model10_config()
    config['model_id'] = args.model_id
    config['seed'] = args.seed
    config['data_dir'] = Path(args.data_dir)
    config['output_dir'] = Path(args.output_dir)
    config['logging_dir'] = Path(args.logging_dir)
    config['submissions_dir'] = Path(args.submissions_dir)
    config['resume'] = args.resume
    config['checkpoint'] = args.checkpoint
    config['inference_only'] = args.inference_only
    config['num_workers'] = args.num_workers if args.num_workers > 0 else config['num_workers']
    config['debug'] = args.debug
    
    # Create directories
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    config['logging_dir'].mkdir(parents=True, exist_ok=True)
    config['submissions_dir'].mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger_instance = setup_logging(
        config['model_id'],
        config['logging_dir'],
        debug=config['debug']
    )
    
    # Print config
    logger_instance.info("="*60)
    logger_instance.info("VideoMAE Model 10 Training")
    logger_instance.info("Sequential Layer Training Strategy")
    logger_instance.info("="*60)
    logger_instance.info(f"Model ID: {config['model_id']}")
    logger_instance.info(f"Seed: {config['seed']}")
    logger_instance.info(f"Architecture: {config['architecture']}")
    logger_instance.info(f"Batch size: {config['batch_size']}")
    logger_instance.info(f"Gradient accumulation: {config['grad_accum_steps']}")
    logger_instance.info(f"Effective batch size: {config['batch_size'] * config['grad_accum_steps']}")
    logger_instance.info(f"Training Strategy:")
    logger_instance.info(f"  Phase 1: Freeze backbone, train head ({config['freeze_backbone_epochs']} epochs)")
    logger_instance.info(f"  Phase 2: Sequential layer training ({config['num_layer_phases']} phases, {config['layer_phase_max_epochs']} max epochs each)")
    logger_instance.info(f"  Phase 3: Freeze all, train head ({config['final_head_epochs']} epochs)")
    logger_instance.info(f"Initialization (Model 9 style):")
    logger_instance.info(f"  - Output gain: {config['init_output_gain']} → std={config['init_output_gain'] * 0.01:.3f}")
    logger_instance.info(f"  - Hidden gain: {config['init_hidden_gain']} → std={config['init_hidden_gain'] * 0.01:.3f}")
    logger_instance.info(f"  - Use normal init: {config['use_normal_init']}")
    logger_instance.info("")
    
    # Inference only mode
    if config.get('inference_only', False):
        checkpoint_path = Path(
            config.get('checkpoint', config['output_dir'] / f'videomae_model_{config["model_id"]}_best.pt')
        )
        if not checkpoint_path.exists():
            logger_instance.error(f"Checkpoint not found: {checkpoint_path}")
            return
        logger_instance.info("Inference mode - to be implemented")
        return
    
    # Set random seed
    set_random_seeds(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger_instance.info(f"Device: {device}")
    log_system_info(logger_instance, device)
    
    # Load datasets
    logger_instance.info("Loading datasets...")
    train_data_dir = config['data_dir'] / 'data_train'
    test_data_dir = config['data_dir'] / 'test'
    
    train_dataset = VideoDataset(
        root=train_data_dir,
        num_frames=config['num_frames'],
        frame_stride=config['frame_stride'],
        image_size=config['img_size'],
        is_train=True,
        val_ratio=config['val_ratio'],
        seed=config['seed'],
        use_temporal_aug=config.get('use_temporal_aug', False),
        use_advanced_spatial=config.get('use_advanced_spatial', False),
        use_advanced_color=config.get('use_advanced_color', False)
    )
    
    val_dataset = VideoDataset(
        root=train_data_dir,
        num_frames=config['num_frames'],
        frame_stride=config['frame_stride'],
        image_size=config['img_size'],
        is_train=False,
        val_ratio=config['val_ratio'],
        seed=config['seed']
    )
    
    logger_instance.info(f"Train samples: {len(train_dataset)}")
    logger_instance.info(f"Val samples: {len(val_dataset)}")
    logger_instance.info(f"Classes: {len(train_dataset.classes)}")
    
    # Create model
    logger_instance.info("Creating VideoMAE model with improved initialization...")
    model = create_model(
        architecture=config['architecture'],
        num_classes=len(train_dataset.classes),
        pretrained_name=None,
        use_adapters=config['use_adapters'],
        dropout=config.get('dropout', 0.1),
        drop_path_rate=config.get('drop_path_rate', 0.0),
        pretrained_ckpt=config.get('pretrained_ckpt', None),
        model_name=config.get('model_name', 'MCG-NJU/videomae-large-finetuned-kinetics'),
        num_frames=config['num_frames'],
        tubelet_size=config.get('tubelet_size', 2),
        image_size=config['img_size'],
        patch_size=config.get('patch_size', 16),
        init_output_gain=config.get('init_output_gain', 2.0),
        init_hidden_gain=config.get('init_hidden_gain', 1.0),
        use_normal_init=config.get('use_normal_init', True)
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger_instance.info(f"Total parameters: {total_params:,}")
    logger_instance.info(f"Trainable parameters: {trainable_params:,}")
    logger_instance.info(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Checkpoint paths
    checkpoint_path = config['output_dir'] / f'videomae_model_{config["model_id"]}_best.pt'
    history_plot_path = config['output_dir'] / f'videomae_model_{config["model_id"]}_training.png'
    
    # Combined history for all phases
    combined_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_overall_val_acc = 0.0
    
    # ===================================================================
    # Phase 1: Freeze Backbone, Train Head
    # ===================================================================
    logger_instance.info("="*60)
    logger_instance.info("PHASE 1: Freeze Backbone, Train Head")
    logger_instance.info("="*60)
    
    # Setup optimizer for Phase 1 (head only)
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'classifier' in name or 'head' in name:
            head_params.append(param)
    
    if len(head_params) == 0:
        raise RuntimeError("ERROR: No head params found!")
    
    # Freeze backbone
    if hasattr(model, 'videomae'):
        for param in model.videomae.parameters():
            param.requires_grad = False
        logger_instance.info("VideoMAE backbone frozen for Phase 1")
    
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": config['head_lr']}],
        weight_decay=config['weight_decay']
    )
    
    # Setup scheduler
    if config.get('use_adaptive_lr', False):
        scheduler = get_adaptive_lr_scheduler(
            optimizer,
            num_epochs=config['freeze_backbone_epochs'],
            warmup_epochs=config['warmup_epochs'],
            min_lr_ratio=config.get('min_lr_ratio', 0.01),
            plateau_patience=config.get('lr_plateau_patience', 7),
            min_delta=config.get('lr_min_delta', 0.005),
            mode='max',
            verbose=True,
            cooldown=config.get('lr_cooldown', 7),
            threshold_mode=config.get('lr_threshold_mode', 'rel'),
            use_val_loss=config.get('use_val_loss_for_lr', False)
        )
    else:
        scheduler = get_lr_scheduler(
            optimizer,
            epochs=config['freeze_backbone_epochs'],
            warmup_epochs=config['warmup_epochs'],
            cosine_start_epoch=config.get('cosine_start_epoch', config['warmup_epochs'] + 5)
        )
    
    scaler = torch.amp.GradScaler()
    
    # DataLoader
    is_windows = os.name == 'nt'
    num_workers = config['num_workers']
    if num_workers == 0 and is_windows:
        logger_instance.info("Windows: Using num_workers=0")
    
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False
    )
    
    # Train Phase 1
    phase1_config = config.copy()
    phase1_config['epochs'] = config['freeze_backbone_epochs']
    phase1_config['early_stop_patience'] = config['freeze_backbone_epochs'] + 1  # No early stopping in Phase 1
    
    phase1_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        config=phase1_config,
        checkpoint_path=checkpoint_path,
        history_plot_path=history_plot_path,
        classes=train_dataset.classes,
        resume_epoch=0,
        resume_history=None
    )
    
    # Merge Phase 1 history
    combined_history['train_loss'].extend(phase1_history.get('train_loss', []))
    combined_history['train_acc'].extend(phase1_history.get('train_acc', []))
    combined_history['val_loss'].extend(phase1_history.get('val_loss', []))
    combined_history['val_acc'].extend(phase1_history.get('val_acc', []))
    
    best_phase1_val_acc = max(phase1_history['val_acc']) if phase1_history.get('val_acc') else 0.0
    if best_phase1_val_acc > best_overall_val_acc:
        best_overall_val_acc = best_phase1_val_acc
        # Save best model after Phase 1
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=config['freeze_backbone_epochs'],
            history=combined_history,
            best_val_acc=best_overall_val_acc,
            train_acc=phase1_history['train_acc'][-1] if phase1_history.get('train_acc') else 0.0,
            classes=train_dataset.classes,
            config=config,
            checkpoint_path=checkpoint_path,
            ema_model=None
        )
        logger_instance.info(f"  ✓ Best model saved after Phase 1 (val_acc: {best_overall_val_acc:.4f})")
    
    logger_instance.info(f"Phase 1 completed. Best val acc: {best_phase1_val_acc:.4f}")
    
    # ===================================================================
    # Phase 2: Sequential Layer Training
    # ===================================================================
    logger_instance.info("="*60)
    logger_instance.info("PHASE 2: Sequential Layer Training")
    logger_instance.info("="*60)
    
    phase2_history, best_phase2_val_acc = sequential_layer_training_loop(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        config=config,
        classes=train_dataset.classes
    )
    
    # Merge Phase 2 history
    combined_history['train_loss'].extend(phase2_history.get('train_loss', []))
    combined_history['train_acc'].extend(phase2_history.get('train_acc', []))
    combined_history['val_loss'].extend(phase2_history.get('val_loss', []))
    combined_history['val_acc'].extend(phase2_history.get('val_acc', []))
    
    if best_phase2_val_acc > best_overall_val_acc:
        best_overall_val_acc = best_phase2_val_acc
        # Save best model after Phase 2 (create temporary optimizer/scheduler for checkpoint)
        # The actual optimizer/scheduler from Phase 2 is not accessible here
        temp_head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'classifier' in name or 'head' in name:
                temp_head_params.append(param)
        
        if temp_head_params:
            temp_optimizer = torch.optim.AdamW([{"params": temp_head_params, "lr": config['final_head_lr']}])
            temp_scheduler = get_adaptive_lr_scheduler(
                temp_optimizer,
                num_epochs=config['final_head_epochs'],
                warmup_epochs=5,
                min_lr_ratio=config.get('min_lr_ratio', 0.01),
                plateau_patience=config.get('lr_plateau_patience', 8),
                min_delta=config.get('lr_min_delta', 0.005),
                mode='max',
                verbose=False,
                cooldown=config.get('lr_cooldown', 3),
                threshold_mode=config.get('lr_threshold_mode', 'rel'),
                use_val_loss=config.get('use_val_loss_for_lr', False)
            )
            save_checkpoint(
                model=model,
                optimizer=temp_optimizer,
                scheduler=temp_scheduler,
                epoch=config['freeze_backbone_epochs'] + len(phase2_history.get('val_acc', [])),
                history=combined_history,
                best_val_acc=best_overall_val_acc,
                train_acc=phase2_history['train_acc'][-1] if phase2_history.get('train_acc') else 0.0,
                classes=train_dataset.classes,
                config=config,
                checkpoint_path=checkpoint_path,
                ema_model=None
            )
            logger_instance.info(f"  ✓ Best model saved after Phase 2 (val_acc: {best_overall_val_acc:.4f})")
    
    logger_instance.info(f"Phase 2 completed. Best val acc: {best_phase2_val_acc:.4f}")
    
    # ===================================================================
    # Phase 3: Final Head Training
    # ===================================================================
    logger_instance.info("="*60)
    logger_instance.info("PHASE 3: Final Head Training (Freeze All, Train Head)")
    logger_instance.info("="*60)
    
    # Freeze all backbone layers
    freeze_all_backbone_layers(model)
    
    # Setup optimizer for Phase 3 (head only)
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'classifier' in name or 'head' in name:
            head_params.append(param)
    
    if len(head_params) == 0:
        raise RuntimeError("ERROR: No head params found!")
    
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": config['final_head_lr']}],
        weight_decay=config.get('weight_decay_unfreeze', config['weight_decay'])
    )
    
    # Setup scheduler for Phase 3
    scheduler = get_adaptive_lr_scheduler(
        optimizer,
        num_epochs=config['final_head_epochs'],
        warmup_epochs=5,
        min_lr_ratio=config.get('min_lr_ratio', 0.01),
        plateau_patience=config.get('lr_plateau_patience', 8),
        min_delta=config.get('lr_min_delta', 0.005),
        mode='max',
        verbose=True,
        cooldown=config.get('lr_cooldown', 3),
        threshold_mode=config.get('lr_threshold_mode', 'rel'),
        use_val_loss=config.get('use_val_loss_for_lr', False)
    )
    
    scaler = torch.amp.GradScaler()
    
    # Train Phase 3
    best_phase3_val_acc, phase3_history = final_head_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        config=config
    )
    
    # Merge Phase 3 history
    combined_history['train_loss'].extend(phase3_history.get('train_loss', []))
    combined_history['train_acc'].extend(phase3_history.get('train_acc', []))
    combined_history['val_loss'].extend(phase3_history.get('val_loss', []))
    combined_history['val_acc'].extend(phase3_history.get('val_acc', []))
    
    if best_phase3_val_acc > best_overall_val_acc:
        best_overall_val_acc = best_phase3_val_acc
    
    # Save final best model
    total_epochs = (
        config['freeze_backbone_epochs'] + 
        len(phase2_history.get('val_acc', [])) +
        len(phase3_history.get('val_acc', []))
    )
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=total_epochs,
        history=combined_history,
        best_val_acc=best_overall_val_acc,
        train_acc=phase3_history['train_acc'][-1] if phase3_history.get('train_acc') else 0.0,
        classes=train_dataset.classes,
        config=config,
        checkpoint_path=checkpoint_path,
        ema_model=None
    )
    
    logger_instance.info(f"Phase 3 completed. Best val acc: {best_phase3_val_acc:.4f}")
    logger_instance.info(f"  ✓ Final best model saved (val_acc: {best_overall_val_acc:.4f})")
    
    # ===================================================================
    # Final Summary
    # ===================================================================
    logger_instance.info("="*60)
    logger_instance.info("Training Completed!")
    logger_instance.info("="*60)
    logger_instance.info(f"Phase 1 best val acc: {best_phase1_val_acc:.4f}")
    logger_instance.info(f"Phase 2 best val acc: {best_phase2_val_acc:.4f}")
    logger_instance.info(f"Phase 3 best val acc: {best_phase3_val_acc:.4f}")
    logger_instance.info(f"Overall best val acc: {best_overall_val_acc:.4f}")
    logger_instance.info(f"Model saved to: {checkpoint_path}")
    logger_instance.info(f"Training plot saved to: {history_plot_path}")
    
    # Update plot with combined history
    plot_training_history(combined_history, history_plot_path, config['model_id'])
    
    logger_instance.info("="*60)


if __name__ == '__main__':
    main()
