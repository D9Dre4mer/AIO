"""
Script để train VideoMAE model 8 cho action recognition.
Model 8: Progressive unfreezing (Phase 1: freeze → Phase 2: unfreeze 6 blocks) để đạt 90% accuracy.
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
from sota_training.utils import setup_logging, log_system_info, load_checkpoint
from sota_training.dataset import VideoDataset, TestDataset
from sota_training.models import create_model
from sota_training.training import train_model, get_lr_scheduler
from sota_training.adaptive_lr_scheduler import get_adaptive_lr_scheduler
from sota_training.inference import run_inference, generate_submission

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_videomae_model8_config():
    """Tạo config cho VideoMAE Model 8: Freeze toàn bộ, dùng hết augmentation."""
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
    
    # Training parameters - Progressive unfreezing để tăng capacity
    # Research: Head-only bị kẹt ở 80% → cần unfreeze một phần backbone
    config['batch_size'] = 8  # FIX: Giảm từ 52 → 8 để tránh OOM khi unfreeze
    config['grad_accum_steps'] = 6  # FIX: Tăng từ 2 → 6 để giữ effective batch size = 48 (8*6=48)
    config['epochs'] = 150  # Tăng epochs lên 150 để train lâu hơn
    config['warmup_epochs'] = 10  # Warmup 10 epochs
    config['cosine_start_epoch'] = 10  # Bắt đầu cosine sau warmup
    
    # Learning rates - 2-phase training với progressive unfreezing
    # Phase 1: Freeze backbone, train head
    config['base_lr'] = 0.0  # Backbone frozen trong Phase 1
    config['head_lr'] = 2e-3  # Head LR cao để học nhanh
    
    # Phase 2: Progressive unfreeze với differential LR
    # FIX: Research cho thấy 1e-6 quá conservative → tăng lên 5e-6 để model có thể adapt tốt hơn
    # Research: VideoMAE đạt 91.3% UCF101 với backbone LR ~1e-5, nhưng 5e-6 an toàn hơn
    config['base_lr_unfreeze'] = 5e-6  # FIX: Tăng từ 1e-6 → 5e-6 (optimal cho small dataset)
    config['head_lr_unfreeze'] = 5e-4  # Giảm head LR khi unfreeze
    
    # Freeze strategy - Progressive unfreezing để tăng capacity
    # Research: Unfreeze từ deeper layers trước (progressive unfreezing)
    # FIX: Tăng số layers unfreeze từ 6 → 8 để tăng capacity nhiều hơn
    config['freeze_backbone_epochs'] = 10  # Phase 1: 10 epochs freeze backbone
    config['unfreeze_partial'] = True  # Phase 2: partial unfreeze
    config['unfreeze_num_blocks'] = 8  # FIX: Tăng từ 6 → 8 layers để tăng capacity (VideoMAE có 24 layers)
    config['skip_unfreeze'] = False  # Enable Phase 2 unfreeze
    
    # Regularization - Phase 1: Regularization vừa phải
    # Research: Phase 1 (freeze) cần regularization vừa phải
    config['weight_decay'] = 0.05  # Weight decay vừa phải
    config['dropout'] = 0.15  # Dropout vừa phải
    config['drop_path_rate'] = 0.0  # Tắt drop path (head nhỏ)
    config['label_smoothing'] = 0.05  # Label smoothing nhẹ
    
    # Phase 2: Giảm regularization khi unfreeze (để model học tốt hơn)
    # Research: Khi unfreeze, giảm regularization để model có thể adapt
    config['weight_decay_unfreeze'] = 0.03  # Giảm weight decay khi unfreeze
    config['dropout_unfreeze'] = 0.1  # Giảm dropout khi unfreeze
    config['drop_path_rate_unfreeze'] = 0.05  # Bật drop path nhẹ khi unfreeze
    config['label_smoothing_unfreeze'] = 0.03  # Giảm label smoothing khi unfreeze
    
    # Augmentation - Phase 1: Augmentation vừa phải
    config['mixup_alpha'] = 0.3  # Mixup vừa phải
    
    # Phase 2: Giảm augmentation khi unfreeze
    config['mixup_alpha_unfreeze'] = 0.2  # Giảm mixup khi unfreeze
    config['cutmix_alpha_unfreeze'] = 0.8  # Giảm cutmix khi unfreeze
    config['use_cutmix_unfreeze'] = True  # Vẫn dùng cutmix
    config['cutmix_alpha'] = 1.0  # Research: CutMix tốt cho video
    config['use_cutmix'] = True  # Bật CutMix
    config['use_temporal_aug'] = True  # Research: Temporal augmentation quan trọng
    config['use_advanced_spatial'] = True  # Research: Spatial augmentation mạnh
    config['use_advanced_color'] = True  # Research: Color jittering quan trọng
    
    # Loss
    config['use_focal_loss'] = False
    
    # Features - FIX: Tắt EMA tạm thời để tránh instability
    # Research: EMA có thể gây vấn đề nếu shadow params chưa tốt (epoch 4 spike)
    # Có thể bật lại sau khi training ổn định
    config['use_ema'] = False  # FIX: Tắt EMA tạm thời để tránh instability
    config['ema_decay'] = 0.9999  # Giữ giá trị cho khi bật lại
    config['use_adapters'] = False
    config['use_progressive_resize'] = False
    config['use_synthetic_data'] = False
    config['use_distillation'] = False
    
    # Adaptive LR - Tối ưu để đạt 90% accuracy
    # Research: Với LR cao, cần scheduler nhạy và linh hoạt
    config['use_adaptive_lr'] = True
    config['lr_plateau_patience'] = 5  # Patience ngắn hơn để giảm LR nhanh khi cần
    config['lr_min_delta'] = 0.003  # Min delta nhỏ hơn (0.3%) để nhạy với improvement nhỏ
    config['lr_threshold_mode'] = 'rel'
    config['lr_cooldown'] = 2  # Cooldown ngắn (2 epochs) để responsive
    config['use_val_loss_for_lr'] = False
    config['min_lr_ratio'] = 0.005  # Min LR ratio nhỏ hơn (0.5%) để train lâu hơn
    
    # Early stopping - Tăng patience để train lâu hơn đạt 90%
    config['early_stop_patience'] = 20  # Patience 20 để train lâu hơn và đạt accuracy cao nhất
    
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
        description='Train VideoMAE model 8 (Freeze toàn bộ, dùng hết augmentation)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model-id', type=int, default=8, help='Model ID')
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
    config = get_videomae_model8_config()
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
    logger_instance.info("VideoMAE Model 8 Training")
    logger_instance.info("Progressive Unfreezing: Phase 1 (freeze) → Phase 2 (unfreeze 6 blocks)")
    logger_instance.info("="*60)
    logger_instance.info(f"Model ID: {config['model_id']}")
    logger_instance.info(f"Seed: {config['seed']}")
    logger_instance.info(f"Architecture: {config['architecture']}")
    logger_instance.info(f"Batch size: {config['batch_size']}")
    logger_instance.info(f"Gradient accumulation: {config['grad_accum_steps']}")
    logger_instance.info(f"Effective batch size: {config['batch_size'] * config['grad_accum_steps']}")
    logger_instance.info(f"Learning rate - Base: {config['base_lr']}, Head: {config['head_lr']}")
    logger_instance.info(f"Epochs: {config['epochs']}")
    logger_instance.info(f"Freeze strategy: Phase 1 (freeze {config['freeze_backbone_epochs']} epochs) → Phase 2 (unfreeze {config['unfreeze_num_blocks']} blocks)")
    logger_instance.info(f"Augmentation:")
    logger_instance.info(f"  - Mixup: {config['mixup_alpha']}")
    logger_instance.info(f"  - CutMix: {config['use_cutmix']} (alpha={config['cutmix_alpha']})")
    logger_instance.info(f"  - Temporal: {config['use_temporal_aug']}")
    logger_instance.info(f"  - Advanced Spatial: {config['use_advanced_spatial']}")
    logger_instance.info(f"  - Advanced Color: {config['use_advanced_color']}")
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
    logger_instance.info("Creating VideoMAE model...")
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
        patch_size=config.get('patch_size', 16)
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger_instance.info(f"Total parameters: {total_params:,}")
    logger_instance.info(f"Trainable parameters: {trainable_params:,}")
    logger_instance.info(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Setup optimizer
    logger_instance.info("Setting up optimizer...")
    backbone_params = []
    adapter_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'classifier' in name or 'head' in name:
            head_params.append(param)
        elif 'adapter' in name or 'temporal_attention' in name or 'space_time_attention' in name:
            adapter_params.append(param)
        else:
            backbone_params.append(param)
    
    base_lr = config['base_lr']
    head_lr = config['head_lr']
    
    logger_instance.info(f"Learning rate:")
    logger_instance.info(f"  Base LR: {base_lr:.6f}")
    logger_instance.info(f"  Head LR: {head_lr:.6f}")
    logger_instance.info(f"  Backbone params: {len(backbone_params)}")
    logger_instance.info(f"  Adapter params: {len(adapter_params)}")
    logger_instance.info(f"  Head params: {len(head_params)}")
    
    if len(head_params) == 0:
        raise RuntimeError("ERROR: No head params found!")
    
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": base_lr})
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": head_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    
    if len(param_groups) == 0:
        raise RuntimeError("CRITICAL: No trainable parameters found!")
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config['weight_decay'])
    
    # Setup scheduler
    if config.get('use_adaptive_lr', False):
        scheduler = get_adaptive_lr_scheduler(
            optimizer,
            num_epochs=config['epochs'],
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
            epochs=config['epochs'],
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
    
    # Resume from checkpoint
    resume_epoch = 0
    resume_history = None
    if config.get('resume'):
        checkpoint = load_checkpoint(Path(config['resume']), device, logger_instance)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        resume_epoch = checkpoint.get('epoch', 0)
        resume_history = checkpoint.get('history')
        logger_instance.info(f"Resumed from epoch {resume_epoch}")
    
    # Checkpoint paths
    checkpoint_path = config['output_dir'] / f'videomae_model_{config["model_id"]}_best.pt'
    history_plot_path = config['output_dir'] / f'videomae_model_{config["model_id"]}_training.png'
    
    # Train
    logger_instance.info("="*60)
    logger_instance.info("Starting Training")
    logger_instance.info("="*60)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        config=config,
        checkpoint_path=checkpoint_path,
        history_plot_path=history_plot_path,
        classes=train_dataset.classes,
        resume_epoch=resume_epoch,
        resume_history=resume_history
    )
    
    logger_instance.info("Training completed!")
    logger_instance.info(f"Best val accuracy: {max(history['val_acc']) if history['val_acc'] else 0.0:.4f}")
    logger_instance.info(f"Model saved to: {checkpoint_path}")


if __name__ == '__main__':
    main()
