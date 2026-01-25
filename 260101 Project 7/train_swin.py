"""
Script riêng để train Swin Transformer với config tối ưu cho phát hiện hành động con người.

Dựa trên:
- Best practices từ Video Swin Transformer paper và implementation
- Config hiện tại trong project (Model 4)
- Nghiên cứu về hyperparameters tối ưu cho action recognition

Key optimizations:
- Learning rate: Backbone 3e-5, Head 3e-4 (theo research) hoặc conservative 5e-5/1e-3
- Batch size: 32-64 (tùy GPU memory, không quá cao)
- Drop path: 0.3 cho Swin-B (theo research)
- Weight decay: 0.05 cho Swin-B (theo research)
- Warmup: 3 epochs (theo research)
- Num frames: 32 (theo research) hoặc 16 (nếu memory limited)
- Frame stride: 2
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
# Enable cuDNN benchmark để tăng tốc (nếu input size không đổi)
torch.backends.cudnn.benchmark = True  # Enable để training nhanh hơn

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
    torch.cuda.manual_seed_all(seed)
    # Không set deterministic=True và benchmark=False vì sẽ làm chậm training
    # Chỉ set seed để đảm bảo reproducibility


def get_swin_optimized_config():
    """
    Tạo config tối ưu cho Swin Transformer dựa trên research và best practices.
    
    Best practices từ Video Swin Transformer paper:
    - Batch size: 64 (có thể điều chỉnh theo GPU)
    - Learning rate: Backbone ~3e-5, Head ~3e-4 (tỷ lệ 10x)
    - Drop path: Swin-B: 0.3
    - Weight decay: Swin-B: ~0.05
    - Warmup: 2-3 epochs
    - Cosine decay
    - 32 frames với stride 2 (cho Kinetics-style datasets)
    
    Returns:
        dict: Config tối ưu cho Swin
    """
    default_config = get_default_config()
    
    # Config tối ưu cho Swin Transformer
    swin_config = default_config.copy()
    
    # Architecture
    swin_config['architecture'] = 'swin'
    swin_config['pretrained_name'] = 'swin_base_patch4_window7_224'
    
    # Video parameters - Tối ưu cho tốc độ: giảm xuống 16 frames
    # 32 frames quá nhiều → training chậm (100 videos × 32 frames = 3,200 images/batch)
    swin_config['num_frames'] = 16  # Giảm từ 32 → 16 để training nhanh hơn 2x
    swin_config['frame_stride'] = 2
    swin_config['img_size'] = 224
    
    # Training parameters - Tối ưu cho tốc độ
    # Batch size 100 quá lớn → CPU bottleneck (phải load 100 videos × 16 frames = 1,600 images)
    swin_config['batch_size'] = 64  # Giảm từ 100 → 64 để giảm CPU bottleneck
    swin_config['grad_accum_steps'] = 2  # Gradient accumulation để giữ effective batch size ~128
    swin_config['epochs'] = 100
    swin_config['warmup_epochs'] = 3  # Theo research: 2-3 epochs
    swin_config['cosine_start_epoch'] = 3  # Cosine bắt đầu ngay sau warmup
    
    # Learning rates - theo research: Backbone 3e-5, Head 3e-4 (tỷ lệ 10x)
    # Conservative hơn: 5e-5 và 1e-3 để training ổn định hơn
    swin_config['base_lr'] = 3e-5  # Theo research (có thể dùng 5e-5 nếu muốn conservative hơn)
    swin_config['head_lr'] = 3e-4  # Theo research (có thể dùng 1e-3 nếu muốn conservative hơn)
    
    # Regularization - theo research cho Swin-B
    swin_config['weight_decay'] = 0.05  # Theo research: Swin-B dùng ~0.05
    swin_config['dropout'] = 0.3  # Moderate dropout
    swin_config['drop_path_rate'] = 0.3  # Theo research: Swin-B dùng 0.3
    swin_config['label_smoothing'] = 0.1  # Moderate label smoothing
    
    # Augmentation - TẮT HẾT để training nhanh nhất
    swin_config['mixup_alpha'] = 0.0  # Tắt Mixup
    swin_config['cutmix_alpha'] = 0.0  # Tắt CutMix
    swin_config['use_cutmix'] = False  # Tắt CutMix
    swin_config['use_temporal_aug'] = False  # Tắt temporal augmentation
    swin_config['use_advanced_spatial'] = False  # Tắt RandAugment/AutoAugment
    swin_config['use_advanced_color'] = False  # Tắt color jitter
    
    # Loss function
    swin_config['use_focal_loss'] = True  # Focal loss tốt cho class imbalance
    swin_config['focal_alpha'] = 0.25
    swin_config['focal_gamma'] = 2.0
    
    # Advanced features
    swin_config['use_ema'] = False  # User không muốn dùng EMA
    swin_config['use_progressive_resize'] = False  # Tắt vì chưa implement đầy đủ
    swin_config['use_synthetic_data'] = False  # Tắt để training nhanh hơn
    swin_config['use_pseudo_labeling'] = False  # Standalone model, không cần pseudo-labeling
    
    # Adaptive LR scheduler
    swin_config['use_adaptive_lr'] = True
    swin_config['lr_plateau_patience'] = 7
    swin_config['lr_min_delta'] = 0.005
    swin_config['lr_threshold_mode'] = 'rel'
    swin_config['lr_cooldown'] = 7
    swin_config['use_val_loss_for_lr'] = False
    swin_config['min_lr_ratio'] = 0.01
    
    # Early stopping
    swin_config['early_stop_patience'] = 12
    
    # Knowledge distillation: Not used (disabled)
    swin_config['use_distillation'] = False
    swin_config['teacher_checkpoints'] = []
    
    # Data
    swin_config['val_ratio'] = 0.20
    # Tối ưu num_workers: thử 2-4 workers để giảm CPU bottleneck
    # Windows có overhead nhưng với batch size lớn vẫn nhanh hơn
    swin_config['num_workers'] = 2  # Tăng từ 0 → 2 để giảm CPU bottleneck
    
    # Output
    swin_config['output_dir'] = Path('./checkpoints')
    swin_config['logging_dir'] = Path('./logging')
    swin_config['submissions_dir'] = Path('./submissions')
    
    return swin_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Swin Transformer với config tối ưu cho action recognition',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--model-id', type=int, default=4, help='Model ID (default: 4)')
    parser.add_argument('--seed', type=int, default=789, help='Random seed (default: 789 cho Model 4)')
    
    # Data configuration
    parser.add_argument('--data-dir', type=str, default='./kaggle_data/data', help='Path to dataset directory')
    parser.add_argument('--num-frames', type=int, default=None, help='Number of frames (default: 16, có thể tăng lên 32 nếu muốn accuracy cao hơn)')
    parser.add_argument('--frame-stride', type=int, default=2, help='Frame stride')
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (default: 64, có thể tăng lên 100 nếu GPU đủ)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (default: 100)')
    parser.add_argument('--base-lr', type=float, default=None, help='Base learning rate (default: 3e-5)')
    parser.add_argument('--head-lr', type=float, default=None, help='Head learning rate (default: 3e-4)')
    parser.add_argument('--weight-decay', type=float, default=None, help='Weight decay (default: 0.05)')
    parser.add_argument('--drop-path-rate', type=float, default=None, help='Drop path rate (default: 0.3)')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--logging-dir', type=str, default='./logging', help='Logging directory')
    parser.add_argument('--submissions-dir', type=str, default='./submissions', help='Submissions directory')
    
    # Training control
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--inference-only', action='store_true', help='Run inference only')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path for inference')
    
    # System
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    return args




def main():
    """Main function."""
    args = parse_args()
    
    # Get optimized config
    config = get_swin_optimized_config()
    
    # Override with command-line arguments
    config['model_id'] = args.model_id
    config['seed'] = args.seed
    config['data_dir'] = Path(args.data_dir)
    
    if args.num_frames is not None:
        config['num_frames'] = args.num_frames
    if args.frame_stride is not None:
        config['frame_stride'] = args.frame_stride
    if args.img_size is not None:
        config['img_size'] = args.img_size
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.base_lr is not None:
        config['base_lr'] = args.base_lr
    if args.head_lr is not None:
        config['head_lr'] = args.head_lr
    if args.weight_decay is not None:
        config['weight_decay'] = args.weight_decay
    if args.drop_path_rate is not None:
        config['drop_path_rate'] = args.drop_path_rate
    
    # Knowledge distillation: Disabled by default (not used)
    
    config['output_dir'] = Path(args.output_dir)
    config['logging_dir'] = Path(args.logging_dir)
    config['submissions_dir'] = Path(args.submissions_dir)
    config['resume'] = args.resume
    config['inference_only'] = args.inference_only
    config['checkpoint'] = args.checkpoint
    config['num_workers'] = args.num_workers
    config['debug'] = args.debug
    
    # Create output directories
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    config['logging_dir'].mkdir(parents=True, exist_ok=True)
    config['submissions_dir'].mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(
        model_id=config['model_id'],
        logging_dir=config['logging_dir'],
        debug=config.get('debug', False)
    )
    
    # Print welcome message with emoji (similar to train_model1_to_model2.py)
    print("\n" + "="*80)
    print("🚀 Swin Transformer Training - Optimized Config")
    print("="*80)
    print(f"Model ID: {config['model_id']}")
    print(f"Seed: {config['seed']}")
    print(f"Architecture: {config['architecture']}")
    print(f"Pretrained: {config['pretrained_name']}")
    print("")
    print("Optimized Hyperparameters (based on research):")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Num frames: {config['num_frames']} (stride: {config['frame_stride']})")
    print(f"  Image size: {config['img_size']}")
    print(f"  Base LR: {config['base_lr']:.6f}")
    print(f"  Head LR: {config['head_lr']:.6f}")
    print(f"  Weight decay: {config['weight_decay']}")
    print(f"  Drop path rate: {config['drop_path_rate']}")
    print(f"  Warmup epochs: {config['warmup_epochs']}")
    print(f"  Total epochs: {config['epochs']}")
    print("")
    print("Augmentation: TẮT HẾT (để training nhanh nhất)")
    print("="*80 + "\n")
    
    # Also log to file
    logger.info("="*60)
    logger.info("Swin Transformer Training - Optimized Config")
    logger.info("="*60)
    logger.info(f"Model ID: {config['model_id']}")
    logger.info(f"Seed: {config['seed']}")
    logger.info(f"Architecture: {config['architecture']}")
    logger.info(f"Pretrained: {config['pretrained_name']}")
    logger.info("")
    logger.info("Optimized Hyperparameters (based on research):")
    logger.info(f"  Batch size: {config['batch_size']}")
    logger.info(f"  Num frames: {config['num_frames']} (stride: {config['frame_stride']})")
    logger.info(f"  Image size: {config['img_size']}")
    logger.info(f"  Base LR: {config['base_lr']:.6f}")
    logger.info(f"  Head LR: {config['head_lr']:.6f}")
    logger.info(f"  Weight decay: {config['weight_decay']}")
    logger.info(f"  Drop path rate: {config['drop_path_rate']}")
    logger.info(f"  Warmup epochs: {config['warmup_epochs']}")
    logger.info(f"  Total epochs: {config['epochs']}")
    logger.info("")
    
    # Inference only mode
    if config.get('inference_only', False):
        checkpoint_path = Path(config.get('checkpoint', config['output_dir'] / f'swin_model_{config["model_id"]}_best.pt'))
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
        # Run inference (simplified - có thể mở rộng sau)
        logger.info("Inference mode - to be implemented")
        return
    
    # Set random seed
    set_random_seeds(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    log_system_info(logger, device)
    
    # Load datasets
    logger.info("Loading datasets...")
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
        use_temporal_aug=config.get('use_temporal_aug', True),
        use_advanced_spatial=config.get('use_advanced_spatial', True),
        use_advanced_color=config.get('use_advanced_color', True)
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
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Classes: {len(train_dataset.classes)}")
    
    # Create model
    print("🔧 Creating Swin Transformer model...")
    logger.info("Creating Swin Transformer model...")
    model = create_model(
        architecture=config['architecture'],
        num_classes=len(train_dataset.classes),
        pretrained_name=config['pretrained_name'],
        use_adapters=config['use_adapters'],
        dropout=config.get('dropout', 0.3),
        drop_path_rate=config.get('drop_path_rate', 0.3)
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    print(f"📊 Trainable ratio: {trainable_params/total_params*100:.2f}%")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Knowledge distillation: Not used (disabled)
    teacher = None
    
    # DataLoader settings (Windows-compatible)
    num_workers = config['num_workers']
    if num_workers == 0 and os.name == 'nt':
        logger.info("Windows: Using num_workers=0 to avoid shared memory errors")
    
    is_windows = os.name == 'nt'
    if num_workers > 0 and is_windows:
        use_persistent_workers = False
        prefetch_factor = 2
        import multiprocessing
        multiprocessing_context = multiprocessing.get_context('spawn')
    elif num_workers > 0:
        use_persistent_workers = True
        prefetch_factor = 4
        multiprocessing_context = None
    else:
        use_persistent_workers = False
        prefetch_factor = None
        multiprocessing_context = None
    
    use_pin_memory = torch.cuda.is_available() and config['batch_size'] <= 128
    
    dataloader_kwargs = {
        'batch_size': config['batch_size'],
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': use_pin_memory,
        'persistent_workers': use_persistent_workers,
        'prefetch_factor': prefetch_factor,
        'drop_last': False
    }
    if multiprocessing_context is not None:
        dataloader_kwargs['multiprocessing_context'] = multiprocessing_context
    
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs)
    
    val_dataloader_kwargs = dataloader_kwargs.copy()
    val_dataloader_kwargs['shuffle'] = False
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_dataloader_kwargs)
    
    # Memory estimation
    log_system_info(
        logger=logger,
        device=device,
        batch_size=config['batch_size'],
        num_frames=config['num_frames'],
        img_size=config['img_size'],
        model_params=total_params
    )
    
    # Setup optimizer
    logger.info("Setting up optimizer...")
    backbone_params = []
    adapter_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'head' in name:
            head_params.append(param)
        elif 'adapter' in name or 'temporal_attention' in name:
            adapter_params.append(param)
        else:
            backbone_params.append(param)
    
    base_lr = config['base_lr']
    head_lr = config['head_lr']
    
    logger.info(f"Learning rate:")
    logger.info(f"  Base LR: {base_lr:.6f}")
    logger.info(f"  Head LR: {head_lr:.6f}")
    
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": base_lr})
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": head_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config['weight_decay'])
    cosine_start_epoch = config.get('cosine_start_epoch', config['warmup_epochs'] + 5)
    
    # Resume from checkpoint if specified
    resume_epoch = 0
    resume_history = None
    if config.get('resume'):
        checkpoint = load_checkpoint(Path(config['resume']), device, logger)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        resume_epoch = checkpoint.get('epoch', 0)
        resume_history = checkpoint.get('history')
        logger.info(f"Resumed from epoch {resume_epoch}")
    
    # Create scheduler
    if config.get('use_adaptive_lr', False):
        logger.info("Using Adaptive LR Scheduler")
        scheduler = get_adaptive_lr_scheduler(
            optimizer=optimizer,
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
        scheduler = get_lr_scheduler(optimizer, config['epochs'], config['warmup_epochs'], cosine_start_epoch)
    
    if config.get('resume') and checkpoint.get('scheduler') and scheduler:
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("  ✓ Scheduler state loaded from checkpoint")
    
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    
    logger.info(f"Optimizer: AdamW")
    logger.info(f"  Backbone LR: {base_lr:.6f} (params: {len(backbone_params)})")
    logger.info(f"  Adapter/Head LR: {head_lr:.6f} (params: {len(adapter_params) + len(head_params)})")
    
    # Training paths
    checkpoint_path = config['output_dir'] / f'swin_model_{config["model_id"]}_best.pt'
    history_plot_path = config['output_dir'] / f'swin_model_{config["model_id"]}_training.png'
    
    # Train
    print("\n" + "="*80)
    print("🚀 Starting Training")
    print("="*80 + "\n")
    logger.info("="*60)
    logger.info("Starting Training")
    logger.info("="*60)
    
    train_model(
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
        resume_history=resume_history,
        teacher=teacher
    )
    
    print("\n" + "="*80)
    print("✅ Training completed!")
    print(f"Model saved to: {checkpoint_path}")
    print(f"Training plot saved to: {history_plot_path}")
    print("="*80)
    logger.info("="*60)
    logger.info("Training completed!")
    logger.info(f"Model saved to: {checkpoint_path}")
    logger.info(f"Training plot saved to: {history_plot_path}")
    logger.info("="*60)
    
    # Run inference
    print("\n🔍 Running inference...")
    logger.info("Running inference...")
    test_dataset = TestDataset(
        root=test_data_dir,
        num_frames=config['num_frames'],
        frame_stride=config['frame_stride'],
        image_size=config['img_size']
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=use_pin_memory
    )
    
    model.eval()
    predictions = run_inference(
        model=model,
        test_loader=test_loader,
        device=device,
        classes=train_dataset.classes,
        use_tta=True,
        num_crops=10,
        num_flips=2,
        expected_num_frames=config['num_frames']  # Truyền num_frames để tránh warning
    )
    
    submission_path = config['submissions_dir'] / f'submission_swin_model_{config["model_id"]}.csv'
    generate_submission(predictions, submission_path, logger)
    
    print(f"✅ Inference completed! Submission saved to: {submission_path}")
    print("="*80 + "\n")
    logger.info(
        f"Inference completed! Submission saved to: {submission_path}"
    )


if __name__ == '__main__':
    main()
