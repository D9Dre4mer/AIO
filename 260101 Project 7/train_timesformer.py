"""
Script riêng để train TimeSformer với config tối ưu cho phát hiện hành động con người.

TimeSformer sử dụng divided space-time attention, tốt hơn model Swin hiện tại vì:
- Có spatiotemporal modeling thực sự (không chỉ xử lý từng frame riêng)
- Divided space-time attention hiệu quả hơn simple temporal attention
- Tốt cho temporal reasoning (Something-Something V2: ~69.6% vs Video Swin)

Dựa trên:
- Best practices từ TimeSformer paper và official implementation
- Config hiện tại trong project (Model 3)
- Nghiên cứu về hyperparameters tối ưu cho action recognition

Key optimizations:
- Learning rate: Backbone 5e-5, Head 1e-3 (theo research cho TimeSformer)
- Batch size: 32-64 (tùy GPU memory)
- Num frames: 16 (tốt cho temporal modeling, không quá chậm)
- Dropout: 0.3-0.4 (TimeSformer cần regularization mạnh)
- Weight decay: 0.05-0.1 (theo research)
- Warmup: 3-5 epochs
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


def get_timesformer_optimized_config():
    """
    Tạo config tối ưu cho TimeSformer dựa trên research và best practices.
    
    Best practices từ TimeSformer paper:
    - Frames: 8-16 (standard), 16 tốt cho temporal modeling
    - Batch size: 64 global (8 per GPU × 8 GPUs), có thể điều chỉnh
    - Learning rate: 0.005 với SGD, hoặc 1e-4 đến 5e-4 với AdamW
    - Weight decay: 5e-5 (theo paper) hoặc 0.05-0.1 (theo config project)
    - Dropout: 0.1 (theo paper) hoặc 0.3-0.4 (theo config project)
    - Warmup: 1-5 epochs
    - Epochs: 15-30 (theo paper), hoặc 100 (theo config project)
    - Attention type: Divided space-time (divST) - đã có trong code
    
    Returns:
        dict: Config tối ưu cho TimeSformer
    """
    default_config = get_default_config()
    
    # Config tối ưu cho TimeSformer
    timesformer_config = default_config.copy()
    
    # Architecture
    timesformer_config['architecture'] = 'timesformer'
    timesformer_config['pretrained_name'] = 'vit_base_patch16_224'
    
    # Video parameters - TĂNG để đạt accuracy cao hơn
    # Nhiều frames và image size lớn hơn → tốt hơn cho accuracy
    timesformer_config['num_frames'] = 32  # Tăng từ 16 → 32 (tốt hơn cho temporal modeling)
    timesformer_config['frame_stride'] = 2
    timesformer_config['img_size'] = 256  # Tăng từ 224 → 256 (tốt hơn cho spatial features)
    
    # Training parameters - Tối ưu cho accuracy cao
    # Batch size nhỏ hơn do tăng frames và image size
    timesformer_config['batch_size'] = 8  # Giảm từ 16 → 8 (do tăng frames và image size)
    timesformer_config['epochs'] = 150  # Tăng từ 100 → 150 (train lâu hơn để đạt accuracy cao)
    timesformer_config['warmup_epochs'] = 10  # Tăng warmup cho training ổn định hơn
    timesformer_config['cosine_start_epoch'] = 10  # Cosine bắt đầu sau warmup
    
    # Learning rates - TĂNG LR để model học tốt hơn (LR hiện tại quá thấp → plateau)
    # Từ log: LR đã giảm xuống 0.000000 → model không học được thêm
    timesformer_config['base_lr'] = 5e-5  # Tăng từ 3e-5 → 5e-5 (để model học tốt hơn)
    timesformer_config['head_lr'] = 1e-3  # Tăng từ 8e-4 → 1e-3 (giữ ratio 20x)
    
    # Regularization - GIẢM để model học tốt hơn (hiện tại đang plateau)
    # Từ log: val acc plateau ở 60.88% → có thể regularization quá mạnh
    timesformer_config['weight_decay'] = 0.05  # Giảm từ 0.1 → 0.05 (để model học tốt hơn)
    timesformer_config['dropout'] = 0.3  # Giảm từ 0.4 → 0.3 (để model có capacity học)
    timesformer_config['drop_path_rate'] = 0.1  # Giảm từ 0.15 → 0.1 (cân bằng)
    timesformer_config['label_smoothing'] = 0.15  # Giảm từ 0.2 → 0.15 (để model học tốt hơn)
    
    # Augmentation - BẬT TẤT CẢ để tăng accuracy (quan trọng!)
    # Augmentation tạo diversity → model học features tốt hơn → accuracy cao hơn
    timesformer_config['mixup_alpha'] = 0.5  # BẬT Mixup (kết hợp với CutMix)
    timesformer_config['cutmix_alpha'] = 1.0  # BẬT CutMix
    timesformer_config['use_cutmix'] = True  # BẬT CutMix
    timesformer_config['use_temporal_aug'] = True  # BẬT temporal augmentation (quan trọng cho video!)
    timesformer_config['use_advanced_spatial'] = True  # BẬT RandAugment (tốt cho accuracy)
    timesformer_config['use_advanced_color'] = True  # BẬT color jitter (tốt cho accuracy)
    
    # Loss function
    timesformer_config['use_focal_loss'] = True  # Focal loss tốt cho class imbalance
    timesformer_config['focal_alpha'] = 0.25
    timesformer_config['focal_gamma'] = 2.0
    
    # Advanced features - BẬT các features tốt cho accuracy
    timesformer_config['use_ema'] = False  # Tắt EMA (user không muốn dùng)
    timesformer_config['use_progressive_resize'] = True  # BẬT progressive resize - tốt cho accuracy
    timesformer_config['use_synthetic_data'] = True  # BẬT synthetic data - tăng data diversity
    timesformer_config['synthetic_method'] = 'temporal_mixup'  # Temporal mixup tốt cho video
    timesformer_config['synthetic_ratio'] = 0.3  # 30% synthetic data
    timesformer_config['use_pseudo_labeling'] = False  # Giữ tắt (có thể gây nhiễu)
    
    # Adaptive LR scheduler - Điều chỉnh để tránh LR quá thấp
    timesformer_config['use_adaptive_lr'] = True
    timesformer_config['lr_plateau_patience'] = 7  # Giảm từ 10 → 7 (giảm LR nhanh hơn khi plateau)
    timesformer_config['lr_min_delta'] = 0.002  # Tăng từ 0.001 → 0.002 (ít nhạy hơn, tránh giảm LR quá sớm)
    timesformer_config['lr_threshold_mode'] = 'rel'
    timesformer_config['lr_cooldown'] = 3  # Giảm từ 5 → 3 (cooldown ngắn hơn)
    timesformer_config['use_val_loss_for_lr'] = False  # Dùng val acc thay vì val loss (tốt hơn cho accuracy)
    timesformer_config['min_lr_ratio'] = 0.01  # Tăng từ 0.001 → 0.01 (LR không giảm quá thấp)
    
    # Early stopping - Tăng patience để train lâu hơn
    timesformer_config['early_stop_patience'] = 20  # Tăng từ 8 → 20 (train lâu hơn để đạt accuracy cao)
    
    # Knowledge distillation: Not used (disabled)
    timesformer_config['use_distillation'] = False
    timesformer_config['teacher_checkpoints'] = []
    
    # Data
    timesformer_config['val_ratio'] = 0.20
    # Tối ưu num_workers: thử 2-4 workers để giảm CPU bottleneck
    timesformer_config['num_workers'] = 2  # Tăng từ 0 → 2 để giảm CPU bottleneck
    
    # Output
    timesformer_config['output_dir'] = Path('./checkpoints')
    timesformer_config['logging_dir'] = Path('./logging')
    timesformer_config['submissions_dir'] = Path('./submissions')
    
    return timesformer_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train TimeSformer với config tối ưu cho action recognition',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--model-id', type=int, default=3, help='Model ID (default: 3)')
    parser.add_argument('--seed', type=int, default=456, help='Random seed (default: 456 cho Model 3)')
    
    # Data configuration
    parser.add_argument('--data-dir', type=str, default='./kaggle_data/data', help='Path to dataset directory')
    parser.add_argument('--num-frames', type=int, default=None, help='Number of frames (default: 16)')
    parser.add_argument('--frame-stride', type=int, default=2, help='Frame stride')
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (default: 32, có thể tăng lên 64 nếu GPU đủ)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (default: 100)')
    parser.add_argument('--base-lr', type=float, default=None, help='Base learning rate (default: 5e-5)')
    parser.add_argument('--head-lr', type=float, default=None, help='Head learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=None, help='Weight decay (default: 0.1)')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate (default: 0.4)')
    
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
    config = get_timesformer_optimized_config()
    
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
    if args.dropout is not None:
        config['dropout'] = args.dropout
    
    config['output_dir'] = Path(args.output_dir)
    config['logging_dir'] = Path(args.logging_dir)
    config['submissions_dir'] = Path(args.submissions_dir)
    config['resume'] = args.resume
    config['inference_only'] = args.inference_only
    config['checkpoint'] = args.checkpoint
    config['num_workers'] = args.num_workers if args.num_workers > 0 else config['num_workers']
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
    print("🚀 TimeSformer Training - Optimized Config")
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
    print(f"  Dropout: {config['dropout']}")
    print(f"  Warmup epochs: {config['warmup_epochs']}")
    print(f"  Total epochs: {config['epochs']}")
    print("")
    print("TimeSformer advantages:")
    print("  ✓ Divided space-time attention (better than simple temporal attention)")
    print("  ✓ True spatiotemporal modeling (not frame-by-frame)")
    print("  ✓ Better temporal reasoning than current Swin implementation")
    print("")
    print("🎯 Config tối ưu để đạt VAL ACCURACY >90%:")
    print("  ✓ Frames: 32 (tăng từ 16)")
    print("  ✓ Image size: 256 (tăng từ 224)")
    print("  ✓ Epochs: 150 (tăng từ 100)")
    print("  ✓ Augmentation: TẤT CẢ BẬT (CutMix, Mixup, Temporal, Spatial, Color)")
    print("  ✓ EMA: TẮT (user không muốn dùng)")
    print("  ✓ Progressive resize: BẬT")
    print("  ✓ Synthetic data: BẬT (temporal mixup)")
    print("  ✓ Regularization: GIẢM (dropout=0.3, weight_decay=0.05) - để model học tốt hơn")
    print("  ✓ Learning rate: TĂNG (base_lr=5e-5) - tránh LR quá thấp → plateau")
    print("="*80 + "\n")
    
    # Also log to file
    logger.info("="*60)
    logger.info("TimeSformer Training - Optimized Config")
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
    logger.info(f"  Dropout: {config['dropout']}")
    logger.info(f"  Warmup epochs: {config['warmup_epochs']}")
    logger.info(f"  Total epochs: {config['epochs']}")
    logger.info("")
    logger.info("TimeSformer advantages:")
    logger.info("  ✓ Divided space-time attention (better than simple temporal attention)")
    logger.info("  ✓ True spatiotemporal modeling (not frame-by-frame)")
    logger.info("  ✓ Better temporal reasoning than current Swin implementation")
    logger.info("")
    logger.info("🎯 Config tối ưu để đạt VAL ACCURACY >90%:")
    logger.info("  ✓ Frames: 32 (tăng từ 16) - tốt hơn cho temporal modeling")
    logger.info("  ✓ Image size: 256 (tăng từ 224) - tốt hơn cho spatial features")
    logger.info("  ✓ Epochs: 150 (tăng từ 100) - train lâu hơn")
    logger.info("  ✓ Augmentation: TẤT CẢ BẬT (CutMix, Mixup, Temporal, Spatial, Color)")
    logger.info("  ✓ EMA: TẮT (user không muốn dùng)")
    logger.info("  ✓ Progressive resize: BẬT (tốt cho accuracy)")
    logger.info("  ✓ Synthetic data: BẬT (temporal mixup - tăng diversity)")
    logger.info("  ✓ Regularization: GIẢM (dropout=0.3, weight_decay=0.05) - để model học tốt hơn")
    logger.info("  ✓ Learning rate: TĂNG (base_lr=5e-5) - tránh LR quá thấp → plateau")
    logger.info("  ✓ Early stopping patience=20 (train lâu hơn)")
    logger.info("")
    logger.info("⚠️  Lưu ý từ log trước:")
    logger.info("  - Best val acc: 60.88% (epoch 73)")
    logger.info("  - Val acc plateau 20 epochs → LR quá thấp (0.000000)")
    logger.info("  - Đã điều chỉnh: Tăng LR, giảm regularization")
    logger.info("")
    
    # Inference only mode
    if config.get('inference_only', False):
        checkpoint_path = Path(
            config.get('checkpoint', config['output_dir'] / f'timesformer_model_{config["model_id"]}_best.pt')
        )
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
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
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Classes: {len(train_dataset.classes)}")
    
    # Create model
    print("🔧 Creating TimeSformer model...")
    logger.info("Creating TimeSformer model...")
    model = create_model(
        architecture=config['architecture'],
        num_classes=len(train_dataset.classes),
        pretrained_name=config['pretrained_name'],
        use_adapters=config['use_adapters'],
        dropout=config.get('dropout', 0.4)
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
        elif 'adapter' in name or 'temporal_attention' in name or 'space_time_attention' in name:
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
            plateau_patience=config.get('lr_plateau_patience', 5),
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
    checkpoint_path = config['output_dir'] / f'timesformer_model_{config["model_id"]}_best.pt'
    history_plot_path = config['output_dir'] / f'timesformer_model_{config["model_id"]}_training.png'
    
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
    
    submission_path = config['submissions_dir'] / f'submission_timesformer_model_{config["model_id"]}.csv'
    generate_submission(predictions, submission_path, logger)
    
    print(f"✅ Inference completed! Submission saved to: {submission_path}")
    print("="*80 + "\n")
    logger.info(
        f"Inference completed! Submission saved to: {submission_path}"
    )


if __name__ == '__main__':
    main()
