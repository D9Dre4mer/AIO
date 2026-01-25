"""
Main entry point for SOTA Training Pipeline.
"""

import sys
import os

# Fix OpenMP duplicate library error on Windows
# This happens when multiple OpenMP runtimes are linked (e.g., from different packages)
# Setting this before importing numpy/torch to avoid the error
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import numpy as np
import torch
from pathlib import Path
import logging
import warnings

# GPU optimization settings
# CUDA_LAUNCH_BLOCKING: '1' = synchronous (safer, slower), '0' = asynchronous (faster)
# For RTX 5090, start with '0' for better performance, fallback to '1' if issues
cuda_launch_blocking = os.environ.get('CUDA_LAUNCH_BLOCKING', '0')
os.environ['CUDA_LAUNCH_BLOCKING'] = cuda_launch_blocking
# cuDNN benchmark: Will be enabled/disabled based on config
torch.backends.cudnn.benchmark = False  # Default: False for RTX 5090 compatibility

from .config import parse_args, get_hyperparameter_variations
from .utils import setup_logging, log_system_info, load_checkpoint
from .dataset import VideoDataset, TestDataset
from .models import SOTAViTForAction, EMAModel, create_model
from .training import train_model, get_lr_scheduler
from .adaptive_lr_scheduler import get_adaptive_lr_scheduler
from .knowledge_distillation import EnsembleTeacher
from .inference import run_inference, generate_submission, run_ensemble_inference

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_and_auto_set_pretrained_name(
    pretrained_name: str,
    architecture: str,
    default_pretrained_names: dict,
    logger: logging.Logger = None,
    is_from_default_config: bool = False
) -> str:
    """
    Validate and auto-set pretrained_name based on architecture.
    
    Args:
        pretrained_name: Current pretrained_name (can be None)
        architecture: Model architecture (e.g., 'vit_large', 'vit_base')
        default_pretrained_names: Dict mapping architecture to default pretrained_name
        logger: Optional logger for warnings/info
        is_from_default_config: If True, pretrained_name is from default config (not explicitly set).
                                In this case, don't log warning when auto-setting.
    
    Returns:
        Validated pretrained_name (None if should use default from create_model)
    """
    # Check if pretrained_name needs to be auto-set
    needs_auto_set = pretrained_name is None
    if pretrained_name is not None:
        # Check if pretrained_name matches architecture by comparing with default
        # If architecture has a default pretrained_name, check if current one matches
        if architecture in default_pretrained_names:
            expected_pretrained_name = default_pretrained_names[architecture]
            # If current pretrained_name doesn't match expected, auto-set
            if pretrained_name != expected_pretrained_name:
                needs_auto_set = True
    
    # Auto-set from default if needed
    if needs_auto_set:
        if architecture in default_pretrained_names:
            # Only log warning if pretrained_name was explicitly set (not from default config)
            if pretrained_name is not None and logger and not is_from_default_config:
                logger.info(f"⚠️  pretrained_name '{pretrained_name}' doesn't match architecture '{architecture}'. Auto-setting to '{default_pretrained_names[architecture]}'")
            return default_pretrained_names[architecture]
        else:
            if logger:
                logger.warning(f"⚠️  No default pretrained_name for architecture '{architecture}'. Using None (create_model will handle)")
            return None
    
    return pretrained_name


def setup_distillation_teacher(config: dict, train_dataset: VideoDataset, device: torch.device, logger: logging.Logger):
    """
    Setup knowledge distillation teacher from checkpoints.
    
    Returns:
        teacher: EnsembleTeacher instance or None if setup failed
        config: Updated config (use_distillation may be disabled if setup fails)
    """
    teacher = None
    if config.get('use_distillation', False):
        teacher_checkpoints = config.get('teacher_checkpoints', [])
        if not teacher_checkpoints:
            logger.warning("⚠️  use_distillation=True but teacher_checkpoints is empty. Disabling distillation.")
            config['use_distillation'] = False
        else:
            logger.info("="*60)
            logger.info("Setting up Knowledge Distillation")
            logger.info("="*60)
            logger.info(f"Teacher checkpoints: {teacher_checkpoints}")
            logger.info(f"Temperature: {config.get('distillation_temperature', 3.0)}")
            logger.info(f"Alpha: {config.get('distillation_alpha', 0.7)}")
            
            # Load teacher models
            try:
                # Validate all teacher checkpoints exist before loading
                missing_checkpoints = []
                for cp_path in teacher_checkpoints:
                    cp = Path(cp_path)
                    if not cp.exists():
                        missing_checkpoints.append(str(cp))
                
                if missing_checkpoints:
                    logger.error(f"❌ Teacher checkpoint(s) not found:")
                    for cp in missing_checkpoints:
                        logger.error(f"   - {cp}")
                    logger.error("   Disabling distillation. Please train teacher model(s) first.")
                    logger.error("   Expected training order: Model 2 → Model 1 → Model 3 → Model 4 → Model 5")
                    config['use_distillation'] = False
                else:
                    # Determine teacher architecture from first checkpoint
                    first_checkpoint = Path(teacher_checkpoints[0])
                    try:
                        checkpoint_data = torch.load(first_checkpoint, map_location=device, weights_only=False)
                        checkpoint_config = checkpoint_data.get('config', {})
                        teacher_architecture = checkpoint_config.get('architecture', 'vit_base')
                        teacher_pretrained_name = checkpoint_config.get('pretrained_name', None)
                        
                        # Detect actual architecture from checkpoint state_dict if available
                        # This is more reliable than config which may be incorrect
                        actual_architecture = None
                        try:
                            checkpoint_temp = torch.load(first_checkpoint, map_location='cpu', weights_only=False)
                            state_dict = checkpoint_temp.get('model') or checkpoint_temp.get('model_state_dict') or checkpoint_temp
                            if 'vit.cls_token' in state_dict:
                                embed_dim = state_dict['vit.cls_token'].shape[-1]
                                # Count unique blocks by extracting block numbers from keys
                                block_keys = [k for k in state_dict.keys() if 'vit.blocks.' in k and '.norm1.weight' in k]
                                unique_blocks = set()
                                for k in block_keys:
                                    parts = k.split('vit.blocks.')
                                    if len(parts) > 1:
                                        block_num = parts[1].split('.')[0]
                                        try:
                                            unique_blocks.add(int(block_num))
                                        except ValueError:
                                            pass
                                num_blocks = len(unique_blocks)
                                
                                if embed_dim == 768 and num_blocks == 12:
                                    actual_architecture = 'vit_base'
                                elif embed_dim == 1024 and num_blocks == 24:
                                    actual_architecture = 'vit_large'
                                
                                if actual_architecture and actual_architecture != teacher_architecture:
                                    logger.warning(f"⚠️  Checkpoint config says '{teacher_architecture}' but actual model is '{actual_architecture}'. Using actual architecture.")
                                    teacher_architecture = actual_architecture
                                    # Validate and auto-set pretrained_name for teacher
                                    default_pretrained_names = checkpoint_config.get('default_pretrained_names', {})
                                    teacher_pretrained_name = validate_and_auto_set_pretrained_name(
                                        teacher_pretrained_name,
                                        actual_architecture,
                                        default_pretrained_names,
                                        logger
                                    )
                        except Exception as e:
                            logger.warning(f"⚠️  Could not detect actual architecture from checkpoint: {e}. Using config architecture.")
                        
                        # Fallback: If actual_architecture is None, validate against teacher_architecture
                        if teacher_pretrained_name is not None and actual_architecture is None:
                            default_pretrained_names = checkpoint_config.get('default_pretrained_names', {})
                            teacher_pretrained_name = validate_and_auto_set_pretrained_name(
                                teacher_pretrained_name,
                                teacher_architecture,
                                default_pretrained_names,
                                logger
                            )
                        
                        # Extract teacher model ID from checkpoint path for logging
                        teacher_id_str = first_checkpoint.stem.replace('sota_vit_model_', '').replace('_best', '')
                        try:
                            teacher_id = int(teacher_id_str)
                            logger.info(f"Teacher: Model {teacher_id} ({teacher_architecture})")
                        except ValueError:
                            logger.info(f"Teacher architecture: {teacher_architecture}")
                        
                        # Load teacher models directly
                        teacher_models = []
                        for checkpoint_path in teacher_checkpoints:
                            checkpoint_path_obj = Path(checkpoint_path)
                            
                            # Create teacher model with same architecture
                            # create_model() will use default from config if pretrained_name is None
                            teacher_model = create_model(
                                architecture=teacher_architecture,
                                num_classes=len(train_dataset.classes),
                                pretrained_name=teacher_pretrained_name,  # Use from checkpoint config (or None if mismatch)
                                use_adapters=checkpoint_config.get('use_adapters', True),
                                dropout=checkpoint_config.get('dropout', 0.1),
                                scales=checkpoint_config.get('multiscale_sizes', None) if teacher_architecture == 'multiscale' else None
                            )
                            
                            # Load checkpoint (load_state_dict will raise error if size mismatch)
                            checkpoint_teacher = torch.load(checkpoint_path_obj, map_location=device, weights_only=False)
                            if 'model' in checkpoint_teacher:
                                teacher_model.load_state_dict(checkpoint_teacher['model'])
                            elif 'model_state_dict' in checkpoint_teacher:
                                teacher_model.load_state_dict(checkpoint_teacher['model_state_dict'])
                            else:
                                teacher_model.load_state_dict(checkpoint_teacher)
                            
                            teacher_model.to(device)
                            teacher_model.eval()
                            teacher_models.append(teacher_model)
                            
                            # Extract model ID for logging
                            cp_id_str = checkpoint_path_obj.stem.replace('sota_vit_model_', '').replace('_best', '')
                            try:
                                cp_id = int(cp_id_str)
                                logger.info(f"  ✓ Loaded teacher Model {cp_id} from: {checkpoint_path_obj.name}")
                            except ValueError:
                                logger.info(f"  ✓ Loaded teacher from: {checkpoint_path_obj.name}")
                        
                        if teacher_models:
                            teacher = EnsembleTeacher(teacher_models)
                            logger.info(f"✅ Teacher ensemble loaded successfully ({len(teacher_models)} model(s))")
                        else:
                            logger.error("❌ No valid teacher models loaded. Disabling distillation.")
                            config['use_distillation'] = False
                            teacher = None
                    except Exception as e:
                        logger.error(f"❌ Failed to load teacher checkpoint: {e}")
                        logger.error("   Disabling distillation.")
                        config['use_distillation'] = False
                        teacher = None
            except Exception as e:
                logger.error(f"❌ Failed to load teacher model: {e}")
                logger.error("   Disabling distillation.")
                config['use_distillation'] = False
                teacher = None
    
    return teacher, config


def train_single_model(config: dict, model_id: int, seed: int):
    """Train a single model."""
    logger.info("="*60)
    logger.info(f"Training Model {model_id}")
    logger.info("="*60)
    
    # Set random seed
    set_random_seeds(seed)
    logger.info(f"Random seed: {seed}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Log system info (will be updated with memory estimation after model creation)
    log_system_info(logger, device)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_data_dir = config['data_dir'] / 'data_train'
    test_data_dir = config['data_dir'] / 'test'
    
    # Use advanced augmentation if enabled
    use_temporal_aug = config.get('use_temporal_aug', True)
    use_advanced_spatial = config.get('use_advanced_spatial', True)
    use_advanced_color = config.get('use_advanced_color', True)
    
    train_dataset = VideoDataset(
        root=train_data_dir,
        num_frames=config['num_frames'],
        frame_stride=config['frame_stride'],
        image_size=config['img_size'],
        is_train=True,
        val_ratio=config['val_ratio'],
        seed=seed,
        use_temporal_aug=use_temporal_aug,
        use_advanced_spatial=use_advanced_spatial,
        use_advanced_color=use_advanced_color
    )
    
    val_dataset = VideoDataset(
        root=train_data_dir,
        num_frames=config['num_frames'],
        frame_stride=config['frame_stride'],
        image_size=config['img_size'],
        is_train=False,
        val_ratio=config['val_ratio'],
        seed=seed
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Classes: {len(train_dataset.classes)}")
    
    # Create model
    logger.info("Creating model...")
    architecture = config.get('architecture', 'vit_base')
    logger.info(f"Architecture: {architecture}")
    
    # Auto-set pretrained_name based on architecture if not set or doesn't match
    pretrained_name = config.get('pretrained_name', None)
    default_pretrained_names = config.get('default_pretrained_names', {})
    # Check if pretrained_name is from default config (not explicitly set in variation)
    # If pretrained_name doesn't match current architecture but matches another architecture's default,
    # it's likely from default config (which has 'vit_base_patch16_224')
    is_from_default_config = False
    if pretrained_name is not None and architecture in default_pretrained_names:
        expected_pretrained_name = default_pretrained_names[architecture]
        if pretrained_name != expected_pretrained_name:
            # Check if pretrained_name matches any other architecture's default
            # If yes, it's likely from default config, not explicitly set
            for other_arch, other_default in default_pretrained_names.items():
                if other_arch != architecture and pretrained_name == other_default:
                    is_from_default_config = True
                    break
    pretrained_name = validate_and_auto_set_pretrained_name(
        pretrained_name,
        architecture,
        default_pretrained_names,
        logger,
        is_from_default_config=is_from_default_config
    )
    config['pretrained_name'] = pretrained_name  # Update config for consistency
    
    # Get multiscale sizes if needed
    scales = None
    if architecture == 'multiscale':
        scales = config.get('multiscale_sizes', [224, 256, 288])
    
    model = create_model(
        architecture=architecture,
        num_classes=len(train_dataset.classes),
        pretrained_name=pretrained_name,
        use_adapters=config['use_adapters'],
        dropout=config.get('dropout', 0.1),
        drop_path_rate=config.get('drop_path_rate', 0.0),  # Stochastic Depth (DropPath)
        scales=scales
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Setup knowledge distillation teacher if enabled
    teacher, config = setup_distillation_teacher(config, train_dataset, device, logger)
    
    # Try to enable cuDNN benchmark for better performance (if enabled and compatible)
    if device.type == 'cuda' and config.get('enable_cudnn_benchmark', False):
        try:
            torch.backends.cudnn.benchmark = True
            logger.info("✓ cuDNN benchmark enabled (may improve training speed)")
        except Exception as e:
            logger.warning(f"⚠️  Could not enable cuDNN benchmark: {e}")
            torch.backends.cudnn.benchmark = False
    
    # Compile model with torch.compile for faster training (if enabled)
    if config.get('use_torch_compile', True) and device.type == 'cuda':
        try:
            # torch.compile requires PyTorch 2.0+ and Triton
            if hasattr(torch, 'compile'):
                # Check if Triton is available by trying to import it
                try:
                    import triton
                    logger.info("Compiling model with torch.compile for faster training...")
                    model = torch.compile(model, mode='reduce-overhead')  # Optimize for training
                    logger.info("✓ Model compiled successfully")
                except ImportError:
                    logger.warning("⚠️  Triton not available - torch.compile requires Triton")
                    logger.warning("   Continuing without compilation (install triton for faster training)")
                    config['use_torch_compile'] = False  # Disable to avoid errors later
            else:
                logger.warning("⚠️  torch.compile not available (requires PyTorch 2.0+)")
                config['use_torch_compile'] = False
        except Exception as e:
            # Catch any other errors (including TritonMissing at runtime)
            error_msg = str(e)
            if 'Triton' in error_msg or 'triton' in error_msg.lower():
                logger.warning("⚠️  Triton error detected - disabling torch.compile")
                logger.warning("   Continuing without compilation (install triton for faster training)")
                config['use_torch_compile'] = False
            else:
                logger.warning(f"⚠️  Could not compile model: {e}")
                logger.warning("   Continuing without compilation")
                config['use_torch_compile'] = False
    
    # Adjust batch size for distillation and architecture-specific requirements
    original_batch_size = config['batch_size']
    architecture = config.get('architecture', 'vit_base')
    default_batch_size = 24
    batch_size_explicitly_set = (original_batch_size != default_batch_size)
    
    # First: Reduce batch size to 12 when distillation is enabled (for ViT-Large teacher compatibility)
    # Only reduce if using default batch size (24), not if user explicitly set a different value
    if config.get('use_distillation', False) and teacher is not None:
        if not batch_size_explicitly_set:
            # Default batch size (24) → reduce to 12 for distillation
            config['batch_size'] = 12
            logger.info(f"⚠️  Distillation enabled: Reducing batch size from {original_batch_size} → 12 to save VRAM")
            original_batch_size = 12  # Update for subsequent checks
        else:
            # User explicitly set batch size → keep it as is (don't auto-reduce)
            logger.info(f"✓  Distillation enabled: Using batch size {original_batch_size} (explicitly set, not auto-reduced)")
    
    # Second: Additional adjustment for TimeSformer (space-time attention is memory-intensive)
    if architecture == 'timesformer':
        current_batch_size = config['batch_size']
        
        if batch_size_explicitly_set and current_batch_size <= 20:
            # Batch size was explicitly set and already <= 20, keep it as is
            logger.info(f"✓  TimeSformer: Using batch size {current_batch_size} (explicitly set)")
        elif not config.get('use_distillation', False) or teacher is None:
            # TimeSformer without distillation: reduce batch size (24 → 16, or 20 → 14 if distillation was applied)
            timesformer_batch_size = max(12, int(current_batch_size * 0.67))  # 33% reduction
            if timesformer_batch_size != current_batch_size:
                config['batch_size'] = timesformer_batch_size
                logger.info(f"⚠️  TimeSformer architecture: Reducing batch size from {current_batch_size} → {timesformer_batch_size} to save VRAM")
                logger.info(f"   (Space-time attention is memory-intensive)")
        # If distillation is enabled, batch size is already reduced to 20, no further reduction needed
    
    # Optimize num_workers: use 0 for Windows (multiprocessing is very slow on Windows)
    # However, with large batch sizes, num_workers=2-4 may help reduce CPU bottleneck
    # For Linux/Mac, auto-detect optimal number
    num_workers = config['num_workers']
    if num_workers == 0 and os.name != 'nt':  # Not Windows
        import multiprocessing
        num_workers = min(4, multiprocessing.cpu_count())
        logger.info(f"Auto-detected num_workers: {num_workers}")
    elif num_workers == 0 and os.name == 'nt':
        # On Windows, always use num_workers=0 to avoid shared memory errors (error 1455)
        # Even num_workers=1 can cause RuntimeError: Couldn't open shared file mapping on Windows
        # The performance trade-off is acceptable compared to training crashes
        logger.info("Windows: Using num_workers=0 to avoid shared memory errors (error 1455). Multiprocessing on Windows is unstable with PyTorch DataLoader.")
    
    # Windows-specific settings for num_workers > 0
    # To use num_workers > 0 on Windows without errors, we need:
    # 1. persistent_workers=False (critical! prevents shared memory accumulation)
    # 2. Lower prefetch_factor (reduces memory pressure)
    # 3. multiprocessing_context='spawn' (explicit, default on Windows but safer to set)
    # 4. Ensure if __name__ == '__main__': guard (already in train.py)
    is_windows = os.name == 'nt'
    if num_workers > 0 and is_windows:
        # On Windows, disable persistent_workers to avoid shared memory errors
        # This means workers are recreated each epoch, but prevents error 1455
        use_persistent_workers = False
        # Use lower prefetch_factor on Windows to reduce memory pressure
        prefetch_factor = 2  # Lower than Linux/Mac to reduce memory usage
        # Use 'spawn' context explicitly (default on Windows but safer to set)
        import multiprocessing
        multiprocessing_context = multiprocessing.get_context('spawn')
        logger.info(f"Windows detected: Using num_workers={num_workers} with persistent_workers=False, prefetch_factor={prefetch_factor}, and spawn context to avoid shared memory errors")
    elif num_workers > 0:
        use_persistent_workers = True
        prefetch_factor = 4 if config.get('batch_size', 12) >= 64 else 2
        multiprocessing_context = None  # Use default (fork on Linux/Mac)
    else:
        use_persistent_workers = False
        prefetch_factor = None
        multiprocessing_context = None
    # Disable pin_memory for very large batch sizes to avoid OOM
    # pin_memory requires extra memory and can cause OOM with large batches
    use_pin_memory = torch.cuda.is_available() and config['batch_size'] <= 128
    
    # Build DataLoader kwargs
    dataloader_kwargs = {
        'batch_size': config['batch_size'],
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': use_pin_memory,
        'persistent_workers': use_persistent_workers,
        'prefetch_factor': prefetch_factor,
        'drop_last': False
    }
    # Add multiprocessing_context only on Windows with num_workers > 0
    if multiprocessing_context is not None:
        dataloader_kwargs['multiprocessing_context'] = multiprocessing_context
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        **dataloader_kwargs
    )
    
    # Val loader (same settings but no shuffle)
    val_dataloader_kwargs = dataloader_kwargs.copy()
    val_dataloader_kwargs['shuffle'] = False
    val_dataloader_kwargs['prefetch_factor'] = prefetch_factor if num_workers > 0 else None
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        **val_dataloader_kwargs
    )
    
    # Memory estimation with actual model parameters
    logger.info("")
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
    
    # Use learning rate directly from config (no batch size scaling)
    base_lr = config['base_lr']
    head_lr = config['head_lr']
    
    logger.info(f"Learning rate (fixed, no batch size scaling):")
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
    is_resume = False
    checkpoint_original_epochs = None
    
    if config.get('resume'):
        checkpoint = load_checkpoint(Path(config['resume']), device, logger)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        resume_epoch = checkpoint.get('epoch', 0)
        resume_history = checkpoint.get('history')
        is_resume = True
        
        # Get config from checkpoint and merge into current config
        # Priority: checkpoint config > current config (except for epochs if explicitly set)
        checkpoint_config = checkpoint.get('config', {})
        checkpoint_original_epochs = checkpoint_config.get('epochs')
        
        # Save current epochs if explicitly set from command line
        new_epochs = config.get('epochs')
        epochs_explicitly_set = config.get('_epochs_explicitly_set', False)
        
        # Merge checkpoint config into current config
        # Checkpoint config takes priority for training hyperparameters
        logger.info("="*60)
        logger.info("Merging config from checkpoint...")
        logger.info("="*60)
        
        # List of config keys to merge from checkpoint (training hyperparameters)
        # Note: early_stop_patience is NOT merged - use current config value to allow changes
        keys_to_merge = [
            'base_lr', 'head_lr', 'weight_decay', 'batch_size', 'grad_accum_steps',
            'label_smoothing', 'dropout', 'mixup_alpha', 'cutmix_alpha',
            'warmup_epochs', 'cosine_start_epoch',
            'use_adaptive_lr', 'lr_plateau_patience', 'lr_min_delta', 'lr_threshold_mode',
            'lr_cooldown', 'use_val_loss_for_lr', 'min_lr_ratio',
            'use_distillation', 'distillation_temperature', 'distillation_alpha', 'teacher_checkpoints',
            'use_progressive_resize', 'progressive_sizes', 'progressive_resize_patience',
            'progressive_resize_min_improvement', 'progressive_resize_force_final_epochs',
            'progressive_resize_min_epochs_per_stage',
            'use_cutmix', 'use_focal_loss', 'focal_alpha', 'focal_gamma',
            'use_ema', 'ema_decay',
            'num_frames', 'frame_stride', 'img_size',
            'architecture', 'pretrained_name', 'use_adapters', 'multiscale_sizes'
        ]
        
        merged_count = 0
        for key in keys_to_merge:
            if key in checkpoint_config:
                old_value = config.get(key)
                new_value = checkpoint_config[key]
                if old_value != new_value:
                    config[key] = new_value
                    logger.info(f"  ✓ {key}: {old_value} → {new_value}")
                    merged_count += 1
        
        # Restore epochs if explicitly set from command line
        if epochs_explicitly_set:
            config['epochs'] = new_epochs
            logger.info(f"  ⚠️  epochs: {checkpoint_original_epochs} → {new_epochs} (overridden by command line)")
        elif checkpoint_original_epochs:
            config['epochs'] = checkpoint_original_epochs
            logger.info(f"  ✓ epochs: {checkpoint_original_epochs} (from checkpoint)")
        
        if merged_count == 0:
            logger.info("  (No config changes needed - already matches checkpoint)")
        logger.info("="*60)
        
        logger.info(f"Resumed from epoch {resume_epoch}")
        if checkpoint_original_epochs and checkpoint_original_epochs != config['epochs']:
            logger.info(f"  Original training: {checkpoint_original_epochs} epochs")
            logger.info(f"  Continuing to: {config['epochs']} epochs")
            logger.info(f"  ⚠️  Scheduler will be recreated to match new epoch count")
        
        # Setup teacher again if use_distillation was restored from checkpoint and teacher is None
        # This handles the case where teacher failed to load initially but config was restored from checkpoint
        if config.get('use_distillation', False) and teacher is None:
            logger.info("="*60)
            logger.info("Resume: Setting up teacher for distillation (restored from checkpoint)")
            logger.info("="*60)
            teacher, config = setup_distillation_teacher(config, train_dataset, device, logger)
            if teacher is None:
                logger.warning("⚠️  Failed to setup teacher after resume. Distillation will be disabled.")
            else:
                logger.info("✅ Teacher setup completed after resume")
    
    # Create scheduler (adaptive or standard)
    if config.get('use_adaptive_lr', False):
        logger.info("Using Adaptive LR Scheduler (chỉ giảm LR khi model không cải thiện)")
        logger.info(f"  Plateau patience: {config.get('lr_plateau_patience', 3)} epochs")
        logger.info(f"  Min delta: {config.get('lr_min_delta', 0.0001)} ({'relative' if config.get('lr_threshold_mode', 'rel') == 'rel' else 'absolute'})")
        logger.info(f"  Cooldown: {config.get('lr_cooldown', 7)} epochs")
        logger.info(f"  Use val loss: {config.get('use_val_loss_for_lr', False)}")
        scheduler = get_adaptive_lr_scheduler(
            optimizer=optimizer,
            num_epochs=config['epochs'],
            warmup_epochs=config['warmup_epochs'],
            min_lr_ratio=config.get('min_lr_ratio', 0.01),
            plateau_patience=config.get('lr_plateau_patience', 3),
            min_delta=config.get('lr_min_delta', 0.0001),
            mode='max',  # 'max' for accuracy
            verbose=True,
            cooldown=config.get('lr_cooldown', 7),
            threshold_mode=config.get('lr_threshold_mode', 'rel'),
            use_val_loss=config.get('use_val_loss_for_lr', False)
        )
    else:
        scheduler = get_lr_scheduler(optimizer, config['epochs'], config['warmup_epochs'], cosine_start_epoch)
    
    # If resuming with different num_epochs, need to handle scheduler carefully
    if is_resume and checkpoint_original_epochs and checkpoint_original_epochs != config['epochs']:
        # Don't load old scheduler state - it was created with different num_epochs
        # Calculate relative progress to maintain similar LR position in new schedule
        logger.info(f"  Adjusting scheduler for new epoch count...")
        
        # Get current LR from optimizer (loaded from checkpoint) - this is the actual LR model was trained with
        current_lr_backbone = optimizer.param_groups[0]['lr']
        current_lr_head = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else None
        
        # Calculate relative progress in old schedule
        # For cosine annealing: progress = (epoch - warmup) / (total_epochs - warmup)
        warmup_epochs = config['warmup_epochs']
        if resume_epoch >= warmup_epochs:
            old_progress = (resume_epoch - warmup_epochs) / max(1, checkpoint_original_epochs - warmup_epochs)
            # Map to new schedule: find epoch in new schedule with similar progress
            new_epoch_for_similar_progress = int(warmup_epochs + old_progress * (config['epochs'] - warmup_epochs))
            # Step scheduler to that epoch to get similar LR
            steps_needed = min(new_epoch_for_similar_progress, resume_epoch)
        else:
            # Still in warmup phase
            steps_needed = resume_epoch
        
        # Step scheduler to get LR that matches progress in new schedule
        # Suppress warning about step() before optimizer.step() - we're just syncing state
        # Only step standard scheduler (adaptive scheduler needs metrics)
        if not config.get('use_adaptive_lr', False):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*")
                for _ in range(steps_needed):
                    scheduler.step()
        
        new_lr_backbone = optimizer.param_groups[0]['lr']
        new_lr_head = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else None
        
        # If LR changed too much, restore original LR to avoid instability
        lr_change_ratio = abs(new_lr_backbone - current_lr_backbone) / max(current_lr_backbone, 1e-8)
        if lr_change_ratio > 0.5:  # If LR changed more than 50%, restore original
            logger.warning(f"  ⚠️  LR change too large ({lr_change_ratio:.2%}), restoring original LR to avoid instability")
            optimizer.param_groups[0]['lr'] = current_lr_backbone
            if current_lr_head and len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = current_lr_head
            
            # Find epoch in new schedule that matches the restored LR
            # This ensures scheduler state matches the LR we restored
            base_lr = config['base_lr']
            warmup_epochs = config['warmup_epochs']
            num_epochs = config['epochs']
            
            # Calculate lambda value for backbone LR (ratio of current LR to base LR)
            lr_lambda_backbone = current_lr_backbone / base_lr
            
            # Find epoch that gives this lambda value in the new schedule
            # LambdaLR uses epoch index (0-based), so epoch 0 gives lambda(0)
            target_epoch = None
            
            # If resume_epoch >= warmup_epochs, we're definitely in cosine annealing phase
            # Otherwise, check if lambda matches warmup phase
            if resume_epoch >= warmup_epochs:
                # Definitely in cosine annealing phase
                # lambda = 0.5 * (1 + cos(pi * progress))
                # Solve: progress = arccos(2*lambda - 1) / pi
                # epoch = warmup_epochs + progress * (num_epochs - warmup_epochs)
                if lr_lambda_backbone > 0:
                    cos_arg = max(-1.0, min(1.0, 2 * lr_lambda_backbone - 1))
                    progress = np.arccos(cos_arg) / np.pi
                    target_epoch = int(warmup_epochs + progress * (num_epochs - warmup_epochs))
                else:
                    target_epoch = num_epochs - 1
            else:
                # resume_epoch < warmup_epochs, could be in warmup
                # Check if lambda matches warmup: lambda = (epoch + 1) / warmup_epochs
                # So epoch = lambda * warmup_epochs - 1
                warmup_epoch = lr_lambda_backbone * warmup_epochs - 1
                if warmup_epoch >= 0 and warmup_epoch < warmup_epochs:
                    # In warmup phase
                    target_epoch = int(warmup_epoch)
                else:
                    # Lambda doesn't match warmup, use cosine annealing calculation
                    if lr_lambda_backbone > 0:
                        cos_arg = max(-1.0, min(1.0, 2 * lr_lambda_backbone - 1))
                        progress = np.arccos(cos_arg) / np.pi
                        target_epoch = int(warmup_epochs + progress * (num_epochs - warmup_epochs))
                    else:
                        target_epoch = num_epochs - 1
            
            # Clamp target_epoch to valid range and ensure it's at least resume_epoch
            # This ensures scheduler is at least at resume_epoch position
            target_epoch = max(resume_epoch, min(target_epoch, num_epochs - 1))
            
            # Reset scheduler by creating new one and stepping to target epoch
            # This ensures scheduler state matches the restored LR
            if config.get('use_adaptive_lr', False):
                scheduler = get_adaptive_lr_scheduler(
                    optimizer=optimizer,
                    num_epochs=num_epochs,
                    warmup_epochs=warmup_epochs,
                    min_lr_ratio=config.get('min_lr_ratio', 0.01),
                    plateau_patience=config.get('lr_plateau_patience', 3),
                    min_delta=config.get('lr_min_delta', 0.0001),
                    mode='max',
                    verbose=True,
                    cooldown=config.get('lr_cooldown', 7),
                    threshold_mode=config.get('lr_threshold_mode', 'rel'),
                    use_val_loss=config.get('use_val_loss_for_lr', False)
                )
                
                # Restore adaptive scheduler state from checkpoint if available
                if checkpoint.get('scheduler'):
                    old_state = checkpoint['scheduler']
                    checkpoint_last_epoch = old_state.get('last_epoch', -1)
                    
                    # Lấy last_epoch từ checkpoint (bằng với epoch trong checkpoint)
                    # Nếu checkpoint không có last_epoch, dùng resume_epoch - 1
                    if checkpoint_last_epoch >= 0:
                        restored_last_epoch = checkpoint_last_epoch
                    else:
                        # Fallback: dùng resume_epoch - 1 (vì scheduler.step() sẽ increment)
                        restored_last_epoch = resume_epoch - 1
                    
                    # QUAN TRỌNG: Đảm bảo last_epoch >= warmup_epochs để tắt warmup khi resume
                    # Nếu resume từ epoch sau warmup, phải đảm bảo last_epoch >= warmup_epochs
                    if resume_epoch > warmup_epochs:
                        restored_last_epoch = max(warmup_epochs, restored_last_epoch, resume_epoch - 1)
                    
                    # Tạm thời override last_epoch trong old_state để load_state_dict() không ghi đè
                    old_state_modified = old_state.copy()
                    old_state_modified['last_epoch'] = restored_last_epoch
                    
                    # Restore scheduler state từ checkpoint (bao gồm base_lrs và current_lr_multiplier)
                    scheduler.load_state_dict(old_state_modified)
                    
                    # Đảm bảo base_lrs được restore từ checkpoint
                    checkpoint_base_lrs = old_state.get('base_lrs', None)
                    if checkpoint_base_lrs is not None:
                        scheduler.base_lrs = checkpoint_base_lrs
                        logger.info(f"  ✓ Restored base_lrs from checkpoint: {checkpoint_base_lrs}")
                    
                    # Đảm bảo last_epoch đúng (phòng trường hợp load_state_dict ghi đè)
                    scheduler.last_epoch = restored_last_epoch
                    
                    # Log trạng thái warmup để debug
                    will_warmup = scheduler.last_epoch < warmup_epochs
                    logger.info(f"  ✓ Resume from epoch {resume_epoch} → Set last_epoch to {scheduler.last_epoch}")
                    logger.info(f"    Warmup epochs: {warmup_epochs}, Checkpoint last_epoch: {checkpoint_last_epoch}")
                    logger.info(f"    Past warmup: {scheduler.last_epoch >= warmup_epochs}")
                    if will_warmup:
                        logger.warning(f"    ⚠️  WARMUP WILL BE APPLIED! (last_epoch={scheduler.last_epoch} < warmup_epochs={warmup_epochs})")
                    else:
                        logger.info(f"    ✅ WARMUP DISABLED (last_epoch={scheduler.last_epoch} >= warmup_epochs={warmup_epochs})")
                    
                    # Sync best_metric với best_val_acc từ checkpoint để đảm bảo consistency
                    # Nếu checkpoint có best_val_acc, dùng nó làm best_metric (nếu best_metric không có hoặc thấp hơn)
                    checkpoint_best_val_acc = checkpoint.get('val_acc', None)
                    if checkpoint_best_val_acc is not None:
                        if scheduler.best_metric is None or checkpoint_best_val_acc > scheduler.best_metric:
                            scheduler.best_metric = checkpoint_best_val_acc
                            logger.info(f"  ✓ Synced best_metric with checkpoint best_val_acc: {checkpoint_best_val_acc:.4f}")
                        elif scheduler.best_metric and scheduler.best_metric > checkpoint_best_val_acc:
                            # Scheduler best_metric cao hơn checkpoint → giữ scheduler best_metric
                            logger.info(f"  ✓ Keeping scheduler best_metric ({scheduler.best_metric:.4f}) > checkpoint best_val_acc ({checkpoint_best_val_acc:.4f})")
                    
                    logger.info(f"  ✓ Adaptive scheduler state restored:")
                    logger.info(f"    Best metric: {scheduler.best_metric:.4f}" if scheduler.best_metric else "    Best metric: None")
                    logger.info(f"    Plateau counter: {scheduler.plateau_counter}/{config.get('lr_plateau_patience', 7)}")
                    logger.info(f"    Cooldown counter: {scheduler.cooldown_counter}/{config.get('lr_cooldown', 7)}")
                    logger.info(f"    Last epoch: {scheduler.last_epoch} (warmup_epochs: {warmup_epochs}, past warmup: {scheduler.last_epoch >= warmup_epochs})")
                    logger.info(f"    Current LR multiplier: {scheduler.current_lr_multiplier:.4f}")
                    logger.info(f"    Base LR: {scheduler.base_lrs[0]:.6f}" + (f", {scheduler.base_lrs[1]:.6f}" if len(scheduler.base_lrs) > 1 else ""))
                else:
                    # No scheduler state in checkpoint, set last_epoch >= warmup_epochs to disable warmup
                    restored_last_epoch = max(warmup_epochs, resume_epoch - 1)
                    scheduler.last_epoch = restored_last_epoch
                    will_warmup = scheduler.last_epoch < warmup_epochs
                    logger.info(f"  ⚠️  No scheduler state in checkpoint, set last_epoch to {scheduler.last_epoch}")
                    if will_warmup:
                        logger.warning(f"    ⚠️  WARMUP WILL BE APPLIED! (last_epoch={scheduler.last_epoch} < warmup_epochs={warmup_epochs})")
                    else:
                        logger.info(f"    ✅ WARMUP DISABLED (last_epoch={scheduler.last_epoch} >= warmup_epochs={warmup_epochs})")
                
                # QUAN TRỌNG: Đảm bảo last_epoch >= warmup_epochs sau TẤT CẢ các bước restore
                # Nếu resume từ epoch sau warmup, PHẢI set last_epoch >= warmup_epochs để tắt warmup
                if is_resume and resume_epoch > warmup_epochs:
                    final_last_epoch = max(warmup_epochs, resume_epoch - 1, scheduler.last_epoch)
                    if scheduler.last_epoch != final_last_epoch:
                        logger.warning(f"  ⚠️  Fixing last_epoch: {scheduler.last_epoch} → {final_last_epoch} (to disable warmup)")
                        scheduler.last_epoch = final_last_epoch
                    logger.info(f"  ✓ Final check: last_epoch={scheduler.last_epoch}, warmup_epochs={warmup_epochs}, warmup_disabled={scheduler.last_epoch >= warmup_epochs}")
            else:
                scheduler = get_lr_scheduler(optimizer, num_epochs, warmup_epochs, cosine_start_epoch)
            
            # Restore LR từ checkpoint (scheduler creation có thể reset LR)
            # Đơn giản: lấy LR từ optimizer state trong checkpoint và dùng trực tiếp
            optimizer.param_groups[0]['lr'] = current_lr_backbone
            if current_lr_head and len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = current_lr_head
            
            # Đối với adaptive scheduler: đảm bảo current_lr_multiplier match với LR thực tế
            # Để scheduler.step() không thay đổi LR khi được gọi
            if config.get('use_adaptive_lr', False) and hasattr(scheduler, 'base_lrs') and hasattr(scheduler, 'current_lr_multiplier'):
                # Tính current_lr_multiplier từ LR thực tế và base_lrs để đảm bảo consistency
                # Khi scheduler.step() được gọi, nó sẽ dùng current_lr_multiplier * base_lrs
                # Nếu không sync, LR có thể bị thay đổi
                if len(scheduler.base_lrs) > 0 and scheduler.base_lrs[0] > 0:
                    actual_multiplier = current_lr_backbone / scheduler.base_lrs[0]
                    scheduler.current_lr_multiplier = actual_multiplier
                    logger.info(f"  ✓ Synced current_lr_multiplier to match LR: {actual_multiplier:.4f} (LR: {current_lr_backbone:.6f}, base: {scheduler.base_lrs[0]:.6f})")
                
                # QUAN TRỌNG: Đảm bảo last_epoch >= warmup_epochs sau khi sync current_lr_multiplier
                # Nếu resume từ epoch sau warmup, phải đảm bảo last_epoch >= warmup_epochs để tắt warmup
                if is_resume and resume_epoch > warmup_epochs and scheduler.last_epoch < warmup_epochs:
                    restored_last_epoch = max(warmup_epochs, resume_epoch - 1)
                    scheduler.last_epoch = restored_last_epoch
                    will_warmup = scheduler.last_epoch < warmup_epochs
                    logger.info(f"  ✓ Resume detected → Set last_epoch to {scheduler.last_epoch}")
                    logger.info(f"    Warmup epochs: {warmup_epochs}, Past warmup: {scheduler.last_epoch >= warmup_epochs}")
                    if will_warmup:
                        logger.warning(f"    ⚠️  WARMUP WILL BE APPLIED! (last_epoch={scheduler.last_epoch} < warmup_epochs={warmup_epochs})")
                    else:
                        logger.info(f"    ✅ WARMUP DISABLED (last_epoch={scheduler.last_epoch} >= warmup_epochs={warmup_epochs})")
            
            # For adaptive scheduler, don't step (it needs metrics to work correctly)
            # For standard scheduler, step to target epoch
            if not config.get('use_adaptive_lr', False):
                # Step scheduler to (target_epoch - 1) so that when training loop calls
                # scheduler.step() for the first time, it will step to target_epoch and LR should be close to restored LR
                # Note: LambdaLR's step() increments last_epoch, so we step to target_epoch - 1
                steps_to_take = max(0, target_epoch - 1)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*")
                    for _ in range(steps_to_take):
                        scheduler.step()
                logger.info(f"  Scheduler stepped to epoch {steps_to_take}, LR manually set to {current_lr_backbone:.6f}")
            else:
                # Adaptive scheduler: already set last_epoch = resume_epoch - 1
                # Will step with metrics in training loop
                logger.info(f"  Adaptive scheduler: last_epoch set to {resume_epoch - 1}, will step with metrics in training loop")
            
            # After stepping, manually set LR to restored value
            # This ensures LR matches what we want, regardless of scheduler state
            # Note: current_lr_backbone/head may have been updated by sync logic above
            # So we use the current values from optimizer (which may have been updated)
            final_lr_backbone = optimizer.param_groups[0]['lr']
            final_lr_head = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else None
            
            # Only update if different (to avoid unnecessary changes)
            if abs(final_lr_backbone - current_lr_backbone) > 1e-8:
                logger.info(f"  ✓ LR updated by sync logic: {current_lr_backbone:.6f} → {final_lr_backbone:.6f}")
                current_lr_backbone = final_lr_backbone
            else:
                optimizer.param_groups[0]['lr'] = current_lr_backbone
                
            if current_lr_head and len(optimizer.param_groups) > 1:
                if final_lr_head and abs(final_lr_head - current_lr_head) > 1e-8:
                    logger.info(f"  ✓ Head LR updated by sync logic: {current_lr_head:.6f} → {final_lr_head:.6f}")
                    current_lr_head = final_lr_head
                else:
                    optimizer.param_groups[1]['lr'] = current_lr_head
            
            logger.info(f"  Note: LR will be maintained at restored value for stability")
            
            # Store restored LR in config so training loop can maintain it
            config['restored_lr_backbone'] = current_lr_backbone
            config['restored_lr_head'] = current_lr_head
            config['maintain_restored_lr_epochs'] = 3  # Maintain for first 3 epochs after resume
            
            # Format head LR string properly
            head_lr_str = f"{current_lr_head:.6f}" if current_lr_head else "N/A"
            logger.info(f"  ✓ LR restored: Backbone={current_lr_backbone:.6f}, Head={head_lr_str}")
        else:
            logger.info(f"  LR transition:")
            logger.info(f"    Backbone: {current_lr_backbone:.6f} → {new_lr_backbone:.6f} ({lr_change_ratio:.1%} change)")
            if current_lr_head:
                logger.info(f"    Head: {current_lr_head:.6f} → {new_lr_head:.6f}")
            logger.info(f"  ✓ Scheduler adjusted for new epoch count")
    elif is_resume and checkpoint.get('scheduler') and scheduler:
        # Same num_epochs - can safely load scheduler state
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info(f"  ✓ Scheduler state loaded from checkpoint")
    
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    
    logger.info(f"Optimizer: AdamW")
    logger.info(f"  Backbone LR: {base_lr:.6f} (params: {len(backbone_params)})")
    logger.info(f"  Adapter/Head LR: {head_lr:.6f} (params: {len(adapter_params) + len(head_params)})")
    
    # Training paths
    checkpoint_path = config['output_dir'] / f'sota_vit_model_{model_id}_best.pt'
    # If resuming, create separate plot file to avoid overwriting original training plot
    if is_resume:
        history_plot_path = config['output_dir'] / f'sota_vit_model_{model_id}_training_from_epoch_{resume_epoch + 1}.png'
        logger.info(f"Plot will be saved to separate file: {history_plot_path.name} (to preserve original training plot)")
    else:
        history_plot_path = config['output_dir'] / f'sota_vit_model_{model_id}_training.png'
    
    # Log warmup status before training starts and FIX if needed
    if config.get('use_adaptive_lr', False) and hasattr(scheduler, 'last_epoch') and hasattr(scheduler, 'warmup_epochs'):
        warmup_epochs = scheduler.warmup_epochs
        current_last_epoch = scheduler.last_epoch
        
        # QUAN TRỌNG: Nếu resume từ epoch sau warmup và last_epoch < warmup_epochs, FIX ngay lập tức
        if is_resume and resume_epoch > warmup_epochs and current_last_epoch < warmup_epochs:
            fixed_last_epoch = max(warmup_epochs, resume_epoch - 1)
            logger.warning(f"  ⚠️  FIXING last_epoch: {current_last_epoch} → {fixed_last_epoch} (to disable warmup)")
            scheduler.last_epoch = fixed_last_epoch
            current_last_epoch = fixed_last_epoch
        
        will_warmup = current_last_epoch < warmup_epochs
        logger.info("="*60)
        logger.info("WARMUP STATUS CHECK (Before Training)")
        logger.info("="*60)
        logger.info(f"  Scheduler last_epoch: {current_last_epoch}")
        logger.info(f"  Warmup epochs: {warmup_epochs}")
        logger.info(f"  Will warmup be applied? {will_warmup}")
        if will_warmup:
            logger.warning(f"  ⚠️  WARMUP WILL BE APPLIED! (last_epoch={current_last_epoch} < warmup_epochs={warmup_epochs})")
            logger.warning(f"      Next scheduler.step() will increment to {current_last_epoch + 1}, which is < {warmup_epochs}")
        else:
            logger.info(f"  ✅ WARMUP DISABLED (last_epoch={current_last_epoch} >= warmup_epochs={warmup_epochs})")
            logger.info(f"      Next scheduler.step() will increment to {current_last_epoch + 1}, which is >= {warmup_epochs}")
        logger.info("="*60)
    
    # Train
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
        resume_history=resume_history,
        teacher=teacher  # Pass teacher for knowledge distillation
    )
    
    logger.info(f"Model {model_id} training completed!")
    return checkpoint_path


def run_inference_pipeline(config: dict, checkpoint_path: Path):
    """Run inference pipeline."""
    logger.info("="*60)
    logger.info("Running Inference Pipeline")
    logger.info("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device, logger)
    classes = checkpoint['classes']
    model_config = checkpoint.get('config', {})
    
    # Create model
    logger.info("Creating model...")
    architecture = model_config.get('architecture', config.get('architecture', 'vit_base'))
    logger.info(f"Architecture: {architecture}")
    
    model = create_model(
        architecture=architecture,
        num_classes=len(classes),
        pretrained_name=model_config.get('pretrained_name', config.get('pretrained_name', None)),
        use_adapters=model_config.get('use_adapters', config['use_adapters']),
        dropout=model_config.get('dropout', config.get('dropout', 0.1)),
        scales=model_config.get('multiscale_sizes', config.get('multiscale_sizes', [224, 256, 288])) if architecture == 'multiscale' else None
    ).to(device)
    
    # Load weights (prefer EMA if available)
    if 'ema_model' in checkpoint:
        logger.info("Loading EMA model weights")
        model.load_state_dict(checkpoint['ema_model'])
    else:
        model.load_state_dict(checkpoint['model'])
    
    model.eval()
    
    # Load test dataset
    logger.info("Loading test dataset...")
    test_data_dir = config['data_dir'] / 'test'
    test_dataset = TestDataset(
        root=test_data_dir,
        num_frames=config['num_frames'],
        frame_stride=config['frame_stride'],
        image_size=config['img_size']
    )
    
    # Optimize num_workers for test loader
    num_workers = config['num_workers']
    if num_workers == 0 and os.name != 'nt':  # Not Windows
        import multiprocessing
        num_workers = min(4, multiprocessing.cpu_count())
    elif num_workers == 0 and os.name == 'nt':
        # Windows: always use num_workers=0 to avoid shared memory errors
        num_workers = 0
    
    # Windows-specific settings for num_workers > 0
    is_windows = os.name == 'nt'
    if num_workers > 0 and is_windows:
        use_persistent_workers = False
        prefetch_factor = 2
        import multiprocessing
        multiprocessing_context = multiprocessing.get_context('spawn')
    elif num_workers > 0:
        use_persistent_workers = True
        prefetch_factor = 2
        multiprocessing_context = None
    else:
        use_persistent_workers = False
        prefetch_factor = None
        multiprocessing_context = None
    
    test_loader_kwargs = {
        'batch_size': config['batch_size'],
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': True if torch.cuda.is_available() else False,
        'persistent_workers': use_persistent_workers,
        'prefetch_factor': prefetch_factor
    }
    if multiprocessing_context is not None:
        test_loader_kwargs['multiprocessing_context'] = multiprocessing_context
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        **test_loader_kwargs
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Run inference
    predictions = run_inference(
        model=model,
        test_loader=test_loader,
        device=device,
        classes=classes,
        use_tta=True,
        num_crops=10,
        num_flips=2,
        expected_num_frames=config.get('num_frames', None)  # Truyền num_frames để tránh warning
    )
    
    # Generate submission
    submission_path = config['submissions_dir'] / f'submission_model_{config["model_id"]}.csv'
    generate_submission(predictions, submission_path, logger)
    
    logger.info("Inference pipeline completed!")
    return submission_path


def main():
    """Main function."""
    # Parse arguments
    config = parse_args()
    
    # Setup logging
    logger_instance = setup_logging(
        model_id=config['model_id'],
        logging_dir=config['logging_dir'],
        debug=config.get('debug', False)
    )
    
    logger.info("="*60)
    logger.info("SOTA Training Pipeline - Video Action Recognition")
    logger.info("="*60)
    logger.info(f"Model ID: {config['model_id']}")
    logger.info(f"Seed: {config['seed']}")
    
    # Auto-set pretrained_name based on architecture before logging config (for single model mode)
    if config.get('num_models', 1) == 1:
        architecture = config.get('architecture', 'vit_base')
        pretrained_name = config.get('pretrained_name', None)
        default_pretrained_names = config.get('default_pretrained_names', {})
        # Check if pretrained_name is from default config
        is_from_default_config = False
        if pretrained_name is not None and architecture in default_pretrained_names:
            expected_pretrained_name = default_pretrained_names[architecture]
            if pretrained_name != expected_pretrained_name:
                # Check if pretrained_name matches any other architecture's default
                for other_arch, other_default in default_pretrained_names.items():
                    if other_arch != architecture and pretrained_name == other_default:
                        is_from_default_config = True
                        break
        pretrained_name = validate_and_auto_set_pretrained_name(
            pretrained_name,
            architecture,
            default_pretrained_names,
            logger,
            is_from_default_config=is_from_default_config
        )
        config['pretrained_name'] = pretrained_name  # Update config
    
    logger.info(f"Configuration:")
    for key, value in config.items():
        if key not in ['data_dir', 'output_dir', 'logging_dir', 'submissions_dir', 'resume', 'checkpoint']:
            logger.info(f"  {key}: {value}")
    
    # Inference only mode
    if config.get('inference_only', False):
        checkpoint_path = Path(config.get('checkpoint', config['output_dir'] / f'sota_vit_model_{config["model_id"]}_best.pt'))
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
        run_inference_pipeline(config, checkpoint_path)
        return
    
    # Training mode
    num_models = config.get('num_models', 1)
    seeds = [42, 123, 456, 789, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
    checkpoint_paths = []
    checkpoint_infos = []
    
    # Check if auto mode with variations
    is_auto_mode = num_models > 1
    auto_use_variations = config.get('auto_use_variations', False)
    variations = get_hyperparameter_variations() if (is_auto_mode and auto_use_variations) else {}
    
    logger.info(f"Training {num_models} model(s)...")
    if is_auto_mode and auto_use_variations:
        logger.info("Auto mode: Using optimized architecture and feature variations")
        logger.info("Recommended: Train 5 models (Models 1-5) with diverse architectures")
        logger.info("  - Model 1: ViT-Large (base teacher, larger capacity)")
        logger.info("  - Model 2: ViT-Base (student from Model 1)")
        logger.info("  - Model 3: TimeSformer (student from Model 2, space-time attention)")
        logger.info("  - Model 4: Swin Transformer (student from Model 3, hierarchical)")
        logger.info("  - Model 5: Multi-scale ViT (student from Model 4, multi-scale fusion)")
        logger.info("  - Model 7: VideoMAEv2 (ViT-Large + divided space-time attention, SOTA)")
    
    for i in range(num_models):
        model_id = config['model_id'] + i if num_models > 1 else config['model_id']
        seed = seeds[model_id - 1] if model_id <= len(seeds) else config['seed'] + model_id
        
        # Update config for this model
        model_config = config.copy()
        model_config['model_id'] = model_id
        model_config['seed'] = seed
        
        # Apply variation for this model if in auto mode
        if model_id in variations:
            variation = variations[model_id].copy()
            # Update hyperparameters and SOTA features from variation
            for key, value in variation.items():
                model_config[key] = value
            
            # Auto-set pretrained_name based on architecture before logging config
            architecture = model_config.get('architecture', 'vit_base')
            pretrained_name = model_config.get('pretrained_name', None)
            default_pretrained_names = model_config.get('default_pretrained_names', {})
            # Check if pretrained_name is from default config
            is_from_default_config = False
            if pretrained_name is not None and architecture in default_pretrained_names:
                expected_pretrained_name = default_pretrained_names[architecture]
                if pretrained_name != expected_pretrained_name:
                    # Check if pretrained_name matches any other architecture's default
                    for other_arch, other_default in default_pretrained_names.items():
                        if other_arch != architecture and pretrained_name == other_default:
                            is_from_default_config = True
                            break
            pretrained_name = validate_and_auto_set_pretrained_name(
                pretrained_name,
                architecture,
                default_pretrained_names,
                logger,
                is_from_default_config=is_from_default_config
            )
            model_config['pretrained_name'] = pretrained_name  # Update config
            
            logger.info(f"Model {model_id} config: Architecture={architecture}, "
                       f"EMA={model_config.get('use_ema')}, "
                       f"CutMix={model_config.get('use_cutmix')}, "
                       f"FocalLoss={model_config.get('use_focal_loss')}, "
                       f"ProgressiveResize={model_config.get('use_progressive_resize')}")
        
        # Setup logging for this model
        if num_models > 1:
            logger_instance = setup_logging(
                model_id=model_id,
                logging_dir=config['logging_dir'],
                debug=config.get('debug', False)
            )
            logger.info(f"\n{'='*60}")
            logger.info(f"Training Model {model_id}/{num_models}")
            logger.info(f"{'='*60}")
        
        # Train model
        checkpoint_path = train_single_model(model_config, model_id, seed)
        checkpoint_paths.append(checkpoint_path)
        
        # Load checkpoint to get validation accuracy for weighting
        checkpoint = load_checkpoint(checkpoint_path, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), logger)
        val_acc = checkpoint.get('val_acc', 0.0)
        checkpoint_infos.append({
            'model_id': model_id,
            'checkpoint_path': checkpoint_path,
            'val_acc': val_acc,
            'classes': checkpoint.get('classes')
        })
        logger.info(f"Model {model_id} completed. Val Acc: {val_acc:.4f}")
        
        # Generate submission for this individual model
        if num_models > 1:
            # Only generate individual submissions when training multiple models
            # Single model mode will generate submission at the end
            logger.info(f"Generating submission for Model {model_id}...")
            run_inference_pipeline(model_config, checkpoint_path)
    
    logger.info("="*60)
    logger.info("All models training completed!")
    logger.info("="*60)
    
    # Ensemble inference if multiple models (auto_ensemble is enabled by default for multiple models)
    if num_models > 1:
        logger.info("\n" + "="*60)
        logger.info("Starting Ensemble Inference")
        logger.info("="*60)
        
        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load all models
        models = []
        weights = []
        classes = None
        
        for info in checkpoint_infos:
            checkpoint = load_checkpoint(info['checkpoint_path'], device, logger)
            
            if classes is None:
                classes = checkpoint['classes']
            
            # Create model
            checkpoint_config = checkpoint.get('config', {})
            architecture = checkpoint_config.get('architecture', config.get('architecture', 'vit_base'))
            
            model = create_model(
                architecture=architecture,
                num_classes=len(classes),
                pretrained_name=checkpoint_config.get('pretrained_name', config.get('pretrained_name', None)),
                use_adapters=checkpoint_config.get('use_adapters', config['use_adapters']),
                dropout=checkpoint_config.get('dropout', config.get('dropout', 0.1)),
                scales=checkpoint_config.get('multiscale_sizes', config.get('multiscale_sizes', [224, 256, 288])) if architecture == 'multiscale' else None
            ).to(device)
            
            # Load weights (prefer EMA if available)
            if 'ema_model' in checkpoint:
                logger.info(f"Loading EMA model weights for Model {info['model_id']}")
                model.load_state_dict(checkpoint['ema_model'])
            else:
                model.load_state_dict(checkpoint['model'])
            
            model.eval()
            models.append(model)
            
            # Weight by validation accuracy
            weights.append(info['val_acc'])
            logger.info(f"Model {info['model_id']} loaded. Val Acc: {info['val_acc']:.4f}")
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        logger.info(f"\nEnsemble weights: {[f'{w:.4f}' for w in weights]}")
        
        # Load test dataset
        logger.info("Loading test dataset...")
        test_data_dir = config['data_dir'] / 'test'
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
            pin_memory=True
        )
        
        logger.info(f"Test samples: {len(test_dataset)}")
        
        # Run ensemble inference with improved method (weighted average of probabilities)
        # This ensures all models have the same scale (0-1) before ensemble
        predictions = run_ensemble_inference(
            models=models,
            test_loader=test_loader,
            device=device,
            classes=classes,
            weights=weights,
            use_tta=True,
            num_crops=10,
            num_flips=2,
            ensemble_method='weighted_avg_probs',  # Use probabilities instead of logits
            use_softmax=True  # Apply softmax before ensemble (recommended)
        )
        
        # Generate final submission
        ensemble_submission_path = config['submissions_dir'] / 'ensemble_submission.csv'
        generate_submission(predictions, ensemble_submission_path, logger)
        
        logger.info("="*60)
        logger.info("Ensemble inference completed!")
        logger.info(f"Final submission saved to: {ensemble_submission_path}")
        logger.info("="*60)
    elif num_models == 1:
        # Single model inference
        run_inference_pipeline(config, checkpoint_paths[0])


if __name__ == '__main__':
    main()
