"""
Utility functions for SOTA Training Pipeline.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import matplotlib.pyplot as plt
import numpy as np


def setup_logging(
    model_id: int,
    logging_dir: Path,
    debug: bool = False
) -> logging.Logger:
    """
    Setup logging with file handler and console handler.
    
    Args:
        model_id: Model ID for log file naming
        logging_dir: Directory to save log files
        debug: Enable debug logging level
    
    Returns:
        Logger instance
    """
    # Create logging directory if it doesn't exist
    logging_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('sota_training')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logging_dir / f'training_model_{model_id}_{timestamp}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def auto_adjust_batch_size(
    initial_batch_size: int,
    gpu_memory_gb: float,
    num_frames: int = 16,
    channels: int = 3,
    img_size: int = 224,
    model_params_mb: float = None,
    use_mixed_precision: bool = True,
    target_memory_ratio: float = 0.90,  # Target 90% VRAM utilization
    min_batch_size: int = 8,
    max_batch_size: int = None
) -> Tuple[int, Dict[str, float]]:
    """
    Automatically adjust batch size to maximize VRAM and GPU utilization.
    
    With mixed precision training:
    - Reduces memory by ~40-50%
    - Real-world: batch_size=12 uses ~4.7GB VRAM
    - Memory per batch_size with mixed precision: ~0.392 GB (at small batches)
    - At larger batches, memory efficiency improves: ~0.30-0.35 GB per batch_size
    
    Args:
        initial_batch_size: Initial batch size
        gpu_memory_gb: GPU memory in GB
        num_frames: Number of frames per video
        channels: Image channels (3 for RGB)
        img_size: Image size (224x224)
        model_params_mb: Model parameters in MB
        use_mixed_precision: Whether using mixed precision
        target_memory_ratio: Target memory usage ratio (default: 0.90 = 90%)
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size (None = no limit)
    
    Returns:
        Tuple of (optimized_batch_size, memory_estimate_dict)
    """
    # Real-world calibration with mixed precision + torch.compile:
    # - batch_size=12 uses ~4.7GB → 4.7/12 = 0.392 GB per batch_size (without compile)
    # - torch.compile can reduce memory by 30-50% at larger batches
    # - At large batches (64-80), with compile + mixed precision: ~0.30-0.32 GB per batch_size
    # For target calculation, use 0.32 GB/batch (balanced for 64-80 batch size range)
    # This avoids CPU bottleneck while maximizing VRAM usage
    MEMORY_PER_BATCH_SIZE = 0.32  # GB per batch_size (balanced for num_workers=0)
    
    # Calculate target memory (90% of VRAM)
    target_memory_gb = gpu_memory_gb * target_memory_ratio
    
    # Calculate optimal batch size
    optimal_batch_size = int(target_memory_gb / MEMORY_PER_BATCH_SIZE)
    
    # Apply limits
    if max_batch_size is not None:
        optimal_batch_size = min(optimal_batch_size, max_batch_size)
    
    optimal_batch_size = max(optimal_batch_size, min_batch_size)
    
    # Round to nearest multiple of 4 for efficiency (better GPU utilization)
    optimal_batch_size = (optimal_batch_size // 4) * 4
    if optimal_batch_size < min_batch_size:
        optimal_batch_size = min_batch_size
    
    # Estimate memory for the optimal batch size
    # Use progressive scaling for accurate estimation (matches estimate_memory logic)
    # Account for torch.compile memory reduction at large batches
    if optimal_batch_size <= 12:
        est_memory_per_batch = 0.392  # Small batches: no compile benefit
    elif optimal_batch_size <= 32:
        est_memory_per_batch = 0.35   # Medium batches: slight compile benefit
    elif optimal_batch_size <= 64:
        est_memory_per_batch = 0.32   # Large batches: moderate compile benefit
    else:
        est_memory_per_batch = 0.30   # Very large batches: significant compile benefit
    
    total_gb = optimal_batch_size * est_memory_per_batch
    total_mb = total_gb * 1024
    
    best_estimate = {
        'total_gb': total_gb,
        'total_mb': total_mb,
        'estimated_actual_gb': total_gb
    }
    
    return optimal_batch_size, best_estimate


def estimate_memory(
    batch_size: int,
    num_frames: int = 16,
    channels: int = 3,
    img_size: int = 224,
    model_params_mb: float = None,
    use_mixed_precision: bool = True
) -> Dict[str, float]:
    """
    Estimate memory usage for video training.
    Simple formula based on MEMORY_CALCULATION.md and real-world calibration.
    
    Real-world: batch_size=12 uses ~4.7GB actual VRAM
    Simple calculation: memory = batch_size × 0.392 GB
    
    Args:
        batch_size: Batch size
        num_frames: Number of frames per video
        channels: Image channels (3 for RGB)
        img_size: Image size (224x224)
        model_params_mb: Model parameters in MB (not used, kept for compatibility)
        use_mixed_precision: Whether using mixed precision (not used, kept for compatibility)
    
    Returns:
        Dictionary with memory estimates:
        - total_gb: Total memory in GB (real-world estimate)
        - total_mb: Total memory in MB
    """
    # Progressive scaling: memory per batch_size decreases at larger batches
    # Based on real-world test data from test_max_batch_size:
    # - batch_size=12: ~4.7GB → 0.392 GB/batch (real-world calibration)
    # - batch_size=176: ~20.48GB → 0.116 GB/batch
    # - batch_size=222: ~25.45GB → 0.115 GB/batch
    # - batch_size=233: ~26.68GB → 0.115 GB/batch
    # Accounts for torch.compile memory reduction at large batches
    # NOTE: With adapters, memory usage is higher, so we use real-world calibration
    if batch_size <= 12:
        MEMORY_PER_BATCH_SIZE = 4.7 / 12  # ~0.392 GB/batch (real-world: batch_size=12 uses ~4.7GB)
    elif batch_size <= 32:
        MEMORY_PER_BATCH_SIZE = 0.35   # Slightly more efficient at medium batches
    elif batch_size <= 64:
        MEMORY_PER_BATCH_SIZE = 0.32   # More efficient at large batches
    elif batch_size <= 128:
        MEMORY_PER_BATCH_SIZE = 0.20   # More efficient at very large batches
    elif batch_size <= 200:
        MEMORY_PER_BATCH_SIZE = 0.12   # Very efficient at very large batches (with compile)
    else:
        MEMORY_PER_BATCH_SIZE = 0.115  # Extremely efficient at huge batches (tested: batch_size=222 uses ~25.45GB)
    
    total_gb = batch_size * MEMORY_PER_BATCH_SIZE
    total_mb = total_gb * 1024
    
    return {
        'total_gb': total_gb,
        'total_mb': total_mb
    }


def log_system_info(
    logger: logging.Logger,
    device: torch.device,
    batch_size: int = None,
    num_frames: int = None,
    img_size: int = None,
    model_params: int = None
):
    """
    Log system information and memory estimates.
    
    Args:
        logger: Logger instance
        device: Device (cuda/cpu)
        batch_size: Batch size for memory estimation
        num_frames: Number of frames per video
        img_size: Image size
        model_params: Number of model parameters
    """
    logger.info("="*60)
    logger.info("System Information")
    logger.info("="*60)
    logger.info(f"PyTorch version: {torch.__version__}")
    
    gpu_memory_gb = None
    if device.type == 'cuda':
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Memory: {gpu_memory_gb:.2f} GB")
        
        if "5090" in torch.cuda.get_device_name(0):
            logger.warning("RTX 5090 detected - may have CUDA compatibility issues")
    else:
        logger.warning("No GPU detected - training will be very slow on CPU")
    
    # Memory estimation if parameters provided
    if batch_size is not None and device.type == 'cuda':
        logger.info("")
        logger.info("Memory Estimation")
        logger.info("-" * 60)
        
        num_frames = num_frames or 16
        img_size = img_size or 224
        model_params_mb = (model_params * 4 / (1024**2)) if model_params else None
        
        mem_estimate = estimate_memory(
            batch_size=batch_size,
            num_frames=num_frames,
            img_size=img_size,
            model_params_mb=model_params_mb,
            use_mixed_precision=True
        )
        
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Frames per video: {num_frames}")
        logger.info(f"Image size: {img_size}x{img_size}")
        logger.info(f"Mixed precision: Enabled (reduces memory ~50%)")
        logger.info("")
        logger.info("Estimated Memory Usage (Real-world):")
        logger.info(f"  TOTAL ESTIMATE:    {mem_estimate['total_mb']:.2f} MB ({mem_estimate['total_gb']:.2f} GB)")
        logger.info("")
        logger.info("  Note: Based on real-world calibration (batch_size=12 uses ~4.7GB)")
        logger.info("")
        
        if gpu_memory_gb:
            estimated_actual_gb = mem_estimate['total_gb']  # Already real-world estimate
            logger.info(f"GPU Memory: {gpu_memory_gb:.2f} GB")
            logger.info(f"Estimated actual usage: ~{estimated_actual_gb:.2f} GB ({estimated_actual_gb/gpu_memory_gb*100:.1f}%)")
            
            if estimated_actual_gb > gpu_memory_gb * 0.9:
                logger.warning(f"⚠️  WARNING: Estimated memory ({estimated_actual_gb:.2f} GB) may exceed GPU memory ({gpu_memory_gb:.2f} GB)!")
                logger.warning("   Consider reducing batch_size or using gradient accumulation.")
            elif estimated_actual_gb > gpu_memory_gb * 0.7:
                logger.warning(f"⚠️  CAUTION: Estimated memory ({estimated_actual_gb:.2f} GB) is high ({gpu_memory_gb * 0.7:.2f} GB threshold).")
                logger.warning("   Monitor GPU memory usage during training.")
            else:
                logger.info(f"✓ Estimated memory ({estimated_actual_gb:.2f} GB) is safe for GPU ({gpu_memory_gb:.2f} GB).")
        
        logger.info("-" * 60)
    
    logger.info("="*60)


def plot_training_history(
    history: Dict[str, list],
    save_path: Path,
    model_id: int
):
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with 'train_loss', 'train_acc', 'val_loss', 'val_acc' lists
                 Optionally 'train_clean_loss', 'train_clean_acc' for clean training metrics
        save_path: Path to save the plot
        model_id: Model ID for title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss - Show 2 or 3 lines depending on whether clean training loss is available
    axes[0].plot(epochs, history['train_loss'], label='Train (with aug)', marker='o', linewidth=2, alpha=0.7)
    
    # Plot clean training loss if available (for fair comparison with validation)
    # Only plot if there are non-None values (clean training loss was enabled)
    if 'train_clean_loss' in history and len(history['train_clean_loss']) > 0:
        clean_losses = [x for x in history['train_clean_loss'] if x is not None]
        if len(clean_losses) > 0:
            # Filter out None values for plotting
            clean_loss_epochs = [i+1 for i, x in enumerate(history['train_clean_loss']) if x is not None]
            clean_loss_values = [x for x in history['train_clean_loss'] if x is not None]
            axes[0].plot(clean_loss_epochs, clean_loss_values, label='Train Clean (no aug)', 
                        marker='^', linewidth=2, linestyle='--', color='green', alpha=0.8)
    
    axes[0].plot(epochs, history['val_loss'], label='Val', marker='s', linewidth=2, color='orange')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'Training Loss - Model {model_id}', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], label='Train', marker='o', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], label='Val', marker='s', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'Training Accuracy - Model {model_id}', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Fix for NumPy 2.4.0rc1 compatibility
    if np.__version__.startswith('2.4'):
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.3)
        plt.savefig(save_path, dpi=100, bbox_inches=None)
    else:
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.close()
    
    # Log plot save location
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Training plot saved to: {save_path} (epochs: {len(history['train_loss'])})")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    history: Dict[str, list],
    best_val_acc: float,
    train_acc: float,
    classes: list,
    config: Dict[str, Any],
    checkpoint_path: Path,
    ema_model: Optional[torch.nn.Module] = None
):
    """
    Save model checkpoint with metadata.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        history: Training history
        best_val_acc: Best validation accuracy
        train_acc: Current training accuracy
        classes: Class names
        config: Configuration dictionary
        checkpoint_path: Path to save checkpoint
        ema_model: EMA model (optional)
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'history': history,
        'val_acc': best_val_acc,
        'train_acc': train_acc,
        'classes': classes,
        'config': config,
        'model_id': config.get('model_id', 0),
        'seed': config.get('seed', 0),
    }
    
    if ema_model is not None:
        checkpoint['ema_model'] = ema_model.state_dict()
    
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        logger: Logger instance
    
    Returns:
        Checkpoint dictionary
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # PyTorch 2.6+ requires weights_only=False for loading checkpoints with custom objects
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    logger.info(f"Checkpoint loaded: Epoch {checkpoint.get('epoch', 'unknown')}, "
                f"Val Acc: {checkpoint.get('val_acc', 0.0):.4f}")
    
    return checkpoint
