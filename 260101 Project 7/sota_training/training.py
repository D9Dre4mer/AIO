"""
Training functions and training loop for video action recognition.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, Any, Optional, Tuple
import logging

from .losses import get_loss_function
from .augmentation import mixup_data, mixup_criterion, cutmix_data, cutmix_criterion
from .models import EMAModel
from .data_generation import generate_synthetic_batch
from .knowledge_distillation import DistillationLoss

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = 1,
    use_mixup: bool = True,
    use_cutmix: bool = False,
    mixup_alpha: float = 0.4,
    cutmix_alpha: float = 1.0,
    label_smoothing: float = 0.0,
    use_focal_loss: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    ema_model: Optional[EMAModel] = None,
    use_synthetic_data: bool = False,
    synthetic_method: str = 'frame_mixup',
    synthetic_ratio: float = 0.3,
    teacher: Optional[Any] = None,  # EnsembleTeacher for knowledge distillation
    use_distillation: bool = False,
    distillation_temperature: float = 3.0,
    distillation_alpha: float = 0.7
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        loader: Training data loader
        optimizer: Optimizer
        scaler: Gradient scaler for mixed precision
        device: Device to train on
        grad_accum_steps: Gradient accumulation steps
        use_mixup: Whether to use Mixup
        use_cutmix: Whether to use CutMix
        mixup_alpha: Mixup alpha parameter
        cutmix_alpha: CutMix alpha parameter
        label_smoothing: Label smoothing factor
        use_focal_loss: Whether to use Focal Loss
        focal_alpha: Focal Loss alpha
        focal_gamma: Focal Loss gamma
        ema_model: EMA model (optional)
    
    Returns:
        Average loss and accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
    
    # Get loss function
    base_criterion = get_loss_function(
        use_focal_loss=use_focal_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing
    )
    
    # Setup distillation loss if enabled
    distillation_criterion = None
    if use_distillation and teacher is not None:
        distillation_criterion = DistillationLoss(
            temperature=distillation_temperature,
            alpha=distillation_alpha,
            base_criterion=base_criterion
        )
        # Move teacher to device if not already
        teacher.to(device)
        logger.info(f"Knowledge distillation enabled: temperature={distillation_temperature}, alpha={distillation_alpha}")
    
    # Use distillation criterion if available, otherwise use base criterion
    criterion = distillation_criterion if distillation_criterion is not None else base_criterion
    
    progress = tqdm(loader, desc="Train", leave=False)
    for batch_idx, (videos, labels) in enumerate(progress):
        # Non-blocking transfer for better GPU utilization (overlaps with computation)
        videos = videos.to(device, non_blocking=True)
        # CRITICAL: Labels must be LongTensor (int64) for CrossEntropyLoss
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        
        # Generate synthetic data if enabled
        if use_synthetic_data and random.random() < synthetic_ratio:
            synthetic_videos, synthetic_labels_a, synthetic_labels_b, synthetic_lam = generate_synthetic_batch(
                videos, labels, method=synthetic_method, alpha=mixup_alpha
            )
            # Mix synthetic data with real data (50% synthetic, 50% real)
            if random.random() < 0.5:
                videos = synthetic_videos
                if synthetic_labels_b is not None:
                    # Mixup-style labels
                    labels_a, labels_b = synthetic_labels_a, synthetic_labels_b
                    lam = synthetic_lam
                else:
                    labels_a, labels_b, lam = labels, labels, 1.0
            else:
                # Keep original batch, synthetic data will be used as additional augmentation
                labels_a, labels_b, lam = labels, labels, 1.0
        else:
            labels_a, labels_b, lam = labels, labels, 1.0
        
        # Choose augmentation: Mixup or CutMix
        # Use 50% probability (giống notebook) - Vẫn có raw data để học features cơ bản
        use_aug = False
        if use_mixup and use_cutmix:
            # Randomly choose between Mixup and CutMix
            use_aug = random.random() < 0.5  # 50% probability (giống notebook)
            aug_type = 'mixup' if random.random() < 0.5 else 'cutmix'
        elif use_mixup:
            use_aug = random.random() < 0.5  # 50% probability (giống notebook)
            aug_type = 'mixup'
        elif use_cutmix:
            use_aug = random.random() < 0.5  # 50% probability (giống notebook)
            aug_type = 'cutmix'
        
        # Apply augmentation
        if use_aug and aug_type == 'mixup':
            # Mixup augmentation overrides synthetic data labels
            mixed_videos, labels_a, labels_b, lam = mixup_data(videos, labels, mixup_alpha)
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                logits = model(mixed_videos)
                # For distillation, get teacher logits for mixed videos
                if use_distillation and teacher is not None and distillation_criterion is not None:
                    with torch.no_grad():
                        teacher_logits_mixed = teacher.predict(mixed_videos)
                    # Compute mixup loss with distillation: mixup of distillation losses
                    loss_a = distillation_criterion(logits, teacher_logits_mixed, labels_a)
                    loss_b = distillation_criterion(logits, teacher_logits_mixed, labels_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss = mixup_criterion(base_criterion, logits, labels_a, labels_b, lam)
            # For accuracy calculation
            preds = logits.argmax(dim=1)
            correct += (lam * (preds == labels_a).float() + 
                       (1 - lam) * (preds == labels_b).float()).sum().item()
        elif use_aug and aug_type == 'cutmix':
            # CutMix augmentation overrides synthetic data labels
            mixed_videos, labels_a, labels_b, lam = cutmix_data(videos, labels, cutmix_alpha)
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                logits = model(mixed_videos)
                # For distillation, get teacher logits for mixed videos
                if use_distillation and teacher is not None and distillation_criterion is not None:
                    with torch.no_grad():
                        teacher_logits_mixed = teacher.predict(mixed_videos)
                    # Compute cutmix loss with distillation: cutmix of distillation losses
                    loss_a = distillation_criterion(logits, teacher_logits_mixed, labels_a)
                    loss_b = distillation_criterion(logits, teacher_logits_mixed, labels_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss = cutmix_criterion(base_criterion, logits, labels_a, labels_b, lam)
            # For accuracy calculation
            preds = logits.argmax(dim=1)
            correct += (lam * (preds == labels_a).float() + 
                       (1 - lam) * (preds == labels_b).float()).sum().item()
        elif labels_b is not None and lam < 1.0:
            # Synthetic data with mixup-style labels (no additional augmentation)
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                logits = model(videos)
                if use_distillation and teacher is not None and distillation_criterion is not None:
                    with torch.no_grad():
                        teacher_logits = teacher.predict(videos)
                    # Compute mixup loss with distillation
                    loss_a = distillation_criterion(logits, teacher_logits, labels_a)
                    loss_b = distillation_criterion(logits, teacher_logits, labels_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss = mixup_criterion(base_criterion, logits, labels_a, labels_b, lam)
            preds = logits.argmax(dim=1)
            correct += (lam * (preds == labels_a).float() + 
                       (1 - lam) * (preds == labels_b).float()).sum().item()
        else:
            # No augmentation or synthetic data - standard training
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                logits = model(videos)
                if use_distillation and teacher is not None and distillation_criterion is not None:
                    with torch.no_grad():
                        teacher_logits = teacher.predict(videos)
                    # Use distillation loss
                    loss = criterion(logits, teacher_logits, labels)
                else:
                    loss = criterion(logits, labels)
            
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
        
        total += labels.size(0)
        loss_value = loss.item()
        
        loss = loss / grad_accum_steps
        scaler.scale(loss).backward()
        
        should_step = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == len(loader))
        if should_step:
            # Gradient clipping
            scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update EMA
            if ema_model is not None:
                ema_model.update()
        
        batch_size = videos.size(0)
        total_loss += loss_value * batch_size
        progress.set_postfix(loss=f"{loss_value:.4f}", acc=f"{correct / max(total, 1):.4f}")
    
    avg_loss = total_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_smoothing: float = 0.0,
    use_focal_loss: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0
) -> Tuple[float, float]:
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        loader: Validation data loader
        device: Device to evaluate on
        label_smoothing: Label smoothing factor
        use_focal_loss: Whether to use Focal Loss
        focal_alpha: Focal Loss alpha
        focal_gamma: Focal Loss gamma
    
    Returns:
        Average loss and accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    criterion = get_loss_function(
        use_focal_loss=use_focal_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing
    )
    
    with torch.no_grad():
        progress = tqdm(loader, desc="Val", leave=False)
        for batch_idx, (videos, labels) in enumerate(progress):
            # Non-blocking transfer for better GPU utilization
            videos = videos.to(device)
            # CRITICAL: Labels must be LongTensor (int64) for CrossEntropyLoss
            labels = labels.to(device, dtype=torch.long)
            
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                logits = model(videos)
                loss = criterion(logits, labels)
            
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.4f}")
    
    avg_loss = total_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


def evaluate_clean_training_loss(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None
) -> Tuple[float, float]:
    """
    Evaluate model on training set WITHOUT augmentation (clean loss).
    This gives a fair comparison with validation loss.
    
    Args:
        model: Model to evaluate
        train_loader: Training data loader
        device: Device to evaluate on
        max_samples: Maximum number of samples to evaluate (None = all)
    
    Returns:
        Average loss and accuracy on clean training data
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Use standard CrossEntropyLoss without label smoothing for fair comparison
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        progress = tqdm(train_loader, desc="Train (clean)", leave=False)
        for batch_idx, (videos, labels) in enumerate(progress):
            if max_samples is not None and total >= max_samples:
                break
                
            # Non-blocking transfer for better GPU utilization
            videos = videos.to(device)
            # CRITICAL: Labels must be LongTensor (int64) for CrossEntropyLoss
            labels = labels.to(device, dtype=torch.long)
            
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                logits = model(videos)
                loss = criterion(logits, labels)
            
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            batch_size = labels.size(0)
            total += batch_size
            total_loss += loss.item() * batch_size
            
            if max_samples is not None and total >= max_samples:
                # Adjust for partial batch
                break
            
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.4f}")
    
    avg_loss = total_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


def get_lr_scheduler(optimizer: torch.optim.Optimizer, num_epochs: int, warmup_epochs: int = 15, cosine_start_epoch: int = 20):
    """
    Create learning rate scheduler with warmup and cosine annealing (giống notebook).
    
    Args:
        optimizer: Optimizer
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs (LR increases linearly from 0 to 1.0)
        cosine_start_epoch: Epoch when cosine annealing starts (unused, kept for compatibility)
    
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup: linear increase from 0 to 1.0 (giống notebook)
            # Notebook: return (epoch + 1) / warmup_epochs
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing: từ 1.0 → 0 (giống notebook)
            # Notebook: return 0.5 * (1 + np.cos(np.pi * progress))
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ProgressiveResizeManager:
    """
    Manages progressive resizing based on validation accuracy improvement and LR scheduling.
    
    Logic:
    - Tăng size khi val acc đạt plateau (không cải thiện trong N epochs)
    - Liên hệ với LR scheduler: khi LR giảm → có thể tăng size
    - Đảm bảo đạt size cuối cùng trước khi training kết thúc
    """
    
    def __init__(
        self,
        sizes: list,
        total_epochs: int,
        patience: int = 5,  # Số epochs không cải thiện trước khi tăng size
        min_improvement: float = 0.005,  # Min improvement để coi là "cải thiện" (0.5%)
        force_final_size_epochs: int = 10,  # Force size cuối cùng trong N epochs cuối
        min_epochs_per_stage: int = 3  # Min epochs ở mỗi stage trước khi tăng size
    ):
        """
        Args:
            sizes: List of image sizes to progress through (e.g., [160, 192, 224])
            total_epochs: Total number of epochs
            patience: Số epochs không cải thiện trước khi tăng size
            min_improvement: Minimum improvement để coi là "cải thiện" (relative)
            force_final_size_epochs: Force size cuối cùng trong N epochs cuối
            min_epochs_per_stage: Min epochs ở mỗi stage trước khi tăng size
        """
        self.sizes = sizes if sizes else [224]
        self.total_epochs = total_epochs
        self.patience = patience
        self.min_improvement = min_improvement
        self.force_final_size_epochs = force_final_size_epochs
        self.min_epochs_per_stage = min_epochs_per_stage
        
        # State tracking
        self.current_stage = 0
        self.current_size = self.sizes[0]
        self.best_val_acc_at_stage = None
        self.epochs_at_current_stage = 0
        self.no_improvement_counter = 0
        self.last_lr_reduction_epoch = -1
        
    def get_size(
        self,
        epoch: int,
        val_acc: float = None,
        lr_reduced: bool = False
    ) -> int:
        """
        Get image size for current epoch based on validation accuracy and LR.
        
        Args:
            epoch: Current epoch (0-indexed)
            val_acc: Current validation accuracy (optional)
            lr_reduced: Whether LR was reduced in this epoch
        
        Returns:
            Image size for current epoch
        """
        # Force final size trong N epochs cuối
        epochs_remaining = self.total_epochs - epoch - 1
        if epochs_remaining <= self.force_final_size_epochs:
            if self.current_stage < len(self.sizes) - 1:
                logger.info(f"  Progressive resize: Force final size {self.sizes[-1]} "
                          f"(còn {epochs_remaining} epochs)")
                self.current_stage = len(self.sizes) - 1
                self.current_size = self.sizes[-1]
                self.epochs_at_current_stage = 0
                self.no_improvement_counter = 0
            return self.sizes[-1]
        
        # Nếu đã ở stage cuối, giữ nguyên
        if self.current_stage >= len(self.sizes) - 1:
            return self.sizes[-1]
        
        # Track epochs at current stage
        self.epochs_at_current_stage += 1
        
        # Nếu chưa đủ min epochs ở stage hiện tại, giữ nguyên
        if self.epochs_at_current_stage < self.min_epochs_per_stage:
            return self.current_size
        
        # Nếu có val_acc, check improvement
        # FIX: Update best_val_acc_at_stage ngay cả khi không vượt threshold
        # để đảm bảo counter chỉ tăng khi thực sự không cải thiện
        if val_acc is not None:
            # Initialize best val acc at stage
            if self.best_val_acc_at_stage is None:
                self.best_val_acc_at_stage = val_acc
                self.no_improvement_counter = 0
                return self.current_size
            
            # Store old best để check improvement
            old_best_val_acc_at_stage = self.best_val_acc_at_stage
            
            # QUAN TRỌNG: Update best_val_acc_at_stage ngay cả khi không vượt threshold
            # Điều này đảm bảo best_val_acc_at_stage luôn là giá trị tốt nhất ở stage hiện tại
            if val_acc >= old_best_val_acc_at_stage:
                self.best_val_acc_at_stage = val_acc
            
            # Check if improved significantly (vượt threshold)
            improvement = val_acc - old_best_val_acc_at_stage
            relative_improvement = improvement / max(old_best_val_acc_at_stage, 0.01)  # Avoid division by zero
            
            if relative_improvement >= self.min_improvement:
                # Model đang cải thiện đáng kể → reset counter
                self.no_improvement_counter = 0
            else:
                # Không cải thiện đáng kể → tăng counter
                self.no_improvement_counter += 1
        
        # Tăng size nếu:
        # 1. LR giảm (model đã học tốt ở size hiện tại)
        # 2. Hoặc val acc không cải thiện trong N epochs (plateau)
        should_increase_size = False
        
        if lr_reduced:
            # LR giảm → model đã học tốt ở size hiện tại → tăng size
            should_increase_size = True
            logger.info(f"  Progressive resize: LR reduced → tăng size từ {self.current_size} → {self.sizes[self.current_stage + 1]}")
        elif val_acc is not None and self.no_improvement_counter >= self.patience:
            # Val acc plateau → tăng size để model học tiếp
            should_increase_size = True
            logger.info(f"  Progressive resize: Val acc plateau ({self.no_improvement_counter} epochs) → "
                       f"tăng size từ {self.current_size} → {self.sizes[self.current_stage + 1]}")
        
        if should_increase_size:
            self.current_stage += 1
            self.current_size = self.sizes[self.current_stage]
            self.epochs_at_current_stage = 0
            self.no_improvement_counter = 0
            self.best_val_acc_at_stage = None  # Reset để track best ở stage mới
        
        return self.current_size
    
    def get_current_size(self) -> int:
        """Get current image size."""
        return self.current_size


def get_progressive_image_size(epoch: int, total_epochs: int, sizes: list) -> int:
    """
    Legacy function for backward compatibility.
    Use ProgressiveResizeManager for new code.
    
    Get image size for progressive resizing (time-based only).
    
    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        sizes: List of image sizes to progress through
    
    Returns:
        Image size for current epoch
    """
    if not sizes or len(sizes) == 1:
        return sizes[0] if sizes else 224
    
    # Divide epochs into stages
    num_stages = len(sizes)
    
    # Handle edge case: if total_epochs < num_stages, use largest size
    if total_epochs < num_stages:
        return sizes[-1]
    
    stage_size = total_epochs // num_stages
    
    # Handle edge case: if stage_size is 0, use largest size
    if stage_size == 0:
        return sizes[-1]
    
    stage = min(epoch // stage_size, num_stages - 1)
    return sizes[stage]


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: Dict[str, Any],
    checkpoint_path: Any,
    history_plot_path: Any,
    classes: list,
    resume_epoch: int = 0,
    resume_history: Optional[Dict[str, list]] = None,
    teacher: Optional[Any] = None  # EnsembleTeacher for knowledge distillation
) -> Dict[str, list]:
    """
    Main training loop.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler
        device: Device to train on
        config: Configuration dictionary
        checkpoint_path: Path to save checkpoints
        history_plot_path: Path to save training history plot
        classes: Class names
        resume_epoch: Epoch to resume from
        resume_history: Training history to resume from
    
    Returns:
        Training history
    """
    from .utils import save_checkpoint, plot_training_history
    
    # Initialize EMA if enabled
    ema_model = None
    if config.get('use_ema', False):
        ema_model = EMAModel(model, decay=config.get('ema_decay', 0.9999))
        logger.info(f"EMA enabled with decay={config.get('ema_decay', 0.9999)}")
    
    # Initialize history
    if resume_history:
        history = resume_history
        # Ensure train_clean_loss exists in history (for backward compatibility)
        if 'train_clean_loss' not in history:
            history['train_clean_loss'] = []
        if 'train_clean_acc' not in history:
            history['train_clean_acc'] = []
        best_val_acc = max(history['val_acc']) if history['val_acc'] else 0.0
    else:
        history = {
            'train_loss': [], 'train_acc': [], 
            'train_clean_loss': [], 'train_clean_acc': [],  # Clean training loss (no augmentation)
            'val_loss': [], 'val_acc': []
        }
        best_val_acc = 0.0
    
    patience_counter = 0
    epochs = config['epochs']
    start_epoch = resume_epoch
    
    # Initialize Adaptive Distillation Disabling (universal feature)
    # Tự động áp dụng cho bất kỳ model nào có use_distillation=True
    distillation_disabled = False
    best_val_acc_with_distillation = None
    distillation_patience_counter = 0
    use_distillation_flag = config.get('use_distillation', False)
    distillation_disable_patience = config.get('distillation_disable_patience', 10)
    distillation_min_delta = config.get('distillation_min_delta', 0.005)
    
    if use_distillation_flag:
        logger.info("="*60)
        logger.info("Adaptive Distillation Disabling Enabled")
        logger.info("="*60)
        logger.info(f"  Patience: {distillation_disable_patience} epochs")
        logger.info(f"  Min delta: {distillation_min_delta*100:.2f}% (relative)")
        logger.info(f"  Will disable distillation if val acc doesn't improve for {distillation_disable_patience} epochs")
    
    # Initialize Progressive Resize Manager (if enabled)
    progressive_resize_manager = None
    if config.get('use_progressive_resize', False):
        progressive_sizes = config.get('progressive_sizes', [160, 192, 224])
        progressive_resize_manager = ProgressiveResizeManager(
            sizes=progressive_sizes,
            total_epochs=epochs,
            patience=config.get('progressive_resize_patience', 5),
            min_improvement=config.get('progressive_resize_min_improvement', 0.005),
            force_final_size_epochs=config.get('progressive_resize_force_final_epochs', 10),
            min_epochs_per_stage=config.get('progressive_resize_min_epochs_per_stage', 3)
        )
        
        # If resuming, restore progressive resize state based on img_size from config
        # Config from checkpoint should have img_size that matches the stage model was at
        if resume_epoch > 0:
            current_img_size = config.get('img_size', 224)
            # Find which stage this img_size corresponds to
            if current_img_size in progressive_sizes:
                stage_index = progressive_sizes.index(current_img_size)
                if stage_index > 0:
                    # Model was at a higher stage, restore it
                    progressive_resize_manager.current_stage = stage_index
                    progressive_resize_manager.current_size = current_img_size
                    logger.info(f"  ✓ Restored progressive resize state: stage {stage_index}, size {current_img_size}")
                else:
                    logger.info(f"  ✓ Progressive resize: starting from size {current_img_size} (stage 0)")
            else:
                # img_size doesn't match any progressive size, use closest or default
                logger.warning(f"  img_size {current_img_size} not in progressive_sizes {progressive_sizes}, using default")
        logger.info("Progressive Resize enabled (adaptive based on val acc & LR)")
        logger.info(f"  Sizes: {progressive_resize_manager.sizes}")
        logger.info(f"  Current stage: {progressive_resize_manager.current_stage}, Current size: {progressive_resize_manager.current_size}")
        logger.info(f"  Patience: {progressive_resize_manager.patience} epochs")
        logger.info(f"  Min improvement: {progressive_resize_manager.min_improvement*100:.2f}%")
        logger.info(f"  Force final size in last {progressive_resize_manager.force_final_size_epochs} epochs")
    
    # Track LR reduction for progressive resize
    previous_lr = None
    lr_reduced_this_epoch = False
    
    logger.info("="*60)
    logger.info("Starting Training")
    logger.info("="*60)
    logger.info(f"Total epochs: {epochs}")
    logger.info(f"Starting from epoch: {start_epoch + 1}")
    
    # Freeze/unfreeze backbone logic (for VideoMAE and similar models)
    freeze_backbone_epochs = config.get('freeze_backbone_epochs', 0)
    backbone_frozen = False
    if freeze_backbone_epochs > 0:
        # Freeze backbone initially
        if hasattr(model, 'videomae'):
            # VideoMAE model từ HuggingFace
            for param in model.videomae.parameters():
                param.requires_grad = False
            backbone_frozen = True
            logger.info(f"VideoMAE backbone frozen for first {freeze_backbone_epochs} epochs")
        elif hasattr(model, 'vit'):
            for param in model.vit.parameters():
                param.requires_grad = False
            backbone_frozen = True
            logger.info(f"Backbone frozen for first {freeze_backbone_epochs} epochs")
        elif hasattr(model, 'swin'):
            for param in model.swin.parameters():
                param.requires_grad = False
            backbone_frozen = True
            logger.info(f"Backbone frozen for first {freeze_backbone_epochs} epochs")
    
    for epoch in range(start_epoch, epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Unfreeze backbone after freeze_backbone_epochs
        # ⚠️ QUAN TRỌNG: Phải rebuild optimizer + giảm regularization khi unfreeze
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs and backbone_frozen:
            skip_unfreeze = config.get('skip_unfreeze', False)
            if skip_unfreeze:
                logger.info(f"⏭️  Skipping unfreeze (skip_unfreeze=True). Keeping backbone frozen + adapter/head only.")
                logger.info(f"   This is research-correct for small datasets (HMDB51).")
            else:
                # Check if partial unfreeze
                unfreeze_partial = config.get('unfreeze_partial', False)
                unfreeze_num_blocks = config.get('unfreeze_num_blocks', 6)
                
                if hasattr(model, 'videomae'):
                    # VideoMAE model từ HuggingFace
                    if unfreeze_partial:
                        # Unfreeze only last N encoder layers
                        total_blocks = len(model.videomae.encoder.layer)
                        blocks_to_unfreeze = min(unfreeze_num_blocks, total_blocks)
                        start_block = total_blocks - blocks_to_unfreeze
                        
                        logger.info(f"Unfreezing last {blocks_to_unfreeze} VideoMAE encoder layers (layers {start_block}-{total_blocks-1} of {total_blocks})")
                        for i in range(start_block, total_blocks):
                            for param in model.videomae.encoder.layer[i].parameters():
                                param.requires_grad = True
                        # Keep earlier layers frozen
                        for i in range(start_block):
                            for param in model.videomae.encoder.layer[i].parameters():
                                param.requires_grad = False
                        # Keep embeddings frozen
                        for param in model.videomae.embeddings.parameters():
                            param.requires_grad = False
                    else:
                        # Unfreeze all encoder layers (but keep embeddings frozen for safety)
                        for param in model.videomae.encoder.parameters():
                            param.requires_grad = True
                        # Optionally unfreeze embeddings too
                        # for param in model.videomae.embeddings.parameters():
                        #     param.requires_grad = True
                        logger.info(f"All VideoMAE encoder layers unfrozen at epoch {epoch + 1}")
                elif hasattr(model, 'vit'):
                    if unfreeze_partial:
                        # Unfreeze only last N blocks (SOTA approach for small datasets)
                        total_blocks = len(model.vit.blocks)
                        blocks_to_unfreeze = min(unfreeze_num_blocks, total_blocks)
                        start_block = total_blocks - blocks_to_unfreeze
                        
                        logger.info(f"Unfreezing last {blocks_to_unfreeze} ViT blocks (blocks {start_block}-{total_blocks-1} of {total_blocks})")
                        for i in range(start_block, total_blocks):
                            for param in model.vit.blocks[i].parameters():
                                param.requires_grad = True
                        # Keep earlier blocks frozen
                        for i in range(start_block):
                            for param in model.vit.blocks[i].parameters():
                                param.requires_grad = False
                    else:
                        # Unfreeze all blocks
                        for param in model.vit.parameters():
                            param.requires_grad = True
                        logger.info(f"All backbone unfrozen at epoch {epoch + 1}")
                    
                    backbone_frozen = False
                    
                    # BẮT BUỘC: Rebuild optimizer (không được reuse optimizer cũ!)
                    logger.info("Rebuilding optimizer (must reset momentum/variance stats)")
                    
                    # Collect new parameter groups
                    # FIX: Không dùng string "videomae", chỉ tách theo classifier/head vs rest
                    backbone_params = []
                    head_params = []
                    
                    for name, param in model.named_parameters():
                        if not param.requires_grad:
                            continue
                        if 'classifier' in name or 'head' in name:
                            head_params.append(param)
                        else:
                            backbone_params.append(param)
                    
                    # Use unfreeze LR (cực nhỏ cho backbone) - ĐẢM BẢO ĐÚNG CONFIG
                    base_lr_unfreeze = config.get('base_lr_unfreeze', 1e-5)
                    head_lr_unfreeze = config.get('head_lr_unfreeze', 1e-4)  # Fix: dùng 1e-4 thay vì 5e-5
                    
                    param_groups = []
                    if backbone_params:
                        param_groups.append({"params": backbone_params, "lr": base_lr_unfreeze})
                    if head_params:
                        param_groups.append({"params": head_params, "lr": head_lr_unfreeze})
                    
                    # Create NEW optimizer (reset momentum/variance)
                    optimizer = torch.optim.AdamW(
                        param_groups,
                        weight_decay=config.get('weight_decay_unfreeze', config.get('weight_decay', 0.05))
                    )
                    
                    logger.info(f"  New optimizer created:")
                    logger.info(f"     Backbone LR: {base_lr_unfreeze:.6f} (CỰC NHỎ để tránh catastrophic forgetting)")
                    logger.info(f"     Head/Adapter LR: {head_lr_unfreeze:.6f}")
                    logger.info(f"     Weight decay: {config.get('weight_decay_unfreeze', config.get('weight_decay', 0.05))}")
                    
                    # BẮT BUỘC: Rebuild scheduler với optimizer mới (không chỉ update!)
                    # Scheduler cần được rebuild để reset state và tính lại warmup/cosine từ đầu
                    from .training import get_lr_scheduler
                    from .adaptive_lr_scheduler import get_adaptive_lr_scheduler
                    
                    if config.get('use_adaptive_lr', False):
                        # Rebuild adaptive scheduler
                        scheduler = get_adaptive_lr_scheduler(
                            optimizer=optimizer,
                            num_epochs=config['epochs'],
                            warmup_epochs=config.get('warmup_epochs', 5),
                            min_lr_ratio=config.get('min_lr_ratio', 0.01),
                            plateau_patience=config.get('lr_plateau_patience', 7),
                            min_delta=config.get('lr_min_delta', 0.005),
                            mode='max',
                            verbose=True,
                            cooldown=config.get('lr_cooldown', 7),
                            threshold_mode=config.get('lr_threshold_mode', 'rel'),
                            use_val_loss=config.get('use_val_loss_for_lr', False)
                        )
                        logger.info("  Adaptive scheduler rebuilt with new optimizer")
                    else:
                        # Rebuild standard scheduler
                        cosine_start_epoch = config.get('cosine_start_epoch', config.get('warmup_epochs', 5) + 5)
                        scheduler = get_lr_scheduler(optimizer, config['epochs'], config.get('warmup_epochs', 5), cosine_start_epoch)
                        logger.info("  Standard scheduler rebuilt with new optimizer")
                    
                    # Giảm regularization khi unfreeze (tránh phá representation)
                    logger.info("📉 Reducing regularization when unfreezing (to avoid breaking VideoMAE features):")
                    config['dropout'] = config.get('dropout_unfreeze', 0.15)
                    config['drop_path_rate'] = config.get('drop_path_rate_unfreeze', 0.1)
                    config['label_smoothing'] = config.get('label_smoothing_unfreeze', 0.05)
                    config['mixup_alpha'] = config.get('mixup_alpha_unfreeze', 0.1)
                    config['cutmix_alpha'] = config.get('cutmix_alpha_unfreeze', 0.0)
                    config['use_cutmix'] = config.get('use_cutmix_unfreeze', False)
                    
                    old_dropout = config.get('dropout', 0.35)
                    old_drop_path = config.get('drop_path_rate', 0.25)
                    old_label_smooth = config.get('label_smoothing', 0.12)
                    old_mixup = config.get('mixup_alpha', 0.4)
                    old_cutmix = config.get('use_cutmix', True)
                    
                    logger.info(f"     Dropout: {config['dropout']} (giảm từ {old_dropout})")
                    logger.info(f"     Drop path: {config['drop_path_rate']} (giảm từ {old_drop_path})")
                    logger.info(f"     Label smoothing: {config['label_smoothing']} (giảm từ {old_label_smooth})")
                    logger.info(f"     Mixup alpha: {config['mixup_alpha']} (giảm từ {old_mixup})")
                    logger.info(f"     CutMix: {config['use_cutmix']} (tắt từ {old_cutmix})")
                    logger.info(f"     Weight decay: {config.get('weight_decay_unfreeze', config.get('weight_decay', 0.05))} (giảm từ {config.get('weight_decay', 0.08)})")
                    
                elif hasattr(model, 'swin'):
                    # Similar logic for Swin (if needed)
                    for param in model.swin.parameters():
                        param.requires_grad = True
                    backbone_frozen = False
                    logger.info(f"Backbone unfrozen at epoch {epoch + 1}")
                    
                    # Rebuild optimizer for Swin
                    # FIX: Không dùng string "videomae", chỉ tách theo classifier/head vs rest
                    backbone_params = []
                    head_params = []
                    
                    for name, param in model.named_parameters():
                        if not param.requires_grad:
                            continue
                        if 'classifier' in name or 'head' in name:
                            head_params.append(param)
                        else:
                            backbone_params.append(param)
                    
                    base_lr_unfreeze = config.get('base_lr_unfreeze', 1e-5)
                    head_lr_unfreeze = config.get('head_lr_unfreeze', 5e-5)
                    
                    param_groups = []
                    if backbone_params:
                        param_groups.append({"params": backbone_params, "lr": base_lr_unfreeze})
                    if head_params:
                        param_groups.append({"params": head_params, "lr": head_lr_unfreeze})
                    
                    optimizer = torch.optim.AdamW(
                        param_groups,
                        weight_decay=config.get('weight_decay_unfreeze', config.get('weight_decay', 0.05))
                    )
                    
                    # Update scheduler
                    if hasattr(scheduler, 'optimizer'):
                        scheduler.optimizer = optimizer
                    
                    # Reduce regularization
                    config['dropout'] = config.get('dropout_unfreeze', 0.15)
                    config['label_smoothing'] = config.get('label_smoothing_unfreeze', 0.05)
                    config['mixup_alpha'] = config.get('mixup_alpha_unfreeze', 0.1)
                    config['use_cutmix'] = config.get('use_cutmix_unfreeze', False)
        
        # Progressive resizing: Get current size (will be updated after validation based on val_acc)
        if progressive_resize_manager is not None:
            current_size = progressive_resize_manager.get_current_size()
            if current_size != config.get('img_size', 224):
                logger.info(f"Progressive resize: using image size {current_size}")
                # Note: This requires recreating datasets, which is complex
                # For now, we'll just log it. Full implementation would require
                # recreating datasets and loaders.
                config['img_size'] = current_size
        
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            grad_accum_steps=config['grad_accum_steps'],
            use_mixup=True,
            use_cutmix=config.get('use_cutmix', False),
            mixup_alpha=config['mixup_alpha'],
            cutmix_alpha=config.get('cutmix_alpha', 1.0),
            label_smoothing=config['label_smoothing'],
            use_focal_loss=config.get('use_focal_loss', False),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
            ema_model=ema_model,
            use_synthetic_data=config.get('use_synthetic_data', False),
            synthetic_method=config.get('synthetic_method', 'frame_mixup'),
            synthetic_ratio=config.get('synthetic_ratio', 0.3),
            teacher=teacher,  # Pass teacher for knowledge distillation
            use_distillation=use_distillation_flag and not distillation_disabled,  # Disable if adaptive logic disabled it
            distillation_temperature=config.get('distillation_temperature', 3.0),
            distillation_alpha=config.get('distillation_alpha', 0.7)
        )
        
        # Validation (use EMA model if available and shadow params đã được train tốt)
        # EMA chỉ được dùng sau epoch 3+ để tránh vấn đề shadow params = initial weights ở epoch đầu
        # Ở epoch 1-2, shadow params chưa tốt → dùng model thường để có accuracy cao hơn
        used_ema_for_validation = False
        if ema_model is not None and epoch >= 3:
            # Chỉ dùng EMA sau epoch 3 (shadow params đã được train tốt)
            ema_model.apply_shadow()
            eval_model = ema_model._model
            used_ema_for_validation = True
        else:
            # Epoch 1-2: dùng model thường (không dùng EMA)
            eval_model = model
        
        # Validation: không dùng label smoothing để đo accuracy chính xác
        # Label smoothing chỉ nên dùng cho training (regularization)
        val_loss, val_acc = evaluate(
            eval_model, val_loader, device,
            label_smoothing=0.0,  # Tắt label smoothing trong validation
            use_focal_loss=config.get('use_focal_loss', False),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0)
        )
        
        # Calculate clean training loss (no augmentation) for fair comparison with validation
        # This helps identify if validation loss being lower is due to augmentation or data issues
        # Can be disabled via config to save training time
        use_clean_train_loss = config.get('use_clean_train_loss', False)  # Default: False to save time
        if use_clean_train_loss:
            max_clean_samples = config.get('clean_train_loss_samples', None)  # None = all, or int = max samples
            train_clean_loss, train_clean_acc = evaluate_clean_training_loss(
                eval_model, train_loader, device, max_samples=max_clean_samples
            )
        else:
            train_clean_loss = None
            train_clean_acc = None
        
        # Restore original parameters if using EMA for validation
        # Chỉ restore nếu đã apply_shadow (tránh lỗi AssertionError)
        if used_ema_for_validation:
            ema_model.restore()
        
        # Track LR before update (for progressive resize)
        if progressive_resize_manager is not None:
            previous_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate (adaptive: chỉ giảm khi model không cải thiện)
        # Pass validation accuracy và loss để scheduler quyết định có giảm LR không
        if hasattr(scheduler, 'step') and hasattr(scheduler, 'best_metric'):
            # Adaptive scheduler: pass metrics and optionally loss
            if hasattr(scheduler, 'use_val_loss') and scheduler.use_val_loss:
                scheduler.step(metrics=val_acc, val_loss=val_loss)
            else:
                scheduler.step(metrics=val_acc)
        else:
            # Standard scheduler: no metrics
            scheduler.step()
        
        # Check if LR was reduced (for progressive resize)
        lr_reduced_this_epoch = False
        if progressive_resize_manager is not None and previous_lr is not None:
            current_lr = optimizer.param_groups[0]['lr']
            # LR giảm nếu current_lr < previous_lr (với tolerance nhỏ để tránh floating point issues)
            if current_lr < previous_lr - 1e-8:
                lr_reduced_this_epoch = True
        
        # Update progressive resize based on val acc and LR reduction
        if progressive_resize_manager is not None:
            new_size = progressive_resize_manager.get_size(
                epoch=epoch,
                val_acc=val_acc,
                lr_reduced=lr_reduced_this_epoch
            )
            if new_size != config['img_size']:
                logger.info(f"Progressive resize: changing image size to {new_size}")
                config['img_size'] = new_size
                # Note: Full implementation would require recreating datasets and loaders here
        
        # If LR was restored after resume, maintain it for a few epochs to avoid instability
        if config.get('restored_lr_backbone') is not None:
            epochs_to_maintain = config.get('maintain_restored_lr_epochs', 0)
            epochs_since_resume = epoch - resume_epoch
            if epochs_since_resume < epochs_to_maintain:
                # Restore LR to maintain stability in early epochs after resume
                optimizer.param_groups[0]['lr'] = config['restored_lr_backbone']
                if config.get('restored_lr_head') is not None and len(optimizer.param_groups) > 1:
                    optimizer.param_groups[1]['lr'] = config['restored_lr_head']
            elif epochs_since_resume == epochs_to_maintain:
                # Clear restored LR flags after maintenance period
                config.pop('restored_lr_backbone', None)
                config.pop('restored_lr_head', None)
                logger.info(f"  LR maintenance period ended, scheduler will control LR normally")
                
                # Sync current_lr_multiplier với LR thực tế (cho adaptive scheduler)
                if hasattr(scheduler, 'current_lr_multiplier') and hasattr(scheduler, 'base_lrs'):
                    actual_multiplier = optimizer.param_groups[0]['lr'] / scheduler.base_lrs[0]
                    scheduler.current_lr_multiplier = actual_multiplier
                    logger.info(f"  Synced current_lr_multiplier to {actual_multiplier:.4f} (LR: {optimizer.param_groups[0]['lr']:.6f})")
        
        # Get LR from all param groups (backbone and adapter/head may have different LRs)
        lr_strs = [f"{pg['lr']:.6f}" for pg in optimizer.param_groups]
        if len(lr_strs) > 1:
            current_lr_str = f"{lr_strs[0]} (backbone), {lr_strs[1]} (adapter/head)"
        else:
            current_lr_str = lr_strs[0]
        current_lr = optimizer.param_groups[0]['lr']  # For history
        
        
        # Adaptive Distillation Disabling Logic (universal feature)
        # Check if distillation should be disabled based on val acc improvement
        # FIX: Update best_val_acc_with_distillation ngay cả khi không vượt threshold
        # để đảm bảo counter chỉ tăng khi thực sự không cải thiện
        if use_distillation_flag and not distillation_disabled:
            if best_val_acc_with_distillation is None:
                # First epoch with distillation: initialize tracking
                best_val_acc_with_distillation = val_acc
                distillation_patience_counter = 0
                logger.info(f"  Distillation tracking started: best_val_acc={val_acc:.4f}")
            else:
                # Store old best để check improvement
                old_best_val_acc_with_distillation = best_val_acc_with_distillation
                
                # QUAN TRỌNG: Update best_val_acc_with_distillation ngay cả khi không vượt threshold
                # Điều này đảm bảo best_val_acc_with_distillation luôn là giá trị tốt nhất
                if val_acc >= old_best_val_acc_with_distillation:
                    best_val_acc_with_distillation = val_acc
                
                # Check if val acc improved significantly (vượt threshold)
                relative_improvement = (val_acc - old_best_val_acc_with_distillation) / old_best_val_acc_with_distillation
                
                if relative_improvement > distillation_min_delta:
                    # Val acc improved significantly: reset patience counter
                    distillation_patience_counter = 0
                    logger.info(f"  Distillation: Val acc improved {relative_improvement*100:.2f}% (best: {best_val_acc_with_distillation:.4f})")
                else:
                    # Val acc didn't improve significantly: increment patience counter
                    distillation_patience_counter += 1
                    logger.info(f"  ⏳ Distillation: No significant improvement ({distillation_patience_counter}/{distillation_disable_patience})")
                    
                    # Check if patience exceeded
                    if distillation_patience_counter >= distillation_disable_patience:
                        distillation_disabled = True
                        use_distillation_flag = False
                        logger.warning("="*60)
                        logger.warning(f"DISTILLATION DISABLED after {distillation_patience_counter} epochs without significant improvement")
                        logger.warning(f"   Best val acc with distillation: {best_val_acc_with_distillation:.4f}")
                        logger.warning(f"   Current val acc: {val_acc:.4f}")
                        logger.warning(f"   Continuing training with hard labels only")
                        logger.warning("="*60)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        if use_clean_train_loss:
            history['train_clean_loss'].append(train_clean_loss)
            history['train_clean_acc'].append(train_clean_acc)
        else:
            # Append None to maintain list length (for backward compatibility with plotting)
            history['train_clean_loss'].append(None)
            history['train_clean_acc'].append(None)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Log training metrics
        if use_clean_train_loss:
            logger.info(f"  Train: Loss={train_loss:.4f} (with aug), Clean Loss={train_clean_loss:.4f} (no aug), Acc={train_acc:.4f}")
            # Warn if clean training loss is still higher than validation loss (potential data issue)
            if train_clean_loss > val_loss + 0.1:  # Significant difference (>0.1)
                logger.warning(f"  ⚠️  Clean train loss ({train_clean_loss:.4f}) > Val loss ({val_loss:.4f}) - Possible data leakage or validation set easier")
            elif train_clean_loss < val_loss:
                logger.info(f"  ✓ Clean train loss ({train_clean_loss:.4f}) < Val loss ({val_loss:.4f}) - Normal behavior")
        else:
            logger.info(f"  Train: Loss={train_loss:.4f} (with aug), Acc={train_acc:.4f}")
        logger.info(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        logger.info(f"  LR: {current_lr_str}")
        if use_distillation_flag and not distillation_disabled:
            logger.info(f"  Distillation: Enabled (patience: {distillation_patience_counter}/{distillation_disable_patience})")
        elif distillation_disabled:
            logger.info(f"  Distillation: Disabled (adaptive)")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Sync scheduler.best_metric với best_val_acc để đảm bảo consistency
            # Khi best model được lưu, scheduler cũng phải update best_metric và reset counter
            # Điều này tránh bug: best model được lưu nhưng scheduler không reset counter
            # FIX: Đảm bảo sync ngay cả khi scheduler đã được gọi trước đó (có thể đã tăng counter sai)
            if hasattr(scheduler, 'best_metric') and hasattr(scheduler, 'plateau_counter'):
                # Update best_metric ngay cả khi không đạt threshold trong scheduler logic
                # Đảm bảo scheduler state sync với best model state
                old_scheduler_best = scheduler.best_metric
                if scheduler.best_metric is None or val_acc > scheduler.best_metric:
                    scheduler.best_metric = val_acc
                    # QUAN TRỌNG: Reset counter khi có best model mới (ngay cả khi scheduler đã tăng counter trước đó)
                    scheduler.plateau_counter = 0
                    # Reset cooldown counter nếu đang trong cooldown (vì model đã cải thiện)
                    if hasattr(scheduler, 'cooldown_counter'):
                        scheduler.cooldown_counter = 0
                    # Format old_scheduler_best safely (có thể là None)
                    old_best_str = f"{old_scheduler_best:.4f}" if old_scheduler_best is not None else "None"
                    logger.info(f"  Scheduler: Updated best_metric from {old_best_str} to {val_acc:.4f}, reset plateau_counter and cooldown")
                elif val_acc == scheduler.best_metric:
                    # Nếu val_acc == best_metric (không tốt hơn nhưng cũng không tệ hơn)
                    # Không reset counter (vì không có improvement thực sự)
                    # Nhưng đảm bảo best_metric được update (đã được update trong scheduler.step())
                    pass
            
            # Save checkpoint (use EMA model if available and shadow params đã được train tốt)
            # Chỉ dùng EMA để save checkpoint sau epoch 3+
            used_ema_for_checkpoint = False
            if ema_model is not None and epoch >= 3:
                # Chỉ dùng EMA sau epoch 3 (shadow params đã được train tốt)
                ema_model.apply_shadow()
                save_model = ema_model._model
                used_ema_for_checkpoint = True
            else:
                # Epoch 1-2: dùng model thường (không dùng EMA)
                save_model = model
            
            save_checkpoint(
                model=save_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                history=history,
                best_val_acc=best_val_acc,
                train_acc=train_acc,
                classes=classes,
                config=config,
                checkpoint_path=checkpoint_path,
                ema_model=ema_model
            )
            
            # Restore original parameters if using EMA for checkpoint
            # Chỉ restore nếu đã apply_shadow (tránh lỗi AssertionError)
            if used_ema_for_checkpoint:
                ema_model.restore()
            
            logger.info(f"  ✓ Best model saved (val_acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{config['early_stop_patience']})")
        
        # Update plot after every epoch (not just when val_acc improves)
        # This ensures plot is always up-to-date with latest training progress
        plot_training_history(history, history_plot_path, config['model_id'])
        
        # Early stopping
        if patience_counter >= config['early_stop_patience']:
            logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Final plot update to ensure latest state is saved
    plot_training_history(history, history_plot_path, config['model_id'])
    
    logger.info("\n" + "="*60)
    logger.info(f"Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Model saved to: {checkpoint_path}")
    logger.info(f"Training plot saved to: {history_plot_path}")
    logger.info("="*60)
    
    return history
