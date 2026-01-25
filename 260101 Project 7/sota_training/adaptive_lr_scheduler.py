"""
Adaptive Learning Rate Scheduler - Chỉ giảm LR khi model không còn cải thiện.
"""

import torch
import numpy as np
from typing import Optional


class AdaptiveCosineAnnealingLR:
    """
    Cosine annealing scheduler với adaptive logic:
    - Chỉ giảm LR khi validation accuracy không cải thiện
    - Giữ LR hiện tại khi model vẫn đang học tốt
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        warmup_epochs: int = 5,
        min_lr_ratio: float = 0.01,
        plateau_patience: int = 3,
        min_delta: float = 0.0001,
        mode: str = 'max',  # 'max' for accuracy, 'min' for loss
        verbose: bool = True,
        cooldown: int = 5,  # Cooldown period after LR reduction
        threshold_mode: str = 'rel',  # 'rel' (relative) or 'abs' (absolute)
        use_val_loss: bool = False  # Also consider validation loss
    ):
        """
        Args:
            optimizer: Optimizer
            num_epochs: Total number of epochs
            warmup_epochs: Number of warmup epochs
            min_lr_ratio: Minimum LR as ratio of base_lr (default: 0.01 = 1%)
            plateau_patience: Số epochs không cải thiện trước khi giảm LR
            min_delta: Minimum change để coi là "cải thiện" (default: 0.0001 = 0.01%)
            mode: 'max' for accuracy (higher is better), 'min' for loss (lower is better)
            verbose: Print messages when LR is reduced
        """
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr_ratio = min_lr_ratio
        self.plateau_patience = plateau_patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.cooldown = cooldown
        self.threshold_mode = threshold_mode
        self.use_val_loss = use_val_loss
        
        # Store base learning rates for each parameter group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Tracking state
        self.last_epoch = -1
        self.best_metric = None
        self.best_val_loss = None  # Track best validation loss
        self.plateau_counter = 0
        self.cooldown_counter = 0
        self.current_lr_multiplier = 1.0  # Current LR multiplier (0-1)
        self.recent_metrics = []  # Track recent metrics for trend analysis
        
    def step(self, metrics: Optional[float] = None, val_loss: Optional[float] = None):
        """
        Update learning rate.
        
        Args:
            metrics: Validation accuracy (if mode='max') or loss (if mode='min')
                     If None, will use cosine schedule regardless of performance
            val_loss: Validation loss (optional, used if use_val_loss=True)
        """
        # Increment last_epoch (but don't go beyond num_epochs - 1)
        if self.last_epoch < self.num_epochs - 1:
            self.last_epoch += 1
        else:
            # Already at last epoch, don't increment
            return
        
        # Warmup phase: always increase LR
        # QUAN TRỌNG: Chỉ warmup nếu last_epoch < warmup_epochs
        # Nếu last_epoch >= warmup_epochs, KHÔNG warmup (skip warmup phase)
        if self.last_epoch < self.warmup_epochs:
            multiplier = (self.last_epoch + 1) / self.warmup_epochs
            if self.verbose:
                print(f"  🔥 Warmup phase: epoch {self.last_epoch}, multiplier: {multiplier:.4f}, LR: {self.base_lrs[0] * multiplier:.6f}")
            self._update_lr(multiplier)
            return
        # Nếu last_epoch >= warmup_epochs, skip warmup và tiếp tục với adaptive logic
        
        # Cooldown period: skip LR reduction during cooldown
        # FIX: Nếu model cải thiện trong cooldown, reset cooldown ngay lập tức
        if self.cooldown_counter > 0:
            if metrics is not None:
                # Check if model is improving during cooldown
                is_improving = self._is_improving(metrics)
                if val_loss is not None and self.use_val_loss:
                    is_improving_loss = self._is_improving_loss(val_loss)
                    if self.use_val_loss and val_loss is not None:
                        is_improving = is_improving or is_improving_loss
                
                # Nếu model cải thiện trong cooldown → reset cooldown ngay
                if is_improving:
                    self.cooldown_counter = 0
                    if self.verbose:
                        print(f"  ✅ Model cải thiện trong cooldown → Reset cooldown")
                    # Tiếp tục với logic bình thường (không return)
                else:
                    # Model không cải thiện → tiếp tục cooldown
                    self.cooldown_counter -= 1
                    return
            else:
                # No metrics → just decrement cooldown
                self.cooldown_counter -= 1
                return
        
        # After warmup: adaptive cosine annealing
        if metrics is not None:
            # Check if model is improving (accuracy)
            is_improving_acc = self._is_improving(metrics)
            
            # Also check validation loss if enabled
            is_improving_loss = True
            if val_loss is not None and self.use_val_loss:
                is_improving_loss = self._is_improving_loss(val_loss)
            
            # Model cải thiện nếu EITHER accuracy tăng HOẶC loss giảm (nếu dùng loss)
            if self.use_val_loss and val_loss is not None:
                is_improving = is_improving_acc or is_improving_loss
            else:
                is_improving = is_improving_acc
            
            # Track recent metrics for trend analysis
            self.recent_metrics.append(metrics)
            if len(self.recent_metrics) > 10:
                self.recent_metrics.pop(0)
            
            if is_improving:
                # Model đang cải thiện → giữ LR hiện tại (không giảm)
                self.plateau_counter = 0
                # QUAN TRỌNG: Đảm bảo LR trong optimizer match với current_lr_multiplier * base_lrs
                # Vì khi không có improvement, _update_lr() không được gọi, nên LR có thể không sync
                self._update_lr(self.current_lr_multiplier)
                if self.verbose and self.last_epoch % 5 == 0:  # Log occasionally
                    current_lr = self.optimizer.param_groups[0]['lr']
                    acc_str = f"acc: {self.best_metric:.4f}" if self.best_metric else "N/A"
                    loss_str = f", loss: {self.best_val_loss:.4f}" if self.best_val_loss and self.use_val_loss else ""
                    print(f"  ✅ Model đang cải thiện ({acc_str}{loss_str}) → Giữ LR: {current_lr:.6f}")
            else:
                # Model không cải thiện → tăng plateau counter
                self.plateau_counter += 1
                
                if self.plateau_counter >= self.plateau_patience:
                    # Đã không cải thiện trong N epochs → giảm LR
                    # Dùng factor 0.1 (giảm 90%) - chuẩn PyTorch ReduceLROnPlateau
                    # Factor 0.1 giúp model escape local minimum tốt hơn so với 0.5
                    # Không giảm xuống dưới min_lr_ratio
                    new_multiplier = max(
                        self.min_lr_ratio,
                        self.current_lr_multiplier * 0.1  # Giảm 90% (chuẩn PyTorch)
                    )
                    
                    # Nếu new_multiplier quá gần min_lr_ratio, dùng cosine schedule để smooth
                    # Nhưng clamp để không nhảy quá lớn (tối đa giảm 90% từ current, phù hợp với factor 0.1)
                    if new_multiplier <= self.min_lr_ratio * 1.5:
                        # Gần min → dùng cosine schedule
                        progress = (self.last_epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)
                        cosine_multiplier = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
                        # Clamp để không nhảy quá lớn (tối đa giảm 90% từ current, tối thiểu = min_lr_ratio)
                        cosine_multiplier = max(self.min_lr_ratio, min(cosine_multiplier, self.current_lr_multiplier * 0.1))
                        new_multiplier = cosine_multiplier
                    
                    self.current_lr_multiplier = new_multiplier
                    self._update_lr(new_multiplier)
                    self.plateau_counter = 0  # Reset counter after reducing LR
                    self.cooldown_counter = self.cooldown  # Start cooldown
                    
                    if self.verbose:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        print(f"  ⚠️  Model không cải thiện trong {self.plateau_patience} epochs → Giảm LR: {current_lr:.6f} (cooldown: {self.cooldown} epochs)")
                else:
                    # Chưa đủ patience → giữ LR hiện tại
                    # QUAN TRỌNG: Đảm bảo LR trong optimizer match với current_lr_multiplier * base_lrs
                    # Vì khi không có improvement, _update_lr() không được gọi, nên LR có thể không sync
                    self._update_lr(self.current_lr_multiplier)
                    if self.verbose and self.plateau_counter == 1:
                        print(f"  📊 Model không cải thiện (counter: {self.plateau_counter}/{self.plateau_patience}) - Giữ LR hiện tại")
        else:
            # No metrics provided → use standard cosine schedule
            progress = (self.last_epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)
            cosine_multiplier = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
            self.current_lr_multiplier = cosine_multiplier
            self._update_lr(cosine_multiplier)
    
    def _is_improving(self, metrics: float) -> bool:
        """Check if metrics is improving and update best_metric.
        
        QUAN TRỌNG: 
        - best_metric phải luôn được update khi metrics tốt hơn hoặc bằng (để sync với training.py)
        - is_improving chỉ True khi vượt threshold (để reset counter đúng)
        """
        if self.best_metric is None:
            self.best_metric = metrics
            return True
        
        # Calculate threshold (dùng best_metric CŨ, trước khi update)
        if self.threshold_mode == 'rel':
            # Relative threshold: min_delta % của best_metric
            threshold = self.best_metric * self.min_delta
        else:
            # Absolute threshold
            threshold = self.min_delta
        
        # Store old best_metric để check improvement
        old_best_metric = self.best_metric
        
        # Check if metrics is better than best (for updating best_metric)
        # QUAN TRỌNG: Update best_metric ngay cả khi không vượt threshold
        # Điều này đảm bảo best_metric luôn là giá trị tốt nhất từ trước đến nay (sync với training.py)
        if self.mode == 'max':
            # For accuracy: higher is better
            if metrics >= old_best_metric:
                self.best_metric = metrics
        else:
            # For loss: lower is better
            if metrics <= old_best_metric:
                self.best_metric = metrics
        
        # Check if improving enough to reset plateau counter (cần vượt threshold)
        # FIX: Dùng old_best_metric để check (trước khi update)
        is_improving = False
        if self.mode == 'max':
            # For accuracy: higher is better
            # Cải thiện nếu metrics > old_best_metric + threshold
            is_improving = metrics > (old_best_metric + threshold)
        else:
            # For loss: lower is better
            # Cải thiện nếu metrics < old_best_metric - threshold
            is_improving = metrics < (old_best_metric - threshold)
        
        return is_improving
    
    def _is_improving_loss(self, val_loss: float) -> bool:
        """Check if validation loss is improving."""
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            return True
        
        # Calculate threshold (relative to best loss)
        if self.threshold_mode == 'rel':
            threshold = self.best_val_loss * self.min_delta
        else:
            threshold = self.min_delta
        
        # For loss: lower is better
        is_improving = val_loss < (self.best_val_loss - threshold)
        
        # Update best_val_loss if improving
        if is_improving:
            self.best_val_loss = val_loss
        
        return is_improving
    
    def _update_lr(self, multiplier: float):
        """Update learning rate for all parameter groups."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[i] * multiplier
    
    def get_last_lr(self):
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Get scheduler state."""
        return {
            'last_epoch': self.last_epoch,
            'best_metric': self.best_metric,
            'best_val_loss': self.best_val_loss,
            'plateau_counter': self.plateau_counter,
            'cooldown_counter': self.cooldown_counter,
            'current_lr_multiplier': self.current_lr_multiplier,
            'base_lrs': self.base_lrs,
            'recent_metrics': self.recent_metrics
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.last_epoch = state_dict.get('last_epoch', -1)
        self.best_metric = state_dict.get('best_metric', None)
        self.best_val_loss = state_dict.get('best_val_loss', None)
        self.plateau_counter = state_dict.get('plateau_counter', 0)
        self.cooldown_counter = state_dict.get('cooldown_counter', 0)
        self.current_lr_multiplier = state_dict.get('current_lr_multiplier', 1.0)
        self.base_lrs = state_dict.get('base_lrs', self.base_lrs)
        self.recent_metrics = state_dict.get('recent_metrics', [])


def get_adaptive_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    warmup_epochs: int = 5,
    min_lr_ratio: float = 0.01,
    plateau_patience: int = 3,
    min_delta: float = 0.0001,
    mode: str = 'max',
    verbose: bool = True,
    cooldown: int = 5,
    threshold_mode: str = 'rel',
    use_val_loss: bool = False
) -> AdaptiveCosineAnnealingLR:
    """
    Create adaptive LR scheduler that only reduces LR when model stops improving.
    
    Args:
        optimizer: Optimizer
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        min_lr_ratio: Minimum LR as ratio of base_lr (default: 0.01 = 1%)
        plateau_patience: Số epochs không cải thiện trước khi giảm LR (default: 3)
        min_delta: Minimum change để coi là "cải thiện" (default: 0.0001 = 0.01%)
        mode: 'max' for accuracy (higher is better), 'min' for loss (lower is better)
        verbose: Print messages when LR is reduced
    
    Returns:
        Adaptive LR scheduler
    """
    return AdaptiveCosineAnnealingLR(
        optimizer=optimizer,
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
        min_lr_ratio=min_lr_ratio,
        plateau_patience=plateau_patience,
        min_delta=min_delta,
        mode=mode,
        verbose=verbose,
        cooldown=cooldown,
        threshold_mode=threshold_mode,
        use_val_loss=use_val_loss
    )
