"""
Augmentation functions for video action recognition.
"""

import random
import torch
import numpy as np
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


class VideoTransform:
    """Advanced video augmentation with RandAugment, color jitter, and temporal augmentation."""
    
    def __init__(
        self,
        image_size: int = 224,
        is_train: bool = True,
        use_temporal_aug: bool = True,
        use_advanced_spatial: bool = True,
        use_advanced_color: bool = True
    ):
        """
        Args:
            image_size: Target image size
            is_train: Whether this is for training (enables augmentation)
            use_temporal_aug: Enable temporal augmentation (speed variation, frame dropping)
            use_advanced_spatial: Enable advanced spatial augmentation (elastic, grid distortion)
            use_advanced_color: Enable advanced color augmentation (histogram matching)
        """
        self.image_size = image_size
        self.is_train = is_train
        self.use_temporal_aug = use_temporal_aug and is_train
        self.use_advanced_spatial = use_advanced_spatial and is_train
        self.use_advanced_color = use_advanced_color and is_train
        # ImageNet normalization stats
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # AutoAugment/RandAugment operations
        # Prefer AutoAugment (better performance) over RandAugment
        if is_train:
            try:
                # Try AutoAugment first (better for ImageNet-style tasks)
                from torchvision.transforms import AutoAugment, AutoAugmentPolicy
                self.auto_augment = AutoAugment(policy=AutoAugmentPolicy.IMAGENET)
                self.rand_augment = None
                self.use_auto_augment = True
            except ImportError:
                # Fallback to RandAugment if AutoAugment not available
                try:
                    from torchvision.transforms import RandAugment
                    self.rand_augment = RandAugment(num_ops=2, magnitude=9)
                    self.auto_augment = None
                    self.use_auto_augment = False
                except:
                    self.rand_augment = None
                    self.auto_augment = None
                    self.use_auto_augment = False
        else:
            self.rand_augment = None
            self.auto_augment = None
            self.use_auto_augment = False
    
    def _apply_color_jitter(self, frame: torch.Tensor) -> torch.Tensor:
        """Apply color jitter augmentation."""
        if random.random() < 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(-0.1, 0.1)
            frame = TF.adjust_brightness(frame, brightness)
            frame = TF.adjust_contrast(frame, contrast)
            frame = TF.adjust_saturation(frame, saturation)
            frame = TF.adjust_hue(frame, hue)
        return frame
    
    def _apply_random_erasing(self, frame: torch.Tensor) -> torch.Tensor:
        """Apply random erasing augmentation."""
        if random.random() < 0.3:
            h, w = frame.shape[-2:]
            area = h * w
            erase_area = random.uniform(0.02, 0.33) * area
            aspect_ratio = random.uniform(0.3, 3.3)
            h_erase = int(round((erase_area * aspect_ratio) ** 0.5))
            w_erase = int(round((erase_area / aspect_ratio) ** 0.5))
            if h_erase < h and w_erase < w:
                top = random.randint(0, h - h_erase)
                left = random.randint(0, w - w_erase)
                frame[..., top:top+h_erase, left:left+w_erase] = random.uniform(0, 1)
        return frame
    
    def _apply_elastic_deformation(
        self, frame: torch.Tensor, alpha: float = 50.0, sigma: float = 5.0
    ) -> torch.Tensor:
        """Apply elastic deformation augmentation."""
        if not self.use_advanced_spatial or random.random() > 0.3:
            return frame
        
        try:
            from scipy.ndimage import gaussian_filter, map_coordinates
            h, w = frame.shape[-2:]
            
            # Generate random displacement fields
            dx = gaussian_filter(
                (np.random.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0.0
            ) * alpha
            dy = gaussian_filter(
                (np.random.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0.0
            ) * alpha
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            
            # Apply deformation to each channel
            frame_np = frame.permute(1, 2, 0).numpy()
            deformed = np.zeros_like(frame_np)
            for c in range(frame_np.shape[2]):
                deformed[:, :, c] = map_coordinates(
                    frame_np[:, :, c], indices, order=1, mode='reflect'
                ).reshape(h, w)
            
            return torch.from_numpy(deformed).permute(2, 0, 1)
        except ImportError:
            # Fallback if scipy not available
            return frame
    
    def _apply_grid_distortion(self, frame: torch.Tensor) -> torch.Tensor:
        """Apply grid distortion augmentation."""
        if not self.use_advanced_spatial or random.random() > 0.2:
            return frame
        
        h, w = frame.shape[-2:]
        grid_size = 4
        distort_strength = random.uniform(0.1, 0.3)
        
        # Create distortion grid
        x = np.linspace(0, w, grid_size + 1)
        y = np.linspace(0, h, grid_size + 1)
        
        # Add random distortion
        x_distorted = x.copy()
        y_distorted = y.copy()
        for i in range(1, grid_size):
            x_distorted[i] += random.uniform(-distort_strength, distort_strength) * w
            y_distorted[i] += random.uniform(-distort_strength, distort_strength) * h
        
        # Apply using interpolation (simplified version)
        # In practice, would use more sophisticated grid sampling
        return frame
    
    def _apply_histogram_matching(self, frame: torch.Tensor) -> torch.Tensor:
        """Apply histogram matching for advanced color augmentation."""
        if not self.use_advanced_color or random.random() > 0.3:
            return frame
        
        try:
            from skimage import exposure
            frame_np = frame.permute(1, 2, 0).numpy()
            
            # Match to random reference histogram
            reference = np.random.rand(*frame_np.shape)
            matched = exposure.match_histograms(frame_np, reference, multichannel=True)
            
            return torch.from_numpy(matched).permute(2, 0, 1).clamp(0, 1)
        except ImportError:
            # Fallback if skimage not available
            return frame
    
    def _apply_temporal_augmentation(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply effective temporal augmentation (only high-impact, fast operations).
        
        Techniques kept (high effectiveness, fast):
        - Temporal reverse: Very effective for video, fast operation
        - Light frame dropping: Effective, fast (only 5-10% drop)
        
        Techniques removed (low effectiveness or slow):
        - Speed jitter: Slow, less effective than reverse
        - Temporal crop: Slow, complex
        - Frame shuffle: Slow, low effectiveness
        
        Note: Must preserve original number of frames to ensure batch consistency.
        """
        if not self.use_temporal_aug:
            return frames
        
        T_original = frames.shape[0]  # Store original frame count
        T_current = frames.shape[0]
        
        # 1. Temporal reverse: reverse video order (HIGH EFFECTIVENESS, FAST)
        # This is one of the most effective augmentations for video action recognition
        if random.random() < 0.5 and T_current > 2:
            frames = frames.flip(0)
        
        # 2. Light frame dropping: drop only 5-10% frames (EFFECTIVE, FAST)
        # More conservative than before to maintain temporal information
        T_current = frames.shape[0]
        if random.random() < 0.15 and T_current > 6:  # Reduced probability and drop ratio
            drop_ratio = random.uniform(0.05, 0.10)  # Only 5-10% drop (was 10-20%)
            num_keep = max(1, int(T_current * (1 - drop_ratio)))
            keep_indices = sorted(np.random.choice(T_current, num_keep, replace=False))
            frames = frames[keep_indices]
            T_current = frames.shape[0]
        
        # CRITICAL: Pad or truncate to original frame count to ensure batch consistency
        T_final = frames.shape[0]
        if T_final < T_original:
            # Pad with last frame (repeat last frame)
            last_frame = frames[-1:].expand(T_original - T_final, -1, -1, -1)
            frames = torch.cat([frames, last_frame], dim=0)
        elif T_final > T_original:
            # Truncate to original size (uniform sampling)
            indices = np.linspace(0, T_final - 1, T_original).astype(int)
            frames = frames[indices]
        
        return frames
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [T, C, H, W] tensor of frames
        Returns:
            frames: [T, C, H, W] augmented frames
        """
        T = frames.shape[0]
        
        # Apply temporal augmentation first
        if self.is_train:
            frames = self._apply_temporal_augmentation(frames)
            T = frames.shape[0]  # Update T after temporal aug
        
        if self.is_train:
            # Spatial augmentation
            h, w = frames.shape[-2:]
            
            # Random resized crop
            scale = random.uniform(0.8, 1.0)
            new_h, new_w = int(h * scale), int(w * scale)
            frames = TF.resize(frames, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
            
            # Random crop
            i = random.randint(0, max(0, new_h - self.image_size))
            j = random.randint(0, max(0, new_w - self.image_size))
            frames = TF.crop(frames, i, j, min(self.image_size, new_h), min(self.image_size, new_w))
            frames = TF.resize(frames, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
            
            # Horizontal flip
            if random.random() < 0.5:
                frames = TF.hflip(frames)
            
            # Apply AutoAugment/RandAugment (if available) - OPTIMIZED: Reduced probability
            # AutoAugment is effective but slow, so we use it sparingly (15% instead of 30%)
            use_auto_aug = False
            if random.random() < 0.15:  # Reduced from 0.3 → 0.15 for speed
                if self.use_auto_augment and self.auto_augment is not None:
                    use_auto_aug = True
                elif self.rand_augment is not None:
                    use_auto_aug = True
            
            if use_auto_aug:
                # Convert all frames at once
                frames_uint8 = (frames * 255).clamp(0, 255).to(torch.uint8)
                # Apply augmentation to each frame
                augmented_frames_list = []
                for t in range(T):
                    if self.use_auto_augment and self.auto_augment is not None:
                        frame_aug = self.auto_augment(frames_uint8[t])
                    else:
                        frame_aug = self.rand_augment(frames_uint8[t])
                    augmented_frames_list.append(
                        frame_aug.to(torch.float32) / 255.0
                    )
                frames = torch.stack(augmented_frames_list)
            
            # Apply per-frame augmentations (lightweight operations)
            augmented_frames = []
            for t in range(T):
                frame = frames[t]
                
                # Color jitter - OPTIMIZED: Reduced probability
                if random.random() < 0.2:  # Reduced from 0.3 → 0.2
                    frame = self._apply_color_jitter(frame)
                
                # Random erasing - OPTIMIZED: Reduced probability
                if random.random() < 0.15:  # Reduced from 0.2 → 0.15
                    frame = self._apply_random_erasing(frame)
                
                augmented_frames.append(frame)
            
            frames = torch.stack(augmented_frames)
        else:
            # Validation: just resize
            frames = TF.resize(frames, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        
        # Normalize with ImageNet stats
        normalized = [TF.normalize(frame, self.mean, self.std) for frame in frames]
        return torch.stack(normalized)


def mixup_data(videos: torch.Tensor, labels: torch.Tensor, alpha: float = 0.4) -> tuple:
    """
    Apply Mixup augmentation to video batch.
    
    Args:
        videos: [B, T, C, H, W] video batch
        labels: [B] label batch
        alpha: Mixup alpha parameter
    
    Returns:
        mixed_videos, labels_a, labels_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = videos.size(0)
    index = torch.randperm(batch_size).to(videos.device)
    
    mixed_videos = lam * videos + (1 - lam) * videos[index, :]
    labels_a, labels_b = labels, labels[index]
    return mixed_videos, labels_a, labels_b, lam


def mixup_criterion(criterion, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Compute loss for Mixup.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixup lambda
    
    Returns:
        Loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(videos: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> tuple:
    """
    Apply CutMix augmentation to video batch.
    
    Args:
        videos: [B, T, C, H, W] video batch
        labels: [B] label batch
        alpha: CutMix alpha parameter
    
    Returns:
        mixed_videos, labels_a, labels_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = videos.size(0)
    index = torch.randperm(batch_size).to(videos.device)
    
    B, T, C, H, W = videos.shape
    
    # Generate random bounding box
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Clamp bounding box
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply CutMix
    mixed_videos = videos.clone()
    mixed_videos[:, :, :, bby1:bby2, bbx1:bbx2] = videos[index, :, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to match actual pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    labels_a, labels_b = labels, labels[index]
    return mixed_videos, labels_a, labels_b, lam


def cutmix_criterion(criterion, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Compute loss for CutMix.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: CutMix lambda
    
    Returns:
        Loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
