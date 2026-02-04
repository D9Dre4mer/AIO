"""
Dataset classes for video action recognition.
"""

import re
import random
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from .augmentation import VideoTransform

logger = logging.getLogger(__name__)


def _safe_is_dir(p: Path) -> bool:
    """Trả về True nếu p là thư mục; bắt OSError (path quá dài, permission, v.v.)."""
    try:
        return p.is_dir()
    except (OSError, PermissionError):
        return False


def _pad_frames_to_same_size(frames: List[torch.Tensor]) -> torch.Tensor:
    """
    Pad list of (C, H, W) tensors to same H, W with black (0); return (T, C, H, W).
    Dùng khi ảnh trong clip khác size (vd person_focus crop) để stack được.
    """
    if not frames:
        return torch.zeros(0)
    max_h = max(f.shape[1] for f in frames)
    max_w = max(f.shape[2] for f in frames)
    out = []
    for f in frames:
        c, h, w = f.shape
        if h == max_h and w == max_w:
            out.append(f)
            continue
        padded = torch.zeros(c, max_h, max_w, dtype=f.dtype, device=f.device)
        padded[:, :h, :w] = f
        out.append(padded)
    return torch.stack(out)


class VideoDataset(Dataset):
    """Video dataset with grouped train/val split to avoid data leakage."""
    
    def __init__(
        self,
        root: Path,
        num_frames: int = 16,
        frame_stride: int = 2,
        image_size: int = 224,
        is_train: bool = True,
        val_ratio: float = 0.15,
        seed: int = 42,
        samples: Optional[List[Tuple[List[Path], int]]] = None,
        use_temporal_aug: bool = True,
        use_advanced_spatial: bool = True,
        use_advanced_color: bool = True
    ):
        """
        Args:
            root: Root directory containing class folders
            num_frames: Number of frames to sample per video
            frame_stride: Stride for frame sampling
            image_size: Target image size
            is_train: Whether this is training dataset
            val_ratio: Validation split ratio
            seed: Random seed for splitting
            samples: Pre-computed samples (for train/val split)
        """
        self.root = Path(root)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.transform = VideoTransform(
            image_size, is_train,
            use_temporal_aug=use_temporal_aug if is_train else False,
            use_advanced_spatial=use_advanced_spatial if is_train else False,
            use_advanced_color=use_advanced_color if is_train else False
        )
        self.to_tensor = transforms.ToTensor()
        
        # Get classes - CRITICAL: Use sorted() to ensure consistent ordering
        # This ensures the same class-to-index mapping across different runs
        class_dirs = [d.name for d in self.root.iterdir() if _safe_is_dir(d)]
        self.classes = sorted(class_dirs)  # Sort alphabetically for consistency
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        
        # Validate: Ensure we have classes
        if len(self.classes) == 0:
            raise ValueError(f"No class directories found in {self.root}")
        
        if samples is None:
            # Collect all samples grouped by video
            grouped_samples = {}  # {(class, base_video_name): [(frame_paths, label), ...]}
            
            for cls in self.classes:
                cls_dir = self.root / cls
                for video_dir in sorted([d for d in cls_dir.iterdir() if _safe_is_dir(d)]):
                    frame_paths = sorted([
                        p for p in video_dir.iterdir()
                        if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
                    ])
                    if frame_paths:
                        # Extract base video name (remove trailing _N)
                        base_name = self._base_video_name(video_dir.name)
                        key = (cls, base_name)
                        if key not in grouped_samples:
                            grouped_samples[key] = []
                        grouped_samples[key].append((frame_paths, self.class_to_idx[cls]))
            
            # Grouped split to avoid data leakage
            group_keys = list(grouped_samples.keys())
            rng = random.Random(seed)
            indices = list(range(len(group_keys)))
            rng.shuffle(indices)
            split_point = int(len(indices) * (1 - val_ratio))
            
            if is_train:
                selected_groups = indices[:split_point]
            else:
                selected_groups = indices[split_point:]
            
            # Collect samples from selected groups
            self.samples = []
            for idx in selected_groups:
                self.samples.extend(grouped_samples[group_keys[idx]])
        else:
            # Use provided samples (for train/val split)
            self.samples = samples
    
    @staticmethod
    def _base_video_name(name: str) -> str:
        """Remove trailing _N from video name for grouping."""
        match = re.match(r"(.+)_\d+$", name)
        return match.group(1) if match else name
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _select_indices(self, total: int) -> torch.Tensor:
        """Select frame indices for sampling.
        
        Strategy:
        - If video has enough frames: sample uniformly with stride
        - If video has fewer frames: reduce stride to use more unique frames
        - Only pad with last frame as last resort (minimize duplicates)
        """
        if total <= 0:
            raise ValueError("No frames")
        if total == 1:
            return torch.zeros(self.num_frames, dtype=torch.long)
        
        # Calculate how many frames we can get with current stride
        max_frames_with_stride = (total + self.frame_stride - 1) // self.frame_stride
        
        # If we need more frames than available with current stride, reduce stride
        if max_frames_with_stride < self.num_frames:
            # Reduce stride to get more frames (minimum stride = 1)
            # Calculate stride needed to get at least num_frames
            required_stride = max(1, total // self.num_frames)
            # Use the smaller of current stride and required stride
            effective_stride = min(self.frame_stride, required_stride)
        else:
            effective_stride = self.frame_stride
        
        # Sample frames with effective stride
        steps = max(self.num_frames * effective_stride, self.num_frames)
        grid = torch.linspace(0, total - 1, steps=min(steps, total))
        idxs = grid[::effective_stride].long()
        
        # Remove duplicates and ensure indices are within bounds
        idxs = torch.unique(idxs)
        idxs = idxs[idxs < total]
        
        # If still not enough frames, pad with last frame (only as last resort)
        if idxs.numel() < self.num_frames:
            pad_count = self.num_frames - idxs.numel()
            pad = idxs.new_full((pad_count,), idxs[-1].item() if idxs.numel() > 0 else 0)
            idxs = torch.cat([idxs, pad], dim=0)
        
        return idxs[:self.num_frames]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a video sample."""
        frame_paths, label = self.samples[idx]
        total = len(frame_paths)
        idxs = self._select_indices(total)
        frames = []
        for i in idxs:
            path = frame_paths[int(i.item())]
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    frames.append(self.to_tensor(img))
            except (OSError, IOError) as e:
                logger.warning(f"Failed to load image {path}: {e}. Using black frame as fallback.")
                # Create a black frame as fallback
                black_frame = torch.zeros(3, self.image_size, self.image_size)
                frames.append(black_frame)
            except Exception as e:
                logger.warning(f"Unexpected error loading image {path}: {e}. Using black frame as fallback.")
                black_frame = torch.zeros(3, self.image_size, self.image_size)
                frames.append(black_frame)
        video = _pad_frames_to_same_size(frames)
        video = self.transform(video)
        return video, label


class TestDataset(Dataset):
    """Test dataset for inference."""
    
    def __init__(
        self,
        root: Path,
        num_frames: int = 16,
        frame_stride: int = 2,
        image_size: int = 224,
        num_clips: int = 10  # Standard: 10 clips per video for evaluation
    ):
        """
        Args:
            root: Root directory containing video folders (named by ID)
            num_frames: Number of frames to sample per video
            frame_stride: Stride for frame sampling
            image_size: Target image size
            num_clips: Number of clips to sample per video (default: 10 for standard evaluation)
        """
        self.root = Path(root)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.transform = VideoTransform(image_size, is_train=False)
        self.to_tensor = transforms.ToTensor()
        
        # Get video directories sorted by ID
        self.video_dirs = sorted(
            [d for d in self.root.iterdir() if d.is_dir()],
            key=lambda x: int(x.name)
        )
        self.video_ids = [int(d.name) for d in self.video_dirs]
        
        # Multi-clip support: 10 clips per video (standard for evaluation)
        self.num_clips = num_clips
        # Create expanded_indices: (video_idx, clip_idx) for each sample
        self.expanded_indices = []
        for video_idx in range(len(self.video_dirs)):
            for clip_idx in range(self.num_clips):
                self.expanded_indices.append((video_idx, clip_idx))
    
    def __len__(self) -> int:
        return len(self.expanded_indices)
    
    def _select_indices(self, total: int) -> torch.Tensor:
        """Select frame indices for sampling.
        
        Strategy:
        - If video has enough frames: sample uniformly with stride
        - If video has fewer frames: reduce stride to use more unique frames
        - Only pad with last frame as last resort (minimize duplicates)
        """
        if total <= 0:
            raise ValueError("No frames")
        if total == 1:
            return torch.zeros(self.num_frames, dtype=torch.long)
        
        # Calculate how many frames we can get with current stride
        max_frames_with_stride = (total + self.frame_stride - 1) // self.frame_stride
        
        # If we need more frames than available with current stride, reduce stride
        if max_frames_with_stride < self.num_frames:
            # Reduce stride to get more frames (minimum stride = 1)
            # Calculate stride needed to get at least num_frames
            required_stride = max(1, total // self.num_frames)
            # Use the smaller of current stride and required stride
            effective_stride = min(self.frame_stride, required_stride)
        else:
            effective_stride = self.frame_stride
        
        # Sample frames with effective stride
        steps = max(self.num_frames * effective_stride, self.num_frames)
        grid = torch.linspace(0, total - 1, steps=min(steps, total))
        idxs = grid[::effective_stride].long()
        
        # Remove duplicates and ensure indices are within bounds
        idxs = torch.unique(idxs)
        idxs = idxs[idxs < total]
        
        # If still not enough frames, pad with last frame (only as last resort)
        if idxs.numel() < self.num_frames:
            pad_count = self.num_frames - idxs.numel()
            pad = idxs.new_full((pad_count,), idxs[-1].item() if idxs.numel() > 0 else 0)
            idxs = torch.cat([idxs, pad], dim=0)
        
        return idxs[:self.num_frames]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a video sample."""
        video_idx, clip_idx = self.expanded_indices[idx]
        video_dir = self.video_dirs[video_idx]
        video_id = self.video_ids[video_idx]
        frame_paths = sorted([
            p for p in video_dir.iterdir()
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        ])
        total = len(frame_paths)
        
        # For multi-clip: sample different temporal segments
        if self.num_clips > 1:
            # Divide video into num_clips segments and sample from segment clip_idx
            segment_size = total / self.num_clips
            start_idx = int(segment_size * clip_idx)
            end_idx = int(segment_size * (clip_idx + 1)) if clip_idx < self.num_clips - 1 else total
            # Sample from this segment
            segment_frame_paths = frame_paths[start_idx:end_idx]
            segment_total = len(segment_frame_paths)
            if segment_total > 0:
                idxs = self._select_indices(segment_total)
                # Map back to original indices
                idxs = torch.tensor([start_idx + int(i.item()) for i in idxs])
            else:
                # Fallback: use original method
                idxs = self._select_indices(total)
        else:
            idxs = self._select_indices(total)
        frames = []
        for i in idxs:
            path = frame_paths[int(i.item())]
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    frames.append(self.to_tensor(img))
            except (OSError, IOError) as e:
                logger.warning(f"Failed to load image {path}: {e}. Using black frame as fallback.")
                # Create a black frame as fallback
                black_frame = torch.zeros(3, self.image_size, self.image_size)
                frames.append(black_frame)
            except Exception as e:
                logger.warning(f"Unexpected error loading image {path}: {e}. Using black frame as fallback.")
                black_frame = torch.zeros(3, self.image_size, self.image_size)
                frames.append(black_frame)
        video = _pad_frames_to_same_size(frames)
        video = self.transform(video)
        return video, video_id


class FilteredVideoDataset(Dataset):
    """Filtered VideoDataset that only includes samples with labels in specified subset."""
    
    def __init__(
        self,
        base_dataset: VideoDataset,
        label_indices: List[int]
    ):
        """
        Args:
            base_dataset: Base VideoDataset to filter
            label_indices: List of label indices to include (e.g., [0, 1, 2, 3, 4, 5, 6])
        """
        self.base_dataset = base_dataset
        self.label_indices = set(label_indices)
        
        # Filter samples to only include those with labels in label_indices
        self.filtered_samples = [
            (frame_paths, label) 
            for frame_paths, label in base_dataset.samples 
            if label in self.label_indices
        ]
        
        # Keep reference to base dataset attributes
        self.root = base_dataset.root
        self.num_frames = base_dataset.num_frames
        self.frame_stride = base_dataset.frame_stride
        self.image_size = base_dataset.image_size
        self.transform = base_dataset.transform
        self.to_tensor = base_dataset.to_tensor
        self.classes = base_dataset.classes
        self.class_to_idx = base_dataset.class_to_idx
    
    def __len__(self) -> int:
        return len(self.filtered_samples)
    
    def _select_indices(self, total: int) -> torch.Tensor:
        """Delegate to base dataset's method."""
        return self.base_dataset._select_indices(total)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a video sample from filtered dataset."""
        frame_paths, label = self.filtered_samples[idx]
        total = len(frame_paths)
        idxs = self._select_indices(total)
        frames = []
        for i in idxs:
            path = frame_paths[int(i.item())]
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    frames.append(self.to_tensor(img))
            except (OSError, IOError) as e:
                logger.warning(f"Failed to load image {path}: {e}. Using black frame as fallback.")
                black_frame = torch.zeros(3, self.image_size, self.image_size)
                frames.append(black_frame)
            except Exception as e:
                logger.warning(f"Unexpected error loading image {path}: {e}. Using black frame as fallback.")
                black_frame = torch.zeros(3, self.image_size, self.image_size)
                frames.append(black_frame)
        video = _pad_frames_to_same_size(frames)
        video = self.transform(video)
        return video, label
