"""
Synthetic data generation for video action recognition.
Includes frame-level mixup, temporal mixup, and GAN-based generation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import random


def frame_level_mixup(
    videos: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply mixup at frame level - mix frames from different videos.

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
        lam = 1.0

    batch_size = videos.size(0)
    index = torch.randperm(batch_size).to(videos.device)

    B, T, C, H, W = videos.shape

    # Mix at frame level
    mixed_videos = videos.clone()
    for t in range(T):
        # Randomly decide which frames to mix
        mix_mask = torch.rand(T, device=videos.device) < lam
        if mix_mask.any():
            mixed_videos[:, t, :, :, :] = (
                lam * videos[:, t, :, :, :] +
                (1 - lam) * videos[index, t, :, :, :]
            )

    labels_a, labels_b = labels, labels[index]
    return mixed_videos, labels_a, labels_b, lam


def temporal_mixup(
    videos: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply mixup along temporal dimension - mix temporal segments.

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
        lam = 1.0

    batch_size = videos.size(0)
    index = torch.randperm(batch_size).to(videos.device)

    B, T, C, H, W = videos.shape

    # Random temporal split point
    split_point = int(T * lam)

    # Mix temporal segments
    mixed_videos = videos.clone()
    mixed_videos[:, :split_point, :, :, :] = videos[:, :split_point, :, :, :]
    mixed_videos[:, split_point:, :, :, :] = videos[index, split_point:, :, :, :]

    labels_a, labels_b = labels, labels[index]
    return mixed_videos, labels_a, labels_b, lam


def frame_shuffle(
    videos: torch.Tensor,
    labels: torch.Tensor,
    shuffle_ratio: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly shuffle frames within videos (temporal augmentation).

    Args:
        videos: [B, T, C, H, W] video batch
        labels: [B] label batch
        shuffle_ratio: Ratio of frames to shuffle

    Returns:
        shuffled_videos, labels
    """
    B, T, C, H, W = videos.shape
    shuffled_videos = videos.clone()

    for b in range(B):
        num_shuffle = max(1, int(T * shuffle_ratio))
        shuffle_indices = torch.randperm(T)[:num_shuffle]
        original_indices = shuffle_indices.clone()
        shuffled_indices = shuffle_indices[torch.randperm(len(shuffle_indices))]

        shuffled_videos[b, shuffle_indices, :, :, :] = (
            videos[b, shuffled_indices, :, :, :]
        )

    return shuffled_videos, labels


def temporal_interpolation(
    videos: torch.Tensor,
    labels: torch.Tensor,
    interpolation_factor: float = 1.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate new frames via temporal interpolation.

    Args:
        videos: [B, T, C, H, W] video batch
        labels: [B] label batch
        interpolation_factor: Factor to increase temporal resolution

    Returns:
        interpolated_videos, labels
    """
    B, T, C, H, W = videos.shape
    new_T = int(T * interpolation_factor)

    interpolated_videos = torch.zeros(
        B, new_T, C, H, W, device=videos.device, dtype=videos.dtype
    )

    for b in range(B):
        for t_new in range(new_T):
            t_old = t_new * (T - 1) / (new_T - 1) if new_T > 1 else 0
            t_low = int(np.floor(t_old))
            t_high = min(int(np.ceil(t_old)), T - 1)
            alpha = t_old - t_low

            if t_low == t_high:
                interpolated_videos[b, t_new] = videos[b, t_low]
            else:
                interpolated_videos[b, t_new] = (
                    (1 - alpha) * videos[b, t_low] +
                    alpha * videos[b, t_high]
                )

    return interpolated_videos, labels


class SimpleVideoGAN(nn.Module):
    """
    Simple GAN for video frame generation.
    This is a basic implementation - for production, use more sophisticated GANs.
    """

    def __init__(self, latent_dim: int = 100, num_classes: int = 51):
        """
        Args:
            latent_dim: Latent dimension for generator
            num_classes: Number of action classes
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Generator
        self.generator = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * 224 * 224),  # Single frame
            nn.Tanh()
        )

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(3 * 224 * 224 + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def generate(
        self,
        num_samples: int,
        labels: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate synthetic video frames.

        Args:
            num_samples: Number of samples to generate
            labels: [num_samples] class labels
            device: Device to generate on

        Returns:
            Generated frames [num_samples, 3, 224, 224]
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        labels_onehot = torch.zeros(
            num_samples, self.num_classes, device=device
        )
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)

        z_with_labels = torch.cat([z, labels_onehot], dim=1)
        generated = self.generator(z_with_labels)
        generated = generated.view(num_samples, 3, 224, 224)

        return generated


def generate_synthetic_batch(
    videos: torch.Tensor,
    labels: torch.Tensor,
    method: str = 'frame_mixup',
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[float]]:
    """
    Generate synthetic video batch using specified method.

    Args:
        videos: [B, T, C, H, W] video batch
        labels: [B] label batch
        method: Generation method
            - 'frame_mixup': Frame-level mixup
            - 'temporal_mixup': Temporal mixup
            - 'frame_shuffle': Frame shuffling
            - 'temporal_interpolation': Temporal interpolation
        **kwargs: Additional arguments for specific methods

    Returns:
        synthetic_videos, labels (or labels_a, labels_b, lam for mixup methods)
    """
    if method == 'frame_mixup':
        alpha = kwargs.get('alpha', 0.4)
        return frame_level_mixup(videos, labels, alpha)
    elif method == 'temporal_mixup':
        alpha = kwargs.get('alpha', 0.4)
        return temporal_mixup(videos, labels, alpha)
    elif method == 'frame_shuffle':
        shuffle_ratio = kwargs.get('shuffle_ratio', 0.3)
        shuffled, labels_out = frame_shuffle(videos, labels, shuffle_ratio)
        return shuffled, labels_out, None, None
    elif method == 'temporal_interpolation':
        factor = kwargs.get('interpolation_factor', 1.5)
        interp, labels_out = temporal_interpolation(videos, labels, factor)
        return interp, labels_out, None, None
    else:
        raise ValueError(f"Unknown generation method: {method}")
