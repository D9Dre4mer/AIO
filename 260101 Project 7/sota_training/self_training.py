"""
Self-training for video action recognition.
Iterative training with high-confidence predictions.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfidenceDataset(Dataset):
    """Dataset with confidence scores for self-training."""

    def __init__(
        self,
        videos: List[torch.Tensor],
        labels: List[int],
        confidences: List[float]
    ):
        """
        Args:
            videos: List of video tensors [T, C, H, W]
            labels: List of labels
            confidences: List of confidence scores
        """
        self.videos = videos
        self.labels = labels
        self.confidences = confidences

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return self.videos[idx], self.labels[idx], self.confidences[idx]


def get_high_confidence_predictions(
    model: nn.Module,
    unlabeled_loader: DataLoader,
    device: torch.device,
    confidence_threshold: float = 0.9,
    top_k: Optional[int] = None
) -> Tuple[List[torch.Tensor], List[int], List[float]]:
    """
    Get high-confidence predictions from unlabeled data.

    Args:
        model: Model to generate predictions
        unlabeled_loader: DataLoader for unlabeled data
        device: Device to run on
        confidence_threshold: Minimum confidence threshold
        top_k: If specified, return top-k most confident predictions

    Returns:
        videos, labels, confidences
    """
    model.eval()
    videos = []
    labels = []
    confidences = []

    with torch.no_grad():
        for batch_videos, _ in unlabeled_loader:
            batch_videos = batch_videos.to(device, non_blocking=True)

            # Get predictions
            logits = model(batch_videos)
            probs = torch.softmax(logits, dim=1)
            max_probs, pred_labels = torch.max(probs, dim=1)

            # Filter by confidence
            high_conf_mask = max_probs >= confidence_threshold
            if high_conf_mask.any():
                high_conf_videos = batch_videos[high_conf_mask].cpu()
                high_conf_labels = pred_labels[high_conf_mask].cpu().numpy()
                high_conf_scores = max_probs[high_conf_mask].cpu().numpy()

                videos.extend([v for v in high_conf_videos])
                labels.extend(high_conf_labels.tolist())
                confidences.extend(high_conf_scores.tolist())

    # Sort by confidence and take top-k if specified
    if top_k is not None and len(videos) > top_k:
        sorted_indices = np.argsort(confidences)[::-1][:top_k]
        videos = [videos[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
        confidences = [confidences[i] for i in sorted_indices]

    logger.info(
        f"Selected {len(videos)} high-confidence samples "
        f"(threshold={confidence_threshold})"
    )

    return videos, labels, confidences


def self_training_iteration(
    model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    confidence_threshold: float = 0.9,
    pseudo_label_ratio: float = 0.5,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Tuple[float, int]:
    """
    Perform one iteration of self-training.

    Args:
        model: Model to train
        labeled_loader: Labeled training data
        unlabeled_loader: Unlabeled data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        confidence_threshold: Confidence threshold for pseudo-labels
        pseudo_label_ratio: Ratio of pseudo-labeled data to use
        scaler: Gradient scaler for mixed precision

    Returns:
        Average loss, number of pseudo-labeled samples added
    """
    model.train()

    # Get high-confidence predictions
    pseudo_videos, pseudo_labels, pseudo_confidences = (
        get_high_confidence_predictions(
            model, unlabeled_loader, device, confidence_threshold
        )
    )

    # Limit pseudo-labeled samples
    num_pseudo = int(len(pseudo_videos) * pseudo_label_ratio)
    if num_pseudo > 0:
        pseudo_videos = pseudo_videos[:num_pseudo]
        pseudo_labels = pseudo_labels[:num_pseudo]
        pseudo_confidences = pseudo_confidences[:num_pseudo]

    # Create pseudo-labeled dataset
    if len(pseudo_videos) > 0:
        pseudo_dataset = ConfidenceDataset(
            pseudo_videos, pseudo_labels, pseudo_confidences
        )
        pseudo_loader = DataLoader(
            pseudo_dataset,
            batch_size=labeled_loader.batch_size,
            shuffle=True,
            num_workers=labeled_loader.num_workers
        )
    else:
        pseudo_loader = None

    # Training loop
    total_loss = 0.0
    num_batches = 0

    from tqdm.auto import tqdm

    # Combine labeled and pseudo-labeled data
    labeled_iter = iter(labeled_loader)
    if pseudo_loader is not None:
        pseudo_iter = iter(pseudo_loader)
        use_pseudo = True
    else:
        use_pseudo = False

    # Train on labeled data
    try:
        while True:
            videos, labels = next(labeled_iter)
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(videos)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(videos)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Add pseudo-labeled batch if available
            if use_pseudo:
                try:
                    pseudo_videos, pseudo_labels, _ = next(pseudo_iter)
                    pseudo_videos = pseudo_videos.to(device, non_blocking=True)
                    pseudo_labels = torch.tensor(
                        pseudo_labels, device=device, dtype=torch.long
                    )

                    optimizer.zero_grad()

                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            pseudo_logits = model(pseudo_videos)
                            pseudo_loss = criterion(pseudo_logits, pseudo_labels)
                        scaler.scale(pseudo_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        pseudo_logits = model(pseudo_videos)
                        pseudo_loss = criterion(pseudo_logits, pseudo_labels)
                        pseudo_loss.backward()
                        optimizer.step()

                    total_loss += pseudo_loss.item()
                    num_batches += 1
                except StopIteration:
                    use_pseudo = False

    except StopIteration:
        pass

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    num_pseudo_added = len(pseudo_videos) if len(pseudo_videos) > 0 else 0

    return avg_loss, num_pseudo_added


def self_training_loop(
    model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_iterations: int = 5,
    confidence_threshold: float = 0.9,
    confidence_increase: float = 0.05,
    pseudo_label_ratio: float = 0.5,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> List[float]:
    """
    Perform multiple iterations of self-training.

    Args:
        model: Model to train
        labeled_loader: Labeled training data
        unlabeled_loader: Unlabeled data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        num_iterations: Number of self-training iterations
        confidence_threshold: Initial confidence threshold
        confidence_increase: Increase threshold each iteration
        pseudo_label_ratio: Ratio of pseudo-labeled data to use
        scaler: Gradient scaler for mixed precision

    Returns:
        List of losses for each iteration
    """
    losses = []
    current_threshold = confidence_threshold

    for iteration in range(num_iterations):
        logger.info(
            f"Self-training iteration {iteration + 1}/{num_iterations} "
            f"(threshold={current_threshold:.2f})"
        )

        loss, num_pseudo = self_training_iteration(
            model, labeled_loader, unlabeled_loader, optimizer, criterion,
            device, current_threshold, pseudo_label_ratio, scaler
        )

        losses.append(loss)
        logger.info(
            f"Iteration {iteration + 1}: loss={loss:.4f}, "
            f"pseudo-labeled={num_pseudo}"
        )

        # Increase confidence threshold
        current_threshold = min(0.99, current_threshold + confidence_increase)

    return losses
