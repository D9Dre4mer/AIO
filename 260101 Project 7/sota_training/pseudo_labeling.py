"""
Pseudo-labeling for video action recognition.
Label test set with best ensemble model and retrain.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PseudoLabeledDataset(Dataset):
    """Dataset with pseudo-labeled data."""

    def __init__(
        self,
        videos: List[torch.Tensor],
        labels: List[int],
        confidences: List[float]
    ):
        """
        Args:
            videos: List of video tensors [T, C, H, W]
            labels: List of pseudo-labels
            confidences: List of confidence scores
        """
        self.videos = videos
        self.labels = labels
        self.confidences = confidences

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return self.videos[idx], self.labels[idx], self.confidences[idx]


def generate_pseudo_labels(
    models: List[nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    confidence_threshold: float = 0.8,
    ensemble_method: str = 'weighted_avg_probs',
    weights: Optional[List[float]] = None
) -> Tuple[List[torch.Tensor], List[int], List[float]]:
    """
    Generate pseudo-labels for test set using ensemble models.

    Args:
        models: List of ensemble models
        test_loader: Test data loader
        device: Device to run on
        confidence_threshold: Minimum confidence for pseudo-labels
        ensemble_method: Ensemble method ('weighted_avg_probs', 'voting')
        weights: Weights for each model (default: uniform)

    Returns:
        videos, labels, confidences
    """
    for model in models:
        model.eval()

    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    videos = []
    labels = []
    confidences = []

    import torch.nn.functional as F
    from tqdm.auto import tqdm

    with torch.no_grad():
        progress = tqdm(test_loader, desc="Generating Pseudo-Labels")
        for batch_videos, video_ids in progress:
            batch_videos = batch_videos.to(device, non_blocking=True)
            B = batch_videos.shape[0]

            # Get predictions from all models
            all_logits = []
            for model in models:
                logits = model(batch_videos)
                all_logits.append(logits)

            # Ensemble predictions
            if ensemble_method == 'weighted_avg_probs':
                ensemble_probs = torch.zeros_like(
                    F.softmax(all_logits[0], dim=1)
                )
                for logits, weight in zip(all_logits, weights):
                    probs = F.softmax(logits, dim=1)
                    ensemble_probs += weight * probs

                max_probs, pred_labels = torch.max(ensemble_probs, dim=1)

            elif ensemble_method == 'voting':
                all_preds = [logits.argmax(dim=1) for logits in all_logits]
                all_preds = torch.stack(all_preds)  # [num_models, B]
                ensemble_preds = torch.mode(all_preds, dim=0)[0]

                # Get confidence from average probabilities
                ensemble_probs = torch.zeros_like(
                    F.softmax(all_logits[0], dim=1)
                )
                for logits, weight in zip(all_logits, weights):
                    probs = F.softmax(logits, dim=1)
                    ensemble_probs += weight * probs

                pred_labels = ensemble_preds
                max_probs = torch.gather(
                    ensemble_probs, 1, pred_labels.unsqueeze(1)
                ).squeeze(1)

            else:
                raise ValueError(f"Unknown ensemble method: {ensemble_method}")

            # Filter by confidence
            high_conf_mask = max_probs >= confidence_threshold
            if high_conf_mask.any():
                high_conf_videos = batch_videos[high_conf_mask].cpu()
                high_conf_labels = pred_labels[high_conf_mask].cpu().numpy()
                high_conf_scores = max_probs[high_conf_mask].cpu().numpy()

                videos.extend([v for v in high_conf_videos])
                labels.extend(high_conf_labels.tolist())
                confidences.extend(high_conf_scores.tolist())

            progress.set_postfix(
                total=len(videos),
                conf_thresh=confidence_threshold
            )

    logger.info(
        f"Generated {len(videos)} pseudo-labels "
        f"(threshold={confidence_threshold})"
    )

    return videos, labels, confidences


def save_pseudo_labels(
    videos: List[torch.Tensor],
    labels: List[int],
    confidences: List[float],
    output_path: Path,
    class_names: List[str]
):
    """
    Save pseudo-labeled data to file.

    Args:
        videos: List of video tensors
        labels: List of labels
        confidences: List of confidence scores
        output_path: Path to save pseudo-labels
        class_names: List of class names
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'video_idx': list(range(len(videos))),
        'label': [class_names[l] for l in labels],
        'label_idx': labels,
        'confidence': confidences
    }

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved {len(videos)} pseudo-labels to {output_path}")


def load_pseudo_labels(
    pseudo_label_path: Path,
    test_dataset: Dataset
) -> PseudoLabeledDataset:
    """
    Load pseudo-labels and create dataset.

    Args:
        pseudo_label_path: Path to pseudo-label CSV
        test_dataset: Original test dataset

    Returns:
        PseudoLabeledDataset
    """
    df = pd.read_csv(pseudo_label_path)

    videos = []
    labels = []
    confidences = []

    for _, row in df.iterrows():
        idx = row['video_idx']
        label = int(row['label_idx'])
        confidence = float(row['confidence'])

        # Get video from test dataset
        video, _ = test_dataset[idx]
        videos.append(video)
        labels.append(label)
        confidences.append(confidence)

    return PseudoLabeledDataset(videos, labels, confidences)


def retrain_with_pseudo_labels(
    model: nn.Module,
    train_loader: DataLoader,
    pseudo_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    pseudo_weight: float = 0.5,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
    """
    Retrain model with pseudo-labeled data.

    Args:
        model: Model to train
        train_loader: Original training data
        pseudo_loader: Pseudo-labeled data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        pseudo_weight: Weight for pseudo-label loss
        scaler: Gradient scaler for mixed precision

    Returns:
        Average loss
    """
    model.train()

    total_loss = 0.0
    num_batches = 0

    from tqdm.auto import tqdm

    train_iter = iter(train_loader)
    pseudo_iter = iter(pseudo_loader)

    try:
        while True:
            # Train on labeled data
            try:
                videos, labels = next(train_iter)
                videos = videos.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = model(videos)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                else:
                    logits = model(videos)
                    loss = criterion(logits, labels)
                    loss.backward()

                total_loss += loss.item()
                num_batches += 1

            except StopIteration:
                break

            # Train on pseudo-labeled data
            try:
                pseudo_videos, pseudo_labels, _ = next(pseudo_iter)
                pseudo_videos = pseudo_videos.to(device, non_blocking=True)
                pseudo_labels = torch.tensor(
                    pseudo_labels, device=device, dtype=torch.long
                )

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        pseudo_logits = model(pseudo_videos)
                        pseudo_loss = (
                            pseudo_weight * criterion(pseudo_logits, pseudo_labels)
                        )
                    scaler.scale(pseudo_loss).backward()
                else:
                    pseudo_logits = model(pseudo_videos)
                    pseudo_loss = (
                        pseudo_weight * criterion(pseudo_logits, pseudo_labels)
                    )
                    pseudo_loss.backward()

                total_loss += pseudo_loss.item()
                num_batches += 1

            except StopIteration:
                pass

            # Update weights
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

    except StopIteration:
        pass

    return total_loss / num_batches if num_batches > 0 else 0.0
