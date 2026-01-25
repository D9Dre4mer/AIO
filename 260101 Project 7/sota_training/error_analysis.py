"""
Error analysis for video action recognition.
Identify hard samples and targeted retraining.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def analyze_errors(
    model: nn.Module,
    data_loader,
    device: torch.device,
    class_names: List[str]
) -> Dict[str, any]:
    """
    Analyze prediction errors.

    Args:
        model: Model to analyze
        data_loader: Data loader
        device: Device to run on
        class_names: List of class names

    Returns:
        Dictionary with error analysis results
    """
    model.eval()

    errors = []
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    hard_samples = []

    with torch.no_grad():
        for videos, labels in data_loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(videos)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                confidence = probs[i, pred_label].item()

                if true_label != pred_label:
                    errors.append({
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'confidence': confidence,
                        'video_idx': len(errors)
                    })
                    confusion_matrix[true_label][pred_label] += 1

                    # Hard sample: low confidence or high confidence but wrong
                    if confidence < 0.5 or (confidence > 0.8 and true_label != pred_label):
                        hard_samples.append({
                            'video': videos[i].cpu(),
                            'true_label': true_label,
                            'pred_label': pred_label,
                            'confidence': confidence
                        })

    # Class-wise error rates
    class_errors = {}
    for true_class in range(len(class_names)):
        total = sum(confusion_matrix[true_class].values())
        if total > 0:
            class_errors[class_names[true_class]] = {
                'total_errors': total,
                'most_confused_with': max(
                    confusion_matrix[true_class].items(),
                    key=lambda x: x[1]
                )[0] if confusion_matrix[true_class] else None
            }

    return {
        'total_errors': len(errors),
        'error_rate': len(errors) / len(data_loader.dataset),
        'confusion_matrix': dict(confusion_matrix),
        'class_errors': class_errors,
        'hard_samples': hard_samples,
        'errors': errors
    }


def identify_hard_samples(
    model: nn.Module,
    data_loader,
    device: torch.device,
    confidence_threshold: float = 0.5,
    max_samples: int = 100
) -> List[Dict]:
    """
    Identify hard samples for targeted retraining.

    Args:
        model: Model to analyze
        data_loader: Data loader
        device: Device to run on
        confidence_threshold: Confidence threshold for hard samples
        max_samples: Maximum number of hard samples to return

    Returns:
        List of hard samples
    """
    model.eval()

    hard_samples = []

    with torch.no_grad():
        for videos, labels in data_loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(videos)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            max_probs = probs.max(dim=1)[0]

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                confidence = max_probs[i].item()

                # Hard sample criteria:
                # 1. Low confidence
                # 2. Wrong prediction with high confidence (overconfident)
                # 3. Correct but low confidence (uncertain)
                is_hard = (
                    confidence < confidence_threshold or
                    (pred_label != true_label and confidence > 0.7) or
                    (pred_label == true_label and confidence < 0.6)
                )

                if is_hard:
                    hard_samples.append({
                        'video': videos[i].cpu(),
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'confidence': confidence
                    })

                    if len(hard_samples) >= max_samples:
                        return hard_samples

    logger.info(f"Identified {len(hard_samples)} hard samples")
    return hard_samples


def create_hard_sample_dataset(hard_samples: List[Dict]) -> torch.utils.data.Dataset:
    """
    Create dataset from hard samples.

    Args:
        hard_samples: List of hard samples

    Returns:
        Dataset
    """
    from torch.utils.data import Dataset

    class HardSampleDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            return sample['video'], sample['true_label']

    return HardSampleDataset(hard_samples)


def targeted_retraining(
    model: nn.Module,
    hard_sample_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 5,
    weight: float = 2.0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
    """
    Retrain model focusing on hard samples.

    Args:
        model: Model to retrain
        hard_sample_loader: Data loader for hard samples
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        num_epochs: Number of epochs
        weight: Weight for hard sample loss
        scaler: Gradient scaler for mixed precision

    Returns:
        Average loss
    """
    model.train()

    total_loss = 0.0
    num_batches = 0

    from tqdm.auto import tqdm

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress = tqdm(hard_sample_loader, desc=f"Hard Sample Training Epoch {epoch+1}")

        for videos, labels in progress:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(videos)
                    loss = weight * criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(videos)
                loss = weight * criterion(logits, labels)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        total_loss += epoch_loss
        logger.info(f"Epoch {epoch+1}: loss={epoch_loss/len(hard_sample_loader):.4f}")

    return total_loss / num_batches if num_batches > 0 else 0.0


def save_error_analysis(
    error_analysis: Dict,
    output_path: Path
):
    """
    Save error analysis results.

    Args:
        error_analysis: Error analysis results
        output_path: Path to save results
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    serializable = {
        'total_errors': error_analysis['total_errors'],
        'error_rate': float(error_analysis['error_rate']),
        'confusion_matrix': {
            str(k): {str(k2): v2 for k2, v2 in v.items()}
            for k, v in error_analysis['confusion_matrix'].items()
        },
        'class_errors': error_analysis['class_errors']
    }

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"Saved error analysis to {output_path}")
