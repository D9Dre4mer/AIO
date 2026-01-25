"""
Knowledge Distillation for video action recognition.
Teacher-student training with ViT-Large ensemble teaching ViT-Base models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss combining:
    - Hard target loss (ground truth labels)
    - Soft target loss (teacher predictions)
    """

    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.7,
        base_criterion: nn.Module = None
    ):
        """
        Args:
            temperature: Temperature for softmax (higher = softer distribution)
            alpha: Weight for soft target loss (1-alpha for hard target)
            base_criterion: Base loss function (default: CrossEntropy)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.base_criterion = (
            base_criterion if base_criterion is not None
            else nn.CrossEntropyLoss()
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distillation loss.

        Args:
            student_logits: [B, num_classes] student predictions
            teacher_logits: [B, num_classes] teacher predictions
            labels: [B] ground truth labels

        Returns:
            Combined loss
        """
        # Soft target loss (KL divergence)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Hard target loss
        hard_loss = self.base_criterion(student_logits, labels)

        # Combined loss
        total_loss = (
            self.alpha * soft_loss +
            (1 - self.alpha) * hard_loss
        )

        return total_loss


class EnsembleTeacher:
    """
    Ensemble of teacher models for knowledge distillation.
    """

    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        """
        Args:
            models: List of teacher models
            weights: Weights for each model (default: uniform)
        """
        self.models = models
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights

        # Set to eval mode
        for model in self.models:
            model.eval()

    def predict(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Get ensemble predictions from teachers.

        Args:
            videos: [B, T, C, H, W] video batch

        Returns:
            Ensemble logits [B, num_classes]
        """
        all_logits = []

        with torch.no_grad():
            for model, weight in zip(self.models, self.weights):
                logits = model(videos)
                all_logits.append(weight * logits)

        # Weighted average
        ensemble_logits = sum(all_logits)

        return ensemble_logits

    def to(self, device: torch.device):
        """Move all models to device."""
        for model in self.models:
            model.to(device)
        return self


def train_with_distillation(
    student_model: nn.Module,
    teacher: EnsembleTeacher,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 3.0,
    alpha: float = 0.7,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
    """
    Train student model with knowledge distillation.

    Args:
        student_model: Student model to train
        teacher: EnsembleTeacher instance
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        temperature: Distillation temperature
        alpha: Weight for soft target loss
        scaler: Gradient scaler for mixed precision

    Returns:
        Average loss
    """
    student_model.train()
    criterion = DistillationLoss(temperature=temperature, alpha=alpha)

    total_loss = 0.0
    num_batches = 0

    from tqdm.auto import tqdm
    progress = tqdm(train_loader, desc="Distillation Training")

    for videos, labels in progress:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Get teacher predictions
        with torch.no_grad():
            teacher_logits = teacher.predict(videos)

        # Get student predictions
        if scaler is not None:
            with torch.cuda.amp.autocast():
                student_logits = student_model(videos)
                loss = criterion(student_logits, teacher_logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            student_logits = student_model(videos)
            loss = criterion(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / num_batches if num_batches > 0 else 0.0


def create_teacher_from_checkpoints(
    checkpoint_paths: List[str],
    model_class,
    num_classes: int = 51,
    device: torch.device = None
) -> EnsembleTeacher:
    """
    Create ensemble teacher from checkpoint paths.

    Args:
        checkpoint_paths: List of checkpoint file paths
        model_class: Model class to instantiate
        num_classes: Number of classes
        device: Device to load models on

    Returns:
        EnsembleTeacher instance
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = []
    for checkpoint_path in checkpoint_paths:
        model = model_class(num_classes=num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        models.append(model)

    return EnsembleTeacher(models)
