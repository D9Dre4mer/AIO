"""
Prediction calibration for video action recognition.
Temperature scaling and Platt scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for calibration.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Temperature parameter (learned if trainable)
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: [B, num_classes] logits

        Returns:
            Scaled logits [B, num_classes]
        """
        return logits / self.temperature

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
        lr: float = 0.01,
        max_iter: int = 50
    ):
        """
        Calibrate temperature on validation set.

        Args:
            logits: [N, num_classes] validation logits
            labels: [N] validation labels
            device: Device to train on
            lr: Learning rate
            max_iter: Maximum iterations
        """
        self.to(device)
        logits = logits.to(device)
        labels = labels.to(device)

        # Optimizer
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            scaled_logits = self(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        logger.info(f"Calibrated temperature: {self.temperature.item():.4f}")


class PlattScaling:
    """
    Platt scaling (logistic regression) for calibration.
    """

    def __init__(self):
        self.model = LogisticRegression()

    def calibrate(
        self,
        logits: np.ndarray,
        labels: np.ndarray
    ):
        """
        Calibrate using Platt scaling.

        Args:
            logits: [N, num_classes] validation logits
            labels: [N] validation labels
        """
        # Convert to probabilities
        probs = F.softmax(torch.FloatTensor(logits), dim=1).numpy()

        # Train logistic regression on max probabilities
        max_probs = probs.max(axis=1)
        self.model.fit(max_probs.reshape(-1, 1), labels)

        logger.info("Platt scaling calibrated")

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """
        Get calibrated probabilities.

        Args:
            logits: [N, num_classes] logits

        Returns:
            Calibrated probabilities [N, num_classes]
        """
        probs = F.softmax(torch.FloatTensor(logits), dim=1).numpy()
        max_probs = probs.max(axis=1)

        # Apply Platt scaling
        calibrated_max = self.model.predict_proba(max_probs.reshape(-1, 1))[:, 1]

        # Adjust probabilities
        calibrated_probs = probs.copy()
        for i in range(len(probs)):
            pred_class = probs[i].argmax()
            calibrated_probs[i, pred_class] = calibrated_max[i]
            # Renormalize
            calibrated_probs[i] = calibrated_probs[i] / calibrated_probs[i].sum()

        return calibrated_probs


def calibrate_predictions(
    model: nn.Module,
    val_loader,
    device: torch.device,
    method: str = 'temperature',
    num_classes: int = 51
) -> nn.Module:
    """
    Calibrate model predictions on validation set.

    Args:
        model: Model to calibrate
        val_loader: Validation data loader
        device: Device to run on
        method: Calibration method ('temperature' or 'platt')
        num_classes: Number of classes

    Returns:
        Calibrated model or calibrator
    """
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(videos)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if method == 'temperature':
        calibrator = TemperatureScaling()
        calibrator.calibrate(all_logits, all_labels, device)
        return calibrator
    elif method == 'platt':
        calibrator = PlattScaling()
        calibrator.calibrate(all_logits.numpy(), all_labels.numpy())
        return calibrator
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def apply_calibration(
    logits: torch.Tensor,
    calibrator,
    method: str = 'temperature'
) -> torch.Tensor:
    """
    Apply calibration to logits.

    Args:
        logits: [B, num_classes] logits
        calibrator: Calibrator (TemperatureScaling or PlattScaling)
        method: Calibration method

    Returns:
        Calibrated probabilities [B, num_classes]
    """
    if method == 'temperature':
        scaled_logits = calibrator(logits)
        return F.softmax(scaled_logits, dim=1)
    elif method == 'platt':
        calibrated_probs = calibrator.predict_proba(logits.numpy())
        return torch.FloatTensor(calibrated_probs)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
