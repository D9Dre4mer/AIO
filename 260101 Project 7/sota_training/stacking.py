"""
Stacking ensemble for video action recognition.
Meta-learner (LightGBM/XGBoost) to combine base models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")


class NeuralMetaLearner(nn.Module):
    """
    Neural network meta-learner for stacking.
    """

    def __init__(
        self,
        num_base_models: int,
        num_classes: int = 51,
        hidden_dim: int = 128
    ):
        """
        Args:
            num_base_models: Number of base models
            num_classes: Number of classes
            hidden_dim: Hidden dimension
        """
        super().__init__()
        self.num_base_models = num_base_models
        self.num_classes = num_classes

        # Input: [B, num_base_models * num_classes] (probabilities from all models)
        self.net = nn.Sequential(
            nn.Linear(num_base_models * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, base_predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            base_predictions: [B, num_base_models, num_classes] probabilities

        Returns:
            Final predictions [B, num_classes]
        """
        B = base_predictions.shape[0]
        # Flatten: [B, num_base_models * num_classes]
        x = base_predictions.view(B, -1)
        return self.net(x)


def get_base_predictions(
    models: List[nn.Module],
    data_loader,
    device: torch.device,
    use_tta: bool = False
) -> np.ndarray:
    """
    Get predictions from all base models.

    Args:
        models: List of base models
        data_loader: Data loader
        device: Device to run on
        use_tta: Whether to use test-time augmentation

    Returns:
        Predictions [N, num_models, num_classes]
    """
    for model in models:
        model.eval()

    all_predictions = []

    with torch.no_grad():
        from tqdm.auto import tqdm
        progress = tqdm(data_loader, desc="Base Predictions")

        for videos, _ in progress:
            videos = videos.to(device, non_blocking=True)
            B = videos.shape[0]

            batch_predictions = []

            for model in models:
                if use_tta:
                    # Simple TTA: original + flipped
                    logits_orig = model(videos)
                    logits_flip = model(torch.flip(videos, dims=[-1]))
                    logits = (logits_orig + logits_flip) / 2
                else:
                    logits = model(videos)

                probs = F.softmax(logits, dim=1)
                batch_predictions.append(probs.cpu().numpy())

            # Stack: [B, num_models, num_classes]
            batch_predictions = np.stack(batch_predictions, axis=1)
            all_predictions.append(batch_predictions)

    # Concatenate: [N, num_models, num_classes]
    all_predictions = np.concatenate(all_predictions, axis=0)

    return all_predictions


def train_lightgbm_meta_learner(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    num_classes: int = 51
) -> lgb.Booster:
    """
    Train LightGBM meta-learner.

    Args:
        X_train: [N, num_models * num_classes] training features
        y_train: [N] training labels
        X_val: [M, num_models * num_classes] validation features
        y_val: [M] validation labels
        num_classes: Number of classes

    Returns:
        Trained LightGBM model
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not available")

    # Reshape for LightGBM: [N, num_models * num_classes]
    if len(X_train.shape) == 3:
        N, num_models, num_classes_in = X_train.shape
        X_train = X_train.reshape(N, -1)

    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train)

    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    valid_sets = [train_data]
    valid_names = ['train']

    if X_val is not None:
        if len(X_val.shape) == 3:
            X_val = X_val.reshape(X_val.shape[0], -1)
        val_data = lgb.Dataset(X_val, label=y_val)
        valid_sets.append(val_data)
        valid_names.append('val')

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)]
    )

    return model


def train_xgboost_meta_learner(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    num_classes: int = 51
) -> xgb.Booster:
    """
    Train XGBoost meta-learner.

    Args:
        X_train: [N, num_models * num_classes] training features
        y_train: [N] training labels
        X_val: [M, num_models * num_classes] validation features
        y_val: [M] validation labels
        num_classes: Number of classes

    Returns:
        Trained XGBoost model
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available")

    # Reshape for XGBoost
    if len(X_train.shape) == 3:
        X_train = X_train.reshape(X_train.shape[0], -1)

    params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'eval_metric': 'mlogloss',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)

    evals = [(dtrain, 'train')]
    if X_val is not None:
        if len(X_val.shape) == 3:
            X_val = X_val.reshape(X_val.shape[0], -1)
        dval = xgb.DMatrix(X_val, label=y_val)
        evals.append((dval, 'val'))

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=10
    )

    return model


def train_neural_meta_learner(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    num_classes: int = 51,
    num_epochs: int = 50,
    device: torch.device = None
) -> NeuralMetaLearner:
    """
    Train neural network meta-learner.

    Args:
        X_train: [N, num_models, num_classes] training predictions
        y_train: [N] training labels
        X_val: [M, num_models, num_classes] validation predictions
        y_val: [M] validation labels
        num_classes: Number of classes
        num_epochs: Number of training epochs
        device: Device to train on

    Returns:
        Trained neural meta-learner
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_base_models = X_train.shape[1]
    model = NeuralMetaLearner(num_base_models, num_classes).to(device)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)

    if X_val is not None:
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    from tqdm.auto import tqdm

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train_tensor)
        loss = criterion(logits, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Validation
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == y_val_tensor).float().mean().item()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: loss={loss.item():.4f}, "
                    f"val_acc={val_acc:.4f}"
                )

    return model


def stacking_predict(
    base_models: List[nn.Module],
    meta_learner,
    data_loader,
    device: torch.device,
    meta_learner_type: str = 'lightgbm',
    use_tta: bool = False
) -> np.ndarray:
    """
    Make predictions using stacking ensemble.

    Args:
        base_models: List of base models
        meta_learner: Trained meta-learner
        data_loader: Data loader
        device: Device to run on
        meta_learner_type: Type of meta-learner ('lightgbm', 'xgboost', 'neural')
        use_tta: Whether to use TTA for base models

    Returns:
        Final predictions [N, num_classes]
    """
    # Get base predictions
    base_preds = get_base_predictions(base_models, data_loader, device, use_tta)

    # Meta-learner prediction
    if meta_learner_type == 'lightgbm':
        if len(base_preds.shape) == 3:
            base_preds = base_preds.reshape(base_preds.shape[0], -1)
        final_preds = meta_learner.predict(base_preds)
        # Convert to probabilities
        final_preds = final_preds / final_preds.sum(axis=1, keepdims=True)

    elif meta_learner_type == 'xgboost':
        if len(base_preds.shape) == 3:
            base_preds = base_preds.reshape(base_preds.shape[0], -1)
        dtest = xgb.DMatrix(base_preds)
        final_preds = meta_learner.predict(dtest)

    elif meta_learner_type == 'neural':
        base_preds_tensor = torch.FloatTensor(base_preds).to(device)
        meta_learner.eval()
        with torch.no_grad():
            logits = meta_learner(base_preds_tensor)
            final_preds = F.softmax(logits, dim=1).cpu().numpy()

    else:
        raise ValueError(f"Unknown meta-learner type: {meta_learner_type}")

    return final_preds
