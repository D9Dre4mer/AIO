"""
Hyperparameter optimization using Optuna with TPE.
Automated search for optimal hyperparameters.
"""

import optuna
from typing import Dict, Any, Callable, Optional
import logging
import torch

logger = logging.getLogger(__name__)


def create_optuna_study(
    direction: str = 'maximize',
    study_name: str = 'hyperparameter_search',
    storage: Optional[str] = None
) -> optuna.Study:
    """
    Create Optuna study for hyperparameter optimization.

    Args:
        direction: Optimization direction ('maximize' or 'minimize')
        study_name: Name of the study
        storage: Storage URL (None for in-memory)

    Returns:
        Optuna study
    """
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        sampler=sampler,
        storage=storage
    )
    return study


def suggest_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Suggest hyperparameters for a trial.

    Args:
        trial: Optuna trial

    Returns:
        Dictionary of suggested hyperparameters
    """
    config = {
        # Learning rates
        'base_lr': trial.suggest_float('base_lr', 1e-5, 1e-3, log=True),
        'head_lr': trial.suggest_float('head_lr', 1e-4, 1e-2, log=True),

        # Regularization
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.1, log=True),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.3),

        # Augmentation
        'mixup_alpha': trial.suggest_float('mixup_alpha', 0.2, 0.8),
        'cutmix_alpha': trial.suggest_float('cutmix_alpha', 0.5, 1.5),

        # Training
        'batch_size': trial.suggest_categorical('batch_size', [16, 24, 32]),
        'warmup_epochs': trial.suggest_int('warmup_epochs', 3, 10),

        # LR scheduling
        'lr_plateau_patience': trial.suggest_int('lr_plateau_patience', 3, 10),
        'lr_min_delta': trial.suggest_float('lr_min_delta', 0.001, 0.01),
        'lr_reduction_factor': trial.suggest_float(
            'lr_reduction_factor', 0.05, 0.5
        ),

        # Early stopping
        'early_stop_patience': trial.suggest_int('early_stop_patience', 5, 15),
    }

    return config


def optimize_hyperparameters(
    objective_fn: Callable[[optuna.Trial], float],
    n_trials: int = 50,
    timeout: Optional[float] = None,
    study_name: str = 'hyperparameter_search'
) -> optuna.Study:
    """
    Optimize hyperparameters using Optuna.

    Args:
        objective_fn: Objective function that takes a trial and returns score
        n_trials: Number of trials to run
        timeout: Timeout in seconds (None for no timeout)
        study_name: Name of the study

    Returns:
        Optimized study
    """
    study = create_optuna_study(study_name=study_name)

    study.optimize(
        objective_fn,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return study


def get_best_hyperparameters(study: optuna.Study) -> Dict[str, Any]:
    """
    Get best hyperparameters from study.

    Args:
        study: Optuna study

    Returns:
        Dictionary of best hyperparameters
    """
    return study.best_params


def create_objective_function(
    train_fn: Callable,
    val_fn: Callable,
    model_fn: Callable,
    train_loader,
    val_loader,
    device: torch.device,
    num_classes: int = 51
) -> Callable[[optuna.Trial], float]:
    """
    Create objective function for Optuna.

    Args:
        train_fn: Training function
        val_fn: Validation function
        model_fn: Function to create model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_classes: Number of classes

    Returns:
        Objective function
    """
    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        config = suggest_hyperparameters(trial)

        # Create model
        model = model_fn(num_classes=num_classes).to(device)

        # Create optimizer with suggested learning rates
        optimizer = torch.optim.AdamW(
            [
                {'params': [p for n, p in model.named_parameters()
                           if 'head' in n or 'temporal' in n or 'adapter' in n],
                 'lr': config['head_lr']},
                {'params': [p for n, p in model.named_parameters()
                           if 'head' not in n and 'temporal' not in n and 'adapter' not in n],
                 'lr': config['base_lr']}
            ],
            weight_decay=config['weight_decay']
        )

        # Train model (simplified - in practice, use full training loop)
        best_val_acc = 0.0
        num_epochs = 10  # Reduced for hyperparameter search

        for epoch in range(num_epochs):
            # Train
            train_fn(model, train_loader, optimizer, device, config)

            # Validate
            val_acc = val_fn(model, val_loader, device)

            # Report to Optuna
            trial.report(val_acc, epoch)

            # Prune if needed
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        return best_val_acc

    return objective
