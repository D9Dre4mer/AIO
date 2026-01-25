"""
SOTA Training Pipeline - Video Action Recognition with ViT

A modular package for training SOTA Vision Transformer models for video action recognition
with advanced techniques including EMA, CutMix, Focal Loss, Progressive Resizing, and Enhanced TTA.
"""

__version__ = "1.0.0"

from .models import SOTAViTForAction, TemporalAttention, Adapter, EMAModel
from .dataset import VideoDataset, TestDataset, FilteredVideoDataset
from .augmentation import (
    VideoTransform,
    mixup_data,
    mixup_criterion,
    cutmix_data,
    cutmix_criterion,
)
from .losses import FocalLoss, LabelSmoothingCrossEntropy
from .training import train_one_epoch, evaluate, get_lr_scheduler, train_model
from .expert_training import train_global_residual_experts
from .sequential_layer_training import (
    sequential_layer_training_loop,
    final_head_training,
    unfreeze_single_layer,
    freeze_single_layer,
    freeze_all_backbone_layers
)
from .inference import enhanced_tta, test_time_augment, run_inference, generate_submission
from .utils import (
    setup_logging,
    plot_training_history,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    # Models
    'SOTAViTForAction',
    'TemporalAttention',
    'Adapter',
    'EMAModel',
    # Datasets
    'VideoDataset',
    'TestDataset',
    'FilteredVideoDataset',
    # Augmentation
    'VideoTransform',
    'mixup_data',
    'mixup_criterion',
    'cutmix_data',
    'cutmix_criterion',
    # Losses
    'FocalLoss',
    'LabelSmoothingCrossEntropy',
    # Training
    'train_one_epoch',
    'evaluate',
    'get_lr_scheduler',
    'train_model',
    'train_global_residual_experts',
    # Sequential Layer Training
    'sequential_layer_training_loop',
    'final_head_training',
    'unfreeze_single_layer',
    'freeze_single_layer',
    'freeze_all_backbone_layers',
    # Inference
    'enhanced_tta',
    'test_time_augment',
    'run_inference',
    'generate_submission',
    # Utils
    'setup_logging',
    'plot_training_history',
    'save_checkpoint',
    'load_checkpoint',
]
