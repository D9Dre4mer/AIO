"""
Train VideoMAE Model 11 with Global + Residual Experts (Multi-Head).

Design:
- Backbone (VideoMAE) frozen
- Global head: predicts all 51 classes
- Residual experts: each predicts a subset and is added into global logits

Training schedule:
1) Stage A: train global head on full 51 labels
2) Stage B: train each expert head on its label subset (dataset filtered)
3) Stage C: joint finetune global + all experts on full 51 labels (calibration)
"""

import os
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from sota_training.config import get_default_config
from sota_training.dataset import VideoDataset
from sota_training.models import create_model
from sota_training.utils import (
    setup_logging,
    log_system_info,
    load_checkpoint,
)
from sota_training.sequential_layer_training import (
    build_contiguous_label_subsets,
)
from sota_training.expert_training import train_global_residual_experts

# IMPORTANT: use the same logger name as setup_logging()
# so messages show up in console + log file (like other training scripts).
logger = logging.getLogger('sota_training')


# Fix OpenMP duplicate library error on Windows
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# GPU optimization settings
cuda_launch_blocking = os.environ.get('CUDA_LAUNCH_BLOCKING', '0')
os.environ['CUDA_LAUNCH_BLOCKING'] = cuda_launch_blocking
torch.backends.cudnn.benchmark = True


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_videomae_model11_experts_config() -> dict:
    default_config = get_default_config()
    config = default_config.copy()

    config['architecture'] = 'videomae_experts'
    config['model_name'] = 'MCG-NJU/videomae-large-finetuned-kinetics'
    config['pretrained_ckpt'] = None

    # Video params
    config['tubelet_size'] = 2
    config['image_size'] = 224
    config['patch_size'] = 16
    config['num_frames'] = 16
    config['frame_stride'] = 2
    config['img_size'] = 224

    # Init (match Model 9 best practice)
    config['init_output_gain'] = 2.0
    config['init_hidden_gain'] = 1.0
    config['use_normal_init'] = True

    # Train
    config['batch_size'] = 24
    config['grad_accum_steps'] = 6
    config['warmup_epochs'] = 5

    # Heads schedule
    config['freeze_backbone_epochs'] = 15          # Stage A epochs
    config['layer_phase_max_epochs'] = 30          # Stage B max epochs / expert
    config['layer_phase_patience'] = 10            # Stage B early stop patience
    config['final_head_epochs'] = 50               # Stage C epochs
    config['early_stop_patience'] = 20             # Stage C early stop patience

    # LRs
    config['head_lr'] = 1e-3                       # Stage A global head LR
    config['expert_lr'] = 2e-4                     # Stage B expert LR
    config['final_head_lr'] = 1e-4                 # Stage C joint LR

    # Regularization / aug (reuse defaults unless overridden)
    config['weight_decay'] = 0.08
    config['weight_decay_unfreeze'] = 0.05
    config['dropout'] = 0.2
    config['label_smoothing'] = 0.1
    config['use_ema'] = False
    config['use_clean_train_loss'] = False

    # Adaptive LR scheduler
    config['use_adaptive_lr'] = True
    config['lr_plateau_patience'] = 8
    config['lr_min_delta'] = 0.005
    config['lr_threshold_mode'] = 'rel'
    config['lr_cooldown'] = 3
    config['use_val_loss_for_lr'] = False
    config['min_lr_ratio'] = 0.005

    # Data
    config['val_ratio'] = 0.20
    config['num_workers'] = 0

    # Output
    config['output_dir'] = Path('./checkpoints')
    config['logging_dir'] = Path('./logging')

    # Experts
    config['num_experts'] = 8

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train VideoMAE model 11 (Global + Residual Experts)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model-id', type=int, default=11, help='Model ID')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./kaggle_data/data',
        help='Data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./checkpoints',
        help='Output directory'
    )
    parser.add_argument(
        '--logging-dir',
        type=str,
        default='./logging',
        help='Logging directory'
    )
    parser.add_argument(
        '--pretrained-ckpt',
        type=str,
        default=None,
        help='Optional pretrained checkpoint path'
    )
    parser.add_argument(
        '--skip-stage-a',
        action='store_true',
        help=(
            'Skip Stage A (train global head). '
            'Recommended with --resume-from.'
        ),
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Resume from a Model 11 checkpoint (loads weights + history).',
    )
    parser.add_argument(
        '--on-existing',
        type=str,
        choices=['prompt', 'resume', 'skip', 'train'],
        default='prompt',
        help=(
            'What to do if the best checkpoint already exists. '
            'prompt=ask, resume=load it + skip stage A, '
            'skip=exit, train=retrain (requires --overwrite).'
        ),
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Allow overwriting existing best checkpoint when retraining.',
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    return parser.parse_args()


def main():
    args = parse_args()

    config = get_videomae_model11_experts_config()
    config['model_id'] = args.model_id
    config['seed'] = args.seed
    config['data_dir'] = Path(args.data_dir)
    config['output_dir'] = Path(args.output_dir)
    config['logging_dir'] = Path(args.logging_dir)
    config['debug'] = args.debug

    config['output_dir'].mkdir(parents=True, exist_ok=True)
    config['logging_dir'].mkdir(parents=True, exist_ok=True)

    logger_instance = setup_logging(
        config['model_id'],
        config['logging_dir'],
        debug=config['debug']
    )
    logger_instance.info("=" * 60)
    logger_instance.info("VideoMAE Model 11 - Global + Residual Experts")
    logger_instance.info("=" * 60)

    set_random_seeds(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger_instance.info(f"Device: {device}")
    log_system_info(logger_instance, device)

    # Datasets
    train_data_dir = config['data_dir'] / 'data_train'
    train_dataset = VideoDataset(
        root=train_data_dir,
        num_frames=config['num_frames'],
        frame_stride=config['frame_stride'],
        image_size=config['img_size'],
        is_train=True,
        val_ratio=config['val_ratio'],
        seed=config['seed'],
        use_temporal_aug=config.get('use_temporal_aug', False),
        use_advanced_spatial=config.get('use_advanced_spatial', False),
        use_advanced_color=config.get('use_advanced_color', False)
    )
    val_dataset = VideoDataset(
        root=train_data_dir,
        num_frames=config['num_frames'],
        frame_stride=config['frame_stride'],
        image_size=config['img_size'],
        is_train=False,
        val_ratio=config['val_ratio'],
        seed=config['seed']
    )

    num_classes = len(train_dataset.classes)
    logger_instance.info(f"Train samples: {len(train_dataset)}")
    logger_instance.info(f"Val samples: {len(val_dataset)}")
    logger_instance.info(f"Classes: {num_classes}")

    # Label subsets for experts
    label_subsets = build_contiguous_label_subsets(
        num_classes,
        num_experts=config['num_experts']
    )
    config['label_subsets'] = label_subsets
    logger_instance.info(f"Experts: {len(label_subsets)} subsets")

    # Pretrained ckpt (optional)
    pretrained_ckpt = args.pretrained_ckpt
    if pretrained_ckpt is None:
        candidate = config['output_dir'] / 'videomae_model_9_best.pt'
        if candidate.exists():
            pretrained_ckpt = str(candidate)
            logger_instance.info(f"Using Model 9 checkpoint as pretrained: {candidate}")

    # Model
    model = create_model(
        architecture=config['architecture'],
        num_classes=num_classes,
        pretrained_name=None,
        use_adapters=False,
        dropout=config.get('dropout', 0.1),
        pretrained_ckpt=pretrained_ckpt,
        model_name=config.get('model_name'),
        num_frames=config['num_frames'],
        tubelet_size=config.get('tubelet_size', 2),
        image_size=config['img_size'],
        patch_size=config.get('patch_size', 16),
        init_output_gain=config.get('init_output_gain', 2.0),
        init_hidden_gain=config.get('init_hidden_gain', 1.0),
        use_normal_init=config.get('use_normal_init', True),
        label_subsets=label_subsets
    ).to(device)

    checkpoint_path = (
        config['output_dir'] / f'videomae_model_{config["model_id"]}_best.pt'
    )
    history_plot_path = (
        config['output_dir'] / f'videomae_model_{config["model_id"]}_training.png'
    )

    # Handle existing checkpoint (skip/resume/retrain) like train_sequential_distillation.py
    existing_best = checkpoint_path.exists()
    on_existing = args.on_existing
    skip_stage_a = args.skip_stage_a

    if existing_best and on_existing == 'prompt':
        logger_instance.info(
            f"Checkpoint already exists: {checkpoint_path}"
        )
        logger_instance.info("Options: [s] Skip, [r] Resume, [t] Train (overwrite)")
        while True:
            response = input(
                "\nChoose option for existing checkpoint (s/r/t): "
            ).strip().lower()
            if response in {'s', 'r', 't'}:
                break
            logger_instance.info("Invalid option. Enter s/r/t.")
        if response == 's':
            logger_instance.info("Skipping training (existing checkpoint kept).")
            return
        if response == 'r':
            on_existing = 'resume'
        if response == 't':
            on_existing = 'train'

    if existing_best and on_existing == 'skip':
        logger_instance.info("Skipping training (existing checkpoint kept).")
        return

    if existing_best and on_existing == 'train' and not args.overwrite:
        logger_instance.error(
            "Checkpoint exists and --on-existing=train was requested, but "
            "--overwrite was not set. Aborting to avoid overwriting."
        )
        return

    if existing_best and on_existing == 'resume':
        # Resume implies we skip Stage A by default.
        skip_stage_a = True

    # Resume/skip Stage A
    resume_ckpt_path = Path(args.resume_from) if args.resume_from else None
    if (skip_stage_a or on_existing == 'resume') and resume_ckpt_path is None:
        candidate = checkpoint_path
        if candidate.exists():
            resume_ckpt_path = candidate
            logger_instance.info(
                f"Auto resume enabled: {resume_ckpt_path}"
            )

    resume_checkpoint = None
    if resume_ckpt_path is not None:
        if resume_ckpt_path.exists():
            resume_checkpoint = load_checkpoint(
                resume_ckpt_path,
                device=device,
                logger=logger_instance,
            )
            state = resume_checkpoint.get('model', None)
            if state is not None:
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing or unexpected:
                    logger_instance.warning(
                        "Resume checkpoint load_state_dict had "
                        f"missing={len(missing)}, unexpected={len(unexpected)}"
                    )
            logger_instance.info(f"Resumed weights from: {resume_ckpt_path}")
        else:
            logger_instance.warning(f"Resume checkpoint not found: {resume_ckpt_path}")

    if skip_stage_a and resume_checkpoint is None:
        logger_instance.warning(
            "skip-stage-a was requested but no resume checkpoint was loaded. "
            "Falling back to running Stage A."
        )
        skip_stage_a = False

    # Training pipeline is in sota_training/ for consistency.
    _ = train_global_residual_experts(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        config=config,
        checkpoint_path=checkpoint_path,
        history_plot_path=history_plot_path,
        classes=train_dataset.classes,
        label_subsets=label_subsets,
        skip_stage_a=skip_stage_a,
        resume_checkpoint=resume_checkpoint,
    )


if __name__ == '__main__':
    main()
