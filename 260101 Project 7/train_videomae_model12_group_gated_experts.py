"""
Train VideoMAE Model 12: Group-Gated Experts.

Design:
- Frozen VideoMAE backbone
- Group head predicts which label-group the sample belongs to
- Experts predict within each label-group
- Inference uses hard top-1 routing (optional; training uses soft routing)
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
from sota_training.utils import setup_logging, log_system_info, load_checkpoint
from sota_training.sequential_layer_training import (
    build_contiguous_label_subsets,
)
from sota_training.expert_training import train_group_gated_experts

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


def get_videomae_model12_group_gated_config() -> dict:
    default_config = get_default_config()
    config = default_config.copy()

    config['architecture'] = 'videomae_group_gated_experts'
    config['model_name'] = 'MCG-NJU/videomae-large-finetuned-kinetics'
    config['pretrained_ckpt'] = None

    # Video params
    config['tubelet_size'] = 2
    config['image_size'] = 224
    config['patch_size'] = 16
    config['num_frames'] = 16
    config['frame_stride'] = 2
    config['img_size'] = 224

    # Init
    config['init_output_gain'] = 2.0
    config['init_hidden_gain'] = 1.0
    config['use_normal_init'] = True

    # Train
    config['batch_size'] = 24
    config['grad_accum_steps'] = 6
    config['warmup_epochs'] = 5

    # Experts (Stage B)
    config['num_experts'] = 8
    config['layer_phase_max_epochs'] = 30
    config['layer_phase_patience'] = 10
    config['expert_lr'] = 2e-4

    # Group head (Stage G)
    config['group_head_epochs'] = 40
    config['group_head_patience'] = 10
    config['group_head_lr'] = 1e-3
    config['group_target_acc'] = 0.99

    # Joint calibration (Stage C)
    config['final_head_epochs'] = 50
    config['early_stop_patience'] = 20
    config['final_head_lr'] = 1e-4
    config['use_stage_c'] = False

    # Regularization / aug
    config['weight_decay'] = 0.08
    config['weight_decay_unfreeze'] = 0.05
    config['dropout'] = 0.2
    config['label_smoothing'] = 0.1
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

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train VideoMAE model 12 (Group-Gated Experts)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model-id', type=int, default=12, help='Model ID')
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
        '--resume-from',
        type=str,
        default=None,
        help='Resume from a Model 12 checkpoint (loads weights + history).',
    )
    parser.add_argument(
        '--on-existing',
        type=str,
        choices=['prompt', 'resume', 'skip', 'train'],
        default='prompt',
        help=(
            'What to do if the best checkpoint already exists. '
            'prompt=ask, resume=load it, skip=exit, '
            'train=retrain (requires --overwrite).'
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

    config = get_videomae_model12_group_gated_config()
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
    logger_instance.info("VideoMAE Model 12 - Group-Gated Experts")
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

    # Label subsets for experts/groups
    label_subsets = build_contiguous_label_subsets(
        num_classes,
        num_experts=config['num_experts']
    )
    config['label_subsets'] = label_subsets
    logger_instance.info(f"Groups/Experts: {len(label_subsets)} subsets")

    # Pretrained backbone ckpt (optional)
    pretrained_ckpt = args.pretrained_ckpt
    if pretrained_ckpt is None:
        candidate = config['output_dir'] / 'videomae_model_9_best.pt'
        if candidate.exists():
            pretrained_ckpt = str(candidate)
            logger_instance.info(
                f"Using Model 9 checkpoint as pretrained: {candidate}"
            )

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
        label_subsets=label_subsets,
        hard_mask_value=-1e9,
    ).to(device)

    checkpoint_path = (
        config['output_dir'] / f'videomae_model_{config["model_id"]}_best.pt'
    )
    history_plot_path = (
        config['output_dir'] /
        f'videomae_model_{config["model_id"]}_training.png'
    )

    # Handle existing checkpoint (skip/resume/retrain)
    existing_best = checkpoint_path.exists()
    on_existing = args.on_existing
    resume_ckpt_path = Path(args.resume_from) if args.resume_from else None

    if existing_best and on_existing == 'prompt':
        logger_instance.info(f"Checkpoint already exists: {checkpoint_path}")
        logger_instance.info(
            "Options: [s] Skip, [r] Resume, [t] Train (overwrite)"
        )
        while True:
            response = input(
                "\nChoose option for existing checkpoint (s/r/t): "
            ).strip().lower()
            if response in {'s', 'r', 't'}:
                break
            logger_instance.info("Invalid option. Enter s/r/t.")
        if response == 's':
            logger_instance.info(
                "Skipping training (existing checkpoint kept)."
            )
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

    if existing_best and on_existing == 'resume' and resume_ckpt_path is None:
        resume_ckpt_path = checkpoint_path
        logger_instance.info(f"Auto resume enabled: {resume_ckpt_path}")

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
                missing, unexpected = model.load_state_dict(
                    state,
                    strict=False,
                )
                if missing or unexpected:
                    logger_instance.warning(
                        "Resume checkpoint load_state_dict had "
                        f"missing={len(missing)}, unexpected={len(unexpected)}"
                    )
            logger_instance.info(f"Resumed weights from: {resume_ckpt_path}")
        else:
            logger_instance.warning(
                f"Resume checkpoint not found: {resume_ckpt_path}"
            )

    _ = train_group_gated_experts(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        config=config,
        checkpoint_path=checkpoint_path,
        history_plot_path=history_plot_path,
        classes=train_dataset.classes,
        label_subsets=label_subsets,
        resume_checkpoint=resume_checkpoint,
    )


if __name__ == '__main__':
    main()
