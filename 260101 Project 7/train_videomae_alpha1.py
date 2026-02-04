"""
Train VideoMAE Model Alpha 1: Two-phase experts with reserved heads.

Design:
- Phase 1: Train only num_phase1_heads (e.g. 8); num_reserved_heads (e.g. 2) are unused.
  Inference/eval: single-best-expert among first K heads only.
- After phase 1: Evaluate on val (per-class acc), select weak classes.
- Phase 2: Partition weak classes into R groups; replace reserved heads with new
  heads for those groups; train only the R reserved heads. Inference:
  single-best-expert among all K+R heads.

Chạy: conda run -n pytorch_gpu python train_videomae_alpha1.py
  [--num-phase1-heads 8] [--num-reserved-heads 2]

Phase 1 tuning (nếu overall val acc thấp): tăng layer_phase_max_epochs / giảm
layer_phase_patience, tăng warmup_epochs, hoặc tăng num_phase1_heads để mỗi expert
ít class hơn. Phase 1 "best overall" tăng dần theo số expert đã train (chỉ K experts
tham gia single-best nên acc tổng hợp phụ thuộc routing).
"""

import os
import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from sota_training.config import get_default_config
from sota_training.dataset import VideoDataset
from sota_training.models import create_model, create_classification_head
from sota_training.utils import setup_logging, log_system_info, load_checkpoint
from sota_training.sequential_layer_training import (
    build_contiguous_label_subsets,
    freeze_all_backbone_layers,
    freeze_all_heads,
    train_expert_head_phase,
)
from sota_training.expert_training import _make_full_loaders, plot_training_history
from sota_training.training import evaluate
from improve_weak_heads import (
    evaluate_model_detailed,
    print_performance_analysis,
    select_classes_to_improve,
)

logger = logging.getLogger('sota_training')

ALPHA_ARCH = 'videomae_alpha_experts'

if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
cuda_launch_blocking = os.environ.get('CUDA_LAUNCH_BLOCKING', '0')
os.environ['CUDA_LAUNCH_BLOCKING'] = cuda_launch_blocking
torch.backends.cudnn.benchmark = True


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def partition_weak_classes_into_groups(
    weak_class_indices: List[int],
    num_groups: int,
) -> List[List[int]]:
    """Chia weak classes thành num_groups nhóm (contiguous)."""
    if num_groups <= 0 or not weak_class_indices:
        return []
    n = len(weak_class_indices)
    size_per = n // num_groups
    rem = n % num_groups
    groups: List[List[int]] = []
    start = 0
    for i in range(num_groups):
        size = size_per + (1 if i < rem else 0)
        end = start + size
        groups.append(weak_class_indices[start:end])
        start = end
    return groups


def build_phase2_subsets(
    label_subsets_phase1: List[List[int]],
    weak_class_indices: List[int],
    reserved_groups: List[List[int]],
) -> List[List[int]]:
    """Phase 2: K nhóm phase1 đã bỏ weak + R nhóm reserved. Mỗi class 1 nhóm."""
    weak_set = set(weak_class_indices)
    phase2_main: List[List[int]] = []
    for grp in label_subsets_phase1:
        phase2_main.append([c for c in grp if c not in weak_set])
    return phase2_main + reserved_groups


def select_bottom_k_classes(
    results: Dict[str, Any],
    k: int,
) -> List[int]:
    """Chọn k class có accuracy thấp nhất."""
    if k <= 0 or not results.get('class_acc'):
        return []
    sorted_classes = sorted(
        results['class_acc'].items(),
        key=lambda x: x[1],
    )
    return [class_idx for class_idx, _ in sorted_classes[:k]]


def get_alpha1_config(
    num_phase1_heads: int = 8,
    num_reserved_heads: int = 2,
) -> dict:
    default_config = get_default_config()
    config = default_config.copy()

    config['architecture'] = ALPHA_ARCH
    config['model_name'] = 'MCG-NJU/videomae-large-finetuned-kinetics'
    config['pretrained_ckpt'] = None

    config['tubelet_size'] = 2
    config['image_size'] = 224
    config['patch_size'] = 16
    config['num_frames'] = 16
    config['frame_stride'] = 2
    config['img_size'] = 224

    config['init_output_gain'] = 2.0
    config['init_hidden_gain'] = 1.0
    config['use_normal_init'] = True

    config['batch_size'] = 24
    config['grad_accum_steps'] = 6
    config['warmup_epochs'] = 8

    config['num_phase1_heads'] = num_phase1_heads
    config['num_reserved_heads'] = num_reserved_heads
    config['num_experts'] = num_phase1_heads
    config['layer_phase_max_epochs'] = 45
    config['layer_phase_patience'] = 12
    config['expert_lr'] = 1.5e-4

    config['weight_decay'] = 0.08
    config['weight_decay_unfreeze'] = 0.05
    config['dropout'] = 0.2
    config['label_smoothing'] = 0.1
    config['use_focal_loss'] = False
    config['focal_alpha'] = 0.25
    config['focal_gamma'] = 2.0
    config['use_clean_train_loss'] = False

    config['use_adaptive_lr'] = True
    config['lr_plateau_patience'] = 10
    config['lr_min_delta'] = 0.005
    config['lr_threshold_mode'] = 'rel'
    config['lr_cooldown'] = 3
    config['use_val_loss_for_lr'] = False
    config['min_lr_ratio'] = 0.005

    config['val_ratio'] = 0.20
    config['num_workers'] = 0

    config['output_dir'] = Path('./checkpoints')
    config['logging_dir'] = Path('./logging')

    config['inference_single_best_expert'] = True

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Alpha 1: two-phase, reserved heads for weak classes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model-name', type=str, default='alpha1', help='Run name')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--data-dir', type=str, default='./kaggle_data/data')
    parser.add_argument('--output-dir', type=str, default='./checkpoints')
    parser.add_argument('--logging-dir', type=str, default='./logging')
    parser.add_argument('--pretrained-ckpt', type=str, default=None)
    parser.add_argument(
        '--num-phase1-heads',
        type=int,
        default=8,
        help='Number of experts to train in phase 1',
    )
    parser.add_argument(
        '--num-reserved-heads',
        type=int,
        default=2,
        help='Reserved heads for phase 2 (weak classes)',
    )
    parser.add_argument(
        '--weak-class-threshold',
        type=float,
        default=0.70,
        help='Class acc < this is weak (for phase 2)',
    )
    parser.add_argument(
        '--weak-bottom-k-classes',
        type=int,
        default=0,
        help='If > 0, use bottom-k classes by acc instead of threshold',
    )
    parser.add_argument(
        '--on-existing',
        type=str,
        choices=['prompt', 'resume', 'skip', 'train', 's', 'r', 't'],
        default='prompt',
    )
    parser.add_argument(
        '--use-focal-loss',
        action='store_true',
        help='Use Focal Loss for expert training (good for imbalanced subsets)',
    )
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def run_phase1(
    model: torch.nn.Module,
    train_dataset: VideoDataset,
    val_dataset: VideoDataset,
    device: torch.device,
    config: dict,
    label_subsets_phase1: List[List[int]],
    classes: List[str],
    checkpoint_path: Path,
    history_plot_path: Path,
) -> Dict[str, list]:
    """Phase 1: train only experts 0..K-1 (K = num_phase1_heads)."""
    K = config['num_phase1_heads']
    freeze_all_backbone_layers(model)
    freeze_all_heads(model)
    if hasattr(model, 'active_expert_idx'):
        model.active_expert_idx = None
    model.inference_single_best_expert = True
    model.num_active_experts = K

    train_loader, val_loader = _make_full_loaders(train_dataset, val_dataset, config)
    combined_history: Dict[str, list] = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
    }
    best_val_saved = 0.0
    best_train_saved = 0.0

    for i in range(K):
        subset = label_subsets_phase1[i]
        phase_name = f"Alpha1 Phase1 - Expert {i + 1}/{K}"
        best_i, hist_i = train_expert_head_phase(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            config=config,
            expert_idx=i,
            label_subset=subset,
            phase_name=phase_name,
        )
        for k in combined_history:
            combined_history[k].extend(hist_i[k])
        plot_training_history(combined_history, history_plot_path, config['model_id'])

        if hasattr(model, 'active_expert_idx'):
            model.active_expert_idx = None
        model.num_active_experts = K
        model.eval()
        val_loss, val_acc = evaluate(model, val_loader, device)
        train_loss, train_acc = evaluate(model, train_loader, device)
        save = (
            val_acc > best_val_saved
            or (train_acc > best_train_saved and val_acc >= best_val_saved)
        )
        if save:
            best_val_saved = max(best_val_saved, val_acc)
            best_train_saved = max(best_train_saved, train_acc)
            checkpoint = {
                'model': model.state_dict(),
                'epoch': len(combined_history['train_loss']),
                'history': combined_history,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'classes': classes,
                'config': {
                    **config,
                    'label_subsets': config.get('label_subsets'),
                    'num_active_experts': K,
                    'inference_single_best_expert': True,
                    'alpha1_phase': 1,
                },
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(
                "  Phase1 saved (val=%.4f, train=%.4f) [best_val=%.4f, best_train=%.4f]",
                val_acc, train_acc, best_val_saved, best_train_saved,
            )

    return combined_history


def main():
    args = parse_args()

    num_phase1 = args.num_phase1_heads
    num_reserved = args.num_reserved_heads
    if num_phase1 <= 0 or num_reserved <= 0:
        logger.error("num_phase1_heads and num_reserved_heads must be > 0")
        return

    config = get_alpha1_config(num_phase1, num_reserved)
    config['run_name'] = args.model_name
    config['model_id'] = config['run_name']
    config['seed'] = args.seed
    config['data_dir'] = Path(args.data_dir)
    config['output_dir'] = Path(args.output_dir)
    config['logging_dir'] = Path(args.logging_dir)
    config['debug'] = args.debug
    if args.use_focal_loss:
        config['use_focal_loss'] = True

    config['output_dir'].mkdir(parents=True, exist_ok=True)
    config['logging_dir'].mkdir(parents=True, exist_ok=True)

    logger_instance = setup_logging(
        config['run_name'],
        config['logging_dir'],
        debug=config['debug'],
    )
    logger_instance.info("=" * 60)
    logger_instance.info(
        "VideoMAE Model Alpha 1 - Two-phase experts (reserved heads)"
    )
    logger_instance.info("=" * 60)

    set_random_seeds(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger_instance.info("Device: %s", device)
    log_system_info(logger_instance, device)

    train_data_dir = config['data_dir'] / 'data_train'
    train_dataset = VideoDataset(
        root=train_data_dir,
        num_frames=config['num_frames'],
        frame_stride=config['frame_stride'],
        image_size=config['img_size'],
        is_train=True,
        val_ratio=config['val_ratio'],
        seed=config['seed'],
    )
    val_dataset = VideoDataset(
        root=train_data_dir,
        num_frames=config['num_frames'],
        frame_stride=config['frame_stride'],
        image_size=config['img_size'],
        is_train=False,
        val_ratio=config['val_ratio'],
        seed=config['seed'],
    )
    num_classes = len(train_dataset.classes)
    classes = train_dataset.classes
    logger_instance.info("Classes: %d", num_classes)

    # Phase 1: K groups (all classes) + R dummy (output_dim >= 1)
    label_subsets_phase1 = build_contiguous_label_subsets(
        num_classes, num_experts=num_phase1
    )
    dummy_subsets: List[List[int]] = []
    for r in range(num_reserved):
        dummy_subsets.append([r % num_classes])
    label_subsets_for_model = label_subsets_phase1 + dummy_subsets

    pretrained_ckpt = args.pretrained_ckpt
    if pretrained_ckpt is None:
        candidate = config['output_dir'] / 'videomae_model_9_best.pt'
        if candidate.exists():
            pretrained_ckpt = str(candidate)
            logger_instance.info(
                "Using Model 9 checkpoint as pretrained: %s", candidate
            )

    model = create_model(
        architecture=ALPHA_ARCH,
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
        label_subsets=label_subsets_for_model,
        hard_mask_value=-1e9,
    ).to(device)

    model.num_active_experts = num_phase1
    model.inference_single_best_expert = True

    config['label_subsets'] = label_subsets_for_model

    checkpoint_path = (
        config['output_dir'] / f'videomae_model_{config["run_name"]}_best.pt'
    )
    history_plot_path = (
        config['output_dir']
        / f'videomae_model_{config["run_name"]}_training.png'
    )

    _srt = {'s': 'skip', 'r': 'resume', 't': 'train'}
    on_existing = _srt.get(args.on_existing, args.on_existing)
    if checkpoint_path.exists() and on_existing == 'skip':
        logger_instance.info(
            "Checkpoint exists and on_existing=skip; exiting."
        )
        return
    if checkpoint_path.exists() and on_existing == 'resume' and args.resume_from is None:
        args.resume_from = str(checkpoint_path)
    if args.resume_from and Path(args.resume_from).exists():
        resume_ckpt = load_checkpoint(Path(args.resume_from), device, logger_instance)
        if resume_ckpt.get('model'):
            model.load_state_dict(resume_ckpt['model'], strict=False)
            logger_instance.info("Resumed weights from %s", args.resume_from)

    # ---------- Phase 1 ----------
    logger_instance.info("")
    logger_instance.info("STEP 1: PHASE 1 - Train experts 0..%d only", num_phase1 - 1)
    logger_instance.info("=" * 60)
    run_phase1(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        config=config,
        label_subsets_phase1=label_subsets_phase1,
        classes=classes,
        checkpoint_path=checkpoint_path,
        history_plot_path=history_plot_path,
    )

    # ---------- Eval between phases ----------
    logger_instance.info("")
    logger_instance.info(
        "STEP 2: EVALUATE (single-best-expert, %d heads) -> weak classes",
        num_phase1,
    )
    logger_instance.info("=" * 60)
    model.num_active_experts = num_phase1
    model.inference_single_best_expert = True
    model.eval()
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=torch.cuda.is_available(),
    )
    results = evaluate_model_detailed(
        model=model,
        val_loader=val_loader,
        device=device,
        classes=classes,
        label_subsets=label_subsets_phase1,
    )
    print_performance_analysis(results, classes, label_subsets_phase1)

    if args.weak_bottom_k_classes > 0:
        weak_classes = select_bottom_k_classes(
            results, args.weak_bottom_k_classes
        )
        logger_instance.info(
            "Weak classes (bottom-%d by acc): %s",
            args.weak_bottom_k_classes, weak_classes,
        )
    else:
        weak_classes = select_classes_to_improve(
            results, threshold=args.weak_class_threshold
        )
        logger_instance.info(
            "Weak classes (acc < %.2f): %s",
            args.weak_class_threshold, weak_classes,
        )

    if not weak_classes:
        logger_instance.info(
            "No weak classes; skipping phase 2. Phase 1 checkpoint is final."
        )
        return

    reserved_groups = partition_weak_classes_into_groups(weak_classes, num_reserved)
    logger_instance.info("Reserved groups (weak classes): %s", reserved_groups)

    # ---------- Phase 2: replace reserved heads, update subsets, train reserved experts ----------
    logger_instance.info("")
    logger_instance.info(
        "STEP 3: PHASE 2 - Replace reserved heads, train experts %d..%d",
        num_phase1, num_phase1 + num_reserved - 1,
    )
    logger_instance.info("=" * 60)

    label_subsets_phase2 = build_phase2_subsets(
        label_subsets_phase1,
        weak_classes,
        reserved_groups,
    )

    embed_dim = model.embed_dim
    dropout = config.get('dropout', 0.1)
    init_output_gain = config.get('init_output_gain', 2.0)
    init_hidden_gain = config.get('init_hidden_gain', 1.0)
    use_normal_init = config.get('use_normal_init', True)

    for r in range(num_reserved):
        subset_r = reserved_groups[r]
        head_r = create_classification_head(
            embed_dim,
            len(subset_r),
            dropout,
            init_output_gain=init_output_gain,
            init_hidden_gain=init_hidden_gain,
            use_normal_init=use_normal_init,
        )
        model.expert_heads[num_phase1 + r] = head_r
        if device.type == 'cuda':
            model.expert_heads[num_phase1 + r] = (
            model.expert_heads[num_phase1 + r].to(device)
        )

    model.label_subsets = [list(s) for s in label_subsets_phase2]
    model._subset_tensors = [None for _ in label_subsets_phase2]
    # Main experts (0..K-1) still have heads trained on phase1 subsets; reserved heads on reserved_groups.
    # Forward uses this to map logits when current subset is shrunk (weak classes removed).
    model.expert_training_subsets = [list(s) for s in label_subsets_phase1] + [
        list(s) for s in reserved_groups
    ]
    model.num_active_experts = num_phase1 + num_reserved

    freeze_all_backbone_layers(model)
    freeze_all_heads(model)
    if hasattr(model, 'active_expert_idx'):
        model.active_expert_idx = None

    for r in range(num_reserved):
        expert_idx = num_phase1 + r
        subset_r = reserved_groups[r]
        phase_name = (
            f"Alpha1 Phase2 - Expert {expert_idx + 1}/"
            f"{num_phase1 + num_reserved}"
        )
        best_r, _ = train_expert_head_phase(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            config=config,
            expert_idx=expert_idx,
            label_subset=subset_r,
            phase_name=phase_name,
        )
        logger_instance.info("  Phase2 expert %d best val acc: %.4f", expert_idx, best_r)

    if hasattr(model, 'active_expert_idx'):
        model.active_expert_idx = None
    model.num_active_experts = num_phase1 + num_reserved
    val_loss, val_acc = evaluate(model, val_loader, device)
    logger_instance.info("Phase2 overall val acc: %.4f", val_acc)

    final_checkpoint = {
        'model': model.state_dict(),
        'epoch': 0,
        'history': {},
        'val_acc': val_acc,
        'classes': classes,
        'config': {
            **config,
            'label_subsets': label_subsets_phase2,
            'num_active_experts': num_phase1 + num_reserved,
            'inference_single_best_expert': True,
            'alpha1_phase': 2,
            'num_phase1_heads': num_phase1,
            'num_reserved_heads': num_reserved,
        },
    }
    torch.save(final_checkpoint, checkpoint_path)
    logger_instance.info("")
    logger_instance.info("Alpha 1 completed. Checkpoint saved: %s", checkpoint_path)
    logger_instance.info("=" * 60)


if __name__ == '__main__':
    main()
