"""
Cải thiện các expert yếu cho Model Alpha (VideoMAE Alpha - Experts only, no group head).

Luồng chuẩn (đã align với improve Model 12):
1. Evaluate: full model (merge all experts) -> overall + per-group/per-class acc
2. Chọn weak groups: acc < group_threshold HOẶC bottom-k theo acc (--improve-bottom-k-groups)
3. Retrain chỉ các expert của weak groups (train_expert_head_phase; best weights được restore sau early stop)
4. Re-evaluate -> save checkpoint improved (Alpha không có group head nên không retrain group head)

Chạy: python improve_weak_heads_alpha.py [--checkpoint ...] --retrain-experts [--improve-bottom-k-groups 2]
"""

import os

if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from pathlib import Path
import logging
import argparse
from typing import Optional, Dict, List

cuda_launch_blocking = os.environ.get('CUDA_LAUNCH_BLOCKING', '0')
os.environ['CUDA_LAUNCH_BLOCKING'] = cuda_launch_blocking
torch.backends.cudnn.benchmark = True

from sota_training.config import get_default_config
from sota_training.utils import setup_logging, log_system_info, load_checkpoint
from sota_training.dataset import VideoDataset
from sota_training.models import create_model
from sota_training.sequential_layer_training import (
    freeze_all_backbone_layers,
    build_contiguous_label_subsets,
    FilteredVideoDataset,
)
from torch.utils.data import DataLoader

from improve_weak_heads import (
    evaluate_model_detailed,
    print_performance_analysis,
    select_groups_to_improve,
    select_bottom_k_groups,
    select_classes_to_improve,
    retrain_weak_experts,
    save_checkpoint,
    _load_state_dict_robust,
)

logger = logging.getLogger('sota_training')

ALPHA_ARCH = 'videomae_alpha_experts'


def evaluate_per_expert_accuracy(
    model: torch.nn.Module,
    val_dataset,
    device: torch.device,
    label_subsets: List[List[int]],
    batch_size: int,
    num_workers: int = 0,
) -> Dict[int, Dict]:
    """
    Đánh giá từng expert riêng (active_expert_idx = i) trên subset của nó.
    Trả về {expert_idx: {'acc': float, 'correct': int, 'total': int}}.
    """
    model.eval()
    result = {}
    prev_active = getattr(model, "active_expert_idx", None)
    try:
        for expert_idx, subset in enumerate(label_subsets):
            if not subset:
                result[expert_idx] = {'acc': 0.0, 'correct': 0, 'total': 0}
                continue
            filtered = FilteredVideoDataset(val_dataset, subset)
            if len(filtered) == 0:
                result[expert_idx] = {'acc': 0.0, 'correct': 0, 'total': 0}
                continue
            loader = DataLoader(
                filtered,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            model.active_expert_idx = expert_idx
            correct, total = 0, 0
            with torch.no_grad():
                for videos, labels in loader:
                    videos = videos.to(device, non_blocking=True)
                    labels = labels.to(device, dtype=torch.long, non_blocking=True)
                    with torch.amp.autocast(
                        device_type='cuda', enabled=(device.type == 'cuda')
                    ):
                        logits = model(videos)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            result[expert_idx] = {
                'acc': correct / total if total else 0.0,
                'correct': correct,
                'total': total,
            }
    finally:
        model.active_expert_idx = prev_active
    return result


def get_expert_head_sizes_from_state_dict(state_dict: dict) -> List[int]:
    """Lấy số class từng expert từ state_dict (expert_heads.i.5.weight shape[0])."""
    sizes = []
    i = 0
    while True:
        key = f"expert_heads.{i}.5.weight"
        if key not in state_dict:
            break
        sizes.append(state_dict[key].shape[0])
        i += 1
    return sizes


def infer_label_subsets_from_state_dict(
    state_dict: dict,
    num_classes: int,
) -> List[List[int]]:
    """
    Suy label_subsets contiguous từ state_dict chỉ khi sum(sizes)==num_classes (Phase 1).
    Phase 2 (10 experts, sum!=51) phải dùng label_subsets từ config.
    """
    sizes = get_expert_head_sizes_from_state_dict(state_dict)
    if not sizes or sum(sizes) != num_classes:
        return []
    start = 0
    subsets = []
    for s in sizes:
        end = start + s
        subsets.append(list(range(start, end)))
        start = end
    return subsets


def build_phase2_subsets_fallback(
    sizes: List[int],
    num_classes: int,
) -> List[List[int]]:
    """
    Fallback cho Phase 2 khi config không khớp: xây subsets đúng kích thước để load state_dict.
    num_experts đầu có tổng = num_classes -> contiguous; còn lại gán contiguous trong [0..num_classes-1]
    (có overlap, chỉ để shape đúng; semantics có thể sai cho reserved experts).
    """
    if not sizes:
        return []
    subsets = []
    start = 0
    for i, s in enumerate(sizes):
        if start >= num_classes:
            start = 0
        end = min(start + s, num_classes)
        if end <= start:
            subset = list(range(0, min(s, num_classes)))
        else:
            subset = list(range(start, end))
        subsets.append(subset)
        start = end
    return subsets


def fix_alpha1_checkpoint_config(checkpoint_path: Path) -> None:
    """
    Sửa checkpoint Alpha1: ghi config['label_subsets'] khớp state_dict
    (infer hoặc fallback nếu config không khớp). Lần load sau không cần fallback.
    Ghi đè file checkpoint_path.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False
    )
    if 'model' not in checkpoint:
        raise ValueError("Checkpoint không chứa 'model'")
    num_classes = len(checkpoint.get('classes', [])) or 51
    state_dict = checkpoint['model']
    sizes = get_expert_head_sizes_from_state_dict(state_dict)
    if not sizes:
        raise ValueError("Không tìm thấy expert_heads trong state_dict")

    config = checkpoint.get('config') or {}
    label_subsets = (
        config.get('label_subsets')
        or checkpoint.get('label_subsets')
    )
    config_matches = (
        label_subsets
        and len(label_subsets) == len(sizes)
        and all(len(label_subsets[i]) == sizes[i] for i in range(len(sizes)))
    )
    if not config_matches:
        inferred = infer_label_subsets_from_state_dict(state_dict, num_classes)
        if inferred:
            label_subsets = inferred
            logger.info("fix_alpha1: dùng contiguous (Phase 1) cho label_subsets")
        else:
            label_subsets = build_phase2_subsets_fallback(sizes, num_classes)
            logger.info(
                "fix_alpha1: dùng fallback Phase 2 (%d experts) cho label_subsets",
                len(label_subsets),
            )

    if 'config' not in checkpoint:
        checkpoint['config'] = {}
    checkpoint['config']['label_subsets'] = label_subsets
    torch.save(checkpoint, checkpoint_path)
    logger.info(
        "Đã ghi config['label_subsets'] vào checkpoint Alpha1: %s",
        checkpoint_path,
    )


def get_model_config_from_checkpoint_alpha(checkpoint_path: Path) -> dict:
    """Lấy config từ checkpoint Alpha; architecture phải là videomae_alpha_experts."""
    config = get_default_config()
    checkpoint = None

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False
        )
        if 'config' in checkpoint:
            config.update(checkpoint['config'])
            logger.info(
                "Loaded config from checkpoint: architecture=%s",
                config.get('architecture'),
            )
    except Exception as e:
        logger.warning("Không thể load config từ checkpoint: %s", e)

    if checkpoint is not None and 'model' in checkpoint:
        state_keys = set(checkpoint['model'].keys())
        if (
            'expert_heads.0.0.weight' in state_keys
            and 'group_head.0.weight' not in state_keys
            and not any(k.startswith('global_head') for k in state_keys)
        ):
            config['architecture'] = ALPHA_ARCH
            logger.info("Detected Model Alpha from state_dict")
    if config.get('architecture') != ALPHA_ARCH:
        filename = checkpoint_path.name.lower()
        if 'alpha' in filename:
            config['architecture'] = ALPHA_ARCH
            logger.info("Detected Model Alpha from filename")

    if config['architecture'] != ALPHA_ARCH:
        raise ValueError(
            f"Checkpoint không phải Model Alpha (architecture={config.get('architecture')}). "
            "Chỉ hỗ trợ videomae_alpha_experts."
        )

    if 'model_name' not in config:
        config['model_name'] = 'MCG-NJU/videomae-large-finetuned-kinetics'
    config.setdefault('tubelet_size', 2)
    config.setdefault('patch_size', 16)
    config.setdefault('num_frames', 16)
    config.setdefault('frame_stride', 2)
    config.setdefault('img_size', 224)
    config.setdefault('image_size', 224)
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Improve weak experts cho Model Alpha (chỉ expert heads)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Checkpoint Alpha (mặc định: output-dir/videomae_model_alpha_best.pt)',
    )
    parser.add_argument('--data-dir', type=str, default='./kaggle_data/data')
    parser.add_argument('--output-dir', type=str, default='./checkpoints')
    parser.add_argument(
        '--output-name', type=str, default=None,
        help='Tên file checkpoint improved (mặc định: videomae_model_alpha_improved_heads.pt)',
    )
    parser.add_argument(
        '--group-threshold', type=float, default=0.85,
        help='Group có acc < threshold được coi là yếu',
    )
    parser.add_argument(
        '--improve-bottom-k-groups', type=int, default=0,
        help='Luôn retrain k group có acc thấp nhất',
    )
    parser.add_argument('--class-threshold', type=float, default=0.70)
    parser.add_argument(
        '--retrain-experts', action='store_true',
        help='Retrain expert heads của weak groups',
    )
    parser.add_argument('--expert-epochs', type=int, default=30)
    parser.add_argument('--expert-lr', type=float, default=2e-4)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--num-workers', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.checkpoint is None:
        args.checkpoint = str(Path(args.output_dir) / 'videomae_model_alpha_best.pt')

    logging_dir = Path(args.output_dir).parent / 'logging'
    logging_dir.mkdir(parents=True, exist_ok=True)
    setup_logging('alpha', logging_dir)
    logger.info("=" * 60)
    logger.info("IMPROVE WEAK HEADS - Model Alpha (Experts only)")
    logger.info("=" * 60)

    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        logger.error("Checkpoint không tồn tại: %s", checkpoint_path)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device: %s", device)
    log_system_info(logger, device)

    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, device, logger)
    model_config = get_model_config_from_checkpoint_alpha(checkpoint_path)

    train_data_dir = data_dir / 'data_train'
    train_dataset = None
    if train_data_dir.exists():
        train_dataset = VideoDataset(
            root=train_data_dir,
            num_frames=model_config.get('num_frames', 16),
            frame_stride=model_config.get('frame_stride', 2),
            image_size=model_config.get('img_size', 224),
            is_train=True,
            val_ratio=0.2,
            seed=42,
        )
        num_classes = len(train_dataset.classes)
        classes = train_dataset.classes
        logger.info("Found %d classes from dataset", num_classes)
    elif 'classes' in checkpoint:
        classes = checkpoint['classes']
        num_classes = len(classes)
        logger.info("Found %d classes from checkpoint", num_classes)
    else:
        logger.error("Không tìm thấy classes")
        return

    num_experts = model_config.get('num_experts', 8)
    state_dict = checkpoint.get('model', {})
    if 'label_subsets' in model_config and model_config['label_subsets']:
        label_subsets = model_config['label_subsets']
    elif checkpoint.get('config', {}).get('label_subsets'):
        label_subsets = checkpoint['config']['label_subsets']
    elif checkpoint.get('label_subsets'):
        label_subsets = checkpoint['label_subsets']
    else:
        label_subsets = build_contiguous_label_subsets(
            num_classes, num_experts=num_experts
        )
        logger.info("Created label_subsets: %d groups", len(label_subsets))

    # Đảm bảo cấu trúc model khớp checkpoint (Phase 1/2)
    sizes = get_expert_head_sizes_from_state_dict(state_dict)
    config_matches = (
        sizes
        and len(label_subsets) == len(sizes)
        and all(len(label_subsets[i]) == sizes[i] for i in range(len(sizes)))
    )
    used_phase2_fallback = False
    if not config_matches and sizes:
        inferred = infer_label_subsets_from_state_dict(state_dict, num_classes)
        if inferred:
            logger.info(
                "label_subsets từ config không khớp state_dict; dùng contiguous (Phase 1)"
            )
            label_subsets = inferred
        else:
            fallback = build_phase2_subsets_fallback(sizes, num_classes)
            if fallback and len(fallback) == len(sizes) and all(
                len(fallback[i]) == sizes[i] for i in range(len(sizes))
            ):
                logger.warning(
                    "Phase 2: config label_subsets không khớp state_dict; dùng fallback theo sizes. "
                    "Reserved experts (8,9) có thể ánh xạ class sai - nên lưu lại checkpoint từ "
                    "train_videomae_alpha1 với config['label_subsets'] đầy đủ."
                )
                label_subsets = fallback
                used_phase2_fallback = True
            else:
                raise ValueError(
                    "Checkpoint Phase 2 (10 experts, sum(sizes)!=51): không thể suy label_subsets. "
                    "Checkpoint phải được lưu với config['label_subsets'] khớp state_dict "
                    "(train_videomae_alpha1 đã lưu khi chạy Phase 2 xong)."
                )

    ckpt_has_adapters = any(
        k.startswith("adapters.") for k in checkpoint.get("model", {}).keys()
    )
    use_adapters = bool(model_config.get("use_adapters", False)) and ckpt_has_adapters
    pretrained_ckpt = model_config.get("pretrained_ckpt") or str(checkpoint_path)

    logger.info("Creating %s model...", ALPHA_ARCH)
    model = create_model(
        architecture=ALPHA_ARCH,
        num_classes=num_classes,
        pretrained_name=model_config.get('pretrained_name'),
        use_adapters=use_adapters,
        dropout=model_config.get('dropout', 0.1),
        pretrained_ckpt=pretrained_ckpt,
        model_name=model_config.get('model_name'),
        num_frames=model_config.get('num_frames', 16),
        tubelet_size=model_config.get('tubelet_size', 2),
        image_size=model_config.get('img_size', 224),
        patch_size=model_config.get('patch_size', 16),
        init_output_gain=model_config.get('init_output_gain', 2.0),
        init_hidden_gain=model_config.get('init_hidden_gain', 1.0),
        use_normal_init=model_config.get('use_normal_init', True),
        label_subsets=label_subsets,
        hard_mask_value=model_config.get('hard_mask_value', -1e9),
    ).to(device)

    if 'model' not in checkpoint:
        logger.error("Checkpoint không chứa 'model' state dict!")
        return
    _load_state_dict_robust(model, checkpoint['model'], logger)
    if hasattr(model, "inference_single_best_expert"):
        model.inference_single_best_expert = False
    if hasattr(model, "active_expert_idx"):
        model.active_expert_idx = None
    if used_phase2_fallback and hasattr(model, "num_active_experts"):
        model.num_active_experts = 8
        logger.info(
            "Phase 2 fallback: chỉ dùng 8 experts chính để evaluate (tránh ghi đè logits)."
        )
    model.eval()

    label_subsets_eval = label_subsets[:8] if used_phase2_fallback else label_subsets

    if train_dataset is None:
        if not train_data_dir.exists():
            logger.error(
                "Thư mục data không tồn tại: %s. Cần data_train để evaluate/retrain.",
                train_data_dir,
            )
            return
        train_dataset = VideoDataset(
            root=train_data_dir,
            num_frames=model_config.get('num_frames', 16),
            frame_stride=model_config.get('frame_stride', 2),
            image_size=model_config.get('img_size', 224),
            is_train=True,
            val_ratio=0.2,
            seed=42,
        )
    val_dataset = VideoDataset(
        root=train_data_dir,
        num_frames=model_config.get('num_frames', 16),
        frame_stride=model_config.get('frame_stride', 2),
        image_size=model_config.get('img_size', 224),
        is_train=False,
        val_ratio=0.2,
        seed=42,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info("")
    logger.info("STEP 1: EVALUATE CURRENT PERFORMANCE")
    logger.info("=" * 60)
    results = evaluate_model_detailed(
        model=model,
        val_loader=val_loader,
        device=device,
        classes=classes,
        label_subsets=label_subsets_eval,
    )
    if used_phase2_fallback and len(label_subsets) > 8:
        per_expert = evaluate_per_expert_accuracy(
            model=model,
            val_dataset=val_dataset,
            device=device,
            label_subsets=label_subsets,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        for expert_idx in range(8, len(label_subsets)):
            if expert_idx in per_expert:
                pe = per_expert[expert_idx]
                results['group_acc'][expert_idx] = pe['acc']
                results['group_details'][expert_idx] = {
                    'acc': pe['acc'],
                    'correct': pe['correct'],
                    'total': pe['total'],
                    'labels': label_subsets[expert_idx],
                }
        logger.info(
            "Per-Expert (8,9): Expert 8 acc=%.4f (%d/%d), Expert 9 acc=%.4f (%d/%d)",
            per_expert.get(8, {}).get('acc', 0),
            per_expert.get(8, {}).get('correct', 0),
            per_expert.get(8, {}).get('total', 0),
            per_expert.get(9, {}).get('acc', 0),
            per_expert.get(9, {}).get('correct', 0),
            per_expert.get(9, {}).get('total', 0),
        )
    print_performance_analysis(results, classes, label_subsets)

    weak_by_threshold = select_groups_to_improve(
        results, label_subsets, threshold=args.group_threshold
    )
    weak_bottom_k = select_bottom_k_groups(
        results, label_subsets, k=args.improve_bottom_k_groups
    )
    weak_groups = sorted(set(weak_by_threshold) | set(weak_bottom_k))
    if used_phase2_fallback and len(label_subsets) > 8:
        weak_groups_skip = [g for g in weak_groups if g >= 8]
        weak_groups = [g for g in weak_groups if g < 8]
        if weak_groups_skip:
            logger.info(
                "Phase 2 fallback: chỉ retrain experts 0-7; bỏ qua %s (subset không đúng semantics)",
                weak_groups_skip,
            )
    weak_classes = select_classes_to_improve(
        results, threshold=args.class_threshold
    )

    logger.info("")
    logger.info("STEP 2: IDENTIFY WEAK GROUPS")
    logger.info("=" * 60)
    logger.info(
        "Weak Groups (acc < %s or bottom-k=%s): %s",
        args.group_threshold, args.improve_bottom_k_groups, weak_groups,
    )
    for g_idx in weak_groups:
        details = results['group_details'][g_idx]
        logger.info(
            "  Group %s: %s (%s%%)",
            g_idx, details['acc'], details['acc'] * 100,
        )
    logger.info("Weak Classes (acc < %s): %d classes", args.class_threshold, len(weak_classes))

    if not args.retrain_experts:
        logger.info("")
        logger.info("No retraining. Use --retrain-experts to improve weak experts.")
        return

    config = get_default_config()
    config.update(model_config)
    config['batch_size'] = args.batch_size
    config['num_workers'] = args.num_workers
    config['layer_phase_max_epochs'] = args.expert_epochs
    config['expert_lr'] = args.expert_lr
    config['label_subsets'] = label_subsets

    logger.info("")
    logger.info("STEP 3: RETRAIN WEAK EXPERT HEADS")
    logger.info("=" * 60)
    expert_results = retrain_weak_experts(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        config=config,
        weak_group_indices=weak_groups,
        label_subsets=label_subsets,
        classes=classes,
    )
    for g_idx, acc in expert_results.items():
        logger.info("  Group %s: %s (%s%%)", g_idx, acc, acc * 100)

    if hasattr(model, "active_expert_idx"):
        model.active_expert_idx = None
    if hasattr(model, "inference_single_best_expert"):
        model.inference_single_best_expert = False

    logger.info("")
    logger.info("STEP 4: RE-EVALUATE AFTER IMPROVEMENTS")
    logger.info("=" * 60)
    final_results = evaluate_model_detailed(
        model=model,
        val_loader=val_loader,
        device=device,
        classes=classes,
        label_subsets=label_subsets,
    )
    print_performance_analysis(final_results, classes, label_subsets)

    logger.info("")
    logger.info("IMPROVEMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(
        "Overall Accuracy: Before %s -> After %s (delta %s)",
        results['overall_acc'],
        final_results['overall_acc'],
        final_results['overall_acc'] - results['overall_acc'],
    )

    out_name = args.output_name
    if out_name is None:
        out_name = "videomae_model_alpha_improved_heads.pt"
    improved_path = save_checkpoint(
        model=model,
        config=config,
        classes=classes,
        label_subsets=label_subsets,
        epoch=checkpoint.get('epoch', 0),
        val_acc=final_results['overall_acc'],
        output_dir=output_dir,
        output_name=out_name,
        source_checkpoint_path=checkpoint_path,
    )
    logger.info("")
    logger.info("IMPROVEMENT COMPLETED! Checkpoint đã lưu: %s", improved_path)
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
