"""
Script chạy inference trên tập validation để kiểm tra:
- Val accuracy (overall, per-class)
- Model 12: phân bố group head (--log-group-head), hard vs soft (--soft-routing)
"""

import os
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from pathlib import Path
import logging
import argparse
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm

from sota_training.utils import log_system_info, load_checkpoint
from sota_training.dataset import VideoDataset
from sota_training.models import create_model
from sota_training.sequential_layer_training import build_contiguous_label_subsets
from run_inference_from_checkpoint import (
    get_model_config_from_checkpoint,
    list_checkpoints,
    select_checkpoint,
)

logger = logging.getLogger(__name__)


def run_validation(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    log_group_head: bool = False,
) -> tuple:
    """
    Chạy inference trên val_loader, trả về (all_preds, all_labels) và nếu log_group_head
    thì cũng trả về list group_preds (Model 12).
    """
    model.eval()
    all_preds = []
    all_labels = []
    group_preds_list = []

    if log_group_head and hasattr(model, 'group_head'):
        def _hook(module, inp, out):
            group_preds_list.append(out.argmax(dim=1).cpu().numpy())
        handle = model.group_head.register_forward_hook(_hook)

    try:
        with torch.no_grad():
            for videos, labels in tqdm(val_loader, desc="Validation", unit="batch"):
                videos = videos.to(device, non_blocking=True)
                labels = labels.to(device, dtype=torch.long)
                logits = model(videos)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    finally:
        if log_group_head and hasattr(model, 'group_head'):
            handle.remove()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    group_preds = (
        np.concatenate(group_preds_list, axis=0) if group_preds_list else None
    )
    return all_preds, all_labels, group_preds


def print_report(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    classes: list,
    group_preds: np.ndarray = None,
    label_subsets: list = None,
):
    """In báo cáo: overall acc, per-class acc, phân bố group head nếu có."""
    n = len(all_labels)
    acc = (all_preds == all_labels).mean()
    logger.info("=" * 60)
    logger.info("KẾT QUẢ VALIDATION")
    logger.info("=" * 60)
    n_correct = int(np.sum(all_preds == all_labels))
    logger.info(f"Overall accuracy: {acc:.4f} ({n_correct}/{n})")

    # Per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for p, t in zip(all_preds, all_labels):
        class_total[int(t)] += 1
        if p == t:
            class_correct[int(t)] += 1
    per_class_acc = []
    for c in range(len(classes)):
        total = class_total.get(c, 0)
        correct = class_correct.get(c, 0)
        acc_c = correct / total if total > 0 else 0.0
        per_class_acc.append((c, acc_c, total, correct))
    per_class_acc.sort(key=lambda x: x[1])
    logger.info("-" * 60)
    logger.info("Per-class accuracy (worst 10):")
    for c, acc_c, total, correct in per_class_acc[:10]:
        logger.info(f"  {classes[c]!r}: {acc_c:.4f} ({correct}/{total})")
    logger.info("  ...")
    logger.info("Per-class accuracy (best 5):")
    for c, acc_c, total, correct in per_class_acc[-5:][::-1]:
        logger.info(f"  {classes[c]!r}: {acc_c:.4f} ({correct}/{total})")

    # Phân bố group head (Model 12)
    if group_preds is not None and label_subsets is not None:
        logger.info("-" * 60)
        logger.info("Phân bố group head (số sample route vào mỗi group):")
        num_groups = len(label_subsets)
        counts = np.bincount(group_preds, minlength=num_groups)
        for g in range(num_groups):
            sub = label_subsets[g][:3]
            subset_classes = [classes[i] for i in sub]
            subset_str = f"{subset_classes}..." if len(label_subsets[g]) > 3 else str(subset_classes)
            pct = 100.0 * counts[g] / len(group_preds) if len(group_preds) > 0 else 0
            logger.info(f"  Group {g}: {counts[g]} ({pct:.1f}%) — {subset_str}")
        if (counts > 0).sum() <= 2:
            logger.warning("  → Chỉ 1–2 group có sample → group head có thể lệch (collapse).")


def main():
    parser = argparse.ArgumentParser(
        description='Inference trên tập val: accuracy, (Model 12) phân bố group head',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='File checkpoint .pt (nếu không có sẽ hiện menu chọn)')
    parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints',
                        help='Thư mục chứa .pt khi chọn checkpoint tương tác')
    parser.add_argument('--data-dir', type=str, default='./kaggle_data/data',
                        help='Thư mục data (chứa data_train)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Val ratio (khớp với lúc train nếu có)')
    parser.add_argument('--seed', type=int, default=42, help='Seed cho train/val split')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader num_workers')
    parser.add_argument('--log-group-head', action='store_true',
                        help='Model 12: in phân bố group head')
    parser.add_argument('--soft-routing', action='store_true',
                        help='Model 12: soft routing thay vì hard (tránh collapse)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    data_dir = Path(args.data_dir)
    train_data_dir = data_dir / 'data_train'
    if not train_data_dir.exists():
        logger.error(f"Không tìm thấy {train_data_dir}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    log_system_info(logger, device)

    # Chọn checkpoint: từ --checkpoint hoặc menu tương tác
    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path is not None and checkpoint_path.exists():
        selected = checkpoint_path
    else:
        if checkpoint_path is not None:
            logger.warning(f"Checkpoint không tồn tại: {checkpoint_path}")
        checkpoints = list_checkpoints(checkpoints_dir)
        if not checkpoints:
            logger.error(f"Không có checkpoint nào trong {checkpoints_dir}")
            return
        selected = select_checkpoint(checkpoints, checkpoint_path)
    checkpoint_path = selected

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device, logger)
    model_config = get_model_config_from_checkpoint(checkpoint_path)

    # Classes từ checkpoint hoặc dataset
    if 'classes' in checkpoint and checkpoint['classes']:
        classes = list(checkpoint['classes'])
        num_classes = len(classes)
        logger.info(f"Classes từ checkpoint: {num_classes}")
    else:
        _ds = VideoDataset(
            root=train_data_dir,
            num_frames=model_config.get('num_frames', 16),
            frame_stride=model_config.get('frame_stride', 2),
            image_size=model_config.get('img_size', 224),
            is_train=True,
            val_ratio=0.0,
            seed=42,
        )
        classes = _ds.classes
        num_classes = len(classes)
        logger.info(f"Classes từ dataset: {num_classes}")

    # Label subsets cho Model 12/11
    label_subsets = None
    if model_config['architecture'] in ['videomae_group_gated_experts', 'videomae_global_residual_experts']:
        if 'label_subsets' in model_config and model_config['label_subsets']:
            label_subsets = model_config['label_subsets']
        elif checkpoint.get('config') and checkpoint['config'].get('label_subsets'):
            label_subsets = checkpoint['config']['label_subsets']
        elif checkpoint.get('label_subsets'):
            label_subsets = checkpoint['label_subsets']
        else:
            ne = model_config.get('num_experts', 8)
            label_subsets = build_contiguous_label_subsets(num_classes, num_experts=ne)
        logger.info(f"Label subsets: {len(label_subsets)} groups")

    # Tạo model: Model 12/11 cần load backbone từ cùng checkpoint, tránh mismatch với heads
    logger.info(f"Creating model: {model_config['architecture']}")
    ckpt_has_adapters = any(
        k.startswith("adapters.") for k in checkpoint.get("model", {}).keys()
    )
    use_adapters = bool(model_config.get("use_adapters", False)) and ckpt_has_adapters
    if not ckpt_has_adapters and model_config.get("use_adapters", False):
        logger.info("Checkpoint không có adapters → tạo model với use_adapters=False")
    pretrained_ckpt = model_config.get('pretrained_ckpt')
    if pretrained_ckpt is None and model_config['architecture'] in [
        'videomae_group_gated_experts', 'videomae_global_residual_experts'
    ]:
        pretrained_ckpt = str(checkpoint_path) if checkpoint_path.exists() else None
        if pretrained_ckpt:
            logger.info("Using checkpoint as pretrained_ckpt (load backbone from same file)")
    kw = {
        'architecture': model_config['architecture'],
        'num_classes': num_classes,
        'pretrained_name': model_config.get('pretrained_name'),
        'use_adapters': use_adapters,
        'dropout': model_config.get('dropout', 0.1),
        'drop_path_rate': model_config.get('drop_path_rate', 0.0),
        'pretrained_ckpt': pretrained_ckpt,
        'model_name': model_config.get('model_name'),
        'num_frames': model_config.get('num_frames', 16),
        'tubelet_size': model_config.get('tubelet_size', 2),
        'image_size': model_config.get('img_size', 224),
        'patch_size': model_config.get('patch_size', 16),
    }
    if model_config['architecture'] in ['videomae_group_gated_experts', 'videomae_global_residual_experts']:
        kw['label_subsets'] = label_subsets
        kw['init_output_gain'] = model_config.get('init_output_gain', 2.0)
        kw['init_hidden_gain'] = model_config.get('init_hidden_gain', 1.0)
        kw['use_normal_init'] = model_config.get('use_normal_init', True)
        if model_config['architecture'] == 'videomae_group_gated_experts':
            kw['hard_mask_value'] = model_config.get('hard_mask_value', -1e9)
    model = create_model(**kw).to(device)
    load_result = model.load_state_dict(checkpoint['model'], strict=False)
    missing_keys = getattr(load_result, 'missing_keys', [])
    unexpected_keys = getattr(load_result, 'unexpected_keys', [])
    if missing_keys:
        logger.warning("load_state_dict missing_keys: %d — %s", len(missing_keys), missing_keys[:15])
    if unexpected_keys:
        logger.warning("load_state_dict unexpected_keys: %d — %s", len(unexpected_keys), unexpected_keys[:15])
    model.eval()

    # Model 12: cấu hình routing
    if model_config.get('architecture') == 'videomae_group_gated_experts':
        if hasattr(model, 'active_expert_idx'):
            model.active_expert_idx = None
        use_hard = not args.soft_routing
        if hasattr(model, 'hard_routing_enabled'):
            model.hard_routing_enabled = use_hard
        logger.info(f"Model 12: hard_routing={'ON' if use_hard else 'OFF (soft)'}")

    # Val dataset & loader
    val_dataset = VideoDataset(
        root=train_data_dir,
        num_frames=model_config.get('num_frames', 16),
        frame_stride=model_config.get('frame_stride', 2),
        image_size=model_config.get('img_size', 224),
        is_train=False,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    logger.info(
        f"Val samples: {len(val_dataset)} (val_ratio={args.val_ratio}, seed={args.seed})"
    )

    # Chạy validation
    all_preds, all_labels, group_preds = run_validation(
        model, val_loader, device, num_classes,
        log_group_head=args.log_group_head,
    )

    # Báo cáo
    print_report(
        all_preds, all_labels, classes,
        group_preds=group_preds,
        label_subsets=label_subsets if args.log_group_head else None,
    )


if __name__ == '__main__':
    main()
