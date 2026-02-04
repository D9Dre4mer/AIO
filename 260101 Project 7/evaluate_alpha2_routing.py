"""
Evaluate Alpha-2 routing: compare baseline (max softmax), heuristic (max margin), CatBoost.

Metrics: overall accuracy, per-class (weak), routing accuracy (CatBoost vs oracle y_route),
confusion matrix expert_true vs expert_pred.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from sota_training.config import get_default_config
from sota_training.dataset import VideoDataset
from sota_training.models import create_model
from sota_training.utils import setup_logging, log_system_info, load_checkpoint
from sota_training.sequential_layer_training import build_contiguous_label_subsets
from sota_training.meta_head_features import (
    build_meta_features,
    compute_y_route_one_sample,
    get_feature_column_order,
    DEFAULT_TOP_M,
)
from improve_weak_heads_alpha import (
    get_model_config_from_checkpoint_alpha,
    get_expert_head_sizes_from_state_dict,
    infer_label_subsets_from_state_dict,
    build_phase2_subsets_fallback,
)
from improve_weak_heads import _load_state_dict_robust

logger = logging.getLogger('sota_training')
ALPHA_ARCH = 'videomae_alpha_experts'


def _expert_to_global_class(
    all_logits: List[torch.Tensor],
    label_subsets: List[List[int]],
    expert_id: int,
    b: int,
) -> int:
    logits_b = all_logits[expert_id][b]
    if hasattr(logits_b, "cpu"):
        logits_b = logits_b.detach().cpu().numpy()
    else:
        logits_b = np.asarray(logits_b)
    pred_local = int(np.argmax(logits_b))
    return label_subsets[expert_id][pred_local]


def run_eval(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    classes: List[str],
    label_subsets: List[List[int]],
    num_active_experts: int,
    phase_id: int,
    reserved: Set[int],
    top_m: int,
    catboost_model: Optional[Any] = None,
    meta_config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[Dict]]:
    """
    Returns: all_y_true, all_y_route (oracle), pred_baseline, pred_heuristic, pred_catboost, all_rows.
    """
    model.eval()
    all_y_true = []
    all_y_route = []
    pred_baseline = []
    pred_heuristic = []
    pred_catboost = []
    all_rows = []

    with torch.no_grad():
        for videos, labels in tqdm(val_loader, desc="Eval", leave=False):
            videos = videos.to(device, non_blocking=True)
            labels_np = labels.cpu().numpy()
            signals = model.forward_expert_signals(videos)
            rows = build_meta_features(
                signals,
                phase_id=phase_id,
                num_active_experts=num_active_experts,
                reserved_expert_indices=reserved,
                top_m=top_m,
            )
            all_logits = signals["all_logits"]
            B = len(rows)
            for b in range(B):
                y_true = int(labels_np[b])
                all_y_true.append(y_true)
                all_probs_b = [signals["all_probs"][e][b : b + 1] for e in range(len(signals["all_probs"]))]
                all_logits_b = [signals["all_logits"][e][b : b + 1] for e in range(len(signals["all_logits"]))]
                y_route = compute_y_route_one_sample(
                    all_probs_b, all_logits_b, signals["label_subsets"],
                    y_true, num_active_experts,
                )
                all_y_route.append(y_route)
                all_rows.append(rows[b])

                expert_baseline = int(rows[b]["best_expert_id_by_p1"])
                expert_heuristic = int(rows[b]["best_expert_id_by_margin"])
                pred_baseline.append(_expert_to_global_class(all_logits, label_subsets, expert_baseline, b))
                pred_heuristic.append(_expert_to_global_class(all_logits, label_subsets, expert_heuristic, b))

                if catboost_model is not None and meta_config is not None:
                    import pandas as pd
                    feature_cols = meta_config.get("feature_names") or get_feature_column_order(top_m=top_m)
                    X_b = pd.DataFrame([rows[b]])[feature_cols]
                    expert_hat = int(catboost_model.predict(X_b)[0])
                    tau = float(meta_config.get("tau", 0.05))
                    gap = float(rows[b]["gap_best_second"])
                    if expert_hat != expert_baseline and gap < tau:
                        expert_final = expert_baseline
                    else:
                        expert_final = expert_hat
                    pred_catboost.append(_expert_to_global_class(all_logits, label_subsets, expert_final, b))
                else:
                    pred_catboost.append(pred_baseline[-1])

    return all_y_true, all_y_route, pred_baseline, pred_heuristic, pred_catboost, all_rows


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Alpha-2 routing: baseline vs heuristic vs CatBoost',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--meta-model', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default='./kaggle_data/data')
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--weak-class-threshold', type=float, default=0.70)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_system_info(logger, device)

    checkpoint_path = Path(args.checkpoint or './checkpoints/videomae_model_alpha1_best.pt')
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return

    checkpoint = load_checkpoint(checkpoint_path, device, logger)
    model_config = get_model_config_from_checkpoint_alpha(checkpoint_path)
    data_dir = Path(args.data_dir) / 'data_train'
    val_dataset = VideoDataset(
        root=data_dir,
        num_frames=model_config.get('num_frames', 16),
        frame_stride=model_config.get('frame_stride', 2),
        image_size=model_config.get('img_size', 224),
        is_train=False,
        val_ratio=0.2,
        seed=42,
    )
    num_classes = len(val_dataset.classes)
    classes = val_dataset.classes
    label_subsets = (
        model_config.get('label_subsets')
        or checkpoint.get('config', {}).get('label_subsets')
        or build_contiguous_label_subsets(num_classes, num_experts=model_config.get('num_experts', 8))
    )
    state_dict = checkpoint.get('model', {})
    sizes = get_expert_head_sizes_from_state_dict(state_dict)
    config_matches = (
        sizes
        and len(label_subsets) == len(sizes)
        and all(len(label_subsets[i]) == sizes[i] for i in range(len(sizes)))
    )
    if not config_matches and sizes:
        inferred = infer_label_subsets_from_state_dict(state_dict, num_classes)
        if inferred:
            label_subsets = inferred
            logger.info("evaluate_alpha2: dùng contiguous (Phase 1) cho label_subsets")
        else:
            fallback = build_phase2_subsets_fallback(sizes, num_classes)
            if fallback and len(fallback) == len(sizes) and all(
                len(fallback[i]) == sizes[i] for i in range(len(sizes))
            ):
                label_subsets = fallback
                logger.info(
                    "evaluate_alpha2: dùng fallback Phase 2 (%d experts) cho label_subsets",
                    len(label_subsets),
                )
            else:
                raise RuntimeError(
                    "Checkpoint Phase 2: label_subsets không khớp state_dict. "
                    "Chạy: python run_improve_alpha1.py --fix-config"
                )
    num_phase1 = model_config.get('num_phase1_heads', 8)
    num_active_experts = len(label_subsets)
    reserved = set(range(num_phase1, num_active_experts))
    phase_id = model_config.get('alpha1_phase', 2)

    ckpt_has_adapters = any(
        k.startswith("adapters.") for k in checkpoint.get("model", {}).keys()
    )
    use_adapters = bool(model_config.get("use_adapters", False)) and ckpt_has_adapters
    if not ckpt_has_adapters and model_config.get("use_adapters"):
        logger.info("Checkpoint không có adapters → tạo model use_adapters=False")

    model = create_model(
        architecture=ALPHA_ARCH,
        num_classes=num_classes,
        pretrained_name=None,
        use_adapters=use_adapters,
        dropout=model_config.get('dropout', 0.1),
        pretrained_ckpt=str(checkpoint_path),
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
    _load_state_dict_robust(model, checkpoint['model'], logger)
    model.num_active_experts = num_active_experts
    model.eval()

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    catboost_model = None
    meta_config = None
    meta_path = Path(args.meta_model or str(checkpoint_path.parent / 'meta' / 'meta_catboost.cbm'))
    if meta_path.exists():
        try:
            from catboost import CatBoostClassifier
            catboost_model = CatBoostClassifier()
            catboost_model.load_model(str(meta_path))
            config_path = meta_path.parent / 'meta_config.json'
            if config_path.exists():
                with open(config_path, encoding='utf-8') as f:
                    meta_config = json.load(f)
            else:
                fn_path = meta_path.parent / 'feature_names.json'
                with open(fn_path, encoding='utf-8') as f:
                    meta_config = json.load(f)
                meta_config.setdefault('tau', 0.05)
                meta_config.setdefault('num_phase1_heads', num_phase1)
                meta_config.setdefault('num_active_experts', num_active_experts)
            logger.info("Loaded CatBoost from %s", meta_path)
        except Exception as e:
            logger.warning("Could not load CatBoost: %s", e)

    y_true, y_route, pred_base, pred_heur, pred_cb, rows = run_eval(
        model, val_loader, device, classes, label_subsets,
        num_active_experts, phase_id, reserved, DEFAULT_TOP_M,
        catboost_model, meta_config,
    )
    y_true = np.array(y_true)
    y_route = np.array(y_route)
    pred_base = np.array(pred_base)
    pred_heur = np.array(pred_heur)
    pred_cb = np.array(pred_cb)

    acc_base = (pred_base == y_true).mean()
    acc_heur = (pred_heur == y_true).mean()
    acc_cb = (pred_cb == y_true).mean()

    logger.info("=" * 60)
    logger.info("Alpha-2 routing evaluation")
    logger.info("=" * 60)
    logger.info("Baseline (max p1):   overall acc = %.4f", acc_base)
    logger.info("Heuristic (max margin): overall acc = %.4f", acc_heur)
    logger.info("CatBoost routing:   overall acc = %.4f", acc_cb)
    logger.info("Delta (CatBoost - Baseline): %.4f", acc_cb - acc_base)

    if catboost_model is not None:
        expert_pred_cb = []
        for b, row in enumerate(rows):
            import pandas as pd
            feature_cols = meta_config.get("feature_names") or get_feature_column_order(top_m=DEFAULT_TOP_M)
            X_b = pd.DataFrame([row])[feature_cols]
            expert_hat = int(catboost_model.predict(X_b)[0])
            tau = float(meta_config.get("tau", 0.05))
            if expert_hat != row["best_expert_id_by_p1"] and row["gap_best_second"] < tau:
                expert_final = row["best_expert_id_by_p1"]
            else:
                expert_final = expert_hat
            expert_pred_cb.append(expert_final)
        expert_pred_cb = np.array(expert_pred_cb)
        routing_acc = (expert_pred_cb == y_route).mean()
        logger.info("Routing accuracy (CatBoost vs oracle y_route): %.4f", routing_acc)

        from sklearn.metrics import confusion_matrix
        cm_expert = confusion_matrix(y_route, expert_pred_cb)
        logger.info("Confusion matrix (expert_true vs expert_pred):")
        logger.info("\n%s", str(cm_expert))

    weak_mask = np.array([False] * len(y_true))
    for c in range(num_classes):
        acc_c = (y_true == c) & (pred_base == c)
        total_c = (y_true == c).sum()
        if total_c > 0 and acc_c.sum() / total_c < args.weak_class_threshold:
            weak_mask |= (y_true == c)
    if weak_mask.sum() > 0:
        acc_base_weak = (pred_base[weak_mask] == y_true[weak_mask]).mean()
        acc_cb_weak = (pred_cb[weak_mask] == y_true[weak_mask]).mean()
        logger.info("Weak classes (acc < %.2f): baseline acc = %.4f, CatBoost acc = %.4f, delta = %.4f",
                    args.weak_class_threshold, acc_base_weak, acc_cb_weak, acc_cb_weak - acc_base_weak)
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
