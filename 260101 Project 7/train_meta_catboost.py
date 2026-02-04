"""
Train CatBoost meta-head for Alpha-2: routing (which expert to trust).

Uses Alpha 1 checkpoint (phase 2), builds meta_train/meta_val from forward_expert_signals,
computes y_route (target), fits CatBoost, saves .cbm + feature_names.json + meta_config.json (tau).
"""

import os
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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
    SCHEMA_VERSION,
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


def _collect_meta_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    phase_id: int,
    num_active_experts: int,
    reserved_expert_indices: Set[int],
    top_m: int,
) -> List[Dict[str, Any]]:
    rows_all = []
    model.eval()
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc="Meta features", leave=False):
            videos = videos.to(device, non_blocking=True)
            labels = labels.cpu().numpy()
            signals = model.forward_expert_signals(videos)
            batch_rows = build_meta_features(
                signals,
                phase_id=phase_id,
                num_active_experts=num_active_experts,
                reserved_expert_indices=reserved_expert_indices,
                top_m=top_m,
            )
            for b, row in enumerate(batch_rows):
                y_true = int(labels[b])
                all_probs_b = [
                    signals["all_probs"][e][b : b + 1]
                    for e in range(len(signals["all_probs"]))
                ]
                all_logits_b = [
                    signals["all_logits"][e][b : b + 1]
                    for e in range(len(signals["all_logits"]))
                ]
                y_route = compute_y_route_one_sample(
                    all_probs_b,
                    all_logits_b,
                    signals["label_subsets"],
                    y_true,
                    num_active_experts,
                )
                row = dict(row)
                row["y_route"] = y_route
                row["y_true"] = y_true
                rows_all.append(row)
    return rows_all


def main():
    parser = argparse.ArgumentParser(
        description='Train CatBoost meta-head for Alpha-2 routing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Alpha 1 checkpoint (phase 2 recommended)',
    )
    parser.add_argument('--data-dir', type=str, default='./kaggle_data/data')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./checkpoints/meta',
        help='Where to save .cbm, feature_names.json, meta_config.json',
    )
    parser.add_argument('--top-m', type=int, default=DEFAULT_TOP_M)
    parser.add_argument(
        '--tau-percentile',
        type=float,
        default=15.0,
        help='Percentile of gap_best_second on val for tau (10-20)',
    )
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--catboost-iterations', type=int, default=3000)
    parser.add_argument('--catboost-depth', type=int, default=6)
    parser.add_argument('--catboost-lr', type=float, default=0.05)
    parser.add_argument('--od-wait', type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = output_dir.parent / 'logging'
    logging_dir.mkdir(parents=True, exist_ok=True)
    setup_logging('meta_catboost', logging_dir)
    logger.info("Alpha-2: Train CatBoost meta-head (routing)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_system_info(logger, device)

    checkpoint_path = Path(args.checkpoint or str(output_dir.parent / 'videomae_model_alpha1_best.pt'))
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return

    checkpoint = load_checkpoint(checkpoint_path, device, logger)
    model_config = get_model_config_from_checkpoint_alpha(checkpoint_path)

    train_data_dir = Path(args.data_dir) / 'data_train'
    if not train_data_dir.exists():
        logger.error("data_train not found: %s", train_data_dir)
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
    num_classes = len(train_dataset.classes)
    label_subsets = (
        model_config.get('label_subsets')
        or checkpoint.get('config', {}).get('label_subsets')
        or build_contiguous_label_subsets(
            num_classes,
            num_experts=model_config.get('num_experts', 8),
        )
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
            logger.info("train_meta: dùng contiguous (Phase 1) cho label_subsets")
        else:
            fallback = build_phase2_subsets_fallback(sizes, num_classes)
            if fallback and len(fallback) == len(sizes) and all(
                len(fallback[i]) == sizes[i] for i in range(len(sizes))
            ):
                label_subsets = fallback
                logger.info(
                    "train_meta: dùng fallback Phase 2 (%d experts) cho label_subsets",
                    len(label_subsets),
                )
            else:
                raise RuntimeError(
                    "Checkpoint Phase 2: label_subsets không khớp state_dict. "
                    "Chạy: python run_improve_alpha1.py --fix-config"
                )
    num_phase1 = model_config.get('num_phase1_heads') or 8
    num_active_experts = len(label_subsets)
    reserved_expert_indices = set(range(num_phase1, num_active_experts))
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info("Collecting meta train...")
    meta_train_rows = _collect_meta_dataset(
        model, train_loader, device, phase_id,
        num_active_experts, reserved_expert_indices, args.top_m,
    )
    logger.info("Collecting meta val...")
    meta_val_rows = _collect_meta_dataset(
        model, val_loader, device, phase_id,
        num_active_experts, reserved_expert_indices, args.top_m,
    )

    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas required for parquet. pip install pandas pyarrow")
        return

    feature_cols = get_feature_column_order(top_m=args.top_m)
    df_train = pd.DataFrame(meta_train_rows)
    df_val = pd.DataFrame(meta_val_rows)
    X_train = df_train[feature_cols]
    y_train = df_train["y_route"]
    X_val = df_val[feature_cols]
    y_val = df_val["y_route"]

    meta_train_path = output_dir / 'meta_train.parquet'
    meta_val_path = output_dir / 'meta_val.parquet'
    try:
        df_train.to_parquet(meta_train_path, index=False)
        df_val.to_parquet(meta_val_path, index=False)
    except ImportError:
        meta_train_path = output_dir / 'meta_train.csv'
        meta_val_path = output_dir / 'meta_val.csv'
        df_train.to_csv(meta_train_path, index=False)
        df_val.to_csv(meta_val_path, index=False)
        logger.info("pyarrow/fastparquet không có → lưu CSV. Cài pyarrow để dùng parquet: pip install pyarrow")
    logger.info("Saved %s, %s", meta_train_path, meta_val_path)

    gap_val = df_val["gap_best_second"].values
    tau = float(np.percentile(gap_val, args.tau_percentile))
    logger.info("tau (confidence guard) = %.4f (percentile %.1f of gap_best_second)", tau, args.tau_percentile)

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        logger.error("catboost required. pip install catboost")
        return

    cat_features = ["best_expert_id_by_p1", "best_expert_id_by_margin", "phase_id"]
    cat_idx = [feature_cols.index(c) for c in cat_features if c in feature_cols]

    clf = CatBoostClassifier(
        iterations=args.catboost_iterations,
        depth=args.catboost_depth,
        learning_rate=args.catboost_lr,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        od_type='Iter',
        od_wait=args.od_wait,
        use_best_model=True,
        verbose=100,
    )
    clf.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_idx if cat_idx else None,
    )
    cbm_path = output_dir / 'meta_catboost.cbm'
    clf.save_model(str(cbm_path))
    logger.info("Saved %s", cbm_path)

    meta_config = {
        "schema_version": SCHEMA_VERSION,
        "tau": tau,
        "tau_percentile": args.tau_percentile,
        "feature_names": feature_cols,
        "top_m": args.top_m,
        "num_active_experts": num_active_experts,
    }
    config_path = output_dir / 'meta_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(meta_config, f, indent=2)
    logger.info("Saved %s", config_path)

    feature_names_path = output_dir / 'feature_names.json'
    with open(feature_names_path, 'w', encoding='utf-8') as f:
        json.dump({"schema_version": SCHEMA_VERSION, "feature_names": feature_cols}, f, indent=2)
    logger.info("Saved %s", feature_names_path)

    acc_val = (clf.predict(X_val) == y_val.values).mean()
    logger.info("CatBoost val routing accuracy: %.4f", acc_val)
    logger.info("Done. Use --meta-head catboost and --meta-model %s for inference.", cbm_path)


if __name__ == '__main__':
    main()
