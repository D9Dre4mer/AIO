"""
Script để cải thiện các head có group và classes chưa tốt.

Chức năng:
1. Phân tích performance của từng group và class từ checkpoint
2. Xác định các group/classes cần cải thiện
3. Retrain riêng các expert heads cho các group có performance thấp
4. Retrain group head với focus vào các group yếu
5. Save checkpoint mới
"""

import os

# Fix OpenMP duplicate library error on Windows
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from pathlib import Path
import logging
import argparse
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# GPU optimization settings
cuda_launch_blocking = os.environ.get('CUDA_LAUNCH_BLOCKING', '0')
os.environ['CUDA_LAUNCH_BLOCKING'] = cuda_launch_blocking
torch.backends.cudnn.benchmark = True

from sota_training.config import get_default_config
from sota_training.utils import setup_logging, log_system_info, load_checkpoint
from sota_training.dataset import VideoDataset
from sota_training.models import create_model
from sota_training.training import evaluate
from sota_training.sequential_layer_training import (
    train_expert_head_phase,
    freeze_all_heads,
    unfreeze_single_expert_only,
    freeze_all_backbone_layers,
    FilteredVideoDataset,
    build_contiguous_label_subsets
)
from sota_training.expert_training import (
    _train_group_head_stage,
    _make_full_loaders,
)
from sota_training.sequential_layer_training import (
    freeze_group_head,
    unfreeze_group_head_only,
    freeze_all_backbone_layers,
)
from torch.utils.data import DataLoader

# Dùng logger sota_training để đồng bộ với setup_logging(model_id, logging_dir)
logger = logging.getLogger('sota_training')


def get_model_config_from_checkpoint(checkpoint_path: Path) -> dict:
    """Lấy config từ checkpoint hoặc suy luận từ tên file/state_dict."""
    config = get_default_config()
    checkpoint = None
    
    # Thử load checkpoint để lấy config
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            # Merge toàn bộ config từ checkpoint (không chỉ keys có trong default)
            config.update(saved_config)
            logger.info(f"Loaded config from checkpoint: architecture={config.get('architecture')}")
    except Exception as e:
        logger.warning(f"Không thể load config từ checkpoint: {e}")
    
    # Nếu không có config trong checkpoint, suy luận từ state_dict
    if checkpoint is not None and 'model' in checkpoint:
        state_dict = checkpoint['model']
        state_keys = set(state_dict.keys())
        
        # Kiểm tra model 12: có group_head và expert_heads
        if 'group_head.0.weight' in state_keys and 'expert_heads.0.0.weight' in state_keys:
            config['architecture'] = 'videomae_group_gated_experts'
            logger.info("Detected Model 12 (Group-Gated Experts) from state_dict")
        # Kiểm tra model 11: có global_head và expert_heads (không có group_head)
        elif 'global_head' in str(state_keys) and 'expert_heads.0.0.weight' in state_keys:
            config['architecture'] = 'videomae_global_residual_experts'
            logger.info("Detected Model 11 (Global Residual Experts) from state_dict")
        # Kiểm tra model có adapters
        elif 'adapters.0.norm.weight' in state_keys:
            config['use_adapters'] = True
            logger.info("Detected adapters in state_dict")
    
    # Suy luận architecture từ tên file (fallback nếu không có trong config/state_dict)
    if 'architecture' not in config or config['architecture'] not in [
        'videomae', 'videomae_group_gated_experts', 'videomae_global_residual_experts',
        'timesformer', 'vit', 'swin'
    ]:
        filename = checkpoint_path.name.lower()
        if 'model_12' in filename or 'model12' in filename:
            config['architecture'] = 'videomae_group_gated_experts'
            logger.info("Detected Model 12 from filename")
        elif 'model_11' in filename or 'model11' in filename:
            config['architecture'] = 'videomae_global_residual_experts'
            logger.info("Detected Model 11 from filename")
        elif 'videomae' in filename:
            config['architecture'] = 'videomae'
            config['model_name'] = 'MCG-NJU/videomae-large-finetuned-kinetics'
        elif 'timesformer' in filename:
            config['architecture'] = 'timesformer'
        elif 'vit' in filename:
            config['architecture'] = 'vit'
        elif 'swin' in filename:
            config['architecture'] = 'swin'
    
    # Default VideoMAE parameters
    if config['architecture'] in ['videomae', 'videomae_group_gated_experts', 'videomae_global_residual_experts']:
        if 'model_name' not in config:
            config['model_name'] = 'MCG-NJU/videomae-large-finetuned-kinetics'
        config.setdefault('tubelet_size', 2)
        config.setdefault('patch_size', 16)
        config.setdefault('num_frames', 16)
        config.setdefault('frame_stride', 2)
        config.setdefault('img_size', 224)
        config.setdefault('image_size', 224)
    
    return config


def evaluate_model_detailed(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    classes: List[str],
    label_subsets: Optional[List[List[int]]] = None,
) -> Dict:
    """
    Evaluate model và trả về chi tiết performance của từng group và class.
    
    Returns:
        Dict chứa:
            - overall_acc: Overall accuracy
            - class_acc: Dict[class_idx] -> accuracy
            - group_acc: Dict[group_idx] -> accuracy
            - class_details: Dict với thông tin chi tiết từng class
            - group_details: Dict với thông tin chi tiết từng group
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    logger.info("=" * 60)
    logger.info("Evaluating Model Performance")
    logger.info("=" * 60)
    
    with torch.no_grad():
        progress = tqdm(val_loader, desc="Evaluating", leave=False)
        for videos, labels in progress:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            
            with torch.amp.autocast(
                device_type='cuda',
                enabled=(device.type == 'cuda')
            ):
                logits = model(videos)
            
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Overall accuracy
    overall_acc = (all_preds == all_labels).mean()
    
    # Per-class accuracy
    num_classes = len(classes)
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_acc = {}
    
    for true_label, pred_label in zip(all_labels, all_preds):
        class_total[true_label] += 1
        if true_label == pred_label:
            class_correct[true_label] += 1
    
    for class_idx in range(num_classes):
        if class_total[class_idx] > 0:
            class_acc[class_idx] = class_correct[class_idx] / class_total[class_idx]
        else:
            class_acc[class_idx] = 0.0
    
    # Group accuracy (if label_subsets provided)
    group_acc = {}
    group_details = {}
    if label_subsets is not None:
        # Build label to group mapping
        label_to_group = {}
        for group_idx, subset in enumerate(label_subsets):
            for label in subset:
                label_to_group[label] = group_idx
        
        group_correct = defaultdict(int)
        group_total = defaultdict(int)
        
        for true_label, pred_label in zip(all_labels, all_preds):
            true_group = label_to_group.get(true_label, -1)
            pred_group = label_to_group.get(pred_label, -1)
            if true_group >= 0:
                group_total[true_group] += 1
                if true_group == pred_group:
                    group_correct[true_group] += 1
        
        for group_idx in range(len(label_subsets)):
            if group_total[group_idx] > 0:
                group_acc[group_idx] = (
                    group_correct[group_idx] / group_total[group_idx]
                )
                group_details[group_idx] = {
                    'acc': group_acc[group_idx],
                    'correct': group_correct[group_idx],
                    'total': group_total[group_idx],
                    'labels': label_subsets[group_idx]
                }
            else:
                group_acc[group_idx] = 0.0
                group_details[group_idx] = {
                    'acc': 0.0,
                    'correct': 0,
                    'total': 0,
                    'labels': label_subsets[group_idx]
                }
    
    # Class details
    class_details = {}
    for class_idx in range(num_classes):
        class_details[class_idx] = {
            'name': classes[class_idx],
            'acc': class_acc[class_idx],
            'correct': class_correct[class_idx],
            'total': class_total[class_idx]
        }
    
    return {
        'overall_acc': overall_acc,
        'class_acc': class_acc,
        'group_acc': group_acc,
        'class_details': class_details,
        'group_details': group_details
    }


def print_performance_analysis(results: Dict, classes: List[str], label_subsets: Optional[List[List[int]]] = None):
    """In ra phân tích performance."""
    logger.info("")
    logger.info("📊 PERFORMANCE ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Overall Accuracy: {results['overall_acc']:.4f} ({results['overall_acc']*100:.2f}%)")
    logger.info("")
    
    # Top worst classes
    sorted_classes = sorted(
        results['class_acc'].items(),
        key=lambda x: x[1]
    )
    
    logger.info("🔴 Top 15 Worst Performing Classes:")
    for class_idx, acc in sorted_classes[:15]:
        details = results['class_details'][class_idx]
        logger.info(
            f"  {details['name']:30s} | Acc: {acc:.4f} ({acc*100:.2f}%) | "
            f"Correct: {details['correct']}/{details['total']}"
        )
    
    logger.info("")
    logger.info("🟢 Top 10 Best Performing Classes:")
    for class_idx, acc in sorted_classes[-10:][::-1]:
        details = results['class_details'][class_idx]
        logger.info(
            f"  {details['name']:30s} | Acc: {acc:.4f} ({acc*100:.2f}%) | "
            f"Correct: {details['correct']}/{details['total']}"
        )
    
    # Group accuracy (if available)
    if label_subsets is not None and results['group_acc']:
        logger.info("")
        logger.info("📦 Group-Level Accuracy:")
        sorted_groups = sorted(
            results['group_acc'].items(),
            key=lambda x: x[1]
        )
        for group_idx, acc in sorted_groups:
            details = results['group_details'][group_idx]
            logger.info(
                f"  Group {group_idx} (Labels {details['labels'][0]}-{details['labels'][-1]}): "
                f"{acc:.4f} ({acc*100:.2f}%) | Correct: {details['correct']}/{details['total']}"
            )
    
    logger.info("=" * 60)


def select_groups_to_improve(results: Dict, label_subsets: Optional[List[List[int]]] = None,
                             threshold: float = 0.85) -> List[int]:
    """Chọn các group có accuracy thấp hơn threshold."""
    if label_subsets is None or not results['group_acc']:
        return []
    weak_groups = []
    for group_idx, acc in results['group_acc'].items():
        if acc < threshold:
            weak_groups.append(group_idx)
    return sorted(weak_groups)


def select_bottom_k_groups(
    results: Dict,
    label_subsets: Optional[List[List[int]]] = None,
    k: int = 0,
) -> List[int]:
    """Chọn k group có accuracy thấp nhất (để luôn retrain dù trên threshold)."""
    if k <= 0 or label_subsets is None or not results.get('group_acc'):
        return []
    sorted_groups = sorted(
        results['group_acc'].items(),
        key=lambda x: x[1],
    )
    return [group_idx for group_idx, _ in sorted_groups[:k]]


def select_classes_to_improve(results: Dict, threshold: float = 0.70) -> List[int]:
    """Chọn các class có accuracy thấp hơn threshold."""
    weak_classes = []
    for class_idx, acc in results['class_acc'].items():
        if acc < threshold:
            weak_classes.append(class_idx)
    
    return sorted(weak_classes)


def retrain_weak_experts(
    model: torch.nn.Module,
    train_dataset,
    val_dataset,
    device: torch.device,
    config: Dict,
    weak_group_indices: List[int],
    label_subsets: List[List[int]],
    classes: List[str],
) -> Dict[str, float]:
    """
    Retrain các expert heads cho các group yếu.
    
    Returns:
        Dict[group_idx] -> best_val_acc
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("RETRAINING WEAK EXPERT HEADS")
    logger.info("=" * 60)
    
    freeze_all_backbone_layers(model)
    freeze_all_heads(model)
    
    results = {}
    
    for group_idx in weak_group_indices:
        if group_idx >= len(label_subsets):
            logger.warning(f"Group {group_idx} out of range, skipping")
            continue
        
        label_subset = label_subsets[group_idx]
        expert_idx = group_idx  # Assuming group_idx == expert_idx
        
        logger.info("")
        logger.info(f"Retraining Expert {expert_idx} for Group {group_idx}")
        logger.info(f"  Labels: {label_subset}")
        logger.info(f"  Classes: {[classes[i] for i in label_subset]}")
        
        # Retrain expert
        best_acc, history = train_expert_head_phase(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            config=config,
            expert_idx=expert_idx,
            label_subset=label_subset,
            phase_name=f"Improve Expert {expert_idx}"
        )
        
        results[group_idx] = best_acc
        logger.info(f"  ✓ Expert {expert_idx} best val acc: {best_acc:.4f}")
    
    return results


def retrain_group_head(
    model: torch.nn.Module,
    train_dataset,
    val_dataset,
    device: torch.device,
    config: Dict,
    classes: List[str],
    label_subsets: List[List[int]],
    output_dir: Path,
    history_plot_path: Path,
    source_checkpoint_path: Optional[Path] = None,
) -> Tuple[float, Optional[Path]]:
    """
    Retrain group head với focus vào các group yếu.

    Stage G best được lưu vào output_dir / "improve_stage_g_best.pt" để không ghi đè
    checkpoint nguồn. Caller phải load từ path trả về trước khi evaluate/save improved.

    Returns:
        (best_validation_accuracy, path_where_best_was_saved or None)
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("RETRAINING GROUP HEAD")
    logger.info("=" * 60)

    stage_g_save_path = output_dir / "improve_stage_g_best.pt"
    logger.info(f"Stage G best will be saved to: {stage_g_save_path} (source not overwritten)")

    train_loader, val_loader = _make_full_loaders(
        train_dataset,
        val_dataset,
        config
    )

    # Stronger regularization when retraining group head from improve (reduce overfit)
    stage_g_config = dict(config)
    stage_g_config['label_subsets'] = label_subsets
    stage_g_config['group_weight_decay'] = max(
        float(stage_g_config.get('group_weight_decay', 0.05)), 0.15
    )
    stage_g_config['group_label_smoothing'] = max(
        float(stage_g_config.get('group_label_smoothing', 0.1)), 0.2
    )
    logger.info(
        f"Stage G from improve: group_weight_decay={stage_g_config['group_weight_decay']:.2f}, "
        f"group_label_smoothing={stage_g_config['group_label_smoothing']:.2f}"
    )

    combined_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    best_acc, _, _ = _train_group_head_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=stage_g_config,
        combined_history=combined_history,
        checkpoint_path=stage_g_save_path,
        history_plot_path=history_plot_path,
        classes=classes,
        start_epoch_idx=0,
        best_overall=0.0,
        use_overall_class_acc_for_best=True,
    )

    logger.info(f"  ✓ Group head best val acc (overall class): {best_acc:.4f}")
    return best_acc, stage_g_save_path


def save_checkpoint(
    model: torch.nn.Module,
    config: Dict,
    classes: List[str],
    label_subsets: Optional[List[List[int]]] = None,
    epoch: int = 0,
    val_acc: float = 0.0,
    output_dir: Optional[Path] = None,
    output_name: Optional[str] = None,
    source_checkpoint_path: Optional[Path] = None,
):
    """
    Lưu checkpoint cải tiến riêng (không ghi đè checkpoint gốc).

    - output_dir: thư mục lưu (mặc định: ./checkpoints).
    - output_name: tên file (ví dụ videomae_model_12_improved_heads.pt).
      Nếu None và source_checkpoint_path có, dùng {stem}_improved_heads.pt.
    """
    out_dir = output_dir if output_dir is not None else Path("./checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_name is not None:
        name = output_name if output_name.endswith(".pt") else f"{output_name}.pt"
    elif source_checkpoint_path is not None:
        name = f"{source_checkpoint_path.stem}_improved_heads.pt"
    else:
        name = "improved_heads.pt"

    output_path = out_dir / name

    checkpoint = {
        'model': model.state_dict(),
        'config': config,
        'classes': classes,
        'epoch': epoch,
        'val_acc': val_acc,
    }

    if label_subsets is not None:
        checkpoint['label_subsets'] = label_subsets

    torch.save(checkpoint, output_path)
    logger.info(f"Saved improved checkpoint (riêng từ checkpoint 12): {output_path}")
    return output_path


def _load_state_dict_robust(model: torch.nn.Module, state_dict: dict, logger_instance: logging.Logger) -> bool:
    """
    Load state_dict vào model, thử chuẩn hóa key nếu có mismatch (vd. prefix 'module.' từ DataParallel).
    Returns True nếu load thành công (không còn missing keys hoặc đã thử strip module.).
    """
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if not missing_keys:
        logger_instance.info("Model weights loaded (strict match)")
        return True
    # Thử bỏ prefix "module." nếu mọi unexpected key đều có prefix đó (checkpoint từ DataParallel)
    if unexpected_keys and all(k.startswith("module.") for k in unexpected_keys):
        state_dict_stripped = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        missing_keys2, unexpected_keys2 = model.load_state_dict(state_dict_stripped, strict=False)
        if not missing_keys2:
            logger_instance.info("Model weights loaded (after stripping 'module.' prefix)")
            return True
        logger_instance.warning(
            f"After stripping 'module.': still missing={len(missing_keys2)}, unexpected={len(unexpected_keys2)}"
        )
    # Phân loại missing keys theo prefix để debug
    by_prefix = {}
    for k in missing_keys:
        p = k.split(".", 1)[0] if "." in k else k
        by_prefix.setdefault(p, []).append(k)
    logger_instance.warning(f"Missing keys: {len(missing_keys)} (by prefix: {[(p, len(v)) for p, v in by_prefix.items()]})")
    for p in sorted(by_prefix.keys()):
        keys = by_prefix[p][:5]
        logger_instance.warning(f"  {p}* sample: {keys}")
    return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Cải thiện các head có group và classes chưa tốt',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/videomae_model_12_best.pt', help='Checkpoint path (mặc định: checkpoint model 12)')
    parser.add_argument('--data-dir', type=str, default='./kaggle_data/data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./checkpoints', help='Thư mục lưu checkpoint cải tiến (riêng, không ghi đè checkpoint 12)')
    parser.add_argument('--output-name', type=str, default=None, help='Tên file checkpoint cải tiến (mặc định: videomae_model_12_improved_heads.pt khi nguồn là model 12)')
    parser.add_argument('--group-threshold', type=float, default=0.85, help='Group accuracy threshold để xác định group yếu')
    parser.add_argument('--improve-bottom-k-groups', type=int, default=0, help='Luôn retrain expert của k group có acc thấp nhất (vd. 2) bất kể threshold')
    parser.add_argument('--class-threshold', type=float, default=0.70, help='Class accuracy threshold để xác định class yếu')
    parser.add_argument('--retrain-experts', action='store_true', help='Retrain expert heads cho các group yếu')
    parser.add_argument('--retrain-group-head', action='store_true', help='Retrain group head')
    parser.add_argument('--expert-epochs', type=int, default=30, help='Max epochs cho expert training')
    parser.add_argument('--group-head-epochs', type=int, default=50, help='Max epochs cho group head training')
    parser.add_argument('--expert-lr', type=float, default=2e-4, help='Learning rate cho expert training')
    parser.add_argument('--group-head-lr', type=float, default=1e-3, help='Learning rate cho group head training')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Setup logging (model_id=12 vì script dùng cho cải thiện heads từ checkpoint model 12)
    logging_dir = Path(args.output_dir).parent / 'logging'
    logging_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(model_id=12, logging_dir=logging_dir)
    logger.info("=" * 60)
    logger.info("IMPROVE WEAK HEADS SCRIPT")
    logger.info("=" * 60)
    
    # Paths
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint không tồn tại: {checkpoint_path}")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    log_system_info(logger, device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device, logger)
    
    # Get model config
    model_config = get_model_config_from_checkpoint(checkpoint_path)
    
    # Get classes
    train_data_dir = data_dir / 'data_train'
    if train_data_dir.exists():
        train_dataset = VideoDataset(
            root=train_data_dir,
            num_frames=model_config.get('num_frames', 16),
            frame_stride=model_config.get('frame_stride', 2),
            image_size=model_config.get('img_size', 224),
            is_train=True,
            val_ratio=0.2,
            seed=42
        )
        num_classes = len(train_dataset.classes)
        classes = train_dataset.classes
        logger.info(f"Found {num_classes} classes from dataset")
    elif 'classes' in checkpoint:
        classes = checkpoint['classes']
        num_classes = len(classes)
        logger.info(f"Found {num_classes} classes from checkpoint")
    else:
        logger.error("Không tìm thấy classes")
        return
    
    # Get label_subsets
    label_subsets = None
    if model_config['architecture'] in ['videomae_group_gated_experts', 'videomae_global_residual_experts']:
        if 'label_subsets' in model_config:
            label_subsets = model_config['label_subsets']
        elif 'config' in checkpoint and 'label_subsets' in checkpoint['config']:
            label_subsets = checkpoint['config']['label_subsets']
        else:
            num_experts = model_config.get('num_experts', 8)
            label_subsets = build_contiguous_label_subsets(num_classes, num_experts=num_experts)
            logger.info(f"Created label_subsets: {len(label_subsets)} groups")
    
    if label_subsets is None:
        logger.error("Model không phải group-gated experts, không thể cải thiện")
        return
    
    # Create model (ưu tiên load backbone từ checkpoint để trùng cấu trúc với lúc train)
    # Khớp use_adapters với checkpoint: nếu checkpoint không có key "adapters.*" thì bắt buộc use_adapters=False
    ckpt_has_adapters = any(k.startswith("adapters.") for k in checkpoint.get("model", {}).keys())
    use_adapters = bool(model_config.get("use_adapters", False)) and ckpt_has_adapters
    if not ckpt_has_adapters and model_config.get("use_adapters", False):
        logger.info("Checkpoint không có adapters → tạo model với use_adapters=False để khớp cấu trúc")
    logger.info(f"Creating {model_config['architecture']} model...")
    create_kwargs = {
        'architecture': model_config['architecture'],
        'num_classes': num_classes,
        'pretrained_name': model_config.get('pretrained_name'),
        'use_adapters': use_adapters,
        'dropout': model_config.get('dropout', 0.1),
        'drop_path_rate': model_config.get('drop_path_rate', 0.0),
        'pretrained_ckpt': str(checkpoint_path) if checkpoint_path.exists() else model_config.get('pretrained_ckpt'),
        'model_name': model_config.get('model_name'),
        'num_frames': model_config.get('num_frames', 16),
        'tubelet_size': model_config.get('tubelet_size', 2),
        'image_size': model_config.get('img_size', 224),
        'patch_size': model_config.get('patch_size', 16),
    }
    
    if model_config['architecture'] in ['videomae_group_gated_experts', 'videomae_global_residual_experts']:
        create_kwargs['label_subsets'] = label_subsets
        create_kwargs['init_output_gain'] = model_config.get('init_output_gain', 2.0)
        create_kwargs['init_hidden_gain'] = model_config.get('init_hidden_gain', 1.0)
        create_kwargs['use_normal_init'] = model_config.get('use_normal_init', True)
        if model_config['architecture'] == 'videomae_group_gated_experts':
            create_kwargs['hard_mask_value'] = model_config.get('hard_mask_value', -1e9)
    
    model = create_model(**create_kwargs).to(device)
    
    # Load toàn bộ state_dict từ checkpoint (backbone đã load từ pretrained_ckpt trong create_model,
    # nhưng load lại đầy đủ để đồng bộ group_head + expert_heads; chuẩn hóa key nếu cần)
    if 'model' not in checkpoint:
        logger.error("Checkpoint không chứa 'model' state dict!")
        return
    state = checkpoint['model']
    ok = _load_state_dict_robust(model, state, logger)
    if not ok and state.keys():
        # In vài key mẫu từ checkpoint để debug
        sample = list(state.keys())[:5]
        logger.info(f"Checkpoint key sample: {sample}")
    
    model.eval()
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = VideoDataset(
        root=train_data_dir,
        num_frames=model_config.get('num_frames', 16),
        frame_stride=model_config.get('frame_stride', 2),
        image_size=model_config.get('img_size', 224),
        is_train=True,
        val_ratio=0.2,
        seed=42
    )
    val_dataset = VideoDataset(
        root=train_data_dir,
        num_frames=model_config.get('num_frames', 16),
        frame_stride=model_config.get('frame_stride', 2),
        image_size=model_config.get('img_size', 224),
        is_train=False,
        val_ratio=0.2,
        seed=42
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Evaluate current performance
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 1: EVALUATE CURRENT PERFORMANCE")
    logger.info("=" * 60)
    
    results = evaluate_model_detailed(
        model=model,
        val_loader=val_loader,
        device=device,
        classes=classes,
        label_subsets=label_subsets
    )
    
    print_performance_analysis(results, classes, label_subsets)
    
    # Identify weak groups and classes (threshold-based + bottom-k by accuracy)
    weak_by_threshold = select_groups_to_improve(results, label_subsets, threshold=args.group_threshold)
    weak_bottom_k = select_bottom_k_groups(results, label_subsets, k=getattr(args, 'improve_bottom_k_groups', 0))
    weak_groups = sorted(set(weak_by_threshold) | set(weak_bottom_k))
    weak_classes = select_classes_to_improve(results, threshold=args.class_threshold)

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: IDENTIFY WEAK GROUPS AND CLASSES")
    logger.info("=" * 60)
    logger.info(f"Weak Groups (acc < {args.group_threshold} or bottom-k={getattr(args, 'improve_bottom_k_groups', 0)}): {weak_groups}")
    for g_idx in weak_groups:
        details = results['group_details'][g_idx]
        logger.info(f"  Group {g_idx}: {details['acc']:.4f} ({details['acc']*100:.2f}%)")
    
    logger.info(f"Weak Classes (acc < {args.class_threshold}): {len(weak_classes)} classes")
    for c_idx in weak_classes[:10]:  # Show first 10
        details = results['class_details'][c_idx]
        logger.info(f"  {details['name']}: {details['acc']:.4f} ({details['acc']*100:.2f}%)")
    
    if not args.retrain_experts and not args.retrain_group_head:
        logger.info("")
        logger.info("No retraining flags set. Use --retrain-experts and/or --retrain-group-head to improve.")
        return
    
    # Prepare config for training
    config = get_default_config()
    config.update(model_config)
    config['batch_size'] = args.batch_size
    config['num_workers'] = args.num_workers
    config['layer_phase_max_epochs'] = args.expert_epochs
    config['group_head_epochs'] = args.group_head_epochs
    config['expert_lr'] = args.expert_lr
    config['group_head_lr'] = args.group_head_lr
    config['label_subsets'] = label_subsets
    
    # Retrain weak experts
    if args.retrain_experts and weak_groups:
        logger.info("")
        logger.info("=" * 60)
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
            classes=classes
        )
        
        logger.info("")
        logger.info("Expert Retraining Results:")
        for g_idx, acc in expert_results.items():
            logger.info(f"  Group {g_idx}: {acc:.4f} ({acc*100:.2f}%)")
    
    stage_g_best_path: Optional[Path] = None
    if args.retrain_group_head:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 4: RETRAIN GROUP HEAD")
        logger.info("=" * 60)

        history_plot_path = output_dir / "improved_training_history.png"
        _, stage_g_best_path = retrain_group_head(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            config=config,
            classes=classes,
            label_subsets=label_subsets,
            output_dir=output_dir,
            history_plot_path=history_plot_path,
            source_checkpoint_path=checkpoint_path,
        )
        if stage_g_best_path is not None and stage_g_best_path.exists():
            ckpt = torch.load(stage_g_best_path, map_location=device, weights_only=False)
            if "model" in ckpt:
                _load_state_dict_robust(model, ckpt["model"], logger)
                logger.info(f"Loaded best Stage G checkpoint into model before re-eval: {stage_g_best_path}")

    # Re-evaluate after improvements
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: RE-EVALUATE AFTER IMPROVEMENTS")
    logger.info("=" * 60)
    
    final_results = evaluate_model_detailed(
        model=model,
        val_loader=val_loader,
        device=device,
        classes=classes,
        label_subsets=label_subsets
    )
    
    print_performance_analysis(final_results, classes, label_subsets)
    
    # Compare before/after
    logger.info("")
    logger.info("=" * 60)
    logger.info("IMPROVEMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Overall Accuracy:")
    logger.info(f"  Before: {results['overall_acc']:.4f} ({results['overall_acc']*100:.2f}%)")
    logger.info(f"  After:  {final_results['overall_acc']:.4f} ({final_results['overall_acc']*100:.2f}%)")
    logger.info(f"  Improvement: {final_results['overall_acc'] - results['overall_acc']:.4f} ({(final_results['overall_acc'] - results['overall_acc'])*100:.2f}%)")
    
    # Save improved checkpoint riêng từ checkpoint 12 (không ghi đè file gốc)
    logger.info("")
    logger.info("=" * 60)
    logger.info("SAVING IMPROVED CHECKPOINT (RIÊNG)")
    logger.info("=" * 60)

    out_name = args.output_name
    if out_name is None and model_config.get('architecture') == 'videomae_group_gated_experts':
        out_name = "videomae_model_12_improved_heads.pt"

    improved_checkpoint_path = save_checkpoint(
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
    logger.info("=" * 60)
    logger.info("IMPROVEMENT COMPLETED!")
    logger.info(f"Checkpoint cải tiến (riêng từ checkpoint 12) đã lưu: {improved_checkpoint_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
