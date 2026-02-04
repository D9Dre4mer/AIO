"""
Multi-stage training for VideoMAE Global + Residual Experts (Model 11).

This module keeps training logic inside `sota_training/` for consistency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .training import train_one_epoch, evaluate
from .adaptive_lr_scheduler import get_adaptive_lr_scheduler
from .utils import save_checkpoint, plot_training_history
from collections import defaultdict
import numpy as np
from .sequential_layer_training import (
    build_contiguous_label_subsets,
    freeze_all_backbone_layers,
    freeze_all_heads,
    freeze_group_head,
    unfreeze_global_head_only,
    unfreeze_all_heads,
    unfreeze_group_head_only,
    unfreeze_group_head_and_all_experts,
    train_expert_head_phase,
)

logger = logging.getLogger('sota_training')


def _evaluate_and_print_results(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    classes: List[str],
    label_subsets: Optional[List[List[int]]] = None,
) -> None:
    """
    Evaluate model on validation set and print detailed results.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to evaluate on
        classes: List of class names
        label_subsets: Optional label subsets for group analysis
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    logger.info("=" * 60)
    logger.info("Final Evaluation on Validation Set")
    logger.info("=" * 60)
    
    with torch.no_grad():
        progress = tqdm(val_loader, desc="Final Eval", leave=False)
        for videos, labels in progress:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            
            with torch.amp.autocast(
                device_type='cuda',
                enabled=(device.type == 'cuda')
            ):
                logits = model(videos)
            
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
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
            else:
                group_acc[group_idx] = 0.0
    
    # Print results
    logger.info("")
    logger.info("📊 FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Overall Validation Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    logger.info(f"Total Samples: {len(all_labels)}")
    logger.info(f"Correct Predictions: {(all_preds == all_labels).sum()}")
    logger.info("")
    
    # Per-class accuracy (top 10 worst and best)
    sorted_classes = sorted(
        class_acc.items(),
        key=lambda x: x[1]
    )
    
    logger.info("🔴 Top 10 Worst Performing Classes:")
    for class_idx, acc in sorted_classes[:10]:
        class_name = classes[class_idx]
        total = class_total[class_idx]
        correct = class_correct[class_idx]
        logger.info(
            f"  {class_name:30s} | Acc: {acc:.4f} ({acc*100:.2f}%) | "
            f"Correct: {correct}/{total}"
        )
    
    logger.info("")
    logger.info("🟢 Top 10 Best Performing Classes:")
    for class_idx, acc in sorted_classes[-10:][::-1]:
        class_name = classes[class_idx]
        total = class_total[class_idx]
        correct = class_correct[class_idx]
        logger.info(
            f"  {class_name:30s} | Acc: {acc:.4f} ({acc*100:.2f}%) | "
            f"Correct: {correct}/{total}"
        )
    
    # Group accuracy (if available)
    if label_subsets is not None and group_acc:
        logger.info("")
        logger.info("📦 Group-Level Accuracy:")
        for group_idx, acc in sorted(group_acc.items(), key=lambda x: x[1]):
            subset = label_subsets[group_idx]
            logger.info(
                f"  Group {group_idx + 1} (Labels {subset[0]}-{subset[-1]}): "
                f"{acc:.4f} ({acc*100:.2f}%)"
            )
    
    logger.info("=" * 60)


def _make_full_loaders(
    train_dataset,
    val_dataset,
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    num_workers = config.get('num_workers', 0)
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def _train_global_head_stage(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    combined_history: Dict[str, list],
    checkpoint_path: Path,
    history_plot_path: Path,
    classes: list,
    start_epoch_idx: int,
    best_overall: float,
) -> Tuple[float, float, int]:
    """
    Stage A: train only global head on full dataset.

    Returns:
      - best val acc within stage
      - best_overall updated
      - updated global epoch index
    """
    freeze_all_backbone_layers(model)
    unfreeze_global_head_only(model)
    # Stage A: disable all experts so training matches "global head only"
    if hasattr(model, "experts_enabled"):
        model.experts_enabled = False
        model.active_expert_idx = None

    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.global_head.parameters()),
                "lr": config['head_lr'],
            }
        ],
        weight_decay=config.get('weight_decay', 0.05),
    )
    scheduler = get_adaptive_lr_scheduler(
        optimizer,
        num_epochs=config['freeze_backbone_epochs'],
        warmup_epochs=config.get('warmup_epochs', 5),
        min_lr_ratio=config.get('min_lr_ratio', 0.01),
        plateau_patience=config.get('lr_plateau_patience', 8),
        min_delta=config.get('lr_min_delta', 0.005),
        mode='max',
        verbose=True,
        cooldown=config.get('lr_cooldown', 3),
        threshold_mode=config.get('lr_threshold_mode', 'rel'),
        use_val_loss=config.get('use_val_loss_for_lr', False),
    )
    scaler = torch.amp.GradScaler()

    max_epochs = config['freeze_backbone_epochs']
    best_stage = 0.0

    logger.info("=" * 60)
    logger.info("STAGE A: Train GlobalHead (full 51 labels)")
    logger.info(f"  Epochs: {max_epochs}")
    logger.info("=" * 60)

    for epoch in range(max_epochs):
        logger.info(f"\nStage A - Epoch {epoch + 1}/{max_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            grad_accum_steps=config['grad_accum_steps'],
            use_mixup=config.get('use_mixup', True),
            use_cutmix=config.get('use_cutmix', False),
            mixup_alpha=config.get('mixup_alpha', 0.4),
            cutmix_alpha=config.get('cutmix_alpha', 1.0),
            label_smoothing=config.get('label_smoothing', 0.1),
            use_focal_loss=config.get('use_focal_loss', False),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
            ema_model=None,
            use_synthetic_data=config.get('use_synthetic_data', False),
            synthetic_method=config.get('synthetic_method', 'frame_mixup'),
            synthetic_ratio=config.get('synthetic_ratio', 0.3),
            teacher=None,
            use_distillation=False,
        )

        val_loss, val_acc = evaluate(
            model, val_loader, device,
            label_smoothing=0.0,
            use_focal_loss=config.get('use_focal_loss', False),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
        )

        scheduler.step(metrics=val_acc)

        logger.info(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        logger.info(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        combined_history['train_loss'].append(train_loss)
        combined_history['train_acc'].append(train_acc)
        combined_history['val_loss'].append(val_loss)
        combined_history['val_acc'].append(val_acc)
        plot_training_history(
            combined_history,
            history_plot_path,
            config['model_id'],
        )

        epoch_idx = start_epoch_idx + epoch + 1

        if val_acc > best_stage:
            best_stage = val_acc
            logger.info(f"  ✓ New best Stage A val acc: {best_stage:.4f}")

        if val_acc > best_overall:
            best_overall = val_acc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_idx,
                history=combined_history,
                best_val_acc=best_overall,
                train_acc=train_acc,
                classes=classes,
                config=config,
                checkpoint_path=checkpoint_path,
                ema_model=None,
            )
            logger.info(f"  ✓ Best model saved (val_acc: {best_overall:.4f})")

    freeze_all_heads(model)
    if hasattr(model, "experts_enabled"):
        model.experts_enabled = True
    return best_stage, best_overall, start_epoch_idx + max_epochs


def _train_joint_finetune_stage(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    combined_history: Dict[str, list],
    checkpoint_path: Path,
    history_plot_path: Path,
    classes: list,
    start_epoch_idx: int,
    best_overall: float,
) -> Tuple[float, float, int]:
    """
    Stage C: joint finetune global + all experts on full dataset.
    """
    freeze_all_backbone_layers(model)
    unfreeze_all_heads(model)
    if hasattr(model, "experts_enabled"):
        model.experts_enabled = True
        model.active_expert_idx = None

    joint_params: List[torch.nn.Parameter] = []
    joint_params.extend(list(model.global_head.parameters()))
    for head in model.expert_heads:
        joint_params.extend(list(head.parameters()))

    optimizer = torch.optim.AdamW(
        [
            {
                "params": joint_params,
                "lr": config['final_head_lr'],
            }
        ],
        weight_decay=config.get(
            'weight_decay_unfreeze',
            config.get('weight_decay', 0.05),
        ),
    )
    scheduler = get_adaptive_lr_scheduler(
        optimizer,
        num_epochs=config['final_head_epochs'],
        warmup_epochs=config.get('warmup_epochs', 5),
        min_lr_ratio=config.get('min_lr_ratio', 0.01),
        plateau_patience=config.get('lr_plateau_patience', 8),
        min_delta=config.get('lr_min_delta', 0.005),
        mode='max',
        verbose=True,
        cooldown=config.get('lr_cooldown', 3),
        threshold_mode=config.get('lr_threshold_mode', 'rel'),
        use_val_loss=config.get('use_val_loss_for_lr', False),
    )
    scaler = torch.amp.GradScaler()

    max_epochs = config.get('final_head_epochs', 50)
    patience = config.get('early_stop_patience', 20)
    best_stage = 0.0
    patience_counter = 0

    logger.info("=" * 60)
    logger.info("STAGE C: Joint finetune Global + Experts (full 51 labels)")
    logger.info(f"  Max epochs: {max_epochs}, Patience: {patience}")
    logger.info("=" * 60)

    for epoch in range(max_epochs):
        logger.info(f"\nStage C - Epoch {epoch + 1}/{max_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            grad_accum_steps=config['grad_accum_steps'],
            use_mixup=config.get('use_mixup', True),
            use_cutmix=config.get('use_cutmix', False),
            mixup_alpha=config.get('mixup_alpha', 0.4),
            cutmix_alpha=config.get('cutmix_alpha', 1.0),
            label_smoothing=config.get('label_smoothing', 0.1),
            use_focal_loss=config.get('use_focal_loss', False),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
            ema_model=None,
            use_synthetic_data=config.get('use_synthetic_data', False),
            synthetic_method=config.get('synthetic_method', 'frame_mixup'),
            synthetic_ratio=config.get('synthetic_ratio', 0.3),
            teacher=None,
            use_distillation=False,
        )

        val_loss, val_acc = evaluate(
            model, val_loader, device,
            label_smoothing=0.0,
            use_focal_loss=config.get('use_focal_loss', False),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
        )

        scheduler.step(metrics=val_acc)

        logger.info(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        logger.info(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        combined_history['train_loss'].append(train_loss)
        combined_history['train_acc'].append(train_acc)
        combined_history['val_loss'].append(val_loss)
        combined_history['val_acc'].append(val_acc)
        plot_training_history(
            combined_history,
            history_plot_path,
            config['model_id'],
        )

        epoch_idx = start_epoch_idx + epoch + 1

        improved = False
        if val_acc > best_stage:
            best_stage = val_acc
            patience_counter = 0
            improved = True
            logger.info(f"  ✓ New best Stage C val acc: {best_stage:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{patience})")

        if val_acc > best_overall:
            best_overall = val_acc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_idx,
                history=combined_history,
                best_val_acc=best_overall,
                train_acc=train_acc,
                classes=classes,
                config=config,
                checkpoint_path=checkpoint_path,
                ema_model=None,
            )
            logger.info(f"  ✓ Best model saved (val_acc: {best_overall:.4f})")

        if not improved and patience_counter >= patience:
            logger.info(
                "\nStage C - Early stopping: "
                f"No improvement for {patience} epochs"
            )
            break

    return best_stage, best_overall, start_epoch_idx + max_epochs


def train_global_residual_experts(
    model: torch.nn.Module,
    train_dataset,
    val_dataset,
    device: torch.device,
    config: Dict[str, Any],
    checkpoint_path: Path,
    history_plot_path: Path,
    classes: list,
    label_subsets: Optional[List[List[int]]] = None,
    skip_stage_a: bool = False,
    resume_checkpoint: Optional[Dict[str, Any]] = None,
) -> Dict[str, list]:
    """
    Full pipeline for Model 11.

    Returns combined_history (train_loss/train_acc/val_loss/val_acc).
    """
    if label_subsets is None:
        label_subsets = build_contiguous_label_subsets(
            len(classes),
            num_experts=config.get('num_experts', 8),
        )

    train_loader, val_loader = _make_full_loaders(
        train_dataset,
        val_dataset,
        config,
    )

    combined_history: Dict[str, list] = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    best_overall = 0.0
    epoch_idx = 0

    # Resume (optional): restore history/best/epoch counter.
    if resume_checkpoint is not None:
        hist = resume_checkpoint.get('history', {}) or {}
        combined_history = {
            'train_loss': list(hist.get('train_loss', [])),
            'train_acc': list(hist.get('train_acc', [])),
            'val_loss': list(hist.get('val_loss', [])),
            'val_acc': list(hist.get('val_acc', [])),
        }
        best_overall = float(resume_checkpoint.get('val_acc', 0.0) or 0.0)
        epoch_idx = int(
            resume_checkpoint.get('epoch', len(combined_history['train_loss']))
        )
        logger.info(
            "Resumed training state: "
            f"epoch={epoch_idx}, best_val_acc={best_overall:.4f}, "
            f"history_len={len(combined_history['train_loss'])}"
        )

    # Stage A
    best_a = 0.0
    if not skip_stage_a:
        best_a, best_overall, epoch_idx = _train_global_head_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            combined_history=combined_history,
            checkpoint_path=checkpoint_path,
            history_plot_path=history_plot_path,
            classes=classes,
            start_epoch_idx=epoch_idx,
            best_overall=best_overall,
        )
    else:
        logger.info("=" * 60)
        logger.info("STAGE A: Skipped (using resumed weights/state)")
        logger.info("=" * 60)

    # Stage B (subset metrics)
    logger.info("=" * 60)
    logger.info("STAGE B: Train Residual Experts (subset labels)")
    logger.info("=" * 60)

    freeze_all_backbone_layers(model)
    freeze_all_heads(model)

    best_b = 0.0
    for i, subset in enumerate(label_subsets):
        phase_name = f"Stage B - Expert {i + 1}/{len(label_subsets)}"
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
        best_b = max(best_b, best_i)
        # Keep plot moving (stage B uses subset val)
        for k in combined_history:
            combined_history[k].extend(hist_i[k])
        plot_training_history(
            combined_history,
            history_plot_path,
            config['model_id'],
        )

    # Stage C
    best_c, best_overall, epoch_idx = _train_joint_finetune_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        combined_history=combined_history,
        checkpoint_path=checkpoint_path,
        history_plot_path=history_plot_path,
        classes=classes,
        start_epoch_idx=epoch_idx,
        best_overall=best_overall,
    )

    logger.info("=" * 60)
    logger.info("Training completed.")
    logger.info(f"Stage A best val acc: {best_a:.4f}")
    logger.info(f"Stage B best subset val acc: {best_b:.4f}")
    logger.info(f"Stage C best val acc: {best_c:.4f}")
    logger.info(f"Overall best val acc: {best_overall:.4f}")
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    logger.info(f"Saved plot: {history_plot_path}")
    logger.info("=" * 60)

    return combined_history


def _train_group_head_stage(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    combined_history: Dict[str, list],
    checkpoint_path: Path,
    history_plot_path: Path,
    classes: list,
    start_epoch_idx: int,
    best_overall: float,
    use_overall_class_acc_for_best: bool = False,
) -> Tuple[float, float, int]:
    """
    Stage G: train only group head to predict the correct group (supervised).

    Target is group index (0..num_groups-1) derived from class label using
    label_subsets.

    If use_overall_class_acc_for_best is True (e.g. when called from improve
    script), best checkpoint and early-stop use overall class accuracy
    (end-to-end) instead of group accuracy, while still training with group CE.
    """
    freeze_all_backbone_layers(model)
    freeze_all_heads(model)
    freeze_group_head(model)
    unfreeze_group_head_only(model)

    if hasattr(model, "active_expert_idx"):
        model.active_expert_idx = None

    max_epochs = int(config.get('group_head_epochs', 10))
    lr = float(config.get('group_head_lr', 1e-3))
    target_acc = float(config.get('group_target_acc', 0.99))

    optimizer = torch.optim.AdamW(
        [{"params": list(model.group_head.parameters()), "lr": lr}],
        weight_decay=config.get('group_weight_decay', config.get('weight_decay', 0.05)),
    )
    scheduler = get_adaptive_lr_scheduler(
        optimizer,
        num_epochs=max_epochs,
        warmup_epochs=config.get('warmup_epochs', 5),
        min_lr_ratio=config.get('min_lr_ratio', 0.01),
        plateau_patience=config.get('lr_plateau_patience', 8),
        min_delta=config.get('lr_min_delta', 0.005),
        mode='max',
        verbose=True,
        cooldown=config.get('lr_cooldown', 3),
        threshold_mode=config.get('lr_threshold_mode', 'rel'),
        use_val_loss=config.get('use_val_loss_for_lr', False),
    )
    scaler = torch.amp.GradScaler()

    patience = int(config.get('group_head_patience', 10))
    patience_counter = 0
    best_stage = 0.0
    grad_accum_steps = int(config.get('grad_accum_steps', 1))

    # Build label->group mapping
    num_groups = (
        len(config.get('label_subsets', [])) or
        getattr(model, 'num_groups', 0)
    )
    if num_groups <= 0:
        raise ValueError("Cannot determine num_groups for Stage G")
    label_to_group = torch.empty(len(classes), dtype=torch.long)
    label_subsets = config.get('label_subsets', None)
    if label_subsets is None:
        label_subsets = getattr(model, 'label_subsets', None)
    if label_subsets is None:
        raise ValueError("label_subsets missing for Stage G")
    for gi, subset in enumerate(label_subsets):
        for lbl in subset:
            label_to_group[int(lbl)] = gi

    logger.info("=" * 60)
    logger.info("STAGE G: Train GroupHead (supervised group accuracy)")
    if use_overall_class_acc_for_best:
        logger.info("  Best/early-stop metric: overall class accuracy (end-to-end)")
    logger.info(
        f"  Max epochs: {max_epochs}, Patience: {patience}, "
        f"Target acc: {target_acc:.4f}"
    )
    logger.info("=" * 60)

    epochs_ran = 0
    for epoch in range(max_epochs):
        logger.info(f"\nStage G - Epoch {epoch + 1}/{max_epochs}")

        model.train()
        optimizer.zero_grad()
        total = 0
        correct = 0
        total_loss = 0.0
        # Use label smoothing to reduce overfitting
        group_label_smoothing = config.get('group_label_smoothing', 0.1)
        if group_label_smoothing > 0:
            from .losses import LabelSmoothingCrossEntropy
            criterion = LabelSmoothingCrossEntropy(
                smoothing=group_label_smoothing
            )
        else:
            criterion = torch.nn.CrossEntropyLoss()

        label_to_group_dev = label_to_group.to(device)
        progress = tqdm(train_loader, desc="Stage G Train", leave=False)
        for batch_idx, (videos, labels) in enumerate(progress):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            group_labels = label_to_group_dev[labels]

            with torch.amp.autocast(
                device_type='cuda',
                enabled=(device.type == 'cuda')
            ):
                pooled = model._forward_backbone(videos)
                group_logits = model.group_head(pooled)
                loss = criterion(group_logits, group_labels)

            preds = group_logits.argmax(dim=1)
            correct += (preds == group_labels).sum().item()
            total += labels.size(0)
            loss_value = loss.item()
            total_loss += loss_value * labels.size(0)
            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                acc=f"{correct / max(total, 1):.4f}",
            )

            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()

            should_step = (
                ((batch_idx + 1) % grad_accum_steps == 0) or
                (batch_idx + 1 == len(train_loader))
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.group_head.parameters(),
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            total_loss = 0.0
            progress = tqdm(val_loader, desc="Stage G Val", leave=False)
            for videos, labels in progress:
                videos = videos.to(device, non_blocking=True)
                labels = labels.to(device, dtype=torch.long, non_blocking=True)
                group_labels = label_to_group_dev[labels]
                with torch.amp.autocast(
                    device_type='cuda',
                    enabled=(device.type == 'cuda')
                ):
                    pooled = model._forward_backbone(videos)
                    group_logits = model.group_head(pooled)
                    loss = criterion(group_logits, group_labels)
                preds = group_logits.argmax(dim=1)
                correct += (preds == group_labels).sum().item()
                total += labels.size(0)
                loss_value = loss.item()
                total_loss += loss_value * labels.size(0)
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    acc=f"{correct / max(total, 1):.4f}",
                )
            val_loss = total_loss / max(total, 1)
            val_acc = correct / max(total, 1)

        metric_for_best = val_acc
        if use_overall_class_acc_for_best:
            _, overall_class_acc = evaluate(
                model, val_loader, device,
                label_smoothing=0.0,
                use_focal_loss=config.get('use_focal_loss', False),
                focal_alpha=config.get('focal_alpha', 0.25),
                focal_gamma=config.get('focal_gamma', 2.0),
            )
            metric_for_best = overall_class_acc

        scheduler.step(metrics=metric_for_best)

        logger.info(
            f"  Train (group): Loss={train_loss:.4f}, Acc={train_acc:.4f}"
        )
        logger.info(
            f"  Val   (group): Loss={val_loss:.4f}, Acc={val_acc:.4f}"
        )
        if use_overall_class_acc_for_best:
            logger.info(f"  Val   (overall class): Acc={metric_for_best:.4f} (metric for best/early-stop)")

        combined_history['train_loss'].append(train_loss)
        combined_history['train_acc'].append(train_acc)
        combined_history['val_loss'].append(val_loss)
        combined_history['val_acc'].append(val_acc)
        plot_training_history(
            combined_history,
            history_plot_path,
            config['model_id'],
        )

        epoch_idx = start_epoch_idx + epoch + 1
        epochs_ran = epoch + 1

        improved = False
        if metric_for_best > best_stage:
            best_stage = metric_for_best
            improved = True
            patience_counter = 0
            logger.info(f"  ✓ New best Stage G val acc: {best_stage:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{patience})")

        if metric_for_best > best_overall:
            best_overall = metric_for_best
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_idx,
                history=combined_history,
                best_val_acc=best_overall,
                train_acc=train_acc,
                classes=classes,
                config=config,
                checkpoint_path=checkpoint_path,
                ema_model=None,
            )
            logger.info(f"  ✓ Best model saved (val_acc: {best_overall:.4f})")

        if not use_overall_class_acc_for_best and val_acc >= target_acc:
            logger.info(
                "  ✓ Target group acc reached: "
                f"{val_acc:.4f} >= {target_acc:.4f}"
            )
            break

        if not improved and patience_counter >= patience:
            logger.info(
                "\nStage G - Early stopping: "
                f"No improvement for {patience} epochs"
            )
            break

    freeze_group_head(model)
    return best_stage, best_overall, start_epoch_idx + epochs_ran


def _train_group_gated_joint_finetune_stage(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    combined_history: Dict[str, list],
    checkpoint_path: Path,
    history_plot_path: Path,
    classes: list,
    start_epoch_idx: int,
    best_overall: float,
) -> Tuple[float, float, int]:
    """
    Stage C for Model 12: joint finetune group_head + all expert_heads on full
    dataset with class-level CrossEntropy. Optimizes overall accuracy and
    balances group head and experts.
    """
    freeze_all_backbone_layers(model)
    unfreeze_group_head_and_all_experts(model)
    if hasattr(model, "hard_routing_enabled"):
        model.hard_routing_enabled = False
    if hasattr(model, "active_expert_idx"):
        model.active_expert_idx = None

    joint_params: List[torch.nn.Parameter] = []
    joint_params.extend(list(model.group_head.parameters()))
    for head in model.expert_heads:
        joint_params.extend(list(head.parameters()))

    lr = float(config.get('stage_c_lr', config.get('final_head_lr', 5e-4)))
    optimizer = torch.optim.AdamW(
        [{"params": joint_params, "lr": lr}],
        weight_decay=config.get(
            'weight_decay_unfreeze',
            config.get('weight_decay', 0.05),
        ),
    )
    max_epochs = int(config.get('stage_c_epochs', config.get('final_head_epochs', 30)))
    scheduler = get_adaptive_lr_scheduler(
        optimizer,
        num_epochs=max_epochs,
        warmup_epochs=config.get('warmup_epochs', 5),
        min_lr_ratio=config.get('min_lr_ratio', 0.01),
        plateau_patience=config.get('lr_plateau_patience', 8),
        min_delta=config.get('lr_min_delta', 0.005),
        mode='max',
        verbose=True,
        cooldown=config.get('lr_cooldown', 3),
        threshold_mode=config.get('lr_threshold_mode', 'rel'),
        use_val_loss=config.get('use_val_loss_for_lr', False),
    )
    scaler = torch.amp.GradScaler()
    patience = int(config.get('stage_c_patience', config.get('early_stop_patience', 15)))
    best_stage = 0.0
    patience_counter = 0
    epochs_ran = 0

    logger.info("=" * 60)
    logger.info("STAGE C: Joint finetune Group + Experts (full class labels)")
    logger.info(f"  Max epochs: {max_epochs}, Patience: {patience}, LR: {lr}")
    logger.info("=" * 60)

    for epoch in range(max_epochs):
        epochs_ran = epoch + 1
        logger.info(f"\nStage C - Epoch {epoch + 1}/{max_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            grad_accum_steps=config['grad_accum_steps'],
            use_mixup=config.get('use_mixup', True),
            use_cutmix=config.get('use_cutmix', False),
            mixup_alpha=config.get('mixup_alpha', 0.4),
            cutmix_alpha=config.get('cutmix_alpha', 1.0),
            label_smoothing=config.get('label_smoothing', 0.1),
            use_focal_loss=config.get('use_focal_loss', False),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
            ema_model=None,
            use_synthetic_data=config.get('use_synthetic_data', False),
            synthetic_method=config.get('synthetic_method', 'frame_mixup'),
            synthetic_ratio=config.get('synthetic_ratio', 0.3),
            teacher=None,
            use_distillation=False,
        )

        val_loss, val_acc = evaluate(
            model, val_loader, device,
            label_smoothing=0.0,
            use_focal_loss=config.get('use_focal_loss', False),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
        )

        scheduler.step(metrics=val_acc)

        logger.info(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        logger.info(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        combined_history['train_loss'].append(train_loss)
        combined_history['train_acc'].append(train_acc)
        combined_history['val_loss'].append(val_loss)
        combined_history['val_acc'].append(val_acc)
        plot_training_history(
            combined_history,
            history_plot_path,
            config['model_id'],
        )

        epoch_idx = start_epoch_idx + epoch + 1
        improved = False
        if val_acc > best_stage:
            best_stage = val_acc
            patience_counter = 0
            improved = True
            logger.info(f"  ✓ New best Stage C val acc: {best_stage:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{patience})")

        if val_acc > best_overall:
            best_overall = val_acc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_idx,
                history=combined_history,
                best_val_acc=best_overall,
                train_acc=train_acc,
                classes=classes,
                config=config,
                checkpoint_path=checkpoint_path,
                ema_model=None,
            )
            logger.info(f"  ✓ Best model saved (val_acc: {best_overall:.4f})")

        if not improved and patience_counter >= patience:
            logger.info(
                "\nStage C - Early stopping: "
                f"No improvement for {patience} epochs"
            )
            break

    if hasattr(model, "hard_routing_enabled"):
        model.hard_routing_enabled = True
    return best_stage, best_overall, start_epoch_idx + epochs_ran


def train_group_gated_experts(
    model: torch.nn.Module,
    train_dataset,
    val_dataset,
    device: torch.device,
    config: Dict[str, Any],
    checkpoint_path: Path,
    history_plot_path: Path,
    classes: list,
    label_subsets: Optional[List[List[int]]] = None,
    resume_checkpoint: Optional[Dict[str, Any]] = None,
) -> Dict[str, list]:
    """
    Full pipeline for Model 12 (Group-Gated Experts).

    Stages:
      - Stage B: train experts sequentially on subsets
      - Stage G: train group head (supervised group accuracy)
      - Stage C: joint finetune group_head + experts (class-level CE)
    """
    if label_subsets is None:
        label_subsets = build_contiguous_label_subsets(
            len(classes),
            num_experts=config.get('num_experts', 8),
        )

    train_loader, val_loader = _make_full_loaders(
        train_dataset,
        val_dataset,
        config
    )

    combined_history: Dict[str, list] = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    best_overall = 0.0
    epoch_idx = 0

    if resume_checkpoint is not None:
        hist = resume_checkpoint.get('history', {}) or {}
        combined_history = {
            'train_loss': list(hist.get('train_loss', [])),
            'train_acc': list(hist.get('train_acc', [])),
            'val_loss': list(hist.get('val_loss', [])),
            'val_acc': list(hist.get('val_acc', [])),
        }
        best_overall = float(resume_checkpoint.get('val_acc', 0.0) or 0.0)
        epoch_idx = int(
            resume_checkpoint.get(
                'epoch',
                len(combined_history['train_loss'])
            )
        )
        logger.info(
            "Resumed training state: "
            f"epoch={epoch_idx}, best_val_acc={best_overall:.4f}, "
            f"history_len={len(combined_history['train_loss'])}"
        )

    logger.info("=" * 60)
    logger.info("STAGE B: Train Group Experts (subset labels)")
    logger.info("=" * 60)

    freeze_all_backbone_layers(model)
    freeze_all_heads(model)
    freeze_group_head(model)
    if hasattr(model, "hard_routing_enabled"):
        model.hard_routing_enabled = False
    if hasattr(model, "active_expert_idx"):
        model.active_expert_idx = None

    best_b = 0.0
    for i, subset in enumerate(label_subsets):
        phase_name = f"Stage B - Expert {i + 1}/{len(label_subsets)}"
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
        best_b = max(best_b, best_i)
        for k in combined_history:
            combined_history[k].extend(hist_i[k])
        plot_training_history(
            combined_history,
            history_plot_path,
            config['model_id'],
        )

    # Stage G
    best_g, best_overall, epoch_idx = _train_group_head_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        combined_history=combined_history,
        checkpoint_path=checkpoint_path,
        history_plot_path=history_plot_path,
        classes=classes,
        start_epoch_idx=epoch_idx,
        best_overall=best_overall,
    )

    # Stage C: joint finetune group_head + experts (class-level CE)
    stage_c_epochs = int(config.get('stage_c_epochs', config.get('final_head_epochs', 30)))
    if stage_c_epochs > 0:
        best_c, best_overall, epoch_idx = _train_group_gated_joint_finetune_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            combined_history=combined_history,
            checkpoint_path=checkpoint_path,
            history_plot_path=history_plot_path,
            classes=classes,
            start_epoch_idx=epoch_idx,
            best_overall=best_overall,
        )
        logger.info("=" * 60)
        logger.info("Training completed.")
        logger.info(f"Stage B best subset val acc: {best_b:.4f}")
        logger.info(f"Stage G best group val acc: {best_g:.4f}")
        logger.info(f"Stage C best val acc: {best_c:.4f}")
        logger.info(f"Overall best val acc: {best_overall:.4f}")
    else:
        logger.info("=" * 60)
        logger.info("Training completed.")
        logger.info(f"Stage B best subset val acc: {best_b:.4f}")
        logger.info(f"Stage G best group val acc: {best_g:.4f}")
        logger.info(f"Overall best val acc: {best_overall:.4f}")
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    logger.info(f"Saved plot: {history_plot_path}")
    logger.info("=" * 60)
    
    # Final evaluation and detailed results
    _evaluate_and_print_results(
        model=model,
        val_loader=val_loader,
        device=device,
        classes=classes,
        label_subsets=label_subsets,
    )

    return combined_history


def train_alpha_experts(
    model: torch.nn.Module,
    train_dataset,
    val_dataset,
    device: torch.device,
    config: Dict[str, Any],
    checkpoint_path: Path,
    history_plot_path: Path,
    classes: list,
    label_subsets: Optional[List[List[int]]] = None,
    resume_checkpoint: Optional[Dict[str, Any]] = None,
) -> Dict[str, list]:
    """
    Full pipeline for Alpha (experts only, no group head).

    Only Stage B: train each expert on its label subset; no Stage G, no Stage C.
    Saves checkpoint after each expert if overall val acc improves (evaluate full model).
    """
    if label_subsets is None:
        label_subsets = build_contiguous_label_subsets(
            len(classes),
            num_experts=config.get('num_experts', 8),
        )

    train_loader, val_loader = _make_full_loaders(
        train_dataset,
        val_dataset,
        config,
    )

    combined_history: Dict[str, list] = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    best_overall = 0.0
    epoch_idx = 0

    if resume_checkpoint is not None:
        hist = resume_checkpoint.get('history', {}) or {}
        combined_history = {
            'train_loss': list(hist.get('train_loss', [])),
            'train_acc': list(hist.get('train_acc', [])),
            'val_loss': list(hist.get('val_loss', [])),
            'val_acc': list(hist.get('val_acc', [])),
        }
        best_overall = float(resume_checkpoint.get('val_acc', 0.0) or 0.0)
        epoch_idx = int(
            resume_checkpoint.get('epoch', len(combined_history['train_loss']))
        )
        logger.info(
            "Resumed training state: "
            f"epoch={epoch_idx}, best_val_acc={best_overall:.4f}, "
            f"history_len={len(combined_history['train_loss'])}"
        )

    logger.info("=" * 60)
    logger.info("Alpha: Train Experts Only (Stage B only, no group head)")
    logger.info("=" * 60)

    freeze_all_backbone_layers(model)
    freeze_all_heads(model)
    if hasattr(model, "group_head"):
        freeze_group_head(model)
    if hasattr(model, "active_expert_idx"):
        model.active_expert_idx = None
    if hasattr(model, "inference_single_best_expert"):
        model.inference_single_best_expert = config.get(
            "inference_single_best_expert", True
        )

    best_b = 0.0
    for i, subset in enumerate(label_subsets):
        phase_name = f"Alpha Stage B - Expert {i + 1}/{len(label_subsets)}"
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
        best_b = max(best_b, best_i)
        for k in combined_history:
            combined_history[k].extend(hist_i[k])
        plot_training_history(
            combined_history,
            history_plot_path,
            config['model_id'],
        )

        # After each expert: evaluate full model (all experts) and save if best
        if hasattr(model, "active_expert_idx"):
            model.active_expert_idx = None
        val_loss, val_acc = evaluate(model, val_loader, device)
        if val_acc > best_overall:
            best_overall = val_acc
            epoch_idx = len(combined_history['train_loss'])
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch_idx,
                'history': combined_history,
                'val_acc': best_overall,
                'classes': classes,
                'config': {**config, 'label_subsets': label_subsets},
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"  ✓ Best overall val acc: {best_overall:.4f} -> saved")

    logger.info("=" * 60)
    logger.info("Alpha training completed.")
    logger.info(f"Stage B best subset val acc: {best_b:.4f}")
    logger.info(f"Overall best val acc: {best_overall:.4f}")
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    logger.info(f"Saved plot: {history_plot_path}")
    logger.info("=" * 60)

    _evaluate_and_print_results(
        model=model,
        val_loader=val_loader,
        device=device,
        classes=classes,
        label_subsets=label_subsets,
    )

    return combined_history
