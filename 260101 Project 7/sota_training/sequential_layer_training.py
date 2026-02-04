"""
Sequential Layer Training - train từng layer/expert riêng (Model 10, 11, 12, Alpha).
Re-export FilteredVideoDataset for improve_weak_heads.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List
import logging

from .training import train_one_epoch, evaluate
from .dataset import FilteredVideoDataset

logger = logging.getLogger(__name__)


def build_contiguous_label_subsets(
    num_classes: int, num_experts: int = 8
) -> List[List[int]]:
    """Split labels [0..num_classes-1] into contiguous subsets."""
    if num_experts <= 0:
        raise ValueError("num_experts must be > 0")
    labels_per = num_classes // num_experts
    remainder = num_classes % num_experts
    subsets: List[List[int]] = []
    start = 0
    for i in range(num_experts):
        size = labels_per + (1 if i < remainder else 0)
        end = start + size
        subsets.append(list(range(start, end)))
        start = end
    return subsets


def freeze_all_backbone_layers(model: nn.Module) -> None:
    """Freeze backbone (videomae, vit, or swin)."""
    if hasattr(model, 'videomae'):
        for p in model.videomae.parameters():
            p.requires_grad = False
        logger.info("Frozen all backbone layers (encoder + embeddings)")
    elif hasattr(model, 'vit'):
        for p in model.vit.parameters():
            p.requires_grad = False
    elif hasattr(model, 'swin'):
        for p in model.swin.parameters():
            p.requires_grad = False


def freeze_all_heads(model: nn.Module) -> None:
    """Freeze global head + all expert heads (if present)."""
    if hasattr(model, "global_head"):
        for p in model.global_head.parameters():
            p.requires_grad = False
    if hasattr(model, "expert_heads"):
        for head in model.expert_heads:
            for p in head.parameters():
                p.requires_grad = False
    if hasattr(model, "group_head"):
        for p in model.group_head.parameters():
            p.requires_grad = False


def unfreeze_global_head_only(model: nn.Module) -> None:
    """Unfreeze only global head; freeze all experts."""
    freeze_all_heads(model)
    if not hasattr(model, "global_head"):
        raise ValueError("Model does not have global_head")
    for p in model.global_head.parameters():
        p.requires_grad = True


def unfreeze_single_expert_only(model: nn.Module, expert_idx: int) -> None:
    """Unfreeze only expert head at index; freeze global + other experts."""
    freeze_all_heads(model)
    if not hasattr(model, "expert_heads"):
        raise ValueError("Model does not have expert_heads")
    if expert_idx < 0 or expert_idx >= len(model.expert_heads):
        raise ValueError(f"expert_idx {expert_idx} out of range")
    for p in model.expert_heads[expert_idx].parameters():
        p.requires_grad = True


def unfreeze_all_heads(model: nn.Module) -> None:
    """Unfreeze global head + all experts."""
    if hasattr(model, "global_head"):
        for p in model.global_head.parameters():
            p.requires_grad = True
    if hasattr(model, "expert_heads"):
        for head in model.expert_heads:
            for p in head.parameters():
                p.requires_grad = True


def freeze_group_head(model: nn.Module) -> None:
    """Freeze group head (if present)."""
    if hasattr(model, "group_head"):
        for p in model.group_head.parameters():
            p.requires_grad = False


def unfreeze_group_head_only(model: nn.Module) -> None:
    """Unfreeze only group head."""
    freeze_all_heads(model)
    if not hasattr(model, "group_head"):
        raise ValueError("Model does not have group_head")
    for p in model.group_head.parameters():
        p.requires_grad = True


def unfreeze_group_head_and_all_experts(model: nn.Module) -> None:
    """Unfreeze group head + all expert heads."""
    freeze_all_backbone_layers(model)
    if hasattr(model, "group_head"):
        for p in model.group_head.parameters():
            p.requires_grad = True
    if hasattr(model, "expert_heads"):
        for head in model.expert_heads:
            for p in head.parameters():
                p.requires_grad = True


def train_expert_head_phase(
    model: nn.Module,
    train_dataset,
    val_dataset,
    device: torch.device,
    config: Dict[str, Any],
    expert_idx: int,
    label_subset: List[int],
    phase_name: str,
) -> Tuple[float, Dict[str, list]]:
    """Train a single expert head on its label subset."""
    filtered_train_dataset = FilteredVideoDataset(train_dataset, label_subset)
    filtered_val_dataset = FilteredVideoDataset(val_dataset, label_subset)

    logger.info("=" * 60)
    logger.info("%s: Expert %s, Labels %s", phase_name, expert_idx, label_subset)
    logger.info("Filtered train samples: %s", len(filtered_train_dataset))
    logger.info("Filtered val samples: %s", len(filtered_val_dataset))
    logger.info("=" * 60)

    if len(filtered_train_dataset) == 0 or len(filtered_val_dataset) == 0:
        logger.warning("No data for this subset; skipping expert phase.")
        return 0.0, {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    num_workers = config.get('num_workers', 0)
    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        filtered_train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        filtered_val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    unfreeze_single_expert_only(model, expert_idx)
    if hasattr(model, "experts_enabled"):
        model.experts_enabled = True
    if hasattr(model, "active_expert_idx"):
        model.active_expert_idx = expert_idx
    if hasattr(model, "hard_routing_enabled"):
        model.hard_routing_enabled = False

    expert_params = [
        p for p in model.expert_heads[expert_idx].parameters() if p.requires_grad
    ]
    if not expert_params:
        logger.warning("No trainable expert params; skipping.")
        return 0.0, {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    expert_lr = config.get('expert_lr', config.get('head_lr_unfreeze', 2e-4))
    lr_mult = config.get('expert_lr_multiplier', {}) or {}
    if expert_idx in lr_mult:
        expert_lr = expert_lr * float(lr_mult[expert_idx])
    optimizer = torch.optim.AdamW(
        [{"params": expert_params, "lr": expert_lr}],
        weight_decay=config.get(
            'weight_decay_unfreeze', config.get('weight_decay', 0.05)
        ),
    )

    from .adaptive_lr_scheduler import get_adaptive_lr_scheduler
    max_epochs = config.get('layer_phase_max_epochs', 30)
    max_epochs_overrides = config.get('expert_max_epochs_overrides', {}) or {}
    if expert_idx in max_epochs_overrides:
        max_epochs = int(max_epochs_overrides[expert_idx])

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

    patience = config.get('layer_phase_patience', 10)
    patience_overrides = config.get('expert_patience_overrides', {}) or {}
    if expert_idx in patience_overrides:
        patience = int(patience_overrides[expert_idx])

    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0.0
    patience_counter = 0
    best_expert_state = None  # restore best weights after early stop

    for epoch in range(max_epochs):
        logger.info("\n%s - Epoch %s/%s", phase_name, epoch + 1, max_epochs)

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

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        logger.info("  Train: Loss=%.4f, Acc=%.4f", train_loss, train_acc)
        logger.info("  Val:   Loss=%.4f, Acc=%.4f", val_loss, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if hasattr(model, 'expert_heads') and expert_idx < len(model.expert_heads):
                best_expert_state = {
                    k: v.cpu().clone()
                    for k, v in model.expert_heads[expert_idx].state_dict().items()
                }
            logger.info("  ✓ New best val acc: %.4f", best_val_acc)
        else:
            patience_counter += 1
            logger.info("  No improvement (%s/%s)", patience_counter, patience)

        if patience_counter >= patience:
            logger.info(
                "\n%s - Early stopping: No improvement for %s epochs",
                phase_name, patience,
            )
            logger.info("  Best val acc: %.4f", best_val_acc)
            break

    if best_expert_state is not None and hasattr(model, 'expert_heads'):
        model.expert_heads[expert_idx].load_state_dict(best_expert_state, strict=True)
        logger.info("  Restored best expert weights (early stop may have kept last epoch)")
    freeze_all_heads(model)
    if hasattr(model, "active_expert_idx"):
        model.active_expert_idx = None
    return best_val_acc, history


def unfreeze_single_layer(model: nn.Module, layer_idx: int) -> None:
    """Unfreeze a single backbone layer (stub for Model 10)."""
    if hasattr(model, 'videomae') and hasattr(model.videomae, 'encoder'):
        layers = model.videomae.encoder.layer
        if 0 <= layer_idx < len(layers):
            for p in layers[layer_idx].parameters():
                p.requires_grad = True


def freeze_single_layer(model: nn.Module, layer_idx: int) -> None:
    """Freeze a single backbone layer (stub for Model 10)."""
    if hasattr(model, 'videomae') and hasattr(model.videomae, 'encoder'):
        layers = model.videomae.encoder.layer
        if 0 <= layer_idx < len(layers):
            for p in layers[layer_idx].parameters():
                p.requires_grad = False


def sequential_layer_training_loop(
    model: nn.Module,
    train_dataset,
    val_dataset,
    device: torch.device,
    config: Dict[str, Any],
    classes: list,
) -> Tuple[Dict[str, list], float]:
    """
    Sequential layer training loop (Model 10).
    Returns (history_dict, best_val_acc).
    """
    freeze_all_backbone_layers(model)
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'classifier' in name or 'head' in name:
            head_params.append(param)
    if not head_params:
        logger.warning("No head params found; returning empty history.")
        return (
            {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []},
            0.0,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 24),
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 24),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": config.get('final_head_lr', 1e-4)}],
        weight_decay=config.get('weight_decay_unfreeze', 0.05),
    )
    from .adaptive_lr_scheduler import get_adaptive_lr_scheduler
    max_epochs = config.get('final_head_epochs', 50)
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

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0.0
    patience = config.get('early_stop_patience', 20)
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            grad_accum_steps=config.get('grad_accum_steps', 1),
            use_mixup=config.get('use_mixup', True),
            use_cutmix=config.get('use_cutmix', False),
            mixup_alpha=config.get('mixup_alpha', 0.4),
            cutmix_alpha=config.get('cutmix_alpha', 1.0),
            label_smoothing=config.get('label_smoothing', 0.1),
            use_focal_loss=config.get('use_focal_loss', False),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
            ema_model=None,
            use_synthetic_data=False,
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
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    return history, best_val_acc


def final_head_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
) -> Tuple[float, Dict[str, list]]:
    """
    Final head training phase (freeze backbone, train head).
    Returns (best_val_acc, history).
    """
    freeze_all_backbone_layers(model)
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'classifier' in name or 'head' in name:
            head_params.append(param)
    if not head_params:
        return 0.0, {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": config.get('final_head_lr', 1e-4)}],
        weight_decay=config.get('weight_decay_unfreeze', 0.05),
    )
    from .adaptive_lr_scheduler import get_adaptive_lr_scheduler
    max_epochs = config.get('final_head_epochs', 50)
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
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            grad_accum_steps=config.get('grad_accum_steps', 1),
            use_mixup=config.get('use_mixup', True),
            use_cutmix=config.get('use_cutmix', False),
            mixup_alpha=config.get('mixup_alpha', 0.4),
            cutmix_alpha=config.get('cutmix_alpha', 1.0),
            label_smoothing=config.get('label_smoothing', 0.1),
            use_focal_loss=config.get('use_focal_loss', False),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
            ema_model=None,
            use_synthetic_data=False,
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
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    return best_val_acc, history
