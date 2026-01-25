"""
Sequential Layer Training for Model 10.
Train each layer separately on a subset of labels until peak performance, then freeze and move to next layer.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List
import logging

from .training import train_one_epoch, evaluate
from .dataset import FilteredVideoDataset

logger = logging.getLogger(__name__)


def build_contiguous_label_subsets(num_classes: int, num_experts: int = 8) -> List[List[int]]:
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


def freeze_all_heads(model: nn.Module) -> None:
    """Freeze global head + all expert heads (if present)."""
    if hasattr(model, "global_head"):
        for p in model.global_head.parameters():
            p.requires_grad = False
    if hasattr(model, "expert_heads"):
        for head in model.expert_heads:
            for p in head.parameters():
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
    if not hasattr(model, "global_head") or not hasattr(model, "expert_heads"):
        raise ValueError("Model does not have global_head/expert_heads")
    for p in model.global_head.parameters():
        p.requires_grad = True
    for head in model.expert_heads:
        for p in head.parameters():
            p.requires_grad = True


def freeze_group_head(model: nn.Module) -> None:
    """Freeze group head (if present)."""
    if hasattr(model, "group_head"):
        for p in model.group_head.parameters():
            p.requires_grad = False


def unfreeze_group_head_only(model: nn.Module) -> None:
    """Unfreeze only group head; freeze experts/global (if present)."""
    freeze_all_heads(model)
    freeze_group_head(model)
    if not hasattr(model, "group_head"):
        raise ValueError("Model does not have group_head")
    for p in model.group_head.parameters():
        p.requires_grad = True


def unfreeze_group_head_and_all_experts(model: nn.Module) -> None:
    """Unfreeze group head + all experts (for joint calibration)."""
    if not hasattr(model, "group_head") or not hasattr(model, "expert_heads"):
        raise ValueError("Model does not have group_head/expert_heads")
    for p in model.group_head.parameters():
        p.requires_grad = True
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
    """
    Train one residual expert head on a subset dataset.

    Backbone must be frozen externally (recommended).
    Global head remains frozen during expert training.
    """
    # Filter datasets
    filtered_train_dataset = FilteredVideoDataset(train_dataset, label_subset)
    filtered_val_dataset = FilteredVideoDataset(val_dataset, label_subset)

    logger.info("=" * 60)
    logger.info(f"{phase_name}: Expert {expert_idx}, Labels {label_subset}")
    logger.info(f"Filtered train samples: {len(filtered_train_dataset)}")
    logger.info(f"Filtered val samples: {len(filtered_val_dataset)}")
    logger.info("=" * 60)

    if len(filtered_train_dataset) == 0 or len(filtered_val_dataset) == 0:
        logger.warning("No data for this subset; skipping expert phase.")
        return 0.0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    num_workers = config.get('num_workers', 0)
    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        filtered_train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False
    )
    val_loader = DataLoader(
        filtered_val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False
    )

    # Trainable params: only this expert
    unfreeze_single_expert_only(model, expert_idx)
    # IMPORTANT: isolate logits competition in Stage B.
    # Only allow `global + active_expert` to contribute logits.
    if hasattr(model, "experts_enabled"):
        model.experts_enabled = True
    if hasattr(model, "active_expert_idx"):
        model.active_expert_idx = expert_idx
    if hasattr(model, "hard_routing_enabled"):
        model.hard_routing_enabled = False
    expert_params = [p for p in model.expert_heads[expert_idx].parameters() if p.requires_grad]
    if not expert_params:
        logger.warning("No trainable expert params; skipping.")
        return 0.0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    expert_lr = config.get('expert_lr', config.get('head_lr_unfreeze', 2e-4))
    optimizer = torch.optim.AdamW(
        [{"params": expert_params, "lr": expert_lr}],
        weight_decay=config.get('weight_decay_unfreeze', config.get('weight_decay', 0.05))
    )

    from .adaptive_lr_scheduler import get_adaptive_lr_scheduler
    max_epochs = config.get('layer_phase_max_epochs', 30)
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
        use_val_loss=config.get('use_val_loss_for_lr', False)
    )
    scaler = torch.amp.GradScaler()

    patience = config.get('layer_phase_patience', 10)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(max_epochs):
        logger.info(f"\n{phase_name} - Epoch {epoch + 1}/{max_epochs}")

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
            use_distillation=False
        )

        val_loss, val_acc = evaluate(
            model, val_loader, device,
            label_smoothing=0.0,
            use_focal_loss=config.get('use_focal_loss', False),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0)
        )

        scheduler.step(metrics=val_acc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        logger.info(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        logger.info(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            logger.info(f"  ✓ New best val acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            logger.info(f"\n{phase_name} - Early stopping: No improvement for {patience} epochs")
            logger.info(f"  Best val acc: {best_val_acc:.4f}")
            break

    # Freeze this expert after training
    freeze_all_heads(model)
    if hasattr(model, "active_expert_idx"):
        model.active_expert_idx = None
    return best_val_acc, history


def joint_finetune_all_heads(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: Dict[str, Any],
) -> Tuple[float, Dict[str, list]]:
    """Joint finetune global + all experts on full dataset (backbone frozen externally)."""
    unfreeze_all_heads(model)

    max_epochs = config.get('final_head_epochs', 50)
    patience = config.get('early_stop_patience', 20)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    best_val_acc = 0.0
    patience_counter = 0

    logger.info("=" * 60)
    logger.info("Joint Finetune Heads (Global + Experts)")
    logger.info(f"  Max epochs: {max_epochs}, Patience: {patience}")
    logger.info("=" * 60)

    for epoch in range(max_epochs):
        logger.info(f"\nJoint Finetune - Epoch {epoch + 1}/{max_epochs}")

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
            use_distillation=False
        )

        val_loss, val_acc = evaluate(
            model, val_loader, device,
            label_smoothing=0.0,
            use_focal_loss=config.get('use_focal_loss', False),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0)
        )

        scheduler.step(metrics=val_acc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        logger.info(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        logger.info(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            logger.info(f"  ✓ New best val acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            logger.info(f"\nJoint Finetune - Early stopping: No improvement for {patience} epochs")
            logger.info(f"  Best val acc: {best_val_acc:.4f}")
            break

    return best_val_acc, history


def unfreeze_single_layer(model: nn.Module, layer_idx: int, total_layers: int) -> None:
    """
    Unfreeze a single layer in VideoMAE encoder.
    
    Args:
        model: VideoMAE model
        layer_idx: Index of layer to unfreeze (0-based, where 0 is first layer)
        total_layers: Total number of layers in encoder
    """
    if not hasattr(model, 'videomae'):
        raise ValueError("Model does not have 'videomae' attribute")
    
    # Ensure layer_idx is valid
    if layer_idx < 0 or layer_idx >= total_layers:
        raise ValueError(f"Layer index {layer_idx} out of range [0, {total_layers-1}]")
    
    # Freeze all layers first
    for i in range(total_layers):
        for param in model.videomae.encoder.layer[i].parameters():
            param.requires_grad = False
    
    # Unfreeze only the specified layer
    for param in model.videomae.encoder.layer[layer_idx].parameters():
        param.requires_grad = True
    
    logger.info(f"Unfrozen layer {layer_idx} (keeping other {total_layers-1} layers frozen)")


def freeze_single_layer(model: nn.Module, layer_idx: int) -> None:
    """
    Freeze a single layer in VideoMAE encoder.
    
    Args:
        model: VideoMAE model
        layer_idx: Index of layer to freeze
    """
    if not hasattr(model, 'videomae'):
        raise ValueError("Model does not have 'videomae' attribute")
    
    for param in model.videomae.encoder.layer[layer_idx].parameters():
        param.requires_grad = False
    
    logger.info(f"Frozen layer {layer_idx}")


def freeze_all_backbone_layers(model: nn.Module) -> None:
    """
    Freeze all backbone layers (encoder + embeddings).
    
    Args:
        model: VideoMAE model
    """
    if not hasattr(model, 'videomae'):
        raise ValueError("Model does not have 'videomae' attribute")
    
    # Freeze all encoder layers
    for layer in model.videomae.encoder.layer:
        for param in layer.parameters():
            param.requires_grad = False
    
    # Freeze embeddings
    for param in model.videomae.embeddings.parameters():
        param.requires_grad = False
    
    logger.info("Frozen all backbone layers (encoder + embeddings)")


def train_single_layer_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    layer_idx: int,
    label_subset: List[int],
    config: Dict[str, Any],
    phase_name: str = "Layer Phase"
) -> Tuple[float, Dict[str, list]]:
    """
    Train a single layer phase on subset of labels until val accuracy peaks.
    
    Args:
        model: Model to train
        train_loader: Training data loader (filtered for label subset)
        val_loader: Validation data loader (filtered for label subset)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler
        device: Device to train on
        layer_idx: Index of layer being trained
        label_subset: List of label indices in this phase
        config: Configuration dictionary
        phase_name: Name of this phase for logging
    
    Returns:
        Best validation accuracy and training history
    """
    max_epochs = config.get('layer_phase_max_epochs', 30)
    patience = config.get('layer_phase_patience', 10)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0
    
    logger.info("="*60)
    logger.info(f"{phase_name}: Training layer {layer_idx} on labels {label_subset}")
    logger.info(f"  Max epochs: {max_epochs}, Patience: {patience}")
    logger.info("="*60)
    
    for epoch in range(max_epochs):
        logger.info(f"\n{phase_name} - Epoch {epoch + 1}/{max_epochs}")
        
        # Training
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
            ema_model=None,  # No EMA for layer phases
            use_synthetic_data=config.get('use_synthetic_data', False),
            synthetic_method=config.get('synthetic_method', 'frame_mixup'),
            synthetic_ratio=config.get('synthetic_ratio', 0.3),
            teacher=None,
            use_distillation=False
        )
        
        # Validation
        val_loss, val_acc = evaluate(
            model, val_loader, device,
            label_smoothing=0.0,  # No label smoothing for validation
            use_focal_loss=config.get('use_focal_loss', False),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0)
        )
        
        # Update scheduler
        if hasattr(scheduler, 'step'):
            if hasattr(scheduler, 'best_metric'):
                scheduler.step(metrics=val_acc)
            else:
                scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        logger.info(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            logger.info(f"  ✓ New best val acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"\n{phase_name} - Early stopping: No improvement for {patience} epochs")
            logger.info(f"  Best val acc: {best_val_acc:.4f}")
            break
    
    logger.info(f"\n{phase_name} completed. Best val acc: {best_val_acc:.4f}")
    return best_val_acc, history


def sequential_layer_training_loop(
    model: nn.Module,
    train_dataset,
    val_dataset,
    device: torch.device,
    config: Dict[str, Any],
    classes: List[str]
) -> Tuple[Dict[str, list], float]:
    """
    Sequential layer training: train each layer separately on subset of labels.
    
    Args:
        model: Model to train
        train_dataset: Full training dataset
        val_dataset: Full validation dataset
        device: Device to train on
        config: Configuration dictionary
        classes: List of class names
    
    Returns:
        Combined training history and best overall val accuracy
    """
    num_classes = len(classes)
    num_layer_phases = config.get('num_layer_phases', 8)
    
    # Calculate label subsets (divide 51 labels into 8 parts)
    labels_per_phase = num_classes // num_layer_phases
    remainder = num_classes % num_layer_phases
    
    label_subsets = []
    start_idx = 0
    for i in range(num_layer_phases):
        # Distribute remainder labels across first few phases
        size = labels_per_phase + (1 if i < remainder else 0)
        end_idx = start_idx + size
        label_subsets.append(list(range(start_idx, end_idx)))
        start_idx = end_idx
    
    # Get total layers in VideoMAE encoder
    if not hasattr(model, 'videomae'):
        raise ValueError("Model does not have 'videomae' attribute")
    
    total_layers = len(model.videomae.encoder.layer)
    
    # Calculate which layers to train (last 8 layers: layers 16-23)
    # VideoMAE has 24 layers (0-23), we train layers 16-23 (8 layers)
    start_layer = total_layers - num_layer_phases  # 24 - 8 = 16
    layer_indices = list(range(start_layer, total_layers))  # [16, 17, 18, 19, 20, 21, 22, 23]
    
    logger.info("="*60)
    logger.info("Sequential Layer Training Phase")
    logger.info("="*60)
    logger.info(f"Total layers: {total_layers}")
    logger.info(f"Training layers: {layer_indices} (last {num_layer_phases} layers)")
    logger.info("Label subsets:")
    for i, subset in enumerate(label_subsets):
        logger.info(f"  Phase {i+1}: Layer {layer_indices[i]}, Labels {subset} ({len(subset)} labels)")
    logger.info("="*60)
    
    # Combined history
    combined_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_overall_val_acc = 0.0
    
    # Freeze all layers initially
    freeze_all_backbone_layers(model)
    
    # Setup optimizer for layer training (only train unfrozen layer + head)
    base_lr_unfreeze = config.get('base_lr_unfreeze', 2e-6)
    head_lr_unfreeze = config.get('head_lr_unfreeze', 2e-4)
    
    for phase_idx, (layer_idx, label_subset) in enumerate(zip(layer_indices, label_subsets)):
        phase_name = f"Phase {phase_idx + 1}/{num_layer_phases}"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{phase_name}: Layer {layer_idx}, Labels {label_subset}")
        logger.info(f"{'='*60}")
        
        # Unfreeze this layer
        unfreeze_single_layer(model, layer_idx, total_layers)
        
        # Filter datasets for this label subset
        filtered_train_dataset = FilteredVideoDataset(train_dataset, label_subset)
        filtered_val_dataset = FilteredVideoDataset(val_dataset, label_subset)
        
        logger.info(f"Filtered train samples: {len(filtered_train_dataset)}")
        logger.info(f"Filtered val samples: {len(filtered_val_dataset)}")
        
        if len(filtered_train_dataset) == 0:
            logger.warning(f"No training samples for labels {label_subset}, skipping phase")
            freeze_single_layer(model, layer_idx)
            continue
        
        # Create data loaders
        num_workers = config.get('num_workers', 0)
        use_pin_memory = torch.cuda.is_available()
        
        train_loader = DataLoader(
            filtered_train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            drop_last=False
        )
        
        val_loader = DataLoader(
            filtered_val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            drop_last=False
        )
        
        # Setup optimizer for this phase (only unfrozen layer + head)
        # Collect parameters from the unfrozen layer
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'classifier' in name or 'head' in name:
                head_params.append(param)
            elif 'videomae' in name and f'encoder.layer.{layer_idx}' in name:
                # This layer is unfrozen
                backbone_params.append(param)
        
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": base_lr_unfreeze})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr_unfreeze})
        
        if len(param_groups) == 0:
            logger.warning(f"No trainable parameters for layer {layer_idx}, skipping")
            freeze_single_layer(model, layer_idx)
            continue
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=config.get('weight_decay_unfreeze', config.get('weight_decay', 0.05))
        )
        
        # Setup scheduler
        from .adaptive_lr_scheduler import get_adaptive_lr_scheduler
        scheduler = get_adaptive_lr_scheduler(
            optimizer,
            num_epochs=config.get('layer_phase_max_epochs', 30),
            warmup_epochs=config.get('warmup_epochs', 5),
            min_lr_ratio=config.get('min_lr_ratio', 0.01),
            plateau_patience=config.get('lr_plateau_patience', 8),
            min_delta=config.get('lr_min_delta', 0.005),
            mode='max',
            verbose=True,
            cooldown=config.get('lr_cooldown', 3),
            threshold_mode=config.get('lr_threshold_mode', 'rel'),
            use_val_loss=config.get('use_val_loss_for_lr', False)
        )
        
        scaler = torch.amp.GradScaler()
        
        # Train this layer phase
        best_val_acc, phase_history = train_single_layer_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            layer_idx=layer_idx,
            label_subset=label_subset,
            config=config,
            phase_name=phase_name
        )
        
        # Freeze this layer after training
        freeze_single_layer(model, layer_idx)
        
        # Update combined history
        combined_history['train_loss'].extend(phase_history['train_loss'])
        combined_history['train_acc'].extend(phase_history['train_acc'])
        combined_history['val_loss'].extend(phase_history['val_loss'])
        combined_history['val_acc'].extend(phase_history['val_acc'])
        
        # Track best overall val acc
        if best_val_acc > best_overall_val_acc:
            best_overall_val_acc = best_val_acc
        
        logger.info(f"{phase_name} completed. Best val acc: {best_val_acc:.4f}")
        logger.info(f"Layer {layer_idx} frozen. Moving to next phase...")
    
    logger.info("="*60)
    logger.info("Sequential Layer Training completed!")
    logger.info(f"Best overall val acc: {best_overall_val_acc:.4f}")
    logger.info("="*60)
    
    return combined_history, best_overall_val_acc


def final_head_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: Dict[str, Any]
) -> Tuple[float, Dict[str, list]]:
    """
    Final head training phase: freeze all backbone, train head on all labels.
    
    Args:
        model: Model to train
        train_loader: Training data loader (all labels)
        val_loader: Validation data loader (all labels)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler
        device: Device to train on
        config: Configuration dictionary
    
    Returns:
        Best validation accuracy and training history
    """
    # Freeze all backbone layers
    freeze_all_backbone_layers(model)
    
    max_epochs = config.get('final_head_epochs', 50)
    patience = config.get('early_stop_patience', 20)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0
    
    logger.info("="*60)
    logger.info("Final Head Training Phase")
    logger.info("  Freeze all backbone, train head on all labels")
    logger.info(f"  Max epochs: {max_epochs}, Patience: {patience}")
    logger.info("="*60)
    
    for epoch in range(max_epochs):
        logger.info(f"\nFinal Head Training - Epoch {epoch + 1}/{max_epochs}")
        
        # Training
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
            use_distillation=False
        )
        
        # Validation
        val_loss, val_acc = evaluate(
            model, val_loader, device,
            label_smoothing=0.0,
            use_focal_loss=config.get('use_focal_loss', False),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0)
        )
        
        # Update scheduler
        if hasattr(scheduler, 'step'):
            if hasattr(scheduler, 'best_metric'):
                scheduler.step(metrics=val_acc)
            else:
                scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        logger.info(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            logger.info(f"  ✓ New best val acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"\nFinal Head Training - Early stopping: No improvement for {patience} epochs")
            logger.info(f"  Best val acc: {best_val_acc:.4f}")
            break
    
    logger.info(f"\nFinal Head Training completed. Best val acc: {best_val_acc:.4f}")
    return best_val_acc, history
