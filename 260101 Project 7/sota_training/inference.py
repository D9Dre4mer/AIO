"""
Inference functions with Enhanced TTA for video action recognition.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)


def enhanced_tta(
    video: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    num_crops: int = 20,
    num_flips: int = 2,
    scales: List[float] = None,
    multi_scale_sizes: List[int] = None,
    use_temporal_aug: bool = True,
    confidence_threshold: float = None,
    expected_num_frames: int = None  # Expected number of frames (from training config)
) -> torch.Tensor:
    """
    Apply Enhanced Test-Time Augmentation to a single video.

    Args:
        video: [B, T, C, H, W] tensor (B=1 for single video)
        model: Model to use for inference
        device: Device to run on
        num_crops: Number of spatial crops (default: 20 for more coverage)
        num_flips: Number of flips (1=original, 2=original+flipped)
        scales: List of scales for multi-scale crops
        multi_scale_sizes: List of image sizes for multi-scale inference
        use_temporal_aug: Whether to use temporal augmentation (forward/backward)
        confidence_threshold: If specified, only use TTA predictions above threshold

    Returns:
        Averaged logits [num_classes]
    """
    if scales is None:
        scales = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
    if multi_scale_sizes is None:
        multi_scale_sizes = [224, 256, 288, 320]

    # Ensure video has correct shape [B, T, C, H, W]
    if video.dim() == 4:
        # If shape is [T, C, H, W], add batch dimension
        video = video.unsqueeze(0)
    # Ensure video has correct shape [B, T, C, H, W]
    if video.dim() == 4:
        # If shape is [T, C, H, W], add batch dimension
        video = video.unsqueeze(0)
    
    B, T, C, H, W = video.shape
    video = video.to(device)
    
    # #region agent log
    import json
    log_path = Path('.cursor/debug.log')
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            log_entry = {
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "A",
                "location": "inference.py:59",
                "message": "Enhanced TTA entry - video shape",
                "data": {
                    "B": int(B),
                    "T": int(T),
                    "C": int(C),
                    "H": int(H),
                    "W": int(W),
                    "video_size": int(video.numel()),
                    "num_crops": int(num_crops),
                    "num_flips": int(num_flips)
                },
                "timestamp": int(time.time() * 1000)
            }
            f.write(json.dumps(log_entry) + '\n')
    except Exception:
        pass
    # #endregion
    
    # Validate shape - T should match expected num_frames
    # If expected_num_frames is provided, check against it; otherwise skip check
    if expected_num_frames is not None and T != expected_num_frames:
        # Log warning but continue - model should handle variable T
        logger.warning(f"Video has {T} frames, expected {expected_num_frames}. This may cause issues in TTA.")
    elif expected_num_frames is None and T != 16:
        # Legacy check: if expected_num_frames not provided, default to 16
        logger.warning(f"Video has {T} frames, expected 16 (default). This may cause issues in TTA.")

    all_logits = []
    all_confidences = []

    # Original video
    with torch.no_grad():
        logits = model(video)
        probs = F.softmax(logits, dim=1)
        max_conf = probs.max(dim=1)[0]
        all_logits.append(logits)
        all_confidences.append(max_conf)

    # Horizontal flip
    if num_flips > 1:
        video_flipped = torch.flip(video, dims=[-1])
        with torch.no_grad():
            logits = model(video_flipped)
            probs = F.softmax(logits, dim=1)
            max_conf = probs.max(dim=1)[0]
            all_logits.append(logits)
            all_confidences.append(max_conf)

    # Temporal augmentation: forward and backward
    if use_temporal_aug and T > 1:
        # Backward (reversed temporal order)
        video_backward = torch.flip(video, dims=[1])
        with torch.no_grad():
            logits = model(video_backward)
            probs = F.softmax(logits, dim=1)
            max_conf = probs.max(dim=1)[0]
            all_logits.append(logits)
            all_confidences.append(max_conf)

    # Multiple crops (enhanced: more crops)
    if num_crops > 1:
        base_crop_size = int(H * 0.875)

        # Standard 5 crops: center + 4 corners
        standard_crops = [
            (0, 0),  # Top-left
            (0, W - base_crop_size),  # Top-right
            (H - base_crop_size, 0),  # Bottom-left
            (H - base_crop_size, W - base_crop_size),  # Bottom-right
            ((H - base_crop_size) // 2, (W - base_crop_size) // 2),  # Center
        ]

        # Apply standard crops
        for top, left in standard_crops[:min(num_crops, 5)]:
            video_crop = video[:, :, :, top:top+base_crop_size, left:left+base_crop_size]
            # Get actual T from video_crop shape (may differ from original T)
            B_crop, T_crop, C_crop, H_crop, W_crop = video_crop.shape
            
            # #region agent log
            import json
            log_path = Path('.cursor/debug.log')
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A",
                        "location": "inference.py:117",
                        "message": "Before view operation - standard crops",
                        "data": {
                            "B_crop": int(B_crop),
                            "T_crop": int(T_crop),
                            "C_crop": int(C_crop),
                            "H_crop": int(H_crop),
                            "W_crop": int(W_crop),
                            "base_crop_size": int(base_crop_size),
                            "video_crop_size": int(video_crop.numel()),
                            "expected_view_size": int(B_crop * T_crop * C_crop * base_crop_size * base_crop_size),
                            "top": int(top),
                            "left": int(left),
                            "H": int(H),
                            "W": int(W)
                        },
                        "timestamp": int(time.time() * 1000)
                    }
                    f.write(json.dumps(log_entry) + '\n')
            except Exception:
                pass
            # #endregion
            
            # Use actual H_crop and W_crop instead of base_crop_size to ensure consistency
            video_crop = F.interpolate(
                video_crop.view(B_crop*T_crop, C_crop, H_crop, W_crop),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            video_crop = video_crop.view(B_crop, T_crop, C_crop, H, W)

            with torch.no_grad():
                logits = model(video_crop)
                probs = F.softmax(logits, dim=1)
                max_conf = probs.max(dim=1)[0]
                all_logits.append(logits)
                all_confidences.append(max_conf)

        # Additional crops: more corner/edge crops
        if num_crops > 5:
            additional_crops = [
                (H // 4, W // 4),  # Quarter positions
                (H // 4, 3 * W // 4),
                (3 * H // 4, W // 4),
                (3 * H // 4, 3 * W // 4),
            ]
            for top, left in additional_crops[:min(num_crops-5, 4)]:
                video_crop = video[:, :, :, top:top+base_crop_size, left:left+base_crop_size]
                # Get actual T from video_crop shape (may differ from original T)
                B_crop, T_crop, C_crop, H_crop, W_crop = video_crop.shape
                
                # #region agent log
                import json
                log_path = Path('.cursor/debug.log')
                try:
                    with open(log_path, 'a', encoding='utf-8') as f:
                        log_entry = {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "A",
                            "location": "inference.py:144",
                            "message": "Before view operation - additional crops",
                            "data": {
                                "B_crop": int(B_crop),
                                "T_crop": int(T_crop),
                                "C_crop": int(C_crop),
                                "H_crop": int(H_crop),
                                "W_crop": int(W_crop),
                                "base_crop_size": int(base_crop_size),
                                "video_crop_size": int(video_crop.numel()),
                                "expected_view_size": int(B_crop * T_crop * C_crop * base_crop_size * base_crop_size),
                                "top": int(top),
                                "left": int(left),
                                "H": int(H),
                                "W": int(W)
                            },
                            "timestamp": int(time.time() * 1000)
                        }
                        f.write(json.dumps(log_entry) + '\n')
                except Exception:
                    pass
                # #endregion
                
                # Use actual H_crop and W_crop instead of base_crop_size to ensure consistency
                video_crop = F.interpolate(
                    video_crop.view(B_crop*T_crop, C_crop, H_crop, W_crop),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
                video_crop = video_crop.view(B_crop, T_crop, C_crop, H, W)

                with torch.no_grad():
                    logits = model(video_crop)
                    probs = F.softmax(logits, dim=1)
                    max_conf = probs.max(dim=1)[0]
                    all_logits.append(logits)
                    all_confidences.append(max_conf)

        # Multi-scale center crops
        if num_crops > 9:
            for scale in scales[:num_crops-9]:
                crop_size = int(base_crop_size * scale)
                if 0 < crop_size < min(H, W):
                    top = max(0, (H - crop_size) // 2)
                    left = max(0, (W - crop_size) // 2)
                    video_crop = video[:, :, :, top:top+crop_size, left:left+crop_size]
                    # Get actual T from video_crop shape (may differ from original T)
                    B_crop, T_crop, C_crop, H_crop, W_crop = video_crop.shape
                    
                    # #region agent log
                    import json
                    log_path = Path('.cursor/debug.log')
                    try:
                        with open(log_path, 'a', encoding='utf-8') as f:
                            log_entry = {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "A",
                                "location": "inference.py:169",
                                "message": "Before view operation - multi-scale crops",
                                "data": {
                                    "B_crop": int(B_crop),
                                    "T_crop": int(T_crop),
                                    "C_crop": int(C_crop),
                                    "H_crop": int(H_crop),
                                    "W_crop": int(W_crop),
                                    "crop_size": int(crop_size),
                                    "video_crop_size": int(video_crop.numel()),
                                    "expected_view_size": int(B_crop * T_crop * C_crop * crop_size * crop_size),
                                    "top": int(top),
                                    "left": int(left),
                                    "H": int(H),
                                    "W": int(W),
                                    "scale": float(scale)
                                },
                                "timestamp": int(time.time() * 1000)
                            }
                            f.write(json.dumps(log_entry) + '\n')
                    except Exception:
                        pass
                    # #endregion
                    
                    # Use actual H_crop and W_crop instead of crop_size to ensure consistency
                    video_crop = F.interpolate(
                        video_crop.view(B_crop*T_crop, C_crop, H_crop, W_crop),
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    )
                    video_crop = video_crop.view(B_crop, T_crop, C_crop, H, W)

                    with torch.no_grad():
                        logits = model(video_crop)
                        probs = F.softmax(logits, dim=1)
                        max_conf = probs.max(dim=1)[0]
                        all_logits.append(logits)
                        all_confidences.append(max_conf)

    # Multi-scale inference
    if multi_scale_sizes:
        for size in multi_scale_sizes:
            if size != H:  # Skip if same as original
                # Get actual T from video shape (may differ from original T)
                B_scale, T_scale, C_scale, H_scale, W_scale = video.shape
                video_resized = F.interpolate(
                    video.view(B_scale*T_scale, C_scale, H_scale, W_scale),
                    size=(size, size),
                    mode='bilinear',
                    align_corners=False
                )
                video_resized = video_resized.view(B_scale, T_scale, C_scale, size, size)
                video_resized = F.interpolate(
                    video_resized.view(B_scale*T_scale, C_scale, size, size),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
                video_resized = video_resized.view(B_scale, T_scale, C_scale, H, W)

                with torch.no_grad():
                    logits = model(video_resized)
                    probs = F.softmax(logits, dim=1)
                    max_conf = probs.max(dim=1)[0]
                    all_logits.append(logits)
                    all_confidences.append(max_conf)

    # Filter by confidence if threshold specified
    if confidence_threshold is not None:
        all_logits_tensor = torch.stack(all_logits)
        all_confidences_tensor = torch.stack(all_confidences)
        mask = all_confidences_tensor >= confidence_threshold
        if mask.any():
            all_logits = [all_logits[i] for i in range(len(all_logits)) if mask[i].item()]
        if len(all_logits) == 0:
            # Fallback to all if none pass threshold
            all_logits = [torch.stack(all_logits).mean(dim=0)]

    # Average all predictions
    all_logits = torch.stack(all_logits)
    avg_logits = all_logits.mean(dim=0)

    return avg_logits


def test_time_augment(
    video: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    num_crops: int = 10,
    num_flips: int = 2
) -> torch.Tensor:
    """
    Apply Test-Time Augmentation to a single video (alias for enhanced_tta).
    
    Args:
        video: [B, T, C, H, W] tensor
        model: Model to use for inference
        device: Device to run on
        num_crops: Number of spatial crops
        num_flips: Number of flips
    
    Returns:
        Averaged logits [num_classes]
    """
    return enhanced_tta(video, model, device, num_crops, num_flips)


def run_inference(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    classes: List[str],
    use_tta: bool = True,
    num_crops: int = 10,
    num_flips: int = 2,
    expected_num_frames: int = None,  # Expected number of frames (from training config)
    return_label_as_index: bool = False,  # True → (video_id, pred_int); False → (video_id, class_name)
) -> List[Tuple[int, object]]:
    """
    Run inference on test dataset.

    Model 12 (Group-Gated): forward = group_head → argmax(group) → expert[group] cho subset;
    output logits [B, num_classes] với chỉ indices thuộc subset được fill, còn lại -1e9 → argmax = class index.

    Args:
        model: Model to use for inference
        test_loader: Test data loader
        device: Device to run on
        classes: List of class names (index i = class name for logit index i)
        use_tta: Whether to use Test-Time Augmentation
        num_crops: Number of crops for TTA
        num_flips: Number of flips for TTA
        return_label_as_index: If True, return (video_id, pred_int); else (video_id, classes[pred])

    Returns:
        List of (video_id, label) where label is int if return_label_as_index else str
    """
    model.eval()
    predictions = []

    logger.info("Running inference on test set...")
    if use_tta:
        logger.info(f"Using Enhanced TTA: {num_crops} crops, {num_flips} flips")

    # Multi-clip aggregation: collect predictions per video
    video_logits = {}  # {video_id: [logits from all clips]}

    with torch.no_grad():
        progress = tqdm(test_loader, desc="Inference")
        for videos, video_ids in progress:
            videos = videos.to(device, non_blocking=True)

            if use_tta:
                batch_logits = []
                for i in range(videos.shape[0]):
                    video = videos[i:i+1]
                    logits = enhanced_tta(
                        video, model, device, num_crops, num_flips,
                        expected_num_frames=expected_num_frames
                    )
                    batch_logits.append(logits)
                logits = torch.cat(batch_logits, dim=0)
            else:
                logits = model(videos)

            for video_id, logit in zip(video_ids, logits):
                video_id = int(video_id)
                if video_id not in video_logits:
                    video_logits[video_id] = []
                video_logits[video_id].append(logit.cpu())

    for video_id, clip_logits in video_logits.items():
        avg_logits = torch.stack(clip_logits).mean(dim=0)
        pred = avg_logits.argmax().item()
        label = pred if return_label_as_index else classes[pred]
        predictions.append((video_id, label))

    logger.info(f"Inference completed. Total predictions: {len(predictions)}")
    return predictions


def generate_submission(
    predictions: List[Tuple[int, object]],
    output_path: Path,
    logger: logging.Logger = None,
    template_format: bool = False,
):
    """
    Generate submission CSV file.

    template_format=False (mặc định):
        - Cột id, label; id = video_id (tên thư mục test/); label = tên class hoặc index.

    template_format=True (khớp kaggle_data/submission_template.csv):
        - Mẫu submission_template: id 0,1,2,... tương ứng số thứ tự của các folder trong kaggle_data/data/test.
        - Cột id, class; id = 0, 1, 2, ... theo thứ tự thư mục trong test (sort theo int(tên thư mục), giống TestDataset).
        - id=0 = thư mục đầu tiên, id=1 = thư mục thứ hai, ...
        - class = tên class hoặc index (giống cột label nhưng tên cột là "class").
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # predictions: (video_id, label) — label có thể str hoặc int
    df = pd.DataFrame(predictions, columns=['video_id', 'label'])
    df = df.drop_duplicates(subset='video_id', keep='first')
    df = df.sort_values('video_id').reset_index(drop=True)

    if template_format:
        # id = 0, 1, 2, ... theo thứ tự thư mục trong test (đã sort theo video_id = int(tên thư mục))
        df_out = pd.DataFrame({
            'id': range(len(df)),
            'class': df['label'].values,
        })
    else:
        df_out = df.rename(columns={'video_id': 'id', 'label': 'label'})
        df_out = df_out[['id', 'label']]

    df_out.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Submission saved to: {output_path}")
    logger.info(f"Total predictions: {len(df_out)}")
    logger.info(f"Format: {'template (id, class)' if template_format else 'id, label'}")
    logger.info(f"Sample:\n{df_out.head(10)}")


def run_ensemble_inference(
    models: List[torch.nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    classes: List[str],
    weights: List[float] = None,
    use_tta: bool = True,
    num_crops: int = 10,
    num_flips: int = 2,
    ensemble_method: str = 'weighted_avg_probs',  # Best method: weighted average of probabilities
    use_softmax: bool = True  # Always use softmax (required for weighted_avg_probs)
) -> List[Tuple[int, str]]:
    """
    Run ensemble inference with multiple models.
    
    Args:
        models: List of models for ensemble
        test_loader: Test data loader
        device: Device to run on
        classes: List of class names
        weights: Weights for each model (default: uniform or based on val_acc)
        use_tta: Whether to use Test-Time Augmentation
        num_crops: Number of crops for TTA
        num_flips: Number of flips for TTA
        ensemble_method: Method to use for ensemble (default: 'weighted_avg_probs' - best method)
            - 'weighted_avg_probs': Weighted average of probabilities (BEST - uses softmax)
            - 'weighted_avg_logits': Weighted average of logits (old method, not recommended)
            - 'voting': Majority vote
            - 'geometric_mean': Geometric mean of probabilities
        use_softmax: Apply softmax before ensemble (always True for weighted_avg_probs)
    
    Returns:
        List of (video_id, predicted_class) tuples
    """
    for model in models:
        model.eval()
    
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    logger.info(f"Running ensemble inference with {len(models)} models")
    logger.info(f"Ensemble method: {ensemble_method}")
    logger.info(f"Use softmax: {use_softmax}")
    logger.info(f"Model weights: {[f'{w:.4f}' for w in weights]}")
    if use_tta:
        logger.info(f"Using Enhanced TTA: {num_crops} crops, {num_flips} flips")
    
    predictions = []
    
    with torch.no_grad():
        progress = tqdm(test_loader, desc="Ensemble Inference")
        for videos, video_ids in progress:
            videos = videos.to(device, non_blocking=True)
            
            # Collect logits from all models
            all_logits = []
            
            for model in models:
                if use_tta:
                    # Apply TTA to each video in batch
                    batch_logits = []
                    for i in range(videos.shape[0]):
                        video = videos[i:i+1]  # [1, T, C, H, W]
                        logits = enhanced_tta(video, model, device, num_crops, num_flips)
                        batch_logits.append(logits)
                    logits = torch.cat(batch_logits, dim=0)
                else:
                    logits = model(videos)
                all_logits.append(logits)
            
            # Ensemble methods
            if ensemble_method == 'weighted_avg_logits':
                # Old method: Weighted average of logits (not recommended)
                ensemble_logits = torch.zeros_like(all_logits[0])
                for logits, weight in zip(all_logits, weights):
                    ensemble_logits += weight * logits
                preds = ensemble_logits.argmax(dim=1).cpu().numpy()
                
            elif ensemble_method == 'weighted_avg_probs':
                # Recommended: Weighted average of probabilities
                ensemble_probs = torch.zeros_like(F.softmax(all_logits[0], dim=1))
                for logits, weight in zip(all_logits, weights):
                    probs = F.softmax(logits, dim=1)
                    ensemble_probs += weight * probs
                preds = ensemble_probs.argmax(dim=1).cpu().numpy()
                
            elif ensemble_method == 'voting':
                # Majority vote
                all_preds = [logits.argmax(dim=1) for logits in all_logits]
                all_preds = torch.stack(all_preds)  # [num_models, batch_size]
                ensemble_preds = torch.mode(all_preds, dim=0)[0]
                preds = ensemble_preds.cpu().numpy()
                
            elif ensemble_method == 'geometric_mean':
                # Geometric mean of probabilities
                ensemble_probs = torch.ones_like(F.softmax(all_logits[0], dim=1))
                for logits, weight in zip(all_logits, weights):
                    probs = F.softmax(logits, dim=1)
                    # Avoid numerical issues with very small probabilities
                    probs = torch.clamp(probs, min=1e-8)
                    ensemble_probs *= probs ** weight
                preds = ensemble_probs.argmax(dim=1).cpu().numpy()
                
            else:
                raise ValueError(f"Unknown ensemble method: {ensemble_method}")
            
            for video_id, pred in zip(video_ids, preds):
                predicted_class = classes[pred]
                predictions.append((int(video_id), predicted_class))
    
    logger.info(f"Ensemble inference completed. Total predictions: {len(predictions)}")
    return predictions
