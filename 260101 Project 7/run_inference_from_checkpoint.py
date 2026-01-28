"""
Script chọn checkpoint và chạy inference tạo submission.
Mặc định infer giống kiểm tra trên tập val: 1 clip/video, không TTA. Dùng --tta (và --num-clips 10 nếu cần) để bật TTA.
"""

import os

# Fix OpenMP duplicate library error on Windows
# MUST be set BEFORE importing torch/numpy to prevent OpenMP initialization conflict
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from pathlib import Path
import logging
import argparse
from typing import Optional

# GPU optimization settings
cuda_launch_blocking = os.environ.get('CUDA_LAUNCH_BLOCKING', '0')
os.environ['CUDA_LAUNCH_BLOCKING'] = cuda_launch_blocking
torch.backends.cudnn.benchmark = True

from sota_training.config import get_default_config
from sota_training.utils import setup_logging, log_system_info, load_checkpoint
from sota_training.dataset import VideoDataset, TestDataset
from sota_training.models import create_model
from sota_training.inference import run_inference, generate_submission
from sota_training.sequential_layer_training import build_contiguous_label_subsets

logger = logging.getLogger(__name__)


def list_checkpoints(checkpoints_dir: Path) -> list:
    """List tất cả checkpoint files (.pt) trong thư mục."""
    checkpoints = []
    
    # Tìm tất cả file .pt
    for pt_file in checkpoints_dir.rglob('*.pt'):
        # Bỏ qua các file trong subdirectories không mong muốn
        if 'models--' in str(pt_file):
            continue
        checkpoints.append(pt_file)
    
    # Sắp xếp theo tên
    checkpoints.sort(key=lambda x: x.name)
    
    return checkpoints


def select_checkpoint(checkpoints: list, checkpoint_path: Optional[Path] = None) -> Path:
    """Cho phép user chọn checkpoint từ danh sách."""
    if checkpoint_path is not None:
        if checkpoint_path.exists():
            return checkpoint_path
        else:
            logger.warning(f"Checkpoint path không tồn tại: {checkpoint_path}")
    
    if len(checkpoints) == 0:
        raise ValueError("Không tìm thấy checkpoint nào trong thư mục!")
    
    print("\n" + "="*60)
    print("Danh sách checkpoint có sẵn:")
    print("="*60)
    for i, ckpt in enumerate(checkpoints, 1):
        # Lấy thông tin checkpoint nếu có
        try:
            checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
            epoch = checkpoint.get('epoch', 'unknown')
            val_acc = checkpoint.get('val_acc', 0.0)
            print(f"{i}. {ckpt.name}")
            print(f"   Epoch: {epoch}, Val Acc: {val_acc:.4f}")
            print(f"   Path: {ckpt}")
        except Exception as e:
            print(f"{i}. {ckpt.name} (không thể đọc metadata: {e})")
            print(f"   Path: {ckpt}")
        print()
    
    while True:
        try:
            choice = input(f"Chọn checkpoint (1-{len(checkpoints)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                selected = checkpoints[idx]
                print(f"\nĐã chọn: {selected.name}")
                return selected
            else:
                print(f"Vui lòng chọn số từ 1 đến {len(checkpoints)}")
        except ValueError:
            print("Vui lòng nhập số hợp lệ")
        except KeyboardInterrupt:
            print("\nĐã hủy.")
            raise


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
        if 'model_12' in filename or 'model12' in filename or 'improved_heads' in filename:
            config['architecture'] = 'videomae_group_gated_experts'
            logger.info("Detected Model 12 (or improved) from filename")
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


def run_inference_core(
    selected_checkpoint: Path,
    args,
    data_dir: Path,
    submissions_dir: Path,
) -> None:
    """
    Load checkpoint, tạo model, chạy inference và ghi submission.
    args cần có: num_clips, tta, num_crops, num_flips, batch_size, num_workers,
    label_format, submission_template.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    log_system_info(logger, device)

    logger.info(f"Loading checkpoint: {selected_checkpoint}")
    checkpoint = load_checkpoint(selected_checkpoint, device, logger)
    model_config = get_model_config_from_checkpoint(selected_checkpoint)

    train_data_dir = data_dir / 'data_train'
    if 'classes' in checkpoint and checkpoint['classes']:
        classes = list(checkpoint['classes'])
        num_classes = len(classes)
        logger.info(f"Using {num_classes} classes from checkpoint (khớp với model đã train)")
    elif train_data_dir.exists():
        train_dataset = VideoDataset(
            root=train_data_dir,
            num_frames=model_config.get('num_frames', 16),
            frame_stride=model_config.get('frame_stride', 2),
            image_size=model_config.get('img_size', 224),
            is_train=True,
            val_ratio=0.0,
            seed=42
        )
        num_classes = len(train_dataset.classes)
        classes = train_dataset.classes
        logger.info(f"Found {num_classes} classes from dataset (checkpoint không có 'classes')")
    else:
        logger.warning("Không tìm thấy số classes, dùng default 51")
        num_classes = 51
        classes = [f"class_{i}" for i in range(num_classes)]

    label_subsets = None
    if model_config['architecture'] in ['videomae_group_gated_experts', 'videomae_global_residual_experts']:
        if 'label_subsets' in model_config and model_config['label_subsets']:
            label_subsets = model_config['label_subsets']
            logger.info(f"Loaded label_subsets from checkpoint config: {len(label_subsets)} groups")
        elif 'config' in checkpoint and checkpoint.get('config') and checkpoint['config'].get('label_subsets'):
            label_subsets = checkpoint['config']['label_subsets']
            logger.info(f"Loaded label_subsets from checkpoint config: {len(label_subsets)} groups")
        elif 'label_subsets' in checkpoint and checkpoint['label_subsets']:
            label_subsets = checkpoint['label_subsets']
            logger.info(f"Loaded label_subsets from checkpoint (top-level, model 12 improved): {len(label_subsets)} groups")
        else:
            num_experts = model_config.get('num_experts', 8)
            label_subsets = build_contiguous_label_subsets(num_classes, num_experts=num_experts)
            logger.info(f"Created label_subsets: {len(label_subsets)} groups (num_experts={num_experts})")

    logger.info(f"Creating {model_config['architecture']} model...")
    ckpt_has_adapters = any(k.startswith("adapters.") for k in checkpoint.get("model", {}).keys())
    use_adapters = bool(model_config.get("use_adapters", False)) and ckpt_has_adapters
    if not ckpt_has_adapters and model_config.get("use_adapters", False):
        logger.info("Checkpoint không có adapters → tạo model với use_adapters=False")
    pretrained_ckpt = model_config.get("pretrained_ckpt")
    if pretrained_ckpt is None and model_config["architecture"] in [
        "videomae_group_gated_experts", "videomae_global_residual_experts"
    ]:
        pretrained_ckpt = str(selected_checkpoint) if selected_checkpoint.exists() else None
        if pretrained_ckpt:
            logger.info("Using checkpoint as pretrained_ckpt (load backbone from same file)")
    create_kwargs = {
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
        create_kwargs['label_subsets'] = label_subsets
        create_kwargs['init_output_gain'] = model_config.get('init_output_gain', 2.0)
        create_kwargs['init_hidden_gain'] = model_config.get('init_hidden_gain', 1.0)
        create_kwargs['use_normal_init'] = model_config.get('use_normal_init', True)
        if model_config['architecture'] == 'videomae_group_gated_experts':
            create_kwargs['hard_mask_value'] = model_config.get('hard_mask_value', -1e9)
    model = create_model(**create_kwargs).to(device)

    if 'model' in checkpoint:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        if missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
        logger.info("Model weights loaded from checkpoint")
    else:
        raise ValueError("Checkpoint không chứa 'model' state dict!")
    model.eval()
    if model_config.get('architecture') == 'videomae_group_gated_experts':
        if hasattr(model, 'active_expert_idx'):
            model.active_expert_idx = None
        if hasattr(model, 'hard_routing_enabled'):
            model.hard_routing_enabled = True
        logger.info("Model 12: inference mode (active_expert_idx=None, hard_routing_enabled=True)")

    test_data_dir = data_dir / 'test'
    if not test_data_dir.exists():
        logger.error(f"Test directory không tồn tại: {test_data_dir}")
        return
    logger.info("Loading test dataset...")
    num_clips = getattr(args, 'num_clips', 1)
    test_dataset = TestDataset(
        root=test_data_dir,
        num_frames=model_config.get('num_frames', 16),
        frame_stride=model_config.get('frame_stride', 2),
        image_size=model_config.get('img_size', 224),
        num_clips=num_clips
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    logger.info(f"Test samples: {len(test_dataset)}")

    use_tta = getattr(args, 'tta', False)
    logger.info("="*60)
    logger.info("Running Inference (mặc định: val-style — 1 clip, no TTA)")
    logger.info("="*60)
    logger.info(f"TTA: {'ON' if use_tta else 'OFF (val-style)'}")
    logger.info(f"  - Num clips per video: {num_clips}")
    if use_tta:
        logger.info(f"  - Num crops: {args.num_crops}")
        logger.info(f"  - Num flips: {args.num_flips}")
    model.eval()
    return_label_as_index = (getattr(args, 'label_format', 'name') == 'index')
    logger.info(f"Label format: {'index (0-50)' if return_label_as_index else 'name (class string)'}")
    predictions = run_inference(
        model=model,
        test_loader=test_loader,
        device=device,
        classes=classes,
        use_tta=use_tta,
        num_crops=args.num_crops,
        num_flips=args.num_flips,
        expected_num_frames=model_config.get('num_frames', 16),
        return_label_as_index=return_label_as_index,
    )
    checkpoint_name = selected_checkpoint.stem
    submission_path = submissions_dir / f'submission_{checkpoint_name}.csv'
    logger.info("="*60)
    logger.info("Generating Submission")
    logger.info("="*60)
    use_template = getattr(args, 'submission_template', False)
    generate_submission(predictions=predictions, output_path=submission_path, logger=logger, template_format=use_template)
    logger.info("="*60)
    logger.info("Inference Completed!")
    logger.info(f"Submission saved to: {submission_path}")
    logger.info("="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Chọn checkpoint và chạy inference tạo submission',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints', help='Checkpoints directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path (nếu không có sẽ hiện menu chọn)')
    parser.add_argument('--data-dir', type=str, default='./kaggle_data/data', help='Data directory')
    parser.add_argument('--submissions-dir', type=str, default='./submissions', help='Submissions directory')
    parser.add_argument('--tta', action='store_true', help='Bật TTA (mặc định: tắt, infer giống tập val)')
    parser.add_argument('--num-crops', type=int, default=10, help='Số crop cho TTA (chỉ khi --tta)')
    parser.add_argument('--num-flips', type=int, default=2, help='Số flip cho TTA (chỉ khi --tta)')
    parser.add_argument('--num-clips', type=int, default=1, help='Số clip mỗi video (mặc định 1 = giống val)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--label-format', type=str, default='name', choices=['name', 'index'],
                        help='Label trong CSV: name = tên class (str), index = 0-50 (int)')
    parser.add_argument('--submission-template', action='store_true',
                        help='Xuất khớp kaggle_data/submission_template.csv: cột id,class; id=0,1,2,... theo thứ tự folder trong test')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    # Paths
    checkpoints_dir = Path(args.checkpoints_dir)
    data_dir = Path(args.data_dir)
    submissions_dir = Path(args.submissions_dir)
    submissions_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Tìm checkpoint trong: {checkpoints_dir}")
    checkpoints = list_checkpoints(checkpoints_dir)
    if len(checkpoints) == 0:
        logger.error(f"Không tìm thấy checkpoint nào trong {checkpoints_dir}")
        return
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    selected_checkpoint = select_checkpoint(checkpoints, checkpoint_path)
    run_inference_core(selected_checkpoint, args, data_dir, submissions_dir)


if __name__ == '__main__':
    main()
