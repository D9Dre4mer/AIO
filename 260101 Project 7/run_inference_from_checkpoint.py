"""
Script để chọn checkpoint và chạy inference tạo kết quả submission.
Cho phép user chọn 1 checkpoint từ danh sách có sẵn.
"""

import os
import torch
from pathlib import Path
import logging
import argparse
from typing import Optional

# Fix OpenMP duplicate library error on Windows
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# GPU optimization settings
cuda_launch_blocking = os.environ.get('CUDA_LAUNCH_BLOCKING', '0')
os.environ['CUDA_LAUNCH_BLOCKING'] = cuda_launch_blocking
torch.backends.cudnn.benchmark = True

from sota_training.config import get_default_config
from sota_training.utils import setup_logging, log_system_info, load_checkpoint
from sota_training.dataset import VideoDataset, TestDataset
from sota_training.models import create_model
from sota_training.inference import run_inference, generate_submission

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
    """Lấy config từ checkpoint hoặc suy luận từ tên file."""
    config = get_default_config()
    
    # Thử load checkpoint để lấy config
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            # Merge với default config
            for key, value in saved_config.items():
                if key in config:
                    config[key] = value
    except Exception as e:
        logger.warning(f"Không thể load config từ checkpoint: {e}")
    
    # Suy luận architecture từ tên file
    filename = checkpoint_path.name.lower()
    if 'videomae' in filename:
        config['architecture'] = 'videomae'
        config['model_name'] = 'MCG-NJU/videomae-large-finetuned-kinetics'
    elif 'timesformer' in filename:
        config['architecture'] = 'timesformer'
    elif 'vit' in filename:
        config['architecture'] = 'vit'
    elif 'swin' in filename:
        config['architecture'] = 'swin'
    
    # Default VideoMAE parameters
    if config['architecture'] == 'videomae':
        config['tubelet_size'] = 2
        config['patch_size'] = 16
        config['num_frames'] = 16
        config['frame_stride'] = 2
        config['img_size'] = 224
    
    return config


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
    parser.add_argument('--use-tta', action='store_true', default=True, help='Use Test-Time Augmentation')
    parser.add_argument('--num-crops', type=int, default=10, help='Number of crops for TTA (standard: 10)')
    parser.add_argument('--num-flips', type=int, default=2, help='Number of flips for TTA (standard: 1-2)')
    parser.add_argument('--num-clips', type=int, default=10, help='Number of clips per video (standard: 10)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loader workers')
    
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
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    log_system_info(logger, device)
    
    # List checkpoints
    logger.info(f"Tìm checkpoint trong: {checkpoints_dir}")
    checkpoints = list_checkpoints(checkpoints_dir)
    
    if len(checkpoints) == 0:
        logger.error(f"Không tìm thấy checkpoint nào trong {checkpoints_dir}")
        return
    
    # Select checkpoint
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    
    selected_checkpoint = select_checkpoint(checkpoints, checkpoint_path)
    
    # Load checkpoint và lấy config
    logger.info(f"Loading checkpoint: {selected_checkpoint}")
    checkpoint = load_checkpoint(selected_checkpoint, device, logger)
    
    # Get model config
    model_config = get_model_config_from_checkpoint(selected_checkpoint)
    
    # Cần số classes - load từ train dataset hoặc từ checkpoint
    train_data_dir = data_dir / 'data_train'
    if train_data_dir.exists():
        train_dataset = VideoDataset(
            root=train_data_dir,
            num_frames=model_config.get('num_frames', 16),
            frame_stride=model_config.get('frame_stride', 2),
            image_size=model_config.get('img_size', 224),
            is_train=True,
            val_ratio=0.0,  # Không cần split
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
        # Default: HMDB51 có 51 classes
        logger.warning("Không tìm thấy số classes, dùng default 51")
        num_classes = 51
        classes = [f"class_{i}" for i in range(num_classes)]
    
    # Create model
    logger.info(f"Creating {model_config['architecture']} model...")
    model = create_model(
        architecture=model_config['architecture'],
        num_classes=num_classes,
        pretrained_name=model_config.get('pretrained_name'),
        use_adapters=model_config.get('use_adapters', False),
        dropout=model_config.get('dropout', 0.1),
        drop_path_rate=model_config.get('drop_path_rate', 0.0),
        pretrained_ckpt=model_config.get('pretrained_ckpt'),
        model_name=model_config.get('model_name'),
        num_frames=model_config.get('num_frames', 16),
        tubelet_size=model_config.get('tubelet_size', 2),
        image_size=model_config.get('img_size', 224),
        patch_size=model_config.get('patch_size', 16)
    ).to(device)
    
    # Load model weights
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        logger.info("Model weights loaded from checkpoint")
    else:
        raise ValueError("Checkpoint không chứa 'model' state dict!")
    
    model.eval()
    
    # Load test dataset
    test_data_dir = data_dir / 'test'
    if not test_data_dir.exists():
        logger.error(f"Test directory không tồn tại: {test_data_dir}")
        return
    
    logger.info("Loading test dataset...")
    # Standard evaluation: 10 clips per video
    num_clips = args.num_clips if hasattr(args, 'num_clips') else 10
    test_dataset = TestDataset(
        root=test_data_dir,
        num_frames=model_config.get('num_frames', 16),
        frame_stride=model_config.get('frame_stride', 2),
        image_size=model_config.get('img_size', 224),
        num_clips=num_clips  # Standard: 10 clips per video for evaluation
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Run inference
    logger.info("="*60)
    logger.info("Running Inference")
    logger.info("="*60)
    logger.info(f"Using TTA: {args.use_tta}")
    logger.info(f"  - Num clips per video: {num_clips}")
    if args.use_tta:
        logger.info(f"  - Num crops: {args.num_crops}")
        logger.info(f"  - Num flips: {args.num_flips}")
    
    # Set model to eval mode
    model.eval()
    
    predictions = run_inference(
        model=model,
        test_loader=test_loader,
        device=device,
        classes=classes,
        use_tta=args.use_tta,
        num_crops=args.num_crops,
        num_flips=args.num_flips,
        expected_num_frames=model_config.get('num_frames', 16)
    )
    
    # Generate submission
    checkpoint_name = selected_checkpoint.stem
    submission_path = submissions_dir / f'submission_{checkpoint_name}.csv'
    
    logger.info("="*60)
    logger.info("Generating Submission")
    logger.info("="*60)
    
    generate_submission(
        predictions=predictions,
        output_path=submission_path,
        logger=logger
    )
    
    logger.info("="*60)
    logger.info("Inference Completed!")
    logger.info(f"Submission saved to: {submission_path}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
