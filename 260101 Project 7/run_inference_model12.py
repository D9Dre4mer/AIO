"""
Script chỉ chạy inference cho Model 12, Improved Heads và Model Alpha.
Liệt kê checkpoint có tên: model_12, model12, improved_heads, model_13, model13, alpha.
Mặc định: 1 clip/video, không TTA (giống val).
"""

import os
import sys

if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from pathlib import Path
import logging
import argparse

cuda_launch_blocking = os.environ.get('CUDA_LAUNCH_BLOCKING', '0')
os.environ['CUDA_LAUNCH_BLOCKING'] = cuda_launch_blocking
torch.backends.cudnn.benchmark = True

from run_inference_from_checkpoint import (  # noqa: E402
    list_checkpoints,
    select_checkpoint,
    get_model_config_from_checkpoint,
    run_inference_core,
)

logger = logging.getLogger(__name__)

MODEL12_ARCH = 'videomae_group_gated_experts'
ALPHA_ARCH = 'videomae_alpha_experts'


def _is_model12_or_alpha_filename(name: str) -> bool:
    n = name.lower()
    return (
        'model_12' in n or 'model12' in n or 'improved_heads' in n
        or 'model_13' in n or 'model13' in n or 'alpha' in n
    )


def list_model12_checkpoints(checkpoints_dir: Path) -> list:
    """Chỉ lấy các checkpoint Model 12 / Improved / Alpha (theo tên file)."""
    all_ckpts = list_checkpoints(checkpoints_dir)
    return [p for p in all_ckpts if _is_model12_or_alpha_filename(p.name)]


def main():
    parser = argparse.ArgumentParser(
        description='Inference chỉ cho Model 12 và Improved Heads',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints', help='Thư mục checkpoints')
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Checkpoint (ưu tiên: videomae_model_12_improved_heads.pt hoặc _best.pt)'
    )
    parser.add_argument('--data-dir', type=str, default='./kaggle_data/data', help='Thư mục data')
    parser.add_argument('--submissions-dir', type=str, default='./submissions', help='Thư mục ghi submission')
    parser.add_argument('--tta', action='store_true', help='Bật TTA (mặc định: tắt, infer giống val)')
    parser.add_argument('--num-crops', type=int, default=10, help='Số crop TTA (khi --tta)')
    parser.add_argument('--num-flips', type=int, default=2, help='Số flip TTA (khi --tta)')
    parser.add_argument('--num-clips', type=int, default=1, help='Số clip/video (mặc định 1 = giống val)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader num_workers')
    parser.add_argument('--label-format', type=str, default='name', choices=['name', 'index'])
    parser.add_argument(
        '--submission-template', action='store_true',
        help='Xuất cột id,class (theo template Kaggle)'
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    checkpoints_dir = Path(args.checkpoints_dir)
    data_dir = Path(args.data_dir)
    submissions_dir = Path(args.submissions_dir)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = list_model12_checkpoints(checkpoints_dir)
    if len(checkpoints) == 0:
        logger.error(
            "Không tìm thấy checkpoint Model 12 / Improved / Alpha trong %s. "
            "Tên file cần chứa: model_12, model12, improved_heads, model_13, model13 hoặc alpha.",
            checkpoints_dir,
        )
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path is not None and checkpoint_path.exists() and checkpoint_path not in checkpoints:
        if not _is_model12_or_alpha_filename(checkpoint_path.name):
            logger.warning(
                "File %s không match model_12/improved/alpha theo tên; vẫn thử dùng nếu architecture khớp.",
                checkpoint_path.name,
            )
        checkpoints = [checkpoint_path] + [p for p in checkpoints if p != checkpoint_path]

    selected_checkpoint = select_checkpoint(checkpoints, checkpoint_path)
    model_config = get_model_config_from_checkpoint(selected_checkpoint)
    arch = model_config.get('architecture')
    if arch not in (MODEL12_ARCH, ALPHA_ARCH):
        logger.error(
            "Checkpoint không phải Model 12 hoặc Alpha (architecture=%s). Chỉ hỗ trợ videomae_group_gated_experts, videomae_alpha_experts.",
            arch,
        )
        sys.exit(1)
    logger.info("Đã xác nhận %s. Chạy inference...", "Model 12 (Group-Gated Experts)" if arch == MODEL12_ARCH else "Model Alpha (Experts only)")
    run_inference_core(selected_checkpoint, args, data_dir, submissions_dir)


if __name__ == '__main__':
    main()
