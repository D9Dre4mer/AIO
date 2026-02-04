"""
Inference chỉ cho Model Alpha (VideoMAE Alpha - Experts only, no group head).

- Chỉ liệt kê checkpoint có tên chứa "alpha".
- Mặc định: single best expert per sample (expert có confidence cao nhất).
- Mặc định: 1 clip/video, không TTA (giống val).
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

ALPHA_ARCH = 'videomae_alpha_experts'


def _is_alpha_filename(name: str) -> bool:
    return 'alpha' in name.lower()


def list_alpha_checkpoints(checkpoints_dir: Path) -> list:
    """Chỉ lấy checkpoint Alpha (tên file chứa 'alpha')."""
    all_ckpts = list_checkpoints(checkpoints_dir)
    return [p for p in all_ckpts if _is_alpha_filename(p.name)]


def main():
    parser = argparse.ArgumentParser(
        description='Inference chỉ cho Model Alpha (Experts only)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--checkpoints-dir', type=str, default='./checkpoints',
        help='Thư mục checkpoints',
    )
    parser.add_argument('--data-dir', type=str, default='./kaggle_data/data')
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Checkpoint (ưu tiên: videomae_model_alpha_best.pt)',
    )
    parser.add_argument('--submissions-dir', type=str, default='./submissions')
    parser.add_argument(
        '--merge-experts', action='store_true',
        help='Merge logits tất cả expert (mặc định: single best per sample)',
    )
    parser.add_argument('--tta', action='store_true', help='Bật TTA')
    parser.add_argument('--num-crops', type=int, default=10)
    parser.add_argument('--num-flips', type=int, default=2)
    parser.add_argument('--num-clips', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument(
        '--label-format', type=str, default='name', choices=['name', 'index'],
    )
    parser.add_argument('--submission-template', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    checkpoints_dir = Path(args.checkpoints_dir)
    data_dir = Path(args.data_dir)
    submissions_dir = Path(args.submissions_dir)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = list_alpha_checkpoints(checkpoints_dir)
    if len(checkpoints) == 0:
        logger.error(
            "Không tìm thấy checkpoint Alpha trong %s. Tên file cần chứa 'alpha'.",
            checkpoints_dir,
        )
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path is not None and checkpoint_path.exists():
        not_alpha_name = not _is_alpha_filename(checkpoint_path.name)
        if checkpoint_path not in checkpoints and not_alpha_name:
            logger.warning(
                "File %s không chứa 'alpha'; vẫn thử nếu architecture khớp.",
                checkpoint_path.name,
            )
        checkpoints = [checkpoint_path] + [p for p in checkpoints if p != checkpoint_path]

    selected_checkpoint = select_checkpoint(checkpoints, checkpoint_path)
    model_config = get_model_config_from_checkpoint(selected_checkpoint)
    arch = model_config.get('architecture')
    if arch != ALPHA_ARCH:
        logger.error(
            "Checkpoint không phải Model Alpha (architecture=%s). "
            "Chỉ hỗ trợ videomae_alpha_experts.", arch,
        )
        sys.exit(1)

    if args.merge_experts:
        model_config['inference_single_best_expert'] = False
        logger.info("Chế độ inference: merge logits tất cả expert")
    else:
        model_config['inference_single_best_expert'] = True
        logger.info("Chế độ inference: single best expert per sample (mặc định)")

    logger.info("Đã xác nhận Model Alpha. Chạy inference...")
    run_inference_core(selected_checkpoint, args, data_dir, submissions_dir)


if __name__ == '__main__':
    main()
