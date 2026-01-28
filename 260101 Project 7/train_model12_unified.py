"""
Train Model 12 thống nhất: mặc định train một lần (B → G → C), chỉ dùng Improve khi cần.

- Mặc định: chạy train một lần (B → G → C) với các fix đã đưa vào training chính.
- Chỉ dùng --improve khi: sau 1 run thấy 1–2 group/class tụt rõ rệt.

Usage:
  python train_model12_unified.py [train args...]
  python train_model12_unified.py --improve [improve args...]
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Model 12 (B→G→C). Chỉ dùng --improve khi sau 1 run thấy 1–2 group/class tụt.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Train (mặc định) ---
    parser.add_argument('--model-id', type=int, default=12, help='Model ID')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--data-dir', type=str, default='./kaggle_data/data')
    parser.add_argument('--output-dir', type=str, default='./checkpoints')
    parser.add_argument('--logging-dir', type=str, default='./logging')
    parser.add_argument('--pretrained-ckpt', type=str, default=None)
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument(
        '--on-existing',
        type=str,
        choices=['prompt', 'resume', 'skip', 'train', 's', 'r', 't'],
        default='prompt',
        help='Khi checkpoint best đã tồn tại: prompt/s/r/t',
    )
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    # --- Improve (chỉ khi --improve) ---
    parser.add_argument('--improve', action='store_true',
                        help='Chạy Improve weak heads (chỉ khi 1–2 group/class tụt rõ sau 1 run)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint khi --improve (mặc định: output-dir/videomae_model_12_best.pt)')
    parser.add_argument('--output-name', type=str, default=None,
                        help='Tên file checkpoint improved khi --improve')
    parser.add_argument('--group-threshold', type=float, default=0.85)
    parser.add_argument('--improve-bottom-k-groups', type=int, default=0,
                        help='Luôn retrain k group thấp nhất (1–2 khi thấy 1–2 group tụt)')
    parser.add_argument('--class-threshold', type=float, default=0.70)
    parser.add_argument('--retrain-experts', action='store_true')
    parser.add_argument('--retrain-group-head', action='store_true')
    parser.add_argument('--expert-epochs', type=int, default=30)
    parser.add_argument('--group-head-epochs', type=int, default=80)
    parser.add_argument('--expert-lr', type=float, default=2e-4)
    parser.add_argument('--group-head-lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--num-workers', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.improve:
        from improve_weak_heads import main_improve
        if not args.checkpoint:
            args.checkpoint = str(Path(args.output_dir) / f'videomae_model_{args.model_id}_best.pt')
        main_improve(args, logging_dir=Path(args.logging_dir))
    else:
        from train_videomae_model12_group_gated_experts import run_train
        run_train(args)


if __name__ == '__main__':
    main()
