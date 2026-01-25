"""
Train ViT-Base Model với các phương pháp hiệu quả nhất.

Script để train ViT-Base với các SOTA techniques đã nghiên cứu:
- Temporal augmentation hiệu quả (temporal reverse, light frame dropping)
- AutoAugment (15% probability - hiệu quả nhưng không quá chậm)
- Color jitter, random erasing (lightweight, hiệu quả)
- CutMix, Mixup (strong augmentation)
- Focal Loss (handle class imbalance)
- DropPath (Stochastic Depth) - SOTA regularization
- 24 frames (tăng từ 16 để capture temporal tốt hơn)
- Progressive Resize (nếu implement đầy đủ)
- Optimized learning rates và regularization
"""

import subprocess
import sys
from pathlib import Path
import time

# Checkpoint paths
CHECKPOINT_DIR = Path('./checkpoints')
CHECKPOINT_PATTERN = 'sota_vit_model_{}_best.pt'
MODEL_ID = 8  # New model ID for optimized ViT-Base


def check_checkpoint_exists(model_id: int) -> bool:
    """Check if checkpoint exists for model."""
    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_PATTERN.format(model_id)
    return checkpoint_path.exists()


def shutdown_computer(delay_seconds: int = 60):
    """
    Shutdown Windows computer after delay.
    
    Args:
        delay_seconds: Delay in seconds before shutdown (default: 60)
    """
    print(f"\n⏰ Computer will shutdown in {delay_seconds} seconds...")
    
    # Windows shutdown command
    subprocess.run(
        [
            "shutdown", "/s", "/t", str(delay_seconds), "/c",
            f"Training completed. Shutting down in {delay_seconds} seconds."
        ],
        check=False
    )
    print("✓ Shutdown scheduled. To cancel: shutdown /a")


def train_model(seed: int = 42, epochs: int = None, resume: bool = False, auto_shutdown: bool = False, shutdown_delay: int = 60, data_dir: str = None):
    """
    Train ViT-Base with optimized SOTA techniques.
    
    Args:
        seed: Random seed
        epochs: Number of epochs (None = use config default)
        resume: Resume from checkpoint if exists
        auto_shutdown: Automatically shutdown computer after training completes
        shutdown_delay: Delay in seconds before shutdown (default: 60)
        data_dir: Path to data directory (None = use config default: ./kaggle_data/data)
    """
    print("="*60)
    print(f"Training Model {MODEL_ID} (ViT-Base Optimized)")
    print("="*60)
    print(f"Seed: {seed}")
    if epochs:
        print(f"Epochs: {epochs}")
    print("="*60)
    
    # Build command (use -m to run as module to support relative imports)
    cmd = [
        sys.executable, '-m', 'sota_training.main',
        '--model-id', str(MODEL_ID),
        '--seed', str(seed),
        '--output-dir', './checkpoints',
        '--logging-dir', './logging',
        '--submissions-dir', './submissions'
    ]
    
    # Add data-dir if specified, otherwise use config default (./kaggle_data/data)
    if data_dir is not None:
        cmd.extend(['--data-dir', data_dir])
    
    if epochs:
        cmd.extend(['--epochs', str(epochs)])
    
    # Resume from checkpoint if requested
    if resume:
        checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_PATTERN.format(MODEL_ID)
        if checkpoint_path.exists():
            cmd.extend(['--resume', str(checkpoint_path)])
            print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
        else:
            print(f"⚠️  Resume requested but checkpoint not found. Starting new training.")
    
    # Run training
    print(f"\n🚀 Starting training...")
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ Training completed successfully!")
        print(f"⏱️  Total time: {elapsed_time / 3600:.2f} hours")
        
        # Check if checkpoint was created
        if check_checkpoint_exists(MODEL_ID):
            checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_PATTERN.format(MODEL_ID)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
        else:
            print("⚠️  Warning: Checkpoint not found after training")
    else:
        print(f"\n❌ Training failed with exit code {result.returncode}")
        print(f"⏱️  Total time: {elapsed_time / 3600:.2f} hours")
    
    # Auto-shutdown if requested (both success and failure)
    if auto_shutdown:
        shutdown_computer(shutdown_delay)
    
    # Exit with training result code
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train ViT-Base Model với các phương pháp hiệu quả nhất',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for training'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of epochs (None = use config default)'
    )
    parser.add_argument(
        '--skip-if-exists', action='store_true',
        help='Skip training if checkpoint already exists'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume training from checkpoint if exists'
    )
    parser.add_argument(
        '--auto-shutdown', action='store_true',
        help='Automatically shutdown computer after training completes successfully'
    )
    parser.add_argument(
        '--shutdown-delay', type=int, default=60,
        help='Delay in seconds before shutdown (default: 60)'
    )
    parser.add_argument(
        '--data-dir', type=str, default=None,
        help='Path to data directory (default: ./kaggle_data/data from config)'
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if args.skip_if_exists and check_checkpoint_exists(MODEL_ID):
        checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_PATTERN.format(MODEL_ID)
        print(f"✓ Checkpoint already exists: {checkpoint_path}")
        print("Skipping training (use --resume to continue training)")
        return
    
    # Train model
    train_model(
        seed=args.seed, 
        epochs=args.epochs, 
        resume=args.resume,
        auto_shutdown=args.auto_shutdown,
        shutdown_delay=args.shutdown_delay,
        data_dir=args.data_dir
    )


if __name__ == '__main__':
    main()
