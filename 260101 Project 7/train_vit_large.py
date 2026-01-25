"""
Train ViT-Large Model Only.

Script đơn giản để train chỉ Model 1 (ViT-Large) với các SOTA techniques:
- Temporal augmentation mạnh (frame dropping, speed jitter, temporal reverse)
- 32 frames (tăng từ 16)
- DropPath (Stochastic Depth) regularization
"""

import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime

# Checkpoint paths
CHECKPOINT_DIR = Path('./checkpoints')
CHECKPOINT_PATTERN = 'sota_vit_model_1_best.pt'
MODEL_ID = 1


def check_checkpoint_exists() -> bool:
    """Check if checkpoint exists for Model 1."""
    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_PATTERN.format(MODEL_ID)
    return checkpoint_path.exists()


def train_model(seed: int = 42, epochs: int = None):
    """
    Train Model 1 (ViT-Large).
    
    Args:
        seed: Random seed
        epochs: Number of epochs (None = use config default)
    """
    print("="*60)
    print(f"Training Model {MODEL_ID} (ViT-Large)")
    print("="*60)
    print(f"Seed: {seed}")
    if epochs:
        print(f"Epochs: {epochs}")
    print("="*60)
    
    # Build command
    cmd = [
        sys.executable, 'train.py',
        '--model-id', str(MODEL_ID),
        '--seed', str(seed)
    ]
    
    if epochs:
        cmd.extend(['--epochs', str(epochs)])
    
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
        if check_checkpoint_exists():
            checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_PATTERN.format(MODEL_ID)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
        else:
            print("⚠️  Warning: Checkpoint not found after training")
    else:
        print(f"\n❌ Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train ViT-Large Model Only',
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
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if args.skip_if_exists and check_checkpoint_exists():
        checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_PATTERN.format(MODEL_ID)
        print(f"✓ Checkpoint already exists: {checkpoint_path}")
        print("Skipping training (use --resume to continue training)")
        return
    
    # Train model
    train_model(seed=args.seed, epochs=args.epochs)


if __name__ == '__main__':
    main()
