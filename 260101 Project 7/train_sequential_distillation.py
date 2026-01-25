"""
Sequential Knowledge Distillation Training Script.

Tự động train các models theo thứ tự sequential distillation:
Model 2 (Base Teacher) → Model 1 → Model 3 → Model 4 → Model 5

Mỗi model sẽ học từ model trước đã được cải thiện, tạo ra progressive improvement chain.
"""

import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime

# Import config to get teacher dependencies
sys.path.insert(0, '.')
from sota_training.config import get_hyperparameter_variations

# Sequential training order (Model 1 is base teacher, no dependencies)
TRAINING_ORDER = [1, 2, 3, 4, 5]

# Checkpoint paths
CHECKPOINT_DIR = Path('./checkpoints')
CHECKPOINT_PATTERN = 'sota_vit_model_{}_best.pt'


def check_checkpoint_exists(model_id: int) -> bool:
    """Check if checkpoint exists for a model."""
    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_PATTERN.format(model_id)
    return checkpoint_path.exists()


def wait_for_checkpoint(model_id: int, timeout: int = 3600, check_interval: int = 60) -> bool:
    """
    Wait for checkpoint to be created.
    
    Args:
        model_id: Model ID to wait for
        timeout: Maximum time to wait in seconds (default: 1 hour)
        check_interval: Interval between checks in seconds (default: 60 seconds)
    
    Returns:
        True if checkpoint exists, False if timeout
    """
    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_PATTERN.format(model_id)
    start_time = time.time()
    
    print(f"⏳ Waiting for checkpoint: {checkpoint_path}")
    print(f"   Timeout: {timeout // 60} minutes, Check interval: {check_interval} seconds")
    
    while not checkpoint_path.exists():
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"❌ Timeout: Checkpoint not found after {timeout // 60} minutes")
            return False
        
        # Print progress every check_interval
        if int(elapsed) % check_interval == 0:
            print(f"   Still waiting... ({int(elapsed // 60)} minutes elapsed)")
        
        time.sleep(1)
    
    elapsed = time.time() - start_time
    print(f"✅ Checkpoint found! (waited {int(elapsed // 60)} minutes)")
    return True


def train_model(model_id: int, seed: int = 42, resume: bool = False) -> bool:
    """
    Train a single model.
    
    Args:
        model_id: Model ID to train
        seed: Random seed
        resume: Whether to resume from checkpoint
    
    Returns:
        True if training successful, False otherwise
    """
    print("\n" + "="*80)
    if resume:
        print(f"🔄 Resuming Training Model {model_id}")
    else:
        print(f"🚀 Training Model {model_id}")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Build command
    cmd = [
        sys.executable,
        'train.py',
        '--model-id', str(model_id),
        '--seed', str(seed),
        '--epochs', '100'  # Train tất cả models đến 100 epochs
    ]
    
    # Add resume flag if needed
    if resume:
        checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_PATTERN.format(model_id)
        cmd.extend(['--resume', str(checkpoint_path)])
        print(f"Resume from: {checkpoint_path}")
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        if resume:
            print(f"\n✅ Model {model_id} training resumed and completed successfully!")
        else:
            print(f"\n✅ Model {model_id} training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Model {model_id} training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Model {model_id} training interrupted by user")
        return False


def main():
    """Main function to run sequential distillation training."""
    print("="*80)
    print("Sequential Knowledge Distillation Training")
    print("="*80)
    print("\nTraining Order:")
    print("  Model 1 (Base Teacher) → Model 2 → Model 3 → Model 4 → Model 5")
    print("\nKnowledge Chain:")
    print("  Model 1 (ViT-Large, no distillation)")
    print("    ↓")
    print("  Model 2 (learns from Model 1)")
    print("    ↓")
    print("  Model 3 (learns from Model 2 → Model 1)")
    print("    ↓")
    print("  Model 4 (learns from Model 3 → Model 2 → Model 1)")
    print("    ↓")
    print("  Model 5 (learns from Model 4 → Model 3 → Model 2 → Model 1)")
    print("\n" + "="*80)
    
    # Check checkpoint directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {CHECKPOINT_DIR.absolute()}")
    
    # Track training results
    results = {}
    start_time = time.time()
    
    # Sequential training
    for idx, model_id in enumerate(TRAINING_ORDER):
        print(f"\n{'='*80}")
        print(f"Step {idx + 1}/{len(TRAINING_ORDER)}: Model {model_id}")
        print(f"{'='*80}")
        
        # Check if checkpoint already exists (skip/resume/retrain options)
        should_resume = False
        if check_checkpoint_exists(model_id):
            checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_PATTERN.format(model_id)
            print(f"⚠️  Checkpoint already exists for Model {model_id}: {checkpoint_path}")
            print(f"\n   Options:")
            print(f"     [s] Skip - Skip training this model")
            print(f"     [r] Resume - Resume training from checkpoint")
            print(f"     [t] Train - Retrain from scratch (will overwrite checkpoint)")
            
            while True:
                response = input(f"\n   Choose option for Model {model_id} (s/r/t): ").strip().lower()
                if response == 's':
                    print(f"⏭️  Skipping Model {model_id}")
                    results[model_id] = True
                    should_resume = False
                    break
                elif response == 'r':
                    print(f"🔄 Will resume training Model {model_id} from checkpoint")
                    should_resume = True
                    break
                elif response == 't':
                    print(f"🔄 Will retrain Model {model_id} from scratch")
                    should_resume = False
                    break
                else:
                    print(f"   Invalid option. Please enter 's' (skip), 'r' (resume), or 't' (train)")
            
            # If skip, continue to next model
            if response == 's':
                continue
        
        # Check teacher dependencies from config (for sequential distillation)
        variations = get_hyperparameter_variations()
        if model_id in variations:
            model_config = variations[model_id]
            use_distillation = model_config.get('use_distillation', False)
            
            if use_distillation:
                teacher_checkpoints = model_config.get('teacher_checkpoints', [])
                if teacher_checkpoints:
                    # Extract teacher model ID from checkpoint path
                    # Format: 'checkpoints/sota_vit_model_{id}_best.pt'
                    teacher_checkpoint_path = Path(teacher_checkpoints[0])
                    
                    # Validate teacher checkpoint exists
                    if not teacher_checkpoint_path.exists():
                        print(f"❌ Teacher checkpoint not found: {teacher_checkpoint_path}")
                        print(f"   Model {model_id} requires teacher from: {teacher_checkpoint_path}")
                        print(f"   Please train the teacher model first!")
                        print(f"   Expected training order: Model 1 → Model 2 → Model 3 → Model 4 → Model 5")
                        results[model_id] = False
                        break
                    else:
                        # Extract teacher model ID for display
                        teacher_id_str = teacher_checkpoint_path.stem.replace('sota_vit_model_', '').replace('_best', '')
                        try:
                            teacher_id = int(teacher_id_str)
                            print(f"✅ Teacher checkpoint found: {teacher_checkpoint_path}")
                            print(f"   Model {model_id} will learn from Model {teacher_id}")
                        except ValueError:
                            print(f"✅ Teacher checkpoint found: {teacher_checkpoint_path}")
                else:
                    print(f"⚠️  Model {model_id} has use_distillation=True but teacher_checkpoints is empty")
                    print(f"   Will train without distillation")
            else:
                # Model 1 is base teacher, no distillation needed
                if model_id == 1:
                    print(f"✅ Model {model_id} is base teacher (ViT-Large, no distillation needed)")
                else:
                    print(f"ℹ️  Model {model_id} does not use distillation")
        
        # Train model (with resume flag if needed)
        success = train_model(model_id, seed=42, resume=should_resume)
        results[model_id] = success
        
        if not success:
            print(f"\n❌ Training failed for Model {model_id}")
            print("   Stopping sequential training")
            break
        
        # Verify checkpoint was created
        if not check_checkpoint_exists(model_id):
            print(f"⚠️  Warning: Checkpoint not found after training Model {model_id}")
            print(f"   Waiting for checkpoint to be created...")
            if not wait_for_checkpoint(model_id, timeout=300):  # Wait 5 minutes
                print(f"❌ Checkpoint not created for Model {model_id}")
                results[model_id] = False
                break
        
        print(f"\n✅ Step {idx + 1} completed: Model {model_id}")
    
    # Summary
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    
    print("\n" + "="*80)
    print("Training Summary")
    print("="*80)
    print(f"Total time: {hours}h {minutes}m")
    print("\nResults:")
    for model_id in TRAINING_ORDER:
        status = "✅ Success" if results.get(model_id, False) else "❌ Failed"
        checkpoint_status = "✅ Exists" if check_checkpoint_exists(model_id) else "❌ Missing"
        print(f"  Model {model_id}: {status} | Checkpoint: {checkpoint_status}")
    
    # Check if all models trained successfully
    all_success = all(results.get(model_id, False) for model_id in TRAINING_ORDER)
    
    if all_success:
        print("\n🎉 All models trained successfully!")
        print("\nNext steps:")
        print("  1. Run ensemble inference:")
        print("     python run_ensemble_all.py")
        print("  2. Check submissions in: ./submissions/")
    else:
        print("\n⚠️  Some models failed. Please check logs and retrain failed models.")
    
    print("="*80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
