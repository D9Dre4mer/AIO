"""
Script tự động kiểm tra và train tiếp các model có tiềm năng cải thiện.
"""

import sys
sys.path.insert(0, '.')

import torch
from pathlib import Path
import logging
from typing import Dict, List, Optional
import subprocess

from sota_training.utils import load_checkpoint

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def analyze_model_potential(checkpoint_path: Path, device: torch.device) -> Optional[Dict]:
    """
    Phân tích tiềm năng cải thiện của một model.
    
    Note: Dùng CPU device để load checkpoint vì chỉ cần metadata,
    không cần model weights → tránh load weights vào GPU không cần thiết.
    
    Returns:
        Dict với các metrics và đánh giá
    """
    try:
        # Load checkpoint với CPU để tránh load weights vào GPU
        # Chỉ cần metadata: epoch, val_acc, train_acc, history
        checkpoint = load_checkpoint(checkpoint_path, device, logger)
        
        if checkpoint is None:
            logger.error(f"Failed to load checkpoint: {checkpoint_path}")
            return None
        
        epoch = checkpoint.get('epoch', 0)
        val_acc = checkpoint.get('val_acc', 0.0)
        train_acc = checkpoint.get('train_acc', 0.0)
        history = checkpoint.get('history', {})
        
        # Validate data
        if epoch < 0:
            logger.warning(f"Invalid epoch in checkpoint: {epoch}")
            return None
        
        # Tính train/val gap
        gap = train_acc - val_acc if train_acc > 0 else 0
        
        # Phân tích trend (val acc có tăng trong 5 epochs gần nhất không)
        val_history = history.get('val_acc', [])
        trend_score = 0
        if len(val_history) >= 5:
            recent_vals = val_history[-5:]
            if recent_vals[-1] > recent_vals[0]:
                trend_score = 1  # Đang tăng
            elif recent_vals[-1] < recent_vals[0]:
                trend_score = -1  # Đang giảm
        
        # Đánh giá potential
        # Logic mới: train thêm 50 epochs từ epoch hiện tại
        additional_epochs = 50  # Số epochs sẽ train thêm
        epochs_remaining = additional_epochs  # Luôn có thể train thêm 50 epochs
        
        # Potential score (0-100)
        potential_score = 0
        
        # 1. Val acc chưa cao (< 70%) → có room để cải thiện
        if val_acc < 0.70:
            potential_score += 30
        elif val_acc < 0.65:
            potential_score += 50
        
        # 2. Gap không quá lớn (< 30%) → chưa overfit nặng
        if gap < 0.30:
            potential_score += 20
        elif gap < 0.25:
            potential_score += 30
        
        # 3. Còn epochs để train (luôn có 50 epochs để train thêm)
        potential_score += 20  # Luôn có 50 epochs để train thêm
        
        # 4. Trend đang tăng hoặc ổn định
        if trend_score >= 0:
            potential_score += 10
        
        # 5. Val acc > 60% → model tốt, đáng train tiếp
        if val_acc > 0.60:
            potential_score += 10
        
        # Đánh giá tổng thể
        if potential_score >= 70:
            should_continue = True
            reason = "High potential: val acc chưa cao, gap nhỏ, còn epochs"
        elif potential_score >= 50:
            should_continue = True
            reason = "Medium potential: có thể cải thiện"
        else:
            should_continue = False
            reason = "Low potential: đã tốt hoặc overfit"
        
        return {
            'checkpoint_path': checkpoint_path,
            'epoch': epoch,
            'val_acc': val_acc,
            'train_acc': train_acc,
            'gap': gap,
            'epochs_remaining': epochs_remaining,  # Luôn là 50 (sẽ train thêm 50 epochs)
            'additional_epochs': additional_epochs,  # Số epochs sẽ train thêm
            'trend_score': trend_score,
            'potential_score': potential_score,
            'should_continue': should_continue,
            'reason': reason,
            'history': history
        }
    except Exception as e:
        logger.error(f"Error analyzing {checkpoint_path}: {e}")
        return None


def find_models_to_continue(checkpoint_dir: Path = Path('./checkpoints')) -> List[Dict]:
    """
    Tìm các model có tiềm năng train tiếp.
    """
    # Dùng CPU để load checkpoint vì chỉ cần metadata, không cần model weights
    # Tránh load weights vào GPU không cần thiết
    device = torch.device('cpu')
    
    checkpoint_files = sorted(checkpoint_dir.glob('sota_vit_model_*_best.pt'))
    
    if not checkpoint_files:
        logger.warning("Không tìm thấy checkpoint nào!")
        return []
    
    logger.info(f"Phân tích {len(checkpoint_files)} checkpoints...")
    
    results = []
    for cp in checkpoint_files:
        analysis = analyze_model_potential(cp, device)
        if analysis:
            results.append(analysis)
    
    # Sort by potential score (descending)
    results.sort(key=lambda x: x['potential_score'], reverse=True)
    
    return results


def print_analysis(results: List[Dict]):
    """In kết quả phân tích."""
    print("\n" + "="*70)
    print("PHÂN TÍCH: MODELS CÓ TIỀM NĂNG TRAIN TIẾP")
    print("="*70)
    
    # Logic mới: train thêm 50 epochs từ epoch hiện tại
    additional_epochs = 50
    
    print(f"\n{'Model':<10} {'Epoch':<8} {'Val Acc':<10} {'Gap':<8} {'Epochs Add':<12} {'New Max':<10} {'Potential':<10} {'Continue':<10}")
    print("-" * 90)
    
    for r in results:
        model_id = r['checkpoint_path'].stem.replace('sota_vit_model_', '').replace('_best', '')
        continue_str = "✅ YES" if r['should_continue'] else "❌ NO"
        new_max = r['epoch'] + additional_epochs
        print(f"{model_id:<10} {r['epoch']:<8} {r['val_acc']:<10.4f} {r['gap']:<8.4f} {additional_epochs:<12} {new_max:<10} {r['potential_score']:<10.0f} {continue_str:<10}")
    
    print("\n" + "="*70)
    print("CHI TIẾT:")
    print("="*70)
    
    # Logic mới: train thêm 50 epochs từ epoch hiện tại
    additional_epochs = 50
    
    for r in results:
        model_id = r['checkpoint_path'].stem.replace('sota_vit_model_', '').replace('_best', '')
        new_max = r['epoch'] + additional_epochs
        print(f"\n📁 Model {model_id}:")
        print(f"   Current epoch: {r['epoch']}")
        print(f"   Sẽ train thêm: {additional_epochs} epochs")
        print(f"   New max epochs: {new_max}")
        print(f"   Training range: epoch {r['epoch'] + 1} → {new_max}")
        print(f"   Val Acc: {r['val_acc']:.4f}")
        print(f"   Train Acc: {r['train_acc']:.4f}")
        print(f"   Gap: {r['gap']:.4f}")
        print(f"   Potential score: {r['potential_score']}/100")
        print(f"   Should continue: {r['should_continue']}")
        print(f"   Reason: {r['reason']}")


def train_continue_model(model_id: int, checkpoint_path: Path, current_epoch: int, additional_epochs: int = 50) -> bool:
    """
    Train tiếp một model từ checkpoint.
    
    Args:
        model_id: Model ID
        checkpoint_path: Path to checkpoint
        current_epoch: Current epoch (sẽ train từ epoch này + 1)
        additional_epochs: Số epochs sẽ train thêm (default: 50)
    
    Returns:
        True nếu thành công
    """
    new_max_epochs = current_epoch + additional_epochs
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Training tiếp Model {model_id}")
    logger.info(f"  - Current epoch: {current_epoch}")
    logger.info(f"  - Sẽ train thêm: {additional_epochs} epochs")
    logger.info(f"  - New max epochs: {new_max_epochs}")
    logger.info(f"  - Training range: epoch {current_epoch + 1} → {new_max_epochs}")
    logger.info(f"{'='*70}")
    
    try:
        # Chạy train.py với resume
        # Code sẽ tự động resume từ checkpoint và train đến new_max_epochs
        cmd = [
            sys.executable,
            'train.py',
            '--resume', str(checkpoint_path),
            '--model-id', str(model_id),
            '--epochs', str(new_max_epochs)
        ]
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        if result.returncode == 0:
            logger.info(f"✅ Model {model_id} training completed!")
            return True
        else:
            logger.error(f"❌ Model {model_id} training failed!")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error training Model {model_id}: {e}")
        return False
    except KeyboardInterrupt:
        logger.warning(f"⚠️  Model {model_id} training interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error training Model {model_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_ensemble(checkpoint_dir: Path = Path('./checkpoints')):
    """
    Chạy ensemble inference sau khi train xong.
    """
    logger.info(f"\n{'='*70}")
    logger.info("Running Ensemble Inference...")
    logger.info(f"{'='*70}")
    
    try:
        # Chạy train.py với inference-only
        # Code sẽ tự động detect tất cả checkpoints và ensemble
        cmd = [
            sys.executable,
            'train.py',
            '--inference-only'
        ]
        
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info("Code sẽ tự động tìm tất cả checkpoints và ensemble")
        
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        if result.returncode == 0:
            logger.info("✅ Ensemble inference completed!")
            return True
        else:
            logger.error("❌ Ensemble inference failed!")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error running ensemble: {e}")
        return False
    except KeyboardInterrupt:
        logger.warning(f"⚠️  Ensemble inference interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error running ensemble: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main function."""
    print("="*70)
    print("AUTO CONTINUE TRAINING SCRIPT")
    print("="*70)
    
    # 1. Tìm models có tiềm năng
    print("\n1. Đang phân tích các models...")
    results = find_models_to_continue()
    
    if not results:
        logger.error("Không tìm thấy model nào để phân tích!")
        return
    
    # 2. In kết quả phân tích
    print_analysis(results)
    
    # 3. Lọc models cần train tiếp
    models_to_continue = [r for r in results if r['should_continue']]
    
    if not models_to_continue:
        print("\n✅ Không có model nào cần train tiếp!")
        print("   Tất cả models đã đạt potential tối đa hoặc overfit.")
        return
    
    # Logic mới: train thêm 50 epochs từ epoch hiện tại
    additional_epochs = 50
    
    print(f"\n{'='*70}")
    print(f"Tìm thấy {len(models_to_continue)} model(s) cần train tiếp:")
    print("="*70)
    for r in models_to_continue:
        model_id = r['checkpoint_path'].stem.replace('sota_vit_model_', '').replace('_best', '')
        new_max = r['epoch'] + additional_epochs
        print(f"  - Model {model_id}: {r['reason']} (Potential: {r['potential_score']}/100)")
        print(f"    → Sẽ train thêm {additional_epochs} epochs (từ epoch {r['epoch'] + 1} → {new_max})")
    
    # 4. Hỏi xác nhận
    # Logic mới: train thêm 50 epochs từ epoch hiện tại
    additional_epochs = 50
    
    # Tính tổng số epochs sẽ train (mỗi model train thêm 50 epochs)
    total_epochs_to_train = len(models_to_continue) * additional_epochs
    
    print(f"\n{'='*70}")
    print(f"Tổng kết:")
    print(f"  - Tổng số models: {len(results)}")
    print(f"  - Models cần train tiếp: {len(models_to_continue)}")
    print(f"  - Models đã tốt: {len(results) - len(models_to_continue)}")
    print(f"  - Số epochs sẽ train thêm cho mỗi model: {additional_epochs}")
    print(f"  - Tổng số epochs sẽ train: {total_epochs_to_train} (mỗi model {additional_epochs} epochs)")
    print("="*70)
    
    response = input(f"\nBạn có muốn train tiếp {len(models_to_continue)} model(s) này không? (y/n): ")
    
    if response.lower() != 'y':
        print("Đã hủy.")
        return
    
    # 5. Train tiếp từng model
    print(f"\n{'='*70}")
    print("Bắt đầu training tiếp...")
    print("="*70)
    
    # Logic mới: train thêm 50 epochs từ epoch hiện tại
    additional_epochs = 50
    
    success_count = 0
    for r in models_to_continue:
        try:
            model_id_str = r['checkpoint_path'].stem.replace('sota_vit_model_', '').replace('_best', '')
            model_id = int(model_id_str)
            checkpoint_path = r['checkpoint_path']
            current_epoch = r['epoch']
            
            # Train thêm 50 epochs từ epoch hiện tại
            if train_continue_model(model_id, checkpoint_path, current_epoch, additional_epochs):
                success_count += 1
            else:
                logger.warning(f"Model {model_id} training failed, skipping...")
        except (ValueError, KeyError) as e:
            logger.error(f"Error processing model: {e}")
            logger.error(f"Checkpoint path: {r.get('checkpoint_path', 'unknown')}")
            continue
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            break
    
    print(f"\n{'='*70}")
    print(f"Training completed: {success_count}/{len(models_to_continue)} models thành công")
    print("="*70)
    
    # 6. Chạy ensemble
    if success_count > 0:
        print(f"\n{'='*70}")
        print("Training tiếp đã hoàn tất!")
        print("="*70)
        print("\nBước tiếp theo: Chạy ensemble inference với tất cả models")
        print("(bao gồm cả models vừa train tiếp và models đã tốt)")
        print("\nBạn có thể chạy ensemble bằng lệnh:")
        print("  python run_ensemble_all.py")
        
        response = input("\nBạn có muốn chạy ensemble inference ngay không? (y/n): ")
        
        if response.lower() == 'y':
            # Chạy ensemble với tất cả checkpoints
            print("\nĐang chạy ensemble inference...")
            try:
                from run_ensemble_all import run_ensemble_all
                run_ensemble_all()
            except ImportError:
                # Fallback: chạy bằng subprocess
                cmd = [sys.executable, 'run_ensemble_all.py']
                result = subprocess.run(cmd, check=True)
                if result.returncode == 0:
                    logger.info("✅ Ensemble completed!")
                else:
                    logger.error("❌ Ensemble failed!")
            except Exception as e:
                logger.error(f"Error running ensemble: {e}")
                print("Vui lòng chạy ensemble manual:")
                print("  python run_ensemble_all.py")
    else:
        print("\n⚠️ Không có model nào train thành công, bỏ qua ensemble.")
    
    print("\n✅ Hoàn tất!")


if __name__ == '__main__':
    main()
