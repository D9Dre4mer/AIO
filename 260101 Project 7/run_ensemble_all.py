"""
Script để chạy ensemble inference với tất cả checkpoints có sẵn.
Đã được cải thiện với các phương pháp ensemble tốt hơn.
"""

import sys
sys.path.insert(0, '.')

from pathlib import Path
import torch
import logging
import argparse
from sota_training.main import main
from sota_training.config import get_default_config, parse_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_ensemble_all():
    """
    Chạy ensemble với tất cả checkpoints có sẵn.
    Sử dụng phương pháp tốt nhất: weighted average of probabilities (softmax trước khi ensemble).
    """
    checkpoint_dir = Path('./checkpoints')
    checkpoint_files = sorted(checkpoint_dir.glob('sota_vit_model_*_best.pt'))
    
    if not checkpoint_files:
        logger.error("Không tìm thấy checkpoint nào!")
        return
    
    logger.info(f"Tìm thấy {len(checkpoint_files)} checkpoints")
    logger.info("Chạy ensemble với tất cả checkpoints...")
    
    # Tạo fake args để trigger ensemble mode
    # Code sẽ tự động detect và ensemble tất cả checkpoints
    sys.argv = ['train.py', '--num-models', str(len(checkpoint_files))]
    
    # Override main để chạy ensemble
    config = get_default_config()
    config['num_models'] = len(checkpoint_files)
    config['model_id'] = 1
    
    # Import và chạy ensemble logic
    from sota_training.dataset import TestDataset
    from sota_training.utils import load_checkpoint
    from sota_training.models import create_model
    from sota_training.inference import run_ensemble_inference, generate_submission
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load all checkpoints
    models = []
    weights = []
    classes = None
    
    for cp_file in checkpoint_files:
        checkpoint = load_checkpoint(cp_file, device, logger)
        model_id = checkpoint.get('model_id', 0)
        val_acc = checkpoint.get('val_acc', 0.0)
        
        if classes is None:
            classes = checkpoint['classes']
        
        # Create model with correct architecture
        checkpoint_config = checkpoint.get('config', {})
        architecture = checkpoint_config.get('architecture', config.get('architecture', 'vit_base'))
        
        model = create_model(
            architecture=architecture,
            num_classes=len(classes),
            pretrained_name=checkpoint_config.get('pretrained_name', config.get('pretrained_name', None)),
            use_adapters=checkpoint_config.get('use_adapters', config['use_adapters']),
            dropout=checkpoint_config.get('dropout', config.get('dropout', 0.1)),
            scales=checkpoint_config.get('multiscale_sizes', config.get('multiscale_sizes', [224, 256, 288])) if architecture == 'multiscale' else None
        ).to(device)
        
        # Load weights
        if 'ema_model' in checkpoint:
            logger.info(f"Loading EMA model weights for Model {model_id}")
            model.load_state_dict(checkpoint['ema_model'])
        else:
            model.load_state_dict(checkpoint['model'])
        
        model.eval()
        models.append(model)
        weights.append(val_acc)
        logger.info(f"Model {model_id} loaded. Val Acc: {val_acc:.4f}")
    
    # Calculate weights: Weighted by validation accuracy (normalized)
    # Note: Val acc weights are used by default. If models have similar performance,
    # uniform weights might work better, but val_acc weights are generally better
    # when there's clear performance difference.
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    logger.info(f"\nEnsemble weights (based on val_acc): {[f'{w:.4f}' for w in weights]}")
    
    # Load test dataset
    logger.info("Loading test dataset...")
    # Convert string to Path if needed
    data_dir = Path(config['data_dir']) if isinstance(config['data_dir'], str) else config['data_dir']
    test_data_dir = data_dir / 'test'
    test_dataset = TestDataset(
        root=test_data_dir,
        num_frames=config['num_frames'],
        frame_stride=config['frame_stride'],
        image_size=config['img_size']
    )
    
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Run ensemble inference with BEST method: weighted average of probabilities
    # This applies softmax before ensemble to ensure all models have same scale (0-1)
    # This is better than ensemble logits directly because logits can have different scales
    predictions = run_ensemble_inference(
        models=models,
        test_loader=test_loader,
        device=device,
        classes=classes,
        weights=weights,
        use_tta=True,
        num_crops=10,
        num_flips=2,
        ensemble_method='weighted_avg_probs',  # Best method: weighted avg of probabilities
        use_softmax=True  # Always use softmax (recommended)
    )
    
    # Generate final submission
    # Convert string to Path if needed
    submissions_dir = Path(config['submissions_dir']) if isinstance(config['submissions_dir'], str) else config['submissions_dir']
    ensemble_submission_path = submissions_dir / 'ensemble_submission.csv'
    generate_submission(predictions, ensemble_submission_path, logger)
    
    logger.info("="*60)
    logger.info("Ensemble inference completed!")
    logger.info(f"Final submission saved to: {ensemble_submission_path}")
    logger.info("="*60)


if __name__ == '__main__':
    run_ensemble_all()
