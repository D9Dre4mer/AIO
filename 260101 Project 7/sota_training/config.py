"""
Configuration management for SOTA Training Pipeline.
"""

import argparse
from pathlib import Path
from typing import Dict, Any


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        # Model parameters
        'num_frames': 16,
        'frame_stride': 2,
        'img_size': 224,
        'pretrained_name': 'vit_base_patch16_224',
        'use_adapters': True,
        
        # Training parameters
        # Batch size 24 for better memory efficiency
        # Note: Will be reduced to 16 when distillation is enabled (to save VRAM)
        'batch_size': 24,
        'distillation_batch_size_ratio': 0.67,  # Reduce batch size by 33% when distillation enabled
        'epochs': 100,  # Tăng từ 50 → 100 để tất cả models có đủ thời gian học
        'base_lr': 1e-4,  # Increased from 5e-5 to 1e-4 (2x) - Tăng để học nhanh hơn, vẫn thấp hơn nhiều so với 1e-3
        'head_lr': 2e-3,  # Increased from 1e-3 to 2e-3 (2x) - Vẫn 20x base_lr, tăng để học nhanh hơn
        'weight_decay': 0.08,  # Tăng từ 0.05 → 0.08 để giảm overfitting mạnh hơn
        'grad_accum_steps': 4,  # Keep same to maintain effective batch size ~256 (64*4)
        'val_ratio': 0.20,  # Increased from 0.15 to have more validation samples
        'label_smoothing': 0.18,  # Tăng từ 0.15 → 0.18 để tăng regularization
        'dropout': 0.35,  # Tăng từ 0.3 → 0.35 để giảm overfitting mạnh hơn
        'mixup_alpha': 0.5,  # Tăng từ 0.4 → 0.5 để tăng augmentation mạnh hơn
        'cutmix_alpha': 1.0,
        'warmup_epochs': 5,  # Warmup epochs (giống notebook)
        'cosine_start_epoch': 5,  # Cosine starts ngay sau warmup (giống notebook, không có constant phase)
        'early_stop_patience': 12,  # Tăng từ 10 → 12 để model có thêm thời gian học với các features mới
        'use_adaptive_lr': True,  # Chỉ giảm LR khi model không cải thiện
        'lr_plateau_patience': 7,  # Số epochs không cải thiện trước khi giảm LR (tăng từ 5 → 7 để model có thêm thời gian học)
        'lr_min_delta': 0.005,  # Minimum change để coi là "cải thiện" (0.5% relative) - Tăng từ 0.001 để tránh noise
        'lr_threshold_mode': 'rel',  # 'rel' (relative) or 'abs' (absolute)
        'lr_cooldown': 7,  # Cooldown period sau khi giảm LR (epochs)
        'use_val_loss_for_lr': False,  # Xem xét cả validation loss (default: False, chỉ dùng accuracy)
        'min_lr_ratio': 0.01,  # Minimum LR = 1% của base_lr
        
        # Phase 1 features
        'use_ema': False,  # User không muốn dùng EMA
        'ema_decay': 0.9999,
        'use_cutmix': True,  # Enable CutMix để tăng augmentation
        'use_focal_loss': True,  # Enable Focal Loss để handle class imbalance tốt hơn
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'use_progressive_resize': True,  # Enable Progressive Resize để tăng accuracy
        'progressive_sizes': [160, 192, 224],
        'progressive_resize_patience': 5,  # Số epochs không cải thiện trước khi tăng size
        'progressive_resize_min_improvement': 0.005,  # Min improvement để coi là "cải thiện" (0.5%)
        'progressive_resize_force_final_epochs': 12,  # Force size cuối cùng trong N epochs cuối (tăng từ 10 → 12)
        'progressive_resize_min_epochs_per_stage': 5,  # Min epochs ở mỗi stage trước khi tăng size (tăng từ 3 → 5)

        # Architecture options
        'architecture': 'vit_base',  # 'vit_base', 'vit_large', 'timesformer', 'swin', 'multiscale'
        'multiscale_sizes': [224, 256, 288],
        # Default pretrained names for each architecture (used as fallback)
        'default_pretrained_names': {
            'vit_base': 'vit_base_patch16_224',
            'vit_large': 'vit_large_patch16_224',
            'swin': 'swin_base_patch4_window7_224',
            'timesformer': 'vit_base_patch16_224',  # TimeSformer uses ViT-Base as backbone
            'multiscale': 'vit_base_patch16_224'  # Multi-scale uses ViT-Base as backbone
            # Note: 'videomae' removed - VideoMAE uses HuggingFace model, not timm ViT
        },

        # Advanced augmentation
        'use_temporal_aug': True,
        'use_advanced_spatial': True,
        'use_advanced_color': True,

        # Synthetic data generation
        'use_synthetic_data': True,  # Enable synthetic data để tăng training data
        'synthetic_method': 'frame_mixup',  # 'frame_mixup', 'temporal_mixup', 'frame_shuffle', 'temporal_interpolation'
        'synthetic_ratio': 0.3,  # Ratio of synthetic data to real data

        # Knowledge distillation
        'use_distillation': False,
        'distillation_temperature': 4.0,  # Tăng từ 3.0 → 4.0 để soft distribution mềm hơn, dễ học hơn
        'distillation_alpha': 0.5,  # Giảm từ 0.7 → 0.5 để giảm phụ thuộc vào teacher, tăng trọng số hard labels
        'teacher_checkpoints': [],
        # Adaptive distillation disabling (universal feature)
        'distillation_disable_patience': 10,  # Epochs không cải thiện trước khi tắt distillation
        'distillation_min_delta': 0.005,  # 0.5% relative improvement threshold

        # Self-training
        'use_self_training': False,
        'self_training_iterations': 5,
        'confidence_threshold': 0.9,
        'pseudo_label_ratio': 0.5,

        # Pseudo-labeling
        'use_pseudo_labeling': True,  # Enable pseudo-labeling để sử dụng test data
        'pseudo_label_confidence': 0.85,  # Tăng từ 0.8 → 0.85 để chỉ dùng high-confidence labels
        'pseudo_label_weight': 0.3,  # Giảm từ 0.5 → 0.3 để conservative hơn

        # Stacking ensemble
        'use_stacking': False,
        'stacking_method': 'lightgbm',  # 'lightgbm', 'xgboost', 'neural'

        # Enhanced TTA
        'tta_num_crops': 30,  # Tăng từ 20 → 30 để tăng accuracy thêm 0.5-1%
        'tta_multi_scale_sizes': [224, 256, 288, 320, 352],  # Thêm scale 352
        'tta_use_temporal': True,
        'tta_confidence_threshold': None,

        # Calibration
        'use_calibration': True,  # Enable calibration để cải thiện confidence scores
        'calibration_method': 'temperature',  # 'temperature', 'platt'

        # Error analysis
        'use_error_analysis': False,
        'hard_sample_weight': 2.0,
        
        # Data
        'data_dir': './kaggle_data/data',
        'num_workers': 0,  # Windows: use 0
        
        # GPU optimization
        'auto_adjust_batch_size': False,  # Disabled - using batch_size=24 for better memory efficiency
        'use_torch_compile': False,  # Disabled by default (requires Triton, often not available on Windows)
        'enable_cudnn_benchmark': False,  # Disable by default for RTX 5090 compatibility
        
        # Output
        'output_dir': './checkpoints',
        'logging_dir': './logging',
        'submissions_dir': './submissions',
    }


def get_hyperparameter_variations() -> Dict[int, Dict[str, Any]]:
    """
    Get hyperparameter variations for different models in auto mode.
    Creates diversity through:
    - Learning rates, mixup/cutmix alpha, label smoothing
    - Augmentation strategies (Mixup vs CutMix)
    - Loss functions (CrossEntropy vs Focal Loss)
    - Progressive resize (some models)
    
    Uses default config values and only overrides what's necessary.
    """
    # Get default config to use as base
    default = get_default_config()
    
    return {
        # Optimized: 5 models với architecture đa dạng thay vì 7 models cùng ViT-Base
        # Strategy: Diversity trong architecture > nhiều models cùng architecture
        
        # Model 1: ViT-Large + All Advanced Features (trừ EMA) - BASE TEACHER MODEL
        # Strategy: Train đầu tiên để làm base teacher cho knowledge distillation
        # Expected: 70-73% Val Acc (cao nhất trong các models)
        # Optimizations: Giảm LR để học ổn định hơn, tăng epochs và lr_plateau_patience để học đủ
        # Batch size: 8 (ViT-Large có 316M params, cần batch size nhỏ để tránh OOM)
        # Note: pretrained_name sẽ tự động set dựa trên architecture
        1: {
            'architecture': 'vit_large',
            'batch_size': 8,  # Giảm từ 24 → 8 để tránh OOM (ViT-Large có 316M params)
            'num_frames': 16,  # Giữ 16 để training nhanh (24 quá chậm, 32 frames chỉ tăng ~1-2% acc nhưng chậm 2x)
            'frame_stride': 2,  # Giữ stride 2
            'base_lr': 5e-5,  # Giảm từ 7.5e-5 → 5e-5 để training ổn định hơn (tránh overfitting sau epoch 3)
            'head_lr': 1e-4,  # Giảm từ 1.5e-3 → 1e-4 để training ổn định hơn
            'epochs': 100,  # Tăng từ 60 → 100 để model có đủ thời gian học với size 224
            'use_ema': False, 'use_cutmix': True, 'use_focal_loss': True, 
            'use_progressive_resize': False,  # TẮT: Progressive resize chưa được implement đầy đủ (không recreate datasets)
            'mixup_alpha': 0.5, 'label_smoothing': 0.18,
            'dropout': 0.35, 'weight_decay': 0.08,  # Giữ nguyên regularization
            'drop_path_rate': 0.1,  # Stochastic Depth (DropPath) - SOTA regularization technique
            'use_temporal_aug': True, 'use_advanced_spatial': False, 'use_advanced_color': False,  # TẮT advanced augmentations tốn thời gian (elastic, histogram matching)
            'use_synthetic_data': False,  # TẮT: Synthetic data generation tốn thời gian, impact không cao
            'synthetic_method': 'frame_mixup', 'synthetic_ratio': 0.3,
            'use_pseudo_labeling': False,  # TẮT: Model 1 là teacher, không cần pseudo labeling (tránh học sai)
            'pseudo_label_confidence': 0.85, 'pseudo_label_weight': 0.3,
            'use_calibration': True,
            'use_distillation': False,  # Model 1 là base teacher, không dùng distillation
            'early_stop_patience': 12,  # Đặt lại 12 để model có thêm thời gian học
        },
        
        # Model 2: ViT-Base + All Advanced Features + Knowledge Distillation (STUDENT)
        # Strategy: Sequential distillation - Dùng Model 1 (ViT-Large) làm teacher
        # Training order: Model 1 → Model 2
        # Optimizations: Giảm LR để học ổn định hơn với distillation
        2: {
            'architecture': 'vit_base',
            'batch_size': 16,  # Set batch_size = 16 khi train với teacher Model 1 (ViT-Large)
            'epochs': 100,  # Train đến 100 epochs
            'base_lr': 5e-5,  # Giảm từ 7.5e-5 → 5e-5 để học ổn định hơn với distillation (tránh overfit)
            'head_lr': 1e-3,  # Giảm từ 1.5e-3 → 1e-3 (giữ ratio 20x, tránh overfit)
            'use_ema': False,  # User không muốn dùng EMA
            'use_cutmix': True, 'use_focal_loss': True, 'use_progressive_resize': True,
            'mixup_alpha': 0.6, 'label_smoothing': 0.22,  # Tăng regularization
            'dropout': 0.45, 'weight_decay': 0.12,  # Tăng regularization mạnh hơn để giảm overfitting (dropout 0.40→0.45, weight_decay 0.10→0.12)
            'use_temporal_aug': True, 'use_advanced_spatial': True, 'use_advanced_color': True,
            'use_synthetic_data': True, 'synthetic_method': 'frame_mixup', 'synthetic_ratio': 0.3,
            'use_pseudo_labeling': False,  # TẮT: Pseudo-labeling có thể gây nhiễu khi kết hợp với distillation
            'pseudo_label_confidence': 0.85, 'pseudo_label_weight': 0.3,
            'use_calibration': True,
            'use_distillation': True,  # Enable knowledge distillation
            'teacher_checkpoints': ['checkpoints/sota_vit_model_1_best.pt'],  # Sequential: Model 1 → Model 2
            'distillation_temperature': 3.5,  # Giảm từ 4.0 → 3.5 để soft distribution không quá mềm
            'distillation_alpha': 0.6,  # Tăng từ 0.5 → 0.6 để tăng trọng số teacher (học nhiều hơn từ teacher)
            'distillation_disable_patience': 10,  # Tăng từ 5 → 10 để distillation không tắt quá sớm
            'lr_plateau_patience': 5,  # LR giảm nhanh hơn khi model không cải thiện
            'early_stop_patience': 12,  # Đặt lại 12 để model có thêm thời gian học
        },
        
        # Model 3: TimeSformer + All Advanced Features + Knowledge Distillation (STUDENT)
        # Strategy: Sequential distillation - Dùng Model 2 làm teacher (đã học từ Model 1)
        # Training order: Model 1 → Model 2 → Model 3
        # Note: Progressive resize được khuyến nghị cho TimeSformer (theo research)
        # Dropout giảm xuống 0.30 vì TimeSformer đã có strong regularization từ attention
        # Optimizations: Giảm LR xuống 5e-5 vì TimeSformer cần LR thấp hơn, đặc biệt với distillation
        3: {
            'architecture': 'timesformer',
            'epochs': 100,  # Train đến 100 epochs
            'batch_size': 18,  # Tăng batch size lên 18 cho Model 3 (TimeSformer)
            'base_lr': 5e-5,  # Giảm từ 1e-4 → 5e-5 vì TimeSformer cần LR thấp hơn, đặc biệt với distillation
            'head_lr': 1e-3,  # Giảm từ 2e-3 → 1e-3 (giữ ratio 20x)
            'use_ema': False, 'use_cutmix': True, 'use_focal_loss': True, 'use_progressive_resize': True,
            'mixup_alpha': 0.6, 'label_smoothing': 0.22,  # Tăng regularization
            'dropout': 0.40, 'weight_decay': 0.10,  # Tăng mạnh regularization để giảm overfitting
            'use_temporal_aug': True, 'use_advanced_spatial': True, 'use_advanced_color': True,
            'use_synthetic_data': True, 'synthetic_method': 'temporal_mixup', 'synthetic_ratio': 0.3,
            'use_pseudo_labeling': True, 'pseudo_label_confidence': 0.85, 'pseudo_label_weight': 0.3,
            'use_calibration': True,
            'use_distillation': True,  # Enable knowledge distillation
            'teacher_checkpoints': ['checkpoints/sota_vit_model_2_best.pt'],  # Sequential: Model 2 → Model 3
            'distillation_temperature': 4.0,  # Tăng từ 3.0 → 4.0 để soft distribution mềm hơn, dễ học hơn
            'distillation_alpha': 0.5,  # Giảm từ 0.7 → 0.5 để giảm phụ thuộc vào teacher, tăng trọng số hard labels
            'lr_plateau_patience': 5,  # LR giảm nhanh hơn khi model không cải thiện
            'early_stop_patience': 9,  # Tránh overfit lâu, nhưng vẫn đủ thời gian để model học
        },
        
        # Model 4: Swin Transformer + All Advanced Features + Knowledge Distillation (STUDENT)
        # Strategy: Sequential distillation - Dùng Model 3 làm teacher (đã học từ Model 2 → Model 1)
        # Training order: Model 1 → Model 2 → Model 3 → Model 4
        # Optimizations: Giảm LR để học ổn định hơn với distillation
        4: {
            'architecture': 'swin',
            'epochs': 100,  # Train đến 100 epochs
            'batch_size': 100,  # Batch size cho Swin model
            'base_lr': 7.5e-5,  # Giảm từ 1e-4 → 7.5e-5 để học ổn định hơn với distillation
            'head_lr': 1.5e-3,  # Giảm từ 2e-3 → 1.5e-3 (giữ ratio 20x)
            'use_ema': False, 'use_cutmix': True, 'use_focal_loss': True, 'use_progressive_resize': True,
            'mixup_alpha': 0.6, 'label_smoothing': 0.22,  # Tăng regularization
            'dropout': 0.40, 'weight_decay': 0.10,  # Tăng regularization để giảm overfitting
            'use_temporal_aug': True, 'use_advanced_spatial': True, 'use_advanced_color': True,
            'use_synthetic_data': True, 'synthetic_method': 'frame_mixup', 'synthetic_ratio': 0.3,
            'use_pseudo_labeling': True, 'pseudo_label_confidence': 0.85, 'pseudo_label_weight': 0.3,
            'use_calibration': True,
            'use_distillation': True,  # Enable knowledge distillation
            'teacher_checkpoints': ['checkpoints/sota_vit_model_3_best.pt'],  # Sequential: Model 3 → Model 4
            'distillation_temperature': 4.0,  # Tăng từ 3.0 → 4.0 để soft distribution mềm hơn, dễ học hơn
            'distillation_alpha': 0.5,  # Giảm từ 0.7 → 0.5 để giảm phụ thuộc vào teacher, tăng trọng số hard labels
            'lr_plateau_patience': 5,  # LR giảm nhanh hơn khi model không cải thiện
            'early_stop_patience': 9,  # Tránh overfit lâu, nhưng vẫn đủ thời gian để model học
        },
        
        # Model 5: Multi-scale ViT + All Advanced Features + Knowledge Distillation (STUDENT)
        # Strategy: Sequential distillation - Dùng Model 4 làm teacher (đã học từ Model 3 → Model 2 → Model 1)
        # Training order: Model 1 → Model 2 → Model 3 → Model 4 → Model 5
        # Optimizations: Giảm LR để học ổn định hơn với distillation
        5: {
            'architecture': 'multiscale',
            'epochs': 100,  # Train đến 100 epochs
            'base_lr': 7.5e-5,  # Giảm từ 1e-4 → 7.5e-5 để học ổn định hơn với distillation
            'head_lr': 1.5e-3,  # Giảm từ 2e-3 → 1.5e-3 (giữ ratio 20x)
            'use_ema': False, 'use_cutmix': True, 'use_focal_loss': True, 'use_progressive_resize': False,
            'mixup_alpha': 0.5, 'label_smoothing': 0.18,
            'dropout': 0.35, 'weight_decay': 0.08,
            'use_temporal_aug': True, 'use_advanced_spatial': True, 'use_advanced_color': True,
            'use_synthetic_data': True, 'synthetic_method': 'frame_mixup', 'synthetic_ratio': 0.3,
            'use_pseudo_labeling': True, 'pseudo_label_confidence': 0.85, 'pseudo_label_weight': 0.3,
            'use_calibration': True,
            'use_distillation': True,  # Enable knowledge distillation
            'teacher_checkpoints': ['checkpoints/sota_vit_model_4_best.pt'],  # Sequential: Model 4 → Model 5
            'distillation_temperature': 4.0,  # Tăng từ 3.0 → 4.5 để soft distribution mềm hơn, dễ học hơn
            'distillation_alpha': 0.5,  # Giảm từ 0.7 → 0.5 để giảm phụ thuộc vào teacher, tăng trọng số hard labels
        },
        
        # Model 6: ViT-Base, CutMix, CrossEntropy, Progressive Resize (Best từ trước)
        6: {
            'architecture': 'vit_base',
            'use_ema': False, 'use_cutmix': True, 'use_focal_loss': False, 'use_progressive_resize': True
        },
        
        # Model 7: VideoMAEv2-style (ViT-Large với spatiotemporal modeling tối ưu)
        # Strategy: SOTA model với accuracy cao nhất (90% trên Kinetics-400)
        # Sử dụng ViT-Large với divided space-time attention (tốt hơn TimeSformer)
        # VideoMAEv2 thực sự cần pretraining phức tạp, nên dùng ViT-Large + divided attention
        7: {
            'architecture': 'videomae',  # VideoMAE-style model
            'epochs': 100,
            'batch_size': 16,  # ViT-Large tốn memory, batch size nhỏ hơn
            'num_frames': 16,  # Tốt cho temporal modeling
            'base_lr': 1e-4,  # LR cao hơn một chút cho ViT-Large
            'head_lr': 2e-3,  # Giữ ratio 20x
            'use_ema': False, 'use_cutmix': True, 'use_focal_loss': True, 'use_progressive_resize': False,
            'mixup_alpha': 0.5, 'label_smoothing': 0.2,
            'dropout': 0.3, 'weight_decay': 0.05,
            'drop_path_rate': 0.2,  # Stochastic Depth cho ViT-Large
            'use_temporal_aug': False,  # Tắt để training nhanh hơn
            'use_advanced_spatial': False,  # Tắt để training nhanh hơn
            'use_advanced_color': False,  # Tắt để training nhanh hơn
            'use_synthetic_data': False,  # Tắt để training nhanh hơn
            'use_pseudo_labeling': False,  # Standalone model
            'use_calibration': True,
            'use_distillation': False,  # Standalone model, không dùng distillation
            'early_stop_patience': 12,
            'lr_plateau_patience': 7,
        },
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='SOTA Training Pipeline for Video Action Recognition',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--model-id', type=int, default=None, help='Model ID (1-7). If not specified, will train 5 models automatically')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-models', type=int, default=None, help='Number of models to train (default: 5 for auto mode, 1 for single model)')
    parser.add_argument('--no-auto-ensemble', action='store_true', help='Disable automatic ensemble after training (default: enabled when training multiple models)')
    parser.add_argument('--use-vit-large', action='store_true', help='Use ViT-Large instead of ViT-Base')
    
    # Phase 1 features
    parser.add_argument('--use-ema', action='store_true', help='Enable Exponential Moving Average')
    parser.add_argument('--use-cutmix', action='store_true', help='Enable CutMix augmentation')
    parser.add_argument('--use-focal-loss', action='store_true', help='Enable Focal Loss')
    parser.add_argument('--use-progressive-resize', action='store_true', help='Enable Progressive Resizing')
    
    # Data configuration
    parser.add_argument('--data-dir', type=str, default='./kaggle_data/data', help='Path to dataset directory')
    parser.add_argument('--num-frames', type=int, default=16, help='Number of frames per video')
    parser.add_argument('--frame-stride', type=int, default=2, help='Frame stride for sampling')
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    
    # Training configuration
    # Use None as default so we can check if user explicitly set the value
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (default: 64 from config for better generalization)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (default: 50 from config)')
    parser.add_argument('--base-lr', type=float, default=None, help='Base learning rate (overrides model variation)')
    parser.add_argument('--head-lr', type=float, default=None, help='Head learning rate (overrides model variation)')
    parser.add_argument('--weight-decay', type=float, default=None, help='Weight decay (default: 0.05 from config)')
    parser.add_argument('--grad-accum-steps', type=int, default=None, help='Gradient accumulation steps (default: 4 from config)')
    parser.add_argument('--val-ratio', type=float, default=None, help='Validation split ratio (default: 0.20 from config)')
    parser.add_argument('--label-smoothing', type=float, default=None, help='Label smoothing (overrides model variation)')
    parser.add_argument('--mixup-alpha', type=float, default=None, help='Mixup alpha (overrides model variation)')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--early-stop-patience', type=int, default=12, help='Early stopping patience')
    
    # Focal Loss parameters
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Focal Loss alpha')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal Loss gamma')
    
    # EMA parameters
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='EMA decay rate')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='./checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--logging-dir', type=str, default='./logging', help='Logging directory')
    parser.add_argument('--submissions-dir', type=str, default='./submissions', help='Submissions directory')
    
    # Training control
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--inference-only', action='store_true', help='Run inference only')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path for inference')
    
    # System
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loader workers (0 for Windows)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Auto mode: train 5 models với diverse architectures nếu BOTH model-id AND num-models không được chỉ định
    is_auto_mode = args.model_id is None and args.num_models is None
    if is_auto_mode:
        args.num_models = 5  # Optimized: 5 models với diverse architectures thay vì 7 models cùng ViT-Base
        args.model_id = 1  # Start from model 1
    
    # Check if user explicitly set SOTA features
    user_set_sota_features = args.use_ema or args.use_cutmix or args.use_focal_loss or args.use_progressive_resize
    
    # Build config dict
    config = get_default_config()
    
    # Update with model variation if available (only for single model mode)
    # Note: In auto mode (multiple models), variations will be applied per-model in main.py
    variations = get_hyperparameter_variations()
    if args.model_id in variations and not is_auto_mode:
        # Single model mode: apply variation for this specific model
        variation = variations[args.model_id].copy()
        # Apply ALL variation settings (including SOTA features) for single model mode
        # User can still override with command-line arguments if needed
        config.update(variation)
        
    
    # Override with command-line arguments
    if args.use_vit_large:
        config['pretrained_name'] = 'vit_large_patch16_224'
        # Adjust batch size for ViT-Large (reduce from default 24)
        # Only adjust if user didn't explicitly set batch_size
        if args.batch_size is None and config.get('batch_size', 24) == 24:
            config['batch_size'] = 8
    
    config['model_id'] = args.model_id if args.model_id is not None else 1
    config['seed'] = args.seed
    config['num_models'] = args.num_models if args.num_models is not None else 1
    
    # Apply SOTA features: use variation if available, otherwise use command-line args
    if is_auto_mode and not user_set_sota_features:
        # Will be set per-model in main.py from variations
        config['use_ema'] = False  # Placeholder, will be overridden per model
        config['use_cutmix'] = False
        config['use_focal_loss'] = False
        config['use_progressive_resize'] = False
        # auto_use_variations will be set later if num_models > 1
    elif not is_auto_mode and args.model_id in variations and not user_set_sota_features:
        # Single model mode: variation đã được apply ở trên, giữ nguyên
        # Chỉ override nếu user explicitly set via command-line
        pass  # Variation đã có SOTA features, không cần override
    else:
        # Use command-line arguments (only if user explicitly set them)
        if user_set_sota_features:
            config['use_ema'] = args.use_ema
            config['use_cutmix'] = args.use_cutmix
            config['use_focal_loss'] = args.use_focal_loss
            config['use_progressive_resize'] = args.use_progressive_resize
        # Nếu không set, giữ nguyên từ variation hoặc default
    config['ema_decay'] = args.ema_decay
    config['focal_alpha'] = args.focal_alpha
    config['focal_gamma'] = args.focal_gamma
    
    config['data_dir'] = Path(args.data_dir)
    config['num_frames'] = args.num_frames
    config['frame_stride'] = args.frame_stride
    config['img_size'] = args.img_size
    
    # Only override config values if user explicitly provided command-line arguments
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['epochs'] = args.epochs
        config['_epochs_explicitly_set'] = True  # Flag to track if epochs was set from command line
    else:
        config['_epochs_explicitly_set'] = False
    if args.weight_decay is not None:
        config['weight_decay'] = args.weight_decay
    if args.grad_accum_steps is not None:
        config['grad_accum_steps'] = args.grad_accum_steps
    if args.val_ratio is not None:
        config['val_ratio'] = args.val_ratio
    
    config['warmup_epochs'] = args.warmup_epochs
    # Only override early_stop_patience if user explicitly provided it (not using default)
    # This allows model variations to set their own early_stop_patience
    # Check if config already has early_stop_patience from variation (different from default 12)
    if 'early_stop_patience' in config and config['early_stop_patience'] != 12:
        # Keep variation value, only override if user explicitly set a different value
        if args.early_stop_patience != 12 and args.early_stop_patience != config['early_stop_patience']:
            config['early_stop_patience'] = args.early_stop_patience
        # Otherwise keep variation value
    else:
        # No variation value, use command line argument (or default 12)
        config['early_stop_patience'] = args.early_stop_patience
    config['num_workers'] = args.num_workers
    config['debug'] = args.debug
    
    if args.base_lr is not None:
        config['base_lr'] = args.base_lr
    if args.head_lr is not None:
        config['head_lr'] = args.head_lr
    if args.label_smoothing is not None:
        config['label_smoothing'] = args.label_smoothing
    if args.mixup_alpha is not None:
        config['mixup_alpha'] = args.mixup_alpha
    
    config['output_dir'] = Path(args.output_dir)
    config['logging_dir'] = Path(args.logging_dir)
    config['submissions_dir'] = Path(args.submissions_dir)
    config['resume'] = args.resume
    config['inference_only'] = args.inference_only
    config['checkpoint'] = args.checkpoint
    
    # Set auto_ensemble and auto_use_variations: True by default if training multiple models
    if config.get('num_models', 1) > 1 and not config.get('inference_only', False):
        config['auto_ensemble'] = not getattr(args, 'no_auto_ensemble', False)
        # Auto-enable variations for diverse architectures (only if user didn't set SOTA features)
        if not user_set_sota_features:
            config['auto_use_variations'] = True
        else:
            config['auto_use_variations'] = False
    else:
        config['auto_ensemble'] = False
        config['auto_use_variations'] = False
    
    # Create output directories
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    config['logging_dir'].mkdir(parents=True, exist_ok=True)
    config['submissions_dir'].mkdir(parents=True, exist_ok=True)
    
    return config
