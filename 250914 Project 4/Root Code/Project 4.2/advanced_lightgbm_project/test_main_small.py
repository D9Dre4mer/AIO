"""
Test main.py với dữ liệu nhỏ để kiểm tra việc lưu biểu đồ
"""

import os
import sys
import yaml
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from main import AdvancedLightGBMPipeline

def create_small_config():
    """Tạo config nhỏ cho test"""
    config = {
        'data': {
            'dataset_name': 'synthetic_small',
            'target_column': 'target',
            'test_size': 0.3,
            'random_state': 42
        },
        'feature_engineering': {
            'enable_polynomial': True,
            'polynomial_degree': 2,
            'enable_interaction': True,
            'enable_statistical': True,
            'enable_target_encoding': False,
            'feature_selection': True,
            'max_features': 20
        },
        'hyperparameter_optimization': {
            'enable': True,
            'n_trials': 5,  # Giảm số trials
            'timeout': 60,  # Giảm timeout
            'cv_folds': 3
        },
        'lightgbm': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 10,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        },
        'ensemble': {
            'enable': True,
            'methods': ['voting_soft', 'voting_hard', 'stacking'],
            'cv_folds': 3
        },
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc_roc'],
            'cv_folds': 3
        },
        'output': {
            'results_dir': 'results',
            'save_models': True,
            'save_plots': True
        }
    }
    return config

def create_small_synthetic_data():
    """Tạo dữ liệu synthetic nhỏ"""
    print("🔧 Creating small synthetic dataset...")
    
    np.random.seed(42)
    n_samples = 200  # Tăng lên một chút để có đủ dữ liệu
    n_features = 8
    
    # Tạo features
    X = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Tạo target với logic đơn giản
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    # Tạo DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Lưu dữ liệu
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Lưu các file
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train
    train_df.to_csv(data_dir / "synthetic_small_train.csv", index=False)
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['target'] = y_test
    test_df.to_csv(data_dir / "synthetic_small_test.csv", index=False)
    
    val_df = train_df.sample(n=len(X_test), random_state=42)
    val_df.to_csv(data_dir / "synthetic_small_val.csv", index=False)
    
    print(f"   ✅ Created dataset: {len(X_train)} train, {len(X_test)} test, {len(val_df)} val samples")
    return len(X_train), len(X_test), len(val_df)

def test_main_plots():
    """Test main.py với dữ liệu nhỏ"""
    print("🚀 Testing main.py plot saving with small data...")
    print("=" * 60)
    
    try:
        # Tạo dữ liệu nhỏ
        n_train, n_test, n_val = create_small_synthetic_data()
        
        # Tạo config nhỏ
        config = create_small_config()
        
        # Lưu config
        config_path = "config/test_config.yaml"
        os.makedirs("config", exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"📁 Config saved to: {config_path}")
        
        # Khởi tạo pipeline
        pipeline = AdvancedLightGBMPipeline(config_path)
        
        # Chạy pipeline với dataset có sẵn
        print("\n🚀 Running pipeline...")
        results = pipeline.run_complete_pipeline('fe')
        
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 Results: {list(results.keys())}")
        
        # Kiểm tra thư mục kết quả
        results_dir = Path("results")
        if results_dir.exists():
            latest_run = max([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('run_')], 
                           key=lambda x: x.stat().st_mtime, default=None)
            
            if latest_run:
                plots_dir = latest_run / "plots"
                if plots_dir.exists():
                    plot_files = list(plots_dir.glob("*.png"))
                    print(f"\n📈 Plots saved in: {plots_dir}")
                    print(f"📊 Total plots: {len(plot_files)}")
                    
                    for i, plot_file in enumerate(sorted(plot_files), 1):
                        print(f"   {i:2d}. {plot_file.name}")
                else:
                    print("⚠️  No plots directory found")
            else:
                print("⚠️  No run directory found")
        else:
            print("⚠️  No results directory found")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_main_plots()
