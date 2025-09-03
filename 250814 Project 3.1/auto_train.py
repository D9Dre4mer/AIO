#!/usr/bin/env python3
"""
Auto Training Script - Chạy training tự động với file CSV trong cache
Tự động đọc file CSV và kích hoạt tất cả chức năng cần thiết

Usage:
    python auto_train.py

Features:
- Tự động đọc file 2cls_spam_text_cls.csv từ cache/
- Tự động cấu hình: text column = 'Message', label column = 'Category'
- Kích hoạt tất cả preprocessing options
- Chạy training với tất cả models (7 models) và vectorization methods (3 methods)
- Tạo cache tự động
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import progress tracking
from utils.progress_tracker import create_training_progress


def print_banner():
    """In banner chào mừng"""
    print("=" * 80)
    print("🤖 TOPIC MODELING - AUTO CLASSIFIER")
    print("🚀 AUTO TRAINING SCRIPT")
    print("=" * 80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def check_file_exists(file_path):
    """Kiểm tra file có tồn tại không"""
    if os.path.exists(file_path):
        print(f"✅ Found file: {file_path}")
        return True
    else:
        print(f"❌ File not found: {file_path}")
        return False


def load_dataset(file_path):
    """Đọc dataset từ file CSV"""
    try:
        print(f"📂 Loading dataset from: {file_path}")
        
        # Đọc file CSV
        df = pd.read_csv(file_path)
        
        print("✅ Dataset loaded successfully!")
        print(f"   📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   📋 Columns: {list(df.columns)}")
        
        # Kiểm tra cột Message và Category
        if 'Message' not in df.columns:
            print("❌ Column 'Message' not found!")
            print(f"   Available columns: {list(df.columns)}")
            return None
            
        if 'Category' not in df.columns:
            print("❌ Column 'Category' not found!")
            print(f"   Available columns: {list(df.columns)}")
            return None
        
        # Hiển thị thông tin về dữ liệu
        print(f"   📝 Text samples: {len(df['Message'].dropna()):,}")
        print(f"   🏷️  Unique categories: {df['Category'].nunique()}")
        print("   📊 Category distribution:")
        category_counts = df['Category'].value_counts()
        for category, count in category_counts.items():
            print(f"      - {category}: {count:,} samples")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


def create_auto_config(df, mode='full'):
    """Tạo cấu hình tự động cho training"""
    print(f"\n🔧 Creating auto configuration ({mode} mode)...")
    
    # Step 1: Dataset configuration
    max_samples = 10000 if mode == 'full' else 1000
    step1_config = {
        'dataframe': df,
        'file_path': 'cache/2cls_spam_text_cls.csv',
        'sampling_config': {
            'num_samples': min(max_samples, len(df)),
            'sampling_strategy': 'Stratified (Recommended)'
        },
        'dataset_size': len(df)
    }
    
    # Step 2: Column selection & preprocessing
    step2_config = {
        'text_column': 'Message',
        'label_column': 'Category',
        'text_samples': len(df['Message'].dropna()),
        'unique_classes': df['Category'].nunique(),
        'distribution': 'Balanced' if df['Category'].nunique() <= 5 else 'Multi-class',
        'avg_length': df['Message'].astype(str).str.len().mean(),
        'avg_length_words': df['Message'].astype(str).str.split().str.len().mean(),
        'unique_words': len(set(' '.join(df['Message'].astype(str).dropna()).lower().split())),
        'validation_errors': [],
        'validation_warnings': [],
        # Kích hoạt tất cả preprocessing options
        'text_cleaning': True,
        'category_mapping': True,
        'data_validation': True,
        'memory_optimization': True,
        # Advanced preprocessing options
        'rare_words_removal': True,
        'rare_words_threshold': 2,
        'lemmatization': True,
        'context_aware_stopwords': True,
        'stopwords_aggressiveness': 'Moderate',
        'phrase_detection': True,
        'min_phrase_freq': 3,
        'completed': True
    }
    
    # Step 3: Model configuration
    if mode == 'full':
        selected_models = [
            'K-Nearest Neighbors',
            'Decision Tree', 
            'Naive Bayes',
            'K-Means Clustering',
            'Support Vector Machine',
            'Logistic Regression',
            'Linear SVC'
        ]
        selected_vectorization = [
            'BoW',
            'TF-IDF', 
            'Word Embeddings'
        ]
        ensemble_enabled = True
        cv_folds = 5
    else:  # quick mode
        selected_models = [
            'K-Nearest Neighbors',
            'Naive Bayes',
            'Logistic Regression'
        ]
        selected_vectorization = [
            'TF-IDF',
            'Word Embeddings'
        ]
        ensemble_enabled = False
        cv_folds = 3
    
    step3_config = {
        'data_split': {
            'training': 80,
            'test': 20
        },
        'cross_validation': {
            'cv_folds': cv_folds,
            'random_state': 42
        },
        'selected_models': selected_models,
        'selected_vectorization': selected_vectorization,
        'ensemble_learning': {
            'eligible': ensemble_enabled,
            'enabled': ensemble_enabled,
            'final_estimator': 'voting' if ensemble_enabled else None
        },
        'knn_config': {
            'optimization_method': 'Manual Input',
            'k_value': 15,
            'weights': 'distance',
            'metric': 'cosine',
            'cv_folds': cv_folds,
            'scoring_metric': 'f1_weighted'
        },
        'validation_errors': [],
        'validation_warnings': [],
        'completed': True
    }
    
    print("✅ Auto configuration created!")
    print(f"   📊 Models: {len(step3_config['selected_models'])} models")
    print(f"   🔤 Vectorization: {len(step3_config['selected_vectorization'])} methods")
    print(f"   🚀 Ensemble Learning: {'Enabled' if step3_config['ensemble_learning']['enabled'] else 'Disabled'}")
    print(f"   📈 Total combinations: {len(step3_config['selected_models']) * len(step3_config['selected_vectorization'])}")
    print(f"   🎯 KNN Configuration: K={step3_config['knn_config']['k_value']}, "
          f"Weights={step3_config['knn_config']['weights']}, "
          f"Metric={step3_config['knn_config']['metric']}")
    
    return step1_config, step2_config, step3_config


def run_training(step1_config, step2_config, step3_config):
    """Chạy training với cấu hình đã tạo"""
    print("\n🚀 Starting training...")
    
    try:
        # Import training pipeline
        from training_pipeline import execute_streamlit_training
        
        # Progress callback function
        def progress_callback(phase, message, progress):
            print(f"   [{phase}] {message} ({progress:.1%})")
        
        # GPU Optimization: Convert sparse matrices to dense for GPU acceleration
        print(f"\n🚀 ENABLING GPU OPTIMIZATION...")
        print(f"   • Converting sparse matrices (BoW, TF-IDF) to dense arrays")
        print(f"   • This enables GPU acceleration for all vectorization methods")
        print(f"   • Memory usage will increase but performance will improve")
        
        # Execute training
        result = execute_streamlit_training(
            step1_config['dataframe'],
            step1_config,
            step2_config, 
            step3_config,
            progress_callback=progress_callback
        )
        
        if result['status'] == 'success':
            print("✅ Training completed successfully!")
            print(f"   📊 Successful combinations: {result.get('successful_combinations', 0)}")
            print(f"   ⏱️  Total time: {result.get('evaluation_time', 0):.1f}s")
            return result
        else:
            print(f"❌ Training failed: {result.get('message', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None


def display_results(result):
    """Hiển thị kết quả training"""
    if not result:
        print("❌ No results to display")
        return
    
    print("\n📊 TRAINING RESULTS")
    print("=" * 50)
    
    # Best model
    best_combinations = result.get('best_combinations', {})
    best_overall = best_combinations.get('best_overall', {})
    
    if best_overall:
        print(f"🥇 Best Model: {best_overall.get('combination_key', 'N/A')}")
        print(f"   📈 F1 Score: {best_overall.get('f1_score', 0):.3f}")
        print(f"   🎯 Test Accuracy: {best_overall.get('test_accuracy', 0):.3f}")
        print(f"   ⏱️  Training Time: {best_overall.get('training_time', 0):.1f}s")
    
    # Summary statistics
    comprehensive_results = result.get('comprehensive_results', [])
    successful_results = [r for r in comprehensive_results if r.get('status') == 'success']
    
    if successful_results:
        print("\n📊 Summary Statistics:")
        print(f"   ✅ Successful models: {len(successful_results)}")
        print(f"   📈 Average F1 Score: {np.mean([r.get('f1_score', 0) for r in successful_results]):.3f}")
        print(f"   🎯 Average Accuracy: {np.mean([r.get('test_accuracy', 0) for r in successful_results]):.3f}")
        print(f"   ⏱️  Average Training Time: {np.mean([r.get('training_time', 0) for r in successful_results]):.1f}s")
        
        # Top 3 models
        print("\n🏆 Top 3 Models:")
        top_models = sorted(successful_results, key=lambda x: x.get('f1_score', 0), reverse=True)[:3]
        for i, model in enumerate(top_models, 1):
            model_name = f"{model.get('model_name', 'Unknown')} + {model.get('embedding_name', 'Unknown')}"
            f1_score = model.get('f1_score', 0)
            print(f"   {i}. {model_name}: {f1_score:.3f}")


def main():
    """Hàm main chính"""
    print_banner()
    
    # Đường dẫn file CSV
    csv_file = "cache/2cls_spam_text_cls.csv"
    
    # Kiểm tra file có tồn tại không
    if not check_file_exists(csv_file):
        print(f"\n💡 Please make sure the file exists at: {csv_file}")
        print("   You can:")
        print("   1. Create the file manually")
        print("   2. Download a sample dataset")
        print("   3. Use a different file path")
        return
    
    # Đọc dataset
    df = load_dataset(csv_file)
    if df is None:
        return
    
    # Chọn mode
    print("\n🔧 Training Mode Selection:")
    print("   1. Quick Mode (3 models, 2 vectorization, ~2-3 minutes)")
    print("   2. Full Mode (7 models, 3 vectorization, ~5-10 minutes)")
    
    try:
        choice = input("\nEnter your choice (1 or 2, default=1): ").strip()
        mode = 'quick' if choice != '2' else 'full'
    except KeyboardInterrupt:
        print("\n\n❌ Training cancelled by user")
        return
    
    print(f"\n🚀 Selected: {mode.upper()} MODE")
    
    # Tạo cấu hình tự động
    step1_config, step2_config, step3_config = create_auto_config(df, mode)
    
    # Chạy training
    result = run_training(step1_config, step2_config, step3_config)
    
    # Hiển thị kết quả
    display_results(result)
    
    print("\n" + "=" * 80)
    print("🎉 AUTO TRAINING COMPLETED!")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if result:
        print("💾 Cache has been created for future use!")
        print("🚀 You can now use the Streamlit app with cached results!")


if __name__ == "__main__":
    main()
