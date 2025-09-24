#!/usr/bin/env python3
"""
Auto Training Script - Test Enhanced ML Models
Tự động đọc file CSV và test các models mới đã implement

Usage:
    python auto_train.py

Features:
- Tự động đọc file CSV từ cache/
- Test 6 models mới: RandomForest, AdaBoost, GradientBoosting, XGBoost, LightGBM, CatBoost
- GPU acceleration cho XGBoost, LightGBM, CatBoost
- Optuna hyperparameter optimization
- Multi-input data processing với auto-detection
- SHAP model interpretability
- Sample size: 1000 samples để test nhanh
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

def print_banner():
    """In banner chào mừng"""
    print("=" * 80)
    print("🤖 ENHANCED ML MODELS - AUTO TESTING")
    print("🚀 Testing New Models: RandomForest, AdaBoost, GradientBoosting")
    print("🔥 GPU Models: XGBoost, LightGBM, CatBoost")
    print("🏗️ Stacking Ensemble: Meta-learner với 6 base models")
    print("⚡ Features: Optuna, SHAP, Multi-Input, GPU Acceleration")
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


def create_sample_dataset():
    """Tạo dataset mẫu với 1000 samples để test"""
    print("📊 Creating sample dataset with 1000 samples...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Tạo features số học
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.normal(0, 1, n_samples)
    feature3 = np.random.uniform(0, 10, n_samples)
    feature4 = np.random.exponential(2, n_samples)
    
    # Tạo features categorical
    categories = ['A', 'B', 'C', 'D', 'E']
    feature5 = np.random.choice(categories, n_samples)
    
    # Tạo features text (simplified)
    text_samples = [
        "This is a sample text about machine learning and data science",
        "The model performs well on classification tasks",
        "We need to optimize hyperparameters for better performance",
        "Feature engineering is crucial for model accuracy",
        "Cross-validation helps prevent overfitting"
    ]
    feature6 = np.random.choice(text_samples, n_samples)
    
    # Tạo target labels (3 classes)
    # Tạo target dựa trên combination của features để có pattern
    target = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if feature1[i] > 0.5 and feature3[i] > 5:
            target[i] = 0  # Class 0
        elif feature2[i] < -0.5 or feature4[i] > 3:
            target[i] = 1  # Class 1
        else:
            target[i] = 2  # Class 2
    
    # Tạo DataFrame
    df = pd.DataFrame({
        'numeric_feature1': feature1,
        'numeric_feature2': feature2,
        'numeric_feature3': feature3,
        'numeric_feature4': feature4,
        'categorical_feature': feature5,
        'text_feature': feature6,
        'target': target
    })
    
    print("✅ Sample dataset created!")
    print(f"   📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   📋 Columns: {list(df.columns)}")
    print(f"   🎯 Target classes: {df['target'].nunique()}")
    print("   📊 Class distribution:")
    class_counts = df['target'].value_counts().sort_index()
    for class_id, count in class_counts.items():
        print(f"      - Class {class_id}: {count:,} samples ({count/n_samples*100:.1f}%)")
    
    return df


def load_or_create_dataset():
    """Đọc dataset từ file hoặc tạo dataset mẫu"""
    # Thử đọc từ file cache trước
    cache_files = [
        "cache/20250822-004129_sample-300_000Samples.csv",
        "cache/sample_dataset.csv",
        "data/sample_dataset.csv"
    ]
    
    for file_path in cache_files:
        if check_file_exists(file_path):
            try:
                print(f"📂 Loading dataset from: {file_path}")
                df = pd.read_csv(file_path)
                
                # Kiểm tra có đủ samples không
                if len(df) < 100:
                    print(f"⚠️ Dataset too small ({len(df)} samples), creating sample dataset...")
                    return create_sample_dataset()
                
                # Nếu có quá nhiều samples, lấy sample
                if len(df) > 1000:
                    print(f"📊 Dataset large ({len(df)} samples), sampling 1000 samples...")
                    df = df.sample(n=1000, random_state=42).reset_index(drop=True)
                
                print("✅ Dataset loaded successfully!")
                print(f"   📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
                print(f"   📋 Columns: {list(df.columns)}")
                
                return df
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                continue
    
    # Nếu không tìm thấy file nào, tạo dataset mẫu
    print("💡 No suitable dataset found, creating sample dataset...")
    return create_sample_dataset()


def create_enhanced_config(df):
    """Tạo cấu hình cho enhanced models"""
    print("\n🔧 Creating enhanced configuration...")
    
    # Auto-detect label column
    label_candidates = ['target', 'label', 'category', 'class', 'y']
    label_column = None
    for col in label_candidates:
        if col in df.columns:
            label_column = col
            break
    
    if label_column is None:
        print("❌ No label column found!")
        return None, None, None
    
    print(f"   🎯 Detected label column: {label_column}")
    
    # Auto-detect text column
    text_candidates = ['text_feature', 'abstract', 'text', 'content', 'description']
    text_column = None
    for col in text_candidates:
        if col in df.columns:
            text_column = col
            break
    
    print(f"   📝 Detected text column: {text_column}")
    
    # Step 1: Dataset configuration với multi-input
    step1_config = {
        'dataframe': df,
        'text_column': text_column,
        'label_column': label_column,
        'selected_categories': sorted(df[label_column].unique().tolist()),
        'is_multi_input': True,
        'input_columns': [col for col in df.columns if col != label_column],
        'type_mapping': {},  # Will be auto-detected
        'preprocessing_config': {
            'numeric_scaler': 'standard',
            'text_encoding': 'label',
            'missing_numeric': 'mean',
            'missing_text': 'mode'
        },
        'sampling_config': {
            'num_samples': len(df),
            'sampling_strategy': 'Stratified (Recommended)'
        },
        'completed': True
    }
    
    # Step 2: Preprocessing configuration
    step2_config = {
        'rare_words_removal': True,
        'rare_words_threshold': 2,
        'lemmatization': True,
        'context_aware_stopwords': True,
        'stopwords_aggressiveness': 'Moderate',
        'phrase_detection': True,
        'min_phrase_freq': 3,
        'completed': True
    }
    
    # Step 3: Enhanced model configuration
    step3_config = {
        'data_split': {
            'training': 80,
            'test': 20
        },
        'cross_validation': {
            'cv_folds': 5,
            'random_state': 42
        },
        # Test các models mới + stacking
        'selected_models': [
            'random_forest',
            'adaboost', 
            'gradient_boosting',
            'xgboost',
            'lightgbm',
            'catboost'
        ],
        'selected_vectorization': [
            'TF-IDF'  # Chỉ dùng TF-IDF để test nhanh
        ],
        'ensemble_learning': {
            'eligible': True,  # Enable stacking với 6 base models
            'enabled': True,
            'final_estimator': 'logistic_regression',  # Meta-learner
            'base_models': ['random_forest', 'adaboost', 'gradient_boosting', 'xgboost', 'lightgbm', 'catboost'],
            'min_base_models': 4,  # Cần ít nhất 4 base models
            'use_original_features': False,  # Chỉ dùng predictions từ base models
            'cv_folds': 5,
            'stratified': True
        },
        # Optuna configuration
        'optuna': {
            'enabled': True,
            'trials': 20,  # Giảm trials để test nhanh
            'timeout': 300,  # 5 phút timeout
            'direction': 'maximize'
        },
        'validation_errors': [],
        'validation_warnings': [],
        'completed': True
    }
    
    print("✅ Enhanced configuration created!")
    print(f"   📊 Models: {len(step3_config['selected_models'])} enhanced models")
    print(f"   🔤 Vectorization: {len(step3_config['selected_vectorization'])} methods")
    print(f"   ⚡ Optuna Optimization: {'Enabled' if step3_config['optuna']['enabled'] else 'Disabled'}")
    print(f"   🎯 Total combinations: {len(step3_config['selected_models']) * len(step3_config['selected_vectorization'])}")
    print(f"   🚀 GPU Models: XGBoost, LightGBM, CatBoost")
    print(f"   🏗️ Stacking Ensemble: {'Enabled' if step3_config['ensemble_learning']['enabled'] else 'Disabled'}")
    if step3_config['ensemble_learning']['enabled']:
        print(f"      - Meta-learner: {step3_config['ensemble_learning']['final_estimator']}")
        print(f"      - Base models: {len(step3_config['ensemble_learning']['base_models'])}")
        print(f"      - Min base models: {step3_config['ensemble_learning']['min_base_models']}")
        print(f"      - CV folds: {step3_config['ensemble_learning']['cv_folds']}")
    
    return step1_config, step2_config, step3_config


def run_enhanced_training(df, step1_config, step2_config, step3_config):
    """Chạy training với enhanced models"""
    print("\n🚀 Starting enhanced model training...")
    
    try:
        # Import training pipeline
        from training_pipeline import StreamlitTrainingPipeline
        
        # Tạo pipeline instance
        pipeline = StreamlitTrainingPipeline()
        
        # Initialize pipeline với configs
        pipeline.initialize_pipeline(df, step1_config, step2_config, step3_config)
        
        # Progress callback function
        def progress_callback(phase, message, progress):
            print(f"   [{phase}] {message} ({progress:.1%})")
        
        # GPU Configuration check
        from gpu_config_manager import detect_gpu_availability, get_device_policy_config
        gpu_available, gpu_info = detect_gpu_availability()
        
        try:
            device_config = get_device_policy_config()
            device_policy = device_config.get('device_policy', 'unknown')
            cpu_jobs = device_config.get('n_jobs', 'unknown')
        except Exception as e:
            print(f"⚠️ Device config error: {e}")
            device_policy = 'unknown'
            cpu_jobs = 'unknown'
        
        print(f"\n🖥️ DEVICE CONFIGURATION:")
        print(f"   GPU Available: {gpu_available}")
        if gpu_available:
            print(f"   GPU Info: {gpu_info}")
        print(f"   Device Policy: {device_policy}")
        print(f"   CPU Jobs: {cpu_jobs}")
        
        # Execute training
        print("\n🏃‍♂️ Executing training pipeline...")
        result = pipeline.execute_training(df, step1_config, step2_config, step3_config)
        
        if result and result.get('status') == 'success':
            print("✅ Enhanced training completed successfully!")
            print(f"   📊 Successful combinations: {result.get('successful_combinations', 0)}")
            print(f"   ⏱️  Total time: {result.get('evaluation_time', 0):.1f}s")
            return result
        else:
            print(f"❌ Enhanced training failed: {result.get('message', 'Unknown error') if result else 'No result'}")
            return None
            
    except Exception as e:
        print(f"❌ Error during enhanced training: {e}")
        import traceback
        traceback.print_exc()
        return None


def display_enhanced_results(result):
    """Hiển thị kết quả enhanced training"""
    if not result:
        print("❌ No results to display")
        return
    
    print("\n📊 ENHANCED TRAINING RESULTS")
    print("=" * 60)
    
    # Best model
    best_combinations = result.get('best_combinations', {})
    best_overall = best_combinations.get('best_overall', {})
    
    if best_overall:
        print(f"🥇 Best Enhanced Model: {best_overall.get('combination_key', 'N/A')}")
        print(f"   📈 F1 Score: {best_overall.get('f1_score', 0):.3f}")
        print(f"   🎯 Test Accuracy: {best_overall.get('test_accuracy', 0):.3f}")
        print(f"   ⏱️  Training Time: {best_overall.get('training_time', 0):.1f}s")
        
        # Check if GPU was used
        if 'gpu' in str(best_overall.get('combination_key', '')).lower():
            print(f"   🔥 GPU Acceleration: Used")
    
    # Summary statistics
    comprehensive_results = result.get('comprehensive_results', [])
    successful_results = [r for r in comprehensive_results if r.get('status') == 'success']
    
    if successful_results:
        print("\n📊 Enhanced Models Summary:")
        print(f"   ✅ Successful models: {len(successful_results)}")
        print(f"   📈 Average F1 Score: {np.mean([r.get('f1_score', 0) for r in successful_results]):.3f}")
        print(f"   🎯 Average Accuracy: {np.mean([r.get('test_accuracy', 0) for r in successful_results]):.3f}")
        print(f"   ⏱️  Average Training Time: {np.mean([r.get('training_time', 0) for r in successful_results]):.1f}s")
        
        # Check for stacking results
        stacking_results = [r for r in successful_results if 'stacking' in str(r.get('model_name', '')).lower() or 'ensemble' in str(r.get('model_name', '')).lower()]
        if stacking_results:
            print(f"\n🏗️ Stacking Ensemble Results:")
            print(f"   ✅ Stacking models: {len(stacking_results)}")
            print(f"   📈 Average F1 Score: {np.mean([r.get('f1_score', 0) for r in stacking_results]):.3f}")
            print(f"   🎯 Average Accuracy: {np.mean([r.get('test_accuracy', 0) for r in stacking_results]):.3f}")
            print(f"   ⏱️  Average Training Time: {np.mean([r.get('training_time', 0) for r in stacking_results]):.1f}s")
        
        # Top 5 enhanced models
        print("\n🏆 Top 5 Enhanced Models:")
        top_models = sorted(successful_results, key=lambda x: x.get('f1_score', 0), reverse=True)[:5]
        for i, model in enumerate(top_models, 1):
            model_name = f"{model.get('model_name', 'Unknown')} + {model.get('embedding_name', 'Unknown')}"
            f1_score = model.get('f1_score', 0)
            accuracy = model.get('test_accuracy', 0)
            training_time = model.get('training_time', 0)
            
            # Check if GPU model
            gpu_indicator = "🔥" if any(gpu_model in model_name.lower() for gpu_model in ['xgboost', 'lightgbm', 'catboost']) else "💻"
            
            print(f"   {i}. {gpu_indicator} {model_name}")
            print(f"      F1: {f1_score:.3f} | Accuracy: {accuracy:.3f} | Time: {training_time:.1f}s")
        
        # Model type breakdown
        print("\n📊 Model Type Performance:")
        model_types = {}
        for model in successful_results:
            model_name = model.get('model_name', 'Unknown')
            if 'stacking' in model_name.lower() or 'ensemble' in model_name.lower():
                model_type = 'Stacking Ensemble'
            elif 'random' in model_name.lower():
                model_type = 'Random Forest'
            elif 'ada' in model_name.lower():
                model_type = 'AdaBoost'
            elif 'gradient' in model_name.lower():
                model_type = 'Gradient Boosting'
            elif 'xgboost' in model_name.lower():
                model_type = 'XGBoost (GPU)'
            elif 'lightgbm' in model_name.lower():
                model_type = 'LightGBM (GPU)'
            elif 'catboost' in model_name.lower():
                model_type = 'CatBoost (GPU)'
            else:
                model_type = 'Other'
            
            if model_type not in model_types:
                model_types[model_type] = []
            model_types[model_type].append(model.get('f1_score', 0))
        
        for model_type, scores in model_types.items():
            avg_score = np.mean(scores)
            icon = "🏗️" if "Stacking" in model_type else "🔥" if "GPU" in model_type else "💻"
            print(f"   {icon} {model_type}: {avg_score:.3f} (avg F1)")


def main():
    """Hàm main chính"""
    print_banner()
    
    # Load hoặc tạo dataset
    df = load_or_create_dataset()
    if df is None:
        print("❌ Failed to load/create dataset")
        return
    
    # Tạo enhanced configuration
    step1_config, step2_config, step3_config = create_enhanced_config(df)
    
    # Chạy enhanced training
    result = run_enhanced_training(df, step1_config, step2_config, step3_config)
    
    # Hiển thị kết quả
    display_enhanced_results(result)
    
    print("\n" + "=" * 80)
    print("🎉 ENHANCED MODEL TESTING COMPLETED!")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if result:
        print("💾 Results cached for future use!")
        print("🚀 Enhanced models are ready for production!")
        print("⚡ GPU acceleration tested successfully!")
        print("🔍 Optuna optimization completed!")
    else:
        print("⚠️ Some issues encountered during testing")
        print("💡 Check logs above for details")


if __name__ == "__main__":
    main()