#!/usr/bin/env python3
"""
Comprehensive Test Script for Heart Dataset
Tests all models with numerical data preprocessing including ensemble/stacking
Using the heart dataset: cache/heart.csv
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import traceback
import json
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import project modules
try:
    from optuna_optimizer import OptunaOptimizer, OPTUNA_AVAILABLE
    from models import model_registry
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import (
        StandardScaler, LabelEncoder, MinMaxScaler
    )
    print("All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)



def load_heart_dataset(
    sample_size: int = None
) -> Tuple[pd.DataFrame, List[str], str]:
    """Load heart dataset"""
    try:
        print("📊 Loading heart dataset...")
        df = pd.read_csv('data/heart.csv')

        # Sample the dataset if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(
                drop=True
            )

        print(f"✅ Dataset loaded: {len(df)} samples, "
              f"{len(df.columns)} columns")
        print(f"📊 Columns: {df.columns.tolist()}")

        # Heart dataset structure: last column is target
        feature_columns = df.columns[:-1].tolist()
        label_column = df.columns[-1]  # 'target'

        print(f"📊 Feature columns: {feature_columns}")
        print(f"📊 Label column: {label_column}")
        print(f"📊 Class distribution: "
              f"{df[label_column].value_counts().to_dict()}")

        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(f"⚠️ Found {missing_values} missing values")
            df = df.dropna()
            print(f"📊 After dropping missing values: {len(df)} samples")

        # Data types info
        print("📊 Data types:")
        for col in df.columns:
            print(f"   {col}: {df[col].dtype}")

        return df, feature_columns, label_column

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise



def preprocess_numerical_data(
    X: pd.DataFrame, method: str, **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Preprocess numerical data using specified method"""
    try:
        print(f"📝 Preprocessing with {method}...")

        if method == "StandardScaler":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            return X_scaled, {
                'method': method,
                'features': X_scaled.shape[1],
                'scaler': 'StandardScaler'
            }

        elif method == "MinMaxScaler":
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            return X_scaled, {
                'method': method,
                'features': X_scaled.shape[1],
                'scaler': 'MinMaxScaler'
            }

        elif method == "NoScaling":
            X_array = X.values
            return X_array, {
                'method': method,
                'features': X_array.shape[1],
                'scaler': 'None'
            }

        else:
            raise ValueError(f"Unknown preprocessing method: {method}")

    except Exception as e:
        print(f"❌ Error preprocessing with {method}: {e}")
        raise



def get_all_models() -> List[str]:
    """Get all available models including ensemble models"""
    try:
        # Get all classification models from registry
        base_models = model_registry.list_models()
        classification_models = [m for m in base_models if m != 'kmeans']
        
        # Add ensemble models
        ensemble_models = [
            'voting_ensemble_hard',
            'voting_ensemble_soft',
            'stacking_ensemble_logistic_regression',
            'stacking_ensemble_random_forest',
            'stacking_ensemble_xgboost'
        ]

        all_models = classification_models + ensemble_models

        print(f"📋 Base models: {classification_models}")
        print(f"📋 Ensemble models: {ensemble_models}")
        print(f"📋 Total models: {len(all_models)}")

        return all_models

    except Exception as e:
        print(f"❌ Error getting models: {e}")
        return []

def get_all_preprocessing_methods() -> List[Dict[str, Any]]:
    """Get all available preprocessing methods with configurations"""
    try:
        preprocessing_methods = [
            {
                'name': 'StandardScaler',
                'params': {}
            },
            {
                'name': 'MinMaxScaler', 
                'params': {}
            },
            {
                'name': 'NoScaling',
                'params': {}
            }
        ]
        
        print(f"📝 Available preprocessing methods: {[m['name'] for m in preprocessing_methods]}")
        return preprocessing_methods
        
    except Exception as e:
        print(f"❌ Error getting preprocessing methods: {e}")
        return []

def test_model_with_preprocessing(model_name: str, X: np.ndarray, y: np.ndarray, 
                                preprocessing_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single model with specific preprocessing method using cache system"""
    try:
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {model_name} + {preprocessing_info['method']}")
        print(f"{'='*60}")
        
        # Split data: 80% train, 10% val, 10% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        print(f"📊 Data split: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
        
        # Import cache manager
        from cache_manager import CacheManager
        import os
        
        # Create cache manager
        cache_manager = CacheManager()
        
        # Generate cache identifiers
        model_key = model_name
        dataset_id = f"heart_dataset_{preprocessing_info['method']}"
        config_hash = cache_manager.generate_config_hash({
            'model': model_name,
            'preprocessing': preprocessing_info['method'],
            'trials': config.get('trials', 10),
            'random_state': 42
        })
        dataset_fingerprint = cache_manager.generate_dataset_fingerprint(
            dataset_path="data/heart.csv",
            dataset_size=os.path.getsize("data/heart.csv"),
            num_rows=len(X_train)
        )
        
        # Check if cache exists
        cache_exists, cached_data = cache_manager.check_cache_exists(
            model_key, dataset_id, config_hash, dataset_fingerprint
        )
        
        if cache_exists:
            # Load full cache data including metrics
            try:
                full_cache_data = cache_manager.load_model_cache(model_key, dataset_id, config_hash)
                cached_metrics = full_cache_data.get('metrics', {})
                has_enhanced_metrics = 'cv_mean' in cached_metrics and 'cv_std' in cached_metrics
                
                if has_enhanced_metrics:
                    print(f"💾 Cache hit! Loading cached results for {model_name}")
                    return {
                        'model': model_name,
                        'preprocessing': preprocessing_info['method'],
                        'score': cached_metrics.get('accuracy', 0.0),
                        'params': full_cache_data.get('params', {}),
                        'time': 0.0,  # Cached, no training time
                        'features': preprocessing_info['features'],
                        'status': 'SUCCESS',
                        'cached': True,
                        'cv_mean': cached_metrics.get('cv_mean', 0.0),
                        'cv_std': cached_metrics.get('cv_std', 0.0),
                        'f1_score': cached_metrics.get('f1_score', 0.0),
                        'precision': cached_metrics.get('precision', 0.0),
                        'recall': cached_metrics.get('recall', 0.0)
                    }
                else:
                    print(f"🔄 Cache cũ detected! Retraining {model_name} with enhanced features...")
                    # Continue to training with enhanced features
            except Exception as e:
                print(f"⚠️ Error loading cache data: {e}")
                print(f"🔄 Cache miss! Training {model_name} with Optuna...")
                # Continue to training
        
        print(f"🔄 Cache miss! Training {model_name} with Optuna...")
        
        # Create Optuna optimizer
        optimizer = OptunaOptimizer(config)
        print(f"✅ Optuna optimizer created")
        
        # Get model class first
        model_class = model_registry.get_model(model_name)
        if not model_class:
            raise ValueError(f"Model {model_name} not found")
        
        # Optimize model using the same pipeline as app.py
        start_time = time.time()
        optimization_result = optimizer.optimize_model(
            model_name=model_name,
            model_class=model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )
        end_time = time.time()
        
        best_score = optimization_result['best_score']
        best_params = optimization_result['best_params']
        
        print(f"✅ Optimization completed in {end_time - start_time:.2f}s")
        print(f"📊 Best score: {best_score:.4f}")
        print(f"📊 Best params: {best_params}")
        
        # Train final model with best params for caching and CV
        print("🔄 Training final model with best params for cache and CV...")
        final_model = model_class(**best_params)
        
        # Special handling for ensemble models
        if model_name.startswith(('voting_ensemble', 'stacking_ensemble')):
            # Create base estimators for ensemble
            base_estimators = []
            for model_name_base in ['knn', 'decision_tree', 'naive_bayes']:
                try:
                    model_class_base = model_registry.get_model(model_name_base)
                    if model_class_base:
                        model_instance_base = model_class_base()
                        base_estimators.append((model_name_base, model_instance_base))
                except Exception as e:
                    print(f"⚠️ Error creating {model_name_base}: {e}")
            
            # Create the ensemble classifier
            if base_estimators:
                final_model.create_stacking_classifier(base_estimators)
        
        final_model.fit(X_train, y_train)
        
        # Final evaluation on test set
        print("🔄 Calculating final metrics on test set...")
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        
        y_pred_test = final_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average='weighted')
        precision = precision_score(y_test, y_pred_test, average='weighted')
        recall = recall_score(y_test, y_pred_test, average='weighted')
        
        print(f"📊 Test Accuracy: {test_accuracy:.4f}")
        print(f"📊 Test F1-Score: {f1:.4f}")
        print(f"📊 Test Precision: {precision:.4f}")
        print(f"📊 Test Recall: {recall:.4f}")
        
        # Save to cache
        if final_model is not None:
            try:
                metrics = {
                    'accuracy': test_accuracy,  # Use test accuracy as final metric
                    'validation_accuracy': best_score,  # Keep validation accuracy for reference
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall
                }
                
                cache_config = {
                    'model': model_name,
                    'preprocessing': preprocessing_info['method'],
                    'trials': config.get('trials', 10),
                    'random_state': 42,
                    'test_size': 0.1
                }
                
                cache_path = cache_manager.save_model_cache(
                    model_key=model_key,
                    dataset_id=dataset_id,
                    config_hash=config_hash,
                    dataset_fingerprint=dataset_fingerprint,
                    model=final_model,
                    params=best_params,
                    metrics=metrics,
                    config=cache_config,
                    feature_names=[f"feature_{i}" for i in range(X.shape[1])],
                    label_mapping={0: 'no_heart_disease', 1: 'heart_disease'}
                )
                
                print(f"💾 Cache saved for {model_name} at {cache_path}")
                
            except Exception as cache_error:
                print(f"⚠️ Cache save failed: {cache_error}")
        
        return {
            'model': model_name,
            'preprocessing': preprocessing_info['method'],
            'score': test_accuracy,  # Use test accuracy as final score
            'validation_score': best_score,  # Keep validation score for reference
            'params': best_params,
            'time': end_time - start_time,
            'features': preprocessing_info['features'],
            'status': 'SUCCESS',
            'cached': False,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }
        
    except Exception as e:
        print(f"❌ Error testing {model_name} + {preprocessing_info['method']}: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        return {
            'model': model_name,
            'preprocessing': preprocessing_info['method'],
            'score': 0.0,
            'params': {},
            'time': 0.0,
            'features': 0,
            'status': 'FAILED',
            'error': str(e)
        }

def test_with_app_pipeline(df: pd.DataFrame, feature_columns: List[str], label_column: str) -> Dict[str, Any]:
    """Test using the same pipeline as app.py to ensure cache creation"""
    try:
        print(f"\n{'='*80}")
        print("🚀 TESTING WITH APP.PY PIPELINE (WITH CACHE)")
        print(f"{'='*80}")
        
        # Create step data similar to app.py for numerical data
        step1_data = {
            'dataframe': df,
            'uploaded_file': {'name': 'heart.csv'},
            'selected_categories': sorted(df[label_column].unique().tolist()),
            'sampling_config': {'num_samples': len(df)},
            'is_single_input': False,  # Numerical data, not single text input
            'feature_columns': feature_columns,
            'label_column': label_column
        }
        
        step2_data = {
            'preprocessing_config': {
                'numerical_preprocessing': True,
                'scaling_methods': ['StandardScaler', 'MinMaxScaler', 'NoScaling']
            }
        }
        
        step3_data = {
            'optuna_config': {
                'trials': 2,
                'timeout': 30,
                'direction': 'maximize'
            },
            'selected_models': get_all_models(),  # Test all models
            'preprocessing_config': {
                'selected_methods': ['StandardScaler', 'MinMaxScaler', 'NoScaling']
            }
        }
        
        # For numerical data, we'll test directly without using streamlit training
        # since it's designed for text data
        print("🔄 Testing numerical data pipeline...")
        
        # Extract features and labels
        X = df[feature_columns]
        y = df[label_column]
        
        # Test with StandardScaler
        X_scaled, preprocessing_info = preprocess_numerical_data(X, 'StandardScaler')
        
        # Test one model as proof of concept
        test_model = get_all_models()[0]
        config = {
            'trials': 2,
            'timeout': 30,
            'direction': 'maximize',
            'study_name': 'heart_dataset_test'
        }
        
        result = test_model_with_preprocessing(test_model, X_scaled, y.values, preprocessing_info, config)
        
        if result['status'] == 'SUCCESS':
            print("✅ Heart dataset pipeline test successful!")
            return {
                'status': 'success',
                'results': result,
                'cache_created': True,
                'message': 'Used numerical data pipeline with preprocessing'
            }
        else:
            print(f"❌ Heart dataset pipeline test failed: {result.get('error', 'Unknown error')}")
            return {
                'status': 'failed',
                'error': result.get('error', 'Unknown error'),
                'cache_created': False
            }
        
    except Exception as e:
        print(f"❌ Error testing with numerical pipeline: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        return {
            'status': 'failed',
            'error': str(e),
            'cache_created': False
        }

def test_all_combinations(df: pd.DataFrame, feature_columns: List[str], label_column: str) -> Dict[str, Any]:
    """Test all model and preprocessing combinations"""
    try:
        print(f"\n{'='*80}")
        print("🚀 COMPREHENSIVE TESTING: All Models × All Preprocessing Methods")
        print(f"{'='*80}")
        
        # Extract features and labels
        X = df[feature_columns]
        y = df[label_column].values
        
        # Encode labels if necessary
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        # Get all models and preprocessing methods
        all_models = get_all_models()
        all_preprocessing_methods = get_all_preprocessing_methods()
        
        # Test all models for comprehensive evaluation
        test_models = all_models  # Test all models
        
        # Optuna config
        config = {
            'trials': 10,  # Increased to 10 trials
            'timeout': 60,
            'direction': 'maximize',
            'study_name': 'heart_dataset_comprehensive_test'
        }
        
        all_results = []
        total_combinations = len(test_models) * len(all_preprocessing_methods)
        current_combination = 0
        
        print(f"📊 Total combinations to test: {total_combinations}")
        print(f"🚀 Starting Optuna optimization for all models...")
        print(f"⏱️ This may take several minutes. Please wait...")
        
        # Test each preprocessing method
        for preprocessing_method in all_preprocessing_methods:
            print(f"\n📝 Testing Preprocessing Method: {preprocessing_method['name']}")
            print("-" * 50)
            
            # Preprocess data
            X_processed, preprocessing_info = preprocess_numerical_data(X, preprocessing_method['name'], **preprocessing_method['params'])
            print(f"✅ Preprocessed: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
            
            # Test base models
            base_models = [m for m in test_models if not m.startswith(('voting_ensemble', 'stacking_ensemble'))]
            print(f"📋 Testing {len(base_models)} base models...")
            
            for model_name in base_models:
                current_combination += 1
                print(f"\n[{current_combination}/{total_combinations}] Testing: {model_name} + {preprocessing_method['name']}")
                
                result = test_model_with_preprocessing(model_name, X_processed, y, preprocessing_info, config)
                all_results.append(result)
                
                print(f"✅ Completed: {model_name} + {preprocessing_method['name']} ({current_combination}/{total_combinations})")
            
            # Test ensemble models (if any in test_models)
            ensemble_models = [m for m in test_models if m.startswith(('voting_ensemble', 'stacking_ensemble'))]
            if ensemble_models:
                print(f"📋 Testing {len(ensemble_models)} ensemble models...")
                
                for model_name in ensemble_models:
                    current_combination += 1
                    print(f"\n[{current_combination}/{total_combinations}] Testing: {model_name} + {preprocessing_method['name']}")
                    
                    result = test_model_with_preprocessing(model_name, X_processed, y, preprocessing_info, config)
                    all_results.append(result)
                    
                    print(f"✅ Completed: {model_name} + {preprocessing_method['name']} ({current_combination}/{total_combinations})")
        
        print(f"\n🎉 All Optuna optimization completed!")
        print(f"📊 Total combinations trained: {len(all_results)}")
        print(f"🚀 Now generating final results table...")
        
        return {
            'all_results': all_results,
            'total_combinations': total_combinations,
            'models_tested': test_models,
            'preprocessing_methods': all_preprocessing_methods
        }
        
    except Exception as e:
        print(f"❌ Error in comprehensive testing: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        return {}

def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze test results and generate insights"""
    try:
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE RESULTS ANALYSIS")
        print(f"{'='*80}")
        
        all_results = results['all_results']
        
        # Overall statistics
        successful_results = [r for r in all_results if r['status'] == 'SUCCESS']
        failed_results = [r for r in all_results if r['status'] == 'FAILED']
        
        print(f"📊 Overall Statistics:")
        print(f"   Total combinations tested: {len(all_results)}")
        print(f"   Successful: {len(successful_results)}")
        print(f"   Failed: {len(failed_results)}")
        print(f"   Success rate: {len(successful_results)/len(all_results)*100:.1f}%")
        
        # Best performing combinations
        if successful_results:
            best_results = sorted(successful_results, key=lambda x: x['score'], reverse=True)[:10]
            print(f"\n🏆 Top 10 Best Performing Combinations:")
            for i, result in enumerate(best_results, 1):
                print(f"   {i:2d}. {result['model']} + {result['preprocessing']}: {result['score']:.4f}")
        
        # Analysis by preprocessing method
        print(f"\n📊 Performance by Preprocessing Method:")
        preprocessing_stats = {}
        for result in successful_results:
            method = result['preprocessing']
            if method not in preprocessing_stats:
                preprocessing_stats[method] = {'scores': [], 'count': 0}
            preprocessing_stats[method]['scores'].append(result['score'])
            preprocessing_stats[method]['count'] += 1
        
        for method, stats in preprocessing_stats.items():
            avg_score = np.mean(stats['scores'])
            max_score = np.max(stats['scores'])
            print(f"   {method}: Avg={avg_score:.4f}, Max={max_score:.4f}, Count={stats['count']}")
        
        # Analysis by model type
        print(f"\n📊 Performance by Model Type:")
        model_stats = {}
        for result in successful_results:
            model = result['model']
            if model not in model_stats:
                model_stats[model] = {'scores': [], 'count': 0}
            model_stats[model]['scores'].append(result['score'])
            model_stats[model]['count'] += 1
        
        # Sort by average score
        sorted_models = sorted(model_stats.items(), key=lambda x: np.mean(x[1]['scores']), reverse=True)
        for model, stats in sorted_models:
            avg_score = np.mean(stats['scores'])
            max_score = np.max(stats['scores'])
            print(f"   {model}: Avg={avg_score:.4f}, Max={max_score:.4f}, Count={stats['count']}")
        
        return {
            'overall_stats': {
                'total': len(all_results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'success_rate': len(successful_results)/len(all_results)*100
            },
            'best_combinations': best_results[:10] if successful_results else [],
            'preprocessing_stats': preprocessing_stats,
            'model_stats': model_stats
        }
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return {}

def main():
    """Main test function"""
    print("Comprehensive Heart Dataset Test Suite")
    print("=" * 70)
    
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna is not available. Please install optuna.")
        return
    
    print("✅ Optuna is available")
    print("🚀 Starting comprehensive test with heart dataset...")
    print("=" * 80)
    
    try:
        # 1. Load dataset
        print("\n1️⃣ Loading Heart Dataset...")
        df, feature_columns, label_column = load_heart_dataset()
        
        # 2. Test with numerical pipeline
        print("\n2️⃣ Testing with numerical data pipeline...")
        app_pipeline_results = test_with_app_pipeline(df, feature_columns, label_column)
        
        if app_pipeline_results.get('status') == 'success':
            print("✅ Numerical pipeline test successful!")
        else:
            print(f"❌ Numerical pipeline test failed: {app_pipeline_results.get('error')}")
        
        # 3. Run comprehensive testing
        print("\n3️⃣ Running comprehensive testing...")
        results = test_all_combinations(df, feature_columns, label_column)
        
        if not results:
            print("❌ No results obtained from testing")
            return
        
        # 4. Analyze results
        print("\n4️⃣ Analyzing results...")
        analysis = analyze_results(results)
        
        # 5. Save results
        print("\n5️⃣ Saving results...")
        results_data = {
            'dataset_info': {
                'name': 'heart_dataset',
                'samples': len(df),
                'feature_columns': feature_columns,
                'label_column': label_column,
                'source_file': 'data/heart.csv'
            },
            'test_config': {
                'trials': 10,
                'timeout': 60,
                'direction': 'maximize'
            },
            'numerical_pipeline_results': app_pipeline_results,
            'comprehensive_results': results,
            'analysis': analysis
        }
        
        # Convert DataFrame objects to serializable format
        def make_serializable(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif hasattr(obj, '__class__') and 'Model' in obj.__class__.__name__:
                return f"<{obj.__class__.__name__} object>"
            elif hasattr(obj, '__dict__'):
                return f"<{obj.__class__.__name__} object>"
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return obj
        
        # Make results serializable
        serializable_results = make_serializable(results_data)
        
        # Ensure training_results directory exists
        os.makedirs('cache/training_results', exist_ok=True)
        
        with open('cache/training_results/comprehensive_heart_dataset_results.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: cache/training_results/comprehensive_heart_dataset_results.json")
        
        # 6. Final summary
        print(f"\n{'='*80}")
        print("🎯 FINAL SUMMARY")
        print(f"{'='*80}")
        
        print(f"📊 Numerical Pipeline Test:")
        print(f"   Status: {'✅ SUCCESS' if app_pipeline_results.get('status') == 'success' else '❌ FAILED'}")
        print(f"   Cache Created: {'✅ YES' if app_pipeline_results.get('cache_created') else '❌ NO'}")
        
        if analysis.get('overall_stats'):
            stats = analysis['overall_stats']
            print(f"\n📊 Comprehensive Test:")
            print(f"   Total combinations tested: {stats['total']}")
            print(f"   Success rate: {stats['success_rate']:.1f}%")
            if analysis['best_combinations']:
                best = analysis['best_combinations'][0]
                print(f"   Best performing combination: {best['model']} + {best['preprocessing']} = {best['score']:.4f}")
        
        print(f"\n🎉 Heart dataset comprehensive testing completed!")
        
        # Debug: Show detailed results and errors
        print(f"\n🔍 DETAILED DEBUG INFORMATION:")
        
        # Use actual results from the test run
        all_results = results['all_results']
        successful_results = [r for r in all_results if r['status'] == 'SUCCESS']
        failed_results = [r for r in all_results if r['status'] == 'FAILED']
        
        # Show successful results
        if successful_results:
            print(f"\n✅ SUCCESSFUL COMBINATIONS ({len(successful_results)}):")
            for i, result in enumerate(successful_results[:10], 1):  # Show first 10
                print(f"  {i}. {result['model']} + {result['preprocessing']}: {result['score']:.4f}")
                print(f"     CV: {result.get('cv_mean', 'N/A')} ± {result.get('cv_std', 'N/A')}")
                print(f"     F1: {result.get('f1_score', 'N/A')}, Precision: {result.get('precision', 'N/A')}, Recall: {result.get('recall', 'N/A')}")
                print(f"     Time: {result.get('time', 0):.2f}s, Cached: {result.get('cached', False)}")
            
            if len(successful_results) > 10:
                print(f"     ... and {len(successful_results) - 10} more successful combinations")
            
            # Show failed results with error analysis
            if failed_results:
                print(f"\n❌ FAILED COMBINATIONS ({len(failed_results)}):")
                error_counts = {}
                for result in failed_results:
                    error = result.get('error', 'Unknown error')
                    error_counts[error] = error_counts.get(error, 0) + 1
                
                print(f"\n🔍 ERROR ANALYSIS:")
                for error, count in error_counts.items():
                    print(f"  • {error}: {count} times")
                
                print(f"\n🔍 DETAILED FAILED RESULTS:")
                for i, result in enumerate(failed_results[:5], 1):  # Show first 5
                    print(f"  {i}. {result['model']} + {result['preprocessing']}")
                    print(f"     Error: {result.get('error', 'Unknown error')}")
            
            # Debug: Show cache statistics
            print(f"\n💾 CACHE STATISTICS:")
            cache_hits = len([r for r in all_results if r.get('cached', False)])
            cache_misses = len([r for r in all_results if not r.get('cached', False)])
            print(f"  • Cache hits: {cache_hits}")
            print(f"  • Cache misses: {cache_misses}")
            print(f"  • Cache hit rate: {cache_hits/(cache_hits+cache_misses)*100:.1f}%" if (cache_hits+cache_misses) > 0 else "  • Cache hit rate: N/A")
            
            # Debug: Show performance statistics
            if successful_results:
                scores = [r['score'] for r in successful_results]
                times = [r.get('time', 0) for r in successful_results]
                print(f"\n📈 PERFORMANCE STATISTICS:")
                print(f"  • Best accuracy: {max(scores):.4f}")
                print(f"  • Worst accuracy: {min(scores):.4f}")
                print(f"  • Average accuracy: {sum(scores)/len(scores):.4f}")
                print(f"  • Average training time: {sum(times)/len(times):.2f}s")
                print(f"  • Total training time: {sum(times):.2f}s")
            
            # Debug: Show top performing models
            if successful_results:
                print(f"\n🏆 TOP 5 PERFORMING MODELS:")
                sorted_results = sorted(successful_results, key=lambda x: x['score'], reverse=True)
                for i, result in enumerate(sorted_results[:5], 1):
                    print(f"  {i}. {result['model']} + {result['preprocessing']}: {result['score']:.4f}")
                    print(f"     CV: {result.get('cv_mean', 'N/A')} ± {result.get('cv_std', 'N/A')}")
                    print(f"     F1: {result.get('f1_score', 'N/A')}, Precision: {result.get('precision', 'N/A')}, Recall: {result.get('recall', 'N/A')}")
            
            # Debug: Show preprocessing method performance
            if successful_results:
                print(f"\n🔧 PREPROCESSING METHOD ANALYSIS:")
                preprocessing_stats = {}
                for result in successful_results:
                    method = result['preprocessing']
                    if method not in preprocessing_stats:
                        preprocessing_stats[method] = []
                    preprocessing_stats[method].append(result['score'])
                
                for method, scores in preprocessing_stats.items():
                    avg_score = sum(scores) / len(scores)
                    print(f"  • {method}: {len(scores)} models, avg accuracy: {avg_score:.4f}")
            
            # Debug: Show model type performance
            if successful_results:
                print(f"\n🤖 MODEL TYPE ANALYSIS:")
                model_stats = {}
                for result in successful_results:
                    model = result['model']
                    if model not in model_stats:
                        model_stats[model] = []
                    model_stats[model].append(result['score'])
                
                for model, scores in model_stats.items():
                    avg_score = sum(scores) / len(scores)
                    print(f"  • {model}: {len(scores)} preprocessing methods, avg accuracy: {avg_score:.4f}")
            
            print(f"\n🎯 DEBUG SUMMARY:")
            print(f"  • Total tests: {len(all_results)}")
            print(f"  • Success rate: {len(successful_results)/len(all_results)*100:.1f}%" if len(all_results) > 0 else "  • Success rate: N/A")
            print(f"  • Cache efficiency: {cache_hits/(cache_hits+cache_misses)*100:.1f}%" if (cache_hits+cache_misses) > 0 else "  • Cache efficiency: N/A")
            print(f"  • Best model: {sorted_results[0]['model']} + {sorted_results[0]['preprocessing']} = {sorted_results[0]['score']:.4f}" if successful_results else "  • Best model: N/A")
        else:
            print(f"  • No analysis data available for debug information")
            print(f"  • Analysis object: {type(analysis)}")
            if analysis:
                print(f"  • Analysis keys: {list(analysis.keys()) if hasattr(analysis, 'keys') else 'No keys method'}")
        
        # 7. Check cache creation
        print(f"\n7️⃣ Checking cache creation...")
        check_cache_creation()
        
    except Exception as e:
        print(f"❌ Critical error in main: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")


def check_cache_creation():
    """Check what cache files were created"""
    try:
        print(f"\n{'='*60}")
        print("📁 CACHE CREATION CHECK")
        print(f"{'='*60}")
        
        import os
        
        # Check cache/models/ (per-model cache)
        cache_models_dir = "cache/models"
        if os.path.exists(cache_models_dir):
            print(f"✅ {cache_models_dir} exists")
            model_count = 0
            for root, dirs, files in os.walk(cache_models_dir):
                if files:  # Only count directories with files
                    model_count += 1
            print(f"📊 Per-model cache entries: {model_count}")
        else:
            print(f"❌ {cache_models_dir} does not exist")
        
        # Check cache/training_results/ (overall cache)
        cache_training_dir = "cache/training_results"
        if os.path.exists(cache_training_dir):
            print(f"\n✅ {cache_training_dir} exists")
            files = os.listdir(cache_training_dir)
            print(f"📊 Training results cache files: {len(files)}")
        else:
            print(f"\n❌ {cache_training_dir} does not exist")
        
        # Summary
        print(f"\n📊 Cache Summary:")
        print(f"   Per-model cache: {'✅ YES' if os.path.exists(cache_models_dir) else '❌ NO'}")
        print(f"   Training results cache: {'✅ YES' if os.path.exists(cache_training_dir) else '❌ NO'}")
        
    except Exception as e:
        print(f"❌ Error checking cache: {e}")


if __name__ == "__main__":
    main()
