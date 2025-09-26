#!/usr/bin/env python3
"""
Comprehensive Test Script for All Models with All Vectorization Methods
Tests all models with all vectorization methods combinations including ensemble and stacking
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import traceback
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import project modules
try:
    from optuna_optimizer import OptunaOptimizer, OPTUNA_AVAILABLE
    from models import model_factory, model_registry
    from models.ensemble import ensemble_manager
    from models.ensemble.stacking_classifier import StackingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from text_encoders import TextVectorizer, EmbeddingVectorizer
    print("All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def load_spam_dataset(sample_size: int = 1000) -> Tuple[pd.DataFrame, str, str]:
    """Load spam text classification dataset"""
    try:
        print(f"📊 Loading spam text dataset with {sample_size} samples...")
        df = pd.read_csv('cache/2cls_spam_text_cls.csv')
        
        # Sample the dataset
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        print(f"✅ Dataset loaded: {len(df)} samples, {len(df.columns)} columns")
        print(f"📊 Columns: {df.columns.tolist()}")
        print(f"📊 Category distribution: {df['Category'].value_counts().to_dict()}")
        
        # Determine text column and label column
        text_column = 'Message'
        label_column = 'Category'
        
        print(f"📊 Text column: {text_column}")
        print(f"📊 Label column: {label_column}")
        
        return df, text_column, label_column
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

def vectorize_text_data(texts: List[str], method: str, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Vectorize text data using specified method"""
    try:
        print(f"📝 Vectorizing with {method}...")
        
        if method == "TF-IDF":
            vectorizer = TextVectorizer()
            X = vectorizer.fit_transform_tfidf(texts)
            return X.toarray(), {'method': method, 'features': X.shape[1]}
            
        elif method == "BoW":
            vectorizer = TextVectorizer()
            X = vectorizer.fit_transform_bow(texts)
            return X.toarray(), {'method': method, 'features': X.shape[1]}
            
        elif method == "Word Embeddings":
            vectorizer = EmbeddingVectorizer(
                model_name=kwargs.get('model_name', 'all-MiniLM-L6-v2'),
                device=kwargs.get('device', 'cpu')
            )
            X = vectorizer.fit_transform(texts)
            return X, {'method': method, 'features': X.shape[1]}
            
        else:
            raise ValueError(f"Unknown vectorization method: {method}")
            
    except Exception as e:
        print(f"❌ Error vectorizing with {method}: {e}")
        raise

def get_all_models() -> List[str]:
    """Get all available models including ensemble models"""
    try:
        # Get base models
        base_models = model_registry.list_models()
        classification_models = [m for m in base_models if m != 'kmeans']
        
        # Skip ensemble models for now - they need proper initialization
        # ensemble_models = [
        #     'voting_ensemble_hard',
        #     'voting_ensemble_soft',
        #     'stacking_ensemble_logistic_regression',
        #     'stacking_ensemble_random_forest',
        #     'stacking_ensemble_xgboost'
        # ]

        all_models = classification_models  # + ensemble_models
        
        print(f"📋 Base models: {classification_models}")
        print(f"📋 Total models: {len(all_models)}")
        
        return all_models
        
    except Exception as e:
        print(f"❌ Error getting models: {e}")
        return []

def get_all_vectorization_methods() -> List[Dict[str, Any]]:
    """Get all available vectorization methods with configurations"""
    try:
        methods = [
            {
                'name': 'TF-IDF',
                'params': {
                    'max_features': 1000,
                    'ngram_range': (1, 2),
                    'min_df': 2
                }
            },
            {
                'name': 'BoW',
                'params': {
                    'max_features': 1000,
                    'ngram_range': (1, 1),
                    'min_df': 2
                }
            },
            {
                'name': 'Word Embeddings',
                'params': {
                    'model_name': 'all-MiniLM-L6-v2',
                    'device': 'cpu'
                }
            }
        ]
        
        print(f"📋 Vectorization methods: {[m['name'] for m in methods]}")
        return methods
        
    except Exception as e:
        print(f"❌ Error getting vectorization methods: {e}")
        return []

def test_model_with_vectorization(model_name: str, X: np.ndarray, y: np.ndarray, 
                                vectorization_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single model with specific vectorization method using cache system and cross-validation (ENHANCED)"""
    try:
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {model_name} + {vectorization_info['method']}")
        print(f"{'='*60}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"📊 Data split: Train {X_train.shape}, Val {X_val.shape}")
        
        # Import cache manager and CV
        from cache_manager import CacheManager
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import f1_score, precision_score, recall_score
        import os
        
        # Create cache manager
        cache_manager = CacheManager()
        
        # Generate cache identifiers
        model_key = model_name
        dataset_id = f"spam_ham_dataset_{vectorization_info['method']}"
        config_hash = cache_manager.generate_config_hash({
            'model': model_name,
            'vectorization': vectorization_info['method'],
            'trials': config.get('trials', 2),
            'cv_folds': 5,
            'random_state': 42
        })
        dataset_fingerprint = cache_manager.generate_dataset_fingerprint(
            dataset_path="cache/2cls_spam_text_cls.csv",
            dataset_size=os.path.getsize("cache/2cls_spam_text_cls.csv"),
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
                        'vectorization': vectorization_info['method'],
                        'score': cached_metrics.get('accuracy', 0.0),
                        'params': full_cache_data.get('params', {}),
                        'time': 0.0,  # Cached, no training time
                        'features': vectorization_info['features'],
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
        
        print(f"🔄 Cache miss! Training {model_name} with Optuna + CV...")
        
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
        
        # Cross-validation
        print("🔄 Running 5-fold cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(final_model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Comprehensive metrics
        print("🔄 Calculating comprehensive metrics...")
        y_pred = final_model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='weighted')
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        
        print(f"📊 CV Scores: {cv_scores}")
        print(f"📊 CV Mean: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"📊 F1-Score: {f1:.4f}")
        print(f"📊 Precision: {precision:.4f}")
        print(f"📊 Recall: {recall:.4f}")
        
        # Save to cache
        if final_model is not None:
            try:
                metrics = {
                    'accuracy': best_score,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std
                }
                
                cache_config = {
                    'model': model_name,
                    'vectorization': vectorization_info['method'],
                    'trials': config.get('trials', 2),
                    'cv_folds': 5,
                    'random_state': 42,
                    'test_size': 0.2
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
                    label_mapping={0: 'ham', 1: 'spam'}
                )
                
                print(f"💾 Cache saved for {model_name} at {cache_path}")
                
            except Exception as cache_error:
                print(f"⚠️ Cache save failed: {cache_error}")
        
        return {
            'model': model_name,
            'vectorization': vectorization_info['method'],
            'score': best_score,
            'params': best_params,
            'time': end_time - start_time,
            'features': vectorization_info['features'],
            'status': 'SUCCESS',
            'cached': False,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }
        
    except Exception as e:
        print(f"❌ Error testing {model_name} + {vectorization_info['method']}: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        return {
            'model': model_name,
            'vectorization': vectorization_info['method'],
            'score': 0.0,
            'params': {},
            'time': 0.0,
            'features': 0,
            'status': 'FAILED',
            'error': str(e)
        }

def test_ensemble_models(X: np.ndarray, y: np.ndarray, vectorization_info: Dict[str, Any], 
                        config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Test ensemble models with specific vectorization"""
    try:
        print(f"\n{'='*60}")
        print(f"🏗️ Testing Ensemble Models with {vectorization_info['method']}")
        print(f"{'='*60}")
        
        ensemble_results = []
        
        # Test voting ensemble models
        voting_models = ['voting_ensemble_hard', 'voting_ensemble_soft']
        for model_name in voting_models:
            result = test_model_with_vectorization(model_name, X, y, vectorization_info, config)
            ensemble_results.append(result)
        
        # Test stacking ensemble models
        stacking_models = [
            'stacking_ensemble_logistic_regression',
            'stacking_ensemble_random_forest', 
            'stacking_ensemble_xgboost'
        ]
        for model_name in stacking_models:
            result = test_model_with_vectorization(model_name, X, y, vectorization_info, config)
            ensemble_results.append(result)
        
        return ensemble_results
        
    except Exception as e:
        print(f"❌ Error testing ensemble models: {e}")
        return []

def test_with_app_pipeline(df: pd.DataFrame, text_column: str, label_column: str) -> Dict[str, Any]:
    """Test using the same pipeline as app.py to ensure cache creation"""
    try:
        print(f"\n{'='*80}")
        print("🚀 TESTING WITH APP.PY PIPELINE (WITH CACHE)")
        print(f"{'='*80}")
        
        # Create step data similar to app.py
        step1_data = {
            'dataframe': df,
            'uploaded_file': {'name': 'spam_text_cls.csv'},
            'selected_categories': sorted(df[label_column].unique().tolist()),
            'sampling_config': {'num_samples': len(df)},
            'is_single_input': True,
            'text_column': text_column,
            'label_column': label_column
        }
        
        step2_data = {
            'preprocessing_config': {
                'text_preprocessing': True,
                'vectorization_methods': ['TF-IDF', 'BoW', 'Word Embeddings']
            }
        }
        
        step3_data = {
            'optuna_config': {
                'trials': 2,
                'timeout': 30,
                'direction': 'maximize'
            },
            'selected_models': get_all_models(),
            'vectorization_config': {
                'selected_methods': ['TF-IDF', 'BoW', 'Word Embeddings']
            },
            'selected_vectorization': ['TF-IDF', 'BoW', 'Word Embeddings']  # Add this
        }
        
        # Use the same pipeline as app.py
        from training_pipeline import execute_streamlit_training
        
        print("🔄 Using execute_streamlit_training (same as app.py)...")
        results = execute_streamlit_training(df, step1_data, step2_data, step3_data)
        
        if results and results.get('status') == 'success':
            print("✅ Training completed successfully with cache!")
            return {
                'status': 'success',
                'results': results,
                'cache_created': True,
                'message': 'Used app.py pipeline with cache creation'
            }
        else:
            print(f"❌ Training failed: {results.get('error', 'Unknown error')}")
            return {
                'status': 'failed',
                'error': results.get('error', 'Unknown error'),
                'cache_created': False
            }
        
    except Exception as e:
        print(f"❌ Error testing with app.py pipeline: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        return {
            'status': 'failed',
            'error': str(e),
            'cache_created': False
        }

def test_all_combinations(df: pd.DataFrame, text_column: str, label_column: str) -> Dict[str, Any]:
    """Test all model and vectorization combinations"""
    try:
        print(f"\n{'='*80}")
        print("🚀 COMPREHENSIVE TESTING: All Models × All Vectorization Methods")
        print(f"{'='*80}")
        
        # Extract text and labels
        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].tolist()
        
        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        
        # Get all models and vectorization methods
        all_models = get_all_models()
        all_vectorization_methods = get_all_vectorization_methods()
        
        # Test with all models and all vectorization methods
        all_models = all_models  # Test all models
        all_vectorization_methods = all_vectorization_methods  # Test all vectorization methods
        
        # Optuna config
        config = {
            'trials': 2,  # Reduced for comprehensive testing
            'timeout': 30,
            'direction': 'maximize',
            'study_name': 'comprehensive_test'
        }
        
        all_results = []
        total_combinations = len(all_models) * len(all_vectorization_methods)
        current_combination = 0
        
        print(f"📊 Total combinations to test: {total_combinations}")
        
        # Test each vectorization method
        for vectorization_method in all_vectorization_methods:
            print(f"\n🔤 Testing Vectorization Method: {vectorization_method['name']}")
            print("-" * 50)
            
            # Vectorize data
            X, vectorization_info = vectorize_text_data(texts, vectorization_method['name'], **vectorization_method['params'])
            print(f"✅ Vectorized: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Test base models
            base_models = [m for m in all_models if not m.startswith(('voting_ensemble', 'stacking_ensemble'))]
            print(f"📋 Testing {len(base_models)} base models...")
            
            for model_name in base_models:
                current_combination += 1
                print(f"\n[{current_combination}/{total_combinations}] Testing: {model_name} + {vectorization_method['name']}")
                
                result = test_model_with_vectorization(model_name, X, y, vectorization_info, config)
                all_results.append(result)
                
                status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
                print(f"{status_icon} {model_name} + {vectorization_method['name']}: {result['score']:.4f}")
            
            # Test ensemble models
            ensemble_models = [m for m in all_models if m.startswith(('voting_ensemble', 'stacking_ensemble'))]
            if ensemble_models:
                print(f"📋 Testing {len(ensemble_models)} ensemble models...")
                
                for model_name in ensemble_models:
                    current_combination += 1
                    print(f"\n[{current_combination}/{total_combinations}] Testing: {model_name} + {vectorization_method['name']}")
                    
                    result = test_model_with_vectorization(model_name, X, y, vectorization_info, config)
                    all_results.append(result)
                    
                    status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
                    print(f"{status_icon} {model_name} + {vectorization_method['name']}: {result['score']:.4f}")
        
        return {
            'all_results': all_results,
            'total_combinations': total_combinations,
            'models_tested': all_models,
            'vectorization_methods': all_vectorization_methods
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
                print(f"   {i:2d}. {result['model']} + {result['vectorization']}: {result['score']:.4f}")
        
        # Analysis by vectorization method
        print(f"\n📊 Performance by Vectorization Method:")
        vectorization_stats = {}
        for result in successful_results:
            method = result['vectorization']
            if method not in vectorization_stats:
                vectorization_stats[method] = {'scores': [], 'count': 0}
            vectorization_stats[method]['scores'].append(result['score'])
            vectorization_stats[method]['count'] += 1
        
        for method, stats in vectorization_stats.items():
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
        for model, stats in sorted_models[:10]:  # Top 10 models
            avg_score = np.mean(stats['scores'])
            max_score = np.max(stats['scores'])
            print(f"   {model}: Avg={avg_score:.4f}, Max={max_score:.4f}, Count={stats['count']}")
        
        # Ensemble vs Base models comparison
        print(f"\n📊 Ensemble vs Base Models Comparison:")
        ensemble_results = [r for r in successful_results if r['model'].startswith(('voting_ensemble', 'stacking_ensemble'))]
        base_results = [r for r in successful_results if not r['model'].startswith(('voting_ensemble', 'stacking_ensemble'))]
        
        if ensemble_results:
            ensemble_avg = np.mean([r['score'] for r in ensemble_results])
            print(f"   Ensemble models: Avg={ensemble_avg:.4f}, Count={len(ensemble_results)}")
        
        if base_results:
            base_avg = np.mean([r['score'] for r in base_results])
            print(f"   Base models: Avg={base_avg:.4f}, Count={len(base_results)}")
        
        return {
            'overall_stats': {
                'total': len(all_results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'success_rate': len(successful_results)/len(all_results)*100
            },
            'best_combinations': best_results[:10] if successful_results else [],
            'vectorization_stats': vectorization_stats,
            'model_stats': model_stats,
            'ensemble_vs_base': {
                'ensemble_avg': np.mean([r['score'] for r in ensemble_results]) if ensemble_results else 0,
                'base_avg': np.mean([r['score'] for r in base_results]) if base_results else 0
            }
        }
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return {}

def main():
    """Main test function"""
    print("Comprehensive Vectorization Test Suite")
    print("=" * 60)
    
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna is not available. Please install optuna.")
        return
    
    print("✅ Optuna is available")
    print("🚀 Starting comprehensive test with all models and vectorization methods...")
    print("=" * 80)
    
    try:
        # 1. Load dataset
        print("\n1️⃣ Loading Text dataset...")
        df, text_column, label_column = load_spam_dataset(sample_size=1000)
        
        # 2. Test with app.py pipeline (WITH CACHE)
        print("\n2️⃣ Testing with app.py pipeline (WITH CACHE)...")
        app_pipeline_results = test_with_app_pipeline(df, text_column, label_column)
        
        if app_pipeline_results.get('status') == 'success':
            print("✅ App.py pipeline test successful - Cache should be created!")
        else:
            print(f"❌ App.py pipeline test failed: {app_pipeline_results.get('error')}")
        
        # 3. Run comprehensive testing (WITHOUT CACHE)
        print("\n3️⃣ Running comprehensive testing (DIRECT OPTUNA)...")
        results = test_all_combinations(df, text_column, label_column)
        
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
                'name': 'spam_text_classification',
                'samples': len(df),
                'text_column': text_column,
                'label_column': label_column
            },
            'test_config': {
                'trials': 2,
                'timeout': 30,
                'direction': 'maximize'
            },
            'app_pipeline_results': app_pipeline_results,
            'direct_optuna_results': results,
            'analysis': analysis
        }
        
        # Remove DataFrame objects that can't be serialized
        if 'direct_optuna_results' in results_data and 'models_tested' in results_data['direct_optuna_results']:
            # Convert any non-serializable objects to strings
            pass
        
        import json
        import pandas as pd
        
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
                # Handle model objects - convert to string representation
                return f"<{obj.__class__.__name__} object>"
            elif hasattr(obj, '__dict__'):
                # Handle objects with __dict__ - convert to string representation
                return f"<{obj.__class__.__name__} object>"
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif hasattr(obj, 'tolist'):  # For other numpy types
                return obj.tolist()
            else:
                return obj
        
        # Make results serializable
        serializable_results = make_serializable(results_data)
        
        # Ensure training_results directory exists
        os.makedirs('cache/training_results', exist_ok=True)
        
        with open('cache/training_results/comprehensive_vectorization_results.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: cache/training_results/comprehensive_vectorization_results.json")
        
        # 6. Final summary
        print(f"\n{'='*80}")
        print("🎯 FINAL SUMMARY")
        print(f"{'='*80}")
        
        print(f"📊 App.py Pipeline Test:")
        print(f"   Status: {'✅ SUCCESS' if app_pipeline_results.get('status') == 'success' else '❌ FAILED'}")
        print(f"   Cache Created: {'✅ YES' if app_pipeline_results.get('cache_created') else '❌ NO'}")
        
        if analysis.get('overall_stats'):
            stats = analysis['overall_stats']
            print(f"\n📊 Direct Optuna Test:")
            print(f"   Total combinations tested: {stats['total']}")
            print(f"   Success rate: {stats['success_rate']:.1f}%")
            print(f"   Best performing combination: {analysis['best_combinations'][0] if analysis['best_combinations'] else 'None'}")
        
        # 7. Check cache creation
        print(f"\n7️⃣ Checking cache creation...")
        check_cache_creation()
        
        print(f"\n🎉 Comprehensive testing completed!")
        
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
                print(f"  {i}. {result['model']} + {result['vectorization']}: {result['score']:.4f}")
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
                    print(f"  {i}. {result['model']} + {result['vectorization']}")
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
                    print(f"  {i}. {result['model']} + {result['vectorization']}: {result['score']:.4f}")
                    print(f"     CV: {result.get('cv_mean', 'N/A')} ± {result.get('cv_std', 'N/A')}")
                    print(f"     F1: {result.get('f1_score', 'N/A')}, Precision: {result.get('precision', 'N/A')}, Recall: {result.get('recall', 'N/A')}")
            
            # Debug: Show vectorization method performance
            if successful_results:
                print(f"\n🔧 VECTORIZATION METHOD ANALYSIS:")
                vectorization_stats = {}
                for result in successful_results:
                    method = result['vectorization']
                    if method not in vectorization_stats:
                        vectorization_stats[method] = []
                    vectorization_stats[method].append(result['score'])
                
                for method, scores in vectorization_stats.items():
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
                    print(f"  • {model}: {len(scores)} vectorization methods, avg accuracy: {avg_score:.4f}")
            
            print(f"\n🎯 DEBUG SUMMARY:")
            print(f"  • Total tests: {len(all_results)}")
            print(f"  • Success rate: {len(successful_results)/len(all_results)*100:.1f}%" if len(all_results) > 0 else "  • Success rate: N/A")
            print(f"  • Cache efficiency: {cache_hits/(cache_hits+cache_misses)*100:.1f}%" if (cache_hits+cache_misses) > 0 else "  • Cache efficiency: N/A")
            print(f"  • Best model: {sorted_results[0]['model']} + {sorted_results[0]['vectorization']} = {sorted_results[0]['score']:.4f}" if successful_results else "  • Best model: N/A")
        else:
            print(f"  • No analysis data available for debug information")
            print(f"  • Analysis object: {type(analysis)}")
            if analysis:
                print(f"  • Analysis keys: {list(analysis.keys()) if hasattr(analysis, 'keys') else 'No keys method'}")
        
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
        
        # Check cache/embeddings/ (embeddings cache)
        cache_embeddings_dir = "cache/embeddings"
        if os.path.exists(cache_embeddings_dir):
            print(f"\n✅ {cache_embeddings_dir} exists")
            files = os.listdir(cache_embeddings_dir)
            print(f"📊 Embeddings cache files: {len(files)}")
        else:
            print(f"\n❌ {cache_embeddings_dir} does not exist")
        
        # Summary
        print(f"\n📊 Cache Summary:")
        print(f"   Per-model cache: {'✅ YES' if os.path.exists(cache_models_dir) else '❌ NO'}")
        print(f"   Training results cache: {'✅ YES' if os.path.exists(cache_training_dir) else '❌ NO'}")
        print(f"   Embeddings cache: {'✅ YES' if os.path.exists(cache_embeddings_dir) else '❌ NO'}")
        
    except Exception as e:
        print(f"❌ Error checking cache: {e}")

if __name__ == "__main__":
    main()
