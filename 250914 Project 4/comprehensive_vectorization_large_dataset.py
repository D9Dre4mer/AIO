#!/usr/bin/env python3
"""
Comprehensive Test Script for Large Dataset (300K samples)
Tests all models with all vectorization methods combinations including ensemble and stacking
Using the large dataset: cache\20250822-004129_sample-300_000Samples.csv
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
    from models import model_factory, model_registry
    from models.ensemble import ensemble_manager
    from models.ensemble.stacking_classifier import StackingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from text_encoders import TextVectorizer, EmbeddingVectorizer
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def load_large_dataset(sample_size: int = 1000) -> Tuple[pd.DataFrame, str, str]:
    """Load large dataset (300K samples)"""
    try:
        print(f"📊 Loading large dataset with {sample_size} samples...")
        df = pd.read_csv('cache/20250822-004129_sample-300_000Samples.csv')
        
        # Sample the dataset
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        print(f"✅ Dataset loaded: {len(df)} samples, {len(df.columns)} columns")
        print(f"📊 Columns: {df.columns.tolist()}")
        
        # Check for label column
        label_column = None
        for col in ['Category', 'label', 'category', 'class', 'target', 'y']:
            if col in df.columns:
                label_column = col
                break
        
        if label_column:
            print(f"📊 Label column: {label_column}")
            print(f"📊 Category distribution: {df[label_column].value_counts().to_dict()}")
        else:
            print("⚠️ No label column found, using first column as label")
            label_column = df.columns[0]
            print(f"📊 Using first column as label: {label_column}")
        
        # Check for text column
        text_column = None
        for col in ['Message', 'text', 'content', 'description', 'data', 'title', 'abstract']:
            if col in df.columns:
                text_column = col
                break
        
        if text_column:
            print(f"📊 Text column: {text_column}")
        else:
            print("⚠️ No text column found, using second column as text")
            text_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            print(f"📊 Using second column as text: {text_column}")
        
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
        # DEBUG MODE: Test one model at a time for debugging
        debug_models = [
            'lightgbm',  # Test LightGBM cache fix
        ]
        
        print(f"🔧 TESTING LIGHTGBM CACHE FIX: {debug_models}")
        print(f"📋 Total models: {len(debug_models)}")
        
        return debug_models
        
    except Exception as e:
        print(f"❌ Error getting models: {e}")
        return []

def get_all_vectorization_methods() -> List[Dict[str, Any]]:
    """Get all available vectorization methods with configurations"""
    try:
        # DEBUG MODE: Test all vectorization methods to check cache key fix
        debug_methods = [
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
        
        print(f"🔧 TESTING ALL VECTORIZATION METHODS: {[m['name'] for m in debug_methods]}")
        return debug_methods
        
    except Exception as e:
        print(f"❌ Error getting vectorization methods: {e}")
        return []

def test_model_with_vectorization(model_name: str, X: np.ndarray, y: np.ndarray, 
                                vectorization_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single model with specific vectorization method using app.py pipeline"""
    try:
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {model_name} + {vectorization_info['method']}")
        print(f"{'='*60}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"📊 Data split: Train {X_train.shape}, Val {X_val.shape}")
        
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
        
        return {
            'model': model_name,
            'vectorization': vectorization_info['method'],
            'score': best_score,
            'params': best_params,
            'time': end_time - start_time,
            'features': vectorization_info['features'],
            'status': 'SUCCESS'
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

def test_with_app_pipeline(df: pd.DataFrame, text_column: str, label_column: str) -> Dict[str, Any]:
    """Test using the same pipeline as app.py to ensure cache creation"""
    try:
        print(f"\n{'='*80}")
        print("🚀 TESTING WITH APP.PY PIPELINE (WITH CACHE)")
        print(f"{'='*80}")
        
        # Create step data similar to app.py
        step1_data = {
            'dataframe': df,
            'uploaded_file': {'name': '20250822-004129_sample-300_000Samples.csv'},
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
            'selected_vectorization': ['TF-IDF', 'BoW', 'Word Embeddings']
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
        # all_models = all_models  # Test all models
        # all_vectorization_methods = all_vectorization_methods  # Test all vectorization methods
        
        # Optuna config
        config = {
            'trials': 2,  # Reduced for comprehensive testing
            'timeout': 30,
            'direction': 'maximize',
            'study_name': 'large_dataset_test'
        }
        
        all_results = []
        total_combinations = len(all_models) * len(all_vectorization_methods)
        current_combination = 0
        
        print(f"📊 Total combinations to test: {total_combinations}")
        print(f"🚀 Starting Optuna optimization for all models...")
        print(f"⏱️ This may take several minutes. Please wait...")
        
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
                
                # Don't print individual results during training - wait for final summary
                print(f"✅ Completed: {model_name} + {vectorization_method['name']} ({current_combination}/{total_combinations})")
            
            # Test ensemble models
            ensemble_models = [m for m in all_models if m.startswith(('voting_ensemble', 'stacking_ensemble'))]
            if ensemble_models:
                print(f"📋 Testing {len(ensemble_models)} ensemble models...")
                
                for model_name in ensemble_models:
                    current_combination += 1
                    print(f"\n[{current_combination}/{total_combinations}] Testing: {model_name} + {vectorization_method['name']}")
                    
                    result = test_model_with_vectorization(model_name, X, y, vectorization_info, config)
                    all_results.append(result)
                    
                    # Don't print individual results during training - wait for final summary
                    print(f"✅ Completed: {model_name} + {vectorization_method['name']} ({current_combination}/{total_combinations})")
        
        print(f"\n🎉 All Optuna optimization completed!")
        print(f"📊 Total combinations trained: {len(all_results)}")
        print(f"🚀 Now generating final results table...")
        
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
    print("🧪 Comprehensive Vectorization Test Suite - Large Dataset")
    print("=" * 70)
    
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna is not available. Please install optuna.")
        return
    
    print("✅ Optuna is available")
    print("🚀 Starting comprehensive test with large dataset (300K samples)...")
    print("=" * 80)
    
    try:
        # 1. Load dataset
        print("\n1️⃣ Loading Large Dataset...")
        df, text_column, label_column = load_large_dataset(sample_size=1000)
        
        # 2. Test with app.py pipeline (WITH CACHE)
        print("\n2️⃣ Testing with app.py pipeline (WITH CACHE)...")
        app_pipeline_results = test_with_app_pipeline(df, text_column, label_column)
        
        if app_pipeline_results.get('status') == 'success':
            print("✅ App.py pipeline test successful - Cache should be created!")
        else:
            print(f"❌ App.py pipeline test failed: {app_pipeline_results.get('error')}")
        
        # 3. Run comprehensive testing (DIRECT OPTUNA)
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
                'name': 'large_dataset_300k_samples',
                'samples': len(df),
                'text_column': text_column,
                'label_column': label_column,
                'source_file': 'cache/20250822-004129_sample-300_000Samples.csv'
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
        
        with open('comprehensive_vectorization_large_dataset_results.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: comprehensive_vectorization_large_dataset_results.json")
        
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
                    level = root.replace(cache_models_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        print(f"{subindent}{file}")
            print(f"📊 Per-model cache entries: {model_count}")
        else:
            print(f"❌ {cache_models_dir} does not exist")
        
        # Check cache/training_results/ (overall cache)
        cache_training_dir = "cache/training_results"
        if os.path.exists(cache_training_dir):
            print(f"\n✅ {cache_training_dir} exists")
            files = os.listdir(cache_training_dir)
            for file in files:
                file_path = os.path.join(cache_training_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"  📄 {file} ({file_size:,} bytes)")
            print(f"📊 Training results cache files: {len(files)}")
        else:
            print(f"\n❌ {cache_training_dir} does not exist")
        
        # Check cache/embeddings/ (embeddings cache)
        cache_embeddings_dir = "cache/embeddings"
        if os.path.exists(cache_embeddings_dir):
            print(f"\n✅ {cache_embeddings_dir} exists")
            files = os.listdir(cache_embeddings_dir)
            for file in files:
                file_path = os.path.join(cache_embeddings_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"  📄 {file} ({file_size:,} bytes)")
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
