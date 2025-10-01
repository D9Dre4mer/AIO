"""
Training Pipeline for Streamlit Wizard UI
Executes comprehensive training evaluation like main.py
Integrates with existing project modules and comprehensive_evaluation.py
"""

import warnings
import numpy as np
import pandas as pd
import time
import os
import json
import pickle
# import threading  # Removed to avoid ScriptRunContext issues
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Parallel processing disabled - using pickle instead

# CRITICAL: Add comprehensive temp file cleanup
try:
    import tempfile
    import shutil
    import glob
    import atexit
    
    def cleanup_temp_files():
        """Clean up temporary files created by various processes"""
        try:
            temp_dir = tempfile.gettempdir()
            # Clean up temp files
            temp_patterns = [
                os.path.join(temp_dir, "tmp*"),
                os.path.join(temp_dir, "*.tmp")
            ]
            
            for pattern in temp_patterns:
                for temp_file in glob.glob(pattern):
                    try:
                        if os.path.isdir(temp_file):
                            shutil.rmtree(temp_file, ignore_errors=True)
                        else:
                            os.remove(temp_file)
                        print(f"SUCCESS: Cleaned up temp file: {temp_file}")
                    except Exception as cleanup_error:
                        print(f"WARNING: Failed to clean up temp file {temp_file}: {cleanup_error}")
            
            print("SUCCESS: Temp file cleanup completed")
        except Exception as cleanup_error:
            print(f"WARNING: Temp file cleanup failed: {cleanup_error}")
    
    # Register cleanup function to run on exit
    atexit.register(cleanup_temp_files)
    print("SUCCESS: Temp file cleanup registered in training_pipeline")
    
except Exception as temp_error:
    print(f"WARNING: Failed to setup temp file cleanup in training_pipeline: {temp_error}")

# Suppress warnings
warnings.filterwarnings("ignore")

# Import cache manager
from cache_manager import cache_manager

# Import project modules
try:
    from data_loader import DataLoader
    from text_encoders import TextVectorizer
    from models import validation_manager, model_factory
    from visualization import (
        plot_confusion_matrix,
        create_output_directories,
        plot_model_comparison,
        print_model_results
    )
    from comprehensive_evaluation import ComprehensiveEvaluator
    from optuna_optimizer import OptunaOptimizer, optimize_model_with_optuna
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    # Create dummy classes for fallback
    class DataLoader:
        pass
    class TextVectorizer:
        pass
    class validation_manager:
        pass
    class model_factory:
        pass
    class ComprehensiveEvaluator:
        pass
    class OptunaOptimizer:
        pass
    def optimize_model_with_optuna(*args, **kwargs):
        return None


class StreamlitTrainingPipeline:
    """Training pipeline specifically designed for Streamlit Wizard UI"""
    
    def __init__(self):
        """Initialize the training pipeline"""
        self.results = {}
        self.training_status = "idle"
        self.current_model = None
        self.current_phase = "initializing"
        self.models_completed = 0
        self.total_models = 0
        self.start_time = None
        self.elapsed_time = 0
        
        # Initialize stop mechanism
        # self._stop_event = threading.Event()  # Removed to avoid ScriptRunContext issues
        # self._training_lock = threading.Lock()  # Removed to avoid ScriptRunContext issues
        
        # Initialize cache system
        self.cache_dir = "cache/training_results"
        self._ensure_cache_directory()
        self.cache_metadata_file = os.path.join(self.cache_dir, "cache_metadata.json")
        self.cache_metadata = self._load_cache_metadata()
        
        # Initialize Optuna optimizer
        self.optuna_optimizer = None
        self.optuna_config = None
    
    def stop_training(self):
        """Stop the current training process"""
        with self._training_lock:
            self._stop_event.set()
            self.training_status = "stopped"
            print("🛑 Training stop requested")
    
    def is_training_stopped(self) -> bool:
        """Check if training should stop"""
        return self._stop_event.is_set()
    
    def reset_stop_flag(self):
        """Reset the stop flag for new training"""
        with self._training_lock:
            self._stop_event.clear()
            if self.training_status == "stopped":
                self.training_status = "idle"
    
    def configure_optuna(self, optuna_config: Dict[str, Any]):
        """Configure Optuna optimization settings
        
        Args:
            optuna_config: Configuration dictionary for Optuna
        """
        self.optuna_config = optuna_config
        
        # Create Optuna optimizer if enabled
        if optuna_config.get('enable', False):
            try:
                self.optuna_optimizer = OptunaOptimizer(optuna_config)
                print(f"✅ Optuna optimization enabled: {optuna_config.get('trials', 100)} trials")
            except ImportError:
                print("⚠️ Optuna not available, skipping optimization")
                self.optuna_optimizer = None
        else:
            self.optuna_optimizer = None
            print("ℹ️ Optuna optimization disabled")
        
    def initialize_pipeline(self, df: pd.DataFrame, step1_data: Dict, 
                          step2_data: Dict, step3_data: Dict) -> Dict:
        """Initialize the training pipeline with configuration from previous steps"""
        
        try:
            self.current_phase = "initializing"
            
            # Extract configuration from previous steps
            sampling_config = step1_data.get('sampling_config', {})
            text_column = step2_data.get('text_column')
            label_column = step2_data.get('label_column')
            
            # Store preprocessing config in instance for later use
            self.preprocessing_config = {
                'text_cleaning': step2_data.get('text_cleaning', True),
                'category_mapping': step2_data.get('category_mapping', True),
                'data_validation': step2_data.get('data_validation', True),
                'memory_optimization': step2_data.get('memory_optimization', True),
                # Advanced preprocessing options
                'rare_words_removal': step2_data.get('rare_words_removal', False),
                'rare_words_threshold': step2_data.get('rare_words_threshold', 2),
                'lemmatization': step2_data.get('lemmatization', False),
                'context_aware_stopwords': step2_data.get('context_aware_stopwords', False),
                'stopwords_aggressiveness': step2_data.get('stopwords_aggressiveness', 'Moderate'),
                'phrase_detection': step2_data.get('phrase_detection', False),
                'min_phrase_freq': step2_data.get('min_phrase_freq', 3)
            }
            
            # Create local variable for return
            preprocessing_config = self.preprocessing_config
            
            model_config = step3_data.get('data_split', {}) if step3_data and isinstance(step3_data, dict) else {}
            selected_models = step3_data.get('selected_models', []) if step3_data and isinstance(step3_data, dict) else []
            selected_vectorization = step3_data.get('selected_vectorization', []) if step3_data and isinstance(step3_data, dict) else []
            cv_config = step3_data.get('cross_validation', {}) if step3_data and isinstance(step3_data, dict) else {}
            optuna_config = step3_data.get('optuna', {}) if step3_data and isinstance(step3_data, dict) else {}
            
            # Configure Optuna if provided
            if optuna_config:
                self.configure_optuna(optuna_config)
            
            # Calculate total models to train
            self.total_models = len(selected_models) * len(selected_vectorization)
            
            # Create output directories
            try:
                create_output_directories()
            except:
                pass  # Directory might already exist
            
            return {
                'status': 'success',
                'message': 'Pipeline initialized successfully',
                'total_models': self.total_models,
                'config': {
                    'sampling': sampling_config,
                    'preprocessing': preprocessing_config,
                    'model': model_config,
                    'vectorization': selected_vectorization,
                    'cv': cv_config
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Pipeline initialization failed: {str(e)}',
                'error': str(e)
            }
    
    def _ensure_cache_directory(self):
        """Ensure cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _load_cache_metadata(self) -> Dict:
        """Load cache metadata from file"""
        try:
            if os.path.exists(self.cache_metadata_file):
                with open(self.cache_metadata_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Warning: Could not load cache metadata: {e}")
            return {}
    
    def _save_cache_metadata(self):
        """Save cache metadata to file with atomic write"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.cache_metadata_file), exist_ok=True)
            
            # Atomic write: write to temp file first, then rename
            temp_file = self.cache_metadata_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
            # Atomic rename - remove target file first on Windows
            if os.path.exists(self.cache_metadata_file):
                os.remove(self.cache_metadata_file)
            os.rename(temp_file, self.cache_metadata_file)
        except Exception as e:
            print(f"❌ Error saving cache metadata: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_cache_key(self, step1_data: Dict, step2_data: Dict, step3_data: Dict) -> str:
        """Generate unique cache key based on configuration with human-readable naming"""
        # Extract key configuration details for naming
        sampling_config = step1_data.get('sampling_config', {})
        selected_models = step3_data.get('selected_models', []) if step3_data and isinstance(step3_data, dict) else []
        selected_vectorization = step3_data.get('selected_vectorization', []) if step3_data and isinstance(step3_data, dict) else []
        text_column = step2_data.get('text_column', 'text')
        label_column = step2_data.get('label_column', 'label')
        
        # CRITICAL FIX: Include dataset information in cache key
        dataset_name = "unknown_dataset"
        dataset_hash = "no_hash"
        
        # Get dataset name from step1_data
        if 'uploaded_file' in step1_data and step1_data['uploaded_file']:
            uploaded_file = step1_data['uploaded_file']
            if isinstance(uploaded_file, dict) and 'name' in uploaded_file:
                dataset_name = uploaded_file['name'].replace('.csv', '').replace('.xlsx', '').replace('.json', '').replace('.txt', '')
            elif hasattr(uploaded_file, 'name'):
                dataset_name = uploaded_file.name.replace('.csv', '').replace('.xlsx', '').replace('.json', '').replace('.txt', '')
        
        # Get dataset content hash from dataframe
        if 'dataframe' in step1_data and step1_data['dataframe'] is not None:
            df = step1_data['dataframe']
            try:
                # Create hash from dataset content (first 100 rows + column names)
                import hashlib
                content_str = str(list(df.columns)) + str(df.head(100).values.tolist())
                dataset_hash = hashlib.md5(content_str.encode()).hexdigest()[:8]
            except Exception as e:
                print(f"⚠️ [CACHE_KEY] Error creating dataset hash: {e}")
                dataset_hash = "hash_err"
        
        # Get selected categories for label differentiation
        selected_categories = step1_data.get('selected_categories', [])
        
        # CRITICAL FIX: Try to extract categories from dataframe if not provided
        if not selected_categories and 'dataframe' in step1_data and step1_data['dataframe'] is not None:
            df = step1_data['dataframe']
            label_column = step2_data.get('label_column', 'label')
            
            # Try to find label column
            for col in ['Category', 'label', 'category', 'class', 'target', 'y']:
                if col in df.columns:
                    label_column = col
                    break
            
            if label_column in df.columns:
                try:
                    # Get unique labels from dataframe
                    unique_labels = df[label_column].unique()
                    selected_categories = sorted(unique_labels.tolist())
    
                except Exception as e:
                    print(f"⚠️ [CACHE_KEY] Error extracting categories: {e}")
        
        # Ensure all categories are strings before joining
        if selected_categories:
            categories_str = "_".join(str(cat)[:10] for cat in selected_categories[:3])
        else:
            categories_str = "no_cats"
        if len(categories_str) > 20:
            categories_str = categories_str[:20] + "..."
        
        # Create human-readable cache name
        sample_count = sampling_config.get('num_samples', 'full')
        if sample_count == 'full':
            sample_str = "full_dataset"
        else:
            sample_str = f"{sample_count}samples"
        
        # Get first few models and vectorization methods for name
        model_str = "_".join(selected_models[:3])  # First 3 models
        vector_str = "_".join(selected_vectorization[:2])  # First 2 vectorization methods
        
        # Truncate if too long
        if len(model_str) > 20:
            model_str = model_str[:20] + "..."
        if len(vector_str) > 15:
            vector_str = vector_str[:15] + "..."
        
        # Create human-readable name with dataset info
        human_name = f"{model_str}_{vector_str}_{sample_str}_{dataset_name}_{categories_str}"
        
        # Add test data information to cache name if available
        if hasattr(self, 'data_dict') and 'y_test' in self.data_dict:
            test_samples = len(self.data_dict['y_test'])
            human_name += f"_test{test_samples}"
        
        # Also create hash for uniqueness
        config_hash = {
            'dataset': {
                'name': dataset_name,
                'hash': dataset_hash,
                'categories': selected_categories,
                'columns': list(df.columns) if 'dataframe' in step1_data and step1_data['dataframe'] is not None else []
            },
            'sampling': sampling_config,
            'preprocessing': {
                'text_column': text_column,
                'label_column': label_column,
                'text_cleaning': step2_data.get('text_cleaning', True),
                'category_mapping': step2_data.get('category_mapping', True),
                'data_validation': step2_data.get('data_validation', True),
                'memory_optimization': step2_data.get('memory_optimization', True),
                'rare_words_removal': step2_data.get('rare_words_removal', False),
                'rare_words_threshold': step2_data.get('rare_words_threshold', 2),
                'lemmatization': step2_data.get('lemmatization', False),
                'context_aware_stopwords': step2_data.get('context_aware_stopwords', False),
                'stopwords_aggressiveness': step2_data.get('stopwords_aggressiveness', 'Moderate'),
                'phrase_detection': step2_data.get('phrase_detection', False),
                'min_phrase_freq': step2_data.get('min_phrase_freq', 3)
            },
            'model': step3_data.get('data_split', {}) if step3_data and isinstance(step3_data, dict) else {},
            'vectorization': selected_vectorization,
            'cv': step3_data.get('cross_validation', {}) if step3_data and isinstance(step3_data, dict) else {}
        }
        
        # Create hash for uniqueness
        import hashlib
        config_str = json.dumps(config_hash, sort_keys=True)
        config_hash_str = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Return human-readable name with hash
        return f"{human_name}_{config_hash_str}"
    
    def _check_cache(self, cache_key: str) -> Dict:
        """Check if results exist in cache - NO TIME EXPIRY"""
        if cache_key in self.cache_metadata:
            cache_info = self.cache_metadata[cache_key]
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            # Check if cache file exists (NO TIME EXPIRY)
            if os.path.exists(cache_file):
                cache_age = time.time() - cache_info['timestamp']
                
                try:
                    with open(cache_file, 'rb') as f:
                        cached_results = pickle.load(f)
                    
                    # Display cache hit information
                    cache_name = cache_info.get('cache_name', cache_key)
                    print(f"✅ Using cached results: {cache_name}")
                    print(f"   Age: {cache_age/3600:.1f}h | File: {cache_key}")
                    
                    return cached_results
                except Exception as e:
                    print(f"Warning: Could not load cached results: {e}")
        
        # If exact cache not found, try to find compatible cache
        return self._find_compatible_cache(cache_key)
    
    def _find_compatible_cache(self, target_cache_key: str) -> Dict:
        """Find compatible cache when exact match not found"""
        print(f"🔍 Exact cache not found: {target_cache_key}")
        print("🔍 Searching for compatible cache...")
        
        # Extract key info from target cache key
        target_parts = target_cache_key.split('_')
        target_samples = None
        target_categories = None
        
        for part in target_parts:
            if 'samples' in part:
                target_samples = part
            elif any(cat in part for cat in ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']):
                target_categories = part
        
        best_match = None
        best_score = 0
        
        for cache_key, cache_info in self.cache_metadata.items():
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            if not os.path.exists(cache_file):
                continue
                
            # Calculate compatibility score
            score = 0
            cache_parts = cache_key.split('_')
            
            # Check samples match
            for part in cache_parts:
                if 'samples' in part and target_samples and part == target_samples:
                    score += 3
                elif 'full' in part and target_samples and 'full' in target_samples:
                    score += 3
            
            # Check categories match
            for part in cache_parts:
                if target_categories and part in target_categories:
                    score += 2
                elif any(cat in part for cat in ['astro-ph', 'cond-mat', 'cs']):
                    score += 1
            
            # Check if it's a recent cache
            cache_age = time.time() - cache_info['timestamp']
            if cache_age < 7 * 24 * 3600:  # Within 7 days
                score += 1
            
            if score > best_score:
                best_score = score
                best_match = (cache_key, cache_info)
        
        if best_match and best_score >= 2:  # Minimum compatibility threshold
            cache_key, cache_info = best_match
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            cache_age = time.time() - cache_info['timestamp']
            
            try:
                with open(cache_file, 'rb') as f:
                    cached_results = pickle.load(f)
                
                print(f"✅ Using compatible cache: {cache_key[:50]}...")
                print(f"   Compatibility score: {best_score}/5")
                print(f"   Age: {cache_age/3600:.1f}h")
                print(f"   ⚠️  Note: Using cache with different config - results may vary")
                
                return cached_results
            except Exception as e:
                print(f"Warning: Could not load compatible cache: {e}")
        
        print("❌ No compatible cache found")
        return None
    
    def _prepare_results_for_cache(self, results: Dict) -> Dict:
        """Prepare results for cache by handling non-serializable objects like FAISS index"""
        import copy
        import faiss
        import pickle
        
        def is_faiss_object(obj):
            """Check if object is a FAISS index or contains FAISS objects"""
            if obj is None:
                return False
            try:
                # Check for FAISS index classes
                if hasattr(obj, '__class__'):
                    class_name = str(obj.__class__)
                    if 'faiss' in class_name.lower() or 'Index' in class_name:
                        return True
                # Check for FAISS index attributes
                if hasattr(obj, 'ntotal') and hasattr(obj, 'd'):
                    return True
                # Check if object can be pickled (FAISS objects cannot)
                try:
                    pickle.dumps(obj)
                    return False
                except Exception as e:
                    # If pickle fails and it's not a basic type, it might be FAISS
                    if not isinstance(obj, (str, int, float, bool, list, dict, tuple)):
                        return True
                    return False
            except:
                return False
        
        def replace_faiss_recursive(obj, path=""):
            """Recursively replace FAISS objects with serializable info"""
            if isinstance(obj, dict):
                new_dict = {}
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if is_faiss_object(value):
                        print(f"🔍 Found FAISS object at {current_path}: {type(value)}")
                        try:
                            new_dict[key] = {
                                '_faiss_index_info': f'FAISS index with {value.ntotal} vectors',
                                '_faiss_index_type': str(type(value)),
                                '_faiss_index_serialized': False
                            }
                        except:
                            new_dict[key] = {
                                '_faiss_index_info': f'FAISS index (unable to get ntotal)',
                                '_faiss_index_type': str(type(value)),
                                '_faiss_index_serialized': False
                            }
                    elif isinstance(value, (dict, list)):
                        new_dict[key] = replace_faiss_recursive(value, current_path)
                    else:
                        new_dict[key] = value
                return new_dict
            elif isinstance(obj, list):
                new_list = []
                for i, item in enumerate(obj):
                    current_path = f"{path}[{i}]" if path else f"[{i}]"
                    if is_faiss_object(item):
                        print(f"🔍 Found FAISS object at {current_path}: {type(item)}")
                        try:
                            new_list.append({
                                '_faiss_index_info': f'FAISS index with {item.ntotal} vectors',
                                '_faiss_index_type': str(type(item)),
                                '_faiss_index_serialized': False
                            })
                        except:
                            new_list.append({
                                '_faiss_index_info': f'FAISS index (unable to get ntotal)',
                                '_faiss_index_type': str(type(item)),
                                '_faiss_index_serialized': False
                            })
                    elif isinstance(item, (dict, list)):
                        new_list.append(replace_faiss_recursive(item, current_path))
                    else:
                        new_list.append(item)
                return new_list
            else:
                return obj
        
        print("🔍 Preparing results for cache - scanning for FAISS objects...")
        # Create a completely new structure with FAISS objects replaced
        results_copy = replace_faiss_recursive(results)
        print("✅ FAISS objects replaced successfully")
        
        # Final verification - try to pickle the result
        try:
            pickle.dumps(results_copy)
            print("✅ Results are now serializable")
        except Exception as e:
            print(f"⚠️ Warning: Results may still contain non-serializable objects: {e}")
        
        return results_copy
    
    def _save_to_cache(self, cache_key: str, results: Dict):
        """Save results to cache with human-readable information and better error handling"""
        try:
            # Ensure cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)
            
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            # CRITICAL FIX: Ensure label mapping is included in results
            if hasattr(self, 'original_label_mapping') and self.original_label_mapping:
                # Add label mapping to results if not already present
                if 'label_mapping' not in results:
                    results['label_mapping'] = self.original_label_mapping

                
                # Add original label values if available
                if hasattr(self, 'original_label_values') and self.original_label_values:
                    results['original_label_values'] = self.original_label_values

                
                # Add label encoder info if available
                if hasattr(self, 'label_encoder') and self.label_encoder:
                    results['label_encoder_classes'] = self.label_encoder.classes_.tolist()

            
            # CRITICAL FIX: Handle FAISS index serialization issue
            # FAISS index cannot be pickled directly, need to convert to serializable format
            results_copy = self._prepare_results_for_cache(results)
            
            # Save results with atomic write (write to temp file first, then rename)
            temp_file = cache_file + '.tmp'
            try:
                with open(temp_file, 'wb') as f:
                    pickle.dump(results_copy, f)
                # Atomic rename - remove target file first on Windows
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                os.rename(temp_file, cache_file)
            except Exception as e:
                # Clean up temp file if it exists
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                raise e
            
            # Update metadata with human-readable info
            self.cache_metadata[cache_key] = {
                'timestamp': time.time(),
                'file_path': cache_file,
                'cache_name': cache_key,  # Human-readable name
                'results_summary': {
                    'successful_combinations': results.get('successful_combinations', 0),
                    'total_combinations': results.get('total_combinations', 0),
                    'evaluation_time': results.get('evaluation_time', 0)
                }
            }
            
            # DISABLED: Old cache system - using new per-model cache instead
            print(f"🔄 Using new per-model cache system (old cache disabled)")
            
        except Exception as e:
            print(f"❌ Error saving to cache: {e}")
            import traceback
            traceback.print_exc()
    
    def show_cache_status(self):
        """Display cache status in a user-friendly format"""
        cached_results = self.list_cached_results()
        
        print("\n" + "="*70)
        print("📊 CACHE STATUS REPORT")
        print("="*70)
        print(f"📍 Cache Directory: {self.cache_dir}")
        print(f"📁 Total Entries: {len(cached_results)}")
        print("-"*70)
        
        if cached_results:
            print("📋 CACHED RESULTS:")
            for i, item in enumerate(cached_results[:10], 1):  # Show first 10
                cache_name = item.get('cache_name', item['cache_key'])
                age_hours = item['age_hours']
                results = item['results_summary']
                
                print(f"{i:2d}. {cache_name}")
                print(f"    Age: {age_hours:.1f}h | Key: {item['cache_key'][:12]}...")
                print(f"    Results: {results.get('successful_combinations', 0)}/{results.get('total_combinations', 0)} combinations")
                print(f"    Time: {results.get('evaluation_time', 0):.1f}s")
                print()
        else:
            print("📭 No cached results found")
        
        print("="*70)
    
    def get_cached_results(self, step1_data: Dict, step2_data: Dict, step3_data: Dict) -> Dict:
        """Get cached results if available"""
        cache_key = self._generate_cache_key(step1_data, step2_data, step3_data)
        return self._check_cache(cache_key)
    
    def clear_cache(self, cache_key: str = None):
        """Clear specific cache or all cache"""
        try:
            if cache_key:
                # Clear specific cache
                if cache_key in self.cache_metadata:
                    cache_file = self.cache_metadata[cache_key]['file_path']
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                    del self.cache_metadata[cache_key]
                    print(f"✅ Cleared cache: {cache_key}")
            else:
                # Clear all cache
                # First, clear files from metadata
                for key in list(self.cache_metadata.keys()):
                    cache_file = self.cache_metadata[key]['file_path']
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                self.cache_metadata = {}
                
                # Then, clear all .pkl files in cache directory (including orphaned files)
                if os.path.exists(self.cache_dir):
                    cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
                    for cache_file in cache_files:
                        file_path = os.path.join(self.cache_dir, cache_file)
                        try:
                            os.remove(file_path)
                            print(f"✅ Removed orphaned cache file: {cache_file}")
                        except Exception as e:
                            print(f"⚠️ Could not remove {cache_file}: {e}")
                
                print("✅ Cleared all cache (including orphaned files)")
            
            self._save_cache_metadata()
            
        except Exception as e:
            print(f"Warning: Could not clear cache: {e}")
    
    def list_cached_results(self) -> List[Dict]:
        """List all cached results with metadata"""
        cached_results = []
        orphaned_keys = []
        
        # First pass: collect valid results and identify orphaned entries
        for cache_key, metadata in self.cache_metadata.items():
            # Check if cache file actually exists
            cache_file = metadata.get('file_path', '')
            if os.path.exists(cache_file):
                cached_results.append({
                    'cache_key': cache_key,
                    'cache_name': metadata.get('cache_name', cache_key),
                    'timestamp': metadata['timestamp'],
                    'age_hours': (time.time() - metadata['timestamp']) / 3600,
                    'results_summary': metadata.get('results_summary', {}),
                    'file_path': metadata['file_path']
                })
            else:
                # Cache file doesn't exist, mark for removal
                orphaned_keys.append(cache_key)
                print(f"⚠️ Cache file not found, marking for removal: {cache_key}")
        
        # Second pass: remove orphaned entries
        if orphaned_keys:
            for cache_key in orphaned_keys:
                if cache_key in self.cache_metadata:
                    del self.cache_metadata[cache_key]
            
            # Save updated metadata
            self._save_cache_metadata()
            print(f"✅ Cleaned up {len(orphaned_keys)} orphaned cache entries")
        
        return cached_results
    
    def execute_training(self, df: pd.DataFrame, step1_data: Dict,
                         step2_data: Dict, step3_data: Dict,
                         progress_callback=None) -> Dict:
        """Execute comprehensive training evaluation like main.py"""

        try:
            # Reset stop flag for new training
            self.reset_stop_flag()
            
            self.start_time = time.time()
            self.training_status = "training"
            self.models_completed = 0

            # Check if training was stopped before starting
            if self.is_training_stopped():
                return {
                    'status': 'stopped',
                    'message': 'Training was stopped before starting',
                    'results': {},
                    'models_completed': 0,
                    'elapsed_time': 0
                }

            # Check cache first
            self.current_phase = "cache_check"
            if progress_callback:
                progress_callback(self.current_phase, "Checking cache for existing results...", 0.02)
            
            cache_key = self._generate_cache_key(step1_data, step2_data, step3_data)
            
            # FIXED: Store cache key in instance for fallback sampling
            self.current_cache_key = cache_key

            
            # DISABLED: Old cache system - using new per-model cache instead
            print("🔄 Using new per-model cache system (old cache disabled)")
            cached_results = None

            # Initialize pipeline
            self.current_phase = "initialization"
            if progress_callback:
                progress_callback(self.current_phase, "Initializing comprehensive evaluation...", 0.05)

            # Initialize pipeline to get preprocessing config
            init_result = self.initialize_pipeline(df, step1_data, step2_data, step3_data)
            if init_result['status'] != 'success':
                return init_result

            # Extract configuration
            text_column = step2_data.get('text_column')
            label_column = step2_data.get('label_column')
            cv_config = step3_data.get('cross_validation', {}) if step3_data and isinstance(step3_data, dict) else {}
            cv_folds = cv_config.get('cv_folds', 5)
            data_split = step3_data.get('data_split', {}) if step3_data and isinstance(step3_data, dict) else {}
            
            # Calculate test size from step 3 configuration
            test_size = data_split.get('test', 20) / 100.0
            # No separate validation size - CV will handle it
            validation_size = 0.0
            
            # CRITICAL: Preserve original label mapping before any data processing
            # This ensures we can restore meaningful labels later
            if label_column in df.columns:
                unique_labels = sorted(df[label_column].unique().tolist())
                
                # Check if labels are already numeric (processed data)
                if all(str(label).isdigit() for label in unique_labels):
                    print(f"⚠️ [PIPELINE] Labels are already numeric - original names may be lost")
                    print(f"   - Found numeric labels: {unique_labels}")
                    
                    # Try to find original labels in step1_data
                    if step1_data and 'selected_categories' in step1_data:
                        original_categories = step1_data['selected_categories']
                        if len(original_categories) == len(unique_labels):
                            # Create mapping from original categories
                            self.original_label_mapping = {i: cat for i, cat in enumerate(sorted(original_categories))}
        
                        else:
                            print(f"⚠️ [PIPELINE] Mismatch: {len(original_categories)} categories vs {len(unique_labels)} labels")
                    else:
                        print(f"⚠️ [PIPELINE] No selected_categories found - will use generic labels as fallback")
                else:
                    # Labels are still text - preserve them
                    self.original_label_mapping = {i: label for i, label in enumerate(unique_labels)}

            else:
                print(f"⚠️ [PIPELINE] Label column '{label_column}' not found in dataframe")
                print(f"   - Available columns: {list(df.columns)}")

            # Prepare data for comprehensive evaluation
            self.current_phase = "data_preparation"
            if progress_callback:
                progress_callback(self.current_phase, "Preparing data for evaluation...", 0.1)

            # Apply sampling if configured
            sampling_config = step1_data.get('sampling_config', {})
            
            # FIXED: Debug session state issue
            if not step1_data or not sampling_config:
                print(f"⚠️ [PIPELINE] WARNING: Session state issue detected!")
                print(f"   - step1_data exists: {step1_data is not None}")
                print(f"   - sampling_config exists: {sampling_config is not None}")
                
                # FIXED: Try to extract sampling info from step1_data keys first
                if step1_data:
                    # Look for any key that might contain sample count
                    for key, value in step1_data.items():
                        if isinstance(value, dict) and 'num_samples' in value:
                            sampling_config = value
                            print(f"✅ [PIPELINE] Found sampling config in step1_data['{key}']: {sampling_config}")
                            break
                        elif key == 'num_samples':
                            sampling_config = {'num_samples': value, 'sampling_strategy': 'Stratified (Recommended)'}
                            print(f"✅ [PIPELINE] Created sampling config from step1_data['{key}']: {sampling_config}")
                            break
                
                # FIXED: If still no config, try to extract from cache key if available
                if not sampling_config and hasattr(self, 'current_cache_key') and self.current_cache_key:
                    import re
                    sample_match = re.search(r'(\d+)samples', self.current_cache_key)
                    if sample_match:
                        extracted_samples = int(sample_match.group(1))
                        
                        # Create fallback sampling config
                        fallback_config = {
                            'num_samples': extracted_samples,
                            'sampling_strategy': 'Stratified (Recommended)'
                        }
                        print(f"🔄 [PIPELINE] Using fallback sampling config: {fallback_config}")
                        sampling_config = fallback_config
                    else:
                        print(f"❌ [PIPELINE] Could not extract sample count from cache key")
                        # Create default fallback config
                        fallback_config = {
                            'num_samples': 1000,
                            'sampling_strategy': 'Stratified (Recommended)'
                        }
                        print(f"🔄 [PIPELINE] Using default fallback sampling config: {fallback_config}")
                        sampling_config = fallback_config
                
                # FIXED: Final fallback - check if there are any recent cache files
                if not sampling_config:

                    try:
                        import os
                        import pickle
                        import re
                        
                        cache_dir = "cache/training_results"
                        if os.path.exists(cache_dir):
                            cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
                            if cache_files:
                                # Get most recent cache file
                                cache_files.sort(key=lambda x: os.path.getmtime(os.path.join(cache_dir, x)), reverse=True)
                                recent_cache = os.path.join(cache_dir, cache_files[0])
                                
                                with open(recent_cache, 'rb') as f:
                                    cached_data = pickle.load(f)
                                
                                cache_key = cached_data.get('cache_key', '')
                                if 'samples' in cache_key:
                                    sample_match = re.search(r'(\d+)samples', cache_key)
                                    if sample_match:
                                        extracted_samples = int(sample_match.group(1))

                                        
                                        fallback_config = {
                                            'num_samples': extracted_samples,
                                            'sampling_strategy': 'Stratified (Recommended)'
                                        }
                                        print(f"🔄 [PIPELINE] Using fallback sampling config from cache: {fallback_config}")
                                        sampling_config = fallback_config
                    except Exception as e:
                        print(f"⚠️ [PIPELINE] Failed to extract from cache files: {e}")
                
                if not sampling_config:
                    print(f"❌ [PIPELINE] No fallback sampling config found, sampling will be skipped")
            
            if sampling_config and sampling_config.get('num_samples'):
                print(f"✅ [PIPELINE] Sampling will be applied: {sampling_config}")
                original_size = len(df)
                df = self._apply_sampling(df, sampling_config, label_column)
                sampled_size = len(df)
                print(f"✅ [PIPELINE] Sampling result: {original_size:,} → {sampled_size:,} samples")
                
                # FIXED: Verify sampling was actually applied
                if sampled_size == original_size:
                    print(f"⚠️ [PIPELINE] WARNING: Sampling did not reduce dataset size!")
                    print(f"   This suggests sampling logic may have failed")
                else:
                    print(f"✅ [PIPELINE] Sampling verified: {original_size:,} → {sampled_size:,}")
                
                # FIXED: Update step1_data with sampled dataframe to ensure consistency
                if sampling_config and sampling_config.get('num_samples'):
                    print(f"🎯 [TRAINING_PIPELINE] Updating step1_data with sampled dataframe: {len(df):,} samples")
                    step1_data['dataframe'] = df.copy()  # Use the already sampled dataframe
            else:
                print(f"❌ [PIPELINE] No sampling applied, using full dataset ({len(df):,} samples)")
                print(f"   Reason: sampling_config={sampling_config}, num_samples={sampling_config.get('num_samples') if sampling_config else 'N/A'}")

            # Apply preprocessing
            df = self._apply_preprocessing(df, step2_data)

            # Save processed data to temporary CSV for DataLoader
            import tempfile
            
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            df.to_csv(temp_file.name, index=False)
            temp_file.close()

            # Initialize Comprehensive Evaluator
            self.current_phase = "evaluator_setup"
            if progress_callback:
                progress_callback(self.current_phase, "Setting up comprehensive evaluator...", 0.2)

            try:
                # CRITICAL FIX: Create DataLoader with labels before passing to ComprehensiveEvaluator
                from data_loader import DataLoader
                
                # Create DataLoader instance with labels from step1_data
                data_loader = DataLoader()
                
                # Set up label mapping if we have selected_categories
                if step1_data and 'selected_categories' in step1_data:
                    selected_categories = step1_data['selected_categories']
                    sorted_categories = sorted(selected_categories)
                    
                    # Create label mappings
                    data_loader.label_to_id = {label: i for i, label in enumerate(sorted_categories)}
                    data_loader.id_to_label = {i: label for i, label in enumerate(sorted_categories)}
                    
                    print(f"✅ [TRAINING_PIPELINE] Created DataLoader with labels: {data_loader.id_to_label}")
                else:
                    print(f"⚠️ [TRAINING_PIPELINE] No selected_categories - creating empty DataLoader")
                
                # Create evaluator with proper model factory and validation manager
                # CRITICAL FIX: Pass DataLoader with labels to ComprehensiveEvaluator
                evaluator = ComprehensiveEvaluator(
                    cv_folds=cv_folds,
                    validation_size=validation_size,
                    test_size=test_size,
                    random_state=cv_config.get('random_state', 42),
                    data_loader=data_loader  # Pass the DataLoader with labels!
                )
                
                # Ensure evaluator has access to model factory and validation manager
                if hasattr(evaluator, 'model_trainer') and evaluator.model_trainer:
                    evaluator.model_trainer.model_factory = model_factory
                    evaluator.model_trainer.validation_manager = validation_manager

                # Temporarily override DataLoader's file path
                original_file_path = getattr(evaluator.data_loader, 'file_path', None)
                evaluator.data_loader.file_path = temp_file.name
                
                # Set text and label columns
                evaluator.data_loader.text_column = text_column
                evaluator.data_loader.label_column = label_column
                
                # CRITICAL FIX: Transfer actual label mapping from step data
                # This ensures ComprehensiveEvaluator has the real labels from data processing
                original_labels_found = False
                
                # Source 1: Check if we have original label mapping from preprocessing
                if hasattr(self, 'original_label_mapping') and self.original_label_mapping:
                    print(f"✅ [TRAINING_PIPELINE] Found original label mapping from preprocessing")
                    evaluator.data_loader.id_to_label = self.original_label_mapping.copy()
                    evaluator.data_loader.label_to_id = {v: k for k, v in self.original_label_mapping.items()}
                    original_labels_found = True
                    
                    print(f"✅ [TRAINING_PIPELINE] Using original labels from preprocessing:")
                    print(f"   - ID to Label: {evaluator.data_loader.id_to_label}")
                    print(f"   - Original values: {getattr(self, 'original_label_values', 'N/A')}")
                
                # Source 2: Try to use selected_categories from step1_data
                elif step1_data and 'selected_categories' in step1_data:
                    selected_categories = step1_data['selected_categories']
                    sorted_categories = sorted(selected_categories)
                    
                    # Create the same mapping as DataLoader.create_label_mappings()
                    evaluator.data_loader.label_to_id = {label: i for i, label in enumerate(sorted_categories)}
                    evaluator.data_loader.id_to_label = {i: label for i, label in enumerate(sorted_categories)}
                    original_labels_found = True
                    
                    print(f"✅ [TRAINING_PIPELINE] Transferred label mapping from step1_data:")
                    print(f"   - Categories: {sorted_categories}")
                    print(f"   - ID to Label: {evaluator.data_loader.id_to_label}")
                
                # Source 3: Fallback - create generic labels
                if not original_labels_found:
                    print(f"⚠️ [TRAINING_PIPELINE] No original labels found, creating generic labels")
                    print(f"   - step1_data: {step1_data}")
                    print(f"   - step1_data keys: {list(step1_data.keys()) if step1_data else 'None'}")
                    
                    # Create generic labels based on unique values in dataframe
                    if 'dataframe' in step1_data and step1_data['dataframe'] is not None:
                        df_temp = step1_data['dataframe']
                        if label_column in df_temp.columns:
                            unique_labels = sorted(df_temp[label_column].unique())
                            generic_mapping = {i: f"Category_{label}" for i, label in enumerate(unique_labels)}
                            
                            evaluator.data_loader.id_to_label = generic_mapping
                            evaluator.data_loader.label_to_id = {v: k for k, v in generic_mapping.items()}
                            
                            print(f"✅ [TRAINING_PIPELINE] Created generic labels from dataframe:")
                            print(f"   - Unique values: {unique_labels}")
                            print(f"   - Generic mapping: {generic_mapping}")
                        else:
                            print(f"❌ [TRAINING_PIPELINE] Label column '{label_column}' not found in dataframe")
                    else:
                        print(f"❌ [TRAINING_PIPELINE] No dataframe available for generic label creation")
                    
                    # Source 4: Check if we have preprocessed samples with original labels
                    if not original_labels_found and hasattr(self, 'preprocessed_samples') and self.preprocessed_samples:
                        # Extract original labels from preprocessed samples
                        original_labels = set()
                        for sample in self.preprocessed_samples:
                            if 'label' in sample and not str(sample['label']).isdigit():
                                original_labels.add(sample['label'])
                        
                        if original_labels:
                            sorted_original_labels = sorted(list(original_labels))
                            evaluator.data_loader.id_to_label = {i: label for i, label in enumerate(sorted_original_labels)}
                            evaluator.data_loader.label_to_id = {label: i for i, label in enumerate(sorted_original_labels)}
                            original_labels_found = True
                            
                            print(f"✅ [TRAINING_PIPELINE] Extracted labels from preprocessed samples:")
                            print(f"   - ID to Label: {evaluator.data_loader.id_to_label}")
                    
                    # Source 5: Extract categories from the processed dataframe
                    if not original_labels_found and label_column in df.columns:
                        unique_categories = sorted(df[label_column].unique().tolist())
                        print(f"🔍 [TRAINING_PIPELINE] Found unique categories in dataframe: {unique_categories}")
                        
                        # Check if categories are numeric strings (indicating processed data)
                        if all(str(cat).isdigit() for cat in unique_categories):
                            print(f"⚠️ [TRAINING_PIPELINE] Categories are numeric strings - data already processed")
                            print(f"   - This means real category names were lost during preprocessing")
                            print(f"   - Will use generic Class_X labels")
                            
                            # Create generic meaningful labels
                            evaluator.data_loader.label_to_id = {f"Class_{cat}": cat for cat in unique_categories}
                            evaluator.data_loader.id_to_label = {cat: f"Class_{cat}" for cat in unique_categories}
                        else:
                            print(f"✅ [TRAINING_PIPELINE] Categories are text labels - using as-is")
                            # Create normal mapping
                            evaluator.data_loader.label_to_id = {label: i for i, label in enumerate(unique_categories)}
                            evaluator.data_loader.id_to_label = {i: label for i, label in enumerate(unique_categories)}
                        
                        print(f"✅ [TRAINING_PIPELINE] Final label mapping:")
                        print(f"   - ID to Label: {evaluator.data_loader.id_to_label}")
                    
                    # Final check - if still no labels found
                    if not original_labels_found:
                        print(f"❌ [TRAINING_PIPELINE] Could not find any label mapping, using fallback")
                        # Create simple numeric labels
                        evaluator.data_loader.id_to_label = {0: 'Class_0', 1: 'Class_1'}
                        evaluator.data_loader.label_to_id = {'Class_0': 0, 'Class_1': 1}
                        print(f"⚠️ [TRAINING_PIPELINE] Using fallback labels: {evaluator.data_loader.id_to_label}")

                # Run comprehensive evaluation
                self.current_phase = "comprehensive_evaluation"
                if progress_callback:
                    progress_callback(self.current_phase, "Running comprehensive evaluation...", 0.3)

                # Check if training should stop before starting evaluation
                if self.is_training_stopped():
                    return {
                        'status': 'stopped',
                        'message': 'Training stopped during setup',
                        'results': {},
                        'models_completed': 0,
                        'elapsed_time': time.time() - self.start_time
                    }

                # Get selected models and vectorization methods from step 3
                selected_models = step3_data.get('selected_models', []) if step3_data and isinstance(step3_data, dict) else []
                selected_vectorization = step3_data.get('selected_vectorization', []) if step3_data and isinstance(step3_data, dict) else []
                
                # FIXED: Update step1_data with sampled dataframe to ensure consistency
                if sampling_config and sampling_config.get('num_samples'):
                    print(f"🎯 [TRAINING_PIPELINE] Updating step1_data with sampled dataframe: {len(df):,} samples")
                    step1_data['dataframe'] = df.copy()  # Use the already sampled dataframe
                
                # Debug: Check what we're passing to evaluator
                print(f"🔍 [TRAINING_PIPELINE] Debug - Data being passed to evaluator:")
                print(f"   • step1_data exists: {step1_data is not None}")
                if step1_data:
                    print(f"   • step1_data type: {type(step1_data)}")
                    print(f"   • step1_data keys: {list(step1_data.keys()) if isinstance(step1_data, dict) else 'Not a dict'}")
                    if isinstance(step1_data, dict) and 'dataframe' in step1_data:
                        print(f"   • step1_data['dataframe'] size: {len(step1_data['dataframe']):,}")
                print(f"   • step2_data exists: {step2_data is not None}")
                if step2_data:
                    print(f"   • step2_data type: {type(step2_data)}")
                    print(f"   • step2_data keys: {list(step2_data.keys()) if isinstance(step2_data, dict) else 'Not a dict'}")
                
                # Check if ensemble learning is enabled
                if step3_data and isinstance(step3_data, dict):
                    ensemble_config = step3_data.get('ensemble_learning', {})
                    ensemble_enabled = ensemble_config.get('enabled', False)
                else:
                    ensemble_config = {}
                    ensemble_enabled = False
                
                print(f"🚀 [TRAINING_PIPELINE] Ensemble Learning: {'Enabled' if ensemble_enabled else 'Disabled'}")
                if ensemble_enabled:
                    print(f"   • Final Estimator: {ensemble_config.get('final_estimator', 'logistic_regression')}")
                    print(f"   • Base Models: KNN + Decision Tree + Naive Bayes")
                
                # Run comprehensive evaluation with selected models and embeddings
                # FIXED: Pass the already sampled dataframe to evaluator
                evaluation_results = evaluator.run_comprehensive_evaluation(
                    max_samples=None,  # Sampling already handled in pipeline
                    skip_csv_prompt=True,
                    sampling_config=sampling_config,  # Pass sampling config
                    selected_models=selected_models,
                    selected_embeddings=selected_vectorization,
                    stop_callback=self.is_training_stopped,  # Pass stop callback
                    step3_data=step3_data,  # Pass Step 3 configuration for KNN
                    preprocessing_config=self.preprocessing_config,  # Pass preprocessing config from instance
                    step1_data=step1_data,  # Pass Step 1 data with sampled dataframe
                    step2_data=step2_data,  # Pass Step 2 data with column configuration
                    ensemble_config=ensemble_config  # Pass ensemble learning configuration
                )
                
                # Update progress based on evaluation progress
                self.current_phase = "analysis"
                if progress_callback:
                    progress_callback(self.current_phase, "Analyzing results...", 0.8)

                # Get comprehensive results
                comprehensive_results = evaluation_results.get('all_results', [])
                successful_results = [r for r in comprehensive_results if r['status'] == 'success']
                
                self.models_completed = len(successful_results)
                self.total_models = evaluation_results.get('total_combinations', 0)

                # Generate summary report
                self.current_phase = "report_generation"
                if progress_callback:
                    progress_callback(self.current_phase, "Generating comprehensive report...", 0.9)

                # Display results (this will print the comprehensive report)
                evaluator.save_results()

                # Clean up temp file
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

                # Restore original file path
                if original_file_path:
                    evaluator.data_loader.file_path = original_file_path

                # Save results to cache
                self.current_phase = "caching_results"
                if progress_callback:
                    progress_callback(self.current_phase, "Saving results to cache...", 0.95)
                
                try:
                    # CRITICAL FIX: Include step data and label mapping in cache
                    # This ensures cache contains all necessary information for label mapping
                    cache_results = {
                        'status': 'success',
                        'message': 'Comprehensive evaluation completed successfully',
                        'results': evaluation_results,
                        'comprehensive_results': comprehensive_results,
                        'successful_combinations': evaluation_results.get('successful_combinations', 0),
                        'total_combinations': evaluation_results.get('total_combinations', 0),
                        'best_combinations': evaluator.best_combinations if hasattr(evaluator, 'best_combinations') else {},
                        'total_models': self.total_models,
                        'models_completed': self.models_completed,
                        'elapsed_time': time.time() - self.start_time,
                        'evaluation_time': evaluation_results.get('evaluation_time', 0),
                        'data_info': evaluation_results.get('data_info', {}),
                        'embedding_info': evaluation_results.get('embedding_info', {}),
                        'from_cache': False,
                        'cache_key': cache_key,
                        # CRITICAL: Include step data for label mapping
                        'step1_data': step1_data,
                        'step2_data': step2_data,
                        'step3_data': step3_data,
                        # CRITICAL: Include label mapping at top level
                        # Use self.original_label_mapping if available, otherwise fallback to evaluator
                        'labels': self.original_label_mapping if hasattr(self, 'original_label_mapping') and self.original_label_mapping else (evaluator.data_loader.id_to_label if hasattr(evaluator, 'data_loader') and evaluator.data_loader else {}),
                        'label_mapping': self.original_label_mapping if hasattr(self, 'original_label_mapping') and self.original_label_mapping else (evaluator.data_loader.id_to_label if hasattr(evaluator, 'data_loader') and evaluator.data_loader else {})
                    }
                    
                    # Save to cache
                    self._save_to_cache(cache_key, cache_results)
                    
                except Exception as e:
                    print(f"Warning: Could not save results to cache: {e}")

                # Finalize results
                self.current_phase = "completed"
                self.training_status = "completed"
                if progress_callback:
                    progress_callback(self.current_phase, "Comprehensive evaluation completed!", 1.0)

                return cache_results

            except Exception as e:
                # Clean up temp file on error
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
                raise e

        except Exception as e:
            self.training_status = "error"
            return {
                'status': 'error',
                'message': f'Comprehensive evaluation failed: {str(e)}',
                'error': str(e)
            }
    
    def _apply_sampling(self, df: pd.DataFrame, sampling_config: Dict, label_column: str = None) -> pd.DataFrame:
        """Apply sampling configuration to dataset"""
        try:
            num_samples = sampling_config.get('num_samples', len(df))
            strategy = sampling_config.get('sampling_strategy', 'Random')
            
            print(f"📊 Dataset size: {len(df):,}, Requested samples: {num_samples:,}, Strategy: {strategy}")
            
            if num_samples >= len(df):
                print(f"ℹ️ Requested samples >= dataset size, using full dataset")
                return df
            
            if 'Stratified' in strategy and label_column and label_column in df.columns:
                # Stratified sampling
                print(f"🎯 Applying stratified sampling with label column: {label_column}")
                from sklearn.model_selection import train_test_split
                try:
                    df_sample, _ = train_test_split(
                        df, 
                        train_size=num_samples, 
                        stratify=df[label_column],
                        random_state=42
                    )
                    print(f"✅ Stratified sampling successful: {len(df):,} → {len(df_sample):,}")
                    return df_sample
                except Exception as e:
                    print(f"⚠️ Stratified sampling failed ({e}), falling back to random sampling")
                    strategy = 'Random'
            
            if 'Stratified' in strategy:
                print(f"⚠️ Stratified sampling requested but label column '{label_column}' not available, using random sampling")
            
            # Random sampling
            print(f"🎲 Applying random sampling")
            df_sample = df.sample(n=num_samples, random_state=42)
            print(f"✅ Random sampling successful: {len(df):,} → {len(df_sample):,}")
            return df_sample
                
        except Exception as e:
            print(f"❌ Sampling failed: {e}, using full dataset")
            return df
    
    def _apply_preprocessing(self, df: pd.DataFrame, step2_data: Dict) -> pd.DataFrame:
        """Apply preprocessing options to dataset"""
        try:
            text_column = step2_data.get('text_column')
            label_column = step2_data.get('label_column')
            
            # Text cleaning
            if step2_data.get('text_cleaning', True):
                df[text_column] = df[text_column].astype(str).str.replace(
                    r'[^\w\s]', '', regex=True
                ).str.strip()
            
            # Data validation (remove nulls)
            if step2_data.get('data_validation', True):
                df = df.dropna(subset=[text_column, label_column])
            
            # Category mapping
            if step2_data.get('category_mapping', True):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                
                # Lưu trữ labels gốc trước khi encode
                original_labels = df[label_column].unique()
                self.original_label_values = original_labels.tolist()
                self.label_encoder = le
                
                # Encode labels
                df[label_column] = le.fit_transform(df[label_column])
                
                # CRITICAL FIX: Only set original_label_mapping if not already set
                # This prevents overwriting labels from step1_data
                if not hasattr(self, 'original_label_mapping') or not self.original_label_mapping:
                    self.original_label_mapping = {i: label for i, label in enumerate(le.classes_)}
                    print(f"✅ [PREPROCESSING] Set original label mapping: {self.original_label_mapping}")
                else:
                    print(f"✅ [PREPROCESSING] Preserved existing label mapping: {self.original_label_mapping}")
                    print(f"   - Labels from step1_data preserved, not overwritten")
                
                print(f"   - Original labels: {self.original_label_values}")
                print(f"   - Encoded labels: {list(le.classes_)}")
            
            # Memory optimization
            if step2_data.get('memory_optimization', True):
                df[text_column] = df[text_column].astype('category')
            
            return df
            
        except Exception as e:
            print(f"Warning: Preprocessing failed: {e}")
            return df
    
    def _create_data_splits(self, X: np.ndarray, y: np.ndarray, 
                           data_split: Dict) -> Tuple:
        """Create train/test splits (validation handled by CV)"""
        try:
            # Use validation manager if available
            if hasattr(validation_manager, 'split_data'):
                return validation_manager.split_data(X, y)
            else:
                # Fallback to sklearn - only split into train and test
                from sklearn.model_selection import train_test_split
                
                # Split: separate test set only
                test_size = data_split.get('test', 0.2)
                
                # Check if we have enough samples for stratified split
                unique_classes = len(np.unique(y))
                test_samples = int(len(y) * test_size)
                
                if test_samples >= unique_classes:
                    # Use stratified split if we have enough test samples
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                else:
                    # Use non-stratified split for small datasets
                    print(f"⚠️ Small dataset detected: {len(y)} samples, {unique_classes} classes")
                    print(f"   Test samples: {test_samples} < {unique_classes} classes")
                    print(f"   Using non-stratified split to avoid error")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                
                # No separate validation set - CV will handle it
                X_val, y_val = np.array([]), np.array([])
                
                return X_train, X_val, X_test, y_train, y_val, y_test
                
        except Exception as e:
            print(f"Warning: Data splitting failed, using simple split: {e}")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, np.array([]), X_test, y_train, np.array([]), y_test
    
    def _prepare_vectorized_data(self, X_train: np.ndarray, X_val: np.ndarray, 
                                X_test: np.ndarray, y_train: np.ndarray, 
                                y_val: np.ndarray, y_test: np.ndarray,
                                vectorization_methods: List[str]) -> Dict:
        """Prepare vectorized data for all selected methods"""
        
        vectorized_data = {}
        
        try:
            # Initialize text vectorizer
            if hasattr(TextVectorizer, '__init__'):
                text_vectorizer = TextVectorizer()
            else:
                # Fallback vectorizer
                text_vectorizer = None
            
            for method in vectorization_methods:
                if method == 'Bag of Words (BoW)':
                    vectorized_data['bow'] = self._vectorize_bow(
                        text_vectorizer, X_train, X_val, X_test
                    )
                elif method == 'TF-IDF':
                    vectorized_data['tfidf'] = self._vectorize_tfidf(
                        text_vectorizer, X_train, X_val, X_test
                    )
                elif method == 'Word Embeddings':
                    vectorized_data['embeddings'] = self._vectorize_embeddings(
                        text_vectorizer, X_train, X_val, X_test
                    )
            
            # Add labels separately
            vectorized_data['labels'] = {
                'train': y_train,
                'val': y_val,
                'test': y_test
            }
            
        except Exception as e:
            print(f"Warning: Vectorization failed: {e}")
            # Create simple fallback vectorization
            vectorized_data = self._create_fallback_vectorization(
                X_train, X_val, X_test, y_train, y_val, y_test
            )
        
        return vectorized_data
    
    def _create_fallback_vectorization(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Create fallback vectorization when main methods fail"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Simple TF-IDF fallback
            tfidf = TfidfVectorizer(max_features=1000)
            X_train_vec = tfidf.fit_transform(X_train)
            X_val_vec = tfidf.transform(X_val) if len(X_val) > 0 else None
            X_test_vec = tfidf.transform(X_test)
            
            return {
                'bow': {
                    'train': X_train_vec,
                    'val': X_val_vec,
                    'test': X_test_vec
                },
                'tfidf': {
                    'train': X_train_vec,
                    'val': X_val_vec,
                    'test': X_test_vec
                },
                'embeddings': {
                    'train': X_train_vec,
                    'val': X_val_vec,
                    'test': X_test_vec
                },
                'labels': {
                    'train': y_train,
                    'val': y_val,
                    'test': y_test
                }
            }
        except Exception as e:
            print(f"Warning: Fallback vectorization failed: {e}")
            return {
                'labels': {
                    'train': y_train,
                    'val': y_val,
                    'test': y_test
                }
            }
    
    def _vectorize_bow(self, vectorizer, X_train, X_val, X_test):
        """Vectorize data using Bag of Words with GPU optimization"""
        try:
            if hasattr(vectorizer, 'fit_transform_bow'):
                X_train_bow = vectorizer.fit_transform_bow(X_train)
                X_val_bow = vectorizer.transform_bow(X_val) if len(X_val) > 0 else None
                X_test_bow = vectorizer.transform_bow(X_test)
            else:
                # Fallback to sklearn
                from sklearn.feature_extraction.text import CountVectorizer
                cv = CountVectorizer(max_features=1000)
                X_train_bow = cv.fit_transform(X_train)
                X_val_bow = cv.transform(X_val) if len(X_val) > 0 else None
                X_test_bow = cv.transform(X_test)
            
            # MEMORY OPTIMIZATION: Keep sparse matrices for memory efficiency
            from scipy import sparse
            from config import ENABLE_GPU_OPTIMIZATION, FORCE_DENSE_CONVERSION
            
            if sparse.issparse(X_train_bow):
                print(f"   📊 Using BoW sparse matrix format for memory efficiency")
                print(f"   💾 Memory saved: Keeping sparse matrices (no dense conversion)")
                # Keep sparse matrices - modern models handle them efficiently
                # No conversion to prevent memory overflow
            
            return {
                'train': X_train_bow,
                'val': X_val_bow,
                'test': X_test_bow
            }
        except Exception as e:
            print(f"Warning: BoW vectorization failed: {e}")
            return None
    
    def _vectorize_tfidf(self, vectorizer, X_train, X_val, X_test):
        """Vectorize data using TF-IDF with GPU optimization"""
        try:
            if hasattr(vectorizer, 'fit_transform_tfidf'):
                X_train_tfidf = vectorizer.fit_transform_tfidf(X_train)
                X_val_tfidf = vectorizer.transform_tfidf(X_val) if len(X_val) > 0 else None
                X_test_tfidf = vectorizer.transform_tfidf(X_test)
            else:
                # Fallback to sklearn
                from sklearn.feature_extraction.text import TfidfVectorizer
                tfidf = TfidfVectorizer(max_features=1000)
                X_train_tfidf = tfidf.fit_transform(X_train)
                X_val_tfidf = tfidf.transform(X_val) if len(X_val) > 0 else None
                X_test_tfidf = tfidf.transform(X_test)
            
            # MEMORY OPTIMIZATION: Keep sparse matrices for memory efficiency
            from scipy import sparse
            from config import ENABLE_GPU_OPTIMIZATION, FORCE_DENSE_CONVERSION
            
            if sparse.issparse(X_train_tfidf):
                print(f"   📊 Using TF-IDF sparse matrix format for memory efficiency")
                print(f"   💾 Memory saved: Keeping sparse matrices (no dense conversion)")
                # Keep sparse matrices - modern models handle them efficiently
                # No conversion to prevent memory overflow
            
            return {
                'train': X_train_tfidf,
                'val': X_val_tfidf,
                'test': X_test_tfidf
            }
        except Exception as e:
            print(f"Warning: TF-IDF vectorization failed: {e}")
            return None
    
    def _vectorize_embeddings(self, vectorizer, X_train, X_val, X_test):
        """Vectorize data using Word Embeddings"""
        try:
            if hasattr(vectorizer, 'transform_embeddings'):
                X_train_emb = vectorizer.transform_embeddings(X_train)
                X_val_emb = vectorizer.transform_embeddings(X_val) if len(X_val) > 0 else None
                X_test_emb = vectorizer.transform_embeddings(X_test)
            else:
                # Fallback to simple embeddings
                X_train_emb = self._create_simple_embeddings(X_train)
                X_val_emb = self._create_simple_embeddings(X_val) if len(X_val) > 0 else None
                X_test_emb = self._create_simple_embeddings(X_test)
            
            return {
                'train': X_train_emb,
                'val': X_val_emb,
                'test': X_test_emb
            }
        except Exception as e:
            print(f"Warning: Embeddings vectorization failed: {e}")
            return None
    
    def _create_simple_embeddings(self, X):
        """Create simple embeddings as fallback"""
        try:
            # Simple character-level encoding
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(max_features=100, analyzer='char')
            return tfidf.fit_transform(X)
        except:
            # Last resort: random embeddings
            return np.random.rand(len(X), 100)
    
    def _train_all_models(self, vectorized_data: Dict, selected_models: List[str], 
                          cv_config: Dict, step3_data: Dict = None, progress_callback=None) -> Dict:
        """Train all selected models with all vectorization methods"""
        
        results = {}
        cv_folds = cv_config.get('cv_folds', 5)
        random_state = cv_config.get('random_state', 42)
        
        # Get labels from vectorized_data
        labels_dict = vectorized_data.get('labels', {})
        if not labels_dict:
            print("Warning: No labels found in vectorized data")
            return results
        
        for model_name in selected_models:
            for vec_method, vec_data in vectorized_data.items():
                if vec_method == 'labels' or vec_data is None:
                    continue
                
                if progress_callback:
                    progress = 0.4 + (0.4 * (self.models_completed / self.total_models))
                    progress_callback(
                        "model_training", 
                        f"Training {model_name} with {vec_method}...", 
                        progress
                    )
                
                try:
                    # Train model with labels
                    model_result = self._train_single_model(
                        model_name, vec_method, vec_data, labels_dict, cv_folds, random_state, step3_data
                    )
                    
                    if model_result:
                        key = f"{model_name}_{vec_method}"
                        results[key] = model_result
                        self.models_completed += 1
                        
                        if progress_callback:
                            progress = 0.4 + (0.4 * (self.models_completed / self.total_models))
                            progress_callback(
                                "model_training", 
                                f"Completed {model_name} with {vec_method}", 
                                progress
                            )
                
                except Exception as e:
                    print(f"Warning: Failed to train {model_name} with {vec_method}: {e}")
                    continue
        
        return results
    
    def _train_single_model(self, model_name: str, vec_method: str, 
                           vec_data: Dict, labels_dict: Dict, cv_folds: int, random_state: int, step3_data: Dict = None) -> Dict:
        """Train a single model with specific vectorization method"""
        
        try:
            # Check if vec_data has the required structure
            if not isinstance(vec_data, dict) or 'train' not in vec_data:
                print(f"Warning: Invalid vectorization data structure for {model_name} with {vec_method}")
                return None
            
            # Check cache first
            cache_result = self._check_model_cache(model_name, vec_method, vec_data, labels_dict, cv_folds, random_state, step3_data)
            if cache_result:
                print(f"✅ Cache HIT for {model_name}_{vec_method}")
                return cache_result
            
            print(f"❌ Cache MISS for {model_name}_{vec_method} - Training new model")
            
            # Check if labels_dict has the required structure
            if not isinstance(labels_dict, dict) or 'train' not in labels_dict:
                print(f"Warning: Invalid labels structure for {model_name} with {vec_method}")
                return None
            
            # Use NewModelTrainer if available and properly configured
            if hasattr(NewModelTrainer, '__init__'):
                try:
                    # Try to create NewModelTrainer with proper arguments
                    if hasattr(NewModelTrainer, 'train_validate_test_model'):
                        # Check if we can create an instance without arguments first
                        try:
                            model_trainer = NewModelTrainer()
                        except TypeError as e:
                            # If constructor requires arguments, try with defaults
                            try:
                                model_trainer = NewModelTrainer(
                                    cv_folds=cv_folds,
                                    validation_size=0.0,  # No separate validation set
                                    test_size=0.2
                                )
                            except Exception as e2:
                                print(f"Warning: NewModelTrainer constructor failed: {e2}")
                                # Skip to sklearn fallback
                                return self._train_sklearn_fallback(
                                    model_name, vec_method, vec_data, labels_dict, cv_folds, random_state, step3_data
                                )
                        
                        # Map model names to trainer method names
                        model_mapping = {
                            'K-Nearest Neighbors': 'knn',
                            'Decision Tree': 'decision_tree',
                            'Naive Bayes': 'naive_bayes',
                            'K-Means Clustering': 'kmeans',
                            'Support Vector Machine (SVM)': 'svm'
                        }
                        
                        trainer_method = model_mapping.get(model_name, model_name.lower())
                        
                        try:
                            labels, _, _, _, accuracy, report = model_trainer.train_validate_test_model(
                                trainer_method,
                                vec_data['train'], labels_dict['train'],
                                vec_data['val'], labels_dict['val'],
                                vec_data['test'], labels_dict['test']
                            )
                            
                            return {
                                'labels': labels,
                                'accuracy': accuracy,
                                'report': report,
                                'vectorization': vec_method
                            }
                        except Exception as e3:
                            print(f"Warning: NewModelTrainer training failed for {model_name}: {e3}")
                            # Fall through to sklearn fallback
                            
                except Exception as e:
                    print(f"Warning: NewModelTrainer failed for {model_name}: {e}")
                    # Fall through to sklearn fallback
            
            # Fallback to sklearn models
            return self._train_sklearn_fallback(
                model_name, vec_method, vec_data, labels_dict, cv_folds, random_state, step3_data
            )
            
        except Exception as e:
            print(f"Warning: Model training failed for {model_name}: {e}")
            return None
    
    def _train_sklearn_fallback(self, model_name: str, vec_method: str, 
                               vec_data: Dict, labels_dict: Dict, cv_folds: int, random_state: int, step3_data: Dict = None) -> Dict:
        """Fallback training using sklearn models with optional Optuna optimization"""
        
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import accuracy_score, classification_report
            
            # Check if Optuna optimization is enabled for this model
            use_optuna = (self.optuna_optimizer is not None and 
                         self.optuna_config.get('enable', False) and
                         self._is_optuna_compatible_model(model_name))
            
            if use_optuna:
                print(f"🚀 Running Optuna optimization for {model_name}...")
                
                # Get model class for Optuna optimization
                model_class = self._get_model_class(model_name)
                if model_class:
                    # Run Optuna optimization
                    optuna_results = self.optuna_optimizer.optimize_model(
                        model_name.lower().replace(' ', '_').replace('-', '_'),
                        vec_data['train'], labels_dict['train'],
                        vec_data['val'], labels_dict['val'],
                        model_class
                    )
                    
                    if optuna_results:
                        print(f"✅ Optuna optimization completed for {model_name}")
                        print(f"   Best score: {optuna_results['best_score']:.4f}")
                        
                        # Use best parameters to create final model
                        best_params = optuna_results['best_params']
                        model = self._create_sklearn_model(model_name, random_state, step3_data, best_params)
                    else:
                        print(f"⚠️ Optuna optimization failed for {model_name}, using default parameters")
                        model = self._create_sklearn_model(model_name, random_state, step3_data)
                else:
                    print(f"⚠️ Model class not found for {model_name}, skipping Optuna optimization")
                    model = self._create_sklearn_model(model_name, random_state, step3_data)
            else:
                # Create model without optimization
                model = self._create_sklearn_model(model_name, random_state, step3_data)
            
            # Train model
            model.fit(vec_data['train'], labels_dict['train'])
            
            # Make predictions
            y_pred = model.predict(vec_data['test'])
            
            # Calculate accuracy
            accuracy = accuracy_score(labels_dict['test'], y_pred)
            
            # Generate report
            report = classification_report(
                labels_dict['test'], y_pred, output_dict=True
            )
            
            result = {
                'labels': y_pred,
                'accuracy': accuracy,
                'report': report,
                'vectorization': vec_method,
                'model': model
            }
            
            # Add Optuna results if available
            if use_optuna and 'optuna_results' in locals():
                result['optuna_results'] = optuna_results
            
            # Save to per-model cache
            try:
                # Prepare evaluation predictions for cache
                eval_predictions = self._prepare_eval_predictions_for_cache(
                    vec_data, labels_dict, model, vec_method
                )
                
                # Create SHAP sample for caching
                shap_sample = None
                try:
                    # Use a subset of test data for SHAP sample
                    if eval_predictions is not None and len(eval_predictions) > 0:
                        # Extract only feature columns (exclude true_labels and predictions)
                        feature_columns = [col for col in eval_predictions.columns if col not in ['true_labels', 'predictions']]
                        test_data = eval_predictions[feature_columns]
                        
                        # Sample for SHAP (max 1000 samples)
                        if len(test_data) > 1000:
                            import numpy as np
                            indices = np.random.choice(len(test_data), 1000, replace=False)
                            shap_sample = test_data.iloc[indices]
                        else:
                            shap_sample = test_data
                        
                        print(f"💾 Created SHAP sample for {model_name}: {len(shap_sample)} samples")
                except Exception as shap_error:
                    print(f"Warning: Failed to create SHAP sample for {model_name}: {shap_error}")
                
                # Save cache
                cache_path = self._save_model_cache(
                    model_name=model_name,
                    vec_method=vec_method,
                    vec_data=vec_data,
                    labels_dict=labels_dict,
                    cv_folds=cv_folds,
                    random_state=random_state,
                    model=model,
                    params=model.get_params() if hasattr(model, 'get_params') else {},
                    metrics={
                        'accuracy': accuracy,
                        'cv_scores': cv_scores.tolist() if 'cv_scores' in locals() else [],
                        'cv_mean': cv_scores.mean() if 'cv_scores' in locals() else 0,
                        'cv_std': cv_scores.std() if 'cv_scores' in locals() else 0
                    },
                    config={
                        'model_name': model_name,
                        'vec_method': vec_method,
                        'cv_folds': cv_folds,
                        'random_state': random_state
                    },
                    eval_predictions=eval_predictions,
                    shap_sample=shap_sample,
                    step3_data=step3_data
                )
                
                if cache_path:
                    result['cache_path'] = cache_path
                    result['cache_saved'] = True
                    
            except Exception as cache_error:
                print(f"Warning: Failed to save cache for {model_name}_{vec_method}: {cache_error}")
                result['cache_saved'] = False
            
            return result
            
        except Exception as e:
            print(f"Warning: Sklearn fallback failed: {e}")
            return None
    
    def _is_optuna_compatible_model(self, model_name: str) -> bool:
        """Check if model is compatible with Optuna optimization"""
        compatible_models = [
            'Random Forest', 'AdaBoost', 'Gradient Boosting',
            'XGBoost', 'LightGBM', 'CatBoost'
        ]
        return any(model in model_name for model in compatible_models)
    
    def _get_model_class(self, model_name: str):
        """Get model class for Optuna optimization"""
        try:
            if 'Random Forest' in model_name:
                from models.classification.random_forest_model import RandomForestModel
                return RandomForestModel
            elif 'AdaBoost' in model_name:
                from models.classification.adaboost_model import AdaBoostModel
                return AdaBoostModel
            elif 'Gradient Boosting' in model_name:
                from models.classification.gradient_boosting_model import GradientBoostingModel
                return GradientBoostingModel
            elif 'XGBoost' in model_name:
                from models.classification.xgboost_model import XGBoostModel
                return XGBoostModel
            elif 'LightGBM' in model_name:
                from models.classification.lightgbm_model import LightGBMModel
                return LightGBMModel
            elif 'CatBoost' in model_name:
                from models.classification.catboost_model import CatBoostModel
                return CatBoostModel
            else:
                return None
        except ImportError as e:
            print(f"Warning: Could not import model class for {model_name}: {e}")
            return None
    
    def _check_model_cache(self, model_name: str, vec_method: str, vec_data: Dict, 
                          labels_dict: Dict, cv_folds: int, random_state: int, 
                          step3_data: Dict = None) -> Optional[Dict]:
        """Check if model cache exists and is valid"""
        try:
            # Generate cache identifiers
            model_key = self._get_model_key(model_name)
            dataset_id = self._generate_dataset_id(vec_data, labels_dict)
            config_hash = self._generate_config_hash(model_name, vec_method, cv_folds, random_state, step3_data)
            dataset_fingerprint = self._generate_dataset_fingerprint(vec_data, labels_dict)
            
            # Check cache
            cache_exists, cache_info = cache_manager.check_cache_exists(
                model_key, dataset_id, config_hash, dataset_fingerprint
            )
            
            if cache_exists:
                # Load cached model
                cache_data = cache_manager.load_model_cache(model_key, dataset_id, config_hash)
                
                # Convert to expected format
                result = {
                    'model_name': model_name,
                    'embedding_name': vec_method,
                    'status': 'success',
                    'model': cache_data['model'],
                    'params': cache_data['params'],
                    'metrics': cache_data['metrics'],
                    'config': cache_data['config'],
                    'eval_predictions': cache_data['eval_predictions'],
                    'shap_sample': cache_data['shap_sample'],
                    'feature_names': cache_data['feature_names'],
                    'label_mapping': cache_data['label_mapping'],
                    'cache_hit': True,
                    'cache_path': cache_data['cache_path']
                }
                
                return result
            
            return None
            
        except Exception as e:
            print(f"Warning: Cache check failed for {model_name}_{vec_method}: {e}")
            return None
    
    def _save_model_cache(self, model_name: str, vec_method: str, vec_data: Dict,
                         labels_dict: Dict, cv_folds: int, random_state: int,
                         model, params: Dict, metrics: Dict, config: Dict,
                         eval_predictions: Optional[pd.DataFrame] = None,
                         shap_sample: Optional[pd.DataFrame] = None,
                         feature_names: Optional[List[str]] = None,
                         label_mapping: Optional[Dict] = None,
                         step3_data: Dict = None) -> str:
        """Save model cache"""
        try:
            # Generate cache identifiers
            model_key = self._get_model_key(model_name)
            dataset_id = self._generate_dataset_id(vec_data, labels_dict)
            config_hash = self._generate_config_hash(model_name, vec_method, cv_folds, random_state, step3_data)
            dataset_fingerprint = self._generate_dataset_fingerprint(vec_data, labels_dict)
            
            # Save cache
            cache_path = cache_manager.save_model_cache(
                model_key=model_key,
                dataset_id=dataset_id,
                config_hash=config_hash,
                dataset_fingerprint=dataset_fingerprint,
                model=model,
                params=params,
                metrics=metrics,
                config=config,
                eval_predictions=eval_predictions,
                shap_sample=shap_sample,
                feature_names=feature_names,
                label_mapping=label_mapping
            )
            
            print(f"💾 Cache saved for {model_name}_{vec_method} at {cache_path}")
            return cache_path
            
        except Exception as e:
            print(f"Warning: Cache save failed for {model_name}_{vec_method}: {e}")
            return ""
    
    def _get_model_key(self, model_name: str) -> str:
        """Get model key for cache"""
        # Map display names to cache keys
        model_mapping = {
            'Random Forest': 'random_forest',
            'AdaBoost': 'adaboost',
            'Gradient Boosting': 'gradient_boosting',
            'XGBoost': 'xgboost',
            'LightGBM': 'lightgbm',
            'CatBoost': 'catboost',
            'K-Nearest Neighbors': 'knn',
            'Decision Tree': 'decision_tree',
            'Naive Bayes': 'naive_bayes',
            'SVM': 'svm',
            'Logistic Regression': 'logistic_regression',
            'Linear SVC': 'linear_svc',
            'K-Means Clustering': 'kmeans'
        }
        return model_mapping.get(model_name, model_name.lower().replace(' ', '_'))
    
    def _generate_dataset_id(self, vec_data: Dict, labels_dict: Dict) -> str:
        """Generate dataset ID for cache"""
        # Use data shape and label info to generate ID
        train_shape = vec_data.get('train', np.array([])).shape if 'train' in vec_data else (0, 0)
        test_shape = vec_data.get('test', np.array([])).shape if 'test' in vec_data else (0, 0)
        num_classes = len(set(labels_dict.get('train', []))) if 'train' in labels_dict else 0
        
        dataset_info = f"{train_shape[0]}x{train_shape[1]}_{test_shape[0]}x{test_shape[1]}_{num_classes}classes"
        import hashlib
        return hashlib.md5(dataset_info.encode()).hexdigest()[:8]
    
    def _generate_config_hash(self, model_name: str, vec_method: str, cv_folds: int,
                             random_state: int, step3_data: Dict = None) -> str:
        """Generate configuration hash for cache"""
        config = {
            'model_name': model_name,
            'vec_method': vec_method,
            'cv_folds': cv_folds,
            'random_state': random_state,
            'timestamp': datetime.now().isoformat()
        }
        
        if step3_data:
            config.update({
                'optuna_enabled': step3_data.get('optuna', {}).get('enabled', False),
                'optuna_trials': step3_data.get('optuna', {}).get('trials', 100),
                'optuna_timeout': step3_data.get('optuna', {}).get('timeout', None)
            })
        
        return cache_manager.generate_config_hash(config)
    
    def _generate_dataset_fingerprint(self, vec_data: Dict, labels_dict: Dict) -> str:
        """Generate dataset fingerprint for cache"""
        train_shape = vec_data.get('train', np.array([])).shape if 'train' in vec_data else (0, 0)
        test_shape = vec_data.get('test', np.array([])).shape if 'test' in vec_data else (0, 0)
        num_classes = len(set(labels_dict.get('train', []))) if 'train' in labels_dict else 0
        
        dataset_info = {
            'train_shape': train_shape,
            'test_shape': test_shape,
            'num_classes': num_classes,
            'timestamp': datetime.now().isoformat()
        }
        
        return cache_manager.generate_dataset_fingerprint(
            dataset_path="",
            dataset_size=0,
            num_rows=train_shape[0] + test_shape[0]
        )
    
    def _prepare_eval_predictions_for_cache(self, vec_data: Dict, labels_dict: Dict, 
                                          model, vec_method: str) -> pd.DataFrame:
        """Prepare evaluation predictions for cache storage
        
        Args:
            vec_data: Vectorized data
            labels_dict: Labels dictionary
            model: Trained model
            vec_method: Vectorization method
            
        Returns:
            DataFrame with evaluation predictions
        """
        try:
            # Get test data
            X_test = vec_data.get('test')
            y_test = labels_dict.get('test')
            
            if X_test is None or y_test is None:
                return None
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                y_pred = model.predict(X_test)
                
                # Create DataFrame with probabilities
                proba_df = pd.DataFrame(y_pred_proba, 
                                      columns=[f'proba__class_{i}' for i in range(y_pred_proba.shape[1])])
                
                eval_df = pd.DataFrame({
                    'y_true': y_test,
                    'y_pred': y_pred
                })
                
                # Combine with probabilities
                eval_df = pd.concat([eval_df, proba_df], axis=1)
                
            else:
                y_pred = model.predict(X_test)
                eval_df = pd.DataFrame({
                    'y_true': y_test,
                    'y_pred': y_pred
                })
            
            return eval_df
            
        except Exception as e:
            print(f"Warning: Failed to prepare eval predictions: {e}")
            return None
    
    def _create_sklearn_model(self, model_name: str, random_state: int, step3_data: Dict = None, optuna_params: Dict = None):
        """Create sklearn model instance with configuration from Step 3 and optional Optuna parameters"""
        
        if 'K-Nearest Neighbors' in model_name:
            # Use custom KNN model with advanced optimization
            from models.classification.knn_model import KNNModel
            
            # Get KNN configuration from Step 3 if available
            knn_config = step3_data.get('knn_config', {}) if step3_data else {}
            optimization_method = knn_config.get('optimization_method', 'Default K=5')
            
            print(f"\n🔍 [KNN MODEL CREATION] Debug information:")
            print(f"   • Step3 data exists: {bool(step3_data)}")
            print(f"   • KNN config exists: {bool(knn_config)}")
            print(f"   • Optimization method: {optimization_method}")
            print(f"   • Full KNN config: {knn_config}")
            print(f"   • Step3 data keys: {list(step3_data.keys()) if step3_data else 'None'}")
            print(f"   • KNN config keys: {list(knn_config.keys()) if knn_config else 'None'}")
            print(f"   • K value in config: {knn_config.get('k_value', 'NOT FOUND')}")
            print(f"   • Optimization method check: {optimization_method in ['Optimal K (Cosine Metric)', 'Grid Search (All Parameters)']}")
            
            if optimization_method == "Manual Input":
                # Use manual configuration
                k_value = knn_config.get('k_value', 5)
                weights = knn_config.get('weights', 'uniform')
                metric = knn_config.get('metric', 'cosine')
                
                print(f"🎯 [KNN] Using MANUAL configuration:")
                print(f"   • K Value: {k_value}")
                print(f"   • Weights: {weights}")
                print(f"   • Metric: {metric}")
                
                knn_model = KNNModel(
                    n_neighbors=k_value,
                    weights=weights,
                    metric=metric,
                    random_state=random_state
                )
                
                print(f"✅ [KNN] Model created with K={k_value}")
                return knn_model
                
            elif optimization_method in ["Optimal K (Cosine Metric)", "Grid Search (All Parameters)"]:
                # Use the BEST K found from optimization in Step 3
                best_k = knn_config.get('k_value', 5)
                best_weights = knn_config.get('weights', 'uniform')
                best_metric = knn_config.get('metric', 'cosine')
                best_score = knn_config.get('best_score', 'N/A')
                
                print(f"🎯 [KNN] Using OPTIMIZED configuration from Step 3:")
                print(f"   • Best K: {best_k}")
                print(f"   • Best Weights: {best_weights}")
                print(f"   • Best Metric: {best_metric}")
                print(f"   • Best Score: {best_score}")
                print(f"   • Optimization Method: {optimization_method}")
                
                # Ensure we have valid values
                if best_k is None or best_k == 5:
                    print(f"⚠️ [KNN] Warning: Best K is {best_k}, falling back to default")
                    best_k = 5
                
                knn_model = KNNModel(
                    n_neighbors=best_k,
                    weights=best_weights,
                    metric=best_metric,
                    random_state=random_state
                )
                
                print(f"✅ [KNN] Model created with OPTIMIZED parameters:")
                print(f"   • K={best_k} (from optimization)")
                print(f"   • Weights={best_weights}")
                print(f"   • Metric={best_metric}")
                
                return knn_model
                
            else:
                # Fallback to default configuration
                print(f"⚠️ [KNN] No valid configuration found, using default:")
                print(f"   • K Value: 5 (default)")
                print(f"   • Weights: uniform (default)")
                print(f"   • Metric: euclidean (default)")
                
                knn_model = KNNModel(
                    n_neighbors=5,
                    weights='uniform',
                    metric='euclidean',
                    random_state=random_state
                )
                
                print(f"✅ [KNN] Model created with DEFAULT parameters: K=5")
                return knn_model
        
        elif ('Decision Tree' in model_name or 'decision_tree' in model_name.lower()):
            # Use custom Decision Tree model with advanced pruning
            from models.classification.decision_tree_model import DecisionTreeModel
            
            # Get Decision Tree configuration from Step 3 if available
            dt_config = step3_data.get('decision_tree_config', {}) if step3_data else {}
            pruning_method = dt_config.get('pruning_method', 'none')  # Default to 'none' for Windows compatibility
            cv_folds = dt_config.get('cv_folds', 5)
            max_depth = dt_config.get('max_depth', None)
            min_samples_split = dt_config.get('min_samples_split', 2)
            min_samples_leaf = dt_config.get('min_samples_leaf', 1)
            
            print(f"🎯 [Decision Tree] Using CUSTOM model with advanced pruning:")
            print(f"   • Pruning Method: {pruning_method}")
            print(f"   • CV Folds: {cv_folds}")
            print(f"   • Max Depth: {max_depth}")
            print(f"   • Min Samples Split: {min_samples_split}")
            print(f"   • Min Samples Leaf: {min_samples_leaf}")
            print(f"   • GPU Acceleration: True")
            print(f"   • GPU Library: auto")
            
            # GPU configuration - AUTO ENABLE by default
            use_gpu = True  # Always enable GPU for performance
            gpu_library = 'auto'  # Auto-select best GPU library
            
            # Ensure random_state is an integer
            safe_random_state = random_state if isinstance(random_state, int) else 42
            
            dt_model = DecisionTreeModel(
                pruning_method=pruning_method,
                cv_folds=cv_folds,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=safe_random_state,
                use_gpu=use_gpu,           # Enable GPU acceleration
                gpu_library=gpu_library    # GPU library selection
            )
            
            # Force GPU initialization if requested
            if use_gpu:
                dt_model._init_gpu_libraries()
                # IMPORTANT: After GPU init, check if pruning was disabled on Windows
                if dt_model.pruning_method == 'none':
                    print(f"   ✅ Pruning method automatically disabled: {pruning_method} → none")
            
            print(f"✅ [Decision Tree] Custom model created with pruning + GPU: {use_gpu}")
            return dt_model
        
        elif 'Naive Bayes' in model_name:
            # Use custom Naive Bayes model with automatic type selection
            from models.classification.naive_bayes_model import NaiveBayesModel
            
            print(f"🎯 [Naive Bayes] Using CUSTOM model with automatic type selection")
            print(f"   • CV Folds: None (no CV for Naive Bayes)")
            
            nb_model = NaiveBayesModel(random_state=random_state)
            
            print(f"✅ [Naive Bayes] Custom model created with automatic type selection")
            return nb_model
        
        elif 'K-Means' in model_name:
            # Use custom K-Means model with optimal K detection
            from models.clustering.kmeans_model import KMeansModel
            
            # Get K-Means configuration from Step 3 if available
            kmeans_config = step3_data.get('kmeans_config', {}) if step3_data else {}
            n_clusters = kmeans_config.get('n_clusters', 5)
            use_optimal_k = kmeans_config.get('use_optimal_k', False)
            
            print(f"🎯 [K-Means] Using CUSTOM model with optimal K detection:")
            print(f"   • Initial Clusters: {n_clusters}")
            print(f"   • Use Optimal K: {use_optimal_k}")
            
            kmeans_model = KMeansModel(
                n_clusters=n_clusters,
                random_state=random_state
            )
            
            print(f"✅ [K-Means] Custom model created with optimal K detection")
            return kmeans_model
        
        elif 'SVM' in model_name:
            # Use custom SVM model with advanced features
            from models.classification.svm_model import SVMModel
            
            print(f"🎯 [SVM] Using CUSTOM model with advanced features")
            
            svm_model = SVMModel(random_state=random_state)
            
            print(f"✅ [SVM] Custom model created with advanced features")
            return svm_model
        
        elif 'Logistic Regression' in model_name:
            # Use custom Logistic Regression model with multinomial support
            from models.classification.logistic_regression_model import LogisticRegressionModel
            
            print(f"🎯 [Logistic Regression] Using CUSTOM model with multinomial support")
            
            lr_model = LogisticRegressionModel(random_state=random_state)
            
            print(f"✅ [Logistic Regression] Custom model created with multinomial support")
            return lr_model
        
        elif 'Linear SVC' in model_name:
            # Use custom Linear SVC model with advanced features
            from models.classification.linear_svc_model import LinearSVCModel
            
            print(f"🎯 [Linear SVC] Using CUSTOM model with advanced features")
            
            lsvc_model = LinearSVCModel(random_state=random_state)
            
            print(f"✅ [Linear SVC] Custom model created with advanced features")
            return lsvc_model
        
        # Enhanced ML Models with Optuna support
        elif 'Random Forest' in model_name:
            from models.classification.random_forest_model import RandomForestModel
            
            print(f"🎯 [Random Forest] Using enhanced model with Optuna support")
            
            # Use Optuna parameters if available
            if optuna_params:
                rf_model = RandomForestModel(**optuna_params, random_state=random_state)
                print(f"✅ [Random Forest] Model created with Optuna parameters: {optuna_params}")
            else:
                rf_model = RandomForestModel(random_state=random_state)
                print(f"✅ [Random Forest] Model created with default parameters")
            
            return rf_model
        
        elif 'AdaBoost' in model_name:
            from models.classification.adaboost_model import AdaBoostModel
            
            print(f"🎯 [AdaBoost] Using enhanced model with Optuna support")
            
            # Use Optuna parameters if available
            if optuna_params:
                ab_model = AdaBoostModel(**optuna_params, random_state=random_state)
                print(f"✅ [AdaBoost] Model created with Optuna parameters: {optuna_params}")
            else:
                ab_model = AdaBoostModel(random_state=random_state)
                print(f"✅ [AdaBoost] Model created with default parameters")
            
            return ab_model
        
        elif 'Gradient Boosting' in model_name:
            from models.classification.gradient_boosting_model import GradientBoostingModel
            
            print(f"🎯 [Gradient Boosting] Using enhanced model with Optuna support")
            
            # Use Optuna parameters if available
            if optuna_params:
                gb_model = GradientBoostingModel(**optuna_params, random_state=random_state)
                print(f"✅ [Gradient Boosting] Model created with Optuna parameters: {optuna_params}")
            else:
                gb_model = GradientBoostingModel(random_state=random_state)
                print(f"✅ [Gradient Boosting] Model created with default parameters")
            
            return gb_model
        
        elif 'XGBoost' in model_name:
            from models.classification.xgboost_model import XGBoostModel
            
            print(f"🎯 [XGBoost] Using enhanced model with GPU support and Optuna")
            
            # Use Optuna parameters if available
            if optuna_params:
                xgb_model = XGBoostModel(**optuna_params, random_state=random_state)
                print(f"✅ [XGBoost] Model created with Optuna parameters: {optuna_params}")
            else:
                xgb_model = XGBoostModel(random_state=random_state)
                print(f"✅ [XGBoost] Model created with default parameters")
            
            return xgb_model
        
        elif 'LightGBM' in model_name:
            from models.classification.lightgbm_model import LightGBMModel
            
            print(f"🎯 [LightGBM] Using enhanced model with GPU support and Optuna")
            
            # Use Optuna parameters if available
            if optuna_params:
                lgb_model = LightGBMModel(**optuna_params, random_state=random_state)
                print(f"✅ [LightGBM] Model created with Optuna parameters: {optuna_params}")
            else:
                lgb_model = LightGBMModel(random_state=random_state)
                print(f"✅ [LightGBM] Model created with default parameters")
            
            return lgb_model
        
        elif 'CatBoost' in model_name:
            from models.classification.catboost_model import CatBoostModel
            
            print(f"🎯 [CatBoost] Using enhanced model with GPU support and Optuna")
            
            # Use Optuna parameters if available
            if optuna_params:
                cb_model = CatBoostModel(**optuna_params, random_state=random_state)
                print(f"✅ [CatBoost] Model created with Optuna parameters: {optuna_params}")
            else:
                cb_model = CatBoostModel(random_state=random_state)
                print(f"✅ [CatBoost] Model created with default parameters")
            
            return cb_model
        
        else:
            # Default to custom Decision Tree model
            from models.classification.decision_tree_model import DecisionTreeModel
            
            # GPU configuration - AUTO ENABLE by default
            use_gpu = True  # Always enable GPU for performance
            gpu_library = 'auto'  # Auto-select best GPU library
            
            print(f"🎯 [Default] Using CUSTOM Decision Tree model as fallback")
            print(f"   • GPU Acceleration: True")
            print(f"   • GPU Library: auto")
            
            # Ensure random_state is an integer
            safe_random_state = random_state if isinstance(random_state, int) else 42
            
            dt_model = DecisionTreeModel(
                pruning_method='none',     # Disable pruning for Windows compatibility
                random_state=safe_random_state,
                use_gpu=use_gpu,           # Enable GPU acceleration
                gpu_library=gpu_library    # GPU library selection
            )
            
            # Force GPU initialization if requested
            if use_gpu:
                dt_model._init_gpu_libraries()
                # IMPORTANT: After GPU init, check if pruning was disabled on Windows
                if dt_model.pruning_method == 'none':
                    print(f"   ✅ Pruning method automatically disabled: ccp → none")
            
            print(f"✅ [Default] Custom Decision Tree model created as fallback + GPU: {use_gpu}")
            return dt_model
    
    def _generate_visualizations(self, training_results: Dict, y_test: np.ndarray):
        """Generate confusion matrices and other visualizations"""
        
        try:
            # Apply label processing method like main.py (map numeric IDs to text labels)
            try:
                # Get unique numeric labels from data
                unique_numeric_labels = sorted(list(set(y_test)))
                
                # Create text labels mapping (sử dụng label mapping động từ data_loader)
                sorted_labels = []
                try:
                    # Sử dụng label mapping động từ data_loader nếu có
                    if hasattr(self, 'data_loader') and hasattr(self.data_loader, 'id_to_label') and self.data_loader.id_to_label:
                        for label_id in unique_numeric_labels:
                            if label_id in self.data_loader.id_to_label:
                                sorted_labels.append(self.data_loader.id_to_label[label_id])
                            else:
                                sorted_labels.append(f"Class_{label_id}")
                        print(f"✅ Sử dụng label mapping động từ data_loader: {sorted_labels}")
                    else:
                        # Fallback: sử dụng numeric labels
                        sorted_labels = [f"Class_{i}" for i in unique_numeric_labels]
                        print(f"⚠️  Sử dụng fallback labels: {sorted_labels}")
                        
                except Exception as e:
                    print(f"Warning: Label mapping failed: {e}")
                    # Fallback: use numeric labels
                    sorted_labels = [f"Class_{i}" for i in unique_numeric_labels]
                    print(f"⚠️  Using fallback labels: {sorted_labels}")
                
            except Exception as e:
                print(f"Warning: Label mapping failed: {e}")
                # Fallback: use numeric labels
                sorted_labels = [f"Class_{i}" for i in unique_numeric_labels]
                print(f"⚠️  Using fallback labels: {sorted_labels}")
            
            # Use unique_numeric_labels for confusion matrix calculation
            unique_labels = unique_numeric_labels
            
            for result_key, result_data in training_results.items():
                if 'labels' not in result_data:
                    continue
                
                # Extract model name and vectorization method
                parts = result_key.split('_')
                if len(parts) >= 2:
                    model_name = parts[0]
                    vec_method = '_'.join(parts[1:])
                else:
                    model_name = result_key
                    vec_method = 'unknown'
                
                # Generate confusion matrix
                try:
                    if hasattr(plot_confusion_matrix, '__call__'):
                        # Use sorted_labels if available (same as main.py)
                        if sorted_labels and len(sorted_labels) > 0:
                            plot_confusion_matrix(
                                y_test, result_data['labels'], sorted_labels,
                                f"{model_name} Confusion Matrix ({vec_method})",
                                f"pdf/Figures/{model_name.lower()}_{vec_method}_confusion_matrix.pdf"
                            )
                        else:
                            plot_confusion_matrix(
                                y_test, result_data['labels'], unique_labels,
                                f"{model_name} Confusion Matrix ({vec_method})",
                                f"pdf/Figures/{model_name.lower()}_{vec_method}_confusion_matrix.pdf"
                            )
                    else:
                        # Fallback confusion matrix
                        self._create_fallback_confusion_matrix(
                            y_test, result_data['labels'], unique_labels,
                            model_name, vec_method
                        )
                except Exception as e:
                    print(f"Warning: Failed to create confusion matrix for {result_key}: {e}")
                    continue
            
            # Create model comparison plot
            try:
                if hasattr(plot_model_comparison, '__call__'):
                    # Prepare results for comparison
                    comparison_results = {}
                    for key, data in training_results.items():
                        if 'accuracy' in data:
                            comparison_results[key] = data['accuracy']
                    
                    plot_model_comparison(
                        comparison_results,
                        "pdf/Figures/model_comparison.pdf"
                    )
                else:
                    # Fallback model comparison
                    self._create_fallback_model_comparison(training_results)
            except Exception as e:
                print(f"Warning: Failed to create model comparison: {e}")
                # Try fallback
                try:
                    self._create_fallback_model_comparison(training_results)
                except Exception as e2:
                    print(f"Warning: Fallback model comparison also failed: {e2}")
                
        except Exception as e:
            print(f"Warning: Visualization generation failed: {e}")
    
    def _create_fallback_confusion_matrix(self, y_true, y_pred, labels, 
                                        model_name: str, vec_method: str):
        """Create confusion matrix using matplotlib as fallback"""
        
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            # Ensure y_true and y_pred are numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Handle empty arrays
            if len(y_true) == 0 or len(y_pred) == 0:
                print(f"Warning: Empty arrays for confusion matrix - {model_name} with {vec_method}")
                return
            
            # Apply label processing method like main.py (map numeric IDs to text labels)
            try:
                # Get unique numeric labels from data
                unique_numeric_labels = sorted(list(set(np.concatenate([y_true, y_pred]))))
                
                # Create text labels mapping (sử dụng label từ data_loader)
                # Since we don't have data_loader here, we'll create meaningful text labels
                class_names = []
                try:
                    # Sử dụng label mapping động từ data_loader nếu có
                    if hasattr(self, 'data_loader') and hasattr(self.data_loader, 'id_to_label') and self.data_loader.id_to_label:
                        for label_id in unique_numeric_labels:
                            if label_id in self.data_loader.id_to_label:
                                class_names.append(self.data_loader.id_to_label[label_id])
                            else:
                                class_names.append(f"Class_{label_id}")
                        print(f"✅ Sử dụng label mapping động từ data_loader: {class_names}")
                    else:
                        # Fallback: sử dụng numeric labels
                        class_names = [f"Class_{i}" for i in unique_numeric_labels]
                        print(f"⚠️  Sử dụng fallback labels: {class_names}")
                        
                except Exception as e:
                    print(f"Warning: Label mapping failed: {e}")
                    # Fallback: use numeric labels
                    class_names = [f"Class_{i}" for i in unique_numeric_labels]
                    print(f"⚠️  Using fallback labels: {class_names}")
                
            except Exception as e:
                print(f"Warning: Label mapping failed: {e}")
                # Fallback: use numeric labels
                class_names = [f"Class_{i}" for i in unique_numeric_labels]
                print(f"⚠️  Using fallback labels: {class_names}")
            
            # Use unique_numeric_labels for confusion matrix calculation
            labels = unique_numeric_labels
            
            # Create confusion matrix
            try:
                cm = confusion_matrix(y_true, y_pred, labels=labels)
            except Exception as e:
                print(f"Warning: Confusion matrix calculation failed: {e}")
                # Try without labels
                cm = confusion_matrix(y_true, y_pred)
                # Update labels based on actual values
                labels = sorted(list(set(np.concatenate([y_true, y_pred]))))
            
            # Create plot with proper labels (same as main.py approach)
            plt.figure(figsize=(10, 8))
            
            # Create annotations with raw values
            annotations = np.empty_like(cm).astype(str)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    annotations[i, j] = str(cm[i, j])
            
            # Plot heatmap with text labels (same as main.py approach)
            sns.heatmap(cm, annot=annotations, fmt="", cmap="Blues",
                       xticklabels=class_names, yticklabels=class_names,
                       cbar=True, linewidths=1, linecolor='black')
            
            plt.title(f'{model_name} Confusion Matrix ({vec_method})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Save plot
            import os
            os.makedirs('pdf/Figures', exist_ok=True)
            plt.savefig(f'pdf/Figures/{model_name.lower()}_{vec_method}_confusion_matrix.pdf', 
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"✅ Confusion matrix created for {model_name} with {vec_method}")
            print(f"   Labels used: {class_names}")
            
        except Exception as e:
            print(f"Warning: Fallback confusion matrix failed for {model_name} with {vec_method}: {e}")
    
    def _create_fallback_model_comparison(self, training_results: Dict):
        """Create model comparison using matplotlib as fallback"""
        try:
            import matplotlib.pyplot as plt
            
            # Prepare results for comparison
            comparison_results = {}
            for key, data in training_results.items():
                if 'accuracy' in data:
                    comparison_results[key] = data['accuracy']
            
            if not comparison_results:
                print("Warning: No accuracy results to compare")
                return
            
            # Create comparison plot
            plt.figure(figsize=(10, 6))
            models = list(comparison_results.keys())
            accuracies = list(comparison_results.values())
            
            plt.bar(range(len(models)), accuracies, color='skyblue', edgecolor='navy')
            plt.xlabel('Models')
            plt.ylabel('Accuracy')
            plt.title('Model Performance Comparison')
            plt.xticks(range(len(models)), models, rotation=45, ha='right')
            plt.ylim(0, 1)
            
            # Add accuracy values on bars
            for i, v in enumerate(accuracies):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            import os
            os.makedirs('pdf/Figures', exist_ok=True)
            plt.savefig('pdf/Figures/model_comparison_fallback.pdf')
            plt.close()
            
            print("✅ Fallback model comparison created successfully")
            
        except Exception as e:
            print(f"Warning: Fallback model comparison failed: {e}")
    
    def plot_confusion_matrices_from_cache(self, cached_results: Dict) -> bool:
        """
        Vẽ confusion matrices từ cached results
        Sử dụng dữ liệu đã được lưu trong cache
        """
        try:
            print("🎨 Vẽ confusion matrices từ cache...")
            
            if 'comprehensive_results' not in cached_results:
                print("❌ Không có comprehensive_results trong cache")
                return False
            
            comprehensive_results = cached_results['comprehensive_results']
            successful_results = [r for r in comprehensive_results if r.get('status') == 'success']
            
            if not successful_results:
                print("❌ Không có kết quả thành công trong cache")
                return False
            
            print(f"✅ Tìm thấy {len(successful_results)} kết quả thành công")
            
            # Vẽ confusion matrix cho từng combination
            for result in successful_results:
                model_name = result.get('model_name', 'Unknown')
                embedding_name = result.get('embedding_name', 'Unknown')
                
                print(f"   🎯 Vẽ confusion matrix cho {model_name} + {embedding_name}")
                
                # Kiểm tra dữ liệu cần thiết
                if 'predictions' in result and 'true_labels' in result:
                    predictions = result['predictions']
                    true_labels = result['true_labels']
                    label_mapping = result.get('label_mapping', {})
                    
                    # Vẽ confusion matrix
                    self._create_confusion_matrix_from_cache(
                        true_labels, predictions, label_mapping,
                        model_name, embedding_name
                    )
                elif model_name == 'Ensemble Learning' and 'ensemble_info' in result:
                    # Xử lý đặc biệt cho Ensemble Learning
                    print(f"      🎯 Xử lý Ensemble Learning - tạo confusion matrix từ base models")
                    print(f"      🔍 Ensemble info keys: {list(result.get('ensemble_info', {}).keys())}")
                    self._create_ensemble_confusion_matrix_from_cache(result)
                else:
                    print(f"      ⚠️  Thiếu dữ liệu cho {model_name} + {embedding_name}")
                    print(f"         Có: {list(result.keys())}")
            
            print("✅ Hoàn thành vẽ confusion matrices từ cache!")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi vẽ confusion matrices từ cache: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_confusion_matrix_from_cache(self, y_true, y_pred, label_mapping: Dict,
                                          model_name: str, embedding_name: str):
        """
        Tạo confusion matrix từ dữ liệu cache với label mapping
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            # Đảm bảo y_true và y_pred là numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Xử lý labels
            unique_labels = sorted(list(set(np.concatenate([y_true, y_pred]))))
            
            # Sử dụng label mapping từ cache nếu có
            if label_mapping:
                class_names = [label_mapping.get(label_id, f"Class_{label_id}") 
                              for label_id in unique_labels]
                print(f"      ✅ Sử dụng label mapping: {class_names}")
            else:
                # Fallback: tạo labels đơn giản
                class_names = [f"Class_{label_id}" for label_id in unique_labels]
                print(f"      ⚠️  Sử dụng fallback labels: {class_names}")
            
            # Tính confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
            
            # Tạo plot
            plt.figure(figsize=(10, 8))
            
            # Tạo annotations
            annotations = np.empty_like(cm).astype(str)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    annotations[i, j] = str(cm[i, j])
            
            # Vẽ heatmap với text labels
            sns.heatmap(cm, annot=annotations, fmt="", cmap="Blues",
                       xticklabels=class_names, yticklabels=class_names,
                       cbar=True, linewidths=1, linecolor='black')
            
            plt.title(f'{model_name} Confusion Matrix ({embedding_name}) - From Cache')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Tạo thư mục nếu chưa có
            os.makedirs('pdf/Figures', exist_ok=True)
            
            # Lưu plot
            filename = f'pdf/Figures/{model_name.lower()}_{embedding_name}_confusion_matrix_cache.pdf'
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"      ✅ Confusion matrix đã lưu: {filename}")
            
        except Exception as e:
            print(f"      ❌ Lỗi khi tạo confusion matrix: {e}")
    
    def _create_ensemble_confusion_matrix_from_cache(self, ensemble_result: Dict):
        """
        Tạo confusion matrix cho Ensemble Learning từ dữ liệu ensemble model
        """
        try:
            print(f"         🔍 Tạo confusion matrix cho Ensemble Learning...")
            print(f"         🔍 Ensemble result keys: {list(ensemble_result.keys())}")
            
            # Lấy dữ liệu trực tiếp từ ensemble model (không phải từ base models)
            predictions = ensemble_result.get('predictions', [])
            true_labels = ensemble_result.get('true_labels', [])
            label_mapping = ensemble_result.get('label_mapping', {})
            
            if not predictions or not true_labels:
                print(f"         ❌ Không có predictions hoặc true_labels trong ensemble result")
                return
            
            print(f"         ✅ Sử dụng dữ liệu từ ensemble model với {len(predictions)} predictions")
            
            # Tạo confusion matrix cho ensemble
            embedding_name = ensemble_result.get('embedding_name', 'Unknown')
            self._create_confusion_matrix_from_cache(
                true_labels, predictions, label_mapping,
                'Ensemble Learning', embedding_name
            )
            
        except Exception as e:
            print(f"         ❌ Lỗi khi tạo ensemble confusion matrix: {e}")
            import traceback
            traceback.print_exc()
    
    def get_training_status(self) -> Dict:
        """Get current training status"""
        return {
            'status': self.training_status,
            'phase': self.current_phase,
            'current_model': self.current_model,
            'models_completed': self.models_completed,
            'total_models': self.total_models,
            'elapsed_time': self.elapsed_time,
            'start_time': self.start_time
        }
    
    def stop_training(self):
        """Stop training process"""
        self.training_status = "stopped"
        self.current_phase = "stopped"
    
    def reset_pipeline(self):
        """Reset pipeline to initial state"""
        self.results = {}
        self.training_status = "idle"
        self.current_model = None
        self.current_phase = "initializing"
        self.models_completed = 0
        self.total_models = 0
        self.start_time = None
        self.elapsed_time = 0


# Global pipeline instance for Streamlit
training_pipeline = StreamlitTrainingPipeline()

def global_stop_check():
    """Global function to check if training should stop"""
    return training_pipeline.is_training_stopped()


def execute_streamlit_training(df: pd.DataFrame, step1_data: Dict, 
                             step2_data: Dict, step3_data: Dict,
                             progress_callback=None) -> Dict:
    """Main function to execute training from Streamlit"""
    
    print(f"\n🚀 [TRAINING] Starting execute_streamlit_training...")
    print(f"📊 [TRAINING] Input dataset size: {len(df):,}")
    print(f"📊 [TRAINING] Step1 data keys: {list(step1_data.keys())}")
    print(f"📊 [TRAINING] Step2 data keys: {list(step2_data.keys())}")
    print(f"📊 [TRAINING] Step3 data keys: {list(step3_data.keys())}")
    
    # CRITICAL FIX: Ensure step1_data contains all necessary data
    # This ensures cache has complete information for label mapping
    if 'uploaded_file' not in step1_data:
        print(f"⚠️ [TRAINING] WARNING: step1_data missing 'uploaded_file'")
        print(f"   - This will cause cache key to show 'unknown_dataset'")
        print(f"   - Available keys: {list(step1_data.keys())}")
    
    if 'selected_categories' not in step1_data:
        print(f"⚠️ [TRAINING] WARNING: step1_data missing 'selected_categories'")
        print(f"   - This will cause cache key to show 'no_cats'")
        print(f"   - Available keys: {list(step1_data.keys())}")
        
        # Try to extract categories from dataframe if available
        if 'dataframe' in step1_data and step1_data['dataframe'] is not None:
            df_step1 = step1_data['dataframe']
            print(f"🔍 [TRAINING] Attempting to extract categories from dataframe...")
            
            # Look for label column (try multiple possible names)
            label_col = None
            for col in ['Category', 'label', 'category', 'class', 'target', 'y']:
                if col in df_step1.columns:
                    label_col = col
                    break
            
            if label_col:
                try:
                    unique_categories = sorted(df_step1[label_col].unique().tolist())
                    step1_data['selected_categories'] = unique_categories
                    print(f"✅ [TRAINING] Extracted categories from '{label_col}': {unique_categories}")
                    
                    # Also try to extract from the main dataframe if different
                    if df is not None and len(df) > 0:
                        main_label_col = None
                        for col in ['Category', 'label', 'category', 'class', 'target', 'y']:
                            if col in df.columns:
                                main_label_col = col
                                break
                        
                        if main_label_col and main_label_col != label_col:
                            main_categories = sorted(df[main_label_col].unique().tolist())
                            if main_categories != unique_categories:
                                print(f"⚠️ [TRAINING] Categories mismatch between step1_data and main dataframe")
                                print(f"   - Step1 categories: {unique_categories}")
                                print(f"   - Main dataframe categories: {main_categories}")
                                # Use main dataframe categories as they're more current
                                step1_data['selected_categories'] = main_categories
                                print(f"✅ [TRAINING] Updated to main dataframe categories: {main_categories}")
                except Exception as e:
                    print(f"⚠️ [TRAINING] Error extracting categories: {e}")
            else:
                print(f"⚠️ [TRAINING] No label column found in dataframe")
                print(f"   - Available columns: {list(df_step1.columns)}")
                
                # Try to extract from main dataframe as fallback
                if df is not None and len(df) > 0:
                    for col in ['Category', 'label', 'category', 'class', 'target', 'y']:
                        if col in df.columns:
                            try:
                                main_categories = sorted(df[col].unique().tolist())
                                step1_data['selected_categories'] = main_categories
                                print(f"✅ [TRAINING] Extracted categories from main dataframe '{col}': {main_categories}")
                                break
                            except Exception as e:
                                print(f"⚠️ [TRAINING] Error extracting from main dataframe: {e}")
    
    # Check sampling config specifically
    if 'sampling_config' in step1_data:
        sampling_config = step1_data['sampling_config']
        print(f"💾 [TRAINING] Sampling config received: {sampling_config}")
        print(f"📊 [TRAINING] Requested samples: {sampling_config.get('num_samples', 'N/A')}")
    else:
        print(f"❌ [TRAINING] No sampling config in step1_data!")
    
    global training_pipeline
    
    # Reset pipeline if needed
    if training_pipeline.training_status == "training":
        training_pipeline.stop_training()
    
    training_pipeline.reset_pipeline()
    
    # Execute training
    result = training_pipeline.execute_training(
        df, step1_data, step2_data, step3_data, progress_callback
    )
    
    return result


def get_training_status() -> Dict:
    """Get current training status for Streamlit"""
    global training_pipeline
    return training_pipeline.get_training_status()


def stop_training():
    """Stop training process from Streamlit"""
    global training_pipeline
    training_pipeline.stop_training()


def reset_training():
    """Reset training pipeline from Streamlit"""
    global training_pipeline
    training_pipeline.reset_pipeline()
