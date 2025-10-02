"""
Cache Manager for Enhanced ML Models
Implements per-model caching with config_hash and dataset_fingerprint

Features:
- Per-model cache with config_hash and dataset_fingerprint
- Cache hit/miss detection
- Force retrain option
- Model artifacts storage
- Evaluation predictions caching
- SHAP sample caching
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import pandas as pd
import numpy as np
import logging

# CRITICAL: Disable parallel processing to prevent memory mapping errors
# Only apply restrictions when not running in Streamlit
import os
if not os.environ.get('STREAMLIT_RUNNING'):
    os.environ['JOBLIB_MULTIPROCESSING'] = '0'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

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
    print("SUCCESS: Temp file cleanup registered in cache_manager")
    
except Exception as temp_error:
    print(f"WARNING: Failed to setup temp file cleanup in cache_manager: {temp_error}")

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages per-model caching with config_hash and dataset_fingerprint"""
    
    def __init__(self, cache_root_dir: str = "cache/models/"):
        """Initialize cache manager
        
        Args:
            cache_root_dir: Root directory for model caches
        """
        # Keep cache in project directory as requested
        self.cache_root_dir = Path(cache_root_dir)
        self.cache_root_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache structure: cache/models/{model_key}/{dataset_id}/{config_hash}/
        self.cache_structure = {
            'model_artifact': 'model.{ext}',
            'params': 'params.json',
            'metrics': 'metrics.json', 
            'config': 'config.json',
            'fingerprint': 'fingerprint.json',
            'eval_predictions': 'eval_predictions.parquet',
            'shap_sample': 'shap_sample.parquet',
            'feature_names': 'feature_names.txt',
            'label_mapping': 'label_mapping.json'
        }
    
    def generate_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate SHA256 hash from configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            SHA256 hash string
        """
        # Normalize config by sorting keys and converting to JSON
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def generate_dataset_fingerprint(self, dataset_path: str, dataset_size: int, 
                                   num_rows: int) -> str:
        """Generate dataset fingerprint
        
        Args:
            dataset_path: Path to dataset file
            dataset_size: Size of dataset in bytes
            num_rows: Number of rows in dataset
            
        Returns:
            Dataset fingerprint string
        """
        fingerprint_data = {
            'path': dataset_path,
            'size': dataset_size,
            'rows': num_rows
        }
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()
    
    def get_cache_path(self, model_key: str, dataset_id: str, 
                      config_hash: str) -> Path:
        """Get cache path for model
        
        Args:
            model_key: Model identifier (e.g., 'random_forest')
            dataset_id: Dataset identifier
            config_hash: Configuration hash
            
        Returns:
            Path to cache directory
        """
        return self.cache_root_dir / model_key / dataset_id / config_hash
    
    def check_cache_exists(self, model_key: str, dataset_id: str, 
                          config_hash: str, dataset_fingerprint: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if cache exists and is valid
        
        Args:
            model_key: Model identifier
            dataset_id: Dataset identifier  
            config_hash: Configuration hash
            dataset_fingerprint: Dataset fingerprint
            
        Returns:
            Tuple of (cache_exists, cache_info)
        """
        cache_path = self.get_cache_path(model_key, dataset_id, config_hash)
        
        if not cache_path.exists():
            return False, {}
        
        # Check fingerprint file
        fingerprint_file = cache_path / self.cache_structure['fingerprint']
        if not fingerprint_file.exists():
            return False, {}
        
        try:
            with open(fingerprint_file, 'r') as f:
                fingerprint_data = json.load(f)
            
            # Check if dataset fingerprint matches
            if fingerprint_data.get('dataset_fingerprint') != dataset_fingerprint:
                logger.info(f"Dataset fingerprint mismatch for {model_key}")
                return False, {}
            
            # Check if all required files exist
            required_files = ['params', 'metrics', 'config']
            for file_key in required_files:
                file_path = cache_path / self.cache_structure[file_key]
                if not file_path.exists():
                    logger.info(f"Missing {file_key} file for {model_key}")
                    return False, {}
            
            # Check model artifact (with dynamic extension)
            model_artifact_found = False
            for ext in ['pkl', 'json', 'txt', 'cbm']:
                model_file = cache_path / f'model.{ext}'
                if model_file.exists():
                    model_artifact_found = True
                    break
            
            if not model_artifact_found:
                logger.info(f"Missing model_artifact file for {model_key}")
                return False, {}
            
            # Load cache info
            cache_info = {
                'cache_path': str(cache_path),
                'created_at': fingerprint_data.get('created_at'),
                'config_hash': config_hash,
                'dataset_fingerprint': dataset_fingerprint,
                'model_key': model_key,
                'dataset_id': dataset_id
            }
            
            return True, cache_info
            
        except Exception as e:
            logger.error(f"Error checking cache for {model_key}: {e}")
            return False, {}
    
    def save_model_cache(self, model_key: str, dataset_id: str, config_hash: str,
                        dataset_fingerprint: str, model, params: Dict[str, Any],
                        metrics: Dict[str, Any], config: Dict[str, Any],
                        eval_predictions: Optional[pd.DataFrame] = None,
                        shap_sample: Optional[pd.DataFrame] = None,
                        feature_names: Optional[List[str]] = None,
                        label_mapping: Optional[Dict[str, Any]] = None) -> str:
        """Save model cache
        
        Args:
            model_key: Model identifier
            dataset_id: Dataset identifier
            config_hash: Configuration hash
            dataset_fingerprint: Dataset fingerprint
            model: Trained model object
            params: Model parameters
            metrics: Model metrics
            config: Configuration used
            eval_predictions: Evaluation predictions
            shap_sample: SHAP sample data
            feature_names: Feature names
            label_mapping: Label mapping
            
        Returns:
            Cache path string
        """
        try:
            print(f"DEBUG: Starting cache save for {model_key}")
            
            cache_path = self.get_cache_path(model_key, dataset_id, config_hash)
            
            # Create cache directory with error handling
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
                print(f"DEBUG: Cache directory created: {cache_path}")
            except Exception as dir_error:
                print(f"ERROR: Failed to create cache directory: {dir_error}")
                raise dir_error
            # Save model artifact with error handling
            try:
                model_ext = self._get_model_extension(model_key)
                model_file = cache_path / f"model.{model_ext}"
                self._save_model_artifact(model, model_file, model_key)
                print(f"DEBUG: Model artifact saved: {model_file}")
            except Exception as model_error:
                print(f"ERROR: Failed to save model artifact: {model_error}")
                raise model_error
            
            # Custom JSON serializer to handle non-serializable objects
            def safe_json_serializer(obj):
                """Convert non-serializable objects to strings"""
                try:
                    if hasattr(obj, '__module__'):
                        return f"<{obj.__class__.__name__} object>"
                    elif hasattr(obj, '__dict__'):
                        return str(obj)
                    else:
                        return str(obj)
                except Exception as e:
                    print(f"WARNING: JSON serialization failed for object: {e}")
                    return f"<SerializationError: {str(e)}>"
            
            # Save params with error handling
            try:
                params_file = cache_path / self.cache_structure['params']
                with open(params_file, 'w') as f:
                    json.dump(params, f, indent=2, default=safe_json_serializer)
                print(f"DEBUG: Params saved: {params_file}")
            except Exception as params_error:
                print(f"ERROR: Failed to save params: {params_error}")
                raise params_error
            
            # Save metrics with error handling
            try:
                metrics_file = cache_path / self.cache_structure['metrics']
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2, default=safe_json_serializer)
                print(f"DEBUG: Metrics saved: {metrics_file}")
            except Exception as metrics_error:
                print(f"ERROR: Failed to save metrics: {metrics_error}")
                raise metrics_error
            
            # Save config with error handling
            try:
                config_file = cache_path / self.cache_structure['config']
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2, default=safe_json_serializer)
                print(f"DEBUG: Config saved: {config_file}")
            except Exception as config_error:
                print(f"ERROR: Failed to save config: {config_error}")
                raise config_error
            
            # Save fingerprint
            fingerprint_data = {
                'config_hash': config_hash,
                'dataset_fingerprint': dataset_fingerprint,
                'created_at': datetime.now().isoformat(),
                'model_key': model_key,
                'dataset_id': dataset_id,
                'library_versions': self._get_library_versions()
            }
            fingerprint_file = cache_path / self.cache_structure['fingerprint']
            with open(fingerprint_file, 'w') as f:
                json.dump(fingerprint_data, f, indent=2, default=safe_json_serializer)
            
            # Save eval predictions if provided
            if eval_predictions is not None:
                eval_file = cache_path / self.cache_structure['eval_predictions']
                eval_predictions.to_parquet(eval_file, index=False)
            
            # Save SHAP sample if provided
            if shap_sample is not None:
                shap_file = cache_path / self.cache_structure['shap_sample']
                shap_sample.to_parquet(shap_file, index=False)
            
            # Save feature names if provided
            if feature_names is not None:
                feature_file = cache_path / self.cache_structure['feature_names']
                with open(feature_file, 'w') as f:
                    f.write('\n'.join(feature_names))
            
            # Save label mapping if provided
            if label_mapping is not None:
                label_file = cache_path / self.cache_structure['label_mapping']
                with open(label_file, 'w') as f:
                    json.dump(label_mapping, f, indent=2, default=safe_json_serializer)
            
            logger.info(f"Cache saved for {model_key} at {cache_path}")
            
            # CRITICAL: Force cleanup to prevent memory mapping errors
            try:
                import gc
                gc.collect()
                print(f"DEBUG: Garbage collection completed for {model_key}")
            except Exception as cleanup_error:
                print(f"WARNING: Failed to cleanup: {cleanup_error}")
            
            # Force garbage collection after operations
            try:
                import gc
                gc.collect()
                print(f"DEBUG: Garbage collection completed for {model_key}")
            except Exception as gc_error:
                print(f"WARNING: Garbage collection failed: {gc_error}")
            
            return str(cache_path)
            
        except Exception as e:
            logger.error(f"Error saving cache for {model_key}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't raise exception, just log the error
            # This allows the training to continue even if cache fails
            return None
    
    def load_model_cache(self, model_key: str, dataset_id: str, 
                        config_hash: str) -> Dict[str, Any]:
        """Load model cache
        
        Args:
            model_key: Model identifier
            dataset_id: Dataset identifier
            config_hash: Configuration hash
            
        Returns:
            Dictionary containing cached data
        """
        cache_path = self.get_cache_path(model_key, dataset_id, config_hash)
        
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found for {model_key}")
        
        try:
            # Load model artifact
            model_ext = self._get_model_extension(model_key)
            model_file = cache_path / f"model.{model_ext}"
            model = self._load_model_artifact(model_file, model_key, config_hash)
            
            # Load params
            params_file = cache_path / self.cache_structure['params']
            with open(params_file, 'r') as f:
                params = json.load(f)
            
            # Load metrics
            metrics_file = cache_path / self.cache_structure['metrics']
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Load config
            config_file = cache_path / self.cache_structure['config']
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Load eval predictions if exists
            eval_file = cache_path / self.cache_structure['eval_predictions']
            eval_predictions = None
            if eval_file.exists():
                eval_predictions = pd.read_parquet(eval_file)
            
            # Load SHAP sample if exists
            shap_file = cache_path / self.cache_structure['shap_sample']
            shap_sample = None
            if shap_file.exists():
                shap_sample = pd.read_parquet(shap_file)
            
            # Load feature names if exists
            feature_file = cache_path / self.cache_structure['feature_names']
            feature_names = None
            if feature_file.exists():
                with open(feature_file, 'r') as f:
                    feature_names = f.read().strip().split('\n')
            
            # Load label mapping if exists
            label_file = cache_path / self.cache_structure['label_mapping']
            label_mapping = None
            if label_file.exists():
                with open(label_file, 'r') as f:
                    label_mapping = json.load(f)
            
            cache_data = {
                'model': model,
                'params': params,
                'metrics': metrics,
                'config': config,
                'eval_predictions': eval_predictions,
                'shap_sample': shap_sample,
                'feature_names': feature_names,
                'label_mapping': label_mapping,
                'cache_path': str(cache_path)
            }
            
            logger.info(f"Cache loaded for {model_key} from {cache_path}")
            return cache_data
            
        except Exception as e:
            logger.error(f"Error loading cache for {model_key}: {e}")
            raise
    
    def _get_model_extension(self, model_key: str) -> str:
        """Get file extension for model based on type
        
        Args:
            model_key: Model identifier
            
        Returns:
            File extension string
        """
        if model_key in ['xgboost']:
            return 'json'
        elif model_key in ['lightgbm']:
            return 'txt'
        elif model_key in ['catboost']:
            return 'cbm'
        else:
            return 'pkl'
    
    def _save_model_artifact(self, model, file_path: Path, model_key: str):
        """Save model artifact based on type
        
        Args:
            model: Model object
            file_path: Path to save model
            model_key: Model identifier
        """
        if model_key in ['xgboost']:
            # Use XGBoost's native save method directly
            import xgboost as xgb
            # Access the actual XGBoost model instance
            xgb_model = model.model if hasattr(model, 'model') else model
            if hasattr(xgb_model, 'get_booster'):
                xgb_model.get_booster().save_model(str(file_path))
            else:
                xgb_model.save_model(str(file_path))
        elif model_key in ['lightgbm']:
            # Use LightGBM's native save method directly
            import lightgbm as lgb
            # Access the actual LightGBM model instance
            lgb_model = model.model if hasattr(model, 'model') else model
            if hasattr(lgb_model, 'booster_'):
                lgb_model.booster_.save_model(str(file_path))
            else:
                lgb_model.save_model(str(file_path))
            
            # Save additional metadata for compatibility checking
            metadata_file = file_path.parent / f"{file_path.stem}_metadata.json"
            metadata = {
                'vectorization_method': getattr(model, '_vectorization_method', 'unknown'),
                'config_hash': getattr(model, '_config_hash', 'unknown'),
                'n_features_in_': getattr(model, 'n_features_in_', 0),
                'classes_': getattr(model, 'classes_', []).tolist() if hasattr(getattr(model, 'classes_', None), 'tolist') else getattr(model, 'classes_', [])
            }
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        elif model_key in ['catboost']:
            # Use CatBoost's native save method directly
            import catboost as cb
            # Access the actual CatBoost model instance
            cb_model = model.model if hasattr(model, 'model') else model
            if hasattr(cb_model, 'save_model'):
                cb_model.save_model(str(file_path))
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(cb_model, f)
        elif model_key.startswith('stacking_ensemble_') or model_key.startswith('voting_ensemble_'):
            # Save ensemble model data (contains ensemble_manager and metadata)
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            # For custom model classes, save the underlying sklearn model
            if hasattr(model, 'model'):
                # Custom wrapper class - save underlying model
                with open(file_path, 'wb') as f:
                    pickle.dump(model.model, f)
            else:
                # Direct sklearn model
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f)
    
    def _load_model_artifact(self, file_path: Path, model_key: str, config_hash: str = None):
        """Load model artifact based on type
        
        Args:
            file_path: Path to model file
            model_key: Model identifier
            config_hash: Configuration hash to check compatibility
            
        Returns:
            Loaded model object
        """
        if model_key in ['xgboost']:
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(str(file_path))
            return model
        elif model_key in ['lightgbm']:
            import lightgbm as lgb
            import numpy as np
            # Load the saved model directly
            booster = lgb.Booster(model_file=str(file_path))
            
            # Create a sklearn-compatible wrapper using TrainedModelWrapper
            from models.ensemble.ensemble_manager import TrainedModelWrapper
            
            class LightGBMCacheWrapper:
                def __init__(self, booster):
                    self.booster = booster
                    self._fitted = True
                    self._is_fitted = True
                    self._n_features = booster.num_feature()
                    
                    # Try to get num_classes from metadata, fallback to 2
                    metadata_file = file_path.parent / f"{file_path.stem}_metadata.json"
                    if metadata_file.exists():
                        try:
                            import json
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            cached_classes = metadata.get('classes_', [])
                            self._n_classes = len(cached_classes) if cached_classes else 2
                        except Exception:
                            self._n_classes = 2
                    else:
                        self._n_classes = 2  # Default to binary classification
                    
                    # Set classes based on booster
                    if self._n_classes == 2:
                        self.classes_ = np.array([0, 1])
                    else:
                        self.classes_ = np.array(list(range(self._n_classes)))
                    
                    # Set n_features_in_ as a property
                    self.n_features_in_ = self._n_features
                
                def predict(self, X):
                    """Predict using the booster directly"""
                    predictions = self.booster.predict(X)
                    if self._n_classes == 2:
                        return (predictions > 0.5).astype(int)
                    else:
                        return np.argmax(predictions, axis=1)
                
                def predict_proba(self, X):
                    """Predict probabilities using the booster directly"""
                    predictions = self.booster.predict(X)
                    if self._n_classes == 2:
                        # Binary classification: return probabilities for both classes
                        proba_1 = predictions
                        proba_0 = 1 - predictions
                        return np.column_stack([proba_0, proba_1])
                    else:
                        # Multi-class: predictions are already probabilities
                        return predictions
            
            # Wrap the LightGBM wrapper with TrainedModelWrapper for sklearn compatibility
            lgb_wrapper = LightGBMCacheWrapper(booster)
            return TrainedModelWrapper(lgb_wrapper, model_name="LightGBM")
        elif model_key in ['catboost']:
            import catboost as cb
            model = cb.CatBoostClassifier()
            model.load_model(str(file_path))
            return model
        elif model_key.startswith('stacking_ensemble_') or model_key.startswith('voting_ensemble_'):
            # Load ensemble model data (contains ensemble_manager and metadata)
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            # Load underlying sklearn model and wrap it
            with open(file_path, 'rb') as f:
                sklearn_model = pickle.load(f)
            
            # Try to recreate the custom wrapper class
            try:
                from models.utils.model_factory import model_factory
                wrapper_class = model_factory.get_model_class(model_key)
                if wrapper_class:
                    # Create new instance and set the loaded model
                    wrapper_instance = wrapper_class()
                    wrapper_instance.model = sklearn_model
                    return wrapper_instance
            except Exception:
                pass
            
            # Fallback: return the sklearn model directly
            return sklearn_model
    
    def _get_library_versions(self) -> Dict[str, str]:
        """Get library versions for fingerprinting
        
        Returns:
            Dictionary of library versions
        """
        versions = {}
        
        try:
            import sklearn
            versions['sklearn'] = sklearn.__version__
        except ImportError:
            pass
        
        try:
            import xgboost as xgb
            versions['xgboost'] = xgb.__version__
        except ImportError:
            pass
        
        try:
            import lightgbm as lgb
            versions['lightgbm'] = lgb.__version__
        except ImportError:
            pass
        
        try:
            import catboost as cb
            versions['catboost'] = cb.__version__
        except ImportError:
            pass
        
        try:
            import optuna
            versions['optuna'] = optuna.__version__
        except ImportError:
            pass
        
        try:
            import shap
            versions['shap'] = shap.__version__
        except ImportError:
            pass
        
        return versions
    
    def list_cached_models(self, model_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all cached models
        
        Args:
            model_key: Optional model key to filter
            
        Returns:
            List of cached model information
        """
        cached_models = []
        
        if model_key:
            model_dir = self.cache_root_dir / model_key
            if model_dir.exists():
                for dataset_id in model_dir.iterdir():
                    if dataset_id.is_dir():
                        for config_hash in dataset_id.iterdir():
                            if config_hash.is_dir():
                                cached_models.append({
                                    'model_key': model_key,
                                    'dataset_id': dataset_id.name,
                                    'config_hash': config_hash.name,
                                    'cache_path': str(config_hash),
                                    'created_at': self._get_cache_created_at(config_hash)
                                })
        else:
            for model_dir in self.cache_root_dir.iterdir():
                if model_dir.is_dir():
                    for dataset_id in model_dir.iterdir():
                        if dataset_id.is_dir():
                            for config_hash in dataset_id.iterdir():
                                if config_hash.is_dir():
                                    cached_models.append({
                                        'model_key': model_dir.name,
                                        'dataset_id': dataset_id.name,
                                        'config_hash': config_hash.name,
                                        'cache_path': str(config_hash),
                                        'created_at': self._get_cache_created_at(config_hash)
                                    })
        
        return cached_models
    
    def _get_cache_created_at(self, cache_path: Path) -> Optional[str]:
        """Get cache creation timestamp
        
        Args:
            cache_path: Path to cache directory
            
        Returns:
            Creation timestamp string or None
        """
        fingerprint_file = cache_path / self.cache_structure['fingerprint']
        if fingerprint_file.exists():
            try:
                with open(fingerprint_file, 'r') as f:
                    fingerprint_data = json.load(f)
                return fingerprint_data.get('created_at')
            except Exception:
                pass
        return None
    
    def clear_cache(self, model_key: Optional[str] = None, 
                   dataset_id: Optional[str] = None,
                   config_hash: Optional[str] = None) -> int:
        """Clear cache entries
        
        Args:
            model_key: Optional model key to clear
            dataset_id: Optional dataset ID to clear
            config_hash: Optional config hash to clear
            
        Returns:
            Number of cache entries cleared
        """
        cleared_count = 0
        
        if model_key and dataset_id and config_hash:
            # Clear specific cache
            cache_path = self.get_cache_path(model_key, dataset_id, config_hash)
            if cache_path.exists():
                import shutil
                shutil.rmtree(cache_path)
                cleared_count = 1
        else:
            # Clear multiple caches
            cached_models = self.list_cached_models(model_key)
            for model_info in cached_models:
                if dataset_id and model_info['dataset_id'] != dataset_id:
                    continue
                if config_hash and model_info['config_hash'] != config_hash:
                    continue
                
                cache_path = Path(model_info['cache_path'])
                if cache_path.exists():
                    import shutil
                    shutil.rmtree(cache_path)
                    cleared_count += 1
        
        logger.info(f"Cleared {cleared_count} cache entries")
        return cleared_count


class TrainingResultsCacheManager:
    """Manages caching of comprehensive training results"""
    
    def __init__(self, cache_root_dir: str = "cache/training_results/"):
        """Initialize training results cache manager
        
        Args:
            cache_root_dir: Root directory for training results cache
        """
        self.cache_root_dir = Path(cache_root_dir)
        self.cache_root_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache structure for training results
        self.cache_structure = {
            'training_results': 'training_results.json',
            'comprehensive_results': 'comprehensive_results.json',
            'step_data': 'step_data.json',
            'metadata': 'metadata.json',
            'cache_info': 'cache_info.json'
        }
    
    def generate_session_key(self, step1_data: Dict, step2_data: Dict, step3_data: Dict) -> str:
        """Generate unique session key from step data"""
        session_data = {
            'step1_dataset': step1_data.get('dataset_size', 0),
            'step1_sampling': step1_data.get('sampling_config', {}),
            'step2_preprocessing': step2_data.get('preprocessing_config', {}),
            'step2_scalers': step2_data.get('selected_scalers', []),
            'step3_models': step3_data.get('selected_models', []),
            'step3_config': step3_data.get('optuna_config', {}),
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        session_str = json.dumps(session_data, sort_keys=True, default=str)
        return hashlib.sha256(session_str.encode()).hexdigest()[:16]
    
    def save_training_results(self, session_key: str, training_results: Dict[str, Any]) -> str:
        """Save comprehensive training results to cache"""
        try:
            session_cache_dir = self.cache_root_dir / session_key
            session_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract components for separate storage
            comprehensive_results = training_results.get('comprehensive_results', [])
            step_data = {
                'step1_data': training_results.get('step1_data', {}),
                'step2_data': training_results.get('step2_data', {}),
                'step3_data': training_results.get('step3_data', {})
            }
            
            # Create metadata
            metadata = {
                'session_key': session_key,
                'created_at': datetime.now().isoformat(),
                'total_models': training_results.get('total_models', 0),
                'successful_combinations': training_results.get('successful_combinations', 0),
                'total_combinations': training_results.get('total_combinations', 0),
                'elapsed_time': training_results.get('elapsed_time', 0),
                'status': training_results.get('status', 'unknown')
            }
            
            # Custom JSON serializer with better handling
            def safe_json_serializer(obj):
                try:
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
                    elif hasattr(obj, '__class__') and 'Classifier' in obj.__class__.__name__:
                        return f"<{obj.__class__.__name__} object>"
                    elif hasattr(obj, '__class__') and 'Regressor' in obj.__class__.__name__:
                        return f"<{obj.__class__.__name__} object>"
                    elif hasattr(obj, '__class__') and 'Scaler' in obj.__class__.__name__:
                        return f"<{obj.__class__.__name__} object>"
                    elif isinstance(obj, dict):
                        return {k: safe_json_serializer(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [safe_json_serializer(item) for item in obj]
                    elif isinstance(obj, tuple):
                        return list(safe_json_serializer(item) for item in obj)
                    elif hasattr(obj, '__dict__'):
                        # For any object with __dict__, convert to string representation
                        return f"<{obj.__class__.__name__} object>"
                    else:
                        return str(obj) if obj is not None else None
                except Exception as e:
                    return f"<SerializationError: {str(e)}>"
            
            # Save training results
            training_results_file = session_cache_dir / self.cache_structure['training_results']
            with open(training_results_file, 'w', encoding='utf-8') as f:
                serializable_results = safe_json_serializer(training_results)
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            # Save comprehensive results
            comprehensive_results_file = session_cache_dir / self.cache_structure['comprehensive_results']
            with open(comprehensive_results_file, 'w', encoding='utf-8') as f:
                serializable_comprehensive = safe_json_serializer(comprehensive_results)
                json.dump(serializable_comprehensive, f, indent=2, ensure_ascii=False)
            
            # Save step data
            step_data_file = session_cache_dir / self.cache_structure['step_data']
            with open(step_data_file, 'w', encoding='utf-8') as f:
                serializable_step_data = safe_json_serializer(step_data)
                json.dump(serializable_step_data, f, indent=2, ensure_ascii=False)
            
            # Save metadata
            metadata_file = session_cache_dir / self.cache_structure['metadata']
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Training results cached successfully: {session_cache_dir}")
            print(f"SUCCESS: Training results cached to {session_cache_dir}")
            
            return str(session_cache_dir)
            
        except Exception as e:
            logger.error(f"Failed to save training results: {e}")
            print(f"ERROR: Failed to save training results: {e}")
            raise
    
    def load_training_results(self, session_key: str) -> Dict[str, Any]:
        """Load comprehensive training results from cache"""
        try:
            session_cache_dir = self.cache_root_dir / session_key
            
            if not session_cache_dir.exists():
                raise FileNotFoundError(f"Training results cache not found: {session_key}")
            
            # Load all components
            training_results_file = session_cache_dir / self.cache_structure['training_results']
            with open(training_results_file, 'r', encoding='utf-8') as f:
                training_results = json.load(f)
            
            comprehensive_results_file = session_cache_dir / self.cache_structure['comprehensive_results']
            with open(comprehensive_results_file, 'r', encoding='utf-8') as f:
                comprehensive_results = json.load(f)
            
            step_data_file = session_cache_dir / self.cache_structure['step_data']
            with open(step_data_file, 'r', encoding='utf-8') as f:
                step_data = json.load(f)
            
            metadata_file = session_cache_dir / self.cache_structure['metadata']
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Reconstruct complete training results
            complete_results = training_results.copy()
            complete_results['comprehensive_results'] = comprehensive_results
            complete_results.update(step_data)
            complete_results['metadata'] = metadata
            complete_results['from_cache'] = True
            complete_results['cache_path'] = str(session_cache_dir)
            
            logger.info(f"Training results loaded from cache: {session_key}")
            print(f"SUCCESS: Training results loaded from cache: {session_key}")
            
            return complete_results
            
        except Exception as e:
            logger.error(f"Failed to load training results: {e}")
            print(f"ERROR: Failed to load training results: {e}")
            raise
    
    def check_training_results_exists(self, session_key: str) -> bool:
        """Check if training results cache exists"""
        session_cache_dir = self.cache_root_dir / session_key
        
        if not session_cache_dir.exists():
            return False
        
        required_files = [
            self.cache_structure['training_results'],
            self.cache_structure['comprehensive_results'],
            self.cache_structure['step_data'],
            self.cache_structure['metadata']
        ]
        
        for file_name in required_files:
            if not (session_cache_dir / file_name).exists():
                return False
        
        return True


# Global cache manager instances
cache_manager = CacheManager()
training_results_cache = TrainingResultsCacheManager()
