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
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages per-model caching with config_hash and dataset_fingerprint"""
    
    def __init__(self, cache_root_dir: str = "cache/models/"):
        """Initialize cache manager
        
        Args:
            cache_root_dir: Root directory for model caches
        """
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
            for ext in ['joblib', 'json', 'txt', 'cbm']:
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
        cache_path = self.get_cache_path(model_key, dataset_id, config_hash)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model artifact
            model_ext = self._get_model_extension(model_key)
            model_file = cache_path / f"model.{model_ext}"
            self._save_model_artifact(model, model_file, model_key)
            
            # Custom JSON serializer to handle non-serializable objects
            def safe_json_serializer(obj):
                """Convert non-serializable objects to strings"""
                if hasattr(obj, '__module__'):
                    return f"<{obj.__class__.__name__} object>"
                elif hasattr(obj, '__dict__'):
                    return str(obj)
                else:
                    return str(obj)
            
            # Save params
            params_file = cache_path / self.cache_structure['params']
            with open(params_file, 'w') as f:
                json.dump(params, f, indent=2, default=safe_json_serializer)
            
            # Save metrics
            metrics_file = cache_path / self.cache_structure['metrics']
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=safe_json_serializer)
            
            # Save config
            config_file = cache_path / self.cache_structure['config']
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2, default=safe_json_serializer)
            
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
            model = self._load_model_artifact(model_file, model_key)
            
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
            return 'joblib'
    
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
        elif model_key in ['catboost']:
            # Use CatBoost's native save method directly
            import catboost as cb
            # Access the actual CatBoost model instance
            cb_model = model.model if hasattr(model, 'model') else model
            if hasattr(cb_model, 'save_model'):
                cb_model.save_model(str(file_path))
            else:
                joblib.dump(cb_model, file_path)
        elif model_key.startswith('stacking_ensemble_') or model_key.startswith('voting_ensemble_'):
            # Save ensemble model data (contains ensemble_manager and metadata)
            joblib.dump(model, file_path)
        else:
            joblib.dump(model, file_path)
    
    def _load_model_artifact(self, file_path: Path, model_key: str):
        """Load model artifact based on type
        
        Args:
            file_path: Path to model file
            model_key: Model identifier
            
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
            # Load the saved model directly
            booster = lgb.Booster(model_file=str(file_path))
            # Create a wrapper classifier
            classifier = lgb.LGBMClassifier()
            classifier._Booster = booster
            # Mark as fitted and set necessary attributes
            classifier._fitted = True
            classifier._n_features = booster.num_feature()
            # Try to get num_classes, fallback to 2 if not available
            try:
                classifier._n_classes = booster.num_class()
            except AttributeError:
                classifier._n_classes = 2  # Default to binary classification
            classifier._classes = list(range(classifier._n_classes))
            classifier._estimator_type = 'classifier'
            # Set additional required attributes
            classifier._evals_result = {}
            classifier._train_set = None
            classifier._valid_sets = None
            return classifier
        elif model_key in ['catboost']:
            import catboost as cb
            model = cb.CatBoostClassifier()
            model.load_model(str(file_path))
            return model
        elif model_key.startswith('stacking_ensemble_') or model_key.startswith('voting_ensemble_'):
            # Load ensemble model data (contains ensemble_manager and metadata)
            return joblib.load(file_path)
        else:
            return joblib.load(file_path)
    
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


# Global cache manager instance
cache_manager = CacheManager()
