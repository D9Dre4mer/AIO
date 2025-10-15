"""
CatBoost Classification Model with GPU Support
"""

from typing import Dict, Any, Union, Tuple
import numpy as np
from scipy import sparse

from ..base.base_model import BaseModel
from ..base.metrics import ModelMetrics


class CatBoostModel(BaseModel):
    """CatBoost classification model with GPU-first configuration"""
    
    def __init__(self, **kwargs):
        """Initialize CatBoost model"""
        super().__init__(**kwargs)
        
        # Import CatBoost
        try:
            import catboost as cb
            self.cb = cb
        except ImportError:
            raise ImportError("CatBoost is required but not installed. Please install with: pip install catboost")
        
        # Default parameters (can be overridden)
        default_params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3.0,
            'border_count': 128,
            'random_seed': 42,
            'train_dir': None,  # Disable training directory logging
            'logging_level': 'Silent',  # Disable verbose logging
            'allow_writing_files': False  # Completely disable file writing (prevents catboost_info/)
        }
        
        # Configure GPU/CPU based on device policy
        self._configure_device_params(default_params)
        
        # Update with user-provided parameters
        default_params.update(kwargs)
        self.model_params = default_params
        
    def _configure_device_params(self, params: Dict[str, Any]):
        """Configure device-specific parameters based on GPU availability"""
        try:
            from gpu_config_manager import configure_model_device
            
            device_config = configure_model_device("catboost")
            
            if device_config["use_gpu"]:
                params.update(device_config["device_params"])
                print(f"🚀 CatBoost configured for GPU: {device_config['gpu_info']}")
            else:
                params.update({
                    "task_type": "CPU"
                })
                print(f"💻 CatBoost configured for CPU")
                
        except ImportError:
            # Fallback to CPU if gpu_config_manager not available
            params.update({
                "task_type": "CPU"
            })
            print(f"💻 CatBoost configured for CPU (fallback)")
    
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'CatBoostModel':
        """Fit CatBoost model to training data"""
        
        # Create model with parameters
        self.model = self.cb.CatBoostClassifier(**self.model_params)
        
        # Fit the model
        self.model.fit(X, y)
        
        self.is_fitted = True
        self.training_history.append({
            'action': 'fit',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'parameters': self.model_params.copy()
        })
        
        return self
    
    def predict(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Make predictions on new data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, importance_type: str = 'PredictionValuesChange') -> np.ndarray:
        """Get feature importance from CatBoost
        
        Args:
            importance_type: Type of importance ('PredictionValuesChange', 'LossFunctionChange', 'FeatureImportance')
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'get_feature_importance'):
            return self.model.get_feature_importance(type=importance_type)
        return None
    
    def get_feature_names(self) -> list:
        """Get feature names"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature names")
        
        if hasattr(self.model, 'feature_names_'):
            return list(self.model.feature_names_)
        return None
    
    def get_best_iteration(self) -> int:
        """Get the best iteration from training"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting best iteration")
        
        if hasattr(self.model, 'get_best_iteration'):
            return self.model.get_best_iteration()
        return None
    
    def get_evals_result(self) -> dict:
        """Get evaluation results from training"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting evaluation results")
        
        if hasattr(self.model, 'get_evals_result'):
            return self.model.get_evals_result()
        return None
    
    def train_and_test(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train and test CatBoost model"""
        
        # Fit the model
        self.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Compute metrics
        metrics = ModelMetrics.compute_classification_metrics(y_test, y_pred)
        
        return y_pred, metrics['accuracy'], metrics['classification_report']
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        info = super().get_model_info()
        info.update({
            'has_feature_importance': True,
            'has_probabilities': True,
            'supports_sparse': True,
            'model_type': 'CatBoostClassifier',
            'supports_gpu': True
        })
        
        # Add CatBoost-specific information if available
        if self.is_fitted:
            info.update({
                'task_type': self.model_params.get('task_type', 'CPU'),
                'feature_names': self.get_feature_names(),
                'best_iteration': self.get_best_iteration()
            })
        
        return info
