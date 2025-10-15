"""
LightGBM Classification Model with GPU Support
"""

from typing import Dict, Any, Union, Tuple
import numpy as np
from scipy import sparse

from ..base.base_model import BaseModel
from ..base.metrics import ModelMetrics


class LightGBMModel(BaseModel):
    """LightGBM classification model with GPU-first configuration"""
    
    def __init__(self, **kwargs):
        """Initialize LightGBM model"""
        super().__init__(**kwargs)
        
        # Import LightGBM
        try:
            import lightgbm as lgb
            self.lgb = lgb
        except ImportError:
            raise ImportError("LightGBM is required but not installed. Please install with: pip install lightgbm")
        
        # Default parameters (can be overridden)
        default_params = {
            'n_estimators': 100,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'min_child_samples': 20,
            'lambda_l1': 0.0,
            'lambda_l2': 0.0,
            'random_state': 42,
            'verbosity': -1
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
            
            device_config = configure_model_device("lightgbm")
            
            if device_config["use_gpu"]:
                params.update(device_config["device_params"])
                print(f"LightGBM configured for GPU: {device_config['gpu_info']}")
            else:
                params.update({
                    "device_type": "cpu"
                })
                print(f"LightGBM configured for CPU")
                
        except ImportError:
            # Fallback to CPU if gpu_config_manager not available
            params.update({
                "device_type": "cpu"
            })
            print(f"LightGBM configured for CPU (fallback)")
    
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'LightGBMModel':
        """Fit LightGBM model to training data"""
        
        # Create model with parameters
        self.model = self.lgb.LGBMClassifier(**self.model_params)
        
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
    
    def get_feature_importance(self, importance_type: str = 'split') -> np.ndarray:
        """Get feature importance from LightGBM
        
        Args:
            importance_type: Type of importance ('split', 'gain')
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    def get_feature_names(self) -> list:
        """Get feature names"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature names")
        
        if hasattr(self.model, 'feature_name_'):
            return list(self.model.feature_name_)
        return None
    
    def get_booster(self):
        """Get the underlying booster object"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting booster")
        
        if hasattr(self.model, 'booster_'):
            return self.model.booster_
        return None
    
    def train_and_test(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train and test LightGBM model"""
        
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
            'model_type': 'LGBMClassifier',
            'supports_gpu': True
        })
        
        # Add LightGBM-specific information if available
        if self.is_fitted:
            info.update({
                'device_type': self.model_params.get('device_type', 'cpu'),
                'feature_names': self.get_feature_names()
            })
        
        return info
