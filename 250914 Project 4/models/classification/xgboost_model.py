"""
XGBoost Classification Model with GPU Support
"""

from typing import Dict, Any, Union, Tuple
import numpy as np
from scipy import sparse

from ..base.base_model import BaseModel
from ..base.metrics import ModelMetrics


class XGBoostModel(BaseModel):
    """XGBoost classification model with GPU-first configuration"""
    
    def __init__(self, **kwargs):
        """Initialize XGBoost model"""
        super().__init__(**kwargs)
        
        # Import XGBoost
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("XGBoost is required but not installed. Please install with: pip install xgboost")
        
        # Default parameters (can be overridden)
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'eta': 0.3,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'min_child_weight': 1,
            'reg_lambda': 1.0,
            'reg_alpha': 0.0,
            'random_state': 42,
            'verbosity': 0
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
            
            device_config = configure_model_device("xgboost")
            
            if device_config["use_gpu"]:
                params.update(device_config["device_params"])
                print(f"XGBoost configured for GPU: {device_config['gpu_info']}")
            else:
                params.update({
                    "tree_method": "hist",
                    "predictor": "auto"
                })
                print(f"💻 XGBoost configured for CPU")
                
        except ImportError:
            # Fallback to CPU if gpu_config_manager not available
            params.update({
                "tree_method": "hist",
                "predictor": "auto"
            })
            print(f"💻 XGBoost configured for CPU (fallback)")
    
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'XGBoostModel':
        """Fit XGBoost model to training data"""
        
        # Create model with parameters
        self.model = self.xgb.XGBClassifier(**self.model_params)
        
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
    
    def get_feature_importance(self, importance_type: str = 'weight') -> np.ndarray:
        """Get feature importance from XGBoost
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'get_booster'):
            booster = self.model.get_booster()
            importance_dict = booster.get_score(importance_type=importance_type)
            
            # Convert to array format
            feature_names = self.model.feature_names_in_
            importance_array = np.zeros(len(feature_names))
            
            for i, feature in enumerate(feature_names):
                if feature in importance_dict:
                    importance_array[i] = importance_dict[feature]
            
            return importance_array
        
        return None
    
    def get_feature_names(self) -> list:
        """Get feature names"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature names")
        
        if hasattr(self.model, 'feature_names_in_'):
            return list(self.model.feature_names_in_)
        return None
    
    def train_and_test(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train and test XGBoost model"""
        
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
            'model_type': 'XGBClassifier',
            'supports_gpu': True
        })
        
        # Add XGBoost-specific information if available
        if self.is_fitted:
            info.update({
                'tree_method': self.model_params.get('tree_method', 'hist'),
                'predictor': self.model_params.get('predictor', 'auto'),
                'feature_names': self.get_feature_names()
            })
        
        return info
