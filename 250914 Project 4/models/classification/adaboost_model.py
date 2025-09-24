"""
AdaBoost Classification Model
"""

from typing import Dict, Any, Union, Tuple
import numpy as np
from scipy import sparse
from sklearn.ensemble import AdaBoostClassifier

from ..base.base_model import BaseModel
from ..base.metrics import ModelMetrics


class AdaBoostModel(BaseModel):
    """AdaBoost classification model with GPU-first configuration"""
    
    def __init__(self, **kwargs):
        """Initialize AdaBoost model"""
        super().__init__(**kwargs)
        
        # Default parameters (can be overridden)
        default_params = {
            'n_estimators': 50,
            'learning_rate': 1.0,
            'algorithm': 'SAMME.R',
            'random_state': 42
        }
        
        # Update with user-provided parameters
        default_params.update(kwargs)
        self.model_params = default_params
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'AdaBoostModel':
        """Fit AdaBoost model to training data"""
        
        # Create model with parameters
        self.model = AdaBoostClassifier(**self.model_params)
        
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
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from AdaBoost"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    def get_estimator_errors(self) -> np.ndarray:
        """Get errors for each estimator"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting estimator errors")
        
        if hasattr(self.model, 'estimator_errors_'):
            return self.model.estimator_errors_
        return None
    
    def get_estimator_weights(self) -> np.ndarray:
        """Get weights for each estimator"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting estimator weights")
        
        if hasattr(self.model, 'estimator_weights_'):
            return self.model.estimator_weights_
        return None
    
    def train_and_test(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train and test AdaBoost model"""
        
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
            'model_type': 'AdaBoostClassifier'
        })
        
        # Add estimator information if available
        if self.is_fitted and hasattr(self.model, 'estimators_'):
            info.update({
                'n_estimators': len(self.model.estimators_),
                'estimator_errors': self.get_estimator_errors(),
                'estimator_weights': self.get_estimator_weights()
            })
        
        return info
