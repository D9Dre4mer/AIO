"""
Gradient Boosting Classification Model
"""

from typing import Dict, Any, Union, Tuple
import numpy as np
from scipy import sparse
from sklearn.ensemble import GradientBoostingClassifier

from ..base.base_model import BaseModel
from ..base.metrics import ModelMetrics


class GradientBoostingModel(BaseModel):
    """Gradient Boosting classification model with GPU-first configuration"""
    
    def __init__(self, **kwargs):
        """Initialize Gradient Boosting model"""
        super().__init__(**kwargs)
        
        # Default parameters (can be overridden)
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 1.0,
            'max_features': None,
            'random_state': 42,
            'verbose': 0
        }
        
        # Update with user-provided parameters
        default_params.update(kwargs)
        self.model_params = default_params
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'GradientBoostingModel':
        """Fit Gradient Boosting model to training data"""
        
        # Create model with parameters
        self.model = GradientBoostingClassifier(**self.model_params)
        
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
        """Get feature importance from Gradient Boosting"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    def get_train_score(self) -> np.ndarray:
        """Get training score for each boosting stage"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting training score")
        
        if hasattr(self.model, 'train_score_'):
            return self.model.train_score_
        return None
    
    def get_oob_improvement(self) -> np.ndarray:
        """Get out-of-bag improvement for each boosting stage"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting OOB improvement")
        
        if hasattr(self.model, 'oob_improvement_'):
            return self.model.oob_improvement_
        return None
    
    def train_and_test(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train and test Gradient Boosting model"""
        
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
            'model_type': 'GradientBoostingClassifier'
        })
        
        # Add training information if available
        if self.is_fitted and hasattr(self.model, 'train_score_'):
            info.update({
                'n_estimators': len(self.model.train_score_),
                'train_score': self.get_train_score(),
                'oob_improvement': self.get_oob_improvement()
            })
        
        return info
