"""
Linear Support Vector Classification Model
"""

from typing import Dict, Any, Union, Tuple
import numpy as np
from scipy import sparse
from sklearn.svm import LinearSVC

from ..base.base_model import BaseModel
from ..base.metrics import ModelMetrics


class LinearSVCModel(BaseModel):
    """Linear Support Vector Classification model"""
    
    def __init__(self, **kwargs):
        """Initialize Linear SVC model"""
        super().__init__(**kwargs)
        
        # Default parameters (can be overridden)
        default_params = {
            'random_state': 42,
            'max_iter': 2000,
            'C': 1.0,
            'loss': 'squared_hinge',
            'dual': True
        }
        
        # Update with user-provided parameters
        default_params.update(kwargs)
        self.model_params = default_params
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'LinearSVCModel':
        """Fit Linear SVC model to training data"""
        
        # Create model with parameters
        self.model = LinearSVC(**self.model_params)
        
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
        """Get prediction probabilities (LinearSVC doesn't support this by default)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # LinearSVC doesn't support predict_proba by default
        # Return None to indicate this feature is not available
        return None
    
    def decision_function(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Get decision function values (distance from hyperplane)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting decision function")
        
        return self.model.decision_function(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance (coefficients) for linear models"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # LinearSVC provides coefficients as feature importance
        if hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])  # Take first class for binary
        return None
    
    def get_coefficients(self) -> np.ndarray:
        """Get model coefficients"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting coefficients")
        
        if hasattr(self.model, 'coef_'):
            return self.model.coef_
        return None
    
    def get_intercept(self) -> np.ndarray:
        """Get model intercept"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting intercept")
        
        if hasattr(self.model, 'intercept_'):
            return self.model.intercept_
        return None
    
    def get_support_vectors(self) -> np.ndarray:
        """Get support vectors (LinearSVC doesn't store these by default)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting support vectors")
        
        # LinearSVC doesn't store support vectors by default
        # Return None to indicate this feature is not available
        return None
    
    def train_and_test(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train and test Linear SVC model"""
        
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
            'has_probabilities': False,  # LinearSVC doesn't support predict_proba
            'supports_sparse': True,
            'model_type': 'LinearSVC',
            'has_decision_function': True
        })
        
        # Add coefficient information if available
        if self.is_fitted and hasattr(self.model, 'coef_'):
            info.update({
                'n_coefficients': self.model.coef_.shape[0],
                'n_features': self.model.coef_.shape[1]
            })
        
        return info
