"""
Logistic Regression Classification Model
"""

from typing import Dict, Any, Union, Tuple
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

# Parallel processing disabled - using pickle instead

from ..base.base_model import BaseModel
from ..base.metrics import ModelMetrics


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classification model with automatic parameter optimization"""
    
    def __init__(self, **kwargs):
        """Initialize Logistic Regression model"""
        super().__init__(**kwargs)
        
        # Default parameters (can be overridden)
        default_params = {
            'max_iter': 2000,
            'multi_class': 'multinomial',
            'n_jobs': -1,  # Use all CPU cores by default
            'random_state': 42,
            'C': 1.0,
            'solver': 'lbfgs'
        }
        
        # Update with user-provided parameters
        default_params.update(kwargs)
        self.model_params = default_params
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'LogisticRegressionModel':
        """Fit Logistic Regression model to training data"""
        
        # Create model with parameters
        self.model = LogisticRegression(**self.model_params)
        
        # Display multithreading info
        n_jobs = self.model_params.get('n_jobs', -1)
        if n_jobs == -1:
            import os
            cpu_count = os.cpu_count()
            print(f"🔄 CPU multithreading: Using all {cpu_count} available cores")
        else:
            print(f"🔄 CPU multithreading: Using {n_jobs} parallel jobs")
        
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
        """Get feature importance (coefficients) for linear models"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Logistic Regression provides coefficients as feature importance
        if hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])  # Take first class for binary, or average for multiclass
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
    
    def train_and_test(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train and test Logistic Regression model"""
        
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
            'model_type': 'LogisticRegression'
        })
        
        # Add coefficient information if available
        if self.is_fitted and hasattr(self.model, 'coef_'):
            info.update({
                'n_coefficients': self.model.coef_.shape[0],
                'n_features': self.model.coef_.shape[1]
            })
        
        return info
