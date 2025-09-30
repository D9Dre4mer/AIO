"""
Random Forest Classification Model
"""

from typing import Dict, Any, Union, Tuple
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier

from ..base.base_model import BaseModel
from ..base.metrics import ModelMetrics


class RandomForestModel(BaseModel):
    """Random Forest classification model with GPU-first configuration"""
    
    def __init__(self, **kwargs):
        """Initialize Random Forest model"""
        super().__init__(**kwargs)
        
        # Default parameters (can be overridden)
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'max_features': 'sqrt',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1,  # Use all CPU cores by default
            'verbose': 0
        }
        
        # Update with user-provided parameters
        default_params.update(kwargs)
        self.model_params = default_params
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'RandomForestModel':
        """Fit Random Forest model to training data"""
        
        # Create model with parameters
        self.model = RandomForestClassifier(**self.model_params)
        
        # Display multithreading info
        n_jobs = self.model_params.get('n_jobs', -1)
        if n_jobs == -1:
            import os
            cpu_count = os.cpu_count()
            print(f"CPU multithreading: Using all {cpu_count} available cores")
        else:
            print(f"CPU multithreading: Using {n_jobs} parallel jobs")
        
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
        """Get feature importance from Random Forest"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    def get_feature_importance_std(self) -> np.ndarray:
        """Get standard deviation of feature importance across trees"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance std")
        
        if hasattr(self.model, 'estimators_'):
            importances = np.array([tree.feature_importances_ for tree in self.model.estimators_])
            return np.std(importances, axis=0)
        return None
    
    def get_oob_score(self) -> float:
        """Get out-of-bag score if available"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting OOB score")
        
        if hasattr(self.model, 'oob_score_'):
            return self.model.oob_score_
        return None
    
    def train_and_test(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train and test Random Forest model"""
        
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
            'model_type': 'RandomForestClassifier',
            'supports_oob_score': True
        })
        
        # Add feature importance information if available
        if self.is_fitted and hasattr(self.model, 'feature_importances_'):
            info.update({
                'n_features': len(self.model.feature_importances_),
                'oob_score': self.get_oob_score()
            })
        
        return info
