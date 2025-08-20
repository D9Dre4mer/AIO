"""
Decision Tree Classification Model
"""

from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier

from ..base.base_model import BaseModel
from ..base.metrics import ModelMetrics


class DecisionTreeModel(BaseModel):
    """Decision Tree classification model"""
    
    def __init__(self, random_state: int = 42, **kwargs):
        """Initialize Decision Tree model"""
        super().__init__(random_state=random_state, **kwargs)
        self.random_state = random_state
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'DecisionTreeModel':
        """Fit Decision Tree model to training data"""
        
        self.model = DecisionTreeClassifier(random_state=self.random_state)
        self.model.fit(X, y)
        
        self.is_fitted = True
        self.training_history.append({
            'action': 'fit',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'random_state': self.random_state
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
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores"""
        if not self.is_fitted:
            return None
        return self.model.feature_importances_
    
    def train_and_test(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train and test Decision Tree model"""
        
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
            'random_state': self.random_state,
            'has_feature_importance': self.get_feature_importance() is not None
        })
        return info
