"""
K-Nearest Neighbors Classification Model
"""

from typing import Dict, Any, Union, Tuple
import numpy as np
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier

from ..base.base_model import BaseModel
from ..base.metrics import ModelMetrics
from config import KNN_N_NEIGHBORS


class KNNModel(BaseModel):
    """K-Nearest Neighbors classification model"""
    
    def __init__(self, n_neighbors: int = KNN_N_NEIGHBORS, **kwargs):
        """Initialize KNN model"""
        super().__init__(n_neighbors=n_neighbors, **kwargs)
        self.n_neighbors = n_neighbors
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'KNNModel':
        """Fit KNN model to training data"""
        
        # Choose algorithm based on data type
        algorithm = 'brute' if sparse.issparse(X) else 'auto'
        
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors, 
            algorithm=algorithm
        )
        self.model.fit(X, y)
        
        self.is_fitted = True
        self.training_history.append({
            'action': 'fit',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_neighbors': self.n_neighbors,
            'algorithm': algorithm
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
    
    def get_feature_importance(self) -> None:
        """KNN doesn't provide feature importance"""
        return None
    
    def train_and_test(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train and test KNN model"""
        
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
        # Determine algorithm based on model state
        if hasattr(self, 'model') and self.model is not None:
            algorithm = self.model.algorithm
        else:
            algorithm = 'unknown'
            
        info.update({
            'n_neighbors': self.n_neighbors,
            'algorithm': algorithm
        })
        return info
