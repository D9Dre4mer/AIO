"""
Naive Bayes Classification Model
"""

from typing import Dict, Any, Union, Tuple
import numpy as np
from scipy import sparse
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from ..base.base_model import BaseModel
from ..base.metrics import ModelMetrics


class NaiveBayesModel(BaseModel):
    """Naive Bayes classification model with automatic type selection"""
    
    def __init__(self, **kwargs):
        """Initialize Naive Bayes model"""
        super().__init__(**kwargs)
        self.nb_type = None
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'NaiveBayesModel':
        """Fit Naive Bayes model to training data"""
        
        # Choose appropriate Naive Bayes variant
        if sparse.issparse(X):
            print("📊 Using MultinomialNB for sparse text features")
            self.model = MultinomialNB()
            self.nb_type = 'MultinomialNB'
        else:
            print("📊 Using GaussianNB for dense features")
            self.model = GaussianNB()
            self.nb_type = 'GaussianNB'
        
        self.model.fit(X, y)
        
        self.is_fitted = True
        self.training_history.append({
            'action': 'fit',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'nb_type': self.nb_type
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
        """Naive Bayes doesn't provide feature importance"""
        return None
    
    def train_and_test(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train and test Naive Bayes model"""
        
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
            'nb_type': self.nb_type,
            'has_feature_importance': False
        })
        return info
