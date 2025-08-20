"""
Support Vector Machine Model Implementation
"""

from typing import Union, Dict, Any
import numpy as np
from scipy import sparse
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from ..base.base_model import BaseModel


class SVMModel(BaseModel):
    """Support Vector Machine model for classification"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Default SVM parameters
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'random_state': 42
        }
        
        # Update with user parameters
        default_params.update(kwargs)
        self.model_params = default_params
        
        # Initialize SVM model
        self.model = SVC(**self.model_params)
    
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'SVMModel':
        """Train the SVM model"""
        
        # Convert sparse to dense if needed (SVM works better with dense)
        if sparse.issparse(X):
            X = X.toarray()
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Store training info
        self.training_history.append({
            'action': 'fit',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y))
        })
        
        return self
    
    def predict(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Make predictions using the SVM model"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert sparse to dense if needed
        if sparse.issparse(X):
            X = X.toarray()
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Get prediction probabilities (if probability=True)"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert sparse to dense if needed
        if sparse.issparse(X):
            X = X.toarray()
        
        # Check if model supports probability predictions
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model was not trained with probability=True")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the SVM model"""
        
        info = {
            'model_type': 'Support Vector Machine',
            'algorithm': 'SVM',
            'library': 'scikit-learn',
            'parameters': self.model_params.copy(),
            'is_fitted': self.is_fitted,
            'supports_sparse': True,
            'supports_multiclass': True,
            'supports_probability': 'probability' in self.model_params and 
                                  self.model_params.get('probability', False)
        }
        
        if self.is_fitted:
            info.update({
                'n_support_vectors': self.model.n_support_.sum(),
                'support_vectors_per_class': self.model.n_support_.tolist(),
                'kernel': self.model.kernel,
                'gamma': self.model.gamma if hasattr(self.model, 'gamma') else None
            })
        
        return info
    
    def evaluate(self, X: Union[np.ndarray, sparse.csr_matrix], 
                 y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        
        return {'accuracy': accuracy}
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance (for linear SVM only)"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if self.model.kernel == 'linear':
            # For linear SVM, coefficients indicate importance
            return np.abs(self.model.coef_[0])
        else:
            raise ValueError("Feature importance only available for linear SVM")
