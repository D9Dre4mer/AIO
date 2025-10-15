"""
Abstract base class for all machine learning models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import numpy as np
from scipy import sparse


class BaseModel(ABC):
    """Abstract base class that all models must inherit from"""
    
    def __init__(self, **kwargs):
        """Initialize base model with parameters"""
        self.model = None
        self.is_fitted = False
        self.model_params = kwargs
        self.training_history = []
        self.validation_metrics = {}
        
    @abstractmethod
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'BaseModel':
        """Fit the model to training data"""
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Make predictions on new data"""
        pass
    
    def score(self, X: Union[np.ndarray, sparse.csr_matrix], y: np.ndarray) -> float:
        """Score the model on test data (accuracy for classification)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y)
        return accuracy
    
    def validate(
        self, 
        X_val: Union[np.ndarray, sparse.csr_matrix], 
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Validate the model on validation set"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before validation")
        
        # Make predictions on validation set
        y_val_pred = self.predict(X_val)
        
        # Compute validation metrics
        from .metrics import ModelMetrics
        metrics = ModelMetrics.compute_classification_metrics(y_val, y_val_pred)
        
        # Store validation metrics
        self.validation_metrics = metrics
        
        # Add to training history
        self.training_history.append({
            'action': 'validate',
            'n_samples': X_val.shape[0],
            'validation_metrics': metrics
        })
        
        return metrics
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters (compatible with scikit-learn)"""
        return self.model_params.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters"""
        self.model_params.update(params)
        return self
    
    def is_model_fitted(self) -> bool:
        """Check if model has been fitted"""
        return self.is_fitted
    
    def get_training_history(self) -> list:
        """Get training history"""
        return self.training_history.copy()
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation metrics if available"""
        return self.validation_metrics.copy() if self.validation_metrics else {}
    
    def save_model(self, path: str) -> None:
        """Save the trained model using pickle"""
        import pickle
        import os
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model data
        model_data = {
            'model_instance': self,
            'model_params': self.model_params,
            'is_fitted': self.is_fitted,
            'training_history': self.training_history,
            'validation_metrics': self.validation_metrics,
            'model_type': self.__class__.__name__
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path: str) -> None:
        """Load a saved model using pickle"""
        import pickle
        import os
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore model state
        self.model_params = model_data.get('model_params', {})
        self.is_fitted = model_data.get('is_fitted', False)
        self.training_history = model_data.get('training_history', [])
        self.validation_metrics = model_data.get('validation_metrics', {})
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            'model_type': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'parameters': self.model_params,
            'training_history_length': len(self.training_history),
            'has_validation_metrics': bool(self.validation_metrics)
        }
