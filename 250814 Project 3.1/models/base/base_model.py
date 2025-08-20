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
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
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
        """Save the trained model (to be implemented by subclasses)"""
        raise NotImplementedError(
            "Model saving not implemented for this model type"
        )
    
    def load_model(self, path: str) -> None:
        """Load a saved model (to be implemented by subclasses)"""
        raise NotImplementedError(
            "Model loading not implemented for this model type"
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            'model_type': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'parameters': self.model_params,
            'training_history_length': len(self.training_history),
            'has_validation_metrics': bool(self.validation_metrics)
        }
