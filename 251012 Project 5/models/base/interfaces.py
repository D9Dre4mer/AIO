"""
Model interfaces and protocols for Topic Modeling Project
Defines the contract that all models must implement
"""

from typing import Protocol, Tuple, Dict, Any, Union, Optional
import numpy as np
from scipy import sparse


class ModelInterface(Protocol):
    """Protocol defining the interface for all machine learning models"""
    
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'ModelInterface':
        """Fit the model to the training data"""
        ...
    
    def predict(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Make predictions on new data"""
        ...
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        ...
    
    def set_params(self, **params) -> 'ModelInterface':
        """Set model parameters"""
        ...


class TrainableModelInterface(Protocol):
    """Protocol for models that can be trained and evaluated"""
    
    def train_and_test(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train the model and test it, returning predictions, accuracy, and report"""
        ...
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model type and parameters"""
        ...


class ClusteringModelInterface(Protocol):
    """Protocol for clustering models"""
    
    def fit_predict(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Fit the model and return cluster labels"""
        ...
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers"""
        ...
    
    def get_n_clusters(self) -> int:
        """Get number of clusters"""
        ...


class ClassificationModelInterface(Protocol):
    """Protocol for classification models"""
    
    def predict_proba(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Get prediction probabilities"""
        ...
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores if available"""
        ...


class DeepLearningModelInterface(Protocol):
    """Protocol for deep learning models"""
    
    def save_model(self, path: str) -> None:
        """Save the trained model"""
        ...
    
    def load_model(self, path: str) -> None:
        """Load a saved model"""
        ...
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        ...
