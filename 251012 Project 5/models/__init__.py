"""
Models package for Topic Modeling Project
Provides modular architecture for different machine learning models
"""

# New modular architecture
from .base.base_model import BaseModel
from .base.interfaces import ModelInterface
from .base.metrics import ModelMetrics
from .clustering.kmeans_model import KMeansModel
from .classification.knn_model import KNNModel
from .classification.decision_tree_model import DecisionTreeModel
from .classification.naive_bayes_model import NaiveBayesModel
from .classification.logistic_regression_model import LogisticRegressionModel
from .classification.linear_svc_model import LinearSVCModel
from .ensemble.ensemble_manager import EnsembleManager
from .ensemble.stacking_classifier import EnsembleStackingClassifier
from .utils.model_factory import ModelFactory
from .utils.model_registry import ModelRegistry
from .utils.validation_manager import ValidationManager

# Import and register models first
from .register_models import register_all_models

# Create global instances
validation_manager = ValidationManager()
model_registry = ModelRegistry()
model_factory = ModelFactory(registry=model_registry)

# Register all models in the global registry
register_all_models(model_registry)


__all__ = [
    # New modular architecture
    'BaseModel',
    'ModelInterface', 
    'ModelMetrics',
    'KMeansModel',
    'KNNModel',
    'DecisionTreeModel',
    'NaiveBayesModel',
    'LogisticRegressionModel',
    'LinearSVCModel',
    'EnsembleManager',
    'EnsembleStackingClassifier',
    'ModelFactory',
    'ModelRegistry',
    'ValidationManager',
    
    # Global instances
    'validation_manager',
    'model_factory',
    'model_registry'
]

# Version info
__version__ = "4.0.0"
__author__ = "Topic Modeling Project Team"
