"""
Models package for Topic Modeling Project
Provides modular architecture for different machine learning models
"""

from .base.base_model import BaseModel
from .base.interfaces import ModelInterface
from .base.metrics import ModelMetrics
from .clustering.kmeans_model import KMeansModel
from .classification.knn_model import KNNModel
from .classification.decision_tree_model import DecisionTreeModel
from .classification.naive_bayes_model import NaiveBayesModel
from .utils.model_factory import ModelFactory
from .utils.model_registry import ModelRegistry

__all__ = [
    'BaseModel',
    'ModelInterface', 
    'ModelMetrics',
    'KMeansModel',
    'KNNModel',
    'DecisionTreeModel',
    'NaiveBayesModel',
    'ModelFactory',
    'ModelRegistry'
]

# Version info
__version__ = "3.3.0"
__author__ = "Topic Modeling Project Team"
