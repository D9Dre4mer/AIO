"""
Ensemble Learning Package for Topic Modeling Project
Provides ensemble learning capabilities using StackingClassifier
"""

from .ensemble_manager import EnsembleManager
from .stacking_classifier import EnsembleStackingClassifier

__all__ = [
    'EnsembleManager',
    'EnsembleStackingClassifier'
]

# Version info
__version__ = "1.0.0"
__author__ = "Topic Modeling Project Team"
