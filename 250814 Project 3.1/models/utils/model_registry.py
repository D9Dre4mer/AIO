"""
Model Registry for managing available models
"""

from typing import Dict, Type, List, Any
from ..base.base_model import BaseModel


class ModelRegistry:
    """Registry for managing available machine learning models"""
    
    def __init__(self):
        """Initialize model registry"""
        self._models: Dict[str, Type[BaseModel]] = {}
        self._model_metadata: Dict[str, Dict[str, Any]] = {}
        
    def register_model(
        self, 
        name: str, 
        model_class: Type[BaseModel], 
        metadata: Dict[str, Any] = None
    ) -> None:
        """Register a new model class"""
        self._models[name] = model_class
        self._model_metadata[name] = metadata or {}
        
    def get_model(self, name: str) -> Type[BaseModel]:
        """Get a model class by name"""
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found in registry")
        return self._models[name]
    
    def list_models(self) -> List[str]:
        """List all registered model names"""
        return list(self._models.keys())
    
    def get_model_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a specific model"""
        if name not in self._model_metadata:
            return {}
        return self._model_metadata[name].copy()
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all models"""
        return self._model_metadata.copy()
    
    def is_model_registered(self, name: str) -> bool:
        """Check if a model is registered"""
        return name in self._models
    
    def unregister_model(self, name: str) -> None:
        """Unregister a model"""
        if name in self._models:
            del self._models[name]
        if name in self._model_metadata:
            del self._model_metadata[name]
    
    def get_model_categories(self) -> Dict[str, List[str]]:
        """Get models grouped by category"""
        categories = {}
        for name, metadata in self._model_metadata.items():
            category = metadata.get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        return categories
