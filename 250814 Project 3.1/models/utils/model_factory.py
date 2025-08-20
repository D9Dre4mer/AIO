"""
Model Factory for creating model instances
"""

from typing import Dict, Any
from ..base.base_model import BaseModel
from .model_registry import model_registry


class ModelFactory:
    """Factory for creating model instances"""
    
    def __init__(self):
        """Initialize model factory"""
        self.registry = model_registry
        
    def create_model(
        self, 
        model_name: str, 
        **kwargs
    ) -> BaseModel:
        """Create a model instance by name"""
        
        # Get model class from registry
        model_class = self.registry.get_model(model_name)
        
        # Create and return instance
        return model_class(**kwargs)
    
    def create_model_with_config(
        self, 
        model_name: str, 
        config: Dict[str, Any]
    ) -> BaseModel:
        """Create a model instance with configuration dictionary"""
        
        # Get model class from registry
        model_class = self.registry.get_model(model_name)
        
        # Create instance with config
        return model_class(**config)
    
    def create_models_batch(
        self, 
        model_names: list, 
        **kwargs
    ) -> Dict[str, BaseModel]:
        """Create multiple model instances"""
        
        models = {}
        for name in model_names:
            try:
                models[name] = self.create_model(name, **kwargs)
            except Exception as e:
                print(f"⚠️ Warning: Could not create model '{name}': {e}")
                continue
        
        return models
    
    def get_available_models(self) -> list:
        """Get list of available model names"""
        return self.registry.list_models()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        return self.registry.get_model_metadata(model_name)
    
    def get_model_categories(self) -> Dict[str, list]:
        """Get models grouped by category"""
        return self.registry.get_model_categories()
    
    def validate_model_name(self, model_name: str) -> bool:
        """Check if a model name is valid"""
        return self.registry.is_model_registered(model_name)
    
    def suggest_models(
        self, 
        task_type: str = None, 
        data_type: str = None
    ) -> list:
        """Suggest models based on task and data type"""
        
        suggestions = []
        all_metadata = self.registry.get_all_metadata()
        
        for name, metadata in all_metadata.items():
            if task_type and metadata.get('task_type') != task_type:
                continue
            if data_type and metadata.get('data_type') != data_type:
                continue
            suggestions.append(name)
        
        return suggestions


# Global model factory instance
model_factory = ModelFactory()
