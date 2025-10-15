"""
Model Factory for creating model instances
"""

from typing import Dict, Any
from ..base.base_model import BaseModel


class ModelFactory:
    """Factory for creating model instances"""
    
    def __init__(self, registry=None):
        """Initialize model factory"""
        self.registry = registry
        
    def create_model(
        self, 
        model_name: str, 
        **kwargs
    ) -> BaseModel:
        """Create a model instance by name"""
        
        if not self.registry:
            raise ValueError("Model registry not set. Please set registry first.")
        
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
        
        if not self.registry:
            raise ValueError("Model registry not set. Please set registry first.")
        
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
    
    def create_ensemble_model(
        self, 
        base_model_names: list,
        final_estimator: str = 'logistic_regression',
        cv_folds: int = 5,
        random_state: int = 42
    ) -> 'EnsembleManager':
        """Create ensemble learning model"""
        
        try:
            from ..ensemble.ensemble_manager import EnsembleManager
            
            ensemble = EnsembleManager(
                base_models=base_model_names,
                final_estimator=final_estimator,
                cv_folds=cv_folds,
                random_state=random_state
            )
            
            print(f"✅ Ensemble model created successfully")
            return ensemble
            
        except Exception as e:
            print(f"❌ Error creating ensemble model: {e}")
            raise
    
    def get_available_models(self) -> list:
        """Get list of available model names"""
        if not self.registry:
            return []
        return self.registry.list_models()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if not self.registry:
            return {}
        return self.registry.get_model_metadata(model_name)
    
    def get_model_categories(self) -> Dict[str, list]:
        """Get models grouped by category"""
        if not self.registry:
            return {}
        return self.registry.get_model_categories()
    
    def validate_model_name(self, model_name: str) -> bool:
        """Check if a model name is valid"""
        if not self.registry:
            return False
        return self.registry.is_model_registered(model_name)
    
    def suggest_models(
        self, 
        task_type: str = None, 
        data_type: str = None
    ) -> list:
        """Suggest models based on task and data type"""
        
        if not self.registry:
            return []
        
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
