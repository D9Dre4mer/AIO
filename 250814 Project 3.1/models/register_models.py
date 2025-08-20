"""
Script to register all models in the registry
"""

from .clustering.kmeans_model import KMeansModel
from .classification.knn_model import KNNModel
from .classification.decision_tree_model import DecisionTreeModel
from .classification.naive_bayes_model import NaiveBayesModel
from .classification.svm_model import SVMModel
from .utils.model_registry import model_registry


def register_all_models():
    """Register all available models in the registry"""
    
    # Register clustering models
    model_registry.register_model(
        'kmeans',
        KMeansModel,
        {
            'category': 'clustering',
            'task_type': 'unsupervised',
            'data_type': 'numerical',
            'description': 'K-Means clustering with SVD optimization',
            'parameters': ['n_clusters'],
            'supports_sparse': True
        }
    )
    
    # Register classification models
    model_registry.register_model(
        'knn',
        KNNModel,
        {
            'category': 'classification',
            'task_type': 'supervised',
            'data_type': 'numerical',
            'description': 'K-Nearest Neighbors classifier',
            'parameters': ['n_neighbors'],
            'supports_sparse': True
        }
    )
    
    model_registry.register_model(
        'decision_tree',
        DecisionTreeModel,
        {
            'category': 'classification',
            'task_type': 'supervised',
            'data_type': 'mixed',
            'description': 'Decision Tree classifier',
            'parameters': ['random_state'],
            'supports_sparse': False,
            'has_feature_importance': True
        }
    )
    
    model_registry.register_model(
        'naive_bayes',
        NaiveBayesModel,
        {
            'category': 'classification',
            'task_type': 'supervised',
            'data_type': 'mixed',
            'description': 'Naive Bayes classifier with automatic type selection',
            'parameters': [],
            'supports_sparse': True,
            'has_feature_importance': False
        }
    )
    
    model_registry.register_model(
        'svm',
        SVMModel,
        {
            'category': 'classification',
            'task_type': 'supervised',
            'data_type': 'numerical',
            'description': 'Support Vector Machine classifier',
            'parameters': ['C', 'kernel', 'gamma'],
            'supports_sparse': True,
            'has_feature_importance': True,  # Only for linear kernel
            'supports_probability': False  # Set to True if probability=True
        }
    )
    
    print("✅ All models registered successfully!")
    print(f"📊 Available models: {model_registry.list_models()}")
    
    # Print model categories
    categories = model_registry.get_model_categories()
    print("\n📁 Model Categories:")
    for category, models in categories.items():
        print(f"  {category}: {', '.join(models)}")


if __name__ == "__main__":
    register_all_models()
