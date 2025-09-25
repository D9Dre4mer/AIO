"""
Script to register all models in the registry
"""

from .clustering.kmeans_model import KMeansModel
from .classification.knn_model import KNNModel
from .classification.decision_tree_model import DecisionTreeModel
from .classification.naive_bayes_model import NaiveBayesModel
from .classification.svm_model import SVMModel
from .classification.logistic_regression_model import LogisticRegressionModel
from .classification.linear_svc_model import LinearSVCModel

# Enhanced ML Models
from .classification.random_forest_model import RandomForestModel
from .classification.adaboost_model import AdaBoostModel
from .classification.gradient_boosting_model import GradientBoostingModel
from .classification.xgboost_model import XGBoostModel
from .classification.lightgbm_model import LightGBMModel
from .classification.catboost_model import CatBoostModel

# Import ModelRegistry
from .utils.model_registry import ModelRegistry

# Import ensemble models
from .ensemble.stacking_classifier import EnsembleStackingClassifier
from .ensemble.ensemble_manager import EnsembleManager

# Create global registry instance
registry = ModelRegistry()

def register_all_models(registry):
    """Register all available models in the registry"""
    
    # Register clustering models
    registry.register_model(
        'kmeans',
        KMeansModel,
        {
            'category': 'clustering',
            'task_type': 'unsupervised',
            'data_type': 'numerical',
            'description': 'K-Means clustering with optimal K detection and SVD optimization',
            'parameters': ['n_clusters', 'use_optimal_k'],
            'supports_sparse': True,
            'has_optimal_k_detection': True,
            'has_optimization_plots': True
        }
    )
    
    # Register classification models
    registry.register_model(
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
    
    registry.register_model(
        'decision_tree',
        DecisionTreeModel,
        {
            'category': 'classification',
            'task_type': 'supervised',
            'data_type': 'mixed',
            'description': 'Decision Tree classifier with advanced pruning techniques',
            'parameters': ['random_state', 'pruning_method', 'cv_folds'],
            'supports_sparse': False,
            'has_feature_importance': True,
            'has_pruning': True,
            'pruning_methods': ['ccp', 'rep', 'mdl', 'cv_optimization']
        }
    )
    
    registry.register_model(
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
    
    registry.register_model(
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
    
    registry.register_model(
        'logistic_regression',
        LogisticRegressionModel,
        {
            'category': 'classification',
            'task_type': 'supervised',
            'data_type': 'numerical',
            'description': 'Logistic Regression classifier with multinomial support',
            'parameters': ['C', 'max_iter', 'multi_class', 'solver'],
            'supports_sparse': True,
            'has_feature_importance': True,
            'supports_probability': True
        }
    )
    
    registry.register_model(
        'linear_svc',
        LinearSVCModel,
        {
            'category': 'classification',
            'task_type': 'supervised',
            'data_type': 'numerical',
            'description': 'Linear Support Vector Classification',
            'parameters': ['C', 'loss', 'max_iter', 'dual'],
            'supports_sparse': True,
            'has_feature_importance': True,
            'supports_probability': False
        }
    )
    
    # Enhanced ML Models Registration
    
    registry.register_model(
        'random_forest',
        RandomForestModel,
        {
            'category': 'classification',
            'task_type': 'supervised',
            'data_type': 'mixed',
            'description': 'Random Forest classifier with ensemble learning',
            'parameters': ['n_estimators', 'max_depth', 'max_features', 'min_samples_split', 'min_samples_leaf'],
            'supports_sparse': True,
            'has_feature_importance': True,
            'supports_probability': True,
            'supports_gpu': False,
            'supports_oob_score': True
        }
    )
    
    registry.register_model(
        'adaboost',
        AdaBoostModel,
        {
            'category': 'classification',
            'task_type': 'supervised',
            'data_type': 'mixed',
            'description': 'AdaBoost classifier with adaptive boosting',
            'parameters': ['n_estimators', 'learning_rate', 'algorithm'],
            'supports_sparse': True,
            'has_feature_importance': True,
            'supports_probability': True,
            'supports_gpu': False
        }
    )
    
    registry.register_model(
        'gradient_boosting',
        GradientBoostingModel,
        {
            'category': 'classification',
            'task_type': 'supervised',
            'data_type': 'mixed',
            'description': 'Gradient Boosting classifier with sequential learning',
            'parameters': ['n_estimators', 'learning_rate', 'max_depth', 'subsample'],
            'supports_sparse': True,
            'has_feature_importance': True,
            'supports_probability': True,
            'supports_gpu': False
        }
    )
    
    registry.register_model(
        'xgboost',
        XGBoostModel,
        {
            'category': 'classification',
            'task_type': 'supervised',
            'data_type': 'mixed',
            'description': 'XGBoost classifier with GPU acceleration support',
            'parameters': ['n_estimators', 'max_depth', 'eta', 'subsample', 'colsample_bytree', 'min_child_weight'],
            'supports_sparse': True,
            'has_feature_importance': True,
            'supports_probability': True,
            'supports_gpu': True,
            'gpu_params': ['tree_method', 'predictor']
        }
    )
    
    registry.register_model(
        'lightgbm',
        LightGBMModel,
        {
            'category': 'classification',
            'task_type': 'supervised',
            'data_type': 'mixed',
            'description': 'LightGBM classifier with GPU acceleration support',
            'parameters': ['n_estimators', 'num_leaves', 'learning_rate', 'feature_fraction', 'bagging_fraction'],
            'supports_sparse': True,
            'has_feature_importance': True,
            'supports_probability': True,
            'supports_gpu': True,
            'gpu_params': ['device_type']
        }
    )
    
    registry.register_model(
        'catboost',
        CatBoostModel,
        {
            'category': 'classification',
            'task_type': 'supervised',
            'data_type': 'mixed',
            'description': 'CatBoost classifier with GPU acceleration support',
            'parameters': ['iterations', 'depth', 'learning_rate', 'l2_leaf_reg', 'border_count'],
            'supports_sparse': True,
            'has_feature_importance': True,
            'supports_probability': True,
            'supports_gpu': True,
            'gpu_params': ['task_type', 'devices']
        }
    )
    
    # Register ensemble models
    registry.register_model(
        'voting_ensemble_hard',
        EnsembleStackingClassifier,
        {
            'category': 'ensemble',
            'task_type': 'supervised',
            'data_type': 'mixed',
            'description': 'Hard Voting Ensemble classifier',
            'parameters': ['base_models', 'final_estimator'],
            'supports_sparse': True,
            'has_feature_importance': False,
            'supports_probability': False,
            'ensemble_type': 'voting',
            'voting_type': 'hard'
        }
    )
    
    registry.register_model(
        'voting_ensemble_soft',
        EnsembleStackingClassifier,
        {
            'category': 'ensemble',
            'task_type': 'supervised',
            'data_type': 'mixed',
            'description': 'Soft Voting Ensemble classifier',
            'parameters': ['base_models', 'final_estimator'],
            'supports_sparse': True,
            'has_feature_importance': False,
            'supports_probability': True,
            'ensemble_type': 'voting',
            'voting_type': 'soft'
        }
    )
    
    registry.register_model(
        'stacking_ensemble_logistic_regression',
        EnsembleStackingClassifier,
        {
            'category': 'ensemble',
            'task_type': 'supervised',
            'data_type': 'mixed',
            'description': 'Stacking Ensemble with Logistic Regression final estimator',
            'parameters': ['base_models', 'final_estimator', 'cv_folds'],
            'supports_sparse': True,
            'has_feature_importance': True,
            'supports_probability': True,
            'ensemble_type': 'stacking',
            'final_estimator': 'logistic_regression'
        }
    )
    
    registry.register_model(
        'stacking_ensemble_random_forest',
        EnsembleStackingClassifier,
        {
            'category': 'ensemble',
            'task_type': 'supervised',
            'data_type': 'mixed',
            'description': 'Stacking Ensemble with Random Forest final estimator',
            'parameters': ['base_models', 'final_estimator', 'cv_folds'],
            'supports_sparse': True,
            'has_feature_importance': True,
            'supports_probability': True,
            'ensemble_type': 'stacking',
            'final_estimator': 'random_forest'
        }
    )
    
    registry.register_model(
        'stacking_ensemble_xgboost',
        EnsembleStackingClassifier,
        {
            'category': 'ensemble',
            'task_type': 'supervised',
            'data_type': 'mixed',
            'description': 'Stacking Ensemble with XGBoost final estimator',
            'parameters': ['base_models', 'final_estimator', 'cv_folds'],
            'supports_sparse': True,
            'has_feature_importance': True,
            'supports_probability': True,
            'ensemble_type': 'stacking',
            'final_estimator': 'xgboost'
        }
    )
    
    print("✅ All models registered successfully!")
    print(f"📊 Available models: {registry.list_models()}")
    
    # Print model categories
    categories = registry.get_model_categories()
    print("\n📁 Model Categories:")
    for category, models in categories.items():
        print(f"  {category}: {', '.join(models)}")


# Register all models in global registry
register_all_models(registry)

if __name__ == "__main__":
    # For standalone testing
    from .utils.model_registry import ModelRegistry
    test_registry = ModelRegistry()
    register_all_models(test_registry)
