"""
Demo Script for Advanced LightGBM Optimization

This script demonstrates the key features of the optimization pipeline
with a simplified example.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def create_demo_data():
    """Create demo dataset for testing"""
    print("📊 Creating demo dataset...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with some relationship to features
    y = (X[:, 0] + X[:, 1] * 2 + X[:, 2] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=42, stratify=y_series)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"   Class distribution: {y_train.value_counts(normalize=True).round(3).to_dict()}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def demo_feature_engineering():
    """Demonstrate advanced feature engineering"""
    print("\n🔧 DEMO: Advanced Feature Engineering")
    print("-" * 50)
    
    from feature_engineering import AdvancedFeatureEngineer
    
    # Create demo data
    X_train, y_train, X_val, y_val, X_test, y_test = create_demo_data()
    
    # Create configuration
    config = {
        'feature_engineering': {
            'polynomial_degree': 2,
            'interaction_only': False,
            'include_bias': False,
            'target_encoding': False,
            'statistical_features': True,
            'feature_selection': True,
            'max_features': 20
        }
    }
    
    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer(config)
    
    # Create comprehensive features
    X_train_processed, X_val_processed, X_test_processed = feature_engineer.create_comprehensive_features(
        X_train, y_train, X_val, X_test
    )
    
    print(f"✅ Feature engineering completed!")
    print(f"   Original features: {X_train.shape[1]}")
    print(f"   Processed features: {X_train_processed.shape[1]}")
    print(f"   Feature expansion: {X_train_processed.shape[1] / X_train.shape[1]:.2f}x")
    
    return X_train_processed, y_train, X_val_processed, y_val, X_test_processed, y_test


def demo_hyperparameter_optimization():
    """Demonstrate hyperparameter optimization"""
    print("\n🎯 DEMO: Hyperparameter Optimization")
    print("-" * 50)
    
    from hyperparameter_optimizer import HyperparameterOptimizer
    
    # Create demo data
    X_train, y_train, X_val, y_val, X_test, y_test = demo_feature_engineering()
    
    # Create configuration
    config = {
        'optimization': {
            'n_trials': 20,  # Reduced for demo
            'timeout': 300,  # 5 minutes
            'cv_folds': 3,
            'direction': 'maximize'
        },
        'model': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42
        },
        'performance': {
            'use_gpu': False  # Disable GPU for demo
        }
    }
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(config)
    
    # Run optimization
    study = optimizer.optimize_with_optuna(X_train, y_train, X_val, y_val, metric='accuracy')
    
    print(f"✅ Hyperparameter optimization completed!")
    print(f"   Best score: {optimizer.get_best_score():.4f}")
    print(f"   Best parameters: {optimizer.get_best_params()}")
    
    return optimizer.get_best_params(), X_train, y_train, X_val, y_val, X_test, y_test


def demo_advanced_lightgbm():
    """Demonstrate advanced LightGBM training"""
    print("\n🚀 DEMO: Advanced LightGBM Training")
    print("-" * 50)
    
    from lightgbm_advanced import AdvancedLightGBM
    
    # Get optimized parameters and data
    best_params, X_train, y_train, X_val, y_val, X_test, y_test = demo_hyperparameter_optimization()
    
    # Create configuration
    config = {
        'model': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42
        },
        'performance': {
            'use_gpu': False  # Disable GPU for demo
        }
    }
    
    # Initialize advanced LightGBM
    lgb_model = AdvancedLightGBM(config, use_gpu=False)
    
    # Train model
    lgb_model.train_model(X_train, y_train, X_val, y_val, params=best_params)
    
    # Evaluate model
    metrics = lgb_model.evaluate_comprehensive(X_test, y_test, "Test")
    
    print(f"✅ Advanced LightGBM training completed!")
    print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Test F1-Score: {metrics['f1']:.4f}")
    print(f"   Test AUC-ROC: {metrics['auc_roc']:.4f}")
    
    return lgb_model, metrics


def demo_ensemble_methods():
    """Demonstrate ensemble methods"""
    print("\n🎭 DEMO: Ensemble Methods")
    print("-" * 50)
    
    from ensemble_methods import EnsembleMethods
    
    # Create demo data
    X_train, y_train, X_val, y_val, X_test, y_test = demo_feature_engineering()
    
    # Create configuration
    config = {
        'ensemble': {
            'voting': True,
            'stacking': True,
            'blending': True,
            'n_estimators': 50,  # Reduced for demo
            'cv_folds': 3
        },
        'performance': {
            'use_gpu': False
        }
    }
    
    # Initialize ensemble methods
    ensemble_methods = EnsembleMethods(config)
    
    # Create base models
    ensemble_methods.create_base_models(use_gpu=False)
    
    # Create voting ensemble
    voting_ensemble = ensemble_methods.create_voting_ensemble(X_train, y_train, X_val, y_val, 'soft')
    
    # Evaluate ensemble
    y_pred = voting_ensemble.predict(X_test)
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"✅ Ensemble methods completed!")
    print(f"   Voting Ensemble Accuracy: {accuracy:.4f}")
    print(f"   Voting Ensemble F1-Score: {f1:.4f}")
    
    return voting_ensemble, accuracy, f1


def demo_model_evaluation():
    """Demonstrate model evaluation"""
    print("\n📊 DEMO: Model Evaluation")
    print("-" * 50)
    
    from model_evaluator import ModelEvaluator
    
    # Create demo data
    X_train, y_train, X_val, y_val, X_test, y_test = demo_feature_engineering()
    
    # Create configuration
    config = {
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'log_loss'],
            'visualization': {
                'roc_curve': True,
                'precision_recall_curve': True,
                'confusion_matrix': True,
                'feature_importance': True
            }
        }
    }
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    
    # Create a simple model for demonstration
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate comprehensive metrics
    metrics = evaluator.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
    
    print(f"✅ Model evaluation completed!")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1-Score: {metrics['f1']:.4f}")
    print(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
    
    return metrics


def demo_model_interpretability():
    """Demonstrate model interpretability"""
    print("\n🔍 DEMO: Model Interpretability")
    print("-" * 50)
    
    from lightgbm_advanced import AdvancedLightGBM
    
    # Create demo data
    X_train, y_train, X_val, y_val, X_test, y_test = demo_feature_engineering()
    
    # Create configuration
    config = {
        'model': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42
        },
        'performance': {
            'use_gpu': False
        }
    }
    
    # Initialize and train model
    lgb_model = AdvancedLightGBM(config, use_gpu=False)
    lgb_model.train_model(X_train, y_train, X_val, y_val)
    
    # Setup SHAP explainer
    lgb_model.setup_shap_explainer(X_train, X_val)
    
    # Plot feature importance
    lgb_model.plot_feature_importance(top_n=10)
    
    print(f"✅ Model interpretability analysis completed!")
    print(f"   Feature importance calculated")
    print(f"   SHAP explainer setup")
    
    return lgb_model


def run_complete_demo():
    """Run complete demonstration"""
    print("🚀 ADVANCED LIGHTGBM OPTIMIZATION - COMPLETE DEMO")
    print("=" * 80)
    
    try:
        # 1. Feature Engineering Demo
        demo_feature_engineering()
        
        # 2. Hyperparameter Optimization Demo
        demo_hyperparameter_optimization()
        
        # 3. Advanced LightGBM Demo
        demo_advanced_lightgbm()
        
        # 4. Ensemble Methods Demo
        demo_ensemble_methods()
        
        # 5. Model Evaluation Demo
        demo_model_evaluation()
        
        # 6. Model Interpretability Demo
        demo_model_interpretability()
        
        print("\n🎉 COMPLETE DEMO FINISHED SUCCESSFULLY!")
        print("=" * 80)
        print("✅ All features demonstrated successfully!")
        print("📚 Check the generated plots and outputs")
        print("🚀 Ready to use the full pipeline!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        raise


if __name__ == "__main__":
    run_complete_demo()
