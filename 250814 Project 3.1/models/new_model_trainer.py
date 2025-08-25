"""
New Model Trainer using modular architecture with validation and cross-validation support
"""

from typing import Dict, Any, Union, Tuple, List
import numpy as np
from scipy import sparse

# Will receive instances via constructor
from .base.metrics import ModelMetrics


class NewModelTrainer:
    """New model trainer using modular architecture with validation and cross-validation"""
    
    def __init__(self, test_size: float = 0.2, validation_size: float = 0.2, 
                 cv_folds: int = 5, cv_stratified: bool = True,
                 model_factory=None, validation_manager=None):
        """Initialize new model trainer with validation and cross-validation support"""
        # Models are now registered in models/__init__.py
        
        # Set validation parameters
        self.test_size = test_size
        self.validation_size = validation_size
        
        # Set cross-validation parameters
        self.cv_folds = cv_folds
        self.cv_stratified = cv_stratified
        
        # Store instances
        self.model_factory = model_factory
        self.validation_manager = validation_manager
        
        # Update validation manager with new CV parameters if provided
        if self.validation_manager:
            self.validation_manager.set_cv_parameters(cv_folds, cv_stratified)
    
    def cross_validate_with_precomputed_embeddings(
        self,
        model_name: str,
        cv_embeddings: Dict[str, Any],
        metrics: List[str] = None,
        **model_params
    ) -> Dict[str, Any]:
        """Perform cross-validation using pre-computed embeddings for fair comparison"""
        
        # Create model instance
        if not self.model_factory:
            raise ValueError("Model factory not set. Please provide model_factory in constructor.")
        model = self.model_factory.create_model(model_name, **model_params)
        
        # Perform cross-validation
        if not self.validation_manager:
            raise ValueError("Validation manager not set. Please provide validation_manager in constructor.")
        cv_results = self.validation_manager.cross_validate_with_precomputed_embeddings(
            model, cv_embeddings, metrics
        )
        
        # Print summary
        self.validation_manager.print_cv_summary(cv_results)
        
        # Get recommendations
        recommendations = self.validation_manager.get_cv_recommendations(cv_results)
        if recommendations:
            print("\n💡 Recommendations:")
            for rec in recommendations:
                print(f"  {rec}")
        
        return cv_results

    def cross_validate_model(
        self,
        model_name: str,
        X: Union[np.ndarray, sparse.csr_matrix],
        y: np.ndarray,
        metrics: List[str] = None,
        is_embeddings: bool = False,
        texts: List[str] = None,
        vectorizer = None,
        **model_params
    ) -> Dict[str, Any]:
        """Perform cross-validation on a specific model"""
        
        # Create model instance
        if not self.model_factory:
            raise ValueError("Model factory not set. Please provide model_factory in constructor.")
        model = self.model_factory.create_model(model_name, **model_params)
        
        # Perform cross-validation
        if not self.validation_manager:
            raise ValueError("Validation manager not set. Please provide validation_manager in constructor.")
        cv_results = self.validation_manager.cross_validate_model(
            model, X, y, metrics, is_embeddings, texts, vectorizer
        )
        
        # Print summary
        self.validation_manager.print_cv_summary(cv_results)
        
        # Get recommendations
        recommendations = self.validation_manager.get_cv_recommendations(cv_results)
        if recommendations:
            print("\n💡 Recommendations:")
            for rec in recommendations:
                print(f"  {rec}")
        
        return cv_results
    
    def cross_validate_all_models(
        self,
        X: Union[np.ndarray, sparse.csr_matrix],
        y: np.ndarray,
        metrics: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Perform cross-validation on all available models"""
        
        results = {}
        if not self.model_factory:
            raise ValueError("Model factory not set. Please provide model_factory in constructor.")
        available_models = self.model_factory.get_available_models()
        
        print(f"🔄 Cross-Validating {len(available_models)} models...")
        
        for model_name in available_models:
            try:
                print(f"\n{'='*50}")
                print(f"📊 {model_name.upper()} - Cross-Validation")
                print(f"{'='*50}")
                
                cv_result = self.cross_validate_model(model_name, X, y, metrics)
                results[model_name] = cv_result
                
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                continue
        
        return results
    
    def compare_cv_results(
        self,
        cv_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare cross-validation results across models"""
        
        comparison = {}
        
        for model_name, result in cv_results.items():
            overall = result['overall_results']
            comparison[model_name] = {
                'accuracy_mean': overall['accuracy_mean'],
                'accuracy_std': overall['accuracy_std'],
                'accuracy_range': overall['accuracy_max'] - overall['accuracy_min'],
                'stability_score': 1.0 - overall['accuracy_std']  # Higher is more stable
            }
        
        # Sort by accuracy (descending)
        sorted_by_accuracy = sorted(
            comparison.items(),
            key=lambda x: x[1]['accuracy_mean'],
            reverse=True
        )
        
        # Sort by stability (descending)
        sorted_by_stability = sorted(
            comparison.items(),
            key=lambda x: x[1]['stability_score'],
            reverse=True
        )
        
        return {
            'model_comparison': comparison,
            'sorted_by_accuracy': sorted_by_accuracy,
            'sorted_by_stability': sorted_by_stability,
            'best_accuracy_model': sorted_by_accuracy[0] if sorted_by_accuracy else None,
            'most_stable_model': sorted_by_stability[0] if sorted_by_stability else None
        }
    
    def print_cv_comparison(self, cv_results: Dict[str, Dict[str, Any]]) -> None:
        """Print comprehensive cross-validation comparison"""
        
        print("\n" + "="*60)
        print("🏆 CROSS-VALIDATION MODEL COMPARISON")
        print("="*60)
        
        comparison = self.compare_cv_results(cv_results)
        
        print("\n📊 Model Performance Ranking (by Mean Accuracy):")
        for i, (model_name, metrics) in enumerate(comparison['sorted_by_accuracy'], 1):
            acc_mean = metrics['accuracy_mean']
            acc_std = metrics['accuracy_std']
            stability = metrics['stability_score']
            
            stability_indicator = "✅" if stability > 0.9 else "⚠️" if stability > 0.8 else "❌"
            
            print(f"{i:2d}. {model_name:15s}: "
                  f"Acc: {acc_mean:.4f} ± {acc_std:.4f} "
                  f"{stability_indicator} (Stability: {stability:.3f})")
        
        print(f"\n🎯 Best Accuracy Model: {comparison['best_accuracy_model'][0]}")
        print(f"🛡️ Most Stable Model: {comparison['most_stable_model'][0]}")
        
        # Stability analysis
        print("\n📈 Stability Analysis:")
        for model_name, metrics in comparison['model_comparison'].items():
            stability = metrics['stability_score']
            if stability > 0.9:
                print(f"✅ {model_name}: Very stable ({stability:.3f})")
            elif stability > 0.8:
                print(f"⚠️  {model_name}: Moderately stable ({stability:.3f})")
            else:
                print(f"❌ {model_name}: Unstable ({stability:.3f})")
    
    # Existing methods (keep for backward compatibility)
    def train_validate_test_model(
        self,
        model_name: str,
        X: Union[np.ndarray, sparse.csr_matrix],
        y: np.ndarray,
        X_val: Union[np.ndarray, sparse.csr_matrix] = None,
        y_val: np.ndarray = None,
        X_test: Union[np.ndarray, sparse.csr_matrix] = None,
        y_test: np.ndarray = None,
        step3_data: Dict = None,
        **model_params
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, Dict[str, Any]]:
        """Train, validate and test a specific model with 3-way split"""
        
        # Use provided split data or create new split
        if X_val is not None and y_val is not None and X_test is not None and y_test is not None:
            # Use provided split data
            X_train, y_train = X, y
            print(f"📊 Using provided data split:")
            print(f"   • Train: {X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train)} | Val: {X_val.shape[0] if hasattr(X_val, 'shape') else len(X_val)} | Test: {X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)}")
        else:
            # Split data into train/test only (validation handled by CV)
            if not self.validation_manager:
                raise ValueError("Validation manager not set. Please provide validation_manager in constructor.")
            X_train, X_test, y_train, y_test = self.validation_manager.split_data_train_test(
                X, y, test_size=self.test_size, stratify=y
            )
            # Create empty validation set (CV will handle it)
            X_val, y_val = np.array([]), np.array([])
            
                    # Print split summary
        print(f"📊 Created 2-way data split (validation handled by CV):")
        print(f"   • Train: {X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train)} | Test: {X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)}")
        print(f"   • Validation: Empty (CV folds will handle validation)")
        
        # Create model instance with KNN configuration if available
        if not self.model_factory:
            raise ValueError("Model factory not set. Please provide model_factory in constructor.")
        
        # Special handling for KNN model with Step 3 configuration
        if model_name == 'knn' and step3_data and 'knn_config' in step3_data:
            knn_config = step3_data['knn_config']
            optimization_method = knn_config.get('optimization_method', 'Default K=5')
            
            if optimization_method == "Default K=5":
                # Use manual configuration
                model_params.update({
                    'n_neighbors': knn_config.get('k_value', 5),
                    'weights': knn_config.get('weights', 'uniform'),
                    'metric': knn_config.get('metric', 'cosine')
                })
                print(f"🎯 KNN Configuration: K={model_params.get('n_neighbors')}, "
                      f"Weights={model_params.get('weights')}, Metric={model_params.get('metric')}")
            elif optimization_method in ["Optimal K (Cosine Metric)", "Grid Search (All Parameters)"]:
                # Use the BEST K found from optimization in Step 3
                best_k = knn_config.get('k_value', 5)
                best_weights = knn_config.get('weights', 'uniform')
                best_metric = knn_config.get('metric', 'cosine')
                
                model_params.update({
                    'n_neighbors': best_k,
                    'weights': best_weights,
                    'metric': best_metric
                })
                
                print(f"🎯 KNN Configuration: Using OPTIMIZED parameters from Step 3:")
                print(f"   • Best K: {best_k}")
                print(f"   • Best Weights: {best_weights}")
                print(f"   • Best Metric: {best_metric}")
                print(f"   • Optimization Method: {optimization_method}")
            else:
                print(f"🎯 KNN Configuration: Will use Grid Search for all parameters")
        
        model = self.model_factory.create_model(model_name, **model_params)
        
        # Log KNN model parameters if it's KNN
        if model_name == 'knn' and hasattr(model, 'n_neighbors'):
            print(f"🎯 [KNN TRAINING] Model parameters:")
            print(f"   • K (n_neighbors): {model.n_neighbors}")
            # Check if model has sklearn attributes
            if hasattr(model, 'model') and hasattr(model.model, 'weights'):
                print(f"   • Weights: {model.model.weights}")
                print(f"   • Metric: {model.model.metric}")
                print(f"   • Algorithm: {model.model.algorithm}")
            else:
                print(f"   • Weights: N/A (custom KNNModel)")
                print(f"   • Metric: N/A (custom KNNModel)")
                print(f"   • Algorithm: N/A (custom KNNModel)")
        
        # Train model
        print(f"\n🚀 Training {model_name} model...")
        
        # Special handling for KNN model with GPU acceleration
        if model_name == 'knn':
            try:
                model.fit(X_train, y_train, use_gpu=True)
                print(f"✅ KNN model trained successfully")
            except Exception as e:
                print(f"⚠️ GPU training failed, falling back to CPU: {e}")
                model.fit(X_train, y_train, use_gpu=False)
        else:
            model.fit(X_train, y_train)
        
        # Validate model (only if validation set exists)
        if len(X_val) > 0 and len(y_val) > 0:
            print(f"🔍 Validating {model_name} model...")
            y_val_pred = model.predict(X_val)
            val_metrics = ModelMetrics.compute_classification_metrics(y_val, y_val_pred)
            val_accuracy = val_metrics['accuracy']
        else:
            print(f"🔍 Skipping validation (no validation set - CV handles it)")
            y_val_pred = np.array([])
            val_metrics = None
            val_accuracy = 0.0
        
        # Test model
        print(f"🧪 Testing {model_name} model...")
        y_test_pred = model.predict(X_test)
        test_metrics = ModelMetrics.compute_classification_metrics(y_test, y_test_pred)
        test_accuracy = test_metrics['accuracy']
        
        return y_test_pred, y_val_pred, y_test, val_accuracy, test_accuracy, test_metrics
    
    def train_validate_test_all_models(
        self,
        X: Union[np.ndarray, sparse.csr_matrix],
        y: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """Train, validate and test all available models"""
        
        results = {}
        if not self.model_factory:
            raise ValueError("Model factory not set. Please provide model_factory in constructor.")
        available_models = self.model_factory.get_available_models()
        
        print(f"🚀 Training {len(available_models)} models with validation...")
        
        for model_name in available_models:
            try:
                print(f"\n{'='*50}")
                print(f"📊 {model_name.upper()}")
                print(f"{'='*50}")
                
                y_test_pred, y_val, y_test, val_acc, test_acc, test_metrics = \
                    self.train_validate_test_model(model_name, X, y)
                
                results[model_name] = {
                    'predictions': y_test_pred,
                    'validation_accuracy': val_acc,
                    'test_accuracy': test_acc,
                    'test_metrics': test_metrics,
                    'validation_predictions': y_val,
                    'test_ground_truth': y_test
                }
                
                print(f"✅ {model_name}: Validation Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
                
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                continue
        
        return results
    
    def get_model_comparison_with_validation(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare performance of all models including validation"""
        
        # Extract accuracies
        model_comparison = {}
        for model_name, result in results.items():
            model_comparison[model_name] = {
                'validation_accuracy': result['validation_accuracy'],
                'test_accuracy': result['test_accuracy'],
                'overfitting_score': result['validation_accuracy'] - result['test_accuracy']
            }
        
        # Sort by test accuracy
        sorted_by_test = sorted(
            model_comparison.items(),
            key=lambda x: x[1]['test_accuracy'],
            reverse=True
        )
        
        # Sort by validation accuracy
        sorted_by_val = sorted(
            model_comparison.items(),
            key=lambda x: x[1]['validation_accuracy'],
            reverse=True
        )
        
        return {
            'model_comparison': model_comparison,
            'sorted_by_test': sorted_by_test,
            'sorted_by_validation': sorted_by_val,
            'best_test_model': sorted_by_test[0] if sorted_by_test else None,
            'best_validation_model': sorted_by_val[0] if sorted_by_val else None
        }
    
    def print_validation_summary(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Print comprehensive validation summary"""
        
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE VALIDATION SUMMARY")
        print("="*60)
        
        comparison = self.get_model_comparison_with_validation(results)
        
        print("\n🏆 Model Performance Ranking (by Test Accuracy):")
        for i, (model_name, metrics) in enumerate(comparison['sorted_by_test'], 1):
            overfitting = metrics['overfitting_score']
            overfitting_indicator = "⚠️" if overfitting > 0.05 else "✅"
            print(f"{i:2d}. {model_name:15s}: "
                  f"Val: {metrics['validation_accuracy']:.4f}, "
                  f"Test: {metrics['test_accuracy']:.4f} "
                  f"{overfitting_indicator} (Overfitting: {overfitting:+.4f})")
        
        print(f"\n🎯 Best Test Model: {comparison['best_test_model'][0]}")
        print(f"🔍 Best Validation Model: {comparison['best_validation_model'][0]}")
        
        # Overfitting analysis
        print("\n📈 Overfitting Analysis:")
        for model_name, metrics in comparison['model_comparison'].items():
            overfitting = metrics['overfitting_score']
            if overfitting > 0.05:
                print(f"⚠️  {model_name}: High overfitting ({overfitting:+.4f})")
            elif overfitting < -0.05:
                print(f"📉 {model_name}: Underfitting ({overfitting:+.4f})")
            else:
                print(f"✅ {model_name}: Good generalization ({overfitting:+.4f})")
    
    # Keep existing methods for backward compatibility
    def train_and_test_model(self, *args, **kwargs):
        """Backward compatibility method - returns (y_pred, accuracy, report)"""
        y_test_pred, y_val, y_test, val_acc, test_acc, test_metrics = \
            self.train_validate_test_model(*args, **kwargs)
        return y_test_pred, test_acc, test_metrics
    
    def train_and_test_all_models(self, *args, **kwargs):
        """Backward compatibility method"""
        return self.train_validate_test_all_models(*args, **kwargs)
    
    def get_model_comparison(self, *args, **kwargs):
        """Backward compatibility method"""
        return self.get_model_comparison_with_validation(*args, **kwargs)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if not self.model_factory:
            raise ValueError("Model factory not set. Please provide model_factory in constructor.")
        return self.model_factory.get_model_info(model_name)
    
    def list_available_models(self) -> list:
        """List all available models"""
        if not self.model_factory:
            raise ValueError("Model factory not set. Please provide model_factory in constructor.")
        return self.model_factory.get_available_models()
    
    def suggest_models_for_task(
        self, 
        task_type: str = None, 
        data_type: str = None
    ) -> list:
        """Suggest models for specific task and data type"""
        if not self.model_factory:
            raise ValueError("Model factory not set. Please provide model_factory in constructor.")
        return self.model_factory.suggest_models(task_type, data_type)
