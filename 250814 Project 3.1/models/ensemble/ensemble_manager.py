"""
Ensemble Learning Manager for Topic Modeling Project
Manages ensemble learning operations and coordinates between individual models
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
import warnings

from ..base.base_model import BaseModel

warnings.filterwarnings("ignore")


class EnsembleManager:
    """
    Manages ensemble learning operations for the Topic Modeling project
    Automatically activates when all 3 base models are selected
    """
    
    def __init__(self, 
                 base_models: List[str] = None,
                 final_estimator: str = 'logistic_regression',
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize the ensemble manager
        
        Args:
            base_models: List of base model names to use in ensemble
            final_estimator: Final estimator for stacking 
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.base_models = (base_models or 
                          ['knn', 'decision_tree', 'naive_bayes'])
        self.final_estimator = final_estimator
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Ensemble components
        self.ensemble_model = None
        self.individual_models = {}
        self.ensemble_results = {}
        self.individual_results = {}
        
        # Performance tracking
        self.training_times = {}
        self.prediction_times = {}
        self.performance_metrics = {}
        
        # Status tracking
        self.is_trained = False
        self.is_ensemble_ready = False
        
        print(f"🚀 Ensemble Manager initialized with:")
        print(f"   • Base Models: {', '.join(self.base_models)}")
        print(f"   • Final Estimator: {final_estimator}")
        print(f"   • CV Folds: {cv_folds}")
        print(f"   • Random State: {random_state}")
    
    def check_ensemble_eligibility(self, selected_models: List[str]) -> bool:
        """
        Check if ensemble learning should be activated
        
        Args:
            selected_models: List of models selected by user
            
        Returns:
            True if ensemble learning should be activated
        """
        # Check if all 3 base models are selected
        required_models = set(self.base_models)
        selected_set = set(selected_models)
        
        is_eligible = required_models.issubset(selected_set)
        
        if is_eligible:
            print(f"🎯 Ensemble Learning Eligible: All base models selected")
            print(f"   • Required: {', '.join(required_models)}")
            print(f"   • Selected: {', '.join(selected_models)}")
        else:
            print(f"ℹ️ Ensemble Learning Not Eligible: Missing base models")
            print(f"   • Required: {', '.join(required_models)}")
            print(f"   • Selected: {', '.join(selected_models)}")
            print(f"   • Missing: {', '.join(required_models - selected_set)}")
        
        return is_eligible
    
    def create_ensemble_model(self, model_instances: Dict[str, BaseModel]):
        """
        Create the ensemble model using StackingClassifier or VotingClassifier
        
        Args:
            model_instances: Dictionary of fitted base model instances
            
        Returns:
            Configured ensemble model (StackingClassifier or VotingClassifier)
        """
        # Prepare base estimators
        base_estimators = []
        print(f"🔍 Debug: Processing {len(model_instances)} model instances")
        print(f"🔍 Debug: Required base models: {self.base_models}")
        
        for model_name in self.base_models:
            if model_name in model_instances:
                # Get the underlying scikit-learn model
                base_model = model_instances[model_name]
                print(f"🔍 Debug: Processing {model_name}: type={type(base_model)}")
                
                if base_model is None:
                    print(f"❌ Error: Model '{model_name}' is None")
                    continue
                
                if hasattr(base_model, 'model'):
                    # Use the underlying scikit-learn model
                    print(f"🔍 Debug: Using base_model.model for {model_name}")
                    base_estimators.append((model_name, base_model.model))
                else:
                    # Use the model directly if it's already a scikit-learn model
                    print(f"🔍 Debug: Using base_model directly for {model_name}")
                    base_estimators.append((model_name, base_model))
            else:
                print(f"⚠️ Warning: Model '{model_name}' not found in instances")
                continue
        
        if len(base_estimators) < 2:
            raise ValueError("Need at least 2 base models for ensemble learning")
        
        # Create ensemble model based on final_estimator type
        if self.final_estimator == 'voting':
            print("🔧 Creating ensemble model with VotingClassifier...")
            
            # Create VotingClassifier with soft voting (uses predict_proba)
            try:
                self.ensemble_model = VotingClassifier(
                    estimators=base_estimators,
                    voting='soft',  # Use soft voting for better performance
                    n_jobs=-1
                )
            except Exception as e:
                print(f"⚠️ Error creating VotingClassifier: {e}")
                # Fallback to hard voting
                self.ensemble_model = VotingClassifier(
                    estimators=base_estimators,
                    voting='hard',
                    n_jobs=-1
                )
                print("⚠️ Using hard voting as fallback")
            
            print(f"✅ VotingClassifier created successfully")
            print(f"   • Base Estimators: {len(base_estimators)}")
            print(f"   • Voting Type: {'soft' if hasattr(self.ensemble_model, 'voting') and self.ensemble_model.voting == 'soft' else 'hard'}")
            
        else:
            # Use StackingClassifier for other final estimators
            print("🔧 Creating ensemble model with StackingClassifier...")
            
            # Create final estimator
            if self.final_estimator == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                final_estimator = LogisticRegression(random_state=self.random_state, max_iter=1000)
            elif self.final_estimator == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                final_estimator = RandomForestClassifier(random_state=self.random_state, n_estimators=100)
            else:
                # Default to logistic regression
                from sklearn.linear_model import LogisticRegression
                final_estimator = LogisticRegression(random_state=self.random_state, max_iter=1000)
            
            # Create StackingClassifier with version-compatible parameters
            try:
                # Try with random_state (newer scikit-learn versions)
                self.ensemble_model = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=final_estimator,
                    cv=self.cv_folds,
                    stack_method='predict_proba',
                    n_jobs=-1,
                    random_state=self.random_state
                )
            except TypeError:
                # Fallback for older scikit-learn versions without random_state
                self.ensemble_model = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=final_estimator,
                    cv=self.cv_folds,
                    stack_method='predict_proba',
                    n_jobs=-1
                )
                print(f"⚠️ Using StackingClassifier without random_state (older scikit-learn version)")
            
            print(f"✅ StackingClassifier created successfully")
            print(f"   • Base Estimators: {len(base_estimators)}")
            print(f"   • Final Estimator: {type(final_estimator).__name__}")
            print(f"   • CV Strategy: {self.cv_folds}-fold cross-validation")
        
        return self.ensemble_model
    
    def get_ensemble_weights(self) -> Dict[str, Any]:
        """
        Get learned weights/coefficients from the trained ensemble model
        
        Returns:
            Dictionary containing weight information
        """
        if not self.is_trained or self.ensemble_model is None:
            return {"error": "Ensemble model not trained yet"}
        
        try:
            # Check if it's a VotingClassifier
            if isinstance(self.ensemble_model, VotingClassifier):
                weights_info = {
                    "final_estimator_type": "VotingClassifier",
                    "base_models": self.base_models,
                    "voting_type": getattr(self.ensemble_model, 'voting', 'unknown'),
                    "weights": {}
                }
                
                # For VotingClassifier, all models have equal weight (1/n_models)
                equal_weight = 1.0 / len(self.base_models)
                for model_name in self.base_models:
                    weights_info["weights"][model_name] = {
                        "weight": equal_weight,
                        "relative_importance": equal_weight,
                        "description": "Equal voting weight"
                    }
                
                return weights_info
            
            # For StackingClassifier, extract weights from final estimator
            final_estimator = self.ensemble_model.final_estimator_
            weights_info = {
                "final_estimator_type": type(final_estimator).__name__,
                "base_models": self.base_models,
                "weights": {}
            }
            
            if hasattr(final_estimator, 'coef_'):
                # LogisticRegression - Linear coefficients
                coefficients = final_estimator.coef_[0]  # Shape: (n_features,)
                
                # Each base model contributes n_classes features (probability outputs)
                # Get n_classes from training data dynamically
                if 'training_labels' in self.ensemble_results:
                    n_classes = len(np.unique(
                        self.ensemble_results['training_labels']
                    ))
                else:
                    # This should not happen if training_labels is stored properly
                    raise ValueError("training_labels not found in ensemble_results")
                features_per_model = n_classes
                
                for i, model_name in enumerate(self.base_models):
                    start_idx = i * features_per_model
                    end_idx = start_idx + features_per_model
                    model_coeffs = coefficients[start_idx:end_idx]
                    
                    weights_info["weights"][model_name] = {
                        "coefficients": model_coeffs.tolist(),
                        "mean_abs_weight": float(np.mean(np.abs(model_coeffs))),
                        "max_abs_weight": float(np.max(np.abs(model_coeffs))),
                        "feature_range": f"{start_idx}:{end_idx}"
                    }
                
                # Overall model importance (sum of absolute coefficients)
                total_importance = sum(info["mean_abs_weight"] for info in weights_info["weights"].values())
                for model_name in weights_info["weights"]:
                    relative_importance = weights_info["weights"][model_name]["mean_abs_weight"] / total_importance
                    weights_info["weights"][model_name]["relative_importance"] = float(relative_importance)
                    
            elif hasattr(final_estimator, 'feature_importances_'):
                # RandomForest - Feature importances
                importances = final_estimator.feature_importances_
                # Get n_classes from training data dynamically
                if 'training_labels' in self.ensemble_results:
                    n_classes = len(np.unique(
                        self.ensemble_results['training_labels']
                    ))
                else:
                    # This should not happen if training_labels is stored properly
                    raise ValueError("training_labels not found in ensemble_results")
                features_per_model = n_classes
                
                for i, model_name in enumerate(self.base_models):
                    start_idx = i * features_per_model
                    end_idx = start_idx + features_per_model
                    model_importances = importances[start_idx:end_idx]
                    
                    weights_info["weights"][model_name] = {
                        "feature_importances": model_importances.tolist(),
                        "mean_importance": float(np.mean(model_importances)),
                        "total_importance": float(np.sum(model_importances)),
                        "feature_range": f"{start_idx}:{end_idx}"
                    }
            else:
                weights_info["error"] = "Final estimator doesn't have extractable weights"
                
            return weights_info
            
        except Exception as e:
            return {"error": f"Failed to extract weights: {str(e)}"}
    
    def print_ensemble_weights(self):
        """Print ensemble weights in a readable format"""
        weights_info = self.get_ensemble_weights()
        
        if "error" in weights_info:
            print(f"❌ {weights_info['error']}")
            return
        
        print(f"\n🔍 Ensemble Model Weights Analysis")
        print(f"=" * 50)
        print(f"Final Estimator: {weights_info['final_estimator_type']}")
        print(f"Base Models: {', '.join(weights_info['base_models'])}")
        print()
        
        if weights_info['final_estimator_type'] == 'VotingClassifier':
            voting_type = weights_info.get('voting_type', 'unknown')
            print(f"📊 Model Importance (Voting Classifier - {voting_type} voting):")
            for model_name, info in weights_info['weights'].items():
                weight = info['weight'] * 100
                print(f"   • {model_name}: {weight:.1f}% (equal voting weight)")
        
        elif weights_info['final_estimator_type'] == 'LogisticRegression':
            print("📊 Model Importance (Logistic Regression Coefficients):")
            for model_name, info in weights_info['weights'].items():
                rel_importance = info['relative_importance'] * 100
                print(f"   • {model_name}: {rel_importance:.1f}% (mean |coef|: {info['mean_abs_weight']:.4f})")
        
        elif weights_info['final_estimator_type'] == 'RandomForestClassifier':
            print("📊 Model Importance (Random Forest Feature Importance):")
            for model_name, info in weights_info['weights'].items():
                total_importance = info['total_importance'] * 100
                print(f"   • {model_name}: {total_importance:.1f}% (total importance: {info['total_importance']:.4f})")
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train the ensemble model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary containing training results and metrics
        """
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not created. Call create_ensemble_model first.")
        
        print("🚀 Training ensemble model...")
        start_time = time.time()
        
        try:
            # Perform cross-validation for ensemble before final training
            from sklearn.model_selection import cross_val_score
            print("📊 Performing cross-validation for ensemble...")
            cv_scores = cross_val_score(self.ensemble_model, X_train, y_train, 
                                      cv=self.cv_folds, scoring='accuracy', n_jobs=-1)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            print(f"   • CV Accuracy: {cv_mean:.3f}±{cv_std:.3f}")
            
            # Train ensemble model on full training data
            self.ensemble_model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            self.training_times['ensemble'] = training_time
            
            # Calculate training accuracy
            train_accuracy = self.ensemble_model.score(X_train, y_train)
            
            # Calculate validation accuracy if validation data provided
            val_accuracy = None
            if X_val is not None and y_val is not None:
                val_accuracy = self.ensemble_model.score(X_val, y_val)
            
            # Store results including CV metrics
            self.ensemble_results = {
                'training_accuracy': train_accuracy,
                'validation_accuracy': val_accuracy,
                'cv_mean_accuracy': cv_mean,
                'cv_std_accuracy': cv_std,
                'training_time': training_time,
                'training_labels': y_train,  # Store for weights calculation
                'is_trained': True
            }
            
            self.is_trained = True
            
            print(f"✅ Ensemble model training completed")
            print(f"   • Training Accuracy: {train_accuracy:.4f}")
            if val_accuracy is not None:
                print(f"   • Validation Accuracy: {val_accuracy:.4f}")
            print(f"   • Training Time: {training_time:.2f}s")
            
            # Display learned weights
            self.print_ensemble_weights()
            
            return self.ensemble_results
            
        except Exception as e:
            training_time = time.time() - start_time
            print(f"❌ Ensemble training failed: {e}")
            self.ensemble_results = {
                'error': str(e),
                'training_time': training_time,
                'is_trained': False
            }
            return self.ensemble_results
    
    def predict_ensemble(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the ensemble model
        
        Args:
            X: Features to predict on
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Ensemble model not trained. Call train_ensemble first.")
        
        start_time = time.time()
        
        try:
            # Make predictions
            predictions = self.ensemble_model.predict(X)
            probabilities = self.ensemble_model.predict_proba(X)
            
            prediction_time = time.time() - start_time
            self.prediction_times['ensemble'] = prediction_time
            
            return predictions, probabilities
            
        except Exception as e:
            print(f"❌ Ensemble prediction failed: {e}")
            raise
    
    def evaluate_ensemble(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate ensemble model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Ensemble model not trained. Call train_ensemble first.")
        
        print("📊 Evaluating ensemble model performance...")
        
        try:
            # Make predictions
            y_pred, y_proba = self.predict_ensemble(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Generate classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Store performance metrics
            self.performance_metrics['ensemble'] = {
                'accuracy': accuracy,
                'classification_report': report,
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
            print(f"✅ Ensemble evaluation completed")
            print(f"   • Test Accuracy: {accuracy:.4f}")
            print(f"   • Prediction Time: {self.prediction_times.get('ensemble', 0):.4f}s")
            
            return self.performance_metrics['ensemble']
            
        except Exception as e:
            print(f"❌ Ensemble evaluation failed: {e}")
            return {'error': str(e)}
    
    def compare_performance(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare ensemble performance with individual models
        
        Args:
            individual_results: Results from individual model training
            
        Returns:
            Dictionary containing performance comparison
        """
        if not self.is_trained:
            print("⚠️ Ensemble not trained, cannot compare performance")
            return {}
        
        print("📈 Comparing ensemble vs individual model performance...")
        
        comparison = {
            'ensemble_accuracy': self.performance_metrics.get('ensemble', {}).get('accuracy', 0),
            'individual_accuracies': {},
            'improvement': {},
            'training_time_comparison': {}
        }
        
        # Compare accuracies
        for model_name, results in individual_results.items():
            if 'accuracy' in results:
                individual_acc = results['accuracy']
                comparison['individual_accuracies'][model_name] = individual_acc
                
                # Calculate improvement
                ensemble_acc = comparison['ensemble_accuracy']
                improvement = ensemble_acc - individual_acc
                comparison['improvement'][model_name] = improvement
                
                # Training time comparison
                ensemble_time = self.training_times.get('ensemble', 0)
                individual_time = results.get('training_time', 0)
                comparison['training_time_comparison'][model_name] = {
                    'ensemble_time': ensemble_time,
                    'individual_time': individual_time,
                    'ratio': ensemble_time / individual_time if individual_time > 0 else float('inf')
                }
        
        # Print comparison summary
        print(f"📊 Performance Comparison Summary:")
        print(f"   • Ensemble Accuracy: {comparison['ensemble_accuracy']:.4f}")
        for model_name, acc in comparison['individual_accuracies'].items():
            improvement = comparison['improvement'][model_name]
            print(f"   • {model_name}: {acc:.4f} (Improvement: {improvement:+.4f})")
        
        return comparison
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """
        Get information about the ensemble system
        
        Returns:
            Dictionary containing ensemble information
        """
        return {
            'base_models': self.base_models,
            'final_estimator': self.final_estimator,
            'cv_folds': self.cv_folds,
            'is_trained': self.is_trained,
            'is_ensemble_ready': self.is_ensemble_ready,
            'training_times': self.training_times,
            'prediction_times': self.prediction_times,
            'performance_metrics': self.performance_metrics
        }
    
    def reset_ensemble(self):
        """Reset ensemble system to initial state"""
        self.ensemble_model = None
        self.ensemble_results = {}
        self.individual_results = {}
        self.performance_metrics = {}
        self.training_times = {}
        self.prediction_times = {}
        self.is_trained = False
        self.is_ensemble_ready = False
        
        print("🔄 Ensemble system reset to initial state")
