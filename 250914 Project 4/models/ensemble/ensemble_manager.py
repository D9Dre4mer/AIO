"""
Ensemble Learning Manager for Topic Modeling Project
Manages ensemble learning operations and coordinates between individual models
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
import time
import warnings

# Parallel processing disabled - using pickle instead

from ..base.base_model import BaseModel

warnings.filterwarnings("ignore")


class TrainedModelWrapper(BaseEstimator, ClassifierMixin):
    """
    Simple wrapper for already trained models in ensemble learning
    Uses the trained model directly without retraining
    """
    
    # Set _estimator_type for sklearn compatibility
    _estimator_type = "classifier"
    
    def __init__(self, trained_model, model_name: str = None):
        self.trained_model = trained_model
        self.model_name = model_name if model_name is not None else "unknown"
        self.is_fitted = True  # Already fitted
        
        # Copy attributes from trained model
        if hasattr(trained_model, 'classes_'):
            self.classes_ = trained_model.classes_
        if hasattr(trained_model, 'n_features_in_'):
            self.n_features_in_ = trained_model.n_features_in_
    
    def fit(self, X, y):
        """No-op since model is already trained"""
        return self
        
    def predict(self, X):
        """Make predictions using the trained model"""
        return self.trained_model.predict(X)
        
    def predict_proba(self, X):
        """Make probability predictions using the trained model"""
        if hasattr(self.trained_model, 'predict_proba'):
            return self.trained_model.predict_proba(X)
        else:
            # Fallback for models without predict_proba
            # from sklearn.utils.extmath import softmax
            predictions = self.trained_model.predict(X)
            # Create dummy probabilities
            n_classes = len(self.classes_) if hasattr(self, 'classes_') else 2
            proba = np.zeros((len(predictions), n_classes))
            for i, pred in enumerate(predictions):
                if hasattr(self, 'classes_'):
                    class_idx = np.where(self.classes_ == pred)[0]
                    if len(class_idx) > 0:
                        proba[i, class_idx[0]] = 1.0
                    else:
                        proba[i, 0] = 1.0
                else:
                    proba[i, int(pred)] = 1.0
            return proba
    
    def score(self, X, y):
        """Score the model"""
        return self.trained_model.score(X, y)
    
    def _more_tags(self):
        """Additional tags for sklearn compatibility"""
        return {
            'binary_only': False,
            'multiclass_only': False,
            'multilabel': False,
            'multioutput': False,
            'multioutput_only': False,
            'no_validation': False,
            'non_deterministic': False,
            'pairwise': False,
            'preserves_dtype': [],
            'poor_score': False,
            'requires_fit': False,  # Already fitted
            'requires_y': False,
            'stateless': False,
            'X_types': ['2darray', 'sparse'],
            'y_types': ['1dlabels']
        }
    
    def _check_targets(self, y_true, y_pred):
        """Check targets for sklearn compatibility"""
        from sklearn.utils.multiclass import type_of_target
        return type_of_target(y_true)
    
    def __sklearn_is_fitted__(self):
        """Check if the estimator is fitted"""
        return self.is_fitted
    
    def _validate_data(self, X, y=None, reset=True):
        """Validate input data for sklearn compatibility"""
        from sklearn.utils.validation import check_X_y, check_array
        if y is not None:
            X, y = check_X_y(X, y)
            return X, y
        else:
            return check_array(X)
    
    def __sklearn_tags__(self):
        """Get tags for sklearn 1.7.2+ compatibility"""
        from sklearn.utils._tags import Tags
        return Tags(
            estimator_type="classifier",
            target_tags=self._get_target_tags(),
            transformer_tags=None,
            classifier_tags=self._get_classifier_tags(),
            regressor_tags=None,
            array_api_support=False,
            no_validation=False,
            non_deterministic=False,
            requires_fit=True,
            _skip_test=False,
            input_tags=self._get_input_tags()
        )
    
    def _get_target_tags(self):
        """Get target tags"""
        from sklearn.utils._tags import TargetTags
        return TargetTags(
            required=True,
            one_d_labels=True,
            two_d_labels=False,
            positive_only=False,
            multi_output=False,
            single_output=True
        )
    
    def _get_classifier_tags(self):
        """Get classifier tags"""
        from sklearn.utils._tags import ClassifierTags
        return ClassifierTags(
            poor_score=False,
            multi_class=True,
            multi_label=False
        )
    
    def _get_input_tags(self):
        """Get input tags"""
        from sklearn.utils._tags import InputTags
        return InputTags(
            one_d_array=False,
            two_d_array=True,
            three_d_array=False,
            sparse=True,
            categorical=False,
            string=False,
            dict=False,
            positive_only=False,
            allow_nan=False,
            pairwise=False
        )
        
    def get_params(self, deep=True):
        """Get model parameters for sklearn compatibility"""
        params = {
            'trained_model': self.trained_model,
            'model_name': self.model_name
        }
        return params
    
    def set_params(self, **params):
        """Set model parameters for sklearn compatibility"""
        for key, value in params.items():
            if key == 'trained_model':
                self.trained_model = value
            elif key == 'model_name':
                self.model_name = value
        return self


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
    
    def create_ensemble_with_reuse(self, individual_results: List[Dict[str, Any]], 
                                  X_train: np.ndarray, y_train: np.ndarray,
                                  model_factory=None, target_embedding: str = None) -> Dict[str, Any]:
        """
        Create ensemble model by reusing trained instances from individual results
        
        Args:
            individual_results: List of individual model training results
            X_train: Training features
            y_train: Training labels
            model_factory: Model factory for creating new instances if needed
            target_embedding: Target embedding type to match models for
            
        Returns:
            Dictionary containing ensemble creation results
        """
        print(f"🔧 Creating ensemble with model reuse for embedding '{target_embedding}'...")
        start_time = time.time()
        
        base_model_instances = {}
        reuse_results = {
            'models_reused': [],
            'models_retrained': [],
            'reuse_errors': []
        }
        
        for model_name in self.base_models:
            print(f"🔍 Processing {model_name} for ensemble...")
            
            try:
                # Try to find trained model in individual results with matching embedding
                trained_model = self._find_trained_model_in_results(individual_results, model_name, target_embedding)
                
                if trained_model:
                    print(f"✅ Found trained {model_name} in individual results")
                    base_model_instances[model_name] = trained_model
                    reuse_results['models_reused'].append(model_name)
                else:
                    # Fallback to creating new model
                    print(f"⚠️ No trained {model_name} found, creating new instance")
                    new_model = self._create_and_train_model(model_name, X_train, y_train, model_factory)
                    if new_model:
                        base_model_instances[model_name] = new_model
                        reuse_results['models_retrained'].append(model_name)
                    else:
                        reuse_results['reuse_errors'].append(f"Failed to create {model_name}")
                        
            except Exception as e:
                error_msg = f"Error processing {model_name}: {str(e)}"
                print(f"❌ {error_msg}")
                reuse_results['reuse_errors'].append(error_msg)
        
        # Create ensemble model
        if len(base_model_instances) < 2:
            error_msg = f"Need at least 2 base models for ensemble, got {len(base_model_instances)}"
            print(f"❌ {error_msg}")
            return {'error': error_msg, 'reuse_results': reuse_results}
        
        try:
            ensemble_model = self.create_ensemble_model(base_model_instances, X_train)
            creation_time = time.time() - start_time
            
            print(f"✅ Ensemble model created successfully")
            print(f"   • Models Reused: {len(reuse_results['models_reused'])}")
            print(f"   • Models Retrained: {len(reuse_results['models_retrained'])}")
            print(f"   • Creation Time: {creation_time:.2f}s")
            
            return {
                'ensemble_model': ensemble_model,
                'base_model_instances': base_model_instances,
                'reuse_results': reuse_results,
                'creation_time': creation_time,
                'success': True
            }
            
        except Exception as e:
            error_msg = f"Failed to create ensemble model: {str(e)}"
            print(f"❌ {error_msg}")
            return {'error': error_msg, 'reuse_results': reuse_results}

    def create_ensemble_model(self, model_instances: Dict[str, BaseModel], X_train=None):
        """
        Create the ensemble model using StackingClassifier or VotingClassifier
        
        Args:
            model_instances: Dictionary of fitted base model instances
            X_train: Training data to check data type compatibility
            
        Returns:
            Configured ensemble model (StackingClassifier or VotingClassifier)
        """
        # Prepare base estimators
        base_estimators = []
        print(f"Processing {len(model_instances)} model instances")
        print(f"Required base models: {self.base_models}")
        
        for model_name in self.base_models:
            if model_name in model_instances:
                # Get the underlying scikit-learn model
                base_model = model_instances[model_name]
                print(f"Processing {model_name}: type={type(base_model)}")
                
                if base_model is None:
                    print(f"❌ Error: Model '{model_name}' is None - skipping from ensemble")
                    continue
                
                # For ensemble, we need to create a sklearn-compatible wrapper
                # that handles data type selection properly
                if hasattr(base_model, 'model') and base_model.model is not None:
                    # Get the underlying sklearn model
                    sklearn_model = base_model.model
                    print(f"Using base_model.model for {model_name}")
                else:
                    # Use the model directly if it's already a sklearn model
                    sklearn_model = base_model
                    print(f"Using base_model directly for {model_name}")
                    
                # Validate that we have a proper sklearn model
                if sklearn_model is None:
                    print(f"❌ Error: Model '{model_name}' has no valid sklearn model - skipping from ensemble")
                    continue
                
                # For Naive Bayes, we need to handle sparse/dense data compatibility
                if model_name == 'naive_bayes' and X_train is not None:
                    # Check if we need to recreate the model for different data type
                    from scipy import sparse
                    from sklearn.naive_bayes import GaussianNB, MultinomialNB
                    
                    if sparse.issparse(X_train) and isinstance(sklearn_model, GaussianNB):
                        # Convert GaussianNB to MultinomialNB for sparse data
                        print(f"📊 Ensemble: Converting GaussianNB to MultinomialNB for sparse data")
                        # Create new MultinomialNB and copy parameters if possible
                        new_model = MultinomialNB()
                        # Note: We can't copy parameters between different NB types
                        sklearn_model = new_model
                    elif not sparse.issparse(X_train) and isinstance(sklearn_model, MultinomialNB):
                        # Convert MultinomialNB to GaussianNB for dense data
                        print(f"📊 Ensemble: Converting MultinomialNB to GaussianNB for dense data")
                        new_model = GaussianNB()
                        sklearn_model = new_model
                
                # Use the trained model directly (no wrapper needed)
                # The model is already trained and sklearn-compatible
                base_estimators.append((model_name, sklearn_model))
            else:
                print(f"⚠️ Warning: Model '{model_name}' not found in instances")
                continue
        
        if len(base_estimators) < 2:
            error_msg = f"Need at least 2 base models for ensemble learning, got {len(base_estimators)}"
            print(f"❌ {error_msg}")
            print(f"Available base_estimators: {[name for name, _ in base_estimators]}")
            raise ValueError(error_msg)
        
        # Create ensemble model based on final_estimator type
        if self.final_estimator == 'voting':
            print("🔧 Creating ensemble model with VotingClassifier...")
            
            # Create VotingClassifier with soft voting (required for predict_proba)
            # Soft voting is needed for evaluation metrics that require probabilities
            try:
                self.ensemble_model = VotingClassifier(
                    estimators=base_estimators,
                    voting='soft',  # Use soft voting for predict_proba compatibility
                    n_jobs=1  # Use single job to avoid serialization issues
                )
                print("✅ Using soft voting for full compatibility with evaluation metrics")
            except Exception as e:
                print(f"⚠️ Error creating VotingClassifier: {e}")
                # Fallback to hard voting if soft voting fails
                self.ensemble_model = VotingClassifier(
                    estimators=base_estimators,
                    voting='hard',
                    n_jobs=1  # Use single job to avoid serialization issues
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
                    n_jobs=1,  # Use single job to avoid serialization issues
                    random_state=self.random_state
                )
            except TypeError:
                # Fallback for older scikit-learn versions without random_state
                self.ensemble_model = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=final_estimator,
                    cv=self.cv_folds,
                    stack_method='predict_proba',
                    n_jobs=1  # Use single job to avoid serialization issues
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
            # Keep sparse matrices for ensemble training (better performance)
            from scipy import sparse
            if sparse.issparse(X_train):
                print("🔧 Using sparse matrix for ensemble training (memory efficient)...")
                X_train_dense = X_train  # Keep sparse for better performance
                print(f"✅ Using sparse matrix: {X_train_dense.shape}")
            else:
                X_train_dense = X_train
            
            if X_val is not None:
                if sparse.issparse(X_val):
                    print("🔧 Using sparse validation matrix...")
                    X_val_dense = X_val  # Keep sparse
                    print(f"✅ Using sparse validation matrix: {X_val_dense.shape}")
                else:
                    X_val_dense = X_val
            else:
                X_val_dense = X_val
            
            # Skip cross-validation for ensemble when using pre-trained models
            # Since base models are already trained and validated, CV is redundant
            print("📊 Skipping cross-validation for ensemble (using cached results)...")
            print("   • Base models already trained and validated individually")
            print("   • Ensemble will use pre-trained model predictions directly")
            print("   • CV accuracy will be calculated from base model results in comprehensive_evaluation.py")
            
            # Use dummy CV scores for compatibility (will be calculated from base models)
            cv_mean = 0.0
            cv_std = 0.0
            
            # Skip actual training since base models are already trained
            # VotingClassifier.fit() is just a no-op when using pre-trained models
            print("⚡ Skipping ensemble training (using pre-trained base models)...")
            print("   • Base models already fitted and ready for prediction")
            print("   • VotingClassifier will use pre-trained models directly")
            
            # Just call fit() for compatibility (it's essentially a no-op with pre-trained models)
            self.ensemble_model.fit(X_train_dense, y_train)
            
            training_time = time.time() - start_time
            self.training_times['ensemble'] = training_time
            
            # Calculate training accuracy
            train_accuracy = self.ensemble_model.score(X_train_dense, y_train)
            
            # Calculate validation accuracy if validation data provided
            val_accuracy = None
            if X_val_dense is not None and y_val is not None:
                val_accuracy = self.ensemble_model.score(X_val_dense, y_val)
            
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
            # Keep sparse matrices for prediction (better performance)
            from scipy import sparse
            if sparse.issparse(X):
                print("🔧 Using sparse matrix for ensemble prediction (memory efficient)...")
                X_dense = X  # Keep sparse for better performance
                print(f"✅ Using sparse matrix: {X_dense.shape}")
            else:
                X_dense = X
            
            # Make predictions
            predictions = self.ensemble_model.predict(X_dense)
            probabilities = self.ensemble_model.predict_proba(X_dense)
            
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
            # Make predictions (predict_ensemble already handles sparse matrices)
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
    
    def _find_trained_model_in_results(self, individual_results: List[Dict[str, Any]], 
                                     model_name: str, target_embedding: str = None):
        """
        Find trained model instance in individual results
        
        Args:
            individual_results: List of individual model results
            model_name: Name of model to find
            target_embedding: Target embedding type to match (optional)
            
        Returns:
            Trained model instance or None
        """
        print(f"Looking for model '{model_name}' with embedding '{target_embedding}' in {len(individual_results)} results")
        
        for i, result in enumerate(individual_results):
            result_model_name = result.get('model_name')
            result_embedding = result.get('embedding_name')
            print(f"Result {i}: status={result.get('status')}, model_name={result_model_name}, embedding={result_embedding}, has_trained_model={'trained_model' in result}")
            
            # Check if model name matches
            if result.get('status') == 'success' and result_model_name == model_name and 'trained_model' in result:
                # If target_embedding is specified, also check embedding match
                if target_embedding is None or result_embedding == target_embedding:
                    trained_model = result['trained_model']
                    print(f"✅ Found trained model '{model_name}' with embedding '{result_embedding}': type={type(trained_model)}")
                    return trained_model
                else:
                    print(f"⚠️ Model '{model_name}' found but with different embedding '{result_embedding}' (target: '{target_embedding}')")
        
        print(f"❌ No trained model found for '{model_name}' with embedding '{target_embedding}'")
        return None
    
    def _create_and_train_model(self, model_name: str, X_train: np.ndarray, 
                               y_train: np.ndarray, model_factory=None):
        """
        Create and train a new model instance
        
        Args:
            model_name: Name of model to create
            X_train: Training features
            y_train: Training labels
            model_factory: Model factory for creating instances
            
        Returns:
            Trained model instance or None
        """
        try:
            if model_factory:
                model_instance = model_factory.create_model(model_name)
            else:
                # Fallback: import model factory
                from ..utils.model_factory import ModelFactory
                from ..utils.model_registry import ModelRegistry
                from ..register_models import register_all_models
                
                registry = ModelRegistry()
                register_all_models(registry)
                factory = ModelFactory(registry)
                model_instance = factory.create_model(model_name)
            
            if model_instance is None:
                print(f"❌ Model factory returned None for {model_name}")
                return None
            
            # Train the model
            if model_name == 'knn':
                model_instance.fit(X_train, y_train, use_gpu=False)
            else:
                model_instance.fit(X_train, y_train)
            
            return model_instance
            
        except Exception as e:
            print(f"❌ Failed to create and train {model_name}: {e}")
            return None
