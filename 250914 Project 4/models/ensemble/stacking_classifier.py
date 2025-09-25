"""
Ensemble Stacking Classifier for Topic Modeling Project
Implements StackingClassifier with KNN + Decision Tree + Naive Bayes
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")


class EnsembleStackingClassifier:
    """
    Ensemble Classifier implementation
    Supports both StackingClassifier and VotingClassifier
    Combines KNN, Decision Tree, and Naive Bayes models
    """
    
    def __init__(self, 
                 base_models: List[str] = None,
                 final_estimator: str = 'logistic_regression',
                 cv_folds: int = 5,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize the stacking classifier
        
        Args:
            base_models: List of base model names
            final_estimator: Final estimator type
            cv_folds: Cross-validation folds
            random_state: Random seed
        """
        self.base_models = base_models or ['knn', 'decision_tree', 'naive_bayes']
        self.final_estimator = final_estimator
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Initialize components
        self.stacking_classifier = None
        self.base_estimators = []
        self.final_estimator_instance = None
        
        # Performance tracking
        self.training_time = 0
        self.prediction_time = 0
        self.is_fitted = False
        
        print(f"🔧 Ensemble Stacking Classifier initialized")
        print(f"   • Base Models: {', '.join(self.base_models)}")
        print(f"   • Final Estimator: {final_estimator}")
        print(f"   • CV Folds: {cv_folds}")
        
        # Set sklearn compatibility attributes
        self._estimator_type = "classifier"
        self.classes_ = None
        self.n_features_in_ = None
    
    def create_final_estimator(self) -> Any:
        """
        Create the final estimator for stacking
        
        Returns:
            Configured final estimator
        """
        if self.final_estimator == 'logistic_regression':
            return LogisticRegression(
                random_state=self.random_state, 
                max_iter=1000
            )
        elif self.final_estimator == 'random_forest':
            return RandomForestClassifier(
                random_state=self.random_state, 
                n_estimators=100
            )
        else:
            # Default to logistic regression
            return LogisticRegression(
                random_state=self.random_state, 
                max_iter=1000
            )
    
    def create_ensemble_classifier(self, 
                                 base_estimators: List[Tuple[str, Any]]):
        """
        Create the ensemble classifier (StackingClassifier or VotingClassifier)
        
        Args:
            base_estimators: List of (name, estimator) tuples
            
        Returns:
            Configured ensemble classifier
        """
        if self.final_estimator == 'voting':
            print("🔧 Creating VotingClassifier...")
            
            # Create VotingClassifier with soft voting (uses predict_proba)
            try:
                self.stacking_classifier = VotingClassifier(
                    estimators=base_estimators,
                    voting='soft',  # Use soft voting for better performance
                    n_jobs=1  # Use single job to avoid serialization issues
                )
            except Exception as e:
                print(f"⚠️ Error creating VotingClassifier: {e}")
                # Fallback to hard voting
                self.stacking_classifier = VotingClassifier(
                    estimators=base_estimators,
                    voting='hard',
                    n_jobs=1  # Use single job to avoid serialization issues
                )
                print("⚠️ Using hard voting as fallback")
            
            print(f"✅ VotingClassifier created successfully")
            print(f"   • Base Estimators: {len(base_estimators)}")
            print(f"   • Voting Type: {'soft' if hasattr(self.stacking_classifier, 'voting') and self.stacking_classifier.voting == 'soft' else 'hard'}")
            
        else:
            print("🔧 Creating StackingClassifier...")
            
            # Create final estimator
            self.final_estimator_instance = self.create_final_estimator()
            
            # Create StackingClassifier with version-compatible parameters
            try:
                # Try with random_state (newer scikit-learn versions)
                self.stacking_classifier = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=self.final_estimator_instance,
                    cv=self.cv_folds,
                    stack_method='predict_proba',
                    n_jobs=1,  # Use single job to avoid serialization issues
                    random_state=self.random_state
                )
            except TypeError:
                # Fallback for older scikit-learn versions without random_state
                self.stacking_classifier = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=self.final_estimator_instance,
                    cv=self.cv_folds,
                    stack_method='predict_proba',
                    n_jobs=1  # Use single job to avoid serialization issues
                )
                print(f"⚠️ Using StackingClassifier without random_state (older scikit-learn version)")
            
            print(f"✅ StackingClassifier created successfully")
            print(f"   • Base Estimators: {len(base_estimators)}")
            print(f"   • Final Estimator: {type(self.final_estimator_instance).__name__}")
        
        return self.stacking_classifier
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleStackingClassifier':
        """
        Fit the stacking classifier
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self for method chaining
        """
        if self.stacking_classifier is None:
            raise ValueError("Stacking classifier not created. Call create_stacking_classifier first.")
        
        print("🚀 Training stacking classifier...")
        
        try:
            # Fit the stacking classifier
            self.stacking_classifier.fit(X, y)
            self.is_fitted = True
            
            print("✅ Stacking classifier training completed")
            return self
            
        except Exception as e:
            print(f"❌ Stacking classifier training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.stacking_classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Features to predict on
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.stacking_classifier.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Accuracy score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        return self.stacking_classifier.score(X, y)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from final estimator
        
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_fitted:
            return {}
        
        if hasattr(self.final_estimator_instance, 'feature_importances_'):
            # Random Forest
            return {
                f'feature_{i}': importance 
                for i, importance in enumerate(self.final_estimator_instance.feature_importances_)
            }
        elif hasattr(self.final_estimator_instance, 'coef_'):
            # Logistic Regression
            return {
                f'feature_{i}': abs(coef) 
                for i, coef in enumerate(self.final_estimator_instance.coef_[0])
            }
        else:
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model
        
        Returns:
            Dictionary containing model information
        """
        return {
            'base_models': self.base_models,
            'final_estimator': self.final_estimator,
            'cv_folds': self.cv_folds,
            'is_fitted': self.is_fitted,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }
    
    def reset(self):
        """Reset the model to initial state"""
        self.stacking_classifier = None
        self.base_estimators = []
        self.final_estimator_instance = None
        self.training_time = 0
        self.prediction_time = 0
        self.is_fitted = False
        
        print("🔄 Ensemble Stacking Classifier reset")
