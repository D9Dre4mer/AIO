"""
Advanced Ensemble Methods Module

This module provides comprehensive ensemble methods for improving model performance:
- Voting Classifier
- Stacking Classifier
- Blending Ensemble
- Advanced ensemble strategies
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import lightgbm as lgb
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


class EnsembleMethods:
    """
    Advanced ensemble methods for improving model performance
    
    Features:
    - Voting Classifier with multiple algorithms
    - Stacking Classifier with meta-learner
    - Blending Ensemble with holdout validation
    - Advanced ensemble strategies
    - Performance comparison
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize EnsembleMethods
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ensemble_config = config.get('ensemble', {})
        self.models = {}
        self.ensembles = {}
        self.performance_results = {}
        
    def create_base_models(self, use_gpu: bool = False) -> Dict[str, Any]:
        """
        Create base models for ensemble
        
        Args:
            use_gpu: Whether to use GPU for supported models
            
        Returns:
            Dictionary of base models
        """
        print("🔧 Creating base models for ensemble...")
        
        # LightGBM models with different configurations
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42
        }
        
        if use_gpu:
            lgb_params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })
        
        models = {
            'lgb_fast': lgb.LGBMClassifier(
                **lgb_params,
                num_leaves=31,
                learning_rate=0.1,
                n_estimators=100
            ),
            'lgb_deep': lgb.LGBMClassifier(
                **lgb_params,
                num_leaves=100,
                learning_rate=0.05,
                max_depth=10,
                n_estimators=200
            ),
            'lgb_wide': lgb.LGBMClassifier(
                **lgb_params,
                num_leaves=200,
                learning_rate=0.03,
                feature_fraction=0.8,
                n_estimators=300
            ),
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'rf_deep': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'lr': LogisticRegression(
                random_state=42,
                max_iter=1000,
                n_jobs=-1
            ),
            'svm': SVC(
                probability=True,
                random_state=42,
                kernel='rbf'
            )
        }
        
        # Add XGBoost if available
        try:
            models['xgb'] = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
        except ImportError:
            print("⚠️  XGBoost not available, skipping...")
        
        # Add CatBoost if available
        try:
            import catboost as cb
            models['catboost'] = cb.CatBoostClassifier(
                random_state=42,
                verbose=False,
                iterations=100
            )
        except ImportError:
            print("⚠️  CatBoost not available, skipping...")
        
        self.models = models
        print(f"✅ Created {len(models)} base models")
        
        return models
    
    def create_voting_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series,
                             voting_type: str = 'soft') -> VotingClassifier:
        """
        Create voting ensemble classifier
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            voting_type: 'hard' or 'soft' voting
            
        Returns:
            Trained voting classifier
        """
        print(f"🔧 Creating voting ensemble ({voting_type})...")
        
        # Create base models if not already created
        if not self.models:
            self.create_base_models()
        
        # Select best models for voting (top 5)
        model_scores = {}
        for name, model in self.models.items():
            try:
                # Quick cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=3, scoring='accuracy', n_jobs=-1
                )
                model_scores[name] = np.mean(cv_scores)
            except Exception as e:
                print(f"⚠️  Error evaluating {name}: {e}")
                model_scores[name] = 0.0
        
        # Select top models
        top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        selected_models = [(name, self.models[name]) for name, _ in top_models]
        
        print(f"   Selected models: {[name for name, _ in selected_models]}")
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=selected_models,
            voting=voting_type
        )
        
        # Train ensemble
        voting_clf.fit(X_train, y_train)
        
        # Evaluate
        val_pred = voting_clf.predict(X_val)
        val_score = accuracy_score(y_val, val_pred)
        
        print(f"✅ Voting ensemble created with validation accuracy: {val_score:.4f}")
        
        self.ensembles['voting'] = voting_clf
        return voting_clf
    
    def create_stacking_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series) -> StackingClassifier:
        """
        Create stacking ensemble classifier
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Trained stacking classifier
        """
        print("🔧 Creating stacking ensemble...")
        
        # Create base models if not already created
        if not self.models:
            self.create_base_models()
        
        # Select models for stacking (top 4)
        model_scores = {}
        for name, model in self.models.items():
            try:
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=3, scoring='accuracy', n_jobs=-1
                )
                model_scores[name] = np.mean(cv_scores)
            except Exception as e:
                print(f"⚠️  Error evaluating {name}: {e}")
                model_scores[name] = 0.0
        
        # Select top models
        top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:4]
        selected_models = [(name, self.models[name]) for name, _ in top_models]
        
        print(f"   Selected models: {[name for name, _ in selected_models]}")
        
        # Meta-learner
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        # Create stacking classifier
        stacking_clf = StackingClassifier(
            estimators=selected_models,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # Train ensemble
        stacking_clf.fit(X_train, y_train)
        
        # Evaluate
        val_pred = stacking_clf.predict(X_val)
        val_score = accuracy_score(y_val, val_pred)
        
        print(f"✅ Stacking ensemble created with validation accuracy: {val_score:.4f}")
        
        self.ensembles['stacking'] = stacking_clf
        return stacking_clf
    
    def create_blending_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[np.ndarray, Dict]:
        """
        Create blending ensemble with holdout validation
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            
        Returns:
            Tuple of (test_predictions, blending_info)
        """
        print("🔧 Creating blending ensemble...")
        
        # Create base models if not already created
        if not self.models:
            self.create_base_models()
        
        # Split training data for blending
        from sklearn.model_selection import train_test_split
        X_train_blend, X_val_blend, y_train_blend, y_val_blend = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Train individual models and get predictions
        model_predictions = {}
        model_scores = {}
        
        for name, model in self.models.items():
            try:
                print(f"   Training {name}...")
                model.fit(X_train_blend, y_train_blend)
                
                # Get predictions on validation set
                if hasattr(model, 'predict_proba'):
                    val_pred_proba = model.predict_proba(X_val_blend)[:, 1]
                else:
                    val_pred_proba = model.predict(X_val_blend)
                
                # Get predictions on test set
                if hasattr(model, 'predict_proba'):
                    test_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    test_pred_proba = model.predict(X_test)
                
                model_predictions[name] = {
                    'val': val_pred_proba,
                    'test': test_pred_proba
                }
                
                # Calculate validation score
                val_pred_binary = (val_pred_proba > 0.5).astype(int)
                val_score = accuracy_score(y_val_blend, val_pred_binary)
                model_scores[name] = val_score
                
                print(f"     {name} validation accuracy: {val_score:.4f}")
                
            except Exception as e:
                print(f"⚠️  Error training {name}: {e}")
                continue
        
        # Create blending dataset
        val_predictions = np.column_stack([
            model_predictions[name]['val'] for name in model_scores.keys()
        ])
        test_predictions = np.column_stack([
            model_predictions[name]['test'] for name in model_scores.keys()
        ])
        
        # Train meta-learner
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        meta_learner.fit(val_predictions, y_val_blend)
        
        # Get final predictions
        final_predictions = meta_learner.predict_proba(test_predictions)[:, 1]
        
        # Calculate final score
        final_pred_binary = (final_predictions > 0.5).astype(int)
        final_score = accuracy_score(y_test, final_pred_binary)
        
        print(f"✅ Blending ensemble created with test accuracy: {final_score:.4f}")
        
        # Store blending info
        blending_info = {
            'model_scores': model_scores,
            'meta_learner': meta_learner,
            'model_predictions': model_predictions,
            'final_score': final_score
        }
        
        self.ensembles['blending'] = blending_info
        return final_predictions, blending_info
    
    def create_advanced_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Create advanced ensemble with multiple strategies
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Dictionary of ensemble methods and their performance
        """
        print("🚀 Creating advanced ensemble methods...")
        
        results = {}
        
        # 1. Voting Ensemble (Soft)
        try:
            voting_soft = self.create_voting_ensemble(X_train, y_train, X_val, y_val, 'soft')
            val_pred = voting_soft.predict(X_val)
            val_score = accuracy_score(y_val, val_pred)
            results['voting_soft'] = {
                'model': voting_soft,
                'accuracy': val_score
            }
        except Exception as e:
            print(f"⚠️  Error creating voting ensemble: {e}")
        
        # 2. Voting Ensemble (Hard)
        try:
            voting_hard = self.create_voting_ensemble(X_train, y_train, X_val, y_val, 'hard')
            val_pred = voting_hard.predict(X_val)
            val_score = accuracy_score(y_val, val_pred)
            results['voting_hard'] = {
                'model': voting_hard,
                'accuracy': val_score
            }
        except Exception as e:
            print(f"⚠️  Error creating hard voting ensemble: {e}")
        
        # 3. Stacking Ensemble
        try:
            stacking = self.create_stacking_ensemble(X_train, y_train, X_val, y_val)
            val_pred = stacking.predict(X_val)
            val_score = accuracy_score(y_val, val_pred)
            results['stacking'] = {
                'model': stacking,
                'accuracy': val_score
            }
        except Exception as e:
            print(f"⚠️  Error creating stacking ensemble: {e}")
        
        # 4. Weighted Ensemble (based on individual performance)
        try:
            weighted_ensemble = self.create_weighted_ensemble(X_train, y_train, X_val, y_val)
            val_pred = weighted_ensemble.predict(X_val)
            val_score = accuracy_score(y_val, val_pred)
            results['weighted'] = {
                'model': weighted_ensemble,
                'accuracy': val_score
            }
        except Exception as e:
            print(f"⚠️  Error creating weighted ensemble: {e}")
        
        self.performance_results = results
        return results
    
    def create_weighted_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series) -> Any:
        """
        Create weighted ensemble based on individual model performance
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Weighted ensemble model
        """
        print("🔧 Creating weighted ensemble...")
        
        # Create base models if not already created
        if not self.models:
            self.create_base_models()
        
        # Train models and calculate weights
        model_weights = {}
        model_predictions = {}
        
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                val_score = accuracy_score(y_val, val_pred)
                model_weights[name] = val_score
                model_predictions[name] = model
                print(f"   {name}: {val_score:.4f}")
            except Exception as e:
                print(f"⚠️  Error training {name}: {e}")
                continue
        
        # Normalize weights
        total_weight = sum(model_weights.values())
        model_weights = {name: weight/total_weight for name, weight in model_weights.items()}
        
        print(f"   Weights: {model_weights}")
        
        # Create weighted ensemble class
        class WeightedEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
            
            def predict(self, X):
                predictions = np.zeros(X.shape[0])
                for name, model in self.models.items():
                    if name in self.weights:
                        pred = model.predict(X)
                        predictions += self.weights[name] * pred
                return (predictions > 0.5).astype(int)
            
            def predict_proba(self, X):
                probabilities = np.zeros((X.shape[0], 2))
                for name, model in self.models.items():
                    if name in self.weights:
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(X)
                        else:
                            pred = model.predict(X)
                            proba = np.column_stack([1-pred, pred])
                        probabilities += self.weights[name] * proba
                return probabilities
        
        weighted_ensemble = WeightedEnsemble(model_predictions, model_weights)
        return weighted_ensemble
    
    def compare_ensembles(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Compare performance of all ensemble methods
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            DataFrame with performance comparison
        """
        print("📊 Comparing ensemble methods...")
        
        if not self.performance_results:
            print("⚠️  No ensemble results available. Run create_advanced_ensemble first.")
            return pd.DataFrame()
        
        comparison_results = []
        
        for name, result in self.performance_results.items():
            try:
                model = result['model']
                test_pred = model.predict(X_test)
                test_accuracy = accuracy_score(y_test, test_pred)
                
                # Additional metrics
                if hasattr(model, 'predict_proba'):
                    test_proba = model.predict_proba(X_test)[:, 1]
                    test_auc = roc_auc_score(y_test, test_proba)
                    test_f1 = f1_score(y_test, test_pred)
                else:
                    test_auc = 0.0
                    test_f1 = f1_score(y_test, test_pred)
                
                comparison_results.append({
                    'Ensemble': name,
                    'Test_Accuracy': test_accuracy,
                    'Test_AUC': test_auc,
                    'Test_F1': test_f1,
                    'Val_Accuracy': result['accuracy']
                })
                
            except Exception as e:
                print(f"⚠️  Error evaluating {name}: {e}")
                continue
        
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('Test_Accuracy', ascending=False)
        
        print("✅ Ensemble comparison completed!")
        print(comparison_df.round(4))
        
        return comparison_df
    
    def get_best_ensemble(self) -> Tuple[str, Any]:
        """
        Get the best performing ensemble
        
        Returns:
            Tuple of (ensemble_name, ensemble_model)
        """
        if not self.performance_results:
            return None, None
        
        best_name = max(self.performance_results.keys(), 
                       key=lambda x: self.performance_results[x]['accuracy'])
        best_model = self.performance_results[best_name]['model']
        
        return best_name, best_model
    
    def save_ensemble_models(self, save_dir: str = "models"):
        """
        Save trained ensemble models
        
        Args:
            save_dir: Directory to save models
        """
        import joblib
        import os
        
        os.makedirs(save_dir, exist_ok=True)
        
        for name, result in self.performance_results.items():
            try:
                model = result['model']
                model_path = os.path.join(save_dir, f"ensemble_{name}.joblib")
                joblib.dump(model, model_path)
                print(f"✅ Saved {name} ensemble to {model_path}")
            except Exception as e:
                print(f"⚠️  Error saving {name}: {e}")
        
        # Save performance results
        results_path = os.path.join(save_dir, "ensemble_performance.json")
        import json
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for name, result in self.performance_results.items():
                json_results[name] = {
                    'accuracy': float(result['accuracy'])
                }
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Saved ensemble performance results to {results_path}")
