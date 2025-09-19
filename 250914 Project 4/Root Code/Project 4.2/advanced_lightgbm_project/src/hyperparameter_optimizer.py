"""
Advanced Hyperparameter Optimization Module

This module provides comprehensive hyperparameter optimization using multiple techniques:
- Optuna with TPE sampler
- Bayesian optimization with Gaussian Processes
- Multi-objective optimization
- Advanced pruning strategies
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Callable
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import warnings
import time

warnings.filterwarnings('ignore')


class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization with multiple algorithms
    
    Features:
    - Optuna with TPE sampler
    - Bayesian optimization with Gaussian Processes
    - Multi-objective optimization
    - Advanced pruning strategies
    - Cross-validation with multiple metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HyperparameterOptimizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.optimization_config = config.get('optimization', {})
        self.model_config = config.get('model', {})
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        
    def create_lightgbm_params(self, trial: Optional[optuna.Trial] = None, 
                              params_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create LightGBM parameters for optimization
        
        Args:
            trial: Optuna trial object
            params_dict: Custom parameters dictionary
            
        Returns:
            Dictionary of LightGBM parameters
        """
        if params_dict:
            return params_dict
        
        # Base parameters
        params = {
            'objective': self.model_config.get('objective', 'binary'),
            'metric': self.model_config.get('metric', 'binary_logloss'),
            'boosting_type': self.model_config.get('boosting_type', 'gbdt'),
            'verbose': self.model_config.get('verbose', -1),
            'random_state': self.model_config.get('random_state', 42)
        }
        
        if trial:
            # Optuna parameter suggestions
            params.update({
                'num_leaves': trial.suggest_int('num_leaves', 10, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
                'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 20.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 20.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 20.0),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            })
            
            # Advanced parameters
            params.update({
                'force_col_wise': trial.suggest_categorical('force_col_wise', [True, False]),
                'force_row_wise': trial.suggest_categorical('force_row_wise', [True, False]),
                'max_bin': trial.suggest_int('max_bin', 200, 300),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
                'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 1.0, log=True),
                'num_threads': -1,
            })
            
            # GPU parameters if available
            if self.config.get('performance', {}).get('use_gpu', False):
                params.update({
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0,
                    'gpu_use_dp': trial.suggest_categorical('gpu_use_dp', [True, False]),
                    'gpu_max_memory_usage': trial.suggest_float('gpu_max_memory_usage', 0.5, 0.95),
                })
        
        return params
    
    def create_objective_function(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series,
                                metric: str = 'accuracy') -> Callable:
        """
        Create objective function for optimization
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            metric: Optimization metric
            
        Returns:
            Objective function
        """
        def objective(trial):
            try:
                # Get parameters
                params = self.create_lightgbm_params(trial)
                
                # Cross-validation
                cv_folds = self.optimization_config.get('cv_folds', 5)
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                                   random_state=self.model_config.get('random_state', 42))
                
                scores = []
                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_tr, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # Create datasets
                    train_data = lgb.Dataset(X_tr, label=y_tr)
                    val_data = lgb.Dataset(X_val_cv, label=y_val_cv, reference=train_data)
                    
                    # Train model
                    model = lgb.train(
                        params,
                        train_data,
                        valid_sets=[val_data],
                        num_boost_round=2000,
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=100),
                            lgb.log_evaluation(0)
                        ]
                    )
                    
                    # Make predictions
                    pred = model.predict(X_val_cv, num_iteration=model.best_iteration)
                    pred_binary = (pred > 0.5).astype(int)
                    
                    # Calculate metric
                    if metric == 'accuracy':
                        score = accuracy_score(y_val_cv, pred_binary)
                    elif metric == 'auc':
                        score = roc_auc_score(y_val_cv, pred)
                    elif metric == 'f1':
                        from sklearn.metrics import f1_score
                        score = f1_score(y_val_cv, pred_binary)
                    else:
                        score = accuracy_score(y_val_cv, pred_binary)
                    
                    scores.append(score)
                
                return np.mean(scores)
                
            except Exception as e:
                print(f"⚠️  Error in trial: {e}")
                return 0.0
        
        return objective
    
    def optimize_with_optuna(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           metric: str = 'accuracy') -> optuna.Study:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            metric: Optimization metric
            
        Returns:
            Optuna study object
        """
        print("🔍 Starting Optuna optimization...")
        
        # Create study
        direction = 'maximize' if metric in ['accuracy', 'auc', 'f1'] else 'minimize'
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(
                seed=self.model_config.get('random_state', 42),
                n_startup_trials=20,
                n_ei_candidates=24
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5,
                interval_steps=1
            )
        )
        
        # Create objective function
        objective = self.create_objective_function(X_train, y_train, X_val, y_val, metric)
        
        # Optimize
        n_trials = self.optimization_config.get('n_trials', 200)
        timeout = self.optimization_config.get('timeout', 3600)
        
        print(f"   Trials: {n_trials}, Timeout: {timeout}s, Metric: {metric}")
        
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        end_time = time.time()
        
        # Store results
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"✅ Optuna optimization completed!")
        print(f"   Best score: {self.best_score:.4f}")
        print(f"   Time taken: {end_time - start_time:.2f}s")
        print(f"   Best parameters: {self.best_params}")
        
        return study
    
    def optimize_with_bayesian(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series,
                             metric: str = 'accuracy') -> Dict[str, Any]:
        """
        Optimize hyperparameters using Bayesian optimization (scikit-optimize)
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            metric: Optimization metric
            
        Returns:
            Dictionary with optimization results
        """
        print("🔍 Starting Bayesian optimization...")
        
        # Define search space
        space = [
            Integer(10, 500, name='num_leaves'),
            Real(0.001, 0.5, prior='log-uniform', name='learning_rate'),
            Real(0.3, 1.0, name='feature_fraction'),
            Real(0.3, 1.0, name='bagging_fraction'),
            Integer(1, 10, name='bagging_freq'),
            Integer(1, 100, name='min_child_samples'),
            Real(0.001, 20.0, prior='log-uniform', name='min_child_weight'),
            Real(0.0, 20.0, name='reg_alpha'),
            Real(0.0, 20.0, name='reg_lambda'),
            Integer(3, 20, name='max_depth'),
            Real(0.0, 1.0, name='min_split_gain'),
            Real(0.5, 1.0, name='subsample'),
            Real(0.5, 1.0, name='colsample_bytree')
        ]
        
        @use_named_args(space)
        def objective(**params):
            try:
                # Add base parameters
                lgb_params = {
                    'objective': self.model_config.get('objective', 'binary'),
                    'metric': self.model_config.get('metric', 'binary_logloss'),
                    'boosting_type': self.model_config.get('boosting_type', 'gbdt'),
                    'verbose': -1,
                    'random_state': self.model_config.get('random_state', 42),
                    **params
                }
                
                # Add GPU parameters if available
                if self.config.get('performance', {}).get('use_gpu', False):
                    lgb_params.update({
                        'device': 'gpu',
                        'gpu_platform_id': 0,
                        'gpu_device_id': 0
                    })
                
                # Cross-validation
                cv_folds = self.optimization_config.get('cv_folds', 5)
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                                   random_state=self.model_config.get('random_state', 42))
                
                scores = []
                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_tr, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    train_data = lgb.Dataset(X_tr, label=y_tr)
                    val_data = lgb.Dataset(X_val_cv, label=y_val_cv, reference=train_data)
                    
                    model = lgb.train(
                        lgb_params,
                        train_data,
                        valid_sets=[val_data],
                        num_boost_round=2000,
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=100),
                            lgb.log_evaluation(0)
                        ]
                    )
                    
                    pred = model.predict(X_val_cv, num_iteration=model.best_iteration)
                    pred_binary = (pred > 0.5).astype(int)
                    
                    if metric == 'accuracy':
                        score = accuracy_score(y_val_cv, pred_binary)
                    elif metric == 'auc':
                        score = roc_auc_score(y_val_cv, pred)
                    elif metric == 'f1':
                        from sklearn.metrics import f1_score
                        score = f1_score(y_val_cv, pred_binary)
                    else:
                        score = accuracy_score(y_val_cv, pred_binary)
                    
                    scores.append(score)
                
                return -np.mean(scores)  # Minimize negative score
                
            except Exception as e:
                print(f"⚠️  Error in Bayesian optimization: {e}")
                return 1.0
        
        # Run optimization
        n_calls = self.optimization_config.get('n_trials', 200)
        print(f"   Calls: {n_calls}, Metric: {metric}")
        
        start_time = time.time()
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
        end_time = time.time()
        
        # Extract best parameters
        best_params = dict(zip([dim.name for dim in space], result.x))
        best_score = -result.fun
        
        # Store results
        self.best_params = best_params
        self.best_score = best_score
        
        print(f"✅ Bayesian optimization completed!")
        print(f"   Best score: {self.best_score:.4f}")
        print(f"   Time taken: {end_time - start_time:.2f}s")
        print(f"   Best parameters: {self.best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_time': end_time - start_time,
            'n_calls': n_calls
        }
    
    def multi_objective_optimization(self, X_train: pd.DataFrame, y_train: pd.Series,
                                   X_val: pd.DataFrame, y_val: pd.Series) -> optuna.Study:
        """
        Multi-objective optimization (accuracy and speed)
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Optuna study object
        """
        print("🔍 Starting multi-objective optimization...")
        
        def multi_objective(trial):
            try:
                params = self.create_lightgbm_params(trial)
                
                # Cross-validation
                cv_folds = self.optimization_config.get('cv_folds', 5)
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                                   random_state=self.model_config.get('random_state', 42))
                
                scores = []
                training_times = []
                
                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_tr, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    train_data = lgb.Dataset(X_tr, label=y_tr)
                    val_data = lgb.Dataset(X_val_cv, label=y_val_cv, reference=train_data)
                    
                    # Measure training time
                    start_time = time.time()
                    model = lgb.train(
                        params,
                        train_data,
                        valid_sets=[val_data],
                        num_boost_round=1000,  # Reduced for speed measurement
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=50),
                            lgb.log_evaluation(0)
                        ]
                    )
                    end_time = time.time()
                    
                    pred = model.predict(X_val_cv, num_iteration=model.best_iteration)
                    pred_binary = (pred > 0.5).astype(int)
                    score = accuracy_score(y_val_cv, pred_binary)
                    
                    scores.append(score)
                    training_times.append(end_time - start_time)
                
                accuracy = np.mean(scores)
                avg_training_time = np.mean(training_times)
                
                # Return both objectives (maximize accuracy, minimize time)
                return accuracy, -avg_training_time
                
            except Exception as e:
                print(f"⚠️  Error in multi-objective optimization: {e}")
                return 0.0, 0.0
        
        # Create multi-objective study
        study = optuna.create_study(
            directions=['maximize', 'maximize'],  # accuracy, -time
            sampler=optuna.samplers.NSGAIISampler(
                population_size=50,
                mutation_prob=0.1,
                crossover_prob=0.9
            )
        )
        
        # Optimize
        n_trials = self.optimization_config.get('n_trials', 100)
        study.optimize(multi_objective, n_trials=n_trials)
        
        print(f"✅ Multi-objective optimization completed!")
        print(f"   Number of Pareto solutions: {len(study.best_trials)}")
        
        return study
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters from optimization"""
        return self.best_params
    
    def get_best_score(self) -> float:
        """Get best score from optimization"""
        return self.best_score
    
    def plot_optimization_history(self, study: optuna.Study, save_path: Optional[str] = None):
        """
        Plot optimization history
        
        Args:
            study: Optuna study object
            save_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get trial values
            trials = study.trials
            values = [trial.value for trial in trials if trial.value is not None]
            numbers = [trial.number for trial in trials if trial.value is not None]
            
            plt.figure(figsize=(12, 6))
            
            # Plot optimization history
            plt.subplot(1, 2, 1)
            plt.plot(numbers, values, 'b-', alpha=0.6)
            plt.plot(numbers, np.maximum.accumulate(values), 'r-', linewidth=2)
            plt.xlabel('Trial Number')
            plt.ylabel('Objective Value')
            plt.title('Optimization History')
            plt.grid(True, alpha=0.3)
            
            # Plot parameter importance
            plt.subplot(1, 2, 2)
            try:
                importance = optuna.importance.get_param_importances(study)
                params = list(importance.keys())
                importances = list(importance.values())
                
                plt.barh(params, importances)
                plt.xlabel('Importance')
                plt.title('Parameter Importance')
                plt.tight_layout()
            except:
                plt.text(0.5, 0.5, 'Parameter importance\nnot available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Optimization history plot saved to {save_path}")
            else:
                plt.show()
            
        except ImportError:
            print("⚠️  Matplotlib not available for plotting")
        except Exception as e:
            print(f"⚠️  Error plotting optimization history: {e}")
    
    def save_optimization_results(self, study: optuna.Study, filepath: str):
        """
        Save optimization results to file
        
        Args:
            study: Optuna study object
            filepath: Path to save results
        """
        import json
        
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(study.trials),
            'study_name': study.study_name,
            'direction': study.direction.name if hasattr(study.direction, 'name') else str(study.direction),
            'optimization_time': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Optimization results saved to {filepath}")
