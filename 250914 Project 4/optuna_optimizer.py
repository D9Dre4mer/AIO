"""
Optuna Optimization Module for Enhanced ML Models
Provides hyperparameter optimization using Optuna
"""

import numpy as np
import logging
from typing import Dict, Any, Union
from scipy import sparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning(
        "Optuna not available. Please install with: pip install optuna"
    )


class OptunaOptimizer:
    """Optuna-based hyperparameter optimizer for ML models"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Optuna optimizer
        
        Args:
            config: Configuration dictionary with Optuna settings
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required but not installed")
        
        self.config = config or {}
        self.study = None
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        
        # Default configuration
        self.default_config = {
            'trials': 100,
            'timeout': None,
            'direction': 'maximize',
            'study_name': 'ml_optimization',
            'storage': None,
            'sampler': 'TPE',
            'pruner': 'MedianPruner'
        }
        
        # Update with provided config
        self.default_config.update(self.config)
        
    def create_study(self, study_name: str = None) -> optuna.Study:
        """Create Optuna study for optimization
        
        Args:
            study_name: Name for the study
            
        Returns:
            optuna.Study: Created study object
        """
        study_name = study_name or self.default_config['study_name']
        
        # Configure sampler
        if self.default_config['sampler'] == 'TPE':
            sampler = optuna.samplers.TPESampler(seed=42)
        elif self.default_config['sampler'] == 'Random':
            sampler = optuna.samplers.RandomSampler(seed=42)
        else:
            sampler = optuna.samplers.TPESampler(seed=42)
        
        # Configure pruner
        if self.default_config['pruner'] == 'MedianPruner':
            pruner = optuna.pruners.MedianPruner()
        elif self.default_config['pruner'] == 'PercentilePruner':
            pruner = optuna.pruners.PercentilePruner(25.0)
        else:
            pruner = optuna.pruners.MedianPruner()
        
        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            direction=self.default_config['direction'],
            sampler=sampler,
            pruner=pruner,
            storage=self.default_config['storage']
        )
        
        logger.info(f"Created Optuna study: {study_name}")
        return self.study
    
    def optimize_model(self, model_name: str,
                      X_train: Union[np.ndarray, sparse.csr_matrix],
                      y_train: np.ndarray,
                      X_val: Union[np.ndarray, sparse.csr_matrix],
                      y_val: np.ndarray,
                      model_class,
                      search_space: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific model
        
        Args:
            model_name: Name of the model to optimize
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_class: Model class to optimize
            search_space: Custom search space (optional)
            
        Returns:
            Dict containing optimization results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required but not installed")
        
        # Create study if not exists
        if self.study is None:
            self.create_study(f"{model_name}_optimization")
        
        # Get search space for the model
        if search_space is None:
            search_space = self._get_default_search_space(model_name)
        
        def objective(trial):
            """Objective function for Optuna optimization"""
            try:
                # Sample hyperparameters
                params = {}
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )
                
                # Create and train model
                model = model_class(**params)
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                y_pred = model.predict(X_val)
                
                # Calculate accuracy
                accuracy = np.mean(y_pred == y_val)
                
                return accuracy
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0  # Return worst possible score
        
        # Run optimization
        logger.info(f"Starting optimization for {model_name}...")
        
        self.study.optimize(
            objective,
            n_trials=self.default_config['trials'],
            timeout=self.default_config['timeout']
        )
        
        # Store results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        # Store optimization history
        self.optimization_history.append({
            'model_name': model_name,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.study.trials),
            'study_name': self.study.study_name
        })
        
        logger.info(f"Optimization completed for {model_name}")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.study.trials),
            'study_name': self.study.study_name,
            'optimization_history': self.optimization_history
        }
    
    def _get_default_search_space(self, model_name: str) -> Dict[str, Any]:
        """Get default search space for a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict containing search space configuration
        """
        search_spaces = {
            'random_forest': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'max_features': {
                    'type': 'categorical', 'choices': ['sqrt', 'log2', None]
                },
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10}
            },
            'adaboost': {
                'n_estimators': {'type': 'int', 'low': 20, 'high': 200},
                'learning_rate': {
                    'type': 'float', 'low': 0.01, 'high': 2.0, 'log': True
                }
            },
            'gradient_boosting': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'learning_rate': {
                    'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True
                },
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
            },
            'xgboost': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'eta': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
                'reg_lambda': {
                    'type': 'float', 'low': 0.0, 'high': 10.0, 'log': True
                },
                'reg_alpha': {
                    'type': 'float', 'low': 0.0, 'high': 10.0, 'log': True
                }
            },
            'lightgbm': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'num_leaves': {'type': 'int', 'low': 10, 'high': 100},
                'learning_rate': {
                    'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True
                },
                'feature_fraction': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'bagging_fraction': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'min_child_samples': {'type': 'int', 'low': 5, 'high': 50},
                'lambda_l1': {
                    'type': 'float', 'low': 0.0, 'high': 10.0, 'log': True
                },
                'lambda_l2': {
                    'type': 'float', 'low': 0.0, 'high': 10.0, 'log': True
                }
            },
            'catboost': {
                'iterations': {'type': 'int', 'low': 50, 'high': 500},
                'depth': {'type': 'int', 'low': 3, 'high': 10},
                'learning_rate': {
                    'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True
                },
                'l2_leaf_reg': {
                    'type': 'float', 'low': 1.0, 'high': 10.0, 'log': True
                },
                'border_count': {'type': 'int', 'low': 32, 'high': 255}
            }
        }
        
        return search_spaces.get(model_name, {})
    
    def get_optimization_results(self) -> Dict[str, Any]:
        """Get optimization results summary
        
        Returns:
            Dict containing optimization results
        """
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'study': self.study
        }
    
    def save_optimization_results(self, filepath: str):
        """Save optimization results to file
        
        Args:
            filepath: Path to save the results
        """
        import json
        
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'config': self.default_config
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization results saved to {filepath}")
    
    def load_optimization_results(self, filepath: str):
        """Load optimization results from file
        
        Args:
            filepath: Path to load the results from
        """
        import json
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.best_params = results['best_params']
        self.best_score = results['best_score']
        self.optimization_history = results['optimization_history']
        
        logger.info(f"Optimization results loaded from {filepath}")


def optimize_model_with_optuna(model_name: str, model_class,
                              X_train: Union[np.ndarray, sparse.csr_matrix],
                              y_train: np.ndarray,
                              X_val: Union[np.ndarray, sparse.csr_matrix],
                              y_val: np.ndarray,
                              config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function to optimize a single model with Optuna
    
    Args:
        model_name: Name of the model to optimize
        model_class: Model class to optimize
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Optuna configuration
        
    Returns:
        Dict containing optimization results
    """
    optimizer = OptunaOptimizer(config)
    return optimizer.optimize_model(
        model_name, X_train, y_train, X_val, y_val, model_class
    )
