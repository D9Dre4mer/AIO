"""
Optuna integration with MLflow for hyperparameter optimization
"""

import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import mlflow
import mlflow.sklearn
import optuna

logger = logging.getLogger(__name__)


class OptunaMLflowIntegration:
    """Optuna integration with MLflow for nested runs"""

    def __init__(self, study_name: str, experiment_name: str = "optuna_study"):
        self.study_name = study_name
        self.experiment_name = experiment_name
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)

        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"Could not setup MLflow experiment: {e}")
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment(self.experiment_name)

    def create_study(
        self, direction: str = "maximize", sampler: Optional[optuna.samplers.BaseSampler] = None
    ):
        """Create Optuna study"""
        storage_url = f"sqlite:///optuna_studies/{self.study_name}.db"

        study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            sampler=sampler or optuna.samplers.TPESampler(),
            storage=storage_url,
            load_if_exists=True,
        )

        return study

    def optimize_with_mlflow(
        self,
        objective_func: Callable,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        direction: str = "maximize",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
    ):
        """Run Optuna optimization with MLflow logging"""

        study = self.create_study(direction=direction, sampler=sampler)

        def mlflow_objective(trial):
            """Wrapper for objective function with MLflow logging"""

            # Start nested MLflow run
            with mlflow.start_run(nested=True):
                # Log trial parameters
                trial_params = {}
                for param_name, param_value in trial.params.items():
                    trial_params[f"trial_{param_name}"] = param_value

                mlflow.log_params(trial_params)
                mlflow.log_param("trial_number", trial.number)

                # Run objective function
                try:
                    result = objective_func(trial)

                    # Log result as metric
                    if isinstance(result, dict):
                        mlflow.log_metrics(result)
                        # Use first metric as objective value
                        objective_value = list(result.values())[0]
                    else:
                        mlflow.log_metric("objective_value", result)
                        objective_value = result

                    # Log trial info
                    trial_info = {
                        "trial_number": trial.number,
                        "params": trial.params,
                        "value": objective_value,
                        "state": trial.state.name,
                        "timestamp": datetime.now().isoformat(),
                    }
                    mlflow.log_dict(trial_info, "trial_info.json")

                    return objective_value

                except Exception as e:
                    logger.error(f"Trial {trial.number} failed: {e}")
                    mlflow.log_param("error", str(e))
                    raise optuna.TrialPruned()

        # Run optimization
        study.optimize(mlflow_objective, n_trials=n_trials, timeout=timeout)

        # Log best trial to main run
        with mlflow.start_run(run_name=f"best_trial_{self.study_name}"):
            best_trial = study.best_trial

            mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
            mlflow.log_metric("best_value", best_trial.value)
            mlflow.log_param("best_trial_number", best_trial.number)

            # Log study summary
            study_summary = {
                "study_name": self.study_name,
                "n_trials": len(study.trials),
                "best_value": best_trial.value,
                "best_params": best_trial.params,
                "best_trial_number": best_trial.number,
                "direction": study.direction.name,
                "timestamp": datetime.now().isoformat(),
            }
            mlflow.log_dict(study_summary, "study_summary.json")

        return study


def create_optuna_objective(
    model_class,
    X_train,
    y_train,
    X_val,
    y_val,
    param_space: Dict[str, Any],
    metric_name: str = "accuracy",
):
    """Create Optuna objective function for model optimization"""

    def objective(trial):
        """Optuna objective function"""

        # Sample parameters from space
        params = {}
        for param_name, param_config in param_space.items():
            if param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, param_config["choices"])
            elif param_config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", 1),
                )
            elif param_config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", None),
                    log=param_config.get("log", False),
                )

        # Create and train model
        model = model_class(**params)
        model.fit(X_train, y_train)

        # Evaluate model
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_val)
            y_pred = model.predict(X_val)
        else:
            y_pred = model.predict(X_val)
            y_pred_proba = None

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        # Return primary metric
        metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

        # Log all metrics to MLflow
        mlflow.log_metrics(metrics)

        return metrics[metric_name]

    return objective


# Example parameter spaces for common models
PARAM_SPACES = {
    "random_forest": {
        "n_estimators": {"type": "int", "low": 10, "high": 200},
        "max_depth": {"type": "int", "low": 3, "high": 20},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
    },
    "xgboost": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
    },
    "lightgbm": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        "num_leaves": {"type": "int", "low": 10, "high": 100},
    },
    "knn": {
        "n_neighbors": {"type": "int", "low": 3, "high": 50},
        "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
        "metric": {"type": "categorical", "choices": ["euclidean", "manhattan", "minkowski"]},
    },
    "svm": {
        "C": {"type": "float", "low": 0.1, "high": 100, "log": True},
        "kernel": {"type": "categorical", "choices": ["linear", "poly", "rbf", "sigmoid"]},
        "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
    },
}
