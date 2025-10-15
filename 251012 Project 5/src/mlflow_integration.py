"""
MLflow integration module for experiment tracking and model registry
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional

import git
import mlflow
import mlflow.pytorch
import mlflow.sklearn


class MLflowTracker:
    """MLflow experiment tracking wrapper"""

    def __init__(self, experiment_name: str = "ml_project"):
        self.experiment_name = experiment_name
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        # Set tracking URI from environment
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            print(f"Warning: Could not setup MLflow experiment: {e}")
            # Fallback to local file store
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run"""
        tags = tags or {}

        # Add git information
        try:
            repo = git.Repo(search_parent_directories=True)
            tags.update(
                {
                    "git_sha": repo.head.object.hexsha[:8],
                    "git_branch": repo.active_branch.name,
                    "git_url": repo.remotes.origin.url if repo.remotes else "local",
                }
            )
        except Exception:
            pass

        # Add timestamp
        tags["timestamp"] = datetime.now().isoformat()

        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, artifact_path: str, registered_model_name: Optional[str] = None):
        """Log model to MLflow"""
        if hasattr(model, "predict"):
            # Sklearn model
            mlflow.sklearn.log_model(
                model, artifact_path=artifact_path, registered_model_name=registered_model_name
            )
        else:
            # Generic model
            mlflow.log_model(
                model, artifact_path=artifact_path, registered_model_name=registered_model_name
            )

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact"""
        mlflow.log_artifact(local_path, artifact_path)

    def log_dict(self, dictionary: Dict[str, Any], artifact_path: str):
        """Log dictionary as JSON artifact"""
        mlflow.log_dict(dictionary, artifact_path)

    def end_run(self):
        """End current run"""
        mlflow.end_run()


class ModelRegistry:
    """MLflow Model Registry wrapper"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def register_model(self, run_id: str, model_path: str, stage: str = "Staging"):
        """Register model to registry"""
        try:
            model_uri = f"runs:/{run_id}/{model_path}"
            model_version = mlflow.register_model(model_uri, self.model_name)

            # Transition to stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=self.model_name, version=model_version.version, stage=stage
            )

            return model_version
        except Exception as e:
            print(f"Warning: Could not register model: {e}")
            return None

    def get_model(self, stage: str = "Production"):
        """Get model from registry"""
        try:
            model_uri = f"models:/{self.model_name}/{stage}"
            return mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            return None

    def list_models(self):
        """List all registered models"""
        try:
            client = mlflow.tracking.MlflowClient()
            return client.search_registered_models()
        except Exception as e:
            print(f"Warning: Could not list models: {e}")
            return []


def log_experiment_results(
    model_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model,
    artifacts: Optional[Dict[str, str]] = None,
):
    """Convenience function to log complete experiment results"""

    tracker = MLflowTracker()
    registry = ModelRegistry(model_name)

    with tracker.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters and metrics
        tracker.log_params(params)
        tracker.log_metrics(metrics)

        # Log model
        tracker.log_model(model, "model", registered_model_name=model_name)

        # Log artifacts
        if artifacts:
            for artifact_name, artifact_path in artifacts.items():
                tracker.log_artifact(artifact_path, artifact_name)

        # Log model info
        model_info = {
            "model_name": model_name,
            "params": params,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        tracker.log_dict(model_info, "model_info.json")

        # Register model
        run_id = mlflow.active_run().info.run_id
        registry.register_model(run_id, "model", "Staging")

    return tracker, registry
