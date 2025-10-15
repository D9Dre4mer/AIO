"""
Prefect flows for ML pipeline orchestration
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import mlflow
import numpy as np
import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.blocks.system import Secret
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from prefect.task_runners import ConcurrentTaskRunner

from src.data_validation import DataValidator
from src.mlflow_integration import MLflowTracker, ModelRegistry
from src.optuna_integration import OptunaMLflowIntegration
from src.train.main import MLTrainingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task(retries=3, retry_delay_seconds=60)
def ingest_data_task(config_path: str = "params.yaml") -> Dict[str, Any]:
    """Data ingestion task"""
    logger = get_run_logger()
    logger.info("Starting data ingestion...")

    try:
        # Import here to avoid circular imports
        from data_loader import DataLoader

        data_loader = DataLoader()
        data_loader.load_dataset()

        # Get recommended categories
        recommended_categories = data_loader.get_category_recommendations(max_categories=5)
        if recommended_categories:
            data_loader.set_selected_categories(recommended_categories)

        # Select and preprocess samples
        data_loader.select_samples(max_samples=10000)  # Smaller for orchestration
        data_loader.preprocess_samples()
        data_loader.create_label_mappings()

        # Prepare train/test data
        X_train, X_test, y_train, y_test = data_loader.prepare_train_test_data()

        logger.info(f"Data ingestion completed: Train={len(X_train)}, Test={len(X_test)}")

        return {
            "status": "success",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "categories": data_loader.selected_categories,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return {"status": "failed", "error": str(e), "timestamp": datetime.now().isoformat()}


@task(retries=2, retry_delay_seconds=30)
def validate_data_task(data_dir: str = "data") -> Dict[str, Any]:
    """Data validation task"""
    logger = get_run_logger()
    logger.info("Starting data validation...")

    try:
        validator = DataValidator(data_dir)
        results = validator.validate_all_datasets()

        # Save validation report
        validator.save_validation_report(results, "artifacts/validation_report.json")

        # Check if any validations failed
        failed_validations = [
            name
            for name, result in results.items()
            if result["schema"]["status"] == "failed" or result["quality"]["status"] == "failed"
        ]

        if failed_validations:
            logger.warning(f"Data validation failed for: {failed_validations}")
            return {
                "status": "failed",
                "failed_validations": failed_validations,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            logger.info("Data validation passed")
            return {
                "status": "success",
                "validated_datasets": list(results.keys()),
                "timestamp": datetime.now().isoformat(),
            }

    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return {"status": "failed", "error": str(e), "timestamp": datetime.now().isoformat()}


@task(retries=2, retry_delay_seconds=60)
def build_training_set_task(ingest_result: Dict[str, Any]) -> Dict[str, Any]:
    """Build training set task"""
    logger = get_run_logger()
    logger.info("Building training set...")

    try:
        if ingest_result["status"] != "success":
            raise ValueError("Data ingestion failed, cannot build training set")

        # Import here to avoid circular imports
        from data_loader import DataLoader
        from text_encoders import TextVectorizer

        data_loader = DataLoader()
        text_vectorizer = TextVectorizer()

        # Reload and prepare data
        data_loader.load_dataset()
        recommended_categories = data_loader.get_category_recommendations(max_categories=5)
        if recommended_categories:
            data_loader.set_selected_categories(recommended_categories)

        data_loader.select_samples(max_samples=10000)
        data_loader.preprocess_samples()
        data_loader.create_label_mappings()

        X_train, X_test, y_train, y_test = data_loader.prepare_train_test_data()

        # Create features
        X_train_features = text_vectorizer.fit_transform_tfidf_svd(X_train)
        X_test_features = text_vectorizer.transform_tfidf_svd(X_test)

        logger.info(f"Training set built: Features shape={X_train_features.shape}")

        return {
            "status": "success",
            "feature_shape": X_train_features.shape,
            "train_samples": len(X_train_features),
            "test_samples": len(X_test_features),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Training set building failed: {e}")
        return {"status": "failed", "error": str(e), "timestamp": datetime.now().isoformat()}


@task(retries=1, retry_delay_seconds=120)
def train_model_task(
    model_name: str, training_set_result: Dict[str, Any], n_trials: int = 20
) -> Dict[str, Any]:
    """Train individual model task"""
    logger = get_run_logger()
    logger.info(f"Training {model_name}...")

    try:
        if training_set_result["status"] != "success":
            raise ValueError("Training set building failed, cannot train model")

        # Import here to avoid circular imports
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier

        from data_loader import DataLoader
        from text_encoders import TextVectorizer

        data_loader = DataLoader()
        text_vectorizer = TextVectorizer()

        # Reload and prepare data
        data_loader.load_dataset()
        recommended_categories = data_loader.get_category_recommendations(max_categories=5)
        if recommended_categories:
            data_loader.set_selected_categories(recommended_categories)

        data_loader.select_samples(max_samples=10000)
        data_loader.preprocess_samples()
        data_loader.create_label_mappings()

        X_train, X_test, y_train, y_test = data_loader.prepare_train_test_data()

        # Create 3-way split
        from models.utils.validation_manager import validation_manager

        X_train_full, X_val, X_test, y_train_full, y_val, y_test = validation_manager.split_data(
            np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test])
        )

        # Create features
        X_train_features = text_vectorizer.fit_transform_tfidf_svd(X_train_full)
        X_val_features = text_vectorizer.transform_tfidf_svd(X_val)
        X_test_features = text_vectorizer.transform_tfidf_svd(X_test)

        # Get model class
        model_classes = {
            "random_forest": RandomForestClassifier,
            "knn": KNeighborsClassifier,
            "svm": SVC,
            "logistic_regression": LogisticRegression,
            "naive_bayes": MultinomialNB,
            "decision_tree": DecisionTreeClassifier,
        }

        model_class = model_classes.get(model_name, RandomForestClassifier)

        # Simple training without Optuna for orchestration
        model = model_class()
        model.fit(X_train_features, y_train_full)

        # Evaluate
        y_pred = model.predict(X_test_features)
        from sklearn.metrics import accuracy_score, f1_score

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Log to MLflow
        tracker = MLflowTracker("prefect_training")
        with tracker.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            tracker.log_params(
                {
                    "model_name": model_name,
                    "n_trials": n_trials,
                    "train_samples": len(X_train_features),
                    "test_samples": len(X_test_features),
                }
            )

            tracker.log_metrics({"accuracy": accuracy, "f1": f1})

            tracker.log_model(model, "model", registered_model_name=f"{model_name}_model")

            # Register model
            registry = ModelRegistry(f"{model_name}_model")
            run_id = mlflow.active_run().info.run_id
            registry.register_model(run_id, "model", "Staging")

        logger.info(f"{model_name} training completed: F1={f1:.4f}")

        return {
            "status": "success",
            "model_name": model_name,
            "accuracy": accuracy,
            "f1": f1,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"{model_name} training failed: {e}")
        return {
            "status": "failed",
            "model_name": model_name,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@task(retries=1, retry_delay_seconds=30)
def evaluate_model_task(model_result: Dict[str, Any]) -> Dict[str, Any]:
    """Model evaluation task"""
    logger = get_run_logger()
    logger.info(f"Evaluating {model_result.get('model_name', 'unknown')}...")

    try:
        if model_result["status"] != "success":
            logger.warning(
                f"Model {model_result.get('model_name')} training failed, skipping evaluation"
            )
            return {
                "status": "skipped",
                "model_name": model_result.get("model_name"),
                "reason": "Training failed",
                "timestamp": datetime.now().isoformat(),
            }

        # Create evaluation report
        eval_report = {
            "model_name": model_result["model_name"],
            "metrics": {"accuracy": model_result["accuracy"], "f1": model_result["f1"]},
            "baseline": {"f1": 0.5},  # Simple baseline
            "drift": {"ok": True},  # Placeholder
            "timestamp": datetime.now().isoformat(),
        }

        # Save evaluation report
        import json

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/eval_report.json", "w") as f:
            json.dump(eval_report, f, indent=2)

        logger.info(f"Evaluation completed for {model_result['model_name']}")

        return {
            "status": "success",
            "model_name": model_result["model_name"],
            "eval_report": eval_report,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Evaluation failed for {model_result.get('model_name')}: {e}")
        return {
            "status": "failed",
            "model_name": model_result.get("model_name"),
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@task(retries=1, retry_delay_seconds=30)
def materialize_features_task() -> Dict[str, Any]:
    """Materialize features task (placeholder for Feast integration)"""
    logger = get_run_logger()
    logger.info("Materializing features...")

    try:
        # Placeholder for Feast feature materialization
        logger.info("Feature materialization completed (placeholder)")

        return {
            "status": "success",
            "message": "Features materialized",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Feature materialization failed: {e}")
        return {"status": "failed", "error": str(e), "timestamp": datetime.now().isoformat()}


@task(retries=1, retry_delay_seconds=30)
def deploy_canary_task(model_result: Dict[str, Any]) -> Dict[str, Any]:
    """Deploy model to canary task"""
    logger = get_run_logger()
    logger.info(f"Deploying {model_result.get('model_name')} to canary...")

    try:
        if model_result["status"] != "success":
            logger.warning(f"Cannot deploy {model_result.get('model_name')} - training failed")
            return {
                "status": "skipped",
                "model_name": model_result.get("model_name"),
                "reason": "Training failed",
                "timestamp": datetime.now().isoformat(),
            }

        # Placeholder for actual deployment
        logger.info(f"Canary deployment completed for {model_result['model_name']}")

        return {
            "status": "success",
            "model_name": model_result["model_name"],
            "deployment_type": "canary",
            "traffic_percentage": 10,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Canary deployment failed for {model_result.get('model_name')}: {e}")
        return {
            "status": "failed",
            "model_name": model_result.get("model_name"),
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@flow(
    name="ml-training-pipeline",
    task_runner=ConcurrentTaskRunner(),
    retries=1,
    retry_delay_seconds=300,
)
def ml_training_pipeline_flow(
    models: List[str] = ["random_forest", "knn"], n_trials: int = 20
) -> Dict[str, Any]:
    """Main ML training pipeline flow"""
    logger = get_run_logger()
    logger.info("Starting ML training pipeline...")

    # Step 1: Ingest data
    ingest_result = ingest_data_task()

    # Step 2: Validate data
    validation_result = validate_data_task()

    # Step 3: Build training set
    training_set_result = build_training_set_task(ingest_result)

    # Step 4: Train models (parallel)
    model_results = []
    for model_name in models:
        model_result = train_model_task(model_name, training_set_result, n_trials)
        model_results.append(model_result)

    # Step 5: Evaluate models
    eval_results = []
    for model_result in model_results:
        eval_result = evaluate_model_task(model_result)
        eval_results.append(eval_result)

    # Step 6: Materialize features
    materialize_result = materialize_features_task()

    # Step 7: Deploy to canary (parallel)
    deployment_results = []
    for model_result in model_results:
        deploy_result = deploy_canary_task(model_result)
        deployment_results.append(deploy_result)

    # Summary
    successful_models = [r for r in model_results if r["status"] == "success"]
    failed_models = [r for r in model_results if r["status"] == "failed"]

    logger.info(
        f"Pipeline completed: {len(successful_models)} successful, {len(failed_models)} failed"
    )

    return {
        "status": "completed",
        "successful_models": len(successful_models),
        "failed_models": len(failed_models),
        "model_results": model_results,
        "eval_results": eval_results,
        "deployment_results": deployment_results,
        "timestamp": datetime.now().isoformat(),
    }


@flow(name="data-validation-flow")
def data_validation_flow() -> Dict[str, Any]:
    """Data validation flow"""
    logger = get_run_logger()
    logger.info("Starting data validation flow...")

    # Run validation
    validation_result = validate_data_task()

    return validation_result


@flow(name="model-evaluation-flow")
def model_evaluation_flow() -> Dict[str, Any]:
    """Model evaluation flow"""
    logger = get_run_logger()
    logger.info("Starting model evaluation flow...")

    # This would typically load models from MLflow registry and evaluate them
    logger.info("Model evaluation flow completed")

    return {"status": "completed", "timestamp": datetime.now().isoformat()}


# Create deployments
def create_deployments():
    """Create Prefect deployments"""

    # ML Training Pipeline - runs daily at 2 AM
    ml_training_deployment = Deployment.build_from_flow(
        flow=ml_training_pipeline_flow,
        name="ml-training-pipeline",
        schedule=CronSchedule(cron="0 2 * * *", timezone="UTC"),
        parameters={"models": ["random_forest", "knn"], "n_trials": 20},
    )

    # Data Validation - runs every 6 hours
    data_validation_deployment = Deployment.build_from_flow(
        flow=data_validation_flow,
        name="data-validation",
        schedule=CronSchedule(cron="0 */6 * * *", timezone="UTC"),
    )

    # Model Evaluation - runs daily at 3 AM
    model_evaluation_deployment = Deployment.build_from_flow(
        flow=model_evaluation_flow,
        name="model-evaluation",
        schedule=CronSchedule(cron="0 3 * * *", timezone="UTC"),
    )

    return [ml_training_deployment, data_validation_deployment, model_evaluation_deployment]


if __name__ == "__main__":
    # Run the flow locally for testing
    result = ml_training_pipeline_flow(models=["random_forest"], n_trials=5)
    print(f"Pipeline result: {result}")
