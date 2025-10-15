"""
Training module with MLflow integration
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# from models import NewModelTrainer  # Removed - not available
from data_loader import DataLoader
from src.mlflow_integration import MLflowTracker, ModelRegistry
from src.optuna_integration import (
    PARAM_SPACES,
    OptunaMLflowIntegration,
    create_optuna_objective,
)
from text_encoders import TextVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLTrainingPipeline:
    """ML Training Pipeline with MLflow integration"""

    def __init__(self, config_path: str = "params.yaml"):
        self.config = self._load_config(config_path)
        self.tracker = MLflowTracker("ml_training")
        self.optuna_integration = OptunaMLflowIntegration("hyperparameter_optimization")

        # Initialize components
        self.data_loader = DataLoader()
        self.text_vectorizer = TextVectorizer()
        # self.model_trainer = NewModelTrainer(  # Removed - not available
        #     cv_folds=self.config["train"]["cv_folds"],
        #     validation_size=self.config["validation_size"],
        #     test_size=self.config["test_size"]
        # )

        # Create output directories
        self._create_directories()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _create_directories(self):
        """Create necessary output directories"""
        directories = ["models", "artifacts", "metrics", "data/processed", "data/features"]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def load_and_preprocess_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess data"""
        logger.info("Loading and preprocessing data...")

        # Load dataset
        self.data_loader.load_dataset()

        # Discover categories first
        self.data_loader.discover_categories()

        # Get recommended categories
        recommended_categories = self.data_loader.get_category_recommendations(max_categories=5)
        if recommended_categories:
            self.data_loader.set_selected_categories(recommended_categories)
        else:
            # Fallback to default categories if no recommendations
            logger.warning("No category recommendations found, using default categories")
            default_categories = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.MA"]
            self.data_loader.set_selected_categories(default_categories)

        # Select and preprocess samples
        max_samples = self.config["ingest"]["max_samples"]
        self.data_loader.select_samples(max_samples=max_samples)
        self.data_loader.preprocess_samples()
        self.data_loader.create_label_mappings()

        # Prepare train/test data
        X_train, X_test, y_train, y_test = self.data_loader.prepare_train_test_data()

        # Create 3-way split
        from models.utils.validation_manager import validation_manager

        X_train_full, X_val, X_test, y_train_full, y_val, y_test = validation_manager.split_data(
            np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test])
        )

        logger.info(f"Data split: Train={len(X_train_full)}, Val={len(X_val)}, Test={len(X_test)}")

        return X_train_full, X_val, X_test, y_train_full, y_val, y_test

    def create_features(
        self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create features using text vectorization"""
        logger.info("Creating features...")

        # Use TF-IDF with SVD for speed
        X_train_features = self.text_vectorizer.fit_transform_tfidf_svd(X_train)
        X_val_features = self.text_vectorizer.transform_tfidf_svd(X_val)
        X_test_features = self.text_vectorizer.transform_tfidf_svd(X_test)

        logger.info(
            f"Feature shapes: Train={X_train_features.shape}, Val={X_val_features.shape}, Test={X_test_features.shape}"
        )

        return X_train_features, X_val_features, X_test_features

    def train_model_with_optuna(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train model with Optuna hyperparameter optimization"""

        logger.info(f"Training {model_name} with Optuna optimization...")

        # Get model class and parameter space
        model_class = self._get_model_class(model_name)
        param_space = PARAM_SPACES.get(model_name, {})

        if not param_space:
            logger.warning(f"No parameter space defined for {model_name}, using default parameters")
            model = model_class()
            model.fit(X_train, y_train)
            return model, {}

        # Create objective function
        objective = create_optuna_objective(
            model_class, X_train, y_train, X_val, y_val, param_space, metric_name="f1"
        )

        # Run optimization
        study = self.optuna_integration.optimize_with_mlflow(
            objective_func=objective, n_trials=n_trials, timeout=None, direction="maximize"
        )

        # Train final model with best parameters
        best_params = study.best_params
        final_model = model_class(**best_params)
        final_model.fit(X_train, y_train)

        logger.info(f"Best parameters for {model_name}: {best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")

        return final_model, best_params

    def _get_model_class(self, model_name: str):
        """Get model class by name"""
        from sklearn.cluster import KMeans
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier

        model_classes = {
            "random_forest": RandomForestClassifier,
            "knn": KNeighborsClassifier,
            "svm": SVC,
            "logistic_regression": LogisticRegression,
            "naive_bayes": MultinomialNB,
            "decision_tree": DecisionTreeClassifier,
            "kmeans": KMeans,
        }

        return model_classes.get(model_name, RandomForestClassifier)

    def evaluate_model(
        self, model: Any, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model and return metrics"""

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

        return metrics

    def train_all_models(self) -> Dict[str, Any]:
        """Train all models specified in config"""

        logger.info("Starting training pipeline...")

        # Load and preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_preprocess_data()

        # Create features
        X_train_features, X_val_features, X_test_features = self.create_features(
            X_train, X_val, X_test
        )

        # Train models
        models = {}
        results = {}

        for model_name in self.config["train"]["models"]:
            logger.info(f"Training {model_name}...")

            try:
                # Train with Optuna
                model, best_params = self.train_model_with_optuna(
                    model_name,
                    X_train_features,
                    y_train,
                    X_val_features,
                    y_val,
                    n_trials=self.config["train"]["optuna"]["n_trials"],
                )

                # Evaluate on test set
                test_metrics = self.evaluate_model(model, X_test_features, y_test)

                # Log to MLflow
                with self.tracker.start_run(
                    run_name=f"{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                ):
                    # Log parameters
                    self.tracker.log_params(
                        {
                            "model_name": model_name,
                            "n_trials": self.config["train"]["optuna"]["n_trials"],
                            **best_params,
                        }
                    )

                    # Log metrics
                    self.tracker.log_metrics(test_metrics)

                    # Log model
                    self.tracker.log_model(
                        model, "model", registered_model_name=f"{model_name}_model"
                    )

                    # Log model info
                    model_info = {
                        "model_name": model_name,
                        "best_params": best_params,
                        "test_metrics": test_metrics,
                        "feature_shape": X_train_features.shape,
                        "timestamp": pd.Timestamp.now().isoformat(),
                    }
                    self.tracker.log_dict(model_info, "model_info.json")

                    # Register model
                    registry = ModelRegistry(f"{model_name}_model")
                    run_id = mlflow.active_run().info.run_id
                    registry.register_model(run_id, "model", "Staging")

                models[model_name] = model
                results[model_name] = {
                    "model": model,
                    "params": best_params,
                    "metrics": test_metrics,
                }

                logger.info(f"Completed {model_name}: F1={test_metrics['f1']:.4f}")

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue

        # Save results
        self._save_results(results)

        logger.info("Training pipeline completed!")
        return results

    def _save_results(self, results: Dict[str, Any]):
        """Save training results"""

        # Save metrics
        metrics_data = {}
        for model_name, result in results.items():
            metrics_data[model_name] = result["metrics"]

        with open("metrics/train.json", "w") as f:
            json.dump(metrics_data, f, indent=2)

        # Save model info
        model_info = {}
        for model_name, result in results.items():
            model_info[model_name] = {"params": result["params"], "metrics": result["metrics"]}

        with open("artifacts/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

        logger.info("Results saved to metrics/train.json and artifacts/model_info.json")


def main():
    """Main training function"""

    # Initialize training pipeline
    pipeline = MLTrainingPipeline()

    # Run training
    results = pipeline.train_all_models()

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for model_name, result in results.items():
        metrics = result["metrics"]
        print(f"{model_name:20s}: F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
