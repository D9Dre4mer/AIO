"""
Training Pipeline mới sử dụng DataManager
Tích hợp với MLflow và UI management
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import mlflow
import mlflow.sklearn

from src.data_manager import DataManager
from src.mlflow_integration import MLflowTracker, ModelRegistry

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Training Pipeline sử dụng DataManager"""
    
    def __init__(self, data_manager: DataManager = None, experiment_name: str = "ml_training"):
        self.experiment_name = experiment_name
        self.data_manager = data_manager or DataManager()
        self.mlflow_tracker = MLflowTracker(experiment_name)
        self.model_registry = ModelRegistry("ml_model")
        
        # Available models
        self.models = {
            "random_forest": RandomForestClassifier,
            "logistic_regression": LogisticRegression,
            "svm": SVC,
            "knn": KNeighborsClassifier,
            "decision_tree": DecisionTreeClassifier
        }
        
        # Model parameters
        self.model_params = {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "logistic_regression": {
                "random_state": 42,
                "max_iter": 1000
            },
            "svm": {
                "random_state": 42,
                "probability": True
            },
            "knn": {
                "n_neighbors": 5
            },
            "decision_tree": {
                "random_state": 42,
                "max_depth": 10
            }
        }
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List available datasets"""
        return self.data_manager.list_available_datasets()
    
    def load_dataset(self, dataset_name: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Load dataset"""
        return self.data_manager.load_dataset(dataset_name, max_samples)
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset information"""
        return self.data_manager.get_dataset_info(dataset_name)
    
    def train_model(self, dataset_name: str, model_name: str, 
                   target_column: str, feature_columns: Optional[List[str]] = None,
                   test_size: float = 0.2, val_size: float = 0.2) -> Dict[str, Any]:
        """Train a single model"""
        
        logger.info(f"Training {model_name} on dataset {dataset_name}")
        
        # Load dataset if not already loaded
        if dataset_name not in self.data_manager.datasets:
            self.data_manager.load_dataset(dataset_name)
        
        # Prepare features and target
        X, y = self.data_manager.prepare_features_and_target(
            dataset_name, target_column, feature_columns
        )
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_manager.split_data(
            X, y, test_size=test_size, val_size=val_size
        )
        
        # Get model class and parameters
        model_class = self.models[model_name]
        model_params = self.model_params[model_name].copy()
        
        # Start MLflow run
        run_name = f"{model_name}_{dataset_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self.mlflow_tracker.start_run(run_name=run_name):
            # Log parameters
            self.mlflow_tracker.log_params({
                "dataset_name": dataset_name,
                "model_name": model_name,
                "target_column": target_column,
                "feature_columns": str(feature_columns) if feature_columns else "auto",
                "test_size": test_size,
                "val_size": val_size,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                **model_params
            })
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
            val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
            val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
            
            # Evaluate on test set
            y_test_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            
            # Log metrics
            self.mlflow_tracker.log_metrics({
                "val_accuracy": val_accuracy,
                "val_f1": val_f1,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "test_accuracy": test_accuracy,
                "test_f1": test_f1,
                "test_precision": test_precision,
                "test_recall": test_recall
            })
            
            # Log model
            self.mlflow_tracker.log_model(model, "model", registered_model_name=f"{model_name}_{dataset_name}")
            
            # Log classification report
            report = classification_report(y_test, y_test_pred, output_dict=True)
            report_path = f"artifacts/{run_name}_classification_report.json"
            os.makedirs("artifacts", exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.mlflow_tracker.log_artifact(report_path)
            
            # Register model
            run_id = mlflow.active_run().info.run_id
            self.model_registry.register_model(run_id, "model", "Staging")
            
            # Prepare results
            results = {
                "status": "success",
                "model_name": model_name,
                "dataset_name": dataset_name,
                "run_id": run_id,
                "metrics": {
                    "val_accuracy": val_accuracy,
                    "val_f1": val_f1,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "test_accuracy": test_accuracy,
                    "test_f1": test_f1,
                    "test_precision": test_precision,
                    "test_recall": test_recall
                },
                "classification_report": report,
                "model_params": model_params,
                "data_info": {
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                    "test_samples": len(X_test),
                    "n_features": X.shape[1],
                    "n_classes": len(np.unique(y))
                }
            }
            
            logger.info(f"Training completed: {model_name} - Test F1: {test_f1:.4f}")
            
            return results
    
    def train_all_models(self, dataset_name: str, target_column: str, 
                        feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train all available models"""
        
        logger.info(f"Training all models on dataset {dataset_name}")
        
        results = {}
        
        for model_name in self.models.keys():
            try:
                result = self.train_model(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    target_column=target_column,
                    feature_columns=feature_columns
                )
                results[model_name] = result
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                results[model_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Find best model
        best_model = None
        best_f1 = 0
        
        for model_name, result in results.items():
            if result["status"] == "success":
                f1_score = result["metrics"]["test_f1"]
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_model = model_name
        
        # Log summary
        summary = {
            "dataset_name": dataset_name,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "best_model": best_model,
            "best_f1": best_f1,
            "results": results,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Save summary
        summary_path = f"artifacts/training_summary_{dataset_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("artifacts", exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Training summary saved to {summary_path}")
        logger.info(f"Best model: {best_model} with F1: {best_f1:.4f}")
        
        return summary
    
    def predict(self, dataset_name: str, model_name: str, 
               target_column: str, feature_columns: Optional[List[str]] = None,
               input_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Make predictions using trained model"""
        
        # Load dataset if not already loaded
        if dataset_name not in self.data_manager.datasets:
            self.data_manager.load_dataset(dataset_name)
        
        # Prepare features
        if input_data is None:
            # Use test data from dataset
            X, y = self.data_manager.prepare_features_and_target(
                dataset_name, target_column, feature_columns
            )
            _, _, X_test, _, _, y_test = self.data_manager.split_data(X, y)
            input_data = pd.DataFrame(X_test)
        else:
            # Use provided input data
            if feature_columns:
                X = input_data[feature_columns].values
            else:
                X = input_data.values
            
            # Scale if needed
            if hasattr(self.data_manager, 'scaler'):
                X = self.data_manager.scaler.transform(X)
        
        # Load model from MLflow
        model_uri = f"models:/{model_name}_{dataset_name}/Staging"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = None
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
        
        results = {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist() if probabilities is not None else None,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "n_predictions": len(predictions)
        }
        
        return results


def main():
    """Test TrainingPipeline"""
    pipeline = TrainingPipeline()
    
    # List available datasets
    datasets = pipeline.list_datasets()
    print("Available datasets:")
    for dataset in datasets:
        print(f"- {dataset['name']}: {dataset['type']}")
    
    if datasets:
        # Use first dataset
        first_dataset = datasets[0]
        dataset_name = first_dataset['name']
        
        # Load dataset
        dataset_info = pipeline.load_dataset(dataset_name, max_samples=1000)
        print(f"\nLoaded dataset: {dataset_info['name']}")
        print(f"Shape: {dataset_info['shape']}")
        
        # Get dataset info
        info = pipeline.get_dataset_info(dataset_name)
        print(f"Columns: {info['columns']}")
        
        # Train a single model
        if info['type'] == 'classification':
            # Find target column
            target_candidates = [col for col in info['columns'] if 'target' in col.lower() or 'label' in col.lower()]
            if target_candidates:
                target_column = target_candidates[0]
                print(f"\nTraining Random Forest on {dataset_name} with target: {target_column}")
                
                result = pipeline.train_model(
                    dataset_name=dataset_name,
                    model_name="random_forest",
                    target_column=target_column
                )
                
                print(f"Training result: {result['status']}")
                if result['status'] == 'success':
                    print(f"Test F1: {result['metrics']['test_f1']:.4f}")


if __name__ == "__main__":
    main()
