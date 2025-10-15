"""
Unit tests for MLflow integration
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.mlflow_integration import MLflowTracker, ModelRegistry, log_experiment_results


class TestMLflowTracker:
    """Test MLflowTracker class"""

    def test_init(self):
        """Test MLflowTracker initialization"""
        tracker = MLflowTracker("test_experiment")
        assert tracker.experiment_name == "test_experiment"

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    def test_setup_mlflow(self, mock_set_experiment, mock_set_tracking_uri):
        """Test MLflow setup"""
        tracker = MLflowTracker("test_experiment")
        mock_set_tracking_uri.assert_called_once()
        mock_set_experiment.assert_called_once_with("test_experiment")

    @patch("mlflow.start_run")
    def test_start_run(self, mock_start_run):
        """Test starting a run"""
        tracker = MLflowTracker("test_experiment")
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run

        with tracker.start_run(run_name="test_run", tags={"test": "tag"}):
            pass

        mock_start_run.assert_called_once()

    @patch("mlflow.log_params")
    def test_log_params(self, mock_log_params):
        """Test logging parameters"""
        tracker = MLflowTracker("test_experiment")
        params = {"param1": "value1", "param2": 42}

        tracker.log_params(params)
        mock_log_params.assert_called_once_with(params)

    @patch("mlflow.log_metrics")
    def test_log_metrics(self, mock_log_metrics):
        """Test logging metrics"""
        tracker = MLflowTracker("test_experiment")
        metrics = {"accuracy": 0.95, "f1": 0.92}

        tracker.log_metrics(metrics)
        mock_log_metrics.assert_called_once_with(metrics, step=None)


class TestModelRegistry:
    """Test ModelRegistry class"""

    def test_init(self):
        """Test ModelRegistry initialization"""
        registry = ModelRegistry("test_model")
        assert registry.model_name == "test_model"

    @patch("mlflow.register_model")
    @patch("mlflow.tracking.MlflowClient")
    def test_register_model(self, mock_client_class, mock_register_model):
        """Test model registration"""
        registry = ModelRegistry("test_model")
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_register_model.return_value = MagicMock(version="1")

        result = registry.register_model("run_123", "model_path", "Staging")

        mock_register_model.assert_called_once()
        mock_client.transition_model_version_stage.assert_called_once()


class TestLogExperimentResults:
    """Test log_experiment_results function"""

    @patch("src.mlflow_integration.MLflowTracker")
    @patch("src.mlflow_integration.ModelRegistry")
    @patch("mlflow.start_run")
    @patch("mlflow.active_run")
    def test_log_experiment_results(
        self, mock_active_run, mock_start_run, mock_registry_class, mock_tracker_class
    ):
        """Test logging experiment results"""
        # Mock active run
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_active_run.return_value = mock_run

        mock_tracker = MagicMock()
        mock_registry = MagicMock()
        mock_tracker_class.return_value = mock_tracker
        mock_registry_class.return_value = mock_registry

        model = MagicMock()
        params = {"param1": "value1"}
        metrics = {"accuracy": 0.95}
        artifacts = {"plot": "plot.png"}

        tracker, registry = log_experiment_results("test_model", params, metrics, model, artifacts)

        assert tracker == mock_tracker
        assert registry == mock_registry
        mock_tracker.log_params.assert_called_once_with(params)
        mock_tracker.log_metrics.assert_called_once_with(metrics)
        mock_tracker.log_model.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
