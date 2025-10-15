"""
Unit tests for FastAPI serving app
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serve.app import app, load_model


class TestFastAPIApp:
    """Test FastAPI application"""

    def test_root_endpoint(self):
        """Test root endpoint"""
        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data

    @patch("src.serve.app.load_model")
    def test_health_endpoint(self, mock_load_model):
        """Test health endpoint"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "model_loaded" in data

    @patch("src.serve.app.load_model")
    def test_predict_endpoint(self, mock_load_model):
        """Test predict endpoint"""
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        mock_load_model.return_value = mock_model

        # Mock MLflow client
        with patch("src.serve.app.mlflow.tracking.MlflowClient") as mock_client_class:
            mock_client = MagicMock()
            mock_version = MagicMock()
            mock_version.version = "1"
            mock_client.get_latest_versions.return_value = [mock_version]
            mock_client_class.return_value = mock_client

            client = TestClient(app)
            response = client.post(
                "/predict",
                json={
                    "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    "model_name": "test_model",
                    "model_stage": "Staging",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "model_name" in data
            assert "model_version" in data
            assert len(data["predictions"]) == 3

    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        client = TestClient(app)
        response = client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    @patch("src.serve.app.mlflow.tracking.MlflowClient")
    def test_models_endpoint(self, mock_client_class):
        """Test models endpoint"""
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.name = "test_model"
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_version.current_stage = "Staging"
        mock_version.creation_timestamp = 1234567890
        mock_model.latest_versions = [mock_version]
        mock_client.search_registered_models.return_value = [mock_model]
        mock_client_class.return_value = mock_client

        client = TestClient(app)
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "test_model"


class TestLoadModel:
    """Test load_model function"""

    @patch("src.serve.app.mlflow.sklearn.load_model")
    @patch("src.serve.app.mlflow.set_tracking_uri")
    def test_load_model_success(self, mock_set_tracking_uri, mock_load_model):
        """Test successful model loading"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        result = load_model("test_model", "Staging")

        assert result == mock_model
        mock_set_tracking_uri.assert_called_once()
        mock_load_model.assert_called_once()

    @patch("src.serve.app.mlflow.sklearn.load_model")
    @patch("src.serve.app.mlflow.set_tracking_uri")
    def test_load_model_failure(self, mock_set_tracking_uri, mock_load_model):
        """Test model loading failure"""
        mock_load_model.side_effect = Exception("Model not found")

        with pytest.raises(Exception):
            load_model("nonexistent_model", "Staging")


if __name__ == "__main__":
    pytest.main([__file__])
