"""
FastAPI serving application for ML models
"""

import logging
import os
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint"])
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds", "HTTP request duration", ["method", "endpoint"]
)
PREDICTION_COUNT = Counter("predictions_total", "Total predictions made")
PREDICTION_DURATION = Histogram("prediction_duration_seconds", "Prediction duration")

# Initialize FastAPI app
app = FastAPI(
    title="ML Model Serving API",
    description="API for serving machine learning models",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for predictions"""

    data: List[List[float]] = Field(..., description="Input features")
    model_name: str = Field(default="app_model", description="Model name")
    model_stage: str = Field(default="Staging", description="Model stage")


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    predictions: List[Any] = Field(..., description="Model predictions")
    model_name: str = Field(..., description="Model name used")
    model_version: str = Field(..., description="Model version")
    confidence: Optional[List[float]] = Field(None, description="Prediction confidence scores")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")


# Global model cache
model_cache: Dict[str, Any] = {}


def load_model(model_name: str, model_stage: str = "Staging") -> Any:
    """Load model from MLflow registry"""
    try:
        cache_key = f"{model_name}:{model_stage}"

        if cache_key in model_cache:
            return model_cache[cache_key]

        # Set MLflow tracking URI
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)

        # Load model from registry
        model_uri = f"models:/{model_name}/{model_stage}"
        model = mlflow.sklearn.load_model(model_uri)

        # Cache model
        model_cache[cache_key] = model

        logger.info(f"Loaded model {model_name} from stage {model_stage}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    REQUEST_COUNT.labels(method="GET", endpoint="/health").inc()

    try:
        # Try to load a model to check MLflow connectivity
        model_name = os.getenv("MODEL_NAME", "app_model")
        model_stage = os.getenv("MODEL_STAGE", "Staging")

        try:
            load_model(model_name, model_stage)
            model_loaded = True
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            model_loaded = False

        return HealthResponse(
            status="healthy", timestamp=pd.Timestamp.now().isoformat(), model_loaded=model_loaded
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unhealthy")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Prediction endpoint"""
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    PREDICTION_COUNT.inc()

    with REQUEST_DURATION.labels(method="POST", endpoint="/predict").time():
        with PREDICTION_DURATION.time():
            try:
                # Load model
                model = load_model(request.model_name, request.model_stage)

                # Convert input to numpy array
                X = np.array(request.data)

                # Make predictions
                predictions = model.predict(X)

                # Get confidence scores if available
                confidence = None
                if hasattr(model, "predict_proba"):
                    try:
                        confidence_scores = model.predict_proba(X)
                        confidence = confidence_scores.max(axis=1).tolist()
                    except Exception as e:
                        logger.debug(f"Could not get confidence scores: {e}")
                        pass

                # Get model version info
                try:
                    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
                    mlflow.set_tracking_uri(tracking_uri)
                    client = mlflow.tracking.MlflowClient()
                    latest_version = client.get_latest_versions(
                        request.model_name, stages=[request.model_stage]
                    )[0]
                    model_version = latest_version.version
                except Exception as e:
                    logger.debug(f"Could not get model version: {e}")
                    model_version = "unknown"

                return PredictionResponse(
                    predictions=predictions.tolist(),
                    model_name=request.model_name,
                    model_version=model_version,
                    confidence=confidence,
                )

            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    REQUEST_COUNT.labels(method="GET", endpoint="/metrics").inc()

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/models")
async def list_models():
    """List available models"""
    REQUEST_COUNT.labels(method="GET", endpoint="/models").inc()

    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

        models = client.search_registered_models()
        return {
            "models": [
                {
                    "name": model.name,
                    "latest_versions": [
                        {
                            "version": version.version,
                            "stage": version.current_stage,
                            "creation_timestamp": version.creation_timestamp,
                        }
                        for version in model.latest_versions
                    ],
                }
                for model in models
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Model Serving API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "models": "/models",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(app, host=host, port=port)
