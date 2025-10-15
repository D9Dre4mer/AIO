# API Documentation

## Tổng quan API

FastAPI serving application cho Topic Modeling với các endpoints cho prediction, health check, và metrics monitoring.

## Base URL

```
http://localhost:8000
```

## Authentication

Hiện tại API không yêu cầu authentication. Trong production, nên implement API keys hoặc JWT tokens.

## Endpoints

### 1. Health Check

#### GET /health
Kiểm tra trạng thái health của service và model.

**Request:**
```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "timestamp": "2025-01-14T10:30:00Z"
}
```

**Status Codes:**
- `200 OK`: Service healthy
- `503 Service Unavailable`: Model not loaded

**Example:**
```bash
curl http://localhost:8000/health
```

### 2. Model Prediction

#### POST /predict
Thực hiện prediction trên text input.

**Request:**
```http
POST /predict
Content-Type: application/json

{
  "texts": [
    "This is a sample text for topic modeling",
    "Another text to classify"
  ]
}
```

**Request Schema:**
```json
{
  "texts": ["string"]
}
```

**Response:**
```json
{
  "predictions": [
    "topic_1",
    "topic_2"
  ],
  "confidence_scores": [
    0.85,
    0.92
  ],
  "processing_time_ms": 150
}
```

**Response Schema:**
```json
{
  "predictions": ["string"],
  "confidence_scores": ["number"],
  "processing_time_ms": "number"
}
```

**Status Codes:**
- `200 OK`: Prediction successful
- `422 Unprocessable Entity`: Invalid input
- `500 Internal Server Error`: Prediction failed
- `503 Service Unavailable`: Model not loaded

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Machine learning is fascinating", "Data science applications"]}'
```

### 3. Model Information

#### GET /model/info
Lấy thông tin về model hiện tại.

**Request:**
```http
GET /model/info
```

**Response:**
```json
{
  "model_name": "TopicModel",
  "model_version": "1.0.0",
  "model_stage": "Production",
  "model_type": "sklearn",
  "training_date": "2025-01-14T08:00:00Z",
  "performance_metrics": {
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.91,
    "f1_score": 0.90
  }
}
```

**Status Codes:**
- `200 OK`: Model info retrieved
- `503 Service Unavailable`: Model not loaded

**Example:**
```bash
curl http://localhost:8000/model/info
```

### 4. Batch Prediction

#### POST /predict/batch
Thực hiện prediction trên nhiều texts với batch processing.

**Request:**
```http
POST /predict/batch
Content-Type: application/json

{
  "texts": [
    "Text 1",
    "Text 2",
    "Text 3"
  ],
  "batch_size": 100,
  "return_probabilities": true
}
```

**Request Schema:**
```json
{
  "texts": ["string"],
  "batch_size": "number",
  "return_probabilities": "boolean"
}
```

**Response:**
```json
{
  "predictions": [
    "topic_1",
    "topic_2", 
    "topic_3"
  ],
  "probabilities": [
    [0.1, 0.8, 0.1],
    [0.2, 0.1, 0.7],
    [0.5, 0.3, 0.2]
  ],
  "batch_processing_time_ms": 500,
  "total_texts": 3
}
```

**Status Codes:**
- `200 OK`: Batch prediction successful
- `422 Unprocessable Entity`: Invalid input
- `500 Internal Server Error`: Prediction failed

**Example:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2"], "batch_size": 50, "return_probabilities": true}'
```

### 5. Metrics

#### GET /metrics
Prometheus metrics endpoint cho monitoring.

**Request:**
```http
GET /metrics
```

**Response:**
```
# HELP http_requests_total Total HTTP Requests
# TYPE http_requests_total counter
http_requests_total{method="GET",endpoint="/health"} 10
http_requests_total{method="POST",endpoint="/predict"} 25

# HELP http_request_duration_seconds HTTP Request Latency
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{method="POST",endpoint="/predict",le="0.1"} 5
http_request_duration_seconds_bucket{method="POST",endpoint="/predict",le="0.5"} 20
http_request_duration_seconds_bucket{method="POST",endpoint="/predict",le="1.0"} 25

# HELP predict_requests_total Total Prediction Requests
# TYPE predict_requests_total counter
predict_requests_total 25

# HELP predict_errors_total Total Prediction Errors
# TYPE predict_errors_total counter
predict_errors_total 0
```

**Status Codes:**
- `200 OK`: Metrics retrieved

**Example:**
```bash
curl http://localhost:8000/metrics
```

### 6. API Documentation

#### GET /docs
Interactive API documentation (Swagger UI).

**Request:**
```http
GET /docs
```

**Response:** HTML page với interactive API documentation.

**Example:**
```bash
# Open in browser
open http://localhost:8000/docs
```

#### GET /redoc
Alternative API documentation (ReDoc).

**Request:**
```http
GET /redoc
```

**Response:** HTML page với ReDoc documentation.

**Example:**
```bash
# Open in browser
open http://localhost:8000/redoc
```

## Error Handling

### Error Response Format
```json
{
  "detail": "Error message",
  "error_code": "ERROR_CODE",
  "timestamp": "2025-01-14T10:30:00Z"
}
```

### Common Error Codes

#### 422 Unprocessable Entity
```json
{
  "detail": [
    {
      "loc": ["body", "texts"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

#### 500 Internal Server Error
```json
{
  "detail": "Prediction failed: Model error",
  "error_code": "PREDICTION_ERROR",
  "timestamp": "2025-01-14T10:30:00Z"
}
```

#### 503 Service Unavailable
```json
{
  "detail": "Model not loaded yet",
  "error_code": "MODEL_NOT_LOADED",
  "timestamp": "2025-01-14T10:30:00Z"
}
```

## Rate Limiting

Hiện tại không có rate limiting. Trong production, nên implement rate limiting để bảo vệ API.

**Recommended Limits:**
- **Per IP**: 100 requests/minute
- **Per API Key**: 1000 requests/minute
- **Burst**: 200 requests/minute

## Performance Considerations

### 1. Response Times
- **Health Check**: < 10ms
- **Single Prediction**: < 100ms
- **Batch Prediction**: < 500ms (100 texts)

### 2. Throughput
- **Concurrent Requests**: 100 requests/second
- **Batch Processing**: 1000 texts/second

### 3. Resource Usage
- **Memory**: ~2GB per instance
- **CPU**: 2-4 cores recommended
- **GPU**: Optional for large models

## Monitoring và Alerting

### 1. Key Metrics
- **Request Rate**: Requests per second
- **Response Time**: P95, P99 latency
- **Error Rate**: 4xx, 5xx error rates
- **Model Performance**: Prediction accuracy

### 2. Alerts
- **High Error Rate**: > 5% errors
- **High Latency**: P95 > 500ms
- **Model Drift**: Performance degradation
- **Service Down**: Health check failures

### 3. Dashboards
- **Grafana**: Application metrics
- **MLflow**: Model performance
- **Prometheus**: System metrics

## Testing

### 1. Unit Tests
```bash
# Run unit tests
pytest tests/test_serve_app.py -v
```

### 2. Integration Tests
```bash
# Test API endpoints
pytest tests/test_api_integration.py -v
```

### 3. Load Testing
```bash
# Load test với locust
locust -f tests/load_test.py --host=http://localhost:8000
```

## Security Best Practices

### 1. Input Validation
- Validate input text length
- Sanitize input text
- Check for malicious content

### 2. Output Sanitization
- Validate prediction outputs
- Limit response size
- Prevent information leakage

### 3. Authentication (Future)
- Implement API key authentication
- Use JWT tokens
- Implement rate limiting per user

## Deployment Considerations

### 1. Environment Variables
```bash
# Required environment variables
MODEL_NAME=topic_model
MODEL_STAGE=Production
MLFLOW_TRACKING_URI=http://mlflow:5000
```

### 2. Health Checks
- Implement proper health checks
- Use Kubernetes liveness/readiness probes
- Monitor model loading status

### 3. Scaling
- Use horizontal pod autoscaling
- Implement load balancing
- Use connection pooling

---
*API documentation created on: 2025-01-14*
*FastAPI serving with MLflow integration*
