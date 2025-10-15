# Deployment Guide

## Tổng quan Deployment

Hướng dẫn deploy hệ thống MLOps Topic Modeling từ development đến production.

## Prerequisites

### 1. System Requirements
- **OS**: Linux/macOS/Windows
- **Python**: 3.11+
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Conda**: Latest version
- **Git**: Latest version

### 2. External Services
- **S3 Bucket**: Cho MLflow artifacts và DVC storage
- **PostgreSQL**: Cho MLflow backend (có thể dùng managed service)
- **GitHub**: Cho CI/CD pipeline

## Environment Setup

### 1. Clone Repository
```bash
git clone <repository_url>
cd topic-modeling-project
```

### 2. Setup Conda Environment
```bash
# Create environment từ environment.yml
conda env create -f environment.yml
conda activate PJ3.1

# Generate conda-lock file (optional)
conda-lock -f environment.yml --platform linux-64
```

### 3. Install Dependencies
```bash
# Install pip dependencies
pip install -r requirements.txt

# Install DVC với S3 support
pip install dvc[s3]
```

### 4. Initialize DVC
```bash
# Initialize DVC
dvc init

# Add S3 remote
dvc remote add -d s3remote s3://your-mlflow-bucket/dvc

# Configure S3 credentials
dvc remote modify s3remote endpoint_url https://s3.amazonaws.com
dvc remote modify s3remote access_key_id YOUR_ACCESS_KEY
dvc remote modify s3remote secret_access_key YOUR_SECRET_KEY
```

## Configuration

### 1. Environment Variables
```bash
# Copy example environment file
cp .env.example .env

# Edit .env với your actual values
nano .env
```

**Required Environment Variables:**
```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Model Configuration
MODEL_NAME=topic_model
MODEL_STAGE=Staging

# Database Configuration
DATABASE_URL=postgresql://mlflow:mlflow@localhost:5432/mlflow
```

### 2. DVC Parameters
```bash
# Edit params.yaml với your parameters
nano params.yaml
```

**Example params.yaml:**
```yaml
train:
  random_state: 42
  n_estimators: 100
  learning_rate: 0.1
  max_depth: 5
  
data:
  test_size: 0.2
  validation_size: 0.2
  
model:
  name: "TopicModel"
  stage: "Staging"
```

## Deployment Scenarios

### 1. Local Development

#### Start Infrastructure
```bash
# Start development stack
make up

# Verify services
docker ps
```

**Services Started:**
- PostgreSQL: `localhost:5432`
- MinIO: `localhost:9000` (API), `localhost:9001` (UI)
- MLflow: `localhost:5000`

#### Run Training Pipeline
```bash
# Run training với MLflow tracking
make train

# Check MLflow UI
open http://localhost:5000
```

#### Start Model Serving
```bash
# Start FastAPI serving
make serve

# Test API
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

#### Start Monitoring
```bash
# Generate monitoring configs
python src/monitoring_config.py

# Start monitoring stack
docker compose -f monitoring/docker-compose.monitoring.yml up -d

# Access Grafana
open http://localhost:3000
# Username: admin, Password: admin
```

### 2. Staging Environment

#### Infrastructure Setup
```bash
# Deploy infrastructure
docker compose -f infra/docker-compose.staging.yml up -d

# Setup monitoring
docker compose -f monitoring/docker-compose.monitoring.yml up -d
```

#### CI/CD Pipeline
```bash
# Push to staging branch
git push origin staging

# GitHub Actions sẽ tự động:
# 1. Run tests
# 2. Train model
# 3. Deploy to staging
# 4. Run canary deployment
```

### 3. Production Environment

#### Infrastructure Setup
```bash
# Deploy production infrastructure
docker compose -f infra/docker-compose.prod.yml up -d

# Setup monitoring với production configs
docker compose -f monitoring/docker-compose.prod.yml up -d
```

#### Production Deployment
```bash
# Push to main branch
git push origin main

# GitHub Actions sẽ tự động:
# 1. Run full test suite
# 2. Train model với production data
# 3. Deploy canary (10% traffic)
# 4. Monitor canary performance
# 5. Promote to production hoặc rollback
```

## Monitoring và Maintenance

### 1. Health Checks

#### Application Health
```bash
# Check FastAPI health
curl http://localhost:8000/health

# Check MLflow health
curl http://localhost:5000/health

# Check Prometheus metrics
curl http://localhost:9090/metrics
```

#### Infrastructure Health
```bash
# Check Docker containers
docker ps

# Check logs
docker logs mlflow_server
docker logs postgres
docker logs minio
```

### 2. Monitoring Dashboards

#### Grafana Dashboards
- **Application Metrics**: Request latency, error rates, throughput
- **ML Metrics**: Model performance, prediction accuracy
- **Infrastructure Metrics**: CPU, memory, disk usage

#### MLflow UI
- **Experiments**: Training runs và metrics
- **Model Registry**: Model versions và stages
- **Artifacts**: Model files và data

### 3. Log Management

#### Application Logs
```bash
# FastAPI logs
docker logs fastapi_app

# Training logs
tail -f logs/training.log
```

#### Infrastructure Logs
```bash
# All services logs
docker compose logs -f

# Specific service logs
docker compose logs -f mlflow
```

## Troubleshooting

### 1. Common Issues

#### MLflow Connection Issues
```bash
# Check MLflow server status
curl http://localhost:5000/health

# Restart MLflow server
docker compose restart mlflow
```

#### Database Connection Issues
```bash
# Check PostgreSQL status
docker compose logs postgres

# Restart PostgreSQL
docker compose restart postgres
```

#### S3/MinIO Issues
```bash
# Check MinIO status
curl http://localhost:9000/minio/health/live

# Restart MinIO
docker compose restart minio
```

### 2. Performance Issues

#### High Memory Usage
```bash
# Check memory usage
docker stats

# Optimize model serving
# Reduce batch size trong FastAPI
```

#### Slow Training
```bash
# Check GPU usage
nvidia-smi

# Optimize training parameters
# Reduce dataset size cho testing
```

### 3. Data Issues

#### Data Validation Failures
```bash
# Check data validation logs
tail -f logs/validation.log

# Re-run data validation
python src/data_validation.py
```

#### Model Performance Degradation
```bash
# Check drift reports
ls reports/drift_*

# Re-run drift monitoring
python src/drift_monitoring.py
```

## Security Considerations

### 1. Access Control
- **API Authentication**: Implement API keys hoặc JWT tokens
- **Database Security**: Use strong passwords và SSL connections
- **S3 Security**: Implement proper IAM policies

### 2. Data Protection
- **Data Encryption**: Encrypt data at rest và in transit
- **Access Logging**: Log all data access
- **Backup Strategy**: Regular backups của database và artifacts

### 3. Infrastructure Security
- **Container Security**: Use minimal base images
- **Network Security**: Implement proper firewall rules
- **Secret Management**: Use proper secret management tools

## Backup và Recovery

### 1. Data Backup
```bash
# Backup PostgreSQL database
docker exec postgres pg_dump -U mlflow mlflow > backup/mlflow_backup.sql

# Backup MinIO data
docker exec minio mc mirror /data backup/minio_backup/
```

### 2. Model Backup
```bash
# Export models từ MLflow
mlflow models export -m models:/TopicModel/Production -o backup/models/

# Backup MLflow artifacts
aws s3 sync s3://your-mlflow-bucket backup/s3_backup/
```

### 3. Recovery Procedures
```bash
# Restore PostgreSQL database
docker exec -i postgres psql -U mlflow mlflow < backup/mlflow_backup.sql

# Restore MinIO data
docker exec minio mc mirror backup/minio_backup/ /data

# Restore models
mlflow models import -m backup/models/ -n TopicModel
```

## Scaling Considerations

### 1. Horizontal Scaling
- **FastAPI**: Deploy multiple instances với load balancer
- **MLflow**: Use MLflow server clustering
- **Database**: Implement read replicas

### 2. Vertical Scaling
- **GPU Resources**: Add more GPU instances
- **Memory**: Increase memory allocation
- **Storage**: Scale storage capacity

### 3. Performance Optimization
- **Caching**: Implement Redis caching
- **CDN**: Use CDN cho static assets
- **Database Optimization**: Optimize queries và indexes

---
*Deployment guide created on: 2025-01-14*
*Production-ready MLOps deployment*
