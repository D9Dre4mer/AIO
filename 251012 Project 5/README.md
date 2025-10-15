# ML Project Migration - Production Ready MLOps

This project has been migrated from a simple ML application to a production-ready MLOps system following the requirements in `.cursor/rules/requirement.md`.

## 🚀 Migration Overview

The migration transforms the project from a minimal ML application to a comprehensive MLOps platform with:

- **Experiment Tracking**: MLflow with Postgres backend & S3 artifact storage
- **Data Management**: DVC with S3 remote for version control
- **Model Serving**: FastAPI with MLflow Model Registry integration
- **Hyperparameter Optimization**: Optuna with MLflow nested runs
- **Orchestration**: Prefect flows for automated pipelines
- **Monitoring**: Prometheus/Grafana with Evidently drift detection
- **CI/CD**: GitHub Actions with canary deployment & rollback
- **Environment Management**: conda-lock for reproducibility

## 📁 Project Structure

```text
├── src/                          # Source code modules
│   ├── ingest/                   # Data ingestion
│   ├── fe/                       # Feature engineering
│   ├── train/                    # Model training
│   ├── eval/                     # Model evaluation
│   ├── serve/                    # Model serving (FastAPI)
│   ├── mlflow_integration.py     # MLflow tracking & registry
│   ├── optuna_integration.py     # Optuna HPO with MLflow
│   ├── prefect_flows.py          # Prefect orchestration
│   ├── drift_monitoring.py       # Evidently drift detection
│   ├── monitoring_config.py      # Prometheus/Grafana config
│   └── cd_pipeline.py           # CD with canary deployment
├── tests/                        # Unit tests
├── monitoring/                   # Monitoring configurations
│   ├── grafana/                  # Grafana dashboards & datasources
│   └── prometheus/               # Prometheus config & rules
├── infra/                        # Infrastructure as code
│   └── docker-compose.dev.yml    # Development stack
├── dvc.yaml                      # DVC pipeline definition
├── params.yaml                   # Pipeline parameters
├── environment.yml               # Conda environment
├── Dockerfile                    # Application container
├── Makefile                      # Development commands
└── .github/workflows/            # CI/CD pipelines
```

## 🛠️ Quick Start Guide

### ⚠️ **IMPORTANT NOTES**
- **Always run in conda environment**: `conda activate <your_env_name>`
- **Ensure Docker is running** before starting infrastructure
- **Check ports**: 5000, 8000, 9000, 9001 should not be in conflict

### 🔧 **Windows Setup (Important)**

#### **🚨 Vấn đề với `make` trên Windows:**
- `make` không có sẵn trên Windows PowerShell
- Cần cài đặt hoặc sử dụng lệnh trực tiếp

#### **✅ Giải pháp: Sử dụng lệnh trực tiếp**

**Thay vì Makefile commands, sử dụng:**

```bash
# Thay vì: make up
docker compose -f infra/docker-compose.dev.yml up -d --build

# Thay vì: make down  
docker compose -f infra/docker-compose.dev.yml down

# Thay vì: make train
python -m src.train.main

# Thay vì: make serve
uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 --reload

# Thay vì: make test
python -m pytest tests/ -v

# Thay vì: make mlflow-ui
mlflow ui --host 0.0.0.0 --port 5000

# Thay vì: make ui
streamlit run app.py --server.port 8501
```

#### **🔧 Cài đặt Make cho Windows (Tùy chọn):**

**Option 1: Chocolatey**
```powershell
# Cài Chocolatey (nếu chưa có)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Cài make
choco install make
```

**Option 2: Scoop**
```powershell
# Cài Scoop (nếu chưa có)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex

# Cài make
scoop install make
```

**Option 3: WSL (Windows Subsystem for Linux)**
```bash
# Cài WSL
wsl --install

# Trong WSL
sudo apt update
sudo apt install make
```

#### **Install Streamlit:**
```bash
# Install streamlit
conda activate <your_env_name>
pip install streamlit

# Verify streamlit installation
python -m streamlit --version
```

#### **Fix Docker Compose Issues:**
```bash
# On Windows, use docker-compose instead of docker compose
# Makefile has been updated for Windows compatibility
```

### ✅ **Quick Start Checklist**

```bash
# 🎯 COPY-PASTE COMMANDS (Run line by line)

# 1. Setup Environment
conda activate <your_env_name>
pip install -r requirements.txt

# 2. Start Infrastructure
make up
# Wait 30 seconds for services to start

# 3. Verify Services
curl http://localhost:5000/health
curl http://localhost:9000/minio/health/live

# 4. Run Training
make train

# 5. Start Serving
make serve

# 6. Test API
curl http://localhost:8000/health

# 7. Open UIs
# MLflow: http://localhost:5000
# FastAPI: http://localhost:8000/docs
```

### 1. Environment Setup

```bash
# ✅ STEP 1: Activate conda environment (REQUIRED)
conda activate <your_env_name>

# ✅ STEP 2: Check Python version
python --version  # Must be Python 3.11+

# ✅ STEP 3: Install dependencies
pip install -r requirements.txt

# ✅ STEP 4: Verify installation
python -c "import mlflow, dvc, fastapi; print('✅ All dependencies installed')"
```

### 2. Start Development Stack

```bash
# ✅ STEP 1: Check Docker
docker --version
docker compose --version

# ✅ STEP 2: Start infrastructure (Postgres + MinIO + MLflow)
make up

# 🔍 OR run manually if make doesn't work:
docker compose -f infra/docker-compose.dev.yml up -d --build

# ✅ STEP 3: Verify services are running
docker ps
# Should see: postgres, minio, mlflow containers

# ✅ STEP 4: Test connections
curl http://localhost:5000/health  # MLflow
curl http://localhost:9000/minio/health/live  # MinIO
```

### 3. Run Training Pipeline

```bash
# ✅ STEP 1: Check data is available
ls data/  # Should have CSV files

# ✅ STEP 2: Run training with MLflow tracking
make train

# 🔍 OR run manually:
python src/train/main.py

# ✅ STEP 3: Check MLflow UI
# Open browser: http://localhost:5000
# Should see experiment "ml_training" with runs
```

### 4. Start Model Serving

```bash
# ✅ STEP 1: Start FastAPI server
make serve

# 🔍 OR run manually:
uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 --reload

# ✅ STEP 2: Test API endpoints
curl http://localhost:8000/health
# Response: {"status": "ok", "model_loaded": true}

# ✅ STEP 3: Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Machine learning is fascinating"]}'
```

### 5. Access Services & UIs

| Service | URL | Username/Password | Description |
|---------|-----|-------------------|-------------|
| **MLflow UI** | http://localhost:5000 | - | Experiment tracking, Model registry |
| **FastAPI Docs** | http://localhost:8000/docs | - | Interactive API documentation |
| **API Health** | http://localhost:8000/health | - | Health check endpoint |
| **MinIO UI** | http://localhost:9001 | minio/minio123 | S3-compatible storage |
| **Prometheus** | http://localhost:9090 | - | Metrics collection |

### 6. Monitoring Setup (Optional)

```bash
# ✅ STEP 1: Generate monitoring configs
python src/monitoring_config.py

# ✅ STEP 2: Start monitoring stack
docker compose -f monitoring/docker-compose.monitoring.yml up -d

# ✅ STEP 3: Access Grafana
# URL: http://localhost:3000
# Username: admin
# Password: admin
```

### 🎯 **Expected Results**

#### ✅ **After successful execution:**

1. **Docker Containers**: 3 containers running
   ```bash
   docker ps
   # postgres, minio, mlflow
   ```

2. **MLflow UI**: http://localhost:5000
   - See experiment "ml_training"
   - Training runs with metrics
   - Model registered in Model Registry

3. **FastAPI API**: http://localhost:8000/docs
   - Health endpoint returns: `{"status": "ok", "model_loaded": true}`
   - Predict endpoint works with sample data
   - Metrics endpoint has Prometheus metrics

4. **MinIO UI**: http://localhost:9001
   - Bucket "mlflow" created
   - Artifacts from MLflow runs

5. **Verification**: 100% PASSED
   ```bash
   python verify_migration.py
   # Success Rate: 100.0%
   ```

#### ❌ **If there are errors:**
- Check logs: `docker compose logs -f`
- Restart services: `make down && make up`
- Check ports: `netstat -an | findstr :5000`

## 🚨 **Troubleshooting**

### **Common Windows Issues:**

#### **1. Error "make is not recognized":**
```bash
# Solution 1: Use direct commands instead of make
docker compose -f infra/docker-compose.dev.yml up -d --build
python -m src.train.main
streamlit run app.py --server.port 8501

# Solution 2: Install make (optional)
conda activate <your_env_name>
conda install make -c conda-forge

# Verify
make --version
```

#### **2. Error "streamlit is not recognized":**
```bash
# Solution: Install streamlit
conda activate <your_env_name>
pip install streamlit

# Use python -m streamlit instead of streamlit directly
python -m streamlit run app.py --server.port 8501
```

#### **3. Error "docker compose" not working:**
```bash
# Solution: Use docker-compose (Windows)
# Makefile has been updated for Windows compatibility
make up  # Will use docker-compose automatically

# Or use direct command:
docker-compose -f infra/docker-compose.dev.yml up -d --build
```

#### **4. Error "No module named streamlit":**
```bash
# Check environment
conda activate <your_env_name>
python -c "import streamlit; print('Streamlit OK')"

# If error, reinstall
pip install streamlit
```

#### **5. Error MLflow connection:**
```bash
# MLflow server not running
mlflow ui --host 0.0.0.0 --port 5000

# Or run training without MLflow server
python -c "from src.training_pipeline import TrainingPipeline; tp = TrainingPipeline(); print('OK')"
```

#### **6. Port conflicts on Windows:**
```bash
# Check ports in use
netstat -an | findstr :5000
netstat -an | findstr :8000
netstat -an | findstr :8501

# Kill process if needed
taskkill /F /PID <PID>
```

### **Debug Commands:**
```bash
# Check conda environment
conda info --envs

# Check Python packages
pip list | grep -E "(streamlit|mlflow|fastapi)"

# Check ports
netstat -an | findstr ":8501\|:5000\|:8000"

# Check Docker
docker ps
docker-compose ps
```

### **Alternative Commands (If make doesn't work):**

#### **Linux/Mac:**
```bash
# Instead of make ui
python -m streamlit run app.py --server.port 8501

# Instead of make train
python -m src.train.main

# Instead of make serve
uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 --reload

# Instead of make mlflow-ui
mlflow ui --host 0.0.0.0 --port 5000
```

#### **Windows:**
```bash
# Instead of make up
docker compose -f infra/docker-compose.dev.yml up -d --build

# Instead of make down
docker compose -f infra/docker-compose.dev.yml down

# Instead of make ui
streamlit run app.py --server.port 8501

# Instead of make train
python -m src.train.main

# Instead of make serve
uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 --reload

# Instead of make mlflow-ui
mlflow ui --host 0.0.0.0 --port 5000

# Instead of make test
python -m pytest tests/ -v

# Instead of make clean
rmdir /s data\processed data\features models artifacts metrics cache
docker compose -f infra/docker-compose.dev.yml down -v
```

## 🔧 Available Commands

### 📋 **Makefile Commands (Recommended)**

```bash
# 🚀 Infrastructure
make up              # Start development stack (Postgres + MinIO + MLflow)
make down            # Stop development stack
make restart         # Restart development stack

# 🤖 ML Pipeline
make train           # Run training pipeline with MLflow tracking
make ui              # Start Streamlit UI (port 8501)
make serve           # Start FastAPI server
make test            # Run unit tests
make lint            # Run code quality checks
make format          # Format code (black + isort)

# 📊 Data & Models
make dvc-repro       # Run DVC pipeline
make dvc-push        # Push data to DVC remote
make dvc-pull        # Pull data from DVC remote
make mlflow-ui       # Start MLflow UI

# 🧹 Cleanup
make clean           # Clean up generated files and caches
make clean-docker    # Clean Docker containers and images
make clean-data      # Clean data cache and artifacts

# 🔧 Environment
make conda-lock      # Generate conda lock file
make verify          # Run migration verification
```

### 🛠️ **Manual Commands (If Makefile doesn't work)**

#### **Linux/Mac Commands:**
```bash
# Infrastructure
docker compose -f infra/docker-compose.dev.yml up -d --build
docker compose -f infra/docker-compose.dev.yml down

# Training
python src/train/main.py

# UI
python -m streamlit run app.py --server.port 8501

# Serving
uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 --reload

# Testing
pytest tests/ -v
flake8 src/ tests/
black src/ tests/
isort src/ tests/

# DVC
dvc repro
dvc push
dvc pull

# Verification
python verify_migration.py
```

#### **🚀 Windows Quick Start Commands:**

```bash
# 1. Setup Environment
conda activate PJ3.1
pip install streamlit

# 2. Start Infrastructure (nếu có Docker)
docker compose -f infra/docker-compose.dev.yml up -d --build

# 3. Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# 4. Start Streamlit App
streamlit run app.py --server.port 8501

# 5. Start FastAPI Server
uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 --reload
```

#### **📊 Access URLs:**
- **Streamlit UI**: http://localhost:8501
- **MLflow UI**: http://localhost:5000
- **FastAPI**: http://localhost:8000
- **FastAPI Docs**: http://localhost:8000/docs

#### **💡 Windows Tips:**
1. **Luôn chạy trong môi trường conda**: `conda activate PJ3.1`
2. **Kiểm tra ports**: 5000, 8000, 8501 không bị conflict
3. **Sử dụng lệnh trực tiếp** thay vì `make` trên Windows
4. **Docker Desktop** cần chạy để sử dụng Docker commands
5. **Streamlit** đã được cài đặt và sẵn sàng sử dụng

#### **🎯 Windows Recommended Workflow:**
```bash
# 1. Setup
conda activate PJ3.1

# 2. Start MLflow (optional)
mlflow ui --host 0.0.0.0 --port 5000

# 3. Start Streamlit App
streamlit run app.py --server.port 8501

# 4. Open browser: http://localhost:8501
# 5. Use "Use CSV Files from Data Folder" option
# 6. Select dataset and train models
```

## 📊 Monitoring & Observability

### Prometheus Metrics

The FastAPI application exposes Prometheus metrics at `/metrics`:

- `http_requests_total` - Total HTTP requests by method and endpoint
- `http_request_duration_seconds` - Request duration histogram
- `predictions_total` - Total predictions made
- `prediction_duration_seconds` - Prediction duration histogram

### Grafana Dashboard

A comprehensive ML monitoring dashboard is available at `monitoring/grafana/dashboards/ml-monitoring-dashboard.json` with:

- Request rate and response time
- Prediction metrics
- Error rates
- Model performance metrics

### Drift Monitoring

Evidently-based drift detection monitors:

- Data drift between reference and current data
- Target drift for model performance
- Data quality metrics

## 🚀 CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline (`.github/workflows/ci-cd.yml`) includes:

1. **Testing**: Unit tests, linting, code formatting
2. **Data Validation**: Great Expectations checkpoints
3. **Training Test**: Limited training run with MLflow logging
4. **Deployment**: Staging deployment with smoke tests

### Canary Deployment

The CD pipeline supports canary deployment with:

- 10% traffic routing to new model
- 30-60 minute monitoring period
- Automatic promotion or rollback based on SLOs
- Prometheus metrics for decision making

## 🔄 Prefect Orchestration

Prefect flows automate the ML pipeline:

- **Data Ingestion**: Load and preprocess data
- **Data Validation**: Schema and quality checks
- **Training**: Model training with Optuna HPO
- **Evaluation**: Model performance assessment
- **Deployment**: Canary deployment with monitoring

Scheduled flows:
- ML Training Pipeline: Daily at 2 AM UTC
- Data Validation: Every 6 hours
- Model Evaluation: Daily at 3 AM UTC

## 📈 MLflow Integration

### Experiment Tracking

All training runs are logged to MLflow with:

- Parameters and hyperparameters
- Metrics (accuracy, precision, recall, F1)
- Artifacts (models, plots, reports)
- Git commit information
- DVC revision tags

### Model Registry

Models are registered with stages:

- **Staging**: Newly trained models
- **Production**: Promoted models
- **Archived**: Previous versions

### Model Serving

FastAPI loads models from MLflow Registry:

```python
# Load model from registry
model_uri = f"models:/{model_name}/{stage}"
model = mlflow.sklearn.load_model(model_uri)
```

## 🔍 Data Management

### DVC Pipeline

The `dvc.yaml` defines the ML pipeline:

1. **Ingest**: Load and preprocess data
2. **Feature Engineering**: Create features
3. **Training**: Train models with HPO
4. **Evaluation**: Evaluate model performance
5. **Serving**: Deploy model for serving

### Data Validation

Great Expectations and Pandera provide:

- Schema validation
- Data quality checks
- Range and freshness validation
- Automated reporting

## 🖥️ **UI Integration**

### **Streamlit UI with DataManager**

The project has successfully integrated **DataManager** and **TrainingPipeline** into the existing Streamlit UI:

#### **UI Features:**
- ✅ **Dataset Management**: Load datasets from `data/` folder
- ✅ **Dataset Preview**: Display detailed dataset information
- ✅ **Training Integration**: Use new TrainingPipeline with MLflow
- ✅ **MLflow Integration**: Display Run ID and Model Registry info
- ✅ **User-friendly**: Friendly interface with dataset selection

#### **How to use UI:**
```bash
# Start UI
make ui
# Or
python -m streamlit run app.py --server.port 8501

# Access: http://localhost:8501
```

#### **UI Workflow:**
1. **Step 1**: Select "Use CSV Files from Data Folder"
2. **Step 2**: Choose dataset from dropdown (heart, spam, arxiv, etc.)
3. **Step 3**: Configure preprocessing and model settings
4. **Step 4**: Click "Start Training" - uses new TrainingPipeline
5. **Step 5**: View results with MLflow integration

#### **Available Datasets:**
- **heart** (38KB) - Heart disease classification
- **Heart_disease_cleveland_new** (11KB) - Cleveland heart dataset  
- **2cls_spam_text_cls** (9MB) - Spam text classification
- **arxiv_dataset_backup** (388KB) - ArXiv abstracts
- **20250822-004129_sample-300_000Samples** (295MB) - Large sample dataset

#### **MLflow Integration in UI:**
- **Experiment Name**: `ml_training_ui`
- **Automatic Logging**: Parameters, metrics, models
- **Model Registry**: Automatic model registration
- **UI Links**: Direct links to MLflow UI (http://localhost:5000)

## 🛡️ Security & Best Practices

### Environment Variables

Sensitive configuration is managed via environment variables:

```bash
# Copy example environment file
cp env.example .env

# Edit with your values
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=minio123
```

### Secrets Management

- Database credentials via environment variables
- S3 credentials via AWS environment variables
- API keys via CI/CD secrets

## 📋 Migration Checklist

- [x] **FR-01**: DVC pipeline with `dvc.yaml` and `params.yaml`
- [x] **FR-02**: MLflow tracking & registry with Postgres + S3
- [x] **FR-03**: FastAPI serving with `/health`, `/predict`, `/metrics`
- [x] **FR-04**: Optuna HPO with MLflow nested runs
- [x] **FR-05**: Data validation with Great Expectations
- [x] **FR-06**: Prefect orchestration flows
- [x] **FR-07**: Prometheus/Grafana monitoring
- [x] **FR-08**: Canary deployment & rollback
- [x] **NFR**: Performance, reliability, security, reproducibility

## 🚨 Troubleshooting

### ❌ **Common Issues and Solutions**

#### 1. **Port Conflict Error**
```bash
# Check ports in use
netstat -an | findstr :5000
netstat -an | findstr :8000

# Kill process if needed
taskkill /F /PID <PID>

# Or change ports in docker-compose.dev.yml
```

#### 2. **Docker Won't Start**
```bash
# Check Docker is running
docker --version
docker ps

# Restart Docker Desktop if needed
# Or restart computer
```

#### 3. **Import Module Errors**
```bash
# Ensure you're in conda environment
conda activate <your_env_name>

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 4. **MLflow Connection Error**
```bash
# Check MLflow server
curl http://localhost:5000/health

# Restart MLflow container
docker compose restart mlflow

# Check logs
docker logs mlflow_server
```

#### 5. **Model Loading Error**
```bash
# Check if model exists in registry
# Access: http://localhost:5000
# Go to Model Registry tab

# Or check logs
docker logs fastapi_app
```

### 🔍 **Debug Commands**

```bash
# Check system health
python verify_migration.py

# Check all services
curl http://localhost:8000/health
curl http://localhost:5000/health
curl http://localhost:9000/minio/health/live

# View logs
docker compose logs -f
docker logs <container_name>

# Check containers
docker ps -a
docker images
```

## 🎯 Next Steps

### 🚀 **Immediate Actions**
1. **Test Full Pipeline**: Run end-to-end pipeline
2. **Add Your Data**: Replace sample data with real data
3. **Customize Models**: Add models suitable for your use case
4. **Setup Monitoring**: Configure alerts and dashboards

### 📈 **Production Ready**
1. **Production Deployment**: Deploy to production environment
2. **Feast Integration**: Add feature store for online features
3. **Advanced Monitoring**: Add custom business metrics
4. **A/B Testing**: Implement model comparison framework
5. **Multi-Environment**: Add staging and production environments

## 📚 Documentation & Resources

### 📖 **Project Documentation**
- **Migration Summary**: `info/MIGRATION_SUMMARY.md`
- **Code Cleanup**: `info/CODE_CLEANUP.md`
- **Architecture**: `info/MLOPS_ARCHITECTURE.md`
- **Deployment Guide**: `info/DEPLOYMENT_GUIDE.md`
- **API Documentation**: `info/API_DOCUMENTATION.md`

### 🔗 **External Documentation**
- [MLflow Documentation](https://mlflow.org/docs/)
- [DVC Documentation](https://dvc.org/doc)
- [Prefect Documentation](https://docs.prefect.io/)
- [Evidently Documentation](https://docs.evidently.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### 🎥 **Quick Reference**

#### **Essential URLs**
- MLflow UI: http://localhost:5000
- FastAPI Docs: http://localhost:8000/docs
- MinIO UI: http://localhost:9001 (minio/minio123)
- Grafana: http://localhost:3000 (admin/admin)

#### **Key Commands**
```bash
# Start everything
conda activate <your_env_name> && make up && make train && make serve

# Test API
curl http://localhost:8000/health

# Check MLflow
open http://localhost:5000
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run CI pipeline
5. Submit pull request

## 📄 License

This project is licensed under the MIT License.