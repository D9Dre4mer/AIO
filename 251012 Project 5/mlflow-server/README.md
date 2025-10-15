# MLflow Server Setup Guide

## 📋 Tổng quan

Thư mục này chứa tất cả các file cần thiết để cài đặt và cấu hình MLflow server cho dự án ML của bạn. MLflow server sẽ chạy độc lập và có thể được truy cập từ các ứng dụng khác nhau.

## 🚀 Cài đặt nhanh

### 1. Cài đặt dependencies
```bash
# Linux/Mac
pip install -r requirements.txt

# Windows
pip install -r requirements.txt
```

### 2. Khởi động server
```bash
# Linux/Mac
chmod +x start_server.sh
./start_server.sh

# Windows
start_server.bat
```

### 3. Truy cập MLflow UI
Mở trình duyệt và truy cập: http://localhost:5000

## 📁 Cấu trúc thư mục

```
mlflow-server/
├── requirements.txt          # Python dependencies
├── start_server.sh           # Linux/Mac startup script
├── start_server.bat          # Windows startup script
├── mlflow.conf               # Configuration file
├── docker-compose.yml        # Docker setup
├── configure_client.py       # Client configuration script
├── health_check.py           # Health monitoring script
├── monitoring/               # Monitoring configuration
│   ├── prometheus.yml        # Prometheus config
│   └── mlflow_rules.yml      # Alert rules
├── mlruns/                   # Artifact storage (auto-created)
├── logs/                     # Log files (auto-created)
└── config/                   # Configuration files (auto-created)
```

## ⚙️ Cấu hình

### Environment Variables

Bạn có thể tùy chỉnh cấu hình bằng cách set các biến môi trường:

```bash
# Server configuration
export MLFLOW_HOST=0.0.0.0
export MLFLOW_PORT=5000
export MLFLOW_WORKERS=4

# Database configuration
export MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db

# Artifact storage
export MLFLOW_ARTIFACT_ROOT=./mlruns
```

### Cấu hình Database

#### SQLite (Mặc định - cho development)
```bash
export MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
```

#### PostgreSQL (Khuyến nghị cho production)
```bash
export MLFLOW_BACKEND_STORE_URI=postgresql://username:password@localhost:5432/mlflow
```

#### MySQL
```bash
export MLFLOW_BACKEND_STORE_URI=mysql+pymysql://username:password@localhost:3306/mlflow
```

### Cấu hình Artifact Storage

#### Local filesystem (Mặc định)
```bash
export MLFLOW_ARTIFACT_ROOT=./mlruns
```

#### AWS S3
```bash
export MLFLOW_ARTIFACT_ROOT=s3://your-bucket-name/mlflow-artifacts
```

#### Azure Blob Storage
```bash
export MLFLOW_ARTIFACT_ROOT=wasbs://container@account.blob.core.windows.net/mlflow-artifacts
```

## 🐳 Docker Setup

### Sử dụng Docker Compose

```bash
# Development (SQLite)
docker-compose up -d

# Production (PostgreSQL)
docker-compose --profile production up -d

# Với monitoring
docker-compose --profile monitoring up -d
```

### Build custom image

```bash
docker build -t mlflow-server .
docker run -p 5000:5000 mlflow-server
```

## 🔧 Cấu hình Client

### 1. Chạy script cấu hình
```bash
python configure_client.py
```

### 2. Sử dụng trong Python code
```python
import mlflow_client_config  # Import config
import mlflow

# MLflow đã được cấu hình tự động
with mlflow.start_run():
    mlflow.log_param("param1", "value1")
    mlflow.log_metric("metric1", 0.95)
```

### 3. Cấu hình thủ công
```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("my_experiment")

# Start run
with mlflow.start_run():
    # Your ML code here
    pass
```

## 📊 Monitoring

### Health Check
```bash
python health_check.py
```

### Prometheus Metrics
- Prometheus UI: http://localhost:9090
- MLflow metrics endpoint: http://localhost:5000/metrics

### Log Files
- Access logs: `logs/access.log`
- Error logs: `logs/error.log`
- Health check logs: `logs/health_check.log`

## 🔐 Authentication (Optional)

Để bật authentication, thêm vào `mlflow.conf`:

```bash
MLFLOW_AUTH_ENABLED=true
MLFLOW_AUTH_USERNAME=admin
MLFLOW_AUTH_PASSWORD=your_password
```

## 🌐 Production Deployment

### 1. Sử dụng Gunicorn
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 mlflow.server:app
```

### 2. Sử dụng Nginx (Reverse Proxy)
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. SSL/HTTPS
```bash
# Sử dụng Let's Encrypt
certbot --nginx -d your-domain.com
```

## 🛠️ Troubleshooting

### Port đã được sử dụng
```bash
# Linux/Mac
lsof -ti:5000 | xargs kill -9

# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Database connection issues
```bash
# Test database connection
python -c "import sqlalchemy; print('Database OK')"
```

### Permission issues
```bash
# Linux/Mac
chmod +x start_server.sh
chmod 755 mlruns/
```

## 📚 API Documentation

### REST API Endpoints
- Health check: `GET /health`
- Experiments: `GET /api/2.0/mlflow/experiments/search`
- Runs: `GET /api/2.0/mlflow/runs/search`
- Models: `GET /api/2.0/mlflow/registered-models/search`

### Python API
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiments = client.search_experiments()
runs = client.search_runs(experiment_ids=[experiment.experiment_id for experiment in experiments])
```

## 🔄 Backup và Restore

### Backup
```bash
# Backup database
cp mlflow.db mlflow_backup_$(date +%Y%m%d).db

# Backup artifacts
tar -czf mlruns_backup_$(date +%Y%m%d).tar.gz mlruns/
```

### Restore
```bash
# Restore database
cp mlflow_backup_20231014.db mlflow.db

# Restore artifacts
tar -xzf mlruns_backup_20231014.tar.gz
```

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra logs trong thư mục `logs/`
2. Chạy health check: `python health_check.py`
3. Kiểm tra port và firewall settings
4. Xem MLflow documentation: https://mlflow.org/docs/

## 🎯 Next Steps

1. **Khởi động MLflow server**
2. **Cấu hình client** bằng `configure_client.py`
3. **Tích hợp vào dự án** của bạn
4. **Thiết lập monitoring** (optional)
5. **Deploy lên production** (optional)

---

**Chúc bạn thành công với MLflow! 🚀**
