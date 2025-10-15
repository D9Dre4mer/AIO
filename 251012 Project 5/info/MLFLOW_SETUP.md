# MLflow Setup Guide

## ✅ MLflow đã được cài đặt và cấu hình thành công!

### 📋 Trạng thái hiện tại:
- **MLflow Version**: 3.4.0 ✅
- **MLflow Server**: Đang chạy trên http://localhost:5000 ✅
- **Database**: SQLite (mlflow.db) ✅
- **Artifacts**: ./mlruns ✅

### 🚀 Cách khởi động MLflow Server:

#### Phương pháp 1: Sử dụng script tự động
```bash
conda activate PJ3.1
python start_mlflow.py
```

#### Phương pháp 2: Khởi động thủ công
```bash
conda activate PJ3.1
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### 🌐 Truy cập MLflow UI:
- **MLflow UI**: http://localhost:5000
- **Tracking URI**: http://localhost:5000

### 📊 Các tính năng đã sẵn sàng:
- ✅ Experiment Tracking
- ✅ Model Registry
- ✅ Artifact Storage
- ✅ Parameter & Metric Logging
- ✅ Model Versioning

### 🔧 Cấu hình trong ứng dụng:
MLflow đã được cấu hình tự động trong `src/mlflow_integration.py`:
```python
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
```

### 🎯 Bước tiếp theo:
1. **Khởi động MLflow server** (nếu chưa chạy):
   ```bash
   python start_mlflow.py
   ```

2. **Chạy Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Kiểm tra kết nối**: Trong sidebar của app sẽ hiển thị "MLflow: ✅ Connected"

### 🛠️ Troubleshooting:

#### Nếu MLflow không kết nối:
1. **Kiểm tra server có chạy không**:
   ```bash
   curl http://localhost:5000/health
   ```

2. **Khởi động lại server**:
   ```bash
   python start_mlflow.py
   ```

3. **Kiểm tra port 5000 có bị chiếm không**:
   ```bash
   netstat -an | findstr :5000
   ```

#### Nếu gặp lỗi encoding:
- Đảm bảo terminal hỗ trợ UTF-8
- Hoặc sử dụng PowerShell thay vì Command Prompt

### 📚 Tài liệu tham khảo:
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
