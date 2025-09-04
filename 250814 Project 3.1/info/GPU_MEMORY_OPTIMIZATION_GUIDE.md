# 🔧 GPU & Memory Optimization Guide

## 📋 Tổng quan

Hướng dẫn này giải thích cách quản lý GPU optimization và memory usage trong Topic Modeling Auto Classifier.

---

## 🔍 Vấn đề chuyển đổi Sparse → Dense Matrix

### ❌ **Tại sao có việc chuyển đổi?**

1. **GPU không hỗ trợ Sparse Matrix**:
   - BoW và TF-IDF trả về sparse matrices (tiết kiệm memory)
   - GPU acceleration (PyTorch, cuML) chỉ hỗ trợ dense arrays
   - Phải chuyển đổi sparse → dense để sử dụng GPU

2. **Memory tăng đáng kể**:
   - Sparse matrix: ~47KB cho 1K samples
   - Dense matrix: ~8.6MB cho 1K samples (**185x tăng**)
   - Với dataset lớn: có thể tăng từ 3GB → 1.5TB!

3. **Hiệu suất không cải thiện**:
   - Chỉ có **KNN model** thực sự sử dụng GPU
   - Các models khác (SVM, Naive Bayes, Logistic Regression) vẫn dùng CPU
   - Chuyển đổi tốn thời gian nhưng không có lợi ích

---

## ⚙️ Cấu hình GPU Optimization

### 📁 **File cấu hình**: `config.py`

```python
# GPU Optimization Settings
ENABLE_GPU_OPTIMIZATION = False  # Use sparse matrices (memory efficient)
FORCE_DENSE_CONVERSION = False   # Force sparse->dense conversion for GPU
```

### 🎛️ **Các chế độ hoạt động**:

| Chế độ | ENABLE_GPU_OPTIMIZATION | FORCE_DENSE_CONVERSION | Kết quả |
|--------|------------------------|------------------------|---------|
| **Memory Optimization** | `False` | `False` | Sparse matrices, CPU, tiết kiệm memory |
| **GPU Optimization** | `True` | `False` | Dense matrices, GPU acceleration |
| **Dense Only** | `False` | `True` | Dense matrices, không GPU |
| **Force Dense** | `True` | `True` | Dense matrices, GPU acceleration |

---

## 🛠️ Quản lý cấu hình

### 📜 **Script quản lý**: `gpu_config_manager.py`

```bash
# Hiển thị cấu hình hiện tại
python gpu_config_manager.py show

# Bật GPU optimization (dense matrices + GPU)
python gpu_config_manager.py gpu

# Bật memory optimization (sparse matrices + CPU)
python gpu_config_manager.py memory

# Bật dense conversion (dense matrices, không GPU)
python gpu_config_manager.py dense

# Tắt dense conversion (sparse matrices)
python gpu_config_manager.py sparse
```

### 🔄 **Chuyển đổi nhanh**:

```bash
# Cho dataset nhỏ (< 10K samples) - sử dụng GPU
python gpu_config_manager.py gpu

# Cho dataset lớn (> 50K samples) - tiết kiệm memory
python gpu_config_manager.py memory
```

---

## 📊 So sánh hiệu suất

### 💾 **Memory Usage**:

| Dataset Size | Sparse (KB) | Dense (KB) | Memory Increase |
|--------------|-------------|------------|-----------------|
| 1,000 samples | 47 | 8,688 | **185x** |
| 5,000 samples | 272 | 142,656 | **525x** |
| 10,000 samples | 3,189 | 1,518,594 | **476x** |

### ⚡ **Computation Speed**:

| Model | Sparse (CPU) | Dense (GPU) | Recommendation |
|-------|--------------|-------------|----------------|
| **KNN** | ⚡ Fast | 🚀 Very Fast | Dense + GPU |
| **SVM** | ⚡ Fast | 🐌 Slow | Sparse + CPU |
| **Naive Bayes** | ⚡ Fast | ⚡ Fast | Sparse + CPU |
| **Logistic Regression** | ⚡ Fast | ⚡ Fast | Sparse + CPU |
| **Decision Tree** | ⚡ Fast | 🚀 Fast (Linux only) | Sparse + CPU |

### 🎯 **Accuracy Impact**:
- **✅ KHÔNG ảnh hưởng tới accuracy**
- **Predictions**: Identical hoặc gần như identical
- **Model Performance**: Không thay đổi

---

## 🎯 Khuyến nghị sử dụng

### ✅ **Sử dụng Memory Optimization (Mặc định)**:
- **Dataset lớn** (> 10K samples)
- **Memory hạn chế** (< 16GB RAM)
- **Tất cả models** (trừ KNN)
- **Production environment**

### ✅ **Sử dụng GPU Optimization**:
- **Dataset nhỏ** (< 10K samples)
- **Memory đủ** (> 32GB RAM)
- **Chỉ KNN model**
- **Development/testing**

### ⚠️ **Tránh Dense Conversion**:
- **Dataset rất lớn** (> 100K samples)
- **Memory thấp** (< 8GB RAM)
- **Không có GPU mạnh**

---

## 🔧 Troubleshooting

### ❌ **Lỗi "Out of Memory"**:
```bash
# Chuyển sang memory optimization
python gpu_config_manager.py memory
```

### ❌ **GPU không hoạt động**:
```bash
# Kiểm tra cấu hình
python gpu_config_manager.py show

# Bật GPU optimization
python gpu_config_manager.py gpu
```

### ❌ **Training quá chậm**:
```bash
# Sử dụng sparse matrices (nhanh hơn cho hầu hết models)
python gpu_config_manager.py memory
```

---

## 📈 Monitoring

### 🔍 **Kiểm tra logs**:

**Memory Optimization Mode**:
```
💾 MEMORY OPTIMIZATION MODE...
   • Using sparse matrices (BoW, TF-IDF) for memory efficiency
   • GPU acceleration disabled to save memory
   • Models will use CPU with sparse matrices (faster for most cases)

📊 Using BoW sparse matrix format for memory efficiency
💾 Memory saved: Avoiding dense conversion (GPU optimization disabled)
```

**GPU Optimization Mode**:
```
🚀 ENABLING GPU OPTIMIZATION...
   • Converting sparse matrices (BoW, TF-IDF) to dense arrays
   • This enables GPU acceleration for all vectorization methods
   • Memory usage will increase but performance will improve

🔄 Converting BoW from sparse to dense for GPU acceleration...
✅ BoW converted to dense arrays for GPU
```

---

## 🎉 Kết luận

### ✅ **Mặc định**: Memory Optimization
- Sử dụng sparse matrices
- Tiết kiệm memory tối đa
- Phù hợp cho hầu hết trường hợp

### 🚀 **Khi cần**: GPU Optimization
- Chỉ cho KNN model
- Dataset nhỏ
- Memory đủ

### 💡 **Lời khuyên**:
- **Bắt đầu với Memory Optimization**
- **Chỉ chuyển sang GPU khi cần thiết**
- **Monitor memory usage**
- **Test với dataset thực tế**
