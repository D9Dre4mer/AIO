# RAPIDS cuML Installation Guide

## Tổng quan
Hướng dẫn cài đặt RAPIDS cuML để sử dụng KMeans trên GPU với cơ chế fallback về CPU.

## Yêu cầu hệ thống

### GPU Requirements
- **NVIDIA GPU**: Compute Capability 6.0+ (Pascal, Volta, Turing, Ampere, Ada Lovelace, Hopper)
- **CUDA**: 11.2+ (khuyến nghị 12.6)
- **Driver**: NVIDIA Driver 450.80.02+

### Software Requirements
- **Python**: 3.9, 3.10, hoặc 3.11
- **Operating System**: Linux (Ubuntu 20.04+), Windows 10/11, hoặc macOS
- **Memory**: Tối thiểu 8GB RAM, khuyến nghị 16GB+

## Cài đặt RAPIDS cuML

### Phương pháp 1: Conda (Khuyến nghị)

#### Cài đặt với GPU support:
```bash
# Tạo environment mới
conda create -n rapids-24.08 python=3.11

# Kích hoạt environment
conda activate rapids-24.08

# Cài đặt RAPIDS cuML với GPU support
conda install -c rapidsai -c nvidia -c conda-forge \
    cuml=24.08 python=3.11 cuda-version=12.6
```

#### Cài đặt CPU-only (fallback):
```bash
# Tạo environment mới
conda create -n rapids-cpu-24.08 python=3.11

# Kích hoạt environment
conda activate rapids-cpu-24.08

# Cài đặt RAPIDS cuML CPU-only
conda install -c rapidsai -c conda-forge \
    cuml-cpu=24.08 python=3.11
```

### Phương pháp 2: Pip (Thử nghiệm)

```bash
# Cài đặt RAPIDS cuML qua pip
pip install cuml-cpu==24.08.0

# Hoặc cho GPU (nếu có CUDA 12.6)
pip install cuml==24.08.0
```

## Kiểm tra cài đặt

### Test script:
```python
# Chạy test script
python test_rapids_kmeans.py
```

### Manual test:
```python
import cuml
from cuml.cluster import KMeans
import numpy as np

# Test basic functionality
X = np.random.rand(100, 10).astype(np.float32)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
predictions = kmeans.predict(X)
print(f"Predictions: {predictions}")
```

## Cấu hình trong project

### File config.py:
```python
# RAPIDS cuML Settings
ENABLE_RAPIDS_CUML = True        # Enable RAPIDS cuML for GPU acceleration
RAPIDS_FALLBACK_TO_CPU = True    # Fallback to CPU if GPU not available
RAPIDS_AUTO_DETECT_DEVICE = True # Automatically detect best device (GPU/CPU)
```

### File requirements.txt:
```
# RAPIDS cuML for GPU-accelerated machine learning
# Note: Install with conda for best compatibility:
# conda install -c rapidsai -c nvidia -c conda-forge cuml=24.08 python=3.11 cuda-version=12.6
# For CPU-only fallback: conda install -c rapidsai -c conda-forge cuml-cpu=24.08
cuml-cpu>=24.08.0  # CPU fallback version
```

## Sử dụng trong code

### KMeansModel tự động sử dụng RAPIDS cuML:
```python
from models.clustering.kmeans_model import KMeansModel

# Tự động detect và sử dụng GPU nếu có
kmeans = KMeansModel(n_clusters=5)
kmeans.fit(X, y)
predictions = kmeans.predict(X_test)
```

### Thông tin model:
```python
info = kmeans.get_model_info()
print(f"Model type: {info['model_type']}")  # RAPIDS cuML KMeans hoặc scikit-learn KMeans
print(f"Device type: {info['device_type']}")  # gpu hoặc cpu
print(f"RAPIDS enabled: {info['rapids_enabled']}")
```

## Troubleshooting

### Lỗi thường gặp:

#### 1. ImportError: No module named 'cuml'
```bash
# Giải pháp: Cài đặt lại RAPIDS cuML
conda install -c rapidsai -c nvidia -c conda-forge cuml=24.08
```

#### 2. CUDA out of memory
```python
# Giải pháp: Giảm kích thước batch hoặc sử dụng CPU
from cuml.common.device_selection import set_global_device_type
set_global_device_type('cpu')
```

#### 3. GPU not detected
```python
# Kiểm tra CUDA availability
from cuml.common import cuda
print(f"CUDA available: {cuda.is_available()}")
print(f"GPU count: {cuda.gpu_count()}")
```

### Performance tuning:

#### 1. Memory optimization:
```python
# Sử dụng sparse matrices cho dataset lớn
from scipy import sparse
X_sparse = sparse.csr_matrix(X_dense)
kmeans.fit(X_sparse)
```

#### 2. Batch processing:
```python
# Xử lý dataset lớn theo batch
for batch in batches:
    kmeans.partial_fit(batch)
```

## Benchmark Performance

### Test performance:
```python
from utils.rapids_detector import benchmark_kmeans_performance

results = benchmark_kmeans_performance(
    n_clusters=5,
    n_samples=10000,
    n_features=100
)

print(f"Speedup: {results['speedup']:.2f}x")
print(f"Device: {results['device_type']}")
```

### Expected performance:
- **GPU**: 2-10x speedup tùy thuộc vào kích thước dataset
- **CPU**: Tương đương hoặc nhanh hơn scikit-learn một chút
- **Memory**: RAPIDS cuML cần nhiều memory hơn scikit-learn

## Tài liệu tham khảo

- [RAPIDS cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
- [RAPIDS cuML Installation Guide](https://docs.rapids.ai/install/)
- [RAPIDS cuML KMeans API](https://docs.rapids.ai/api/cuml/stable/api.html#cuml.cluster.KMeans)
- [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)

## Support

Nếu gặp vấn đề, hãy:
1. Kiểm tra log trong `test_rapids_kmeans.py`
2. Xem troubleshooting section ở trên
3. Kiểm tra compatibility matrix của RAPIDS cuML
4. Tạo issue trên GitHub repository
