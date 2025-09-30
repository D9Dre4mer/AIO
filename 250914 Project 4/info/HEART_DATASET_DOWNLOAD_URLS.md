# Heart Dataset - Đường dẫn Download trên Internet

## Nguồn gốc chính thức

### 1. UCI Machine Learning Repository (Nguồn gốc)
- **URL**: https://archive.ics.uci.edu/ml/datasets/Heart%2BDisease
- **Mô tả**: Bộ dữ liệu gốc từ UCI, bao gồm 4 datasets từ Cleveland, Hungary, Switzerland và VA Long Beach
- **Cleveland dataset**: Thường được sử dụng nhất trong nghiên cứu ML
- **Format**: Có thể download dưới dạng ZIP hoặc individual files

### 2. IEEE DataPort
- **URL**: https://ieee-dataport.org/open-access/heart-disease-dataset-comprehensive
- **Mô tả**: Dataset tổng hợp từ 5 nguồn (Cleveland, Hungary, Switzerland, Long Beach VA, Statlog)
- **Kích thước**: 1,190 samples với 11 features chung
- **Ưu điểm**: Dataset đã được làm sạch và chuẩn hóa

### 3. Hugging Face Datasets
- **URL**: https://huggingface.co/datasets/nezahatkorkmaz/heart-disease-dataset
- **Mô tả**: Heart disease dataset với các features như age, sex, blood pressure, cholesterol
- **Format**: Có thể load trực tiếp bằng Hugging Face datasets library
- **Code example**:
```python
from datasets import load_dataset
dataset = load_dataset("nezahatkorkmaz/heart-disease-dataset")
```

### 4. GitHub Gist
- **URL**: https://gist.github.com/notreallyme2/8d20f6b5bcc3606e0541cbdf2ee0a7a6
- **Mô tả**: Copy của UCI Heart Disease dataset trên GitHub
- **Format**: CSV files có thể download trực tiếp

### 5. Data.World
- **URL**: https://data.world/uci/heart-disease
- **Mô tả**: UCI Heart Disease dataset trên platform Data.World
- **Features**: Có thể explore và download dataset

## Kaggle Datasets (Cần tìm kiếm cụ thể)

### Các Kaggle datasets phổ biến:
1. **Heart Disease Dataset** - Tìm kiếm trên kaggle.com
2. **Heart Disease Prediction** - Nhiều competitions và datasets
3. **UCI Heart Disease** - Copy của dataset gốc

### Cách tìm trên Kaggle:
1. Truy cập https://kaggle.com/datasets
2. Search "heart disease"
3. Filter theo "Most Downloaded" hoặc "Most Viewed"

## Download Scripts

### Python script để download từ UCI:
```python
import urllib.request
import zipfile
import os

def download_uci_heart_dataset():
    """Download Heart Disease dataset từ UCI"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    
    # Download các files
    files = [
        "processed.cleveland.data",
        "processed.hungarian.data", 
        "processed.switzerland.data",
        "processed.va.data"
    ]
    
    for file in files:
        file_url = url + file
        urllib.request.urlretrieve(file_url, f"cache/{file}")
        print(f"Downloaded: {file}")

# Sử dụng
download_uci_heart_dataset()
```

### Download từ Hugging Face:
```python
from datasets import load_dataset
import pandas as pd

def download_huggingface_heart_dataset():
    """Download từ Hugging Face"""
    dataset = load_dataset("nezahatkorkmaz/heart-disease-dataset")
    
    # Convert to pandas
    df = dataset['train'].to_pandas()
    
    # Save to cache
    df.to_csv('cache/heart_huggingface.csv', index=False)
    print(f"Downloaded: {len(df)} samples")

# Sử dụng
download_huggingface_heart_dataset()
```

## So sánh các nguồn

| Nguồn | Kích thước | Format | Missing Values | Ưu điểm |
|-------|------------|--------|----------------|---------|
| UCI Original | 4 datasets | .data | Có ('?') | Nguồn gốc, đầy đủ |
| IEEE DataPort | 1,190 samples | CSV | Đã xử lý | Tổng hợp, sạch |
| Hugging Face | Variable | Dataset | Depends | Dễ sử dụng |
| GitHub Gist | Variable | CSV | Depends | Nhanh, đơn giản |

## Khuyến nghị

### Cho nghiên cứu:
- **UCI Original**: Nguồn gốc, đầy đủ metadata
- **IEEE DataPort**: Dataset tổng hợp, đã được làm sạch

### Cho development:
- **Hugging Face**: Dễ integrate vào code
- **GitHub Gist**: Download nhanh, đơn giản

### Cho production:
- **UCI Cleveland dataset**: Dataset chuẩn, được sử dụng rộng rãi

## Lưu ý quan trọng

1. **Missing Values**: UCI dataset gốc có missing values được ký hiệu bằng '?'
2. **Data Processing**: Cần xử lý missing values trước khi sử dụng
3. **Column Names**: UCI dataset không có header, cần thêm column names
4. **Encoding**: Một số datasets có thể có encoding khác nhau

## Ngày cập nhật
Ngày: 2024-12-19
Nguồn: Web search và UCI ML Repository
