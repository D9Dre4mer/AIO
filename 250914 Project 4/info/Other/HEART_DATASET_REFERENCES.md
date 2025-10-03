# Heart Dataset References và Đường dẫn trong Codebase

## Tổng quan
Tài liệu này liệt kê tất cả các tham chiếu và đường dẫn đến Heart datasets trong codebase.

## Datasets có sẵn trong Cache
```
cache/
├── Heart_disease_cleveland_new.csv  (303 rows × 14 columns)
├── heart.csv                        (1025 rows × 14 columns)
└── heart+disease.zip               (Archive file)
```

## Files tham chiếu đến Heart Datasets

### 1. Scripts phân tích
- `compare_heart_datasets.py` - Script so sánh hai datasets
- `check_missing_data.py` - Script kiểm tra missing values
- `deep_missing_analysis.py` - Script phân tích sâu về missing values

### 2. Training scripts
- `auto_train_heart_dataset.py` - Script training tự động với heart dataset
  - Load dataset từ: `cache/heart.csv`
  - Function: `load_heart_dataset(sample_size=None)`

### 3. Model cache directories
```
cache/models/
├── fixed_gradient_boosting/heart_dataset_fixed/
├── fixed_logistic_regression/heart_dataset_fixed/
├── fixed_random_forest/heart_dataset_fixed/
├── random_forest_heart_test/heart_dataset/
└── test_random_forest_step5/heart_dataset_step5/
```

### 4. Training results
- `cache/training_results/comprehensive_heart_dataset_results.json`

## Đường dẫn cụ thể trong Code

### auto_train_heart_dataset.py
```python
def load_heart_dataset(sample_size: int = None):
    df = pd.read_csv('cache/heart.csv')  # Line 44
```

### compare_heart_datasets.py
```python
def load_datasets():
    cleveland_df = pd.read_csv('C:/Users/User/OneDrive/Cloud/0. Studying/250516 AIO2025/z. Git/AIO/250914 Project 4/cache/Heart_disease_cleveland_new.csv')
    heart_df = pd.read_csv('C:/Users/User/OneDrive/Cloud/0. Studying/250516 AIO2025/z. Git/AIO/250914 Project 4/cache/heart.csv')
```

## Dataset Usage Patterns

### 1. Primary Usage
- **heart.csv** được sử dụng làm dataset chính trong `auto_train_heart_dataset.py`
- **Heart_disease_cleveland_new.csv** chỉ được sử dụng trong các script phân tích

### 2. Model Training
- Các models được train với heart dataset được lưu trong:
  - `heart_dataset_fixed/` - Fixed configurations
  - `heart_dataset/` - Standard configurations
  - `heart_dataset_step5/` - Step 5 configurations

### 3. Dataset Fingerprints
Các model fingerprints sử dụng:
- `heart_fixed_fixed_gradient_boosting`
- `heart_fixed_fixed_logistic_regression`
- `heart_fixed_fixed_random_forest`
- `heart_step5_test`
- `heart_test`

## Recommendations

### 1. Dataset Selection
- **Cho production**: Sử dụng `Heart_disease_cleveland_new.csv` (sạch, không missing values)
- **Cho research**: Có thể sử dụng `heart.csv` để nghiên cứu missing value handling

### 2. Code Updates
- Cập nhật `auto_train_heart_dataset.py` để sử dụng Cleveland dataset
- Thêm option để chọn dataset trong UI

### 3. Documentation
- Cập nhật README với thông tin về hai datasets
- Thêm warning về missing values trong heart.csv

## Files được tạo trong quá trình phân tích
1. `info/HEART_DATASETS_ANALYSIS_REPORT.md` - Báo cáo phân tích chi tiết
2. `compare_heart_datasets.py` - Script so sánh
3. `check_missing_data.py` - Script kiểm tra missing values
4. `deep_missing_analysis.py` - Script phân tích sâu
5. `heart_datasets_comparison.png` - Visualization (nếu được tạo)

## Ngày cập nhật
Ngày: 2024-12-19
Người phân tích: AI Assistant
