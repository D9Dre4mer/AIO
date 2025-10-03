# Heart Dataset - Nguồn gốc chính xác từ Kaggle

## Nguồn gốc thực tế của heart.csv

### Kaggle Dataset
- **URL**: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data
- **Uploader**: johnsmith88
- **Mô tả**: Heart Disease Dataset từ năm 1988

### Thông tin chi tiết
- **Năm**: 1988
- **Nguồn**: UCI Machine Learning Repository (4 databases)
- **Databases**: Cleveland, Hungary, Switzerland, và Long Beach V
- **Tổng attributes**: 76 attributes
- **Attributes sử dụng**: 14 attributes (subset được publish)
- **Target field**: 
  - 0 = no disease (không có bệnh tim)
  - 1 = disease (có bệnh tim)

## Phân tích lại dựa trên thông tin mới

### 1. Dataset Structure
```
heart.csv (1025 rows × 14 columns)
├── 14 attributes được chọn từ 76 attributes gốc
├── Target: 0 (no disease) / 1 (disease)
└── Nguồn: 4 databases từ UCI (1988)
```

### 2. Giải thích Missing Values
Dựa trên thông tin này, missing values trong heart.csv có thể do:

**A. Data Aggregation từ 4 databases:**
- Cleveland database: Có thể có missing values
- Hungary database: Có thể có missing values  
- Switzerland database: Có thể có missing values
- Long Beach V database: Có thể có missing values

**B. Attribute Selection:**
- Chỉ chọn 14/76 attributes
- Một số attributes có thể có missing values
- Missing values được encode thành 0 và 4

### 3. So sánh với Cleveland dataset
```
Heart_disease_cleveland_new.csv (303 rows)
├── Chỉ từ Cleveland database
├── Đã được làm sạch (no missing values)
└── Dataset nhỏ hơn, sạch hơn

heart.csv (1025 rows)  
├── Từ 4 databases tổng hợp
├── Có missing values (ca=4, thal=0)
├── Dataset lớn hơn, có missing values
└── Đã được augment/duplicate
```

## Cập nhật kết luận

### Quy trình thực tế:
```
UCI Original (1988) - 4 databases
         ↓
Kaggle Dataset (johnsmith88) - 14 attributes selected
         ↓
heart.csv - Aggregated + Missing values encoded
         ↓
Heart_disease_cleveland_new.csv - Cleveland subset cleaned
```

### Giải thích Missing Values:
1. **ca=4**: Có thể là missing value từ Hungary/Switzerland/Long Beach databases
2. **thal=0**: Có thể là missing value từ các databases khác Cleveland
3. **Data augmentation**: heart.csv được duplicate để tăng kích thước

### Khuyến nghị cập nhật:

#### 1. Dataset Selection
- **Cho research**: Sử dụng heart.csv (đầy đủ 4 databases)
- **Cho production**: Sử dụng Heart_disease_cleveland_new.csv (sạch, chỉ Cleveland)
- **Cho comparison**: Sử dụng cả hai để so sánh

#### 2. Missing Value Handling
- **ca=4**: Treat as missing value, có thể impute hoặc remove
- **thal=0**: Treat as missing value, có thể impute hoặc remove
- **Strategy**: Có thể sử dụng median/mode cho missing values

#### 3. Data Preprocessing
```python
def preprocess_heart_dataset(df):
    """Preprocess heart dataset với missing value handling"""
    
    # Handle missing values
    df.loc[df['ca'] == 4, 'ca'] = np.nan
    df.loc[df['thal'] == 0, 'thal'] = np.nan
    
    # Impute missing values
    df['ca'] = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
    
    return df
```

## Files cần cập nhật

### 1. auto_train_heart_dataset.py
```python
def load_heart_dataset(sample_size: int = None):
    """Load heart dataset với preprocessing"""
    df = pd.read_csv('cache/heart.csv')
    
    # Handle missing values
    df.loc[df['ca'] == 4, 'ca'] = np.nan
    df.loc[df['thal'] == 0, 'thal'] = np.nan
    
    # Impute missing values
    df['ca'] = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
    
    return df, feature_columns, label_column
```

### 2. Documentation Updates
- Cập nhật README với thông tin về nguồn gốc
- Thêm warning về missing values
- Hướng dẫn preprocessing

## Kết luận

**heart.csv** là dataset tổng hợp từ 4 databases UCI (1988) với missing values được encode. **Heart_disease_cleveland_new.csv** là subset Cleveland đã được làm sạch. Cả hai đều có giá trị sử dụng khác nhau tùy theo mục đích.

## Ngày cập nhật
Ngày: 2024-12-19
Nguồn: Kaggle Dataset - johnsmith88/heart-disease-dataset
