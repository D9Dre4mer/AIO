# Phân Tích Các Hardcode Có Thể Ảnh Hưởng Đến Kết Quả

## Tổng Quan

Sau khi nghiên cứu toàn bộ codebase, đã phát hiện nhiều giá trị hardcode có thể ảnh hưởng đến kết quả training và testing. Báo cáo này phân tích từng loại hardcode và đánh giá mức độ tác động.

## 1. Hardcode Liên Quan Đến Random State

### 🔴 **Mức Độ Tác Động: CAO**

#### **Các chỗ hardcode `random_state=42`:**

**a) Data Splitting:**
```python
# Trong tất cả comprehensive files
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Trong app.py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**b) Cross-Validation:**
```python
# Trong tất cả comprehensive files
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**c) Data Sampling:**
```python
# Trong comprehensive files
df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
```

**d) Ensemble Models:**
```python
# Trong app.py
final_estimator = LogisticRegression(random_state=42, max_iter=1000)
final_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
```

#### **🚨 Tác Động:**
- **Reproducibility**: Kết quả luôn giống nhau
- **Bias**: Có thể tạo bias do luôn chọn cùng một subset
- **Overfitting**: Model có thể overfit với split cụ thể
- **Evaluation**: Không đánh giá được độ ổn định của model

#### **💡 Khuyến Nghị:**
- Sử dụng multiple random seeds để test
- Tạo config cho random_state thay vì hardcode
- Chạy experiment với nhiều random_state khác nhau

## 2. Hardcode Liên Quan Đến Optuna Configuration

### 🟡 **Mức Độ Tác Động: TRUNG BÌNH**

#### **Các chỗ hardcode Optuna:**

**a) Trials và Timeout:**
```python
# Trong tất cả comprehensive files
config = {
    'trials': 2,  # RẤT THẤP - chỉ để test nhanh
    'timeout': 30,
    'direction': 'maximize'
}
```

**b) Default Values trong app.py:**
```python
# Trong app.py UI
n_trials = st.number_input("Number of Trials", min_value=10, max_value=200, value=50)
timeout = st.number_input("Timeout (minutes)", min_value=5, max_value=120, value=30)
```

#### **🚨 Tác Động:**
- **Underoptimization**: `trials=2` quá thấp, không tìm được optimal hyperparameters
- **Inconsistent**: Comprehensive files dùng 2 trials, app.py dùng 50 trials
- **Poor Performance**: Model performance không optimal do hyperparameter tuning kém

#### **💡 Khuyến Nghị:**
- Tăng `trials` lên ít nhất 20-50 cho comprehensive testing
- Tạo config file cho Optuna parameters
- Sử dụng adaptive trials dựa trên dataset size

## 3. Hardcode Liên Quan Đến Data Split

### 🟡 **Mức Độ Tác Động: TRUNG BÌNH**

#### **Các chỗ hardcode test_size:**

```python
# Trong tất cả files
test_size=0.2  # 80-20 split
```

#### **🚨 Tác Động:**
- **Fixed Split**: Luôn dùng 80-20 split
- **Small Datasets**: Với dataset nhỏ, 20% test có thể quá ít
- **Large Datasets**: Với dataset lớn, 20% test có thể thừa

#### **💡 Khuyến Nghị:**
- Dynamic test_size dựa trên dataset size
- Smaller datasets: 70-30 split
- Larger datasets: 90-10 split

## 4. Hardcode Liên Quan Đến Vectorization

### 🟡 **Mức Độ Tác Động: TRUNG BÌNH**

#### **Các chỗ hardcode vectorization parameters:**

**a) TF-IDF và BoW:**
```python
# Trong comprehensive files - QUÁ THẤP
'max_features': 1000,
'ngram_range': (1, 2),
'min_df': 2

# Trong app.py - TƯƠNG ĐỐI TỐT
'max_features': 10000,  # Default value
'ngram_range': (1, 2),
'min_df': 2
```

**b) Text Encoders:**
```python
# Trong text_encoders.py
min_df=2,           # Ignore words appearing in < 2 documents
max_df=0.95,        # Ignore words appearing in > 95% documents
```

#### **🚨 Tác Động:**
- **Under-representation**: `max_features=1000` quá thấp có thể mất thông tin
- **Inconsistency**: Comprehensive files (1000) vs app.py (10000)
- **Poor Text Representation**: Có thể ảnh hưởng đến accuracy của text classification

#### **💡 Khuyến Nghị:**
- Tăng `max_features` trong comprehensive files lên 5000-10000
- Tạo adaptive max_features dựa trên dataset size
- Test với nhiều ngram_range khác nhau

## 5. Hardcode Liên Quan Đến Cross-Validation

### 🟡 **Mức Độ Tác Động: TRUNG BÌNH**

#### **Các chỗ hardcode CV:**

```python
# Trong tất cả files
n_splits=5  # 5-fold CV
cv_folds=5
```

#### **🚨 Tác Động:**
- **Fixed Folds**: Luôn dùng 5-fold
- **Small Datasets**: 5-fold có thể tạo fold quá nhỏ
- **Large Datasets**: 5-fold có thể đủ nhưng có thể tối ưu hơn

#### **💡 Khuyến Nghị:**
- Dynamic CV folds dựa trên dataset size
- Small datasets (n<500): 3-fold CV
- Medium datasets (500-5000): 5-fold CV
- Large datasets (>5000): 10-fold CV

## 6. Hardcode Liên Quan Đến Sample Size

### 🟡 **Mức Độ Tác Động: TRUNG BÌNH**

#### **Các chỗ hardcode sample_size:**

```python
# Trong comprehensive files
def load_large_dataset(sample_size: int = 1000)
def load_spam_dataset(sample_size: int = 1000)

# Trong main functions
df, text_column, label_column = load_large_dataset(sample_size=1000)
```

#### **🚨 Tác Động:**
- **Limited Testing**: Chỉ test trên 1000 samples từ 300K dataset
- **Unrepresentative**: 1000 samples có thể không đại diện cho toàn bộ dataset
- **Performance Bias**: Kết quả có thể khác khi chạy trên full dataset

#### **💡 Khuyến Nghị:**
- Tăng sample_size lên 5000-10000 cho large dataset
- Tạo multiple sample sizes để test
- So sánh performance giữa các sample sizes

## 7. Hardcode Trong Model Parameters

### 🟢 **Mức Độ Tác Động: THẤP**

#### **Các chỗ hardcode model params:**

```python
# Trong app.py ensemble
final_estimator = LogisticRegression(random_state=42, max_iter=1000)
final_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
```

#### **🚨 Tác Động:**
- **Suboptimal**: Các parameters này có thể không optimal
- **Limited**: Không test các configuration khác

#### **💡 Khuyến Nghị:**
- Để Optuna tự optimize các parameters này
- Không hardcode model-specific parameters

## 8. Các Hardcode Khác

### 🟢 **Mức Độ Tác Động: THẤP - KHÔNG ĐÁNG KỂ**

#### **Formatting và Display:**
```python
# Các hardcode không ảnh hưởng kết quả
print(f"   Success rate: {len(successful_results)/len(all_results)*100:.1f}%")
'max_iter': 1000  # Trong LogisticRegression - đủ cho hầu hết trường hợp
```

## Đánh Giá Tổng Thể

### **🔴 Critical Issues (Cần Sửa Ngay):**

1. **`trials=2` trong comprehensive files**
   - Quá thấp, không đủ để tìm optimal hyperparameters
   - Tác động trực tiếp đến model performance

2. **`max_features=1000` trong vectorization**
   - Quá thấp cho text data, có thể mất thông tin quan trọng
   - Tác động đến text classification accuracy

3. **`sample_size=1000` cho large dataset**
   - Không đại diện cho 300K samples
   - Kết quả không reflect performance thực tế

### **🟡 Medium Issues (Nên Cải Thiện):**

1. **`random_state=42` ở mọi nơi**
   - Cần test với multiple random seeds
   - Đánh giá model stability

2. **Fixed `test_size=0.2`**
   - Nên adaptive dựa trên dataset size

3. **Fixed `n_splits=5` cho CV**
   - Nên adaptive dựa trên dataset size

### **🟢 Low Issues (Có Thể Bỏ Qua):**

1. Formatting parameters
2. Display configurations
3. Reasonable default values (như max_iter=1000)

## Khuyến Nghị Cải Thiện

### **1. Tạo Configuration System:**

```python
# config.py
DEFAULT_CONFIG = {
    'data_split': {
        'test_size': 0.2,
        'random_state': 42,
        'stratify': True
    },
    'optuna': {
        'trials': 50,  # Tăng từ 2
        'timeout': 300,  # 5 phút
        'direction': 'maximize'
    },
    'vectorization': {
        'max_features': 10000,  # Tăng từ 1000
        'ngram_range': (1, 2),
        'min_df': 2
    },
    'cross_validation': {
        'n_splits': 5,
        'shuffle': True,
        'random_state': 42
    }
}
```

### **2. Dynamic Parameters:**

```python
def get_adaptive_config(dataset_size):
    if dataset_size < 500:
        return {'cv_folds': 3, 'test_size': 0.3}
    elif dataset_size < 5000:
        return {'cv_folds': 5, 'test_size': 0.2}
    else:
        return {'cv_folds': 10, 'test_size': 0.1}
```

### **3. Multiple Random Seeds Testing:**

```python
def test_with_multiple_seeds(seeds=[42, 123, 456, 789, 999]):
    results = []
    for seed in seeds:
        result = train_model(random_state=seed)
        results.append(result)
    return analyze_stability(results)
```

### **4. Increased Optuna Trials:**

```python
# Comprehensive files
config = {
    'trials': 50,  # Tăng từ 2
    'timeout': 600,  # 10 phút thay vì 30 giây
    'direction': 'maximize'
}
```

### **5. Improved Vectorization:**

```python
# Comprehensive files
'max_features': 10000,  # Tăng từ 1000
'ngram_range': [(1,1), (1,2), (1,3)],  # Test multiple ranges
```

## Kết Luận

### **Tác Động Lên Kết Quả Hiện Tại:**

1. **Heart Dataset**: Ít bị ảnh hưởng vì dataset dễ classify
2. **Text Datasets**: Bị ảnh hưởng nhiều hơn do `max_features=1000` và `trials=2`
3. **Overall Performance**: Có thể chưa optimal do underoptimization

### **Priority Actions:**

1. **Immediately**: Tăng `trials` từ 2 lên 50 trong comprehensive files
2. **High Priority**: Tăng `max_features` từ 1000 lên 10000
3. **Medium Priority**: Test với multiple random seeds
4. **Low Priority**: Tạo adaptive configuration system

### **Expected Improvements:**

- **Text Classification**: +2-5% accuracy với max_features tăng
- **All Models**: +1-3% accuracy với trials tăng
- **Stability**: Better understanding với multiple seeds
- **Reliability**: More robust evaluation với improved configuration

---

*Báo cáo này dựa trên phân tích comprehensive toàn bộ codebase và đánh giá tác động của từng hardcode đến model performance.*
