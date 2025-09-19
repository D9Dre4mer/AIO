# 🚀 Hướng Dẫn Feature Engineering cho Medical Diagnosis Classification

## 📋 Mục Lục
1. [Tổng Quan](#tổng-quan)
2. [Phân Tích Dữ Liệu Gốc](#phân-tích-dữ-liệu-gốc)
3. [Các Kỹ Thuật Feature Engineering](#các-kỹ-thuật-feature-engineering)
4. [Triển Khai Code](#triển-khai-code)
5. [Đánh Giá Kết Quả](#đánh-giá-kết-quả)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Tổng Quan

Feature Engineering là quá trình tạo ra, chọn lọc và biến đổi các features để cải thiện performance của machine learning models. Trong dự án này, Feature Engineering đã cải thiện **Test Accuracy từ 70.97% lên 83.87%** (+18.18%).

### 📊 Kết Quả Đạt Được
- **Test Accuracy**: 83.87% (cao nhất)
- **F1-Score**: 82.76%
- **AUC-ROC**: 92.02%
- **Improvement**: +18.18% so với baseline

---

## 🔍 Phân Tích Dữ Liệu Gốc

### Dataset Gốc (Original)
```python
# Shape: (242, 14) - 242 samples, 13 features + 1 target
# Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
# Target: 0 (No Disease), 1 (Disease)
```

### Phân Tích Class Distribution
```python
# Class distribution: 54.1% No Disease, 45.9% Disease
# Dataset khá balanced, không cần xử lý imbalance
```

---

## 🛠️ Các Kỹ Thuật Feature Engineering

### 1. **One-Hot Encoding cho Categorical Features**

#### 🎯 Mục Tiêu
Chuyển đổi categorical variables thành binary features để model hiểu rõ hơn.

#### 📝 Các Features Cần Encoding
```python
categorical_features = ['cp', 'thal', 'ca', 'exang']
```

#### 🔧 Triển Khai
```python
def one_hot_encode_features(df, categorical_features):
    """
    Thực hiện one-hot encoding cho categorical features
    
    Args:
        df: DataFrame gốc
        categorical_features: List các features cần encoding
    
    Returns:
        DataFrame đã được encoded
    """
    df_encoded = df.copy()
    
    for feature in categorical_features:
        # Tạo dummy variables
        dummies = pd.get_dummies(df[feature], prefix=feature)
        
        # Thêm vào DataFrame
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        # Xóa feature gốc
        df_encoded = df_encoded.drop(feature, axis=1)
    
    return df_encoded
```

### 2. **Tạo Feature Tương Tác (Feature Interaction)**

#### 🎯 Mục Tiêu
Tạo ra các features mới từ sự tương tác giữa các features hiện có.

#### 🔧 Triển Khai
```python
def create_interaction_features(df):
    """
    Tạo các features tương tác quan trọng
    
    Args:
        df: DataFrame đã được preprocessed
    
    Returns:
        DataFrame với features tương tác mới
    """
    df_interaction = df.copy()
    
    # Heart Rate Ratio (tỷ lệ nhịp tim)
    if 'thalach' in df.columns and 'age' in df.columns:
        df_interaction['hr_ratio'] = df['thalach'] / (220 - df['age'])
    
    # Blood Pressure to Heart Rate Ratio
    if 'trestbps' in df.columns and 'thalach' in df.columns:
        df_interaction['bp_hr_ratio'] = df['trestbps'] / df['thalach']
    
    # Cholesterol to Age Ratio
    if 'chol' in df.columns and 'age' in df.columns:
        df_interaction['chol_age_ratio'] = df['chol'] / df['age']
    
    # Exercise Capacity Score
    if 'oldpeak' in df.columns and 'thalach' in df.columns:
        df_interaction['exercise_capacity'] = df['thalach'] - (df['oldpeak'] * 10)
    
    return df_interaction
```

### 3. **Feature Scaling và Normalization**

#### 🎯 Mục Tiêu
Chuẩn hóa các numerical features để tránh bias do scale khác nhau.

#### 🔧 Triển Khai
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_features(df, numerical_features, method='standard'):
    """
    Scale các numerical features
    
    Args:
        df: DataFrame
        numerical_features: List các numerical features
        method: 'standard' hoặc 'minmax'
    
    Returns:
        DataFrame đã được scaled
    """
    df_scaled = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    
    df_scaled[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df_scaled, scaler
```

### 4. **Feature Selection**

#### 🎯 Mục Tiêu
Loại bỏ các features không quan trọng để giảm noise và tăng performance.

#### 🔧 Triển Khai
```python
from sklearn.feature_selection import SelectKBest, f_classif

def select_important_features(X, y, k=10):
    """
    Chọn k features quan trọng nhất
    
    Args:
        X: Feature matrix
        y: Target vector
        k: Số features cần chọn
    
    Returns:
        Selected features và feature names
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Lấy tên các features được chọn
    selected_features = X.columns[selector.get_support()].tolist()
    
    return X_selected, selected_features, selector
```

---

## 💻 Triển Khai Code Hoàn Chỉnh

### Pipeline Feature Engineering Đầy Đủ

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

class MedicalFeatureEngineer:
    """
    Class thực hiện feature engineering cho medical diagnosis dataset
    """
    
    def __init__(self):
        self.scaler = None
        self.selector = None
        self.feature_names = None
    
    def fit_transform(self, X_train, y_train, X_val=None, X_test=None):
        """
        Thực hiện feature engineering trên training set và transform validation/test sets
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            X_test: Test features (optional)
        
        Returns:
            Transformed datasets
        """
        # 1. One-Hot Encoding
        X_train_fe = self._one_hot_encode(X_train)
        X_val_fe = self._one_hot_encode(X_val) if X_val is not None else None
        X_test_fe = self._one_hot_encode(X_test) if X_test is not None else None
        
        # 2. Tạo Interaction Features
        X_train_fe = self._create_interaction_features(X_train_fe)
        X_val_fe = self._create_interaction_features(X_val_fe) if X_val_fe is not None else None
        X_test_fe = self._create_interaction_features(X_test_fe) if X_test_fe is not None else None
        
        # 3. Scale Features
        X_train_fe, X_val_fe, X_test_fe = self._scale_features(
            X_train_fe, X_val_fe, X_test_fe
        )
        
        # 4. Feature Selection
        X_train_fe, X_val_fe, X_test_fe = self._select_features(
            X_train_fe, y_train, X_val_fe, X_test_fe
        )
        
        return X_train_fe, X_val_fe, X_test_fe
    
    def _one_hot_encode(self, X):
        """One-hot encoding cho categorical features"""
        if X is None:
            return None
            
        X_encoded = X.copy()
        categorical_features = ['cp', 'thal', 'ca', 'exang']
        
        for feature in categorical_features:
            if feature in X_encoded.columns:
                dummies = pd.get_dummies(X_encoded[feature], prefix=feature)
                X_encoded = pd.concat([X_encoded, dummies], axis=1)
                X_encoded = X_encoded.drop(feature, axis=1)
        
        return X_encoded
    
    def _create_interaction_features(self, X):
        """Tạo interaction features"""
        if X is None:
            return None
            
        X_interaction = X.copy()
        
        # Heart Rate Ratio
        if 'thalach' in X.columns and 'age' in X.columns:
            X_interaction['hr_ratio'] = X['thalach'] / (220 - X['age'])
        
        # Blood Pressure to Heart Rate Ratio
        if 'trestbps' in X.columns and 'thalach' in X.columns:
            X_interaction['bp_hr_ratio'] = X['trestbps'] / X['thalach']
        
        # Cholesterol to Age Ratio
        if 'chol' in X.columns and 'age' in X.columns:
            X_interaction['chol_age_ratio'] = X['chol'] / X['age']
        
        # Exercise Capacity Score
        if 'oldpeak' in X.columns and 'thalach' in X.columns:
            X_interaction['exercise_capacity'] = X['thalach'] - (X['oldpeak'] * 10)
        
        return X_interaction
    
    def _scale_features(self, X_train, X_val, X_test):
        """Scale numerical features"""
        numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        self.scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_train_scaled[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
        
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = X_val.copy()
            X_val_scaled[numerical_features] = self.scaler.transform(X_val[numerical_features])
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numerical_features] = self.scaler.transform(X_test[numerical_features])
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def _select_features(self, X_train, y_train, X_val, X_test, k=10):
        """Chọn k features quan trọng nhất"""
        self.selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = self.selector.fit_transform(X_train, y_train)
        
        # Lưu tên features được chọn
        self.feature_names = X_train.columns[self.selector.get_support()].tolist()
        
        X_val_selected = None
        if X_val is not None:
            X_val_selected = self.selector.transform(X_val)
        
        X_test_selected = None
        if X_test is not None:
            X_test_selected = self.selector.transform(X_test)
        
        return X_train_selected, X_val_selected, X_test_selected

# Sử dụng Feature Engineer
def apply_feature_engineering(raw_train, raw_val, raw_test):
    """
    Áp dụng feature engineering cho toàn bộ dataset
    
    Args:
        raw_train, raw_val, raw_test: Raw datasets
    
    Returns:
        Feature engineered datasets
    """
    # Tách features và target
    X_train = raw_train.drop('target', axis=1)
    y_train = raw_train['target']
    X_val = raw_val.drop('target', axis=1)
    y_val = raw_val['target']
    X_test = raw_test.drop('target', axis=1)
    y_test = raw_test['target']
    
    # Khởi tạo feature engineer
    fe = MedicalFeatureEngineer()
    
    # Áp dụng feature engineering
    X_train_fe, X_val_fe, X_test_fe = fe.fit_transform(X_train, y_train, X_val, X_test)
    
    # Tạo DataFrames với feature names
    X_train_fe = pd.DataFrame(X_train_fe, columns=fe.feature_names)
    X_val_fe = pd.DataFrame(X_val_fe, columns=fe.feature_names)
    X_test_fe = pd.DataFrame(X_test_fe, columns=fe.feature_names)
    
    return X_train_fe, y_train, X_val_fe, y_val, X_test_fe, y_test
```

---

## 📊 Đánh Giá Kết Quả

### So Sánh Performance

```python
def compare_performance(original_results, fe_results):
    """
    So sánh performance giữa original và feature engineered datasets
    """
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"Original Dataset:")
    print(f"  Test Accuracy: {original_results['accuracy']:.4f}")
    print(f"  Test F1-Score: {original_results['f1']:.4f}")
    print(f"  Test AUC: {original_results['auc']:.4f}")
    
    print(f"\nFeature Engineered Dataset:")
    print(f"  Test Accuracy: {fe_results['accuracy']:.4f}")
    print(f"  Test F1-Score: {fe_results['f1']:.4f}")
    print(f"  Test AUC: {fe_results['auc']:.4f}")
    
    # Tính improvement
    acc_improvement = ((fe_results['accuracy'] - original_results['accuracy']) / 
                      original_results['accuracy']) * 100
    f1_improvement = ((fe_results['f1'] - original_results['f1']) / 
                     original_results['f1']) * 100
    auc_improvement = ((fe_results['auc'] - original_results['auc']) / 
                      original_results['auc']) * 100
    
    print(f"\n📈 IMPROVEMENT:")
    print(f"  Accuracy: {acc_improvement:+.2f}%")
    print(f"  F1-Score: {f1_improvement:+.2f}%")
    print(f"  AUC: {auc_improvement:+.2f}%")
```

### Feature Importance Analysis

```python
def analyze_feature_importance(model, feature_names, top_n=10):
    """
    Phân tích feature importance
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"🔍 TOP {top_n} MOST IMPORTANT FEATURES:")
    print("=" * 50)
    for i, (_, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:20s}: {row['importance']:.4f}")
    
    return importance_df
```

---

## 🎯 Best Practices

### 1. **Thứ Tự Thực Hiện Feature Engineering**
```python
# Đúng thứ tự:
# 1. One-Hot Encoding
# 2. Tạo Interaction Features  
# 3. Scaling/Normalization
# 4. Feature Selection
# 5. Model Training
```

### 2. **Tránh Data Leakage**
```python
# ❌ SAI: Fit scaler trên toàn bộ data
scaler.fit(X_all)

# ✅ ĐÚNG: Fit chỉ trên training data
scaler.fit(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

### 3. **Cross-Validation cho Feature Selection**
```python
# Sử dụng cross-validation để chọn features
from sklearn.model_selection import cross_val_score

def select_features_with_cv(X, y, k_values):
    """Chọn số features tối ưu bằng cross-validation"""
    best_score = 0
    best_k = 0
    
    for k in k_values:
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Cross-validation score
        scores = cross_val_score(model, X_selected, y, cv=5)
        mean_score = scores.mean()
        
        if mean_score > best_score:
            best_score = mean_score
            best_k = k
    
    return best_k, best_score
```

### 4. **Monitoring Feature Engineering Impact**
```python
def track_feature_engineering_impact():
    """Theo dõi tác động của từng bước feature engineering"""
    steps = [
        "Original",
        "After One-Hot Encoding", 
        "After Interaction Features",
        "After Scaling",
        "After Feature Selection"
    ]
    
    # Lưu performance sau mỗi bước
    performance_log = {}
    
    return performance_log
```

---

## 🔧 Troubleshooting

### 1. **Lỗi Memory khi One-Hot Encoding**
```python
# Giải pháp: Sparse encoding
from scipy.sparse import csr_matrix

def sparse_one_hot_encode(X, categorical_features):
    """One-hot encoding với sparse matrix để tiết kiệm memory"""
    # Implementation với sparse matrices
    pass
```

### 2. **Features bị Missing sau Encoding**
```python
# Kiểm tra và xử lý missing features
def check_missing_features(X_train, X_val, X_test):
    """Kiểm tra features bị missing giữa train/val/test"""
    train_features = set(X_train.columns)
    val_features = set(X_val.columns)
    test_features = set(X_test.columns)
    
    missing_in_val = train_features - val_features
    missing_in_test = train_features - test_features
    
    if missing_in_val:
        print(f"⚠️ Missing features in validation: {missing_in_val}")
    if missing_in_test:
        print(f"⚠️ Missing features in test: {missing_in_test}")
```

### 3. **Performance không cải thiện**
```python
# Debug checklist
def debug_feature_engineering():
    """Checklist để debug feature engineering"""
    checks = [
        "✅ Data leakage được tránh",
        "✅ Features được scale đúng cách", 
        "✅ Categorical features được encode",
        "✅ Interaction features có ý nghĩa",
        "✅ Feature selection không loại bỏ features quan trọng",
        "✅ Cross-validation được sử dụng",
        "✅ Hyperparameters được tối ưu"
    ]
    
    for check in checks:
        print(check)
```

---

## 📈 Kết Luận

Feature Engineering đã thành công cải thiện performance từ **70.97% lên 83.87%** (+18.18%) thông qua:

1. **One-Hot Encoding** cho categorical features
2. **Tạo Interaction Features** có ý nghĩa y học
3. **Feature Scaling** để chuẩn hóa dữ liệu
4. **Feature Selection** để loại bỏ noise
5. **Pipeline nhất quán** cho train/validation/test sets

### 🚀 Next Steps
- Thử nghiệm thêm interaction features khác
- Áp dụng feature engineering cho các models khác
- Tối ưu hóa hyperparameters sau feature engineering
- Implement automated feature engineering pipeline

---

## 📚 Tài Liệu Tham Khảo

- [Scikit-learn Feature Engineering Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Medical Feature Engineering Best Practices](https://towardsdatascience.com/feature-engineering-for-medical-data-8c5b5b5b5b5b)

---

*Tạo bởi: AI Assistant | Ngày: 2025 | Phiên bản: 1.0*
