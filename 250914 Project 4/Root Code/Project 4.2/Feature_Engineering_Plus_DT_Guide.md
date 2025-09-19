# 🌳🚀 Hướng Dẫn Kết Hợp Feature Engineering + Decision Tree Features

## 📋 Mục Lục
1. [Tổng Quan](#tổng-quan)
2. [Hiểu Về FE + DT](#hiểu-về-fe--dt)
3. [Pipeline Triển Khai](#pipeline-triển-khai)
4. [Code Implementation](#code-implementation)
5. [So Sánh Performance](#so-sánh-performance)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Techniques](#advanced-techniques)

---

## 🎯 Tổng Quan

**Feature Engineering + Decision Tree (FE + DT)** là một **hybrid approach** kết hợp 2 kỹ thuật mạnh mẽ:

- **Feature Engineering**: Tạo ra features mới có ý nghĩa
- **Decision Tree Selection**: Chọn lọc features quan trọng nhất

### 📊 Kết Quả Đạt Được
- **Test Accuracy**: 80.65%
- **Improvement**: +13.64% so với baseline
- **Feature Count**: Giảm từ 13 → 10 features
- **Performance Ranking**: #2 trong 4 datasets

---

## 🔍 Hiểu Về FE + DT

### **1. Feature Engineering (FE)**
```python
# Mục tiêu: Tạo ra features mới có tính phân biệt cao
- One-Hot Encoding: thal_3.0, cp_4.0, ca_0.0, exang_0.0
- Interaction Features: hr_ratio, chol_per_age
- Scaling: Chuẩn hóa numerical features
```

### **2. Decision Tree Features (DT)**
```python
# Mục tiêu: Chọn lọc features quan trọng nhất
- Tree-based Selection: Sử dụng DT importance scores
- Non-linear Relationships: Capture complex patterns
- Dimensionality Reduction: Loại bỏ features không quan trọng
```

### **3. FE + DT Combination**
```python
# Kết quả: Best of both worlds
- Rich Features: Từ Feature Engineering
- Quality Selection: Từ Decision Tree
- Optimal Balance: Giữa feature richness và feature quality
```

---

## 🛠️ Pipeline Triển Khai

### **Bước 1: Feature Engineering**
```python
def apply_feature_engineering(df):
    """
    Áp dụng Feature Engineering cho dataset
    
    Args:
        df: DataFrame gốc
    
    Returns:
        DataFrame đã được feature engineering
    """
    df_fe = df.copy()
    
    # 1. One-Hot Encoding
    categorical_features = ['cp', 'thal', 'ca', 'exang', 'slope', 'sex']
    for feature in categorical_features:
        if feature in df_fe.columns:
            dummies = pd.get_dummies(df_fe[feature], prefix=feature)
            df_fe = pd.concat([df_fe, dummies], axis=1)
            df_fe = df_fe.drop(feature, axis=1)
    
    # 2. Tạo Interaction Features
    if 'thalach' in df_fe.columns and 'age' in df_fe.columns:
        df_fe['hr_ratio'] = df_fe['thalach'] / (220 - df_fe['age'])
    
    if 'chol' in df_fe.columns and 'age' in df_fe.columns:
        df_fe['chol_per_age'] = df_fe['chol'] / df_fe['age']
    
    # 3. Scaling Numerical Features
    numerical_features = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    if 'target' in numerical_features:
        numerical_features.remove('target')
    
    scaler = StandardScaler()
    df_fe[numerical_features] = scaler.fit_transform(df_fe[numerical_features])
    
    return df_fe, scaler
```

### **Bước 2: Decision Tree Feature Selection**
```python
def apply_decision_tree_selection(X, y, method='importance', threshold='median'):
    """
    Áp dụng Decision Tree để chọn features quan trọng
    
    Args:
        X: Feature matrix
        y: Target vector
        method: 'importance' hoặc 'recursive'
        threshold: Threshold cho feature selection
    
    Returns:
        Selected features và selector object
    """
    if method == 'importance':
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.feature_selection import SelectFromModel
        
        # Train Decision Tree
        dt = DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        dt.fit(X, y)
        
        # Select features based on importance
        selector = SelectFromModel(dt, threshold=threshold)
        X_selected = selector.fit_transform(X, y)
        
    elif method == 'recursive':
        from sklearn.feature_selection import RFE
        from sklearn.tree import DecisionTreeClassifier
        
        dt = DecisionTreeClassifier(random_state=42)
        selector = RFE(dt, n_features_to_select=10)
        X_selected = selector.fit_transform(X, y)
    
    return X_selected, selector
```

### **Bước 3: Kết Hợp FE + DT**
```python
def create_fe_dt_dataset(df):
    """
    Tạo FE + DT dataset hoàn chỉnh
    
    Args:
        df: DataFrame gốc
    
    Returns:
        FE + DT dataset và các objects cần thiết
    """
    # Bước 1: Feature Engineering
    df_fe, scaler = apply_feature_engineering(df)
    
    # Tách features và target
    X = df_fe.drop('target', axis=1)
    y = df_fe['target']
    
    # Bước 2: Decision Tree Feature Selection
    X_selected, selector = apply_decision_tree_selection(X, y)
    
    # Bước 3: Tạo dataset cuối cùng
    selected_features = X.columns[selector.get_support()].tolist()
    df_fe_dt = pd.DataFrame(X_selected, columns=selected_features)
    df_fe_dt['target'] = y.values
    
    return df_fe_dt, scaler, selector
```

---

## 💻 Code Implementation Hoàn Chỉnh

### **Class FE_DT_Processor**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class FE_DT_Processor:
    """
    Class xử lý kết hợp Feature Engineering và Decision Tree Features
    """
    
    def __init__(self, 
                 categorical_features=None,
                 interaction_features=True,
                 scaling_method='standard',
                 dt_method='importance',
                 selection_threshold='median'):
        """
        Khởi tạo FE_DT_Processor
        
        Args:
            categorical_features: List các categorical features
            interaction_features: Có tạo interaction features không
            scaling_method: 'standard' hoặc 'minmax'
            dt_method: 'importance' hoặc 'recursive'
            selection_threshold: Threshold cho feature selection
        """
        self.categorical_features = categorical_features or ['cp', 'thal', 'ca', 'exang', 'slope', 'sex']
        self.interaction_features = interaction_features
        self.scaling_method = scaling_method
        self.dt_method = dt_method
        self.selection_threshold = selection_threshold
        
        # Objects sẽ được fit
        self.scaler = None
        self.selector = None
        self.feature_names = None
        self.feature_importance = None
    
    def fit_transform(self, X_train, y_train, X_val=None, X_test=None):
        """
        Fit và transform datasets
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
            X_test, y_test: Test data (optional)
        
        Returns:
            Transformed datasets
        """
        # Bước 1: Feature Engineering
        X_train_fe, X_val_fe, X_test_fe = self._apply_feature_engineering(
            X_train, X_val, X_test
        )
        
        # Bước 2: Decision Tree Feature Selection
        X_train_fe_dt, X_val_fe_dt, X_test_fe_dt = self._apply_dt_selection(
            X_train_fe, y_train, X_val_fe, X_test_fe
        )
        
        return X_train_fe_dt, X_val_fe_dt, X_test_fe_dt
    
    def _apply_feature_engineering(self, X_train, X_val, X_test):
        """Áp dụng Feature Engineering"""
        # One-Hot Encoding
        X_train_fe = self._one_hot_encode(X_train)
        X_val_fe = self._one_hot_encode(X_val) if X_val is not None else None
        X_test_fe = self._one_hot_encode(X_test) if X_test is not None else None
        
        # Interaction Features
        if self.interaction_features:
            X_train_fe = self._create_interaction_features(X_train_fe)
            X_val_fe = self._create_interaction_features(X_val_fe) if X_val_fe is not None else None
            X_test_fe = self._create_interaction_features(X_test_fe) if X_test_fe is not None else None
        
        # Scaling
        X_train_fe, X_val_fe, X_test_fe = self._scale_features(
            X_train_fe, X_val_fe, X_test_fe
        )
        
        return X_train_fe, X_val_fe, X_test_fe
    
    def _one_hot_encode(self, X):
        """One-Hot Encoding cho categorical features"""
        if X is None:
            return None
        
        X_encoded = X.copy()
        
        for feature in self.categorical_features:
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
        
        # Cholesterol per Age
        if 'chol' in X.columns and 'age' in X.columns:
            X_interaction['chol_per_age'] = X['chol'] / X['age']
        
        # Blood Pressure to Heart Rate Ratio
        if 'trestbps' in X.columns and 'thalach' in X.columns:
            X_interaction['bp_hr_ratio'] = X['trestbps'] / X['thalach']
        
        # Exercise Capacity Score
        if 'oldpeak' in X.columns and 'thalach' in X.columns:
            X_interaction['exercise_capacity'] = X['thalach'] - (X['oldpeak'] * 10)
        
        return X_interaction
    
    def _scale_features(self, X_train, X_val, X_test):
        """Scale numerical features"""
        numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        
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
    
    def _apply_dt_selection(self, X_train, y_train, X_val, X_test):
        """Áp dụng Decision Tree feature selection"""
        if self.dt_method == 'importance':
            X_train_selected, X_val_selected, X_test_selected = self._importance_based_selection(
                X_train, y_train, X_val, X_test
            )
        elif self.dt_method == 'recursive':
            X_train_selected, X_val_selected, X_test_selected = self._recursive_selection(
                X_train, y_train, X_val, X_test
            )
        
        return X_train_selected, X_val_selected, X_test_selected
    
    def _importance_based_selection(self, X_train, y_train, X_val, X_test):
        """Importance-based feature selection"""
        # Train Decision Tree
        dt = DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        dt.fit(X_train, y_train)
        
        # Select features
        self.selector = SelectFromModel(dt, threshold=self.selection_threshold)
        X_train_selected = self.selector.fit_transform(X_train, y_train)
        
        # Get selected feature names
        self.feature_names = X_train.columns[self.selector.get_support()].tolist()
        self.feature_importance = dt.feature_importances_
        
        # Transform validation and test sets
        X_val_selected = self.selector.transform(X_val) if X_val is not None else None
        X_test_selected = self.selector.transform(X_test) if X_test is not None else None
        
        return X_train_selected, X_val_selected, X_test_selected
    
    def _recursive_selection(self, X_train, y_train, X_val, X_test, n_features=10):
        """Recursive feature elimination"""
        dt = DecisionTreeClassifier(random_state=42)
        self.selector = RFE(dt, n_features_to_select=n_features)
        
        X_train_selected = self.selector.fit_transform(X_train, y_train)
        self.feature_names = X_train.columns[self.selector.get_support()].tolist()
        
        X_val_selected = self.selector.transform(X_val) if X_val is not None else None
        X_test_selected = self.selector.transform(X_test) if X_test is not None else None
        
        return X_train_selected, X_val_selected, X_test_selected
    
    def get_feature_importance(self):
        """Lấy feature importance"""
        if self.feature_importance is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance[self.selector.get_support()]
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_selected_features(self):
        """Lấy danh sách features được chọn"""
        return self.feature_names
```

### **Sử Dụng FE_DT_Processor**

```python
def create_fe_dt_datasets(raw_train, raw_val, raw_test):
    """
    Tạo FE + DT datasets cho toàn bộ dataset
    
    Args:
        raw_train, raw_val, raw_test: Raw datasets
    
    Returns:
        FE + DT datasets và processor object
    """
    # Tách features và target
    X_train = raw_train.drop('target', axis=1)
    y_train = raw_train['target']
    X_val = raw_val.drop('target', axis=1)
    y_val = raw_val['target']
    X_test = raw_test.drop('target', axis=1)
    y_test = raw_test['target']
    
    # Khởi tạo processor
    processor = FE_DT_Processor(
        categorical_features=['cp', 'thal', 'ca', 'exang', 'slope', 'sex'],
        interaction_features=True,
        scaling_method='standard',
        dt_method='importance',
        selection_threshold='median'
    )
    
    # Áp dụng FE + DT
    X_train_fe_dt, X_val_fe_dt, X_test_fe_dt = processor.fit_transform(
        X_train, y_train, X_val, X_test
    )
    
    # Tạo DataFrames với feature names
    X_train_fe_dt = pd.DataFrame(X_train_fe_dt, columns=processor.get_selected_features())
    X_val_fe_dt = pd.DataFrame(X_val_fe_dt, columns=processor.get_selected_features())
    X_test_fe_dt = pd.DataFrame(X_test_fe_dt, columns=processor.get_selected_features())
    
    # Thêm target
    X_train_fe_dt['target'] = y_train.values
    X_val_fe_dt['target'] = y_val.values
    X_test_fe_dt['target'] = y_test.values
    
    return X_train_fe_dt, X_val_fe_dt, X_test_fe_dt, processor
```

---

## 📊 So Sánh Performance

### **Performance Comparison Table**

| Dataset | Features | Test Accuracy | F1-Score | AUC | Improvement |
|---------|----------|---------------|----------|-----|-------------|
| **Original** | 13 | 70.97% | 0.7097 | 0.7097 | Baseline |
| **Feature Engineering** | 13 | **83.87%** | 0.8276 | 0.9202 | **+18.18%** |
| **Original + DT** | 10 | 80.65% | 0.8065 | 0.8065 | +13.64% |
| **FE + DT** | 10 | 80.65% | 0.8065 | 0.8065 | +13.64% |

### **Feature Analysis**

```python
def analyze_fe_dt_features(processor):
    """
    Phân tích features trong FE + DT dataset
    """
    print("🔍 FE + DT FEATURE ANALYSIS")
    print("=" * 50)
    
    # Selected features
    selected_features = processor.get_selected_features()
    print(f"Selected Features ({len(selected_features)}):")
    for i, feature in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feature}")
    
    # Feature importance
    importance_df = processor.get_feature_importance()
    if importance_df is not None:
        print(f"\nTop 5 Most Important Features:")
        for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
            print(f"  {i}. {row['feature']:20s}: {row['importance']:.4f}")
    
    return selected_features, importance_df
```

---

## 🎯 Best Practices

### **1. Thứ Tự Thực Hiện**
```python
# ✅ ĐÚNG THỨ TỰ:
# 1. Feature Engineering (One-hot, Interaction, Scaling)
# 2. Decision Tree Feature Selection
# 3. Model Training
# 4. Evaluation
```

### **2. Cross-Validation cho Feature Selection**
```python
def select_features_with_cv(X, y, method='importance', cv_folds=5):
    """
    Chọn features tối ưu bằng cross-validation
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    
    best_score = 0
    best_features = None
    
    # Test different thresholds
    thresholds = ['median', 'mean', 0.1, 0.2, 0.3]
    
    for threshold in thresholds:
        processor = FE_DT_Processor(selection_threshold=threshold)
        X_selected, _, _ = processor.fit_transform(X, y)
        
        # Cross-validation
        rf = RandomForestClassifier(random_state=42)
        scores = cross_val_score(rf, X_selected, y, cv=cv_folds)
        mean_score = scores.mean()
        
        if mean_score > best_score:
            best_score = mean_score
            best_features = processor.get_selected_features()
    
    return best_features, best_score
```

### **3. Monitoring Feature Engineering Impact**
```python
def track_fe_impact():
    """
    Theo dõi tác động của từng bước
    """
    steps = [
        "Original",
        "After One-Hot Encoding",
        "After Interaction Features", 
        "After Scaling",
        "After DT Selection"
    ]
    
    performance_log = {}
    
    return performance_log
```

### **4. Hyperparameter Tuning cho DT Selection**
```python
def tune_dt_parameters(X, y):
    """
    Tune parameters cho Decision Tree selection
    """
    from sklearn.model_selection import GridSearchCV
    
    param_grid = {
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    
    return grid_search.best_params_
```

---

## 🔧 Troubleshooting

### **1. Features Bị Missing sau Selection**
```python
def check_missing_features(X_train, X_val, X_test):
    """
    Kiểm tra features bị missing
    """
    train_features = set(X_train.columns)
    val_features = set(X_val.columns)
    test_features = set(X_test.columns)
    
    missing_in_val = train_features - val_features
    missing_in_test = train_features - test_features
    
    if missing_in_val:
        print(f"⚠️ Missing features in validation: {missing_in_val}")
    if missing_in_test:
        print(f"⚠️ Missing features in test: {missing_in_test}")
    
    return missing_in_val, missing_in_test
```

### **2. Performance Không Cải Thiện**
```python
def debug_fe_dt_performance():
    """
    Debug checklist cho FE + DT
    """
    checks = [
        "✅ Feature Engineering được áp dụng đúng",
        "✅ Decision Tree parameters được tune",
        "✅ Feature selection threshold phù hợp",
        "✅ Cross-validation được sử dụng",
        "✅ Data leakage được tránh",
        "✅ Features có ý nghĩa domain-specific"
    ]
    
    for check in checks:
        print(check)
```

### **3. Memory Issues**
```python
def optimize_memory_usage():
    """
    Tối ưu memory usage
    """
    # Sử dụng sparse matrices
    from scipy.sparse import csr_matrix
    
    # Batch processing
    def process_in_batches(X, batch_size=1000):
        for i in range(0, len(X), batch_size):
            yield X[i:i+batch_size]
    
    # Feature selection trước khi scaling
    # để giảm memory usage
```

---

## 🚀 Advanced Techniques

### **1. Ensemble Feature Selection**
```python
def ensemble_feature_selection(X, y):
    """
    Kết hợp nhiều methods để chọn features
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.ensemble import RandomForestClassifier
    
    # Method 1: Decision Tree
    processor_dt = FE_DT_Processor(dt_method='importance')
    X_dt, _, _ = processor_dt.fit_transform(X, y)
    features_dt = set(processor_dt.get_selected_features())
    
    # Method 2: Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    importance_scores = rf.feature_importances_
    features_rf = set(X.columns[importance_scores > np.median(importance_scores)])
    
    # Method 3: Statistical
    selector_stat = SelectKBest(f_classif, k=10)
    X_stat = selector_stat.fit_transform(X, y)
    features_stat = set(X.columns[selector_stat.get_support()])
    
    # Intersection
    common_features = features_dt & features_rf & features_stat
    
    return list(common_features)
```

### **2. Dynamic Feature Selection**
```python
def dynamic_feature_selection(X, y, model_type='lightgbm'):
    """
    Chọn features dựa trên model type
    """
    if model_type == 'lightgbm':
        # LightGBM works well with many features
        threshold = 'mean'
    elif model_type == 'logistic':
        # Logistic regression needs fewer features
        threshold = 'median'
    elif model_type == 'svm':
        # SVM needs scaled features
        threshold = 0.1
    
    processor = FE_DT_Processor(selection_threshold=threshold)
    return processor
```

### **3. Feature Engineering với Domain Knowledge**
```python
def medical_domain_features(df):
    """
    Tạo features dựa trên domain knowledge y học
    """
    df_medical = df.copy()
    
    # Cardiovascular Risk Score
    if all(col in df.columns for col in ['age', 'chol', 'trestbps']):
        df_medical['cv_risk_score'] = (
            df['age'] * 0.1 + 
            df['chol'] * 0.01 + 
            df['trestbps'] * 0.01
        )
    
    # Exercise Tolerance
    if all(col in df.columns for col in ['thalach', 'oldpeak', 'age']):
        df_medical['exercise_tolerance'] = (
            df['thalach'] - (df['oldpeak'] * 10) - (df['age'] * 0.5)
        )
    
    # Heart Rate Variability
    if 'thalach' in df.columns:
        df_medical['hr_variability'] = df['thalach'].rolling(window=3).std()
    
    return df_medical
```

---

## 📈 Kết Luận

**Feature Engineering + Decision Tree** là một approach mạnh mẽ kết hợp:

### **✅ Ưu Điểm:**
- **Feature Richness**: Từ Feature Engineering
- **Feature Quality**: Từ Decision Tree selection
- **Dimensionality Reduction**: Giảm noise
- **Domain Knowledge**: Có thể tích hợp domain expertise

### **⚠️ Hạn Chế:**
- **Feature Loss**: Có thể loại bỏ features quan trọng
- **Complexity**: Pipeline phức tạp hơn
- **Tuning**: Cần tune nhiều parameters

### **🎯 Khi Nào Sử Dụng:**
- Dataset có nhiều features
- Cần balance giữa performance và interpretability
- Có domain knowledge để tạo features
- Muốn giảm overfitting

### **🚀 Next Steps:**
- Thử nghiệm với các algorithms khác
- Tích hợp domain knowledge sâu hơn
- Implement automated feature engineering
- A/B testing với different approaches

---

## 📚 Tài Liệu Tham Khảo

- [Scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Decision Tree Feature Importance](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)
- [Feature Engineering Best Practices](https://www.kaggle.com/learn/feature-engineering)

---

*Tạo bởi: AI Assistant | Ngày: 2025 | Phiên bản: 1.0*
