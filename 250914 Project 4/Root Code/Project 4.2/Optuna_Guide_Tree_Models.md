# 🌳 Hướng dẫn sử dụng Optuna cho các mô hình cây

## 📋 Mục lục
1. [Giới thiệu Optuna](#giới-thiệu-optuna)
2. [Cài đặt](#cài-đặt)
3. [Cấu trúc cơ bản](#cấu-trúc-cơ-bản)
4. [XGBoost với Optuna](#xgboost-với-optuna)
5. [LightGBM với Optuna](#lightgbm-với-optuna)
6. [Random Forest với Optuna](#random-forest-với-optuna)
7. [Gradient Boosting với Optuna](#gradient-boosting-với-optuna)
8. [AdaBoost với Optuna](#adaboost-với-optuna)
9. [Visualization](#visualization)
10. [Best Practices](#best-practices)

---

## 🎯 Giới thiệu Optuna

**Optuna** là thư viện Python mạnh mẽ để tối ưu hóa hyperparameters tự động, đặc biệt hiệu quả với các mô hình cây.

### Ưu điểm:
- ✅ **Tự động tìm kiếm**: Không cần test thủ công
- ✅ **Pruning thông minh**: Dừng sớm các trial không có triển vọng
- ✅ **Parallel processing**: Chạy song song nhiều thí nghiệm
- ✅ **Visualization**: Hiển thị quá trình tối ưu hóa
- ✅ **Tương thích cao**: Hỗ trợ hầu hết các mô hình ML

---

## 🔧 Cài đặt

### Cài đặt cơ bản:
```bash
pip install optuna
```

### Cài đặt với các tùy chọn mở rộng:
```bash
# Với visualization
pip install optuna[visualization]

# Với integration cho LightGBM
pip install optuna[lightgbm]

# Với integration cho XGBoost
pip install optuna[xgboost]

# Cài đặt đầy đủ
pip install optuna[all]
```

### Kiểm tra cài đặt:
```python
import optuna
print(f"Optuna version: {optuna.__version__}")
```

---

## 🏗️ Cấu trúc cơ bản

### Template cơ bản:
```python
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def objective(trial):
    # 1. Định nghĩa hyperparameters
    param1 = trial.suggest_float('param1', 0.0, 1.0)
    param2 = trial.suggest_int('param2', 1, 100)
    
    # 2. Tạo model với tham số
    model = YourModel(param1=param1, param2=param2)
    
    # 3. Train và evaluate
    scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # 4. Return metric để optimize
    return scores.mean()

# 5. Tạo study và optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 6. Lấy kết quả
print(f"Best score: {study.best_value}")
print(f"Best params: {study.best_params}")
```

---

## 🚀 XGBoost với Optuna

### Cài đặt:
```bash
pip install xgboost optuna
```

### Code mẫu:
```python
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

# Tạo study
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)

# Optimize
study.optimize(xgb_objective, n_trials=100, timeout=3600)

# Kết quả
print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Train model cuối cùng
best_model = xgb.XGBClassifier(**study.best_params)
best_model.fit(X_train, y_train)
```

---

## 💡 LightGBM với Optuna

### Cài đặt:
```bash
pip install lightgbm optuna
```

### Code mẫu:
```python
import optuna
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

def lgb_objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42,
        'verbose': -1
    }
    
    # Sử dụng LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[train_data], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    # Predict và tính accuracy
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_val, y_pred_binary)
    
    return accuracy

# Tạo study
study = optuna.create_study(direction='maximize')
study.optimize(lgb_objective, n_trials=100)

print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

---

## 🌲 Random Forest với Optuna

### Code mẫu:
```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42
    }
    
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

# Tạo study
study = optuna.create_study(direction='maximize')
study.optimize(rf_objective, n_trials=100)

print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

---

## 📈 Gradient Boosting với Optuna

### Code mẫu:
```python
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def gb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 42
    }
    
    model = GradientBoostingClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

# Tạo study
study = optuna.create_study(direction='maximize')
study.optimize(gb_objective, n_trials=100)

print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

---

## 🎯 AdaBoost với Optuna

### Code mẫu:
```python
import optuna
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def ada_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 2.0, log=True),
        'algorithm': trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R']),
        'random_state': 42
    }
    
    # Base estimator với hyperparameters
    base_estimator = DecisionTreeClassifier(
        max_depth=trial.suggest_int('max_depth', 1, 10),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
        random_state=42
    )
    
    model = AdaBoostClassifier(estimator=base_estimator, **params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

# Tạo study
study = optuna.create_study(direction='maximize')
study.optimize(ada_objective, n_trials=100)

print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

---

## 📊 Visualization

### Cài đặt visualization:
```bash
pip install optuna[visualization]
```

### Các loại biểu đồ:

#### 1. Optimization History:
```python
import optuna.visualization as vis

# Lịch sử tối ưu hóa
vis.plot_optimization_history(study)
```

#### 2. Parameter Importance:
```python
# Tầm quan trọng của từng parameter
vis.plot_param_importances(study)
```

#### 3. Parallel Coordinate:
```python
# Xem tất cả trials trên 1 biểu đồ
vis.plot_parallel_coordinate(study)
```

#### 4. Slice Plot:
```python
# Xem từng parameter riêng lẻ
vis.plot_slice(study)
```

#### 5. Contour Plot:
```python
# Xem tương quan giữa 2 parameters
vis.plot_contour(study, params=['param1', 'param2'])
```

---

## 🏆 Best Practices

### 1. **Chọn Sampler phù hợp:**
```python
# TPESampler - tốt nhất cho hầu hết trường hợp
sampler = optuna.samplers.TPESampler(seed=42)

# RandomSampler - đơn giản, nhanh
sampler = optuna.samplers.RandomSampler(seed=42)

# GridSampler - khi biết rõ không gian tìm kiếm
sampler = optuna.samplers.GridSampler(search_space)
```

### 2. **Sử dụng Pruning:**
```python
# MedianPruner - cân bằng tốt
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,      # Số trials đầu không prune
    n_warmup_steps=10,       # Số steps warmup
    interval_steps=1         # Tần suất check
)

# SuccessiveHalvingPruner - cho large-scale
pruner = optuna.pruners.SuccessiveHalvingPruner()
```

### 3. **Timeout và n_trials:**
```python
# Kết hợp cả hai
study.optimize(objective, n_trials=1000, timeout=3600)  # 1 giờ hoặc 1000 trials
```

### 4. **Logging và Callback:**
```python
def callback(study, trial):
    print(f"Trial {trial.number}: {trial.value:.4f}")

study.optimize(objective, n_trials=100, callbacks=[callback])
```

### 5. **Resume Study:**
```python
# Lưu study
study = optuna.create_study(study_name='my_study', storage='sqlite:///study.db')

# Resume từ checkpoint
study = optuna.load_study(study_name='my_study', storage='sqlite:///study.db')
```

---

## 🔧 Troubleshooting

### Lỗi thường gặp:

#### 1. **Memory Error:**
```python
# Giảm n_trials hoặc tăng timeout
study.optimize(objective, n_trials=50, timeout=1800)
```

#### 2. **Slow Convergence:**
```python
# Tăng n_startup_trials
pruner = optuna.pruners.MedianPruner(n_startup_trials=20)
```

#### 3. **Overfitting:**
```python
# Sử dụng cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5)
return scores.mean()
```

---

## 📝 Template hoàn chỉnh

```python
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import xgboost as xgb

def objective(trial):
    # Định nghĩa hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42
    }
    
    # Tạo model
    model = xgb.XGBClassifier(**params)
    
    # Cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    return scores.mean()

# Tạo study
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
)

# Optimize
study.optimize(objective, n_trials=100, timeout=3600)

# Kết quả
print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Train model cuối cùng
best_model = xgb.XGBClassifier(**study.best_params)
best_model.fit(X_train, y_train)
```

---

## 🎉 Kết luận

Optuna là công cụ mạnh mẽ giúp tối ưu hóa hyperparameters tự động cho các mô hình cây. Với hướng dẫn này, bạn có thể:

- ✅ Cài đặt và sử dụng Optuna
- ✅ Áp dụng cho các mô hình cây phổ biến
- ✅ Sử dụng visualization để phân tích
- ✅ Áp dụng best practices

**Chúc bạn thành công với Optuna! 🚀**
