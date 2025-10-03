# 🗂️ Cấu Trúc Cache System - Tài Liệu Chi Tiết

## 📋 Tổng Quan

Cache System được thiết kế để **tối ưu hóa performance** và **loại bỏ trùng lặp** trong Machine Learning Pipeline. Sau khi tối ưu hóa, hệ thống chỉ sử dụng **Individual Model Cache** thay vì Training Results Cache.

## 🏗️ Kiến Trúc Cache Hiện Tại

### **Cấu Trúc Thư Mục**
```
cache/
├── 📁 models/                           # Individual Model Cache (CHÍNH)
│   ├── 📁 logistic_regression/          # Model Type
│   │   ├── 📁 numeric_dataset_StandardScaler/  # Dataset + Preprocessing
│   │   │   ├── 📁 a1b2c3d4e5f6/        # Config Hash
│   │   │   │   ├── 📄 model.pkl         # Trained Model
│   │   │   │   ├── 📄 metrics.json      # Performance Metrics
│   │   │   │   ├── 📄 config.json       # Training Configuration
│   │   │   │   ├── 📄 params.json       # Model Parameters
│   │   │   │   ├── 📄 fingerprint.json  # Dataset Fingerprint
│   │   │   │   ├── 📄 eval_predictions.parquet  # Test Predictions
│   │   │   │   ├── 📄 shap_sample.parquet       # SHAP Sample Data
│   │   │   │   ├── 📄 shap_explainer.pkl        # SHAP Explainer
│   │   │   │   ├── 📄 shap_values.pkl           # SHAP Values
│   │   │   │   ├── 📄 feature_names.txt         # Feature Names
│   │   │   │   └── 📄 label_mapping.json        # Label Mapping
│   │   │   └── 📁 f6e5d4c3b2a1/        # Another Config Hash
│   │   └── 📁 text_dataset_TFIDF/       # Different Dataset + Preprocessing
│   ├── 📁 random_forest/
│   ├── 📁 xgboost/
│   ├── 📁 voting_ensemble_hard/
│   └── 📁 stacking_ensemble_logistic_regression/
│
├── 📁 shap/                             # SHAP Cache (Legacy - có thể xóa)
│   ├── 📄 *.pkl                        # SHAP cache files
│
├── 📁 confusion_matrices/               # Confusion Matrix Cache
│   ├── 📄 *.png                        # Confusion matrix images
│
├── 📁 training_results/                # Training Results Cache (KHÔNG CÒN SỬ DỤNG)
│   └── 📁 sessions/                     # Session-based cache
│
└── 📄 cache_metadata.json              # Global Cache Metadata
```

## 🔑 Cache Identifiers

### **1. Model Key** (`model_key`)
```python
# Individual Models
"logistic_regression"
"random_forest" 
"xgboost"
"svm"
"naive_bayes"
"knn"
"decision_tree"
"gradient_boosting"
"adaboost"
"catboost"
"lightgbm"

# Ensemble Models
"voting_ensemble_hard"
"voting_ensemble_soft"
"stacking_ensemble_logistic_regression"
"stacking_ensemble_random_forest"
```

### **2. Dataset ID** (`dataset_id`)
```python
# Numeric Datasets
"numeric_dataset_StandardScaler"
"numeric_dataset_RobustScaler"
"numeric_dataset_MinMaxScaler"

# Text Datasets  
"text_dataset_TFIDF"
"text_dataset_CountVectorizer"
"text_dataset_HashingVectorizer"

# Heart Dataset
"heart_dataset_StandardScaler"
"heart_dataset_RobustScaler"
```

### **3. Config Hash** (`config_hash`)
```python
# Generated from:
{
    'model': model_name,
    'preprocessing': scaler_name,
    'cv_folds': 5,
    'random_state': 42,
    'test_size': 0.2,
    'optuna_trials': 50
}
# Example: "a1b2c3d4e5f6789012345678901234567890abcd"
```

### **4. Dataset Fingerprint** (`dataset_fingerprint`)
```python
# Generated from:
{
    'dataset_path': 'heart.csv',
    'dataset_size': 1000,
    'num_rows': 1000,
    'num_features': 13,
    'target_column': 'target'
}
# Example: "def4567890123456789012345678901234567890efgh"
```

## 📄 File Structure Chi Tiết

### **1. Model File** (`model.pkl`)
```python
# Contains: Trained sklearn model object
# Format: Pickle
# Size: ~1-50MB (depending on model complexity)
# Example: LogisticRegression, RandomForestClassifier, etc.
```

### **2. Metrics File** (`metrics.json`)
```json
{
    "accuracy": 0.85,
    "validation_accuracy": 0.83,
    "f1_score": 0.84,
    "precision": 0.82,
    "recall": 0.86,
    "support": 200,
    "cv_mean": 0.83,
    "cv_std": 0.02,
    "training_time": 1.25
}
```

### **3. Config File** (`config.json`)
```json
{
    "model_name": "logistic_regression",
    "vec_method": "StandardScaler",
    "cv_folds": 5,
    "random_state": 42,
    "test_size": 0.2,
    "optuna_trials": 50
}
```

### **4. Params File** (`params.json`)
```json
{
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "max_iter": 1000,
    "random_state": 42
}
```

### **5. Eval Predictions** (`eval_predictions.parquet`)
```python
# DataFrame with columns:
# - feature_1, feature_2, ..., feature_n (test data)
# - true_labels (actual labels)
# - predictions (model predictions)
# - proba_class_0, proba_class_1 (if available)
```

### **6. SHAP Sample** (`shap_sample.parquet`)
```python
# DataFrame with columns:
# - feature_1, feature_2, ..., feature_n (sample data for SHAP)
# Size: Max 1000 samples (reduced from original for memory)
```

### **7. SHAP Explainer** (`shap_explainer.pkl`)
```python
# Contains: SHAP explainer object
# Types: TreeExplainer, KernelExplainer, LinearExplainer
# Size: ~1-10MB
```

### **8. SHAP Values** (`shap_values.pkl`)
```python
# Contains: SHAP values array
# Shape: (n_samples, n_features)
# Size: ~1-50MB (depending on sample size)
```

### **9. Feature Names** (`feature_names.txt`)
```text
age
sex
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
```

### **10. Label Mapping** (`label_mapping.json`)
```json
{
    "0": "no_heart_disease",
    "1": "heart_disease"
}
```

## 🔄 Cache Workflow

### **1. Training Phase**
```
1. Check cache exists → If yes, load from cache
2. If not, train model
3. Save to individual model cache:
   - model.pkl
   - metrics.json (INCLUDING training_time)
   - config.json
   - params.json
   - eval_predictions.parquet
   - shap_sample.parquet
   - shap_explainer.pkl
   - shap_values.pkl
   - feature_names.txt
   - label_mapping.json
```

### **2. Step 5 Analysis Phase**
```
1. Load cached models using cache_manager.list_cached_models()
2. For each model:
   - Load model.pkl
   - Load metrics.json
   - Load shap_explainer.pkl
   - Load shap_values.pkl
   - Load eval_predictions.parquet
3. Generate analysis without retraining
```

## ⚡ Tối Ưu Hóa Đã Thực Hiện

### **1. Loại Bỏ Training Results Cache**
```python
# TRƯỚC (gây delay):
Step 4: Train → Save individual cache → Display results → Save training results cache (DELAY!)

# SAU (tối ưu):
Step 4: Train → Save individual cache → Display results (NO DELAY!)
```

### **2. SHAP Cache Integration**
```python
# TRƯỚC:
Step 5: Load model → Create SHAP explainer → Generate SHAP values (SLOW!)

# SAU:
Step 5: Load model + SHAP explainer + SHAP values (FAST!)
```

### **3. Training Time Fix**
```python
# TRƯỚC:
metrics = {
    'accuracy': 0.85,
    'f1_score': 0.84,
    # ❌ THIẾU training_time
}

# SAU:
metrics = {
    'accuracy': 0.85,
    'f1_score': 0.84,
    'training_time': 1.25  # ✅ ĐÃ THÊM
}
```

## 🛠️ Cache Management

### **Cache Size Estimation**
```
Individual Model Cache:
- Model: ~1-50MB
- Metrics: ~1KB
- Config: ~1KB
- Eval Predictions: ~1-10MB
- SHAP Sample: ~1-5MB
- SHAP Explainer: ~1-10MB
- SHAP Values: ~1-50MB
- Total per model: ~5-125MB

For 10 models: ~50MB-1.25GB
```

### **Cache Cleanup**
```python
# Manual cleanup
cache_manager.clear_cache()  # Clear all
cache_manager.clear_cache(model_key="logistic_regression")  # Clear specific model

# Automatic cleanup (if implemented)
- Remove old cache (>30 days)
- Remove failed training cache
- Compress large files
```

## 🔍 Debugging Cache

### **Check Cache Status**
```python
# List all cached models
cached_models = cache_manager.list_cached_models()
print(f"Found {len(cached_models)} cached models")

# Check specific model
cache_exists, cache_info = cache_manager.check_cache_exists(
    model_key="logistic_regression",
    dataset_id="heart_dataset_StandardScaler", 
    config_hash="abc123",
    dataset_fingerprint="def456"
)
```

### **Common Cache Issues**
1. **Missing training_time**: Fixed in metrics.json
2. **SHAP cache missing**: Check shap_explainer.pkl and shap_values.pkl
3. **Cache corruption**: Delete cache directory and retrain
4. **Memory issues**: Reduce SHAP sample size

## 📊 Performance Impact

### **Before Optimization**
- Step 4: Train + Save individual cache + Save training results cache (SLOW)
- Step 5: Load training results cache (SLOW)
- Total delay: ~30-60 seconds after result table

### **After Optimization**
- Step 4: Train + Save individual cache (FAST)
- Step 5: Load individual model cache (FAST)
- Total delay: ~0 seconds after result table

### **Memory Usage**
- Individual cache: ~5-125MB per model
- Training results cache: ~50-500MB (REMOVED)
- Total savings: ~50-500MB per session

## 🎯 Best Practices

### **Cache Usage**
1. **Always check cache first** before training
2. **Use consistent identifiers** for reproducible results
3. **Monitor cache size** to avoid disk space issues
4. **Clean up old cache** regularly

### **Development**
1. **Test cache loading** after changes
2. **Verify SHAP cache** works correctly
3. **Check training_time** is saved
4. **Validate cache structure** matches expectations

## 🔮 Future Improvements

### **Potential Enhancements**
1. **Cache compression**: Compress large files
2. **Cache versioning**: Handle cache format changes
3. **Distributed cache**: Share cache across machines
4. **Cache analytics**: Track cache hit rates
5. **Auto cleanup**: Automatic cache management

## 📝 Summary

Cache System sau tối ưu hóa:
- ✅ **Individual Model Cache**: Hoàn chỉnh với SHAP và training_time
- ✅ **Performance**: Không còn delay sau result table
- ✅ **Memory**: Tiết kiệm 50-500MB per session
- ✅ **Functionality**: Step 5 hoạt động hoàn toàn với individual cache
- ✅ **Maintainability**: Cấu trúc rõ ràng, dễ debug

**Cache System đã được tối ưu hóa hoàn toàn! 🚀**
