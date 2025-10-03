# Auto Train Scripts Setup Guide

## Tổng Quan

Các file auto_train scripts (`auto_train_spam_ham.py`, `auto_train_large_dataset.py`, `auto_train_heart_dataset.py`) đã được cấu hình để sử dụng cùng cache system với `app.py`, đảm bảo tính nhất quán và khả năng chia sẻ cache giữa các script.

## Cấu Hình Cache System

### 1. Cache Format Thống Nhất

Tất cả auto_train scripts sử dụng cùng cache format với `app.py`:

```
cache/models/{model_name}/{dataset_id}/{config_hash}/
├── model.pkl
├── params.json
├── metrics.json
├── config.json
└── fingerprint.json
```

### 2. Dataset ID Format

- **Text Data**: `{dataset_name}_{vectorization_method}`
  - Ví dụ: `spam_ham_dataset_TF-IDF`, `large_dataset_TF-IDF`
- **Numerical Data**: `{dataset_name}_{preprocessing_method}`
  - Ví dụ: `heart_dataset_StandardScaler`

### 3. Config Hash Structure

```json
{
  "model": "random_forest",
  "vectorization": "TF-IDF",  // hoặc "preprocessing": "StandardScaler"
  "trials": 10,
  "random_state": 42,
  "test_size": 0.2
}
```

## Chi Tiết Từng Script

### 1. auto_train_spam_ham.py

**Dataset**: Spam/Ham Text Classification
- **File**: `data/2cls_spam_text_cls.csv`
- **Text Column**: `Message`
- **Label Column**: `Category`
- **Vectorization Methods**: TF-IDF, BoW, Word Embeddings
- **Cache ID**: `spam_ham_dataset_{method}`

**Setup Parameters**:
```python
step3_data = {
    'optuna_config': {
        'trials': 10,           # Aligned with app.py default
        'timeout': 30,
        'direction': 'maximize'
    },
    'selected_models': get_all_models(),
    'vectorization_config': {
        'selected_methods': ['TF-IDF', 'BoW', 'Word Embeddings']
    },
    'selected_vectorization': ['TF-IDF', 'BoW', 'Word Embeddings']
}
```

### 2. auto_train_large_dataset.py

**Dataset**: Large Text Dataset (ArXiv Papers)
- **File**: `data/20250822-004129_sample-300_000Samples.csv`
- **Text Column**: `title`
- **Label Column**: `label`
- **Vectorization Methods**: TF-IDF, BoW, Word Embeddings
- **Cache ID**: `large_dataset_{method}`

**Setup Parameters**:
```python
step3_data = {
    'optuna_config': {
        'trials': 10,           # Aligned with app.py default
        'timeout': 30,
        'direction': 'maximize'
    },
    'selected_models': get_all_models(),
    'vectorization_config': {
        'selected_methods': ['TF-IDF', 'BoW', 'Word Embeddings']
    },
    'selected_vectorization': ['TF-IDF', 'BoW', 'Word Embeddings']
}
```

### 3. auto_train_heart_dataset.py

**Dataset**: Heart Disease Classification
- **File**: `data/heart.csv`
- **Feature Columns**: All numerical columns except target
- **Label Column**: `target`
- **Preprocessing Methods**: StandardScaler, MinMaxScaler, NoScaling
- **Cache ID**: `heart_dataset_{method}`

**Setup Parameters**:
```python
step3_data = {
    'optuna_config': {
        'trials': 10,           # Aligned with app.py default
        'timeout': 30,
        'direction': 'maximize'
    },
    'selected_models': get_all_models(),
    'preprocessing_config': {
        'selected_methods': ['StandardScaler', 'MinMaxScaler', 'NoScaling']
    }
}
```

## Cache Manager Integration

### Direct Cache Manager Usage

Tất cả auto_train scripts sử dụng `CacheManager` trực tiếp thay vì `ComprehensiveEvaluator` để đảm bảo cache format tương đồng với `app.py`:

```python
from cache_manager import CacheManager

# Generate cache identifiers using app.py format
cache_manager = CacheManager()
model_key = model_name
dataset_id = f"{dataset_name}_{method}"
config_hash = cache_manager.generate_config_hash({
    'model': model_name,
    'vectorization': method,  # hoặc 'preprocessing'
    'trials': 10,
    'random_state': 42,
    'test_size': 0.2
})
dataset_fingerprint = cache_manager.generate_dataset_fingerprint(
    dataset_path=dataset_path,
    dataset_size=os.path.getsize(dataset_path),
    num_rows=len(X_train)
)
```

### Cache Check và Save

```python
# Check if cache exists
cache_exists, cached_data = cache_manager.check_cache_exists(
    model_key, dataset_id, config_hash, dataset_fingerprint
)

if cache_exists:
    print(f"💾 Cache hit! Loading cached results for {model_name}")
    return {'status': 'success', 'cache_hit': True}
else:
    # Train model and save cache
    cache_manager.save_model_cache(
        model_key=model_key,
        dataset_id=dataset_id,
        config_hash=config_hash,
        dataset_fingerprint=dataset_fingerprint,
        model=model,
        params=model.get_params(),
        metrics=metrics,
        config=config
    )
```

## Text Vectorization Methods

### TF-IDF Vectorization

```python
from text_encoders import TextVectorizer
vectorizer = TextVectorizer()

# Create TF-IDF features
X_train_tfidf = vectorizer.fit_transform_tfidf(X_train)
X_test_tfidf = vectorizer.transform_tfidf(X_test)
```

### Bag of Words (BoW)

```python
# Create BoW features
X_train_bow = vectorizer.fit_transform_bow(X_train)
X_test_bow = vectorizer.transform_bow(X_test)
```

### Word Embeddings

```python
from text_encoders import EmbeddingVectorizer
vectorizer = EmbeddingVectorizer(
    model_name='all-MiniLM-L6-v2',
    device='cpu'
)
X_train_embeddings = vectorizer.fit_transform(X_train)
X_test_embeddings = vectorizer.transform(X_test)
```

## Numerical Preprocessing

### StandardScaler

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Data Splitting

Tất cả scripts sử dụng cùng data splitting strategy:

```python
from sklearn.model_selection import train_test_split

# Split data: 80% train, 10% val, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
```

## Model Training và Evaluation

### Model Training

```python
from models import model_registry

# Get model class
model_class = model_registry.get_model(model_name)
model = model_class()

# Train model
model.fit(X_train_processed, y_train)

# Evaluate
y_pred = model.predict(X_test_processed)
```

### Metrics Calculation

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred, average='weighted'),
    'precision': precision_score(y_test, y_pred, average='weighted'),
    'recall': recall_score(y_test, y_pred, average='weighted')
}
```

## Cache Reuse với app.py

### Cache Hit Scenario

Khi auto_train scripts chạy lần thứ 2, chúng sẽ sử dụng cache từ lần chạy đầu tiên:

```
💾 Cache hit! Loading cached results for random_forest
```

### Cache Compatibility

- ✅ **Same Dataset ID Format**: `{dataset_name}_{method}`
- ✅ **Same Config Structure**: Model, method, trials, random_state, test_size
- ✅ **Same Cache Directory**: `cache/models/{model_name}/{dataset_id}/`
- ✅ **Same File Structure**: model.pkl, params.json, metrics.json, config.json, fingerprint.json

## Lưu Ý Quan Trọng

### 1. Environment Setup

Đảm bảo chạy trong môi trường conda `PJ3.1`:

```bash
conda activate PJ3.1
```

### 2. Required Imports

```python
from datetime import datetime  # Required for timestamp
from cache_manager import CacheManager
from models import model_registry
from text_encoders import TextVectorizer, EmbeddingVectorizer
```

### 3. Error Handling

Tất cả scripts có error handling để tránh crash:

```python
try:
    # Training logic
    pass
except Exception as e:
    print(f"❌ Error: {e}")
    return {'status': 'failed', 'error': str(e)}
```

## Kết Luận

Các auto_train scripts đã được cấu hình hoàn hảo để:

1. **Sử dụng cùng cache system** với `app.py`
2. **Chia sẻ cache** giữa các script và app.py
3. **Đảm bảo tính nhất quán** trong training và evaluation
4. **Tối ưu hóa performance** thông qua cache reuse
5. **Không ảnh hưởng** đến hoạt động của `app.py`

Việc setup này cho phép người dùng có thể train models thông qua auto_train scripts và sau đó sử dụng kết quả trong `app.py` mà không cần retrain.
