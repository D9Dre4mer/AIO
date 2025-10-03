# Automated Training Scripts Guide - Comprehensive Machine Learning Platform

## Tổng Quan Automated Training Scripts

Dự án **Comprehensive Machine Learning Platform** bao gồm 3 automated training scripts được thiết kế để tự động hóa quá trình training và testing trên các dataset khác nhau. Các script này sử dụng cùng cache system với `app.py`, đảm bảo tính nhất quán và khả năng chia sẻ cache.

---

## 1. Automated Training Scripts Overview

### 1.1 Danh Sách Scripts

```python
# Automated Training Scripts
AUTOMATED_SCRIPTS = {
    "heart_dataset": "auto_train_heart_dataset.py",
    "large_dataset": "auto_train_large_dataset.py", 
    "spam_ham_dataset": "auto_train_spam_ham.py"
}
```

### 1.2 Mục Đích và Chức Năng

**Mục đích chính:**
- **Automated Testing**: Tự động test tất cả models trên specific datasets
- **Performance Benchmarking**: So sánh performance giữa các models
- **Cache Integration**: Sử dụng cùng cache system với Streamlit app
- **Comprehensive Evaluation**: Đánh giá toàn diện với multiple metrics

**Chức năng:**
- **Model Training**: Train tất cả available models
- **Hyperparameter Optimization**: Optuna integration
- **Ensemble Learning**: Voting và Stacking classifiers
- **Performance Analysis**: Comprehensive metrics và visualization

---

## 2. Auto Train Heart Dataset Script

### 2.1 Script Overview (`auto_train_heart_dataset.py`)

```python
#!/usr/bin/env python3
"""
Comprehensive Test Script for Heart Dataset
Tests all models with numerical data preprocessing including ensemble/stacking
Using the heart dataset: cache/heart.csv
"""
```

### 2.2 Dataset Configuration

```python
# Heart Dataset Configuration
HEART_DATASET_CONFIG = {
    "dataset_path": "cache/heart.csv",
    "dataset_type": "numerical",
    "target_column": "target",
    "preprocessing": {
        "scalers": ["StandardScaler", "MinMaxScaler", "RobustScaler"],
        "remove_duplicates": True,
        "outlier_detection": True
    },
    "data_split": {
        "test_size": 0.2,
        "validation_size": 0.1,
        "random_state": 42
    }
}
```

### 2.3 Models Testing

```python
def test_all_models():
    """Test all available models on heart dataset"""
    
    # Classification Models
    models_to_test = [
        'knn', 'decision_tree', 'naive_bayes', 'logistic_regression',
        'svm', 'random_forest', 'adaboost', 'gradient_boosting',
        'xgboost', 'lightgbm', 'catboost'
    ]
    
    # Ensemble Models
    ensemble_models = [
        'voting_ensemble_hard', 'voting_ensemble_soft',
        'stacking_ensemble_logistic_regression'
    ]
    
    # Test each model with different scalers
    for scaler in ['StandardScaler', 'MinMaxScaler', 'RobustScaler']:
        for model_name in models_to_test + ensemble_models:
            test_model_with_scaler(model_name, scaler)
```

### 2.4 Performance Metrics

```python
def evaluate_model_performance(model, X_test, y_test):
    """Comprehensive model evaluation"""
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    return metrics
```

---

## 3. Auto Train Large Dataset Script

### 3.1 Script Overview (`auto_train_large_dataset.py`)

```python
#!/usr/bin/env python3
"""
Automated Training Script for Large Dataset (300,000+ samples)
Optimized for memory efficiency and performance
"""
```

### 3.2 Large Dataset Configuration

```python
# Large Dataset Configuration
LARGE_DATASET_CONFIG = {
    "dataset_path": "data/20250822-004129_sample-300_000Samples.csv",
    "dataset_type": "text",
    "target_column": "categories",
    "preprocessing": {
        "vectorization_methods": ["TF-IDF", "BoW", "Embeddings"],
        "svd_reduction": True,
        "max_features": 30000,
        "memory_optimization": True
    },
    "training_config": {
        "sample_size": 10000,  # Reduced for testing
        "cv_folds": 3,         # Reduced for speed
        "timeout_per_model": 300  # 5 minutes per model
    }
}
```

### 3.3 Memory Optimization

```python
def optimize_for_large_dataset():
    """Memory optimization strategies for large datasets"""
    
    # 1. Chunked Processing
    chunk_size = 10000
    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
        process_chunk(chunk)
    
    # 2. Sparse Matrix Usage
    from scipy import sparse
    X_sparse = sparse.csr_matrix(X)
    
    # 3. Garbage Collection
    import gc
    gc.collect()
    
    # 4. Memory Monitoring
    import psutil
    memory_usage = psutil.virtual_memory().percent
    if memory_usage > 80:
        reduce_sample_size()
```

### 3.4 Performance Monitoring

```python
def monitor_training_performance():
    """Monitor training performance and resource usage"""
    
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    
    # Training process
    model.fit(X_train, y_train)
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used
    
    performance_stats = {
        'training_time': end_time - start_time,
        'memory_usage': end_memory - start_memory,
        'peak_memory': psutil.virtual_memory().percent
    }
    
    return performance_stats
```

---

## 4. Auto Train Spam Ham Script

### 4.1 Script Overview (`auto_train_spam_ham.py`)

```python
#!/usr/bin/env python3
"""
Automated Training Script for Spam/Ham Dataset
Tests text classification models with different vectorization methods
"""
```

### 4.2 Spam Dataset Configuration

```python
# Spam Dataset Configuration
SPAM_DATASET_CONFIG = {
    "dataset_path": "data/2cls_spam_text_cls.csv",
    "dataset_type": "text",
    "target_column": "label",
    "text_column": "text",
    "preprocessing": {
        "text_cleaning": True,
        "vectorization_methods": ["TF-IDF", "BoW", "Embeddings"],
        "max_features": 10000,
        "ngram_range": (1, 2)
    },
    "models": [
        'naive_bayes', 'logistic_regression', 'svm',
        'random_forest', 'xgboost', 'lightgbm'
    ]
}
```

### 4.3 Text Preprocessing

```python
def preprocess_text_data(texts):
    """Advanced text preprocessing for spam detection"""
    
    # 1. Text Cleaning
    cleaned_texts = []
    for text in texts:
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        cleaned_texts.append(text)
    
    # 2. Vectorization
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X_vectorized = vectorizer.fit_transform(cleaned_texts)
    
    return X_vectorized, vectorizer
```

---

## 5. Cache System Integration

### 5.1 Unified Cache Format

```python
# Cache Structure for Automated Scripts
CACHE_STRUCTURE = {
    "models": "cache/models/{model_name}/{dataset_id}/{config_hash}/",
    "artifacts": {
        "model": "model.pkl",
        "params": "params.json", 
        "metrics": "metrics.json",
        "config": "config.json",
        "fingerprint": "fingerprint.json"
    }
}
```

### 5.2 Dataset ID Format

```python
def generate_dataset_id(dataset_name, preprocessing_method):
    """Generate consistent dataset ID"""
    
    if dataset_name == "heart":
        return f"heart_dataset_{preprocessing_method}"
    elif dataset_name == "spam":
        return f"spam_ham_dataset_{preprocessing_method}"
    elif dataset_name == "large":
        return f"large_dataset_{preprocessing_method}"
    else:
        return f"{dataset_name}_{preprocessing_method}"
```

### 5.3 Config Hash Generation

```python
def generate_config_hash(model_params, preprocessing_config):
    """Generate consistent config hash"""
    
    config_string = json.dumps({
        'model_params': model_params,
        'preprocessing_config': preprocessing_config
    }, sort_keys=True)
    
    return hashlib.md5(config_string.encode()).hexdigest()[:16]
```

---

## 6. Performance Optimization

### 6.1 Memory Management

```python
class MemoryOptimizer:
    """Memory optimization for automated training"""
    
    def __init__(self, max_memory_percent=80):
        self.max_memory_percent = max_memory_percent
        self.memory_threshold = max_memory_percent * 1024**3  # GB to bytes
    
    def check_memory_usage(self):
        """Check current memory usage"""
        memory = psutil.virtual_memory()
        return memory.percent
    
    def optimize_data_loading(self, dataset_path):
        """Optimize data loading based on available memory"""
        available_memory = psutil.virtual_memory().available
        
        if available_memory < self.memory_threshold:
            # Use chunked loading
            return self.load_data_chunked(dataset_path)
        else:
            # Load full dataset
            return pd.read_csv(dataset_path)
```

### 6.2 Parallel Processing

```python
def parallel_model_training(models_config):
    """Parallel training of multiple models"""
    
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit training tasks
        future_to_model = {
            executor.submit(train_single_model, model_config): model_config['name']
            for model_config in models_config
        }
        
        # Collect results
        for future in future_to_model:
            model_name = future_to_model[future]
            try:
                results[model_name] = future.result()
            except Exception as exc:
                print(f'{model_name} generated an exception: {exc}')
    
    return results
```

---

## 7. Error Handling và Recovery

### 7.1 Robust Error Handling

```python
def robust_model_training(model_name, X_train, y_train, X_test, y_test):
    """Robust model training with error handling"""
    
    try:
        # Initialize model
        model = get_model_instance(model_name)
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate model
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        
        return {
            'model': model,
            'metrics': metrics,
            'training_time': training_time,
            'status': 'success'
        }
        
    except MemoryError:
        print(f"Memory error for {model_name}, reducing sample size")
        return retry_with_reduced_data(model_name, X_train, y_train)
        
    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")
        return {
            'model': None,
            'metrics': None,
            'training_time': 0,
            'status': 'failed',
            'error': str(e)
        }
```

### 7.2 Recovery Mechanisms

```python
def retry_with_reduced_data(model_name, X_train, y_train):
    """Retry training with reduced data size"""
    
    # Reduce data size by 50%
    sample_size = len(X_train) // 2
    X_reduced = X_train[:sample_size]
    y_reduced = y_train[:sample_size]
    
    try:
        model = get_model_instance(model_name)
        model.fit(X_reduced, y_reduced)
        
        return {
            'model': model,
            'metrics': None,
            'training_time': 0,
            'status': 'success_reduced',
            'sample_size': sample_size
        }
    except Exception as e:
        return {
            'model': None,
            'metrics': None,
            'training_time': 0,
            'status': 'failed_reduced',
            'error': str(e)
        }
```

---

## 8. Results Analysis và Reporting

### 8.1 Performance Comparison

```python
def generate_performance_report(results):
    """Generate comprehensive performance report"""
    
    report = {
        'summary': {
            'total_models_tested': len(results),
            'successful_models': len([r for r in results if r['status'] == 'success']),
            'failed_models': len([r for r in results if r['status'] == 'failed']),
            'total_training_time': sum([r['training_time'] for r in results])
        },
        'best_models': {
            'highest_accuracy': max(results, key=lambda x: x['metrics']['accuracy']),
            'fastest_training': min(results, key=lambda x: x['training_time']),
            'best_f1_score': max(results, key=lambda x: x['metrics']['f1_score'])
        },
        'detailed_results': results
    }
    
    return report
```

### 8.2 Visualization

```python
def create_performance_visualizations(results):
    """Create performance visualization charts"""
    
    # 1. Accuracy Comparison
    model_names = [r['model_name'] for r in results]
    accuracies = [r['metrics']['accuracy'] for r in results]
    
    plt.figure(figsize=(12, 8))
    plt.bar(model_names, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    
    # 2. Training Time Comparison
    training_times = [r['training_time'] for r in results]
    
    plt.figure(figsize=(12, 8))
    plt.bar(model_names, training_times)
    plt.title('Training Time Comparison')
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('training_time_comparison.png')
```

---

## 9. Usage Instructions

### 9.1 Running Automated Scripts

```bash
# Activate conda environment
conda activate PJ3.1

# Run heart dataset training
python auto_train_heart_dataset.py

# Run large dataset training
python auto_train_large_dataset.py

# Run spam dataset training
python auto_train_spam_ham.py
```

### 9.2 Configuration Options

```python
# Configuration for automated training
AUTOMATED_TRAINING_CONFIG = {
    "enable_optuna": True,
    "optuna_trials": 50,
    "enable_ensemble": True,
    "enable_caching": True,
    "memory_limit_gb": 8,
    "max_training_time_minutes": 30,
    "parallel_training": True,
    "save_results": True,
    "generate_plots": True
}
```

---

## 10. Best Practices

### 10.1 Performance Optimization

1. **Memory Management**: Monitor memory usage và optimize data loading
2. **Parallel Processing**: Sử dụng parallel training khi possible
3. **Caching**: Leverage cache system để avoid retraining
4. **Error Handling**: Implement robust error handling và recovery

### 10.2 Resource Management

1. **CPU Usage**: Monitor CPU usage và adjust parallel workers
2. **Memory Usage**: Set memory limits và implement cleanup
3. **Disk Usage**: Monitor cache size và implement cleanup
4. **Time Limits**: Set timeouts để avoid infinite training

### 10.3 Results Management

1. **Logging**: Comprehensive logging của training process
2. **Metrics**: Consistent metrics calculation và reporting
3. **Visualization**: Generate plots và charts cho analysis
4. **Export**: Export results trong multiple formats

---

## 🎯 Kết Luận

Automated Training Scripts cung cấp một cách thức hiệu quả để:

- **Automate Testing**: Tự động test tất cả models trên different datasets
- **Performance Benchmarking**: So sánh performance giữa các models
- **Cache Integration**: Sử dụng unified cache system
- **Comprehensive Evaluation**: Đánh giá toàn diện với multiple metrics

Các scripts này là essential components của Comprehensive Machine Learning Platform, providing automated testing và benchmarking capabilities cho all supported models và datasets.
