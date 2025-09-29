# Cache System - Hướng dẫn chi tiết

## 📋 Tổng quan

Cache System là một thành phần quan trọng của pipeline, giúp:
- **Tăng tốc độ**: Tránh retrain models không cần thiết
- **Đảm bảo consistency**: Kết quả reproducible
- **Tiết kiệm tài nguyên**: Không waste computational power
- **Tích hợp mượt mà**: Seamless với Step 5 analysis

## 🏗️ Kiến trúc Cache System

### Cache Structure
```
cache/
├── models/                    # Model cache storage
│   ├── {model_name}/         # Model identifier
│   │   ├── {dataset_id}/     # Dataset identifier
│   │   │   ├── {config_hash}/ # Configuration hash
│   │   │   │   ├── model.joblib      # Trained model
│   │   │   │   ├── metrics.json      # Performance metrics
│   │   │   │   ├── config.json       # Training configuration
│   │   │   │   ├── shap_sample.pkl   # SHAP analysis sample
│   │   │   │   └── metadata.json     # Cache metadata
├── training_results/         # Training results cache
├── embeddings/              # Text embeddings cache
└── cache_metadata.json      # Global cache metadata
```

### Cache Identifiers
- **Model Key**: Tên model (e.g., "random_forest", "xgboost")
- **Dataset ID**: Dataset identifier (e.g., "heart_dataset")
- **Config Hash**: Hash của training configuration
- **Dataset Fingerprint**: Hash của dataset characteristics

## 🔧 Cache Manager

### Core Functions

#### 1. Save Model Cache
```python
cache_manager.save_model_cache(
    model_key="random_forest",
    dataset_id="heart_dataset", 
    config_hash="abc123",
    dataset_fingerprint="def456",
    model=trained_model,
    params=model_params,
    metrics=performance_metrics,
    config=training_config,
    eval_predictions=test_data,
    shap_sample=shap_data
)
```

#### 2. Load Model Cache
```python
cache_data = cache_manager.load_model_cache(
    model_key="random_forest",
    dataset_id="heart_dataset",
    config_hash="abc123"
)
```

#### 3. Check Cache Exists
```python
exists, cache_info = cache_manager.check_cache_exists(
    model_key="random_forest",
    dataset_id="heart_dataset", 
    config_hash="abc123"
)
```

#### 4. List Cached Models
```python
cached_models = cache_manager.list_cached_models()
```

### Cache Validation

#### Automatic Validation
- **File Integrity**: Kiểm tra files tồn tại
- **Data Consistency**: Validate data format
- **Model Compatibility**: Check model compatibility
- **Metadata Validation**: Verify metadata

#### Manual Validation
```python
# Check specific cache
is_valid = cache_manager.validate_cache(cache_path)

# Validate all caches
validation_results = cache_manager.validate_all_caches()
```

## 📊 Cache Integration với Training

### Step 4 Training Process

#### 1. Pre-training Check
```python
# Check if model already cached
exists, cache_info = cache_manager.check_cache_exists(
    model_key=model_name,
    dataset_id=dataset_id,
    config_hash=config_hash
)

if exists:
    # Load from cache
    cache_data = cache_manager.load_model_cache(...)
    return cache_data
```

#### 2. Training Execution
```python
# Train model
model.fit(X_train, y_train)

# Evaluate model
metrics = evaluate_model(model, X_test, y_test)

# Prepare cache data
cache_data = {
    'model': model,
    'metrics': metrics,
    'eval_predictions': X_test,
    'shap_sample': X_test[:100]
}
```

#### 3. Post-training Save
```python
# Save to cache
cache_manager.save_model_cache(
    model_key=model_name,
    dataset_id=dataset_id,
    config_hash=config_hash,
    **cache_data
)
```

### Cache Hit/Miss Logic

#### Cache Hit
- **Condition**: Same model + dataset + config
- **Action**: Load from cache
- **Benefit**: Skip training, instant results

#### Cache Miss
- **Condition**: Different model/dataset/config
- **Action**: Train new model
- **Benefit**: Fresh training với updated config

## 🔍 Cache Integration với Step 5

### SHAP Analysis

#### Cache Requirements
- **Model**: Trained model object
- **SHAP Sample**: Sample data cho SHAP analysis
- **Feature Names**: Column names

#### Cache Usage
```python
# Load model với SHAP sample
cache_data = cache_manager.load_model_cache(...)

model = cache_data['model']
shap_sample = cache_data['shap_sample']

# Create SHAP explainer
explainer = create_shap_explainer(model, shap_sample)
shap_values = explainer.shap_values(shap_sample)
```

### Confusion Matrix

#### Cache Requirements
- **Model**: Trained model object
- **Eval Predictions**: Test data
- **Eval Labels**: True labels (optional)

#### Cache Usage
```python
# Load model với evaluation data
cache_data = cache_manager.load_model_cache(...)

model = cache_data['model']
eval_predictions = cache_data['eval_predictions']

# Generate predictions
y_pred = model.predict(eval_predictions)
cm = confusion_matrix(y_true, y_pred)
```

### Model Comparison

#### Cache Requirements
- **Multiple Models**: All trained models
- **Metrics**: Performance metrics
- **Metadata**: Model information

#### Cache Usage
```python
# Load all cached models
cached_models = cache_manager.list_cached_models()

comparison_data = []
for cache_info in cached_models:
    cache_data = cache_manager.load_model_cache(...)
    comparison_data.append({
        'model_name': cache_info['model_key'],
        'metrics': cache_data['metrics'],
        'model_type': type(cache_data['model']).__name__
    })
```

## ⚙️ Cache Configuration

### Cache Settings
```python
# config.py
CACHE_DIR = "cache"
CACHE_ENABLE = True
CACHE_MAX_SIZE = "10GB"
CACHE_CLEANUP_INTERVAL = 7  # days
```

### Cache Policies
- **LRU**: Least Recently Used
- **Size-based**: Remove largest caches first
- **Age-based**: Remove oldest caches first
- **Manual**: User-controlled cleanup

## 🧹 Cache Management

### Automatic Cleanup
```python
# Cleanup old caches
cache_manager.cleanup_old_caches(days=7)

# Cleanup by size
cache_manager.cleanup_by_size(max_size="5GB")

# Cleanup broken caches
cache_manager.cleanup_broken_caches()
```

### Manual Management
```python
# List all caches
cached_models = cache_manager.list_cached_models()

# Remove specific cache
cache_manager.remove_cache(model_key, dataset_id, config_hash)

# Clear all caches
cache_manager.clear_all_caches()
```

### Cache Statistics
```python
# Get cache statistics
stats = cache_manager.get_cache_stats()
print(f"Total caches: {stats['total_caches']}")
print(f"Total size: {stats['total_size']}")
print(f"Oldest cache: {stats['oldest_cache']}")
print(f"Newest cache: {stats['newest_cache']}")
```

## 🔧 Troubleshooting

### Common Issues

#### "Cache not found"
- **Cause**: Cache bị xóa hoặc corrupt
- **Solution**: Retrain model trong Step 4

#### "Cache loading failed"
- **Cause**: Model format không tương thích
- **Solution**: Clear cache và retrain

#### "SHAP sample missing"
- **Cause**: Cache không có SHAP sample
- **Solution**: Retrain với SHAP sample

#### "Cache size too large"
- **Cause**: Quá nhiều cached models
- **Solution**: Cleanup old caches

### Debug Commands
```python
# Check cache status
cache_manager.check_cache_status()

# Validate specific cache
cache_manager.validate_cache(cache_path)

# Get cache info
cache_info = cache_manager.get_cache_info(model_key, dataset_id, config_hash)
```

## 📈 Performance Optimization

### Cache Efficiency
1. **Smart Caching**: Chỉ cache khi cần thiết
2. **Compression**: Compress large models
3. **Lazy Loading**: Load on demand
4. **Parallel Loading**: Load multiple caches

### Memory Management
1. **Model Serialization**: Efficient serialization
2. **Sample Size**: Optimize SHAP sample size
3. **Cleanup**: Regular cleanup old caches
4. **Monitoring**: Monitor cache usage

## 🎯 Best Practices

### Cache Strategy
1. **Cache Key Design**: Meaningful, unique keys
2. **Data Validation**: Validate cached data
3. **Version Control**: Handle model versioning
4. **Error Handling**: Graceful cache failures

### Performance Tips
1. **Cache Hit Rate**: Optimize cache hit rate
2. **Cache Size**: Monitor cache size
3. **Cleanup Schedule**: Regular cleanup
4. **Backup Strategy**: Backup important caches

### Development
1. **Testing**: Test cache functionality
2. **Monitoring**: Monitor cache performance
3. **Documentation**: Document cache structure
4. **Maintenance**: Regular maintenance

## 🔮 Future Enhancements

### Planned Features
- **Distributed Caching**: Multi-machine cache
- **Cloud Storage**: Cloud-based cache storage
- **Cache Analytics**: Detailed cache analytics
- **Auto-optimization**: Automatic cache optimization

### Integration Improvements
- **Real-time Sync**: Real-time cache synchronization
- **Version Control**: Git-like cache versioning
- **API Integration**: REST API cho cache
- **Monitoring Dashboard**: Cache monitoring UI

## 📚 API Reference

### CacheManager Class
```python
class CacheManager:
    def __init__(self, cache_dir="cache"):
        """Initialize cache manager"""
    
    def save_model_cache(self, model_key, dataset_id, config_hash, **kwargs):
        """Save model to cache"""
    
    def load_model_cache(self, model_key, dataset_id, config_hash):
        """Load model from cache"""
    
    def check_cache_exists(self, model_key, dataset_id, config_hash):
        """Check if cache exists"""
    
    def list_cached_models(self):
        """List all cached models"""
    
    def remove_cache(self, model_key, dataset_id, config_hash):
        """Remove specific cache"""
    
    def clear_all_caches(self):
        """Clear all caches"""
    
    def get_cache_stats(self):
        """Get cache statistics"""
    
    def cleanup_old_caches(self, days=7):
        """Cleanup old caches"""
```

## 🎉 Conclusion

Cache System là một thành phần quan trọng giúp:
- **Tăng hiệu suất**: Tránh retrain không cần thiết
- **Đảm bảo consistency**: Kết quả reproducible
- **Tích hợp mượt mà**: Seamless với analysis tools
- **Quản lý hiệu quả**: Smart cache management

Sử dụng cache system để tối ưu hóa workflow và tăng productivity!
