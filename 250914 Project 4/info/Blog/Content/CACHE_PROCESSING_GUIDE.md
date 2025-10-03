# Hướng Dẫn Xử Lý Cache của Dự Án

## Tổng Quan

Dự án này sử dụng một hệ thống cache phức tạp và tinh vi để tối ưu hóa performance, đảm bảo reproducibility và tiết kiệm tài nguyên computational. Cache system được thiết kế với kiến trúc hierarchical và intelligent caching strategies.

## 1. Kiến Trúc Cache System

### 1.1 Cache Hierarchy

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
├── 📁 shap/                             # SHAP Cache (Legacy)
│   ├── 📁 explainers/
│   └── 📁 values/
│
├── 📁 confusion_matrices/               # Confusion Matrix Cache
│   ├── 📄 model_dataset_scaler.png
│   └── 📄 ...
│
└── 📁 training_results/                 # Training Results Cache (Legacy)
    ├── 📁 session_hash/
    └── 📄 ...
```

### 1.2 Cache Identifiers

- **Model Key**: Tên model (e.g., "random_forest", "xgboost", "voting_ensemble_hard")
- **Dataset ID**: Dataset identifier bao gồm dataset type và preprocessing method (e.g., "numeric_dataset_StandardScaler", "text_dataset_TFIDF")
- **Config Hash**: SHA256 hash của training configuration
- **Dataset Fingerprint**: Hash của dataset characteristics

## 2. Cache Manager Core

### 2.1 CacheManager Class

```python
class CacheManager:
    """Manages per-model caching với config_hash và dataset_fingerprint"""
    
    def __init__(self, cache_root_dir: str = "cache/models/"):
        self.cache_root_dir = Path(cache_root_dir)
        self.cache_root_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache structure definition
        self.cache_structure = {
            'model_artifact': 'model.{ext}',
            'params': 'params.json',
            'metrics': 'metrics.json', 
            'config': 'config.json',
            'fingerprint': 'fingerprint.json',
            'eval_predictions': 'eval_predictions.parquet',
            'shap_sample': 'shap_sample.parquet',
            'feature_names': 'feature_names.txt',
            'label_mapping': 'label_mapping.json'
        }
```

### 2.2 Cache Key Generation

```python
def generate_config_hash(self, config: Dict[str, Any]) -> str:
    """Generate SHA256 hash từ configuration"""
    # Normalize config by sorting keys và converting to JSON
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()

def generate_dataset_fingerprint(self, dataset_path: str, dataset_size: int, 
                               num_rows: int) -> str:
    """Generate dataset fingerprint"""
    fingerprint_data = {
        'dataset_path': dataset_path,
        'dataset_size': dataset_size,
        'num_rows': num_rows,
        'timestamp': datetime.now().isoformat()
    }
    fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()
```

### 2.3 Cache Path Generation

```python
def get_cache_path(self, model_key: str, dataset_id: str, config_hash: str) -> Path:
    """Get cache path cho model"""
    return self.cache_root_dir / model_key / dataset_id / config_hash

def get_cache_path_components(self, model_key: str, dataset_id: str, config_hash: str) -> Dict[str, str]:
    """Get human-readable cache path components"""
    return {
        'model_key': model_key,
        'dataset_id': dataset_id,
        'config_hash': config_hash,
        'cache_path': str(self.get_cache_path(model_key, dataset_id, config_hash))
    }
```

## 3. Cache Operations

### 3.1 Save Model Cache

```python
def save_model_cache(self, model_key: str, dataset_id: str, config_hash: str,
                    dataset_fingerprint: str, model, params: Dict[str, Any],
                    metrics: Dict[str, Any], config: Dict[str, Any],
                    eval_predictions: Optional[pd.DataFrame] = None,
                    shap_sample: Optional[pd.DataFrame] = None,
                    shap_explainer: Optional[Any] = None,
                    shap_values: Optional[Any] = None,
                    feature_names: Optional[List[str]] = None,
                    label_mapping: Optional[Dict[str, Any]] = None) -> str:
    """Save model cache với comprehensive error handling"""
    
    try:
        print(f"Starting cache save for {model_key}")
        
        cache_path = self.get_cache_path(model_key, dataset_id, config_hash)
        
        # Create cache directory
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Save model artifact với appropriate extension
        model_ext = self._get_model_extension(model_key)
        model_file = cache_path / f"model.{model_ext}"
        self._save_model_artifact(model, model_file, model_key)
        
        # Custom JSON serializer để handle non-serializable objects
        def safe_json_serializer(obj):
            """Convert non-serializable objects to strings"""
            try:
                if hasattr(obj, '__module__'):
                    return f"<{obj.__class__.__name__} object>"
                elif hasattr(obj, '__dict__'):
                    return str(obj)
                else:
                    return str(obj)
            except Exception as e:
                return f"<SerializationError: {str(e)}>"
        
        # Save params, metrics, config
        for data_type, filename in [('params', 'params.json'), 
                                   ('metrics', 'metrics.json'), 
                                   ('config', 'config.json')]:
            data_file = cache_path / filename
            with open(data_file, 'w') as f:
                json.dump(locals()[data_type], f, indent=2, default=safe_json_serializer)
        
        # Save fingerprint với metadata
        fingerprint_data = {
            'config_hash': config_hash,
            'dataset_fingerprint': dataset_fingerprint,
            'created_at': datetime.now().isoformat(),
            'model_key': model_key,
            'dataset_id': dataset_id,
            'library_versions': self._get_library_versions()
        }
        fingerprint_file = cache_path / 'fingerprint.json'
        with open(fingerprint_file, 'w') as f:
            json.dump(fingerprint_data, f, indent=2, default=safe_json_serializer)
        
        # Save eval predictions nếu provided
        if eval_predictions is not None:
            eval_file = cache_path / 'eval_predictions.parquet'
            eval_predictions.to_parquet(eval_file, index=False)
        
        # Save SHAP data nếu provided
        if shap_sample is not None:
            shap_file = cache_path / 'shap_sample.parquet'
            shap_sample.to_parquet(shap_file, index=False)
        
        if shap_explainer is not None and shap_values is not None:
            # Save SHAP explainer và values
            shap_explainer_file = cache_path / 'shap_explainer.pkl'
            with open(shap_explainer_file, 'wb') as f:
                pickle.dump(shap_explainer, f)
            
            shap_values_file = cache_path / 'shap_values.pkl'
            with open(shap_values_file, 'wb') as f:
                pickle.dump(shap_values, f)
        
        # Save feature names và label mapping
        if feature_names is not None:
            feature_file = cache_path / 'feature_names.txt'
            with open(feature_file, 'w') as f:
                f.write('\n'.join(feature_names))
        
        if label_mapping is not None:
            label_file = cache_path / 'label_mapping.json'
            with open(label_file, 'w') as f:
                json.dump(label_mapping, f, indent=2, default=safe_json_serializer)
        
        print(f"✅ Cache saved successfully: {cache_path}")
        return str(cache_path)
        
    except Exception as e:
        logger.error(f"Failed to save cache for {model_key}: {e}")
        raise
```

### 3.2 Load Model Cache

```python
def load_model_cache(self, model_key: str, dataset_id: str, config_hash: str) -> Dict[str, Any]:
    """Load model cache với comprehensive error handling"""
    
    try:
        cache_path = self.get_cache_path(model_key, dataset_id, config_hash)
        
        if not cache_path.exists():
            logger.warning(f"Cache not found: {cache_path}")
            return {}
        
        cache_data = {}
        
        # Load model artifact
        model_ext = self._get_model_extension(model_key)
        model_file = cache_path / f"model.{model_ext}"
        if model_file.exists():
            cache_data['model'] = self._load_model_artifact(model_file, model_key)
        
        # Load JSON files
        for data_type, filename in [('params', 'params.json'), 
                                   ('metrics', 'metrics.json'), 
                                   ('config', 'config.json')]:
            data_file = cache_path / filename
            if data_file.exists():
                with open(data_file, 'r') as f:
                    cache_data[data_type] = json.load(f)
        
        # Load fingerprint
        fingerprint_file = cache_path / 'fingerprint.json'
        if fingerprint_file.exists():
            with open(fingerprint_file, 'r') as f:
                cache_data['fingerprint'] = json.load(f)
        
        # Load eval predictions
        eval_file = cache_path / 'eval_predictions.parquet'
        if eval_file.exists():
            cache_data['eval_predictions'] = pd.read_parquet(eval_file)
        
        # Load SHAP data
        shap_file = cache_path / 'shap_sample.parquet'
        if shap_file.exists():
            cache_data['shap_sample'] = pd.read_parquet(shap_file)
        
        shap_explainer_file = cache_path / 'shap_explainer.pkl'
        if shap_explainer_file.exists():
            with open(shap_explainer_file, 'rb') as f:
                cache_data['shap_explainer'] = pickle.load(f)
        
        shap_values_file = cache_path / 'shap_values.pkl'
        if shap_values_file.exists():
            with open(shap_values_file, 'rb') as f:
                cache_data['shap_values'] = pickle.load(f)
        
        # Load feature names và label mapping
        feature_file = cache_path / 'feature_names.txt'
        if feature_file.exists():
            with open(feature_file, 'r') as f:
                cache_data['feature_names'] = f.read().strip().split('\n')
        
        label_file = cache_path / 'label_mapping.json'
        if label_file.exists():
            with open(label_file, 'r') as f:
                cache_data['label_mapping'] = json.load(f)
        
        logger.info(f"Cache loaded successfully: {cache_path}")
        return cache_data
        
    except Exception as e:
        logger.error(f"Failed to load cache for {model_key}: {e}")
        return {}
```

### 3.3 Cache Existence Check

```python
def check_cache_exists(self, model_key: str, dataset_id: str, config_hash: str) -> Tuple[bool, Dict[str, Any]]:
    """Check if cache exists và return cache info"""
    
    try:
        cache_path = self.get_cache_path(model_key, dataset_id, config_hash)
        
        if not cache_path.exists():
            return False, {}
        
        # Check fingerprint file
        fingerprint_file = cache_path / 'fingerprint.json'
        if not fingerprint_file.exists():
            return False, {}
        
        # Load fingerprint data
        with open(fingerprint_file, 'r') as f:
            fingerprint_data = json.load(f)
        
        # Validate config hash
        if fingerprint_data.get('config_hash') != config_hash:
            logger.warning(f"Config hash mismatch for {model_key}")
            return False, {}
        
        # Load cache info
        cache_info = {
            'cache_path': str(cache_path),
            'created_at': fingerprint_data.get('created_at'),
            'config_hash': config_hash,
            'dataset_fingerprint': fingerprint_data.get('dataset_fingerprint'),
            'model_key': model_key,
            'dataset_id': dataset_id
        }
        
        return True, cache_info
        
    except Exception as e:
        logger.error(f"Error checking cache for {model_key}: {e}")
        return False, {}
```

## 4. SHAP Cache Manager

### 4.1 SHAPCacheManager Class

```python
class SHAPCacheManager:
    """Manages SHAP explainer và values caching với memory leak protection"""
    
    def __init__(self, cache_dir: str = "cache/shap/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory leak protection
        self._lock = threading.Lock()
        self._active_explainers = {}  # Track active explainers
        self._max_memory_mb = 2000  # Max memory usage in MB
        self._max_sample_size = 500  # Max sample size để prevent memory issues
        
        # Cleanup old cache files on init
        self._cleanup_old_cache()
```

### 4.2 Memory Safety Operations

```python
@contextmanager
def _memory_safe_operation(self):
    """Context manager cho memory-safe operations"""
    try:
        # Force garbage collection before checking memory
        gc.collect()
        
        # Check memory before operation
        if not self._check_memory_usage():
            print("Warning: High memory usage detected, attempting cleanup...")
            # Try aggressive cleanup
            gc.collect()
            time.sleep(0.1)  # Brief pause for cleanup
            
            # Check again after cleanup
            if not self._check_memory_usage():
                print("Warning: Still high memory usage after cleanup, skipping SHAP operation")
                yield False
                return
            else:
                print("INFO: Memory usage reduced after cleanup, proceeding with SHAP operation")
        
        yield True
    finally:
        # Aggressive cleanup after operation
        gc.collect()
        time.sleep(0.05)  # Brief pause for cleanup

def _check_memory_usage(self):
    """Check if memory usage is within limits"""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"DEBUG: Current memory usage: {memory_mb:.1f} MB (limit: {self._max_memory_mb} MB)")
        return memory_mb < self._max_memory_mb
    except Exception:
        return True  # If can't check, assume OK
```

### 4.3 SHAP Cache Operations

```python
def generate_shap_cache_key(self, model, sample_data, model_name=""):
    """Generate cache key cho SHAP explainer và values với safety checks"""
    try:
        # Create hash từ model và sample data (với safety checks)
        try:
            model_params = model.get_params() if hasattr(model, 'get_params') else {}
        except Exception:
            model_params = {}
        
        model_str = str(type(model)) + str(model_params)
        
        # Safe sample data handling
        if hasattr(sample_data, 'shape'):
            sample_str = f"shape_{sample_data.shape}_dtype_{sample_data.dtype}"
        else:
            sample_str = str(type(sample_data))
        
        cache_key_data = f"{model_str}_{sample_str}_{model_name}"
        cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
        
        return cache_key
        
    except Exception as e:
        print(f"Warning: Failed to generate SHAP cache key: {e}")
        return f"fallback_{int(time.time())}"

def save_shap_cache(self, cache_key: str, explainer, shap_values, sample_data):
    """Save SHAP explainer và values to cache"""
    try:
        with self._memory_safe_operation() as safe:
            if not safe:
                return False
            
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            # Save với memory protection
            cache_data = {
                'explainer': explainer,
                'shap_values': shap_values,
                'sample_data': sample_data,
                'created_at': time.time()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"✅ SHAP cache saved: {cache_key}")
            return True
            
    except Exception as e:
        print(f"Warning: Failed to save SHAP cache: {e}")
        return False

def load_shap_cache(self, cache_key: str):
    """Load SHAP explainer và values từ cache"""
    try:
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # Check cache age (7 days max)
        cache_age = time.time() - cache_file.stat().st_mtime
        if cache_age > (7 * 24 * 60 * 60):  # 7 days
            cache_file.unlink()  # Remove old cache
            return None
        
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        print(f"✅ SHAP cache loaded: {cache_key}")
        return cache_data
        
    except Exception as e:
        print(f"Warning: Failed to load SHAP cache: {e}")
        return None
```

## 5. Confusion Matrix Cache

### 5.1 ConfusionMatrixCache Class

```python
class ConfusionMatrixCache:
    """Generates confusion matrices từ cached evaluation predictions"""
    
    def __init__(self, cache_root_dir: str = "cache/models/"):
        self.cache_root_dir = Path(cache_root_dir)
        self.cache_manager = cache_manager
    
    def generate_confusion_matrix_from_cache(self, model_key: str, dataset_id: str, 
                                           config_hash: str, 
                                           normalize: str = "true",
                                           labels_order: Optional[List[str]] = None,
                                           save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate confusion matrix từ cached eval_predictions"""
        
        try:
            # Load cached data
            cache_data = self.cache_manager.load_model_cache(model_key, dataset_id, config_hash)
            
            if cache_data['eval_predictions'] is None:
                raise ValueError(f"No eval_predictions found in cache for {model_key}")
            
            eval_df = cache_data['eval_predictions']
            
            # Extract true labels và predictions
            if 'y_true' in eval_df.columns:
                y_true = eval_df['y_true'].values
            elif 'true_labels' in eval_df.columns:
                y_true = eval_df['true_labels'].values
            else:
                raise ValueError("No true labels column found. Expected 'y_true' or 'true_labels'")
            
            y_pred = self._extract_predictions(eval_df)
            
            # Get label mapping
            label_mapping = cache_data.get('label_mapping', {})
            if not label_mapping:
                # Create default mapping using integer keys
                unique_labels = sorted(set(y_true) | set(y_pred))
                label_mapping = {int(i): f"Class_{i}" for i in unique_labels}
            else:
                # Convert string keys to integer keys nếu needed
                label_mapping = {int(k): v for k, v in label_mapping.items()}
            
            # Apply label ordering nếu provided
            if labels_order:
                label_mapping = {k: v for k, v in label_mapping.items() if v in labels_order}
            
            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=list(label_mapping.keys()))
            
            # Normalize nếu requested
            if normalize:
                cm_normalized = self._normalize_confusion_matrix(cm, normalize)
            else:
                cm_normalized = cm
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot confusion matrix
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                       xticklabels=list(label_mapping.values()),
                       yticklabels=list(label_mapping.values()),
                       ax=ax)
            
            ax.set_title(f'Confusion Matrix: {model_key} on {dataset_id}')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            
            # Save plot nếu requested
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Confusion matrix saved: {save_path}")
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_true, y_pred, label_mapping)
            
            return {
                'confusion_matrix': cm,
                'confusion_matrix_normalized': cm_normalized,
                'plot': fig,
                'metrics': metrics,
                'label_mapping': label_mapping
            }
            
        except Exception as e:
            logger.error(f"Failed to generate confusion matrix: {e}")
            raise
```

## 6. Training Pipeline Cache Integration

### 6.1 Cache Key Generation trong Training Pipeline

```python
def _generate_cache_key(self, step1_data: Dict, step2_data: Dict, step3_data: Dict) -> str:
    """Generate cache key từ wizard steps data"""
    
    # Extract key information từ steps
    dataset_info = step1_data.get('dataset_info', {})
    preprocessing_info = step2_data.get('preprocessing_info', {})
    model_info = step3_data.get('model_info', {})
    
    # Create human-readable name
    dataset_name = dataset_info.get('name', 'unknown_dataset')
    preprocessing_method = preprocessing_info.get('method', 'unknown_method')
    model_name = model_info.get('name', 'unknown_model')
    
    human_name = f"{model_name}_{dataset_name}_{preprocessing_method}"
    
    # Generate config hash
    config_data = {
        'step1': step1_data,
        'step2': step2_data,
        'step3': step3_data
    }
    
    config_hash = self._generate_config_hash(config_data)
    config_hash_str = config_hash[:8]  # Use first 8 characters
    
    # Return human-readable name với hash
    return f"{human_name}_{config_hash_str}"
```

### 6.2 Cache Check và Load

```python
def _check_cache(self, cache_key: str) -> Dict:
    """Check if results exist in cache - NO TIME EXPIRY"""
    if cache_key in self.cache_metadata:
        cache_info = self.cache_metadata[cache_key]
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # Check if cache file exists (NO TIME EXPIRY)
        if os.path.exists(cache_file):
            cache_age = time.time() - cache_info['timestamp']
            
            try:
                with open(cache_file, 'rb') as f:
                    cached_results = pickle.load(f)
                
                # Display cache hit information
                cache_name = cache_info.get('cache_name', cache_key)
                print(f"✅ Using cached results: {cache_name}")
                print(f"   Age: {cache_age/3600:.1f}h | File: {cache_key}")
                
                return cached_results
            except Exception as e:
                print(f"Warning: Could not load cached results: {e}")
    
    # If exact cache not found, try to find compatible cache
    return self._find_compatible_cache(cache_key)
```

### 6.3 Cache Save trong Training Pipeline

```python
def _save_cache(self, cache_key: str, results: Dict, cache_name: str = None):
    """Save results to cache"""
    try:
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # Save results
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Update metadata
        self.cache_metadata[cache_key] = {
            'timestamp': time.time(),
            'cache_name': cache_name or cache_key,
            'file_size': os.path.getsize(cache_file)
        }
        
        # Save metadata
        self._save_cache_metadata()
        
        print(f"✅ Results cached: {cache_name or cache_key}")
        
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")
```

## 7. Cache Configuration

### 7.1 Cache Settings trong config.py

```python
# Cache Configuration
CACHE_MODELS_ROOT_DIR = "cache/models/"
CACHE_STACKING_ROOT_DIR = "cache/stacking/"
CACHE_FORCE_RETRAIN = False
CACHE_USE_CACHE = True

# SHAP Configuration
SHAP_ENABLE = True
SHAP_SAMPLE_SIZE = 5000
SHAP_OUTPUT_DIR = "info/Result/"

# Stacking Configuration
STACKING_CACHE_OUTPUT_DIR = "cache/stacking/"
STACKING_CACHE_FORMAT = "parquet"  # "parquet" | "csv"
```

### 7.2 Memory Optimization Settings

```python
# Memory optimization thresholds
KMEANS_SVD_THRESHOLD = 20000  # Use SVD nếu features > 20K
KMEANS_SVD_COMPONENTS = 2000  # Reduce to 2K dimensions
MAX_VOCABULARY_SIZE = 30000   # Maximum vocabulary cho BoW/TF-IDF

# BoW/TF-IDF SVD dimensionality reduction cho speed optimization
BOW_TFIDF_SVD_COMPONENTS = 400  # Reduce to 400 dimensions
BOW_TFIDF_SVD_THRESHOLD = 200   # Apply SVD nếu features > 200
```

## 8. Cache Best Practices

### 8.1 Cache Key Design

1. **Human-Readable**: Include meaningful names trong cache keys
2. **Unique**: Ensure uniqueness với config hashes
3. **Hierarchical**: Organize theo model → dataset → config
4. **Consistent**: Use consistent naming conventions

### 8.2 Memory Management

1. **Memory Limits**: Set appropriate memory limits cho SHAP operations
2. **Cleanup**: Regular cleanup của old cache files
3. **Garbage Collection**: Force garbage collection sau operations
4. **Sample Size**: Limit sample sizes để prevent memory issues

### 8.3 Error Handling

1. **Graceful Degradation**: Fallback khi cache operations fail
2. **Comprehensive Logging**: Log all cache operations
3. **Validation**: Validate cache integrity before use
4. **Recovery**: Handle corrupted cache files

### 8.4 Performance Optimization

1. **Parallel Processing**: Disable parallel processing để prevent memory mapping errors
2. **File Formats**: Use efficient file formats (parquet, pickle)
3. **Compression**: Consider compression cho large cache files
4. **Indexing**: Maintain cache metadata index

## 9. Cache Troubleshooting

### 9.1 Common Issues

1. **Memory Issues**:
   - Reduce SHAP sample size
   - Enable memory cleanup
   - Use memory-safe operations

2. **Cache Corruption**:
   - Validate cache integrity
   - Implement cache recovery
   - Clear corrupted cache

3. **Performance Issues**:
   - Optimize cache key generation
   - Use efficient serialization
   - Implement cache pruning

4. **Disk Space**:
   - Implement cache size limits
   - Regular cleanup của old files
   - Monitor disk usage

### 9.2 Debugging Cache

```python
def debug_cache(self, model_key: str, dataset_id: str, config_hash: str):
    """Debug cache issues"""
    
    cache_path = self.get_cache_path(model_key, dataset_id, config_hash)
    
    print(f"🔍 Debugging cache: {cache_path}")
    print(f"   Exists: {cache_path.exists()}")
    
    if cache_path.exists():
        # List all files trong cache
        for file_path in cache_path.iterdir():
            print(f"   File: {file_path.name} ({file_path.stat().st_size} bytes)")
        
        # Check fingerprint
        fingerprint_file = cache_path / 'fingerprint.json'
        if fingerprint_file.exists():
            with open(fingerprint_file, 'r') as f:
                fingerprint = json.load(f)
            print(f"   Fingerprint: {fingerprint}")
    
    return cache_path
```

## 10. Cache Statistics và Monitoring

### 10.1 Cache Statistics

```python
def get_cache_stats(self) -> Dict[str, Any]:
    """Get cache statistics"""
    
    stats = {
        'total_models': 0,
        'total_cache_size_mb': 0,
        'cache_breakdown': {},
        'oldest_cache': None,
        'newest_cache': None
    }
    
    for model_dir in self.cache_root_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            model_stats = {
                'datasets': 0,
                'configs': 0,
                'size_mb': 0
            }
            
            for dataset_dir in model_dir.iterdir():
                if dataset_dir.is_dir():
                    model_stats['datasets'] += 1
                    
                    for config_dir in dataset_dir.iterdir():
                        if config_dir.is_dir():
                            model_stats['configs'] += 1
                            
                            # Calculate size
                            config_size = sum(f.stat().st_size for f in config_dir.rglob('*') if f.is_file())
                            model_stats['size_mb'] += config_size / (1024 * 1024)
            
            stats['cache_breakdown'][model_name] = model_stats
            stats['total_models'] += 1
            stats['total_cache_size_mb'] += model_stats['size_mb']
    
    return stats
```

## Kết Luận

Cache system của dự án này là một thành phần phức tạp và tinh vi với:

- **Hierarchical Architecture**: Organized theo model → dataset → config
- **Memory Safety**: Comprehensive memory management và leak protection
- **Error Handling**: Graceful degradation và recovery mechanisms
- **Performance Optimization**: Efficient serialization và file formats
- **Monitoring**: Statistics và debugging capabilities

Cache system đảm bảo:
- **Reproducibility**: Consistent results across runs
- **Performance**: Avoid unnecessary retraining
- **Resource Efficiency**: Optimal use của computational resources
- **Scalability**: Handle large datasets và complex models

Đây là foundation cho một ML platform enterprise-grade với intelligent caching strategies.
