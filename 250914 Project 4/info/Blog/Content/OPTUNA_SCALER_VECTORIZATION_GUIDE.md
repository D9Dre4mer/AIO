# Hướng Dẫn Áp Dụng Optuna, Scaler và Vectorization trong Dự Án

## Tổng Quan

Dự án này tích hợp ba công nghệ quan trọng để tối ưu hóa hiệu suất machine learning:
- **Optuna**: Hyperparameter optimization với Bayesian optimization
- **Scalers**: Normalization và standardization cho dữ liệu số
- **Vectorization**: Chuyển đổi text thành numerical features

## 1. Optuna Hyperparameter Optimization

### 1.1 Cấu Hình Optuna

```python
# config.py
OPTUNA_ENABLE = True
OPTUNA_TRIALS = 100
OPTUNA_TIMEOUT = None  # seconds, None for no timeout
OPTUNA_DIRECTION = "maximize"
```

### 1.2 Implementation trong OptunaOptimizer

```python
class OptunaOptimizer:
    def __init__(self, config: Dict[str, Any] = None):
        self.default_config = {
            'trials': 100,
            'timeout': None,
            'direction': 'maximize',
            'study_name': 'ml_optimization',
            'storage': None,
            'sampler': 'TPE',
            'pruner': 'MedianPruner'
        }
```

### 1.3 Tạo Study và Sampler

```python
def create_study(self, study_name: str = None) -> optuna.Study:
    """Create Optuna study with TPE sampler and MedianPruner"""
    
    # TPE Sampler với cấu hình tối ưu
    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=20,  # Random trials trước khi TPE
        n_ei_candidates=24     # Số candidates cho Expected Improvement
    )
    
    # MedianPruner để dừng sớm các trial kém
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,  # Không prune trong 10 trials đầu
        n_warmup_steps=5,     # Warmup steps
        interval_steps=1      # Check pruning mỗi step
    )
    
    study = optuna.create_study(
        direction=self.default_config['direction'],
        sampler=sampler,
        pruner=pruner,
        study_name=study_name or self.default_config['study_name']
    )
```

### 1.4 Search Space cho Từng Model

```python
def _get_default_search_space(self, model_name: str) -> Dict[str, Any]:
    """Định nghĩa search space cho từng model"""
    
    search_spaces = {
        'random_forest': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'max_depth': {'type': 'int', 'low': 3, 'high': 20},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10}
        },
        'xgboost': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'eta': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
            'reg_lambda': {'type': 'float', 'low': 0.001, 'high': 10.0, 'log': True},
            'reg_alpha': {'type': 'float', 'low': 0.001, 'high': 10.0, 'log': True}
        },
        'lightgbm': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'num_leaves': {'type': 'int', 'low': 10, 'high': 100},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'feature_fraction': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'bagging_fraction': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'min_child_samples': {'type': 'int', 'low': 5, 'high': 50},
            'lambda_l1': {'type': 'float', 'low': 0.001, 'high': 10.0, 'log': True},
            'lambda_l2': {'type': 'float', 'low': 0.001, 'high': 10.0, 'log': True}
        }
    }
```

### 1.5 Objective Function

```python
def objective(trial):
    """Objective function cho Optuna optimization"""
    try:
        # Suggest parameters từ search space
        params = {}
        search_space = self._get_default_search_space(model_name)
        
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config['low'], 
                    param_config['high']
                )
            elif param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, 
                    param_config['low'], 
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, 
                    param_config['choices']
                )
        
        # Tạo và train model
        model = model_class(**params)
        model.fit(X_train, y_train)
        
        # Evaluate trên validation set
        y_pred = model.predict(X_val)
        accuracy = np.mean(y_pred == y_val)
        
        return accuracy
        
    except Exception as e:
        logger.warning(f"Trial failed: {e}")
        return 0.0  # Return worst possible score
```

### 1.6 Sử Dụng trong Training Pipeline

```python
# app.py - train_models_with_scaling function
if optuna_enabled:
    # Sử dụng Optuna optimization
    optimizer = OptunaOptimizer(optuna_config)
    optimization_result = optimizer.optimize_model(
        model_name=mapped_name,
        model_class=model.__class__,
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,  # Sử dụng validation set cho Optuna
        y_val=y_val
    )
    
    best_params = optimization_result['best_params']
    best_score = optimization_result['best_score']
    
    # Train final model với best params
    final_model = model_factory.create_model(mapped_name)
    final_model.set_params(**best_params)
    final_model.fit(X_train_scaled, y_train)
```

## 2. Scaler và Normalization

### 2.1 Các Loại Scaler Được Hỗ Trợ

```python
# Các scaler có sẵn trong dự án
SCALERS = {
    'StandardScaler': StandardScaler(),      # Z-score normalization
    'MinMaxScaler': MinMaxScaler(),          # Min-Max scaling [0,1]
    'RobustScaler': RobustScaler(),          # Robust scaling với median
    'None': None                             # Không scaling
}
```

### 2.2 Implementation trong DataLoader

```python
def preprocess_multi_input_data(self, df, input_columns: List[str], 
                              label_column: str, 
                              preprocessing_config: Dict = None) -> Dict:
    """Preprocess data với multiple scalers"""
    
    # Separate columns by type
    numeric_cols = [col for col in input_columns if type_mapping.get(col) == 'numeric']
    categorical_cols = [col for col in input_columns if type_mapping.get(col) == 'categorical']
    text_cols = [col for col in input_columns if type_mapping.get(col) == 'text']
    
    # Initialize scaler cho numeric columns
    scaler = None
    if numeric_cols:
        scaler_type = preprocessing_config.get('numeric_scaler', 'standard')
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
    
    # Apply scaling
    if numeric_cols and scaler is not None:
        processed_data['numeric_scaled'] = scaler.fit_transform(df[numeric_cols])
        processed_data['scaler'] = scaler
```

### 2.3 Scaling trong Training Pipeline

```python
# app.py - train_numeric_data_directly function
for scaler_idx, scaler_name in enumerate(numeric_scalers):
    # Apply scaling
    if scaler_name == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_name == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_name == 'RobustScaler':
        scaler = RobustScaler()
    elif scaler_name == 'None':
        scaler = None
    
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val
        X_test_scaled = X_test
```

### 2.4 Cấu Hình Scaling

```python
# config.py
DATA_PROCESSING_NUMERIC_SCALER = "standard"  # "standard" | "minmax" | "robust"
```

## 3. Text Vectorization Methods

### 3.1 TextVectorizer Class

```python
class TextVectorizer:
    """Class cho handling different text vectorization methods"""
    
    def __init__(self):
        # BoW vectorizer với memory optimization
        self.bow_vectorizer = CountVectorizer(
            max_features=MAX_VOCABULARY_SIZE,  # 30,000 features max
            min_df=2,           # Ignore words appearing in < 2 documents
            max_df=0.95,        # Ignore words appearing in > 95% documents
            stop_words='english'
        )
        
        # TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=MAX_VOCABULARY_SIZE,
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        # Embedding vectorizer
        self.embedding_vectorizer = EmbeddingVectorizer()
        
        # SVD models cho dimensionality reduction
        self.bow_svd_model = None
        self.tfidf_svd_model = None
```

### 3.2 Bag of Words (BoW) Implementation

```python
def fit_transform_bow(self, texts: List[str]):
    """Fit and transform texts using Bag of Words (returns sparse matrix)"""
    vectors = self.bow_vectorizer.fit_transform(texts)
    print(f"📊 BoW Features: {vectors.shape[1]:,} | Sparsity: {1 - vectors.nnz / (vectors.shape[0] * vectors.shape[1]):.3f}")
    return vectors  # Keep sparse for memory efficiency

def fit_transform_bow_svd(self, texts: List[str]):
    """BoW với SVD dimensionality reduction"""
    vectors = self.bow_vectorizer.fit_transform(texts)
    
    # Apply SVD nếu cần - dynamic reduction cho large datasets
    n_samples = vectors.shape[0]
    if vectors.shape[1] > BOW_TFIDF_SVD_THRESHOLD or n_samples > 100000:
        # Aggressive reduction cho large datasets
        if n_samples > 200000:
            svd_components = min(200, BOW_TFIDF_SVD_COMPONENTS)
            print(f"🔧 Large dataset detected ({n_samples:,} samples), using aggressive SVD reduction")
        else:
            svd_components = BOW_TFIDF_SVD_COMPONENTS
        
        print(f"🔧 Applying SVD to BoW: {vectors.shape[1]:,} → {svd_components} dimensions")
        n_components = min(svd_components, vectors.shape[1] - 1, vectors.shape[0] - 1)
        self.bow_svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        vectors = self.bow_svd_model.fit_transform(vectors)
        explained_variance = self.bow_svd_model.explained_variance_ratio_.sum()
        print(f"✅ BoW SVD completed: {n_components} dimensions | Variance preserved: {explained_variance:.1%}")
    
    return vectors
```

### 3.3 TF-IDF Implementation

```python
def fit_transform_tfidf(self, texts: List[str]):
    """Fit and transform texts using TF-IDF (returns sparse matrix)"""
    vectors = self.tfidf_vectorizer.fit_transform(texts)
    print(f"📊 TF-IDF Features: {vectors.shape[1]:,} | Sparsity: {1 - vectors.nnz / (vectors.shape[0] * vectors.shape[1]):.3f}")
    return vectors

def fit_transform_tfidf_svd(self, texts: List[str]):
    """TF-IDF với SVD dimensionality reduction"""
    vectors = self.tfidf_vectorizer.fit_transform(texts)
    
    # Apply SVD với same logic như BoW
    n_samples = vectors.shape[0]
    if vectors.shape[1] > BOW_TFIDF_SVD_THRESHOLD or n_samples > 100000:
        if n_samples > 200000:
            svd_components = min(200, BOW_TFIDF_SVD_COMPONENTS)
        else:
            svd_components = BOW_TFIDF_SVD_COMPONENTS
        
        print(f"🔧 Applying SVD to TF-IDF: {vectors.shape[1]:,} → {svd_components} dimensions")
        n_components = min(svd_components, vectors.shape[1] - 1, vectors.shape[0] - 1)
        self.tfidf_svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        vectors = self.tfidf_svd_model.fit_transform(vectors)
        explained_variance = self.tfidf_svd_model.explained_variance_ratio_.sum()
        print(f"✅ TF-IDF SVD completed: {n_components} dimensions | Variance preserved: {explained_variance:.1%}")
    
    return vectors
```

### 3.4 Word Embeddings Implementation

```python
class EmbeddingVectorizer:
    """Class cho generating word embeddings using pre-trained models"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME,
                 normalize: bool = EMBEDDING_NORMALIZE,
                 device: str = EMBEDDING_DEVICE):
        # Auto-detect device
        if device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize model với GPU support
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_name = model_name
        self.normalize = normalize

    def transform_with_progress(self, texts: List[str], 
                              mode: Literal['query', 'passage'] = 'query', 
                              batch_size: int = 100, 
                              stop_callback=None) -> List[List[float]]:
        """Transform texts với progress tracking và ETA"""
        
        total_texts = len(texts)
        all_embeddings = []
        
        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Format inputs based on mode
            formatted_texts = self._format_inputs(batch_texts, mode)
            
            # Encode batch
            batch_embeddings = self.model.encode(
                formatted_texts,
                normalize_embeddings=self.normalize,
                show_progress_bar=False
            )
            
            all_embeddings.extend(batch_embeddings.tolist())
            
            # Progress tracking
            progress = (i + batch_size) / total_texts
            elapsed_time = time.time() - start_time
            eta = elapsed_time / progress * (1 - progress)
            
            if stop_callback:
                stop_callback(f"Generating embeddings: {progress:.1%} | ETA: {self._format_time(eta)}")
        
        return all_embeddings
```

### 3.5 Cấu Hình Vectorization

```python
# config.py
MAX_VOCABULARY_SIZE = 30000   # Maximum vocabulary cho BoW/TF-IDF
BOW_TFIDF_SVD_COMPONENTS = 400  # Reduce to 400 dimensions
BOW_TFIDF_SVD_THRESHOLD = 200   # Apply SVD nếu features > 200

# Embedding model configuration
EMBEDDING_MODEL_NAME = 'sentence-transformers/allenai-specter'  # 768d
EMBEDDING_NORMALIZE = True
EMBEDDING_DEVICE = 'auto'
```

## 4. Integration trong Training Pipeline

### 4.1 Comprehensive Evaluation

```python
def create_all_embeddings(self, X_train: List[str], X_val: List[str], X_test: List[str], 
                         selected_embeddings: List[str] = None, stop_callback=None) -> Dict[str, Dict[str, Any]]:
    """Tạo tất cả embeddings với progress tracking"""
    
    embeddings = {}
    embedding_methods = selected_embeddings or ['bow', 'tfidf', 'embeddings']
    
    # 1. Bag of Words (BoW)
    if 'bow' in embedding_methods:
        print("📦 Processing Bag of Words...")
        start_time = time.time()
        X_train_bow = self.text_vectorizer.fit_transform_bow(X_train)
        X_val_bow = self.text_vectorizer.transform_bow(X_val) if len(X_val) > 0 else None
        X_test_bow = self.text_vectorizer.transform_bow(X_test)
        bow_time = time.time() - start_time
        
        embeddings['bow'] = {
            'X_train': X_train_bow,
            'X_val': X_val_bow,
            'X_test': X_test_bow,
            'processing_time': bow_time,
            'sparse': hasattr(X_train_bow, 'nnz'),
            'shape': X_train_bow.shape
        }
    
    # 2. TF-IDF
    if 'tfidf' in embedding_methods:
        print("📊 Processing TF-IDF...")
        start_time = time.time()
        X_train_tfidf = self.text_vectorizer.fit_transform_tfidf(X_train)
        X_val_tfidf = self.text_vectorizer.transform_tfidf(X_val) if len(X_val) > 0 else None
        X_test_tfidf = self.text_vectorizer.transform_tfidf(X_test)
        tfidf_time = time.time() - start_time
        
        embeddings['tfidf'] = {
            'X_train': X_train_tfidf,
            'X_val': X_val_tfidf,
            'X_test': X_test_tfidf,
            'processing_time': tfidf_time,
            'sparse': hasattr(X_train_tfidf, 'nnz'),
            'shape': X_train_tfidf.shape
        }
    
    # 3. Word Embeddings
    if 'embeddings' in embedding_methods:
        print("🧠 Processing Word Embeddings...")
        start_time = time.time()
        X_train_embeddings = self.text_vectorizer.fit_transform_embeddings(X_train, stop_callback)
        X_val_embeddings = self.text_vectorizer.transform_embeddings(X_val, stop_callback) if len(X_val) > 0 else None
        X_test_embeddings = self.text_vectorizer.transform_embeddings(X_test, stop_callback)
        embeddings_time = time.time() - start_time
        
        embeddings['embeddings'] = {
            'X_train': X_train_embeddings,
            'X_val': X_val_embeddings,
            'X_test': X_test_embeddings,
            'processing_time': embeddings_time,
            'sparse': False,
            'shape': X_train_embeddings.shape
        }
    
    return embeddings
```

## 5. Best Practices và Tips

### 5.1 Optuna Best Practices

1. **Startup Trials**: Sử dụng 20 random trials trước khi TPE
2. **Pruning**: Sử dụng MedianPruner để dừng sớm các trial kém
3. **Search Space**: Định nghĩa search space phù hợp với từng model
4. **Validation Set**: Sử dụng validation set riêng cho Optuna, không dùng test set

### 5.2 Scaler Best Practices

1. **Fit trên Training**: Chỉ fit scaler trên training data
2. **Transform tất cả**: Transform cả validation và test với scaler đã fit
3. **Multiple Scalers**: Test nhiều scalers để tìm best performance
4. **Memory Efficiency**: Sử dụng sparse matrices cho BoW/TF-IDF

### 5.3 Vectorization Best Practices

1. **Vocabulary Size**: Giới hạn vocabulary size để tránh memory issues
2. **SVD Reduction**: Sử dụng SVD cho large datasets
3. **Batch Processing**: Process embeddings theo batches với progress tracking
4. **Device Selection**: Auto-detect GPU/CPU cho embeddings

### 5.4 Performance Optimization

```python
# Memory optimization thresholds
KMEANS_SVD_THRESHOLD = 20000  # Use SVD nếu features > 20K
KMEANS_SVD_COMPONENTS = 2000  # Reduce to 2K dimensions
MAX_VOCABULARY_SIZE = 30000   # Maximum vocabulary cho BoW/TF-IDF

# GPU Optimization Settings
ENABLE_GPU_OPTIMIZATION = False  # Use sparse matrices (memory efficient)
FORCE_DENSE_CONVERSION = False   # Force sparse->dense conversion cho GPU

# RAPIDS cuML Settings
ENABLE_RAPIDS_CUML = True        # Enable RAPIDS cuML cho GPU acceleration
RAPIDS_FALLBACK_TO_CPU = True    # Fallback to CPU nếu GPU not available
```

## 6. Troubleshooting

### 6.1 Optuna Issues

- **Memory Issues**: Giảm số trials hoặc timeout
- **Slow Optimization**: Sử dụng pruning và smaller search space
- **Poor Results**: Kiểm tra search space và validation set

### 6.2 Scaler Issues

- **Data Leakage**: Đảm bảo chỉ fit trên training data
- **Memory Issues**: Sử dụng sparse matrices cho large datasets
- **Performance**: Test multiple scalers để tìm best

### 6.3 Vectorization Issues

- **Memory Issues**: Giảm vocabulary size và sử dụng SVD
- **Slow Processing**: Sử dụng batch processing và GPU
- **Quality Issues**: Kiểm tra text preprocessing và model selection

## Kết Luận

Việc tích hợp Optuna, Scaler và Vectorization tạo ra một pipeline ML hoàn chỉnh với:

- **Automatic hyperparameter optimization** với Bayesian methods
- **Flexible data preprocessing** với multiple scaling options
- **Advanced text processing** với BoW, TF-IDF và embeddings
- **Memory optimization** cho large datasets
- **GPU acceleration** cho performance

Đây là foundation cho một ML platform enterprise-grade với khả năng scale và optimize tự động.
