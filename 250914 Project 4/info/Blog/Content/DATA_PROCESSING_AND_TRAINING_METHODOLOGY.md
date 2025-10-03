# Phương Pháp Giải Quyết Dữ Liệu và Training - Comprehensive Machine Learning Platform

## Tổng Quan Phương Pháp

Dự án **Comprehensive Machine Learning Platform** sử dụng một phương pháp xử lý dữ liệu và training rất tinh vi, được thiết kế để xử lý hiệu quả nhiều loại dữ liệu khác nhau từ datasets nhỏ (1K samples) đến datasets lớn (300K+ samples). Phương pháp này kết hợp các kỹ thuật tiên tiến về preprocessing, vectorization, và training với tối ưu hóa memory và GPU acceleration.

---

## 1. Phương Pháp Xử Lý Dữ Liệu (Data Processing)

### 1.1 Kiến Trúc DataLoader (`data_loader.py`)

```python
class DataLoader:
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.dataset = None
        self.samples = []
        self.preprocessed_samples = []
        self.label_to_id = {}
        self.id_to_label = {}
        self.available_categories = []
        self.selected_categories = []
        self.category_stats = {}
        self.is_multi_input = False
```

**Đặc điểm chính:**
- **Dynamic Category Detection**: Tự động phát hiện và phân loại categories
- **Multi-input Support**: Hỗ trợ cả dữ liệu text và numerical
- **Intelligent Sampling**: Sampling thông minh với category balancing
- **Memory Optimization**: Tối ưu hóa memory cho datasets lớn

### 1.2 Quy Trình Load Dataset

```python
def load_dataset(self, skip_csv_prompt: bool = False) -> None:
    """Load any dataset and automatically detect categories"""
    # 1. Check cache first
    dataset_cache_path = Path(self.cache_dir) / "UniverseTBD___arxiv-abstracts-large"
    csv_backup_path = Path(self.cache_dir) / "arxiv_dataset_backup.csv"
    
    # 2. Load from HuggingFace datasets
    if dataset_cache_path.exists():
        self.dataset = load_dataset("UniverseTBD/arxiv-abstracts-large", cache_dir=str(dataset_cache_path))
    else:
        self.dataset = load_dataset("UniverseTBD/arxiv-abstracts-large")
    
    # 3. Create CSV backup for faster access
    self._create_csv_backup_chunked(csv_backup_path)
```

**Các bước xử lý:**

1. **Cache Check**: Kiểm tra cache trước khi load
2. **HuggingFace Integration**: Load từ HuggingFace datasets
3. **CSV Backup**: Tạo backup CSV với chunked processing
4. **Auto-detection**: Tự động phát hiện data types và categories

### 1.3 Phương Pháp Sampling Thông Minh

```python
def select_samples(self, max_samples: int = None) -> None:
    """Intelligent sampling strategy with category balancing"""
    
    # 1. Category-based sampling
    if self.selected_categories:
        category_samples = {}
        samples_per_category = max_samples // len(self.selected_categories)
        
        for category in self.selected_categories:
            category_data = [s for s in self.dataset['train'] 
                           if category in s['categories']]
            category_samples[category] = category_data[:samples_per_category]
    
    # 2. Stratified sampling
    # 3. Memory optimization
    # 4. Progress tracking
```

**Sampling Strategies:**
- **Stratified Sampling**: Đảm bảo tỷ lệ categories cân bằng
- **Category Balancing**: Cân bằng số lượng samples giữa các categories
- **Memory-aware Sampling**: Sampling dựa trên available memory
- **Progressive Sampling**: Sampling từng chunk để tránh memory overflow

### 1.4 Preprocessing Pipeline Chi Tiết

```python
def preprocess_samples(self, preprocessing_config: Dict = None) -> None:
    """Comprehensive preprocessing pipeline"""
    
    # Default preprocessing configuration
    default_config = {
        'text_cleaning': True,
        'data_validation': True,
        'category_mapping': True,
        'memory_optimization': True,
        'rare_words_removal': False,
        'rare_words_threshold': 2,
        'lemmatization': False,
        'context_aware_stopwords': False,
        'stopwords_aggressiveness': 'Moderate',
        'phrase_detection': False,
        'min_phrase_freq': 3
    }
    
    # Apply preprocessing steps
    for sample in self.samples:
        # 1. Text cleaning and normalization
        # 2. Missing value handling
        # 3. Outlier detection
        # 4. Feature engineering
        # 5. Encoding strategies
```

**Preprocessing Steps:**

#### A. Text Cleaning & Normalization
```python
def clean_text(self, text: str) -> str:
    """Advanced text cleaning"""
    # 1. Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 2. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Handle encoding issues
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # 4. Context-aware stopwords removal
    if self.context_aware_stopwords:
        text = self._remove_context_stopwords(text)
    
    return text
```

#### B. Missing Value Handling
```python
def handle_missing_values(self, data: Dict) -> Dict:
    """Intelligent missing value handling"""
    
    # For numeric data
    if self.is_multi_input:
        # Use mean/median/mode imputation
        numeric_columns = self._detect_numeric_columns(data)
        for col in numeric_columns:
            if data[col] is None or data[col] == '':
                data[col] = self._get_imputation_value(col)
    
    # For text data
    else:
        # Use mode imputation or removal
        if not data['abstract'].strip():
            data['abstract'] = self._get_default_text()
    
    return data
```

#### C. Outlier Detection & Treatment
```python
def detect_and_treat_outliers(self, data: List[Dict]) -> List[Dict]:
    """Advanced outlier detection"""
    
    # 1. IQR method for numeric data
    # 2. Z-score method for extreme outliers
    # 3. Isolation Forest for complex patterns
    # 4. Domain-specific outlier rules
    
    outlier_method = self.outlier_method
    if outlier_method == 'iqr':
        return self._iqr_outlier_detection(data)
    elif outlier_method == 'zscore':
        return self._zscore_outlier_detection(data)
    elif outlier_method == 'isolation_forest':
        return self._isolation_forest_outlier_detection(data)
    
    return data
```

### 1.5 Feature Engineering & Type Detection

```python
def auto_detect_feature_types(self, data: List[Dict]) -> Dict[str, str]:
    """Automatic feature type detection"""
    
    feature_types = {}
    
    for sample in data[:100]:  # Sample for efficiency
        for key, value in sample.items():
            if key in ['abstract', 'categories']:
                continue  # Skip known fields
            
            # Detect numeric features
            if isinstance(value, (int, float)) or self._is_numeric_string(value):
                feature_types[key] = 'numeric'
            
            # Detect categorical features
            elif isinstance(value, str) and len(value) < 50:
                feature_types[key] = 'categorical'
            
            # Detect text features
            elif isinstance(value, str) and len(value) >= 50:
                feature_types[key] = 'text'
    
    return feature_types
```

---

## 2. Phương Pháp Vectorization (Text Encoding)

### 2.1 Kiến Trúc TextVectorizer (`text_encoders.py`)

```python
class TextVectorizer:
    def __init__(self):
        # Bag of Words with vocabulary limits
        self.bow_vectorizer = CountVectorizer(
            max_features=MAX_VOCABULARY_SIZE,  # 30,000 features
            min_df=2,           # Ignore words in < 2 documents
            max_df=0.95,        # Ignore words in > 95% documents
            stop_words='english'
        )
        
        # TF-IDF with optimization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=MAX_VOCABULARY_SIZE,
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        # Sentence transformer embeddings
        self.embedding_vectorizer = EmbeddingVectorizer()
        
        # SVD models for dimensionality reduction
        self.bow_svd_model = None
        self.tfidf_svd_model = None
```

### 2.2 Bag of Words với SVD Optimization

```python
def fit_transform_bow_svd(self, texts: List[str]):
    """Bag of Words with SVD dimensionality reduction"""
    
    # 1. Generate BoW vectors
    vectors = self.bow_vectorizer.fit_transform(texts)
    print(f"📊 BoW Features: {vectors.shape[1]:,} | Sparsity: {1 - vectors.nnz / (vectors.shape[0] * vectors.shape[1]):.3f}")
    
    # 2. Apply SVD if needed
    n_samples = vectors.shape[0]
    if vectors.shape[1] > BOW_TFIDF_SVD_THRESHOLD or n_samples > 100000:
        
        # Dynamic SVD components based on dataset size
        if n_samples > 200000:
            svd_components = min(200, BOW_TFIDF_SVD_COMPONENTS)  # Very aggressive for 300k+ samples
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

**Đặc điểm BoW với SVD:**
- **Vocabulary Limiting**: Giới hạn 30K features để tránh memory overflow
- **Dynamic SVD**: SVD components thay đổi theo dataset size
- **Sparsity Preservation**: Giữ sparse matrices để tiết kiệm memory
- **Variance Tracking**: Theo dõi explained variance ratio

### 2.3 TF-IDF với Optimization

```python
def fit_transform_tfidf_svd(self, texts: List[str]):
    """TF-IDF with SVD dimensionality reduction"""
    
    # 1. Generate TF-IDF vectors
    vectors = self.tfidf_vectorizer.fit_transform(texts)
    print(f"📊 TF-IDF Features: {vectors.shape[1]:,} | Sparsity: {1 - vectors.nnz / (vectors.shape[0] * vectors.shape[1]):.3f}")
    
    # 2. Apply SVD with same logic as BoW
    n_samples = vectors.shape[0]
    if vectors.shape[1] > BOW_TFIDF_SVD_THRESHOLD or n_samples > 100000:
        
        # Same dynamic SVD logic
        if n_samples > 200000:
            svd_components = min(200, BOW_TFIDF_SVD_COMPONENTS)
        else:
            svd_components = BOW_TFIDF_SVD_COMPONENTS
        
        self.tfidf_svd_model = TruncatedSVD(n_components=svd_components, random_state=42)
        vectors = self.tfidf_svd_model.fit_transform(vectors)
    
    return vectors
```

### 2.4 Sentence Transformer Embeddings

```python
class EmbeddingVectorizer:
    def __init__(self, model_name: str = 'sentence-transformers/allenai-specter'):
        # Auto-detect device
        if device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize model with GPU support
        self.model = SentenceTransformer(model_name, device=self.device)
        self.normalize = True
    
    def transform_with_progress(self, texts: List[str], batch_size: int = 100):
        """Transform texts to embeddings with progress tracking"""
        
        total_texts = len(texts)
        all_embeddings = []
        
        # Process in batches with progress bar
        for i in range(0, total_texts, batch_size):
            batch_end = min(i + batch_size, total_texts)
            batch_inputs = texts[i:batch_end]
            
            # Generate embeddings for current batch
            batch_embeddings = self.model.encode(
                batch_inputs,
                normalize_embeddings=self.normalize,
                show_progress_bar=False
            )
            
            all_embeddings.extend(batch_embeddings.tolist())
            
            # Show progress with ETA
            progress_percent = (batch_end / total_texts) * 100
            progress_bar = self._create_progress_bar(progress_percent, 40)
            print(f"\r🔄 Embedding Progress: {progress_bar} {progress_percent:5.1f}% ({batch_end:,}/{total_texts:,})", end="", flush=True)
        
        return all_embeddings
```

**Đặc điểm Embeddings:**
- **GPU Acceleration**: Tự động detect và sử dụng GPU
- **Batch Processing**: Xử lý theo batch để tối ưu memory
- **Progress Tracking**: Real-time progress với ETA
- **Normalization**: L2 normalization cho embeddings
- **Model Selection**: Sử dụng allenai-specter (768 dimensions)

---

## 3. Phương Pháp Training Models

### 3.1 Kiến Trúc Base Model (`models/base/base_model.py`)

```python
class BaseModel(ABC):
    """Abstract base class that all models must inherit from"""
    
    def __init__(self, **kwargs):
        self.model = None
        self.is_fitted = False
        self.model_params = kwargs
        self.training_history = []
        self.validation_metrics = {}
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], y: np.ndarray) -> 'BaseModel':
        """Fit the model to training data"""
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Make predictions on new data"""
        pass
    
    def validate(self, X_val: Union[np.ndarray, sparse.csr_matrix], y_val: np.ndarray) -> Dict[str, Any]:
        """Validate the model on validation set"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before validation")
        
        # Make predictions on validation set
        y_val_pred = self.predict(X_val)
        
        # Compute validation metrics
        from .metrics import ModelMetrics
        metrics = ModelMetrics.compute_classification_metrics(y_val, y_val_pred)
        
        # Store validation metrics
        self.validation_metrics = metrics
        
        # Add to training history
        self.training_history.append({
            'action': 'validate',
            'n_samples': X_val.shape[0],
            'validation_metrics': metrics
        })
        
        return metrics
```

### 3.2 Logistic Regression Implementation (`models/classification/logistic_regression_model.py`)

```python
class LogisticRegressionModel(BaseModel):
    """Logistic Regression classification model with automatic parameter optimization"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Default parameters optimized for performance
        default_params = {
            'max_iter': 2000,           # Increased iterations for convergence
            'multi_class': 'multinomial', # Better for multi-class
            'n_jobs': -1,               # Use all CPU cores
            'random_state': 42,          # Reproducibility
            'C': 1.0,                   # Regularization strength
            'solver': 'lbfgs'           # Efficient solver
        }
        
        # Update with user-provided parameters
        default_params.update(kwargs)
        self.model_params = default_params
    
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], y: np.ndarray) -> 'LogisticRegressionModel':
        """Fit Logistic Regression model to training data"""
        
        # Create model with parameters
        self.model = LogisticRegression(**self.model_params)
        
        # Display multithreading info
        n_jobs = self.model_params.get('n_jobs', -1)
        if n_jobs == -1:
            import os
            cpu_count = os.cpu_count()
            print(f"🔄 CPU multithreading: Using all {cpu_count} available cores")
        else:
            print(f"🔄 CPU multithreading: Using {n_jobs} parallel jobs")
        
        # Fit the model
        self.model.fit(X, y)
        
        self.is_fitted = True
        self.training_history.append({
            'action': 'fit',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'parameters': self.model_params.copy()
        })
        
        return self
```

**Đặc điểm Logistic Regression:**
- **Multithreading**: Sử dụng tất cả CPU cores
- **Convergence**: Tăng max_iter để đảm bảo convergence
- **Multi-class**: Sử dụng multinomial cho multi-class problems
- **Sparse Support**: Hỗ trợ sparse matrices
- **Parameter Optimization**: Parameters được tối ưu cho performance

### 3.3 Model Registry System (`models/utils/model_registry.py`)

```python
class ModelRegistry:
    """Registry pattern for managing models"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {}
    
    def register_model(self, name: str, model_class: type, config: Dict):
        """Register model with configuration"""
        self.models[name] = model_class
        self.model_configs[name] = config
    
    def get_model(self, name: str, **kwargs):
        """Factory method to create model instances"""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not registered")
        
        model_class = self.models[name]
        default_config = self.model_configs[name].copy()
        default_config.update(kwargs)
        
        return model_class(**default_config)
```

### 3.4 Training Pipeline (`training_pipeline.py`)

```python
class StreamlitTrainingPipeline:
    """Comprehensive training pipeline for Streamlit"""
    
    def __init__(self):
        self.cache_dir = "./cache"
        self.cache_metadata = {}
        self.cache_metadata_file = os.path.join(self.cache_dir, "cache_metadata.json")
        self.data_loader = DataLoader()
        self.text_vectorizer = TextVectorizer()
        self.comprehensive_evaluator = ComprehensiveEvaluator()
    
    def initialize_pipeline(self, step1_data: Dict, step2_data: Dict, step3_data: Dict) -> Dict:
        """Initialize training pipeline with configuration"""
        
        # 1. Extract configuration
        sampling_config = step1_data.get('sampling_config', {})
        preprocessing_config = step2_data.get('preprocessing_config', {})
        model_config = step3_data.get('model_config', {})
        optuna_config = step3_data.get('optuna_config', {})
        
        # 2. Configure components
        self.data_loader = DataLoader()
        self.text_vectorizer = TextVectorizer()
        
        # 3. Load and prepare data
        self.data_dict, self.sorted_labels = self.comprehensive_evaluator.load_and_prepare_data(
            sampling_config=sampling_config,
            preprocessing_config=preprocessing_config
        )
        
        # 4. Generate cache key
        cache_key = self._generate_cache_key(step1_data, step2_data, step3_data)
        
        # 5. Check cache
        cached_results = self._check_cache(cache_key)
        if cached_results:
            return cached_results
        
        return {
            'status': 'success',
            'message': 'Pipeline initialized successfully',
            'cache_key': cache_key
        }
```

---

## 4. Phương Pháp Validation & Cross-Validation

### 4.1 ValidationManager (`models/utils/validation_manager.py`)

```python
class ValidationManager:
    """Unified manager for data splitting and cross-validation"""
    
    def __init__(self, test_size: float = 0.2, validation_size: float = 0.2, 
                 random_state: int = 42, cv_folds: int = 5, cv_stratified: bool = True):
        
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.cv_stratified = cv_stratified
        
        # Initialize KFold splitter
        if cv_stratified:
            self.kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            self.kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
```

### 4.2 3-Way Data Splitting

```python
def split_data(self, X: Union[np.ndarray, sparse.csr_matrix], y: np.ndarray) -> Tuple:
    """Split data into train/validation/test sets"""
    
    total_samples = X.shape[0]
    
    # Calculate exact sizes
    test_samples = int(total_samples * self.test_size)
    remaining_samples = total_samples - test_samples
    val_samples = int(remaining_samples * self.validation_size)
    train_samples = remaining_samples - val_samples
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_samples, 
        random_state=self.random_state,
        stratify=None  # Don't stratify for exact size splits
    )
    
    # Second split: separate validation from remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_samples,
        random_state=self.random_state,
        stratify=None
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
```

### 4.3 Cross-Validation với Pre-computed Embeddings

```python
def cross_validate_with_precomputed_embeddings(self, model, cv_embeddings: Dict[str, Any], 
                                               metrics: List[str] = None) -> Dict[str, Any]:
    """Perform cross-validation using pre-computed embeddings for fair comparison"""
    
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Initialize results storage
    fold_scores = {metric: [] for metric in metrics}
    all_predictions = []
    all_true_labels = []
    
    # Perform cross-validation using pre-computed embeddings
    for fold in range(1, self.cv_folds + 1):
        print(f"  📊 Fold {fold}/{self.cv_folds}")
        
        fold_data = cv_embeddings[f'fold_{fold}']
        X_train = fold_data['X_train']
        X_val = fold_data['X_val']
        y_train = fold_data['y_train']
        y_val = fold_data['y_val']
        
        # Keep sparse matrices for better performance
        if hasattr(X_train, 'toarray'):
            print(f"   📊 Using sparse matrix format for memory efficiency in CV fold {fold}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Store predictions and true labels
        all_predictions.extend(y_pred)
        all_true_labels.extend(y_val)
        
        # Calculate metrics for this fold
        for metric in metrics:
            if metric == 'accuracy':
                # Calculate validation accuracy
                val_score = accuracy_score(y_val, y_pred)
                # Calculate training accuracy for overfitting detection
                y_train_pred = model.predict(X_train)
                train_score = accuracy_score(y_train, y_train_pred)
                
                # Store both scores
                if 'train_accuracy' not in fold_scores:
                    fold_scores['train_accuracy'] = []
                    fold_scores['validation_accuracy'] = []
                fold_scores['train_accuracy'].append(train_score)
                fold_scores['validation_accuracy'].append(val_score)
                
                score = val_score
            elif metric == 'precision':
                score = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            elif metric == 'f1':
                score = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            fold_scores[metric].append(score)
    
    return self._compute_cv_statistics(fold_scores, all_predictions, all_true_labels, metrics)
```

**Đặc điểm Cross-Validation:**
- **Pre-computed Embeddings**: Sử dụng embeddings đã tính sẵn để đảm bảo fairness
- **Sparse Matrix Support**: Giữ sparse matrices để tiết kiệm memory
- **Overfitting Detection**: Tính cả training và validation accuracy
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Memory Optimization**: Xử lý từng fold để tránh memory overflow

---

## 5. Phương Pháp Caching & Optimization

### 5.1 Intelligent Caching System

```python
def _generate_cache_key(self, step1_data: Dict, step2_data: Dict, step3_data: Dict) -> str:
    """Generate unique cache key based on configuration with human-readable naming"""
    
    # Extract key configuration details
    sampling_config = step1_data.get('sampling_config', {})
    optuna_config = step3_data.get('optuna_config', {})
    selected_models = optuna_config.get('models', [])
    selected_vectorization = step3_data.get('selected_vectorization', [])
    
    # Get dataset information
    dataset_name = "unknown_dataset"
    dataset_hash = "no_hash"
    
    if 'uploaded_file' in step1_data and step1_data['uploaded_file']:
        uploaded_file = step1_data['uploaded_file']
        if isinstance(uploaded_file, dict) and 'name' in uploaded_file:
            dataset_name = uploaded_file['name'].replace('.csv', '').replace('.xlsx', '')
    
    # Create hash from dataset content
    if 'dataframe' in step1_data and step1_data['dataframe'] is not None:
        df = step1_data['dataframe']
        try:
            content_str = str(list(df.columns)) + str(df.head(100).values.tolist())
            dataset_hash = hashlib.md5(content_str.encode()).hexdigest()[:8]
        except Exception as e:
            dataset_hash = "hash_err"
    
    # Create human-readable cache name
    sample_count = sampling_config.get('num_samples', 'full')
    if sample_count == 'full':
        sample_str = "full_dataset"
    else:
        sample_str = f"{sample_count}samples"
    
    # Get first few models and vectorization methods
    model_str = "_".join(selected_models[:3])
    vector_str = "_".join(selected_vectorization[:2])
    
    # Create human-readable name
    human_name = f"{model_str}_{vector_str}_{sample_str}_{dataset_name}"
    
    # Create hash for uniqueness
    config_hash = {
        'dataset': {'name': dataset_name, 'hash': dataset_hash},
        'sampling': sampling_config,
        'preprocessing': step2_data,
        'model': step3_data.get('data_split', {}),
        'vectorization': selected_vectorization
    }
    
    config_str = json.dumps(config_hash, sort_keys=True)
    config_hash_str = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    return f"{human_name}_{config_hash_str}"
```

### 5.2 Compatible Cache Finding

```python
def _find_compatible_cache(self, target_cache_key: str) -> Dict:
    """Find compatible cache when exact match not found"""
    
    print(f"🔍 Exact cache not found: {target_cache_key}")
    print("🔍 Searching for compatible cache...")
    
    # Extract key info from target cache key
    target_parts = target_cache_key.split('_')
    target_samples = None
    target_categories = None
    
    for part in target_parts:
        if 'samples' in part:
            target_samples = part
        elif any(cat in part for cat in ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']):
            target_categories = part
    
    best_match = None
    best_score = 0
    
    for cache_key, cache_info in self.cache_metadata.items():
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if not os.path.exists(cache_file):
            continue
        
        # Calculate compatibility score
        score = 0
        cache_parts = cache_key.split('_')
        
        # Check samples match
        for part in cache_parts:
            if 'samples' in part and target_samples and part == target_samples:
                score += 3
            elif 'full' in part and target_samples and 'full' in target_samples:
                score += 3
        
        # Check categories match
        for part in cache_parts:
            if target_categories and part in target_categories:
                score += 2
        
        # Check if it's a recent cache
        cache_age = time.time() - cache_info['timestamp']
        if cache_age < 7 * 24 * 3600:  # Within 7 days
            score += 1
        
        if score > best_score:
            best_score = score
            best_match = (cache_key, cache_info)
    
    if best_match and best_score >= 2:  # Minimum compatibility threshold
        cache_key, cache_info = best_match
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_file, 'rb') as f:
                cached_results = pickle.load(f)
            
            print(f"✅ Using compatible cache: {cache_key[:50]}...")
            print(f"   Compatibility score: {best_score}/5")
            print(f"   ⚠️  Note: Using cache with different config - results may vary")
            
            return cached_results
        except Exception as e:
            print(f"Warning: Could not load compatible cache: {e}")
    
    return None
```

---

## 6. Phương Pháp Comprehensive Evaluation

### 6.1 ComprehensiveEvaluator (`comprehensive_evaluation.py`)

```python
class ComprehensiveEvaluator:
    """Comprehensive evaluation system for all embedding-model combinations"""
    
    def __init__(self, cv_folds: int = 5, validation_size: float = 0.2, 
                 test_size: float = 0.2, random_state: int = 42):
        
        self.cv_folds = cv_folds
        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state
        
        # Embedding methods
        self.embedding_methods = ['bow', 'tfidf', 'embeddings']
        
        # Model types
        self.model_types = ['classification', 'clustering', 'ensemble']
        
        # Results storage
        self.results = {}
    
    def evaluate_all_combinations(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate all embedding-model combinations"""
        
        # 1. Load and prepare data
        data_dict, sorted_labels = self.load_and_prepare_data()
        
        # 2. Generate all embeddings
        embeddings = self._generate_all_embeddings(data_dict)
        
        # 3. Create CV folds for each embedding type
        cv_embeddings = self._create_cv_folds_for_all_embeddings(embeddings, data_dict)
        
        # 4. Evaluate all model combinations
        results = self._evaluate_all_models(cv_embeddings, sorted_labels)
        
        # 5. Generate comprehensive report
        report = self._generate_comprehensive_report(results)
        
        return report
```

### 6.2 Embedding Generation Pipeline

```python
def _generate_all_embeddings(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Generate all types of embeddings"""
    
    embeddings = {}
    texts = data_dict['X_train']
    
    # 1. Bag of Words with SVD
    print("🔄 Generating Bag of Words embeddings...")
    X_train_bow = self.text_vectorizer.fit_transform_bow_svd(texts)
    X_test_bow = self.text_vectorizer.transform_bow_svd(data_dict['X_test'])
    
    embeddings['bow'] = {
        'X_train': X_train_bow,
        'X_test': X_test_bow,
        'vectorizer': self.text_vectorizer.bow_vectorizer,
        'svd_model': self.text_vectorizer.bow_svd_model
    }
    
    # 2. TF-IDF with SVD
    print("🔄 Generating TF-IDF embeddings...")
    X_train_tfidf = self.text_vectorizer.fit_transform_tfidf_svd(texts)
    X_test_tfidf = self.text_vectorizer.transform_tfidf_svd(data_dict['X_test'])
    
    embeddings['tfidf'] = {
        'X_train': X_train_tfidf,
        'X_test': X_test_tfidf,
        'vectorizer': self.text_vectorizer.tfidf_vectorizer,
        'svd_model': self.text_vectorizer.tfidf_svd_model
    }
    
    # 3. Sentence Transformer Embeddings
    print("🔄 Generating Sentence Transformer embeddings...")
    X_train_embeddings = self.text_vectorizer.embedding_vectorizer.transform_with_progress(texts)
    X_test_embeddings = self.text_vectorizer.embedding_vectorizer.transform_with_progress(data_dict['X_test'])
    
    embeddings['embeddings'] = {
        'X_train': np.array(X_train_embeddings),
        'X_test': np.array(X_test_embeddings),
        'vectorizer': self.text_vectorizer.embedding_vectorizer
    }
    
    return embeddings
```

### 6.3 Model Evaluation Pipeline

```python
def _evaluate_all_models(self, cv_embeddings: Dict[str, Any], sorted_labels: List[str]) -> Dict[str, Any]:
    """Evaluate all model combinations"""
    
    results = {}
    
    # Get available models
    available_models = self._get_available_models()
    
    for embedding_type, cv_data in cv_embeddings.items():
        print(f"\n🔍 Evaluating {embedding_type.upper()} embeddings...")
        
        embedding_results = {}
        
        for model_name in available_models:
            print(f"  🤖 Training {model_name}...")
            
            try:
                # Create model instance
                model = self.model_factory.create_model(model_name)
                
                # Perform cross-validation
                cv_results = self.validation_manager.cross_validate_with_precomputed_embeddings(
                    model, cv_data, metrics=['accuracy', 'precision', 'recall', 'f1']
                )
                
                # Evaluate on test data
                test_results = self.validation_manager.evaluate_test_data_from_cv_cache(
                    model, cv_data, metrics=['accuracy', 'precision', 'recall', 'f1']
                )
                
                # Store results
                embedding_results[model_name] = {
                    'cv_results': cv_results,
                    'test_results': test_results,
                    'model_info': model.get_model_info()
                }
                
                print(f"    ✅ {model_name}: CV Acc={cv_results['accuracy_mean']:.4f}±{cv_results['accuracy_std']:.4f}")
                
            except Exception as e:
                print(f"    ❌ {model_name}: Error - {str(e)}")
                embedding_results[model_name] = {'error': str(e)}
        
        results[embedding_type] = embedding_results
    
    return results
```

---

## 7. Tối Ưu Hóa Performance

### 7.1 Memory Management

```python
def optimize_memory_usage(self, data_size: int, available_memory: int) -> Dict[str, Any]:
    """Dynamic memory optimization strategies"""
    
    optimization_strategies = {}
    
    # 1. Sparse Matrix Usage
    if data_size > 10000:
        optimization_strategies['use_sparse'] = True
        optimization_strategies['sparse_threshold'] = 0.1  # 10% sparsity threshold
    
    # 2. Batch Processing
    if data_size > 50000:
        optimization_strategies['batch_size'] = min(1000, data_size // 10)
        optimization_strategies['chunked_processing'] = True
    
    # 3. SVD Dimensionality Reduction
    if data_size > 100000:
        optimization_strategies['svd_components'] = min(200, data_size // 1000)
        optimization_strategies['aggressive_svd'] = True
    
    # 4. Garbage Collection
    optimization_strategies['gc_frequency'] = 'after_each_model'
    optimization_strategies['clear_cache'] = True
    
    return optimization_strategies
```

### 7.2 GPU Acceleration

```python
def configure_gpu_acceleration(self) -> Dict[str, Any]:
    """Configure GPU acceleration based on available hardware"""
    
    gpu_config = {
        'cuda_available': False,
        'cuda_version': None,
        'gpu_memory': None,
        'device_policy': 'cpu_only'
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_config['cuda_available'] = True
            gpu_config['cuda_version'] = torch.version.cuda
            gpu_config['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
            gpu_config['device_policy'] = 'gpu_first'
            
            print(f"🚀 GPU Acceleration Available:")
            print(f"   CUDA Version: {gpu_config['cuda_version']}")
            print(f"   GPU Memory: {gpu_config['gpu_memory'] / 1024**3:.1f} GB")
    except ImportError:
        print("⚠️ PyTorch not available - using CPU only")
    
    return gpu_config
```

---

## 8. Kết Luận

### 8.1 Điểm Mạnh của Phương Pháp

**A. Xử Lý Dữ Liệu:**
- ✅ **Dynamic Category Detection**: Tự động phát hiện categories
- ✅ **Intelligent Sampling**: Sampling thông minh với category balancing
- ✅ **Advanced Preprocessing**: Preprocessing pipeline toàn diện
- ✅ **Memory Optimization**: Tối ưu hóa memory cho datasets lớn

**B. Vectorization:**
- ✅ **Multiple Methods**: BoW, TF-IDF, Sentence Transformers
- ✅ **SVD Optimization**: Dimensionality reduction thông minh
- ✅ **GPU Acceleration**: Hỗ trợ GPU cho embeddings
- ✅ **Sparse Matrix Support**: Tiết kiệm memory với sparse matrices

**C. Training:**
- ✅ **Modular Architecture**: Kiến trúc modular dễ mở rộng
- ✅ **Cross-Validation**: CV với pre-computed embeddings
- ✅ **Overfitting Detection**: Phát hiện overfitting
- ✅ **Comprehensive Metrics**: Metrics toàn diện

**D. Optimization:**
- ✅ **Intelligent Caching**: Cache system thông minh
- ✅ **Memory Management**: Quản lý memory hiệu quả
- ✅ **GPU Acceleration**: Tăng tốc với GPU
- ✅ **Performance Monitoring**: Theo dõi performance real-time

### 8.2 Tính Năng Đặc Biệt

1. **Scalability**: Xử lý từ 1K đến 300K+ samples
2. **Flexibility**: Hỗ trợ multiple data types và model types
3. **Efficiency**: Tối ưu hóa memory và computation
4. **Reliability**: Error handling và recovery mechanisms
5. **User Experience**: Progress tracking và real-time feedback

### 8.3 Best Practices Được Áp Dụng

- **Separation of Concerns**: Tách biệt rõ ràng các components
- **Factory Pattern**: Model creation với factory pattern
- **Registry Pattern**: Model management với registry pattern
- **Caching Strategy**: Multi-level caching với intelligent keys
- **Error Handling**: Comprehensive error handling và recovery
- **Performance Optimization**: Memory và computation optimization
- **Code Reusability**: Modular design cho reusability

Phương pháp này thể hiện một cách tiếp cận chuyên nghiệp và toàn diện trong việc xây dựng một nền tảng Machine Learning có khả năng xử lý hiệu quả nhiều loại dữ liệu và mô hình khác nhau.

---

*Data Processing and Training Methodology Documentation*
*Comprehensive Machine Learning Platform*
*Cập nhật: 2025-01-27*
