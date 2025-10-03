# Hướng Dẫn Xử Lý Từng Model trong Dự Án

## Tổng Quan

Dự án này hỗ trợ **15+ models** được tổ chức theo kiến trúc modular với BaseModel làm foundation. Mỗi model có cách xử lý riêng biệt nhưng tuân theo interface chung.

## 1. Kiến Trúc Model System

### 1.1 BaseModel Foundation

```python
class BaseModel:
    """Abstract base class cho tất cả ML models"""
    
    def __init__(self, **kwargs):
        self.model = None
        self.is_fitted = False
        self.training_history = []
        self.model_params = {}
        
    def fit(self, X, y):
        """Abstract method - phải implement trong subclass"""
        pass
        
    def predict(self, X):
        """Abstract method - phải implement trong subclass"""
        pass
        
    def score(self, X, y):
        """Calculate model score"""
        pass
        
    def validate(self, X, y):
        """Validate model performance"""
        pass
```

### 1.2 Model Registry System

```python
# models/register_models.py
def register_all_models(registry):
    """Register tất cả available models trong registry"""
    
    # Clustering models
    registry.register_model('kmeans', KMeansModel, {...})
    
    # Classification models  
    registry.register_model('knn', KNNModel, {...})
    registry.register_model('decision_tree', DecisionTreeModel, {...})
    registry.register_model('naive_bayes', NaiveBayesModel, {...})
    registry.register_model('svm', SVMModel, {...})
    registry.register_model('logistic_regression', LogisticRegressionModel, {...})
    registry.register_model('linear_svc', LinearSVCModel, {...})
    registry.register_model('random_forest', RandomForestModel, {...})
    registry.register_model('adaboost', AdaBoostModel, {...})
    registry.register_model('gradient_boosting', GradientBoostingModel, {...})
    registry.register_model('xgboost', XGBoostModel, {...})
    registry.register_model('lightgbm', LightGBMModel, {...})
    registry.register_model('catboost', CatBoostModel, {...})
    
    # Ensemble models
    registry.register_model('voting_ensemble_hard', EnsembleStackingClassifier, {...})
    registry.register_model('voting_ensemble_soft', EnsembleStackingClassifier, {...})
    registry.register_model('stacking_ensemble_logistic_regression', EnsembleStackingClassifier, {...})
```

## 2. Clustering Models

### 2.1 K-Means Model

```python
class KMeansModel(BaseModel):
    """K-Means clustering với optimal K detection và SVD optimization"""
    
    def __init__(self, n_clusters: int = 5, use_optimal_k: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.use_optimal_k = use_optimal_k
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], y: np.ndarray = None):
        """Fit K-Means model với automatic optimization"""
        
        # Memory optimization cho large datasets
        n_samples, n_features = X.shape
        if n_features > KMEANS_SVD_THRESHOLD:  # 20,000 features
            print(f"🔧 Large feature space ({n_features:,}), applying SVD reduction")
            svd = TruncatedSVD(n_components=KMEANS_SVD_COMPONENTS, random_state=42)
            X_reduced = svd.fit_transform(X)
            self.svd_model = svd
        else:
            X_reduced = X
            
        # Optimal K detection
        if self.use_optimal_k:
            self.n_clusters = self._find_optimal_k(X_reduced)
            
        # Fit K-Means
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.model.fit(X_reduced)
        
        return self
        
    def _find_optimal_k(self, X: np.ndarray) -> int:
        """Find optimal K using elbow method"""
        inertias = []
        K_range = range(2, min(11, X.shape[0] // 2))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            
        # Simple elbow detection
        optimal_k = self._detect_elbow(K_range, inertias)
        print(f"🎯 Optimal K detected: {optimal_k}")
        return optimal_k
```

**Đặc điểm xử lý:**
- **SVD Optimization**: Tự động giảm chiều cho datasets lớn (>20K features)
- **Optimal K Detection**: Tự động tìm số clusters tối ưu
- **Memory Efficient**: Sử dụng sparse matrices khi cần
- **Visualization**: Tạo elbow plots cho K selection

## 3. Classification Models

### 3.1 K-Nearest Neighbors (KNN)

```python
class KNNModel(BaseModel):
    """K-Nearest Neighbors với GPU acceleration"""
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform', 
                 metric: str = 'euclidean', **kwargs):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        
        # GPU acceleration setup
        self.faiss_available = self._check_faiss_availability()
        self.faiss_gpu_available = self._check_faiss_gpu_availability()
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray, use_gpu: bool = False):
        """Fit KNN với memory-efficient handling"""
        
        n_samples, n_features = X.shape
        memory_estimate_gb = (n_samples * n_features * 4) / (1024**3)
        is_sparse = sparse.issparse(X)
        
        # Strategy: Different handling cho embeddings vs TF-IDF/BOW
        if is_sparse:
            # Sparse data (TF-IDF/BOW) - use sklearn
            self.model = KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                metric=self.metric,
                n_jobs=-1
            )
            self.model.fit(X, y)
            
        elif memory_estimate_gb > 2.0:  # Large dense data
            # Large dense data - use FAISS
            if use_gpu and self.faiss_gpu_available:
                self._setup_faiss_gpu(X, y)
            elif self.faiss_available:
                self._setup_faiss_cpu(X, y)
            else:
                # Fallback to sklearn
                self.model = KNeighborsClassifier(
                    n_neighbors=self.n_neighbors,
                    weights=self.weights,
                    metric=self.metric,
                    n_jobs=-1
                )
                self.model.fit(X, y)
        else:
            # Small dense data - use sklearn
            self.model = KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                metric=self.metric,
                n_jobs=-1
            )
            self.model.fit(X, y)
            
        return self
```

**Đặc điểm xử lý:**
- **GPU Acceleration**: FAISS GPU/CPU support với fallback
- **Memory Optimization**: Different strategies cho sparse vs dense data
- **Adaptive Algorithm**: Tự động chọn algorithm dựa trên data size
- **Hyperparameter Tuning**: GridSearchCV với multiple metrics

### 3.2 Decision Tree

```python
class DecisionTreeModel(BaseModel):
    """Decision Tree với advanced pruning techniques"""
    
    def __init__(self, random_state: int = 42, pruning_method: str = 'ccp',
                 cv_folds: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.random_state = random_state
        self.pruning_method = pruning_method
        self.cv_folds = cv_folds
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], y: np.ndarray):
        """Fit Decision Tree với pruning"""
        
        # Create base model
        self.model = DecisionTreeClassifier(random_state=self.random_state)
        
        # Apply pruning nếu enabled
        if self.pruning_method == 'ccp':
            self._apply_ccp_pruning(X, y)
        elif self.pruning_method == 'rep':
            self._apply_rep_pruning(X, y)
        elif self.pruning_method == 'mdl':
            self._apply_mdl_pruning(X, y)
        elif self.pruning_method == 'cv_optimization':
            self._apply_cv_optimization(X, y)
        else:
            # No pruning
            self.model.fit(X, y)
            
        return self
        
    def _apply_ccp_pruning(self, X, y):
        """Cost Complexity Pruning"""
        path = self.model.cost_complexity_pruning_path(X, y)
        ccp_alphas = path.ccp_alphas
        
        # Find optimal alpha using cross-validation
        best_alpha = self._find_optimal_alpha(X, y, ccp_alphas)
        
        # Fit với optimal alpha
        self.model = DecisionTreeClassifier(
            ccp_alpha=best_alpha,
            random_state=self.random_state
        )
        self.model.fit(X, y)
```

**Đặc điểm xử lý:**
- **Multiple Pruning Methods**: CCP, REP, MDL, CV optimization
- **Feature Importance**: Automatic feature importance calculation
- **Overfitting Prevention**: Advanced pruning techniques
- **Visualization**: Tree structure và pruning plots

### 3.3 Naive Bayes

```python
class NaiveBayesModel(BaseModel):
    """Naive Bayes với automatic type selection"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nb_type = None  # Will be determined automatically
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], y: np.ndarray):
        """Fit Naive Bayes với automatic type selection"""
        
        # Auto-detect best Naive Bayes type
        self.nb_type = self._detect_best_nb_type(X, y)
        
        if self.nb_type == 'multinomial':
            self.model = MultinomialNB()
        elif self.nb_type == 'gaussian':
            self.model = GaussianNB()
        elif self.nb_type == 'bernoulli':
            self.model = BernoulliNB()
        else:
            # Default to multinomial
            self.model = MultinomialNB()
            
        self.model.fit(X, y)
        return self
        
    def _detect_best_nb_type(self, X, y):
        """Detect best Naive Bayes type based on data characteristics"""
        
        # Check data characteristics
        is_sparse = sparse.issparse(X)
        has_negative = np.any(X < 0) if not is_sparse else np.any(X.data < 0)
        is_binary = np.all(np.isin(X, [0, 1])) if not is_sparse else np.all(np.isin(X.data, [0, 1]))
        
        if is_sparse:
            return 'multinomial'
        elif is_binary:
            return 'bernoulli'
        elif has_negative:
            return 'gaussian'
        else:
            return 'multinomial'
```

**Đặc điểm xử lý:**
- **Automatic Type Selection**: Tự động chọn Multinomial/Gaussian/Bernoulli
- **Sparse Data Support**: Optimized cho sparse matrices
- **Data Type Detection**: Analyze data characteristics để chọn best type
- **Fast Training**: Very fast training và prediction

### 3.4 Support Vector Machine (SVM)

```python
class SVMModel(BaseModel):
    """SVM với kernel selection và GPU optimization"""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, 
                 gamma: str = 'scale', **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], y: np.ndarray):
        """Fit SVM với automatic kernel selection"""
        
        # Auto-select kernel based on data characteristics
        if self.kernel == 'auto':
            self.kernel = self._select_best_kernel(X, y)
            
        # Create SVM model
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            random_state=42,
            probability=True  # Enable probability estimates
        )
        
        self.model.fit(X, y)
        return self
        
    def _select_best_kernel(self, X, y):
        """Select best kernel based on data characteristics"""
        
        n_samples, n_features = X.shape
        
        # Simple heuristic
        if n_features > n_samples:
            return 'linear'  # High-dimensional data
        elif n_features < 10:
            return 'rbf'     # Low-dimensional data
        else:
            return 'poly'    # Medium-dimensional data
```

**Đặc điểm xử lý:**
- **Kernel Selection**: Automatic kernel selection based on data
- **Probability Estimates**: Enable probability predictions
- **Memory Optimization**: Efficient handling cho large datasets
- **Hyperparameter Tuning**: GridSearchCV cho C và gamma

### 3.5 Logistic Regression

```python
class LogisticRegressionModel(BaseModel):
    """Logistic Regression với automatic parameter optimization"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Default parameters với CPU multithreading
        default_params = {
            'max_iter': 2000,
            'multi_class': 'multinomial',
            'n_jobs': -1,  # Use all CPU cores
            'random_state': 42,
            'C': 1.0,
            'solver': 'lbfgs'
        }
        
        default_params.update(kwargs)
        self.model_params = default_params
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], y: np.ndarray):
        """Fit Logistic Regression với multithreading"""
        
        # Create model với parameters
        self.model = LogisticRegression(**self.model_params)
        
        # Display multithreading info
        n_jobs = self.model_params.get('n_jobs', -1)
        if n_jobs == -1:
            import os
            cpu_count = os.cpu_count()
            print(f"🔄 CPU multithreading: Using all {cpu_count} available cores")
        else:
            print(f"🔄 CPU multithreading: Using {n_jobs} parallel jobs")
            
        self.model.fit(X, y)
        return self
        
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance (coefficients)"""
        if hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        return None
        
    def get_coefficients(self) -> np.ndarray:
        """Get model coefficients"""
        if hasattr(self.model, 'coef_'):
            return self.model.coef_
        return None
```

**Đặc điểm xử lý:**
- **CPU Multithreading**: Sử dụng tất cả CPU cores
- **Feature Importance**: Coefficients as feature importance
- **Solver Selection**: Automatic solver selection
- **Convergence**: Robust convergence với max_iter=2000

## 4. Ensemble Models

### 4.1 Random Forest

```python
class RandomForestModel(BaseModel):
    """Random Forest với GPU-first configuration"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'max_features': 'sqrt',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1,  # Use all CPU cores
            'verbose': 0
        }
        
        default_params.update(kwargs)
        self.model_params = default_params
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], y: np.ndarray):
        """Fit Random Forest với multithreading"""
        
        self.model = RandomForestClassifier(**self.model_params)
        
        # Display multithreading info
        n_jobs = self.model_params.get('n_jobs', -1)
        if n_jobs == -1:
            import os
            cpu_count = os.cpu_count()
            print(f"🔄 CPU multithreading: Using all {cpu_count} available cores")
        else:
            print(f"🔄 CPU multithreading: Using {n_jobs} parallel jobs")
            
        self.model.fit(X, y)
        return self
        
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance từ Random Forest"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
        
    def get_feature_importance_std(self) -> np.ndarray:
        """Get standard deviation của feature importance across trees"""
        if hasattr(self.model, 'estimators_'):
            importances = np.array([tree.feature_importances_ for tree in self.model.estimators_])
            return np.std(importances, axis=0)
        return None
```

**Đặc điểm xử lý:**
- **CPU Multithreading**: Parallel tree building
- **Feature Importance**: Mean và std của feature importance
- **Bootstrap Sampling**: Random sampling với replacement
- **Out-of-bag Scoring**: OOB score calculation

### 4.2 XGBoost

```python
class XGBoostModel(BaseModel):
    """XGBoost với GPU-first configuration"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Import XGBoost
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("XGBoost is required but not installed")
            
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'eta': 0.3,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'min_child_weight': 1,
            'reg_lambda': 1.0,
            'reg_alpha': 0.0,
            'random_state': 42,
            'verbosity': 0
        }
        
        # Configure GPU/CPU based on device policy
        self._configure_device_params(default_params)
        
        default_params.update(kwargs)
        self.model_params = default_params
        
    def _configure_device_params(self, params: Dict[str, Any]):
        """Configure device-specific parameters"""
        try:
            from gpu_config_manager import configure_model_device
            
            device_config = configure_model_device("xgboost")
            
            if device_config["use_gpu"]:
                params.update(device_config["device_params"])
                print(f"🚀 XGBoost configured for GPU: {device_config['gpu_info']}")
            else:
                params.update({
                    "tree_method": "hist",
                    "predictor": "auto"
                })
                print(f"💻 XGBoost configured for CPU")
                
        except ImportError:
            # Fallback to CPU
            params.update({
                "tree_method": "hist",
                "predictor": "auto"
            })
            print(f"💻 XGBoost configured for CPU (fallback)")
```

**Đặc điểm xử lý:**
- **GPU Acceleration**: CUDA support với automatic fallback
- **Tree Method**: Histogram-based tree building
- **Regularization**: L1 và L2 regularization
- **Early Stopping**: Prevent overfitting

### 4.3 LightGBM

```python
class LightGBMModel(BaseModel):
    """LightGBM với GPU acceleration"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Import LightGBM
        try:
            import lightgbm as lgb
            self.lgb = lgb
        except ImportError:
            raise ImportError("LightGBM is required but not installed")
            
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'min_child_samples': 20,
            'lambda_l1': 0.0,
            'lambda_l2': 0.0,
            'random_state': 42,
            'verbosity': -1
        }
        
        # Configure GPU/CPU
        self._configure_device_params(default_params)
        
        default_params.update(kwargs)
        self.model_params = default_params
        
    def _configure_device_params(self, params: Dict[str, Any]):
        """Configure device-specific parameters"""
        try:
            from gpu_config_manager import configure_model_device
            
            device_config = configure_model_device("lightgbm")
            
            if device_config["use_gpu"]:
                params.update(device_config["device_params"])
                print(f"🚀 LightGBM configured for GPU: {device_config['gpu_info']}")
            else:
                params.update({
                    "device": "cpu",
                    "tree_learner": "serial"
                })
                print(f"💻 LightGBM configured for CPU")
                
        except ImportError:
            params.update({
                "device": "cpu",
                "tree_learner": "serial"
            })
            print(f"💻 LightGBM configured for CPU (fallback)")
```

**Đặc điểm xử lý:**
- **GPU Acceleration**: CUDA support
- **Leaf-wise Growth**: More efficient than level-wise
- **Categorical Features**: Native categorical feature support
- **Memory Efficient**: Lower memory usage than XGBoost

### 4.4 CatBoost

```python
class CatBoostModel(BaseModel):
    """CatBoost với GPU acceleration"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Import CatBoost
        try:
            import catboost as cb
            self.cb = cb
        except ImportError:
            raise ImportError("CatBoost is required but not installed")
            
        # Default parameters
        default_params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3.0,
            'random_seed': 42,
            'verbose': False
        }
        
        # Configure GPU/CPU
        self._configure_device_params(default_params)
        
        default_params.update(kwargs)
        self.model_params = default_params
        
    def _configure_device_params(self, params: Dict[str, Any]):
        """Configure device-specific parameters"""
        try:
            from gpu_config_manager import configure_model_device
            
            device_config = configure_model_device("catboost")
            
            if device_config["use_gpu"]:
                params.update(device_config["device_params"])
                print(f"🚀 CatBoost configured for GPU: {device_config['gpu_info']}")
            else:
                params.update({
                    "task_type": "CPU"
                })
                print(f"💻 CatBoost configured for CPU")
                
        except ImportError:
            params.update({
                "task_type": "CPU"
            })
            print(f"💻 CatBoost configured for CPU (fallback)")
```

**Đặc điểm xử lý:**
- **GPU Acceleration**: CUDA support
- **Categorical Features**: Native categorical feature handling
- **Overfitting Prevention**: Built-in overfitting detection
- **No Preprocessing**: Handles categorical features natively

## 5. Ensemble Learning

### 5.1 Voting Ensemble

```python
class EnsembleStackingClassifier(BaseModel):
    """Ensemble classifier với voting và stacking"""
    
    def __init__(self, ensemble_type: str = 'voting_hard', **kwargs):
        super().__init__(**kwargs)
        self.ensemble_type = ensemble_type
        self.base_models = []
        self.ensemble_model = None
        
    def create_ensemble_classifier(self, base_models: List[Tuple[str, Any]]):
        """Create ensemble classifier với base models"""
        
        self.base_models = base_models
        
        if self.ensemble_type.startswith('voting'):
            # Voting ensemble
            voting_type = 'hard' if 'hard' in self.ensemble_type else 'soft'
            self.ensemble_model = VotingClassifier(
                estimators=base_models,
                voting=voting_type
            )
        elif self.ensemble_type.startswith('stacking'):
            # Stacking ensemble
            meta_learner = self._get_meta_learner()
            self.ensemble_model = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=5
            )
            
    def _get_meta_learner(self):
        """Get meta-learner cho stacking"""
        if 'logistic_regression' in self.ensemble_type:
            return LogisticRegression(random_state=42)
        elif 'random_forest' in self.ensemble_type:
            return RandomForestClassifier(random_state=42)
        elif 'xgboost' in self.ensemble_type:
            return XGBClassifier(random_state=42)
        else:
            return LogisticRegression(random_state=42)
```

**Đặc điểm xử lý:**
- **Voting**: Hard voting và soft voting
- **Stacking**: Meta-learner training
- **Base Models**: Flexible base model selection
- **Cross-validation**: Built-in CV cho stacking

## 6. Model Factory và Registry

### 6.1 Model Factory

```python
class ModelFactory:
    """Factory cho creating model instances"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        
    def create_model(self, model_name: str, **kwargs):
        """Create model instance từ registry"""
        
        model_class = self.registry.get_model(model_name)
        if model_class is None:
            print(f"❌ Model '{model_name}' not found in registry")
            return None
            
        try:
            model_instance = model_class(**kwargs)
            print(f"✅ Created {model_name} model instance")
            return model_instance
        except Exception as e:
            print(f"❌ Failed to create {model_name}: {e}")
            return None
```

### 6.2 Model Registry

```python
class ModelRegistry:
    """Registry cho managing available models"""
    
    def __init__(self):
        self.models = {}
        
    def register_model(self, name: str, model_class: type, metadata: Dict[str, Any]):
        """Register model trong registry"""
        self.models[name] = {
            'class': model_class,
            'metadata': metadata
        }
        
    def get_model(self, name: str) -> type:
        """Get model class từ registry"""
        if name in self.models:
            return self.models[name]['class']
        return None
        
    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.models.keys())
        
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """Get model metadata"""
        if name in self.models:
            return self.models[name]['metadata']
        return None
```

## 7. Training Pipeline Integration

### 7.1 Model Selection và Creation

```python
# app.py - train_models_with_scaling function
for model_name in selected_models:
    # Map model name
    mapped_name = model_mapping.get(model_name, model_name)
    
    # Create model using factory
    model = model_factory.create_model(mapped_name)
    if model is None:
        st.error(f"❌ Failed to create model: {mapped_name}")
        continue
        
    # Train model
    if optuna_enabled:
        # Use Optuna optimization
        optimizer = OptunaOptimizer(optuna_config)
        optimization_result = optimizer.optimize_model(
            model_name=mapped_name,
            model_class=model.__class__,
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val
        )
        
        # Train final model với best params
        final_model = model_factory.create_model(mapped_name)
        final_model.set_params(**optimization_result['best_params'])
        final_model.fit(X_train_scaled, y_train)
    else:
        # Train without optimization
        model.fit(X_train_scaled, y_train)
        final_model = model
```

### 7.2 Model Evaluation

```python
# Comprehensive evaluation
evaluator = ComprehensiveEvaluator()
results = evaluator.evaluate_model(
    model=final_model,
    X_test=X_test_scaled,
    y_test=y_test,
    model_name=mapped_name,
    embedding_method=embedding_method,
    scaler_name=scaler_name
)
```

## 8. Best Practices

### 8.1 Model Selection Guidelines

1. **Small Datasets (< 1000 samples)**:
   - KNN, Decision Tree, Naive Bayes
   - Avoid complex models (XGBoost, LightGBM)

2. **Medium Datasets (1000-10000 samples)**:
   - Random Forest, SVM, Logistic Regression
   - Consider ensemble methods

3. **Large Datasets (> 10000 samples)**:
   - XGBoost, LightGBM, CatBoost
   - Use GPU acceleration

4. **Text Data**:
   - Naive Bayes (multinomial)
   - SVM với linear kernel
   - Logistic Regression

5. **High-Dimensional Data**:
   - Linear models (Logistic Regression, SVM linear)
   - Avoid tree-based models

### 8.2 Memory Optimization

1. **Sparse Data**: Use sparse matrices cho BoW/TF-IDF
2. **Large Datasets**: Enable SVD reduction
3. **GPU Models**: Use GPU acceleration khi available
4. **Batch Processing**: Process data in batches

### 8.3 Performance Optimization

1. **CPU Multithreading**: Enable n_jobs=-1 cho CPU models
2. **GPU Acceleration**: Use GPU cho XGBoost, LightGBM, CatBoost
3. **Early Stopping**: Prevent overfitting
4. **Hyperparameter Tuning**: Use Optuna optimization

## 9. Troubleshooting

### 9.1 Common Issues

1. **Memory Issues**:
   - Reduce dataset size
   - Use sparse matrices
   - Enable SVD reduction

2. **Slow Training**:
   - Enable GPU acceleration
   - Use CPU multithreading
   - Reduce hyperparameter search space

3. **Poor Performance**:
   - Check data preprocessing
   - Try different models
   - Use ensemble methods

4. **Convergence Issues**:
   - Increase max_iter
   - Adjust learning rate
   - Check data scaling

### 9.2 Model-Specific Issues

1. **KNN**: Memory issues với large datasets
2. **SVM**: Slow với large datasets
3. **XGBoost**: GPU memory issues
4. **LightGBM**: Categorical feature encoding
5. **CatBoost**: Slow với small datasets

## Kết Luận

Dự án này cung cấp một hệ thống model processing toàn diện với:

- **15+ models** với specialized handling
- **Modular architecture** dễ extend
- **GPU acceleration** cho performance
- **Memory optimization** cho large datasets
- **Automatic optimization** với Optuna
- **Ensemble learning** capabilities

Mỗi model có cách xử lý riêng biệt nhưng tuân theo interface chung, tạo ra một hệ thống flexible và scalable cho machine learning tasks.
