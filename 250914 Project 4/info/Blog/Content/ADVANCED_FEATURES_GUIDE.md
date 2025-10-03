# Advanced Features Guide - Comprehensive Machine Learning Platform

## Tổng Quan Advanced Features

Dự án **Comprehensive Machine Learning Platform** tích hợp nhiều tính năng nâng cao được thiết kế để tối ưu hóa performance, memory management, GPU acceleration, và user experience. Các tính năng này bao gồm advanced caching, parallel processing, memory optimization, error recovery, và comprehensive monitoring systems.

---

## 1. Memory Management & Optimization

### 1.1 Advanced Memory Management

```python
class AdvancedMemoryManager:
    """Advanced memory management for large-scale ML operations"""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.memory_threshold = max_memory_gb * 1024**3  # Convert to bytes
        self.memory_monitor = MemoryMonitor()
        
    def optimize_data_loading(self, dataset_path: str, chunk_size: int = 10000):
        """Optimize data loading based on available memory"""
        
        available_memory = psutil.virtual_memory().available
        
        if available_memory < self.memory_threshold:
            # Use chunked loading for large datasets
            return self.load_data_chunked(dataset_path, chunk_size)
        else:
            # Load full dataset if memory allows
            return pd.read_csv(dataset_path)
            
    def load_data_chunked(self, dataset_path: str, chunk_size: int):
        """Load data in chunks to manage memory usage"""
        
        chunks = []
        total_rows = 0
        
        for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
            chunks.append(chunk)
            total_rows += len(chunk)
            
            # Check memory usage
            if self.memory_monitor.get_memory_usage() > 0.8:  # 80% threshold
                break
                
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        else:
            raise MemoryError("Insufficient memory to load dataset")
            
    def cleanup_memory(self):
        """Aggressive memory cleanup"""
        
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Clear unused variables
        for name in list(globals().keys()):
            if not name.startswith('_'):
                try:
                    obj = globals()[name]
                    if hasattr(obj, '__dict__'):
                        obj.__dict__.clear()
                except:
                    pass
                    
        # Final garbage collection
        gc.collect()
        
        print(f"Memory cleanup completed. Current usage: {self.memory_monitor.get_memory_usage():.1f}%")
```

### 1.2 Memory Monitoring System

```python
class MemoryMonitor:
    """Real-time memory monitoring and alerting"""
    
    def __init__(self):
        self.memory_history = []
        self.alert_threshold = 0.85  # 85% memory usage
        
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        
        memory = psutil.virtual_memory()
        usage_percent = memory.percent / 100.0
        
        # Store in history
        self.memory_history.append({
            'timestamp': time.time(),
            'usage_percent': usage_percent,
            'available_gb': memory.available / 1024**3
        })
        
        # Keep only last 100 measurements
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-100:]
            
        return usage_percent
        
    def check_memory_alert(self) -> bool:
        """Check if memory usage exceeds threshold"""
        
        current_usage = self.get_memory_usage()
        
        if current_usage > self.alert_threshold:
            print(f"⚠️ Memory usage alert: {current_usage:.1%}")
            return True
            
        return False
        
    def get_memory_trend(self) -> str:
        """Analyze memory usage trend"""
        
        if len(self.memory_history) < 10:
            return "insufficient_data"
            
        recent_usage = [m['usage_percent'] for m in self.memory_history[-10:]]
        avg_usage = sum(recent_usage) / len(recent_usage)
        
        if avg_usage > 0.8:
            return "high"
        elif avg_usage > 0.6:
            return "medium"
        else:
            return "low"
```

### 1.3 Sparse Matrix Optimization

```python
class SparseMatrixOptimizer:
    """Optimize sparse matrix operations for memory efficiency"""
    
    @staticmethod
    def optimize_sparse_matrix(X, sparsity_threshold: float = 0.1):
        """Convert dense matrix to sparse if beneficial"""
        
        if hasattr(X, 'toarray'):  # Already sparse
            return X
            
        # Calculate sparsity
        total_elements = X.shape[0] * X.shape[1]
        non_zero_elements = np.count_nonzero(X)
        sparsity = 1 - (non_zero_elements / total_elements)
        
        if sparsity > sparsity_threshold:
            # Convert to sparse matrix
            from scipy import sparse
            return sparse.csr_matrix(X)
        else:
            return X
            
    @staticmethod
    def optimize_sparse_operations(X_sparse, operation: str):
        """Optimize sparse matrix operations"""
        
        if operation == "svd":
            # Use sparse SVD for large sparse matrices
            from sklearn.decomposition import TruncatedSVD
            
            if X_sparse.shape[1] > 10000:
                svd = TruncatedSVD(n_components=min(500, X_sparse.shape[1]-1))
                return svd.fit_transform(X_sparse)
            else:
                return X_sparse
                
        elif operation == "clustering":
            # Use sparse clustering algorithms
            from sklearn.cluster import KMeans
            
            if X_sparse.shape[0] > 50000:
                # Use MiniBatchKMeans for large datasets
                from sklearn.cluster import MiniBatchKMeans
                return MiniBatchKMeans(n_clusters=5, random_state=42)
            else:
                return KMeans(n_clusters=5, random_state=42)
```

---

## 2. GPU Acceleration & Optimization

### 2.1 GPU Configuration Manager (`gpu_config_manager.py`)

```python
class GPUConfigurationManager:
    """Advanced GPU configuration and optimization"""
    
    def __init__(self):
        self.gpu_available = self.detect_gpu_availability()
        self.gpu_config = self.get_optimal_gpu_config()
        
    def detect_gpu_availability(self) -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive GPU detection"""
        
        gpu_info = {
            'cuda_available': False,
            'gpu_count': 0,
            'gpu_names': [],
            'gpu_memory': [],
            'driver_version': None
        }
        
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                gpu_info['gpu_count'] = torch.cuda.device_count()
                
                for i in range(gpu_info['gpu_count']):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    
                    gpu_info['gpu_names'].append(gpu_name)
                    gpu_info['gpu_memory'].append(gpu_memory)
                    
                gpu_info['driver_version'] = torch.version.cuda
                
        except ImportError:
            print("PyTorch not available for GPU detection")
            
        return gpu_info['cuda_available'], gpu_info
        
    def get_optimal_gpu_config(self) -> Dict[str, Any]:
        """Get optimal GPU configuration based on available hardware"""
        
        if not self.gpu_available:
            return {'device': 'cpu', 'optimization_level': 'none'}
            
        gpu_info = self.gpu_info
        
        # Determine optimization level based on GPU capabilities
        if gpu_info['gpu_count'] >= 2:
            optimization_level = 'high'
        elif gpu_info['gpu_memory'][0] >= 8:  # 8GB+ VRAM
            optimization_level = 'medium'
        else:
            optimization_level = 'low'
            
        config = {
            'device': 'cuda',
            'optimization_level': optimization_level,
            'gpu_count': gpu_info['gpu_count'],
            'memory_fraction': 0.8,  # Use 80% of GPU memory
            'allow_growth': True
        }
        
        return config
        
    def configure_model_for_gpu(self, model_name: str) -> Dict[str, Any]:
        """Configure specific model for GPU optimization"""
        
        gpu_configs = {
            'xgboost': {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor'
            },
            'lightgbm': {
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            },
            'catboost': {
                'task_type': 'GPU',
                'devices': '0'
            },
            'knn': {
                'algorithm': 'faiss_gpu',
                'metric': 'euclidean'
            }
        }
        
        return gpu_configs.get(model_name, {})
```

### 2.2 RAPIDS cuML Integration (`utils/rapids_detector.py`)

```python
class RapidsDetector:
    """RAPIDS cuML detection and integration"""
    
    def __init__(self):
        self.rapids_available = self.detect_rapids_availability()
        self.cuml_models = self.get_cuml_models()
        
    def detect_rapids_availability(self) -> Dict[str, Any]:
        """Detect RAPIDS cuML availability and capabilities"""
        
        detection_result = {
            'cuml_available': False,
            'gpu_available': False,
            'device_type': 'cpu',
            'error_message': None,
            'version': None
        }
        
        try:
            import cuml
            detection_result['cuml_available'] = True
            detection_result['version'] = cuml.__version__
            
            # Check GPU availability
            try:
                import cupy as cp
                detection_result['gpu_available'] = True
                detection_result['device_type'] = 'gpu'
            except ImportError:
                detection_result['device_type'] = 'cpu'
                
        except ImportError as e:
            detection_result['error_message'] = str(e)
            
        return detection_result
        
    def get_cuml_models(self) -> Dict[str, Any]:
        """Get available cuML models"""
        
        if not self.rapids_available['cuml_available']:
            return {}
            
        try:
            import cuml
            
            models = {
                'kmeans': cuml.KMeans,
                'random_forest': cuml.RandomForestClassifier,
                'svm': cuml.SVM,
                'logistic_regression': cuml.LogisticRegression,
                'pca': cuml.PCA,
                'tsvd': cuml.TruncatedSVD
            }
            
            return models
            
        except ImportError:
            return {}
            
    def optimize_for_rapids(self, model_name: str, X, y=None):
        """Optimize model for RAPIDS cuML"""
        
        if not self.rapids_available['cuml_available']:
            return None
            
        cuml_model = self.cuml_models.get(model_name)
        
        if cuml_model:
            try:
                # Convert data to cuDF if needed
                if hasattr(X, 'to_pandas'):
                    X_cuml = X
                else:
                    import cudf
                    X_cuml = cudf.DataFrame(X)
                    
                # Create and configure model
                model = cuml_model()
                
                if y is not None:
                    if hasattr(y, 'to_pandas'):
                        y_cuml = y
                    else:
                        import cudf
                        y_cuml = cudf.Series(y)
                        
                    model.fit(X_cuml, y_cuml)
                else:
                    model.fit(X_cuml)
                    
                return model
                
            except Exception as e:
                print(f"RAPIDS optimization failed for {model_name}: {e}")
                return None
                
        return None
```

---

## 3. Parallel Processing & Optimization

### 3.1 Advanced Parallel Processing

```python
class ParallelProcessingManager:
    """Advanced parallel processing for ML operations"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(8, os.cpu_count())
        self.thread_pool = None
        self.process_pool = None
        
    def parallel_model_training(self, models_config: List[Dict[str, Any]]):
        """Parallel training of multiple models"""
        
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
        import threading
        
        results = {}
        errors = {}
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            
            # Submit training tasks
            future_to_model = {
                executor.submit(self.train_single_model, config): config['name']
                for config in models_config
            }
            
            # Collect results with progress tracking
            completed = 0
            total = len(future_to_model)
            
            for future in future_to_model:
                model_name = future_to_model[future]
                
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results[model_name] = result
                    
                except Exception as exc:
                    errors[model_name] = str(exc)
                    print(f'{model_name} generated an exception: {exc}')
                    
                completed += 1
                print(f"Progress: {completed}/{total} models completed")
                
        return results, errors
        
    def parallel_cross_validation(self, model, X, y, cv_folds: int = 5):
        """Parallel cross-validation"""
        
        from sklearn.model_selection import KFold
        from concurrent.futures import ThreadPoolExecutor
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        def cv_fold(fold_data):
            train_idx, val_idx = fold_data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train model on fold
            model_copy = clone(model)
            model_copy.fit(X_train_fold, y_train_fold)
            
            # Evaluate on validation set
            score = model_copy.score(X_val_fold, y_val_fold)
            
            return score
            
        # Parallel execution of CV folds
        with ThreadPoolExecutor(max_workers=min(cv_folds, self.max_workers)) as executor:
            scores = list(executor.map(cv_fold, kf.split(X)))
            
        return np.mean(scores), np.std(scores)
```

### 3.2 Progress Tracking System (`utils/progress_tracker.py`)

```python
class ProgressTracker:
    """Advanced progress tracking with time estimation"""
    
    def __init__(self, total_tasks: int, task_name: str = "Training"):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.task_name = task_name
        self.start_time = time.time()
        self.task_times = []
        self.current_task = ""
        self.current_task_start = None
        
        # Threading for real-time updates
        self.update_queue = queue.Queue()
        self.stop_updates = False
        self.update_thread = None
        
    def start_task(self, task_description: str):
        """Start a new task"""
        
        if self.current_task_start:
            # Finish previous task
            task_time = time.time() - self.current_task_start
            self.task_times.append(task_time)
            
        self.current_task = task_description
        self.current_task_start = time.time()
        
        print(f"🔄 Starting: {task_description}")
        
    def complete_task(self):
        """Complete current task"""
        
        if self.current_task_start:
            task_time = time.time() - self.current_task_start
            self.task_times.append(task_time)
            self.completed_tasks += 1
            
            print(f"✅ Completed: {self.current_task} ({task_time:.2f}s)")
            
        self.current_task_start = None
        
    def get_progress_info(self) -> Dict[str, Any]:
        """Get comprehensive progress information"""
        
        elapsed_time = time.time() - self.start_time
        
        if self.completed_tasks > 0:
            avg_task_time = np.mean(self.task_times)
            remaining_tasks = self.total_tasks - self.completed_tasks
            estimated_remaining_time = avg_task_time * remaining_tasks
        else:
            estimated_remaining_time = 0
            
        progress_percent = (self.completed_tasks / self.total_tasks) * 100
        
        return {
            'completed_tasks': self.completed_tasks,
            'total_tasks': self.total_tasks,
            'progress_percent': progress_percent,
            'elapsed_time': elapsed_time,
            'estimated_remaining_time': estimated_remaining_time,
            'current_task': self.current_task,
            'avg_task_time': np.mean(self.task_times) if self.task_times else 0
        }
        
    def print_progress_summary(self):
        """Print detailed progress summary"""
        
        info = self.get_progress_info()
        
        print(f"\n📊 {self.task_name} Progress Summary:")
        print(f"   Completed: {info['completed_tasks']}/{info['total_tasks']} ({info['progress_percent']:.1f}%)")
        print(f"   Elapsed Time: {info['elapsed_time']:.1f}s")
        print(f"   Estimated Remaining: {info['estimated_remaining_time']:.1f}s")
        print(f"   Average Task Time: {info['avg_task_time']:.2f}s")
        
        if info['current_task']:
            print(f"   Current Task: {info['current_task']}")
```

---

## 4. Advanced Caching System

### 4.1 Intelligent Cache Management

```python
class IntelligentCacheManager:
    """Advanced cache management with intelligent strategies"""
    
    def __init__(self, cache_root: str = "cache/"):
        self.cache_root = Path(cache_root)
        self.cache_metadata = self.load_cache_metadata()
        self.cache_policies = self.get_cache_policies()
        
    def get_cache_policies(self) -> Dict[str, Any]:
        """Define intelligent cache policies"""
        
        return {
            'model_cache': {
                'max_size_gb': 5.0,
                'max_age_days': 30,
                'priority': 'high',
                'compression': True
            },
            'shap_cache': {
                'max_size_gb': 2.0,
                'max_age_days': 7,
                'priority': 'medium',
                'compression': True
            },
            'training_results': {
                'max_size_gb': 1.0,
                'max_age_days': 14,
                'priority': 'low',
                'compression': False
            }
        }
        
    def intelligent_cache_cleanup(self):
        """Intelligent cache cleanup based on policies"""
        
        cleanup_stats = {
            'files_removed': 0,
            'space_freed_gb': 0,
            'cache_types_cleaned': []
        }
        
        for cache_type, policy in self.cache_policies.items():
            cache_dir = self.cache_root / cache_type
            
            if cache_dir.exists():
                removed_files, freed_space = self.cleanup_cache_type(
                    cache_dir, policy
                )
                
                cleanup_stats['files_removed'] += removed_files
                cleanup_stats['space_freed_gb'] += freed_space
                cleanup_stats['cache_types_cleaned'].append(cache_type)
                
        return cleanup_stats
        
    def cleanup_cache_type(self, cache_dir: Path, policy: Dict[str, Any]) -> Tuple[int, float]:
        """Cleanup specific cache type based on policy"""
        
        removed_files = 0
        freed_space = 0
        
        current_time = time.time()
        max_age_seconds = policy['max_age_days'] * 24 * 60 * 60
        
        for file_path in cache_dir.rglob('*'):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                file_size = file_path.stat().st_size
                
                # Remove old files
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        removed_files += 1
                        freed_space += file_size
                    except Exception as e:
                        print(f"Failed to remove {file_path}: {e}")
                        
        return removed_files, freed_space / (1024**3)  # Convert to GB
        
    def optimize_cache_access(self, cache_key: str) -> str:
        """Optimize cache access patterns"""
        
        # Check if cache exists
        cache_path = self.get_cache_path(cache_key)
        
        if cache_path.exists():
            # Update access time
            cache_path.touch()
            
            # Move to faster storage if available
            if self.has_fast_storage():
                self.move_to_fast_storage(cache_path)
                
        return str(cache_path)
```

### 4.2 Cache Performance Monitoring

```python
class CachePerformanceMonitor:
    """Monitor cache performance and optimize access patterns"""
    
    def __init__(self):
        self.access_log = []
        self.hit_rates = {}
        self.performance_metrics = {}
        
    def log_cache_access(self, cache_key: str, hit: bool, access_time: float):
        """Log cache access for performance analysis"""
        
        self.access_log.append({
            'timestamp': time.time(),
            'cache_key': cache_key,
            'hit': hit,
            'access_time': access_time
        })
        
        # Update hit rate
        if cache_key not in self.hit_rates:
            self.hit_rates[cache_key] = {'hits': 0, 'misses': 0}
            
        if hit:
            self.hit_rates[cache_key]['hits'] += 1
        else:
            self.hit_rates[cache_key]['misses'] += 1
            
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        total_accesses = len(self.access_log)
        total_hits = sum(1 for log in self.access_log if log['hit'])
        total_misses = total_accesses - total_hits
        
        overall_hit_rate = total_hits / total_accesses if total_accesses > 0 else 0
        
        # Calculate hit rates by cache type
        hit_rates_by_type = {}
        for cache_key, rates in self.hit_rates.items():
            total = rates['hits'] + rates['misses']
            hit_rate = rates['hits'] / total if total > 0 else 0
            hit_rates_by_type[cache_key] = hit_rate
            
        return {
            'total_accesses': total_accesses,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'overall_hit_rate': overall_hit_rate,
            'hit_rates_by_type': hit_rates_by_type,
            'average_access_time': np.mean([log['access_time'] for log in self.access_log])
        }
```

---

## 5. Error Recovery & Resilience

### 5.1 Advanced Error Recovery

```python
class AdvancedErrorRecovery:
    """Advanced error recovery and resilience mechanisms"""
    
    def __init__(self):
        self.recovery_strategies = {
            'memory_error': self.handle_memory_error,
            'gpu_error': self.handle_gpu_error,
            'timeout_error': self.handle_timeout_error,
            'convergence_error': self.handle_convergence_error,
            'data_error': self.handle_data_error
        }
        
    def handle_memory_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle memory errors with automatic recovery"""
        
        print(f"🔄 Memory error detected: {str(error)}")
        
        # Strategy 1: Reduce batch size
        if 'batch_size' in context:
            new_batch_size = context['batch_size'] // 2
            if new_batch_size > 0:
                print(f"   Reducing batch size to {new_batch_size}")
                context['batch_size'] = new_batch_size
                return True
                
        # Strategy 2: Use chunked processing
        if 'data_size' in context and context['data_size'] > 10000:
            print("   Switching to chunked processing")
            context['chunked_processing'] = True
            return True
            
        # Strategy 3: Clear cache and retry
        print("   Clearing cache and retrying")
        self.clear_memory_cache()
        return True
        
    def handle_gpu_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle GPU errors with fallback to CPU"""
        
        print(f"🔄 GPU error detected: {str(error)}")
        
        # Strategy 1: Fallback to CPU
        if context.get('use_gpu', False):
            print("   Falling back to CPU processing")
            context['use_gpu'] = False
            return True
            
        # Strategy 2: Reduce GPU memory usage
        if 'gpu_memory_fraction' in context:
            new_fraction = context['gpu_memory_fraction'] * 0.5
            if new_fraction > 0.1:
                print(f"   Reducing GPU memory fraction to {new_fraction}")
                context['gpu_memory_fraction'] = new_fraction
                return True
                
        return False
        
    def handle_timeout_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle timeout errors with optimization"""
        
        print(f"🔄 Timeout error detected: {str(error)}")
        
        # Strategy 1: Increase timeout
        if 'timeout' in context:
            new_timeout = context['timeout'] * 2
            print(f"   Increasing timeout to {new_timeout}s")
            context['timeout'] = new_timeout
            return True
            
        # Strategy 2: Reduce complexity
        if 'max_iterations' in context:
            new_iterations = context['max_iterations'] // 2
            if new_iterations > 0:
                print(f"   Reducing max iterations to {new_iterations}")
                context['max_iterations'] = new_iterations
                return True
                
        return False
        
    def retry_with_recovery(self, func, max_retries: int = 3, **kwargs):
        """Retry function with automatic error recovery"""
        
        for attempt in range(max_retries):
            try:
                return func(**kwargs)
                
            except Exception as e:
                error_type = type(e).__name__
                
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    
                    # Apply recovery strategy
                    if error_type in self.recovery_strategies:
                        if self.recovery_strategies[error_type](e, kwargs):
                            print(f"   Recovery strategy applied, retrying...")
                            continue
                            
                    # Default recovery: wait and retry
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
                else:
                    print(f"All {max_retries} attempts failed")
                    raise e
```

### 5.2 Circuit Breaker Pattern

```python
class CircuitBreaker:
    """Circuit breaker pattern for resilient operations"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
                
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
            
        except Exception as e:
            self.on_failure()
            raise e
            
    def on_success(self):
        """Handle successful operation"""
        
        self.failure_count = 0
        self.state = 'CLOSED'
        
    def on_failure(self):
        """Handle failed operation"""
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            print(f"Circuit breaker opened after {self.failure_count} failures")
```

---

## 6. Performance Monitoring & Analytics

### 6.1 Comprehensive Performance Monitoring

```python
class PerformanceMonitor:
    """Comprehensive performance monitoring and analytics"""
    
    def __init__(self):
        self.metrics = {
            'training_times': [],
            'memory_usage': [],
            'gpu_usage': [],
            'cache_hit_rates': [],
            'error_rates': []
        }
        self.start_time = time.time()
        
    def start_training_monitor(self, model_name: str):
        """Start monitoring training process"""
        
        self.current_training = {
            'model_name': model_name,
            'start_time': time.time(),
            'start_memory': psutil.virtual_memory().used,
            'start_gpu_memory': self.get_gpu_memory_usage()
        }
        
    def end_training_monitor(self):
        """End training monitoring and record metrics"""
        
        if not hasattr(self, 'current_training'):
            return
            
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        end_gpu_memory = self.get_gpu_memory_usage()
        
        training_time = end_time - self.current_training['start_time']
        memory_usage = end_memory - self.current_training['start_memory']
        gpu_usage = end_gpu_memory - self.current_training['start_gpu_memory']
        
        # Record metrics
        self.metrics['training_times'].append({
            'model_name': self.current_training['model_name'],
            'time': training_time,
            'memory_usage': memory_usage,
            'gpu_usage': gpu_usage
        })
        
        # Clear current training
        delattr(self, 'current_training')
        
    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage"""
        
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024**3  # GB
        except ImportError:
            pass
            
        return 0.0
        
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        total_time = time.time() - self.start_time
        
        # Calculate statistics
        training_times = [m['time'] for m in self.metrics['training_times']]
        memory_usages = [m['memory_usage'] for m in self.metrics['training_times']]
        
        report = {
            'session_duration': total_time,
            'total_models_trained': len(self.metrics['training_times']),
            'average_training_time': np.mean(training_times) if training_times else 0,
            'total_training_time': sum(training_times),
            'peak_memory_usage': max(memory_usages) if memory_usages else 0,
            'average_memory_usage': np.mean(memory_usages) if memory_usages else 0,
            'performance_by_model': self.get_performance_by_model()
        }
        
        return report
        
    def get_performance_by_model(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics by model"""
        
        model_stats = {}
        
        for metric in self.metrics['training_times']:
            model_name = metric['model_name']
            
            if model_name not in model_stats:
                model_stats[model_name] = {
                    'count': 0,
                    'total_time': 0,
                    'total_memory': 0,
                    'times': [],
                    'memory_usages': []
                }
                
            stats = model_stats[model_name]
            stats['count'] += 1
            stats['total_time'] += metric['time']
            stats['total_memory'] += metric['memory_usage']
            stats['times'].append(metric['time'])
            stats['memory_usages'].append(metric['memory_usage'])
            
        # Calculate averages
        for model_name, stats in model_stats.items():
            stats['avg_time'] = stats['total_time'] / stats['count']
            stats['avg_memory'] = stats['total_memory'] / stats['count']
            stats['std_time'] = np.std(stats['times'])
            stats['std_memory'] = np.std(stats['memory_usages'])
            
        return model_stats
```

---

## 7. Advanced Configuration Management

### 7.1 Dynamic Configuration System

```python
class DynamicConfigurationManager:
    """Dynamic configuration management with environment adaptation"""
    
    def __init__(self):
        self.base_config = self.load_base_config()
        self.environment_config = self.detect_environment()
        self.runtime_config = self.generate_runtime_config()
        
    def detect_environment(self) -> Dict[str, Any]:
        """Detect runtime environment and capabilities"""
        
        environment = {
            'cpu_cores': os.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / 1024**3,
            'gpu_available': self.detect_gpu(),
            'disk_space_gb': psutil.disk_usage('/').free / 1024**3,
            'platform': platform.system(),
            'python_version': platform.python_version()
        }
        
        return environment
        
    def generate_runtime_config(self) -> Dict[str, Any]:
        """Generate optimal runtime configuration"""
        
        config = self.base_config.copy()
        
        # Adapt to available resources
        if self.environment_config['memory_gb'] < 8:
            config['max_sample_size'] = 10000
            config['enable_chunked_processing'] = True
        else:
            config['max_sample_size'] = 50000
            config['enable_chunked_processing'] = False
            
        # Adapt to CPU cores
        config['max_workers'] = min(8, self.environment_config['cpu_cores'])
        
        # Adapt to GPU availability
        if self.environment_config['gpu_available']:
            config['enable_gpu_acceleration'] = True
            config['gpu_memory_fraction'] = 0.8
        else:
            config['enable_gpu_acceleration'] = False
            
        return config
        
    def get_adaptive_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get adaptive configuration for specific model"""
        
        base_model_config = self.runtime_config.get('models', {}).get(model_name, {})
        
        # Adapt based on environment
        if model_name in ['xgboost', 'lightgbm', 'catboost']:
            if self.environment_config['gpu_available']:
                base_model_config['device'] = 'gpu'
            else:
                base_model_config['device'] = 'cpu'
                
        # Adapt based on memory
        if self.environment_config['memory_gb'] < 4:
            if 'n_estimators' in base_model_config:
                base_model_config['n_estimators'] = min(100, base_model_config['n_estimators'])
                
        return base_model_config
```

---

## 8. Best Practices & Optimization Guidelines

### 8.1 Memory Optimization Best Practices

1. **Use Sparse Matrices**: Convert dense matrices to sparse when sparsity > 10%
2. **Chunked Processing**: Process large datasets in chunks
3. **Garbage Collection**: Regular garbage collection after heavy operations
4. **Memory Monitoring**: Continuous monitoring of memory usage
5. **Cache Management**: Intelligent cache cleanup và management

### 8.2 GPU Optimization Best Practices

1. **GPU Detection**: Automatic detection và configuration
2. **Memory Management**: Proper GPU memory management
3. **Fallback Strategies**: Automatic fallback to CPU when GPU fails
4. **Batch Processing**: Optimize batch sizes for GPU memory
5. **Mixed Precision**: Use mixed precision training when available

### 8.3 Performance Optimization Best Practices

1. **Parallel Processing**: Use parallel processing for independent operations
2. **Caching**: Implement intelligent caching strategies
3. **Progress Tracking**: Real-time progress tracking và time estimation
4. **Error Recovery**: Comprehensive error handling và recovery
5. **Resource Monitoring**: Continuous monitoring của system resources

---

## 🎯 Kết Luận

Advanced Features của Comprehensive Machine Learning Platform cung cấp:

- **Memory Management**: Advanced memory optimization và monitoring
- **GPU Acceleration**: Comprehensive GPU support với automatic fallback
- **Parallel Processing**: Efficient parallel execution của ML operations
- **Intelligent Caching**: Smart caching với performance optimization
- **Error Recovery**: Robust error handling và recovery mechanisms
- **Performance Monitoring**: Comprehensive performance analytics
- **Dynamic Configuration**: Adaptive configuration based on environment

Các tính năng này đảm bảo rằng platform có thể handle large-scale ML workloads một cách efficient và reliable, với automatic optimization và recovery mechanisms.
