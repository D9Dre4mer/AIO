"""
Support Vector Machine Model Implementation with Detailed Training Logging and GPU Support
"""

from typing import Union, Dict, Any, Optional
import numpy as np
from scipy import sparse
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import logging


from ..base.base_model import BaseModel


class GPUAccelerator:
    """GPU acceleration manager for SVM training"""
    
    def __init__(self):
        self.gpu_available = False
        self.gpu_library = None
        self.gpu_device_info = {}
        self._detect_gpu()
    
    def _detect_gpu(self):
        """Detect available GPU acceleration libraries"""
        self.gpu_available = False
        
        # Try CUDA (PyTorch)
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_library = "PyTorch (CUDA)"
                self.gpu_device_info = {
                    'device_count': torch.cuda.device_count(),
                    'current_device': torch.cuda.current_device(),
                    'device_name': torch.cuda.get_device_name(0),
                    'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                    'memory_free': torch.cuda.memory_reserved(0) / 1024**3
                }
                print(f"🚀 GPU detected: {self.gpu_library}")
                print(f"   Device: {self.gpu_device_info['device_name']}")
                print(f"   Memory: {self.gpu_device_info['memory_total']:.1f} GB")
                return
        except ImportError:
            pass
        
        # Try CuPy (CUDA)
        try:
            import cupy as cp
            if cp.cuda.is_available():
                self.gpu_available = True
                self.gpu_library = "CuPy (CUDA)"
                self.gpu_device_info = {
                    'device_count': cp.cuda.runtime.getDeviceCount(),
                    'current_device': cp.cuda.runtime.getDevice(),
                    'device_name': cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())['name'].decode(),
                    'memory_total': cp.cuda.runtime.memGetInfo()[1] / 1024**3,
                    'memory_free': cp.cuda.runtime.memGetInfo()[0] / 1024**3
                }
                print(f"🚀 GPU detected: {self.gpu_library}")
                print(f"   Device: {self.gpu_device_info['device_name']}")
                print(f"   Memory: {self.gpu_device_info['memory_total']:.1f} GB")
                return
        except ImportError:
            pass
        
        # Try RAPIDS cuML (CUDA)
        try:
            import cuml
            if hasattr(cuml, 'SVC'):
                self.gpu_available = True
                self.gpu_library = "RAPIDS cuML (CUDA)"
                self.gpu_device_info = {
                    'device_count': 1,  # cuML typically uses single GPU
                    'current_device': 0,
                    'device_name': "RAPIDS cuML GPU",
                    'memory_total': 0,  # Not easily accessible
                    'memory_free': 0
                }
                print(f"🚀 GPU detected: {self.gpu_library}")
                return
        except ImportError:
            pass
        
        # Try Apple Metal (MPS)
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.gpu_available = True
                self.gpu_library = "PyTorch (Apple Metal)"
                self.gpu_device_info = {
                    'device_count': 1,
                    'current_device': 0,
                    'device_name': "Apple Metal GPU",
                    'memory_total': 0,
                    'memory_free': 0
                }
                print(f"🚀 GPU detected: {self.gpu_library}")
                return
        except ImportError:
            pass
        
        print("⚠️  No GPU acceleration libraries detected. Using CPU only.")
        self.gpu_available = False
        self.gpu_library = "CPU (scikit-learn)"
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        return {
            'gpu_available': self.gpu_available,
            'gpu_library': self.gpu_library,
            'gpu_device_info': self.gpu_device_info.copy()
        }
    
    def create_gpu_svm(self, **kwargs):
        """Create GPU-accelerated SVM model if available"""
        if not self.gpu_available:
            return None
        
        try:
            if "RAPIDS cuML" in self.gpu_library:
                from cuml.svm import SVC as cuMLSVC
                return cuMLSVC(**kwargs)
            elif "PyTorch" in self.gpu_library:
                # PyTorch doesn't have direct SVM, but we can use it for data transfer
                return None
            elif "CuPy" in self.gpu_library:
                # CuPy doesn't have SVM, but we can use it for data transfer
                return None
        except Exception as e:
            print(f"⚠️  GPU SVM creation failed: {e}")
            return None
        
        return None


class SVMTrainingLogger:
    """Detailed logger for SVM training process with GPU support"""
    
    def __init__(self, model_name: str = "SVM"):
        self.model_name = model_name
        self.training_log = []
        self.start_time = None
        self.end_time = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"{model_name}_Training")
        
    def log_start(self, X_shape: tuple, y_shape: tuple, parameters: Dict, gpu_info: Dict = None):
        """Log training start with GPU information"""
        self.start_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        memory_mb = X_shape[0] * X_shape[1] * 8 / 1024 / 1024
        log_entry = {
            'timestamp': timestamp,
            'action': 'training_start',
            'model_name': self.model_name,
            'data_shape': {
                'X_shape': X_shape,
                'y_shape': y_shape,
                'n_samples': X_shape[0],
                'n_features': X_shape[1],
                'n_classes': len(np.unique(y_shape)) if hasattr(y_shape, '__len__') and len(y_shape) == 1 else 1
            },
            'parameters': parameters,
            'memory_usage': f"{memory_mb:.2f} MB",
            'gpu_info': gpu_info
        }
        
        self.training_log.append(log_entry)
        self.logger.info(f"🚀 Starting {self.model_name} training at {timestamp}")
        # Handle both tuple and array for y_shape
        if hasattr(y_shape, '__len__') and not isinstance(y_shape, (str, bytes)):
            if len(y_shape) == 1:  # y_shape is (n_samples,)
                n_classes = len(np.unique(y_shape))
            else:  # y_shape is (n_samples, n_features)
                n_classes = 1  # Default for regression-like data
        else:
            n_classes = 1
        self.logger.info(f"📊 Data: {X_shape[0]:,} samples, "
                         f"{X_shape[1]:,} features, {n_classes} classes")
        self.logger.info(f"⚙️  Parameters: {parameters}")
        self.logger.info(f"💾 Estimated memory: {log_entry['memory_usage']}")
        
        if gpu_info and gpu_info.get('gpu_available'):
            self.logger.info(f"🚀 GPU Acceleration: {gpu_info['gpu_library']}")
            if gpu_info.get('gpu_device_info', {}).get('device_name'):
                self.logger.info(f"   Device: {gpu_info['gpu_device_info']['device_name']}")
        else:
            self.logger.info("💻 Using CPU (scikit-learn)")
        
    def log_data_preparation(self, X_type: str, conversion_time: float = None, gpu_transfer: bool = False):
        """Log data preparation steps"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            'timestamp': timestamp,
            'action': 'data_preparation',
            'data_type': X_type,
            'conversion_time': conversion_time,
            'gpu_transfer': gpu_transfer
        }
        
        self.training_log.append(log_entry)
        self.logger.info(f"🔧 Data preparation: {X_type}")
        if conversion_time:
            self.logger.info(f"⏱️  Conversion time: {conversion_time:.4f}s")
        if gpu_transfer:
            self.logger.info("🚀 Data transferred to GPU")
            
    def log_training_progress(self, iteration: int = None, n_support_vectors: int = None, gpu_usage: bool = False):
        """Log training progress"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            'timestamp': timestamp,
            'action': 'training_progress',
            'iteration': iteration,
            'n_support_vectors': n_support_vectors,
            'gpu_usage': gpu_usage
        }
        
        self.training_log.append(log_entry)
        if iteration:
            self.logger.info(f"🔄 Training iteration: {iteration}")
        if n_support_vectors:
            self.logger.info(f"🎯 Support vectors found: {n_support_vectors}")
        if gpu_usage:
            self.logger.info("🚀 Training on GPU")
            
    def log_training_complete(self, training_time: float, n_support_vectors: int, gpu_used: bool = False):
        """Log training completion"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            'timestamp': timestamp,
            'action': 'training_complete',
            'training_time': training_time,
            'n_support_vectors': n_support_vectors,
            'gpu_used': gpu_used
        }
        
        self.training_log.append(log_entry)
        accelerator = "GPU" if gpu_used else "CPU"
        self.logger.info(f"✅ Training completed on {accelerator} in {training_time:.4f}s")
        self.logger.info(f"🎯 Total support vectors: {n_support_vectors}")
        
    def log_prediction(self, X_shape: tuple, prediction_time: float, gpu_used: bool = False):
        """Log prediction process"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            'timestamp': timestamp,
            'action': 'prediction',
            'X_shape': X_shape,
            'prediction_time': prediction_time,
            'gpu_used': gpu_used
        }
        
        self.training_log.append(log_entry)
        accelerator = "GPU" if gpu_used else "CPU"
        self.logger.info(f"🔮 Prediction on {X_shape[0]:,} samples using {accelerator} in {prediction_time:.4f}s")
        
    def log_evaluation(self, metrics: Dict[str, float]):
        """Log evaluation results"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            'timestamp': timestamp,
            'action': 'evaluation',
            'metrics': metrics
        }
        
        self.training_log.append(log_entry)
        self.logger.info(f"📊 Evaluation results:")
        for metric, value in metrics.items():
            self.logger.info(f"   • {metric}: {value:.4f}")
            
    def log_error(self, error_message: str, error_type: str = "error"):
        """Log errors during training"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            'timestamp': timestamp,
            'action': 'error',
            'error_type': error_type,
            'error_message': error_message
        }
        
        self.training_log.append(log_entry)
        self.logger.error(f"❌ {error_type.upper()}: {error_message}")
        
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        if not self.start_time:
            return {"status": "No training started"}
            
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        # Count different types of log entries
        action_counts = {}
        for entry in self.training_log:
            action = entry['action']
            action_counts[action] = action_counts.get(action, 0) + 1
            
        return {
            'model_name': self.model_name,
            'total_training_time': total_time,
            'start_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
            'end_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.end_time)),
            'log_entries_count': len(self.training_log),
            'action_counts': action_counts,
            'training_log': self.training_log
        }
        
    def print_summary(self):
        """Print training summary to console"""
        summary = self.get_training_summary()
        
        print(f"\n{'='*60}")
        print(f"📋 {self.model_name} TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"🚀 Start Time: {summary['start_time']}")
        print(f"✅ End Time: {summary['end_time']}")
        print(f"⏱️  Total Time: {summary['total_training_time']:.4f}s")
        print(f"📝 Log Entries: {summary['log_entries_count']}")
        
        print(f"\n📊 Action Breakdown:")
        for action, count in summary['action_counts'].items():
            print(f"   • {action}: {count}")
            
        print(f"\n📋 Detailed Log:")
        for entry in summary['training_log']:
            timestamp = entry['timestamp']
            action = entry['action']
            print(f"   [{timestamp}] {action}")
            
            # Print specific details for each action
            if action == 'training_start':
                data = entry['data_shape']
                print(f"      📊 Data: {data['n_samples']:,} samples, {data['n_features']:,} features, {data['n_classes']} classes")
                print(f"      💾 Memory: {entry['memory_usage']}")
                if entry.get('gpu_info', {}).get('gpu_available'):
                    print(f"      🚀 GPU: {entry['gpu_info']['gpu_library']}")
            elif action == 'training_complete':
                gpu_used = entry.get('gpu_used', False)
                accelerator = "GPU" if gpu_used else "CPU"
                print(f"      ⏱️  Training time ({accelerator}): {entry['training_time']:.4f}s")
                print(f"      🎯 Support vectors: {entry['n_support_vectors']}")
            elif action == 'evaluation':
                print(f"      📈 Metrics: {entry['metrics']}")
            elif action == 'error':
                print(f"      ❌ {entry['error_type']}: {entry['error_message']}")


class SVMModel(BaseModel):
    """Support Vector Machine model for classification with detailed logging and GPU support"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Default SVM parameters - optimized for speed
        default_params = {
            'C': 0.1,  # Reduced regularization for faster training
            'kernel': 'rbf',
            'gamma': 'auto',  # Faster than 'scale'
            'random_state': 42,
            'verbose': False,
            'max_iter': 1000,  # Limit iterations to prevent long training
            'tol': 1e-3,  # Relaxed tolerance for faster convergence
            'auto_fast': True  # Auto-select fastest algorithm
        }
        
        # Update with user parameters
        default_params.update(kwargs)
        self.model_params = default_params
        
        # Initialize GPU accelerator
        self.gpu_accelerator = GPUAccelerator()
        
        # Initialize SVM model (try GPU first, fallback to CPU)
        self.model = self._create_svm_model()
        self.gpu_model = None
        
        # Initialize training logger
        self.training_logger = SVMTrainingLogger("SVM")
        
    def _create_svm_model(self):
        """Create SVM model with GPU acceleration if available"""
        # Try GPU first
        if self.gpu_accelerator.gpu_available:
            gpu_model = self.gpu_accelerator.create_gpu_svm(**self.model_params)
            if gpu_model is not None:
                self.gpu_model = gpu_model
                print(f"🚀 Created GPU-accelerated SVM using {self.gpu_accelerator.gpu_library}")
                return gpu_model
        
        # Try fast SVM alternatives for CPU
        try:
            # Auto-select fastest algorithm if enabled
            if self.model_params.get('auto_fast', True):
                # For linear kernel, LinearSVC is fastest
                if self.model_params.get('kernel') == 'linear':
                    from sklearn.svm import LinearSVC
                    print(f"⚡ Using LinearSVC (fastest for linear)")
                    return LinearSVC(
                        C=self.model_params.get('C', 0.1),
                        random_state=self.model_params.get('random_state', 42),
                        max_iter=self.model_params.get('max_iter', 1000),
                        tol=self.model_params.get('tol', 1e-3)
                    )
                
                # For other kernels, SGDClassifier is fastest
                from sklearn.linear_model import SGDClassifier
                print(f"⚡ Using SGDClassifier (fastest overall)")
                return SGDClassifier(
                    loss='hinge',  # SVM loss
                    alpha=self.model_params.get('C', 0.1),
                    random_state=self.model_params.get('random_state', 42),
                    max_iter=self.model_params.get('max_iter', 1000),
                    tol=self.model_params.get('tol', 1e-3)
                )
            else:
                # Manual selection based on kernel
                if self.model_params.get('kernel') == 'linear':
                    from sklearn.svm import LinearSVC
                    print(f"⚡ Using LinearSVC for linear kernel")
                    return LinearSVC(
                        C=self.model_params.get('C', 0.1),
                        random_state=self.model_params.get('random_state', 42),
                        max_iter=self.model_params.get('max_iter', 1000),
                        tol=self.model_params.get('tol', 1e-3)
                    )
        except ImportError:
            pass
        
        # Remove auto_fast from parameters before passing to SVC
        svc_params = {k: v for k, v in self.model_params.items() if k != 'auto_fast'}
        
        # Fallback to standard SVC with optimized parameters
        print(f"💻 Using CPU SVM (scikit-learn) with speed optimizations")
        return SVC(**svc_params)
    
    def _transfer_to_gpu(self, X: np.ndarray) -> Optional[Any]:
        """Transfer data to GPU if available"""
        if not self.gpu_accelerator.gpu_available:
            return None
        
        try:
            if "PyTorch" in self.gpu_accelerator.gpu_library:
                import torch
                if torch.cuda.is_available():
                    X_gpu = torch.tensor(X, dtype=torch.float32, device='cuda')
                    return X_gpu
            elif "CuPy" in self.gpu_accelerator.gpu_library:
                import cupy as cp
                X_gpu = cp.asarray(X, dtype=cp.float32)
                return X_gpu
        except Exception as e:
            print(f"⚠️  GPU transfer failed: {e}")
            return None
        
        return None
    
    def _transfer_from_gpu(self, X_gpu: Any) -> np.ndarray:
        """Transfer data from GPU back to CPU"""
        try:
            if hasattr(X_gpu, 'cpu'):
                return X_gpu.cpu().numpy()
            elif hasattr(X_gpu, 'get'):
                import cupy as cp
                return cp.asnumpy(X_gpu)
        except Exception as e:
            print(f"⚠️  GPU transfer back failed: {e}")
            return None
        
        return None
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'SVMModel':
        """Train the SVM model with detailed logging and GPU support"""
        
        try:
            # Convert input data to proper format if needed
            if isinstance(X, list):
                print(f"🔧 Converting list to numpy array for SVM: {len(X)} samples")
                X = np.array(X)
                print(f"✅ Converted to numpy array: {X.shape}")
            elif not hasattr(X, 'shape'):
                print(f"🔧 Converting unknown type {type(X)} to numpy array")
                X = np.asarray(X)
                print(f"✅ Converted to numpy array: {X.shape}")
            
            if isinstance(y, list):
                print(f"🔧 Converting y list to numpy array: {len(y)} samples")
                y = np.array(y)
                print(f"✅ Converted y to numpy array: {y.shape}")
            elif not hasattr(y, 'shape'):
                print(f"🔧 Converting y type {type(y)} to numpy array")
                y = np.asarray(y)
                print(f"✅ Converted y to numpy array: {y.shape}")
            
            # Get GPU info for logging
            gpu_info = self.gpu_accelerator.get_gpu_info()
            
            # Log training start
            self.training_logger.log_start(X.shape, y.shape, self.model_params, gpu_info)
            
            # Data preparation logging
            start_time = time.time()
            gpu_transfer = False
            
            if sparse.issparse(X):
                print(f"🔧 Converting sparse matrix to dense for SVM training: {X.shape}")
                self.training_logger.log_data_preparation("sparse_matrix")
                X = X.toarray()
                print(f"✅ Converted to dense: {X.shape}")
                conversion_time = time.time() - start_time
                self.training_logger.log_data_preparation("dense_array", conversion_time)
            else:
                print(f"📊 Using dense array for SVM training: {X.shape}")
                self.training_logger.log_data_preparation("dense_array")
            
            # Try GPU transfer if available
            X_gpu = None
            y_gpu = None
            if self.gpu_accelerator.gpu_available and self.gpu_model is not None:
                try:
                    X_gpu = self._transfer_to_gpu(X)
                    if X_gpu is not None:
                        gpu_transfer = True
                        self.training_logger.log_data_preparation("gpu_transfer", None, True)
                except Exception as e:
                    print(f"⚠️  GPU transfer failed, falling back to CPU: {e}")
            
            # Training with progress monitoring
            training_start = time.time()
            gpu_used = False
            
            if X_gpu is not None and self.gpu_model is not None:
                # Train on GPU
                try:
                    if "RAPIDS cuML" in self.gpu_accelerator.gpu_library:
                        # cuML SVM
                        self.gpu_model.fit(X_gpu, y)
                        gpu_used = True
                    else:
                        # Other GPU libraries - transfer back to CPU for training
                        X = self._transfer_from_gpu(X_gpu)
                        if X is None:
                            X = X  # Use original if transfer failed
                        self.model.fit(X, y)
                except Exception as e:
                    print(f"⚠️  GPU training failed, falling back to CPU: {e}")
                    self.model.fit(X, y)
            else:
                # Train on CPU
                self.model.fit(X, y)
            
            training_time = time.time() - training_start
            
            # Log training completion
            if gpu_used:
                n_support_vectors = self.gpu_model.n_support_.sum() if hasattr(self.gpu_model, 'n_support_') else 0
            else:
                n_support_vectors = self.model.n_support_.sum() if hasattr(self.model, 'n_support_') else 0
            
            self.training_logger.log_training_complete(training_time, n_support_vectors, gpu_used)
            
            self.is_fitted = True
            
            # Store training info
            self.training_history.append({
                'action': 'fit',
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y)),
                'training_time': training_time,
                'n_support_vectors': n_support_vectors,
                'gpu_used': gpu_used
            })
            
            return self
            
        except Exception as e:
            self.training_logger.log_error(str(e), "training_error")
            raise e
    
    def predict(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Make predictions using the SVM model with logging and GPU support"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Convert input data to proper format if needed
            if isinstance(X, list):
                print(f"🔧 Converting list to numpy array for SVM prediction: {len(X)} samples")
                X = np.array(X)
                print(f"✅ Converted to numpy array: {X.shape}")
            elif not hasattr(X, 'shape'):
                print(f"🔧 Converting unknown type {type(X)} to numpy array for prediction")
                X = np.asarray(X)
                print(f"✅ Converted to numpy array: {X.shape}")
            
            # Log prediction start
            start_time = time.time()
            gpu_used = False
            
            # Convert sparse to dense if needed
            if sparse.issparse(X):
                print(f"🔧 Converting sparse matrix to dense for SVM prediction: {X.shape}")
                X = X.toarray()
                print(f"✅ Converted to dense: {X.shape}")
            
            # Try GPU prediction if available
            if self.gpu_accelerator.gpu_available and self.gpu_model is not None:
                try:
                    X_gpu = self._transfer_to_gpu(X)
                    if X_gpu is not None:
                        if "RAPIDS cuML" in self.gpu_accelerator.gpu_library:
                            predictions = self.gpu_model.predict(X_gpu)
                            gpu_used = True
                        else:
                            # Transfer back to CPU for prediction
                            X = self._transfer_from_gpu(X_gpu)
                            if X is None:
                                X = X  # Use original if transfer failed
                            predictions = self.model.predict(X)
                    else:
                        predictions = self.model.predict(X)
                except Exception as e:
                    print(f"⚠️  GPU prediction failed, falling back to CPU: {e}")
                    predictions = self.model.predict(X)
            else:
                predictions = self.model.predict(X)
            
            prediction_time = time.time() - start_time
            
            # Log prediction
            self.training_logger.log_prediction(X.shape, prediction_time, gpu_used)
            
            return predictions
            
        except Exception as e:
            self.training_logger.log_error(str(e), "prediction_error")
            raise e
    
    def predict_proba(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Make probability predictions using the SVM model"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Convert input data to proper format if needed
            if isinstance(X, list):
                print(f"🔧 Converting list to numpy array for SVM predict_proba: {len(X)} samples")
                X = np.array(X)
                print(f"✅ Converted to numpy array: {X.shape}")
            elif not hasattr(X, 'shape'):
                print(f"🔧 Converting unknown type {type(X)} to numpy array for predict_proba")
                X = np.asarray(X)
                print(f"✅ Converted to numpy array: {X.shape}")
            
            # Convert sparse to dense if needed
            if sparse.issparse(X):
                print(f"🔧 Converting sparse matrix to dense for SVM predict_proba: {X.shape}")
                X = X.toarray()
                print(f"✅ Converted to dense: {X.shape}")
            
            # Check if model supports probability predictions
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                raise ValueError("Model was not trained with probability=True")
                
        except Exception as e:
            self.training_logger.log_error(str(e), "probability_prediction_error")
            raise e
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the SVM model"""
        
        info = {
            'model_type': 'Support Vector Machine',
            'algorithm': 'SVM',
            'library': 'scikit-learn',
            'parameters': self.model_params.copy(),
            'is_fitted': self.is_fitted,
            'supports_sparse': True,
            'supports_multiclass': True,
            'supports_probability': 'probability' in self.model_params and 
                                  self.model_params.get('probability', False),
            'gpu_acceleration': self.gpu_accelerator.get_gpu_info()
        }
        
        if self.is_fitted:
            if self.gpu_model is not None:
                model_to_check = self.gpu_model
            else:
                model_to_check = self.model
                
            info.update({
                'n_support_vectors': model_to_check.n_support_.sum() if hasattr(model_to_check, 'n_support_') else 0,
                'support_vectors_per_class': model_to_check.n_support_.tolist() if hasattr(model_to_check, 'n_support_') else [],
                'kernel': model_to_check.kernel if hasattr(model_to_check, 'kernel') else None,
                'gamma': model_to_check.gamma if hasattr(model_to_check, 'gamma') else None
            })
        
        return info
    
    def evaluate(self, X: Union[np.ndarray, sparse.csr_matrix], 
                 y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance with logging"""
        
        try:
            # Make predictions
            predictions = self.predict(X)
            
            # Compute metrics
            accuracy = accuracy_score(y, predictions)
            metrics = {'accuracy': accuracy}
            
            # Log evaluation
            self.training_logger.log_evaluation(metrics)
            
            return metrics
            
        except Exception as e:
            self.training_logger.log_error(str(e), "evaluation_error")
            raise e
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance (for linear SVM only)"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if self.model_params.get('kernel') != 'linear':
            raise ValueError("Feature importance only available for linear kernel SVM")
        
        model_to_check = self.gpu_model if self.gpu_model is not None else self.model
        
        if hasattr(model_to_check, 'coef_'):
            return np.abs(model_to_check.coef_[0])
        else:
            raise ValueError("Model coefficients not available")
    
    def get_training_log(self) -> Dict[str, Any]:
        """Get detailed training log"""
        return self.training_logger.get_training_summary()
    
    def print_training_summary(self):
        """Print detailed training summary"""
        self.training_logger.print_summary()
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU acceleration information"""
        return self.gpu_accelerator.get_gpu_info()
    
    def benchmark_gpu_vs_cpu(self, X: np.ndarray, y: np.ndarray, n_runs: int = 3) -> Dict[str, Any]:
        """Benchmark GPU vs CPU performance"""
        if not self.gpu_accelerator.gpu_available:
            return {"error": "No GPU available for benchmarking"}
        
        print("🚀 Benchmarking GPU vs CPU performance...")
        
        # CPU benchmark
        cpu_times = []
        for i in range(n_runs):
            start_time = time.time()
            cpu_model = SVC(**self.model_params)
            cpu_model.fit(X, y)
            cpu_time = time.time() - start_time
            cpu_times.append(cpu_time)
            print(f"   CPU Run {i+1}: {cpu_time:.4f}s")
        
        # GPU benchmark
        gpu_times = []
        if self.gpu_model is not None:
            for i in range(n_runs):
                start_time = time.time()
                try:
                    X_gpu = self._transfer_to_gpu(X)
                    if X_gpu is not None:
                        self.gpu_model.fit(X_gpu, y)
                        gpu_time = time.time() - start_time
                        gpu_times.append(gpu_time)
                        print(f"   GPU Run {i+1}: {gpu_time:.4f}s")
                    else:
                        print(f"   GPU Run {i+1}: Failed (transfer error)")
                except Exception as e:
                    print(f"   GPU Run {i+1}: Failed ({e})")
        
        # Calculate statistics
        cpu_mean = np.mean(cpu_times)
        cpu_std = np.std(cpu_times)
        
        if gpu_times:
            gpu_mean = np.mean(gpu_times)
            gpu_std = np.std(gpu_times)
            speedup = cpu_mean / gpu_mean
        else:
            gpu_mean = gpu_std = speedup = None
        
        benchmark_results = {
            'cpu_times': cpu_times,
            'cpu_mean': cpu_mean,
            'cpu_std': cpu_std,
            'gpu_times': gpu_times,
            'gpu_mean': gpu_mean,
            'gpu_std': gpu_std,
            'speedup': speedup,
            'n_runs': n_runs
        }
        
        print(f"\n📊 Benchmark Results:")
        print(f"   CPU: {cpu_mean:.4f}s ± {cpu_std:.4f}s")
        if gpu_mean:
            print(f"   GPU: {gpu_mean:.4f}s ± {gpu_std:.4f}s")
            print(f"   Speedup: {speedup:.2f}x")
        else:
            print(f"   GPU: Failed")
        
        return benchmark_results
