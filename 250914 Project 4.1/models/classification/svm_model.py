"""
Support Vector Machine Model Implementation with Clean CPU-Only Training
"""

from typing import Union, Dict, Any
import numpy as np
from scipy import sparse
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
import time
import logging

from ..base.base_model import BaseModel


class SVMTrainingLogger:
    """Clean logger for SVM training process"""
    
    def __init__(self, model_name: str = "SVM"):
        self.model_name = model_name
        self.training_log = []
        self.start_time = None
        self.end_time = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"{model_name}_Training")
        
    def log_start(self, X_shape: tuple, y_shape: tuple, parameters: Dict):
        """Log training start"""
        self.start_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate memory usage
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
            'memory_usage': f"{memory_mb:.2f} MB"
        }
        
        self.training_log.append(log_entry)
        self.logger.info(f"🚀 Starting {self.model_name} training at {timestamp}")
        
        # Handle both tuple and array for y_shape
        if (hasattr(y_shape, '__len__') and 
                not isinstance(y_shape, (str, bytes))):
            if len(y_shape) == 1:  # y_shape is (n_samples,)
                n_classes = len(np.unique(y_shape))
            else:  # y_shape is (n_samples, n_features)
                n_classes = 1  # Default for regression-like data
        else:
            n_classes = 1
            
        self.logger.info(
            f"📊 Data: {X_shape[0]:,} samples, "
            f"{X_shape[1]:,} features, {n_classes} classes"
        )
        self.logger.info(f"⚙️  Parameters: {parameters}")
        self.logger.info(f"💾 Estimated memory: {log_entry['memory_usage']}")
        self.logger.info("💻 Using CPU (scikit-learn) - Fast and reliable")
        
    def log_data_preparation(self, X_type: str, conversion_time: float = None):
        """Log data preparation steps"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            'timestamp': timestamp,
            'action': 'data_preparation',
            'data_type': X_type,
            'conversion_time': conversion_time
        }
        
        self.training_log.append(log_entry)
        self.logger.info(f"🔧 Data preparation: {X_type}")
        if conversion_time:
            self.logger.info(f"⏱️  Conversion time: {conversion_time:.4f}s")
            
    def log_training_progress(self, iteration: int = None, n_support_vectors: int = None):
        """Log training progress"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            'timestamp': timestamp,
            'action': 'training_progress',
            'iteration': iteration,
            'n_support_vectors': n_support_vectors
        }
        
        self.training_log.append(log_entry)
        if iteration:
            self.logger.info(f"🔄 Training iteration: {iteration}")
        if n_support_vectors:
            self.logger.info(f"🎯 Support vectors found: {n_support_vectors}")
            
    def log_training_complete(self, training_time: float, n_support_vectors: int):
        """Log training completion"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            'timestamp': timestamp,
            'action': 'training_complete',
            'training_time': training_time,
            'n_support_vectors': n_support_vectors
        }
        
        self.training_log.append(log_entry)
        self.logger.info(f"✅ Training completed on CPU in {training_time:.4f}s")
        self.logger.info(f"🎯 Total support vectors: {n_support_vectors}")
        
    def log_prediction(self, X_shape: tuple, prediction_time: float):
        """Log prediction process"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            'timestamp': timestamp,
            'action': 'prediction',
            'X_shape': X_shape,
            'prediction_time': prediction_time
        }
        
        self.training_log.append(log_entry)
        self.logger.info(f"🔮 Prediction on {X_shape[0]:,} samples using CPU in {prediction_time:.4f}s")
        
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
            return {}
            
        summary = {
            'model_name': self.model_name,
            'training_log': self.training_log.copy(),
            'total_entries': len(self.training_log)
        }
        
        if self.end_time:
            summary['total_time'] = self.end_time - self.start_time
            
        return summary


class SVMModel(BaseModel):
    """Clean SVM model with CPU-only training and optimized algorithms"""
    
    def __init__(self, **kwargs):
        """Initialize SVM model with clean parameters"""
        super().__init__(**kwargs)
        
        # Clean model parameters
        self.model_params = {
            'C': kwargs.get('C', 1.0),
            'kernel': kwargs.get('kernel', 'rbf'),
            'gamma': kwargs.get('gamma', 'scale'),
            'random_state': kwargs.get('random_state', 42),
            'max_iter': kwargs.get('max_iter', 1000),
            'tol': kwargs.get('tol', 1e-3),
            'auto_fast': kwargs.get('auto_fast', True)
        }
        
        # Initialize components
        self.model = None
        self.training_logger = SVMTrainingLogger("SVM")
        
        # Create optimized model
        self.model = self._create_optimized_svm()
        
    def _create_optimized_svm(self):
        """Create the most efficient SVM model for given parameters"""
        
        # Auto-select fastest algorithm if enabled
        if self.model_params.get('auto_fast', True):
            # For linear kernel, LinearSVC is fastest
            if self.model_params.get('kernel') == 'linear':
                print(f"⚡ Using LinearSVC (fastest for linear kernel)")
                return LinearSVC(
                    C=self.model_params.get('C', 1.0),
                    random_state=self.model_params.get('random_state', 42),
                    max_iter=self.model_params.get('max_iter', 1000),
                    tol=self.model_params.get('tol', 1e-3)
                )
            
            # For other kernels, try SGDClassifier first
            try:
                print(f"⚡ Using SGDClassifier (fastest overall)")
                return SGDClassifier(
                    loss='hinge',  # SVM loss
                    alpha=1.0 / self.model_params.get('C', 1.0),
                    random_state=self.model_params.get('random_state', 42),
                    max_iter=self.model_params.get('max_iter', 1000),
                    tol=self.model_params.get('tol', 1e-3)
                )
            except ImportError:
                pass
        
        # Fallback to standard SVC with optimized parameters
        print(f"💻 Using CPU SVC (scikit-learn) with speed optimizations")
        return SVC(**self.model_params)
    
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'SVMModel':
        """Train the SVM model with clean CPU-only training"""
        
        try:
            # Convert input data to proper format if needed
            if isinstance(X, list):
                print(f"🔧 Converting list to numpy array for SVM: {X.shape[0] if hasattr(X, 'shape') else (X.getnnz() if hasattr(X, 'getnnz') else len(X))} samples")
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
            
            # Log training start
            self.training_logger.log_start(X.shape, y.shape, self.model_params)
            
            # Data preparation logging
            start_time = time.time()
            
            if sparse.issparse(X):
                print(f"📊 Using sparse matrix for SVM training: {X.shape}")
                self.training_logger.log_data_preparation("sparse_matrix")
                # Keep sparse matrix - sklearn SVM handles sparse matrices efficiently
                # No conversion to prevent memory overflow
            else:
                print(f"📊 Using dense array for SVM training: {X.shape}")
                self.training_logger.log_data_preparation("dense_array")
            
            # Training with progress monitoring
            training_start = time.time()
            
            # Train on CPU (clean and honest)
            self.model.fit(X, y)
            
            training_time = time.time() - training_start
            
            # Get number of support vectors
            if hasattr(self.model, 'n_support_'):
                n_support_vectors = self.model.n_support_.sum()
            elif hasattr(self.model, 'support_vectors_'):
                n_support_vectors = len(self.model.support_vectors_)
            else:
                n_support_vectors = 0
            
            # Log training completion
            self.training_logger.log_training_complete(training_time, n_support_vectors)
            
            self.is_fitted = True
            
            # Store training info
            self.training_history.append({
                'action': 'fit',
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y)),
                'training_time': training_time,
                'n_support_vectors': n_support_vectors,
                'gpu_used': False  # Always false - clean and honest
            })
            
            return self
            
        except Exception as e:
            self.training_logger.log_error(str(e), "training_error")
            raise e
    
    def predict(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Make predictions using the SVM model"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Keep sparse matrix for SVM prediction
            if isinstance(X, list):
                X = np.array(X)
            elif sparse.issparse(X):
                print("🔧 Using sparse matrix for SVM prediction")
                # SVM supports sparse matrices directly
            
            # Make predictions
            prediction_start = time.time()
            y_pred = self.model.predict(X)
            prediction_time = time.time() - prediction_start
            
            # Log prediction
            self.training_logger.log_prediction(X.shape, prediction_time)
            
            return y_pred
            
        except Exception as e:
            self.training_logger.log_error(str(e), "prediction_error")
            raise e
    
    def predict_proba(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Get prediction probabilities if available"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Keep sparse matrix for SVM prediction
            if isinstance(X, list):
                X = np.array(X)
            elif sparse.issparse(X):
                print("🔧 Using sparse matrix for SVM prediction")
                # SVM supports sparse matrices directly
            
            # Check if model supports predict_proba
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                print("⚠️  This SVM model doesn't support probability predictions")
                return None
                
        except Exception as e:
            self.training_logger.log_error(str(e), "prediction_proba_error")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        info = super().get_model_info()
        info.update({
            'model_type': 'SVM',
            'algorithm': type(self.model).__name__,
            'parameters': self.model_params,
            'gpu_support': False,  # Clean and honest
            'training_logger': self.training_logger.get_training_summary()
        })
        return info
    
    def get_support_vectors(self) -> np.ndarray:
        """Get support vectors if available"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing support vectors")
        
        if hasattr(self.model, 'support_vectors_'):
            return self.model.support_vectors_
        else:
            print("⚠️  This SVM model doesn't provide support vectors")
            return None
    
    def get_n_support_vectors(self) -> int:
        """Get number of support vectors"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing support vectors")
        
        if hasattr(self.model, 'n_support_'):
            return self.model.n_support_.sum()
        elif hasattr(self.model, 'support_vectors_'):
            return len(self.model.support_vectors_)
        else:
            return 0
