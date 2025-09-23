"""
RAPIDS cuML GPU Detection Utility
Kiểm tra khả năng sử dụng RAPIDS cuML trên GPU và CPU
"""

from typing import Dict, Any
import numpy as np


class RapidsDetector:
    """Detector for RAPIDS cuML availability and GPU support"""
    
    def __init__(self):
        self._gpu_available = None
        self._cuml_available = None
        self._device_info = {}
        self._test_data = None
        
    def detect_rapids_availability(self) -> Dict[str, Any]:
        """
        Detect RAPIDS cuML availability and GPU support
        
        Returns:
            Dictionary with detection results
        """
        results = {
            'cuml_available': False,
            'gpu_available': False,
            'device_type': 'cpu',
            'error_message': None,
            'version_info': {},
            'test_passed': False
        }
        
        # Test 1: Check if cuML can be imported
        try:
            import cuml
            results['cuml_available'] = True
            results['version_info']['cuml_version'] = getattr(cuml, '__version__', 'unknown')
        except ImportError as e:
            results['error_message'] = f"cuML not available: {e}"
            return results
        
        # Test 2: Check GPU availability
        try:
            from cuml.common import cuda
            
            # Check if CUDA is available
            if cuda.is_available():
                results['gpu_available'] = True
                results['device_type'] = 'gpu'
                results['version_info']['cuda_available'] = True
                
                # Get GPU info
                try:
                    gpu_count = cuda.gpu_count()
                    results['version_info']['gpu_count'] = gpu_count
                    results['version_info']['gpu_memory'] = []
                    
                    for i in range(gpu_count):
                        try:
                            memory_info = cuda.gpu_memory_info(i)
                            results['version_info']['gpu_memory'].append({
                                'device_id': i,
                                'total_memory': memory_info.total,
                                'free_memory': memory_info.free
                            })
                        except Exception as e:
                            results['version_info']['gpu_memory'].append({
                                'device_id': i,
                                'error': str(e)
                            })
                except Exception as e:
                    results['version_info']['gpu_info_error'] = str(e)
            else:
                results['gpu_available'] = False
                results['device_type'] = 'cpu'
                results['version_info']['cuda_available'] = False
                
        except ImportError as e:
            results['error_message'] = f"cuML GPU detection failed: {e}"
            results['device_type'] = 'cpu'
        
        # Test 3: Test basic functionality
        try:
            test_result = self._test_cuml_functionality()
            results['test_passed'] = test_result
        except Exception as e:
            results['error_message'] = f"cuML functionality test failed: {e}"
        
        return results
    
    def _test_cuml_functionality(self) -> bool:
        """Test basic cuML functionality with small dataset"""
        try:
            from cuml.cluster import KMeans
            
            # Create small test data
            np.random.seed(42)
            X_test = np.random.rand(100, 10).astype(np.float32)
            
            # Test KMeans
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(X_test)
            predictions = kmeans.predict(X_test)
            
            # Basic validation
            assert len(predictions) == 100
            assert len(set(predictions)) <= 3
            assert hasattr(kmeans, 'cluster_centers_')
            
            return True
            
        except Exception as e:
            print(f"⚠️ cuML functionality test failed: {e}")
            return False
    
    def get_optimal_device_config(self) -> Dict[str, Any]:
        """
        Get optimal device configuration for the current system
        
        Returns:
            Dictionary with recommended configuration
        """
        detection_results = self.detect_rapids_availability()
        
        config = {
            'use_gpu': False,
            'use_cuml': False,
            'fallback_to_cpu': True,
            'device_type': 'cpu',
            'recommendations': []
        }
        
        if not detection_results['cuml_available']:
            config['recommendations'].append("Install cuML: conda install -c rapidsai -c nvidia -c conda-forge cuml=24.08")
            return config
        
        if detection_results['gpu_available'] and detection_results['test_passed']:
            config['use_gpu'] = True
            config['use_cuml'] = True
            config['device_type'] = 'gpu'
            config['fallback_to_cpu'] = True
            config['recommendations'].append("GPU detected and working - using RAPIDS cuML on GPU")
        else:
            config['use_cuml'] = True
            config['device_type'] = 'cpu'
            config['fallback_to_cpu'] = True
            config['recommendations'].append("Using RAPIDS cuML on CPU (GPU not available or not working)")
        
        return config
    
    def create_test_data(self, n_samples: int = 1000, n_features: int = 50) -> np.ndarray:
        """Create test data for benchmarking"""
        if self._test_data is None:
            np.random.seed(42)
            self._test_data = np.random.rand(n_samples, n_features).astype(np.float32)
        return self._test_data
    
    def benchmark_kmeans(self, n_clusters: int = 5, n_samples: int = 1000, n_features: int = 50) -> Dict[str, Any]:
        """
        Benchmark KMeans performance on current system
        
        Returns:
            Dictionary with benchmark results
        """
        import time
        
        # Create test data
        X = self.create_test_data(n_samples, n_features)
        
        results = {
            'device_type': 'unknown',
            'cuml_time': None,
            'sklearn_time': None,
            'speedup': None,
            'error': None
        }
        
        # Test cuML KMeans
        try:
            from cuml.cluster import KMeans as cuMLKMeans
            
            start_time = time.time()
            cuml_kmeans = cuMLKMeans(n_clusters=n_clusters, random_state=42)
            cuml_kmeans.fit(X)
            cuml_predictions = cuml_kmeans.predict(X)
            cuml_time = time.time() - start_time
            
            results['cuml_time'] = cuml_time
            results['device_type'] = 'gpu' if self.detect_rapids_availability()['gpu_available'] else 'cpu'
            
        except Exception as e:
            results['error'] = f"cuML KMeans failed: {e}"
            return results
        
        # Test scikit-learn KMeans for comparison
        try:
            from sklearn.cluster import KMeans as SklearnKMeans
            
            start_time = time.time()
            sklearn_kmeans = SklearnKMeans(n_clusters=n_clusters, random_state=42)
            sklearn_kmeans.fit(X)
            sklearn_predictions = sklearn_kmeans.predict(X)
            sklearn_time = time.time() - start_time
            
            results['sklearn_time'] = sklearn_time
            
            if sklearn_time > 0:
                results['speedup'] = sklearn_time / cuml_time
                
        except Exception as e:
            results['error'] = f"scikit-learn KMeans failed: {e}"
        
        return results

# Global instance for easy access
rapids_detector = RapidsDetector()

def get_rapids_info() -> Dict[str, Any]:
    """Get RAPIDS cuML information and recommendations"""
    return rapids_detector.detect_rapids_availability()

def get_optimal_config() -> Dict[str, Any]:
    """Get optimal configuration for current system"""
    return rapids_detector.get_optimal_device_config()

def benchmark_kmeans_performance(n_clusters: int = 5, n_samples: int = 1000, n_features: int = 50) -> Dict[str, Any]:
    """Benchmark KMeans performance"""
    return rapids_detector.benchmark_kmeans(n_clusters, n_samples, n_features)
