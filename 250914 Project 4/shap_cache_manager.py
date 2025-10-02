
import os
import pickle
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
import threading
import gc
import psutil
import time
from contextlib import contextmanager

class SHAPCacheManager:
    """Manages SHAP explainer and values caching with memory leak protection"""
    
    def __init__(self, cache_dir: str = "cache/shap/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory leak protection
        self._lock = threading.Lock()
        self._active_explainers = {}  # Track active explainers
        self._max_memory_mb = 1000  # Max memory usage in MB
        self._max_sample_size = 500  # Max sample size to prevent memory issues
        
        # Cleanup old cache files on init
        self._cleanup_old_cache()
    
    def _cleanup_old_cache(self):
        """Cleanup cache files older than 7 days"""
        try:
            current_time = time.time()
            cutoff_time = current_time - (7 * 24 * 60 * 60)  # 7 days
            
            for cache_file in self.cache_dir.glob("*.pkl"):
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
        except Exception as e:
            print(f"Warning: Failed to cleanup old cache: {e}")
    
    def _check_memory_usage(self):
        """Check if memory usage is within limits"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb < self._max_memory_mb
        except Exception:
            return True  # If can't check, assume OK
    
    def _cleanup_explainers(self):
        """Cleanup active explainers to prevent memory leaks"""
        with self._lock:
            if len(self._active_explainers) > 5:  # Keep max 5 explainers
                # Remove oldest explainer
                oldest_key = min(self._active_explainers.keys())
                del self._active_explainers[oldest_key]
                gc.collect()  # Force garbage collection
    
    @contextmanager
    def _memory_safe_operation(self):
        """Context manager for memory-safe operations"""
        try:
            # Check memory before operation
            if not self._check_memory_usage():
                print("Warning: High memory usage detected, skipping SHAP operation")
                yield False
                return
            
            yield True
        finally:
            # Cleanup after operation
            gc.collect()
    
    def generate_shap_cache_key(self, model, sample_data, model_name=""):
        """Generate cache key for SHAP explainer and values with safety checks"""
        try:
            # Create hash from model and sample data (with safety checks)
            try:
                model_params = model.get_params() if hasattr(model, 'get_params') else {}
            except Exception:
                model_params = {}
            
            model_str = str(type(model)) + str(model_params)
            sample_str = str(sample_data.shape) + str(sample_data.dtype)
            
            # Create hash
            content = f"{model_str}_{sample_str}_{model_name}"
            cache_key = hashlib.md5(content.encode()).hexdigest()
            
            return cache_key
        except Exception as e:
            print(f"Warning: Failed to generate SHAP cache key: {e}")
            return None
    
    def save_shap_cache(self, model, sample_data, explainer, shap_values, model_name="", feature_names=None):
        """Save SHAP values to cache with memory leak protection and concurrent access safety"""
        with self._memory_safe_operation() as safe:
            if not safe:
                return None
                
            with self._lock:  # Thread-safe file operations
                try:
                    # Safety check: Limit sample size
                    if len(sample_data) > self._max_sample_size:
                        print(f"Warning: Sample size {len(sample_data)} exceeds limit {self._max_sample_size}, truncating")
                        sample_data = sample_data[:self._max_sample_size]
                        shap_values = shap_values[:self._max_sample_size]
                    
                    # Convert SHAP values to numpy array if it's a list
                    if isinstance(shap_values, list):
                        try:
                            import numpy as np
                            shap_values = np.array(shap_values)
                            print(f"INFO: Converted SHAP values from list to numpy array: {shap_values.shape}")
                        except Exception as e:
                            print(f"Warning: Failed to convert SHAP values from list to numpy array: {e}")
                            return None
                    
                    cache_key = self.generate_shap_cache_key(model, sample_data, model_name)
                    if not cache_key:
                        return None
                    
                    cache_file = self.cache_dir / f"{cache_key}.pkl"
                    
                    # Save SHAP values and comprehensive metadata
                    cache_data = {
                        'shap_values': shap_values,
                        'sample_data': sample_data,
                        'model_name': model_name,
                        'model_type': str(type(model)),
                        'feature_names': feature_names,  # Add feature names
                        'model_params': model.get_params() if hasattr(model, 'get_params') else {},  # Add model params
                        'sample_shape': sample_data.shape,  # Add sample shape
                        'shap_shape': shap_values.shape if hasattr(shap_values, 'shape') else 'unknown',  # Add SHAP shape
                        'timestamp': pd.Timestamp.now(),
                        'sample_size': len(sample_data),
                        'cache_version': '1.1'  # Add version for future compatibility
                    }
                    
                    # Atomic file write with temp file
                    temp_file = cache_file.with_suffix('.tmp')
                    with open(temp_file, 'wb') as f:
                        pickle.dump(cache_data, f)
                    
                    # Atomic move
                    temp_file.replace(cache_file)
                    
                    print(f"SUCCESS: SHAP cache saved for {model_name} at {cache_file}")
                    return cache_file
                    
                except Exception as e:
                    print(f"Warning: Failed to save SHAP cache: {e}")
                    return None
    
    def load_shap_cache(self, model, sample_data, model_name=""):
        """Load SHAP values from cache with concurrent access safety"""
        with self._lock:  # Thread-safe file operations
            try:
                cache_key = self.generate_shap_cache_key(model, sample_data, model_name)
                if not cache_key:
                    return None, None
                
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                
                if not cache_file.exists():
                    return None, None
                
                # Check file age (avoid corrupted files)
                file_age = time.time() - cache_file.stat().st_mtime
                if file_age > 24 * 60 * 60:  # Older than 24 hours
                    print(f"Warning: Cache file {cache_file} is older than 24 hours, removing")
                    cache_file.unlink()
                    return None, None
                
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                print(f"SUCCESS: SHAP cache loaded for {model_name} from {cache_file}")
                # Return explainer (None), SHAP values, and metadata
                return None, cache_data['shap_values'], cache_data
                
            except Exception as e:
                print(f"Warning: Failed to load SHAP cache: {e}")
                return None, None
    
    def clear_shap_cache(self, model_name=None):
        """Clear SHAP cache with thread safety"""
        with self._lock:  # Thread-safe file operations
            try:
                cleared_count = 0
                
                for cache_file in self.cache_dir.glob("*.pkl"):
                    if model_name:
                        # Load cache to check model name
                        try:
                            with open(cache_file, 'rb') as f:
                                cache_data = pickle.load(f)
                            if cache_data.get('model_name') != model_name:
                                continue
                        except:
                            continue
                    
                    cache_file.unlink()
                    cleared_count += 1
                
                print(f"SUCCESS: Cleared {cleared_count} SHAP cache entries")
                return cleared_count
                
            except Exception as e:
                print(f"Warning: Failed to clear SHAP cache: {e}")
                return 0
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0
    
    def find_shap_cache(self, model, sample_data, model_name=""):
        """Find SHAP cache with multiple possible keys"""
        possible_keys = [
            model_name,
            f"{model_name}_StandardScaler",
            f"{model_name}_MinMaxScaler", 
            f"{model_name}_RobustScaler",
            f"{model_name}_None"
        ]
        
        for key in possible_keys:
            cache_key = self.generate_shap_cache_key(model, sample_data, key)
            if cache_key:
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                if cache_file.exists():
                    print(f"SUCCESS: Found SHAP cache with key '{key}' -> {cache_file}")
                    return self.load_shap_cache(model, sample_data, key)
        
        print(f"INFO: No SHAP cache found for any of these keys: {possible_keys}")
        return None, None, None
    
    def cleanup_all(self):
        """Cleanup all resources and force garbage collection"""
        with self._lock:
            self._active_explainers.clear()
            gc.collect()
            print("SUCCESS: SHAP cache manager cleanup completed")

# Global SHAP cache manager instance
shap_cache_manager = SHAPCacheManager()
