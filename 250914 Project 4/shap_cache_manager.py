
import os
import pickle
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path

class SHAPCacheManager:
    """Manages SHAP explainer and values caching"""
    
    def __init__(self, cache_dir: str = "cache/shap/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_shap_cache_key(self, model, sample_data, model_name=""):
        """Generate cache key for SHAP explainer and values"""
        try:
            # Create hash from model and sample data
            model_str = str(type(model)) + str(model.get_params() if hasattr(model, 'get_params') else {})
            sample_str = str(sample_data.shape) + str(sample_data.dtype)
            
            # Create hash
            content = f"{model_str}_{sample_str}_{model_name}"
            cache_key = hashlib.md5(content.encode()).hexdigest()
            
            return cache_key
        except Exception as e:
            print(f"Warning: Failed to generate SHAP cache key: {e}")
            return None
    
    def save_shap_cache(self, model, sample_data, explainer, shap_values, model_name=""):
        """Save SHAP values to cache (not explainer due to pickle issues)"""
        try:
            cache_key = self.generate_shap_cache_key(model, sample_data, model_name)
            if not cache_key:
                return None
            
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            # Save only SHAP values and metadata (not explainer due to pickle issues)
            cache_data = {
                'shap_values': shap_values,
                'sample_data': sample_data,
                'model_name': model_name,
                'model_type': str(type(model)),
                'timestamp': pd.Timestamp.now()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"SUCCESS: SHAP cache saved for {model_name} at {cache_file}")
            return cache_file
            
        except Exception as e:
            print(f"Warning: Failed to save SHAP cache: {e}")
            return None
    
    def load_shap_cache(self, model, sample_data, model_name=""):
        """Load SHAP values from cache"""
        try:
            cache_key = self.generate_shap_cache_key(model, sample_data, model_name)
            if not cache_key:
                return None, None
            
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if not cache_file.exists():
                return None, None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            print(f"SUCCESS: SHAP cache loaded for {model_name} from {cache_file}")
            # Return None for explainer (will recreate), and cached SHAP values
            return None, cache_data['shap_values']
            
        except Exception as e:
            print(f"Warning: Failed to load SHAP cache: {e}")
            return None, None
    
    def clear_shap_cache(self, model_name=None):
        """Clear SHAP cache"""
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

# Global SHAP cache manager instance
shap_cache_manager = SHAPCacheManager()
