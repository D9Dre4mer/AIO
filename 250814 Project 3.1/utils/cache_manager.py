# =========================================
# Cache Manager for Academic Paper Classification System
# Handles dataset caching and resource management
# =========================================

import json
from pathlib import Path
from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages caching for datasets and resources.
    Automatically downloads and caches datasets if they don't exist.
    """

    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cached resources
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.cache_dir / "datasets").mkdir(exist_ok=True)
        (self.cache_dir / "models").mkdir(exist_ok=True)
        (self.cache_dir / "features").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)
        
        # Cache metadata file
        self.metadata_file = self.cache_dir / "metadata" / "cache_info.json"
        self._load_cache_metadata()

    def _load_cache_metadata(self):
        """Load cache metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.cache_info = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self.cache_info = {}
        else:
            self.cache_info = {}

    def _save_cache_metadata(self):
        """Save cache metadata to file."""
        try:
            self.metadata_file.parent.mkdir(exist_ok=True)
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def get_dataset_path(self, dataset_name: str) -> Path:
        """
        Get the path where dataset should be cached.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Path to cached dataset
        """
        # Create a safe filename from dataset name
        safe_name = dataset_name.replace("/", "_").replace("\\", "_")
        return self.cache_dir / "datasets" / f"{safe_name}"

    def is_dataset_cached(self, dataset_name: str) -> bool:
        """
        Check if dataset is already cached.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            True if dataset is cached, False otherwise
        """
        dataset_path = self.get_dataset_path(dataset_name)
        
        # Check if basic cache structure exists
        if not dataset_path.exists():
            return False
            
        # Check for HuggingFace datasets cache structure
        if (dataset_path / "default").exists():
            # Look for actual data files in the cache
            data_files = list(dataset_path.rglob("*.arrow"))
            return len(data_files) > 0
            
        # Check for our custom cache format
        if (dataset_path / "dataset_info.json").exists():
            return True
            
        return False

    def _is_hf_datasets_cache(self, dataset_path: Path) -> bool:
        """
        Check if the cache path contains HuggingFace datasets cache structure.
        
        Args:
            dataset_path: Path to check
            
        Returns:
            True if it's a HuggingFace datasets cache
        """
        return (dataset_path / "default").exists()
    
    def _get_hf_cache_path(self, dataset_path: Path) -> Optional[Path]:
        """
        Get the actual data path within HuggingFace datasets cache.
        
        Args:
            dataset_path: Base cache path
            
        Returns:
            Path to actual data files or None
        """
        default_path = dataset_path / "default"
        if not default_path.exists():
            return None
            
        # Look for version directories
        version_dirs = [d for d in default_path.iterdir() if d.is_dir()]
        if not version_dirs:
            return None
            
        # Get the latest version (usually the first one)
        latest_version = version_dirs[0]
        
        # Look for hash directories
        hash_dirs = [d for d in latest_version.iterdir() if d.is_dir()]
        if not hash_dirs:
            return None
            
        return hash_dirs[0]

    def get_cached_dataset(self, dataset_name: str) -> Optional[Dataset]:
        """
        Load cached dataset if it exists.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Cached dataset or None if not found
        """
        if not self.is_dataset_cached(dataset_name):
            logger.warning(f"Dataset {dataset_name} is not cached")
            return None
        
        try:
            dataset_path = self.get_dataset_path(dataset_name)
            logger.info(f"Loading cached dataset from: {dataset_path}")
            
            # Check if this is a HuggingFace datasets cache
            if self._is_hf_datasets_cache(dataset_path):
                # This is a HuggingFace datasets cache, load using the 
                # original dataset name
                logger.info(f"Loading HuggingFace datasets cache: {dataset_name}")
                return load_dataset(
                    dataset_name, 
                    cache_dir=str(self.cache_dir / "datasets")
                )
            else:
                # Try to load as saved dataset
                logger.info(f"Loading custom cache format from: {dataset_path}")
                return load_dataset("json", data_dir=str(dataset_path))
                
        except Exception as e:
            logger.error(
                f"Failed to load cached dataset {dataset_name}: {e}"
            )
            logger.error(f"Cache path: {dataset_path}")
            
            # Get cache structure info
            if dataset_path.exists():
                cache_structure = list(dataset_path.iterdir())
            else:
                cache_structure = 'Path does not exist'
            
            logger.error(
                f"Cache structure: {cache_structure}"
            )
            return None

    def cache_dataset(self, dataset: Dataset, dataset_name: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Cache a dataset to disk.
        
        Args:
            dataset: Dataset to cache
            dataset_name: Name of the dataset
            metadata: Additional metadata to store
            
        Returns:
            True if caching successful, False otherwise
        """
        try:
            dataset_path = self.get_dataset_path(dataset_name)
            
            # Save dataset in parquet format (more efficient than JSON)
            dataset.save_to_disk(str(dataset_path))
            
            # Store metadata
            features = (list(dataset.features.keys()) 
                       if hasattr(dataset, 'features') else [])
            split_names = (list(dataset.keys()) 
                          if hasattr(dataset, 'keys') else [])
            
            cache_info = {
                "dataset_name": dataset_name,
                "cached_at": str(Path().cwd()),
                "dataset_size": len(dataset),
                "features": features,
                "split_names": split_names,
                "metadata": metadata or {}
            }
            
            self.cache_info[dataset_name] = cache_info
            self._save_cache_metadata()
            
            logger.info(f"Dataset cached successfully to: {dataset_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache dataset: {e}")
            return False

    def load_or_download_dataset(self, dataset_name: str, 
                                cache_dir: Optional[str] = None,
                                **kwargs) -> Dataset:
        """
        Load dataset from cache if available, otherwise download and cache it.
        
        Args:
            dataset_name: Name of the dataset (e.g., "UniverseTBD/arxiv-abstracts-large")
            cache_dir: Custom cache directory for this dataset
            **kwargs: Additional arguments for load_dataset
            
        Returns:
            Loaded dataset
        """
        # Check if dataset is already cached
        if self.is_dataset_cached(dataset_name):
            logger.info(f"Loading dataset from cache: {dataset_name}")
            cached_dataset = self.get_cached_dataset(dataset_name)
            if cached_dataset is not None:
                return cached_dataset
        
        # Download dataset if not cached
        logger.info(f"Downloading dataset: {dataset_name}")
        
        # Use custom cache directory if provided
        if cache_dir:
            kwargs['cache_dir'] = cache_dir
        else:
            kwargs['cache_dir'] = str(self.cache_dir / "datasets")
        
        try:
            dataset = load_dataset(dataset_name, **kwargs)
            
            # Cache the downloaded dataset
            if isinstance(dataset, dict):
                # If dataset has multiple splits, cache the first one
                first_split = list(dataset.values())[0]
                self.cache_dataset(first_split, dataset_name)
            else:
                self.cache_dataset(dataset, dataset_name)
            
            logger.info(f"Dataset downloaded and cached successfully: {dataset_name}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_name}: {e}")
            raise

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached resources.
        
        Returns:
            Dictionary with cache information
        """
        return {
            "cache_directory": str(self.cache_dir),
            "total_datasets": len(self.cache_info),
            "cached_datasets": list(self.cache_info.keys()),
            "cache_size_mb": self._get_cache_size_mb(),
            "cache_info": self.cache_info
        }

    def _get_cache_size_mb(self) -> float:
        """Calculate total cache size in MB."""
        try:
            total_size = 0
            for path in self.cache_dir.rglob('*'):
                if path.is_file():
                    total_size += path.stat().st_size
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0

    def clear_cache(self, dataset_name: Optional[str] = None):
        """
        Clear cache for specific dataset or entire cache.
        
        Args:
            dataset_name: Name of dataset to clear, or None to clear all
        """
        if dataset_name is None:
            # Clear entire cache
            import shutil
            try:
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
                self.cache_info = {}
                self._save_cache_metadata()
                logger.info("Entire cache cleared successfully")
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
        else:
            # Clear specific dataset
            dataset_path = self.get_dataset_path(dataset_name)
            if dataset_path.exists():
                import shutil
                try:
                    shutil.rmtree(dataset_path)
                    if dataset_name in self.cache_info:
                        del self.cache_info[dataset_name]
                    self._save_cache_metadata()
                    logger.info(f"Cache cleared for dataset: {dataset_name}")
                except Exception as e:
                    logger.error(f"Failed to clear cache for {dataset_name}: {e}")

    def get_dataset_stats(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a cached dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset statistics or None if not found
        """
        if dataset_name not in self.cache_info:
            return None
        
        return self.cache_info[dataset_name]

    def list_cached_datasets(self) -> list:
        """
        List all cached datasets.
        
        Returns:
            List of cached dataset names
        """
        return list(self.cache_info.keys())


# Global cache manager instance
cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.
    
    Returns:
        CacheManager instance
    """
    return cache_manager


# Convenience functions
def load_cached_dataset(dataset_name: str, **kwargs) -> Dataset:
    """
    Load dataset from cache or download if not available.
    
    Args:
        dataset_name: Name of the dataset
        **kwargs: Additional arguments for load_dataset
        
    Returns:
        Loaded dataset
    """
    return cache_manager.load_or_download_dataset(dataset_name, **kwargs)


def is_dataset_cached(dataset_name: str) -> bool:
    """
    Check if dataset is cached.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        True if dataset is cached
    """
    return cache_manager.is_dataset_cached(dataset_name)


def get_cache_info() -> Dict[str, Any]:
    """
    Get cache information.
    
    Returns:
        Cache information dictionary
    """
    return cache_manager.get_cache_info()
