"""
Data Loading and Preprocessing Module

This module handles data loading, preprocessing, and validation for the LightGBM optimization project.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import yaml
from pathlib import Path
import gdown
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """
    Advanced data loader with preprocessing capabilities
    
    Features:
    - Automatic dataset download from Google Drive
    - Data validation and cleaning
    - Memory optimization
    - Categorical encoding
    - Train/validation/test splitting
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataLoader with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data = {}
        self.scalers = {}
        self.encoders = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def download_datasets(self, force_download: bool = False) -> None:
        """
        Download datasets from Google Drive if not present
        
        Args:
            force_download: Force download even if files exist
        """
        # Google Drive folder URL
        folder_url = "https://drive.google.com/drive/folders/1cMoqIDEgGYDVzv8B7cKp3csxujQ4OFp7"
        
        # Check if data directory exists
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Check if files already exist
        required_files = [
            "fe_train.csv", "fe_val.csv", "fe_test.csv",
            "raw_train.csv", "raw_val.csv", "raw_test.csv",
            "dt_train.csv", "dt_val.csv", "dt_test.csv",
            "fe_dt_train.csv", "fe_dt_val.csv", "fe_dt_test.csv"
        ]
        
        files_exist = all((data_dir / file).exists() for file in required_files)
        
        if not files_exist or force_download:
            print("📥 Downloading datasets from Google Drive...")
            try:
                gdown.download_folder(folder_url, output="data", quiet=False, use_cookies=False)
                print("✅ Datasets downloaded successfully!")
            except Exception as e:
                print(f"❌ Error downloading datasets: {e}")
                print("💡 Please ensure you have internet connection and gdown installed")
        else:
            print("✅ Datasets already exist, skipping download")
    
    def load_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load specific dataset with train/val/test splits
        
        Args:
            dataset_name: Name of dataset to load ('raw', 'fe', 'dt', 'fe_dt')
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        data_dir = Path("data")
        
        # Define file paths based on dataset name
        file_mapping = {
            'raw': ('raw_train.csv', 'raw_val.csv', 'raw_test.csv'),
            'fe': ('fe_train.csv', 'fe_val.csv', 'fe_test.csv'),
            'dt': ('dt_train.csv', 'dt_val.csv', 'dt_test.csv'),
            'fe_dt': ('fe_dt_train.csv', 'fe_dt_val.csv', 'fe_dt_test.csv')
        }
        
        if dataset_name not in file_mapping:
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {list(file_mapping.keys())}")
        
        train_file, val_file, test_file = file_mapping[dataset_name]
        
        # Load datasets
        print(f"📊 Loading {dataset_name} dataset...")
        
        train_df = pd.read_csv(data_dir / train_file)
        val_df = pd.read_csv(data_dir / val_file)
        test_df = pd.read_csv(data_dir / test_file)
        
        # Extract features and target
        target_col = self.config['data']['target_column']
        
        X_train = train_df.drop(target_col, axis=1)
        y_train = train_df[target_col]
        X_val = val_df.drop(target_col, axis=1)
        y_val = val_df[target_col]
        X_test = test_df.drop(target_col, axis=1)
        y_test = test_df[target_col]
        
        # Data validation
        self._validate_data(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Memory optimization
        X_train = self._optimize_memory(X_train)
        X_val = self._optimize_memory(X_val)
        X_test = self._optimize_memory(X_test)
        
        print(f"✅ {dataset_name} dataset loaded successfully!")
        print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print(f"   Class distribution: {y_train.value_counts(normalize=True).round(3).to_dict()}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _validate_data(self, X_train, y_train, X_val, y_val, X_test, y_test) -> None:
        """Validate data integrity and consistency"""
        # Check for missing values
        for name, data in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
            if data.isnull().any().any():
                print(f"⚠️  Warning: Missing values found in {name}")
        
        # Check for infinite values
        for name, data in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
            if np.isinf(data.select_dtypes(include=[np.number])).any().any():
                print(f"⚠️  Warning: Infinite values found in {name}")
        
        # Check target consistency
        unique_train = set(y_train.unique())
        unique_val = set(y_val.unique())
        unique_test = set(y_test.unique())
        
        if not unique_train == unique_val == unique_test:
            print("⚠️  Warning: Inconsistent target classes across splits")
        
        # Check feature consistency
        if not (X_train.columns.equals(X_val.columns) and X_val.columns.equals(X_test.columns)):
            print("⚠️  Warning: Inconsistent feature columns across splits")
    
    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage by converting data types"""
        df_optimized = df.copy()
        
        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype
            
            if col_type != 'object':
                c_min = df_optimized[col].min()
                c_max = df_optimized[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df_optimized[col] = df_optimized[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df_optimized[col] = df_optimized[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df_optimized[col] = df_optimized[col].astype(np.int32)
                    else:
                        df_optimized[col] = df_optimized[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df_optimized[col] = df_optimized[col].astype(np.float32)
                    else:
                        df_optimized[col] = df_optimized[col].astype(np.float64)
            else:
                # Convert object columns to category if they have few unique values
                if df_optimized[col].nunique() < 0.5 * len(df_optimized):
                    df_optimized[col] = df_optimized[col].astype('category')
        
        return df_optimized
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a dataset
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            Dictionary with dataset information
        """
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_dataset(dataset_name)
        
        info = {
            'name': dataset_name,
            'train_shape': X_train.shape,
            'val_shape': X_val.shape,
            'test_shape': X_test.shape,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0] + X_val.shape[0] + X_test.shape[0],
            'class_distribution': y_train.value_counts(normalize=True).to_dict(),
            'feature_types': {
                'numeric': X_train.select_dtypes(include=[np.number]).shape[1],
                'categorical': X_train.select_dtypes(include=['category', 'object']).shape[1]
            },
            'missing_values': X_train.isnull().sum().sum(),
            'memory_usage': X_train.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        return info
    
    def create_custom_split(self, X: pd.DataFrame, y: pd.Series, 
                          test_size: float = 0.2, val_size: float = 0.2,
                          stratify: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Create custom train/validation/test split
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            val_size: Proportion of validation set
            stratify: Whether to stratify based on target
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        stratify_param = y if stratify else None
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config['data']['random_state'],
            stratify=stratify_param
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        stratify_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.config['data']['random_state'],
            stratify=stratify_temp
        )
        
        return X_train, y_train, X_val, y_val, X_test, y_test
