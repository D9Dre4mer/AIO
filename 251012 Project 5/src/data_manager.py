"""
Data Manager - Load và quản lý datasets từ thư mục data
Tích hợp với MLflow và UI management
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)


class DataManager:
    """Data Manager để load và quản lý datasets từ thư mục data"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.datasets = {}
        self.current_dataset = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def list_available_datasets(self) -> List[Dict[str, Any]]:
        """Liệt kê tất cả datasets có sẵn trong thư mục data"""
        datasets = []
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return datasets
            
        for file_path in self.data_dir.glob("*.csv"):
            try:
                # Load một sample để kiểm tra
                sample_df = pd.read_csv(file_path, nrows=5)
                
                dataset_info = {
                    "name": file_path.stem,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "columns": list(sample_df.columns),
                    "sample_rows": len(sample_df),
                    "file_type": "csv"
                }
                
                # Detect dataset type
                dataset_info["type"] = self._detect_dataset_type(sample_df)
                
                datasets.append(dataset_info)
                
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                
        return datasets
    
    def _detect_dataset_type(self, df: pd.DataFrame) -> str:
        """Detect loại dataset dựa trên columns"""
        columns = [col.lower() for col in df.columns]
        
        if any("text" in col for col in columns) and any("category" in col for col in columns):
            return "text_classification"
        elif any("target" in col for col in columns) or any("label" in col for col in columns):
            return "classification"
        elif any("price" in col for col in columns) or any("value" in col for col in columns):
            return "regression"
        else:
            return "unknown"
    
    def load_dataset(self, dataset_name: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Load dataset từ file"""
        dataset_path = self.data_dir / f"{dataset_name}.csv"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found in {self.data_dir}")
        
        try:
            # Load dataset
            if max_samples:
                df = pd.read_csv(dataset_path, nrows=max_samples)
            else:
                df = pd.read_csv(dataset_path)
            
            # Store dataset info
            dataset_info = {
                "name": dataset_name,
                "data": df,
                "shape": df.shape,
                "columns": list(df.columns),
                "type": self._detect_dataset_type(df),
                "loaded_at": pd.Timestamp.now().isoformat()
            }
            
            self.datasets[dataset_name] = dataset_info
            self.current_dataset = dataset_name
            
            logger.info(f"Loaded dataset {dataset_name}: {df.shape}")
            
            return dataset_info
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get thông tin chi tiết về dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        dataset_info = self.datasets[dataset_name]
        df = dataset_info["data"]
        
        info = {
            "name": dataset_name,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "unique_values": {col: df[col].nunique() for col in df.columns},
            "sample_data": df.head().to_dict(),
            "type": dataset_info["type"]
        }
        
        return info
    
    def prepare_features_and_target(self, dataset_name: str, target_column: str, 
                                  feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Chuẩn bị features và target cho training"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        df = self.datasets[dataset_name]["data"]
        
        # Select features
        if feature_columns is None:
            # Auto-select features (exclude target column)
            feature_columns = [col for col in df.columns if col != target_column]
        
        # Extract features and target
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Handle text data
        if self.datasets[dataset_name]["type"] == "text_classification":
            # For text data, we'll use TF-IDF later
            pass
        
        # Encode target if needed
        if y.dtype == 'object' or len(np.unique(y)) < 20:
            y = self.label_encoder.fit_transform(y)
        
        # Scale features if numeric
        if X.dtype in ['float64', 'int64']:
            X = self.scaler.fit_transform(X)
        
        logger.info(f"Prepared features: {X.shape}, target: {y.shape}")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, val_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data thành train/validation/test"""
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get danh sách models có sẵn"""
        models = [
            {
                "name": "Random Forest",
                "class": "RandomForestClassifier",
                "module": "sklearn.ensemble",
                "description": "Ensemble method with multiple decision trees"
            },
            {
                "name": "Logistic Regression", 
                "class": "LogisticRegression",
                "module": "sklearn.linear_model",
                "description": "Linear classification model"
            },
            {
                "name": "SVM",
                "class": "SVC", 
                "module": "sklearn.svm",
                "description": "Support Vector Machine"
            },
            {
                "name": "KNN",
                "class": "KNeighborsClassifier",
                "module": "sklearn.neighbors", 
                "description": "K-Nearest Neighbors"
            },
            {
                "name": "Decision Tree",
                "class": "DecisionTreeClassifier",
                "module": "sklearn.tree",
                "description": "Single decision tree"
            }
        ]
        
        return models
    
    def log_dataset_to_mlflow(self, dataset_name: str, experiment_name: str = "dataset_exploration"):
        """Log dataset info to MLflow"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        dataset_info = self.get_dataset_info(dataset_name)
        
        with mlflow.start_run(run_name=f"dataset_{dataset_name}"):
            # Log dataset parameters
            mlflow.log_params({
                "dataset_name": dataset_name,
                "dataset_type": dataset_info["type"],
                "n_samples": dataset_info["shape"][0],
                "n_features": dataset_info["shape"][1],
                "missing_values_total": sum(dataset_info["missing_values"].values())
            })
            
            # Log dataset metrics
            mlflow.log_metrics({
                "dataset_size_mb": self.datasets[dataset_name]["data"].memory_usage(deep=True).sum() / 1024 / 1024,
                "avg_unique_values": np.mean(list(dataset_info["unique_values"].values()))
            })
            
            # Log dataset as artifact
            dataset_path = f"artifacts/{dataset_name}_info.json"
            os.makedirs("artifacts", exist_ok=True)
            
            import json
            with open(dataset_path, 'w') as f:
                json.dump(dataset_info, f, indent=2, default=str)
            
            mlflow.log_artifact(dataset_path)
            
            logger.info(f"Logged dataset {dataset_name} to MLflow")


def main():
    """Test DataManager"""
    data_manager = DataManager()
    
    # List available datasets
    datasets = data_manager.list_available_datasets()
    print("Available datasets:")
    for dataset in datasets:
        print(f"- {dataset['name']}: {dataset['type']} ({dataset['size']} bytes)")
    
    if datasets:
        # Load first dataset
        first_dataset = datasets[0]
        dataset_info = data_manager.load_dataset(first_dataset['name'])
        print(f"\nLoaded dataset: {dataset_info['name']}")
        print(f"Shape: {dataset_info['shape']}")
        print(f"Type: {dataset_info['type']}")


if __name__ == "__main__":
    main()
