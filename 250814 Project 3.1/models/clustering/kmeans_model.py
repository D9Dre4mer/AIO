"""
K-Means Clustering Model Implementation
"""

from collections import Counter
from typing import Dict, Any, Union, Tuple
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from ..base.base_model import BaseModel
from ..base.metrics import ModelMetrics
from config import KMEANS_N_CLUSTERS, KMEANS_SVD_THRESHOLD, KMEANS_SVD_COMPONENTS


class KMeansModel(BaseModel):
    """K-Means clustering model with SVD optimization"""
    
    def __init__(self, n_clusters: int = KMEANS_N_CLUSTERS, **kwargs):
        """Initialize K-Means model"""
        super().__init__(n_clusters=n_clusters, **kwargs)
        self.n_clusters = n_clusters
        self.svd_model = None
        self.cluster_to_label = {}
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'KMeansModel':
        """Fit K-Means model with optional SVD preprocessing"""
        
        # Handle sparse matrices and high-dimensional data
        X_processed, X_test_placeholder = self._preprocess_data(X)
        
        # Fit K-Means
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_ids = self.model.fit_predict(X_processed)
        
        # Create cluster to label mapping
        self._create_cluster_mapping(cluster_ids, y)
        
        self.is_fitted = True
        self.training_history.append({
            'action': 'fit',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_clusters': self.n_clusters
        })
        
        return self
    
    def predict(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Predict cluster labels for new data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Preprocess data
        X_processed, _ = self._preprocess_data(X)
        
        # Get cluster predictions
        cluster_ids = self.model.predict(X_processed)
        
        # Map clusters to labels
        y_pred = [self.cluster_to_label.get(cluster_id, cluster_id) 
                  for cluster_id in cluster_ids]
        
        return np.array(y_pred)
    
    def _preprocess_data(self, X: Union[np.ndarray, sparse.csr_matrix]) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data with SVD if needed"""
        
        if sparse.issparse(X):
            print(f"💾 K-Means preprocessing: {X.shape[0]:,} samples × {X.shape[1]:,} features")
            
            if X.shape[1] > KMEANS_SVD_THRESHOLD:
                print(f"📊 High-dimensional data detected ({X.shape[1]:,} > {KMEANS_SVD_THRESHOLD:,} features)")
                print(f"🔧 Applying Truncated SVD dimensionality reduction...")
                
                n_components = min(KMEANS_SVD_COMPONENTS, X.shape[1] - 1, X.shape[0] - 1)
                self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
                
                from tqdm import tqdm
                print(f"⚡ Reducing {X.shape[1]:,} → {n_components:,} dimensions...")
                
                # Apply SVD
                print("🔄 Performing SVD transformation...")
                X_processed = self.svd_model.fit_transform(
                    tqdm(X, desc="SVD transformation", unit="samples", total=X.shape[0])
                )
                
                explained_variance = self.svd_model.explained_variance_ratio_.sum()
                print(f"✅ SVD completed: {X.shape[1]:,} dimensions | "
                      f"Variance preserved: {explained_variance:.1%}")
                
                return X_processed, X_processed  # Placeholder for test data
            else:
                print(f"✅ Converting to dense matrix ({X.shape[1]:,} features ≤ {KMEANS_SVD_THRESHOLD:,} threshold)")
                return X.toarray(), X.toarray()
        else:
            return X, X
    
    def _create_cluster_mapping(self, cluster_ids: np.ndarray, y: np.ndarray) -> None:
        """Create mapping from cluster IDs to true labels"""
        for cluster_id in set(cluster_ids):
            labels_in_cluster = [
                y[i] for i in range(len(y)) 
                if cluster_ids[i] == cluster_id
            ]
            if labels_in_cluster:
                most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
                self.cluster_to_label[cluster_id] = most_common_label
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing cluster centers")
        return self.model.cluster_centers_
    
    def get_n_clusters(self) -> int:
        """Get number of clusters"""
        return self.n_clusters
    
    def train_and_test(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train and test K-Means model"""
        
        # Fit the model
        self.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Compute metrics
        metrics = ModelMetrics.compute_clustering_metrics(y_test, y_pred)
        
        return y_pred, metrics['accuracy'], metrics['classification_report']
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        info = super().get_model_info()
        info.update({
            'n_clusters': self.n_clusters,
            'has_svd': self.svd_model is not None,
            'cluster_mapping': self.cluster_to_label
        })
        return info
