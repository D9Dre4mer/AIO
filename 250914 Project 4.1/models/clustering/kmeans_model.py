"""
K-Means Clustering Model Implementation
"""

from collections import Counter
from typing import Dict, Any, Union, Tuple, List
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from ..base.base_model import BaseModel
from ..base.metrics import ModelMetrics
from config import (KMEANS_N_CLUSTERS, KMEANS_SVD_THRESHOLD,
                   KMEANS_SVD_COMPONENTS, ENABLE_RAPIDS_CUML,
                   RAPIDS_FALLBACK_TO_CPU, RAPIDS_AUTO_DETECT_DEVICE)

# Import RAPIDS cuML with fallback
try:
    from cuml.cluster import KMeans as cuMLKMeans
    from cuml.common.device_selection import set_global_device_type
    from cuml.common import cuda
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False
    cuMLKMeans = None
    set_global_device_type = None
    cuda = None


class KMeansModel(BaseModel):
    """K-Means clustering model with SVD optimization and optimal K detection"""
    
    def __init__(self, n_clusters: int = KMEANS_N_CLUSTERS, **kwargs):
        """Initialize K-Means model"""
        super().__init__(n_clusters=n_clusters, **kwargs)
        self.n_clusters = n_clusters
        self.svd_model = None
        self.cluster_to_label = {}
        self.optimal_k = None
        self.optimization_results = {}
        
        # RAPIDS cuML configuration
        self.use_rapids = ENABLE_RAPIDS_CUML and RAPIDS_AVAILABLE
        self.device_type = 'cpu'  # Will be determined during fit
        self.rapids_initialized = False
    
    def _initialize_rapids(self) -> bool:
        """Initialize RAPIDS cuML and determine device type"""
        if not self.use_rapids or not RAPIDS_AVAILABLE:
            return False
        
        try:
            if RAPIDS_AUTO_DETECT_DEVICE:
                # Auto-detect best device
                if cuda and cuda.is_available():
                    self.device_type = 'gpu'
                    if set_global_device_type:
                        set_global_device_type('gpu')
                    print("🚀 RAPIDS cuML: Using GPU acceleration")
                else:
                    self.device_type = 'cpu'
                    if set_global_device_type:
                        set_global_device_type('cpu')
                    print("💻 RAPIDS cuML: Using CPU (GPU not available)")
            else:
                # Use CPU by default
                self.device_type = 'cpu'
                if set_global_device_type:
                    set_global_device_type('cpu')
                print("💻 RAPIDS cuML: Using CPU")
            
            self.rapids_initialized = True
            return True
            
        except Exception as e:
            print(f"⚠️ RAPIDS cuML initialization failed: {e}")
            self.device_type = 'cpu'
            self.rapids_initialized = False
            return False
    
    def _create_kmeans_model(self, n_clusters: int, random_state: int = 42):
        """Create appropriate KMeans model (RAPIDS cuML or scikit-learn)"""
        if self.use_rapids and self.rapids_initialized and cuMLKMeans:
            try:
                # Use RAPIDS cuML KMeans
                return cuMLKMeans(
                    n_clusters=n_clusters,
                    random_state=random_state,
                    init='scalable-k-means++'  # RAPIDS cuML default
                )
            except Exception as e:
                print(f"⚠️ RAPIDS cuML KMeans creation failed: {e}")
                if RAPIDS_FALLBACK_TO_CPU:
                    print("🔄 Falling back to scikit-learn KMeans")
                    return KMeans(n_clusters=n_clusters, random_state=random_state)
                else:
                    raise
        else:
            # Use scikit-learn KMeans
            return KMeans(n_clusters=n_clusters, random_state=random_state)
    
    def _convert_to_dense_if_needed(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Convert sparse matrix to dense if using RAPIDS cuML on GPU"""
        if (self.use_rapids and self.device_type == 'gpu' and 
            sparse.issparse(X) and RAPIDS_AVAILABLE):
            print("🔄 Converting sparse matrix to dense for RAPIDS cuML GPU")
            return X.toarray().astype(np.float32)
        elif sparse.issparse(X):
            return X
        else:
            return X.astype(np.float32) if X.dtype != np.float32 else X
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray = None) -> 'KMeansModel':
        """Fit K-Means model with optional SVD preprocessing"""
        
        # Initialize RAPIDS cuML if enabled
        if self.use_rapids and not self.rapids_initialized:
            self._initialize_rapids()
        
        # Handle sparse matrices and high-dimensional data
        X_processed, X_test_placeholder = self._preprocess_data(X)
        
        # Convert to appropriate format for RAPIDS cuML if needed
        X_processed = self._convert_to_dense_if_needed(X_processed)
        
        # Create and fit K-Means model
        self.model = self._create_kmeans_model(self.n_clusters, random_state=42)
        cluster_ids = self.model.fit_predict(X_processed)
        
        # Create cluster to label mapping if labels provided
        if y is not None:
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
        
        # Preprocess data (not training)
        X_processed, _ = self._preprocess_data(X, is_training=False)
        
        # Convert to appropriate format for RAPIDS cuML if needed
        X_processed = self._convert_to_dense_if_needed(X_processed)
        
        # Get cluster predictions
        cluster_ids = self.model.predict(X_processed)
        
        # Map clusters to labels
        y_pred = [self.cluster_to_label.get(cluster_id, cluster_id) 
                  for cluster_id in cluster_ids]
        
        return np.array(y_pred)
    
    def _preprocess_data(self, X: Union[np.ndarray, sparse.csr_matrix], is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data with SVD if needed"""
        
        if sparse.issparse(X):
            print(f"💾 K-Means preprocessing: {X.shape[0]:,} samples × {X.shape[1]:,} features")
            
            if X.shape[1] > KMEANS_SVD_THRESHOLD:
                if is_training:
                    # Training: Create new SVD model
                    print(f"📊 High-dimensional data detected "
                          f"({X.shape[1]:,} > {KMEANS_SVD_THRESHOLD:,} features)")
                    print("🔧 Applying Truncated SVD dimensionality reduction...")
                    
                    n_components = min(KMEANS_SVD_COMPONENTS, 
                                     X.shape[1] - 1, X.shape[0] - 1)
                    self.svd_model = TruncatedSVD(n_components=n_components, 
                                                random_state=42)
                    
                    print(f"⚡ Reducing {X.shape[1]:,} → {n_components:,} dimensions...")
                    
                    # Apply SVD
                    print("🔄 Performing SVD transformation...")
                    X_processed = self.svd_model.fit_transform(X)
                    
                    explained_variance = self.svd_model.explained_variance_ratio_.sum()
                    print(f"✅ SVD completed: {X.shape[1]:,} dimensions | "
                          f"Variance preserved: {explained_variance:.1%}")
                    
                    return X_processed, X_processed
                else:
                    # Prediction: Use existing fitted SVD model
                    if self.svd_model is None:
                        raise ValueError("SVD model not fitted. Please train the model first.")
                    
                    print(f"🔧 Using fitted SVD model for prediction...")
                    print(f"⚡ Reducing {X.shape[1]:,} → {self.svd_model.n_components:,} dimensions...")
                    
                    # Apply fitted SVD
                    X_processed = self.svd_model.transform(X)
                    
                    print(f"✅ SVD transformation completed for prediction")
                    return X_processed, X_processed
            else:
                print(f"✅ Using sparse matrix for K-Means "
                      f"({X.shape[1]:,} features ≤ {KMEANS_SVD_THRESHOLD:,} threshold)")
                return X, X
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
            'cluster_mapping': self.cluster_to_label,
            'rapids_enabled': self.use_rapids,
            'device_type': self.device_type,
            'rapids_available': RAPIDS_AVAILABLE,
            'model_type': 'RAPIDS cuML KMeans' if (self.use_rapids and self.rapids_initialized) else 'scikit-learn KMeans'
        })
        return info

    def find_optimal_k(self, X: Union[np.ndarray, sparse.csr_matrix], 
                       k_range: List[int] = None, 
                       method: str = 'auto') -> Dict[str, Any]:
        """
        Find optimal number of clusters using multiple methods
        
        Args:
            X: Input data
            k_range: Range of K values to test (default: 2 to min(20, n_samples//2))
            method: 'auto', 'elbow', 'silhouette', or 'both'
        
        Returns:
            Dictionary with optimal K and optimization results
        """
        if k_range is None:
            max_k = min(20, X.shape[0] // 2)
            k_range = list(range(2, max_k + 1))
        
        # Preprocess data
        X_processed, _ = self._preprocess_data(X)
        
        results = {
            'k_range': k_range,
            'elbow_scores': {},
            'silhouette_scores': {},
            'recommendations': {}
        }
        
        if method in ['auto', 'elbow', 'both']:
            results['elbow_scores'] = self._elbow_method(X_processed, k_range)
        
        if method in ['auto', 'silhouette', 'both']:
            results['silhouette_scores'] = self._silhouette_analysis(X_processed, k_range)
        
        # Determine optimal K based on method
        if method == 'auto':
            optimal_k = self._auto_select_optimal_k(results)
        elif method == 'elbow':
            optimal_k = self._select_elbow_k(results['elbow_scores'])
        elif method == 'silhouette':
            optimal_k = self._select_silhouette_k(results['silhouette_scores'])
        else:  # 'both'
            optimal_k = self._select_optimal_k_both_methods(results)
        
        results['optimal_k'] = optimal_k
        results['recommendations'] = self._generate_recommendations(results, optimal_k)
        
        # Store results for later use
        self.optimal_k = optimal_k
        self.optimization_results = results
        
        return results
    
    def _elbow_method(self, X: np.ndarray, k_range: List[int]) -> Dict[int, float]:
        """Calculate Within-Cluster Sum of Squares (WCSS) for different K values"""
        wcss_scores = {}
        
        # Convert to appropriate format for RAPIDS cuML if needed
        X_processed = self._convert_to_dense_if_needed(X)
        
        for k in k_range:
            kmeans = self._create_kmeans_model(k, random_state=42)
            kmeans.fit(X_processed)
            wcss_scores[k] = kmeans.inertia_
        
        return wcss_scores
    
    def _silhouette_analysis(self, X: np.ndarray, k_range: List[int]) -> Dict[int, float]:
        """Calculate silhouette scores for different K values"""
        silhouette_scores = {}
        
        # Convert to appropriate format for RAPIDS cuML if needed
        X_processed = self._convert_to_dense_if_needed(X)
        
        for k in k_range:
            if k == 1:  # Silhouette score requires at least 2 clusters
                silhouette_scores[k] = 0.0
                continue
                
            kmeans = self._create_kmeans_model(k, random_state=42)
            cluster_labels = kmeans.fit_predict(X_processed)
            silhouette_scores[k] = silhouette_score(X_processed, cluster_labels)
        
        return silhouette_scores
    
    def _auto_select_optimal_k(self, results: Dict[str, Any]) -> int:
        """Automatically select optimal K using both methods"""
        if not results['elbow_scores'] and not results['silhouette_scores']:
            return self.n_clusters
        
        # If both methods available, use weighted combination
        if results['elbow_scores'] and results['silhouette_scores']:
            elbow_k = self._select_elbow_k(results['elbow_scores'])
            silhouette_k = self._select_silhouette_k(results['silhouette_scores'])
            
            # Weighted average (elbow method gets higher weight for stability)
            optimal_k = int(round(0.6 * elbow_k + 0.4 * silhouette_k))
            
            # Ensure optimal_k is within range
            k_range = results['k_range']
            if optimal_k < min(k_range):
                optimal_k = min(k_range)
            elif optimal_k > max(k_range):
                optimal_k = max(k_range)
            
            return optimal_k
        
        # Fallback to available method
        elif results['elbow_scores']:
            return self._select_elbow_k(results['elbow_scores'])
        else:
            return self._select_silhouette_k(results['silhouette_scores'])
    
    def _select_elbow_k(self, elbow_scores: Dict[int, float]) -> int:
        """Select optimal K using elbow method"""
        if not elbow_scores:
            return self.n_clusters
        
        k_values = list(elbow_scores.keys())
        wcss_values = list(elbow_scores.values())
        
        # Calculate second derivative to find elbow point
        if len(wcss_values) < 3:
            return k_values[0]
        
        # Simple elbow detection: find point with maximum curvature
        second_derivatives = []
        for i in range(1, len(wcss_values) - 1):
            # Second derivative approximation
            second_deriv = wcss_values[i-1] - 2*wcss_values[i] + wcss_values[i+1]
            second_derivatives.append(second_deriv)
        
        if second_derivatives:
            # Find maximum curvature (minimum second derivative)
            max_curvature_idx = np.argmin(second_derivatives)
            optimal_k = k_values[max_curvature_idx + 1]  # +1 because we skipped first element
        else:
            optimal_k = k_values[0]
        
        return optimal_k
    
    def _select_silhouette_k(self, silhouette_scores: Dict[int, float]) -> int:
        """Select optimal K using silhouette score"""
        if not silhouette_scores:
            return self.n_clusters
        
        # Find K with highest silhouette score
        optimal_k = max(silhouette_scores, key=silhouette_scores.get)
        return optimal_k
    
    def _select_optimal_k_both_methods(self, results: Dict[str, Any]) -> int:
        """Select optimal K when both methods are available"""
        return self._auto_select_optimal_k(results)
    
    def _generate_recommendations(self, results: Dict[str, Any], optimal_k: int) -> Dict[str, Any]:
        """Generate recommendations based on optimization results"""
        recommendations = {
            'optimal_k': optimal_k,
            'confidence': 'medium',
            'method_agreement': 'unknown',
            'notes': []
        }
        
        if results['elbow_scores'] and results['silhouette_scores']:
            elbow_k = self._select_elbow_k(results['elbow_scores'])
            silhouette_k = self._select_silhouette_k(results['silhouette_scores'])
            
            # Check if methods agree
            if abs(elbow_k - silhouette_k) <= 1:
                recommendations['method_agreement'] = 'high'
                recommendations['confidence'] = 'high'
                recommendations['notes'].append(f"Both methods suggest similar K: Elbow={elbow_k}, Silhouette={silhouette_k}")
            else:
                recommendations['method_agreement'] = 'low'
                recommendations['confidence'] = 'medium'
                recommendations['notes'].append(f"Methods disagree: Elbow={elbow_k}, Silhouette={silhouette_k}")
        
        # Add interpretation notes
        if 'silhouette_scores' in results and optimal_k in results['silhouette_scores']:
            silhouette_score = results['silhouette_scores'][optimal_k]
            if silhouette_score > 0.7:
                recommendations['notes'].append(f"Excellent clustering structure (Silhouette={silhouette_score:.3f})")
            elif silhouette_score > 0.5:
                recommendations['notes'].append(f"Good clustering structure (Silhouette={silhouette_score:.3f})")
            elif silhouette_score > 0.25:
                recommendations['notes'].append(f"Fair clustering structure (Silhouette={silhouette_score:.3f})")
            else:
                recommendations['notes'].append(f"Weak clustering structure (Silhouette={silhouette_score:.3f})")
        
        return recommendations
    
    def plot_optimization_results(self, save_path: str = None) -> plt.Figure:
        """Plot optimization results for visual analysis"""
        if not hasattr(self, 'optimization_results') or not self.optimization_results:
            # Try to find optimal K if not already done
            if hasattr(self, 'optimal_k') and self.optimal_k:
                # Create basic results structure for plotting
                self.optimization_results = {
                    'optimal_k': self.optimal_k,
                    'elbow_scores': {},
                    'silhouette_scores': {}
                }
            else:
                raise ValueError("No optimization results available. "
                               "Run find_optimal_k() first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elbow plot
        if self.optimization_results['elbow_scores']:
            k_values = list(self.optimization_results['elbow_scores'].keys())
            wcss_values = list(self.optimization_results['elbow_scores'].values())
            
            ax1.plot(k_values, wcss_values, 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('Number of Clusters (K)')
            ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
            ax1.set_title('Elbow Method for Optimal K Selection')
            ax1.grid(True, alpha=0.3)
            
                    # Highlight optimal K
        if 'optimal_k' in self.optimization_results:
            optimal_k = self.optimization_results['optimal_k']
            if optimal_k in wcss_values:
                optimal_idx = k_values.index(optimal_k)
                ax1.plot(optimal_k, wcss_values[optimal_idx], 'ro', 
                         markersize=12, label=f'Optimal K={optimal_k}')
                ax1.legend()
        
        # Silhouette plot
        if self.optimization_results['silhouette_scores']:
            k_values = list(self.optimization_results['silhouette_scores'].keys())
            silhouette_values = list(self.optimization_results['silhouette_scores'].values())
            
            ax2.plot(k_values, silhouette_values, 'go-', linewidth=2, markersize=8)
            ax2.set_xlabel('Number of Clusters (K)')
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title('Silhouette Analysis for Optimal K Selection')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(-0.1, 1.1)
            
            # Highlight optimal K
            if 'optimal_k' in self.optimization_results:
                optimal_k = self.optimization_results['optimal_k']
                if optimal_k in silhouette_values:
                    optimal_idx = k_values.index(optimal_k)
                    ax2.plot(optimal_k, silhouette_values[optimal_idx], 'ro', markersize=12, label=f'Optimal K={optimal_k}')
                    ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_optimal_k(self) -> int:
        """Get the optimal number of clusters found by optimization"""
        return self.optimal_k if self.optimal_k else self.n_clusters
    
    def get_optimization_results(self) -> Dict[str, Any]:
        """Get detailed optimization results"""
        return self.optimization_results.copy()
