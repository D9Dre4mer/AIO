"""
K-Nearest Neighbors Classification Model with GridSearchCV Optimization
"""

from typing import Dict, Any, Union, Tuple, List, Optional
import numpy as np
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

from ..base.base_model import BaseModel
from ..base.metrics import ModelMetrics
from config import KNN_N_NEIGHBORS


class KNNModel(BaseModel):
    """K-Nearest Neighbors classification model"""
    
    def __init__(self, n_neighbors: int = KNN_N_NEIGHBORS, weights: str = 'uniform', metric: str = 'euclidean', **kwargs):
        """Initialize KNN model"""
        super().__init__(n_neighbors=n_neighbors, **kwargs)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], 
            y: np.ndarray) -> 'KNNModel':
        """Fit KNN model to training data"""
        
        # Choose algorithm based on data type
        algorithm = 'brute' if sparse.issparse(X) else 'auto'
        
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            algorithm=algorithm
        )
        self.model.fit(X, y)
        
        self.is_fitted = True
        self.training_history.append({
            'action': 'fit',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_neighbors': self.n_neighbors,
            'algorithm': algorithm
        })
        
        return self
    
    def predict(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Make predictions on new data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> None:
        """KNN doesn't provide feature importance"""
        return None
    
    def train_and_test(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train and test KNN model"""
        
        # Fit the model
        self.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Compute metrics
        metrics = ModelMetrics.compute_classification_metrics(y_test, y_pred)
        
        return y_pred, metrics['accuracy'], metrics['classification_report']
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        info = super().get_model_info()
        # Determine algorithm based on model state
        if hasattr(self, 'model') and self.model is not None:
            algorithm = self.model.algorithm
        else:
            algorithm = 'unknown'
            
        info.update({
            'n_neighbors': self.n_neighbors,
            'algorithm': algorithm
        })
        return info
    
    def tune_hyperparameters(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray,
        cv_folds: int = 3,
        scoring: str = 'f1_macro',
        k_range: List[int] = None,
        n_jobs: int = -1,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Tune KNN hyperparameters using GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of CV folds (default: 3)
            scoring: Scoring metric (default: 'f1_macro')
            k_range: List of K values to test (default: [3,5,7,...,31])
            n_jobs: Number of parallel jobs (default: -1 for all)
            verbose: int = 1
            
        Returns:
            Dictionary with tuning results and best model
        """
        # Use provided k_range or default range
        if k_range is None:
            k_range = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
        
        n_samples = X_train.shape[0]  # Handle sparse matrix
        print(f"🔍 Tuning KNN hyperparameters with {cv_folds}-fold CV...")
        print(f"📊 Sample size: {n_samples}, Testing K values: {k_range}")
        
        # Define parameter grid (adaptive based on sample size)
        param_grid = {
            'n_neighbors': k_range,
            'weights': ['uniform', 'distance'],
            'metric': ['cosine', 'euclidean', 'manhattan']
        }
        
        # Choose algorithm based on data type
        algorithm = 'brute' if sparse.issparse(X_train) else 'auto'
        
        # Create base KNN model
        base_knn = KNeighborsClassifier(algorithm=algorithm)
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_knn,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=False
        )
        
        # Fit GridSearchCV
        print(f"🔄 Fitting {len(param_grid['n_neighbors'])} K values × "
              f"{len(param_grid['weights'])} weights × "
              f"{len(param_grid['metric'])} metrics = "
              f"{len(param_grid['n_neighbors']) * len(param_grid['weights']) * len(param_grid['metric'])} combinations...")
        
        grid_search.fit(X_train, y_train)
        
        # Get best parameters and model
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        
        print(f"✅ Best KNN parameters: {best_params}")
        print(f"✅ Best CV score ({scoring}): {best_score:.4f}")
        
        # Update current model with best one
        self.model = best_model
        self.n_neighbors = best_params['n_neighbors']
        self.is_fitted = True
        
        # Store tuning results
        self.tuning_results = {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': grid_search.cv_results_,
            'param_grid': param_grid,
            'cv_folds': cv_folds,
            'scoring': scoring
        }
        
        # Update training history
        self.training_history.append({
            'action': 'hyperparameter_tuning',
            'best_params': best_params,
            'best_score': best_score,
            'cv_folds': cv_folds,
            'scoring': scoring
        })
        
        return self.tuning_results
    
    def determine_optimal_k(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray,
        cv_folds: int = 3,
        scoring: str = 'f1_macro',
        k_range: List[int] = None,
        n_jobs: int = -1,
        verbose: int = 1,
        plot_results: bool = True
    ) -> Dict[str, Any]:
        """
        Determine optimal K value for KNN using GridSearchCV with cosine metric
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of CV folds (default: 3)
            scoring: Scoring metric (default: 'f1_macro')
            k_range: List of K values to test (default: [3,5,7,...,31])
            n_jobs: Number of parallel jobs (default: -1 for all)
            verbose: Verbosity level (default: 1)
            plot_results: Whether to plot benchmark results (default: True)
            
        Returns:
            Dictionary with optimal K results and benchmark data
        """
        # Use provided k_range or default range
        if k_range is None:
            k_range = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
        
        n_samples = X_train.shape[0]  # Handle sparse matrix
        print(f"🎯 Determining optimal K for KNN with {cv_folds}-fold CV...")
        print(f"📊 Sample size: {n_samples}, Testing K values: {k_range}")
        
        # Define parameter grid focused on K values with cosine metric
        param_grid = {
            'n_neighbors': k_range,
            'weights': ['uniform', 'distance']
        }
        
        # Choose algorithm based on data type
        algorithm = 'brute' if sparse.issparse(X_train) else 'auto'
        
        # Create base KNN model with cosine metric
        base_knn = KNeighborsClassifier(metric='cosine', algorithm=algorithm)
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_knn,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=False
        )
        
        # Fit GridSearchCV
        print(f"🔄 Fitting {len(param_grid['n_neighbors'])} K values × "
              f"{len(param_grid['weights'])} weights = "
              f"{len(param_grid['n_neighbors']) * len(param_grid['weights'])} combinations...")
        
        grid_search.fit(X_train, y_train)
        
        # Get best parameters and model
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        
        print(f"✅ Best KNN params for cosine metric: {best_params}")
        print(f"✅ Best CV score ({scoring}): {best_score:.4f}")
        
        # Get benchmark results for each K value
        results = grid_search.cv_results_
        benchmark_results = {}
        
        print(f"\n📊 Benchmark Results for KNN (cosine metric):")
        for k in [3, 5, 7, 9, 11, 13, 15]:
            mask = results['param_n_neighbors'] == k
            if np.any(mask):
                mean_f1 = results['mean_test_score'][mask].mean()
                std_f1 = results['std_test_score'][mask].mean()
                benchmark_results[k] = {
                    'mean_f1': mean_f1,
                    'std_f1': std_f1
                }
                print(f"K = {k}: Mean Macro F1 = {mean_f1:.4f} (± {std_f1:.4f})")
        
        # Plot benchmark F1-scores for K values if requested
        if plot_results:
            self._plot_k_benchmark(results, param_grid)
        
        # Update current model with best one
        self.model = best_model
        self.n_neighbors = best_params['n_neighbors']
        self.is_fitted = True
        
        # Store optimal K results
        self.optimal_k_results = {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': results,
            'param_grid': param_grid,
            'cv_folds': cv_folds,
            'scoring': scoring,
            'benchmark_results': benchmark_results,
            'optimal_k': best_params['n_neighbors']
        }
        
        # Update training history
        self.training_history.append({
            'action': 'optimal_k_determination',
            'best_params': best_params,
            'best_score': best_score,
            'optimal_k': best_params['n_neighbors'],
            'cv_folds': cv_folds,
            'scoring': scoring
        })
        
        return self.optimal_k_results
    
    def _plot_k_benchmark(self, cv_results: Dict[str, Any], param_grid: Dict[str, Any]):
        """Plot benchmark F1-scores for different K values"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Plot for each weight type
            for weight in param_grid['weights']:
                mask = cv_results['param_weights'] == weight
                if np.any(mask):
                    k_values = []
                    scores = []
                    
                    for k in sorted(param_grid['n_neighbors']):
                        k_mask = (cv_results['param_n_neighbors'] == k) & mask
                        if np.any(k_mask):
                            k_values.append(k)
                            scores.append(cv_results['mean_test_score'][k_mask].mean())
                    
                    if k_values and scores:
                        plt.plot(
                            k_values, scores, marker='o', 
                            label=f'Weight = {weight}', 
                            linewidth=2, markersize=8
                        )
            
            plt.xlabel('Number of Neighbors (K)', fontsize=12)
            plt.ylabel('Mean Macro F1 Score', fontsize=12)
            plt.title(
                'KNN Benchmark Performance (Cosine Metric)', 
                fontsize=14, fontweight='bold'
            )
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.xticks(sorted(param_grid['n_neighbors']))
            
            # Add best K marker
            if hasattr(self, 'optimal_k_results'):
                best_k = self.optimal_k_results['optimal_k']
                best_score = self.optimal_k_results['best_score']
                plt.annotate(
                    f'Best K={best_k}\nScore={best_score:.4f}', 
                    xy=(best_k, best_score), 
                    xytext=(best_k+2, best_score+0.02),
                    arrowprops=dict(
                        arrowstyle='->', color='red', lw=2
                    ),
                    fontsize=10, color='red', fontweight='bold'
                )
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Warning: Could not create benchmark plot: {e}")
    
    def get_optimal_k_summary(self) -> Dict[str, Any]:
        """Get summary of optimal K determination results"""
        if not hasattr(self, 'optimal_k_results'):
            return {}
        
        results = self.optimal_k_results
        return {
            'optimal_k': results['optimal_k'],
            'best_score': results['best_score'],
            'best_params': results['best_params'],
            'benchmark_results': results['benchmark_results'],
            'cv_folds': results['cv_folds'],
            'scoring': results['scoring']
        }
    
    def print_optimal_k_summary(self):
        """Print summary of optimal K determination results"""
        if not hasattr(self, 'optimal_k_results'):
            print("⚠️  No optimal K results available. Run determine_optimal_k() first.")
            return
        
        results = self.optimal_k_results
        print(f"\n🎯 KNN Optimal K Summary:")
        print(f"   Optimal K: {results['optimal_k']}")
        print(f"   Best Score ({results['scoring']}): {results['best_score']:.4f}")
        print(f"   Best Parameters: {results['best_params']}")
        print(f"   CV Folds: {results['cv_folds']}")
        
        if results['benchmark_results']:
            print(f"\n📊 K Value Benchmark:")
            for k, metrics in results['benchmark_results'].items():
                print(f"   K = {k}: F1 = {metrics['mean_f1']:.4f} (± {metrics['std_f1']:.4f})")
    
    def get_benchmark_results(self) -> Dict[str, Any]:
        """
        Get benchmark results for different K values
        
        Returns:
            Dictionary with benchmark results
        """
        if not hasattr(self, 'tuning_results'):
            return {}
        
        cv_results = self.tuning_results['cv_results']
        param_grid = self.tuning_results['param_grid']
        
        # Create benchmark table
        benchmark = {}
        for k in param_grid['n_neighbors']:
            k_results = []
            for weight in param_grid['weights']:
                for metric in param_grid['metric']:
                    mask = ((cv_results['param_n_neighbors'] == k) & 
                           (cv_results['param_weights'] == weight) & 
                           (cv_results['param_metric'] == metric))
                    
                    if np.any(mask):
                        mean_score = cv_results['mean_test_score'][mask].mean()
                        std_score = cv_results['std_test_score'][mask].mean()
                        k_results.append({
                            'weights': weight,
                            'metric': metric,
                            'mean_score': mean_score,
                            'std_score': std_score
                        })
            
            if k_results:
                # Get best result for this K
                best_k_result = max(k_results, key=lambda x: x['mean_score'])
                benchmark[f'K={k}'] = best_k_result
        
        return benchmark
    
    def print_benchmark_summary(self):
        """Print benchmark summary for different K values"""
        benchmark = self.get_benchmark_results()
        
        if not benchmark:
            print("⚠️  No benchmark results available. Run tune_hyperparameters() first.")
            return
        
        print(f"\n📊 KNN Benchmark Results (Best per K value):")
        print(f"{'K':<4} {'Weights':<10} {'Metric':<10} {'Mean Score':<12} {'Std':<8}")
        print("-" * 50)
        
        for k_str, result in benchmark.items():
            k = k_str.split('=')[1]
            print(f"{k:<4} {result['weights']:<10} {result['metric']:<10} "
                  f"{result['mean_score']:<12.4f} {result['std_score']:<8.4f}")
        
        # Find overall best K
        best_k = max(benchmark.keys(), 
                    key=lambda x: benchmark[x]['mean_score'])
        best_score = benchmark[best_k]['mean_score']
        
        print(f"\n🏆 Best K: {best_k} with score: {best_score:.4f}")
        print(f"   Parameters: {benchmark[best_k]['weights']} weights, "
              f"{benchmark[best_k]['metric']} metric")
