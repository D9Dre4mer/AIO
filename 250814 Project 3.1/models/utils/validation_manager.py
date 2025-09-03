"""
Unified Validation Manager for handling both train/validation/test splits and cross-validation
"""

from typing import Tuple, Dict, Any, Union, List
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ValidationManager:
    """Unified manager for data splitting and cross-validation"""
    
    def __init__(self, test_size: float = 0.2, validation_size: float = 0.2, 
                 random_state: int = 42, cv_folds: int = 5, cv_stratified: bool = True):
        """Initialize validation manager
        
        Args:
            test_size: Proportion of data for test set (default: 0.2)
            validation_size: Proportion of remaining data for validation (default: 0.2)
            random_state: Random seed for reproducibility
            cv_folds: Number of folds for cross-validation (default: 5)
            cv_stratified: Whether to use stratified sampling for CV (default: True)
        """
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.cv_stratified = cv_stratified
        
        # Initialize KFold splitter for cross-validation
        if cv_stratified:
            self.kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            self.kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # ==================== TRAIN/VALIDATION/TEST SPLIT METHODS ====================
    
    def split_data(
        self, 
        X: Union[np.ndarray, sparse.csr_matrix], 
        y: np.ndarray,
        stratify: np.ndarray = None
    ) -> Tuple[Union[np.ndarray, sparse.csr_matrix], 
               Union[np.ndarray, sparse.csr_matrix], 
               Union[np.ndarray, sparse.csr_matrix], 
               np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/validation/test sets
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        
        total_samples = X.shape[0]
        
        # Calculate exact sizes
        test_samples = int(total_samples * self.test_size)
        remaining_samples = total_samples - test_samples
        val_samples = int(remaining_samples * self.validation_size)
        train_samples = remaining_samples - val_samples
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_samples, 
            random_state=self.random_state,
            stratify=None  # Don't stratify for exact size splits
        )
        
        # Second split: separate validation from remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_samples,
            random_state=self.random_state,
            stratify=None  # Don't stratify for exact size splits
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def split_data_train_test(
        self, 
        X: Union[np.ndarray, sparse.csr_matrix], 
        y: np.ndarray,
        test_size: float = None,
        stratify: np.ndarray = None
    ) -> Tuple[Union[np.ndarray, sparse.csr_matrix], 
               Union[np.ndarray, sparse.csr_matrix], 
               np.ndarray, np.ndarray]:
        """Split data into train/test sets only (validation handled by CV)
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        
        # Use provided test_size or default
        if test_size is None:
            test_size = self.test_size
        
        # Split data into train/test only
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_split_info(
        self, 
        X: Union[np.ndarray, sparse.csr_matrix], 
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Get information about data splits"""
        
        total_samples = X.shape[0]
        test_samples = int(total_samples * self.test_size)
        remaining_samples = total_samples - test_samples
        val_samples = int(remaining_samples * self.validation_size)
        train_samples = remaining_samples - val_samples
        
        return {
            'total_samples': total_samples,
            'train_samples': train_samples,
            'validation_samples': val_samples,
            'test_samples': test_samples,
            'train_ratio': train_samples / total_samples,
            'validation_ratio': val_samples / total_samples,
            'test_ratio': test_samples / total_samples,
            'split_config': {
                'test_size': self.test_size,
                'validation_size': self.validation_size,
                'random_state': self.random_state
            }
        }
    
    def print_split_summary(self, split_info: Dict[str, Any]) -> None:
        """Print a formatted summary of data splits"""
        
        print("\n📊 Data Split Summary")
        print("=" * 40)
        print(f"Total samples: {split_info['total_samples']:,}")
        print(f"Train set:     {split_info['train_samples']:,} ({split_info['train_ratio']:.1%})")
        print(f"Validation set: {split_info['validation_samples']:,} ({split_info['validation_ratio']:.1%})")
        print(f"Test set:       {split_info['test_samples']:,} ({split_info['test_ratio']:.1%})")
        print("=" * 40)
    
    # ==================== CROSS-VALIDATION METHODS ====================
    
    def cross_validate_with_precomputed_embeddings(
        self,
        model,
        cv_embeddings: Dict[str, Any],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Perform cross-validation using pre-computed embeddings for fair comparison
        
        Args:
            model: Model instance to evaluate
            cv_embeddings: Pre-computed embeddings for each fold
            metrics: List of metrics to compute
            
        Returns:
            Dictionary containing CV results
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Initialize results storage
        fold_scores = {metric: [] for metric in metrics}
        all_predictions = []
        all_true_labels = []
        
        # Perform cross-validation using pre-computed embeddings
        for fold in range(1, self.cv_folds + 1):
            print(f"  📊 Fold {fold}/{self.cv_folds}")
            
            fold_data = cv_embeddings[f'fold_{fold}']
            X_train = fold_data['X_train']
            X_val = fold_data['X_val']
            y_train = fold_data['y_train']
            y_val = fold_data['y_val']
            
            # GPU Optimization: Convert sparse matrices to dense arrays if needed
            if hasattr(X_train, 'toarray'):  # Sparse matrix
                print(f"   🔄 Converting sparse matrix to dense for GPU acceleration in CV fold {fold}...")
                X_train = X_train.toarray()
                X_val = X_val.toarray()
                print(f"   ✅ Sparse matrix converted to dense arrays for GPU")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Store predictions and true labels
            all_predictions.extend(y_pred)
            all_true_labels.extend(y_val)
            
            # Calculate metrics for this fold
            for metric in metrics:
                if metric == 'accuracy':
                    # Calculate validation accuracy (model on validation data)
                    val_score = accuracy_score(y_val, y_pred)
                    # Calculate training accuracy (model on training data)
                    y_train_pred = model.predict(X_train)
                    train_score = accuracy_score(y_train, y_train_pred)
                    
                    # Store both scores for overfitting detection
                    if 'train_accuracy' not in fold_scores:
                        fold_scores['train_accuracy'] = []
                        fold_scores['validation_accuracy'] = []
                    fold_scores['train_accuracy'].append(train_score)
                    fold_scores['validation_accuracy'].append(val_score)
                    
                    # Store validation accuracy in regular accuracy list
                    score = val_score
                elif metric == 'precision':
                    score = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                elif metric == 'recall':
                    score = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                elif metric == 'f1':
                    score = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                else:
                    print(f"⚠️ Unknown metric: {metric}")
                    continue
                
                fold_scores[metric].append(score)
        
        # Calculate overall statistics
        cv_results = {}
        for metric in metrics:
            if fold_scores[metric]:  # Only process if we have scores
                scores = np.array(fold_scores[metric])
                cv_results[f'{metric}_scores'] = scores
                cv_results[f'{metric}_mean'] = np.mean(scores)
                cv_results[f'{metric}_std'] = np.std(scores)
        
        # Add additional info
        cv_results['cv_folds'] = self.cv_folds
        cv_results['all_predictions'] = all_predictions
        cv_results['all_true_labels'] = all_true_labels
        
        # Add cv_config for compatibility with print_cv_summary
        cv_results['cv_config'] = {
            'n_folds': self.cv_folds,
            'stratified': self.cv_stratified,
            'shuffle': True
        }
        
        # Add fold_results for compatibility with print_cv_summary
        cv_results['fold_results'] = []
        for fold in range(1, self.cv_folds + 1):
            fold_data = cv_embeddings[f'fold_{fold}']
            fold_result = {
                'fold': fold,
                'accuracy': fold_scores['accuracy'][fold-1] if fold_scores['accuracy'] else 0,
                'n_train': fold_data['y_train'].shape[0] if hasattr(fold_data['y_train'], 'shape') else len(fold_data['y_train']),
                'n_val': fold_data['y_val'].shape[0] if hasattr(fold_data['y_val'], 'shape') else len(fold_data['y_val']),
                'n_test': fold_data['y_test'].shape[0] if 'y_test' in fold_data and fold_data['y_test'] is not None and hasattr(fold_data['y_test'], 'shape') else (len(fold_data['y_test']) if 'y_test' in fold_data and fold_data['y_test'] is not None else 0),  # ← THÊM SỐ SAMPLE TEST
                # Use real training vs validation accuracy for overfitting detection
                'train_accuracy': fold_scores['train_accuracy'][fold-1] if 'train_accuracy' in fold_scores and fold_scores['train_accuracy'] else 0,
                'validation_accuracy': fold_scores['validation_accuracy'][fold-1] if 'validation_accuracy' in fold_scores and fold_scores['validation_accuracy'] else 0,
                # Add sample counts for debugging and analysis
                'n_train_samples': fold_data.get('n_train_samples', fold_data['y_train'].shape[0] if hasattr(fold_data['y_train'], 'shape') else len(fold_data['y_train'])),
                'n_val_samples': fold_data.get('n_val_samples', fold_data['y_val'].shape[0] if hasattr(fold_data['y_val'], 'shape') else len(fold_data['y_val'])),
                'n_test_samples': fold_data.get('n_test_samples', fold_data['y_test'].shape[0] if 'y_test' in fold_data and fold_data['y_test'] is not None and hasattr(fold_data['y_test'], 'shape') else (len(fold_data['y_test']) if 'y_test' in fold_data and fold_data['y_test'] is not None else 0))
            }
            cv_results['fold_results'].append(fold_result)
        
        # Add overall_results for compatibility with print_cv_summary
        cv_results['overall_results'] = {}
        for metric in metrics:
            if fold_scores[metric]:
                scores = np.array(fold_scores[metric])
                cv_results['overall_results'][f'{metric}_mean'] = np.mean(scores)
                cv_results['overall_results'][f'{metric}_std'] = np.std(scores)
                cv_results['overall_results'][f'{metric}_min'] = np.min(scores)
                cv_results['overall_results'][f'{metric}_max'] = np.max(scores)
        
        return cv_results
    
    def evaluate_test_data_from_cv_cache(self, model, cv_embeddings: Dict[str, Any], 
                                       metrics: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate model on test data using pre-computed CV embeddings cache
        
        Args:
            model: Trained model to evaluate
            cv_embeddings: CV embeddings cache containing test data
            metrics: List of metrics to compute
            
        Returns:
            Dictionary with test evaluation results
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Get test data from first fold (all folds should have same test data)
        first_fold = cv_embeddings[f'fold_1']
        if 'X_test' not in first_fold or 'y_test' not in first_fold:
            print("⚠️ Test data not found in CV cache")
            return {}
        
        X_test = first_fold['X_test']
        y_test = first_fold['y_test']
        
        if X_test is None or y_test is None:
            print("⚠️ Test data is None in CV cache")
            print("   💡 This is normal for some embedding types - using CV validation results instead")
            return {}
        
        print(f"  📊 Test data: {X_test.shape if hasattr(X_test, 'shape') else len(X_test)} samples")
        
        # Make predictions on test data
        y_test_pred = model.predict(X_test)
        
        # Calculate test metrics
        test_results = {}
        for metric in metrics:
            if metric == 'accuracy':
                score = accuracy_score(y_test, y_test_pred)
            elif metric == 'precision':
                score = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            elif metric == 'f1':
                score = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            else:
                print(f"⚠️ Unknown metric: {metric}")
                continue
            
            test_results[f'test_{metric}'] = score
        
        # Add additional information
        test_results['test_samples'] = len(y_test)
        test_results['test_predictions'] = y_test_pred
        test_results['test_true_labels'] = y_test
        
        print(f"✅ Test evaluation completed: {test_results}")
        return test_results

    def cross_validate_model(
        self,
        model,
        X: Union[np.ndarray, sparse.csr_matrix],
        y: np.ndarray,
        metrics: List[str] = None,
        is_embeddings: bool = False,
        texts: List[str] = None,
        vectorizer = None
    ) -> Dict[str, Any]:
        """Perform K-Fold Cross-Validation on a model
        
        Args:
            model: Model instance with fit() and predict() methods
            X: Feature matrix
            y: Target vector
            metrics: List of metrics to compute (default: ['accuracy'])
            
        Returns:
            Dictionary with cross-validation results
        """
        if metrics is None:
            metrics = ['accuracy']
        
        # Initialize results storage
        fold_results = []
        all_predictions = []
        all_true_labels = []
        
        # Ensure X and y are properly formatted for CV
        # Handle sparse matrices and ensure proper dimensions
        if hasattr(X, 'shape'):
            X_array = X
        else:
            X_array = np.asarray(X)
            
        if hasattr(y, 'shape'):
            y_array = y
        else:
            y_array = np.asarray(y)
        
        print(f"🔍 Data shapes: X={X_array.shape}, y={y_array.shape}")
        print(f"🔍 Data types: X={type(X_array)}, y={type(y_array)}")
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(X_array, y_array), 1):
            print(f"  📊 Fold {fold}/{self.cv_folds}")
            
            # Handle embeddings differently to prevent data leakage
            if is_embeddings and texts is not None and vectorizer is not None:
                # For embeddings: create new vectorizer instance for each fold
                # to prevent data leakage between folds
                fold_texts_train = [texts[i] for i in train_idx]
                fold_texts_val = [texts[i] for i in val_idx]
                
                # Create new embedding vectorizer for this fold
                fold_vectorizer = vectorizer.fit_embeddings_only(fold_texts_train)
                
                # Transform texts to embeddings using fold-specific vectorizer
                X_train = np.array(fold_vectorizer.transform_with_progress(fold_texts_train, stop_callback=None))
                X_val = np.array(fold_vectorizer.transform_with_progress(fold_texts_val, stop_callback=None))
                
                y_train = y_array[train_idx]
                y_val = y_array[val_idx]
            else:
                # Standard processing for BoW/TF-IDF or pre-computed features
                # Split data for this fold - handle sparse matrices properly
                if hasattr(X_array, 'toarray'):  # Sparse matrix
                    # MEMORY OPTIMIZATION: Keep sparse matrices for memory efficiency
                    print(f"   📊 Using sparse matrix format for memory efficiency in CV fold {fold}")
                    X_train = X_array[train_idx]  # Keep sparse
                    X_val = X_array[val_idx]      # Keep sparse
                else:  # Dense array
                    X_train = X_array[train_idx]
                    X_val = X_array[val_idx]
                    
                y_train = y_array[train_idx]
                y_val = y_array[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Store predictions and true labels
            all_predictions.extend(y_pred)
            all_true_labels.extend(y_val)
            
            # Compute metrics for this fold
            fold_metrics = self._compute_metrics(y_val, y_pred, metrics)
            fold_metrics['fold'] = fold
            fold_metrics['n_train'] = X_train.shape[0]
            fold_metrics['n_val'] = X_val.shape[0]
            
            # Add training accuracy for overfitting detection
            y_train_pred = model.predict(X_train)
            fold_metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
            fold_metrics['validation_accuracy'] = fold_metrics['accuracy']  # Rename for clarity
            
            fold_results.append(fold_metrics)
            
            print(f"    ✅ Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, "
                  f"Acc: {fold_metrics['accuracy']:.4f}")
        
        # Compute overall statistics
        overall_results = self._compute_overall_statistics(
            fold_results, all_true_labels, all_predictions, metrics
        )
        
        return {
            'fold_results': fold_results,
            'overall_results': overall_results,
            'all_predictions': np.array(all_predictions),
            'all_true_labels': np.array(all_true_labels),
            'cv_config': {
                'n_folds': self.cv_folds,
                'shuffle': True,
                'random_state': self.random_state,
                'stratified': self.cv_stratified
            }
        }
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: List[str]
    ) -> Dict[str, float]:
        """Compute specified metrics for predictions"""
        results = {}
        
        for metric in metrics:
            if metric == 'accuracy':
                results[metric] = accuracy_score(y_true, y_pred)
            elif metric == 'precision':
                results[metric] = precision_score(
                    y_true, y_pred, average='weighted', zero_division=0
                )
            elif metric == 'recall':
                results[metric] = recall_score(
                    y_true, y_pred, average='weighted', zero_division=0
                )
            elif metric == 'f1':
                results[metric] = f1_score(
                    y_true, y_pred, average='weighted', zero_division=0
                )
            else:
                print(f"⚠️ Unknown metric: {metric}")
        
        return results
    
    def _compute_overall_statistics(
        self,
        fold_results: List[Dict[str, Any]],
        all_true_labels: List,
        all_predictions: List,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Compute overall statistics across all folds"""
        overall = {}
        
        for metric in metrics:
            if metric in fold_results[0]:
                values = [fold[metric] for fold in fold_results]
                overall[f'{metric}_mean'] = np.mean(values)
                overall[f'{metric}_std'] = np.std(values)
                overall[f'{metric}_min'] = np.min(values)
                overall[f'{metric}_max'] = np.max(values)
        
        # Compute overall metrics on all predictions
        overall_metrics = self._compute_metrics(all_true_labels, all_predictions, metrics)
        for metric, value in overall_metrics.items():
            overall[f'{metric}_overall'] = value
        
        return overall
    
    def print_cv_summary(self, cv_results: Dict[str, Any]) -> None:
        """Print comprehensive cross-validation summary"""
        
        print("\n" + "="*60)
        print("📊 CROSS-VALIDATION SUMMARY")
        print("="*60)
        
        config = cv_results['cv_config']
        print(f"🔧 Configuration: {config['n_folds']}-Fold CV, "
              f"Stratified: {config['stratified']}, Shuffle: {config['shuffle']}")
        
        # Print fold-by-fold results
        print("\n📈 Fold-by-Fold Results:")
        for fold_result in cv_results['fold_results']:
            fold = fold_result['fold']
            acc = fold_result['accuracy']
            n_train = fold_result['n_train']
            n_val = fold_result['n_val']
            print(f"  Fold {fold}: Train={n_train}, Val={n_val}, Acc={acc:.4f}")
        
        # Print overall statistics
        overall = cv_results['overall_results']
        print("\n🏆 Overall Statistics:")
        print(f"  Accuracy: {overall['accuracy_mean']:.4f} ± {overall['accuracy_std']:.4f}")
        print(f"  Range: {overall['accuracy_min']:.4f} - {overall['accuracy_max']:.4f}")
        
        if 'precision_mean' in overall:
            print(f"  Precision: {overall['precision_mean']:.4f} ± {overall['precision_std']:.4f}")
        if 'recall_mean' in overall:
            print(f"  Recall: {overall['recall_mean']:.4f} ± {overall['recall_std']:.4f}")
        if 'f1_mean' in overall:
            print(f"  F1-Score: {overall['f1_mean']:.4f} ± {overall['f1_std']:.4f}")
        
        print("="*60)
    
    def get_cv_recommendations(self, cv_results: Dict[str, Any]) -> List[str]:
        """Get recommendations based on cross-validation results"""
        
        recommendations = []
        overall = cv_results['overall_results']
        
        # Check for high variance (potential overfitting)
        if overall['accuracy_std'] > 0.1:
            recommendations.append(
                "⚠️ High variance detected - consider regularization or more data"
            )
        
        # Check for low accuracy
        if overall['accuracy_mean'] < 0.6:
            recommendations.append(
                "📉 Low accuracy - consider feature engineering or different model"
            )
        
        # Check for good performance
        if overall['accuracy_mean'] > 0.9 and overall['accuracy_std'] < 0.05:
            recommendations.append("✅ Excellent performance with low variance")
        
        # Check for overfitting signs
        if overall['accuracy_max'] - overall['accuracy_min'] > 0.2:
            recommendations.append(
                "🎲 High fold-to-fold variation - consider ensemble methods"
            )
        
        return recommendations
    
    # ==================== UTILITY METHODS ====================
    
    def set_cv_parameters(self, n_folds: int = None, stratified: bool = None) -> None:
        """Update cross-validation parameters"""
        if n_folds is not None:
            self.cv_folds = n_folds
        if stratified is not None:
            self.cv_stratified = stratified
        
        # Reinitialize KFold splitter
        if self.cv_stratified:
            self.kf = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
        else:
            self.kf = KFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of all configuration parameters"""
        return {
            'split_config': {
                'test_size': self.test_size,
                'validation_size': self.validation_size,
                'random_state': self.random_state
            },
            'cv_config': {
                'n_folds': self.cv_folds,
                'stratified': self.cv_stratified,
                'shuffle': True
            }
        }


# Global validation manager instance
validation_manager = ValidationManager()
