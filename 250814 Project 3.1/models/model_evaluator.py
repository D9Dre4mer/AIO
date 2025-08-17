# =========================================
# Model Evaluator for Academic Paper Classification
# Comprehensive performance evaluation and optimization
# =========================================

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV,
    train_test_split, StratifiedKFold, learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.pipeline import Pipeline
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator for academic paper classification.
    Handles cross-validation, hyperparameter tuning, and performance analysis.
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize model evaluator.
        
        Args:
            cache_dir: Directory to store evaluation results
        """
        self.cache_dir = Path(cache_dir)
        self.evaluation_dir = self.cache_dir / "evaluations"
        self.evaluation_dir.mkdir(exist_ok=True)
        
        # Store evaluation results
        self.evaluation_results = {}
        self.best_models = {}
        self.performance_history = {}
        
        # Cross-validation settings
        self.cv_folds = 5
        self.random_state = 42
        
        # Hyperparameter tuning settings
        self.n_iter_random_search = 20
        
        logger.info(f"ModelEvaluator initialized with cache directory: {self.cache_dir}")
    
    def evaluate_model_performance(self, model_name: str, model, X_train: np.ndarray, 
                                 y_train: np.ndarray, X_test: np.ndarray, 
                                 y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a single model's performance comprehensively.
        
        Args:
            model_name: Name of the model
            model: Model instance (will be trained)
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary containing all performance metrics
        """
        logger.info(f"Evaluating model: {model_name}")
        
        # Train the model first
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'training_time': training_time,
            'n_samples_train': X_train.shape[0],
            'n_samples_test': X_test.shape[0],
            'n_features': X_train.shape[1]
        }
        
        # Add ROC AUC if possible
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            except:
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = None
        
        # Store detailed results
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_test,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        logger.info(f"Model {model_name} evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def cross_validate_model(self, model_name: str, model, X: np.ndarray, 
                           y: np.ndarray, cv_folds: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform cross-validation for a model.
        
        Args:
            model_name: Name of the model
            model: Model instance
            X, y: Training data
            cv_folds: Number of CV folds (default: self.cv_folds)
            
        Returns:
            Cross-validation results
        """
        if cv_folds is None:
            cv_folds = self.cv_folds
            
        logger.info(f"Performing {cv_folds}-fold cross-validation for {model_name}")
        
        # Create cross-validation strategy
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Cross-validate different metrics
        cv_results = {}
        
        # Accuracy
        cv_results['accuracy'] = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        # Precision, Recall, F1
        cv_results['precision'] = cross_val_score(model, X, y, cv=cv, scoring='precision_macro')
        cv_results['recall'] = cross_val_score(model, X, y, cv=cv, scoring='recall_macro')
        cv_results['f1'] = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
        
        # Calculate statistics
        cv_summary = {}
        for metric, scores in cv_results.items():
            cv_summary[f'{metric}_mean'] = np.mean(scores)
            cv_summary[f'{metric}_std'] = np.std(scores)
            cv_summary[f'{metric}_min'] = np.min(scores)
            cv_summary[f'{metric}_max'] = np.max(scores)
        
        cv_summary['cv_folds'] = cv_folds
        cv_summary['model_name'] = model_name
        
        # Store CV results
        if model_name not in self.evaluation_results:
            self.evaluation_results[model_name] = {}
        self.evaluation_results[model_name]['cross_validation'] = {
            'scores': cv_results,
            'summary': cv_summary
        }
        
        logger.info(f"CV completed for {model_name}. Mean accuracy: {cv_summary['accuracy_mean']:.4f} ± {cv_summary['accuracy_std']:.4f}")
        return cv_summary
    
    def hyperparameter_tuning(self, model_name: str, model, param_grid: Dict[str, List], 
                            X: np.ndarray, y: np.ndarray, 
                            method: str = 'grid', cv_folds: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
        
        Args:
            model_name: Name of the model
            model: Base model instance
            param_grid: Parameter grid for tuning
            X, y: Training data
            method: 'grid' or 'random'
            cv_folds: Number of CV folds
            
        Returns:
            Tuning results with best parameters
        """
        if cv_folds is None:
            cv_folds = self.cv_folds
            
        logger.info(f"Starting hyperparameter tuning for {model_name} using {method} search")
        
        # Create cross-validation strategy
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Choose search method
        if method.lower() == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring='f1_macro', 
                n_jobs=-1, verbose=1
            )
        elif method.lower() == 'random':
            search = RandomizedSearchCV(
                model, param_grid, cv=cv, scoring='f1_macro',
                n_iter=self.n_iter_random_search, n_jobs=-1, 
                verbose=1, random_state=self.random_state
            )
        else:
            raise ValueError("Method must be 'grid' or 'random'")
        
        # Perform search
        start_time = time.time()
        search.fit(X, y)
        tuning_time = time.time() - start_time
        
        # Store results
        tuning_results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'cv_results': search.cv_results_,
            'tuning_time': tuning_time,
            'method': method,
            'cv_folds': cv_folds
        }
        
        # Store in evaluation results
        if model_name not in self.evaluation_results:
            self.evaluation_results[model_name] = {}
        self.evaluation_results[model_name]['hyperparameter_tuning'] = tuning_results
        
        # Store best model
        self.best_models[model_name] = search.best_estimator_
        
        logger.info(f"Hyperparameter tuning completed for {model_name}")
        logger.info(f"Best score: {search.best_score_:.4f}")
        logger.info(f"Best parameters: {search.best_params_}")
        
        return tuning_results
    
    def generate_performance_report(self, save_to_file: bool = True) -> pd.DataFrame:
        """
        Generate comprehensive performance report for all evaluated models.
        
        Args:
            save_to_file: Whether to save report to file
            
        Returns:
            DataFrame with performance summary
        """
        logger.info("Generating performance report")
        
        report_data = []
        
        for model_name, results in self.evaluation_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                row = {
                    'Model': model_name,
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision (Macro)': f"{metrics['precision_macro']:.4f}",
                    'Recall (Macro)': f"{metrics['recall_macro']:.4f}",
                    'F1-Score (Macro)': f"{metrics['f1_macro']:.4f}",
                    'Precision (Weighted)': f"{metrics['precision_weighted']:.4f}",
                    'Recall (Weighted)': f"{metrics['recall_weighted']:.4f}",
                    'F1-Score (Weighted)': f"{metrics['f1_weighted']:.4f}",
                    'Training Time (s)': f"{metrics['training_time']:.3f}",
                    'Train Samples': metrics['n_samples_train'],
                    'Test Samples': metrics['n_samples_test'],
                    'Features': metrics['n_features']
                }
                
                # Add CV results if available
                if 'cross_validation' in results:
                    cv_summary = results['cross_validation']['summary']
                    row['CV Accuracy'] = f"{cv_summary['accuracy_mean']:.4f} ± {cv_summary['accuracy_std']:.4f}"
                    row['CV F1-Score'] = f"{cv_summary['f1_mean']:.4f} ± {cv_summary['f1_std']:.4f}"
                
                # Add hyperparameter tuning results if available
                if 'hyperparameter_tuning' in results:
                    tuning = results['hyperparameter_tuning']
                    row['Best CV Score'] = f"{tuning['best_score']:.4f}"
                    row['Tuning Method'] = tuning['method']
                
                report_data.append(row)
        
        # Create DataFrame
        report_df = pd.DataFrame(report_data)
        
        # Save to file if requested
        if save_to_file:
            report_path = self.evaluation_dir / f"performance_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            report_df.to_csv(report_path, index=False)
            logger.info(f"Performance report saved to: {report_path}")
        
        return report_df
    
    def plot_model_comparison(self, save_plots: bool = True) -> None:
        """
        Create visualization plots for model comparison.
        
        Args:
            save_plots: Whether to save plots to files
        """
        logger.info("Creating model comparison plots")
        
        if not self.evaluation_results:
            logger.warning("No evaluation results available for plotting")
            return
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        model_names = []
        accuracies = []
        f1_scores = []
        training_times = []
        
        for model_name, results in self.evaluation_results.items():
            if 'metrics' in results:
                model_names.append(model_name)
                metrics = results['metrics']
                accuracies.append(metrics['accuracy'])
                f1_scores.append(metrics['f1_macro'])
                training_times.append(metrics['training_time'])
        
        # 1. Accuracy comparison
        axes[0, 0].bar(model_names, accuracies, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, 1)
        
        # 2. F1-Score comparison
        axes[0, 1].bar(model_names, f1_scores, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Model F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylim(0, 1)
        
        # 3. Training time comparison
        axes[1, 0].bar(model_names, training_times, color='salmon', alpha=0.7)
        axes[1, 0].set_title('Model Training Time Comparison')
        axes[1, 0].set_ylabel('Training Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Radar chart for multiple metrics
        if len(model_names) > 0:
            # Prepare data for radar chart
            metrics_data = []
            for model_name in model_names:
                results = self.evaluation_results[model_name]
                if 'metrics' in results:
                    metrics = results['metrics']
                    metrics_data.append([
                        metrics['accuracy'],
                        metrics['precision_macro'],
                        metrics['recall_macro'],
                        metrics['f1_macro']
                    ])
            
            if metrics_data:
                # Create radar chart
                categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]  # Complete the circle
                
                ax_radar = axes[1, 1]
                ax_radar.remove()
                ax_radar = fig.add_subplot(2, 2, 4, projection='polar')
                
                for i, (model_name, metrics) in enumerate(zip(model_names, metrics_data)):
                    metrics += metrics[:1]  # Complete the circle
                    ax_radar.plot(angles, metrics, 'o-', linewidth=2, label=model_name)
                    ax_radar.fill(angles, metrics, alpha=0.1)
                
                ax_radar.set_xticks(angles[:-1])
                ax_radar.set_xticklabels(categories)
                ax_radar.set_ylim(0, 1)
                ax_radar.set_title('Performance Radar Chart')
                ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots:
            plot_path = self.evaluation_dir / f"model_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to: {plot_path}")
        
        plt.show()
    
    def save_evaluation_results(self, filename: Optional[str] = None) -> str:
        """
        Save all evaluation results to file.
        
        Args:
            filename: Custom filename (optional)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"evaluation_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        file_path = self.evaluation_dir / filename
        
        # Prepare data for saving (remove non-serializable objects)
        save_data = {}
        for model_name, results in self.evaluation_results.items():
            save_data[model_name] = {}
            for key, value in results.items():
                if key in ['metrics', 'cross_validation', 'hyperparameter_tuning']:
                    save_data[model_name][key] = value
                elif key in ['predictions', 'probabilities', 'true_labels']:
                    # Convert numpy arrays to lists for saving
                    if isinstance(value, np.ndarray):
                        save_data[model_name][key] = value.tolist()
                    else:
                        save_data[model_name][key] = value
                elif key in ['classification_report', 'confusion_matrix']:
                    save_data[model_name][key] = value
        
        # Save to file
        joblib.dump(save_data, file_path)
        logger.info(f"Evaluation results saved to: {file_path}")
        
        return str(file_path)
    
    def load_evaluation_results(self, filepath: str) -> bool:
        """
        Load evaluation results from file.
        
        Args:
            filepath: Path to saved evaluation results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            loaded_data = joblib.load(filepath)
            self.evaluation_results = loaded_data
            logger.info(f"Evaluation results loaded from: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load evaluation results: {e}")
            return False
    
    def get_best_model(self, metric: str = 'f1_macro') -> Tuple[str, Any]:
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric: Metric to use for comparison ('accuracy', 'f1_macro', etc.)
            
        Returns:
            Tuple of (model_name, best_model)
        """
        if not self.evaluation_results:
            return None, None
        
        best_score = -1
        best_model_name = None
        best_model = None
        
        for model_name, results in self.evaluation_results.items():
            if 'metrics' in results and metric in results['metrics']:
                score = results['metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_model = self.best_models.get(model_name)
        
        return best_model_name, best_model


# Convenience functions
def create_model_evaluator(cache_dir: str = "cache") -> ModelEvaluator:
    """Create and return a ModelEvaluator instance."""
    return ModelEvaluator(cache_dir)


def evaluate_all_models(models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray, 
                       perform_cv: bool = True, perform_tuning: bool = False,
                       param_grids: Optional[Dict[str, Dict]] = None) -> ModelEvaluator:
    """
    Evaluate all models comprehensively.
    
    Args:
        models: Dictionary of {model_name: model_instance}
        X_train, y_train, X_test, y_test: Data splits
        perform_cv: Whether to perform cross-validation
        perform_tuning: Whether to perform hyperparameter tuning
        param_grids: Parameter grids for tuning (optional)
        
    Returns:
        ModelEvaluator instance with results
    """
    evaluator = ModelEvaluator()
    
    for model_name, model in models.items():
        logger.info(f"Evaluating model: {model_name}")
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate performance
        evaluator.evaluate_model_performance(model_name, model, X_train, y_train, X_test, y_test)
        
        # Cross-validation
        if perform_cv:
            evaluator.cross_validate_model(model_name, model, X_train, y_train)
        
        # Hyperparameter tuning
        if perform_tuning and param_grids and model_name in param_grids:
            evaluator.hyperparameter_tuning(model_name, model, param_grids[model_name], X_train, y_train)
    
    return evaluator
