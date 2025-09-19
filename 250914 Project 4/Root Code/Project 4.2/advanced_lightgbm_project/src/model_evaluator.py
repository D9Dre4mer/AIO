"""
Advanced Model Evaluation Module

This module provides comprehensive model evaluation capabilities:
- Advanced metrics calculation
- Visualization tools
- Performance comparison
- Statistical significance testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, roc_curve, precision_recall_curve,
                           confusion_matrix, classification_report, log_loss,
                           matthews_corrcoef, cohen_kappa_score, average_precision_score)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Advanced model evaluation with comprehensive metrics and visualizations
    
    Features:
    - Comprehensive metrics calculation
    - Advanced visualizations
    - Performance comparison
    - Statistical significance testing
    - Model interpretability analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelEvaluator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.evaluation_config = config.get('evaluation', {})
        self.results = {}
        
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Probability-based metrics
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['auc_pr'] = average_precision_score(y_true, y_pred_proba)
        metrics['log_loss'] = log_loss(y_true, np.clip(y_pred_proba, 1e-15, 1-1e-15))
        
        # Advanced metrics
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Additional metrics
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        metrics['f2'] = (1 + 2**2) * (metrics['precision'] * metrics['recall']) / (2**2 * metrics['precision'] + metrics['recall']) if (2**2 * metrics['precision'] + metrics['recall']) > 0 else 0
        metrics['f0_5'] = (1 + 0.5**2) * (metrics['precision'] * metrics['recall']) / (0.5**2 * metrics['precision'] + metrics['recall']) if (0.5**2 * metrics['precision'] + metrics['recall']) > 0 else 0
        
        return metrics
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      model_name: str = "Model", save_path: Optional[str] = None):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of model for legend
            save_path: Path to save plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title(f'ROC Curve - {model_name}\n(Classification Performance)', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                  model_name: str = "Model", save_path: Optional[str] = None):
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of model for legend
            save_path: Path to save plot
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        auc_pr = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'{model_name} (AUC-PR = {auc_pr:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity)', fontsize=12)
        plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}\n(Classification Performance)', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = "Model", save_path: Optional[str] = None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of model for title
            save_path: Path to save plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['True 0', 'True 1'])
        plt.title(f'Confusion Matrix - {model_name}\n(Classification Accuracy)', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_prediction_distribution(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   model_name: str = "Model", save_path: Optional[str] = None):
        """
        Plot prediction probability distribution
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of model for title
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Probability distribution
        plt.subplot(1, 2, 1)
        plt.hist(y_pred_proba[y_true == 0], bins=20, alpha=0.7, label='Class 0', color='blue', density=True)
        plt.hist(y_pred_proba[y_true == 1], bins=20, alpha=0.7, label='Class 1', color='red', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title(f'Prediction Distribution - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Calibration plot
        plt.subplot(1, 2, 2)
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_proba, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f'{model_name}')
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_metrics_comparison(self, results_dict: Dict[str, Dict[str, float]],
                              metrics: List[str] = None, save_path: Optional[str] = None):
        """
        Plot metrics comparison across models
        
        Args:
            results_dict: Dictionary with model results
            metrics: List of metrics to plot
            save_path: Path to save plot
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        
        # Prepare data
        model_names = list(results_dict.keys())
        metric_values = {metric: [results_dict[model].get(metric, 0) for model in model_names] 
                        for metric in metrics}
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        
        for i, metric in enumerate(metrics):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.bar(model_names, metric_values[metric], color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
            metric_names = {
                'accuracy': 'Độ chính xác',
                'precision': 'Độ chính xác dự đoán',
                'recall': 'Độ nhạy',
                'f1': 'F1-Score',
                'auc_roc': 'AUC-ROC'
            }
            plt.title(f'So sánh {metric_names.get(metric, metric.upper())}', fontweight='bold')
            plt.ylabel(metric_names.get(metric, metric.upper()), fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for j, v in enumerate(metric_values[metric]):
                plt.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_radar_chart(self, results_dict: Dict[str, Dict[str, float]],
                        metrics: List[str] = None, save_path: Optional[str] = None):
        """
        Plot radar chart for model comparison
        
        Args:
            results_dict: Dictionary with model results
            metrics: List of metrics to plot
            save_path: Path to save plot
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        
        # Prepare data
        model_names = list(results_dict.keys())
        metric_values = {metric: [results_dict[model].get(metric, 0) for model in model_names] 
                        for metric in metrics}
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        for i, model in enumerate(model_names):
            values = [metric_values[metric][i] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        metric_labels = {
            'accuracy': 'Độ chính xác',
            'precision': 'Độ chính xác dự đoán', 
            'recall': 'Độ nhạy',
            'f1': 'F1-Score',
            'auc_roc': 'AUC-ROC'
        }
        ax.set_xticklabels([metric_labels.get(m, m) for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Biểu đồ Radar So sánh Mô hình\n(Hiệu suất tổng thể)', size=16, pad=20, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        else:
            plt.show()
    
    def statistical_significance_test(self, y_true: np.ndarray, 
                                    model1_pred: np.ndarray, model2_pred: np.ndarray,
                                    test_type: str = 'mcnemar') -> Dict[str, Any]:
        """
        Perform statistical significance test between two models
        
        Args:
            y_true: True labels
            model1_pred: Predictions from model 1
            model2_pred: Predictions from model 2
            test_type: Type of test ('mcnemar', 'wilcoxon', 't_test')
            
        Returns:
            Dictionary with test results
        """
        if test_type == 'mcnemar':
            # McNemar's test for paired binary data
            from statsmodels.stats.contingency_tables import mcnemar
            
            # Create contingency table
            table = np.array([
                [np.sum((y_true == 0) & (model1_pred == 0) & (model2_pred == 0)),
                 np.sum((y_true == 0) & (model1_pred == 0) & (model2_pred == 1))],
                [np.sum((y_true == 0) & (model1_pred == 1) & (model2_pred == 0)),
                 np.sum((y_true == 0) & (model1_pred == 1) & (model2_pred == 1))]
            ])
            
            result = mcnemar(table, exact=True)
            
            return {
                'test_type': 'mcnemar',
                'statistic': result.statistic,
                'p_value': result.pvalue,
                'significant': result.pvalue < 0.05,
                'contingency_table': table
            }
        
        elif test_type == 'wilcoxon':
            # Wilcoxon signed-rank test
            from scipy.stats import wilcoxon
            
            # Calculate accuracy for each sample
            model1_acc = (y_true == model1_pred).astype(int)
            model2_acc = (y_true == model2_pred).astype(int)
            
            statistic, p_value = wilcoxon(model1_acc, model2_acc)
            
            return {
                'test_type': 'wilcoxon',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        elif test_type == 't_test':
            # Paired t-test
            from scipy.stats import ttest_rel
            
            # Calculate accuracy for each sample
            model1_acc = (y_true == model1_pred).astype(int)
            model2_acc = (y_true == model2_pred).astype(int)
            
            statistic, p_value = ttest_rel(model1_acc, model2_acc)
            
            return {
                'test_type': 't_test',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    def cross_validation_analysis(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                                cv_folds: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Perform cross-validation analysis for multiple models
        
        Args:
            models: Dictionary of models
            X: Features
            y: Target
            cv_folds: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        print(f"🔄 Performing {cv_folds}-fold cross-validation analysis...")
        
        cv_results = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                cv_results[name] = {
                    'scores': scores,
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
                print(f"   {name}: {cv_results[name]['mean']:.4f} (+/- {cv_results[name]['std']:.4f})")
            except Exception as e:
                print(f"⚠️  Error in CV for {name}: {e}")
                cv_results[name] = None
        
        return cv_results
    
    def plot_cv_results(self, cv_results: Dict[str, Any], save_path: Optional[str] = None):
        """
        Plot cross-validation results
        
        Args:
            cv_results: Cross-validation results
            save_path: Path to save plot
        """
        # Prepare data
        model_names = []
        means = []
        stds = []
        
        for name, result in cv_results.items():
            if result is not None:
                model_names.append(name)
                means.append(result['mean'])
                stds.append(result['std'])
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        x_pos = np.arange(len(model_names))
        bars = plt.bar(x_pos, means, yerr=stds, capsize=5, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
        
        plt.xlabel('Models')
        plt.ylabel('Cross-Validation Score')
        plt.title('Cross-Validation Results Comparison')
        plt.xticks(x_pos, model_names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            plt.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_evaluation_report(self, results_dict: Dict[str, Dict[str, float]],
                                 save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            results_dict: Dictionary with model results
            save_path: Path to save report
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Model comparison
        report.append("MODEL PERFORMANCE COMPARISON")
        report.append("-" * 40)
        
        # Create comparison table
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'matthews_corrcoef']
        model_names = list(results_dict.keys())
        
        # Header
        header = f"{'Model':<20} " + " ".join([f"{metric.upper():<10}" for metric in metrics])
        report.append(header)
        report.append("-" * len(header))
        
        # Data rows
        for model in model_names:
            row = f"{model:<20} "
            for metric in metrics:
                value = results_dict[model].get(metric, 0)
                row += f"{value:<10.4f} "
            report.append(row)
        
        report.append("")
        
        # Best model analysis
        report.append("BEST MODEL ANALYSIS")
        report.append("-" * 40)
        
        # Find best model for each metric
        for metric in metrics:
            best_model = max(model_names, key=lambda x: results_dict[x].get(metric, 0))
            best_score = results_dict[best_model].get(metric, 0)
            report.append(f"Best {metric.upper()}: {best_model} ({best_score:.4f})")
        
        report.append("")
        
        # Overall best model
        overall_scores = {}
        for model in model_names:
            # Calculate average rank across all metrics
            ranks = []
            for metric in metrics:
                values = [results_dict[m].get(metric, 0) for m in model_names]
                sorted_values = sorted(values, reverse=True)
                rank = sorted_values.index(results_dict[model].get(metric, 0)) + 1
                ranks.append(rank)
            overall_scores[model] = np.mean(ranks)
        
        best_overall = min(overall_scores.keys(), key=lambda x: overall_scores[x])
        report.append(f"Overall Best Model: {best_overall} (avg rank: {overall_scores[best_overall]:.2f})")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"✅ Evaluation report saved to {save_path}")
        
        return report_text
