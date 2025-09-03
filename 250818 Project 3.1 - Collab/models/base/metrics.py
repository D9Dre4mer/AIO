"""
Common evaluation metrics for machine learning models
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)


class ModelMetrics:
    """Class for computing and managing model evaluation metrics"""
    
    @staticmethod
    def compute_classification_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        labels: List[str] = None
    ) -> Dict[str, Any]:
        """Compute comprehensive classification metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted',
                                  zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted',
                             zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Detailed classification report
        report = classification_report(
            y_true, y_pred, 
            labels=labels,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'metrics_summary': {
                'accuracy': f"{accuracy:.4f}",
                'precision': f"{precision:.4f}",
                'recall': f"{recall:.4f}",
                'f1_score': f"{f1:.4f}"
            }
        }
    
    @staticmethod
    def compute_clustering_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Compute clustering evaluation metrics"""
        
        # For clustering, we need to handle label mapping
        from collections import Counter
        
        # Create label mapping from clusters to true labels
        cluster_to_label = {}
        for cluster_id in set(y_pred):
            labels_in_cluster = [
                y_true[i] for i in range(len(y_true)) 
                if y_pred[i] == cluster_id
            ]
            if labels_in_cluster:
                most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
                cluster_to_label[cluster_id] = most_common_label
        
        # Map cluster predictions to true labels for evaluation
        y_pred_mapped = [cluster_to_label[cluster_id] for cluster_id in y_pred]
        
        # Compute classification metrics on mapped predictions
        return ModelMetrics.compute_classification_metrics(y_true, y_pred_mapped)
    
    @staticmethod
    def print_metrics_summary(metrics: Dict[str, Any]) -> None:
        """Print a formatted summary of metrics"""
        print("\n📊 Model Performance Summary")
        print("=" * 40)
        
        if 'metrics_summary' in metrics:
            summary = metrics['metrics_summary']
            print(f"Accuracy:  {summary['accuracy']}")
            print(f"Precision: {summary['precision']}")
            print(f"Recall:    {summary['recall']}")
            print(f"F1-Score:  {summary['f1_score']}")
        
        if 'accuracy' in metrics:
            print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        
        print("=" * 40)
    
    @staticmethod
    def compare_models(
        model_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare multiple models based on their metrics"""
        
        comparison = {}
        
        for model_name, results in model_results.items():
            if 'accuracy' in results:
                comparison[model_name] = {
                    'accuracy': results['accuracy'],
                    'precision': results.get('precision', 0),
                    'recall': results.get('recall', 0),
                    'f1_score': results.get('f1_score', 0)
                }
        
        # Sort by accuracy
        sorted_models = sorted(
            comparison.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )
        
        return {
            'comparison': comparison,
            'sorted_models': sorted_models,
            'best_model': sorted_models[0] if sorted_models else None
        }
