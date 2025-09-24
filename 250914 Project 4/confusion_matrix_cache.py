"""
Confusion Matrix from Cache
Generates confusion matrices using cached eval_predictions

Features:
- Read eval_predictions from model cache
- Generate confusion matrix with normalization options
- Support for binary and multiclass classification
- Configurable threshold for binary classification
- Label ordering and mapping support
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
from sklearn.metrics import confusion_matrix
from cache_manager import cache_manager

logger = logging.getLogger(__name__)

class ConfusionMatrixCache:
    """Generates confusion matrices from cached evaluation predictions"""
    
    def __init__(self, cache_root_dir: str = "cache/models/"):
        """Initialize confusion matrix cache
        
        Args:
            cache_root_dir: Root directory for model caches
        """
        self.cache_root_dir = Path(cache_root_dir)
        self.cache_manager = cache_manager
    
    def generate_confusion_matrix_from_cache(self, model_key: str, dataset_id: str, 
                                           config_hash: str, 
                                           threshold: float = 0.5,
                                           normalize: str = "true",
                                           labels_order: Optional[List[str]] = None,
                                           save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate confusion matrix from cached eval_predictions
        
        Args:
            model_key: Model identifier
            dataset_id: Dataset identifier
            config_hash: Configuration hash
            threshold: Threshold for binary classification
            normalize: Normalization method ('true', 'pred', 'all', None)
            labels_order: Order of labels for display
            save_path: Path to save confusion matrix plot
            
        Returns:
            Dictionary containing confusion matrix data and plot
        """
        try:
            # Load cached data
            cache_data = self.cache_manager.load_model_cache(model_key, dataset_id, config_hash)
            
            if cache_data['eval_predictions'] is None:
                raise ValueError(f"No eval_predictions found in cache for {model_key}")
            
            eval_df = cache_data['eval_predictions']
            
            # Extract true labels and predictions
            y_true = eval_df['y_true'].values
            y_pred = self._extract_predictions(eval_df, threshold)
            
            # Get label mapping
            label_mapping = cache_data.get('label_mapping', {})
            if not label_mapping:
                # Create default mapping
                unique_labels = sorted(set(y_true) | set(y_pred))
                label_mapping = {i: f"Class_{i}" for i in unique_labels}
            
            # Apply label ordering if provided
            if labels_order:
                label_mapping = {k: v for k, v in label_mapping.items() if v in labels_order}
            
            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=list(label_mapping.keys()))
            
            # Normalize if requested
            if normalize:
                cm_normalized = self._normalize_confusion_matrix(cm, normalize)
            else:
                cm_normalized = cm
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(cm_normalized, 
                       annot=True, 
                       fmt='.3f' if normalize else 'd',
                       cmap='Blues',
                       xticklabels=list(label_mapping.values()),
                       yticklabels=list(label_mapping.values()),
                       ax=ax)
            
            # Set title and labels
            title = f"Confusion Matrix - {model_key}"
            if normalize:
                title += f" (Normalized: {normalize})"
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            
            # Add raw counts as text if normalized
            if normalize:
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j + 0.5, i + 0.7, f'({cm[i, j]})', 
                               ha='center', va='center', fontsize=8, color='gray')
            
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to {save_path}")
            
            # Calculate metrics
            metrics = self._calculate_metrics(cm, label_mapping)
            
            result = {
                'confusion_matrix': cm,
                'confusion_matrix_normalized': cm_normalized,
                'plot': fig,
                'metrics': metrics,
                'label_mapping': label_mapping,
                'threshold': threshold,
                'normalize': normalize,
                'model_key': model_key,
                'dataset_id': dataset_id,
                'config_hash': config_hash
            }
            
            logger.info(f"Confusion matrix generated for {model_key}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating confusion matrix for {model_key}: {e}")
            raise
    
    def _extract_predictions(self, eval_df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Extract predictions from evaluation dataframe
        
        Args:
            eval_df: Evaluation dataframe
            threshold: Threshold for binary classification
            
        Returns:
            Array of predictions
        """
        if 'y_pred' in eval_df.columns:
            # Use existing predictions
            return eval_df['y_pred'].values
        
        # Extract from probability columns
        proba_cols = [col for col in eval_df.columns if col.startswith('proba__class_')]
        
        if not proba_cols:
            raise ValueError("No prediction columns found in eval_predictions")
        
        if len(proba_cols) == 2:
            # Binary classification
            proba_class_1 = eval_df[proba_cols[1]].values
            y_pred = (proba_class_1 >= threshold).astype(int)
        else:
            # Multiclass classification
            proba_values = eval_df[proba_cols].values
            y_pred = np.argmax(proba_values, axis=1)
        
        return y_pred
    
    def _normalize_confusion_matrix(self, cm: np.ndarray, normalize: str) -> np.ndarray:
        """Normalize confusion matrix
        
        Args:
            cm: Confusion matrix
            normalize: Normalization method
            
        Returns:
            Normalized confusion matrix
        """
        if normalize == 'true':
            # Normalize by true labels (rows)
            return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        elif normalize == 'pred':
            # Normalize by predicted labels (columns)
            return cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        elif normalize == 'all':
            # Normalize by total
            return cm.astype('float') / cm.sum()
        else:
            return cm.astype('float')
    
    def _calculate_metrics(self, cm: np.ndarray, label_mapping: Dict[int, str]) -> Dict[str, Any]:
        """Calculate metrics from confusion matrix
        
        Args:
            cm: Confusion matrix
            label_mapping: Label mapping
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Overall accuracy
        accuracy = np.trace(cm) / np.sum(cm)
        metrics['accuracy'] = accuracy
        
        # Per-class metrics
        class_metrics = {}
        for i, (class_id, class_name) in enumerate(label_mapping.items()):
            if i < cm.shape[0]:
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics[class_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'support': tp + fn
                }
        
        metrics['class_metrics'] = class_metrics
        
        # Macro averages
        precisions = [m['precision'] for m in class_metrics.values()]
        recalls = [m['recall'] for m in class_metrics.values()]
        f1_scores = [m['f1_score'] for m in class_metrics.values()]
        
        metrics['macro_precision'] = np.mean(precisions)
        metrics['macro_recall'] = np.mean(recalls)
        metrics['macro_f1'] = np.mean(f1_scores)
        
        # Weighted averages
        supports = [m['support'] for m in class_metrics.values()]
        total_support = sum(supports)
        
        metrics['weighted_precision'] = sum(p * s for p, s in zip(precisions, supports)) / total_support
        metrics['weighted_recall'] = sum(r * s for r, s in zip(recalls, supports)) / total_support
        metrics['weighted_f1'] = sum(f * s for f, s in zip(f1_scores, supports)) / total_support
        
        return metrics
    
    def list_available_caches(self) -> List[Dict[str, Any]]:
        """List all available model caches with eval_predictions
        
        Returns:
            List of cache information
        """
        cached_models = self.cache_manager.list_cached_models()
        available_caches = []
        
        for model_info in cached_models:
            try:
                cache_data = self.cache_manager.load_model_cache(
                    model_info['model_key'],
                    model_info['dataset_id'],
                    model_info['config_hash']
                )
                
                if cache_data['eval_predictions'] is not None:
                    available_caches.append({
                        **model_info,
                        'has_eval_predictions': True,
                        'eval_predictions_shape': cache_data['eval_predictions'].shape,
                        'has_shap_sample': cache_data['shap_sample'] is not None
                    })
                else:
                    available_caches.append({
                        **model_info,
                        'has_eval_predictions': False,
                        'eval_predictions_shape': None,
                        'has_shap_sample': cache_data['shap_sample'] is not None
                    })
                    
            except Exception as e:
                logger.warning(f"Could not load cache for {model_info['model_key']}: {e}")
                continue
        
        return available_caches
    
    def generate_confusion_matrix_summary(self, save_dir: str = "info/Result/") -> Dict[str, Any]:
        """Generate confusion matrices for all available cached models
        
        Args:
            save_dir: Directory to save confusion matrix plots
            
        Returns:
            Summary of generated confusion matrices
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        available_caches = self.list_available_caches()
        generated_matrices = []
        
        for cache_info in available_caches:
            if not cache_info['has_eval_predictions']:
                continue
            
            try:
                # Generate confusion matrix
                result = self.generate_confusion_matrix_from_cache(
                    model_key=cache_info['model_key'],
                    dataset_id=cache_info['dataset_id'],
                    config_hash=cache_info['config_hash'],
                    normalize='true'
                )
                
                # Save plot
                plot_filename = f"{cache_info['model_key']}_{cache_info['dataset_id']}_confusion_matrix.png"
                plot_path = save_path / plot_filename
                
                result['plot'].savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(result['plot'])  # Close to free memory
                
                generated_matrices.append({
                    'model_key': cache_info['model_key'],
                    'dataset_id': cache_info['dataset_id'],
                    'config_hash': cache_info['config_hash'],
                    'plot_path': str(plot_path),
                    'metrics': result['metrics'],
                    'accuracy': result['metrics']['accuracy']
                })
                
                logger.info(f"Generated confusion matrix for {cache_info['model_key']}")
                
            except Exception as e:
                logger.error(f"Failed to generate confusion matrix for {cache_info['model_key']}: {e}")
                continue
        
        # Sort by accuracy
        generated_matrices.sort(key=lambda x: x['accuracy'], reverse=True)
        
        summary = {
            'total_matrices': len(generated_matrices),
            'matrices': generated_matrices,
            'best_model': generated_matrices[0] if generated_matrices else None,
            'save_directory': str(save_path)
        }
        
        logger.info(f"Generated {len(generated_matrices)} confusion matrices")
        return summary


# Global confusion matrix cache instance
confusion_matrix_cache = ConfusionMatrixCache()
