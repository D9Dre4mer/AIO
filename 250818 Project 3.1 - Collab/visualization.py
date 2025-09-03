"""
Visualization module for Topic Modeling Project
Handles plotting and visualization functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import FIGURES_DIR


def plot_confusion_matrix(
    y_true, 
    y_pred, 
    label_list, 
    figure_name="Confusion Matrix", 
    save_path=None
):
    """
    Plots a confusion matrix with raw counts and normalized values using Seaborn.

    Parameters:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        label_list (list or dict): Class names (list or dict from ID to name).
        figure_name (str): Title of the plot.
        save_path (str, optional): Path to save the figure. If None, the figure will not be saved.
    """
    # Compute confusion matrix and normalize
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Map class indices to names
    labels = np.unique(y_true)
    if isinstance(label_list, dict):
        class_names = [label_list[i] for i in labels]
    else:
        class_names = [label_list[i] for i in labels]

    # Create annotations with raw + normalized values
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            raw = cm[i, j]
            norm = cm_normalized[i, j]
            annotations[i, j] = f"{raw}\n({norm:.2%})"

    # Plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, 
        annot=annotations, 
        fmt="", 
        cmap="Blues",
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar=False, 
        linewidths=1, 
        linecolor='black'
    )

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(figure_name)
    plt.tight_layout()

    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()


def create_output_directories():
    """Create necessary output directories"""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"✅ Created output directory: {FIGURES_DIR}")


def plot_model_comparison(results_dict, save_path=None):
    """
    Plot comparison of different models and vectorization methods
    
    Parameters:
        results_dict (dict): Dictionary containing results for different models
        save_path (str, optional): Path to save the figure
    """
    # Define specific models to plot
    models = ['kmeans', 'knn', 'decision_tree', 'naive_bayes']
    vectorization_methods = ['bow', 'tfidf', 'embeddings']
    
    # Extract accuracies
    accuracies = {}
    for model in models:
        accuracies[model] = []
        for method in vectorization_methods:
            key = f"{model}_{method}_accuracy"
            if key in results_dict:
                accuracies[model].append(results_dict[key])
            else:
                accuracies[model].append(0)
    
    # Create bar plot
    x = np.arange(len(vectorization_methods))
    width = 0.2
    
    plt.figure(figsize=(10, 6))
    
    for i, (model, accs) in enumerate(accuracies.items()):
        plt.bar(
            x + i * width, 
            accs, 
            width, 
            label=model.replace('_', ' ').title()
        )
    
    plt.xlabel('Vectorization Method')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width * (len(models) - 1) / 2, vectorization_methods)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def print_model_results(results_dict):
    """Print formatted results for all models"""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    models = ['kmeans', 'knn', 'decision_tree', 'naive_bayes']
    vectorization_methods = ['bow', 'tfidf', 'embeddings']
    
    for model in models:
        print(f"\n{model.replace('_', ' ').title()}:")
        print("-" * 40)
        
        for method in vectorization_methods:
            key = f"{model}_{method}_accuracy"
            if key in results_dict:
                accuracy = results_dict[key]
                print(f"  {method.upper()}: {accuracy:.4f}")
            else:
                print(f"  {method.upper()}: N/A")
    
    print("\n" + "="*60)
