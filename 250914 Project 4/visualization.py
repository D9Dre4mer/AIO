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


# SHAP Visualization Functions

def create_shap_explainer(model, X_sample, model_type="auto"):
    """
    Create SHAP explainer for different model types
    
    Args:
        model: Trained model
        X_sample: Sample data for SHAP analysis
        model_type: Type of model ("tree", "linear", "auto")
        
    Returns:
        SHAP explainer object
    """
    try:
        import shap
        
        # Auto-detect model type if not specified
        if model_type == "auto":
            model_name = model.__class__.__name__.lower()
            if any(tree_model in model_name for tree_model in ['randomforest', 'decisiontree', 'gradientboosting', 'xgboost', 'lightgbm', 'catboost']):
                model_type = "tree"
            elif any(linear_model in model_name for linear_model in ['logisticregression', 'linearsvc', 'svm']):
                model_type = "linear"
            else:
                model_type = "tree"  # Default to tree explainer
        
        # Create appropriate explainer
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            explainer = shap.LinearExplainer(model, X_sample)
        else:
            # Fallback to TreeExplainer
            explainer = shap.TreeExplainer(model)
        
        print(f"✅ SHAP {model_type.title()}Explainer created successfully")
        return explainer
        
    except ImportError:
        print("❌ SHAP not available. Please install with: pip install shap")
        return None
    except Exception as e:
        print(f"❌ Error creating SHAP explainer: {e}")
        return None


def generate_shap_summary_plot(explainer, X_sample, feature_names=None, max_display=20, save_path=None):
    """
    Generate SHAP summary plot (beeswarm plot)
    
    Args:
        explainer: SHAP explainer object
        X_sample: Sample data for SHAP analysis
        feature_names: List of feature names
        max_display: Maximum number of features to display
        save_path: Path to save the plot
        
    Returns:
        matplotlib figure object
    """
    try:
        import shap
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         max_display=max_display, show=False)
        
        plt.title("SHAP Summary Plot (Beeswarm)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ SHAP summary plot saved to: {save_path}")
        
        return plt.gcf()
        
    except Exception as e:
        print(f"❌ Error generating SHAP summary plot: {e}")
        return None


def generate_shap_bar_plot(explainer, X_sample, feature_names=None, max_display=20, save_path=None):
    """
    Generate SHAP bar plot (mean absolute SHAP values)
    
    Args:
        explainer: SHAP explainer object
        X_sample: Sample data for SHAP analysis
        feature_names: List of feature names
        max_display: Maximum number of features to display
        save_path: Path to save the plot
        
    Returns:
        matplotlib figure object
    """
    try:
        import shap
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification
        
        # Create bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         max_display=max_display, plot_type="bar", show=False)
        
        plt.title("SHAP Bar Plot (Mean Absolute SHAP Values)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ SHAP bar plot saved to: {save_path}")
        
        return plt.gcf()
        
    except Exception as e:
        print(f"❌ Error generating SHAP bar plot: {e}")
        return None


def generate_shap_dependence_plot(explainer, X_sample, feature_names=None, 
                                 feature_index=0, interaction_index=None, save_path=None):
    """
    Generate SHAP dependence plot for a specific feature
    
    Args:
        explainer: SHAP explainer object
        X_sample: Sample data for SHAP analysis
        feature_names: List of feature names
        feature_index: Index of the feature to plot
        interaction_index: Index of feature to show interaction with
        save_path: Path to save the plot
        
    Returns:
        matplotlib figure object
    """
    try:
        import shap
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification
        
        # Create dependence plot
        plt.figure(figsize=(10, 6))
        
        if interaction_index is not None:
            shap.dependence_plot(feature_index, shap_values, X_sample, 
                                feature_names=feature_names, 
                                interaction_index=interaction_index, show=False)
        else:
            shap.dependence_plot(feature_index, shap_values, X_sample, 
                                feature_names=feature_names, show=False)
        
        feature_name = feature_names[feature_index] if feature_names else f"Feature {feature_index}"
        plt.title(f"SHAP Dependence Plot: {feature_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ SHAP dependence plot saved to: {save_path}")
        
        return plt.gcf()
        
    except Exception as e:
        print(f"❌ Error generating SHAP dependence plot: {e}")
        return None


def generate_comprehensive_shap_analysis(model, X_sample, feature_names=None, 
                                       model_name="Model", output_dir="info/Result/"):
    """
    Generate comprehensive SHAP analysis with multiple plots
    
    Args:
        model: Trained model
        X_sample: Sample data for SHAP analysis
        feature_names: List of feature names
        model_name: Name of the model for file naming
        output_dir: Directory to save plots
        
    Returns:
        Dict containing paths to saved plots
    """
    try:
        import shap
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create SHAP explainer
        explainer = create_shap_explainer(model, X_sample)
        if explainer is None:
            return None
        
        # Generate timestamp for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate plots
        plots = {}
        
        # 1. Summary plot (beeswarm)
        summary_path = os.path.join(output_dir, f"{model_name}_shap_summary_{timestamp}.png")
        summary_plot = generate_shap_summary_plot(explainer, X_sample, feature_names, 
                                                 save_path=summary_path)
        if summary_plot:
            plots['summary'] = summary_path
        
        # 2. Bar plot
        bar_path = os.path.join(output_dir, f"{model_name}_shap_bar_{timestamp}.png")
        bar_plot = generate_shap_bar_plot(explainer, X_sample, feature_names, 
                                        save_path=bar_path)
        if bar_plot:
            plots['bar'] = bar_path
        
        # 3. Dependence plots for top features
        if feature_names:
            # Get top 3 most important features
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            top_features = np.argsort(mean_shap)[-3:][::-1]
            
            for i, feature_idx in enumerate(top_features):
                dep_path = os.path.join(output_dir, f"{model_name}_shap_dependence_{feature_names[feature_idx]}_{timestamp}.png")
                dep_plot = generate_shap_dependence_plot(explainer, X_sample, feature_names, 
                                                      feature_index=feature_idx, save_path=dep_path)
                if dep_plot:
                    plots[f'dependence_{feature_names[feature_idx]}'] = dep_path
        
        print(f"✅ Comprehensive SHAP analysis completed for {model_name}")
        print(f"   Generated {len(plots)} plots in {output_dir}")
        
        return plots
        
    except Exception as e:
        print(f"❌ Error in comprehensive SHAP analysis: {e}")
        return None


def plot_shap_waterfall(explainer, X_sample, instance_index=0, feature_names=None, save_path=None):
    """
    Generate SHAP waterfall plot for a single instance
    
    Args:
        explainer: SHAP explainer object
        X_sample: Sample data for SHAP analysis
        instance_index: Index of the instance to explain
        feature_names: List of feature names
        save_path: Path to save the plot
        
    Returns:
        matplotlib figure object
    """
    try:
        import shap
        
        # Calculate SHAP values for the instance
        shap_values = explainer.shap_values(X_sample[instance_index:instance_index+1])
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explainer.expected_value, shap_values[0], X_sample[instance_index], 
                           feature_names=feature_names, show=False)
        
        plt.title(f"SHAP Waterfall Plot - Instance {instance_index}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ SHAP waterfall plot saved to: {save_path}")
        
        return plt.gcf()
        
    except Exception as e:
        print(f"❌ Error generating SHAP waterfall plot: {e}")
        return None