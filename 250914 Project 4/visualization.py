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
        cmap="Greens",
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
    Create SHAP explainer for the given model - ENHANCED VERSION
    Supports custom model classes by extracting underlying sklearn models
    
    Args:
        model: Trained model (can be custom wrapper or sklearn model)
        X_sample: Sample data for SHAP analysis
        model_type: Type of model ('tree', 'linear', 'deep', 'auto')
    
    Returns:
        SHAP explainer object
    """
    try:
        import shap
        
        def extract_underlying_model(model):
            """Extract underlying sklearn model from custom wrapper"""
            try:
                # Check if model has a .model attribute (custom wrapper)
                if hasattr(model, 'model') and model.model is not None:
                    return model.model
                
                # Check if model is already a sklearn model
                elif hasattr(model, 'predict_proba') and hasattr(model, 'fit'):
                    return model
                
                # Check if model has sklearn attributes
                elif hasattr(model, 'estimators_') or hasattr(model, 'tree_'):
                    return model
                
                else:
                    return None
                    
            except Exception as e:
                print(f"Error extracting underlying model: {e}")
                return None
        
        # Extract underlying sklearn model
        underlying_model = extract_underlying_model(model)
        if underlying_model is None:
            print("ERROR: Cannot extract underlying sklearn model")
            return None
        
        model_name = model.__class__.__name__
        underlying_name = underlying_model.__class__.__name__
        
        print(f"Creating SHAP explainer for {model_name} -> {underlying_name}")
        
        # Determine model type and create appropriate explainer
        model_type_str = str(type(underlying_model)).lower()
        
        # Try TreeExplainer for tree-based models (with memory protection)
        if any(keyword in model_type_str for keyword in ['randomforest', 'lightgbm', 'xgboost', 'gradientboosting', 'decisiontree']):
            try:
                print(f"Attempting TreeExplainer for {underlying_name}")
                
                # Memory safety check
                if len(X_sample) > 100:  # Increase limit to prevent memory issues
                    print(f"Warning: Sample size {len(X_sample)} too large, using first 100 samples")
                    X_sample = X_sample[:100]
                
                explainer = shap.TreeExplainer(underlying_model)
                
                # Test explainer with small sample to ensure it works
                test_sample = X_sample[:5] if len(X_sample) >= 5 else X_sample
                try:
                    _ = explainer.shap_values(test_sample)
                    print(f"SUCCESS: Created TreeExplainer for {underlying_name}")
                    return explainer
                except Exception as test_error:
                    print(f"WARNING: TreeExplainer test failed for {underlying_name}: {test_error}")
                    del explainer
                    import gc
                    gc.collect()
                    raise test_error
                    
            except Exception as e:
                print(f"WARNING: TreeExplainer failed for {underlying_name}: {e}")
                import gc
                gc.collect()  # Force cleanup
                # Fallback to Explainer with predict_proba
                pass
        
        # Try Explainer with predict_proba for other models (with memory protection)
        try:
            print(f"Attempting Explainer with predict_proba for {underlying_name}")
            
            # Memory safety check
            if len(X_sample) > 50:  # Increase limit to prevent memory issues
                print(f"Warning: Sample size {len(X_sample)} too large, using first 50 samples")
                X_sample = X_sample[:50]
            
            def predict_proba_wrapper(X):
                return underlying_model.predict_proba(X)
            
            # Create explainer with smaller background
            background_sample = X_sample[:10] if len(X_sample) >= 10 else X_sample
            explainer = shap.Explainer(predict_proba_wrapper, background_sample)
            
            # Test explainer
            test_sample = X_sample[:3] if len(X_sample) >= 3 else X_sample
            try:
                _ = explainer(test_sample)
                print(f"SUCCESS: Created Explainer with predict_proba for {underlying_name}")
                return explainer
            except Exception as test_error:
                print(f"WARNING: Explainer test failed for {underlying_name}: {test_error}")
                del explainer
                import gc
                gc.collect()
                raise test_error
            
        except Exception as e:
            print(f"ERROR: All SHAP explainer methods failed for {underlying_name}: {e}")
            import gc
            gc.collect()  # Force cleanup
            return None
            
    except ImportError:
        print("ERROR: SHAP not available")
        return None
    except Exception as e:
        print(f"ERROR: Error creating SHAP explainer: {e}")
        return None


def get_shap_values_definitive(explainer, X_sample):
    """
    Get SHAP values in consistent format
    
    Args:
        explainer: SHAP explainer object
        X_sample: Sample data
    
    Returns:
        tuple: (shap_values, shap_type) where shap_type is 'list', 'array', or 'explanation'
    """
    try:
        if hasattr(explainer, 'shap_values'):
            # TreeExplainer
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                # Multi-class: use positive class (index 1)
                return shap_values[1], 'list'
            else:
                return shap_values, 'array'
        else:
            # Explainer (Exact)
            shap_values = explainer(X_sample)
            if hasattr(shap_values, 'values'):
                # Explanation object
                values = shap_values.values
                if len(values.shape) == 3:  # (n_samples, n_features, n_classes)
                    return values[:, :, 1], 'explanation'  # Positive class
                else:
                    return values, 'explanation'
            else:
                return shap_values, 'array'
    except Exception as e:
        print(f"ERROR: Failed to get SHAP values: {e}")
        return None, None


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
        
        # Get SHAP values using definitive method
        shap_values, shap_type = get_shap_values_definitive(explainer, X_sample)
        if shap_values is None:
            return None
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         max_display=max_display, show=False)
        
        ax.set_title("SHAP Summary Plot (Beeswarm)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SUCCESS: SHAP summary plot saved to: {save_path}")
        
        return fig
        
    except Exception as e:
        print(f"Error generating SHAP summary plot: {e}")
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
        
        # Get SHAP values using definitive method
        shap_values, shap_type = get_shap_values_definitive(explainer, X_sample)
        if shap_values is None:
            return None
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         max_display=max_display, plot_type="bar", show=False)
        
        ax.set_title("SHAP Bar Plot (Mean Absolute SHAP Values)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SUCCESS: SHAP bar plot saved to: {save_path}")
        
        return fig
        
    except Exception as e:
        print(f"Error generating SHAP bar plot: {e}")
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
        
        # Get SHAP values using definitive method
        shap_values, shap_type = get_shap_values_definitive(explainer, X_sample)
        if shap_values is None:
            return None
        
        # Create dependence plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Ensure feature_index is an integer
        if isinstance(feature_index, str):
            try:
                feature_index = int(feature_index)
            except ValueError:
                feature_index = 0
        
        try:
            if interaction_index is not None:
                shap.dependence_plot(feature_index, shap_values, X_sample, 
                                    interaction_index=interaction_index, show=False)
            else:
                shap.dependence_plot(feature_index, shap_values, X_sample, show=False)
            
            # Force set title for all axes after plot creation
            feature_name = feature_names[feature_index] if feature_names and len(feature_names) > feature_index else f"Feature {feature_index}"
            for ax in fig.get_axes():
                if not ax.get_title():  # Only set if title is empty
                    ax.set_title(f"SHAP Dependence Plot: {feature_name}", fontsize=14, fontweight='bold')
            
            # Check if plot has meaningful content
            axes = fig.get_axes()
            has_content = False
            for ax in axes:
                if len(ax.get_lines()) > 0 or len(ax.collections) > 0:
                    # Check if collections have actual data points
                    for collection in ax.collections:
                        if hasattr(collection, 'get_offsets') and len(collection.get_offsets()) > 0:
                            has_content = True
                            break
                    if has_content:
                        break
            
            if not has_content:
                print("Warning: Dependence plot has no meaningful content, creating custom plot")
                # Clear and create custom dependence plot
                fig.clear()
                ax = fig.add_subplot(111)
                
                # Extract feature values and SHAP values
                feature_values = X_sample[:, feature_index]
                shap_feature_values = shap_values[:, feature_index]
                
                # Create scatter plot
                scatter = ax.scatter(feature_values, shap_feature_values, 
                                   alpha=0.7, s=50, c=shap_feature_values, 
                                   cmap='RdBu_r', edgecolors='black', linewidth=0.5)
                
                # Add colorbar
                plt.colorbar(scatter, ax=ax, label='SHAP Value')
                
                # Set labels and title
                feature_name = feature_names[feature_index] if feature_names and len(feature_names) > feature_index else f"Feature {feature_index}"
                ax.set_xlabel(feature_name, fontsize=12)
                ax.set_ylabel(f"SHAP Value for {feature_name}", fontsize=12)
                ax.set_title(f"SHAP Dependence Plot: {feature_name}", fontsize=14, fontweight='bold')
                
                # Add grid
                ax.grid(True, alpha=0.3)
            
        except Exception as dep_error:
            print(f"Warning: Dependence plot failed, creating custom plot: {dep_error}")
            # Create custom dependence plot
            fig.clear()
            ax = fig.add_subplot(111)
            
            # Extract feature values and SHAP values
            feature_values = X_sample[:, feature_index]
            shap_feature_values = shap_values[:, feature_index]
            
            # Create scatter plot
            scatter = ax.scatter(feature_values, shap_feature_values, 
                               alpha=0.7, s=50, c=shap_feature_values, 
                               cmap='RdBu_r', edgecolors='black', linewidth=0.5)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='SHAP Value')
            
            # Set labels and title
            feature_name = feature_names[feature_index] if feature_names and len(feature_names) > feature_index else f"Feature {feature_index}"
            ax.set_xlabel(feature_name, fontsize=12)
            ax.set_ylabel(f"SHAP Value for {feature_name}", fontsize=12)
            ax.set_title(f"SHAP Dependence Plot: {feature_name}", fontsize=14, fontweight='bold')
            
            # Add grid
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SUCCESS: SHAP dependence plot saved to: {save_path}")
        
        return fig
        
    except Exception as e:
        print(f"Error generating SHAP dependence plot: {e}")
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


def generate_shap_summary_plot_from_values(shap_values, X_sample, feature_names=None):
    """Generate SHAP summary plot from cached SHAP values"""
    try:
        import shap
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # Handle numpy array with shape (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
            if shap_values.shape[0] == 2:  # Binary classification: (2, n_samples, n_features)
                shap_values = shap_values[1]  # Use positive class
            elif shap_values.shape[2] == 2:  # Binary classification: (n_samples, n_features, 2)
                shap_values = shap_values[:, :, 1]  # Use positive class
            else:
                shap_values = shap_values[0]  # Use first class for multi-class
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title("SHAP Summary Plot (from cache)")
        plt.tight_layout()
        return plt.gcf()
        
    except Exception as e:
        print(f"Error generating summary plot from cached values: {e}")
        return None

def generate_shap_bar_plot_from_values(shap_values, X_sample, feature_names=None):
    """Generate SHAP bar plot from cached SHAP values"""
    try:
        import shap
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # Handle numpy array with shape (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
            if shap_values.shape[0] == 2:  # Binary classification: (2, n_samples, n_features)
                shap_values = shap_values[1]  # Use positive class
            elif shap_values.shape[2] == 2:  # Binary classification: (n_samples, n_features, 2)
                shap_values = shap_values[:, :, 1]  # Use positive class
            else:
                shap_values = shap_values[0]  # Use first class for multi-class
        
        plt.figure(figsize=(10, 6))
        # Use summary_plot with plot_type="bar" instead of bar_plot
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
        plt.title("SHAP Bar Plot (from cache)")
        plt.tight_layout()
        return plt.gcf()
        
    except Exception as e:
        print(f"Error generating bar plot from cached values: {e}")
        return None

def generate_shap_dependence_plot_from_values(shap_values, X_sample, feature_names=None, feature_index=0):
    """Generate SHAP dependence plot from cached SHAP values"""
    try:
        import shap
        import matplotlib.pyplot as plt
        
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification
        
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feature_index, 
            shap_values, 
            X_sample, 
            feature_names=feature_names,
            show=False
        )
        plt.title(f"SHAP Dependence Plot - {feature_names[feature_index] if feature_names else f'Feature {feature_index}'} (from cache)")
        plt.tight_layout()
        return plt.gcf()
        
    except Exception as e:
        print(f"Error generating dependence plot from cached values: {e}")
        return None

def plot_shap_waterfall_from_values(shap_values, X_sample, instance_index=0, feature_names=None):
    """Generate SHAP waterfall plot from cached SHAP values"""
    try:
        import shap
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # Handle numpy array with shape (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
            if shap_values.shape[0] == 2:  # Binary classification: (2, n_samples, n_features)
                shap_values = shap_values[1]  # Use positive class
            elif shap_values.shape[2] == 2:  # Binary classification: (n_samples, n_features, 2)
                shap_values = shap_values[:, :, 1]  # Use positive class
            else:
                shap_values = shap_values[0]  # Use first class for multi-class
        
        # Ensure shap_values is numpy array
        if not isinstance(shap_values, np.ndarray):
            shap_values = np.array(shap_values)
        
        # Get SHAP values for single instance
        if len(shap_values.shape) >= 2:
            instance_values = shap_values[instance_index]
        else:
            instance_values = shap_values
        
        plt.figure(figsize=(12, 8))
        
        # Create Explanation object for waterfall plot
        try:
            # Calculate expected value (mean of all SHAP values)
            if len(shap_values.shape) >= 2:
                expected_value = np.mean(shap_values)
            else:
                expected_value = np.mean(shap_values)
            
            # Ensure instance_values is 1D
            if len(instance_values.shape) > 1:
                instance_values = instance_values.flatten()
            
            # Create Explanation object for waterfall plot
            explanation = shap.Explanation(
                values=instance_values,
                base_values=expected_value,
                data=X_sample[instance_index] if len(X_sample) > instance_index else X_sample[0],
                feature_names=feature_names
            )
            shap.waterfall_plot(explanation, show=False)
        except Exception as wf_error:
            print(f"Warning: Waterfall plot failed, using bar plot instead: {wf_error}")
            # Fallback to bar plot
            try:
                shap.bar_plot(instance_values, feature_names=feature_names, show=False)
            except Exception as bar_error:
                print(f"Warning: Bar plot also failed: {bar_error}")
                # Final fallback to simple bar plot
                plt.bar(range(len(instance_values)), instance_values)
                if feature_names:
                    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
        
        plt.title(f"SHAP Waterfall Plot - Instance {instance_index} (from cache)")
        plt.tight_layout()
        return plt.gcf()
        
    except Exception as e:
        print(f"Error generating waterfall plot from cached values: {e}")
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
        
        # Get SHAP values for single instance using definitive method
        shap_values, shap_type = get_shap_values_definitive(explainer, X_sample[instance_index:instance_index+1])
        if shap_values is None:
            return None
        
        # Ensure we have the right shape for single instance
        if len(shap_values.shape) > 1:
            instance_values = shap_values[0]
        else:
            instance_values = shap_values
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        
        # Create Explanation object for waterfall plot
        try:
            # Get expected value
            if hasattr(explainer, 'expected_value'):
                expected_value = explainer.expected_value
            else:
                # For Explainer objects, calculate expected value
                expected_value = np.mean(shap_values)
            
            # Ensure instance_values is 1D
            if len(instance_values.shape) > 1:
                instance_values = instance_values.flatten()
            
            # Create Explanation object for waterfall plot
            explanation = shap.Explanation(
                values=instance_values,
                base_values=expected_value,
                data=X_sample[instance_index] if len(X_sample) > instance_index else X_sample[0],
                feature_names=feature_names
            )
            shap.waterfall_plot(explanation, show=False)
        except Exception as wf_error:
            print(f"Warning: Waterfall plot failed, using bar plot instead: {wf_error}")
            # Fallback to bar plot
            try:
                shap.bar_plot(instance_values, feature_names=feature_names, show=False)
            except Exception as bar_error:
                print(f"Warning: Bar plot also failed: {bar_error}")
                # Final fallback to simple bar plot
                plt.bar(range(len(instance_values)), instance_values)
                if feature_names:
                    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
        
        plt.title(f"SHAP Waterfall Plot - Instance {instance_index}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SUCCESS: SHAP waterfall plot saved to: {save_path}")
        
        return plt.gcf()
        
    except Exception as e:
        print(f"Error generating SHAP waterfall plot: {e}")
        return None