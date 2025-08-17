# =========================================
# Academic Paper Classification System - Streamlit App
# Main application interface
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.cache_manager import load_cached_dataset, get_cache_info
from models.model_evaluator import ModelEvaluator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time


# Page configuration
st.set_page_config(
    page_title="Academic Paper Classification System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = []
if 'sample_data_loaded' not in st.session_state:
    st.session_state.sample_data_loaded = False
if 'cache_status' not in st.session_state:
    st.session_state.cache_status = None


def _format_authors(authors_parsed):
    """
    Format authors_parsed field from dataset.
    
    Args:
        authors_parsed: List of author information from dataset
        
    Returns:
        Formatted string of authors
    """
    if not authors_parsed:
        return 'N/A'
    
    try:
        # Handle different formats of authors_parsed
        if isinstance(authors_parsed, list):
            if len(authors_parsed) > 0 and isinstance(authors_parsed[0], list):
                # authors_parsed is list of lists (e.g., [['Smith', 'John'], ['Doe', 'Jane']])
                formatted_authors = []
                for author in authors_parsed:
                    if isinstance(author, list) and len(author) >= 2:
                        # Format as "Last, First"
                        formatted_authors.append(f"{author[0]}, {author[1]}")
                    elif isinstance(author, list) and len(author) == 1:
                        formatted_authors.append(author[0])
                    else:
                        formatted_authors.append(str(author))
                return '; '.join(formatted_authors)
            else:
                # authors_parsed is simple list
                return '; '.join(str(author) for author in authors_parsed)
        else:
            return str(authors_parsed)
    except Exception:
        return 'N/A'


def load_sample_data_from_cache(dataset_name: str, sample_size: int = 100):
    """Load sample data from cached dataset."""
    try:
        # Load cached dataset
        dataset = load_cached_dataset(dataset_name, split="train")
        
        if dataset is None:
            st.error("❌ Failed to load dataset from cache")
            return None
        
        # Handle dataset with splits
        if hasattr(dataset, 'keys') and 'train' in dataset:
            # Dataset has splits, access the train split
            dataset = dataset['train']
        
        # Take a small sample for demo
        sample_size = min(sample_size, len(dataset))
        sample_data = dataset.select(range(sample_size))
        
        # Convert to list of dictionaries
        data_list = []
        for item in sample_data:
            data_list.append({
                'title': item.get('title', 'N/A'),
                'authors': [
                    _format_authors(item.get('authors_parsed', []))
                    for item in [item]
                ][0],  # Get the first (and only) item
                'abstract': item.get('abstract', 'N/A'),
                'categories': item.get('categories', 'N/A')
            })
        
        return data_list
        
    except Exception as e:
        st.error(f"❌ Error loading sample data: {e}")
        return None


def create_sample_classification_data(n_samples: int = 1000):
    """Create sample data for classification testing."""
    # Create sample text data (simulating academic paper abstracts)
    sample_texts = [
        "Machine learning algorithms for healthcare applications",
        "Quantum computing in cryptography and optimization",
        "Economic impact of climate change on global markets",
        "Neural networks for image recognition and classification",
        "Statistical analysis of social media data",
        "Biomedical engineering advances in prosthetics",
        "Chemical synthesis of novel pharmaceutical compounds",
        "Mathematical modeling of population dynamics",
        "Computer vision applications in autonomous vehicles",
        "Environmental science and sustainability research"
    ]
    
    # Create more diverse samples by repeating and modifying
    X = []
    y = []
    
    # Define some sample categories
    categories = ['Computer Science', 'Physics', 'Economics', 'Biology', 'Chemistry']
    
    for i in range(n_samples):
        # Select base text and modify slightly
        base_text = sample_texts[i % len(sample_texts)]
        modified_text = base_text + f" - Research {i} - " + " ".join([f"term{j}" for j in range(i % 5 + 1)])
        X.append(modified_text)
        
        # Assign category based on text content
        if 'machine learning' in modified_text.lower() or 'neural' in modified_text.lower():
            y.append('Computer Science')
        elif 'quantum' in modified_text.lower() or 'physics' in modified_text.lower():
            y.append('Physics')
        elif 'economic' in modified_text.lower() or 'market' in modified_text.lower():
            y.append('Economics')
        elif 'biomedical' in modified_text.lower() or 'biology' in modified_text.lower():
            y.append('Biology')
        elif 'chemical' in modified_text.lower() or 'synthesis' in modified_text.lower():
            y.append('Chemistry')
        else:
            y.append('Computer Science')  # Default
    
    return X, y


def initialize_models():
    """Initialize classification models."""
    models = {
        'SVM': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=300)
    }
    return models


def get_parameter_grids():
    """Get parameter grids for hyperparameter tuning."""
    param_grids = {
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        },
        'Naive Bayes': {
            'alpha': [0.1, 0.5, 1.0, 2.0]
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50, 25), (100, 50), (100, 100)],
            'learning_rate_init': [0.001, 0.01, 0.1]
        }
    }
    return param_grids


def plot_performance_comparison(evaluator):
    """Create interactive performance comparison plots using Plotly."""
    if not evaluator.evaluation_results:
        st.warning("No evaluation results available for plotting")
        return
    
    # Extract data for plotting
    model_names = []
    accuracies = []
    f1_scores = []
    training_times = []
    
    for model_name, results in evaluator.evaluation_results.items():
        if 'metrics' in results:
            model_names.append(model_name)
            metrics = results['metrics']
            accuracies.append(metrics['accuracy'])
            f1_scores.append(metrics['f1_macro'])
            training_times.append(metrics['training_time'])
    
    if not model_names:
        st.warning("No metrics available for plotting")
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy Comparison', 'F1-Score Comparison', 
                       'Training Time Comparison', 'Performance Radar Chart'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "polar"}]]
    )
    
    # 1. Accuracy comparison
    fig.add_trace(
        go.Bar(x=model_names, y=accuracies, name='Accuracy', 
               marker_color='skyblue', opacity=0.7),
        row=1, col=1
    )
    
    # 2. F1-Score comparison
    fig.add_trace(
        go.Bar(x=model_names, y=f1_scores, name='F1-Score', 
               marker_color='lightgreen', opacity=0.7),
        row=1, col=2
    )
    
    # 3. Training time comparison
    fig.add_trace(
        go.Bar(x=model_names, y=training_times, name='Training Time', 
               marker_color='salmon', opacity=0.7),
        row=2, col=1
    )
    
    # 4. Radar chart for multiple metrics
    if len(model_names) > 0:
        # Prepare data for radar chart
        metrics_data = []
        for model_name in model_names:
            results = evaluator.evaluation_results[model_name]
            if 'metrics' in results:
                metrics = results['metrics']
                metrics_data.append([
                    metrics['accuracy'],
                    metrics['precision_macro'],
                    metrics['recall_macro'],
                    metrics['f1_macro']
                ])
        
        if metrics_data:
            categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            for i, (model_name, metrics) in enumerate(zip(model_names, metrics_data)):
                # Complete the circle for radar chart
                metrics_complete = metrics + [metrics[0]]
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                angles += [angles[0]]  # Complete the circle
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=metrics_complete,
                        theta=angles,
                        fill='toself',
                        name=model_name,
                        line=dict(width=2)
                    ),
                    row=2, col=2
                )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Model Performance Comparison",
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Models", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="Models", row=1, col=2)
    fig.update_yaxes(title_text="F1-Score", range=[0, 1], row=1, col=2)
    fig.update_xaxes(title_text="Models", row=2, col=1)
    fig.update_yaxes(title_text="Training Time (seconds)", row=2, col=1)
    
    # Update polar chart
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application function."""
    
    # Header
    st.title("📚 Academic Paper Classification System")
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Cache Management", "🔍 Sample Data", "🤖 Model Evaluation"])
    
    with tab1:
        st.header("📊 Cache Management")
        
        # Cache info
        cache_info = get_cache_info()
        if cache_info:
            st.subheader("Cache Status")
            
            # Display cache overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Cache Directory", cache_info.get('cache_directory', 'N/A'))
                st.metric("Total Datasets", cache_info.get('total_datasets', 0))
            
            with col2:
                cache_size_mb = cache_info.get('cache_size_mb', 0)
                st.metric("Cache Size", f"{cache_size_mb:.1f} MB")
                st.metric("Cache Size (GB)", f"{cache_size_mb/1024:.2f} GB")
            
            with col3:
                st.metric("Status", "✅ Active")
                st.metric("Last Updated", "Just now")
            
            # Display cached datasets
            st.subheader("📚 Cached Datasets")
            
            if 'cached_datasets' in cache_info and cache_info['cached_datasets']:
                for dataset_name in cache_info['cached_datasets']:
                    with st.expander(f"📖 {dataset_name}", expanded=False):
                        dataset_info = cache_info.get('cache_info', {}).get(dataset_name, {})
                        
                        if dataset_info:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Dataset Name:** {dataset_info.get('dataset_name', 'N/A')}")
                                st.write(f"**Cached At:** {dataset_info.get('cached_at', 'N/A')}")
                                st.write(f"**Dataset Size:** {dataset_info.get('dataset_size', 0):,} papers")
                            
                            with col2:
                                features = dataset_info.get('features', [])
                                st.write(f"**Features:** {len(features)}")
                                st.write(f"**Splits:** {len(dataset_info.get('split_names', []))}")
                                
                                # Show first few features
                                if features:
                                    st.write("**Feature List:**")
                                    for i, feature in enumerate(features[:5]):  # Show first 5
                                        st.write(f"  • {feature}")
                                    if len(features) > 5:
                                        st.write(f"  • ... and {len(features) - 5} more")
                        else:
                            st.info("No detailed information available for this dataset")
            else:
                st.info("No datasets found in cache")
            
            # Cache management actions
            st.subheader("🔧 Cache Management")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Refresh Cache Info", type="secondary"):
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear Cache", type="secondary"):
                    st.warning("Cache clearing functionality not implemented yet")
            
            with col3:
                if st.button("📊 Export Cache Info", type="secondary"):
                    # Create a formatted cache report
                    cache_report = {
                        "summary": {
                            "cache_directory": cache_info.get('cache_directory'),
                            "total_datasets": cache_info.get('total_datasets'),
                            "cache_size_mb": cache_info.get('cache_size_mb'),
                            "timestamp": pd.Timestamp.now().isoformat()
                        },
                        "datasets": cache_info.get('cache_info', {})
                    }
                    
                    # Convert to JSON string
                    import json
                    cache_json = json.dumps(cache_report, indent=2, default=str)
                    
                    st.download_button(
                        label="📥 Download Cache Report",
                        data=cache_json,
                        file_name=f"cache_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        else:
            st.warning("No cache information available")
            st.info("💡 Try loading a dataset first to initialize cache")
        
        # Dataset operations
        st.subheader("Dataset Operations")
        dataset_name = st.text_input(
            "Dataset Name", 
            value="UniverseTBD/arxiv-abstracts-large",
            help="Enter the HuggingFace dataset name"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Load Dataset"):
                with st.spinner("Loading dataset..."):
                    dataset = load_cached_dataset(dataset_name)
                    if dataset:
                        st.success(f"✅ Dataset loaded successfully!")
                        
                        # Handle dataset with splits
                        if hasattr(dataset, 'keys') and 'train' in dataset:
                            dataset = dataset['train']
                        
                        st.info(f"Dataset size: {len(dataset):,} papers")
                        
                        # Show sample info
                        if len(dataset) > 0:
                            sample = dataset[0]
                            st.write("**Sample Paper:**")
                            st.write(f"Title: {sample.get('title', 'N/A')[:100]}...")
                            st.write(f"Categories: {sample.get('categories', 'N/A')}")
                    else:
                        st.error("❌ Failed to load dataset")
        
        with col2:
            if st.button("📊 Dataset Info"):
                with st.spinner("Getting dataset information..."):
                    dataset = load_cached_dataset(dataset_name)
                    if dataset:
                        # Handle dataset with splits
                        if hasattr(dataset, 'keys') and 'train' in dataset:
                            dataset = dataset['train']
                        
                        st.write("**Dataset Information:**")
                        st.write(f"• Total papers: {len(dataset):,}")
                        
                        if len(dataset) > 0:
                            sample = dataset[0]
                            features = list(sample.keys())
                            st.write(f"• Features: {len(features)}")
                            st.write(f"• Feature names: {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")
                            
                            # Show categories distribution if available
                            if 'categories' in sample:
                                categories = [item.get('categories', 'Unknown') for item in dataset.select(range(min(1000, len(dataset))))]
                                unique_categories = set()
                                for cat in categories:
                                    if cat:
                                        # Split multiple categories
                                        cat_list = cat.split()
                                        unique_categories.update(cat_list)
                                
                                st.write(f"• Unique categories: {len(unique_categories)}")
                                st.write(f"• Sample categories: {', '.join(list(unique_categories)[:10])}{'...' if len(unique_categories) > 10 else ''}")
                    else:
                        st.error("❌ Failed to load dataset")
    
    with tab2:
        st.header("🔍 Sample Data Viewer")
        
        # Dataset selection for sample data
        sample_dataset = st.text_input(
            "Dataset for Sample Data", 
            value="UniverseTBD/arxiv-abstracts-large",
            help="Enter dataset name to view sample data"
        )
        
        sample_size = st.slider("Sample Size", min_value=10, max_value=200, value=50)
        
        if st.button("📖 Load Sample Data"):
            with st.spinner("Loading sample data..."):
                sample_data = load_sample_data_from_cache(sample_dataset, sample_size)
                
                if sample_data:
                    st.success(f"✅ Loaded {len(sample_data)} sample papers")
                    
                    # Display as DataFrame
                    df = pd.DataFrame(sample_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Show statistics
                    st.subheader("📊 Sample Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Papers", len(sample_data))
                        st.metric("Categories", df['categories'].nunique())
                    
                    with col2:
                        st.metric("Avg Title Length", df['title'].str.len().mean().round(1))
                        st.metric("Avg Abstract Length", df['abstract'].str.len().mean().round(1))
                    
                    # Category distribution
                    if df['categories'].nunique() > 1:
                        st.subheader("🏷️ Category Distribution")
                        category_counts = df['categories'].value_counts()
                        fig = px.bar(x=category_counts.index, y=category_counts.values, 
                                   title="Paper Categories Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ Failed to load sample data")
    
    with tab3:
        st.header("🤖 Model Evaluation & Performance Analysis")
        st.markdown("Comprehensive evaluation of classification models with cross-validation and hyperparameter tuning")
        
        # Configuration
        st.subheader("⚙️ Evaluation Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            data_source = st.selectbox(
                "Data Source",
                ["Sample Data", "Real Dataset"],
                help="Choose between synthetic sample data or real academic papers"
            )
        
        with col2:
            n_samples = st.slider(
                "Number of Samples", 
                min_value=100, 
                max_value=2000, 
                value=500,
                help="Number of samples for training and evaluation"
            )
        
        with col3:
            cv_folds = st.slider(
                "Cross-Validation Folds", 
                min_value=3, 
                max_value=10, 
                value=5,
                help="Number of CV folds for evaluation"
            )
        
        # Model selection
        st.subheader("🤖 Model Selection")
        available_models = ['SVM', 'Random Forest', 'Naive Bayes', 'Neural Network']
        selected_models = st.multiselect(
            "Select Models to Evaluate",
            available_models,
            default=['SVM', 'Random Forest', 'Naive Bayes'],
            help="Choose which models to evaluate"
        )
        
        # Evaluation options
        col1, col2 = st.columns(2)
        
        with col1:
            perform_cv = st.checkbox("Perform Cross-Validation", value=True)
            perform_tuning = st.checkbox("Perform Hyperparameter Tuning", value=False)
        
        with col2:
            save_results = st.checkbox("Save Results", value=True)
            show_plots = st.checkbox("Show Performance Plots", value=True)
        
        # Start evaluation
        if st.button("🚀 Start Model Evaluation", type="primary"):
            if not selected_models:
                st.error("❌ Please select at least one model to evaluate")
                return
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Prepare data
                status_text.text("🔧 Preparing data...")
                progress_bar.progress(20)
                
                if data_source == "Sample Data":
                    X, y = create_sample_classification_data(n_samples)
                    st.info(f"📊 Using synthetic data: {len(X)} samples, {len(set(y))} categories")
                else:
                    # Load real dataset
                    dataset = load_cached_dataset("UniverseTBD/arxiv-abstracts-large", split="train")
                    if dataset is None:
                        st.error("❌ Failed to load real dataset")
                        return
                    
                    # Handle dataset with splits
                    if hasattr(dataset, 'keys') and 'train' in dataset:
                        dataset = dataset['train']
                    
                    # Extract text and categories
                    sample_data = dataset.select(range(min(n_samples, len(dataset))))
                    texts = [item.get('abstract', '') for item in sample_data]
                    categories = [item.get('categories', 'Unknown') for item in sample_data]
                    
                    # Filter and process
                    valid_data = [(text, cat) for text, cat in zip(texts, categories) if text.strip()]
                    if len(valid_data) < 100:
                        st.error("❌ Insufficient valid data for evaluation")
                        return
                    
                    X, y = zip(*valid_data[:n_samples])
                    y = [cat.split()[0] if cat and ' ' in cat else cat for cat in y]
                    st.info(f"📊 Using real data: {len(X)} samples, {len(set(y))} categories")
                
                # Step 2: Vectorize text
                status_text.text("🔤 Vectorizing text data...")
                progress_bar.progress(40)
                
                vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
                X_vectorized = vectorizer.fit_transform(X)
                
                # Step 3: Split data
                status_text.text("✂️ Splitting data...")
                progress_bar.progress(60)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Step 4: Initialize models
                status_text.text("🤖 Initializing models...")
                progress_bar.progress(80)
                
                all_models = initialize_models()
                models = {name: all_models[name] for name in selected_models if name in all_models}
                
                # Step 5: Create evaluator
                status_text.text("📊 Creating model evaluator...")
                progress_bar.progress(90)
                
                evaluator = ModelEvaluator()
                evaluator.cv_folds = cv_folds
                
                # Step 6: Evaluate models
                status_text.text("🔍 Evaluating models...")
                progress_bar.progress(95)
                
                for i, (model_name, model) in enumerate(models.items()):
                    st.write(f"📈 Evaluating {model_name}...")
                    
                    # Evaluate performance
                    evaluator.evaluate_model_performance(
                        model_name, model, X_train, y_train, X_test, y_test
                    )
                    
                    # Cross-validation
                    if perform_cv:
                        cv_results = evaluator.cross_validate_model(model_name, model, X_train, y_train)
                        st.write(f"   CV Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
                        st.write(f"   CV F1-Score: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
                    
                    # Hyperparameter tuning
                    if perform_tuning:
                        param_grids = get_parameter_grids()
                        if model_name in param_grids:
                            st.write(f"🔧 Tuning {model_name} hyperparameters...")
                            tuning_results = evaluator.hyperparameter_tuning(
                                model_name, model, param_grids[model_name], X_train, y_train, method='grid'
                            )
                            st.write(f"   Best parameters: {tuning_results['best_params']}")
                            st.write(f"   Best CV score: {tuning_results['best_score']:.4f}")
                
                progress_bar.progress(100)
                status_text.text("✅ Evaluation completed!")
                
                # Display results
                st.success("🎉 Model evaluation completed successfully!")
                
                # Performance report
                st.subheader("📊 Performance Report")
                performance_report = evaluator.generate_performance_report(save_to_file=save_results)
                st.dataframe(performance_report, use_container_width=True)
                
                # Best model analysis
                st.subheader("🏆 Best Model Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    best_model_name, best_model = evaluator.get_best_model(metric='f1_macro')
                    st.metric("Best Model (F1-Score)", best_model_name or "N/A")
                
                with col2:
                    best_model_name_acc, _ = evaluator.get_best_model(metric='accuracy')
                    st.metric("Best Model (Accuracy)", best_model_name_acc or "N/A")
                
                # Performance plots
                if show_plots:
                    st.subheader("📈 Performance Visualization")
                    plot_performance_comparison(evaluator)
                
                # Save results
                if save_results:
                    results_file = evaluator.save_evaluation_results()
                    st.info(f"💾 Evaluation results saved to: {results_file}")
                
            except Exception as e:
                st.error(f"❌ Error during evaluation: {e}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
