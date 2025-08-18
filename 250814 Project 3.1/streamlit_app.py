"""
Streamlit Web Interface for Topic Modeling Project
Provides user-friendly interface for dataset input, auto classifier selection, 
and result visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# Import project modules
try:
    from data_loader import DataLoader
    from text_encoders import TextVectorizer
    from models import ModelTrainer
    from config import MAX_SAMPLES, TEST_SIZE, RANDOM_STATE
except ImportError as e:
    st.error(f"Error importing project modules: {e}")
    st.info("Please ensure all project files are in the same directory")

# Page configuration
st.set_page_config(
    page_title="Topic Modeling - Auto Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .column-selector {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitTopicModeling:
    """Main class for Streamlit Topic Modeling interface"""
    
    def __init__(self):
        self.data_loader = None
        self.text_vectorizer = None
        self.model_trainer = None
        self.results = {}
        self.current_dataset = None
        self.selected_text_column = None
        self.selected_label_column = None
        
    def initialize_components(self):
        """Initialize project components"""
        try:
            self.data_loader = DataLoader()
            self.text_vectorizer = TextVectorizer()
            self.model_trainer = ModelTrainer()
            return True
        except Exception as e:
            st.error(f"Failed to initialize components: {e}")
            return False
    
    def load_dataset_from_file(self, uploaded_file):
        """Load dataset from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV, JSON, or Excel file.")
                return None
            
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    
    def load_dataset_from_path(self, file_path: str):
        """Load dataset from local file path"""
        try:
            import os
            from pathlib import Path
            
            # Validate file path
            path = Path(file_path.strip())
            if not path.exists():
                st.error(f"❌ File not found: {file_path}")
                return None
            
            if not path.is_file():
                st.error(f"❌ Path is not a file: {file_path}")
                return None
            
            # Check file size (warn if too large)
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > 100:  # 100MB limit
                st.warning(f"⚠️ Large file detected: {file_size_mb:.1f} MB. Loading may take time...")
            
            # Load based on file extension
            file_ext = path.suffix.lower()
            
            # Add debug info
            st.write(f"🔍 Debug: Loading file with extension: {file_ext}")
            
            if file_ext == '.csv':
                df = pd.read_csv(path, encoding='utf-8')
            elif file_ext == '.json':
                df = pd.read_json(path)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(path)
            elif file_ext == '.parquet':
                df = pd.read_parquet(path)
            elif file_ext == '.pickle' or file_ext == '.pkl':
                df = pd.read_pickle(path)
            else:
                st.error(f"❌ Unsupported file format: {file_ext}")
                st.info("Supported formats: CSV, JSON, Excel (.xlsx/.xls), Parquet, Pickle")
                return None
            
            # Debug DataFrame info
            st.write(f"🔍 Debug: DataFrame shape: {df.shape}")
            st.write(f"🔍 Debug: DataFrame columns: {list(df.columns)}")
            st.write(f"🔍 Debug: DataFrame dtypes: {df.dtypes.to_dict()}")
            
            # Ensure DataFrame has proper column names (no special characters)
            df.columns = [str(col).strip() for col in df.columns]
            
            # Basic data cleaning
            df = df.dropna(how='all')  # Remove completely empty rows
            df = df.loc[:, df.columns.notnull()]  # Remove columns with null names
            
            st.write(f"🔍 Debug: After cleaning - DataFrame shape: {df.shape}")
            st.write(f"🔍 Debug: After cleaning - DataFrame columns: {list(df.columns)}")
            
            if df.empty:
                st.error("❌ Dataset is empty after loading!")
                return None
            
            st.success(f"✅ File loaded successfully from: {path}")
            st.info(f"📊 File size: {file_size_mb:.1f} MB")
            st.info(f"📊 Dataset shape: {df.shape}")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error loading file from path: {e}")
            st.write("🔧 Debug info:")
            import traceback
            st.code(traceback.format_exc())
            st.info("💡 Make sure the path is correct and the file is accessible")
            return None
    
    def load_dataset_from_url(self, url):
        """Load dataset from URL"""
        try:
            if url.endswith('.csv'):
                df = pd.read_csv(url)
            elif url.endswith('.json'):
                df = pd.read_json(url)
            else:
                st.error("URL must point to a CSV or JSON file")
                return None
            
            return df
        except Exception as e:
            st.error(f"Error loading from URL: {e}")
            return None
    
    def select_dataset_columns(self, df):
        """Allow user to select text and label columns"""
        try:
            st.markdown("""
            <div class="column-selector">
                <h4>🔧 Column Selection</h4>
                <p>Select which columns to use for text classification:</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Get all columns
            all_columns = list(df.columns)
            st.write(f"🔍 Available columns: {all_columns}")
            
            # Simple column selection without complex session state
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📝 Text Column")
                text_col = st.selectbox(
                    "Select text column:",
                    options=all_columns,
                    index=0,
                    key="simple_text_selector"
                )
                
                # Show preview
                if text_col and text_col in df.columns:
                    st.write("**Preview:**")
                    preview_data = df[text_col].head(3).tolist()
                    st.write(preview_data)
            
            with col2:
                st.subheader("🏷️ Label Column")
                label_col = st.selectbox(
                    "Select label column:",
                    options=all_columns,
                    index=1 if len(all_columns) > 1 else 0,
                    key="simple_label_selector"
                )
                
                # Show preview
                if label_col and label_col in df.columns:
                    st.write("**Preview:**")
                    unique_labels = df[label_col].unique()
                    st.write(f"Unique labels: {unique_labels[:5].tolist()}")
                    if len(unique_labels) > 5:
                        st.write(f"... and {len(unique_labels) - 5} more")
            
            # Simple validation
            if text_col == label_col:
                st.error("❌ Text and label columns must be different!")
                return False
            
            # Update instance variables directly
            self.selected_text_column = text_col
            self.selected_label_column = label_col
            
            # Show current selection
            st.success("✅ Columns selected successfully!")
            st.info(f"📝 Text Column: {text_col}")
            st.info(f"🏷️ Label Column: {label_col}")
            
            return True
            
        except Exception as e:
            st.error(f"Error in column selection: {e}")
            import traceback
            st.code(traceback.format_exc())
            return False
    
    def preprocess_dataset(self, df):
        """Preprocess the dataset with selected columns"""
        try:
            # Create new dataframe with selected columns
            processed_df = pd.DataFrame({
                'text': df[self.selected_text_column],
                'label': df[self.selected_label_column]
            })
            
            # Basic preprocessing
            processed_df = processed_df.dropna(subset=['text', 'label'])
            processed_df['text'] = processed_df['text'].astype(str).str.lower()
            processed_df['label'] = processed_df['label'].astype(str)
            
            # Limit samples if too many
            if len(processed_df) > MAX_SAMPLES:
                processed_df = processed_df.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE)
                st.warning(f"Dataset limited to {MAX_SAMPLES} samples for performance")
            
            return processed_df
        except Exception as e:
            st.error(f"Error preprocessing dataset: {e}")
            return None
    
    def run_auto_classification(self, df):
        """Run automatic classification with all methods"""
        try:
            # Prepare data
            X = df['text'].tolist()
            y = df['label'].tolist()
            
            # Create label mappings
            unique_labels = sorted(list(set(y)))
            label_to_id = {label: i for i, label in enumerate(unique_labels)}
            y_encoded = [label_to_id[label] for label in y]
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=TEST_SIZE, 
                random_state=RANDOM_STATE, stratify=y_encoded
            )
            
            # Text vectorization
            st.info("Processing text vectorization...")
            
            # BoW
            with st.spinner("Processing Bag of Words..."):
                X_train_bow = self.text_vectorizer.fit_transform_bow(X_train)
                X_test_bow = self.text_vectorizer.transform_bow(X_test)
            
            # TF-IDF
            with st.spinner("Processing TF-IDF..."):
                X_train_tfidf = self.text_vectorizer.fit_transform_tfidf(X_train)
                X_test_tfidf = self.text_vectorizer.transform_tfidf(X_test)
            
            # Embeddings
            with st.spinner("Processing Word Embeddings..."):
                X_train_embeddings = self.text_vectorizer.transform_embeddings(X_train)
                X_test_embeddings = self.text_vectorizer.transform_embeddings(X_test)
            
            # Train and evaluate all models
            st.info("Training and evaluating models...")
            results = {}
            
            # K-Means
            with st.spinner("Training K-Means..."):
                km_bow_labels, km_bow_accuracy, km_bow_report = (
                    self.model_trainer.train_and_test_kmeans(
                        X_train_bow, y_train, X_test_bow, y_test
                    )
                )
                results['kmeans_bow'] = {
                    'accuracy': km_bow_accuracy,
                    'predictions': km_bow_labels,
                    'report': km_bow_report,
                    'vectorizer': 'BoW'
                }
                
                km_tfidf_labels, km_tfidf_accuracy, km_tfidf_report = (
                    self.model_trainer.train_and_test_kmeans(
                        X_train_tfidf, y_train, X_test_tfidf, y_test
                    )
                )
                results['kmeans_tfidf'] = {
                    'accuracy': km_tfidf_accuracy,
                    'predictions': km_tfidf_labels,
                    'report': km_tfidf_report,
                    'vectorizer': 'TF-IDF'
                }
                
                km_emb_labels, km_emb_accuracy, km_emb_report = (
                    self.model_trainer.train_and_test_kmeans(
                        X_train_embeddings, y_train, X_test_embeddings, y_test
                    )
                )
                results['kmeans_embeddings'] = {
                    'accuracy': km_emb_accuracy,
                    'predictions': km_emb_labels,
                    'report': km_emb_report,
                    'vectorizer': 'Embeddings'
                }
            
            # KNN
            with st.spinner("Training KNN..."):
                knn_bow_labels, knn_bow_accuracy, knn_bow_report = (
                    self.model_trainer.train_and_test_knn(
                        X_train_bow, y_train, X_test_bow, y_test
                    )
                )
                results['knn_bow'] = {
                    'accuracy': knn_bow_accuracy,
                    'predictions': knn_bow_labels,
                    'report': knn_bow_report,
                    'vectorizer': 'BoW'
                }
                
                knn_tfidf_labels, knn_tfidf_accuracy, knn_tfidf_report = (
                    self.model_trainer.train_and_test_knn(
                        X_train_tfidf, y_train, X_test_tfidf, y_test
                    )
                )
                results['knn_tfidf'] = {
                    'accuracy': knn_tfidf_accuracy,
                    'predictions': knn_tfidf_labels,
                    'report': knn_tfidf_report,
                    'vectorizer': 'TF-IDF'
                }
                
                knn_emb_labels, knn_emb_accuracy, knn_emb_report = (
                    self.model_trainer.train_and_test_knn(
                        X_train_embeddings, y_train, X_test_embeddings, y_test
                    )
                )
                results['knn_embeddings'] = {
                    'accuracy': knn_emb_accuracy,
                    'predictions': knn_emb_labels,
                    'report': knn_emb_report,
                    'vectorizer': 'Embeddings'
                }
            
            # Decision Tree
            with st.spinner("Training Decision Tree..."):
                dt_bow_labels, dt_bow_accuracy, dt_bow_report = (
                    self.model_trainer.train_and_test_decision_tree(
                        X_train_bow, y_train, X_test_bow, y_test
                    )
                )
                results['decision_tree_bow'] = {
                    'accuracy': dt_bow_accuracy,
                    'predictions': dt_bow_labels,
                    'report': dt_bow_report,
                    'vectorizer': 'BoW'
                }
                
                dt_tfidf_labels, dt_tfidf_accuracy, dt_tfidf_report = (
                    self.model_trainer.train_and_test_decision_tree(
                        X_train_tfidf, y_train, X_test_tfidf, y_test
                    )
                )
                results['decision_tree_tfidf'] = {
                    'accuracy': dt_tfidf_accuracy,
                    'predictions': dt_tfidf_labels,
                    'report': dt_tfidf_report,
                    'vectorizer': 'TF-IDF'
                }
                
                dt_emb_labels, dt_emb_accuracy, dt_emb_report = (
                    self.model_trainer.train_and_test_decision_tree(
                        X_train_embeddings, y_train, X_test_embeddings, y_test
                    )
                )
                results['decision_tree_embeddings'] = {
                    'accuracy': dt_emb_accuracy,
                    'predictions': dt_emb_labels,
                    'report': dt_emb_report,
                    'vectorizer': 'Embeddings'
                }
            
            # Naive Bayes
            with st.spinner("Training Naive Bayes..."):
                nb_bow_labels, nb_bow_accuracy, nb_bow_report = (
                    self.model_trainer.train_and_test_naive_bayes(
                        X_train_bow, y_train, X_test_bow, y_test
                    )
                )
                results['naive_bayes_bow'] = {
                    'accuracy': nb_bow_accuracy,
                    'predictions': nb_bow_labels,
                    'report': nb_bow_report,
                    'vectorizer': 'BoW'
                }
                
                nb_tfidf_labels, nb_tfidf_accuracy, nb_tfidf_report = (
                    self.model_trainer.train_and_test_naive_bayes(
                        X_train_tfidf, y_train, X_test_tfidf, y_test
                    )
                )
                results['nb_tfidf'] = {
                    'accuracy': nb_tfidf_accuracy,
                    'predictions': nb_tfidf_labels,
                    'report': nb_tfidf_report,
                    'vectorizer': 'TF-IDF'
                }
                
                nb_emb_labels, nb_emb_accuracy, nb_emb_report = (
                    self.model_trainer.train_and_test_naive_bayes(
                        X_train_embeddings, y_train, X_test_embeddings, y_test
                    )
                )
                results['naive_bayes_embeddings'] = {
                    'accuracy': nb_emb_accuracy,
                    'predictions': nb_emb_labels,
                    'report': nb_emb_report,
                    'vectorizer': 'Embeddings'
                }
            
            return results, y_test, unique_labels
            
        except Exception as e:
            st.error(f"Error in auto classification: {e}")
            return None, None, None
    
    def find_best_classifier(self, results):
        """Find the best performing classifier"""
        best_model = None
        best_accuracy = 0
        
        for model_name, result in results.items():
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model = model_name
        
        return best_model, best_accuracy
    
    def plot_confusion_matrix_plotly(self, y_true, y_pred, labels, title):
        """Create interactive confusion matrix using Plotly"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create annotations
        annotations = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                annotations.append(
                    dict(
                        text=f"{cm[i, j]}<br>({cm_normalized[i, j]:.2%})",
                        x=j, y=i,
                        xref='x', yref='y',
                        showarrow=False,
                        font=dict(color="white" if cm_normalized[i, j] > 0.5 else "black")
                    )
                )
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=600,
            height=500
        )
        
        return fig
    
    def create_performance_comparison(self, results):
        """Create performance comparison chart"""
        models = []
        accuracies = []
        vectorizers = []
        
        for model_name, result in results.items():
            if '_' in model_name:
                model_type = (model_name.split('_')[0] + ' ' + 
                            model_name.split('_')[1])
            else:
                model_type = model_name
            models.append(model_type)
            accuracies.append(result['accuracy'])
            vectorizers.append(result['vectorizer'])
        
        df_compare = pd.DataFrame({
            'Model': models,
            'Accuracy': accuracies,
            'Vectorizer': vectorizers
        })
        
        fig = px.bar(
            df_compare, 
            x='Model', 
            y='Accuracy',
            color='Vectorizer',
            title="Model Performance Comparison",
            barmode='group'
        )
        
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Accuracy",
            yaxis_range=[0, 1],
            width=800,
            height=500
        )
        
        return fig


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Topic Modeling - Auto Classifier</h1>', 
                unsafe_allow_html=True)
    
    # Initialize app
    app = StreamlitTopicModeling()
    
    # Initialize session state for dataset persistence
    if 'loaded_dataset' not in st.session_state:
        st.session_state.loaded_dataset = None
        st.session_state.dataset_source = None
    
    # Sidebar for dataset input
    st.sidebar.header("📁 Dataset Input")
    
    input_method = st.sidebar.selectbox(
        "Choose input method:",
        ["Upload File", "Enter File Path", "Enter URL", "Use Sample Dataset"]
    )
    
    dataset = st.session_state.loaded_dataset
    
    if input_method == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your dataset",
            type=['csv', 'json', 'xlsx'],
            help="Upload CSV, JSON, or Excel file with text and label columns"
        )
        
        if uploaded_file is not None:
            temp_dataset = app.load_dataset_from_file(uploaded_file)
            if temp_dataset is not None:
                st.session_state.loaded_dataset = temp_dataset
                st.session_state.dataset_source = f"Upload: {uploaded_file.name}"
                dataset = temp_dataset
                st.sidebar.success(f"✅ File loaded: {uploaded_file.name}")
                st.sidebar.info(f"📊 Dataset shape: {dataset.shape}")
    
    elif input_method == "Enter File Path":
        file_path = st.sidebar.text_input(
            "Enter local file path:",
            placeholder="C:/Users/Username/Documents/dataset.csv",
            help="Enter the full path to your dataset file on your local machine"
        )
        
        # Add path validation and suggestions
        if file_path:
            st.sidebar.info("💡 Path examples:")
            st.sidebar.code("Windows: C:\\Users\\Username\\Documents\\data.csv")
            st.sidebar.code("Mac/Linux: /home/username/documents/data.csv")
            st.sidebar.code("Relative: ./data/dataset.csv")
        
        if file_path and st.sidebar.button("Load from Path"):
            with st.spinner("Loading dataset from local path..."):
                temp_dataset = app.load_dataset_from_path(file_path)
                if temp_dataset is not None:
                    st.session_state.loaded_dataset = temp_dataset
                    st.session_state.dataset_source = f"File Path: {file_path}"
                    dataset = temp_dataset
                    st.sidebar.success("✅ Dataset loaded from local path")
                    st.sidebar.info(f"📊 Dataset shape: {dataset.shape}")
                    st.rerun()  # Force rerun to update main area
    
    elif input_method == "Enter URL":
        url = st.sidebar.text_input(
            "Enter dataset URL:",
            placeholder="https://example.com/dataset.csv",
            help="Enter URL to CSV or JSON file"
        )
        
        if url and st.sidebar.button("Load from URL"):
            with st.spinner("Loading dataset from URL..."):
                temp_dataset = app.load_dataset_from_url(url)
                if temp_dataset is not None:
                    st.session_state.loaded_dataset = temp_dataset
                    st.session_state.dataset_source = f"URL: {url}"
                    dataset = temp_dataset
                    st.sidebar.success("✅ Dataset loaded from URL")
                    st.sidebar.info(f"📊 Dataset shape: {dataset.shape}")
                    st.rerun()  # Force rerun to update main area
    
    elif input_method == "Use Sample Dataset":
        if st.sidebar.button("Load Sample Dataset"):
            st.sidebar.info("Loading ArXiv dataset...")
            if app.initialize_components():
                try:
                    app.data_loader.load_dataset()
                    app.data_loader.select_samples()
                    app.data_loader.preprocess_samples()
                    
                    # Convert to DataFrame
                    sample_data = app.data_loader.preprocessed_samples
                    temp_dataset = pd.DataFrame(sample_data)
                    
                    st.session_state.loaded_dataset = temp_dataset
                    st.session_state.dataset_source = "Sample: ArXiv Dataset"
                    dataset = temp_dataset
                    st.sidebar.success("✅ Sample dataset loaded")
                    st.sidebar.info(f"📊 Dataset shape: {dataset.shape}")
                    st.rerun()  # Force rerun to update main area
                except Exception as e:
                    st.sidebar.error(f"Error loading sample dataset: {e}")
    
    # Main content area
    if dataset is not None:
        # Dataset preview
        st.header("📊 Dataset Preview")
        if st.session_state.dataset_source:
            st.info(f"📁 Dataset Source: {st.session_state.dataset_source}")
        st.dataframe(dataset.head(10))
        st.info(f"Total samples: {len(dataset)} | Columns: {list(dataset.columns)}")
        
        # Debug info for dataset source
        st.write(f"🔍 Debug: Dataset from session state: {st.session_state.loaded_dataset is not None}")
        st.write(f"🔍 Debug: Dataset shape: {dataset.shape}")
        st.write(f"🔍 Debug: Dataset columns: {list(dataset.columns)}")
        st.write(f"🔍 Debug: Dataset types: {dataset.dtypes.to_dict()}")
        
        # Column selection - always run, don't depend on return value
        app.select_dataset_columns(dataset)
        
        # Check if both columns are selected and different
        if (hasattr(app, 'selected_text_column') and hasattr(app, 'selected_label_column') and
            app.selected_text_column and app.selected_label_column and
            app.selected_text_column != app.selected_label_column):
            
            # Show start classification button
            st.markdown("---")
            st.header("🚀 Ready to Start Classification")
            st.write("Click the button below to begin the automatic classification process:")
            
            if st.button("🚀 Start Auto Classification", type="primary", key="start_classification"):
                with st.spinner("Preprocessing dataset..."):
                    processed_dataset = app.preprocess_dataset(dataset)
                    
                    if processed_dataset is not None:
                        # Initialize components
                        if app.initialize_components():
                            # Run classification
                            st.info("🤖 Running automatic classification...")
                            results, y_test, labels = app.run_auto_classification(processed_dataset)
                            
                            if results:
                                # Find best classifier
                                best_model, best_accuracy = app.find_best_classifier(results)
                                
                                # Display results
                                st.success("🎉 Classification completed successfully!")
                                st.header("🏆 Best Classifier")
                                st.metric("Best Model", best_model.replace('_', ' ').title())
                                st.metric("Best Accuracy", f"{best_accuracy:.3f}")
                                
                                # Performance comparison
                                st.header("📈 Performance Comparison")
                                comparison_fig = app.create_performance_comparison(results)
                                st.plotly_chart(comparison_fig, use_container_width=True)
                            else:
                                st.error("❌ Classification failed!")
                        else:
                            st.error("❌ Failed to initialize components!")
                    else:
                        st.error("❌ Failed to preprocess dataset!")
        else:
            if hasattr(app, 'selected_text_column') and hasattr(app, 'selected_label_column'):
                if app.selected_text_column == app.selected_label_column:
                    st.warning("⚠️ Please select different columns for text and label.")
                else:
                    st.info("ℹ️ Please select both text and label columns to continue.")
            else:
                st.info("ℹ️ Please select columns above to start classification.")
    
    else:
        # Empty state - no dataset loaded
        pass


if __name__ == "__main__":
    main()
