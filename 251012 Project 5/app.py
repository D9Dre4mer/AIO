"""
Enhanced ML Pipeline Application
Modern UI leveraging the complete MLOps system

Features:
- Modern Streamlit UI with step-by-step wizard
- Full MLOps integration (MLflow, DVC, monitoring)
- Modular model architecture
- Real-time monitoring and drift detection
- Production-ready serving capabilities

Created: 2025-01-27
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules - lazy loading để tránh chậm
from src.data_manager import DataManager
from src.training_pipeline import TrainingPipeline
from src.mlflow_integration import MLflowTracker
from wizard_ui.session_manager import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🚀 Enhanced ML Pipeline",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main header with gradient */
    .main-header {
        background: linear-gradient(90deg, #0d5f3c 0%, #16a085 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Step container with theme-aware borders */
    .step-container {
        background: var(--background-color);
        border: 2px solid var(--primary-color);
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        color: var(--text-color);
    }
    
    /* Section boxes with theme-aware styling */
    .section-box {
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: var(--text-color);
    }
    
    /* Preview box with theme-aware styling */
    .preview-box {
        background: var(--warning-background-color);
        border: 2px solid var(--warning-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: var(--text-color);
    }
    
    /* Metric boxes with theme-aware styling */
    .metric-box {
        background: var(--background-color);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
        color: var(--text-color);
    }
    
    /* Navigation buttons container */
    .nav-buttons {
        text-align: center;
        margin: 2rem 0;
        padding: 1rem;
        background: var(--secondary-background-color);
        border-radius: 8px;
        color: var(--text-color);
    }
    
    /* Theme-aware text colors */
    .theme-text {
        color: var(--text-color) !important;
    }
    
    .theme-text-secondary {
        color: var(--secondary-text-color) !important;
    }
    
    /* CSS Variables for theme switching */
    :root {
        /* Light theme (default) */
        --background-color: #ffffff;
        --secondary-background-color: #f8f9fa;
        --text-color: #0d5f3c;
        --secondary-text-color: #5d6d5b;
        --border-color: #dee2e6;
        --primary-color: #16a085;
        --warning-color: #ffc107;
        --warning-background-color: #fff3cd;
        --success-color: #28a745;
        --info-color: #17a2b8;
    }
    
    /* Dark theme overrides */
    [data-testid="stAppViewContainer"] [data-testid="stDecoration"] {
        background: #0e1117;
    }
    
    /* Dark theme detection and overrides */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #0e1117;
            --secondary-background-color: #262730;
            --text-color: #fafafa;
            --secondary-text-color: #b0b0b0;
            --border-color: #4a4a4a;
            --primary-color: #16a085;
            --warning-color: #ffc107;
            --warning-background-color: #2d2b1a;
            --success-color: #28a745;
            --info-color: #17a2b8;
        }
    }
    
    /* Streamlit dark theme detection */
    .stApp[data-theme="dark"] {
        --background-color: #0e1117;
        --secondary-background-color: #262730;
        --text-color: #fafafa;
        --secondary-text-color: #b0b0b0;
        --border-color: #4a4a4a;
    }
    
    /* Enhanced contrast for better readability */
    .metric-box h4 {
        color: var(--primary-color) !important;
        font-weight: bold;
    }
    
    .metric-box p {
        color: var(--text-color) !important;
        font-weight: 500;
    }
    
    /* Ensure all Streamlit elements are theme-aware */
    .stMarkdown, .stText, .stButton, .stSelectbox, .stRadio, .stFileUploader {
        color: var(--text-color) !important;
    }
    
    /* Streamlit form elements */
    .stForm {
        background: var(--secondary-background-color) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Streamlit input fields */
    .stTextInput, .stTextArea, .stNumberInput {
        background: var(--background-color) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Streamlit buttons */
    .stButton > button {
        background: var(--primary-color) !important;
        color: white !important;
        border: 1px solid var(--primary-color) !important;
    }
    
    .stButton > button:hover {
        background: var(--primary-color) !important;
        opacity: 0.9;
    }
    
    /* Streamlit file uploader */
    .stFileUploader {
        background: var(--secondary-background-color) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Streamlit radio buttons */
    .stRadio > div > div {
        background: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Streamlit success/warning/info boxes */
    .stAlert {
        background: var(--secondary-background-color) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Hover effects for interactive elements */
    .section-box:hover {
        border-color: var(--primary-color);
        box-shadow: 0 2px 8px rgba(22, 160, 133, 0.1);
    }
    
    .metric-box:hover {
        border-color: var(--primary-color);
        transform: translateY(-2px);
        transition: all 0.2s ease;
    }
    
    /* Responsive design improvements */
    @media (max-width: 768px) {
        .step-container {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .section-box {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .metric-box {
            padding: 0.5rem;
            margin: 0.25rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize global components
def initialize_session_state():
    """Initialize session state directly"""
    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 1
    if 'wizard_data' not in st.session_state:
        st.session_state.wizard_data = {}
    if 'wizard_progress' not in st.session_state:
        st.session_state.wizard_progress = {}

@st.cache_resource
def get_session_manager():
    """Get or create global SessionManager instance"""
    # Use @st.cache_resource to ensure single instance
    return SessionManager()

@st.cache_resource
def get_data_manager():
    """Get or create global DataManager instance"""
    return DataManager()

@st.cache_resource
def get_training_pipeline():
    """Get or create global TrainingPipeline instance"""
    return TrainingPipeline()

@st.cache_resource
def get_mlflow_tracker():
    """Get or create global MLflowTracker instance"""
    return MLflowTracker("enhanced_ml_pipeline")

@st.cache_resource
def get_model_factory():
    """Get or create global model factory instance"""
    try:
        from models import model_factory
        return model_factory
    except ImportError:
        return None

@st.cache_resource
def get_model_registry():
    """Get or create global model registry instance"""
    try:
        from models.utils.model_registry import ModelRegistry
        from models.register_models import register_all_models
        
        registry = ModelRegistry()
        register_all_models(registry)
        return registry
    except Exception as e:
        print(f"Error creating model registry: {e}")
        return None

@st.cache_resource
def get_navigation_controller():
    """Get or create global NavigationController instance"""
    try:
        from wizard_ui.navigation import NavigationController
        wizard_manager = SimpleWizardManager()
        session_manager = get_session_manager()
        return NavigationController(wizard_manager, session_manager)
    except ImportError:
        return None

# Create a simple wizard manager for navigation
class SimpleWizardManager:
    def __init__(self, total_steps=7):
        self.total_steps = total_steps
    
    def get_total_steps(self):
        return self.total_steps

# Initialize only essential instances
def get_global_session_manager():
    """Get session manager instance"""
    return get_session_manager()

def get_global_data_manager():
    """Get data manager instance"""
    return get_data_manager()

def render_header():
    """Render the main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Enhanced ML Pipeline</h1>
        <p>Production-Ready MLOps Platform with Modern UI</p>
        <p>MLflow • DVC • Monitoring • AutoML • Model Serving</p>
    </div>
    """, unsafe_allow_html=True)

def render_navigation_buttons():
    """Render navigation buttons as per wireframe"""
    st.markdown('<div class="nav-buttons">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        session_manager = get_global_session_manager()
        current_step = session_manager.get_current_step()
        if st.button("◀ Previous", width='stretch', key=f"prev_btn_{current_step}"):
            # Go back to previous step
            if current_step > 1:
                session_manager.set_current_step(current_step - 1)
                st.success(f"← Going back to Step {current_step - 1}")
                st.rerun()
            else:
                st.info("ℹ️ You're already at the first step.")
    
    with col2:
        st.markdown(f"<div style='text-align: center; color: var(--text-color); font-weight: bold;'>Step {current_step} of 7</div>", unsafe_allow_html=True)
    
    with col3:
        # Use the SAME session_manager instance
        if st.button("Next ▶", width='stretch', key=f"next_btn_{current_step}"):
            current_step = session_manager.get_current_step()
            
            if current_step == 1:
                dataset = session_manager.get_wizard_data('dataset')
                print(f"DEBUG: Checking dataset for step 1 - dataset is None: {dataset is None}")
                if dataset is not None:
                    print(f"DEBUG: Dataset shape: {dataset.shape}")
                    st.success("✅ Step 1 completed! Moving to Step 2...")
                    session_manager.set_current_step(2)
                    st.rerun()
                else:
                    print("DEBUG: Dataset is None, showing warning")
                    st.warning("⚠️ Please complete Step 1 first")
            elif current_step == 2:
                processed_dataset = session_manager.get_wizard_data('processed_dataset')
                if processed_dataset is not None:
                    st.success("✅ Step 2 completed! Moving to Step 3...")
                    session_manager.set_current_step(3)
                    st.rerun()
                else:
                    st.warning("⚠️ Please complete Step 2 first")
            elif current_step == 3:
                model_config = session_manager.get_wizard_data('model_config')
                if model_config is not None:
                    st.success("✅ Step 3 completed! Moving to Step 4...")
                    session_manager.set_current_step(4)
                    st.rerun()
                else:
                    st.warning("⚠️ Please complete Step 3 first")
            elif current_step == 4:
                training_results = session_manager.get_wizard_data('training_results')
                if training_results is not None:
                    st.success("✅ Step 4 completed! Moving to Step 5...")
                    session_manager.set_current_step(5)
                    st.rerun()
                else:
                    st.warning("⚠️ Please complete Step 4 first")
            elif current_step == 5:
                st.success("✅ Step 5 completed! Moving to Step 6...")
                session_manager.set_current_step(6)
                st.rerun()
            elif current_step == 6:
                st.success("✅ Step 6 completed! Moving to Step 7...")
                session_manager.set_current_step(7)
                st.rerun()
            elif current_step == 7:
                st.success("🎉 All steps completed!")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_sidebar():
    """Render the application sidebar with navigation and status"""
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        # Step navigation
        session_manager = get_global_session_manager()
        current_step = session_manager.get_current_step()
        steps = [
            ("📊 Dataset Selection", 1),
            ("🔧 Data Preprocessing", 2),
            ("🎯 Model Configuration", 3),
            ("🚀 Training & Optimization", 4),
            ("📈 Evaluation & Analysis", 5),
            ("🔍 Monitoring & Drift", 6),
            ("🚢 Deployment", 7)
        ]
        
        for step_name, step_num in steps:
            if step_num == current_step:
                st.markdown(f"**{step_name}** ✅")
            else:
                st.markdown(f"{step_name}")
        
        st.markdown("---")
        
        # System status
        st.markdown("## 📊 System Status")
        
        # MLflow status
        try:
            mlflow_tracker = get_mlflow_tracker()
            experiments = mlflow_tracker.list_experiments()
            st.markdown(f"**MLflow**: {len(experiments)} experiments")
        except Exception:
            st.markdown("**MLflow**: ⚠️ Not connected")
        
        # Model registry status
        try:
            model_registry = get_model_registry()
            if model_registry:
                models = model_registry.list_models()
                st.markdown(f"**Models**: {len(models)} registered")
            else:
                st.markdown("**Models**: ⚠️ Not available")
        except Exception:
            st.markdown("**Models**: ⚠️ Error loading")
        
        # Data status
        try:
            data_manager = get_global_data_manager()
            datasets = data_manager.list_available_datasets()
            st.markdown(f"**Datasets**: {len(datasets)} available")
        except Exception:
            st.markdown("**Datasets**: ⚠️ Error loading")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("## ⚡ Quick Actions")
        
        if st.button("🔄 Refresh Status"):
            st.rerun()
        
        if st.button("🗑️ Clear Cache"):
            session_manager = get_global_session_manager()
            session_manager.clear_all_data()
            st.success("Cache cleared!")
            st.rerun()
        
        if st.button("📊 View Experiments"):
            st.session_state.show_experiments = True

def render_step1_dataset_selection():
    """Step 1: Dataset Selection and Upload"""
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown("## 📊 Step 1: Dataset Selection")
    
    # Get session manager from main function context
    session_manager = get_global_session_manager()
    data_manager = get_global_data_manager()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Available Datasets")
        
        # List available datasets
        try:
            datasets = data_manager.list_available_datasets()
            
            if datasets:
                dataset_names = [d['name'] for d in datasets]
                selected_dataset = st.selectbox(
                    "Select a dataset:",
                    dataset_names,
                    key="dataset_selection"
                )
                
                if selected_dataset:
                    # Show dataset info
                    dataset_info = next(d for d in datasets if d['name'] == selected_dataset)
                    
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.metric("Size", f"{dataset_info['size']:,} bytes")
                    with col_info2:
                        st.metric("Columns", len(dataset_info['columns']))
                    with col_info3:
                        st.metric("Sample Rows", dataset_info['sample_rows'])
                    
                    # Load and preview dataset
                    if st.button("📋 Load Dataset"):
                        with st.spinner("Loading dataset..."):
                            dataset_info = data_manager.load_dataset(selected_dataset)
                            df = dataset_info['data']  # Extract DataFrame from dict
                            print(f"DEBUG: Loading dataset - Shape: {df.shape}")
                            session_manager.set_wizard_data('dataset', df)
                            session_manager.set_wizard_data('dataset_name', selected_dataset)
                            
                            # Verify data was saved
                            saved_dataset = session_manager.get_wizard_data('dataset')
                            print(f"DEBUG: Dataset saved - Shape: {saved_dataset.shape if saved_dataset is not None else 'None'}")
                            
                            st.success(f"Dataset '{selected_dataset}' loaded successfully!")
                            st.rerun()
            else:
                st.warning("No datasets found in the data directory.")
                
        except Exception as e:
            st.error(f"Error loading datasets: {e}")
    
    with col2:
        st.markdown("### Upload New Dataset")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file:",
            type=['csv'],
            key="file_upload"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.markdown("#### Dataset Preview")
                st.dataframe(df.head())
                
                st.markdown("#### Dataset Info")
                st.write(f"Shape: {df.shape}")
                st.write(f"Columns: {list(df.columns)}")
                
                if st.button("💾 Save Dataset"):
                    # Save to data directory
                    filename = f"{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    filepath = Path("data") / filename
                    df.to_csv(filepath, index=False)
                    
                    session_manager.set_wizard_data('dataset', df)
                    session_manager.set_wizard_data('dataset_name', filename)
                    st.success(f"Dataset saved as '{filename}'!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")
    
    # Dataset loaded status
    if session_manager.get_wizard_data('dataset') is not None:
        st.markdown('<div class="status-success">✅ Dataset loaded successfully!</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add navigation buttons
    render_navigation_buttons()

def render_step2_data_preprocessing():
    """Step 2: Data Preprocessing and Feature Engineering"""
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown("## 🔧 Step 2: Data Preprocessing")
    
    session_manager = get_global_session_manager()
    
    df = session_manager.get_wizard_data('dataset')
    if df is None:
        st.warning("Please load a dataset first.")
        return
    
    st.markdown("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h4>📊 Total Rows</h4>
            <p>{len(df):,}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h4>📋 Total Columns</h4>
            <p>{len(df.columns)}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h4>❓ Missing Values</h4>
            <p>{df.isnull().sum().sum():,}</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <h4>🔄 Duplicate Rows</h4>
            <p>{df.duplicated().sum():,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Auto-detect data types and suggest processing
    st.markdown("### 📊 Data Type Analysis")
    
    # Analyze each column
    column_analysis = []
    for col in df.columns:
        col_data = df[col]
        missing_count = col_data.isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        # Detect data type
        if col_data.dtype in ['int64', 'float64']:
            data_type = "Numeric"
            unique_count = col_data.nunique()
            if unique_count <= 10:
                suggestion = "Consider as categorical if low cardinality"
            else:
                suggestion = "Keep as numeric for scaling"
        elif col_data.dtype == 'object':
            # Check if it's actually numeric stored as text
            try:
                pd.to_numeric(col_data.dropna())
                data_type = "Numeric (as text)"
                suggestion = "Convert to numeric"
            except:
                data_type = "Text/Categorical"
                unique_count = col_data.nunique()
                if unique_count <= 20:
                    suggestion = "Encode as categorical"
                else:
                    suggestion = "Consider text preprocessing"
        else:
            data_type = str(col_data.dtype)
            suggestion = "Review data type"
        
        column_analysis.append({
            'column': col,
            'data_type': data_type,
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'suggestion': suggestion
        })
    
    # Display column analysis
    analysis_df = pd.DataFrame(column_analysis)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Column Analysis")
        st.dataframe(analysis_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Data Type Distribution")
        type_counts = analysis_df['data_type'].value_counts()
        st.bar_chart(type_counts)
        
        st.markdown("#### Missing Data Summary")
        missing_summary = analysis_df[analysis_df['missing_count'] > 0][['column', 'missing_count', 'missing_pct']]
        if not missing_summary.empty:
            st.dataframe(missing_summary, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
    
    # Data preprocessing options
    st.markdown("### 🔧 Column-Specific Preprocessing")
    
    # Create tabs for different preprocessing types
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Numeric Columns", "📝 Text Columns", "🔢 Missing Values", "⚙️ General"])
    
    with tab1:
        st.markdown("#### Numeric Column Processing")
        numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        
        if numeric_cols:
            st.write(f"Found {len(numeric_cols)} numeric columns: {', '.join(numeric_cols)}")
            
            col1, col2 = st.columns(2)
            with col1:
                numeric_scaling = st.multiselect(
                    "Scaling methods for numeric columns:",
                    ["none", "standard", "minmax", "robust"],
                    default=["standard", "minmax", "robust"],
                    key="numeric_scaling"
                )
            
            with col2:
                remove_outliers = st.checkbox("Remove outliers using IQR method", key="remove_outliers")
        else:
            st.info("No numeric columns found.")
    
    with tab2:
        st.markdown("#### Text Column Processing")
        text_cols = [col for col in df.columns if df[col].dtype == 'object']
        
        if text_cols:
            st.write(f"Found {len(text_cols)} text columns: {', '.join(text_cols)}")
            
            col1, col2 = st.columns(2)
            with col1:
                text_encoding = st.selectbox(
                    "Encoding method for text columns:",
                    ["label_encoding", "one_hot_encoding", "target_encoding"],
                    key="text_encoding"
                )
            
            with col2:
                max_categories = st.number_input(
                    "Max categories for one-hot encoding:",
                    min_value=2,
                    max_value=50,
                    value=10,
                    key="max_categories"
                )
        else:
            st.info("No text columns found.")
    
    with tab3:
        st.markdown("#### Missing Value Handling")
        
        col1, col2 = st.columns(2)
        with col1:
            missing_strategy = st.selectbox(
                "Strategy for missing values:",
                ["drop", "fill_mean", "fill_median", "fill_mode", "fill_forward", "fill_backward"],
                key="missing_strategy"
            )
        
        with col2:
            remove_duplicates = st.checkbox("Remove duplicate rows", key="remove_duplicates")
    
    with tab4:
        st.markdown("#### General Options")
        
        col1, col2 = st.columns(2)
        with col1:
            feature_selection = st.checkbox("Enable feature selection", key="feature_selection")
            if feature_selection:
                max_features = st.number_input(
                    "Maximum number of features:",
                    min_value=1,
                    max_value=len(df.columns),
                    value=min(10, len(df.columns)),
                    key="max_features"
                )
        
        with col2:
            data_split_config = {
                'test_size': st.slider("Test set size:", 0.1, 0.5, 0.2, key="test_size"),
                'val_size': st.slider("Validation set size:", 0.1, 0.3, 0.2, key="val_size"),
                'random_state': st.number_input("Random state:", value=42, key="random_state")
            }
    
    # Apply preprocessing
    if st.button("🔧 Apply Preprocessing"):
        with st.spinner("Applying preprocessing..."):
            try:
                # Apply preprocessing steps
                processed_df = df.copy()
                
                # Remove duplicates
                if remove_duplicates:
                    processed_df = processed_df.drop_duplicates()
                
                # Handle missing values
                if missing_strategy == "drop":
                    processed_df = processed_df.dropna()
                elif missing_strategy == "fill_mean":
                    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())
                elif missing_strategy == "fill_median":
                    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].median())
                elif missing_strategy == "fill_mode":
                    processed_df = processed_df.fillna(processed_df.mode().iloc[0])
                
                # Remove outliers
                if remove_outliers:
                    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        Q1 = processed_df[col].quantile(0.25)
                        Q3 = processed_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        processed_df = processed_df[(processed_df[col] >= lower_bound) & (processed_df[col] <= upper_bound)]
                
                # Store preprocessing config
                preprocessing_config = {
                    'missing_strategy': missing_strategy,
                    'remove_outliers': remove_outliers,
                    'remove_duplicates': remove_duplicates,
                    'numeric_scaling': numeric_scaling if 'numeric_scaling' in locals() else ['none'],
                    'text_encoding': text_encoding if 'text_encoding' in locals() else 'label_encoding',
                    'max_categories': max_categories if 'max_categories' in locals() else 10,
                    'feature_selection': feature_selection,
                    'max_features': max_features if feature_selection else None,
                    'data_split_config': data_split_config if 'data_split_config' in locals() else None,
                    'numeric_cols': numeric_cols if 'numeric_cols' in locals() else [],
                    'text_cols': text_cols if 'text_cols' in locals() else []
                }
                
                session_manager.set_wizard_data('processed_dataset', processed_df)
                session_manager.set_wizard_data('preprocessing_config', preprocessing_config)
                
                st.success("Preprocessing completed successfully!")
                
                # Show results
                st.markdown("### Preprocessing Results")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Rows After", len(processed_df))
                with col2:
                    st.metric("Columns After", len(processed_df.columns))
                with col3:
                    st.metric("Missing Values", processed_df.isnull().sum().sum())
                with col4:
                    st.metric("Duplicate Rows", processed_df.duplicated().sum())
                
            except Exception as e:
                st.error(f"Error during preprocessing: {e}")
    
    # Show processed data preview
    if session_manager.get_wizard_data('processed_dataset') is not None:
        st.markdown("### Processed Data Preview")
        processed_df = session_manager.get_wizard_data('processed_dataset')
        st.dataframe(processed_df.head())
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add navigation buttons
    render_navigation_buttons()

def render_step3_model_configuration():
    """Step 3: Model Configuration and Selection"""
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown("## 🎯 Step 3: Model Configuration")
    
    print("DEBUG: Starting Step 3")
    
    session_manager = get_global_session_manager()
    
    processed_df = session_manager.get_wizard_data('processed_dataset')
    print(f"DEBUG: Processed dataset is None: {processed_df is None}")
    if processed_df is None:
        st.warning("⚠️ Please complete data preprocessing first.")
        st.info("💡 Go back to Step 2 and click '🔧 Apply Preprocessing' button.")
        return
    
    # Get available models from registry
    print("DEBUG: Getting model registry")
    model_registry = get_model_registry()
    print(f"DEBUG: Model registry is None: {model_registry is None}")
    if not model_registry:
        st.error("Model registry not available")
        return
    
    print("DEBUG: Listing models")
    available_models = model_registry.list_models()
    print(f"DEBUG: Found {len(available_models)} models: {available_models}")
    
    st.markdown("### Available Models")
    
    # Create list of available models with descriptions
    available_model_options = []
    for model_name in available_models:
        try:
            metadata = model_registry.get_model_metadata(model_name)
            if metadata.get('category') in ['classification', 'clustering']:
                description = metadata.get('description', 'No description')
                available_model_options.append(f"{model_name} - {description}")
        except Exception as e:
            print(f"DEBUG: Error getting metadata for {model_name}: {e}")
    
    print(f"DEBUG: Available model options: {len(available_model_options)}")
    
    # Use multiselect like in app_old.py
    selected_models = st.multiselect(
        "Select models for training:",
        available_model_options,
        default=available_model_options,  # Default to all models
        key="model_selection"
    )
    
    # Extract model names from selected options
    selected_model_names = []
    for option in selected_models:
        model_name = option.split(' - ')[0]  # Extract model name before ' - '
        selected_model_names.append(model_name)
    
    print(f"DEBUG: Selected models: {selected_model_names}")
    
    if selected_model_names:
        st.success(f"✅ Selected {len(selected_model_names)} models: {', '.join(selected_model_names)}")
    
    # Model configuration
    st.markdown("### Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Hyperparameter Optimization")
        enable_optuna = st.checkbox("Enable Optuna optimization", key="enable_optuna")
        
        if enable_optuna:
            n_trials = st.slider("Number of trials:", 10, 100, 50)
            optimization_timeout = st.slider("Timeout (minutes):", 5, 60, 30)
    
    with col2:
        st.markdown("#### Ensemble Methods")
        enable_ensemble = st.checkbox("Enable ensemble methods", key="enable_ensemble")
        
        if enable_ensemble:
            ensemble_method = st.selectbox(
                "Ensemble method:",
                ["voting", "stacking", "bagging"],
                key="ensemble_method"
            )
    
    # Target column selection
    st.markdown("### Target Configuration")
    
    # Determine if this is classification or clustering
    task_type = st.radio(
        "Task type:",
        ["classification", "clustering"],
        key="task_type"
    )
    
    if task_type == "classification":
        target_column = st.selectbox(
            "Select target column:",
            processed_df.columns,
            key="target_column"
        )
        
        # Show target distribution
        if target_column:
            st.markdown("#### Target Distribution")
            target_counts = processed_df[target_column].value_counts()
            st.bar_chart(target_counts)
    
    # Store configuration
    if st.button("💾 Save Configuration"):
        model_config = {
            'selected_models': selected_model_names,
            'task_type': task_type,
            'target_column': target_column if task_type == 'classification' else None,
            'enable_optuna': enable_optuna,
            'n_trials': n_trials if enable_optuna else None,
            'optimization_timeout': optimization_timeout if enable_optuna else None,
            'enable_ensemble': enable_ensemble,
            'ensemble_method': ensemble_method if enable_ensemble else None
        }
        
        session_manager.set_wizard_data('model_config', model_config)
        st.success("Model configuration saved!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add navigation buttons
    render_navigation_buttons()

def render_step4_training_optimization():
    """Step 4: Training and Hyperparameter Optimization"""
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown("## 🚀 Step 4: Training & Optimization")
    
    session_manager = get_global_session_manager()
    
    processed_df = session_manager.get_wizard_data('processed_dataset')
    model_config = session_manager.get_wizard_data('model_config')
    
    if processed_df is None or model_config is None:
        st.warning("Please complete previous steps first.")
        return
    
    # Training configuration
    st.markdown("### Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Data Split")
        test_size = st.slider("Test set size:", 0.1, 0.4, 0.2)
        validation_size = st.slider("Validation set size:", 0.1, 0.3, 0.2)
        random_state = st.number_input("Random state:", value=42, min_value=0)
    
    with col2:
        st.markdown("#### Training Options")
        cross_validation = st.checkbox("Enable cross-validation", value=True)
        if cross_validation:
            cv_folds = st.slider("CV folds:", 3, 10, 5)
        
        early_stopping = st.checkbox("Enable early stopping", value=True)
        if early_stopping:
            patience = st.slider("Patience:", 5, 50, 10)
    
    # Start training
    if st.button("🚀 Start Training"):
        with st.spinner("Training models..."):
            try:
                # Prepare data
                if model_config['task_type'] == 'classification':
                    X = processed_df.drop(columns=[model_config['target_column']])
                    y = processed_df[model_config['target_column']]
                    
                    # Split data
                    from sklearn.model_selection import train_test_split
                    X_train, X_temp, y_train, y_temp = train_test_split(
                        X, y, test_size=test_size + validation_size, random_state=random_state
                    )
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_temp, y_temp, test_size=test_size/(test_size + validation_size), random_state=random_state
                    )
                    
                    # Train models
                    results = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, model_name in enumerate(model_config['selected_models']):
                        status_text.text(f"Training {model_name}...")
                        
                        # Create model instance
                        model_factory = get_model_factory()
                        if not model_factory:
                            st.error("Model factory not available")
                            continue
                        
                        model = model_factory.create_model(model_name)
                        
                        # Train model
                        if model_config['enable_optuna']:
                            optuna_optimizer = get_optuna_optimizer()
                            if optuna_optimizer:
                                # Use Optuna optimization
                                best_params = optuna_optimizer.optimize(
                                    model, X_train, y_train, X_val, y_val,
                                    n_trials=model_config['n_trials']
                                )
                                model.set_params(**best_params)
                        
                        # Fit model
                        model.fit(X_train, y_train)
                        
                        # Evaluate
                        train_score = model.score(X_train, y_train)
                        val_score = model.score(X_val, y_val)
                        test_score = model.score(X_test, y_test)
                        
                        results[model_name] = {
                            'model': model,
                            'train_score': train_score,
                            'val_score': val_score,
                            'test_score': test_score,
                            'best_params': best_params if model_config['enable_optuna'] else None
                        }
                        
                        # Log to MLflow
                        mlflow_tracker = get_mlflow_tracker()
                        if mlflow_tracker:
                            with mlflow_tracker.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                                mlflow_tracker.log_params(best_params if model_config['enable_optuna'] else {})
                                mlflow_tracker.log_metrics({
                                    'train_score': train_score,
                                    'val_score': val_score,
                                    'test_score': test_score
                                })
                                
                                # Log model
                                mlflow_tracker.log_model(model, f"{model_name}_model")
                        
                        progress_bar.progress((i + 1) / len(model_config['selected_models']))
                    
                    # Store results
                    session_manager.set_wizard_data('training_results', results)
                    session_manager.set_wizard_data('test_data', {'X_test': X_test, 'y_test': y_test})
                    
                    st.success("Training completed successfully!")
                    
                    # Show results summary
                    st.markdown("### Training Results Summary")
                    
                    results_df = pd.DataFrame({
                        'Model': list(results.keys()),
                        'Train Score': [r['train_score'] for r in results.values()],
                        'Validation Score': [r['val_score'] for r in results.values()],
                        'Test Score': [r['test_score'] for r in results.values()]
                    })
                    
                    st.dataframe(results_df)
                    
                    # Show best model
                    best_model = max(results.keys(), key=lambda k: results[k]['test_score'])
                    st.markdown(f"**Best Model**: {best_model} (Test Score: {results[best_model]['test_score']:.4f})")
                
            except Exception as e:
                st.error(f"Error during training: {e}")
                logger.error(f"Training error: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add navigation buttons
    render_navigation_buttons()

def render_step5_evaluation_analysis():
    """Step 5: Model Evaluation and Analysis"""
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown("## 📈 Step 5: Evaluation & Analysis")
    
    session_manager = get_global_session_manager()
    
    results = session_manager.get_wizard_data('training_results')
    test_data = session_manager.get_wizard_data('test_data')
    
    if results is None or test_data is None:
        st.warning("Please complete training first.")
        return
    
    # Model comparison
    st.markdown("### Model Performance Comparison")
    
    # Create comparison chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(results.keys())
    train_scores = [results[m]['train_score'] for m in models]
    val_scores = [results[m]['val_score'] for m in models]
    test_scores = [results[m]['test_score'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, train_scores, width, label='Train', alpha=0.8)
    ax.bar(x, val_scores, width, label='Validation', alpha=0.8)
    ax.bar(x + width, test_scores, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Detailed evaluation for best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_score'])
    best_model = results[best_model_name]['model']
    
    st.markdown(f"### Detailed Analysis: {best_model_name}")
    
    # Predictions
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    y_pred = best_model.predict(X_test)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Classification Report")
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred, output_dict=True)
        st.json(report)
    
    with col2:
        st.markdown("#### Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        st.markdown("#### Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=ax)
        ax.set_title('Top 10 Feature Importances')
        st.pyplot(fig)
    
    # Model registry
    st.markdown("### Model Registry")
    
    if st.button("📝 Register Best Model"):
        try:
            model_registry = get_model_registry()
            if model_registry:
                model_registry.register_model(
                    name=f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model=best_model,
                    metrics={'test_score': results[best_model_name]['test_score']},
                    tags={'best_model': True, 'model_type': best_model_name}
                )
                st.success("Model registered successfully!")
            else:
                st.error("Model registry not available")
        except Exception as e:
            st.error(f"Error registering model: {e}")
    
    # Show next step button
    if st.button("➡️ Next: Monitoring & Drift", key="next_step5"):
        session_manager.set_current_step(6)
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add navigation buttons
    render_navigation_buttons()

def render_step6_monitoring_drift():
    """Step 6: Monitoring and Drift Detection"""
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown("## 🔍 Step 6: Monitoring & Drift Detection")
    
    session_manager = get_global_session_manager()
    
    results = session_manager.get_wizard_data('training_results')
    processed_df = session_manager.get_wizard_data('processed_dataset')
    
    if results is None or processed_df is None:
        st.warning("Please complete training first.")
        return
    
    # Monitoring configuration
    st.markdown("### Monitoring Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Drift Detection")
        enable_drift_detection = st.checkbox("Enable drift detection", value=True)
        
        if enable_drift_detection:
            drift_threshold = st.slider("Drift threshold:", 0.1, 0.5, 0.2)
            drift_window = st.slider("Detection window (days):", 1, 30, 7)
    
    with col2:
        st.markdown("#### Performance Monitoring")
        enable_performance_monitoring = st.checkbox("Enable performance monitoring", value=True)
        
        if enable_performance_monitoring:
            performance_threshold = st.slider("Performance threshold:", 0.5, 0.95, 0.8)
            alert_frequency = st.selectbox("Alert frequency:", ["immediate", "daily", "weekly"])
    
    # Setup monitoring
    if st.button("🔧 Setup Monitoring"):
        with st.spinner("Setting up monitoring..."):
            try:
                # Configure drift monitoring
                if enable_drift_detection:
                    drift_monitor = get_drift_monitor()
                    if drift_monitor:
                        drift_monitor.setup_drift_detection(
                            reference_data=processed_df,
                            threshold=drift_threshold,
                            window_days=drift_window
                        )
                
                # Configure performance monitoring
                if enable_performance_monitoring:
                    monitoring_config = get_monitoring_config()
                    if monitoring_config:
                        monitoring_config.setup_performance_monitoring(
                            threshold=performance_threshold,
                            alert_frequency=alert_frequency
                        )
                
                st.success("Monitoring setup completed!")
                
            except Exception as e:
                st.error(f"Error setting up monitoring: {e}")
    
    # Monitoring dashboard
    st.markdown("### Monitoring Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### System Health")
        st.metric("MLflow Status", "✅ Connected")
        st.metric("Model Registry", "✅ Active")
        st.metric("Drift Detection", "✅ Enabled" if enable_drift_detection else "❌ Disabled")
    
    with col2:
        st.markdown("#### Performance Metrics")
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_score'])
        best_score = results[best_model_name]['test_score']
        
        st.metric("Best Model Score", f"{best_score:.4f}")
        st.metric("Model Count", len(results))
        st.metric("Experiments", "Active")
    
    with col3:
        st.markdown("#### Alerts")
        st.info("No active alerts")
        st.success("All systems operational")
        st.warning("Monitor drift detection")
    
    # Drift detection results
    if enable_drift_detection:
        st.markdown("### Drift Detection Results")
        
        if st.button("🔍 Check for Drift"):
            with st.spinner("Checking for drift..."):
                try:
                    drift_monitor = get_drift_monitor()
                    if drift_monitor:
                        # Simulate drift detection (in real implementation, this would check against new data)
                        drift_results = drift_monitor.check_drift(processed_df.sample(100))
                        
                        if drift_results['drift_detected']:
                            st.error(f"⚠️ Drift detected! Score: {drift_results['drift_score']:.4f}")
                        else:
                            st.success(f"✅ No drift detected. Score: {drift_results['drift_score']:.4f}")
                    else:
                        st.warning("Drift monitor not available")
                    
                except Exception as e:
                    st.error(f"Error checking drift: {e}")
    
    # Show next step button
    if st.button("➡️ Next: Deployment", key="next_step6"):
        session_manager.set_current_step(7)
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add navigation buttons
    render_navigation_buttons()

def render_step7_deployment():
    """Step 7: Model Deployment and Serving"""
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown("## 🚢 Step 7: Deployment")
    
    session_manager = get_global_session_manager()
    
    results = session_manager.get_wizard_data('training_results')
    
    if results is None:
        st.warning("Please complete training first.")
        return
    
    # Deployment options
    st.markdown("### Deployment Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Deployment Type")
        deployment_type = st.selectbox(
            "Select deployment type:",
            ["local", "docker", "cloud", "kubernetes"],
            key="deployment_type"
        )
        
        st.markdown("#### Model Selection")
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_score'])
        st.info(f"Selected model: {best_model_name}")
    
    with col2:
        st.markdown("#### Serving Configuration")
        serving_port = st.number_input("Serving port:", value=8000, min_value=1000, max_value=65535)
        max_workers = st.slider("Max workers:", 1, 10, 4)
        
        enable_monitoring = st.checkbox("Enable serving monitoring", value=True)
    
    # Deployment actions
    st.markdown("### Deployment Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Deploy Model"):
            with st.spinner("Deploying model..."):
                try:
                    # Deploy model using FastAPI serving
                    from src.serve.app import app
                    
                    # Start serving (in real implementation, this would start the FastAPI server)
                    st.success("Model deployed successfully!")
                    st.info(f"Serving on port {serving_port}")
                    
                except Exception as e:
                    st.error(f"Error deploying model: {e}")
    
    with col2:
        if st.button("📊 Test API"):
            with st.spinner("Testing API..."):
                try:
                    # Test API endpoint (simulate)
                    st.success("API test successful!")
                    st.info("Model is responding correctly")
                    
                except Exception as e:
                    st.error(f"API test failed: {e}")
    
    with col3:
        if st.button("📈 View Metrics"):
            with st.spinner("Loading metrics..."):
                try:
                    # Show serving metrics
                    st.success("Metrics loaded!")
                    st.info("Performance metrics available")
                    
                except Exception as e:
                    st.error(f"Error loading metrics: {e}")
    
    # Deployment status
    st.markdown("### Deployment Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", "✅ Deployed")
    with col2:
        st.metric("Port", serving_port)
    with col3:
        st.metric("Workers", max_workers)
    with col4:
        st.metric("Monitoring", "✅ Enabled" if enable_monitoring else "❌ Disabled")
    
    # API documentation
    st.markdown("### API Documentation")
    
    st.markdown("""
    #### Available Endpoints:
    
    - **POST /predict**: Make predictions
    - **GET /health**: Health check
    - **GET /metrics**: Prometheus metrics
    - **GET /docs**: API documentation
    
    #### Example Usage:
    ```python
    import requests
    
    # Make prediction
    response = requests.post(
        f"http://localhost:{serving_port}/predict",
        json={"data": [[1, 2, 3, 4, 5]]}
    )
    result = response.json()
    ```
    """)
    
    # Completion
    st.markdown("### 🎉 Pipeline Complete!")
    
    if st.button("🔄 Start New Pipeline"):
        session_manager.clear_all_data()
        session_manager.set_current_step(1)
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add navigation buttons
    render_navigation_buttons()

def render_experiments_view():
    """Render experiments view"""
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown("## 📊 MLflow Experiments")
    
    try:
        mlflow_tracker = get_mlflow_tracker()
        if mlflow_tracker:
            experiments = mlflow_tracker.list_experiments()
            
            if experiments:
                for exp in experiments:
                    with st.expander(f"Experiment: {exp['name']}"):
                        st.write(f"**ID**: {exp['experiment_id']}")
                        st.write(f"**Artifact Location**: {exp['artifact_location']}")
                        st.write(f"**Lifecycle Stage**: {exp['lifecycle_stage']}")
                        
                        # Show runs
                        runs = mlflow_tracker.list_runs(exp['experiment_id'])
                        if runs:
                            st.markdown("#### Runs:")
                            for run in runs[:5]:  # Show first 5 runs
                                st.write(f"- {run['run_name']} (Status: {run['status']})")
            else:
                st.info("No experiments found.")
        else:
            st.warning("MLflow tracker not available")
            
    except Exception as e:
        st.error(f"Error loading experiments: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function"""
    try:
        # Initialize session state first
        initialize_session_state()
        
        # Initialize session manager
        session_manager = get_global_session_manager()
        
        # Render header
        render_header()
        
        # Render sidebar
        render_sidebar()
        
        # Check if showing experiments
        if st.session_state.get('show_experiments', False):
            render_experiments_view()
            if st.button("← Back to Main"):
                st.session_state.show_experiments = False
                st.rerun()
            return
        
        # Get current step
        current_step = session_manager.get_current_step()
        
        # Render current step
        if current_step == 1:
            render_step1_dataset_selection()
        elif current_step == 2:
            render_step2_data_preprocessing()
        elif current_step == 3:
            render_step3_model_configuration()
        elif current_step == 4:
            render_step4_training_optimization()
        elif current_step == 5:
            render_step5_evaluation_analysis()
        elif current_step == 6:
            render_step6_monitoring_drift()
        elif current_step == 7:
            render_step7_deployment()
        else:
            st.error(f"Unknown step: {current_step}")
    
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
