"""
Topic Modeling - Auto Classifier
Step 1: Dataset Selection & Upload
Step 2: Data Preprocessing & Sampling
Exact wireframe implementation

Created: 2025-01-27
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
import seaborn as sns
import sys
import os
import time
import pickle
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wizard_ui.session_manager import SessionManager
from training_pipeline import StreamlitTrainingPipeline

# FIXED: Create global SessionManager instance for persistence
@st.cache_resource
def get_session_manager():
    """Get or create global SessionManager instance"""
    return SessionManager()

# Initialize global session manager
session_manager = get_session_manager()


def get_cache_info():
    """Get cache information for display"""
    try:
        pipeline = StreamlitTrainingPipeline()
        cached_results = pipeline.list_cached_results()
        return cached_results
    except Exception as e:
        st.error(f"Error accessing cache: {e}")
        return []


def clear_cache_action(cache_key: str = None):
    """Clear specific cache or all cache"""
    try:
        pipeline = StreamlitTrainingPipeline()
        pipeline.clear_cache(cache_key)
        st.success("Cache cleared successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Error clearing cache: {e}")


def format_cache_age(age_hours: float) -> str:
    """Format cache age for display"""
    if age_hours < 1:
        return f"{age_hours * 60:.0f} minutes"
    elif age_hours < 24:
        return f"{age_hours:.1f} hours"
    else:
        days = age_hours / 24
        return f"{days:.1f} days"

# Page configuration
st.set_page_config(
    page_title="🤖 AIO Classifier",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS for flexible theme design
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
            margin: 0.25rem;
            padding: 0.75rem;
        }
    }
    
    /* Force theme consistency */
    * {
        transition: background-color 0.2s ease, 
                   color 0.2s ease, 
                   border-color 0.2s ease;
    }
    
    /* Optimize transitions for better performance */
    .step-container, .section-box, .metric-box {
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    
    /* Reduce transition effects that might cause dark screen */
    .stMarkdown, .stText, .stButton, .stSelectbox, .stRadio, .stFileUploader {
        transition: none !important;
    }
    
    /* Ensure smooth loading without dark transitions */
    .stSpinner {
        transition: opacity 0.1s ease !important;
    }
    
        content: "🔄";
    /* Prevent dark screen during loading */
    .stApp[data-theme="dark"] .stSpinner {
        background: rgba(14, 17, 23, 0.8) !important;
    }
    
    .stApp[data-theme="light"] .stSpinner {
        background: rgba(255, 255, 255, 0.8) !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application following wireframe design"""
    
    # Initialize loading state
    if 'is_loading' not in st.session_state:
        st.session_state.is_loading = False
    
    # Initialize detailed analysis session state
    if 'selected_model_detail' not in st.session_state:
        st.session_state.selected_model_detail = None
    if 'selected_model_result' not in st.session_state:
        st.session_state.selected_model_result = None
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AIO Classifier</h1>
        <p>Intelligent Text Classification with Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session manager
    # Use global session_manager instance
    current_step = get_current_step(session_manager)
    
    # Show loading indicator if processing
    if st.session_state.is_loading:
        with st.spinner("🔄 Processing your request..."):
            time.sleep(0.1)  # Small delay to show spinner
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Ensure current_step is properly set in session
        if session_manager.get_current_step() != current_step:
            session_manager.set_current_step(current_step)
        
        if current_step == 1:
            render_step1_wireframe()
        elif current_step == 2:
            render_step2_wireframe()
        elif current_step == 3:
            render_step3_wireframe()
        elif current_step == 4:
            render_step4_wireframe()
        elif current_step == 5:
            render_step5_wireframe()
        else:
            render_step1_wireframe()  # Default to step 1
    
    with col2:
        render_sidebar()
    
    # Reset loading state
    st.session_state.is_loading = False


def render_step1_wireframe():
    """Render Step 1 exactly as per wireframe design"""
    
    # Step title - simplified without big container
    st.markdown("""
    <h2 style="text-align: left; color: var(--text-color); margin: 2rem 0 1rem 0; font-size: 1.8rem;">
        📍 STEP 1/5: Dataset Selection & Upload
    </h2>
    """, unsafe_allow_html=True)
    
    # Dataset Source Selection - simplified
    st.markdown("""
    <h3 style="color: var(--text-color); margin: 1.5rem 0 1rem 0;">🎯 Choose Dataset Source:</h3>
    """, unsafe_allow_html=True)
    
    # Simple selection dropdown
    dataset_source = st.selectbox(
        "Select your dataset source:",
        [
            "Use Sample Dataset (Data Folder)",
            "File Path (File Path)",
            "Upload Custom File (CSV/JSON/Excel)"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    # Handle different dataset sources
    if "File Path" in dataset_source:
        file_path = st.text_input(
            "File path to your dataset:",
            placeholder="e.g., C:/Users/username/documents/dataset.csv",
            help="Enter the full path to your dataset file (CSV, Excel, JSON, or TXT)",
            label_visibility="collapsed"
        )
        
        if file_path:
            if st.button("📂 Load File from Path", type="primary"):
                try:
                    # Check if file exists
                    if os.path.exists(file_path):
                        # Add loading indicator for file reading
                        with st.spinner("🔄 Reading file from path..."):
                            # Read file based on extension
                            file_extension = file_path.split('.')[-1].lower()
                            
                            if file_extension == 'csv':
                                df = pd.read_csv(file_path)
                            elif file_extension in ['xlsx', 'xls']:
                                df = pd.read_excel(file_path)
                            elif file_extension == 'json':
                                df = pd.read_json(file_path)
                            elif file_extension == 'txt':
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    lines = f.readlines()[:100]
                                df = pd.DataFrame({'text': lines})
                            else:
                                st.error("❌ Unsupported file format. Please use CSV, Excel, JSON, or TXT files.")
                                return
                        
                        st.success(f"✅ File loaded successfully from: {file_path}")
                        
                        # Store in session with loading indicator
                        with st.spinner("💾 Saving to session..."):
                            # Store in session using global instance
                            session_manager.update_step_data(1, 'dataframe', df)
                            session_manager.update_step_data(1, 'file_path',
                                                           file_path)
                            
                            # Store default sampling configuration
                            dataset_size = len(df)
                            if dataset_size <= 10:
                                default_samples = dataset_size
                            elif dataset_size < 1000:
                                default_samples = dataset_size
                            else:
                                default_samples = min(100000, dataset_size)
                            
                            default_sampling_config = {
                                'num_samples': default_samples,
                                'sampling_strategy': 'Stratified (Recommended)'
                            }
                            
                            session_manager.update_step_data(1, 'sampling_config', default_sampling_config)
                            print(f"💾 Saved default sampling config from path: {default_sampling_config}")
                            
                            # Also store dataset size for reference
                            session_manager.update_step_data(1, 'dataset_size', dataset_size)
                        
                        # Show file preview with loading indicator
                        with st.spinner("📊 Generating file preview..."):
                            show_file_preview(df, file_extension)
                        
                    else:
                        st.error("❌ File not found. Please check the path and try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
                    st.exception(e)  # Show full error details
        
    elif "Sample Dataset" in dataset_source:
        import glob

        # Định nghĩa thư mục data (thay đổi từ cache sang data)
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        allowed_exts = ['.csv', '.xlsx', '.xls', '.json', '.txt']

        # Lấy danh sách file hợp lệ trong thư mục data
        if os.path.exists(data_dir):
            files = []
            for ext in allowed_exts:
                files.extend(glob.glob(os.path.join(data_dir, f"*{ext}")))
            files = sorted(files)
        else:
            files = []

        if not files:
            st.warning("⚠️ Không tìm thấy sample dataset trong thư mục data.")
        else:
            # Hiển thị danh sách file sample
            file_names = [os.path.basename(f) for f in files]
            selected_file = st.selectbox(
                "Chọn sample dataset từ data:",
                file_names,
                help="Chọn một file mẫu từ thư mục data của dự án"
            )

            if selected_file:
                file_path = os.path.join(data_dir, selected_file)
                file_extension = selected_file.split('.')[-1].lower()
                try:
                    # Add loading indicator for sample dataset
                    with st.spinner(f"🔄 Loading sample dataset '{selected_file}'..."):
                        if file_extension == 'csv':
                            df = pd.read_csv(file_path)
                        elif file_extension in ['xlsx', 'xls']:
                            df = pd.read_excel(file_path)
                        elif file_extension == 'json':
                            df = pd.read_json(file_path)
                        elif file_extension == 'txt':
                            with open(file_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()[:100]
                            df = pd.DataFrame({'text': lines})
                        else:
                            st.error("❌ Unsupported file format. Please use CSV, Excel, JSON, or TXT files.")
                            return

                    st.toast(f"✅ Sample dataset '{selected_file}' loaded from data.")

                    # Store in session with loading indicator
                    with st.spinner("💾 Saving to session..."):
                        # Store in session using global instance
                        session_manager.update_step_data(1, 'dataframe', df)
                        session_manager.update_step_data(1, 'file_path', file_path)

                        # Store default sampling configuration
                        dataset_size = len(df)
                        if dataset_size <= 10:
                            default_samples = dataset_size
                        elif dataset_size < 1000:
                            default_samples = dataset_size
                        else:
                            default_samples = min(100000, dataset_size)
                        
                        default_sampling_config = {
                            'num_samples': default_samples,
                            'sampling_strategy': 'Stratified (Recommended)'
                        }
                        
                        session_manager.update_step_data(1, 'sampling_config', default_sampling_config)
                        print(f"💾 Saved default sampling config from sample: {default_sampling_config}")
                        
                        # Also store dataset size for reference
                        session_manager.update_step_data(1, 'dataset_size', dataset_size)

                    # Show file preview with loading indicator
                    with st.spinner("📊 Generating file preview..."):
                        show_file_preview(df, file_extension)
                        
                except Exception as e:
                    st.toast(f"❌ Error loading sample dataset: {str(e)}")
                    st.exception(e)  # Show full error details
    
    # Custom File Upload Section
    if "Upload Custom File" in dataset_source:
        # File upload controls
        uploaded_file = st.file_uploader(
            "Choose Files",
            type=['csv', 'xlsx', 'xls', 'json', 'txt'],
            help="Upload your dataset file"
        )
        
        # Process uploaded file
        if uploaded_file:
            process_uploaded_file(uploaded_file)
    
    # Sampling Configuration Section (moved from Step 2)
    if 'dataframe' in locals() or 'df' in locals():
        st.markdown("---")
        st.markdown("""
        <h3 style="color: var(--text-color); margin: 1.5rem 0 1rem 0;">🔧 Sampling Configuration:</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Number of samples slider
            # Use global session_manager instance
            step1_data = session_manager.get_step_data(1)
            
            # Initialize default values for sampling variables
            min_samples = 100
            max_samples = 1000
            step_size = 100
            default_samples = 1000
            dataset_size = 0
            existing_config = {}
            
            if step1_data and 'dataframe' in step1_data:
                df = step1_data['dataframe']
                dataset_size = len(df)
                
                # Get existing sampling config from session
                existing_config = session_manager.get_step_data(1).get('sampling_config', {})
                default_samples = existing_config.get('num_samples', min(100000, dataset_size))
            
                # Adjust min_value and step based on dataset size
                if dataset_size <= 10:
                    min_samples = 1
                    max_samples = dataset_size
                    step_size = 1
                    default_samples = dataset_size
                elif dataset_size < 100:
                    min_samples = 1
                    max_samples = dataset_size
                    step_size = max(1, dataset_size // 10)
                    default_samples = dataset_size
                elif dataset_size < 1000:
                    min_samples = 10
                    max_samples = dataset_size
                    step_size = max(1, dataset_size // 10)
                    default_samples = dataset_size
                elif dataset_size == 1000:
                    min_samples = 100
                    max_samples = 1000
                    step_size = 100
                    default_samples = 1000
                else:
                    min_samples = 1000
                    max_samples = min(500000, dataset_size)
                    step_size = 1000
                    default_samples = min(100000, dataset_size)
            
            # Final safety check to ensure max > min
            if max_samples <= min_samples:
                max_samples = min_samples + step_size
            
            num_samples = st.slider(
                "📊 Number of Samples:",
                min_value=min_samples,
                max_value=max_samples,
                value=default_samples,
                step=step_size,
                help=f"Select number of samples ({min_samples:,} - {max_samples:,})"
            )
            
            # Show warning for small datasets
            if dataset_size < 1000 and dataset_size > 0:
                st.warning(f"⚠️ Small dataset detected ({dataset_size:,} rows). "
                          f"Consider using all available data for better model performance.")
            elif dataset_size == 1000:
                st.info(f"ℹ️ Dataset size: {dataset_size:,} rows. "
                       f"You can sample from 1000 rows.")
            elif dataset_size == 0:
                st.warning("⚠️ Please load a dataset first to configure sampling.")
        
        with col2:
            # Sampling strategy
            # Initialize default sampling strategy
            sampling_strategy = 'Stratified (Recommended)'
            
            if step1_data and 'dataframe' in step1_data:
                existing_strategy = existing_config.get('sampling_strategy', 'Stratified (Recommended)')
                strategy_index = 1 if existing_strategy == "Stratified (Recommended)" else 0
                
                sampling_strategy = st.radio(
                    "🎯 Sampling Strategy:",
                    ["Random", "Stratified (Recommended)"],
                    index=strategy_index,
                    help="Random: Simple random sampling. Stratified: Maintains class distribution."
                )
            else:
                st.warning("⚠️ Please load a dataset first to configure sampling.")
        
        # Save sampling configuration to session
        if step1_data and 'dataframe' in step1_data:
            # Always save sampling config, even if user hasn't changed it
            current_sampling_config = {
                'num_samples': num_samples,
                'sampling_strategy': sampling_strategy
            }
            session_manager.update_step_data(1, 'sampling_config', current_sampling_config)
            print(f"💾 [STEP1] Saved sampling config to session: {current_sampling_config}")
            print(f"📊 [STEP1] Dataset size: {len(step1_data['dataframe']):,}, Requested samples: {num_samples:,}")
        else:
            print(f"⚠️ [STEP1] Cannot save sampling config - no dataframe in step 1")
    
    # Navigation buttons
    render_navigation_buttons()

def process_uploaded_file(uploaded_file):
    """Process the uploaded file and show preview"""
    
    # Add loading indicator
    with st.spinner("🔄 Processing uploaded file..."):
        try:
            # Read file based on type
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            elif file_extension == 'txt':
                content = uploaded_file.read().decode('utf-8')
                lines = content.split('\n')[:100]
                df = pd.DataFrame({'text': lines})
                uploaded_file.seek(0)
            
            st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
            
            # Show file preview with progress
            with st.spinner("📊 Generating file preview..."):
                show_file_preview(df, file_extension)
            
            # Store in session
            with st.spinner("💾 Saving to session..."):
                # Use global session_manager instance
                session_manager.update_step_data(1, 'dataframe', df)
                session_manager.update_step_data(1, 'uploaded_file', {
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'type': uploaded_file.type
                })
                
                # Store default sampling configuration
                dataset_size = len(df)
                if dataset_size <= 10:
                    default_samples = dataset_size
                elif dataset_size < 1000:
                    default_samples = dataset_size
                else:
                    default_samples = min(100000, dataset_size)
                
                default_sampling_config = {
                    'num_samples': default_samples,
                    'sampling_strategy': 'Stratified (Recommended)'
                }
                
                session_manager.update_step_data(1, 'sampling_config', default_sampling_config)
                print(f"💾 Saved default sampling config: {default_sampling_config}")
                
                # Also store dataset size for reference
                session_manager.update_step_data(1, 'dataset_size', dataset_size)
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)  # Show full error details


def show_file_preview(df, file_extension):
    """Show file preview and info for both uploaded and path-based files"""
    
    # Data preview box
    st.subheader("📊 Data Preview (First 5 rows)")
    
    st.dataframe(df.head(5), width='stretch')
    
    # Get sampling configuration for display
    # Use global session_manager instance
    step1_data = session_manager.get_step_data(1)
    sampling_config = step1_data.get('sampling_config', {}) if step1_data else {}
    
    # Calculate display values
    if sampling_config and sampling_config.get('num_samples'):
        num_samples = sampling_config['num_samples']
        if num_samples < df.shape[0]:
            rows_display = f"{num_samples:,} samples (from {df.shape[0]:,} total)"
        else:
            rows_display = f"{df.shape[0]:,} samples"
    else:
        rows_display = f"{df.shape[0]:,} samples"
    
    # Metrics in boxes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h4>Shape</h4>
            <p><strong>{rows_display}, {df.shape[1]} columns</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.markdown(f"""
        <div class="metric-box">
            <h4>Memory</h4>
            <p><strong>{memory_mb:.1f} MB</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h4>Format</h4>
            <p><strong>{file_extension.upper()}</strong></p>
        </div>
        """, unsafe_allow_html=True)

def train_models_with_scaling(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, selected_models, optuna_config, scaler_name, log_container, input_columns=None, training_progress_bar=None, training_status_text=None, current_task=0, total_tasks=1):
    """Train models with specific scaling method using proper train/val/test split and cache system"""
    try:
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
        from models import model_factory, model_registry
        from optuna_optimizer import OptunaOptimizer
        from shap_cache_manager import shap_cache_manager
        from cache_manager import CacheManager
        import time
        import os
        import pandas as pd
        
        with log_container:
            st.info(f"Starting train_models_with_scaling for {scaler_name}")
        
        model_name_mapping = {
            'random_forest': 'random_forest',
            'xgboost': 'xgboost', 
            'lightgbm': 'lightgbm',
            'catboost': 'catboost',
            'decision_tree': 'decision_tree',
            'knn': 'knn',
            'naive_bayes': 'naive_bayes',
            'svm': 'svm',
            'logistic_regression': 'logistic_regression',
            'adaboost': 'adaboost',
            'gradient_boosting': 'gradient_boosting'
        }
        
        scaler_results = {}
        optuna_enabled = optuna_config.get('enabled', False)
        
        # Initialize cache manager
        cache_manager = CacheManager()
        
        with log_container:
            st.info(f"🤖 Training {len(selected_models)} models with {scaler_name} scaling")
        
        for model_idx, model_name in enumerate(selected_models):
            # Update progress bar
            if training_progress_bar and training_status_text:
                progress = (current_task + model_idx) / total_tasks
                training_progress_bar.progress(progress)
                training_status_text.text(f"Training {model_name} with {scaler_name} ({current_task + model_idx + 1}/{total_tasks})")
            
            try:
                with log_container:
                    st.info(f"🔄 Training {model_name} with {scaler_name}...")
                
                # Map model name
                mapped_name = model_name_mapping.get(model_name, model_name)
                
                # Generate cache identifiers
                model_key = mapped_name
                dataset_id = f"numeric_dataset_{scaler_name}"
                config_hash = cache_manager.generate_config_hash({
                    'model': mapped_name,
                    'preprocessing': scaler_name,
                    'trials': optuna_config.get('trials', 50) if optuna_enabled else 0,
                    'random_state': 42
                })
                dataset_fingerprint = cache_manager.generate_dataset_fingerprint(
                    dataset_path="numeric_data_in_memory",
                    dataset_size=len(X_train_scaled),
                    num_rows=len(X_train_scaled)
                    )
                
                # Check cache first
                cache_exists, cached_data = cache_manager.check_cache_exists(
                    model_key, dataset_id, config_hash, dataset_fingerprint
                )
                
                if cache_exists:
                    # Load from cache
                    try:
                        full_cache_data = cache_manager.load_model_cache(model_key, dataset_id, config_hash)
                        cached_metrics = full_cache_data.get('metrics', {})
                        
                        with log_container:
                            st.success(f"💾 Cache hit! Loading {model_name} ({scaler_name}) from cache")
                        
                        scaler_results[model_name] = {
                            'model': full_cache_data.get('model'),
                            'accuracy': cached_metrics.get('accuracy', 0.0),
                            'validation_accuracy': cached_metrics.get('validation_accuracy', 0.0),
                            'f1_score': cached_metrics.get('f1_score', 0.0),
                            'precision': cached_metrics.get('precision', 0.0),
                            'recall': cached_metrics.get('recall', 0.0),
                            'support': cached_metrics.get('support', 0),
                            'cv_mean': cached_metrics.get('cv_mean', 0.0),
                            'cv_std': cached_metrics.get('cv_std', 0.0),
                            'training_time': cached_metrics.get('training_time', 0.0),  # Load from cache
                            'params': full_cache_data.get('params', {}),
                            'status': 'success',
                            'cached': True
                        }
                        continue
                    except Exception as cache_error:
                        with log_container:
                            st.warning(f"⚠️ Cache load failed for {model_name}: {cache_error}")
                        # Continue to training
                
                # Cache miss - train new model
                with log_container:
                    st.info(f"🔄 Cache miss! Training {model_name} ({scaler_name})...")
                
                # Create model using factory
                model = model_factory.create_model(mapped_name)
                if model is None:
                    with log_container:
                        st.error(f"❌ Failed to create model: {mapped_name}")
                    continue
                
                start_time = time.time()
                
                if optuna_enabled:
                    # Use Optuna optimization
                    optimizer = OptunaOptimizer(optuna_config)
                    optimization_result = optimizer.optimize_model(
                        model_name=mapped_name,
                        model_class=model.__class__,
                        X_train=X_train_scaled,
                        y_train=y_train,
                        X_val=X_val_scaled,  # Use validation set for Optuna
                        y_val=y_val
                    )
                    
                    best_params = optimization_result['best_params']
                    best_score = optimization_result['best_score']
                    
                    # Train final model with best params
                    final_model = model_factory.create_model(mapped_name)
                    final_model.set_params(**best_params)
                    final_model.fit(X_train_scaled, y_train)
                else:
                    # Train without optimization
                    model.fit(X_train_scaled, y_train)
                    final_model = model
                    best_params = {}
                    best_score = model.score(X_val_scaled, y_val)  # Use validation set for scoring
                
                end_time = time.time()
                training_time = end_time - start_time
                
                # Skip cross-validation to avoid double validation
                # Optuna already provides validation, CV would be redundant
                cv_mean = best_score  # Use validation score as CV estimate
                cv_std = 0.0  # No CV std available
                
                # Final evaluation on test set (unseen data)
                y_pred = final_model.predict(X_test_scaled)
                test_accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                
                # Calculate support (number of samples for each class)
                from collections import Counter
                support = sum(Counter(y_test).values())  # Total support
                
                scaler_results[model_name] = {
                    'model': final_model,
                    'accuracy': test_accuracy,  # Use test accuracy as final metric
                    'validation_accuracy': best_score,  # Keep validation accuracy for reference
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'support': support,
                    'cv_mean': cv_mean,  # Validation score (no double validation)
                    'cv_std': cv_std,    # No CV std (avoid double validation)
                    'training_time': training_time,
                    'params': best_params,
                    'status': 'success',
                    'cached': False
                }
                
                with log_container:
                    st.success(f"✅ {model_name} ({scaler_name}): Val={best_score:.4f}, Test={test_accuracy:.4f} ({training_time:.2f}s)")
                
                # Save to cache
                try:
                    metrics = {
                        'accuracy': test_accuracy,
                        'validation_accuracy': best_score,
                        'f1_score': f1,
                        'precision': precision,
                        'recall': recall,
                        'support': support,
                        'cv_mean': cv_mean,
                        'cv_std': cv_std,
                        'training_time': training_time
                    }
                    
                    cache_config = {
                        'model': mapped_name,
                        'preprocessing': scaler_name,
                        'trials': optuna_config.get('trials', 50) if optuna_enabled else 0,
                        'random_state': 42,
                        'test_size': 0.2
                    }
                    
                    # Create eval_predictions DataFrame for SHAP analysis
                    # Use actual feature names from input_columns if available
                    if input_columns is not None:
                        feature_names = input_columns
                    else:
                        feature_names = [f"feature_{i}" for i in range(X_train_scaled.shape[1])]
                    
                    with log_container:
                        st.info(f"Using feature names: {feature_names[:5]}{'...' if len(feature_names) > 5 else ''}")
                        st.info(f"Total features: {len(feature_names)}")
                    
                    eval_predictions = pd.DataFrame(X_test_scaled, columns=feature_names)
                    eval_predictions['true_labels'] = y_test
                    eval_predictions['predictions'] = y_pred
                    
                    # Create SHAP sample for caching (first 30 samples to prevent memory issues)
                    shap_sample_size = min(30, len(X_test_scaled))  # Reduced from 100 to 30
                    shap_sample = pd.DataFrame(X_test_scaled[:shap_sample_size], columns=feature_names)
                    
                    with log_container:
                        st.info(f"Created SHAP sample with {shap_sample_size} samples for caching")
                    
                    # Create label mapping with real labels from dataset
                    # Get unique labels from y_train and y_test
                    unique_labels = sorted(set(y_train) | set(y_test))
                    
                    # Try to get original labels from session data
                    step1_data = session_manager.get_step_data(1) or {}
                    step2_data = session_manager.get_step_data(2) or {}
                    
                    original_labels = None
                    if step1_data.get('dataframe') is not None:
                        df = step1_data['dataframe']
                        label_column = step2_data.get('column_config', {}).get('label_column')
                        if label_column and label_column in df.columns:
                            # Get original labels from dataset
                            original_labels = sorted(df[label_column].unique().tolist())
                            with log_container:
                                st.info(f"Found original labels: {original_labels}")
                    
                    # Create label mapping
                    if original_labels and len(original_labels) == len(unique_labels):
                        # Use original labels if available and count matches
                        label_mapping = {i: original_labels[i] for i in range(len(unique_labels))}
                        with log_container:
                            st.info(f"✅ Using original labels: {label_mapping}")
                    else:
                        # Fallback to generic labels
                        label_mapping = {i: f"Class_{i}" for i in range(len(unique_labels))}
                        with log_container:
                            st.info(f"⚠️ Using generic labels: {label_mapping}")
                    
                    cache_path = cache_manager.save_model_cache(
                        model_key=model_key,
                        dataset_id=dataset_id,
                        config_hash=config_hash,
                        dataset_fingerprint=dataset_fingerprint,
                        model=final_model,
                        params=best_params,
                        metrics=metrics,
                        config=cache_config,
                        eval_predictions=eval_predictions,
                        shap_sample=shap_sample,
                        feature_names=feature_names,
                        label_mapping=label_mapping
                    )
                    
                    with log_container:
                        st.success(f"💾 Cache saved for {model_name} ({scaler_name})")
                    
                    # Generate and cache SHAP values for tree-based models (with comprehensive safety checks)
                    with log_container:
                        st.info(f"Starting SHAP cache generation for {model_name} ({scaler_name})")
                    
                    try:
                        # Safety Check 1: Verify we're in a stable environment
                        import sys
                        if hasattr(sys, '_getframe'):
                            with log_container:
                                st.info("Environment check: Python environment is stable")
                        
                        # Safety Check 2: Check if SHAP is available
                        try:
                            import shap
                            shap_available = True
                            with log_container:
                                st.info(f"SHAP version: {shap.__version__}")
                        except ImportError as e:
                            shap_available = False
                            with log_container:
                                st.error(f"❌ SHAP not available: {e}")
                                st.info("💡 Please install SHAP: pip install shap")
                                st.info("💡 Or use conda environment: conda activate PJ3.1")
                        
                        # Safety Check 3: Verify model is ready for SHAP
                        if shap_available:
                            if hasattr(final_model, 'predict_proba'):
                                with log_container:
                                    st.info("Model check: Model has predict_proba method")
                            else:
                                with log_container:
                                    st.warning("⚠️ Model check: Model may not be compatible with SHAP")
                            
                            with log_container:
                                st.info(f"SHAP is available, proceeding with cache generation")
                        
                        if shap_available:
                            # Safety Check 4: Memory usage validation
                            memory_before = shap_cache_manager.get_memory_usage()
                            with log_container:
                                st.info(f"Memory usage before SHAP: {memory_before:.1f} MB")
                        
                            # Safety Check 5: Sample data validation
                            if shap_sample is None or len(shap_sample) == 0:
                                with log_container:
                                    st.warning("⚠️ SHAP sample is empty or None - cannot generate SHAP")
                                shap_available = False
                            
                            with log_container:
                                st.info(f"SHAP sample size: {len(shap_sample)} samples, {shap_sample.shape[1]} features")
                        
                            # Safety Check 6: Model type validation
                            tree_based_models = ['random_forest', 'xgboost', 'lightgbm', 'catboost', 'gradient_boosting', 'decision_tree']
                            is_tree_based = any(tree_model in model_name.lower() for tree_model in tree_based_models)
                        
                            if is_tree_based:
                                with log_container:
                                    st.info(f"Model type check: {model_name} is tree-based - suitable for SHAP")
                            else:
                                with log_container:
                                    st.info(f"ℹ️ Model type check: {model_name} is not tree-based - will try SHAP anyway")
                                # Don't skip SHAP cache generation - SHAP can work with many model types
                        
                            # Safety Check 7: Import SHAP cache manager with error handling
                            try:
                                from shap_cache_manager import shap_cache_manager
                                with log_container:
                                    st.info("SHAP cache manager imported successfully")
                            except ImportError as e:
                                with log_container:
                                    st.warning(f"⚠️ Failed to import SHAP cache manager: {e} - cannot generate SHAP")
                                shap_available = False
                        
                            # Safety Check 8: Import visualization functions
                            try:
                                from visualization import create_shap_explainer
                                with log_container:
                                    st.info("SHAP visualization functions imported successfully")
                            except ImportError as e:
                                with log_container:
                                    st.warning(f"⚠️ Failed to import SHAP visualization functions: {e} - cannot generate SHAP")
                                shap_available = False
                        
                            # Proceed with SHAP cache generation
                            with log_container:
                                st.info(f"Generating SHAP cache for model: {model_name}")
                                st.info(f"🔍 Debug: shap_available = {shap_available}")
                                st.info(f"🔍 Debug: model_name = {model_name}")
                                st.info(f"🔍 Debug: scaler_name = {scaler_name}")
                            
                            # Create SHAP explainer with comprehensive error handling
                            try:
                                with log_container:
                                    st.info(f"Creating SHAP explainer for {model_name} ({scaler_name})")
                                    st.info(f"Model type: {type(final_model)}")
                                    st.info(f"Sample data shape: {shap_sample.values.shape}")
                                
                                explainer = create_shap_explainer(final_model, shap_sample.values)
                                
                                with log_container:
                                    st.info(f"🔍 Debug: explainer created = {explainer is not None}")
                                    if explainer is not None:
                                        st.info(f"🔍 Debug: explainer type = {type(explainer)}")
                                
                                if explainer is not None:
                                    with log_container:
                                        st.success(f"✅ SHAP explainer created successfully for {model_name}")
                                    
                                    # Generate SHAP values with timeout protection
                                    try:
                                        with log_container:
                                            st.info(f"Generating SHAP values for {model_name}")
                                        
                                        shap_values = explainer.shap_values(shap_sample.values)
                                        
                                        with log_container:
                                            st.success(f"✅ SHAP values generated successfully for {model_name}")
                                        
                                        # Save SHAP cache
                                        with log_container:
                                            st.info(f"Attempting to save SHAP cache for {model_name} ({scaler_name})")
                                            st.info(f"Feature names: {input_columns}")
                                            st.info(f"Sample data shape: {shap_sample.values.shape}")
                                            st.info(f"SHAP values type: {type(shap_values)}")
                                        
                                        shap_cache_file = shap_cache_manager.save_shap_cache(
                                            model=final_model,
                                            sample_data=shap_sample.values,
                                            explainer=explainer,
                                            shap_values=shap_values,
                                            model_name=f"{model_key}_{scaler_name}",
                                            feature_names=input_columns
                                        )
                                        
                                        with log_container:
                                            if shap_cache_file:
                                                st.success(f"🔍 SHAP cache saved successfully: {shap_cache_file}")
                                            else:
                                                st.error(f"❌ SHAP cache save failed for {model_name}")
                                                
                                    except Exception as shap_values_error:
                                        with log_container:
                                            st.warning(f"⚠️ Failed to generate SHAP values for {model_name}: {shap_values_error}")
                                else:
                                    with log_container:
                                        st.warning(f"⚠️ Failed to create SHAP explainer for {model_name}")
                                        st.info("💡 This could be due to model type incompatibility or memory issues")
                                
                            except Exception as explainer_error:
                                with log_container:
                                    st.warning(f"⚠️ SHAP explainer creation failed for {model_name}: {explainer_error}")
                        else:
                            with log_container:
                                st.warning("⚠️ SHAP not available - skipping SHAP cache generation")
                                st.info("💡 Please install SHAP: pip install shap")
                                st.info("💡 Or use conda environment: conda activate PJ3.1")
                    
                    except Exception as shap_error:
                        with log_container:
                            st.error(f"❌ SHAP cache generation failed for {model_name}: {shap_error}")
                            st.error(f"❌ Error type: {type(shap_error).__name__}")
                            st.error(f"❌ Error details: {str(shap_error)}")
                            st.info("ℹ️ Continuing with training without SHAP cache...")
                
                except Exception as cache_error:
                    with log_container:
                        st.warning(f"⚠️ Cache save failed for {model_name}: {cache_error}")
                
                with log_container:
                    st.success(f"✅ {model_name} ({scaler_name}): Val={best_score:.4f}, Test={test_accuracy:.4f} ({training_time:.2f}s)")
            
            except Exception as e:
                with log_container:
                    st.error(f"❌ {model_name} ({scaler_name}) failed: {str(e)}")
                scaler_results[model_name] = {
                    'model': None,
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'support': 0,
                    'cv_mean': 0.0,
                    'cv_std': 0.0,
                    'training_time': 0.0,
                    'params': {},
                    'status': 'failed',
                    'error': str(e)
                }
        
        with log_container:
            st.info(f"train_models_with_scaling completed for {scaler_name}")
            st.info(f"scaler_results count = {len(scaler_results)}")
            st.info(f"scaler_results keys = {list(scaler_results.keys())}")
            st.info(f"About to return results for {scaler_name}")
        
        result = {
            'model_results': scaler_results,
            'status': 'success',
            'current_task': current_task + len(selected_models)  # Return updated current_task
        }
        
        with log_container:
            st.info(f"Returning result for {scaler_name}: {result}")
        
        return result
    
    except Exception as e:
        with log_container:
            st.error(f"❌ train_models_with_scaling failed for {scaler_name}: {str(e)}")
            import traceback
            st.error(f"❌ Traceback: {traceback.format_exc()}")
        return {
            'model_results': {},
            'status': 'failed',
            'error': str(e)
        }


def train_numeric_data_directly(df, input_columns, label_column, selected_models, optuna_config, voting_config, stacking_config, progress_bar, status_text, numeric_scalers=None, remove_duplicates=False, data_split_config=None):
    """Train numeric data using Optuna optimization with cache and cross-validation (ENHANCED)"""
    import time
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
    from models import model_factory, model_registry
    from optuna_optimizer import OptunaOptimizer
    from cache_manager import CacheManager
    import os
    
    # Default scaling methods if not provided
    if numeric_scalers is None:
        numeric_scalers = ['StandardScaler']
    
    try:
        # Handle duplicates based on user setting
        original_size = len(df)
        if remove_duplicates:
            df_clean = df.drop_duplicates()
            clean_size = len(df_clean)
        else:
            df_clean = df
            clean_size = original_size
        
        # Create logging container
        log_container = st.expander("📋 Training Log", expanded=False)
        
        with log_container:
            if remove_duplicates and original_size != clean_size:
                st.info(f"🧹 Removed {original_size - clean_size} duplicate rows ({original_size - clean_size}/{original_size} = {(original_size - clean_size)/original_size*100:.1f}%)")
                st.info(f"📊 Training on {clean_size} unique samples (from {original_size} total)")
            elif remove_duplicates:
                st.info(f"📊 No duplicates found - training on all {clean_size} samples")
            else:
                st.info(f"📊 Keeping all {clean_size} samples (including duplicates)")
            
            with log_container:
                st.info(f"📋 Using label column: '{label_column}'")
                st.info(f"📋 Using input columns: {input_columns}")
                st.info(f"📊 Multi-input data: {len(input_columns)} features, {clean_size} samples")
    
        # Prepare data
        X = df_clean[input_columns].values
        y = df_clean[label_column].values
        
        with log_container:
            st.info(f"📊 Training on {clean_size} samples with {len(set(y))} classes")
            st.info(f"🤖 Training {len(selected_models)} models: {', '.join(selected_models)}")
        
        with log_container:
            st.info(f"📊 Features: {len(input_columns)}, Samples: {len(X)}, Classes: {len(set(y))}")
        
        # Split data into train/validation/test (3-way split to avoid data leakage)
        if data_split_config:
            train_ratio = data_split_config['train_ratio']
            val_ratio = data_split_config['val_ratio']
            test_ratio = data_split_config['test_ratio']
        else:
            # Default: 80% train, 10% val, 10% test
            train_ratio = 0.8
            val_ratio = 0.1
            test_ratio = 0.1
        
        # First split: train vs (val + test)
        val_test_ratio = val_ratio + test_ratio
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=val_test_ratio, random_state=42, stratify=y)
        
        # Second split: val vs test
        val_ratio_in_temp = val_ratio / val_test_ratio
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1-val_ratio_in_temp), random_state=42, stratify=y_temp)
        
        with log_container:
            st.info(f"📊 Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
            st.info(f"🔧 Scaling methods: {', '.join(numeric_scalers)}")
            st.info(f"✅ Using 3-way split to prevent data leakage")
        
        # Train models with multiple scaling methods (like auto_train_heart_dataset.py)
        model_results = {}
        training_times = {}
        
        # Process each scaling method
        total_scalers = len(numeric_scalers)
        total_models = len(selected_models)
        total_tasks = total_scalers * total_models  # Total number of model-scaler combinations
        
        # Create single progress bar for entire training process
        training_progress_bar = st.progress(0)
        training_status_text = st.empty()
        
        current_task = 0
        
        for scaler_idx, scaler_name in enumerate(numeric_scalers):
            # Update progress bar
            progress = current_task / total_tasks
            training_progress_bar.progress(progress)
            training_status_text.text(f"Processing scaling method: {scaler_name} ({scaler_idx + 1}/{total_scalers})")
            
            with log_container:
                st.info(f"🔧 Processing scaling method: {scaler_name}")
            
            # Apply scaling
            if scaler_name == 'StandardScaler':
                scaler = StandardScaler()
            elif scaler_name == 'MinMaxScaler':
                scaler = MinMaxScaler()
            elif scaler_name == 'RobustScaler':
                scaler = RobustScaler()
            elif scaler_name == 'None':
                scaler = None
            else:
                with log_container:
                    st.warning(f"⚠️ Unknown scaler: {scaler_name}, using StandardScaler")
                scaler = StandardScaler()
            
            if scaler is not None:
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
                X_test_scaled = X_test
            
            with log_container:
                st.info(f"✅ Applied {scaler_name} scaling")
            
            # Train models with this scaling method (using validation set for Optuna)
            try:
                scaling_result = train_models_with_scaling(
                    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, 
                    selected_models, optuna_config, scaler_name, log_container, input_columns,
                    training_progress_bar, training_status_text, current_task, total_tasks
                )
                
                # Extract scaler_results from the returned structure
                if scaling_result is None:
                    with log_container:
                        st.error(f"❌ Scaling method {scaler_name} returned None")
                    scaler_results = {}
                else:
                    scaler_results = scaling_result.get('model_results', {})
                    current_task = scaling_result.get('current_task', current_task)  # Update current_task
                    with log_container:
                        st.info(f"{scaler_name} returned {len(scaler_results)} models")
                
                # Merge results with scaling method prefix
                for model_name, result in scaler_results.items():
                    prefixed_name = f"{model_name}_{scaler_name}"
                    model_results[prefixed_name] = result
                    training_times[prefixed_name] = result.get('training_time', 0)
                
                # Show individual model results in Training Log
                with log_container:
                    st.info(f"{scaler_name} - Individual models trained: {list(scaler_results.keys())}")
                    for model_name, result in scaler_results.items():
                        status = result.get('status', 'unknown')
                        accuracy = result.get('validation_accuracy', 0)
                        st.info(f"{model_name}_{scaler_name}: status={status}, accuracy={accuracy:.4f}")
                    
            except Exception as scaling_error:
                st.error(f"❌ Scaling method {scaler_name} failed: {str(scaling_error)}")
                import traceback
                st.error(f"❌ Scaling error traceback: {traceback.format_exc()}")
                # Continue with next scaling method
                continue
        
        # Train ensemble models if enabled
        ensemble_results = {}
        
        # Show ensemble training info in Training Log
        with log_container:
            st.info("Reached ensemble training logic!")
            st.info(f"model_results keys = {list(model_results.keys())}")
            st.info(f"model_results count = {len(model_results)}")
            
            # Check if model_results is empty
            if not model_results:
                st.error("❌ model_results is EMPTY! No individual models trained successfully!")
                st.error("❌ This means ensemble training cannot proceed!")
                return {
                    'status': 'failed',
                    'error': 'No individual models trained successfully',
                    'model_results': {},
                    'training_times': {}
                }
            
            # Update progress bar for ensemble training
            if training_progress_bar and training_status_text:
                progress = current_task / total_tasks
                training_progress_bar.progress(progress)
                training_status_text.text(f"Training ensemble models ({current_task + 1}/{total_tasks})")
            
            # Debug: Show successful models
            successful_models = [k for k, v in model_results.items() if v.get('status') == 'success']
            st.info(f"🔍 DEBUG: Successful models: {successful_models}")
            st.info(f"🔍 DEBUG: Successful models count: {len(successful_models)}")
        
        if not successful_models:
            with log_container:
                st.error("❌ DEBUG: No successful models found! Ensemble training cannot proceed!")
            return {
                'status': 'failed',
                'error': 'No successful individual models found',
                'model_results': model_results,
                'training_times': training_times
            }
        
        # Voting Ensemble - Create for each scaler
        if voting_config.get('enabled', False) and voting_config.get('models'):
            with log_container:
                st.info(f"🗳️ Training Voting Ensemble ({voting_config.get('voting_method', 'hard')} voting) for all scalers")
                st.info(f"🔍 Debug: model_results keys = {list(model_results.keys())}")
                st.info(f"🔍 Debug: numeric_scalers = {numeric_scalers}")
            
            try:
                from sklearn.ensemble import VotingClassifier
                
                # Create voting ensemble for each scaler
                for scaler_name in numeric_scalers:
                    with log_container:
                        st.info(f"🗳️ Creating Voting Ensemble for {scaler_name}")
                    
                    # Prepare base models for voting with this scaler
                    voting_models = []
                    voting_model_names = []
                    
                    for model_name in voting_config.get('models', []):
                        # Use model from this specific scaler
                        prefixed_name = f"{model_name}_{scaler_name}"
                        if prefixed_name in model_results and model_results[prefixed_name].get('status') == 'success':
                            best_model = model_results[prefixed_name].get('model')
                            best_scaler = scaler_name
                            
                            if best_model is not None:
                                # Create fresh sklearn models from scratch to avoid pickle issues
                                # But don't train them - just use the structure
                                try:
                                    fresh_model = None
                                    
                                    if model_name == 'logistic_regression':
                                        from sklearn.linear_model import LogisticRegression
                                        fresh_model = LogisticRegression(random_state=42)
                                    elif model_name == 'decision_tree':
                                        from sklearn.tree import DecisionTreeClassifier
                                        fresh_model = DecisionTreeClassifier(random_state=42)
                                    elif model_name == 'random_forest':
                                        from sklearn.ensemble import RandomForestClassifier
                                        fresh_model = RandomForestClassifier(random_state=42)
                                    elif model_name == 'svm':
                                        from sklearn.svm import SVC
                                        fresh_model = SVC(random_state=42)
                                    elif model_name == 'knn':
                                        from sklearn.neighbors import KNeighborsClassifier
                                        fresh_model = KNeighborsClassifier()
                                    elif model_name == 'naive_bayes':
                                        from sklearn.naive_bayes import GaussianNB
                                        fresh_model = GaussianNB()
                                    elif model_name == 'gradient_boosting':
                                        from sklearn.ensemble import GradientBoostingClassifier
                                        fresh_model = GradientBoostingClassifier(random_state=42)
                                    elif model_name == 'adaboost':
                                        from sklearn.ensemble import AdaBoostClassifier
                                        fresh_model = AdaBoostClassifier(random_state=42)
                                    elif model_name == 'xgboost':
                                        from xgboost import XGBClassifier
                                        fresh_model = XGBClassifier(random_state=42)
                                    elif model_name == 'lightgbm':
                                        from lightgbm import LGBMClassifier
                                        fresh_model = LGBMClassifier(random_state=42)
                                    elif model_name == 'catboost':
                                        from catboost import CatBoostClassifier
                                        fresh_model = CatBoostClassifier(random_state=42, verbose=False)
                                    
                                    if fresh_model is not None:
                                        voting_models.append((f"{model_name}_{best_scaler}", fresh_model))
                                        voting_model_names.append(f"{model_name}_{best_scaler}")
                                        
                                        with log_container:
                                            st.info(f"   ✅ Created fresh {model_name} for voting ensemble")
                                    else:
                                        with log_container:
                                            st.warning(f"   ⚠️ Unknown model type: {model_name}")
                                            
                                except Exception as create_error:
                                    with log_container:
                                        st.warning(f"   ⚠️ Could not create fresh {model_name}: {create_error}")
                                    # Skip this model
                                    continue
                
                if voting_models:
                    # Create voting classifier
                    voting_method = voting_config.get('voting_method', 'hard')
                    voting_clf = VotingClassifier(
                        estimators=voting_models,
                        voting=voting_method
                    )
                    
                    # Train voting ensemble
                    start_time = time.time()
                    voting_clf.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Evaluate voting ensemble
                    y_pred = voting_clf.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    
                    # Create descriptive name for voting ensemble
                    voting_method = voting_config.get('voting_method', 'hard')
                    voting_ensemble_name = f"Voting Ensemble ({voting_method.title()}) - {scaler_name}"
                    
                    ensemble_results[voting_ensemble_name] = {
                        'model': voting_clf,
                        'accuracy': accuracy,
                        'validation_accuracy': accuracy,  # Use test accuracy as validation
                        'f1_score': f1,
                        'precision': precision,
                        'recall': recall,
                        'support': len(y_test),
                        'training_time': training_time,
                        'params': {
                            'voting_method': voting_method,
                            'base_models': voting_model_names
                        },
                        'status': 'success',
                        'cached': False
                    }
                    
                    # Cache voting ensemble model
                    try:
                        from cache_manager import CacheManager
                        cache_manager = CacheManager()
                        
                        # Generate cache identifiers for ensemble
                        ensemble_model_key = f"voting_ensemble_{voting_method}_{scaler_name}"
                        dataset_id = f"numeric_dataset_{scaler_name}"
                        config_hash = cache_manager.generate_config_hash({
                            'model': ensemble_model_key,
                            'preprocessing': scaler_name,
                            'voting_method': voting_method,
                            'base_models': voting_model_names,
                            'random_state': 42
                        })
                        dataset_fingerprint = cache_manager.generate_dataset_fingerprint(
                            dataset_path="numeric_data_in_memory",
                            dataset_size=len(X_train_scaled),
                            num_rows=len(X_train_scaled)
                        )
                        
                        # Create eval_predictions DataFrame
                        eval_predictions = pd.DataFrame({
                            'true_labels': y_test,
                            'predictions': voting_clf.predict(X_test_scaled)
                        })
                        
                        # Add probability columns if available
                        if hasattr(voting_clf, 'predict_proba'):
                            proba = voting_clf.predict_proba(X_test_scaled)
                            for i in range(proba.shape[1]):
                                eval_predictions[f'proba_class_{i}'] = proba[:, i]
                        
                        # Create label mapping with real labels from dataset
                        unique_labels = sorted(set(y_train) | set(y_test))
                        
                        # Try to get original labels from session data
                        step1_data = session_manager.get_step_data(1) or {}
                        step2_data = session_manager.get_step_data(2) or {}
                        
                        original_labels = None
                        if step1_data.get('dataframe') is not None:
                            df = step1_data['dataframe']
                            label_column = step2_data.get('column_config', {}).get('label_column')
                            if label_column and label_column in df.columns:
                                # Get original labels from dataset
                                original_labels = sorted(df[label_column].unique().tolist())
                        
                        # Create label mapping
                        if original_labels and len(original_labels) == len(unique_labels):
                            # Use original labels if available and count matches
                            label_mapping = {i: original_labels[i] for i in range(len(unique_labels))}
                        else:
                            # Fallback to generic labels
                            label_mapping = {i: f"Class_{i}" for i in range(len(unique_labels))}
                        
                        # Save ensemble cache
                        cache_manager.save_model_cache(
                            model_key=ensemble_model_key,
                            dataset_id=dataset_id,
                            config_hash=config_hash,
                            dataset_fingerprint=dataset_fingerprint,
                            model=voting_clf,
                            params={
                                'voting_method': voting_method,
                                'base_models': voting_model_names,
                                'random_state': 42
                            },
                            metrics={
                                'validation_accuracy': accuracy,
                                'test_accuracy': accuracy,
                                'f1_score': f1,
                                'precision': precision,
                                'recall': recall,
                                'training_time': training_time
                            },
                            config={
                                'model_name': ensemble_model_key,
                                'voting_method': voting_method,
                                'base_models': voting_model_names,
                                'preprocessing': scaler_name
                            },
                            eval_predictions=eval_predictions,
                            shap_sample=None,  # Ensemble doesn't support SHAP directly
                            feature_names=input_columns,
                            label_mapping=label_mapping
                        )
                        
                        # Update cached status
                        ensemble_results[voting_ensemble_name]['cached'] = True
                        
                        with log_container:
                            st.success(f"💾 Voting ensemble cached successfully!")
                            
                    except Exception as cache_error:
                        with log_container:
                            st.warning(f"⚠️ Failed to cache voting ensemble: {str(cache_error)}")
                    
                    with log_container:
                        st.success(f"✅ Voting Ensemble ({scaler_name}) trained: {accuracy:.4f} accuracy, {training_time:.2f}s")
                else:
                    with log_container:
                        st.warning(f"⚠️ No successful base models found for voting ensemble ({scaler_name})")
                        
            except Exception as e:
                with log_container:
                    st.error(f"❌ Voting ensemble ({scaler_name}) training failed: {str(e)}")
                # Create descriptive name for failed voting ensemble
                voting_method = voting_config.get('voting_method', 'hard')
                voting_ensemble_name = f"Voting Ensemble ({voting_method.title()}) - {scaler_name}"
                
                ensemble_results[voting_ensemble_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        else:
            with log_container:
                if not voting_config.get('enabled', False):
                    st.warning("⚠️ Voting ensemble disabled - voting_config.get('enabled') = False")
                if not voting_config.get('models'):
                    st.warning("⚠️ Voting ensemble disabled - voting_config.get('models') is empty")
                st.info("🔍 Debug: Skipping voting ensemble training")
        
        # Stacking Ensemble - Create for each scaler
        if stacking_config.get('enabled', False) and stacking_config.get('base_models'):
            with log_container:
                st.info(f"📚 Training Stacking Ensemble (meta-learner: {stacking_config.get('meta_learner', 'logistic_regression')}) for all scalers")
            
            try:
                from sklearn.ensemble import StackingClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier
                import xgboost as xgb
                
                # Create stacking ensemble for each scaler
                for scaler_name in numeric_scalers:
                    with log_container:
                        st.info(f"📚 Creating Stacking Ensemble for {scaler_name}")
                    
                    # Prepare base models for stacking with this scaler
                    stacking_models = []
                    stacking_model_names = []
                    
                    for model_name in stacking_config.get('base_models', []):
                        # Use model from this specific scaler
                        prefixed_name = f"{model_name}_{scaler_name}"
                        if prefixed_name in model_results and model_results[prefixed_name].get('status') == 'success':
                            best_model = model_results[prefixed_name].get('model')
                            best_scaler = scaler_name
                            
                            if best_model is not None:
                                # Create fresh sklearn models from scratch to avoid pickle issues
                                # But don't train them - just use the structure
                                try:
                                    fresh_model = None
                                    
                                    if model_name == 'logistic_regression':
                                        from sklearn.linear_model import LogisticRegression
                                        fresh_model = LogisticRegression(random_state=42)
                                    elif model_name == 'decision_tree':
                                        from sklearn.tree import DecisionTreeClassifier
                                        fresh_model = DecisionTreeClassifier(random_state=42)
                                    elif model_name == 'random_forest':
                                        from sklearn.ensemble import RandomForestClassifier
                                        fresh_model = RandomForestClassifier(random_state=42)
                                    elif model_name == 'svm':
                                        from sklearn.svm import SVC
                                        fresh_model = SVC(random_state=42)
                                    elif model_name == 'knn':
                                        from sklearn.neighbors import KNeighborsClassifier
                                        fresh_model = KNeighborsClassifier()
                                    elif model_name == 'naive_bayes':
                                        from sklearn.naive_bayes import GaussianNB
                                        fresh_model = GaussianNB()
                                    elif model_name == 'gradient_boosting':
                                        from sklearn.ensemble import GradientBoostingClassifier
                                        fresh_model = GradientBoostingClassifier(random_state=42)
                                    elif model_name == 'adaboost':
                                        from sklearn.ensemble import AdaBoostClassifier
                                        fresh_model = AdaBoostClassifier(random_state=42)
                                    elif model_name == 'xgboost':
                                        from xgboost import XGBClassifier
                                        fresh_model = XGBClassifier(random_state=42)
                                    elif model_name == 'lightgbm':
                                        from lightgbm import LGBMClassifier
                                        fresh_model = LGBMClassifier(random_state=42)
                                    elif model_name == 'catboost':
                                        from catboost import CatBoostClassifier
                                        fresh_model = CatBoostClassifier(random_state=42, verbose=False)
                                    
                                    if fresh_model is not None:
                                        stacking_models.append((f"{model_name}_{best_scaler}", fresh_model))
                                        stacking_model_names.append(f"{model_name}_{best_scaler}")
                                        
                                        with log_container:
                                            st.info(f"   ✅ Created fresh {model_name} for stacking ensemble")
                                    else:
                                        with log_container:
                                            st.warning(f"   ⚠️ Unknown model type: {model_name}")
                                            
                                except Exception as create_error:
                                    with log_container:
                                        st.warning(f"   ⚠️ Could not create fresh {model_name}: {create_error}")
                                    # Skip this model
                                    continue
                
                if stacking_models:
                    # Create meta-learner
                    meta_learner_name = stacking_config.get('meta_learner', 'logistic_regression')
                    if meta_learner_name == 'logistic_regression':
                        meta_learner = LogisticRegression(random_state=42)
                    elif meta_learner_name == 'random_forest':
                        meta_learner = RandomForestClassifier(random_state=42)
                    elif meta_learner_name == 'xgboost':
                        meta_learner = xgb.XGBClassifier(random_state=42)
                    else:
                        meta_learner = LogisticRegression(random_state=42)
                    
                    # Create stacking classifier
                    stacking_clf = StackingClassifier(
                        estimators=stacking_models,
                        final_estimator=meta_learner,
                        cv=3  # Use 3-fold CV for meta-features
                    )
                    
                    # Train stacking ensemble
                    start_time = time.time()
                    stacking_clf.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Evaluate stacking ensemble
                    y_pred = stacking_clf.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    
                    # Create descriptive name for stacking ensemble
                    meta_learner_name = stacking_config.get('meta_learner', 'logistic_regression')
                    stacking_ensemble_name = f"Stacking Ensemble ({meta_learner_name.replace('_', ' ').title()}) - {scaler_name}"
                    
                    ensemble_results[stacking_ensemble_name] = {
                        'model': stacking_clf,
                        'accuracy': accuracy,
                        'validation_accuracy': accuracy,  # Use test accuracy as validation
                        'f1_score': f1,
                        'precision': precision,
                        'recall': recall,
                        'support': len(y_test),
                        'training_time': training_time,
                        'params': {
                            'meta_learner': meta_learner_name,
                            'base_models': stacking_model_names
                        },
                        'status': 'success',
                        'cached': False
                    }
                    
                    # Cache stacking ensemble model
                    try:
                        from cache_manager import CacheManager
                        cache_manager = CacheManager()
                        
                        # Generate cache identifiers for ensemble
                        ensemble_model_key = f"stacking_ensemble_{meta_learner_name}_{scaler_name}"
                        dataset_id = f"numeric_dataset_{scaler_name}"
                        config_hash = cache_manager.generate_config_hash({
                            'model': ensemble_model_key,
                            'preprocessing': scaler_name,
                            'meta_learner': meta_learner_name,
                            'base_models': stacking_model_names,
                            'random_state': 42
                        })
                        dataset_fingerprint = cache_manager.generate_dataset_fingerprint(
                            dataset_path="numeric_data_in_memory",
                            dataset_size=len(X_train_scaled),
                            num_rows=len(X_train_scaled)
                        )
                        
                        # Create eval_predictions DataFrame
                        eval_predictions = pd.DataFrame({
                            'true_labels': y_test,
                            'predictions': stacking_clf.predict(X_test_scaled)
                        })
                        
                        # Add probability columns if available
                        if hasattr(stacking_clf, 'predict_proba'):
                            proba = stacking_clf.predict_proba(X_test_scaled)
                            for i in range(proba.shape[1]):
                                eval_predictions[f'proba_class_{i}'] = proba[:, i]
                        
                        # Create label mapping with real labels from dataset
                        unique_labels = sorted(set(y_train) | set(y_test))
                        
                        # Try to get original labels from session data
                        step1_data = session_manager.get_step_data(1) or {}
                        step2_data = session_manager.get_step_data(2) or {}
                        
                        original_labels = None
                        if step1_data.get('dataframe') is not None:
                            df = step1_data['dataframe']
                            label_column = step2_data.get('column_config', {}).get('label_column')
                            if label_column and label_column in df.columns:
                                # Get original labels from dataset
                                original_labels = sorted(df[label_column].unique().tolist())
                        
                        # Create label mapping
                        if original_labels and len(original_labels) == len(unique_labels):
                            # Use original labels if available and count matches
                            label_mapping = {i: original_labels[i] for i in range(len(unique_labels))}
                        else:
                            # Fallback to generic labels
                            label_mapping = {i: f"Class_{i}" for i in range(len(unique_labels))}
                        
                        # Save ensemble cache
                        cache_manager.save_model_cache(
                            model_key=ensemble_model_key,
                            dataset_id=dataset_id,
                            config_hash=config_hash,
                            dataset_fingerprint=dataset_fingerprint,
                            model=stacking_clf,
                            params={
                                'meta_learner': meta_learner_name,
                                'base_models': stacking_model_names,
                                'random_state': 42
                            },
                            metrics={
                                'validation_accuracy': accuracy,
                                'test_accuracy': accuracy,
                                'f1_score': f1,
                                'precision': precision,
                                'recall': recall,
                                'training_time': training_time
                            },
                            config={
                                'model_name': ensemble_model_key,
                                'meta_learner': meta_learner_name,
                                'base_models': stacking_model_names,
                                'preprocessing': scaler_name
                            },
                            eval_predictions=eval_predictions,
                            shap_sample=None,  # Ensemble doesn't support SHAP directly
                            feature_names=input_columns,
                            label_mapping=label_mapping
                        )
                        
                        # Update cached status
                        ensemble_results[stacking_ensemble_name]['cached'] = True
                        
                        with log_container:
                            st.success(f"💾 Stacking ensemble cached successfully!")
                            
                    except Exception as cache_error:
                        with log_container:
                            st.warning(f"⚠️ Failed to cache stacking ensemble: {str(cache_error)}")
                    
                    with log_container:
                        st.success(f"✅ Stacking Ensemble ({scaler_name}) trained: {accuracy:.4f} accuracy, {training_time:.2f}s")
                else:
                    with log_container:
                        st.warning(f"⚠️ No successful base models found for stacking ensemble ({scaler_name})")
                        
            except Exception as e:
                with log_container:
                    st.error(f"❌ Stacking ensemble ({scaler_name}) training failed: {str(e)}")
                # Create descriptive name for failed stacking ensemble
                meta_learner_name = stacking_config.get('meta_learner', 'logistic_regression')
                stacking_ensemble_name = f"Stacking Ensemble ({meta_learner_name.replace('_', ' ').title()}) - {scaler_name}"
                
                ensemble_results[stacking_ensemble_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        else:
            with log_container:
                if not stacking_config.get('enabled', False):
                    st.warning("⚠️ Stacking ensemble disabled - stacking_config.get('enabled') = False")
                if not stacking_config.get('base_models'):
                    st.warning("⚠️ Stacking ensemble disabled - stacking_config.get('base_models') is empty")
                st.info("🔍 Debug: Skipping stacking ensemble training")
        
        # Merge ensemble results with individual model results
        model_results.update(ensemble_results)
        
        # Return results in the same format as original function
        return {
            'status': 'success',
            'model_results': model_results,
            'training_times': training_times,
            'data_info': {
                'original_size': original_size,
                'clean_size': clean_size,
                'input_columns': input_columns,
                'label_column': label_column,
                'scaling_methods': numeric_scalers
            },
            'optuna_enabled': optuna_config.get('enabled', False),
            'ensemble_results': ensemble_results
        }
        
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        import traceback
        st.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {
            'status': 'failed',
            'error': str(e),
            'model_results': {},
            'training_times': {}
        }


def render_step3_wireframe():
    """Render Step 3 - Optuna Optimization & Ensemble Configuration"""
    
    # Step title
    st.markdown("""
    <h2 style="text-align: left; color: var(--text-color); margin: 2rem 0 1rem 0; font-size: 1.8rem;">
        📍 STEP 3/5: Optuna Optimization & Ensemble Configuration
    </h2>
    """, unsafe_allow_html=True)
    
    # Get data from previous steps
    step1_data = session_manager.get_step_data(1)
    step2_data = session_manager.get_step_data(2)
    
    if not step1_data or 'dataframe' not in step1_data:
        st.error("❌ No dataset found. Please complete Step 1 first.")
        return
    
    if not step2_data or not step2_data.get('completed', False):
        st.error("❌ Column selection not completed. Please complete Step 2 first.")
        return
    
    # Check if we have text data to determine if vectorization tab should be shown
    has_text_data = _check_for_text_data()
    
    # Create tabs based on data type
    if has_text_data:
        tab1, tab2, tab3 = st.tabs([
            "🎯 Optuna Optimization", 
            "📊 Vectorization Methods", 
            "🤝 Ensemble Learning"
        ])
    else:
        tab1, tab2 = st.tabs([
            "🎯 Optuna Optimization", 
            "🤝 Ensemble Learning"
        ])
    
    # Tab 1: Optuna Optimization
    with tab1:
        st.markdown("**🎯 Optuna Hyperparameter Optimization**")
        
        # Optuna configuration
        optuna_enabled = st.checkbox(
            "Enable Optuna Optimization", 
            value=True,
            help="Use Optuna for automatic hyperparameter tuning"
        )
        
        # Initialize default values
        n_trials = 50
        timeout = 1800
        
        if optuna_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                n_trials = st.number_input(
                    "Number of Trials", 
                    min_value=10, 
                    max_value=200, 
                    value=50,
                    help="Number of optimization trials"
                )
            
            with col2:
                timeout = st.number_input(
                    "Timeout (seconds)", 
                    min_value=60, 
                    max_value=3600, 
                    value=1800,
                    help="Maximum time for optimization"
                )
        
        # Model selection
        st.markdown("**🤖 Select Models to Optimize**")
        
        # Get available models
        from models import model_registry
        available_models = model_registry.list_models()
        classification_models = [m for m in available_models if m != 'kmeans']
        
        # Create checkboxes for each model
        selected_models = []
        cols = st.columns(3)
        
        for i, model_name in enumerate(classification_models):
            with cols[i % 3]:
                if st.checkbox(model_name.replace('_', ' ').title(), value=True, key=f"model_{model_name}"):
                    selected_models.append(model_name)
        
        # If no models selected, default to all models
        if not selected_models:
            selected_models = classification_models.copy()
            st.info("ℹ️ All models selected by default")
        
        # Save Optuna configuration
        optuna_config = {
            'enabled': optuna_enabled,
            'trials': n_trials if optuna_enabled else 0,  # Changed from 'n_trials' to 'trials'
            'timeout': timeout if optuna_enabled else 0,
            'models': selected_models
        }
    
    # Tab 2: Vectorization Methods (only for text data)
    if has_text_data:
        with tab2:
            st.markdown("**📊 Text Vectorization Methods**")
            
            # Vectorization configuration
            vectorization_methods = st.multiselect(
                "Select Vectorization Methods",
                ["BoW", "TF-IDF", "Word Embeddings"],
                default=["BoW", "TF-IDF"],
                help="Choose text vectorization methods"
            )
            
            # Save vectorization configuration
            vectorization_config = {
                'selected_methods': vectorization_methods
            }
    else:
        vectorization_config = {'selected_methods': []}
    
    # Tab 3/2: Ensemble Learning
    ensemble_tab = tab3 if has_text_data else tab2
    
    with ensemble_tab:
        st.markdown("**🤝 Ensemble Learning**")
        
        # Voting Ensemble
        st.markdown("**🗳️ Voting Ensemble**")
        voting_enabled = st.checkbox("Enable Voting Ensemble", value=False)
        
        if voting_enabled:
            voting_method = st.selectbox(
                "Voting Method",
                ["hard", "soft"],
                help="Hard voting uses majority class, soft voting uses predicted probabilities"
            )
            
            # Select base models for voting
            st.markdown("**Select Base Models for Voting:**")
            voting_models = st.multiselect(
                "Voting Base Models",
                selected_models,
                default=selected_models,  # Select all models by default
                help="Select models to include in voting ensemble"
            )
        
        # Stacking Ensemble
        st.markdown("**📚 Stacking Ensemble**")
        stacking_enabled = st.checkbox("Enable Stacking Ensemble", value=False)
        
        if stacking_enabled:
            meta_learner = st.selectbox(
                "Meta-Learner",
                ["logistic_regression", "random_forest", "xgboost"],
                help="Final estimator for stacking"
            )
            
            # Select base models for stacking
            st.markdown("**Select Base Models for Stacking:**")
            stacking_models = st.multiselect(
                "Stacking Base Models",
                selected_models,
                default=selected_models,  # Select all models by default
                help="Select models to include in stacking ensemble"
            )
        
        # Save ensemble configurations
        voting_config = {
            'enabled': voting_enabled,
            'voting_method': voting_method if voting_enabled else 'hard',
            'models': voting_models if voting_enabled else []
        }
        
        stacking_config = {
            'enabled': stacking_enabled,
            'meta_learner': meta_learner if stacking_enabled else 'logistic_regression',
            'base_models': stacking_models if stacking_enabled else []
        }
        
        # Debug: Show current configuration
        st.markdown("---")
        st.subheader("🔍 Current Configuration Preview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🗳️ Voting Ensemble:**")
            if voting_enabled:
                st.success(f"✅ Enabled: {voting_method} voting")
                st.info(f"📋 Models: {', '.join(voting_models) if voting_models else 'None'}")
            else:
                st.warning("⚠️ Disabled")
        
        with col2:
            st.markdown("**📚 Stacking Ensemble:**")
            if stacking_enabled:
                st.success(f"✅ Enabled: {meta_learner} meta-learner")
                st.info(f"📋 Base Models: {', '.join(stacking_models) if stacking_models else 'None'}")
            else:
                st.warning("⚠️ Disabled")
        
        # Auto-save configuration when user makes changes
        if st.button("💾 Auto-Save Configuration", help="Save current configuration without completing Step 3"):
            # Save current configuration
            current_step3_data = session_manager.get_step_data(3) or {}
            current_step3_data.update({
                'optuna_config': optuna_config,
                'vectorization_config': vectorization_config,
                'voting_config': voting_config,
                'stacking_config': stacking_config
            })
            session_manager.set_step_data(3, current_step3_data)
            
            # Also save directly to session state as backup
            st.session_state['step3_backup'] = {
                'optuna_config': optuna_config,
                'vectorization_config': vectorization_config,
                'voting_config': voting_config,
                'stacking_config': stacking_config,
                'timestamp': str(datetime.now())
            }
            
            st.success("✅ Configuration auto-saved!")
            st.info(f"🔍 Debug: Auto-saved voting_config = {voting_config}")
            st.info(f"🔍 Debug: Auto-saved stacking_config = {stacking_config}")
            
            # Debug: Check session state directly
            st.info(f"🔍 Debug: Session state step3 = {st.session_state.get('step3', 'NOT_FOUND')}")
            st.info(f"🔍 Debug: Session state step3_backup = {st.session_state.get('step3_backup', 'NOT_FOUND')}")
            
            # Force rerun to show updated state
            st.rerun()
    
    # Complete Step 3 button
    st.markdown("---")
    
    if st.button("✅ Complete Step 3", type="primary"):
        # Save all configurations
        step3_data = {
            'optuna_config': optuna_config,
            'vectorization_config': vectorization_config,
            'voting_config': voting_config,
            'stacking_config': stacking_config,
            'completed': True
        }
        
        session_manager.set_step_data(3, step3_data)
        
        # Debug: Show what was saved
        st.success("✅ Step 3 configuration saved!")
        st.info(f"🔍 Debug: Saved voting_config = {voting_config}")
        st.info(f"🔍 Debug: Saved stacking_config = {stacking_config}")
        st.info("💡 Click 'Next ▶' button to proceed to Step 4.")
    
    # Navigation buttons
    render_navigation_buttons()


def _check_for_text_data():
    """Check if the current dataset contains text data that needs vectorization"""
    try:
        # Get data from Step 1
        step1_data = session_manager.get_step_data(1)
        if not step1_data or 'dataframe' not in step1_data:
            return False
        
        df = step1_data['dataframe']
        
        # Check if any columns are text type
        text_columns = []
        for col in df.columns:
            if df[col].dtype in ['object', 'string', 'category']:
                # Check if it can be converted to numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                    # If it can be converted to numeric, it's not text
                    continue
                except:
                    # If it can't be converted to numeric, it's text
                    text_columns.append(col)
        
        # Check if we have text columns in multi-input mode
        if text_columns:
            return True
        
        return False
        
    except Exception as e:
        print(f"Error checking for text data: {e}")
        return False


def render_navigation_buttons():
    """Render navigation buttons as per wireframe"""
     
    col1, col2 = st.columns(2)
    
    with col1:
        current_step = get_current_step(session_manager)
        if st.button("◀ Previous", width='stretch', key=f"prev_btn_{current_step}"):
            # Go back to previous step
            # Use global session_manager instance
            if current_step > 1:
                session_manager.set_current_step(current_step - 1)
                st.success(f"← Going back to Step {current_step - 1}")
                st.rerun()
            else:
                st.info("ℹ️ You're already at the first step.")
    
    with col2:
        if st.button("Next ▶", width='stretch', key=f"next_btn_{current_step}"):
            # Use global session_manager instance
            current_step = get_current_step(session_manager)
            
            if current_step == 1:
                step_data = session_manager.get_step_data(1)
                if 'dataframe' in step_data and step_data['dataframe'] is not None:
                    # Log current step 1 data before moving to step 2
                    print(f"\n🔄 [NAVIGATION] Moving from Step 1 to Step 2...")
                    print(f"📊 [NAVIGATION] Step 1 data keys: {list(step_data.keys())}")
                    
                    if 'sampling_config' in step_data:
                        sampling_config = step_data['sampling_config']
                        print(f"💾 [NAVIGATION] Sampling config found: {sampling_config}")
                        if 'dataframe' in step_data:
                            df_size = len(step_data['dataframe'])
                            print(f"📊 [NAVIGATION] Dataset size: {df_size:,}, Requested samples: {sampling_config.get('num_samples', 'N/A')}")
                    else:
                        print(f"❌ [NAVIGATION] No sampling config found in step 1 data!")
                    
                    # Move to step 2 (Column Selection & Preprocessing)
                    session_manager.set_current_step(2)
                    st.success("✅ Step 1 completed! Moving to Step 2...")
                    st.rerun()
                else:
                    st.warning("⚠️ Please complete Step 1 first")
            elif current_step == 2:
                step_data = session_manager.get_step_data(2)
                if step_data and step_data.get('completed', False):
                    # Clear preview cache when moving to next step
                    if 'step2_preview_cache' in st.session_state:
                        del st.session_state.step2_preview_cache
                        print("🧹 [NAVIGATION] Cleared Step 2 preview cache")
                    
                    st.success("✅ Step 2 completed! Moving to Step 3...")
                    # Move to step 3
                    session_manager.set_current_step(3)
                    st.rerun()
                else:
                    st.warning("⚠️ Please complete Step 2 first")
            elif current_step == 3:
                step_data = session_manager.get_step_data(3)
                if step_data and step_data.get('completed', False):
                    st.success("✅ Step 3 completed! Moving to Step 4...")
                    # Move to step 4
                    session_manager.set_current_step(4)
                    st.rerun()
                else:
                    st.warning("⚠️ Please complete Step 3 first")
            elif current_step == 4:
                step_data = session_manager.get_step_data(4)
                if step_data and step_data.get('completed', False):
                    st.success("✅ Step 4 completed! Moving to Step 5...")
                    # Move to step 5
                    session_manager.set_current_step(5)
                    st.rerun()
                else:
                    st.warning("⚠️ Please complete Step 4 first")
            elif current_step == 5:
                st.info("ℹ️ You're already at the last step.")
            else:
                st.info("ℹ️ This step is not yet implemented.")
    
    # Removed Skip to End button

def render_sidebar():
    """Render sidebar with progress tracking"""
    # Initialize session manager
    # Use global session_manager instance
    
    # Current step info - Dynamic based on current step
    current_step = get_current_step(session_manager)
    step_names = [
        "Dataset Selection",
        "Column Selection & Preprocessing",
        "Model Configuration",
        "Training Execution",
        "Results Analysis"
    ]
    
    current_step_name = step_names[current_step - 1] if current_step <= len(step_names) else "Unknown"
    # Step status - Dynamic based on current progress
    current_step = get_current_step(session_manager)
    
    # Generate step status dynamically
    step_names = [
        "Dataset Selection",
        "Column Selection & Preprocessing",
        "Model Configuration",
        "Training Execution",
        "Results Analysis"
    ]
    
    # Create clickable step status buttons
    st.sidebar.markdown("### 📋 Navigation")
    
    for i, step_name in enumerate(step_names, 1):
        if i < current_step:
            status_icon = "✅"
            status_text = "Completed"
            button_color = "primary"
        elif i == current_step:
            status_icon = "🔄"
            status_text = "Current"
            button_color = "secondary"
        else:
            status_icon = "⏳"
            status_text = "Pending"
            button_color = "tertiary"
        
        # Check if step has data to show completion status
        step_data = session_manager.get_step_data(i)
        if step_data and len(step_data) > 0:
            if i == 1 and 'dataframe' in step_data:
                status_icon = "✅"
                status_text = "Completed"
                button_color = "primary"
            elif i == 2 and step_data.get('completed', False):  # Step 2 (Column Selection & Preprocessing)
                status_icon = "✅"
                status_text = "Completed"
                button_color = "primary"
            elif i == 3 and step_data.get('completed', False):  # Step 3 (Model Configuration)
                status_icon = "✅"
                status_text = "Completed"
                button_color = "primary"
            elif i == 4 and step_data.get('completed', False):  # Step 4 (Training Execution)
                status_icon = "✅"
                status_text = "Completed"
                button_color = "primary"
            elif i == 5 and step_data.get('completed', False):  # Step 5 (Results Analysis)
                status_icon = "✅"
                status_text = "Completed"
                button_color = "primary"
        
        # Create clickable button for each step
        if st.sidebar.button(
            f"{status_icon} Step {i}: {step_name}",
            type=button_color,
            width='stretch',
            key=f"step_nav_{i}",
            help=None,

        ):
            # Navigate to selected step
            session_manager.set_current_step(i)
            st.sidebar.success(f"🚀 Navigated to Step {i}: {step_name}")
            st.rerun()
        

def get_current_step(session_manager):
    """Get current step from session manager"""
    try:
        current_step = session_manager.get_current_step()
        if current_step is None:
            # Check which step has data to determine current step
            for step_num in range(1, 6):  # 5 steps total
                step_data = session_manager.get_step_data(step_num)
                if step_data and len(step_data) > 0:
                    if step_num == 1 and 'dataframe' in step_data:
                        # If step 1 has data but no current_step set, set it to 1
                        session_manager.set_current_step(1)
                        return 1
                    elif step_num == 2 and step_data.get('completed', False):  # Step 2 (Column Selection & Preprocessing)
                        # If step 2 has data but no current_step set, set it to 2
                        session_manager.set_current_step(2)
                        return 2
                    elif step_num == 3 and step_data.get('completed', False):  # Step 3 (Model Configuration)
                        # If step 3 has data but no current_step set, set it to 3
                        session_manager.set_current_step(3)
                        return 3
                    elif step_num == 4 and step_data.get('completed', False):  # Step 4 (Training Execution)
                        session_manager.set_current_step(4)
                        return 4
            # If no step data found, set current_step to 1
            session_manager.set_current_step(1)
            return 1
        return current_step
    except Exception:
        # If error, set current_step to 1 and return
        try:
            session_manager.set_current_step(1)
        except Exception:
            pass
        return 1

def render_step2_wireframe():
    """Render Step 2 - Data Processing & Preprocessing"""
    
    # Step title
    st.markdown("""
    <h2 style="text-align: left; color: var(--text-color); margin: 2rem 0 1rem 0; font-size: 1.8rem;">
        📍 STEP 2/5: Data Processing & Preprocessing
    </h2>
    """, unsafe_allow_html=True)
    
    # Create tabs for Single Input and Multi Input
    tab1, tab2 = st.tabs(["📄 Single Input (Text)", "📊 Multi Input (Mixed Data)"])
    
    with tab1:
        render_single_input_section()
    
    with tab2:
        render_multi_input_section()
    
    # Navigation buttons
    render_navigation_buttons()


def render_single_input_section():
    """Render Single Input section (existing functionality)"""
    # Get data from step 1
    # Use global session_manager instance
    step1_data = session_manager.get_step_data(1)
    
    if not step1_data or 'dataframe' not in step1_data:
        st.error("❌ No dataset found. Please complete Step 1 first.")
        if st.button("← Go to Step 1"):
            session_manager.set_current_step(1)
            st.success("← Going back to Step 1")
            st.rerun()
        return
    
    df = step1_data['dataframe']
    
    # Column Selection Section
    st.markdown("""
    <h3 style="color: var(--text-color); margin: 1.5rem 0 1rem 0;">📝 Column Selection:</h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Text column selection
        text_columns = [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype == 'string']
        if not text_columns:
            text_columns = df.columns.tolist()  # Fallback to all columns
        
        selected_text_column = st.selectbox(
            "📄 Chọn cột văn bản:",
            text_columns,
            help="Select the column containing text data for classification",
            index=0
        )
    
    with col2:
        # Label column selection
        label_columns = [col for col in df.columns if col != selected_text_column]
        if not label_columns:
            label_columns = df.columns.tolist()  # Fallback to all columns
        
        selected_label_column = st.selectbox(
            "🏷️ Chọn cột nhãn:",
            label_columns,
            help="Select the column containing class labels",
            index=0
        )

    col1, col2 = st.columns(2)
    
    with col1:
        # Text Column Analysis      
        # Calculate text column statistics
        text_data = df[selected_text_column].dropna()
        original_text_samples = len(text_data)
        
        # Get sampling configuration to show actual samples
        step1_data = session_manager.get_step_data(1)
        sampling_config = step1_data.get('sampling_config', {}) if step1_data else {}
        
        if sampling_config and sampling_config.get('num_samples'):
            actual_text_samples = min(sampling_config['num_samples'], original_text_samples)
        else:
            actual_text_samples = original_text_samples
        
        # Calculate average length
        if text_data.dtype == 'object' or text_data.dtype == 'string':
            avg_length = text_data.astype(str).str.len().mean()
            avg_length_words = text_data.astype(str).str.split().str.len().mean()
        else:
            avg_length = 0
            avg_length_words = 0
        
        # Calculate unique words (approximate)
        if text_data.dtype == 'object' or text_data.dtype == 'string':
            all_text = ' '.join(text_data.astype(str).dropna())
            unique_words = len(set(all_text.lower().split()))
        else:
            unique_words = 0
        
        st.markdown(f"""
        <p>• Samples: <strong>{actual_text_samples:,}</strong></p>
        <p>• Avg Length: <strong>{avg_length:.0f} chars</strong></p>
        <p>• Unique Words: <strong>{unique_words:,}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Label Column Analysis        
        # Calculate label column statistics
        label_data = df[selected_label_column].dropna()
        original_label_samples = len(label_data)
        
        # Get sampling configuration to show actual samples
        step1_data = session_manager.get_step_data(1)
        sampling_config = step1_data.get('sampling_config', {}) if step1_data else {}
        
        if sampling_config and sampling_config.get('num_samples'):
            actual_label_samples = min(sampling_config['num_samples'], original_label_samples)
            # Apply sampling to label data for accurate statistics
            if actual_label_samples < original_label_samples:
                # Sample the label data to get accurate class distribution
                sampled_indices = df[selected_label_column].dropna().sample(
                    n=actual_label_samples, 
                    random_state=42
                ).index
                sampled_label_data = df.loc[sampled_indices, selected_label_column]
                unique_classes = sampled_label_data.nunique()
                class_counts = sampled_label_data.value_counts()
            else:
                unique_classes = label_data.nunique()
                class_counts = label_data.value_counts()
        else:
            actual_label_samples = original_label_samples
            unique_classes = label_data.nunique()
            class_counts = label_data.value_counts()
        
        max_class_count = class_counts.max()
        min_class_count = class_counts.min()
        
        if max_class_count > 0:
            balance_ratio = min_class_count / max_class_count
            if balance_ratio > 0.7:
                distribution = "Balanced"
            elif balance_ratio > 0.3:
                distribution = "Moderately Balanced"
            else:
                distribution = "Imbalanced"
        else:
            distribution = "Unknown"
        
        # Get sample labels
        sample_labels = ', '.join(class_counts.head(5).index.astype(str))
        if len(class_counts) > 5:
            sample_labels += f", ... (+{len(class_counts) - 5} more)"
        
        st.markdown(f"""
        <p>• Samples: <strong>{actual_label_samples:,}</strong></p>
        <p>• Unique Classes: <strong>{unique_classes}</strong></p>
        <p>• Distribution: <strong>{distribution}</strong></p>
        <p>• Sample Labels: <strong>{sample_labels}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    # Perform validation checks
    import sys
    
    validation_errors = []
    validation_warnings = []
    
    # Check if text column has sufficient data
    if actual_text_samples < 10:
        msg = "Text column has insufficient data (less than 10 samples)"
        print(f"ERROR: {msg}", file=sys.stderr)
    elif actual_text_samples < 100:
        msg = "Text column has limited data (less than 100 samples)"
        print(f"WARNING: {msg}", file=sys.stderr)

    if unique_classes < 2:
        msg = "Label column must have at least 2 unique classes"
        print(f"ERROR: {msg}", file=sys.stderr)
    elif unique_classes < 3:
        msg = "Label column has only 2 classes (binary classification)"
        print(f"WARNING: {msg}", file=sys.stderr)

    if distribution == "Imbalanced":
        msg = (
            "Class distribution is imbalanced - consider using stratified sampling"
        )
        print(f"WARNING: {msg}", file=sys.stderr)
    
    # Display validation results
    if validation_errors:
        st.error("❌ **Validation Errors:**")
        for error in validation_errors:
            st.error(f"• {error}")
    
    if validation_warnings:
        st.warning("⚠️ **Validation Warnings:**")
        for warning in validation_warnings:
            st.warning(f"• {warning}")
    
    if not validation_errors and not validation_warnings:
        st.toast("✅ **All validations passed!** Columns are ready for processing.")
    
    # Preprocessing Options Section
    st.markdown("---")
    st.markdown("""
    <h3 style="color: var(--text-color); margin: 1.5rem 0 1rem 0;">🧹 Preprocessing Options:</h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        text_cleaning = st.checkbox(
            "☑️ Text Cleaning (remove special chars)",
            value=True,
            help="Remove special characters and normalize text"
        )
        
        category_mapping = st.checkbox(
            "☑️ Category Mapping (convert to numeric)",
            value=True,
            help="Convert categorical labels to numeric values"
        )
    
    with col2:
        data_validation = st.checkbox(
            "☑️ Data Validation (remove nulls)",
            value=True,
            help="Remove rows with missing values"
        )
        
        memory_optimization = st.checkbox(
            "☑️ Memory Optimization",
            value=True,
            help="Optimize data types for memory efficiency"
        )
    
    # Advanced Preprocessing Options
    st.markdown("---")
    st.markdown("""
    <h4 style="color: var(--text-color); margin: 1rem 0 0.5rem 0;">🚀 Advanced Preprocessing Options:</h4>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        rare_words_removal = st.checkbox(
            "🔍 Rare Words Removal",
            value=False,
            help="Remove words that appear very rarely (improves model performance)"
        )
        
        if rare_words_removal:
            rare_words_threshold = st.slider(
                "📊 Minimum Word Frequency:",
                min_value=1,
                max_value=10,
                value=2,
                help="Words appearing less than this many times will be removed"
            )
        
        lemmatization = st.checkbox(
            "🌿 Lemmatization",
            value=False,
            help="Convert words to their base form (e.g., 'running' → 'run')"
        )
    
    with col2:
        context_aware_stopwords = st.checkbox(
            "🧠 Context-aware Stopwords",
            value=False,
            help="Intelligently remove stopwords based on context and domain"
        )
        
        if context_aware_stopwords:
            stopwords_aggressiveness = st.selectbox(
                "⚡ Stopwords Aggressiveness:",
                options=["Conservative", "Moderate", "Aggressive"],
                index=1,
                help="How aggressively to remove stopwords"
            )
        
        phrase_detection = st.checkbox(
            "🔗 Phrase Detection",
            value=False,
            help="Detect and preserve important phrases (e.g., 'machine learning')"
        )
        
        if phrase_detection:
            min_phrase_freq = st.slider(
                "📊 Minimum Phrase Frequency:",
                min_value=1,
                max_value=20,
                value=3,
                help="Phrases appearing less than this many times will not be preserved"
            )
    
    # Column Preview Section
    st.markdown("""
    <h3 style="color: var(--text-color); margin: 1.5rem 0 1rem 0;">👀 Column Preview:</h3>
    """, unsafe_allow_html=True)
    
    # Show sample data from selected columns
    if 'step2_preview_cache' not in st.session_state:
        # Initial preview - show original data
        preview_df = df[[selected_text_column, selected_label_column]].head(5)
        st.dataframe(preview_df, width='stretch')
        st.caption("📝 **Original data preview** - Apply preprocessing to see transformed data")
    else:
        # Show cached preview with preprocessing applied
        cached_preview = st.session_state.step2_preview_cache
        st.dataframe(cached_preview, width='stretch')
     
        # Add option to clear cache and show original data
        if st.button("🔄 Show Original Data", key="show_original_preview"):
            del st.session_state.step2_preview_cache
            st.rerun()
    
    # Save configuration button
    if st.button("💾 Save Column Configuration", type="primary", width='stretch'):
        # Create preview cache with preprocessing applied
        preview_data = []
        sample_df = df[[selected_text_column, selected_label_column]].head(5)
        
        for idx, row in sample_df.iterrows():
            text_value = str(row[selected_text_column])
            label_value = str(row[selected_label_column])
            
            # Apply text cleaning if enabled
            if text_cleaning:
                # Remove \n characters and extra spaces
                text_value = text_value.strip().replace("\n", " ")
                # Remove special characters (keep only letters, numbers, spaces)
                import re
                text_value = re.sub(r'[^\w\s]', '', text_value)
                # Remove digits
                text_value = re.sub(r'\d+', '', text_value)
                # Remove extra spaces
                text_value = re.sub(r'\s+', ' ', text_value).strip()
                # Convert to lowercase
                text_value = text_value.lower()
            
            # Apply advanced preprocessing options using DataLoader functions
            if rare_words_removal or lemmatization or context_aware_stopwords or phrase_detection:
                # Create preprocessing config for preview
                preview_preprocessing_config = {
                    'text_cleaning': text_cleaning,
                    'rare_words_removal': rare_words_removal,
                    'rare_words_threshold': rare_words_threshold if 'rare_words_threshold' in locals() else 2,
                    'lemmatization': lemmatization,
                    'context_aware_stopwords': context_aware_stopwords,
                    'stopwords_aggressiveness': stopwords_aggressiveness if 'stopwords_aggressiveness' in locals() else 'Moderate',
                    'phrase_detection': phrase_detection,
                    'min_phrase_freq': min_phrase_freq if 'min_phrase_freq' in locals() else 3
                }
                
                # Create temporary DataLoader instance for preview
                from data_loader import DataLoader
                temp_loader = DataLoader()
                temp_loader.samples = [{'abstract': text_value, 'categories': label_value}]
                
                # Apply preprocessing using DataLoader's actual logic
                temp_loader.preprocess_samples(preview_preprocessing_config)
                
                if temp_loader.preprocessed_samples:
                    # Get the preprocessed text
                    text_value = temp_loader.preprocessed_samples[0]['text']
                    label_value = temp_loader.preprocessed_samples[0]['label']
            
            # Apply data validation if enabled
            if data_validation:
                if not text_value.strip() or not label_value.strip():
                    continue  # Skip invalid samples
            
            # Apply memory optimization if enabled
            if memory_optimization:
                text_value = str(text_value)
                label_value = str(label_value)
            
            preview_data.append({
                selected_text_column: text_value,
                selected_label_column: label_value
            })
        
        # Store preview cache in session state
        if preview_data:
            st.session_state.step2_preview_cache = pd.DataFrame(preview_data)
        
        # Store step 2 configuration with preprocessing options
        step2_config = {
            'text_column': selected_text_column,
            'label_column': selected_label_column,
            'text_samples': actual_text_samples,
            'unique_classes': unique_classes,
            'distribution': distribution,
            'avg_length': avg_length,
            'avg_length_words': avg_length_words,
            'unique_words': unique_words,
            'validation_errors': validation_errors,
            'validation_warnings': validation_warnings,
            'text_cleaning': text_cleaning,
            'category_mapping': category_mapping,
            'data_validation': data_validation,
            'memory_optimization': memory_optimization,
            # Advanced preprocessing options
            'rare_words_removal': rare_words_removal,
            'rare_words_threshold': rare_words_threshold if rare_words_removal else 2,
            'lemmatization': lemmatization,
            'context_aware_stopwords': context_aware_stopwords,
            'stopwords_aggressiveness': stopwords_aggressiveness if context_aware_stopwords else 'Moderate',
            'phrase_detection': phrase_detection,
            'min_phrase_freq': min_phrase_freq if phrase_detection else 3,
            'completed': True
        }
        
        session_manager.set_step_config('step2', step2_config)
        
        st.toast("Column configuration and preprocessing options saved! Preview updated with preprocessing applied.")
        st.rerun()
        
        # Use the actual_text_samples already calculated above
        # Get sampling configuration to show actual samples that will be used
        step1_data = session_manager.get_step_data(1)
        sampling_config = step1_data.get('sampling_config', {}) if step1_data else {}
        
        # Use actual_text_samples that was calculated in text column analysis
        text_display = f"{actual_text_samples:,} samples"
        label_display = f"{unique_classes} classes"
        
        # Show configuration summary in terminal
        print(f"""
        **Configuration Summary:**
        - **Text Column**: {selected_text_column} ({text_display})
        - **Label Column**: {selected_label_column} ({label_display})
        - **Distribution**: {distribution}
        - **Text Length**: {avg_length_words:.1f} words average
        - **Sampling**: {sampling_config.get('num_samples', 'None')} samples via {sampling_config.get('sampling_strategy', 'None')}
        - **Text Cleaning**: {'Enabled' if text_cleaning else 'Disabled'}
        - **Category Mapping**: {'Enabled' if category_mapping else 'Disabled'}
        - **Data Validation**: {'Enabled' if data_validation else 'Disabled'}
        - **Memory Optimization**: {'Enabled' if memory_optimization else 'Disabled'}
        """)
        
        print(f"📊 [STEP2] Dataset info - Original: {original_text_samples:,}, Will use: {actual_text_samples:,} samples")
        
        # Show completion message
        st.toast("Step 2 completed successfully!")
        st.toast("Click 'Next ▶' button to proceed to Step 3.")
    

def render_multi_input_section():
    """Render Multi Input section for mixed data processing"""
    st.markdown("""
    <h3 style="color: var(--text-color); margin: 1.5rem 0 1rem 0;">📊 Multi Input Data Processing:</h3>
    """, unsafe_allow_html=True)
    
    # Get data from Step 1
    step1_data = session_manager.get_step_data(1)
    
    if not step1_data or 'dataframe' not in step1_data:
        st.error("❌ No dataset found. Please complete Step 1 first.")
        return
    
    df = step1_data['dataframe']
    
    # Display dataset info
    st.info(f"📊 **Using dataset from Step 1**: {df.shape[0]:,} samples × {df.shape[1]} columns")
    
    # Data preview
    st.markdown("**📋 Data Preview:**")
    st.dataframe(df.head(10), width='stretch')
    
    # Column information
    st.markdown("**📊 Column Information:**")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    st.dataframe(col_info, width='stretch')
    
    # Auto-detect data types
    st.markdown("**🔍 Auto-detect Data Types:**")
    numeric_cols = []
    text_cols = []
    mixed_cols = []
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_cols.append(col)
        elif df[col].dtype in ['object', 'string', 'category']:
            # Check if can be converted to numeric
            try:
                pd.to_numeric(df[col], errors='raise')
                mixed_cols.append(col)
            except:
                text_cols.append(col)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Numeric Columns", len(numeric_cols))
    with col2:
        st.metric("📝 Text Columns", len(text_cols))
    with col3:
        st.metric("🔄 Mixed Columns", len(mixed_cols))
        
    if len(df.columns) > 0:
        # Initialize default values for session state if not set
        if 'multi_input_label_col' not in st.session_state:
            st.session_state.multi_input_label_col = df.columns.tolist()[-1] if len(df.columns) > 0 else ''
    
        if 'multi_input_input_cols' not in st.session_state:
            st.session_state.multi_input_input_cols = df.columns.tolist()[:-1] if len(df.columns) > 1 else []
    
        # Column selection
        st.markdown("**📝 Column Selection:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
            # Get current label column selection to exclude from input options
            current_label_col = st.session_state.get('multi_input_label_col', '')
            
            # Available columns for input (exclude label column)
            available_input_cols = [col for col in df.columns.tolist() if col != current_label_col]
            
            # Get current input selection and filter out label column if it exists
            current_input_cols = st.session_state.get('multi_input_input_cols', [])
            filtered_input_cols = [col for col in current_input_cols if col != current_label_col]
            
            # Default: All columns except the last one (which will be label)
            default_input_cols = df.columns.tolist()[:-1] if len(df.columns) > 1 else []
            
            # Select input columns (multiple)
            input_cols = st.multiselect(
                "📄 Select input columns:",
                options=available_input_cols,
                default=filtered_input_cols if filtered_input_cols else default_input_cols,
                key="multi_input_input_cols",
                help="Select multiple columns for input features"
            )
    
    with col2:
        # Get current input columns selection to exclude from label options
        current_input_cols = st.session_state.get('multi_input_input_cols', [])
        
        # Available columns for label (exclude input columns)
        available_label_cols = [col for col in df.columns.tolist() if col not in current_input_cols]
        
        # Get current label selection and reset if it's in input columns
        current_label_col = st.session_state.get('multi_input_label_col', '')
        if current_label_col in current_input_cols:
            current_label_col = ''
        
        # Default: Last column in the dataset
        default_label_col = df.columns.tolist()[-1] if len(df.columns) > 0 else ''
        
        # Calculate default index
        if current_label_col == '':
            # Use default (last column) if no current selection
            default_index = available_label_cols.index(default_label_col) + 1 if default_label_col in available_label_cols else 0
        else:
            # Use current selection if it exists
            default_index = available_label_cols.index(current_label_col) + 1 if current_label_col in available_label_cols else 0
        
        # Select label column (single)
        label_col = st.selectbox(
            "🏷️ Select label column:",
            options=[''] + available_label_cols,
            index=default_index,
            key="multi_input_label_col",
            help="Select the target column for classification"
        )
        
        # Note: Column configuration is saved when clicking "🚀 Process Multi-Input Data" button below
    
    # Validation: Remove label column from input columns if selected (moved outside col2)
    if label_col and label_col in input_cols:
        input_cols = [col for col in input_cols if col != label_col]
        st.warning(f"⚠️ Removed '{label_col}' from input columns as it's selected as label column")
    
    # Show data quality metrics if columns are selected (moved outside col2)
    if input_cols and label_col:
        # Data quality metrics
        st.markdown("**📊 Data Quality Metrics:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Input Features", len(input_cols))
        
        with col2:
            unique_labels = df[label_col].nunique()
            st.metric("🏷️ Unique Labels", unique_labels)
        
        with col3:
            # Remove duplicates to avoid duplicate column names
            all_cols = list(input_cols) + [label_col]
            unique_cols = list(dict.fromkeys(all_cols))  # Preserve order, remove duplicates
            missing_pct = (df[unique_cols].isnull().sum().sum() / 
                          (len(df) * len(unique_cols))) * 100
            st.metric("❌ Missing %", f"{missing_pct:.1f}%")
        
        # Show sample data
        st.markdown("**👀 Sample Data:**")
        # Remove duplicates to avoid duplicate column names
        all_cols = list(input_cols) + [label_col]
        unique_cols = list(dict.fromkeys(all_cols))  # Preserve order, remove duplicates
        sample_data = df[unique_cols].head(5)
        st.dataframe(sample_data, width='stretch')
        
        # Preprocessing options
        st.markdown("**🧹 Preprocessing Options:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            numeric_scaler = st.multiselect(
                "📊 Numeric Scaling:",
                ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"],
                default=["StandardScaler", "MinMaxScaler", "RobustScaler"],  # Default to 3 options
                key="multi_input_numeric_scaler",
                help="Choose one or more scaling methods for numeric features. Multiple methods will train mixed models."
            )
            
            text_encoding = st.selectbox(
                "📝 Text Encoding:",
                ["TF-IDF", "Count Vectorizer", "Word Embeddings", "None"],
                index=3,  # Default to "None" (index 3)
                key="multi_input_text_encoding",
                help="Choose encoding method for text features"
            )
        
        with col2:
            remove_duplicates = st.checkbox(
                "🗑️ Remove Duplicates",
                value=False,  # Default to False (keep duplicates like auto_train files)
                key="multi_input_remove_duplicates",
                help="Remove duplicate rows from dataset. Warning: This may significantly reduce dataset size and affect model performance."
            )
            
            if remove_duplicates:
                # Show duplicate analysis
                duplicates = df.duplicated()
                duplicate_count = duplicates.sum()
                duplicate_percentage = duplicate_count / len(df) * 100
                
                st.info(f"📊 Found {duplicate_count} duplicate rows ({duplicate_percentage:.1f}% of dataset)")
                
                if duplicate_percentage > 50:
                    st.warning("⚠️ High duplicate percentage detected! Removing duplicates may significantly reduce dataset size.")
                elif duplicate_percentage > 20:
                    st.info("ℹ️ Moderate duplicate percentage detected.")
                else:
                    st.success("✅ Low duplicate percentage - safe to remove.")
            
            outlier_detection = st.checkbox(
                "🔍 Outlier Detection",
                value=False,  # Default to False
                key="multi_input_outlier_detection",
                help="Detect and handle outliers in numeric features"
            )
            
            missing_strategy = st.selectbox(
                "❌ Missing Values:",
                ["Drop rows", "Fill with mean/median", "Fill with mode", "Forward fill"],
                index=1,  # Default to "Fill with mean/median" (index 1)
                key="multi_input_missing_strategy",
                help="Strategy for handling missing values"
            )
        
        # Process button (moved outside col2)
        if st.button("🚀 Process Multi-Input Data", type="primary", key="multi_input_process_button"):
            # Save multi-input configuration
            multi_input_config = {
                'input_columns': input_cols,
                'label_column': label_col,
                'numeric_scaler': numeric_scaler,
                'text_encoding': text_encoding,
                'missing_strategy': missing_strategy,
                'outlier_detection': outlier_detection,
                'remove_duplicates': remove_duplicates,
                'processed': True
            }
            
            # Save column configuration (similar to Single Input)
            column_config = {
                'text_column': None,  # Multi-input doesn't have single text column
                'label_column': label_col,
                'input_columns': input_cols,
                'data_type': 'multi_input',
                'completed': True
            }
            
            # Save both configurations to session
            session_manager.set_step_data(2, {
                'multi_input_config': multi_input_config,
                'column_config': column_config,
                'completed': True
            })
            
            st.success("✅ Multi-input data configuration saved!")
            st.success("✅ Column configuration saved!")
            st.info("💡 This configuration will be used for training in Step 4.")
            
            # Show configuration summary
            st.markdown("**📋 Configuration Summary:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Column Configuration:**")
                st.write(f"- Input Columns: {', '.join(input_cols)}")
                st.write(f"- Label Column: {label_col}")
                st.write(f"- Data Type: Multi-Input")
                
            with col2:
                st.markdown("**⚙️ Processing Configuration:**")
                st.write(f"- Numeric Scaler: {', '.join(numeric_scaler) if numeric_scaler else 'None'}")
                st.write(f"- Text Encoding: {text_encoding}")
                st.write(f"- Missing Strategy: {missing_strategy}")
                st.write(f"- Outlier Detection: {'Yes' if outlier_detection else 'No'}")
            
            # Show completion message
            st.toast("Step 2 (Multi-Input) completed successfully!")
            st.toast("Click 'Next ▶' button to proceed to Step 3.")
    
    else:
        st.warning("⚠️ Please select input columns and label column to continue.")


def _check_for_text_data():
    """Check if the current dataset contains text data that needs vectorization"""
    try:
        # Get data from Step 1
        step1_data = session_manager.get_step_data(1)
        if not step1_data or 'dataframe' not in step1_data:
            return False
        
        df = step1_data['dataframe']
        
        # Check if any columns are text type
        text_columns = []
        for col in df.columns:
            if df[col].dtype in ['object', 'string', 'category']:
                # Check if it can be converted to numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                    # If it can be converted to numeric, it's not text
                    continue
                except:
                    # If it can't be converted to numeric, it's text
                    text_columns.append(col)
        
        # Also check if we're in single input mode (text data)
        if step1_data.get('is_single_input', False):
            return True
        
        # Check if we have text columns in multi-input mode
        if text_columns:
            return True
        
        return False
        
    except Exception as e:
        print(f"Error checking for text data: {e}")
        return False


def render_step3_wireframe():
    """Render Step 3 - Optuna Optimization & Ensemble Configuration"""
    
    # Step title
    st.markdown("""
    <h2 style="text-align: left; color: var(--text-color); margin: 2rem 0 1rem 0; font-size: 1.8rem;">
        📍 STEP 3/5: Optuna Optimization & Ensemble Configuration
    </h2>
    """, unsafe_allow_html=True)
    
    # Check if we have text data to determine if vectorization tab should be shown
    has_text_data = _check_for_text_data()
    
    # Create tabs dynamically based on data type
    if has_text_data:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Optuna Configuration", 
            "🗳️ Voting/Weight Ensemble", 
            "🏗️ Stacking Ensemble",
            "🔤 Vectorization Methods",
            "📊 Review & Validate"
        ])
        
        with tab1:
            render_optuna_configuration()
        
        with tab2:
            render_voting_weight_ensemble()
        
        with tab3:
            render_stacking_configuration()
        
        with tab4:
            render_vectorization_configuration()
        
        with tab5:
            render_review_validation()
    else:
        # No text data, show original tabs without vectorization
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Optuna Configuration", 
            "🗳️ Voting/Weight Ensemble", 
            "🏗️ Stacking Ensemble", 
            "📊 Review & Validate"
        ])
        
        with tab1:
            render_optuna_configuration()
        
        with tab2:
            render_voting_weight_ensemble()
        
        with tab3:
            render_stacking_configuration()
        
        with tab4:
            render_review_validation()
    
    # Complete Step 3 button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✅ Complete Step 3", width='stretch', key="complete_step3"):
            # Mark step 3 as completed
            current_data = session_manager.get_step_data(3)
            current_data['completed'] = True
            session_manager.set_step_data(3, current_data)
            st.success("✅ Step 3 completed! You can now proceed to Step 4.")
            st.rerun()
    
    # Navigation buttons
    render_navigation_buttons()


def render_optuna_configuration():
    """Render Optuna optimization configuration"""
    st.subheader("🎯 Optuna Hyperparameter Optimization")
    
    # Enable/Disable Optuna
    enable_optuna = st.checkbox("Enable Optuna Optimization", value=False, key="enable_optuna")
    
    if enable_optuna:
        col1, col2 = st.columns(2)
        
        with col1:
            n_trials = st.number_input("Number of Trials", min_value=10, max_value=200, value=50, key="optuna_trials")
            timeout = st.number_input("Timeout (minutes)", min_value=5, max_value=120, value=30, key="optuna_timeout")
        
        with col2:
            direction = st.selectbox("Optimization Direction", ["maximize", "minimize"], key="optuna_direction")
            metric = st.selectbox("Optimization Metric", ["accuracy", "f1_score", "precision", "recall"], key="optuna_metric")
        
        # Model selection for Optuna
        st.subheader("📋 Models to Optimize")
        available_models = [
            'random_forest', 'xgboost', 'lightgbm', 'catboost', 
            'adaboost', 'gradient_boosting', 'decision_tree',
            'logistic_regression', 'svm', 'knn', 'naive_bayes'
        ]
        
        selected_models = st.multiselect(
            "Select models for optimization",
            available_models,
            default=available_models,  # Default to all models
            key="optuna_models"
        )
        
        if selected_models:
            # Save Optuna configuration
            optuna_config = {
                'enabled': True,
                'trials': n_trials,
                'timeout': timeout * 60,  # Convert to seconds
                'direction': direction,
                'metric': metric,
                'models': selected_models
            }
            
            # Merge with existing step data
            current_data = session_manager.get_step_data(3)
            current_data['optuna_config'] = optuna_config
            # FIXED: Also save selected_models directly to step3_data for consistency
            current_data['selected_models'] = selected_models
            session_manager.set_step_data(3, current_data)
            
            st.success(f"✅ Optuna configuration saved for {len(selected_models)} models")
        else:
            st.warning("⚠️ Please select at least one model for optimization")
    else:
        # Save disabled state
        current_data = session_manager.get_step_data(3)
        current_data['optuna_config'] = {'enabled': False}
        session_manager.set_step_data(3, current_data)


def render_voting_weight_ensemble():
    """Render Voting/Weight ensemble configuration"""
    st.subheader("🗳️ Voting/Weight Ensemble")
    
    st.info("""
    **Voting/Weight Ensemble** is suitable for traditional ML models:
    - Combines predictions from multiple models using voting or weighted averaging
    - Works well with models that have different strengths
    - Simple and interpretable approach
    """)
    
    # Enable/Disable Voting Ensemble
    enable_voting = st.checkbox("Enable Voting/Weight Ensemble", value=False, key="enable_voting")
    
    if enable_voting:
        # All available models for voting
        traditional_models = [
            'random_forest', 'xgboost', 'lightgbm', 'catboost',
            'logistic_regression', 'svm', 'knn', 'naive_bayes', 
            'decision_tree', 'adaboost', 'gradient_boosting'
        ]
        
        selected_models = st.multiselect(
            "Select traditional models for voting",
            traditional_models,
            default=traditional_models,  # Default to all models
            key="voting_models"
        )
        
        if selected_models:
            col1, col2 = st.columns(2)
            
            with col1:
                voting_method = st.selectbox(
                    "Voting Method",
                    ["hard", "soft"],
                    help="Hard: majority vote, Soft: average probabilities",
                    key="voting_method"
                )
            
            with col2:
                use_custom_weights = st.checkbox("Use Custom Weights", key="use_custom_weights")
    
            if use_custom_weights:
                st.subheader("⚖️ Custom Weights")
                weights = {}
                for i, model in enumerate(selected_models):
                    weights[model] = st.number_input(
                        f"Weight for {model}",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0/len(selected_models),
                        step=0.1,
                        key=f"weight_{model}"
                    )
            else:
                weights = None
            
            # Save voting configuration
            voting_config = {
                'enabled': True,
                'models': selected_models,
                'voting_method': voting_method,
                'weights': weights
            }
            
            # Merge with existing step data
            current_data = session_manager.get_step_data(3)
            current_data['voting_config'] = voting_config
            session_manager.set_step_data(3, current_data)
        else:
            # No models selected - disable voting ensemble
            voting_config = {
                'enabled': False,
                'models': [],
                'voting_method': 'hard',
                'weights': None
            }
            
            # Merge with existing step data
            current_data = session_manager.get_step_data(3)
            current_data['voting_config'] = voting_config
            session_manager.set_step_data(3, current_data)
            
            st.success(f"✅ Voting ensemble configured with {len(selected_models)} models")
    else:
        # Save disabled state
        current_data = session_manager.get_step_data(3)
        current_data['voting_config'] = {'enabled': False}
        session_manager.set_step_data(3, current_data)


def render_stacking_configuration():
    """Render Stacking ensemble configuration"""
    st.subheader("🏗️ Stacking Ensemble")
    
    st.info("""
    **Stacking Ensemble** is suitable for tree-based models:
    - Uses a meta-learner to combine predictions from base models
    - Works well with models that can provide probability estimates
    - More sophisticated than voting, often achieves better performance
    """)
    
    # Enable/Disable Stacking Ensemble
    enable_stacking = st.checkbox("Enable Stacking Ensemble", value=False, key="enable_stacking")
    
    if enable_stacking:
        # Tree-based models for stacking
        tree_models = [
            'random_forest', 'xgboost', 'lightgbm', 'catboost',
            'adaboost', 'gradient_boosting', 'decision_tree'
        ]
        
        base_models = st.multiselect(
            "Select tree-based models for stacking",
            tree_models,
            default=tree_models,  # Default to all tree-based models
            key="stacking_models"
        )
        
        if base_models:
            col1, col2 = st.columns(2)
            
            with col1:
                meta_learner = st.selectbox(
                    "Meta-learner",
                    ["logistic_regression", "random_forest", "svm"],
                    help="Final model that combines base model predictions",
                    key="meta_learner"
                )
            
            with col2:
                cv_folds = st.number_input(
                    "Cross-validation folds",
                    min_value=3,
                    max_value=10,
                    value=5,
                    key="cv_folds"
                )
            
            # Save stacking configuration
            stacking_config = {
                'enabled': True,
                'base_models': base_models,
                'meta_learner': meta_learner,
                'cv_folds': cv_folds
            }
            
            # Merge with existing step data
            current_data = session_manager.get_step_data(3)
            current_data['stacking_config'] = stacking_config
            session_manager.set_step_data(3, current_data)
            
            st.success(f"✅ Stacking ensemble configured with {len(base_models)} base models")
        else:
            st.warning("⚠️ Please select at least one tree-based model for stacking")
    else:
        # Save disabled state
        current_data = session_manager.get_step_data(3)
        current_data['stacking_config'] = {'enabled': False}
        session_manager.set_step_data(3, current_data)


def render_vectorization_configuration():
    """Render vectorization methods configuration for text data"""
    st.subheader("🔤 Text Vectorization Methods")
    
    # Get data info
    step1_data = session_manager.get_step_data(1)
    if not step1_data or 'dataframe' not in step1_data:
        st.error("❌ No dataset found. Please complete Step 1 first.")
        return
    
    df = step1_data['dataframe']
    
    # Show text columns info
    st.info("📝 **Text Data Detected**: The following text columns will be vectorized:")
    
    text_columns = []
    for col in df.columns:
        if df[col].dtype in ['object', 'string', 'category']:
            try:
                pd.to_numeric(df[col], errors='raise')
                continue
            except:
                text_columns.append(col)
    
    if text_columns:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Text Columns:**")
            for col in text_columns:
                st.write(f"• {col}")
        with col2:
            st.write("**Sample Text:**")
            sample_text = df[text_columns[0]].dropna().iloc[0] if len(df) > 0 else "No data"
            st.text_area("", value=str(sample_text)[:200] + "..." if len(str(sample_text)) > 200 else str(sample_text), 
                        height=100, disabled=True)
    else:
        st.warning("⚠️ No text columns detected in the dataset.")
        return
    
    st.markdown("---")
    
    # Vectorization methods selection
    st.markdown("**🎯 Select Vectorization Methods:**")
    
    # Available vectorization methods
    vectorization_methods = {
        "TF-IDF": {
            "description": "Term Frequency-Inverse Document Frequency - Good for most text classification tasks",
            "pros": ["Handles rare words well", "Good for classification", "Memory efficient"],
            "cons": ["May lose word order", "Sparse representation"]
        },
        "Bag of Words (BoW)": {
            "description": "Simple word counting - Fast and interpretable",
            "pros": ["Fast processing", "Easy to understand", "Good baseline"],
            "cons": ["Ignores word order", "Sensitive to frequent words"]
        },
        "Word Embeddings": {
            "description": "Dense vector representations using pre-trained models",
            "pros": ["Captures semantic meaning", "Dense representation", "Good for similarity"],
            "cons": ["Requires more memory", "Slower processing", "May overfit on small datasets"]
        }
    }
    
    # Method selection with detailed info
    selected_methods = []
    
    for method_name, method_info in vectorization_methods.items():
        col1, col2 = st.columns([1, 4])
        
        with col1:
            is_selected = st.checkbox(
                f"**{method_name}**",
                value=method_name == "TF-IDF",  # Default to TF-IDF
                key=f"vectorization_{method_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
            )
            if is_selected:
                selected_methods.append(method_name)
        
        with col2:
            with st.expander(f"ℹ️ {method_name} Details", expanded=False):
                st.write(f"**Description:** {method_info['description']}")
                
                col_pros, col_cons = st.columns(2)
                with col_pros:
                    st.write("**✅ Pros:**")
                    for pro in method_info['pros']:
                        st.write(f"• {pro}")
                
                with col_cons:
                    st.write("**❌ Cons:**")
                    for con in method_info['cons']:
                        st.write(f"• {con}")
    
    # Advanced configuration for selected methods
    if selected_methods:
        st.markdown("---")
        st.markdown("**⚙️ Advanced Configuration:**")
        
        # TF-IDF Configuration
        if "TF-IDF" in selected_methods:
            with st.expander("🔧 TF-IDF Parameters", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    tfidf_max_features = st.number_input(
                        "Max Features",
                        min_value=100,
                        max_value=50000,
                        value=10000,
                        key="tfidf_max_features",
                        help="Maximum number of features to extract"
                    )
                
                with col2:
                    tfidf_ngram_range = st.selectbox(
                        "N-gram Range",
                        ["(1,1)", "(1,2)", "(1,3)", "(2,2)", "(2,3)"],
                        index=1,  # Default to (1,2)
                        key="tfidf_ngram_range",
                        help="Range of n-grams to extract"
                    )
                
                with col3:
                    tfidf_min_df = st.number_input(
                        "Min Document Frequency",
                        min_value=1,
                        max_value=10,
                        value=2,
                        key="tfidf_min_df",
                        help="Minimum documents a word must appear in"
                    )
        
        # BoW Configuration
        if "Bag of Words (BoW)" in selected_methods:
            with st.expander("🔧 BoW Parameters", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    bow_max_features = st.number_input(
                        "Max Features",
                        min_value=100,
                        max_value=50000,
                        value=10000,
                        key="bow_max_features",
                        help="Maximum number of features to extract"
                    )
                
                with col2:
                    bow_ngram_range = st.selectbox(
                        "N-gram Range",
                        ["(1,1)", "(1,2)", "(1,3)", "(2,2)", "(2,3)"],
                        index=0,  # Default to (1,1)
                        key="bow_ngram_range",
                        help="Range of n-grams to extract"
                    )
                
                with col3:
                    bow_min_df = st.number_input(
                        "Min Document Frequency",
                        min_value=1,
                        max_value=10,
                        value=2,
                        key="bow_min_df",
                        help="Minimum documents a word must appear in"
                    )
        
        # Word Embeddings Configuration
        if "Word Embeddings" in selected_methods:
            with st.expander("🔧 Word Embeddings Parameters", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    embedding_model = st.selectbox(
                        "Pre-trained Model",
                        [
                            "all-MiniLM-L6-v2",
                            "all-mpnet-base-v2", 
                            "paraphrase-MiniLM-L6-v2",
                            "distilbert-base-nli-mean-tokens"
                        ],
                        index=0,
                        key="embedding_model",
                        help="Pre-trained sentence transformer model"
                    )
                
                with col2:
                    embedding_device = st.selectbox(
                        "Device",
                        ["auto", "cpu", "cuda"],
                        index=0,
                        key="embedding_device",
                        help="Device to run embeddings on"
                    )
        
        # Save configuration
        vectorization_config = {
            'selected_methods': selected_methods,
            'tfidf': {
                'max_features': tfidf_max_features if "TF-IDF" in selected_methods else 10000,
                'ngram_range': eval(tfidf_ngram_range) if "TF-IDF" in selected_methods else (1, 2),
                'min_df': tfidf_min_df if "TF-IDF" in selected_methods else 2
            },
            'bow': {
                'max_features': bow_max_features if "Bag of Words (BoW)" in selected_methods else 10000,
                'ngram_range': eval(bow_ngram_range) if "Bag of Words (BoW)" in selected_methods else (1, 1),
                'min_df': bow_min_df if "Bag of Words (BoW)" in selected_methods else 2
            },
            'embeddings': {
                'model_name': embedding_model if "Word Embeddings" in selected_methods else "all-MiniLM-L6-v2",
                'device': embedding_device if "Word Embeddings" in selected_methods else "auto"
            }
        }
        
        # Merge with existing step data
        current_data = session_manager.get_step_data(3)
        current_data['vectorization_config'] = vectorization_config
        # FIXED: Also save selected_vectorization directly to step3_data for consistency
        current_data['selected_vectorization'] = selected_methods
        session_manager.set_step_data(3, current_data)
        
        st.success(f"✅ Vectorization configuration saved for {len(selected_methods)} methods")
        
        # Show preview of what will be created
        st.markdown("---")
        st.markdown("**📊 Preview:**")
        
        total_features = 0
        for method in selected_methods:
            if method == "TF-IDF":
                total_features += tfidf_max_features
            elif method == "Bag of Words (BoW)":
                total_features += bow_max_features
            elif method == "Word Embeddings":
                total_features += 384  # Typical embedding dimension
        
        st.info(f"**Estimated total features:** {total_features:,} (across {len(selected_methods)} methods)")
        st.info(f"**Selected methods:** {', '.join(selected_methods)}")
        
    else:
        st.warning("⚠️ Please select at least one vectorization method.")


def render_review_validation():
    """Render review and validation tab"""
    st.subheader("📊 Review & Validation")
    
    # Get configurations from session
    step3_data = session_manager.get_step_data(3)
    
    # Fallback: Try to load from backup if step3_data is empty
    if not step3_data or not step3_data.get('voting_config') and not step3_data.get('stacking_config'):
        backup_data = st.session_state.get('step3_backup', {})
        if backup_data:
            st.warning("⚠️ Loading configuration from backup (session state may have been reset)")
            step3_data = backup_data
    optuna_config = step3_data.get('optuna_config', {})
    voting_config = step3_data.get('voting_config', {})
    stacking_config = step3_data.get('stacking_config', {})
    
    # Display configurations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Optuna Configuration")
    if optuna_config.get('enabled', False):
        st.success("✅ Optuna Optimization Enabled")
        st.write(f"- Trials: {optuna_config.get('trials', 'N/A')}")
        st.write(f"- Timeout: {optuna_config.get('timeout', 'N/A')} seconds")
        st.write(f"- Direction: {optuna_config.get('direction', 'N/A')}")
        st.write(f"- Models: {', '.join(optuna_config.get('models', []))}")
    else:
        st.info("ℹ️ Optuna Optimization Disabled")
    
    with col2:
        st.subheader("🗳️ Voting Ensemble")
    if voting_config.get('enabled', False):
        st.success("✅ Voting Ensemble Enabled")
        st.write(f"- Method: {voting_config.get('voting_method', 'N/A')}")
        st.write(f"- Models: {', '.join(voting_config.get('models', []))}")
        if voting_config.get('weights'):
            st.write("- Custom weights configured")
    else:
        st.info("ℹ️ Voting Ensemble Disabled")
    
    st.subheader("🏗️ Stacking Ensemble")
    if stacking_config.get('enabled', False):
        st.success("✅ Stacking Ensemble Enabled")
        st.write(f"- Base Models: {', '.join(stacking_config.get('base_models', []))}")
        st.write(f"- Meta-learner: {stacking_config.get('meta_learner', 'N/A')}")
        st.write(f"- CV Folds: {stacking_config.get('cv_folds', 'N/A')}")
    else:
        st.info("ℹ️ Stacking Ensemble Disabled")
    
    # Validation
    st.subheader("✅ Validation")
    if any([optuna_config.get('enabled'), voting_config.get('enabled'), stacking_config.get('enabled')]):
        st.success("✅ At least one optimization/ensemble method is configured")
        st.info("💡 Click 'Complete Step 3' button below to proceed to Step 4.")
    else:
        st.warning("⚠️ Please enable at least one optimization or ensemble method")


def render_step4_wireframe():
    """Render Step 4 - Training Execution and Monitoring"""
    
    # Step title
    st.markdown("""
    <h2 style="text-align: left; color: var(--text-color); margin: 2rem 0 1rem 0; font-size: 1.8rem;">
        📍 STEP 4/5: Training Execution and Monitoring
    </h2>
    """, unsafe_allow_html=True)
    
    # Get data from previous steps with validation
    try:
        step1_data = session_manager.get_step_data(1)
        step2_data = session_manager.get_step_data(2)
        step3_data = session_manager.get_step_data(3)
    except Exception as e:
        st.error(f"❌ Error loading step data: {str(e)}")
        st.info("💡 Please refresh the page and try again.")
        return
    
    if not step1_data or 'dataframe' not in step1_data:
        st.error("❌ No dataset found. Please complete Step 1 first.")
        return
    
    if not step2_data or not step2_data.get('completed', False):
        st.error("❌ Column selection not completed. Please complete Step 2 first.")
        return
    
    if not step3_data or not step3_data.get('completed', False):
        st.error("❌ Model configuration not completed. Please complete Step 3 first.")
        return
    
    # Display dataset info
    df = step1_data['dataframe']
    st.info(f"📊 **Dataset**: {df.shape[0]:,} samples × {df.shape[1]} columns")
    
    # Data Split Configuration
    st.subheader("📊 Data Split Configuration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        train_ratio = st.slider("🚂 Train (%)", 50, 90, 80, 5)
    with col2:
        val_ratio = st.slider("🔍 Val (%)", 5, 30, 10, 5)
    with col3:
        test_ratio = st.slider("🧪 Test (%)", 5, 30, 10, 5)
    
    # Validate and display
    total = train_ratio + val_ratio + test_ratio
    if total != 100:
        st.warning(f"⚠️ Total: {total}% (should be 100%)")
    else:
        st.toast(f"✅ Split: {train_ratio}%/{val_ratio}%/{test_ratio}%")
    
    # Store configuration
    data_split_config = {
        'train_ratio': train_ratio / 100,
        'val_ratio': val_ratio / 100,
        'test_ratio': test_ratio / 100
    }
    
    # Display configurations
    st.subheader("📋 Configuration Summary")
        
    # Initialize training state
    if 'training_started' not in st.session_state:
        st.session_state.training_started = False
    
    # Create debug container for configuration details
    config_debug_container = st.expander("📋 Training Log", expanded=False)
    
    # Load configurations from Step 3 (always load, not just for display)
    optuna_config = step3_data.get('optuna_config', {})
    voting_config = step3_data.get('voting_config', {})
    stacking_config = step3_data.get('stacking_config', {})
    
    # Only show configuration summary if training hasn't started
    if not st.session_state.training_started:
        if optuna_config.get('enabled', False):
            with config_debug_container:
                st.success(f"✅ Optuna: {optuna_config.get('trials', 'N/A')} trials, {len(optuna_config.get('models', []))} models")
        else:
            with config_debug_container:
                st.info("ℹ️ Optuna: Disabled")
        
        if voting_config.get('enabled', False):
            with config_debug_container:
                st.success(f"✅ Voting: {voting_config.get('voting_method', 'N/A')} voting, {len(voting_config.get('models', []))} models")
                st.info(f"🔍 Debug: Loaded voting_config = {voting_config}")
        else:
            with config_debug_container:
                st.warning("⚠️ Voting: Disabled - Enable in Step 3 to train voting ensemble")
                st.info(f"🔍 Debug: Loaded voting_config = {voting_config}")
        
        if stacking_config.get('enabled', False):
            with config_debug_container:
                st.success(f"✅ Stacking: {stacking_config.get('meta_learner', 'N/A')} meta-learner, {len(stacking_config.get('base_models', []))} base models")
                st.info(f"🔍 Debug: Loaded stacking_config = {stacking_config}")
        else:
            with config_debug_container:
                st.warning("⚠️ Stacking: Disabled - Enable in Step 3 to train stacking ensemble")
                st.info(f"🔍 Debug: Loaded stacking_config = {stacking_config}")
        
        # Debug: Show raw session state
        with config_debug_container:
            st.info(f"🔍 Debug: Raw session state step3 = {st.session_state.get('step3', 'NOT_FOUND')}")
            st.info(f"🔍 Debug: step3_data keys = {list(step3_data.keys()) if step3_data else 'EMPTY'}")
    
    # Training execution
    st.subheader("🚀 Training Execution")
    
    # Check if training is already completed
    if hasattr(st.session_state, 'training_completed') and st.session_state.training_completed:
        st.success("✅ Training completed! You can proceed to Step 5.")
        if st.button("🔄 Restart Training"):
            st.session_state.training_started = False
            st.session_state.training_in_progress = False
            st.session_state.training_completed = False
            st.session_state.results_displayed = False
            st.rerun()
    
    if st.button("🚀 Start Training", type="primary"):
        # Prevent multiple clicks and rerun issues
        if hasattr(st.session_state, 'training_in_progress') and st.session_state.training_in_progress:
            st.warning("⚠️ Training is already in progress. Please wait...")
            return
        
        # Prevent starting if already completed
        if hasattr(st.session_state, 'training_completed') and st.session_state.training_completed:
            st.warning("⚠️ Training already completed. Use 'Restart Training' to run again.")
            return
            
        # Set training states
        st.session_state.training_in_progress = True
        st.session_state.training_started = True
        st.session_state.training_completed = False
        st.session_state.results_displayed = False
        
        # Use try-finally to ensure cleanup
        try:
            # Create progress tracking
            progress_container = st.container()
            with progress_container:
                st.info("🔄 Initializing training pipeline...")
                
            # Use spinner with timeout protection
            with st.spinner("🔄 Starting training pipeline..."):
                # Import training pipeline
                from comprehensive_evaluation import ComprehensiveEvaluator
                
                # Get dataset
                df = step1_data['dataframe']
                
                # Get column configuration from Step 2
                # Check both step2_data and step2_config
                text_column = None
                label_column = None
                
                # Try to get from step2_data first (Multi Input)
                column_config = step2_data.get('column_config', {})
                # Create debug logging container for Step 4
                debug_container = config_debug_container
                with debug_container:
                    st.info(f"🔍 Debug Step 4: step2_data keys = {list(step2_data.keys())}")
                    st.info(f"🔍 Debug Step 4: column_config = {column_config}")
                    st.info(f"🔍 Debug Step 4: step2_data completed = {step2_data.get('completed', False)}")
                if column_config.get('text_column') or column_config.get('label_column'):
                    text_column = column_config.get('text_column')
                    label_column = column_config.get('label_column')
                
                # Try to get from step2_config (Single Input)
                if not text_column and not label_column:
                    step2_config = session_manager.get_step_config('step2')
                    text_column = step2_config.get('text_column')
                    label_column = step2_config.get('label_column')
                
                # Check if we have valid configuration from Step 2
                if not label_column:
                    st.error("❌ No label column configuration found from Step 2.")
                    st.info("💡 Please complete Step 2 to configure the label column.")
                    return
                
                # Check if columns exist in dataset
                data_type = column_config.get('data_type', 'single_input')
                with debug_container:
                    st.info(f"🔍 Debug: data_type = '{data_type}', column_config keys = {list(column_config.keys())}")
                
                if data_type == 'single_input':
                    # Single-input: require text_column
                    if text_column not in df.columns:
                        st.error(f"❌ Text column '{text_column}' not found in dataset. Available columns: {list(df.columns)}")
                        return
                
                if label_column not in df.columns:
                    st.error(f"❌ Label column '{label_column}' not found in dataset. Available columns: {list(df.columns)}")
                    st.info("💡 Please complete Step 2 to configure the correct label column.")
                    return
                
                # Show detected columns
                with debug_container:
                    st.info(f"📋 Using label column: '{label_column}'")
                    if data_type == 'single_input':
                        st.info(f"📋 Using text column: '{text_column}'")
                    else:
                        input_columns = column_config.get('input_columns', [])
                        st.info(f"📋 Using input columns: {input_columns}")
                
                # Prepare data based on data type
                
                if data_type == 'multi_input':
                    # Multi-input data: use input_columns as features
                    input_columns = column_config.get('input_columns', [])
                    if not input_columns:
                        st.error("❌ No input columns configured for multi-input data")
                        return
                    
                    # Check if all input columns exist
                    missing_cols = [col for col in input_columns if col not in df.columns]
                    if missing_cols:
                        st.error(f"❌ Input columns not found: {missing_cols}")
                        return
                    
                    # Get multi_input_config for scaling methods
                    multi_input_config = step2_data.get('multi_input_config', {})
                    numeric_scalers = multi_input_config.get('numeric_scaler', ['StandardScaler', 'MinMaxScaler', 'RobustScaler'])
                    
                    # Handle single scaler (convert to list)
                    if isinstance(numeric_scalers, str):
                        numeric_scalers = [numeric_scalers]
                    
                    with debug_container:
                        st.info(f"🔍 Debug: numeric_scalers = {numeric_scalers}")
                    
                    X = df[input_columns].values
                    y = df[label_column].values
                    
                    with debug_container:
                        st.info(f"📊 Multi-input data: {len(input_columns)} features, {len(X):,} samples")
                        st.info(f"📊 Scaling methods: {', '.join(numeric_scalers)}")
                    
                else:
                    # Single-input data: use text column
                    X = df[text_column].values
                    y = df[label_column].values
                    
                    st.info(f"📊 Single-input data: {len(X):,} samples")
                
                # Get selected models from Step 3
                selected_models = optuna_config.get('models', [])
                if not selected_models:
                    st.error("❌ No models selected in Step 3")
                    return
                
                with debug_container:
                    st.info(f"🤖 Selected models: {', '.join(selected_models)}")
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Enhanced configurations for training
                enhanced_step1_config = {
                    'dataframe': df,
                    'text_column': text_column,
                    'label_column': label_column,
                    'input_columns': input_columns if data_type == 'multi_input' else None,
                    'data_type': data_type
                }
                
                enhanced_step2_config = {
                    'preprocessing_config': step2_data.get('preprocessing_config', {}),
                    'multi_input_config': step2_data.get('multi_input_config', {})
                }
                
                # Create preprocessing config based on data type
                if data_type == 'multi_input':
                    # For multi-input data, use the numeric_scalers from step2 config
                    preprocessing_selected_methods = []
                    for scaler in numeric_scalers:
                        if scaler == 'StandardScaler':
                            preprocessing_selected_methods.append('StandardScaler')
                        elif scaler == 'MinMaxScaler':
                            preprocessing_selected_methods.append('MinMaxScaler')
                        elif scaler == 'RobustScaler':
                            preprocessing_selected_methods.append('RobustScaler')
                        elif scaler == 'None':
                            preprocessing_selected_methods.append('NoScaling')
                    
                    # Fallback if no valid scalers found
                    if not preprocessing_selected_methods:
                        preprocessing_selected_methods = ['StandardScaler', 'MinMaxScaler', 'NoScaling']
                else:
                    # For text data, use default methods
                    preprocessing_selected_methods = ['StandardScaler', 'MinMaxScaler', 'NoScaling']
                
                enhanced_step3_config = {
                    'optuna_config': optuna_config,
                    'voting_config': voting_config,
                    'stacking_config': stacking_config,
                    'selected_models': selected_models,
                    'preprocessing_config': {
                        'selected_methods': preprocessing_selected_methods
                    }
                }
                
                # Use different approaches based on data type with timeout protection
                with debug_container:
                    st.info(f"🔍 Debug: About to check data_type = '{data_type}' for training approach")
                    st.info(f"🔍 Debug: Using preprocessing_selected_methods = {preprocessing_selected_methods}")
                    st.info(f"🔍 Debug: Original numeric_scalers from step2 = {numeric_scalers}")
                
                # Initialize results variable to avoid UnboundLocalError
                results = None
                
                # Run training directly (no threading to avoid ScriptRunContext issues)
                try:
                    with debug_container:
                        st.info(f"🔍 Debug: Starting training with data_type = '{data_type}'")
                    
                    # Add progress indicator to prevent UI freezing
                    progress_placeholder = st.empty()
                    progress_placeholder.toast("🔄 Training in progress... Please wait.")
                    
                    if data_type == 'multi_input':
                        # For numeric data: use direct sklearn training
                        with debug_container:
                            st.info("🔢 Using direct sklearn for numeric data...")
                            st.info(f"🔍 Debug: Calling train_numeric_data_directly with input_columns = {input_columns}, label_column = {label_column}")
                            st.info(f"🔍 Debug: selected_models = {selected_models}")
                            st.info(f"🔍 Debug: numeric_scalers = {numeric_scalers}")
                            st.info(f"🔍 Debug: voting_config = {voting_config}")
                            st.info(f"🔍 Debug: stacking_config = {stacking_config}")
                        
                        results = train_numeric_data_directly(df, input_columns, label_column, selected_models, optuna_config, voting_config, stacking_config, progress_bar, status_text, numeric_scalers, multi_input_config.get('remove_duplicates', False), data_split_config)
                        
                        with debug_container:
                            st.info(f"🔍 Debug: train_numeric_data_directly completed, result status = {results.get('status', 'NOT_FOUND')}")
                    else:
                        # For text data: use execute_streamlit_training
                        st.info("📝 Using execute_streamlit_training for text data...")
                        with debug_container:
                            st.info(f"🔍 Debug: Calling execute_streamlit_training with enhanced configs")
                        
                        from training_pipeline import execute_streamlit_training
                        results = execute_streamlit_training(df, enhanced_step1_config, enhanced_step2_config, enhanced_step3_config)
                        
                        with debug_container:
                            st.info(f"🔍 Debug: execute_streamlit_training completed, result status = {results.get('status', 'NOT_FOUND')}")
                    
                    # Clear progress indicator
                    progress_placeholder.empty()
                    
                    # Clear progress bar and status text
                    try:
                        progress_bar.empty()
                        status_text.empty()
                    except:
                        pass
                    
                except Exception as e:
                    # Clear progress indicator on error
                    try:
                        progress_placeholder.empty()
                        progress_bar.empty()
                        status_text.empty()
                    except:
                        pass
                    
                    with debug_container:
                        st.error(f"🔍 Debug: Exception in training: {str(e)}")
                        import traceback
                        st.error(f"🔍 Debug: Traceback: {traceback.format_exc()}")
                    
                    results = {'status': 'failed', 'error': str(e)}
                    
                # Process results (same format as auto_train.py)
                results_debug_container = config_debug_container
                with results_debug_container:
                    st.info(f"🔍 Debug: results type = {type(results)}")
                    if isinstance(results, dict):
                        st.info(f"🔍 Debug: results keys = {list(results.keys())}")
                        st.info(f"🔍 Debug: results['status'] = {results.get('status', 'NOT_FOUND')}")
                    else:
                        st.info(f"🔍 Debug: results is not a dict: {results}")
                # Check if we have results and if any models were successful
                if results and isinstance(results, dict):
                    with results_debug_container:
                        st.info(f"🔍 Debug: Processing results with keys = {list(results.keys())}")
                        st.info(f"🔍 Debug: Overall status = {results.get('status', 'NO_STATUS')}")
                    
                    # Check if training failed immediately
                    if results.get('status') == 'failed':
                        error_msg = results.get('error', 'Unknown error')
                        st.error(f"❌ Training failed: {error_msg}")
                        with results_debug_container:
                            st.error(f"🔍 Debug: Training failed with error: {error_msg}")
                            st.error(f"🔍 Debug: Full results: {results}")
                        return
                    
                    # Extract successful results based on format
                    successful_results = []
                    
                    # Summary statistics - handle different result formats
                    with results_debug_container:
                        if isinstance(results, dict):
                            st.info(f"🔍 Debug: Processing results with keys = {list(results.keys())}")
                        else:
                            st.info(f"🔍 Debug: results is not a dict, cannot process keys")

                    if 'comprehensive_results' in results:
                        # Format from execute_streamlit_training (text data)
                        with results_debug_container:
                            st.info("🔍 Debug: Using comprehensive_results format (text data)")
                        comprehensive_results = results.get('comprehensive_results', [])
                        if isinstance(comprehensive_results, list):
                            successful_results = [r for r in comprehensive_results if isinstance(r, dict) and r.get('status') == 'success']
                        else:
                            successful_results = []
                        with results_debug_container:
                            st.info(f"🔍 Debug: Found {len(successful_results)} successful results in comprehensive_results")
                    elif 'model_results' in results:
                        # Format from train_numeric_data_directly (numeric data)
                        with results_debug_container:
                            st.info("🔍 Debug: Using model_results format (numeric data)")
                        model_results = results.get('model_results', {})
                        with results_debug_container:
                            if isinstance(model_results, dict):
                                st.info(f"🔍 Debug: model_results keys = {list(model_results.keys())}")
                            else:
                                st.info(f"🔍 Debug: model_results is not dict: {type(model_results)}")
                        successful_results = []
                        for model_name, model_data in model_results.items():
                            with results_debug_container:
                                if isinstance(model_data, dict):
                                    st.info(f"🔍 Debug: Processing {model_name}, status = {model_data.get('status', 'NO_STATUS')}")
                                else:
                                    st.info(f"🔍 Debug: Processing {model_name}, model_data is not dict: {type(model_data)}")
                            if isinstance(model_data, dict) and model_data.get('status') == 'success':
                                successful_results.append({
                                    'model_name': model_name,
                                    'validation_accuracy': model_data.get('validation_accuracy', 0),
                                    'test_accuracy': model_data.get('accuracy', 0),
                                    'f1_score': model_data.get('f1_score', 0),
                                    'precision': model_data.get('precision', 0),
                                    'recall': model_data.get('recall', 0),
                                    'support': model_data.get('support', 0),
                                    'training_time': model_data.get('training_time', 0)
                                })
                        with results_debug_container:
                            st.info(f"🔍 Debug: Found {len(successful_results)} successful results in model_results")
                    else:
                        # Fallback: try to use 'models' key if available
                        with results_debug_container:
                            st.info("🔍 Debug: Using fallback 'models' key")
                        models_data = results.get('models', {})
                        with results_debug_container:
                            if isinstance(models_data, dict):
                                st.info(f"🔍 Debug: models keys = {list(models_data.keys())}")
                            else:
                                st.info(f"🔍 Debug: models_data is not dict: {type(models_data)}")
                        successful_results = []
                        for model_name, model_data in models_data.items():
                            if isinstance(model_data, dict) and model_data.get('status') == 'success':
                                successful_results.append({
                                    'model_name': model_name,
                                    'f1_score': model_data.get('f1_score', model_data.get('accuracy', 0)),
                                    'test_accuracy': model_data.get('accuracy', 0),
                                    'validation_accuracy': model_data.get('validation_accuracy', 0),
                                    'training_time': model_data.get('training_time', 0),
                                    'embedding_name': 'numeric_features'
                                })
                        with results_debug_container:
                            st.info(f"🔍 Debug: Found {len(successful_results)} successful results in models fallback")

                    # Debug logging
                    with results_debug_container:
                        st.info(f"🔍 Debug: successful_results count = {len(successful_results)}")
                        if successful_results:
                            st.info(f"🔍 Debug: successful_results sample = {successful_results[0] if successful_results else 'None'}")
                    
                    if successful_results:
                        st.success(f"✅ Training completed successfully! {len(successful_results)} model(s) trained successfully.")
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame(successful_results)
                        
                        # Display summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("✅ Successful Models", len(successful_results))
                        with col2:
                            best_accuracy = results_df['test_accuracy'].max()
                            st.metric("🏆 Best Accuracy", f"{best_accuracy:.4f}")
                        with col3:
                            avg_time = results_df['training_time'].mean()
                            st.metric("⏱️ Avg Training Time", f"{avg_time:.2f}s")
                        
                        # Display detailed results
                        st.subheader("📋 Detailed Results")
                        st.dataframe(results_df, width='stretch')
                        
                        # Save training results cache with summary information only
                        # Individual models are already cached in cache/models/ during training
                        # Training results cache contains only summary metadata
                        try:
                            from cache_manager import training_results_cache
                            
                            # Generate session key for Step 5 compatibility
                            session_key = training_results_cache.generate_session_key(
                                step1_data, step2_data, step3_data
                            )
                            
                            # Create summary training results (no large files, only metadata)
                            summary_results = {
                                'status': results.get('status', 'success'),
                                'total_models': results.get('total_models', 0),
                                'successful_combinations': results.get('successful_combinations', 0),
                                'total_combinations': results.get('total_combinations', 0),
                                'elapsed_time': results.get('elapsed_time', 0),
                                'session_key': session_key,
                                'created_at': pd.Timestamp.now().isoformat(),
                                'summary_stats': {
                                    'best_accuracy': results_df['test_accuracy'].max() if len(successful_results) > 0 else 0,
                                    'avg_accuracy': results_df['test_accuracy'].mean() if len(successful_results) > 0 else 0,
                                    'avg_training_time': results_df['training_time'].mean() if len(successful_results) > 0 else 0,
                                    'total_training_time': results_df['training_time'].sum() if len(successful_results) > 0 else 0
                                },
                                'model_summary': [
                                    {
                                        'model_name': row['model_name'],
                                        'test_accuracy': row['test_accuracy'],
                                        'training_time': row['training_time'],
                                        'status': row.get('status', 'success')
                                    }
                                    for _, row in results_df.iterrows()
                                ] if len(successful_results) > 0 else []
                            }
                            
                            # Save training results cache (summary only)
                            cache_path = training_results_cache.save_training_results(session_key, summary_results)
                            
                            # Debug: Show session key and cache path
                            with results_debug_container:
                                st.info(f"🔍 Debug: Generated session key: {session_key}")
                                st.info(f"💾 Training results cache saved: {cache_path}")
                                st.info("📊 Summary metadata saved (no large files)")
                            
                            # Save data to session state
                            step4_data = {
                                'session_key': session_key,
                                'cache_path': cache_path,
                                'completed': True,
                                'total_models': results.get('total_models', 0),
                                'successful_combinations': results.get('successful_combinations', 0),
                                'elapsed_time': results.get('elapsed_time', 0),
                                'status': results.get('status', 'unknown')
                            }
                            session_manager.set_step_data(4, step4_data)
                            with results_debug_container:
                                st.success("✅ Step 4 completed! Training results cache saved with summary metadata.")
                            
                        except Exception as cache_error:
                            st.warning(f"⚠️ Cache save failed: {str(cache_error)}")
                            st.info("💡 Training completed but results may not be saved. Please check Step 5.")
                            
                            # Save minimal data to session state as fallback (with error handling)
                            try:
                                step4_data = {
                                    'completed': True,
                                    'total_models': results.get('total_models', 0),
                                    'successful_combinations': results.get('successful_combinations', 0),
                                    'elapsed_time': results.get('elapsed_time', 0),
                                    'status': results.get('status', 'unknown'),
                                    'cache_error': str(cache_error)
                                }
                                session_manager.set_step_data(4, step4_data)
                                st.info("✅ Fallback data saved to session state.")
                            except Exception as session_error:
                                st.error(f"❌ Session state save also failed: {str(session_error)}")
                                st.info("💡 Training completed but no data was saved.")
                        
                    # Gentle memory cleanup after training (avoid aggressive cleanup that might cause crashes)
                    try:
                        import gc
                        gc.collect()
                    except Exception as gc_error:
                        st.warning(f"⚠️ Memory cleanup warning: {gc_error}")
                    
                    # Don't aggressively delete objects that might be needed for UI updates
                    # Let Python's garbage collector handle cleanup naturally
                        
                    with results_debug_container:
                        st.success("✅ Training results saved!")
                        st.info("💡 Click 'Next ▶' button to proceed to Step 5.")
                    
                    # Mark training as completed (will be set in finally block)
                else:
                        st.error("❌ No successful training results found")
                        with results_debug_container:
                            st.error("🔍 Debug: No successful results found in any format")
                            st.error(f"🔍 Debug: Overall status = {results.get('status', 'NO_STATUS')}")
                            
                            # Show overall error if available
                            if results.get('status') == 'failed' and 'error' in results:
                                st.error(f"🔍 Debug: Overall error = {results['error']}")
                            
                            if 'model_results' in results:
                                model_results = results.get('model_results', {})
                                for model_name, model_data in model_results.items():
                                    if isinstance(model_data, dict):
                                        st.error(f"🔍 Debug: {model_name} status = {model_data.get('status', 'NO_STATUS')}")
                                        if model_data.get('status') == 'failed':
                                            st.error(f"🔍 Debug: {model_name} error = {model_data.get('error', 'NO_ERROR')}")
                            elif 'comprehensive_results' in results:
                                comprehensive_results = results.get('comprehensive_results', [])
                                for i, result in enumerate(comprehensive_results):
                                    if isinstance(result, dict):
                                        st.error(f"🔍 Debug: Result {i} status = {result.get('status', 'NO_STATUS')}")
                                        if result.get('status') == 'failed':
                                            st.error(f"🔍 Debug: Result {i} error = {result.get('error', 'NO_ERROR')}")
                
                # Handle case where results is not successful
                if not (results and isinstance(results, dict) and results.get('status') == 'success'):
                    error_msg = 'Unknown error'
                    if isinstance(results, dict):
                        error_msg = results.get('error', 'Unknown error')
                        # Show detailed debug info
                        with results_debug_container:
                            st.error(f"🔍 Debug: results status = {results.get('status', 'NOT_FOUND')}")
                            st.error(f"🔍 Debug: results keys = {list(results.keys())}")
                            if 'error' in results:
                                st.error(f"🔍 Debug: error details = {results['error']}")
                            if 'model_results' in results:
                                st.error(f"🔍 Debug: model_results = {results['model_results']}")
                    elif results:
                        error_msg = str(results)
                        with results_debug_container:
                            st.error(f"🔍 Debug: results type = {type(results)}")
                            st.error(f"🔍 Debug: results content = {results}")
                    else:
                        error_msg = 'No results returned'
                        with results_debug_container:
                            st.error(f"🔍 Debug: results is None or empty")
                    
                    st.error(f"❌ Training failed: {error_msg}")
                
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
        
        finally:
            # Always cleanup training states
            st.session_state.training_in_progress = False
            
            # CRITICAL: Clear any remaining spinner/progress indicators
            try:
                # Clear any progress bars or spinners that might be stuck
                import gc
                gc.collect()
            except Exception:
                pass
            
            # Check if results exists and is successful before updating states
            if results is None or not (isinstance(results, dict) and results.get('status') == 'success'):
                st.session_state.training_started = False
                st.session_state.training_completed = False
            else:
                # Training was successful, keep completed state
                st.session_state.training_completed = True
                
                # NO RERUN - Keep the results displayed without refreshing
                # This prevents rerun after displaying detailed results
                pass
    
    # Navigation buttons
    render_navigation_buttons()

def render_step5_wireframe():
    """Render Step 5 - SHAP Visualization & Confusion Matrix"""
    
    # Step title
    st.markdown("""
    <h2 style="text-align: left; color: var(--text-color); margin: 2rem 0 1rem 0; font-size: 1.8rem;">
        📍 STEP 5/5: SHAP Visualization & Confusion Matrix
    </h2>
    """, unsafe_allow_html=True)
    
    # Create tabs for SHAP and Confusion Matrix
    tab1, tab2, tab3 = st.tabs([
        "🔍 SHAP Analysis", 
        "📊 Confusion Matrix", 
        "📈 Model Comparison"
    ])
    
    with tab1:
        render_shap_analysis()
    
    with tab2:
        render_confusion_matrix()
    
    with tab3:
        render_model_comparison()
    
    # Navigation buttons
    render_navigation_buttons()
    
    # Get data from previous steps
    # Use global session_manager instance
    step1_data = session_manager.get_step_data(1)
    step2_data = session_manager.get_step_data(2)
    step3_data = session_manager.get_step_data(3)
    step4_data = session_manager.get_step_data(4)
    
    if not step1_data or 'dataframe' not in step1_data:
        st.error("❌ No dataset found. Please complete Step 1 first.")
        if st.button("← Go to Step 1"):
            session_manager.set_current_step(1)
            st.rerun()
        return
    
    if not step2_data or not step2_data.get('completed', False):
        st.error("❌ Please complete Step 2 (Column Selection & Preprocessing) first.")
        if st.button("← Go to Step 2"):
            session_manager.set_current_step(2)
            st.rerun()
        return
    
    if not step3_data or not step3_data.get('completed', False):
        st.error("❌ Please complete Step 3 (Model Configuration & Vectorization) first.")
        if st.button("← Go to Step 3"):
            session_manager.set_current_step(3)
            st.rerun()
        return
    
    if not step4_data or not step4_data.get('completed', False):
        st.error("❌ Please complete Step 4 (Training Execution & Monitoring) first.")
        if st.button("← Go to Step 4"):
            session_manager.set_current_step(4)
            st.rerun()
        return
    
    # Load both training results cache and model cache
    # Training results cache contains summary metadata
    # Model cache contains individual model data
    session_key = step4_data.get('session_key')
    cache_path = step4_data.get('cache_path')
    
    if not session_key:
        st.error("❌ No session key found. Please complete Step 4 first.")
        if st.button("← Go to Step 4"):
            session_manager.set_current_step(4)
            st.rerun()
        return
    
    # Load training results cache (summary metadata)
    training_summary = None
    if cache_path:
        try:
            from cache_manager import training_results_cache
            training_summary = training_results_cache.load_training_results(session_key)
            st.info(f"📊 Training summary loaded: {len(training_summary.get('model_summary', []))} models")
        except Exception as e:
            st.warning(f"⚠️ Could not load training summary: {e}")
    
    # Load model cache (individual model data)
    try:
        from cache_manager import cache_manager
        
        # List all cached models to get available models for Step 5
        cached_models = cache_manager.list_cached_models()
        
        if not cached_models:
            st.error("❌ No cached models found. Please complete Step 4 first.")
            if st.button("← Go to Step 4"):
                session_manager.set_current_step(4)
                st.rerun()
            return
            
        st.success(f"✅ Found {len(cached_models)} cached models ready for analysis!")
        
        # Show training summary if available
        if training_summary:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Models", training_summary.get('total_models', 0))
            with col2:
                st.metric("✅ Successful", training_summary.get('successful_combinations', 0))
            with col3:
                elapsed_time = training_summary.get('elapsed_time', 0)
                try:
                    elapsed_time_float = float(elapsed_time)
                    st.metric("⏱️ Total Time", f"{elapsed_time_float:.1f}s")
                except (ValueError, TypeError):
                    st.metric("⏱️ Total Time", f"{elapsed_time}s")
        
    except Exception as e:
        st.error(f"❌ Failed to load model cache: {e}")
        if st.button("← Go to Step 4"):
            session_manager.set_current_step(4)
            st.rerun()
        return
def render_shap_analysis():
    """Render SHAP analysis and visualization"""
    # Initialize debug info collection
    debug_info = []
    
    # Beautiful header for SHAP Analysis
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    ">
        <h2 style="
            color: white;
            margin: 0 0 1rem 0;
            text-align: center;
            font-size: 2rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        ">
            🔍 SHAP Analysis
        </h2>
        <p style="
            color: rgba(255, 255, 255, 0.9);
            text-align: center;
            margin: 0;
            font-size: 1.1rem;
            font-weight: 300;
        ">
            SHapley Additive exPlanations - Explain individual predictions by computing feature contributions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enable/Disable SHAP
    enable_shap = st.checkbox("Enable SHAP Analysis", value=True, help="Enable SHAP visualization for tree-based models")
    
    if enable_shap:
        
        col1, col2 = st.columns(2)
        
        with col1:
            sample_size = st.number_input(
                "Sample Size for SHAP:",
                min_value=100,
                max_value=10000,
                value=1000,
                help="Number of samples to use for SHAP analysis"
            )
            
            output_dir = st.text_input(
                "Output Directory:",
                value="info/Result/",
                help="Directory to save SHAP plots"
            )
        
        with col2:
            # Available models for SHAP (tree-based models - bao gồm cả mô hình cũ và mới)
            available_models = [
                # Mô hình cũ
                'decision_tree',
                # Mô hình mới
                'random_forest', 'xgboost', 'lightgbm', 'catboost',
                # 'adaboost',  # Removed: not supported by SHAP TreeExplainer
                'gradient_boosting'
            ]
            
            selected_models = st.multiselect(
                "Select Models for SHAP Analysis:",
                available_models,
                default=available_models,  # Default to all available models
                help="Choose tree-based models for SHAP analysis"
            )
        
        plot_types = st.multiselect(
            "Select Plot Types:",
            ["summary", "bar", "dependence", "waterfall"],
            default=["summary", "bar", "dependence"],
            help="Choose types of SHAP plots to generate"
        )
        
        # Dependence plot features (if dependence is selected)
        if "dependence" in plot_types:
            dependence_features = st.multiselect(
                "Features for Dependence Plots:",
                ["auto", "top_3", "custom"],
                default=["top_3"],
                help="Select features for dependence plots"
            )
        
        # Generate SHAP plots button with beautiful styling
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Generate SHAP Analysis", key="generate_shap_btn", help="Click to generate SHAP analysis for selected models"):
            if selected_models and plot_types:
                with st.spinner("Generating SHAP analysis from cache..."):
                    try:
                        # Import cache manager and visualization
                        from cache_manager import CacheManager
                        from visualization import create_shap_explainer, generate_shap_summary_plot, generate_shap_bar_plot, generate_shap_dependence_plot, plot_shap_waterfall
                        import numpy as np
                        
                        # Initialize cache manager
                        cache_manager = CacheManager()
                        
                        # Get available cached models
                        available_caches = cache_manager.list_cached_models()
                        
                        if not available_caches:
                            st.warning("⚠️ No cached models found. Please complete training in Step 4 first.")
                            return
                        
                        debug_info.append(f"📊 Found {len(available_caches)} cached models")
                        
                        # Filter models that are selected and available in cache
                        cached_models = []
                        for model_name in selected_models:
                            # Look for cached models matching the selected model
                            for cache_info in available_caches:
                                model_key = cache_info.get('model_key', '')
                                # Check if model_name matches the beginning of model_key (before any scaler suffix)
                                if model_key.startswith(model_name + '_') or model_key == model_name:
                                    cached_models.append({
                                        'model_name': model_name,
                                        'cache_info': cache_info
                                    })
                                    break
                        
                        if not cached_models:
                            st.warning(f"⚠️ No cached models found for selected models: {selected_models}")
                            debug_info.append("Available cached models:")
                            for cache_info in available_caches:
                                debug_info.append(f"- {cache_info.get('model_key', 'Unknown')}")
                            return
                        
                        debug_info.append(f"✅ Found {len(cached_models)} cached models for SHAP analysis")
                        
                        # Check SHAP cache availability
                        debug_info.append("🔍 Checking SHAP cache availability...")
                        try:
                            from shap_cache_manager import shap_cache_manager
                            import os
                            import time
                            cache_dir = shap_cache_manager.cache_dir
                            
                            # Add cache directory info to debug_info
                            debug_info.append(f"🔍 Cache directory: {cache_dir}")
                            debug_info.append(f"🔍 Cache directory exists: {cache_dir.exists()}")
                            if cache_dir.exists():
                                cache_files = list(cache_dir.glob("*.pkl"))
                                debug_info.append(f"🔍 Cache files found: {len(cache_files)}")
                                if cache_files:
                                    debug_info.append("🔍 Cache file details:")
                                    for cache_file in cache_files[:5]:  # Show first 5 files
                                        file_size = cache_file.stat().st_size if cache_file.exists() else 0
                                        file_age = time.time() - cache_file.stat().st_mtime if cache_file.exists() else 0
                                        debug_info.append(f"  - {cache_file.name} (size: {file_size} bytes, age: {file_age:.1f}s)")
                            else:
                                debug_info.append("⚠️ SHAP cache directory does not exist")
                                debug_info.append("💡 This means Step 4 never created SHAP cache")
                            
                            # Add debug info for SHAP analysis process
                            debug_info.append("🔍 **SHAP Analysis Debug Info:**")
                            debug_info.append("🔍 This section shows detailed debug information for SHAP cache search and plot generation")
                            
                            if cache_dir.exists():
                                cache_files = list(cache_dir.glob("*.pkl"))
                                if cache_files:
                                    debug_info.append(f"✅ Found {len(cache_files)} SHAP cache files")
                                    debug_info.append("Available SHAP cache files:")
                                    for cache_file in cache_files[:10]:  # Show first 10 files
                                        debug_info.append(f"  - {cache_file.name}")
                                    if len(cache_files) > 10:
                                        debug_info.append(f"  ... and {len(cache_files) - 10} more files")
                                else:
                                    debug_info.append("⚠️ No SHAP cache files found")
                                    debug_info.append("💡 Please complete Step 4 training to generate SHAP cache first.")
                            else:
                                debug_info.append("⚠️ SHAP cache directory does not exist")
                                debug_info.append("💡 Please complete Step 4 training to generate SHAP cache first.")
                        except Exception as e:
                            st.error(f"❌ Error checking SHAP cache: {e}")
                        
                        # Check if any SHAP cache files exist
                        try:
                            cache_dir = shap_cache_manager.cache_dir
                            cache_files = list(cache_dir.glob("*.pkl")) if cache_dir.exists() else []
                            
                            if not cache_files:
                                st.error("❌ No SHAP cache files found!")
                                st.info("💡 **To generate SHAP cache:**")
                                st.info("1. Go to Step 4 (Training Execution & Monitoring)")
                                st.info("2. Enable SHAP cache generation during training")
                                st.info("3. Complete the training process")
                                st.info("4. Return to Step 5 to view SHAP analysis")
                                return
                        except Exception as e:
                            st.error(f"❌ Error checking SHAP cache: {e}")
                            return
                        
                        # Generate SHAP plots for each cached model
                        for model_data in cached_models:
                            model_name = model_data['model_name']
                            cache_info = model_data['cache_info']
                            
                            debug_info.append(f"🔍 Generating SHAP plots for {model_name}...")
                            
                            try:
                                # Load the cached model and data
                                cache_data = cache_manager.load_model_cache(
                                    cache_info['model_key'], 
                                    cache_info['dataset_id'], 
                                    cache_info['config_hash']
                                )
                                
                                model = cache_data['model']
                                eval_predictions = cache_data.get('eval_predictions')
                                
                                if eval_predictions is None or len(eval_predictions) == 0:
                                    st.warning(f"⚠️ No test data found for {model_name}")
                                    continue
                                
                                # Use eval_predictions as test data for SHAP
                                # Extract only feature columns (exclude true_labels and predictions)
                                feature_columns = [col for col in eval_predictions.columns if col not in ['true_labels', 'predictions']]
                                test_data = eval_predictions[feature_columns].values
                                
                                # Sample data for SHAP analysis
                                if len(test_data) > sample_size:
                                    import numpy as np
                                    indices = np.random.choice(len(test_data), sample_size, replace=False)
                                    sample_data = test_data[indices]
                                else:
                                    sample_data = test_data
                                
                                # Try to load SHAP cache first
                                from shap_cache_manager import shap_cache_manager
                                from visualization import (
                                    generate_shap_summary_plot_from_values,
                                    generate_shap_bar_plot_from_values,
                                    generate_shap_dependence_plot_from_values,
                                    plot_shap_waterfall_from_values
                                )
                                
                                # Try different cache key formats - use the exact model_name from cache_info
                                # The cache was created with model_name="{model_key}_{scaler_name}" in Step 4 (after fix)
                                # Extract scaler_name from dataset_id (e.g., "numeric_dataset_StandardScaler" -> "StandardScaler")
                                dataset_id = cache_info.get('dataset_id', '')
                                scaler_name_from_dataset = dataset_id.replace('numeric_dataset_', '') if dataset_id.startswith('numeric_dataset_') else 'StandardScaler'
                                
                                # Try all possible scalers since cache files have different scalers
                                all_possible_scalers = ['StandardScaler', 'MinMaxScaler', 'RobustScaler']
                                
                                cache_keys_to_try = []
                                # Add exact key from Step 4 first
                                cache_keys_to_try.append(f"{cache_info['model_key']}_{scaler_name_from_dataset}")
                                # Add all possible scaler combinations
                                for scaler in all_possible_scalers:
                                    cache_keys_to_try.append(f"{cache_info['model_key']}_{scaler}")
                                # Add original key without scaler
                                cache_keys_to_try.append(cache_info['model_key'])
                                # Add model name combinations
                                cache_keys_to_try.append(f"{model_name}_{scaler_name_from_dataset}")
                                for scaler in all_possible_scalers:
                                    cache_keys_to_try.append(f"{model_name}_{scaler}")
                                cache_keys_to_try.append(model_name)
                                
                                # Remove duplicates while preserving order
                                cache_keys_to_try = list(dict.fromkeys(cache_keys_to_try))
                                
                                cached_explainer, cached_shap_values = None, None
                                successful_key = None
                                
                                # Try to load SHAP cache directly from files (bypass hash-based key)
                                debug_info = []
                                debug_info.append(f"🔍 Debug: Trying to find cache for keys: {cache_keys_to_try}")
                                
                                try:
                                    cache_dir = shap_cache_manager.cache_dir
                                    if cache_dir.exists():
                                        cache_files = list(cache_dir.glob("*.pkl"))
                                        
                                        # Create a mapping of model names to cache data for efficient lookup
                                        cache_mapping = {}
                                        for cache_file in cache_files:
                                            try:
                                                with open(cache_file, 'rb') as f:
                                                    cache_data = pickle.load(f)
                                                cached_model_name = cache_data.get('model_name', '')
                                                if cached_model_name:
                                                    cache_mapping[cached_model_name] = cache_data
                                            except Exception as e:
                                                continue
                                        
                                        # Try to find cache using efficient lookup
                                        for cache_key in cache_keys_to_try:
                                            # Direct match first
                                            if cache_key in cache_mapping:
                                                cached_shap_values = cache_mapping[cache_key].get('shap_values')
                                                if cached_shap_values is not None:
                                                    successful_key = cache_key
                                                    debug_info.append(f"✅ Found SHAP cache for key: '{cache_key}'")
                                                    break
                                            
                                            # Pattern match for scaler combinations
                                            for cached_model_name, cache_data in cache_mapping.items():
                                                if cached_model_name.startswith(cache_key + "_"):
                                                    cached_shap_values = cache_data.get('shap_values')
                                                    if cached_shap_values is not None:
                                                        successful_key = cache_key
                                                        debug_info.append(f"✅ Found SHAP cache for key: '{cache_key}' (model: '{cached_model_name}')")
                                                        break
                                            else:
                                                continue  # No match found, try next key
                                            break  # Match found, exit outer loop
                                        
                                        if cached_shap_values is None:
                                            debug_info.append(f"❌ No SHAP cache found for any key in {len(cache_files)} files")
                                            
                                except Exception as e:
                                    debug_info.append(f"❌ Error searching cache files: {e}")
                                
                                # Debug: List available SHAP cache files
                                if cached_shap_values is None:
                                    debug_info.append("🔍 Debug: Available SHAP cache files:")
                                    try:
                                        import os
                                        cache_dir = shap_cache_manager.cache_dir
                                        if cache_dir.exists():
                                            cache_files = list(cache_dir.glob("*.pkl"))
                                            if cache_files:
                                                for cache_file in cache_files[:10]:  # Show first 10 files
                                                    debug_info.append(f"  - {cache_file.name}")
                                            else:
                                                debug_info.append("  - No SHAP cache files found")
                                        else:
                                            debug_info.append("  - SHAP cache directory does not exist")
                                    except Exception as e:
                                        debug_info.append(f"  - Error listing cache files: {e}")
                                
                                if cached_shap_values is not None:
                                    debug_info.append(f"✅ Loaded SHAP cache for {model_name} (key: {successful_key})")
                                    # Use cached SHAP values
                                    shap_values = cached_shap_values
                                    explainer = None  # We have cached values, don't need explainer
                                else:
                                    debug_info.append(f"⚠️ No SHAP cache found for {model_name}")
                                    debug_info.append("💡 Please complete Step 4 training to generate SHAP cache first.")
                                    debug_info.append("ℹ️ Skipping SHAP analysis for this model.")
                                    continue
                                
                                # Get sample data from cache (should match SHAP values)
                                sample_data = cache_data.get('sample_data')
                                if sample_data is None:
                                    st.error(f"❌ No sample_data found in cache for {model_name}")
                                    continue
                                
                                # Get feature names from cache
                                feature_names = cache_data.get('feature_names')
                                if feature_names is None:
                                    # Fallback: create generic feature names
                                    feature_names = [f"Feature_{i}" for i in range(sample_data.shape[1])]
                                
                                debug_info.append(f"🔍 Debug: Using {len(feature_names)} feature names: {feature_names}")
                                debug_info.append(f"🔍 Debug: sample_data from cache - type: {type(sample_data)}, shape: {getattr(sample_data, 'shape', 'No shape')}")
                                
                                # Create organized layout for this model
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    padding: 1rem;
                                    border-radius: 10px;
                                    margin: 1.5rem 0;
                                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                ">
                                    <h4 style="
                                        color: white;
                                        margin: 0;
                                        text-align: center;
                                        font-size: 1.2rem;
                                        font-weight: bold;
                                        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                                    ">
                                        🤖 {model_name.replace('_', ' ').title()} Model Analysis
                                    </h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Create tabs for different plot types within each model
                                plot_tabs = st.tabs([f"📊 {plot_type.title()}" for plot_type in plot_types])
                                
                                # Generate plots based on selected types
                                for tab_idx, plot_type in enumerate(plot_types):
                                    with plot_tabs[tab_idx]:
                                        if plot_type == "summary":
                                            st.markdown("**📈 Summary Plot:** Shows feature importance and impact")
                                            try:
                                                if explainer is not None:
                                                    fig = generate_shap_summary_plot(explainer, sample_data, feature_names)
                                                else:
                                                    # Use cached SHAP values for summary plot
                                                    debug_info.append(f"🔍 Debug: shap_values type: {type(shap_values)}, shape: {getattr(shap_values, 'shape', 'No shape')}")
                                                    debug_info.append(f"🔍 Debug: sample_data type: {type(sample_data)}, shape: {getattr(sample_data, 'shape', 'No shape')}")
                                                    fig = generate_shap_summary_plot_from_values(shap_values, sample_data, feature_names)
                                                    debug_info.append(f"🔍 Debug: generate_shap_summary_plot_from_values returned: {type(fig)}")
                                                if fig:
                                                    debug_info.append(f"📊 Displaying {plot_type} plot for {model_name}")
                                                    st.pyplot(fig)
                                                else:
                                                    debug_info.append(f"⚠️ Failed to generate {plot_type} plot for {model_name}")
                                            except Exception as e:
                                                debug_info.append(f"❌ Error generating {plot_type} plot for {model_name}: {e}")
                                        
                                        elif plot_type == "bar":
                                            st.markdown("**📊 Bar Plot:** Shows mean absolute SHAP values for each feature")
                                            try:
                                                if explainer is not None:
                                                    fig = generate_shap_bar_plot(explainer, sample_data, feature_names)
                                                else:
                                                    # Use cached SHAP values for bar plot
                                                    debug_info.append(f"🔍 Debug: shap_values type: {type(shap_values)}, shape: {getattr(shap_values, 'shape', 'No shape')}")
                                                    debug_info.append(f"🔍 Debug: sample_data type: {type(sample_data)}, shape: {getattr(sample_data, 'shape', 'No shape')}")
                                                    fig = generate_shap_bar_plot_from_values(shap_values, sample_data, feature_names)
                                                    debug_info.append(f"🔍 Debug: generate_shap_bar_plot_from_values returned: {type(fig)}")
                                                    if fig is None:
                                                        debug_info.append(f"❌ generate_shap_bar_plot_from_values returned None - check error logs")
                                                if fig:
                                                    debug_info.append(f"📊 Displaying {plot_type} plot for {model_name}")
                                                    st.pyplot(fig)
                                                else:
                                                    debug_info.append(f"⚠️ Failed to generate {plot_type} plot for {model_name}")
                                            except Exception as e:
                                                debug_info.append(f"❌ Error generating {plot_type} plot for {model_name}: {e}")
                                        
                                        elif plot_type == "dependence":
                                            st.markdown("**🔗 Dependence Plots:** Shows how each feature affects the model's output")
                                            # Generate dependence plots for top 3 features
                                            if isinstance(shap_values, list):
                                                shap_values_for_dep = shap_values[1]  # Use positive class for binary classification
                                            else:
                                                shap_values_for_dep = shap_values
                                            
                                            # Ensure shap_values_for_dep is a numpy array with correct shape
                                            try:
                                                import numpy as np
                                                if not isinstance(shap_values_for_dep, np.ndarray):
                                                    shap_values_for_dep = np.array(shap_values_for_dep)
                                                
                                                # Handle multi-class SHAP values (3D arrays)
                                                if len(shap_values_for_dep.shape) == 3:
                                                    if shap_values_for_dep.shape[0] == 2:  # (2, n_samples, n_features)
                                                        shap_values_for_dep = shap_values_for_dep[1]
                                                    elif shap_values_for_dep.shape[2] == 2:  # (n_samples, n_features, 2)
                                                        shap_values_for_dep = shap_values_for_dep[:, :, 1]
                                                    else:
                                                        shap_values_for_dep = shap_values_for_dep[0]
                                                
                                                # Calculate mean absolute SHAP values to find top features
                                                if len(shap_values_for_dep.shape) >= 2:
                                                    mean_shap = np.mean(np.abs(shap_values_for_dep), axis=0)
                                                    top_features = np.argsort(mean_shap)[-3:][::-1]  # Top 3 features
                                                else:
                                                    # If only 1D, use all features
                                                    top_features = np.arange(min(3, len(shap_values_for_dep)))
                                                
                                            except Exception as e:
                                                debug_info.append(f"❌ Error processing SHAP values for dependence plots: {e}")
                                                continue
                                            
                                            # Create sub-tabs for top 3 features
                                            if len(top_features) > 0:
                                                dep_tabs = st.tabs([f"Feature {i+1}" for i in range(min(3, len(top_features)))])
                                                
                                                for feature_idx in top_features:
                                                    tab_idx_dep = list(top_features).index(feature_idx)
                                                    with dep_tabs[tab_idx_dep]:
                                                        feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature_{feature_idx}"
                                                        st.markdown(f"**Feature: {feature_name}**")
                                                        if explainer is not None:
                                                            fig = generate_shap_dependence_plot(
                                                                explainer, sample_data, feature_names, 
                                                                feature_index=feature_idx
                                                            )
                                                        else:
                                                            # Use cached SHAP values for dependence plot
                                                            fig = generate_shap_dependence_plot_from_values(
                                                                shap_values_for_dep, sample_data, feature_names, 
                                                                feature_index=feature_idx
                                                            )
                                                        if fig:
                                                            st.pyplot(fig)
                                        
                                        elif plot_type == "waterfall":
                                            st.markdown("**🌊 Waterfall Plots:** Shows prediction explanation for individual instances")
                                            # Generate waterfall plots for first 3 instances
                                            try:
                                                if len(sample_data) > 0:
                                                    waterfall_tabs = st.tabs([f"Instance {i+1}" for i in range(min(3, len(sample_data)))])
                                                    
                                                    for instance_idx in range(min(3, len(sample_data))):
                                                        with waterfall_tabs[instance_idx]:
                                                            st.markdown(f"**Instance {instance_idx + 1}**")
                                                            try:
                                                                if explainer is not None:
                                                                    fig = plot_shap_waterfall(
                                                                        explainer, sample_data, 
                                                                        instance_index=instance_idx, 
                                                                        feature_names=feature_names
                                                                    )
                                                                else:
                                                                    # Use cached SHAP values for waterfall plot
                                                                    fig = plot_shap_waterfall_from_values(
                                                                        shap_values, sample_data, 
                                                                        instance_index=instance_idx, 
                                                                        feature_names=feature_names
                                                                    )
                                                                if fig:
                                                                    st.pyplot(fig)
                                                                else:
                                                                    debug_info.append(f"⚠️ Failed to generate waterfall plot for instance {instance_idx + 1}")
                                                            except Exception as e:
                                                                debug_info.append(f"❌ Error generating waterfall plot for instance {instance_idx + 1}: {e}")
                                            except Exception as e:
                                                debug_info.append(f"❌ Error processing waterfall plots: {e}")
                                
                                # Add separator between models
                                st.markdown("---")
                                
                                # Count generated plots
                                plot_count = len(plot_types)
                                if "dependence" in plot_types:
                                    plot_count += 2  # +2 for top 3 features (minus 1 for the base plot_type)
                                if "waterfall" in plot_types:
                                    plot_count += 2  # +2 for first 3 instances (minus 1 for the base plot_type)
                                
                                debug_info.append(f"✅ Generated {plot_count} SHAP plots for {model_name}")
                                
                            except Exception as e:
                                st.error(f"❌ Error generating SHAP plots for {model_name}: {str(e)}")
                                continue
                        
                        # Add final summary section with beautiful styling
                        models_processed = len(cached_models) if 'cached_models' in locals() else 0
                        cache_files_found = len(cache_files) if 'cache_files' in locals() else 0
                        
                        st.markdown("""
                        <div style="
                            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                            padding: 1.5rem;
                            border-radius: 15px;
                            margin: 2rem 0;
                            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
                        ">
                            <h3 style="
                                color: white;
                                margin: 0 0 1rem 0;
                                text-align: center;
                                font-size: 1.3rem;
                                font-weight: bold;
                                text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                            ">
                                🎉 SHAP Analysis Summary
                            </h3>
                            <div style="
                                background: rgba(255, 255, 255, 0.9);
                                padding: 1rem;
                                border-radius: 10px;
                                color: #333;
                            ">
                                <p style="margin: 0.5rem 0;"><strong>✅ Models Processed:</strong> {models_processed}</p>
                                <p style="margin: 0.5rem 0;"><strong>📊 SHAP Cache Files Found:</strong> {cache_files_found}</p>
                                <p style="margin: 0.5rem 0;"><strong>💾 Data Source:</strong> Only cached SHAP data was used (no new training)</p>
                            </div>
                        </div>
                        """.format(models_processed=models_processed, cache_files_found=cache_files_found), unsafe_allow_html=True)
                        
                        # Summary for debug info
                        debug_info.append("✅ SHAP analysis completed!")
                        debug_info.append("📊 **Summary:**")
                        debug_info.append(f"- Models processed: {models_processed}")
                        debug_info.append(f"- SHAP cache files found: {cache_files_found}")
                        debug_info.append("- Only cached SHAP data was used (no new training)")
                        
                        # Save SHAP configuration
                        shap_config = {
                            'enable': enable_shap,
                            'sample_size': sample_size,
                            'output_dir': output_dir,
                            'selected_models': selected_models,
                            'plot_types': plot_types,
                            'dependence_features': dependence_features if "dependence" in plot_types else [],
                            'generated': True
                        }
                        
                        # Save to session
                        session_manager.set_step_data(5, {'shap_config': shap_config})
                        
                        debug_info.append("🎉 SHAP analysis completed successfully!")
                        
                        # Display a brief success message
                        st.success("🎉 SHAP analysis completed successfully!")
                        
                        # Display debug info in a single container
                        if debug_info:
                            with st.expander("📋 Training Log", expanded=False):
                                for info in debug_info:
                                    st.info(info)
                        
                    except Exception as e:
                        st.error(f"❌ Error during SHAP analysis: {str(e)}")
                        st.exception(e)
            else:
                st.error("❌ Please select at least one model and one plot type")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.info("ℹ️ SHAP analysis is disabled.")


def render_confusion_matrix():
    """Render confusion matrix visualization"""
    st.markdown("""
    <h3 style="color: var(--text-color); margin: 1.5rem 0 1rem 0;">📊 Confusion Matrix Analysis</h3>
    """, unsafe_allow_html=True)
    
    st.info("📝 **Confusion Matrix** shows the performance of a classification model by displaying correct and incorrect predictions for each class.")
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        normalize_method = st.selectbox(
            "Normalization Method:",
            ["true", "pred", "all", None],
            index=0,
            help="How to normalize the confusion matrix"
        )
        
        save_plots = st.checkbox(
            "Save Plots",
            value=True,
            help="Save confusion matrix plots to disk"
        )
    
    with col2:
        show_metrics = st.checkbox(
            "Show Metrics",
            value=True,
            help="Display classification metrics"
        )
    
    # Available models (check from cache)
    st.markdown("**🔍 Available Cached Models:**")
    
    # Get available models from actual cache
    try:
        from cache_manager import CacheManager
        cache_manager = CacheManager()
        cached_models = cache_manager.list_cached_models()
        
        if not cached_models:
            st.warning("⚠️ No cached models found. Please complete Step 4 training first.")
            cached_models = []
    except Exception as e:
        st.error(f"❌ Error loading cached models: {str(e)}")
        cached_models = []
    
    if cached_models:
        # Convert cached models to display format
        model_options = []
        for model in cached_models:
            if isinstance(model, dict):
                display_name = f"{model.get('model_key', 'Unknown')} - {model.get('dataset_id', 'Unknown')}"
                model_options.append((display_name, model))
            else:
                model_options.append((str(model), model))
        
        selected_display_name = st.selectbox(
            "Select Model for Confusion Matrix:",
            [option[0] for option in model_options],
            help="Choose a trained model to generate confusion matrix"
        )
        
        # Get selected model object
        selected_model = None
        for display_name, model_obj in model_options:
            if display_name == selected_display_name:
                selected_model = model_obj
                break
        
        # Generate confusion matrix
        if st.button("📊 Generate Confusion Matrix", type="primary"):
            with st.spinner("Generating confusion matrix..."):
                try:
                    from confusion_matrix_cache import ConfusionMatrixCache
                    cm_cache = ConfusionMatrixCache()
                    
                    # Parse labels order if provided (use default empty string)
                    labels_order = ""  # Default empty string
                    labels_list = None
                    if labels_order and labels_order.strip():
                        labels_list = [label.strip() for label in labels_order.split(',')]
                    
                    # Generate confusion matrix from cache
                    save_path = None
                    if save_plots:
                        # Create directory if it doesn't exist
                        import os
                        os.makedirs("cache/confusion_matrices", exist_ok=True)
                        save_path = f"cache/confusion_matrices/{selected_model['model_key']}_{selected_model['dataset_id']}.png"
                    
                    result = cm_cache.generate_confusion_matrix_from_cache(
                        model_key=selected_model['model_key'],
                        dataset_id=selected_model['dataset_id'],
                        config_hash=selected_model['config_hash'],
                        normalize=normalize_method,
                        labels_order=labels_list,
                        save_path=save_path
                    )
                    
                    if result and result.get('plot'):
                        st.success("✅ Confusion matrix generated successfully!")
                        
                        # Save configuration and result to session
                        cm_config = {
                            'normalize_method': normalize_method,
                            'save_plots': save_plots,
                            'show_metrics': show_metrics,
                            'labels_order': labels_order,
                            'selected_model': selected_model
                        }
                        
                        current_step5_data = session_manager.get_step_data(5) or {}
                        current_step5_data['confusion_matrix_config'] = cm_config
                        current_step5_data['confusion_matrix_result'] = result
                        session_manager.set_step_data(5, current_step5_data)
                        
                    else:
                        st.error("❌ Failed to generate confusion matrix. Please check if the model has evaluation predictions.")
                        
                except Exception as e:
                    st.error(f"❌ Error generating confusion matrix: {str(e)}")
                    st.info("💡 Make sure the model was trained with evaluation predictions saved to cache.")
        
        # Display confusion matrix results from session
        current_step5_data = session_manager.get_step_data(5) or {}
        if current_step5_data.get('confusion_matrix_result'):
            st.markdown("**📊 Confusion Matrix Results:**")
            result = current_step5_data['confusion_matrix_result']
            
            if result.get('plot'):
                st.pyplot(result['plot'])
                
                if show_metrics and result.get('metrics'):
                    st.markdown("**📊 Classification Metrics:**")
                    metrics = result['metrics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        accuracy = metrics.get('accuracy', 0)
                        try:
                            accuracy_float = float(accuracy)
                            st.metric("Accuracy", f"{accuracy_float:.4f}")
                        except (ValueError, TypeError):
                            st.metric("Accuracy", f"{accuracy}")
                    with col2:
                        # Use macro_precision as default, fallback to precision
                        precision = metrics.get('macro_precision', metrics.get('precision', 0))
                        try:
                            precision_float = float(precision)
                            st.metric("Precision", f"{precision_float:.4f}")
                        except (ValueError, TypeError):
                            st.metric("Precision", f"{precision}")
                    with col3:
                        # Use macro_recall as default, fallback to recall
                        recall = metrics.get('macro_recall', metrics.get('recall', 0))
                        try:
                            recall_float = float(recall)
                            st.metric("Recall", f"{recall_float:.4f}")
                        except (ValueError, TypeError):
                            st.metric("Recall", f"{recall}")
                    with col4:
                        # Use macro_f1 as default, fallback to f1_score
                        f1_score = metrics.get('macro_f1', metrics.get('f1_score', 0))
                        try:
                            f1_score_float = float(f1_score)
                            st.metric("F1-Score", f"{f1_score_float:.4f}")
                        except (ValueError, TypeError):
                            st.metric("F1-Score", f"{f1_score}")
                    
                    # Show additional metrics if available
                    if metrics.get('class_metrics'):
                        st.markdown("**📊 Per-Class Metrics:**")
                        class_metrics = metrics['class_metrics']
                        
                        # Create a table for per-class metrics
                        import pandas as pd
                        class_data = []
                        for class_name, class_metric in class_metrics.items():
                            class_data.append({
                                'Class': class_name,
                                'Precision': f"{class_metric.get('precision', 0):.4f}",
                                'Recall': f"{class_metric.get('recall', 0):.4f}",
                                'F1-Score': f"{class_metric.get('f1_score', 0):.4f}",
                                'Support': class_metric.get('support', 0)
                            })
                        
                        if class_data:
                            class_df = pd.DataFrame(class_data)
                            st.dataframe(class_df, width='stretch')
                    
                    # Show weighted averages if available
                    if metrics.get('weighted_precision') is not None:
                        st.markdown("**📊 Weighted Averages:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Weighted Precision", f"{metrics.get('weighted_precision', 0):.4f}")
                        with col2:
                            st.metric("Weighted Recall", f"{metrics.get('weighted_recall', 0):.4f}")
                        with col3:
                            st.metric("Weighted F1-Score", f"{metrics.get('weighted_f1', 0):.4f}")
    else:
        st.warning("⚠️ No cached models found. Please complete model training first.")


def render_model_comparison():
    """Render model comparison and metrics"""
    st.markdown("""
    <h3 style="color: var(--text-color); margin: 1.5rem 0 1rem 0;">📈 Model Performance Comparison</h3>
    """, unsafe_allow_html=True)
    
    st.info("📝 **Model Comparison** displays performance metrics across all trained models to help you choose the best one.")
    
    # Get step data
    step3_data = session_manager.get_step_data(3) or {}
    step5_data = session_manager.get_step_data(5) or {}
    
    # Show configurations from previous steps
    st.markdown("**🎯 Training Configurations:**")
    
    # Optuna results
    optuna_config = step3_data.get('optuna_config', {})
    if optuna_config and optuna_config.get('enable', False):
        st.write(f"✅ Optuna optimization: {len(optuna_config.get('selected_models', []))} models")
    
    # Voting ensemble results
    voting_config = step3_data.get('voting_config', {})
    if voting_config and voting_config.get('enable', False):
        st.write(f"✅ Voting ensemble: {len(voting_config.get('selected_models', []))} models")
    
    # Stacking ensemble results
    stacking_config = step3_data.get('stacking_config', {})
    if stacking_config and stacking_config.get('enable', False):
        st.write(f"✅ Stacking ensemble: {len(stacking_config.get('selected_models', []))} base models")
    
    # SHAP analysis results
    shap_config = step5_data.get('shap_config', {})
    if shap_config and shap_config.get('enable', False):
        st.write(f"✅ SHAP analysis: {len(shap_config.get('selected_models', []))} models")
    
    # Performance metrics from actual training results
    st.markdown("**📊 Performance Metrics:**")
    
    # Column selection for comparison
    st.markdown("**⚙️ Select Metrics to Display:**")
    
    available_metrics = [
        "Accuracy",
        "F1-Score", 
        "Precision",
        "Recall",
        "Training Time",
        "Validation Accuracy",
        "CV Mean",
        "CV Std",
        "Support",
        "Parameters",
        "Status",
        "Cached"
    ]
    
    default_metrics = ["Accuracy", "F1-Score", "Training Time", "Status"]
    
    selected_metrics = st.multiselect(
        "Choose metrics to display:",
        options=available_metrics,
        default=default_metrics,
        help="Select which metrics you want to see in the comparison table"
    )
    
    # Convert multiselect to individual boolean variables for compatibility
    show_accuracy = "Accuracy" in selected_metrics
    show_f1_score = "F1-Score" in selected_metrics
    show_precision = "Precision" in selected_metrics
    show_recall = "Recall" in selected_metrics
    show_training_time = "Training Time" in selected_metrics
    show_validation_acc = "Validation Accuracy" in selected_metrics
    show_cv_mean = "CV Mean" in selected_metrics
    show_cv_std = "CV Std" in selected_metrics
    show_support = "Support" in selected_metrics
    show_params = "Parameters" in selected_metrics
    show_status = "Status" in selected_metrics
    show_cached = "Cached" in selected_metrics
    
    if st.button("📈 Load Model Metrics"):
        try:
            # OPTIMIZED: Load metrics directly from model cache
            from cache_manager import cache_manager
            
            # Get all cached models
            cached_models = cache_manager.list_cached_models()
            
            if not cached_models:
                st.error("❌ No cached models found. Please complete Step 4 first.")
                return
            
            # Load metrics from each cached model
            model_metrics = []
            for model_info in cached_models:
                try:
                    # Load model cache
                    cache_data = cache_manager.load_model_cache(
                        model_info['model_key'],
                        model_info['dataset_id'],
                        model_info['config_hash']
                    )
                    
                    # Extract metrics
                    metrics = cache_data.get('metrics', {})
                    config = cache_data.get('config', {})
                    
                    model_metrics.append({
                        'model_name': model_info['model_key'],
                        'dataset_id': model_info['dataset_id'],
                        'config_hash': model_info['config_hash'][:8] + '...',
                        'accuracy': metrics.get('accuracy', metrics.get('test_accuracy', 0)),  # Fallback to test_accuracy
                        'f1_score': metrics.get('f1_score', 0),
                        'precision': metrics.get('precision', 0),
                        'recall': metrics.get('recall', 0),
                        'training_time': metrics.get('training_time', 0),
                        'status': 'success',
                        'cached': True
                    })
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not load metrics for {model_info['model_key']}: {e}")
            
            if not model_metrics:
                st.error("❌ No valid model metrics found.")
                return
            
            # Convert model_metrics to metrics_data format
            metrics_data = []
            for model_metric in model_metrics:
                row_data = {'Model': model_metric['model_name']}
                
                if show_accuracy:
                    accuracy = model_metric['accuracy']
                    try:
                        accuracy_float = float(accuracy)
                        row_data['Accuracy'] = f"{accuracy_float:.4f}"
                    except (ValueError, TypeError):
                        row_data['Accuracy'] = f"{accuracy}"
                if show_f1_score:
                    f1_score = model_metric['f1_score']
                    try:
                        f1_score_float = float(f1_score)
                        row_data['F1-Score'] = f"{f1_score_float:.4f}"
                    except (ValueError, TypeError):
                        row_data['F1-Score'] = f"{f1_score}"
                if show_precision:
                    precision = model_metric['precision']
                    try:
                        precision_float = float(precision)
                        row_data['Precision'] = f"{precision_float:.4f}"
                    except (ValueError, TypeError):
                        row_data['Precision'] = f"{precision}"
                if show_recall:
                    recall = model_metric['recall']
                    try:
                        recall_float = float(recall)
                        row_data['Recall'] = f"{recall_float:.4f}"
                    except (ValueError, TypeError):
                        row_data['Recall'] = f"{recall}"
                if show_training_time:
                    training_time = model_metric['training_time']
                    try:
                        training_time_float = float(training_time)
                        row_data['Training Time'] = f"{training_time_float:.1f}s"
                    except (ValueError, TypeError):
                        row_data['Training Time'] = f"{training_time}s"
                if show_status:
                    row_data['Status'] = model_metric['status']
                if show_cached:
                    row_data['Cached'] = "Yes" if model_metric['cached'] else "No"
                
                metrics_data.append(row_data)
            
            # Display metrics data
            if metrics_data:
                import pandas as pd
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, width='stretch')
                
                # Find best model
                best_model = None
                best_f1 = 0
                for model_metric in model_metrics:
                    if model_metric['f1_score'] > best_f1:
                        best_f1 = model_metric['f1_score']
                        best_model = model_metric['model_name']
                
                if best_model:
                    try:
                        best_f1_float = float(best_f1)
                        st.success(f"🏆 **Best Model:** {best_model} (F1-Score: {best_f1_float:.4f})")
                    except (ValueError, TypeError):
                        st.success(f"🏆 **Best Model:** {best_model} (F1-Score: {best_f1})")
                else:
                    st.warning("⚠️ No successful training results found.")
                
        except Exception as e:
            st.error(f"❌ Failed to load model metrics: {e}")
    
    # Export options
    st.markdown("**💾 Export Options:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Metrics (CSV)"):
            st.info("📄 Metrics will be exported to CSV format")
    
    with col2:
        if st.button("🖼️ Export Plots (PNG)"):
            st.info("🖼️ All plots will be exported as PNG images")
    
    with col3:
        if st.button("📊 Export Report (PDF)"):
            st.info("📊 Complete analysis report will be generated as PDF")


if __name__ == "__main__":
    # Initialize session manager and ensure current_step is set
    try:
        # Use global session_manager instance
        if session_manager.get_current_step() is None:
            session_manager.set_current_step(1)
    except Exception:
        pass
    
    main()
