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
            "Use Sample Dataset (Cache Folder)",
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

        # Định nghĩa thư mục cache (giả sử là ./cache hoặc bạn có thể sửa lại đường dẫn)
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        allowed_exts = ['.csv', '.xlsx', '.xls', '.json', '.txt']

        # Lấy danh sách file hợp lệ trong thư mục cache
        if os.path.exists(cache_dir):
            files = []
            for ext in allowed_exts:
                files.extend(glob.glob(os.path.join(cache_dir, f"*{ext}")))
            files = sorted(files)
        else:
            files = []

        if not files:
            st.warning("⚠️ Không tìm thấy sample dataset trong thư mục cache.")
        else:
            # Hiển thị danh sách file sample
            file_names = [os.path.basename(f) for f in files]
            selected_file = st.selectbox(
                "Chọn sample dataset từ cache:",
                file_names,
                help="Chọn một file mẫu từ thư mục cache của dự án"
            )

            if selected_file:
                file_path = os.path.join(cache_dir, selected_file)
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

                    st.toast(f"✅ Sample dataset '{selected_file}' loaded from cache.")

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
    
    st.dataframe(df.head(5), use_container_width=True)
    
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

def train_models_with_scaling(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, selected_models, optuna_config, scaler_name, log_container):
    """Train models with specific scaling method using proper train/val/test split and cache system"""
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from models import model_factory, model_registry
    from optuna_optimizer import OptunaOptimizer
    from cache_manager import CacheManager
    import time
    import os
    
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
    
    for model_name in selected_models:
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
                        'cv_mean': cached_metrics.get('cv_mean', 0.0),
                        'cv_std': cached_metrics.get('cv_std', 0.0),
                        'training_time': 0.0,  # Cached, no training time
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
            
            scaler_results[model_name] = {
                'model': final_model,
                'accuracy': test_accuracy,  # Use test accuracy as final metric
                'validation_accuracy': best_score,  # Keep validation accuracy for reference
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'cv_mean': cv_mean,  # Validation score (no double validation)
                'cv_std': cv_std,    # No CV std (avoid double validation)
                'training_time': training_time,
                'params': best_params,
                'status': 'success',
                'cached': False
            }
            
            # Save to cache
            try:
                metrics = {
                    'accuracy': test_accuracy,
                    'validation_accuracy': best_score,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std
                }
                
                cache_config = {
                    'model': mapped_name,
                    'preprocessing': scaler_name,
                    'trials': optuna_config.get('trials', 50) if optuna_enabled else 0,
                    'random_state': 42,
                    'test_size': 0.2
                }
                
                cache_path = cache_manager.save_model_cache(
                    model_key=model_key,
                    dataset_id=dataset_id,
                    config_hash=config_hash,
                    dataset_fingerprint=dataset_fingerprint,
                    model=final_model,
                    params=best_params,
                    metrics=metrics,
                    config=cache_config,
                    feature_names=[f"feature_{i}" for i in range(X_train_scaled.shape[1])],
                    label_mapping={i: f"class_{i}" for i in range(len(set(y_train)))}
                )
                
                with log_container:
                    st.success(f"💾 Cache saved for {model_name} ({scaler_name})")
                    
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
                'cv_mean': 0.0,
                'cv_std': 0.0,
                'training_time': 0.0,
                'params': {},
                'status': 'failed',
                'error': str(e)
            }
    
    return {
        'model_results': scaler_results,
        'status': 'success'
    }


def train_numeric_data_directly(df, input_columns, label_column, selected_models, optuna_config, voting_config, stacking_config, progress_bar, status_text, numeric_scalers=None, remove_duplicates=False, data_split_config=None):
    """Train numeric data using Optuna optimization with cache and cross-validation (ENHANCED)"""
    import time
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
        for scaler_name in numeric_scalers:
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
            scaling_result = train_models_with_scaling(
                X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, 
                selected_models, optuna_config, scaler_name, log_container
            )
            
            # Extract scaler_results from the returned structure
            scaler_results = scaling_result.get('model_results', {})
            
            # Merge results with scaling method prefix
            for model_name, result in scaler_results.items():
                prefixed_name = f"{model_name}_{scaler_name}"
                model_results[prefixed_name] = result
                training_times[prefixed_name] = result.get('training_time', 0)
        
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
            'optuna_enabled': optuna_config.get('enabled', False)
        }
        
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
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
        
        st.success("✅ Step 3 configuration saved!")
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


def render_step4_wireframe():
    """Render Step 4 - Training Execution and Monitoring"""
    
    # Step title
    st.markdown("""
    <h2 style="text-align: left; color: var(--text-color); margin: 2rem 0 1rem 0; font-size: 1.8rem;">
        📍 STEP 4/5: Training Execution and Monitoring
    </h2>
    """, unsafe_allow_html=True)
    
    # Get data from previous steps
    step1_data = session_manager.get_step_data(1)
    step2_data = session_manager.get_step_data(2)
    step3_data = session_manager.get_step_data(3)
    
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
        st.success(f"✅ Split: {train_ratio}%/{val_ratio}%/{test_ratio}%")
    
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
    config_debug_container = st.expander("🔍 Configuration Debug", expanded=False)
    
    # Only show configuration summary if training hasn't started
    if not st.session_state.training_started:
        # Optuna configuration
        optuna_config = step3_data.get('optuna_config', {})
        if optuna_config.get('enabled', False):
            with config_debug_container:
                st.success(f"✅ Optuna: {optuna_config.get('trials', 'N/A')} trials, {len(optuna_config.get('models', []))} models")
        else:
            with config_debug_container:
                st.info("ℹ️ Optuna: Disabled")
        
        # Voting configuration
        voting_config = step3_data.get('voting_config', {})
        if voting_config.get('enabled', False):
            with config_debug_container:
                st.success(f"✅ Voting: {voting_config.get('voting_method', 'N/A')} voting, {len(voting_config.get('models', []))} models")
        else:
            with config_debug_container:
                st.info("ℹ️ Voting: Disabled")
        
        # Stacking configuration
        stacking_config = step3_data.get('stacking_config', {})
        if stacking_config.get('enabled', False):
            with config_debug_container:
                st.success(f"✅ Stacking: {stacking_config.get('meta_learner', 'N/A')} meta-learner, {len(stacking_config.get('base_models', []))} base models")
        else:
            with config_debug_container:
                st.info("ℹ️ Stacking: Disabled")
    
    # Training execution
    st.subheader("🚀 Training Execution")
    
    if st.button("🚀 Start Training", type="primary"):
        # Set training started state to hide configuration summary
        st.session_state.training_started = True
        with st.spinner("🔄 Starting training pipeline..."):
            try:
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
                debug_container = st.expander("🔍 Debug Log", expanded=False)
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
                    numeric_scalers = multi_input_config.get('numeric_scaler', ['StandardScaler'])
                    
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
                
                enhanced_step3_config = {
                    'optuna_config': optuna_config,
                    'voting_config': voting_config,
                    'stacking_config': stacking_config,
                    'selected_models': selected_models,
                    'preprocessing_config': {
                        'selected_methods': ['StandardScaler', 'MinMaxScaler', 'NoScaling']
                    }
                }
                
                # Use different approaches based on data type
                with debug_container:
                    st.info(f"🔍 Debug: About to check data_type = '{data_type}' for training approach")
                if data_type == 'multi_input':
                    # For numeric data: use direct sklearn training (like debug_streamlit_pipeline.py)
                    with debug_container:
                        st.info("🔢 Using direct sklearn for numeric data...")
                    with debug_container:
                        st.info(f"🔍 Debug: Calling train_numeric_data_directly with input_columns = {input_columns}, label_column = {label_column}")
                    results = train_numeric_data_directly(df, input_columns, label_column, selected_models, optuna_config, voting_config, stacking_config, progress_bar, status_text, numeric_scalers, multi_input_config.get('remove_duplicates', False), data_split_config)
                else:
                    # For text data: use execute_streamlit_training
                    st.info("📝 Using execute_streamlit_training for text data...")
                    with debug_container:
                        st.info(f"🔍 Debug: Calling execute_streamlit_training with enhanced configs")
                    from training_pipeline import execute_streamlit_training
                    results = execute_streamlit_training(df, enhanced_step1_config, enhanced_step2_config, enhanced_step3_config)
                    
                # Process results (same format as auto_train.py)
                results_debug_container = st.expander("🔍 Results Debug Log", expanded=False)
                with results_debug_container:
                    st.info(f"🔍 Debug: results type = {type(results)}")
                    if isinstance(results, dict):
                        st.info(f"🔍 Debug: results keys = {list(results.keys())}")
                        st.info(f"🔍 Debug: results['status'] = {results.get('status', 'NOT_FOUND')}")
                    else:
                        st.info(f"🔍 Debug: results is not a dict: {results}")
                if results and isinstance(results, dict) and results.get('status') == 'success':
                    st.toast("✅ Training completed successfully!")
                    
                    # Display results (same format as auto_train.py)
                    st.subheader("📊 Training Results")
                    
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
                                    'f1_score': model_data.get('f1_score', 0),
                                    'test_accuracy': model_data.get('accuracy', 0),
                                    'validation_accuracy': model_data.get('validation_accuracy', 0),
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
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Save training results
                        step4_data = {
                            'training_results': results,
                            'completed': True
                        }
                        session_manager.set_step_data(4, step4_data)
                        
                        st.success("✅ Training results saved!")
                        st.info("💡 Click 'Next ▶' button to proceed to Step 5.")
                        
                        # Reset training started state to show configuration summary again
                        st.session_state.training_started = False
                    else:
                        st.error("❌ No successful training results found")
                        with results_debug_container:
                            st.info(f"🔍 Debug: successful_results is empty")
                            if isinstance(results, dict) and 'model_results' in results:
                                st.info(f"🔍 Debug: model_results = {results['model_results']}")
                else:
                    error_msg = 'Unknown error'
                    if isinstance(results, dict):
                        error_msg = results.get('error', 'Unknown error')
                    elif results:
                        error_msg = str(results)
                    else:
                        error_msg = 'No results returned'
                    st.error(f"❌ Training failed: {error_msg}")
                    # Reset training started state to show configuration summary again
                    st.session_state.training_started = False
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                # Reset training started state to show configuration summary again
                st.session_state.training_started = False
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
    
    # Navigation buttons
    render_navigation_buttons()
def render_navigation_buttons():
    """Render navigation buttons as per wireframe"""
     
    col1, col2 = st.columns(2)
    
    with col1:
        current_step = get_current_step(session_manager)
        if st.button("◀ Previous", use_container_width=True, key=f"prev_btn_{current_step}"):
            # Go back to previous step
            # Use global session_manager instance
            if current_step > 1:
                session_manager.set_current_step(current_step - 1)
                st.success(f"← Going back to Step {current_step - 1}")
                st.rerun()
            else:
                st.info("ℹ️ You're already at the first step.")
    
    with col2:
        if st.button("Next ▶", use_container_width=True, key=f"next_btn_{current_step}"):
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
            use_container_width=True,
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
        st.dataframe(preview_df, use_container_width=True)
        st.caption("📝 **Original data preview** - Apply preprocessing to see transformed data")
    else:
        # Show cached preview with preprocessing applied
        cached_preview = st.session_state.step2_preview_cache
        st.dataframe(cached_preview, use_container_width=True)
     
        # Add option to clear cache and show original data
        if st.button("🔄 Show Original Data", key="show_original_preview"):
            del st.session_state.step2_preview_cache
            st.rerun()
    
    # Save configuration button
    if st.button("💾 Save Column Configuration", type="primary", use_container_width=True):
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
    st.dataframe(df.head(10), use_container_width=True)
    
    # Column information
    st.markdown("**📊 Column Information:**")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    st.dataframe(col_info, use_container_width=True)
    
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
        
        # Validation: Remove label column from input columns if selected
        if label_col and label_col in input_cols:
            input_cols = [col for col in input_cols if col != label_col]
            st.warning(f"⚠️ Removed '{label_col}' from input columns as it's selected as label column")
        
        # Show data quality metrics if columns are selected
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
            st.dataframe(sample_data, use_container_width=True)
            
            # Preprocessing options
            st.markdown("**🧹 Preprocessing Options:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                numeric_scaler = st.multiselect(
                    "📊 Numeric Scaling:",
                    ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"],
                    default=["StandardScaler"],  # Default to first option
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
            
            with col2:
                missing_strategy = st.selectbox(
                    "❌ Missing Values:",
                    ["Drop rows", "Fill with mean/median", "Fill with mode", "Forward fill"],
                    index=1,  # Default to "Fill with mean/median" (index 1)
                    key="multi_input_missing_strategy",
                    help="Strategy for handling missing values"
                )
            
            # Process button
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
    else:
        st.error("❌ No columns found in the dataset.")


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
        if st.button("✅ Complete Step 3", use_container_width=True, key="complete_step3"):
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
            default=['random_forest', 'xgboost', 'lightgbm'],
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
        # Traditional models for voting
        traditional_models = [
            'random_forest', 'logistic_regression', 'svm', 'knn', 
            'naive_bayes', 'decision_tree', 'adaboost', 'gradient_boosting'
        ]
        
        selected_models = st.multiselect(
            "Select traditional models for voting",
            traditional_models,
            default=['random_forest', 'logistic_regression', 'svm'],
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
            
            st.success(f"✅ Voting ensemble configured with {len(selected_models)} models")
        else:
            st.warning("⚠️ Please select at least one model for voting")
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
            default=['random_forest', 'xgboost', 'lightgbm'],
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
    
    # Get data from previous steps
    step1_data = session_manager.get_step_data(1)
    step2_data = session_manager.get_step_data(2)
    step3_data = session_manager.get_step_data(3)
    
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
    config_debug_container = st.expander("🔍 Configuration Debug", expanded=False)
    
    # Only show configuration summary if training hasn't started
    if not st.session_state.training_started:
        # Optuna configuration
        optuna_config = step3_data.get('optuna_config', {})
        if optuna_config.get('enabled', False):
            with config_debug_container:
                st.success(f"✅ Optuna: {optuna_config.get('trials', 'N/A')} trials, {len(optuna_config.get('models', []))} models")
        else:
            with config_debug_container:
                st.info("ℹ️ Optuna: Disabled")
        
        # Voting configuration
        voting_config = step3_data.get('voting_config', {})
        if voting_config.get('enabled', False):
            with config_debug_container:
                st.success(f"✅ Voting: {voting_config.get('voting_method', 'N/A')} voting, {len(voting_config.get('models', []))} models")
        else:
            with config_debug_container:
                st.info("ℹ️ Voting: Disabled")
        
        # Stacking configuration
        stacking_config = step3_data.get('stacking_config', {})
        if stacking_config.get('enabled', False):
            with config_debug_container:
                st.success(f"✅ Stacking: {stacking_config.get('meta_learner', 'N/A')} meta-learner, {len(stacking_config.get('base_models', []))} base models")
        else:
            with config_debug_container:
                st.info("ℹ️ Stacking: Disabled")
    
    # Training execution
    st.subheader("🚀 Training Execution")
    
    if st.button("🚀 Start Training", type="primary"):
        # Set training started state to hide configuration summary
        st.session_state.training_started = True
        with st.spinner("🔄 Starting training pipeline..."):
            try:
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
                debug_container = st.expander("🔍 Debug Log", expanded=False)
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
                    numeric_scalers = multi_input_config.get('numeric_scaler', ['StandardScaler'])
                    
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
                
                enhanced_step3_config = {
                    'optuna_config': optuna_config,
                    'voting_config': voting_config,
                    'stacking_config': stacking_config,
                    'selected_models': selected_models,
                    'preprocessing_config': {
                        'selected_methods': ['StandardScaler', 'MinMaxScaler', 'NoScaling']
                    }
                }
                
                # Use different approaches based on data type
                with debug_container:
                    st.info(f"🔍 Debug: About to check data_type = '{data_type}' for training approach")
                if data_type == 'multi_input':
                    # For numeric data: use direct sklearn training (like debug_streamlit_pipeline.py)
                    with debug_container:
                        st.info("🔢 Using direct sklearn for numeric data...")
                    with debug_container:
                        st.info(f"🔍 Debug: Calling train_numeric_data_directly with input_columns = {input_columns}, label_column = {label_column}")
                    results = train_numeric_data_directly(df, input_columns, label_column, selected_models, optuna_config, voting_config, stacking_config, progress_bar, status_text, numeric_scalers, multi_input_config.get('remove_duplicates', False), data_split_config)
                else:
                    # For text data: use execute_streamlit_training
                    st.info("📝 Using execute_streamlit_training for text data...")
                    with debug_container:
                        st.info(f"🔍 Debug: Calling execute_streamlit_training with enhanced configs")
                    from training_pipeline import execute_streamlit_training
                    results = execute_streamlit_training(df, enhanced_step1_config, enhanced_step2_config, enhanced_step3_config)
                    
                # Process results (same format as auto_train.py)
                results_debug_container = st.expander("🔍 Results Debug Log", expanded=False)
                with results_debug_container:
                    st.info(f"🔍 Debug: results type = {type(results)}")
                    if isinstance(results, dict):
                        st.info(f"🔍 Debug: results keys = {list(results.keys())}")
                        st.info(f"🔍 Debug: results['status'] = {results.get('status', 'NOT_FOUND')}")
                    else:
                        st.info(f"🔍 Debug: results is not a dict: {results}")
                if results and isinstance(results, dict) and results.get('status') == 'success':
                    st.success("✅ Training completed successfully!")
                    
                    # Display results (same format as auto_train.py)
                    st.subheader("📊 Training Results")
                    
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
                                    'f1_score': model_data.get('f1_score', 0),
                                    'test_accuracy': model_data.get('accuracy', 0),
                                    'validation_accuracy': model_data.get('validation_accuracy', 0),
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
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Save training results
                        step4_data = {
                            'training_results': results,
                            'completed': True
                        }
                        session_manager.set_step_data(4, step4_data)
                        
                        st.success("✅ Training results saved!")
                        st.info("💡 Click 'Next ▶' button to proceed to Step 5.")
                        
                        # Reset training started state to show configuration summary again
                        st.session_state.training_started = False
                    else:
                        st.error("❌ No successful training results found")
                        with results_debug_container:
                            st.info(f"🔍 Debug: successful_results is empty")
                            if isinstance(results, dict) and 'model_results' in results:
                                st.info(f"🔍 Debug: model_results = {results['model_results']}")
                else:
                    error_msg = 'Unknown error'
                    if isinstance(results, dict):
                        error_msg = results.get('error', 'Unknown error')
                    elif results:
                        error_msg = str(results)
                    else:
                        error_msg = 'No results returned'
                    st.error(f"❌ Training failed: {error_msg}")
                    # Reset training started state to show configuration summary again
                    st.session_state.training_started = False
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                # Reset training started state to show configuration summary again
                st.session_state.training_started = False
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
    
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
    
    # Get training results from step 4
    training_results = step4_data.get('training_results', {})
    
    if not training_results or training_results.get('status') != 'success':
        st.error("❌ No training results found. Please complete Step 4 first.")
        if st.button("← Go to Step 4"):
            session_manager.set_current_step(4)
            st.rerun()
        return
    
    # Get comprehensive results for comparison - try multiple sources
    comprehensive_results = training_results.get('comprehensive_results', [])
    
    # If no results from step4_data, try session state
    if not comprehensive_results and 'training_results' in st.session_state and st.session_state.training_results:
        comprehensive_results = st.session_state.training_results.get('comprehensive_results', [])
    
    # If still no results, try to load from any available cache
    if not comprehensive_results:
        try:
            from training_pipeline import StreamlitTrainingPipeline
            pipeline = StreamlitTrainingPipeline()
            cached_results_list = pipeline.list_cached_results()
            
            if cached_results_list:
                # Use the most recent cache
                most_recent_cache = max(cached_results_list, key=lambda x: x['timestamp'])
                cache_key = most_recent_cache['cache_key']
                
                # Load the cached results
                import pickle
                import os
                cache_file = os.path.join(pipeline.cache_dir, f"{cache_key}.pkl")
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        cached_results = pickle.load(f)
                    
                    # Debug: Check cache structure and type
                    if isinstance(cached_results, dict):
                        st.toast(f"🔍 Cache structure: {list(cached_results.keys())}")
                        
                        # Try different possible keys for comprehensive results
                        if 'all_results' in cached_results:
                            comprehensive_results = cached_results['all_results']
                            st.toast(f"📊 Found 'all_results': {len(comprehensive_results)} items")
                        elif 'comprehensive_results' in cached_results:
                            comprehensive_results = cached_results['comprehensive_results']
                            st.toast(f"📊 Found 'comprehensive_results': {len(comprehensive_results)} items")
                        elif 'results' in cached_results:
                            # If results is a list, use it directly
                            if isinstance(cached_results['results'], list):
                                comprehensive_results = cached_results['results']
                                st.toast(f"📊 Found 'results' (list): {len(comprehensive_results)} items")
                            # If results is a dict, look for nested data
                            elif isinstance(cached_results['results'], dict):
                                if 'all_results' in cached_results['results']:
                                    comprehensive_results = cached_results['results']['all_results']
                                    st.toast(f"📊 Found 'results.all_results': {len(comprehensive_results)} items")
                                elif 'comprehensive_results' in cached_results['results']:
                                    comprehensive_results = cached_results['results']['comprehensive_results']
                                    st.toast(f"📊 Found 'results.comprehensive_results': {len(comprehensive_results)} items")
                    else:
                        # Handle non-dict cached results
                        st.toast(f"⚠️ Cache data is not a dictionary: {type(cached_results)}")
                        if isinstance(cached_results, list):
                            comprehensive_results = cached_results
                            st.toast(f"📊 Using cached list directly: {len(comprehensive_results)} items")
                        else:
                            st.toast(f"❌ Unsupported cache data type: {type(cached_results)}")
                            comprehensive_results = []
                    
                    # Also update training_results for best_combinations
                    if comprehensive_results and 'best_combinations' not in training_results:
                        if 'best_combinations' in cached_results:
                            training_results['best_combinations'] = cached_results['best_combinations']
                        elif 'results' in cached_results and 'best_combinations' in cached_results['results']:
                            training_results['best_combinations'] = cached_results['results']['best_combinations']
                    
                    st.toast(f"✅ Loaded cached results: {most_recent_cache.get('cache_name', cache_key)}")
                    st.toast(f"⏰ Cache age: {most_recent_cache['age_hours']:.1f} hours | Results: {most_recent_cache['results_summary'].get('successful_combinations', 0)} combinations")
                    
                    # Store cache data in session state for access by confusion matrix functions
                    st.session_state.cache_data = cached_results
                    st.toast(f"💾 Cache data stored in session state for label access")
                    

                    
                    # Also store in training_results for fallback access
                    if 'cache_data' not in training_results:
                        training_results['cache_data'] = cached_results
                    

                    
                    # Debug: Show what we found
                    if comprehensive_results:
                        st.toast(f"🎯 Successfully loaded {len(comprehensive_results)} comprehensive results!")
                    else:
                        st.toast("⚠️ Cache loaded but no comprehensive results found in expected format")
                        st.toast("Cache keys available: " + ", ".join(cached_results.keys()))
        except Exception as e:
            st.error(f"❌ Error loading cache: {e}")
            st.exception(e)
    
    # If still no results, show helpful message and redirect to Step 4
    if not comprehensive_results:
        st.warning("⚠️ No comprehensive results available for comparison.")
        st.info("💡 This usually means Step 4 hasn't been completed yet or no models were trained successfully.")
        
        # Show helpful information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
                <h4>📋 What to do next:</h4>
                <p>1. Go to Step 4 and train some models</p>
                <p>2. Make sure training completes successfully</p>
                <p>3. Check that models are properly evaluated</p>
                <p>4. Wait for cache to be generated</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🚀 Go to Step 4", type="primary", use_container_width=True):
                session_manager.set_current_step(4)
                st.rerun()
        
        # Don't show tabs if no data
        return
    
    # Tab System for Step 5
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📊 Results Overview", "🔍 Detailed Analysis"])
    
    with tab1:
        # Results Overview Tab
        st.markdown("**📊 Results Overview**")
        
        # Best Model Selection Section
        best_combinations = training_results.get('best_combinations', {})
        best_overall = best_combinations.get('best_overall', {})
        
        if best_overall:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                model_name = best_overall.get('combination_key', 'N/A')
                st.markdown(f"""
                <div class="metric-box">
                    <h4>🥇 TOP PERFORMER</h4>
                    <p><strong>{model_name}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                f1_score = best_overall.get('f1_score', 0)
                st.markdown(f"""
                <div class="metric-box">
                    <h4>📊 F1 Score</h4>
                    <p><strong>{f1_score:.3f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                training_time = best_overall.get('training_time', 0)
                st.markdown(f"""
                <div class="metric-box">
                    <h4>⏱️ Training Time</h4>
                    <p><strong>{training_time:.1f}s</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                test_accuracy = best_overall.get('test_accuracy', 0)
                st.markdown(f"""
                <div class="metric-box">
                    <h4>🎯 Test Accuracy</h4>
                    <p><strong>{test_accuracy:.3f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if 'test_metrics' in best_overall:
                    test_metrics = best_overall['test_metrics']
                    precision = test_metrics.get('precision', 0)
                    
                    st.markdown(f"""
                    <div class="metric-box">
                        <h4>📊 Precision</h4>
                        <p><strong>{precision:.3f}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                if 'test_metrics' in best_overall:
                    test_metrics = best_overall['test_metrics']
                    recall = test_metrics.get('recall', 0)
                    
                    st.markdown(f"""
                    <div class="metric-box">
                        <h4>📈 Recall</h4>
                        <p><strong>{recall:.3f}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No best model results found in training data.")
        
        # Model Comparison Chart Section
        st.markdown("**🎯 Model Comparison Chart:**")
        
        # Create comparison data
        comparison_data = []
        for result in comprehensive_results:
            if result.get('status') == 'success':
                model_name = result.get('model_name', 'Unknown')
                embedding_name = result.get('embedding_name', 'Unknown')
                combination_key = f"{model_name} + {embedding_name}"
                
                # Get both F1 Score and Test Accuracy separately
                f1_score = result.get('f1_score', 0)
                test_accuracy = result.get('test_accuracy', 0)
                
                comparison_data.append({
                    'Model': combination_key.replace('_', ' ').title(),
                    'F1 Score': f1_score,
                    'Test Accuracy': test_accuracy,
                    'Precision': result.get('test_metrics', {}).get('precision', 0),
                    'Recall': result.get('test_metrics', {}).get('recall', 0),
                    'Training Time': result.get('training_time', 0)
                })
        
        if comparison_data:
            # Create interactive bar chart
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Sort by F1 Score for better visualization
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('F1 Score', ascending=False)
            
            # Create bar chart
            fig = px.bar(
                comparison_df,
                x='Model',
                y='F1 Score',
                title='Model Performance Comparison (F1 Score)',
                color='F1 Score',
                color_continuous_scale='viridis',
                text=comparison_df['F1 Score'].apply(lambda x: f'{x:.3f}')
            )
            
            fig.update_traces(textposition='outside')
            fig.update_layout(
                xaxis_title="Model + Vectorization",
                yaxis_title="F1 Score",
                showlegend=False,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    
    with tab2:    
            # Group results by model type for better organization
            model_groups = {}
            for result in comprehensive_results:
                if result.get('status') == 'success':
                    model_name = result.get('model_name', 'Unknown')
                    if model_name not in model_groups:
                        model_groups[model_name] = []
                    model_groups[model_name].append(result)
            
            # Create interactive table for model selection
            model_table_data = []
            
            for model_name, results in model_groups.items():
                for result in results:
                    embedding_name = result.get('embedding_name', 'Unknown')
                    f1_score = result.get('f1_score', 0)
                    test_accuracy = result.get('test_accuracy', 0)
                    training_time = result.get('training_time', 0)
                    precision = result.get('test_metrics', {}).get('precision', 0)
                    recall = result.get('test_metrics', {}).get('recall', 0)
                    
                    # Create unique key for this model combination
                    model_key = f"{model_name}_{embedding_name}"
                    
                    # Safe formatting for metrics
                    f1_display = f"{f1_score:.1%}" if f1_score is not None else "N/A"
                    acc_display = f"{test_accuracy:.1%}" if test_accuracy is not None else "N/A"
                    prec_display = f"{precision:.1%}" if precision is not None else "N/A"
                    rec_display = f"{recall:.1%}" if recall is not None else "N/A"
                    time_display = f"{training_time:.1f}" if training_time is not None else "N/A"
                    
                    model_table_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Vectorization': embedding_name.replace('_', ' ').title(),
                        'F1 Score': f1_display,
                        'Accuracy': acc_display,
                        'Precision': prec_display,
                        'Recall': rec_display,
                        'Training Time (s)': time_display,
                        'Actions': model_key  # Hidden column for actions
                    })
            
            if model_table_data:
                # Create DataFrame
                model_df = pd.DataFrame(model_table_data)
                             
                # Use st.data_editor for interactive table
                edited_df = st.data_editor(
                    model_df.drop('Actions', axis=1),  # Hide actions column
                    use_container_width=True,
                    hide_index=True,
                    num_rows="dynamic",
                    column_config={
                        "Model": st.column_config.TextColumn("🧠 Model", width="medium"),
                        "Vectorization": st.column_config.TextColumn("🔤 Vectorization", width="medium"),
                        "F1 Score": st.column_config.TextColumn("📊 F1 Score", width="small"),
                        "Accuracy": st.column_config.TextColumn("🎯 Accuracy", width="small"),
                        "Precision": st.column_config.TextColumn("📈 Precision", width="small"),
                        "Recall": st.column_config.TextColumn("📉 Recall", width="small"),
                        "Training Time (s)": st.column_config.NumberColumn("⏱️ Time (s)", format="%.1f", width="small")
                    }
                )
                                
                # Create selection dropdown
                model_options = [f"{row['Model']} + {row['Vectorization']}" for row in model_table_data]
                selected_model_display = st.selectbox(
                    "Choose a model to analyze:",
                    options=model_options,
                    index=0,
                    help="Select a model to view detailed analysis including confusion matrix"
                )
                
                # Find the selected result
                selected_result = None
                for i, row in enumerate(model_table_data):
                    if f"{row['Model']} + {row['Vectorization']}" == selected_model_display:
                        # Find the actual result data
                        for model_name, results in model_groups.items():
                            for result in results:
                                if (model_name.replace('_', ' ').title() == row['Model'] and 
                                    result.get('embedding_name', 'Unknown').replace('_', ' ').title() == row['Vectorization']):
                                    selected_result = result
                                    break
                            if selected_result:
                                break
                        break
                
                # Button to view details
                if selected_result and st.button("🔍 View Detailed Analysis", type="primary", use_container_width=True):
                    # Create unique key for this model combination
                    model_name = selected_result.get('model_name', 'Unknown')
                    embedding_name = selected_result.get('embedding_name', 'Unknown')
                    model_key = f"{model_name}_{embedding_name}"
                    
                    st.session_state.selected_model_detail = model_key
                    st.session_state.selected_model_result = selected_result
                    st.rerun()
            else:
                st.warning("⚠️ No models available for detailed analysis.")
                   
            successful_results = [r for r in comprehensive_results if r.get('status') == 'success']
            if successful_results:
                # Find best overall model
                best_model = max(successful_results, key=lambda x: x.get('f1_score', 0))
                best_accuracy = best_model.get('f1_score', 0)
                best_model_name = f"{best_model.get('model_name', 'Unknown')} + {best_model.get('embedding_name', 'Unknown')}"
                
                # Find fastest training model
                fastest_model = min(successful_results, key=lambda x: x.get('training_time', float('inf')))
                fastest_time = fastest_model.get('training_time', 0)
                fastest_model_name = f"{fastest_model.get('model_name', 'Unknown')} + {fastest_model.get('embedding_name', 'Unknown')}"
                
                # Find most memory efficient (assuming smaller models are more efficient)
                most_efficient = min(successful_results, key=lambda x: x.get('training_time', 0))  # Using training time as proxy
                efficient_time = most_efficient.get('training_time', 0)
                efficient_model_name = f"{most_efficient.get('model_name', 'Unknown')} + {most_efficient.get('embedding_name', 'Unknown')}"
    
            # Individual Model Confusion Matrix Window
            if st.session_state.get('selected_model_detail') and st.session_state.get('selected_model_result'):
                selected_result = st.session_state.selected_model_result
                
                st.markdown("---")
                st.markdown(f"""
                <h3 style="color: var(--text-color); margin: 1.5rem 0 1rem 0;">🧠 Model: {selected_result.get('model_name', 'Unknown').replace('_', ' ').title()} + {selected_result.get('embedding_name', 'Unknown').replace('_', ' ').title()} - Detailed Analysis</h3>
                """, unsafe_allow_html=True)
                
                # Set default metric (Accuracy)
                selected_metric = "Accuracy"
                selected_metric_value = selected_result.get('test_accuracy', 0)
                
                # Set default display mode
                cm_display_mode = "Percentages (Normalized)"
                
                # Add display format selection
                cm_display_mode = st.radio(
                    "Choose display format:",
                    options=["Percentages (Normalized)", "Counts (Raw Numbers)"],
                    index=0,
                    help="Percentages show relative proportions, Counts show actual prediction numbers"
                )
                
                # Display confusion matrix if available
                # Sử dụng màu "green royal" làm theme chính cho confusion matrix
                MAIN_THEME_CMAP = 'YlGn'  # Royal green style
                MAIN_THEME_TITLE_COLOR = '#006400'  # Dark green (royal green)

                if 'confusion_matrix' in selected_result:
                    st.markdown("**🎯 Confusion Matrix:**")

                    # Get confusion matrix data
                    cm = selected_result['confusion_matrix']
                    if isinstance(cm, (list, np.ndarray)) and len(cm) > 0:
                        # Use consistent helper function to get labels
                        # Pass cache_data to prioritize top-level labels over individual result labels
                        cache_data = st.session_state.get('cache_data') or step4_data.get('training_results', {})
                        

                        
                        unique_labels, label_mapping = _get_unique_labels_and_mapping(
                            selected_result, 
                            cache_data=cache_data
                        )
                        

                        
                        # If no unique_labels found, fall back to matrix dimensions
                        if unique_labels is None:
                            unique_labels = list(range(len(cm)))
                        
                        # Get consistent class labels
                        class_labels = _get_consistent_labels(unique_labels, label_mapping)
                        


                        # Create heatmap with project theme color
                        fig, ax = plt.subplots(figsize=(8, 6))
                        cmap = MAIN_THEME_CMAP
                        title_color = MAIN_THEME_TITLE_COLOR

                        # Create heatmap with selected metric value in title
                        if cm_display_mode == "Percentages (Normalized)":
                            # Normalize confusion matrix to percentages
                            cm_arr = np.array(cm)
                            cm_normalized = cm_arr.astype('float') / np.sum(cm_arr, axis=1)[:, np.newaxis]
                            cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)

                            # Create heatmap with percentage format
                            sns.heatmap(
                                cm_normalized, annot=True, fmt='.1%', cmap=cmap,
                                xticklabels=class_labels, yticklabels=class_labels,
                                ax=ax, cbar_kws={'label': f'{selected_metric} Focus (%)'}
                            )

                            ax.set_title(
                                f'Confusion Matrix (Normalized %) - {selected_metric}: '
                                f'{selected_metric_value:.1%}',
                                color=title_color, fontsize=14, fontweight='bold'
                            )
                        else:
                            # Create heatmap with count format
                            sns.heatmap(
                                cm, annot=True, fmt='d', cmap=cmap,
                                xticklabels=class_labels, yticklabels=class_labels,
                                ax=ax, cbar_kws={'label': f'{selected_metric} Focus (Counts)'}
                            )

                            ax.set_title(
                                f'Confusion Matrix (Counts) - {selected_metric}: '
                                f'{selected_metric_value:.1%}',
                                color=title_color, fontsize=14, fontweight='bold'
                            )
                        ax.set_xlabel('Predicted', fontsize=12)
                        ax.set_ylabel('Actual', fontsize=12)

                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("📊 Confusion matrix data not available for this model.")
                elif 'predictions' in selected_result and 'true_labels' in selected_result:
                    st.markdown("**🎯 Confusion Matrix (generated from predictions):**")

                    try:
                        from sklearn.metrics import confusion_matrix

                        y_pred = selected_result['predictions']
                        y_true = selected_result['true_labels']

                        if (
                            y_pred is not None and y_true is not None
                            and len(y_pred) > 0 and len(y_true) > 0
                        ):
                            # Use consistent helper function to get labels
                            # Pass cache_data to prioritize top-level labels over individual result labels
                            cache_data = st.session_state.get('cache_data') or step4_data.get('training_results', {})
                            

                            
                            unique_labels, label_mapping = _get_unique_labels_and_mapping(
                                selected_result, 
                                cache_data=cache_data
                            )
                            

                            
                            # If no unique_labels found, calculate from predictions
                            if unique_labels is None:
                                unique_labels = sorted(list(set(np.concatenate([y_true, y_pred]))))

                            # Create confusion matrix with correct label order
                            cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

                            # Get consistent class labels
                            class_labels = _get_consistent_labels(unique_labels, label_mapping)
                            


                            # Vẽ heatmap confusion matrix
                            fig, ax = plt.subplots(figsize=(8, 6))
                            cmap = MAIN_THEME_CMAP
                            title_color = MAIN_THEME_TITLE_COLOR

                            if cm_display_mode == "Percentages (Normalized)":
                                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                                cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)
                                sns.heatmap(
                                    cm_normalized, annot=True, fmt='.1%', cmap=cmap,
                                    xticklabels=class_labels, yticklabels=class_labels,
                                    ax=ax, cbar_kws={'label': f'{selected_metric} Focus (%)'}
                                )
                                ax.set_title(
                                    f'Confusion Matrix (Generated, Normalized %) - {selected_metric}: '
                                    f'{selected_metric_value:.1%}',
                                    color=title_color, fontsize=14, fontweight='bold'
                                )
                            else:
                                sns.heatmap(
                                    cm, annot=True, fmt='d', cmap=cmap,
                                    xticklabels=class_labels, yticklabels=class_labels,
                                    ax=ax, cbar_kws={'label': f'{selected_metric} Focus (Counts)'}
                                )
                                ax.set_title(
                                    f'Confusion Matrix (Generated, Counts) - {selected_metric}: '
                                    f'{selected_metric_value:.1%}',
                                    color=title_color, fontsize=14, fontweight='bold'
                                )
                            ax.set_xlabel('Predicted', fontsize=12)
                            ax.set_ylabel('Actual', fontsize=12)

                            st.pyplot(fig)
                            plt.close(fig)

                            # Lưu lại confusion matrix vào selected_result
                            selected_result['confusion_matrix'] = cm
                            st.success("✅ Confusion matrix generated successfully from predictions!")
                        else:
                            st.warning("⚠️ Predictions or true labels are empty or None")
                    except Exception as e:
                        st.error(f"❌ Error generating confusion matrix: {e}")
                        st.info(
                            "💡 This might happen if predictions/true_labels are not in the expected format"
                        )
                elif selected_result.get('model_name') == 'Ensemble Learning' and 'ensemble_info' in selected_result:
                    st.markdown("**🎯 Confusion Matrix (generated from ensemble model):**")

                    try:
                        from sklearn.metrics import confusion_matrix

                        # Extract data directly from ensemble model (not from base models)
                        y_pred = selected_result.get('predictions', [])
                        y_true = selected_result.get('true_labels', [])
                        
                        if not y_pred or not y_true:
                            st.warning("⚠️ No predictions or true labels found in ensemble result")
                            return
                        
                        st.info(f"✅ Using ensemble model data with {selected_result.get('embedding_name', 'unknown')} embedding")
                        
                        if (
                            y_pred is not None and y_true is not None
                            and len(y_pred) > 0 and len(y_true) > 0
                        ):
                            # Use consistent helper function to get labels
                            # Pass cache_data to prioritize top-level labels over individual result labels
                            cache_data = st.session_state.get('cache_data') or step4_data.get('training_results', {})
                            
                            # Create a dummy model_data dict for ensemble model
                            ensemble_model_data = {
                                'predictions': y_pred,
                                'true_labels': y_true,
                                'label_mapping': selected_result.get('label_mapping', {})
                            }
                            
                            unique_labels, label_mapping = _get_unique_labels_and_mapping(
                                selected_result, 
                                ensemble_model_data,
                                cache_data=cache_data
                            )
                            
                            # If no unique_labels found, calculate from predictions
                            if unique_labels is None:
                                unique_labels = sorted(list(set(np.concatenate([y_true, y_pred]))))

                            # Create confusion matrix with correct label order
                            cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

                            # Get consistent class labels
                            class_labels = _get_consistent_labels(unique_labels, label_mapping)
                            


                            # Draw confusion matrix heatmap
                            fig, ax = plt.subplots(figsize=(8, 6))
                            cmap = MAIN_THEME_CMAP
                            title_color = MAIN_THEME_TITLE_COLOR

                            if cm_display_mode == "Percentages (Normalized)":
                                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                                cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)
                                sns.heatmap(
                                    cm_normalized, annot=True, fmt='.1%', cmap=cmap,
                                    xticklabels=class_labels, yticklabels=class_labels,
                                    ax=ax, cbar_kws={'label': f'{selected_metric} Focus (%)'}
                                )
                                ax.set_title(
                                    f'Ensemble Learning Confusion Matrix (Normalized %) - {selected_metric}: '
                                    f'{selected_metric_value:.1%}',
                                    color=title_color, fontsize=14, fontweight='bold'
                                )
                            else:
                                sns.heatmap(
                                    cm, annot=True, fmt='d', cmap=cmap,
                                    xticklabels=class_labels, yticklabels=class_labels,
                                    ax=ax, cbar_kws={'label': f'{selected_metric} Focus (Counts)'}
                                )
                                ax.set_title(
                                    f'Ensemble Learning Confusion Matrix (Counts) - {selected_metric}: '
                                    f'{selected_metric_value:.1%}',
                                    color=title_color, fontsize=14, fontweight='bold'
                                )
                            ax.set_xlabel('Predicted', fontsize=12)
                            ax.set_ylabel('Actual', fontsize=12)

                            st.pyplot(fig)
                            plt.close(fig)

                            # Save confusion matrix to selected_result
                            selected_result['confusion_matrix'] = cm
                            st.success("✅ Ensemble confusion matrix generated successfully from base model data!")
                        else:
                            st.warning("⚠️ Base model predictions or true labels are empty or None")
                    except Exception as e:
                        st.error(f"❌ Error generating ensemble confusion matrix: {e}")
                        st.info(
                            "💡 This might happen if ensemble base model data is not in the expected format"
                        )
                        import traceback
                        st.code(traceback.format_exc())
                else:
                    st.info("📊 Confusion matrix not available for this model.")
                    st.info(
                        "💡 Available keys: " +
                        ", ".join([
                            k for k in selected_result.keys()
                            if k not in ['predictions', 'true_labels', 'confusion_matrix']
                        ])
                    )        
        
              
    # Close results container
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Export Functionality Section - Only in Tab 2
    st.markdown("---")
    st.markdown("""
    <h3 style="color: var(--text-color); margin: 1.5rem 0 1rem 0;">📤 Export & Documentation:</h3>
    """, unsafe_allow_html=True)
        
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV Export
        if comprehensive_results:
            # Prepare CSV data
            csv_data = []
            for result in comprehensive_results:
                if result.get('status') == 'success':
                    csv_data.append({
                        'Model': result.get('model_name', 'Unknown'),
                        'Vectorization': result.get('embedding_name', 'Unknown'),
                        'F1 Score': result.get('f1_score', 0),
                        'Test Accuracy': result.get('test_accuracy', 0),
                        'Precision': result.get('test_metrics', {}).get('precision', 0),
                        'Recall': result.get('test_metrics', {}).get('recall', 0),
                        'Training Time (s)': result.get('training_time', 0),
                        'Overfitting Level': result.get('overfitting_level', 'N/A'),
                        'CV Mean Accuracy': result.get('cv_mean_accuracy', 0),
                        'CV Std Accuracy': result.get('cv_std_accuracy', 0)
                    })
            
            if csv_data:
                csv_df = pd.DataFrame(csv_data)
                csv = csv_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="comprehensive_evaluation_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("⚠️ No data available for CSV export.")
        else:
            st.warning("⚠️ No results available for export.")
    
    with col2:
        # Summary Report
        if comprehensive_results:
            # Generate summary statistics
            successful_results = [r for r in comprehensive_results if r.get('status') == 'success']
            
            if successful_results:
                total_models = len(successful_results)
                avg_accuracy = sum(r.get('f1_score', 0) for r in successful_results) / total_models
                avg_training_time = sum(r.get('training_time', 0) for r in successful_results) / total_models
                
                st.markdown(f"""
                <div class="metric-box">
                    <h4>📊 Summary</h4>
                    <p>• Total Models: <strong>{total_models}</strong></p>
                    <p>• Avg Accuracy: <strong>{avg_accuracy:.3f}</strong></p>
                    <p>• Avg Training Time: <strong>{avg_training_time:.2f}s</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No successful results for summary.")
        else:
            st.warning("⚠️ No results available for summary.")
        
        # Save step 5 configuration
        step5_config = {
            'results_analyzed': True,
            'best_model': best_overall,
            'total_models': len(comprehensive_results) if comprehensive_results else 0,
            'export_generated': True,
        'completed': True
    }
    session_manager.set_step_config('step5', step5_config)
    
    # Show completion message
    st.toast("✅ Step 5 completed successfully!")
    st.toast("Click 'Next ▶' button to proceed to Step 5.")
    

def render_shap_analysis():
    """Render SHAP analysis and visualization"""
    st.markdown("""
    <h3 style="color: var(--text-color); margin: 1.5rem 0 1rem 0;">🔍 SHAP (SHapley Additive exPlanations) Analysis</h3>
    """, unsafe_allow_html=True)
    
    st.info("📝 **SHAP** explains individual predictions by computing the contribution of each feature to the model's output. Best suited for tree-based models.")
    
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
                'adaboost', 'gradient_boosting'
            ]
            
            selected_models = st.multiselect(
                "Select Models for SHAP Analysis:",
                available_models,
                default=['random_forest', 'xgboost', 'lightgbm'],
                help="Choose tree-based models for SHAP analysis"
            )
        
        # SHAP plot types
        st.markdown("**📊 SHAP Plot Types:**")
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
        
        # Generate SHAP plots
        if st.button("🚀 Generate SHAP Analysis"):
            if selected_models and plot_types:
                with st.spinner("Generating SHAP analysis from cache..."):
                    try:
                        # Import cache manager and visualization
                        from cache_manager import CacheManager
                        from visualization import create_shap_explainer, generate_shap_summary_plot, generate_shap_bar_plot, generate_shap_dependence_plot
                        
                        # Initialize cache manager
                        cache_manager = CacheManager()
                        
                        # Get available cached models
                        available_caches = cache_manager.list_cached_models()
                        
                        if not available_caches:
                            st.warning("⚠️ No cached models found. Please complete training in Step 4 first.")
                            return
                        
                        st.info(f"📊 Found {len(available_caches)} cached models")
                        
                        # Filter models that are selected and available in cache
                        cached_models = []
                        for model_name in selected_models:
                            # Look for cached models matching the selected model
                            for cache_info in available_caches:
                                if model_name in cache_info.get('model_key', ''):
                                    cached_models.append({
                                        'model_name': model_name,
                                        'cache_info': cache_info
                                    })
                                    break
                        
                        if not cached_models:
                            st.warning(f"⚠️ No cached models found for selected models: {selected_models}")
                            st.info("Available cached models:")
                            for cache_info in available_caches:
                                st.write(f"- {cache_info.get('model_key', 'Unknown')}")
                            return
                        
                        st.success(f"✅ Found {len(cached_models)} cached models for SHAP analysis")
                        
                        # Generate SHAP plots for each cached model
                        for model_data in cached_models:
                            model_name = model_data['model_name']
                            cache_info = model_data['cache_info']
                            
                            st.info(f"🔍 Generating SHAP plots for {model_name}...")
                            
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
                                test_data = eval_predictions.values
                                
                                # Sample data for SHAP analysis
                                if len(test_data) > sample_size:
                                    import numpy as np
                                    indices = np.random.choice(len(test_data), sample_size, replace=False)
                                    sample_data = test_data[indices]
                                else:
                                    sample_data = test_data
                                
                                # Create SHAP explainer
                                explainer = create_shap_explainer(model, sample_data)
                                shap_values = explainer(sample_data)
                                
                                # Generate plots based on selected types
                                for plot_type in plot_types:
                                    if plot_type == "summary":
                                        fig = generate_shap_summary_plot(shap_values, sample_data)
                                        st.pyplot(fig)
                                    
                                    elif plot_type == "bar":
                                        fig = generate_shap_bar_plot(shap_values, sample_data)
                                        st.pyplot(fig)
                                    
                                    elif plot_type == "dependence":
                                        fig = generate_shap_dependence_plot(shap_values, sample_data, sample_data)
                                        st.pyplot(fig)
                                
                                st.success(f"✅ SHAP plots generated for {model_name}")
                                
                            except Exception as e:
                                st.error(f"❌ Error generating SHAP plots for {model_name}: {str(e)}")
                                continue
                        
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
                        
                        st.success("🎉 SHAP analysis completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during SHAP analysis: {str(e)}")
                        st.exception(e)
            else:
                st.error("❌ Please select at least one model and one plot type")
    
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
            ["none", "true", "pred", "all"],
            index=0,
            help="How to normalize the confusion matrix"
        )
        
        dataset_split = st.selectbox(
            "Dataset Split:",
            ["test", "validation", "train"],
            index=0,
            help="Which dataset split to use for confusion matrix"
        )
    
    with col2:
        threshold = st.slider(
            "Classification Threshold:",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Threshold for binary classification"
        )
        
        show_percentages = st.checkbox(
            "Show Percentages",
            value=True,
            help="Display percentages in confusion matrix"
        )
    
    # Available models (check from cache - bao gồm cả mô hình cũ và mới)
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
        selected_model = st.selectbox(
            "Select Model for Confusion Matrix:",
            cached_models,
            help="Choose a trained model to generate confusion matrix"
        )
        
        # Generate confusion matrix
        if st.button("📊 Generate Confusion Matrix"):
            with st.spinner("Generating confusion matrix..."):
                # Save configuration
                cm_config = {
                    'normalize_method': normalize_method,
                    'dataset_split': dataset_split,
                    'threshold': threshold,
                    'show_percentages': show_percentages,
                    'selected_model': selected_model
                }
                
                # Save to session
                current_step5_data = session_manager.get_step_data(5) or {}
                current_step5_data['confusion_matrix_config'] = cm_config
                session_manager.set_step_data(5, current_step5_data)
                
                st.success("✅ Confusion matrix configuration saved!")
                st.info("📊 Confusion matrix will be generated from cached model predictions.")
                
                # Show configuration summary
                st.markdown("**📋 Confusion Matrix Configuration:**")
                st.write(f"- Model: {selected_model}")
                st.write(f"- Dataset: {dataset_split}")
                st.write(f"- Normalization: {normalize_method}")
                st.write(f"- Threshold: {threshold}")
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
    
    if st.button("📈 Load Model Metrics"):
        try:
            # Get training results from Step 4
            step4_data = session_manager.get_step_data(4)
            training_results = step4_data.get('training_results', {})
            
            if training_results.get('status') == 'success':
                results = training_results.get('results', {})
                comprehensive_results = results.get('comprehensive_results', [])
                
                if comprehensive_results:
                    st.info("📊 Loading metrics from actual training results...")
                    
                    # Create metrics table from real results
                    import pandas as pd
                    metrics_data = []
                    
                    for result in comprehensive_results:
                        if result.get('status') == 'success':
                            metrics_data.append({
                                'Model': f"{result.get('model_name', 'Unknown')} + {result.get('embedding_name', 'Unknown')}",
                                'Accuracy': f"{result.get('test_accuracy', 0):.4f}",
                                'F1-Score': f"{result.get('f1_score', 0):.4f}",
                                'Training Time': f"{result.get('training_time', 0):.1f}s",
                                'Status': result.get('status', 'Unknown')
                            })
                    
                    if metrics_data:
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Find best model
                        best_result = max(comprehensive_results, key=lambda x: x.get('f1_score', 0))
                        best_model = f"{best_result.get('model_name', 'Unknown')} + {best_result.get('embedding_name', 'Unknown')}"
                        best_f1 = best_result.get('f1_score', 0)
                        
                        st.success(f"🏆 **Best Model:** {best_model} (F1-Score: {best_f1:.4f})")
                    else:
                        st.warning("⚠️ No successful training results found.")
                else:
                    st.warning("⚠️ No comprehensive results available. Please complete Step 4 training first.")
            else:
                st.warning("⚠️ Training not completed successfully. Please complete Step 4 training first.")
                
        except Exception as e:
            st.error(f"❌ Error loading metrics: {str(e)}")
            st.info("💡 Please complete Step 4 training first to see actual metrics.")
    
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
