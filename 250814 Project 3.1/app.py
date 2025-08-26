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
import sys
import os
import time

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wizard_ui.session_manager import SessionManager
from training_pipeline import StreamlitTrainingPipeline


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
    page_title="🤖 Topic Modeling - Auto Classifier",
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
        transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application following wireframe design"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Topic Modeling - Auto Classifier</h1>
        <p>Intelligent Text Classification with Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session manager
    session_manager = SessionManager()
    current_step = get_current_step(session_manager)
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
        else:
            render_step1_wireframe()  # Default to step 1
    
    with col2:
        render_sidebar()

def render_step1_wireframe():
    """Render Step 1 exactly as per wireframe design"""
    
    # Step title - simplified without big container
    st.markdown("""
    <h2 style="text-align: left; color: var(--text-color); margin: 2rem 0 1rem 0; font-size: 1.8rem;">
        📍 STEP 1/6: Dataset Selection & Upload
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
                        
                        # Store in session
                        session_manager = SessionManager()
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
                        
                        # Show file preview
                        show_file_preview(df, file_extension)
                        
                    else:
                        st.error("❌ File not found. Please check the path and try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
        
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

                    # Store in session
                    session_manager = SessionManager()
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

                    # Show file preview
                    show_file_preview(df, file_extension)
                except Exception as e:
                    st.toast(f"❌ Error loading sample dataset: {str(e)}")
    
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
            session_manager = SessionManager()
            step1_data = session_manager.get_step_data(1)
            
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
                if dataset_size < 1000:
                    st.warning(f"⚠️ Small dataset detected ({dataset_size:,} rows). "
                              f"Consider using all available data for better model performance.")
                elif dataset_size == 1000:
                    st.info(f"ℹ️ Dataset size: {dataset_size:,} rows. "
                           f"You can sample from 100 to 1000 rows.")
            else:
                st.warning("⚠️ Please load a dataset first to configure sampling.")
        
        with col2:
            # Sampling strategy
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
        
        # Show file preview
        show_file_preview(df, file_extension)
        
        # Store in session
        session_manager = SessionManager()
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


def show_file_preview(df, file_extension):
    """Show file preview and info for both uploaded and path-based files"""
    
    # Data preview box
    st.subheader("📊 Data Preview (First 5 rows)")
    
    st.dataframe(df.head(5), use_container_width=True)
    
    # Get sampling configuration for display
    session_manager = SessionManager()
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

def render_navigation_buttons():
    """Render navigation buttons as per wireframe"""
     
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("◀ Previous", use_container_width=True):
            # Go back to previous step
            session_manager = SessionManager()
            current_step = get_current_step(session_manager)
            if current_step > 1:
                session_manager.set_current_step(current_step - 1)
                st.success(f"← Going back to Step {current_step - 1}")
                st.rerun()
            else:
                st.info("ℹ️ You're already at the first step.")
    
    with col2:
        if st.button("Next ▶", use_container_width=True):
            session_manager = SessionManager()
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
                step_data = session_manager.get_step_data(5)
                if step_data and step_data.get('completed', False):
                    st.success("✅ Step 5 completed! Moving to Step 6...")
                    # Move to step 6
                    session_manager.set_current_step(6)
                    st.rerun()
                else:
                    st.warning("⚠️ Please complete Step 5 first")
            elif current_step == 6:
                st.info("ℹ️ You're already at the last step.")
            else:
                st.info("ℹ️ This step is not yet implemented.")
    
    # Removed Skip to End button

def render_sidebar():
    """Render sidebar with progress tracking"""
    # Initialize session manager
    session_manager = SessionManager()
    
    # Current step info - Dynamic based on current step
    current_step = get_current_step(session_manager)
    step_names = [
        "Dataset Selection",
        "Column Selection & Preprocessing",
        "Model Configuration",
        "Training Execution",
        "Results Analysis",
        "Text Classification"
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
        "Results Analysis",
        "Text Classification"
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
            for step_num in range(1, 7):  # 6 steps total
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
    """Render Step 2 - Column Selection & Preprocessing"""
    
    # Step title
    st.markdown("""
    <h2 style="text-align: left; color: var(--text-color); margin: 2rem 0 1rem 0; font-size: 1.8rem;">
        📍 STEP 2/6: Column Selection & Preprocessing
    </h2>
    """, unsafe_allow_html=True)
    
    # Get data from step 1
    session_manager = SessionManager()
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
    
    # Navigation buttons
    render_navigation_buttons()


def render_step3_wireframe():
    """Render Step 3 - Model Configuration & Vectorization exactly as per wireframe design"""
    
    # Step title
    st.markdown("""
    <h2 style="text-align: left; color: var(--text-color); margin: 2rem 0 1rem 0; font-size: 1.8rem;">
        📍 STEP 3/6: Model Configuration & Vectorization
    </h2>
    """, unsafe_allow_html=True)
    
    # Get data from previous steps
    session_manager = SessionManager()
    step1_data = session_manager.get_step_data(1)
    step2_data = session_manager.get_step_data(2)
    
    if not step1_data or 'dataframe' not in step1_data:
        st.error("❌ No dataset found. Please complete Step 1 first.")
        if st.button("← Go to Step 1"):
            session_manager.set_current_step(1)
            st.success("← Going back to Step 1")
            st.rerun()
        return
    
    if not step2_data or not step2_data.get('completed', False):
        st.error("❌ Column selection not completed. Please complete Step 2 first.")
        if st.button("← Go to Step 2"):
            session_manager.set_current_step(2)
            st.success("← Going back to Step 2")
            st.rerun()
        return
    
    df = step1_data['dataframe']
    
    # Display dataset info from previous steps with sampling information
    sampling_config = step1_data.get('sampling_config', {})
    if sampling_config and sampling_config.get('num_samples'):
        num_samples = sampling_config['num_samples']
        strategy = sampling_config.get('sampling_strategy', 'Unknown')
        # Chỉ hiển thị số samples đã chọn
        if num_samples < df.shape[0]:
            dataset_display = f"{num_samples:,} samples"
        else:
            dataset_display = f"{df.shape[0]:,} samples"
    else:
        dataset_display = f"{df.shape[0]:,} samples"
    
    st.info(f"📊 **Dataset**: {dataset_display} × {df.shape[1]} columns | "
            f"**Text Column**: {step2_data.get('text_column', 'N/A')} | "
            f"**Label Column**: {step2_data.get('label_column', 'N/A')}")
    
    # Log sampling info
    print(f"📊 [STEP3] Dataset display - Original: {df.shape[0]:,}, Will use: {dataset_display}")
    
    # Data Split Configuration (Simplified: Only Test + Training)
    st.markdown("**📊 Data Split:**")
    col1, col2 = st.columns(2)
    
    with col1:
        test_split = st.slider(
            "Test Set:",
            min_value=15,
            max_value=30,
            value=20,
            step=5,
            help="Test set percentage (15-30%). Fixed for final evaluation."
        )
    
    with col2:
        # Training set (remaining percentage)
        training_split = 100 - test_split
        
        st.info(f"📊 **Training Set**: {training_split}% (for Cross-Validation)")
    
    # Calculate actual percentages
    final_test = test_split
    final_training = training_split
    
    # Display final split information
    st.info(f"📊 **Final Data Split**: Training: {final_training}% | Test: {final_test}%")
    
    # Cross-Validation Configuration
    st.markdown("**🔄 Cross-Validation:**")
    col1, col2 = st.columns(2)
    
    with col1:
        cv_folds = st.slider(
            "CV Folds:",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Number of cross-validation folds (3-10). Each fold uses ~{final_training//cv_folds}% of training data."
        )
        
        # Show cross-validation explanation
        if final_training > 0 and cv_folds > 0:
            fold_percentage = final_training / cv_folds
            st.info(f"📊 **Each CV Fold**: ~{fold_percentage:.1f}% of training data")
   
    with col2:
        random_state = st.number_input(
            "Random State:",
            min_value=0,
            max_value=9999,
            value=42,
            step=1,
            help="Random seed for reproducible results"
        )
        
    # Model Selection Section
    st.markdown("""
    <h3 style="color: var(--text-color); margin: 1.5rem 0 1rem 0;">🎯 Model Selection:</h3>
    """, unsafe_allow_html=True)
    
    # Get existing model selection from session
    existing_config = session_manager.get_step_data(3) or {}
    existing_models = existing_config.get('selected_models', [])
    
    # Model selection checkboxes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Supervised Models:**")
        knn_model = st.checkbox(
            "☑️ K-Nearest Neighbors (Supervised)",
            value="KNN" in existing_models,
            help="K-Nearest Neighbors classifier for text classification"
        )
        
        decision_tree = st.checkbox(
            "☑️ Decision Tree (Supervised)",
            value="Decision Tree" in existing_models,
            help="Decision Tree classifier with interpretable rules"
        )
        
        naive_bayes = st.checkbox(
            "☑️ Naive Bayes (Supervised)",
            value="Naive Bayes" in existing_models,
            help="Naive Bayes classifier for text classification"
        )
    
    with col2:
        st.markdown("**Unsupervised Models:**")
        kmeans_model = st.checkbox(
            "☑️ K-Means Clustering (Unsupervised)",
            value="K-Means" in existing_models,
            help="K-Means clustering for topic discovery"
        )
        
        st.markdown("**Advanced Models:**")
        svm_model = st.checkbox(
            "☑️ Support Vector Machine (Supervised)",
            value="SVM" in existing_models,
            help="SVM classifier with kernel methods"
        )
        
        logistic_regression = st.checkbox(
            "☑️ Logistic Regression (Supervised)",
            value="Logistic Regression" in existing_models,
            help="Logistic Regression classifier with multinomial support"
        )
        
        linear_svc = st.checkbox(
            "☑️ Linear SVC (Supervised)",
            value="Linear SVC" in existing_models,
            help="Linear Support Vector Classification"
        )
    
    # KNN Advanced Configuration Section (only show if KNN is selected)
    if knn_model:
        with st.expander("🎯 KNN Advanced Configuration", expanded=False):
            st.markdown("**🔍 KNN Parameter Optimization:**")
                     
            # Optimization Type Selection with Manual Option
            st.markdown("### 🎯 **Chọn phương pháp tối ưu KNN:**")
            
            col1, col2 = st.columns(2)
            with col1:
                knn_optimization_type = st.selectbox(
                    "🔍 **Optimization Strategy:**",
                    options=["Manual K Input", "Optimal K (Cosine Metric)", "Grid Search (All Parameters)"],
                    index=0,
                    help="Chọn cách thiết lập tham số KNN"
                )
            with col2:
                knn_vectorizer_type = st.selectbox(
                    "🧬 **Vectorization Method:**",
                    options=[
                        "Sentence Embeddings (Recommended)",
                        "TF-IDF Vectorization",
                        "Bag of Words (BoW)"
                    ],
                    index=0,
                    help="Chọn phương pháp vector hóa văn bản cho KNN"
                )
            st.markdown("---")
            
            # Show different UI based on optimization type
            if knn_optimization_type == "Manual K Input":

                # Manual Configuration
                col1, col2, col3 = st.columns(3)
                with col1:
                    manual_k = st.number_input(
                        "🎯 **K Value:**",
                        min_value=1,
                        max_value=50,
                        value=5,
                        step=1,
                        help="Số neighbors gần nhất"
                    )
                    
                with col2:
                    manual_weights = st.selectbox(
                        "⚖️ **Weights:**",
                        options=["uniform", "distance"],
                        index=0,
                        help="Cách tính trọng số cho neighbors"
                    )
                    
                with col3:
                    manual_metric = st.selectbox(
                        "📏 **Distance Metric:**",
                        options=["cosine", "euclidean", "manhattan"],
                        index=0,
                        help="Phương pháp tính khoảng cách"
                    )
                
                # Manual Save Button
                if st.button("💾 Save Manual Configuration", type="secondary", 
                            help="Lưu cấu hình thủ công vào session"):
                    st.session_state.knn_config = {
                        'optimization_method': 'Manual Input',
                        'k_value': manual_k,
                        'weights': manual_weights,
                        'metric': manual_metric,
                        'best_score': None,
                        'cv_folds': None,
                        'scoring': None
                    }
                    st.toast(f"✅ Đã lưu cấu hình: K={manual_k}, Weights={manual_weights}, Metric={manual_metric}")
                
            elif knn_optimization_type == "Optimal K (Cosine Metric)":                
                # CV Configuration for Optimal K
                col1, col2 = st.columns(2)
                with col1:
                    knn_cv_folds = st.slider(
                        "🔄 **Cross-Validation Folds:**",
                        min_value=2,
                        max_value=10,
                        value=3,
                        step=1,
                        help="Số folds cho cross-validation (sẽ tự động điều chỉnh nếu class quá nhỏ)"
                    )
                    
                    # Show warning about CV fold requirements
                    st.caption("⚠️ **Lưu ý**: CV folds sẽ tự động điều chỉnh dựa trên số lượng mẫu nhỏ nhất trong mỗi class")
                    
                with col2:
                    knn_scoring = st.selectbox(
                        "📈 **Scoring Metric:**",
                        options=["f1_macro", "accuracy", "precision_macro", "recall_macro"],
                        index=0,
                        help="Metric đánh giá model"
                    )              
            
            else:  # Grid Search               
                # CV Configuration for Grid Search  
                col1, col2 = st.columns(2)
                with col1:
                    knn_cv_folds = st.slider(
                        "🔄 **Cross-Validation Folds:**",
                        min_value=2,
                        max_value=10,
                        value=3,
                        step=1,
                        help="Số folds cho cross-validation (sẽ tự động điều chỉnh nếu class quá nhỏ)"
                    )
                    
                    # Show warning about CV fold requirements
                    st.caption("⚠️ **Lưu ý**: CV folds sẽ tự động điều chỉnh dựa trên số lượng mẫu nhỏ nhất trong mỗi class")
                    
                with col2:
                    knn_scoring = st.selectbox(
                        "📈 **Scoring Metric:**",
                        options=["f1_macro", "accuracy", "precision_macro", "recall_macro"],
                        index=0,
                        help="Metric đánh giá model"
                    )
            
            # Show current KNN config if exists

            if 'knn_config' in st.session_state:
                current_config = st.session_state.knn_config
                if current_config.get('optimization_method') == 'Manual Input':
                    st.toast(f"✅ **Manual Config**: K={current_config.get('k_value', 'Not set')}, "
                              f"Weights={current_config.get('weights', 'Not set')}, "
                              f"Metric={current_config.get('metric', 'Not set')}")
                else:
                    score_display = f"{current_config.get('best_score', 0):.4f}" if current_config.get('best_score') is not None else "N/A"
            else:
                st.warning("⚠️ **Current Config**: No configuration set")
            
            # Run Optimization Button - Only show for optimization modes
            if knn_optimization_type != "Manual K Input":
                st.markdown("---")
                st.markdown("**🚀 Run KNN Optimization:**")
                
                # Optimization buttons for Optimal K and Grid Search
                if knn_optimization_type == "Optimal K (Cosine Metric)":
                    button_text = "🎯 Run Optimal K Search (Cosine Metric)"
                    button_help = "Tìm K tối ưu với cosine metric (nhanh, phù hợp text data)"
                else:
                    button_text = "🔍 Run Full Grid Search (All Parameters)"
                    button_help = "Tìm combination tối ưu cho tất cả tham số (chậm hơn, toàn diện)"
                
                if st.button(button_text, type="primary", help=button_help):
                    # Check if Step 2 is completed
                    step2_config = session_manager.get_step_data(2)
                    if not step2_config or not step2_config.get('completed', False):
                        st.error("❌ **Step 2 not completed!** Please complete Step 2 (Data Processing) first.")
                        st.stop()
                    
                    # Get column configuration
                    text_column = step2_config.get('text_column')
                    label_column = step2_config.get('label_column')
                    
                    if not text_column or not label_column:
                        st.error("❌ **Column configuration missing!** Please complete Step 2 first.")
                        st.stop()
                    
                    try:
                        # Use sample size from Step 1 configuration
                        step1_data = session_manager.get_step_data(1)
                        sampling_config = step1_data.get('sampling_config', {})
                        configured_sample_size = sampling_config.get('num_samples', len(df))
                        
                        # Use the configured sample size from Step 1
                        sample_size = min(configured_sample_size, len(df))
                        
                        # Create sample data
                        df_sample = df.sample(n=sample_size, random_state=42)
                        X_texts = df_sample[text_column].astype(str).tolist()
                        y_labels = df_sample[label_column].tolist()
                        
                        # Use Sentence Embeddings for KNN optimization (same as Project3_1_A-Son.ipynb)
                        from text_encoders import EmbeddingVectorizer
                        from sklearn.preprocessing import LabelEncoder
                                   
                        # Create sentence embeddings (use raw mode for direct text input)
                        embedding_vectorizer = EmbeddingVectorizer()
                        embedding_vectorizer.fit(X_texts)
                        X_train = embedding_vectorizer.transform(X_texts, mode='raw')
                        
                        label_encoder = LabelEncoder()
                        y_train = label_encoder.fit_transform(y_labels)
                        
                        # Calculate adaptive K range based on sample size (for embeddings)
                        n_samples = X_train.shape[0]  
                        n_classes = len(np.unique(y_train))
                        
                        # Check class distribution and adjust CV folds if needed
                        from collections import Counter
                        class_counts = Counter(y_train)
                        min_class_samples = min(class_counts.values())
                        
                        # Adjust CV folds based on smallest class size
                        original_cv_folds = knn_cv_folds
                        if min_class_samples < knn_cv_folds:
                            knn_cv_folds = max(2, min_class_samples)
                       
                        # Ensure we have enough samples for CV
                        if n_samples < 10:
                            st.error("❌ Sample size too small for KNN optimization. Need at least 10 samples.")
                            return
                        
                        # Always use full K range from 3 to 31 for comprehensive benchmarking
                        k_range = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
                        
                        # Log the K range being used
                        st.toast(f"🔍 **Using comprehensive K range**: {k_range}")
                        
                        # Validate K range
                        if not k_range:
                            st.error("❌ Invalid K range generated. Using default range.")
                            k_range = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
                        # Run optimization using KNNModel methods
                        with st.spinner("🔄 Running KNN optimization with embeddings..."):
                            from models.classification.knn_model import KNNModel
                            knn_model = KNNModel()
                            
                            if knn_optimization_type == "Optimal K (Cosine Metric)":
                                st.info("🎯 **Using determine_optimal_k** from KNNModel (cosine metric only)")
                                results = knn_model.determine_optimal_k(
                                    X_train, y_train, 
                                    cv_folds=knn_cv_folds, 
                                    scoring=knn_scoring,
                                    k_range=k_range,
                                    plot_results=False,
                                    use_gpu=True
                                )
                                
                                best_params = results['best_params']
                                best_score = results['best_score']
                                
                                st.session_state.knn_config = {
                                    'optimization_method': 'Optimal K (Cosine Metric)',
                                    'k_value': best_params['n_neighbors'],
                                    'weights': best_params['weights'],
                                    'metric': 'cosine',
                                    'cv_folds': knn_cv_folds,
                                    'scoring_metric': knn_scoring,
                                    'best_score': best_score
                                }
                                
                                # Store benchmark data for plotting
                                st.session_state.knn_benchmark_data = {
                                    'best_params': best_params,
                                    'best_score': best_score,
                                    'cv_results': results.get('cv_results', results),
                                    'param_grid': results.get('param_grid', {'n_neighbors': k_range, 'weights': ['uniform', 'distance']}),
                                    'n_samples': n_samples,
                                    'n_classes': n_classes
                                }
                                st.session_state.show_knn_benchmark = True
                                
                                st.success(f"✅ **Optimal K Found**: {best_params['n_neighbors']}")
                                st.success(f"🏆 **Best Score**: {best_score:.4f}")
                                
                            else:  # Grid Search
                                # Use the adjusted CV folds from above
                                results = knn_model.tune_hyperparameters(
                                    X_train, y_train,
                                    cv_folds=knn_cv_folds,  # This is already adjusted above
                                    scoring=knn_scoring,
                                    k_range=k_range,
                                    use_gpu=True
                                )
                                
                                best_params = results['best_params']
                                best_score = results['best_score']
                                
                                st.session_state.knn_config = {
                                    'optimization_method': 'Grid Search (All Parameters)',
                                    'k_value': best_params['n_neighbors'],
                                    'weights': best_params['weights'],
                                    'metric': best_params['metric'],
                                    'cv_folds': knn_cv_folds,
                                    'scoring_metric': knn_scoring,
                                    'best_score': best_score
                                }
                                
                                # Store benchmark data for plotting
                                st.session_state.knn_benchmark_data = {
                                    'best_params': best_params,
                                    'best_score': best_score,
                                    'cv_results': results.get('cv_results', results),
                                    'param_grid': results.get('param_grid', {'n_neighbors': k_range, 'weights': ['uniform', 'distance']}),
                                    'n_samples': n_samples,
                                    'n_classes': n_classes
                                }
                                st.session_state.show_knn_benchmark = True
                                
                                st.success(f"✅ **Grid Search Complete**: K={best_params['n_neighbors']}")
                                st.success(f"🏆 **Best Score**: {best_score:.4f}")
                                st.success(f"📏 **Best Metric**: {best_params['metric']}")
                            
                            st.toast("🎉 **Optimization completed successfully!**")
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ Error during KNN optimization: {str(e)}")
            
            # Show current configuration summary
            if 'knn_config' in st.session_state:
                st.markdown("---")
                st.markdown("**📋 Current KNN Configuration:**")
                config = st.session_state.knn_config
                
                # Different display based on optimization method
                if config.get('optimization_method') == 'Manual Input':

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 K Value", config.get('k_value', 'Not set'))
                    with col2:
                        st.metric("⚖️ Weights", config.get('weights', 'Not set'))
                    with col3:
                        st.metric("📏 Metric", config.get('metric', 'Not set'))
                else:
                    st.info(f"🔍 **{config.get('optimization_method', 'Optimized')}**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 K Value", config.get('k_value', 'Not set'))
                    with col2:
                        st.metric("⚖️ Weights", config.get('weights', 'Not set'))
                    with col3:
                        st.metric("📏 Metric", config.get('metric', 'Not set'))
                    
                    if config.get('best_score') is not None:
                        st.metric("🏆 Best Score", f"{config.get('best_score', 0):.4f}")
                
                # Clear config button
                if st.button("🗑️ Clear KNN Config", type="secondary", 
                            help="Xóa cấu hình KNN hiện tại"):
                    del st.session_state.knn_config
                    st.rerun()
                
                # Show benchmark popup with plot if available
                if st.session_state.get('show_knn_benchmark', False) and 'knn_benchmark_data' in st.session_state:
                    with st.expander("📊 KNN Optimization Benchmark Results", expanded=True):
                        benchmark_data = st.session_state.knn_benchmark_data

                        # Show benchmark plot if available
                        if 'cv_results' in benchmark_data and benchmark_data['cv_results']:
                            try:
                                # Get CV results and param grid
                                cv_results = benchmark_data['cv_results']
                                param_grid = benchmark_data.get('param_grid', {})
                                best_params = benchmark_data.get('best_params', {})
                                best_score = benchmark_data.get('best_score', 0.0)

                                if param_grid and 'n_neighbors' in param_grid and 'weights' in param_grid:
                                    # Use the KNN model's plotting method
                                    from models.classification.knn_model import KNNModel

                                    # Create a temporary KNN model instance to use its plotting method
                                    temp_knn = KNNModel()
                                    fig = temp_knn._plot_k_benchmark(
                                        cv_results=cv_results,
                                        param_grid=param_grid,
                                        best_params=best_params,
                                        best_score=best_score
                                    )

                                    if fig is not None:
                                        # Display the plot in Streamlit
                                        st.pyplot(fig)
                                        plt.close(fig)
                                    else:
                                        st.info("📊 Could not generate benchmark plot - check debug info above")

                                else:
                                    st.info("📊 Plot data not available - missing required parameters")

                            except Exception as e:
                                st.error(f"❌ Error creating benchmark plot: {str(e)}")
                                st.write("🔍 **Full Error Details:**")
                                import traceback
                                st.code(traceback.format_exc())

    
    # Text Vectorization Methods Section
    st.markdown("""
    <h3 style="color: var(--text-color); margin: 1.5rem 0 1rem 0;">📚 Text Vectorization Methods:</h3>
    """, unsafe_allow_html=True)
    
    # Get existing vectorization selection from session
    existing_vectorization = existing_config.get('selected_vectorization', [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        bow_vectorization = st.checkbox(
            "☑️ Bag of Words (BoW) - Fast, interpretable",
            value="BoW" in existing_vectorization,
            help="Simple word frequency representation"
        )
        
        tfidf_vectorization = st.checkbox(
            "☑️ TF-IDF - Better than BoW, handles rare words",
            value="TF-IDF" in existing_vectorization,
            help="Term frequency-inverse document frequency"
        )
    
    with col2:
        embeddings_vectorization = st.checkbox(
            "☑️ Word Embeddings - Semantic understanding, slower",
            value="Word Embeddings" in existing_vectorization,
            help="Semantic word representations"
        )
    
    # Collect selected models
    selected_models = []
    if knn_model:
        selected_models.append("K-Nearest Neighbors")
    if decision_tree:
        selected_models.append("Decision Tree")
    if naive_bayes:
        selected_models.append("Naive Bayes")
    if kmeans_model:
        selected_models.append("K-Means Clustering")
    if svm_model:
        selected_models.append("Support Vector Machine")
    if logistic_regression:
        selected_models.append("Logistic Regression")
    if linear_svc:
        selected_models.append("Linear SVC")
    
    # Collect selected vectorization methods
    selected_vectorization = []
    if bow_vectorization:
        selected_vectorization.append("BoW")
    if tfidf_vectorization:
        selected_vectorization.append("TF-IDF")
    if embeddings_vectorization:
        selected_vectorization.append("Word Embeddings")
    
    # Validation
    validation_errors = []
    validation_warnings = []
    
    # Validation logic - only print to terminal, do not show in Streamlit UI
    if not selected_models:
        print("ERROR: At least one model must be selected")
    elif len(selected_models) > 3:
        print("WARNING: More than 3 models selected - training time will be longer")

    if not selected_vectorization:
        print("ERROR: At least one vectorization method must be selected")
    elif len(selected_vectorization) > 2:
        print("WARNING: More than 2 vectorization methods selected - memory usage will be higher")

    # Check if data split is valid
    total_final = final_training + final_test
    if total_final != 100:
        print(f"ERROR: Data split percentages must equal 100%. Current total: {total_final}%")

    # Check if test set is reasonable
    if final_test < 15:
        print("WARNING: Test set is very small (< 15%). Consider increasing for better evaluation.")
    elif final_test > 30:
        print("WARNING: Test set is large (> 30%). Consider reducing to allocate more data for training.")

    # Check if training set is sufficient for cross-validation
    if final_training < 50:
        print("WARNING: Training set is small (< 50%). Cross-validation may have limited data per fold.")

    # Check cross-validation configuration
    if cv_folds > final_training / 10:
        print(f"WARNING: Many CV folds ({cv_folds}) with small training set ({final_training}%). Each fold may have insufficient data.")
    
    # Check if CV folds are reasonable for training set size
    if cv_folds > final_training / 5:
        print(f"WARNING: Too many CV folds ({cv_folds}) for training set ({final_training}%). Each fold may have very little data.")
    
    if not validation_errors and selected_models and selected_vectorization:
        st.toast("✅ **Configuration is valid!** Ready to proceed to training.")
    
    # Save configuration button
    if st.button("💾 Save Model Configuration", type="primary", 
                use_container_width=True):
        if not selected_models or not selected_vectorization:
            st.error("❌ Please select at least one model and "
                    "one vectorization method.")
            return
        
        if total_final != 100:
            st.error("❌ Data split percentages must equal 100%.")
            return
        
        # Store step 3 configuration
        step3_config = {
            'data_split': {
                'training': final_training,
                'test': final_test
            },
            'cross_validation': {
                'cv_folds': cv_folds,
                'random_state': random_state
            },
            'selected_models': selected_models,
            'selected_vectorization': selected_vectorization,
            'validation_errors': validation_errors,
            'validation_warnings': validation_warnings,
            'completed': True
        }
        
        # Add KNN specific configuration if KNN is selected and config exists
        if knn_model and 'knn_config' in st.session_state:
            knn_config = st.session_state.knn_config
            # Ensure all required fields are present for manual input
            if knn_config.get('optimization_method') == 'Manual Input':
                step3_config['knn_config'] = {
                    'optimization_method': 'Manual Input',
                    'k_value': knn_config.get('k_value', 5),
                    'weights': knn_config.get('weights', 'uniform'),
                    'metric': knn_config.get('metric', 'cosine'),
                    'best_score': None,
                    'cv_folds': None,
                    'scoring': None
                }
            else:
                step3_config['knn_config'] = knn_config
        
        session_manager.set_step_config('step3', step3_config)
        
        st.toast("Model configuration saved successfully! Ready for Step 4.")
        
        # Show configuration summary in terminal
        print(f"""
        **Model Configuration Summary:**
        - **Data Split**: Training={final_training}%, Test={final_test}%
        - **Cross-Validation**: {cv_folds} folds, Random State={random_state}
        - **CV Strategy**: Training set ({final_training}%) divided into {cv_folds} folds (~{final_training/cv_folds:.1f}% per fold)
        - **Validation**: Handled automatically by CV folds (no separate validation set)
        - **Selected Models**: {', '.join(selected_models)}
        - **Vectorization Methods**: {', '.join(selected_vectorization)}
        - **Total Combinations**: {len(selected_models) * len(selected_vectorization)} model-vectorization pairs
        """)
        
        # Show KNN configuration if selected
        if knn_model and 'knn_config' in step3_config:
            knn_config = step3_config['knn_config']
            if knn_config.get('optimization_method') == 'Manual Input':
                print(f"""
        **KNN Configuration (Manual Input):**
        - **Method**: Manual Configuration
        - **K Value**: {knn_config.get('k_value', 'Not set')}
        - **Weights**: {knn_config.get('weights', 'Not set')}
        - **Metric**: {knn_config.get('metric', 'Not set')}
        - **Note**: Parameters set manually, no optimization performed
        """)
            else:
                print(f"""
        **KNN Configuration (Optimized):**
        - **Method**: {knn_config.get('optimization_method', 'Not set')}
        - **K Value**: {knn_config.get('k_value', 'Not set')}
        - **Weights**: {knn_config.get('weights', 'Not set')}
        - **Metric**: {knn_config.get('metric', 'Not set')}
        - **Best Score**: {knn_config.get('best_score', 'Not available') if knn_config.get('best_score') is None else f"{knn_config.get('best_score'):.4f}"}
        """)
        
        # Show completion message
        st.toast("Step 3 completed successfully!")
        st.toast("Click 'Next ▶' button to proceed to Step 4.")
    
    # Navigation buttons
    render_navigation_buttons()


def render_step4_wireframe():
    """Render Step 4 - Training Execution & Monitoring with auto-execution"""
    
    # Step title
    st.markdown("""
    <h2 style="text-align: left; color: var(--text-color); margin: 2rem 0 1rem 0; font-size: 1.8rem;">
        📍 STEP 4/6: Training Execution & Monitoring
    </h2>
    """, unsafe_allow_html=True)
    
    # Get data from previous steps
    session_manager = SessionManager()
    step1_data = session_manager.get_step_data(1)
    step2_data = session_manager.get_step_data(2)
    step3_data = session_manager.get_step_data(3)
    
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
    
    df = step1_data['dataframe']
    
    # Initialize result variable to prevent UnboundLocalError
    result = {
        'status': 'not_started',
        'message': 'Training not started',
        'results': {},
        'comprehensive_results': [],
        'successful_combinations': 0,
        'total_combinations': 0,
        'best_combinations': {},
        'total_models': 0,
        'models_completed': 0,
        'elapsed_time': 0,
        'evaluation_time': 0,
        'data_info': {},
        'embedding_info': {},
        'from_cache': False
    }
    
    # Training Control Section
    st.markdown("**🎮 Training Control:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        start_button = st.button("🚀 START TRAINING", type="primary", use_container_width=True)
    
    with col2:
        pause_button = st.button("⏸️ PAUSE", use_container_width=True)
    
    with col3:
        stop_button = st.button("⏹️ STOP", use_container_width=True)
    
    with col4:
        reset_button = st.button("🔄 RESET", use_container_width=True)
    
    # Overall Progress Section
    progress_bar = st.progress(0)
    progress_text = st.empty()
      
    # Current Step Status Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_phase = st.metric("Phase", "Initializing")
    
    with col2:
        current_model = st.metric("Model", "None")
    
    with col3:
        current_status = st.metric("Status", "Ready")
    
    # Real-time Metrics Section
    st.markdown("**📈 Real-time Metrics:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        models_completed = st.metric("Models Completed", "0")
    
    with col2:
        current_accuracy = st.metric("Current Accuracy", "0.00%")
    
    with col3:
        best_accuracy = st.metric("Best Accuracy", "0.00%")
    
            # Cache Management Section
    with st.expander("💾 Cache Management", expanded=False):
        st.markdown("**Cache Information:**")
        
        # Get cache info
        cached_results = get_cache_info()
        
        if cached_results:
            st.toast(f"✅ Found {len(cached_results)} cached training results")
            
            # Display cache table
            cache_data = []
            for cache in cached_results:
                cache_data.append({
                    'Cache Key': cache['cache_key'][:12] + '...',
                    'Age': format_cache_age(cache['age_hours']),
                    'Success Rate': f"{cache['results_summary'].get('successful_combinations', 0)}/{cache['results_summary'].get('total_combinations', 0)}",
                    'Training Time': f"{cache['results_summary'].get('evaluation_time', 0):.1f}s"
                })
            
            cache_df = pd.DataFrame(cache_data)
            st.dataframe(cache_df, use_container_width=True)
            
            # Cache actions
            if st.button("🗑️ Clear All Cache", type="secondary", key="clear_cache_confusion_matrix"):
                clear_cache_action()
        else:
            st.info("ℹ️ No cached results found")
        
        # Confusion Matrix from Cache Section
        if result.get('from_cache', False):
            with st.expander("🎨 Plot Confusion Matrices from Cache", expanded=False):
                st.markdown("**Generate Confusion Matrices from Cached Results:**")
                st.info("💡 Use cached results to create confusion matrices without retraining")
                
                if st.button("🎯 Plot All Confusion Matrices from Cache", type="primary"):
                    try:
                        pipeline = StreamlitTrainingPipeline()
                        success = pipeline.plot_confusion_matrices_from_cache(result)
                        
                        if success:
                            st.success("✅ Confusion matrices generated successfully from cache!")
                            st.info("📁 Check the 'pdf/Figures' folder for generated plots")
                        else:
                            st.error("❌ Failed to generate confusion matrices from cache")
                    except Exception as e:
                        st.error(f"❌ Error generating confusion matrices: {e}")
  
    # Training Log Section
    with st.expander("📝 Training Log", expanded=True):
        training_log = st.empty()
        
        # Initialize training state
        if 'training_started' not in st.session_state:
            st.session_state.training_started = False
            st.session_state.training_results = None
            st.session_state.training_log = []
        
        # Handle button clicks
        if start_button and not st.session_state.training_started:
            st.session_state.training_started = True
            st.session_state.training_log = []
            
            # Start training in background
            st.rerun()

        elif pause_button:
            st.session_state.training_started = False
            st.info("⏸️ Training paused")
        
        elif stop_button:
            st.session_state.training_started = False
            st.session_state.training_results = None
            st.session_state.training_log = []
            
            # Stop the training pipeline
            try:
                from training_pipeline import training_pipeline
                training_pipeline.stop_training()
                st.toast("⏹️ Training stopped - Current process will finish gracefully")
            except Exception as e:
                st.error(f"Error stopping training: {e}")
                st.toast("⏹️ Training stopped (UI only)")
        
        elif reset_button:
            st.session_state.training_started = False
            st.session_state.training_results = None
            st.session_state.training_log = []
            st.success("🔄 Training reset")
    
    # Execute training if started
    if st.session_state.training_started:
        try:
            # Import training pipeline
            from training_pipeline import execute_streamlit_training, get_training_status
            
            # Progress callback function
            def update_progress(phase, message, progress):
                progress_bar.progress(progress)
                progress_text.text(f"{phase}: {message}")
                current_phase.metric("Phase", phase)
                
                # Update log
                st.session_state.training_log.append(f"[{phase}] {message}")
                training_log.text("\n".join(st.session_state.training_log[-10:]))
                
                # Update metrics
                models_completed.metric("Models Completed", "Training...")
                current_accuracy.metric("Current Accuracy", "Training...")
                best_accuracy.metric("Best Accuracy", "Training...")
            
            # Check cache first
            pipeline = StreamlitTrainingPipeline()
            cached_results = pipeline.get_cached_results(step1_data, step2_data, step3_data)
            
            if cached_results:
                st.toast("🎯 Using cached results! No need to retrain.")
                result = {
                    'status': 'success',
                    'message': 'Using cached results',
                    'results': cached_results,
                    'comprehensive_results': cached_results.get('all_results', []),
                    'successful_combinations': cached_results.get('successful_combinations', 0),
                    'total_combinations': cached_results.get('total_combinations', 0),
                    'best_combinations': cached_results.get('best_combinations', {}),
                    'total_models': cached_results.get('total_models', 0),
                    'models_completed': cached_results.get('models_completed', 0),
                    'elapsed_time': 0,
                    'evaluation_time': cached_results.get('evaluation_time', 0),
                    'data_info': cached_results.get('data_info', {}),
                    'embedding_info': cached_results.get('embedding_info', {}),
                    'from_cache': True
                }
                st.session_state.training_log.append("✅ Using cached results (no training needed)")
            else:
                # Execute training
                with st.spinner("🚀 Training in progress..."):
                    result = execute_streamlit_training(
                        df, step1_data, step2_data, step3_data,
                        progress_callback=update_progress
                    )
            
            if result['status'] == 'success':
                st.session_state.training_results = result
                st.session_state.training_started = False
                
                # Update final metrics
                progress_bar.progress(1.0)
                progress_text.text("Comprehensive evaluation completed successfully!")
                current_phase.metric("Phase", "Completed")
                current_model.metric("Model", "All Models")
                current_status.metric("Status", "Completed")
                
                # Display comprehensive evaluation metrics
                successful_combinations = result.get('successful_combinations', 0)
                total_combinations = result.get('total_combinations', 0)
                models_completed.metric("Combinations Evaluated", f"{successful_combinations}/{total_combinations}")
                
                # Show best combination if available
                if 'best_combinations' in result and result['best_combinations']:
                    best_overall = result['best_combinations'].get('best_overall', {})
                    if best_overall:
                        best_accuracy.metric("Best Accuracy", f"{best_overall.get('test_accuracy', 0):.3f}")
                        st.session_state.training_log.append(f"🏆 Best: {best_overall.get('combination_key', 'N/A')}")
                
                # Show completion message
                if result.get('from_cache', False):
                    st.toast("🎯 Using cached results!")
                    st.info(f"📋 Retrieved {total_combinations} model-embedding combinations from cache (no training needed)")
                else:
                    st.toast("🎉 Comprehensive evaluation completed successfully!")
                    st.info(f"Evaluated {total_combinations} model-embedding combinations in {result['elapsed_time']:.2f} seconds")
                
                # ===== COMPREHENSIVE RESULTS DISPLAY =====
                st.markdown("---")
                st.markdown("""
                <h3 style="color: var(--text-color); margin: 1.5rem 0 1rem 0;">🥇 Best Overall Model</h3>
                """, unsafe_allow_html=True)
                
                # Best Model Performance
                if 'best_combinations' in result and result['best_combinations']:
                    best_overall = result['best_combinations'].get('best_overall', {})
                    if best_overall:
                        
                        best_col1, best_col2, best_col3 = st.columns(3)
                        
                        with best_col1:
                            st.metric("Model", best_overall.get('combination_key', 'N/A'))
                        
                        with best_col2:
                            st.metric("F1 Score", f"{best_overall.get('f1_score', 0):.3f}")
                        
                        with best_col3:
                            st.metric("Validation Accuracy", f"{best_overall.get('validation_accuracy', 0):.3f}")
                
                # Detailed Results Table
                if 'comprehensive_results' in result and result['comprehensive_results']:
                    st.markdown("""
                    <h4 style="color: var(--text-color); margin: 1rem 0 0.5rem 0;">📊 Detailed Results Table</h4>
                    """, unsafe_allow_html=True)
                    
                    # Create results dataframe
                    results_data = []
                    for res in result['comprehensive_results']:
                        if res['status'] == 'success':
                            results_data.append({
                                'Model': res['model_name'].replace('_', ' ').title(),
                                'Embedding': res['embedding_name'].replace('_', ' ').title(),
                                'CV Accuracy': f"{res.get('cv_mean_accuracy', 0):.3f}±{res.get('cv_std_accuracy', 0):.3f}",
                                'Test Accuracy': f"{res.get('test_accuracy', 0):.3f}",
                                'Precision': f"{res.get('test_metrics', {}).get('precision', 0):.3f}",
                                'Recall': f"{res.get('test_metrics', {}).get('recall', 0):.3f}",
                                'F1 Score': f"{res.get('f1_score', 0):.3f}",
                                'Overfitting': res.get('overfitting_status', 'N/A').replace('_', ' ').title(),
                                'Training Time': f"{res.get('training_time', 0):.2f}s"
                            })
                    
                    if results_data:
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True, hide_index=True)
                        
                        # Download results button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name="comprehensive_evaluation_results.csv",
                            mime="text/csv"
                        )
                
                
                # Save step 4 configuration
                step4_config = {
                    'training_results': result,
                    'models_completed': result['models_completed'],
                    'elapsed_time': result['elapsed_time'],
                    'completed': True
                }
                session_manager.set_step_config('step4', step4_config)
                
                st.toast("Training results saved! Ready for Step 5.")
                
                # Show next step guidance
                st.success("🎯 **Next Step**: Proceed to Step 5 to analyze and export results!")
                
            else:
                st.error(f"❌ Training failed: {result['message']}")
                st.session_state.training_started = False
                
        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            st.session_state.training_started = False
    
    # Navigation buttons
    render_navigation_buttons()


if __name__ == "__main__":
    # Initialize session manager and ensure current_step is set
    try:
        session_manager = SessionManager()
        if session_manager.get_current_step() is None:
            session_manager.set_current_step(1)
    except Exception:
        pass
    
    main()
