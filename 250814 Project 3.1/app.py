"""
Topic Modeling - Auto Classifier
Step 1: Dataset Selection & Upload
Step 2: Data Preprocessing & Sampling
Exact wireframe implementation

Created: 2025-01-27
"""

import streamlit as st
import pandas as pd
import sys
import os
import time

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wizard_ui.session_manager import SessionManager

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
                        
                        session_manager.update_step_data(1, 'sampling_config', {
                            'num_samples': default_samples,
                            'sampling_strategy': 'Stratified (Recommended)'
                        })
                        
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
                    
                    session_manager.update_step_data(1, 'sampling_config', {
                        'num_samples': default_samples,
                        'sampling_strategy': 'Stratified (Recommended)'
                    })

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
            if 'df' in locals():
                dataset_size = len(df)
                
                # Get existing sampling config from session
                session_manager = SessionManager()
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
        
        with col2:
            # Sampling strategy
            existing_strategy = existing_config.get('sampling_strategy', 'Stratified (Recommended)')
            strategy_index = 1 if existing_strategy == "Stratified (Recommended)" else 0
            
            sampling_strategy = st.radio(
                "🎯 Sampling Strategy:",
                ["Random", "Stratified (Recommended)"],
                index=strategy_index,
                help="Random: Simple random sampling. Stratified: Maintains class distribution."
            )
        
        # Save sampling configuration to session
        if 'df' in locals():
            session_manager.update_step_data(1, 'sampling_config', {
                'num_samples': num_samples,
                'sampling_strategy': sampling_strategy
            })
    
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
        
        session_manager.update_step_data(1, 'sampling_config', {
            'num_samples': default_samples,
            'sampling_strategy': 'Stratified (Recommended)'
        })
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")


def show_file_preview(df, file_extension):
    """Show file preview and info for both uploaded and path-based files"""
    
    # Data preview box
    st.subheader("📊 Data Preview (First 5 rows)")
    
    st.dataframe(df.head(5), use_container_width=True)
    
    # Metrics in boxes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h4>Shape</h4>
            <p><strong>{df.shape[0]} rows, {df.shape[1]} columns</strong></p>
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
                    # Move to step 2 (Column Selection & Preprocessing)
                    session_manager.set_current_step(2)
                    st.success("✅ Step 1 completed! Moving to Step 2...")
                    st.rerun()
                else:
                    st.warning("⚠️ Please complete Step 1 first")
            elif current_step == 2:
                step_data = session_manager.get_step_data(2)
                if step_data and step_data.get('completed', False):
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
    
    st.sidebar.title("🔍 Progress Tracker")
    
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
    
    st.sidebar.markdown(f"""
    <div style="background: var(--secondary-background-color); padding: 1rem; border-radius: 8px; margin: 1rem 0; border: 1px solid var(--border-color);">
        <h4>📍 Current Step</h4>
        <p><strong>Step {current_step}/6:</strong> {current_step_name}</p>
        <p><strong>Status:</strong> In Progress</p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    step_status_html = ""
    for i, step_name in enumerate(step_names, 1):
        if i < current_step:
            status_icon = "✅"
            status_text = "Completed"
        elif i == current_step:
            status_icon = "🔄"
            status_text = "Current"
        else:
            status_icon = "⏳"
            status_text = "Pending"
        
        # Check if step has data to show completion status
        step_data = session_manager.get_step_data(i)
        if step_data and len(step_data) > 0:
            if i == 1 and 'dataframe' in step_data:
                status_icon = "✅"
                status_text = "Completed"
            elif i == 2 and step_data.get('completed', False):  # Step 2 (Column Selection & Preprocessing)
                status_icon = "✅"
                status_text = "Completed"
            elif i == 3 and step_data.get('completed', False):  # Step 3 (Model Configuration)
                status_icon = "✅"
                status_text = "Completed"
            elif i == 4 and step_data.get('completed', False):  # Step 4 (Training Execution)
                status_icon = "✅"
                status_text = "Completed"
        
        step_status_html += f'<p>{status_icon} Step {i}: {step_name} ({status_text})</p>'
    
    st.sidebar.markdown(f"""
    <div style="background: var(--secondary-background-color); padding: 1rem; border-radius: 8px; margin: 1rem 0; border: 1px solid var(--border-color);">
        <h4>📋 Step Status</h4>
        {step_status_html}
    </div>
    """, unsafe_allow_html=True)
    
    # Quick actions
    if st.sidebar.button("🔄 Reset Step", use_container_width=True):
        session_manager = SessionManager()
        current_step = get_current_step(session_manager)
        if current_step > 1:
            # Reset to step 1
            session_manager.set_current_step(1)
            st.sidebar.success(f"Reset from Step {current_step} to Step 1!")
        else:
            # Reset current step data and current_step
            session_manager.reset_session()
            session_manager.set_current_step(1)
            st.sidebar.success("Step 1 reset!")
        st.rerun()
    
    if st.sidebar.button("💾 Save Progress", use_container_width=True):
        session_manager = SessionManager()
        current_step = get_current_step(session_manager)
        # Save current step to ensure progress is preserved
        session_manager.set_current_step(current_step)
        st.sidebar.success(f"Progress saved! Current step: {current_step}")

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
        text_samples = len(text_data)
        
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
        <p>• Samples: <strong>{text_samples:,}</strong></p>
        <p>• Avg Length: <strong>{avg_length:.0f} chars</strong></p>
        <p>• Unique Words: <strong>{unique_words:,}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Label Column Analysis        
        # Calculate label column statistics
        label_data = df[selected_label_column].dropna()
        unique_classes = label_data.nunique()
        
        # Check class distribution
        class_counts = label_data.value_counts()
        total_samples = len(label_data)
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
    if text_samples < 10:
        msg = "Text column has insufficient data (less than 10 samples)"
        print(f"ERROR: {msg}", file=sys.stderr)
    elif text_samples < 100:
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
    
    # Column Preview Section
    st.markdown("""
    <h3 style="color: var(--text-color); margin: 1.5rem 0 1rem 0;">👀 Column Preview:</h3>
    """, unsafe_allow_html=True)
    
    # Show sample data from selected columns
    preview_df = df[[selected_text_column, selected_label_column]].head(10)
    st.dataframe(preview_df, use_container_width=True)
    
    # Save configuration button
    if st.button("💾 Save Column Configuration", type="primary", use_container_width=True):
        # Store step 2 configuration with preprocessing options
        step2_config = {
            'text_column': selected_text_column,
            'label_column': selected_label_column,
            'text_samples': text_samples,
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
            'completed': True
        }
        
        session_manager.set_step_config('step2', step2_config)
        
        st.toast("Column configuration and preprocessing options saved! Ready for Step 3.")
        
        # Show configuration summary in terminal
        print(f"""
        **Configuration Summary:**
        - **Text Column**: {selected_text_column} ({text_samples:,} samples)
        - **Label Column**: {selected_label_column} ({unique_classes} classes)
        - **Distribution**: {distribution}
        - **Text Length**: {avg_length_words:.1f} words average
        - **Text Cleaning**: {'Enabled' if text_cleaning else 'Disabled'}
        - **Category Mapping**: {'Enabled' if category_mapping else 'Disabled'}
        - **Data Validation**: {'Enabled' if data_validation else 'Disabled'}
        - **Memory Optimization**: {'Enabled' if memory_optimization else 'Disabled'}
        """)
        
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
    
    # Display dataset info from previous steps
    st.info(f"📊 **Dataset**: {df.shape[0]:,} rows × {df.shape[1]} columns | "
            f"**Text Column**: {step2_data.get('text_column', 'N/A')} | "
            f"**Label Column**: {step2_data.get('label_column', 'N/A')}")
    
    # Data Split Configuration
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
        # Training + Validation combined (remaining percentage)
        training_validation_split = 100 - test_split
          
        # Sub-division of training + validation
        training_ratio = st.slider(
            f"Training + Validation: {training_validation_split}%*",
            min_value=70,
            max_value=95,
            value=80,
            step=5,
            help="Percentage of training+validation data used for training (70-95%). Rest goes to validation."
        )
    
    # Calculate actual percentages
    final_test = test_split
    final_training = int((training_validation_split * training_ratio) / 100)
    final_validation = training_validation_split - final_training
    
    # Display final split information
    st.info(f"📊 **Final Data Split**: Training: {final_training}% | Validation: {final_validation}% | Test: {final_test}%")
    
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
            help="Number of cross-validation folds (3-10). Each fold uses ~{final_training//5}% of data."
        )
        
        # Show cross-validation explanation
        if final_training > 0 and cv_folds > 0:
            fold_percentage = final_training / cv_folds
   
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
    total_final = final_training + final_validation + final_test
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

    # Check if validation set is reasonable
    if final_validation < 5:
        print("WARNING: Validation set is very small (< 5%). Consider increasing training ratio.")
    elif final_validation > 20:
        print("WARNING: Validation set is large (> 20%). Consider increasing training ratio.")

    # Check cross-validation configuration
    if cv_folds > final_training / 10:
        print(f"WARNING: Many CV folds ({cv_folds}) with small training set ({final_training}%). Each fold may have insufficient data.")

    # Check training ratio
    if training_ratio < 70:
        print("WARNING: Training ratio is low (< 70%). Consider increasing to allocate more data for training.")
    elif training_ratio > 95:
        print("WARNING: Training ratio is very high (> 95%). Consider reducing to have some validation data.")
    
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
                'validation': final_validation,
                'test': final_test,
                'training_ratio': training_ratio
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
        
        session_manager.set_step_config('step3', step3_config)
        
        st.toast("Model configuration saved successfully! Ready for Step 4.")
        
        # Show configuration summary in terminal
        print(f"""
        **Model Configuration Summary:**
        - **Data Split**: Training={final_training}%, Validation={final_validation}%, Test={final_test}%
        - **Training Ratio**: {training_ratio}% of non-test data used for training
        - **Cross-Validation**: {cv_folds} folds, Random State={random_state}
        - **CV Strategy**: Training set ({final_training}%) divided into {cv_folds} folds (~{final_training/cv_folds:.1f}% per fold)
        - **Selected Models**: {', '.join(selected_models)}
        - **Vectorization Methods**: {', '.join(selected_vectorization)}
        - **Total Combinations**: {len(selected_models) * len(selected_vectorization)} model-vectorization pairs
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
            st.warning("⏹️ Training stopped")
        
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
                st.toast("🎉 Comprehensive evaluation completed successfully!")
                st.info(f"Evaluated {total_combinations} model-embedding combinations in {result['elapsed_time']:.2f} seconds")
                
                # ===== COMPREHENSIVE RESULTS DISPLAY =====
                st.markdown("---")
                st.markdown("""
                <h3 style="color: var(--text-color); margin: 1.5rem 0 1rem 0;">🏆 Comprehensive Evaluation Results</h3>
                """, unsafe_allow_html=True)
                
                # Results Summary Cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Combinations", total_combinations)
                
                with col2:
                    st.metric("Successful", successful_combinations)
                
                with col3:
                    st.metric("Success Rate", f"{(successful_combinations/total_combinations)*100:.1f}%")
                
                with col4:
                    st.metric("Total Time", f"{result['elapsed_time']:.1f}s")
                
                # Best Model Performance
                if 'best_combinations' in result and result['best_combinations']:
                    best_overall = result['best_combinations'].get('best_overall', {})
                    if best_overall:
                        st.markdown("""
                        <h4 style="color: var(--text-color); margin: 1rem 0 0.5rem 0;">🥇 Best Overall Model</h4>
                        """, unsafe_allow_html=True)
                        
                        best_col1, best_col2, best_col3 = st.columns(3)
                        
                        with best_col1:
                            st.metric("Model", best_overall.get('combination_key', 'N/A'))
                        
                        with best_col2:
                            st.metric("Test Accuracy", f"{best_overall.get('test_accuracy', 0):.3f}")
                        
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
                                'Val Accuracy': f"{res['validation_accuracy']:.3f}",
                                'Test Accuracy': f"{res['test_accuracy']:.3f}",
                                'CV Accuracy': f"{res.get('cv_mean_accuracy', 0):.3f}±{res.get('cv_std_accuracy', 0):.3f}",
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
                
                # Performance Analysis
                if 'comprehensive_results' in result and result['comprehensive_results']:
                    st.markdown("""
                    <h4 style="color: var(--text-color); margin: 1rem 0 0.5rem 0;">📈 Performance Analysis</h4>
                    """, unsafe_allow_html=True)
                    
                    # Overfitting analysis
                    overfitting_counts = {}
                    for res in result['comprehensive_results']:
                        if res['status'] == 'success':
                            status = res.get('overfitting_status', 'unknown')
                            overfitting_counts[status] = overfitting_counts.get(status, 0) + 1
                    
                    if overfitting_counts:
                        overfitting_col1, overfitting_col2 = st.columns(2)
                        
                        with overfitting_col1:
                            st.markdown("**Overfitting Analysis:**")
                            for status, count in overfitting_counts.items():
                                percentage = (count / successful_combinations) * 100
                                st.markdown(f"• **{status.replace('_', ' ').title()}**: {count} ({percentage:.1f}%)")
                        
                        with overfitting_col2:
                            st.markdown("**Recommendations:**")
                            if 'overfitting' in overfitting_counts:
                                st.markdown("• ⚠️ Some models show overfitting - consider regularization")
                            if 'underfitting' in overfitting_counts:
                                st.markdown("• 📉 Some models show underfitting - consider more features")
                            if 'well_fitted' in overfitting_counts:
                                st.markdown("• ✅ Some models are well-fitted")
                
                # Data Information
                if 'data_info' in result:
                    data_info = result['data_info']
                    st.markdown("""
                    <h4 style="color: var(--text-color); margin: 1rem 0 0.5rem 0;">📋 Dataset Information</h4>
                    """, unsafe_allow_html=True)
                    
                    data_col1, data_col2, data_col3, data_col4 = st.columns(4)
                    
                    with data_col1:
                        st.metric("Training Samples", data_info.get('n_samples', 0))
                    
                    with data_col2:
                        st.metric("Validation Samples", data_info.get('n_validation', 0))
                    
                    with data_col3:
                        st.metric("Test Samples", data_info.get('n_test', 0))
                    
                    with data_col4:
                        st.metric("Classes", data_info.get('n_classes', 0))
                
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
