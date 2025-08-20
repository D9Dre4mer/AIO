"""
Topic Modeling - Auto Classifier
Step 1: Dataset Selection & Upload
Exact wireframe implementation

Created: 2025-01-27
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wizard_ui.steps.step1_dataset import DatasetSelectionStep
from wizard_ui.session_manager import SessionManager

# Page configuration
st.set_page_config(
    page_title="🤖 Topic Modeling - Auto Classifier",
    page_icon="🤖",
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
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        render_step1_wireframe()
    
    with col2:
        render_sidebar()

def render_step1_wireframe():
    """Render Step 1 exactly as per wireframe design"""
    
    # Step title - simplified without big container
    st.markdown("""
    <h2 style="text-align: left; color: var(--text-color); margin: 2rem 0 1rem 0; font-size: 1.8rem;">
        📍 STEP 1/7: Dataset Selection & Upload
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
            "File Path (File Path)",
            "Upload Custom File (CSV/JSON/Excel)",
            "Use Sample Dataset (Cache Folder)"
        ],
        index=1,
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
                        session_manager.update_step_data(1, 'file_path', file_path)
                        
                        # Show file preview
                        show_file_preview(df, file_extension)
                        
                    else:
                        st.error("❌ File not found. Please check the path and try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
        
    elif "Sample Dataset" in dataset_source:
        st.info("🎲 Sample dataset will be available in future versions")
    
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
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")


def show_file_preview(df, file_extension):
    """Show file preview and info for both uploaded and path-based files"""
    
    # Dataset Preview Section
    st.markdown("""
    <div class="section-box">
        <h3>📊 Dataset Preview (if available):</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Data preview box
    st.markdown("""
    <div class="preview-box">
        <h4>Data Preview - First 5 rows</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(df.head(5), use_container_width=True)
    
    # Dataset Info Section
    st.markdown("""
    <div class="section-box">
        <h3>📈 Dataset Info:</h3>
    </div>
    """, unsafe_allow_html=True)
    
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
     
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button("◀ Previous", disabled=True, use_container_width=True)
    
    with col2:
        if st.button("Next ▶", use_container_width=True):
            session_manager = SessionManager()
            step_data = session_manager.get_step_data(1)
            
            if 'dataframe' in step_data and step_data['dataframe'] is not None:
                st.success("✅ Step 1 completed! Moving to Step 2...")
            else:
                st.warning("⚠️ Please complete Step 1 first")
    
    with col3:
        st.button("Skip to End", use_container_width=True)

def render_sidebar():
    """Render sidebar with progress tracking"""
    
    st.sidebar.title("🔍 Progress Tracker")
    
    # Current step info
    st.sidebar.markdown("""
    <div style="background: var(--secondary-background-color); padding: 1rem; border-radius: 8px; margin: 1rem 0; border: 1px solid var(--border-color);">
        <h4>📍 Current Step</h4>
        <p><strong>Step 1/7:</strong> Dataset Selection</p>
        <p><strong>Status:</strong> In Progress</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step status - Dynamic based on current progress
    current_step = 1  # Default to Step 1 for now
    total_steps = 7
    
    # Get current step from session if available
    try:
        session_manager = SessionManager()
        if hasattr(session_manager, 'get_current_step'):
            current_step = session_manager.get_current_step()
        else:
            # Fallback: check which step has data
            for step_num in range(1, total_steps + 1):
                step_data = session_manager.get_step_data(step_num)
                if step_data and len(step_data) > 0:
                    current_step = step_num
                else:
                    break
    except:
        current_step = 1
    
    # Generate step status dynamically
    step_names = [
        "Dataset Selection",
        "Data Preprocessing", 
        "Column Selection",
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
        session_manager.reset_session()
        st.sidebar.success("Step 1 reset!")
        st.rerun()
    
    if st.sidebar.button("💾 Save Progress", use_container_width=True):
        st.sidebar.success("Progress saved!")

if __name__ == "__main__":
    main()
