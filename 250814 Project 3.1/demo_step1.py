"""
Demo App for Step 1: Dataset Selection

Test the Step 1 implementation of the wizard UI

Author: AI Assistant
Created: 2025-01-27
"""

import streamlit as st
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wizard_ui.steps.step1_dataset import DatasetSelectionStep
from wizard_ui.session_manager import SessionManager

def main():
    """Main demo application"""
    st.set_page_config(
        page_title="Wizard UI - Step 1 Demo",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🧪 Wizard UI - Step 1 Demo")
    st.markdown("Testing the Dataset Selection & Upload step")
    
    # Initialize session manager
    session_manager = SessionManager()
    
    # Initialize Step 1
    step1 = DatasetSelectionStep()
    
    # Render Step 1
    step1.render()
    
    # Show session state info
    st.sidebar.title("🔍 Session State Info")
    
    if st.sidebar.button("Show Session Data"):
        step_data = session_manager.get_step_data(1)
        st.sidebar.json(step_data)
    
    if st.sidebar.button("Show Progress"):
        progress = session_manager.get_progress(1)
        st.sidebar.metric("Step 1 Progress", f"{progress:.0%}")
    
    if st.sidebar.button("Validate Step"):
        is_valid = step1.validate_step()
        if is_valid:
            st.sidebar.success("✅ Step 1 is valid!")
        else:
            st.sidebar.warning("⚠️ Step 1 is not complete")
    
    if st.sidebar.button("Reset Session"):
        session_manager.reset_session()
        st.sidebar.success("Session reset!")
        st.rerun()

if __name__ == "__main__":
    main()
