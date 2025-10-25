"""
File Upload Component for Wizard UI

Created: 2025-01-27
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FileUploadComponent:
    """Reusable file upload component for wizard steps"""
    
    def __init__(self, max_size_mb: int = 100):
        self.max_size_mb = max_size_mb
        self.allowed_types = ['.csv', '.xlsx', '.xls', '.json', '.txt']
    
    def render_upload_section(self, key: str = "file_uploader") -> Optional[Any]:
        """Render file upload section"""
        st.subheader("📁 Dataset Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=[ext.lstrip('.') for ext in self.allowed_types],
            help="Upload your dataset file (CSV, Excel, JSON, TXT)",
            key=key
        )
        
        if uploaded_file:
            if self._validate_file(uploaded_file):
                st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
                self._display_file_info(uploaded_file)
                return uploaded_file
            else:
                st.error("❌ File validation failed")
                return None
        
        return None
    
    def _validate_file(self, file) -> bool:
        """Validate uploaded file"""
        try:
            file_size_mb = file.size / (1024 * 1024)
            if file_size_mb > self.max_size_mb:
                st.error(f"File size ({file_size_mb:.1f} MB) exceeds limit ({self.max_size_mb} MB)")
                return False
            
            file_extension = Path(file.name).suffix.lower()
            if file_extension not in self.allowed_types:
                st.error(f"File type '{file_extension}' not supported")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"File validation error: {str(e)}")
            return False
    
    def _display_file_info(self, file) -> None:
        """Display file information"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("File Name", file.name)
        
        with col2:
            file_size_mb = file.size / (1024 * 1024)
            st.metric("File Size", f"{file_size_mb:.1f} MB")
        
        with col3:
            file_type = Path(file.name).suffix.upper()
            st.metric("File Type", file_type)
    
    def get_file_data(self, file) -> Optional[pd.DataFrame]:
        """Get file data as DataFrame"""
        try:
            file_extension = Path(file.name).suffix.lower()
            
            if file_extension == '.csv':
                return pd.read_csv(file)
            elif file_extension in ['.xlsx', '.xls']:
                return pd.read_excel(file)
            elif file_extension == '.json':
                return pd.read_json(file)
            else:
                return None
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
