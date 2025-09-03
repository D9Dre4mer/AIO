"""
Dataset Preview Component for Wizard UI

Created: 2025-01-27
"""

import streamlit as st
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DatasetPreviewComponent:
    """Component for previewing dataset information"""
    
    def __init__(self):
        """Initialize dataset preview component"""
        pass
    
    def render_dataset_info(self, df: pd.DataFrame) -> None:
        """Render dataset information and statistics"""
        if df is None or df.empty:
            st.warning("No dataset available for preview")
            return
        
        st.subheader("📊 Dataset Information")
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rows", len(df))
        
        with col2:
            st.metric("Total Columns", len(df.columns))
        
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("Memory Usage", f"{memory_mb:.1f} MB")
        
        # Column information
        st.subheader("📋 Column Information")
        
        col_info = []
        for col in df.columns:
            col_info.append({
                'Column': col,
                'Type': str(df[col].dtype),
                'Non-Null': df[col].count(),
                'Null': df[col].isnull().sum(),
                'Unique': df[col].nunique()
            })
        
        col_df = pd.DataFrame(col_info)
        st.dataframe(col_df, use_container_width=True)
        
        # Data preview
        st.subheader("👀 Data Preview")
        
        tab1, tab2 = st.tabs(["First 10 Rows", "Last 10 Rows"])
        
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        
        with tab2:
            st.dataframe(df.tail(10), use_container_width=True)
        
        # Missing values visualization
        if df.isnull().sum().sum() > 0:
            st.subheader("⚠️ Missing Values")
            
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            
            if not missing_data.empty:
                st.bar_chart(missing_data)
                
                # Show columns with missing values
                st.write("Columns with missing values:")
                for col, missing_count in missing_data.items():
                    percentage = (missing_count / len(df)) * 100
                    st.write(f"- **{col}**: {missing_count} ({percentage:.1f}%)")
    
    def render_data_types_summary(self, df: pd.DataFrame) -> None:
        """Render data types summary"""
        if df is None or df.empty:
            return
        
        st.subheader("🔍 Data Types Summary")
        
        # Count data types
        type_counts = df.dtypes.value_counts()
        
        # Display type distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Type Distribution:**")
            for dtype, count in type_counts.items():
                st.write(f"- {dtype}: {count}")
        
        with col2:
            # Pie chart of data types
            st.write("**Data Type Chart:**")
            st.bar_chart(type_counts)
    
    def render_sample_data(self, df: pd.DataFrame, 
                          sample_size: int = 5) -> None:
        """Render random sample of data"""
        if df is None or df.empty:
            return
        
        st.subheader("🎲 Random Sample")
        
        # Get random sample
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        st.dataframe(sample_df, use_container_width=True)
        
        # Sample info
        st.info(f"Showing {len(sample_df)} random rows from {len(df)} total rows")
    
    def get_dataset_summary(self, df: pd.DataFrame) -> dict:
        """Get comprehensive dataset summary"""
        if df is None or df.empty:
            return {}
        
        try:
            summary = {
                'shape': df.shape,
                'columns': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'duplicate_rows': df.duplicated().sum()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating dataset summary: {str(e)}")
            return {}
