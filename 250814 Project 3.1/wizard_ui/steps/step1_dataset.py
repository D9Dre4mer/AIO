"""
Step 1: Dataset Selection & Upload

Created: 2025-01-27
"""

import streamlit as st
import pandas as pd
import logging

from ..components.file_upload import FileUploadComponent
from ..components.dataset_preview import DatasetPreviewComponent
from ..session_manager import SessionManager

logger = logging.getLogger(__name__)


class DatasetSelectionStep:
    """Step 1: Dataset Selection & Upload"""
    
    def __init__(self):
        """Initialize Step 1"""
        self.file_uploader = FileUploadComponent()
        self.dataset_preview = DatasetPreviewComponent()
        self.session_manager = SessionManager()
    
    def render(self) -> None:
        """Render the complete Step 1 interface"""
        st.title("📊 Step 1: Dataset Selection & Upload")
        
        st.markdown("""
        **What you'll do here:**
        1. 📁 Upload your dataset file (CSV, Excel, JSON, TXT)
        2. 📊 Preview and validate your data
        3. 🔍 Understand your dataset structure
        4. ✅ Confirm data quality for modeling
        """)
        
        # File upload section
        uploaded_file = self._render_file_upload()
        
        # Dataset preview and validation
        if uploaded_file:
            self._render_dataset_processing(uploaded_file)
        
        # Step completion
        self._render_step_completion()
    
    def _render_file_upload(self):
        """Render file upload section"""
        st.subheader("📁 File Upload")
        
        uploaded_file = self.file_uploader.render_upload_section(
            key="step1_file_uploader"
        )
        
        if uploaded_file:
            file_info = {
                'name': uploaded_file.name,
                'size': uploaded_file.size,
                'type': uploaded_file.type
            }
            self.session_manager.update_step_data(1, 'uploaded_file', file_info)
        
        return uploaded_file
    
    def _render_dataset_processing(self, uploaded_file):
        """Render dataset processing and preview"""
        st.subheader("🔄 Processing Dataset")
        
        with st.spinner("Reading and analyzing your dataset..."):
            df = self.file_uploader.get_file_data(uploaded_file)
            
            if df is not None:
                # CRITICAL FIX: Automatically create selected_categories for ALL datasets
                selected_categories = self._extract_categories_from_dataframe(df)
                
                self.session_manager.update_step_data(1, 'dataframe', df)
                self.session_manager.update_step_data(1, 'selected_categories', selected_categories)
                summary = self.dataset_preview.get_dataset_summary(df)
                self.session_manager.update_step_data(1, 'dataset_summary', summary)
                
                st.success("✅ Dataset processed successfully!")
                
                # Display extracted categories
                if selected_categories:
                    st.info(f"🏷️ Detected {len(selected_categories)} categories: {', '.join(selected_categories[:5])}{'...' if len(selected_categories) > 5 else ''}")
                
                self.dataset_preview.render_dataset_info(df)
                self.dataset_preview.render_data_types_summary(df)
                self.dataset_preview.render_sample_data(df, sample_size=5)
                
                self._render_data_quality_check(df)
            else:
                st.error("❌ Failed to process dataset")
                self.session_manager.update_step_data(1, 'dataframe', None)
    
    def _extract_categories_from_dataframe(self, df):
        """
        Extract categories from dataframe for ANY dataset type
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            List[str]: List of unique category labels
        """
        try:
            # Strategy 1: Look for common label column names
            label_column_candidates = [
                'label', 'category', 'class', 'target', 'y', 'output',
                'categories', 'labels', 'classes', 'targets'
            ]
            
            # Find the label column (usually the last column or one with few unique values)
            label_column = None
            
            # Check if any column name matches our candidates
            for col in df.columns:
                if col.lower() in [c.lower() for c in label_column_candidates]:
                    label_column = col
                    break
            
            # If no match found, use the last column (common convention)
            if label_column is None:
                label_column = df.columns[-1]
            
            # Extract unique categories
            unique_categories = sorted(df[label_column].unique().tolist())
            
            # Clean and validate categories
            cleaned_categories = []
            for cat in unique_categories:
                if cat is not None and str(cat).strip():
                    # Convert to string and clean
                    clean_cat = str(cat).strip()
                    if clean_cat and clean_cat not in cleaned_categories:
                        cleaned_categories.append(clean_cat)
            
            print(f"✅ [STEP1] Extracted {len(cleaned_categories)} categories from column '{label_column}'")
            
            return cleaned_categories
            
        except Exception as e:
            print(f"⚠️ [STEP1] Error extracting categories: {e}")
            # Fallback: return empty list
            return []
    
    def _render_data_quality_check(self, df):
        """Render data quality assessment"""
        st.subheader("🔍 Data Quality Assessment")
        
        quality_score = 0
        total_checks = 5
        
        # Check data size
        if len(df) >= 100:
            st.success("✅ Dataset size: Good (>100 rows)")
            quality_score += 1
        else:
            st.warning("⚠️ Dataset size: Small (<100 rows)")
        
        # Check missing values
        missing_total = df.isnull().sum().sum()
        missing_percentage = (missing_total / (len(df) * len(df.columns))) * 100
        
        if missing_percentage < 10:
            st.success(f"✅ Missing values: Good ({missing_percentage:.1f}%)")
            quality_score += 1
        else:
            st.warning(f"⚠️ Missing values: High ({missing_percentage:.1f}%)")
        
        # Check duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count == 0:
            st.success("✅ Duplicate rows: None found")
            quality_score += 1
        else:
            st.warning(f"⚠️ Duplicate rows: {duplicate_count} found")
        
        # Check text columns
        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            st.success(f"✅ Text columns: {len(text_columns)} found")
            quality_score += 1
        else:
            st.warning("⚠️ Text columns: No text columns found")
        
        # Check data variety
        if df.nunique().mean() > 2:
            st.success("✅ Data variety: Good")
            quality_score += 1
        else:
            st.warning("⚠️ Data variety: Low")
        
        # Overall quality score
        quality_percentage = (quality_score / total_checks) * 100
        
        st.subheader(f"📊 Overall Quality Score: {quality_percentage:.0f}%")
        st.progress(quality_score / total_checks)
        
        self.session_manager.update_step_data(1, 'quality_score', quality_percentage)
        
        if quality_percentage >= 80:
            st.success("🎉 Excellent! Your dataset is ready for modeling.")
        elif quality_percentage >= 60:
            st.info("👍 Good! Your dataset can be used for modeling.")
        else:
            st.warning("⚠️ Consider improving data quality before modeling.")
    
    def _render_step_completion(self):
        """Render step completion section"""
        st.subheader("✅ Step Completion")
        
        step_data = self.session_manager.get_step_data(1)
        is_complete = (
            'dataframe' in step_data and 
            step_data['dataframe'] is not None and
            'quality_score' in step_data
        )
        
        if is_complete:
            st.success("🎯 Step 1 completed successfully!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Dataset", step_data.get('uploaded_file', {}).get('name', 'N/A'))
            
            with col2:
                df = step_data.get('dataframe')
                if df is not None:
                    st.metric("Rows", len(df))
            
            with col3:
                quality = step_data.get('quality_score', 0)
                st.metric("Quality Score", f"{quality:.0f}%")
            
            self.session_manager.set_progress(1, 1.0)
            
        else:
            st.info("📝 Complete the dataset upload to proceed to the next step.")
            self.session_manager.set_progress(1, 0.0)
    
    def validate_step(self) -> bool:
        """Validate if Step 1 is complete"""
        step_data = self.session_manager.get_step_data(1)
        
        required_fields = ['dataframe', 'quality_score']
        for field in required_fields:
            if field not in step_data or step_data[field] is None:
                return False
        
        return True
