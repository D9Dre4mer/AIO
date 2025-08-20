"""
Streamlit Web Interface for Topic Modeling Project
Provides user-friendly interface for dataset input, auto classifier selection, 
and result visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# Import project modules
try:
    from data_loader import DataLoader
    from text_encoders import TextVectorizer
    from models.new_model_trainer import NewModelTrainer
    from config import MAX_SAMPLES, TEST_SIZE, RANDOM_STATE
except ImportError as e:
    st.error(f"Error importing project modules: {e}")
    st.info("Please ensure all project files are in the same directory")

# Page configuration
st.set_page_config(
    page_title="Topic Modeling - Auto Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .column-selector {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitTopicModeling:
    """Main class for Streamlit Topic Modeling interface"""
    
    def __init__(self):
        self.data_loader = None
        self.text_vectorizer = None
        self.model_trainer = None
        self.results = {}
        self.current_dataset = None
        self.selected_text_column = None
        self.selected_label_column = None
        
    def initialize_components(self):
        """Initialize project components"""
        try:
            self.data_loader = DataLoader()
            self.text_vectorizer = TextVectorizer()
            self.model_trainer = NewModelTrainer(cv_folds=5, validation_size=0.2)
            return True
        except Exception as e:
            st.error(f"Failed to initialize components: {e}")
            return False
    
    def load_dataset_from_file(self, uploaded_file):
        """Load dataset from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV, JSON, or Excel file.")
                return None
            
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    
    def load_dataset_from_path(self, file_path: str):
        """Load dataset from local file path"""
        try:
            import os
            from pathlib import Path
            
            # Validate file path
            path = Path(file_path.strip())
            if not path.exists():
                st.error(f"❌ File not found: {file_path}")
                return None
            
            if not path.is_file():
                st.error(f"❌ Path is not a file: {file_path}")
                return None
            
            # Check file size (warn if too large)
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > 100:  # 100MB limit
                st.warning(f"⚠️ Large file detected: {file_size_mb:.1f} MB. Loading may take time...")
            
            # Load based on file extension
            file_ext = path.suffix.lower()
            
            # Add debug info
            st.write(f"🔍 Debug: Loading file with extension: {file_ext}")
            
            if file_ext == '.csv':
                df = pd.read_csv(path, encoding='utf-8')
            elif file_ext == '.json':
                df = pd.read_json(path)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(path)
            elif file_ext == '.parquet':
                df = pd.read_parquet(path)
            elif file_ext == '.pickle' or file_ext == '.pkl':
                df = pd.read_pickle(path)
            else:
                st.error(f"❌ Unsupported file format: {file_ext}")
                st.info("Supported formats: CSV, JSON, Excel (.xlsx/.xls), Parquet, Pickle")
                return None
            
            # Debug DataFrame info
            st.write(f"🔍 Debug: DataFrame shape: {df.shape}")
            st.write(f"🔍 Debug: DataFrame columns: {list(df.columns)}")
            st.write(f"🔍 Debug: DataFrame dtypes: {df.dtypes.to_dict()}")
            
            # Ensure DataFrame has proper column names (no special characters)
            df.columns = [str(col).strip() for col in df.columns]
            
            # Basic data cleaning
            df = df.dropna(how='all')  # Remove completely empty rows
            df = df.loc[:, df.columns.notnull()]  # Remove columns with null names
            
            st.write(f"🔍 Debug: After cleaning - DataFrame shape: {df.shape}")
            st.write(f"🔍 Debug: After cleaning - DataFrame columns: {list(df.columns)}")
            
            if df.empty:
                st.error("❌ Dataset is empty after loading!")
                return None
            
            st.success(f"✅ File loaded successfully from: {path}")
            st.info(f"📊 File size: {file_size_mb:.1f} MB")
            st.info(f"📊 Dataset shape: {df.shape}")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error loading file from path: {e}")
            st.write("🔧 Debug info:")
            import traceback
            st.code(traceback.format_exc())
            st.info("💡 Make sure the path is correct and the file is accessible")
            return None
    
    def load_dataset_from_url(self, url):
        """Load dataset from URL"""
        try:
            if url.endswith('.csv'):
                df = pd.read_csv(url)
            elif url.endswith('.json'):
                df = pd.read_json(url)
            else:
                st.error("URL must point to a CSV or JSON file")
                return None
            
            return df
        except Exception as e:
            st.error(f"Error loading from URL: {e}")
            return None
    
    def select_dataset_columns(self, df):
        """Allow user to select text and label columns"""
        try:
            st.write("Select which columns to use for text classification:")
            
            # Get all columns
            all_columns = list(df.columns)
            st.write(f"🔍 Available columns: {all_columns}")
            
            # Simple column selection without complex session state
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📝 Text Column")
                text_col = st.selectbox(
                    "Select text column:",
                    options=all_columns,
                    index=0,
                    key="simple_text_selector"
                )
                
                # Show preview
                if text_col and text_col in df.columns:
                    st.write("**Preview:**")
                    preview_data = df[text_col].head(3).tolist()
                    st.write(preview_data)
            
            with col2:
                st.subheader("🏷️ Label Column")
                label_col = st.selectbox(
                    "Select label column:",
                    options=all_columns,
                    index=1 if len(all_columns) > 1 else 0,
                    key="simple_label_selector"
                )
                
                # Show preview
                if label_col and label_col in df.columns:
                    st.write("**Preview:**")
                    unique_labels = df[label_col].unique()
                    st.write(f"Unique labels: {unique_labels[:5].tolist()}")
                    if len(unique_labels) > 5:
                        st.write(f"... and {len(unique_labels) - 5} more")
            
            # Simple validation
            if text_col == label_col:
                st.error("❌ Text and label columns must be different!")
                return False
            
            # Update instance variables directly
            self.selected_text_column = text_col
            self.selected_label_column = label_col
        
            
            return True
            
        except Exception as e:
            st.error(f"Error in column selection: {e}")
            import traceback
            st.code(traceback.format_exc())
            return False
    
    def standardize_text_preprocessing(self, text_series):
        """Standard text preprocessing following data_loader.py logic"""
        try:
            # Print detailed processing steps to terminal only
            print("📝 Applying standard text preprocessing (following data_loader.py logic)...")
            print("🔧 Processing steps:")
            print("  • Remove \\n characters and strip whitespace")
            print("  • Remove special characters (keep only \\w\\s)")
            print("  • Remove digits")
            print("  • Remove extra spaces")
            print("  • Convert to lowercase")
            
            processed_text = []
            
            import re
            for i, text in enumerate(text_series):
                if pd.isna(text) or text == '':
                    processed_text.append('empty_text')
                    continue
                
                # Convert to string
                abstract = str(text)
                
                # Remove \n characters in the middle and leading/trailing spaces
                abstract = abstract.strip().replace("\n", " ")
                
                # Remove special characters (keep only word chars and spaces)
                abstract = re.sub(r'[^\w\s]', '', abstract)
                
                # Remove digits
                abstract = re.sub(r'\d+', '', abstract)
                
                # Remove extra spaces
                abstract = re.sub(r'\s+', ' ', abstract).strip()
                
                # Convert to lower case
                abstract = abstract.lower()
                
                # Handle empty results
                if not abstract or len(abstract) < 3:
                    processed_text.append('short_text_removed')
                else:
                    processed_text.append(abstract)
                
                # Show progress in terminal for large datasets
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{len(text_series)} texts...")
            
            # Convert back to pandas Series
            processed_series = pd.Series(processed_text)
            
            # Print sample transformations to terminal
            print("🔍 Sample text transformations:")
            for i in range(min(3, len(text_series))):
                original = str(text_series.iloc[i])[:100] + "..." if len(str(text_series.iloc[i])) > 100 else str(text_series.iloc[i])
                processed = processed_text[i][:100] + "..." if len(processed_text[i]) > 100 else processed_text[i]
                print(f"  '{original}' → '{processed}'")
            
            # Statistics to terminal
            valid_texts = [t for t in processed_text if t not in ['empty_text', 'short_text_removed']]
            avg_length = sum(len(t) for t in valid_texts) / len(valid_texts) if valid_texts else 0
            removed_count = len([t for t in processed_text if t in ['empty_text', 'short_text_removed']])
            
            print(f"📊 Average processed text length: {avg_length:.1f} characters")
            print(f"📊 Removed/empty texts: {removed_count} samples")
            print("✅ Text preprocessing completed!")
            
            # No processing info shown in Streamlit for text preprocessing
            
            return processed_series
            
        except Exception as e:
            st.error(f"Error in text preprocessing: {e}")
            print(f"❌ Error in text preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return text_series

    def universal_label_preprocessing(self, label_series):
        """Universal label preprocessing optimized for all data types"""
        try:
            # Print detailed processing steps to terminal only
            print("🏷️ Applying universal label preprocessing (optimized for all data types)...")
            
            processed_labels = []
            category_set = set()
            
            # Advanced Format Detection (print to terminal)
            sample_labels = label_series.head(20).tolist()
            print(f"🔧 Sample labels for detection: {sample_labels[:5]}...")
            
            # Comprehensive format detection
            has_dots = any('.' in str(label) for label in sample_labels if pd.notna(label))
            has_spaces_with_dots = any(' ' in str(label) and '.' in str(label) for label in sample_labels if pd.notna(label))
            has_separators = any(any(sep in str(label) for sep in ['|', ';', ',', '/']) for label in sample_labels if pd.notna(label))
            is_numeric = all(str(label).replace('.','').replace('-','').replace('_','').isdigit() 
                           for label in sample_labels[:5] if pd.notna(label) and str(label).strip())
            has_underscores = any('_' in str(label) for label in sample_labels if pd.notna(label))
            has_mixed_case = any(str(label) != str(label).lower() for label in sample_labels if pd.notna(label))
            
            print(f"  • Contains dots (.): {has_dots}")
            print(f"  • ArXiv format (spaces + dots): {has_spaces_with_dots}")
            print(f"  • Has separators (|, ;, ,, /): {has_separators}")
            print(f"  • Numeric labels: {is_numeric}")
            print(f"  • Has underscores (_): {has_underscores}")
            print(f"  • Mixed case: {has_mixed_case}")
            
            # Determine primary processing method
            if any('.' in str(l) for l in sample_labels if pd.notna(l)):
                processing_method = "ArXiv-style processing (data_loader.py logic)"
                if has_spaces_with_dots:
                    print("🎯 Detected ArXiv format - using data_loader.py logic...")
            elif any(any(sep in str(l) for sep in ['|', ';', ',', '/']) for l in sample_labels if pd.notna(l)):
                processing_method = "Multi-label separation"
                print("🎯 Detected multi-label format...")
            elif all(str(l).replace('.','').replace('-','').replace('_','').isdigit() 
                    for l in sample_labels[:5] if pd.notna(l) and str(l).strip()):
                processing_method = "Numeric label formatting"
                print("🎯 Detected numeric labels...")
            else:
                processing_method = "Text label normalization"
                print("🎯 Detected text labels...")
            
            # Universal Processing Logic
            for i, label in enumerate(label_series):
                if pd.isna(label) or str(label).strip() == '' or str(label).lower() == 'nan':
                    processed_labels.append('unknown_label')
                    continue
                
                label_str = str(label).strip()
                
                # ArXiv format (highest priority - follows data_loader.py)
                if ' ' in label_str and '.' in label_str:
                    parts = label_str.split(' ')
                    primary_category = parts[0].split('.')[0]
                    category_set.add(primary_category)
                    processed_labels.append(primary_category)
                    
                # Single ArXiv category
                elif '.' in label_str and ' ' not in label_str:
                    primary_category = label_str.split('.')[0]
                    category_set.add(primary_category)
                    processed_labels.append(primary_category)
                    
                # Multi-label with separators
                elif any(sep in label_str for sep in ['|', ';', ',', '/']):
                    for sep in ['|', ';', ',', '/']:
                        if sep in label_str:
                            first_label = label_str.split(sep)[0].strip()
                            # Apply further processing if needed
                            if '.' in first_label:
                                first_label = first_label.split('.')[0]
                            category_set.add(first_label)
                            processed_labels.append(first_label)
                            break
                
                # Numeric labels
                elif label_str.replace('-','').replace('_','').replace('.','').isdigit():
                    numeric_label = f"class_{label_str}"
                    category_set.add(numeric_label)
                    processed_labels.append(numeric_label)
                
                # Text labels with special handling
                else:
                    # Clean text labels
                    clean_label = label_str.lower().strip()
                    
                    # Handle underscores and mixed formatting
                    if '_' in clean_label:
                        clean_label = clean_label.replace('_', ' ')
                    
                    # Remove extra spaces
                    clean_label = ' '.join(clean_label.split())
                    
                    # Handle empty results
                    if not clean_label:
                        clean_label = 'unknown_label'
                    
                    category_set.add(clean_label)
                    processed_labels.append(clean_label)
                
                # Show progress in terminal for large datasets
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{len(label_series)} labels...")
            
            # Convert to pandas Series for easier manipulation
            processed_series = pd.Series(processed_labels)
            
            # Print results to terminal
            unique_labels = sorted(list(category_set), key=lambda x: x.lower())
            print(f"✅ Universal label preprocessing completed!")
            print(f"🏷️ Found {len(unique_labels)} unique labels")
            print(f"🎯 Primary method: {processing_method}")
            
            # Print sample conversions to terminal
            print("🔍 Sample label conversions:")
            sample_original = label_series.head(5).tolist()
            sample_processed = processed_labels[:5]
            for orig, proc in zip(sample_original, sample_processed):
                print(f"  '{orig}' → '{proc}'")
            
            # Print top distribution to terminal
            label_counts = processed_series.value_counts().head(10)
            print("📈 Top 10 Label Distribution:")
            for label, count in label_counts.items():
                percentage = (count / len(processed_series)) * 100
                print(f"  • {label}: {count} ({percentage:.1f}%)")
            
            # No processing info shown in Streamlit for label preprocessing
            
            return processed_series
            
        except Exception as e:
            st.error(f"Error in universal label preprocessing: {e}")
            print(f"❌ Error in universal label preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return label_series
    
    def _try_label_consolidation(self, df):
        """Try to consolidate similar labels to increase sample counts"""
        try:
            label_counts = df['label'].value_counts()
            insufficient_labels = label_counts[label_counts < 2].index.tolist()
            
            # Simple consolidation strategy: group by first few characters
            consolidation_map = {}
            for label in insufficient_labels:
                # Try to find similar labels with sufficient samples
                label_prefix = label[:3].lower()  # First 3 chars
                similar_labels = [l for l in label_counts.index 
                                 if l.lower().startswith(label_prefix) and label_counts[l] >= 2]
                
                if similar_labels:
                    # Consolidate with the most frequent similar label
                    target_label = max(similar_labels, key=lambda x: label_counts[x])
                    consolidation_map[label] = target_label
                    print(f"    • Consolidating '{label}' → '{target_label}'")
            
            if consolidation_map:
                # Apply consolidation
                df['label'] = df['label'].replace(consolidation_map)
                print(f"✅ Consolidated {len(consolidation_map)} labels")
                return True
            else:
                print("❌ No consolidation opportunities found")
                return False
                
        except Exception as e:
            print(f"❌ Label consolidation failed: {e}")
            return False
    
    def _try_top_labels_only(self, df, min_labels=5):
        """Keep only the top N most frequent labels"""
        try:
            label_counts = df['label'].value_counts()
            
            # Find labels with sufficient samples
            sufficient_labels = label_counts[label_counts >= 2]
            
            if len(sufficient_labels) >= min_labels:
                # Keep top labels that have sufficient samples
                top_labels = sufficient_labels.head(min_labels).index.tolist()
                
                # Filter dataframe to keep only top labels
                mask_top_labels = df['label'].isin(top_labels)
                original_len = len(df)
                df.drop(df[~mask_top_labels].index, inplace=True)
                
                print(f"✅ Kept top {len(top_labels)} labels, reduced from {original_len} to {len(df)} samples")
                print(f"📊 Kept labels: {top_labels}")
                return True
            
            print(f"❌ Insufficient labels with adequate samples (need {min_labels}, found {len(sufficient_labels)})")
            return False
            
        except Exception as e:
            print(f"❌ Top labels strategy failed: {e}")
            return False

    def preprocess_dataset(self, df, max_samples=None):
        """Standard dataset preprocessing pipeline"""
        try:
            # Print header to terminal only
            print("🔧 Starting Standard Dataset Preprocessing Pipeline...")
            print("📊 Applying comprehensive data preprocessing for optimal classification performance")
            
            # Phase 1: Data Extraction and Basic Cleaning (terminal only)
            print("📋 Phase 1: Data Extraction and Basic Cleaning")
            raw_df = pd.DataFrame({
                'text': df[self.selected_text_column],
                'label': df[self.selected_label_column]
            })
            
            initial_count = len(raw_df)
            print(f"📊 Initial dataset: {initial_count} samples")
            
            # Remove completely empty rows
            raw_df = raw_df.dropna(subset=['text', 'label'])
            after_na_removal = len(raw_df)
            print(f"📊 After removing missing data: {after_na_removal} samples ({initial_count - after_na_removal} removed)")
            
            # Phase 2: Text Preprocessing (terminal only)
            print("📝 Phase 2: Text Preprocessing")
            processed_text = self.standardize_text_preprocessing(raw_df['text'])
            
            # Phase 3: Label Preprocessing (terminal only)
            print("🏷️ Phase 3: Universal Label Preprocessing")
            processed_labels = self.universal_label_preprocessing(raw_df['label'])
            
            # Phase 4: Final Dataset Assembly (terminal only)
            print("📦 Phase 4: Final Dataset Assembly")
            processed_df = pd.DataFrame({
                'text': processed_text,
                'label': processed_labels
            })
            
            # Remove samples with processed empty content
            mask_valid = ~processed_df['text'].isin(['empty_text', 'short_text_removed'])
            mask_valid &= ~processed_df['label'].isin(['unknown_label'])
            
            if mask_valid.sum() < len(processed_df):
                filtered_count = len(processed_df) - mask_valid.sum()
                print(f"⚠️ Filtering out {filtered_count} samples with invalid content")
                processed_df = processed_df[mask_valid]
            
            # Phase 5: Label Balancing and Validation (terminal only)
            print("⚖️ Phase 5: Label Balancing and Validation")
            
            # Iterative label cleaning - may need multiple passes
            iteration = 1
            max_iterations = 5
            
            while iteration <= max_iterations:
                print(f"🔄 Label validation iteration {iteration}:")
                
                # Check label distribution
                label_counts = processed_df['label'].value_counts()
                print(f"  Total samples: {len(processed_df)}")
                print(f"  Unique labels: {len(label_counts)}")
                
                # Find labels with insufficient samples
                min_samples_per_class = 2  # Minimum for train_test_split
                insufficient_labels = label_counts[label_counts < min_samples_per_class]
                
                if len(insufficient_labels) == 0:
                    print("✅ All labels have sufficient samples!")
                    break
                
                print(f"⚠️ Found {len(insufficient_labels)} labels with < {min_samples_per_class} samples:")
                for label, count in insufficient_labels.items():
                    print(f"    • {label}: {count} sample(s)")
                
                # Check if we can safely remove these labels
                remaining_labels_count = len(label_counts) - len(insufficient_labels)
                if remaining_labels_count >= 2:
                    print(f"🔧 Removing {len(insufficient_labels)} insufficient labels...")
                    mask_sufficient = ~processed_df['label'].isin(insufficient_labels.index)
                    processed_df = processed_df[mask_sufficient]
                    removed_samples = insufficient_labels.sum()
                    print(f"✅ Removed {len(insufficient_labels)} labels ({removed_samples} samples)")
                    
                    # Recompute counts for next iteration
                    label_counts = processed_df['label'].value_counts()
                    iteration += 1
                else:
                    print("❌ Too many labels have insufficient samples. Trying alternative strategies...")
                    
                    # Alternative Strategy 1: Consolidate similar labels
                    print("🔄 Alternative Strategy 1: Label consolidation")
                    if self._try_label_consolidation(processed_df):
                        print("✅ Label consolidation successful")
                        iteration += 1
                        continue
                    
                    # Alternative Strategy 2: Use top N most frequent labels only
                    print("🔄 Alternative Strategy 2: Keep only top labels")
                    if self._try_top_labels_only(processed_df, min_labels=5):
                        print("✅ Top labels strategy successful")
                        iteration += 1
                        continue
                    
                    # Final fallback
                    print("❌ All strategies failed. Cannot proceed with classification.")
                    st.error("❌ Dataset has too many labels with insufficient samples. Cannot proceed.")
                    return None
            
            if iteration > max_iterations:
                print(f"❌ Maximum iterations ({max_iterations}) reached. Cannot balance labels.")
                st.error("❌ Could not balance labels after multiple attempts.")
                return None
            
            # Final validation
            final_label_counts = processed_df['label'].value_counts()
            print(f"📊 Final label distribution after balancing:")
            for label, count in final_label_counts.head(10).items():
                percentage = (count / len(processed_df)) * 100
                print(f"  • {label}: {count} samples ({percentage:.1f}%)")
            if len(final_label_counts) > 10:
                print(f"  ... and {len(final_label_counts) - 10} more labels")
            
            # Phase 6: Quality Assessment (terminal only)
            print("📊 Phase 6: Quality Assessment")
            final_count = len(processed_df)
            unique_labels = processed_df['label'].unique()
            final_label_counts = processed_df['label'].value_counts()
            
            # Print detailed analysis to terminal
            print("📊 Final Quality Assessment:")
            print(f"  Final samples: {final_count}")
            print(f"  Unique labels: {len(unique_labels)}")
            retention_rate = (final_count / initial_count * 100) if initial_count > 0 else 0
            print(f"  Retention rate: {retention_rate:.1f}%")
            min_class_size = final_label_counts.min()
            print(f"  Min class size: {min_class_size}")
            
            # Print sample data to terminal
            print("👀 Sample processed data:")
            sample_df = processed_df.head(3)[['text', 'label']]
            for idx, row in sample_df.iterrows():
                text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
                print(f"  Text: {text_preview}")
                print(f"  Label: {row['label']}")
                print("  ---")
            
            # Check quality issues and print to terminal
            if len(unique_labels) < 2:
                print("❌ Dataset has less than 2 unique labels. Classification cannot proceed.")
                st.error("❌ Dataset has less than 2 unique labels. Classification cannot proceed.")
                return None
            elif len(unique_labels) > 50:
                print("⚠️ Dataset has many labels (>50). Consider label consolidation.")
            
            if final_count < 50:
                print("⚠️ Small dataset (<50 samples). Results may not be reliable.")
            
            # Check class balance
            max_class_size = final_label_counts.max()
            min_class_size = final_label_counts.min()
            imbalance_ratio = max_class_size / min_class_size if min_class_size > 0 else float('inf')
            
            if imbalance_ratio > 10:
                print(f"⚖️ High class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
                print("💡 Consider using balanced sampling or weighted classification")
            
            # Print final distribution to terminal
            print("📈 Final Label Distribution:")
            for label, count in final_label_counts.head(10).items():
                percentage = (count / final_count) * 100
                print(f"  • {label}: {count} ({percentage:.1f}%)")
            
            print("📊 Distribution Statistics:")
            print(f"  • Mean samples per class: {final_label_counts.mean():.1f}")
            print(f"  • Median samples per class: {final_label_counts.median():.1f}")
            print(f"  • Standard deviation: {final_label_counts.std():.1f}")
            print(f"  • Imbalance ratio: {imbalance_ratio:.1f}:1")
            
            # Phase 7: Performance Optimization (terminal only)
            print("🚀 Phase 7: Performance Optimization")
            
            # Use custom max_samples if provided, otherwise use config default
            effective_max_samples = max_samples if max_samples is not None else MAX_SAMPLES
            
            if final_count > effective_max_samples:
                print(f"🔄 Dataset is large ({final_count} samples). Sampling {effective_max_samples:,} for performance...")
                
                # Stratified sampling to maintain label distribution
                try:
                    from sklearn.model_selection import train_test_split
                    print("📊 Applying stratified sampling to maintain label distribution...")
                    # Use stratified sampling to maintain proportions
                    sample_df, _ = train_test_split(
                        processed_df, 
                        test_size=(final_count - effective_max_samples) / final_count,
                        stratify=processed_df['label'],
                        random_state=RANDOM_STATE
                    )
                    processed_df = sample_df
                    print("✅ Applied stratified sampling successfully")
                except Exception as e:
                    # Fallback to simple random sampling
                    print("⚠️ Stratified sampling failed, using random sampling")
                    processed_df = processed_df.sample(n=effective_max_samples, random_state=RANDOM_STATE)
                
                print(f"📊 Using {len(processed_df)} samples for classification")
                
                # Show final distribution after sampling in terminal
                final_sampled_counts = processed_df['label'].value_counts()
                print("📊 Distribution after sampling:")
                for label, count in final_sampled_counts.head(10).items():
                    percentage = (count / len(processed_df)) * 100
                    print(f"  • {label}: {count} ({percentage:.1f}%)")
            else:
                print("✅ Dataset size is optimal for processing")
            
            # Final validation with automatic fixing
            final_label_counts_check = processed_df['label'].value_counts()
            min_final_class_size = final_label_counts_check.min()
            
            print(f"🔍 Final validation - min class size: {min_final_class_size}")
            
            if min_final_class_size < 2:
                print("⚠️ After sampling, some classes have < 2 samples. Auto-fixing...")
                insufficient_final = final_label_counts_check[final_label_counts_check < 2]
                print("Classes with insufficient samples after sampling:")
                for label, count in insufficient_final.items():
                    print(f"  • {label}: {count} sample(s)")
                
                # Auto-fix by removing insufficient labels
                mask_sufficient_final = ~processed_df['label'].isin(insufficient_final.index)
                processed_df = processed_df[mask_sufficient_final]
                removed_final_samples = insufficient_final.sum()
                print(f"🔧 Auto-removed {len(insufficient_final)} labels ({removed_final_samples} samples)")
                
                # Re-check after auto-fix
                final_label_counts_check = processed_df['label'].value_counts()
                min_final_class_size = final_label_counts_check.min()
                
                if len(final_label_counts_check) < 2:
                    print("❌ Less than 2 labels remaining after auto-fix. Cannot proceed.")
                    st.error("❌ Dataset has insufficient labels for classification.")
                    return None
                elif min_final_class_size < 2:
                    print("❌ Auto-fix failed. Still have labels with < 2 samples.")
                    st.error("❌ Cannot resolve label imbalance. Try a different dataset.")
                    return None
                else:
                    print(f"✅ Auto-fix successful! Min class size now: {min_final_class_size}")
            else:
                print("✅ All classes have sufficient samples for classification!")
            
            # Final success message (print to terminal)
            print("🎉 Standard preprocessing pipeline completed successfully!")
            print(f"✅ Ready for classification with {len(processed_df)} samples and {len(final_label_counts_check)} balanced labels")
            
            # Only show final result in Streamlit - no processing details
            st.toast("✅ Dataset processed successfully! Ready for classification")

            
            return processed_df
        except Exception as e:
            st.error(f"Error preprocessing dataset: {e}")
            return None
    
    def run_auto_classification(self, df):
        """Run automatic classification with all methods"""
        try:
            # Print classification info to terminal only
            print("🤖 Starting Automatic Classification")
            
            # Final validation before classification (terminal only)
            label_counts = df['label'].value_counts()
            min_samples = label_counts.min()
            
            print(f"📊 Pre-classification check: {len(df)} samples, {len(label_counts)} labels")
            print(f"📊 Minimum samples per class: {min_samples}")
            
            if min_samples < 2:
                print("❌ Cannot proceed: Some labels have less than 2 samples")
                problematic_labels = label_counts[label_counts < 2]
                print("Problematic labels:")
                for label, count in problematic_labels.items():
                    print(f"  • {label}: {count} sample(s)")
                st.error("❌ Cannot proceed: Some labels have less than 2 samples")
                return None, None, None
            
            # Prepare data
            X = df['text'].tolist()
            y = df['label'].tolist()
            
            # Create label mappings
            unique_labels = sorted(list(set(y)))
            label_to_id = {label: i for i, label in enumerate(unique_labels)}
            y_encoded = [label_to_id[label] for label in y]
            
            # Check if we have enough samples for train/test split (terminal only)
            total_samples = len(df)
            test_size = TEST_SIZE
            
            # Adjust test size for small datasets
            if total_samples < 10:
                test_size = 0.3  # Use smaller test set for very small datasets
                print(f"⚠️ Small dataset detected. Using {test_size*100}% for testing.")
            
            # Split data with enhanced error handling (terminal only)
            from sklearn.model_selection import train_test_split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, 
                    random_state=RANDOM_STATE, stratify=y_encoded
                )
                print(f"✅ Data split: {len(X_train)} training, {len(X_test)} testing samples")
                
            except ValueError as e:
                if "The least populated class in y has only 1 member" in str(e):
                    print("❌ Cannot split data: Some classes have insufficient samples for train/test split")
                    print("💡 This error occurs when labels have very few samples")
                    
                    # Show current distribution in terminal
                    print("Current label distribution:")
                    for label, count in label_counts.items():
                        print(f"  • {label}: {count} samples")
                    
                    # Try without stratification as fallback
                    print("🔄 Attempting split without stratification...")
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y_encoded, test_size=test_size, 
                            random_state=RANDOM_STATE  # Remove stratify
                        )
                        print("✅ Data split successful without stratification")
                    except Exception as e2:
                        print(f"❌ Split failed completely: {e2}")
                        st.error(f"❌ Data split failed: {e2}")
                        return None, None, None
                else:
                    raise e
            
            # Text vectorization (terminal only)
            print("🔧 Processing text vectorization...")
            
            # BoW (terminal only)
            print("🔧 Processing Bag of Words...")
            X_train_bow = self.text_vectorizer.fit_transform_bow(X_train)
            X_test_bow = self.text_vectorizer.transform_bow(X_test)
            print(f"✅ BoW vectorization completed | Shape: {X_train_bow.shape} | Sparse: {hasattr(X_train_bow, 'nnz')}")
            
            # TF-IDF (terminal only)
            print("🔧 Processing TF-IDF...")
            X_train_tfidf = self.text_vectorizer.fit_transform_tfidf(X_train)
            X_test_tfidf = self.text_vectorizer.transform_tfidf(X_test)
            print(f"✅ TF-IDF vectorization completed | Shape: {X_train_tfidf.shape} | Sparse: {hasattr(X_train_tfidf, 'nnz')}")
            
            # Embeddings (terminal only)
            print("🔧 Processing Word Embeddings...")
            print("📊 Using sentence-transformers for high-quality embeddings")
            
            print("🔄 Processing training set embeddings...")
            X_train_embeddings = self.text_vectorizer.transform_embeddings(X_train)
            
            print("🔄 Processing test set embeddings...")
            X_test_embeddings = self.text_vectorizer.transform_embeddings(X_test)
            
            print(f"✅ Word embeddings completed | Train shape: {X_train_embeddings.shape} | Test shape: {X_test_embeddings.shape}")
            
            # Train and evaluate all models (terminal only)
            print("🤖 Training and evaluating models...")
            results = {}
            
            # K-Means (terminal only)
            print("🤖 Training K-Means models...")
            km_bow_labels, km_bow_accuracy, km_bow_report = (
                self.model_trainer.train_and_test_kmeans(
                    X_train_bow, y_train, X_test_bow, y_test
                )
            )
            results['kmeans_bow'] = {
                'accuracy': km_bow_accuracy,
                'predictions': km_bow_labels,
                'report': km_bow_report,
                'vectorizer': 'BoW'
            }
            print(f"  ✅ K-Means + BoW: {km_bow_accuracy:.3f}")
            
            km_tfidf_labels, km_tfidf_accuracy, km_tfidf_report = (
                self.model_trainer.train_and_test_kmeans(
                    X_train_tfidf, y_train, X_test_tfidf, y_test
                )
            )
            results['kmeans_tfidf'] = {
                'accuracy': km_tfidf_accuracy,
                'predictions': km_tfidf_labels,
                'report': km_tfidf_report,
                'vectorizer': 'TF-IDF'
            }
            print(f"  ✅ K-Means + TF-IDF: {km_tfidf_accuracy:.3f}")
            
            km_emb_labels, km_emb_accuracy, km_emb_report = (
                self.model_trainer.train_and_test_kmeans(
                    X_train_embeddings, y_train, X_test_embeddings, y_test
                )
            )
            results['kmeans_embeddings'] = {
                'accuracy': km_emb_accuracy,
                'predictions': km_emb_labels,
                'report': km_emb_report,
                'vectorizer': 'Embeddings'
            }
            print(f"  ✅ K-Means + Embeddings: {km_emb_accuracy:.3f}")
            
            # KNN (terminal only)
            print("🤖 Training KNN models...")
            knn_bow_labels, knn_bow_accuracy, knn_bow_report = (
                self.model_trainer.train_and_test_knn(
                    X_train_bow, y_train, X_test_bow, y_test
                )
            )
            results['knn_bow'] = {
                'accuracy': knn_bow_accuracy,
                'predictions': knn_bow_labels,
                'report': knn_bow_report,
                'vectorizer': 'BoW'
            }
            print(f"  ✅ KNN + BoW: {knn_bow_accuracy:.3f}")
            
            knn_tfidf_labels, knn_tfidf_accuracy, knn_tfidf_report = (
                self.model_trainer.train_and_test_knn(
                    X_train_tfidf, y_train, X_test_tfidf, y_test
                )
            )
            results['knn_tfidf'] = {
                'accuracy': knn_tfidf_accuracy,
                'predictions': knn_tfidf_labels,
                'report': knn_tfidf_report,
                'vectorizer': 'TF-IDF'
            }
            print(f"  ✅ KNN + TF-IDF: {knn_tfidf_accuracy:.3f}")
            
            knn_emb_labels, knn_emb_accuracy, knn_emb_report = (
                self.model_trainer.train_and_test_knn(
                    X_train_embeddings, y_train, X_test_embeddings, y_test
                )
            )
            results['knn_embeddings'] = {
                'accuracy': knn_emb_accuracy,
                'predictions': knn_emb_labels,
                'report': knn_emb_report,
                'vectorizer': 'Embeddings'
            }
            print(f"  ✅ KNN + Embeddings: {knn_emb_accuracy:.3f}")
            
            # Decision Tree (terminal only)
            print("🤖 Training Decision Tree models...")
            dt_bow_labels, dt_bow_accuracy, dt_bow_report = (
                self.model_trainer.train_and_test_decision_tree(
                    X_train_bow, y_train, X_test_bow, y_test
                )
            )
            results['decision_tree_bow'] = {
                'accuracy': dt_bow_accuracy,
                'predictions': dt_bow_labels,
                'report': dt_bow_report,
                'vectorizer': 'BoW'
            }
            print(f"  ✅ Decision Tree + BoW: {dt_bow_accuracy:.3f}")
            
            dt_tfidf_labels, dt_tfidf_accuracy, dt_tfidf_report = (
                self.model_trainer.train_and_test_decision_tree(
                    X_train_tfidf, y_train, X_test_tfidf, y_test
                )
            )
            results['decision_tree_tfidf'] = {
                'accuracy': dt_tfidf_accuracy,
                'predictions': dt_tfidf_labels,
                'report': dt_tfidf_report,
                'vectorizer': 'TF-IDF'
            }
            print(f"  ✅ Decision Tree + TF-IDF: {dt_tfidf_accuracy:.3f}")
            
            dt_emb_labels, dt_emb_accuracy, dt_emb_report = (
                self.model_trainer.train_and_test_decision_tree(
                    X_train_embeddings, y_train, X_test_embeddings, y_test
                )
            )
            results['decision_tree_embeddings'] = {
                'accuracy': dt_emb_accuracy,
                'predictions': dt_emb_labels,
                'report': dt_emb_report,
                'vectorizer': 'Embeddings'
            }
            print(f"  ✅ Decision Tree + Embeddings: {dt_emb_accuracy:.3f}")
            
            # Naive Bayes (terminal only)
            print("🤖 Training Naive Bayes models...")
            nb_bow_labels, nb_bow_accuracy, nb_bow_report = (
                self.model_trainer.train_and_test_naive_bayes(
                    X_train_bow, y_train, X_test_bow, y_test
                )
            )
            results['naive_bayes_bow'] = {
                'accuracy': nb_bow_accuracy,
                'predictions': nb_bow_labels,
                'report': nb_bow_report,
                'vectorizer': 'BoW'
            }
            print(f"  ✅ Naive Bayes + BoW: {nb_bow_accuracy:.3f}")
            
            nb_tfidf_labels, nb_tfidf_accuracy, nb_tfidf_report = (
                self.model_trainer.train_and_test_naive_bayes(
                    X_train_tfidf, y_train, X_test_tfidf, y_test
                )
            )
            results['nb_tfidf'] = {
                'accuracy': nb_tfidf_accuracy,
                'predictions': nb_tfidf_labels,
                'report': nb_tfidf_report,
                'vectorizer': 'TF-IDF'
            }
            print(f"  ✅ Naive Bayes + TF-IDF: {nb_tfidf_accuracy:.3f}")
            
            nb_emb_labels, nb_emb_accuracy, nb_emb_report = (
                self.model_trainer.train_and_test_naive_bayes(
                    X_train_embeddings, y_train, X_test_embeddings, y_test
                )
            )
            results['naive_bayes_embeddings'] = {
                'accuracy': nb_emb_accuracy,
                'predictions': nb_emb_labels,
                'report': nb_emb_report,
                'vectorizer': 'Embeddings'
            }
            print(f"  ✅ Naive Bayes + Embeddings: {nb_emb_accuracy:.3f}")
            
            print("🎉 All model training completed!")
            
            return results, y_test, unique_labels
            
        except Exception as e:
            st.error(f"Error in auto classification: {e}")
            return None, None, None
    
    def find_best_classifier(self, results):
        """Find the best performing classifier"""
        best_model = None
        best_accuracy = 0
        
        for model_name, result in results.items():
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model = model_name
        
        return best_model, best_accuracy
    
    def plot_confusion_matrix_plotly(self, y_true, y_pred, labels, title):
        """Create interactive confusion matrix using Plotly"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create annotations
        annotations = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                annotations.append(
                    dict(
                        text=f"{cm[i, j]}<br>({cm_normalized[i, j]:.2%})",
                        x=j, y=i,
                        xref='x', yref='y',
                        showarrow=False,
                        font=dict(color="white" if cm_normalized[i, j] > 0.5 else "black")
                    )
                )
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=600,
            height=500
        )
        
        return fig
    
    def create_performance_comparison(self, results):
        """Create performance comparison chart"""
        models = []
        accuracies = []
        vectorizers = []
        
        for model_name, result in results.items():
            if '_' in model_name:
                model_type = (model_name.split('_')[0] + ' ' + 
                            model_name.split('_')[1])
            else:
                model_type = model_name
            models.append(model_type)
            accuracies.append(result['accuracy'])
            vectorizers.append(result['vectorizer'])
        
        df_compare = pd.DataFrame({
            'Model': models,
            'Accuracy': accuracies,
            'Vectorizer': vectorizers
        })
        
        fig = px.bar(
            df_compare, 
            x='Model', 
            y='Accuracy',
            color='Vectorizer',
            title="Model Performance Comparison",
            barmode='group'
        )
        
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Accuracy",
            yaxis_range=[0, 1],
            width=800,
            height=500
        )
        
        return fig


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Topic Modeling - Auto Classifier</h1>', 
                unsafe_allow_html=True)
    
    # Initialize app
    app = StreamlitTopicModeling()
    
    # Initialize session state for dataset persistence
    if 'loaded_dataset' not in st.session_state:
        st.session_state.loaded_dataset = None
        st.session_state.dataset_source = None
    
    # Sidebar for dataset input
    st.sidebar.header("📁 Dataset Input")
    
    input_method = st.sidebar.selectbox(
        "Choose input method:",
        ["Upload File", "Enter File Path", "Enter URL", "Use Sample Dataset"]
    )
    
    dataset = st.session_state.loaded_dataset
    
    if input_method == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your dataset",
            type=['csv', 'json', 'xlsx'],
            help="Upload CSV, JSON, or Excel file with text and label columns"
        )
        
        if uploaded_file is not None:
            temp_dataset = app.load_dataset_from_file(uploaded_file)
            if temp_dataset is not None:
                st.session_state.loaded_dataset = temp_dataset
                st.session_state.dataset_source = f"Upload: {uploaded_file.name}"
                dataset = temp_dataset
                st.sidebar.success(f"✅ File loaded: {uploaded_file.name}")
                st.sidebar.info(f"📊 Dataset shape: {dataset.shape}")
    
    elif input_method == "Enter File Path":
        file_path = st.sidebar.text_input(
            "Enter local file path:",
            placeholder="C:/Users/Username/Documents/dataset.csv",
            help="Enter the full path to your dataset file on your local machine"
        )
        
        if file_path and st.sidebar.button("Load from Path"):
            with st.spinner("Loading dataset from local path..."):
                temp_dataset = app.load_dataset_from_path(file_path)
                if temp_dataset is not None:
                    st.session_state.loaded_dataset = temp_dataset
                    st.session_state.dataset_source = f"File Path: {file_path}"
                    dataset = temp_dataset
                    st.sidebar.success("✅ Dataset loaded from local path")
                    st.sidebar.info(f"📊 Dataset shape: {dataset.shape}")
                    st.rerun()  # Force rerun to update main area
    
    elif input_method == "Enter URL":
        url = st.sidebar.text_input(
            "Enter dataset URL:",
            placeholder="https://example.com/dataset.csv",
            help="Enter URL to CSV or JSON file"
        )
        
        if url and st.sidebar.button("Load from URL"):
            with st.spinner("Loading dataset from URL..."):
                temp_dataset = app.load_dataset_from_url(url)
                if temp_dataset is not None:
                    st.session_state.loaded_dataset = temp_dataset
                    st.session_state.dataset_source = f"URL: {url}"
                    dataset = temp_dataset
                    st.sidebar.success("✅ Dataset loaded from URL")
                    st.sidebar.info(f"📊 Dataset shape: {dataset.shape}")
                    st.rerun()  # Force rerun to update main area
    
    elif input_method == "Use Sample Dataset":
        if st.sidebar.button("Load Sample Dataset"):
            st.sidebar.info("Loading ArXiv dataset...")
            if app.initialize_components():
                try:
                    app.data_loader.load_dataset()
                    app.data_loader.select_samples()
                    app.data_loader.preprocess_samples()
                    
                    # Convert to DataFrame
                    sample_data = app.data_loader.preprocessed_samples
                    temp_dataset = pd.DataFrame(sample_data)
                    
                    st.session_state.loaded_dataset = temp_dataset
                    st.session_state.dataset_source = "Sample: ArXiv Dataset"
                    dataset = temp_dataset
                    st.sidebar.success("✅ Sample dataset loaded")
                    st.sidebar.info(f"📊 Dataset shape: {dataset.shape}")
                    st.rerun()  # Force rerun to update main area
                except Exception as e:
                    st.sidebar.error(f"Error loading sample dataset: {e}")
    
    # Main content area
    if dataset is not None:
        # Dataset preview in collapsible section
        with st.expander("📊 Dataset Preview", expanded=False):
            if st.session_state.dataset_source:
                st.info(f"📁 Dataset Source: {st.session_state.dataset_source}")
            st.dataframe(dataset.head(10), use_container_width=True)
            
            # Dataset statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(dataset))
            with col2:
                st.metric("Total Columns", len(dataset.columns))
            with col3:
                st.metric("Memory Usage", f"{dataset.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Column info
            st.write("**Columns:**")
            columns_info = []
            for col in dataset.columns:
                dtype = str(dataset[col].dtype)
                null_count = dataset[col].isnull().sum()
                columns_info.append(f"• `{col}` ({dtype}) - {null_count} nulls")
            
            # Display columns in multiple columns for better layout
            col_chunks = [columns_info[i:i+3] for i in range(0, len(columns_info), 3)]
            for chunk in col_chunks:
                cols = st.columns(min(3, len(chunk)))
                for i, col_info in enumerate(chunk):
                    with cols[i]:
                        st.write(col_info)
        
        # Column selection in collapsible section
        with st.expander("🔧 Column Selection", expanded=True):
            app.select_dataset_columns(dataset)
        
        # Check if both columns are selected and different
        if (hasattr(app, 'selected_text_column') and hasattr(app, 'selected_label_column') and
            app.selected_text_column and app.selected_label_column and
            app.selected_text_column != app.selected_label_column):
            
            # Show start classification button in collapsible section
            with st.expander("🚀 Start Classification", expanded=True):
                st.write("Click the button below to begin the automatic classification process:")
                
                # Sample size input
                col1, col2 = st.columns([2, 1])
                with col1:
                    # Use actual dataset size as max_value
                    dataset_size = len(dataset)
                    default_value = min(100000, dataset_size)  # Use 100K or dataset size, whichever is smaller
                    
                    sample_size = st.number_input(
                        "📊 Number of samples to process:",
                        min_value=min(1000, dataset_size),  # At least 1K or dataset size if smaller
                        max_value=dataset_size,             # Maximum is actual dataset size
                        value=default_value,
                        step=1000,
                        help=f"Enter the number of samples to use for classification. Dataset has {dataset_size:,} samples total. More samples = better accuracy but slower processing."
                    )
                with col2:
                    st.metric("Selected", f"{sample_size:,}")
                    st.caption(f"of {dataset_size:,} total")
                
                if st.button("🚀 Start Auto Classification", type="primary", key="start_classification"):
                    with st.spinner("Preprocessing dataset..."):
                        processed_dataset = app.preprocess_dataset(dataset, max_samples=sample_size)
                        
                        if processed_dataset is not None:
                            # Initialize components
                            if app.initialize_components():
                                # Run classification
                                with st.spinner("🤖 Running automatic classification..."):
                                    results, y_test, labels = app.run_auto_classification(processed_dataset)
                                
                                if results:
                                    # Find best classifier
                                    best_model, best_accuracy = app.find_best_classifier(results)
                                    
                                    # Display results in collapsible section
                                    st.toast("🎉 Classification completed successfully!", icon="🎉")
                                    
                                    with st.expander("🏆 Classification Results", expanded=True):
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric("Best Model", best_model.replace('_', ' ').title())
                                        with col2:
                                            st.metric("Best Accuracy", f"{best_accuracy:.3f}")
                                        
                                        # Performance comparison
                                        st.subheader("📈 Performance Comparison")
                                        comparison_fig = app.create_performance_comparison(results)
                                        st.plotly_chart(comparison_fig, use_container_width=True)
                                else:
                                    st.error("❌ Classification failed!")
                            else:
                                st.error("❌ Failed to initialize components!")
                        else:
                            st.error("❌ Failed to preprocess dataset!")
        else:
            if hasattr(app, 'selected_text_column') and hasattr(app, 'selected_label_column'):
                if app.selected_text_column == app.selected_label_column:
                    st.warning("⚠️ Please select different columns for text and label.")
                else:
                    st.info("ℹ️ Please select both text and label columns to continue.")
            else:
                st.info("ℹ️ Please select columns above to start classification.")
    
    else:
        # Empty state - no dataset loaded
        pass


if __name__ == "__main__":
    main()
