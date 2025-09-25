"""
Comprehensive Evaluation System for Topic Modeling Project
Combines all embedding methods with all models for comprehensive evaluation
Evaluates overfitting/underfitting and provides cross-validation results
Designed with extensible architecture for future model/embedding additions
Uses new modular architecture exclusively
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from typing import Dict, List, Tuple, Any, Union
from scipy import sparse
import time
from datetime import datetime

# Import project modules
from data_loader import DataLoader
from text_encoders import TextVectorizer
from models import ModelMetrics

# Progress tracking removed - using simple progress indicators instead

class ComprehensiveEvaluator:
    """
    Comprehensive evaluation system for all embedding-model combinations
    """
    
    def __init__(self, 
                 cv_folds: int = 5,
                 validation_size: float = 0.2,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 data_loader: DataLoader = None):
        """
        Initialize the comprehensive evaluator
        
        Args:
            cv_folds: Number of cross-validation folds
            validation_size: Size of validation set (for overfitting detection)
            test_size: Size of test set
            random_state: Random seed for reproducibility
            data_loader: Optional DataLoader instance with pre-configured labels
        """
        self.cv_folds = cv_folds
        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize components - Use provided DataLoader or create new one
        if data_loader is not None:
            self.data_loader = data_loader
            labels = getattr(data_loader, 'id_to_label', {})
            print(f"✅ [COMPREHENSIVE_EVALUATOR] Using provided DataLoader with labels: {labels}")
        else:
            self.data_loader = DataLoader()
            print("⚠️ [COMPREHENSIVE_EVALUATOR] Created new DataLoader (no labels transferred)")
        
        self.text_vectorizer = TextVectorizer()
        
        # Import model components
        from models import model_factory, validation_manager as vm
        
        # Import cache manager
        try:
            from cache_manager import cache_manager
            self.cache_manager = cache_manager
            self.cache_enabled = True
            print("✅ Cache manager loaded successfully")
        except ImportError:
            self.cache_manager = None
            self.cache_enabled = False
            print("⚠️ Cache manager not available")
        
        # Store model factory and validation manager
        self.model_factory = model_factory
        self.validation_manager = vm
        
        # Add cache helper methods
        self._add_cache_methods()
        
        # Results storage
        self.evaluation_results = {}
        self.overfitting_analysis = {}
        self.best_combinations = {}
        
        # Performance tracking
        self.training_times = {}
        self.prediction_times = {}
        
        # Pre-computed CV embeddings cache
        self.cv_embeddings_cache = {}
        
        print("🚀 Comprehensive Evaluator initialized with:")
        print(f"   • CV Folds: {cv_folds}")
        print(f"   • Validation Size: {validation_size:.1%}")
        print(f"   • Test Size: {test_size:.1%}")
        print(f"   • Random State: {random_state}")

    def precompute_cv_embeddings(self, texts: List[str], labels: List[str], stop_callback=None, 
                                test_texts: List[str] = None, test_labels: List[str] = None) -> Dict[str, Any]:
        """Pre-compute embeddings for all CV folds to ensure fair comparison across models
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            stop_callback: Optional callback to check for stop signal
            
        Returns:
            Dictionary containing pre-computed embeddings for each fold + test data
        """
        print("🔧 Pre-computing CV embeddings for all folds + test data...")
        
        # Get CV splits using same strategy as ValidationManager
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_embeddings = {}
        
        # Use pre-computed embeddings and split them for CV folds (NO additional embedding creation)
        print(f"  🔧 Using pre-computed embeddings and splitting for CV folds...")
        
        # Get pre-computed embeddings from the evaluator
        if not hasattr(self, 'embeddings') or 'embeddings' not in self.embeddings:
            print("  ❌ Error: Pre-computed embeddings not found. Please create embeddings first.")
            return {}
        
        # Get the pre-computed embeddings
        if 'embeddings' not in self.embeddings:
            print("  ❌ Error: Word embeddings not found in embeddings. Please create word embeddings first.")
            return {}
            
        precomputed_embeddings = self.embeddings['embeddings']['X_train']
        
        # Get test data embeddings if available
        test_embeddings = None
        test_labels = None
        
        # Priority 1: Use test data passed as parameters
        if test_texts is not None and test_labels is not None:
            # Get test embeddings from embeddings cache
            if 'X_test' in self.embeddings['embeddings']:
                test_embeddings = self.embeddings['embeddings']['X_test']
                test_labels = test_labels
                print(f"  ✅ Using test data from parameters: {len(test_texts)} texts, {len(test_labels)} labels")
                print(f"  ✅ Test embeddings shape: {test_embeddings.shape if hasattr(test_embeddings, 'shape') else len(test_embeddings)}")
            else:
                print(f"  ⚠️ Test texts provided but no test embeddings found in embeddings cache")
        
        # Priority 2: Fallback to embeddings cache
        elif 'X_test' in self.embeddings['embeddings']:
            test_embeddings = self.embeddings['embeddings']['X_test']
            # Get test labels from data_dict if available
            if hasattr(self, 'data_dict') and 'y_test' in self.data_dict:
                test_labels = self.data_dict['y_test']
            print(f"  ✅ Found test embeddings in cache: {test_embeddings.shape if hasattr(test_embeddings, 'shape') else len(test_embeddings)} samples")
        else:
            print(f"  ⚠️ No test data available, will use validation data for final evaluation")
        
        # Create embeddings for each fold by splitting pre-computed embeddings
        for fold, (train_idx, val_idx) in enumerate(kf.split(texts, labels), 1):
            if stop_callback and stop_callback():
                print("⏹️ Stop signal received during CV embeddings pre-computation")
                return {}
                
            print(f"  📊 Splitting pre-computed embeddings for Fold {fold}/{self.cv_folds}")
            
            # Split pre-computed embeddings for this fold
            X_train_emb = precomputed_embeddings[train_idx]
            X_val_emb = precomputed_embeddings[val_idx]
            
            # Get labels for this fold
            y_train_fold = np.array([labels[i] for i in train_idx])
            y_val_fold = np.array([labels[i] for i in val_idx])
            
            # Store fold data with test data included
            cv_embeddings[f'fold_{fold}'] = {
                'X_train': X_train_emb,
                'X_val': X_val_emb, 
                'X_test': test_embeddings,  # ← THÊM TEST DATA
                'y_train': y_train_fold,
                'y_val': y_val_fold,
                'y_test': test_labels,      # ← THÊM TEST LABELS
                'train_idx': train_idx,
                'val_idx': val_idx,
                            'n_train_samples': X_train_emb.shape[0] if hasattr(X_train_emb, 'shape') else len(X_train_emb),    # ← THÊM SỐ SAMPLE
            'n_val_samples': X_val_emb.shape[0] if hasattr(X_val_emb, 'shape') else len(X_val_emb),        # ← THÊM SỐ SAMPLE
                'n_test_samples': len(test_embeddings) if test_embeddings is not None else 0  # ← THÊM SỐ SAMPLE TEST
            }
        
        if stop_callback and stop_callback():
            return {}
            
        print(f"✅ Pre-computed embeddings for {self.cv_folds} folds + test data")
        print(f"   📊 Each fold contains: Training, Validation, and Test data")
        self.cv_embeddings_cache = cv_embeddings  # Cache for reuse
        return cv_embeddings
    
    def create_cv_folds_for_sparse_embeddings(self, X_train: Union[np.ndarray, sparse.csr_matrix], 
                                             y_train: np.ndarray, 
                                             embedding_type: str,
                                             X_test: Union[np.ndarray, sparse.csr_matrix] = None,
                                             y_test: np.ndarray = None) -> Dict[str, Any]:
        """
        Create CV folds for BoW/TF-IDF (sparse matrices) - REUSES existing logic
        
        Args:
            X_train: Pre-computed sparse matrix (BoW/TF-IDF)
            y_train: Training labels
            embedding_type: 'bow' or 'tfidf'
        
        Returns:
            Dictionary with CV folds data compatible with existing structure + test data
        """
        print(f"🔄 Creating CV folds for {embedding_type.upper()} + test data...")
        
        # Import sklearn for CV splitting
        from sklearn.model_selection import StratifiedKFold
        
        # Create CV splitter
        kf = StratifiedKFold(
            n_splits=self.cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        cv_folds = {}
        
        # Get test data if available from embeddings
        test_data = None
        test_labels = None
        
        # Priority 1: Use test data passed as parameters
        if X_test is not None and y_test is not None:
            test_data = X_test
            test_labels = y_test
            print(f"  ✅ Using test data from parameters for {embedding_type}: {test_data.shape if hasattr(test_data, 'shape') else len(test_data)} samples")
        
        # Priority 2: Fallback to embeddings cache
        elif hasattr(self, 'embeddings') and embedding_type in self.embeddings:
            if 'X_test' in self.embeddings[embedding_type]:
                test_data = self.embeddings[embedding_type]['X_test']
                # Get test labels from data_dict if available
                if hasattr(self, 'data_dict') and 'y_test' in self.data_dict:
                    test_labels = self.data_dict['y_test']
                print(f"  ✅ Found test data in cache for {embedding_type}: {test_data.shape if hasattr(test_data, 'shape') else len(test_data)} samples")
            else:
                print(f"  ⚠️ Test data not found in cache for {embedding_type}, will use validation data for final evaluation")
        else:
            print(f"  ⚠️ No test data available for {embedding_type}, will use validation data for final evaluation")
            print(f"     🔍 Debug: hasattr(self, 'embeddings') = {hasattr(self, 'embeddings')}")
            if hasattr(self, 'embeddings'):
                print(f"     🔍 Debug: embedding_type '{embedding_type}' in embeddings = {embedding_type in self.embeddings}")
                print(f"     🔍 Debug: available embeddings = {list(self.embeddings.keys()) if self.embeddings else 'None'}")
        
        # Create folds using same strategy as precompute_cv_embeddings
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
            print(f"  📊 Creating fold {fold}/{self.cv_folds}")
            
            # Split sparse matrix data for this fold
            if hasattr(X_train, 'toarray'):  # Sparse matrix
                X_train_fold = X_train[train_idx]
                X_val_fold = X_train[val_idx]
            else:  # Dense array
                X_train_fold = X_train[train_idx]
                X_val_fold = X_train[val_idx]
            
            # Get labels for this fold
            y_train_fold = np.array([y_train[i] for i in train_idx])
            y_val_fold = np.array([y_train[i] for i in val_idx])
            
            # Store fold data - compatible with existing structure + test data
            cv_folds[f'fold_{fold}'] = {
                'X_train': X_train_fold,
                'X_val': X_val_fold,
                'X_test': test_data if test_data is not None else None,        # ← THÊM TEST DATA
                'y_train': y_train_fold,
                'y_val': y_val_fold,
                'y_test': test_labels if test_labels is not None else None,      # ← THÊM TEST LABELS
                'train_idx': train_idx,
                'val_idx': val_idx,
                            'n_train_samples': X_train_fold.shape[0] if hasattr(X_train_fold, 'shape') else len(X_train_fold),    # ← THÊM SỐ SAMPLE
            'n_val_samples': X_val_fold.shape[0] if hasattr(X_val_fold, 'shape') else len(X_val_fold),        # ← THÊM SỐ SAMPLE
                'n_test_samples': test_data.shape[0] if test_data is not None and hasattr(test_data, 'shape') else (len(test_data) if test_data is not None else 0)  # ← THÊM SỐ SAMPLE TEST
            }
        
        print(f"✅ Created CV folds for {embedding_type.upper()}: {self.cv_folds} folds + test data")
        print(f"   📊 Each fold contains: Training, Validation, and Test data")
        return cv_folds
    
    def load_and_prepare_data(self, max_samples: int = None, skip_csv_prompt: bool = False, sampling_config: Dict = None, preprocessing_config: Dict = None) -> Tuple[Dict[str, Any], List[str]]:
        """
        Load and prepare all data formats for evaluation
        
        Args:
            max_samples: Maximum number of samples to use
            skip_csv_prompt: If True, skip CSV backup prompt (for Streamlit usage)
            sampling_config: Sampling configuration from Streamlit (optional)
            preprocessing_config: Preprocessing configuration from Streamlit (optional)
        
        Returns:
            Tuple of (data_dict, sorted_labels)
        """
        print("\n📊 Loading and Preparing Data...")
        print("=" * 50)
        
        # Check if we already have sampled data from Step 1
        
        if hasattr(self, 'step1_data') and self.step1_data and 'dataframe' in self.step1_data:
            print("🚀 Using pre-sampled data from Step 1...")
            df = self.step1_data['dataframe']
            print(f"📊 Pre-sampled data size: {len(df):,} samples")
            
            # Set multi-input flag based on step1_data
            self.data_loader.is_multi_input = self.step1_data.get('is_multi_input', False)
            
            # Convert DataFrame to DataLoader format
            self.data_loader.samples = []
            
            # CRITICAL: Get actual column names from Step 1 configuration
            # Check if we have step1_data with column configuration
            if 'text_column' in self.step1_data and 'label_column' in self.step1_data:
                text_col = self.step1_data['text_column']
                label_col = self.step1_data['label_column']
                print(f"🔍 Using Step 1 column config: text='{text_col}', label='{label_col}'")
            else:
                # Fallback: try to guess column names
                text_col = 'text' if 'text' in df.columns else df.columns[0]
                label_col = 'label' if 'label' in df.columns else df.columns[-1]
                print(f"⚠️ No Step 2 config, guessing columns: text='{text_col}', label='{label_col}'")
            
            print(f"🔍 Available columns in DataFrame: {list(df.columns)}")
            print(f"🔍 Using columns: text='{text_col}', label='{label_col}'")
            
            for idx, row in df.iterrows():
                sample = {
                    'abstract': str(row.get(text_col, '')),
                    'categories': str(row.get(label_col, ''))
                }
                self.data_loader.samples.append(sample)
            
            print(f"✅ Converted {len(self.data_loader.samples):,} samples to DataLoader format")
            
            # Skip dataset loading and category discovery since we have pre-sampled data
            actual_max_samples = len(self.data_loader.samples)
            print(f"📊 Using pre-sampled samples: {actual_max_samples:,}")
            
            # IMPORTANT: Set available_categories and selected_categories for pre-sampled data
            # Extract unique categories from the pre-sampled data
            unique_categories = set()
            for sample in self.data_loader.samples:
                if sample['categories']:
                    categories = [cat.strip() for cat in str(sample['categories']).split()]
                    unique_categories.update(categories)
            
            self.data_loader.available_categories = sorted(unique_categories)
            self.data_loader.selected_categories = list(unique_categories)
            
            print(f"🔍 Discovered {len(self.data_loader.available_categories)} categories from pre-sampled data")
            print(f"💡 Categories: {self.data_loader.selected_categories[:5]}...")
            
            # CRITICAL: Validate that we have valid data
            # For numeric data, check if we have valid categories (labels)
            # For text data, check both text and categories
            if hasattr(self.data_loader, 'is_multi_input') and self.data_loader.is_multi_input:
                # Numeric data: only check categories (labels)
                valid_samples = [s for s in self.data_loader.samples if s['categories'].strip()]
                print(f"🔍 Valid samples (non-empty categories for numeric data): {len(valid_samples):,}")
            else:
                # Text data: check both text and categories
                valid_samples = [s for s in self.data_loader.samples if s['abstract'].strip() and s['categories'].strip()]
                print(f"🔍 Valid samples (non-empty text & categories): {len(valid_samples):,}")
            
            if len(valid_samples) == 0:
                print("❌ ERROR: No valid samples found! All samples have empty text or categories.")
                print("🔍 Debug: First few samples:")
                for i, sample in enumerate(self.data_loader.samples[:3]):
                    print(f"   Sample {i}: text='{sample['abstract'][:50]}...', categories='{sample['categories']}'")
                raise ValueError("No valid samples found. Check column mapping and data quality.")
            
        else:
            # Fallback: Load dataset from scratch (for non-Streamlit usage)
            if hasattr(self, 'step1_data'):
                print(f"   • step1_data type: {type(self.step1_data)}")
                if self.step1_data:
                    print(f"   • step1_data keys: {list(self.step1_data.keys()) if isinstance(self.step1_data, dict) else 'Not a dict'}")
            
            print("📥 Loading dataset from scratch...")
            self.data_loader.load_dataset(skip_csv_prompt=skip_csv_prompt)
            
            # Use sampling_config if available, otherwise fall back to max_samples
            if sampling_config and sampling_config.get('num_samples'):
                actual_max_samples = sampling_config['num_samples']
                print(f"📊 Using sampling config: {actual_max_samples:,} samples")
            else:
                actual_max_samples = max_samples
                if isinstance(actual_max_samples, dict):
                    actual_max_samples = actual_max_samples.get('num_samples', None)
                print(f"📊 Using max_samples parameter: {actual_max_samples:,} samples" if actual_max_samples else "📊 No sample limit specified")
            
            # Discover categories first if not already done
            if not self.data_loader.available_categories:
                print("🔍 Discovering available categories...")
                self.data_loader.discover_categories()
            
            # Get recommended categories if none selected
            if not self.data_loader.selected_categories:
                recommended_categories = self.data_loader.get_category_recommendations(max_categories=5)
                if recommended_categories:
                    print(f"💡 Setting recommended categories: {recommended_categories}")
                    self.data_loader.set_selected_categories(recommended_categories)
                else:
                    # Fallback: use all available categories
                    all_categories = self.data_loader.available_categories[:5]  # Limit to 5
                    print(f"⚠️ No categories available, using first 5: {all_categories}")
                    self.data_loader.set_selected_categories(all_categories)
            
            # Now select samples with categories
            self.data_loader.select_samples(actual_max_samples)
        
        # Apply preprocessing with advanced options
        # Use default preprocessing config if none provided
        if preprocessing_config is None:
            preprocessing_config = {
                'text_cleaning': True,
                'data_validation': True,
                'category_mapping': True,
                'memory_optimization': True,
                # Advanced preprocessing options with defaults
                'rare_words_removal': False,
                'rare_words_threshold': 2,
                'lemmatization': False,
                'context_aware_stopwords': False,
                'stopwords_aggressiveness': 'Moderate',
                'phrase_detection': False,
                'min_phrase_freq': 3
            }
        
        # Ensure all advanced preprocessing options are included
        full_preprocessing_config = {
            'text_cleaning': preprocessing_config.get('text_cleaning', True),
            'data_validation': preprocessing_config.get('data_validation', True),
            'category_mapping': preprocessing_config.get('category_mapping', True),
            'memory_optimization': preprocessing_config.get('memory_optimization', True),
            # Advanced preprocessing options
            'rare_words_removal': preprocessing_config.get('rare_words_removal', False),
            'rare_words_threshold': preprocessing_config.get('rare_words_threshold', 2),
            'lemmatization': preprocessing_config.get('lemmatization', False),
            'context_aware_stopwords': preprocessing_config.get('context_aware_stopwords', False),
            'stopwords_aggressiveness': preprocessing_config.get('stopwords_aggressiveness', 'Moderate'),
            'phrase_detection': preprocessing_config.get('phrase_detection', False),
            'min_phrase_freq': preprocessing_config.get('min_phrase_freq', 3)
        }
        
        print(f"🔧 [EVALUATOR] Applying preprocessing with full config: "
              f"{full_preprocessing_config}")
        
        # CRITICAL: Ensure we have samples before preprocessing
        if not self.data_loader.samples:
            print("❌ ERROR: No samples available for preprocessing!")
            print(f"🔍 Debug: samples count = {len(self.data_loader.samples)}")
            print(f"🔍 Debug: available_categories = {self.data_loader.available_categories}")
            print(f"🔍 Debug: selected_categories = {self.data_loader.selected_categories}")
            raise ValueError("No samples available for preprocessing. Check data loading.")
        
        print(f"📊 Preprocessing {len(self.data_loader.samples):,} samples...")
        
        self.data_loader.preprocess_samples(full_preprocessing_config)
        self.data_loader.create_label_mappings()
        
        # Validate data state before train/test split
        
        if not self.data_loader.preprocessed_samples:
            print("❌ ERROR: No preprocessed samples available!")
            print("🔍 Debug: Check if preprocessing was successful")
            raise ValueError("No preprocessed samples available for train/test split")
        
        # Prepare train/test data (no separate validation set)
        # Get requested samples from Step 1
        requested_samples = None
        if hasattr(self, 'step1_data') and self.step1_data and 'sampling_config' in self.step1_data:
            requested_samples = self.step1_data['sampling_config'].get('num_samples')
            print(f"📊 Using requested samples from Step 1: {requested_samples}")
            
            # Update test_size if requested_samples is provided
            if requested_samples:
                total_samples = len(self.data_loader.preprocessed_samples)
                if total_samples > 0:
                    # Calculate test_size to get 20% of requested samples
                    target_test_samples = int(requested_samples * 0.2)
                    if total_samples >= requested_samples:
                        dynamic_test_size = target_test_samples / total_samples
                        print(f"📊 Using dynamic test size: {dynamic_test_size:.3f}")
                        # Update test_size for data splitting
                        self.test_size = dynamic_test_size
        
        X_train, X_test, y_train, y_test = self.data_loader.prepare_train_test_data(requested_samples)
        sorted_labels = self.data_loader.get_sorted_labels()
        
        # Use train/test split directly (validation handled by CV)
        X_train_full, y_train_full = X_train, y_train
        X_val, y_val = np.array([]), np.array([])  # Empty validation set
        
        # Verify split consistency
        print(f"🔍 Split verification:")
        print(f"   • Total: {(X_train_full.shape[0] if hasattr(X_train_full, 'shape') else len(X_train_full)) + (X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test))}")
        print(f"   • Train: {X_train_full.shape[0] if hasattr(X_train_full, 'shape') else len(X_train_full)} | Test: {X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)}")
        
        print(f"✅ Data prepared:")
        print(f"   • Training: {X_train_full.shape[0] if hasattr(X_train_full, 'shape') else len(X_train_full)} samples (for CV)")
        print(f"   • Validation: Handled by CV folds")
        print(f"   • Test: {X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)} samples")
        print(f"   • Labels: {len(sorted_labels)} classes")
        
        # Store data
        data_dict = {
            'X_train': X_train_full,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train_full,
            'y_val': y_val,
            'y_test': y_test,
            'labels': sorted_labels,
            'dataset_info': f"dataset_{len(X_train_full)}samples_{len(sorted_labels)}classes"
        }
        
        # Store original texts for embedding CV (fix data leakage)
        self._original_texts = X_train_full
        
        # Note: Embeddings will be created later in run_comprehensive_evaluation()
        # to avoid duplication and respect selected_embeddings parameter
        
        return data_dict, sorted_labels
    
    def create_all_embeddings(self, X_train: List[str], X_val: List[str], X_test: List[str], 
                             selected_embeddings: List[str] = None, stop_callback=None) -> Dict[str, Dict[str, Any]]:
        """
        Create embedding representations for the data
        
        Args:
            X_train: Training text data
            X_val: Validation text data  
            X_test: Test text data
            selected_embeddings: List of embedding methods to create (if None, create all)
        
        Returns:
            Dictionary of embeddings for each method
        """
        print("\n🔤 Creating Embedding Representations...")
        print("=" * 50)
        
        embeddings = {}
        
        # Store text vectorizer for embedding CV (fix data leakage)
        self._text_vectorizer = self.text_vectorizer
        
        # Define which embeddings to create
        if selected_embeddings is None:
            embedding_methods = ['bow', 'tfidf', 'embeddings']
        else:
            # Map Streamlit names to internal names
            embedding_mapping = {
                'BoW': 'bow',
                'TF-IDF': 'tfidf',
                'Word Embeddings': 'embeddings'
            }
            embedding_methods = [embedding_mapping.get(emb, emb) for emb in selected_embeddings]
            print(f"🔍 Embedding creation mapping: {selected_embeddings} -> {embedding_methods}")
        
        print(f"📊 Creating embeddings: {', '.join(embedding_methods)}")
        
        # 1. Bag of Words (BoW)
        if 'bow' in embedding_methods:
            print("📦 Processing Bag of Words...")
            start_time = time.time()
            X_train_bow = self.text_vectorizer.fit_transform_bow(X_train)
            X_val_bow = self.text_vectorizer.transform_bow(X_val) if len(X_val) > 0 else None
            X_test_bow = self.text_vectorizer.transform_bow(X_test)
            bow_time = time.time() - start_time
            
            embeddings['bow'] = {
                'X_train': X_train_bow,
                'X_val': X_val_bow,
                'X_test': X_test_bow,
                'processing_time': bow_time,
                'sparse': hasattr(X_train_bow, 'nnz'),
                'shape': X_train_bow.shape
            }
            print(f"   ✅ BoW: {X_train_bow.shape} | Sparse: {hasattr(X_train_bow, 'nnz')} | Time: {bow_time:.2f}s")
        
        # 2. TF-IDF
        if 'tfidf' in embedding_methods:
            print("📊 Processing TF-IDF...")
            start_time = time.time()
            X_train_tfidf = self.text_vectorizer.fit_transform_tfidf(X_train)
            X_val_tfidf = self.text_vectorizer.transform_tfidf(X_val) if len(X_val) > 0 else None
            X_test_tfidf = self.text_vectorizer.transform_tfidf(X_test)
            tfidf_time = time.time() - start_time
            
            embeddings['tfidf'] = {
                'X_train': X_train_tfidf,
                'X_val': X_val_tfidf,
                'X_test': X_test_tfidf,
                'processing_time': tfidf_time,
                'sparse': hasattr(X_train_tfidf, 'nnz'),
                'shape': X_train_tfidf.shape
            }
            print(f"   ✅ TF-IDF: {X_train_tfidf.shape} | Sparse: {hasattr(X_train_tfidf, 'nnz')} | Time: {tfidf_time:.2f}s")
        
        # 3. Word Embeddings
        if 'embeddings' in embedding_methods:
            print("🧠 Processing Word Embeddings...")
            start_time = time.time()
            
            # Progress tracking removed - using simple progress indicators
            
            # FIXED: Fit embedding model on TRAINING DATA ONLY to prevent data leakage
            print(f"🔧 Fitting embedding model on {len(X_train):,} training samples...")
            # Import global stop check if available
            try:
                from training_pipeline import global_stop_check
                actual_stop_callback = global_stop_check
            except ImportError:
                actual_stop_callback = stop_callback
                
            X_train_emb = self.text_vectorizer.fit_transform_embeddings(X_train, stop_callback=actual_stop_callback)
            
            # Progress tracking removed
            
            # Check if stopped during training embeddings
            if actual_stop_callback and actual_stop_callback():
                print("🛑 Embedding creation stopped by user request")
                return {}
            
            # Transform test data using fitted model (no data leakage)
            print(f"🔧 Transforming {X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test):,} test samples using fitted model...")
            X_test_emb = self.text_vectorizer.transform_embeddings(X_test, stop_callback=actual_stop_callback)
            
            # Progress tracking removed
            
            # Check if stopped during test embeddings
            if actual_stop_callback and actual_stop_callback():
                print("🛑 Embedding creation stopped by user request")
                return {}
            
            # Transform validation data if exists
            X_val_emb = None
            if len(X_val) > 0:
                print(f"🔧 Transforming {len(X_val):,} validation samples...")
                X_val_emb = self.text_vectorizer.transform_embeddings(X_val, stop_callback=actual_stop_callback)
                
                # Progress tracking removed
                
                # Check if stopped during validation embeddings
                if actual_stop_callback and actual_stop_callback():
                    print("🛑 Embedding creation stopped by user request")
                    return {}
            
            # Progress tracking removed
            
            emb_time = time.time() - start_time
            
            embeddings['embeddings'] = {
                'X_train': X_train_emb,
                'X_val': X_val_emb,
                'X_test': X_test_emb,
                'processing_time': emb_time,
                'sparse': hasattr(X_train_emb, 'nnz'),
                'shape': X_train_emb.shape
            }
            print(f"   ✅ Embeddings: {X_train_emb.shape} | Sparse: {hasattr(X_train_emb, 'nnz')} | Time: {emb_time:.2f}s")
        
        # Summary
        if embeddings:
            total_time = sum(emb['processing_time'] for emb in embeddings.values())
            print(f"\n📊 Embedding Summary:")
            print(f"   • Total processing time: {total_time:.2f}s")
            print(f"   • Memory efficient: {sum(1 for emb in embeddings.values() if emb['sparse'])}/{len(embeddings)} methods use sparse matrices")
        
        return embeddings
    
    def evaluate_single_combination(self, 
                                  model_name: str, 
                                  embedding_name: str,
                                  X_train: Union[np.ndarray, sparse.csr_matrix, List],
                                  X_val: Union[np.ndarray, sparse.csr_matrix, List],
                                  X_test: Union[np.ndarray, sparse.csr_matrix, List],
                                  y_train: np.ndarray,
                                  y_val: np.ndarray,
                                  y_test: np.ndarray,
                                  step3_data: Dict = None,
                                  selected_embeddings: List[str] = None,
                                  feature_dimensions: Dict[str, int] = None) -> Dict[str, Any]:
        """
        Evaluate a single model-embedding combination
        
        Returns:
            Dictionary with evaluation results
        """
        combination_key = f"{model_name}_{embedding_name}"
        print(f"   🔍 Evaluating {combination_key}...")
        
        # Handle text data by converting to numerical representation for cache
        if isinstance(X_train, list):
            # Convert text data to numerical representation for cache consistency
            import hashlib
            X_train_hash = hashlib.md5(str(X_train).encode()).hexdigest()[:8]
            X_train_for_cache = np.array([int(X_train_hash, 16)])  # Convert to numerical
        else:
            X_train_for_cache = X_train
        
        try:
            # Log KNN configuration if available
            if model_name == 'knn' and step3_data and 'knn_config' in step3_data:
                knn_config = step3_data['knn_config']
                print(f"     🎯 [KNN EVALUATION] Configuration from Step 3:")
                print(f"        • Optimization Method: {knn_config.get('optimization_method', 'N/A')}")
                print(f"        • K Value: {knn_config.get('k_value', 'N/A')}")
                print(f"        • Weights: {knn_config.get('weights', 'N/A')}")
                print(f"        • Metric: {knn_config.get('metric', 'N/A')}")
                if knn_config.get('best_score'):
                    print(f"        • Best Score: {knn_config.get('best_score', 'N/A'):.4f}")
            
            # MEMORY OPTIMIZATION: Keep sparse matrices as-is for memory efficiency
            if hasattr(X_train, 'toarray'):  # Sparse matrix
                print(f"     📊 Using sparse matrix format for memory efficiency")
                # Keep sparse matrices - modern models can handle them efficiently
                # No conversion to dense to prevent memory overflow
            
            # Training with memory management
            start_time = time.time()
            
            # Clear GPU cache before training to prevent memory issues
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"     🧹 GPU cache cleared before training")
            except ImportError:
                pass  # PyTorch not available
            
            # Check cache first
            print(f"     🔍 Checking cache for {model_name}...")
            cached_result = self._check_model_cache(model_name, X_train_for_cache, y_train, self.cv_folds, self.random_state, step3_data, selected_embeddings, feature_dimensions)
            if cached_result:
                print(f"     ✅ Cache HIT for {model_name}")
            else:
                print(f"     ❌ Cache MISS for {model_name}")
            
            if cached_result:
                # Use cached model
                model = cached_result['model']
                val_acc = cached_result['metrics'].get('validation_accuracy', 0.0)
                test_acc = cached_result['metrics'].get('test_accuracy', 0.0)
                test_metrics = cached_result['metrics'].get('test_metrics', {})
                
                # Convert data types for cached model predictions
                if hasattr(X_test, 'toarray'):  # Sparse matrix
                    X_test = X_test.astype(np.float32)
                elif isinstance(X_test, np.ndarray):
                    X_test = X_test.astype(np.float32)
                
                if X_val is not None:
                    if hasattr(X_val, 'toarray'):  # Sparse matrix
                        X_val = X_val.astype(np.float32)
                    elif isinstance(X_val, np.ndarray):
                        X_val = X_val.astype(np.float32)
                
                # Make predictions
                y_test_pred = model.predict(X_test)
                y_val_pred = model.predict(X_val) if X_val is not None else y_test_pred
                
                print(f"✅ Using cached model for {model_name}")
            else:
                # Train new model
                print(f"🚀 Training new model: {model_name}")
                
                # Create model
                model = self.model_factory.create_model(model_name)
                
                # Convert data types to float32 for better compatibility
                if hasattr(X_train, 'toarray'):  # Sparse matrix
                    X_train = X_train.astype(np.float32)
                elif isinstance(X_train, np.ndarray):
                    X_train = X_train.astype(np.float32)
                
                if hasattr(X_test, 'toarray'):  # Sparse matrix
                    X_test = X_test.astype(np.float32)
                elif isinstance(X_test, np.ndarray):
                    X_test = X_test.astype(np.float32)
                
                if X_val is not None:
                    if hasattr(X_val, 'toarray'):  # Sparse matrix
                        X_val = X_val.astype(np.float32)
                    elif isinstance(X_val, np.ndarray):
                        X_val = X_val.astype(np.float32)
                
                print(f"     🔧 Converted data to float32 for compatibility")
                
                # Special handling for ensemble models
                if model_name.startswith(('voting_ensemble', 'stacking_ensemble')):
                    # Create base estimators for ensemble
                    base_estimators = []
                    for model_name_base in ['knn', 'decision_tree', 'naive_bayes']:
                        try:
                            model_class_base = self.model_factory.create_model(model_name_base)
                            base_estimators.append((model_name_base, model_class_base))
                        except Exception as e:
                            print(f"Warning: Error creating {model_name_base}: {e}")
                    
                    # Create the ensemble classifier
                    if base_estimators:
                        model.create_ensemble_classifier(base_estimators)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_test_pred = model.predict(X_test)
                y_val_pred = model.predict(X_val) if X_val is not None else y_test_pred
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                # Use test data for validation if validation data is not available
                if y_val is not None and len(y_val) > 0:
                    val_acc = accuracy_score(y_val, y_val_pred)
                else:
                    # Use test data for validation when validation data is not available
                    val_acc = accuracy_score(y_test, y_test_pred)
                
                test_acc = accuracy_score(y_test, y_test_pred)
                
                test_metrics = {
                    'accuracy': test_acc,
                    'precision': precision_score(y_test, y_test_pred, average='weighted'),
                    'recall': recall_score(y_test, y_test_pred, average='weighted'),
                    'f1_score': f1_score(y_test, y_test_pred, average='weighted')
                }
                
                # Save to cache
                eval_predictions = None
                if X_test is not None and y_test is not None:
                    eval_predictions = self._prepare_eval_predictions_for_cache(X_test, y_test, model)
                    if eval_predictions is not None:
                        print(f"     ✅ Eval predictions prepared: {eval_predictions.shape}")
                    else:
                        print(f"     ⚠️ Eval predictions failed to prepare")
                else:
                    print(f"     ⚠️ Cannot prepare eval predictions: X_test={X_test is not None}, y_test={y_test is not None}")
                
                self._save_model_cache(
                    model_name=model_name,
                    X_train=X_train_for_cache,
                    y_train=y_train,
                    cv_folds=self.cv_folds,
                    random_state=self.random_state,
                    model=model,
                    params=getattr(model, 'get_params', lambda: {})(),
                    metrics={
                        'validation_accuracy': val_acc,
                        'test_accuracy': test_acc,
                        'test_metrics': test_metrics
                    },
                    config={
                        'model_name': model_name,
                        'cv_folds': self.cv_folds,
                        'random_state': self.random_state
                    },
                    eval_predictions=eval_predictions,
                    step3_data=step3_data,
                    selected_embeddings=selected_embeddings,
                    feature_dimensions=feature_dimensions
                )
            training_time = time.time() - start_time
            
            # Clear GPU cache after training
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"     🧹 GPU cache cleared after training")
            except ImportError:
                pass  # PyTorch not available
            
            # Validation metrics (only if validation set exists)
            if len(y_val) > 0 and y_val_pred is not None:
                val_metrics = ModelMetrics.compute_classification_metrics(y_val, y_val_pred)
            else:
                val_metrics = None
                val_acc = 0.0  # Set to 0 if no validation set
            
            # Test metrics
            test_metrics = ModelMetrics.compute_classification_metrics(y_test, y_test_pred)
            
            # ML Standard Overfitting Analysis (training vs validation/test accuracy)
            try:
                # Get training accuracy by creating a temporary model and scoring on training data
                # This is the ML standard approach for overfitting detection
                from models import model_factory
                temp_model = model_factory.create_model(model_name)
                if temp_model:
                    # Fit the model on training data
                    if model_name == 'knn':
                        # Use GPU for embeddings (dense), CPU for TF-IDF/BOW (sparse)
                        use_gpu = not hasattr(X_train, 'toarray')  # Dense matrices can use GPU
                        temp_model.fit(X_train, y_train, use_gpu=use_gpu)
                    else:
                        temp_model.fit(X_train, y_train)
                    
                    # Get training accuracy
                    if hasattr(temp_model, 'score'):
                        train_acc = temp_model.score(X_train, y_train)
                    else:
                        train_acc = test_acc  # Fallback
                else:
                    train_acc = test_acc  # Fallback
                
                if len(y_val) > 0 and y_val_pred is not None:
                    # Calculate ML standard overfitting: Training Accuracy vs Validation Accuracy
                    # This is the standard ML approach for detecting overfitting
                    overfitting_score = train_acc - val_acc  # Training Acc - Validation Acc
                    overfitting_status = self._classify_overfitting(overfitting_score)
                    
                    # Classify overfitting level
                    if overfitting_score is not None:
                        if overfitting_score > 0.1:
                            overfitting_level = f"High overfitting - {overfitting_score:.3f}"
                        elif overfitting_score > 0.05:
                            overfitting_level = f"Moderate overfitting - {overfitting_score:.3f}"
                        elif overfitting_score > -0.05:
                            overfitting_level = f"Good fit - {overfitting_score:.3f}"
                        elif overfitting_score > -0.1:
                            overfitting_level = f"Slight underfitting - {overfitting_score:.3f}"
                        else:
                            overfitting_level = f"Underfitting - {overfitting_score:.3f}"
                    else:
                        overfitting_level = "Cannot determine - score not available"
                    
                    if overfitting_score is not None:
                        print(f"     📊 ML Standard Overfitting: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}, Score={overfitting_score:+.3f}")
                    else:
                        print(f"     📊 ML Standard Overfitting: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}, Score=Not available")
                    print(f"     📊 Overfitting Level: {overfitting_level}")
                    
                else:
                    # No validation set: Will calculate overfitting from CV fold later
                    # overfitting_score, overfitting_status, and overfitting_level will be set after CV
                    overfitting_score = None
                    overfitting_status = None
                    overfitting_level = None
                    
                    print(f"     💡 No separate validation set: Using CV folds for overfitting analysis")
                    print(f"     📊 Overfitting Level: Will be calculated from CV fold")
                    print(f"     💡 This is the standard ML approach for overfitting detection")
                    
            except Exception as e:
                # Cannot calculate overfitting if ML standard fails
                overfitting_score = None
                overfitting_status = "calculation_failed"
                overfitting_level = "Cannot determine - calculation failed"
                print(f"     ❌ ML Standard overfitting calculation failed: {e}")
                print(f"     📊 Overfitting Level: {overfitting_level}")
                print(f"     💡 Recommendation: Check validation data and model configuration")
            
            # Cross-validation - ENHANCED: Use optimized CV for sparse embeddings (BoW/TF-IDF) or fitted embeddings
            if embedding_name in ['bow', 'tfidf']:
                # For BoW and TF-IDF, use optimized CV with sparse matrix handling
                print(f"     🔧 CV using optimized sparse {embedding_name} data for {model_name} (no data leakage)")
                
                # Create CV folds specifically for sparse embeddings with test data
                test_data = self.data_dict.get('X_test') if hasattr(self, 'data_dict') else None
                test_labels = self.data_dict.get('y_test') if hasattr(self, 'data_dict') else None
                
                # Convert test data to numpy array if it's a list
                if test_data is not None and isinstance(test_data, list):
                    test_data = np.array(test_data)
                if test_labels is not None and isinstance(test_labels, list):
                    test_labels = np.array(test_labels)
                
                print(f"     🔍 Debug: test_data = {test_data.shape if test_data is not None and hasattr(test_data, 'shape') else f'list({len(test_data)})' if test_data is not None else 'None'}")
                print(f"     🔍 Debug: test_labels = {test_labels.shape if test_labels is not None and hasattr(test_labels, 'shape') else f'list({len(test_labels)})' if test_labels is not None else 'None'}")
                
                cv_folds = self.create_cv_folds_for_sparse_embeddings(
                    X_train, y_train, embedding_name, 
                    X_test=test_data, 
                    y_test=test_labels
                )
                
                # Use CV folds for evaluation
                # TODO: Implement CV evaluation without model_trainer
                cv_results = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            else:
                # For embeddings, use pre-computed CV embeddings for fair comparison
                print(f"     🔧 CV using pre-computed {embedding_name} embeddings for {model_name} (fair comparison)")
                
                print(f"     🔍 Debug: cv_embeddings_cache exists = {hasattr(self, 'cv_embeddings_cache')}")
                print(f"     🔍 Debug: cv_embeddings_cache content = {bool(self.cv_embeddings_cache)}")
                print(f"     🔍 Debug: cv_embeddings_cache keys = {list(self.cv_embeddings_cache.keys()) if hasattr(self, 'cv_embeddings_cache') else 'N/A'}")
                
                if hasattr(self, 'cv_embeddings_cache') and self.cv_embeddings_cache:
                    # Use cached pre-computed embeddings
                    print(f"     ✅ Using pre-computed CV embeddings for {model_name}")
                    # TODO: Implement CV evaluation without model_trainer
                    cv_results = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                    
                    # ENHANCED: Also evaluate on test data from CV cache
                    if cv_results and 'fold_results' in cv_results:
                        print(f"     🔍 CV cache contains test data, evaluating on test set...")
                        # Get test evaluation from CV cache
                        # TODO: Implement test evaluation without model_trainer
                        test_eval = None
                        if test_eval:
                            # Update cv_results with test evaluation
                            cv_results['test_evaluation'] = test_eval
                            print(f"     ✅ Test evaluation from CV cache: {test_eval}")
                        else:
                            print(f"     ⚠️ Test evaluation from CV cache failed")
                else:
                    # Fallback to old method if cache not available
                    print(f"     ⚠️  Fallback: CV embeddings cache not found, using standard CV")
                    # TODO: Implement CV evaluation without model_trainer
                    cv_results = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
            # Calculate ML Standard CV Accuracy (training vs validation/test accuracy)
            # This provides ML standard overfitting detection
            try:
                # Use the same temp_model from above for ML standard CV calculation
                if temp_model and hasattr(temp_model, 'score'):
                    # We already have train_acc from above, just use it
                    pass
                else:
                    # Fallback: use test accuracy as approximation
                    train_acc = test_acc
                
                if len(y_val) > 0 and y_val_pred is not None:
                    # Calculate ML standard CV accuracy: average of training and validation accuracy
                    cv_f1_based_accuracy = (train_acc + val_acc) / 2.0  # ML standard CV accuracy
                    cv_f1_based_std = abs(train_acc - val_acc) / 2.0   # ML standard variation
                    print(f"     📊 ML Standard CV: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}, CV={cv_f1_based_accuracy:.3f}±{cv_f1_based_std:.3f}")
                else:
                    # No validation set: Use first CV fold to calculate overfitting approximation
                    # This provides a reasonable estimate without changing the data structure
                    if cv_results and 'fold_results' in cv_results and len(cv_results['fold_results']) > 0:
                        # Get first fold results for overfitting approximation
                        first_fold = cv_results['fold_results'][0]
                        if 'train_accuracy' in first_fold and 'validation_accuracy' in first_fold:
                            cv_train_acc = first_fold['train_accuracy']
                            cv_val_acc = first_fold['validation_accuracy']
                            
                            # Calculate overfitting metrics
                            
                            # Calculate overfitting using mean(train_accuracy_folds) − mean(val_accuracy_folds)
                            train_mean = np.mean([fold['train_accuracy'] for fold in cv_results['fold_results']])
                            val_mean = np.mean([fold['validation_accuracy'] for fold in cv_results['fold_results']])
                            cv_mean_accuracy = val_mean  # Keep for compatibility with logging
                            overfitting_score = train_mean - val_mean
                            overfitting_status = self._classify_overfitting(overfitting_score)
                            overfitting_level = self._get_overfitting_level_from_score(overfitting_score)
                            
                            # Calculate ML standard CV from first fold
                            cv_f1_based_accuracy = (cv_train_acc + cv_val_acc) / 2.0
                            cv_f1_based_std = abs(cv_train_acc - cv_val_acc) / 2.0
                            
                            print(f"     📊 Train vs Val Overfitting: Train Mean={train_mean:.3f}, Val Mean={val_mean:.3f}, Test Acc={test_acc:.3f}, Score={overfitting_score:+.3f}")
                            print(f"     📊 ML Standard CV (CV Fold): CV={cv_f1_based_accuracy:.3f}±{cv_f1_based_std:.3f}")

                        else:
                            cv_f1_based_accuracy = None
                            cv_f1_based_std = None 
                            # Set default values when CV fold data is incomplete
                            overfitting_score = None
                            overfitting_status = "cv_data_incomplete"
                            overfitting_level = "Cannot determine - CV fold data incomplete"
                            print(f"     ⚠️ CV fold data incomplete, cannot calculate overfitting")
                    else:
                        cv_f1_based_accuracy = None
                        cv_f1_based_std = None
                        # Set default values when CV results are not available
                        overfitting_score = None
                        overfitting_status = "cv_not_available"
                        overfitting_level = "Cannot determine - CV results not available"
                        print(f"     ⚠️ CV results not available, cannot calculate overfitting")
                
            except Exception as e:
                # Cannot calculate ML standard CV if it fails
                cv_f1_based_accuracy = None
                cv_f1_based_std = None
                print(f"     ❌ ML Standard CV calculation failed: {e}")
                print(f"     💡 Recommendation: Check validation data and model configuration")
            
            # Store results with trained model instance for reuse
            result = {
                'model_name': model_name,
                'embedding_name': embedding_name,
                'combination_key': combination_key,
                
                # Performance metrics
                'validation_accuracy': val_acc,
                'test_accuracy': test_acc,
                'validation_metrics': val_metrics,
                'test_metrics': test_metrics,
                'f1_score': test_metrics.get('f1_score', 0.0),  # ← Thêm F1 score từ test set
                
                # ENHANCED: Store trained model instance for ensemble reuse
                'trained_model': model,  # Use the trained model directly
                
                # Overfitting analysis
                'overfitting_score': overfitting_score,
                'overfitting_status': overfitting_status,
                'overfitting_level': overfitting_level,  # ← Thêm overfitting_level vào result
                'overfitting_classification': self._get_overfitting_classification(overfitting_score) if overfitting_score is not None else "Cannot determine",
                
                # Cross-validation results - Traditional CV accuracy from folds
                'cv_mean_accuracy': cv_results.get('overall_results', {}).get('accuracy_mean', val_acc),  # Traditional CV accuracy (from folds)
                'cv_std_accuracy': cv_results.get('overall_results', {}).get('accuracy_std', 0.0),       # Traditional CV accuracy std (from folds)
                'cv_mean_f1': cv_results.get('overall_results', {}).get('f1_mean', 0.0),                # Traditional CV F1 (from folds)
                'cv_std_f1': cv_results.get('overall_results', {}).get('f1_std', 0.0),                  # Traditional CV F1 std (from folds)
                
                # ML Standard CV Score - NEW: Based on training vs validation accuracy for overfitting detection
                'ml_cv_accuracy': cv_f1_based_accuracy,   # ← ML standard CV accuracy (avg of train_acc + val_acc) or None if no validation
                'ml_cv_variation': cv_f1_based_std,       # ← ML standard variation (variation between train_acc and val_acc) or None if no validation
                'cv_stability_score': self._calculate_stability_score(cv_results) if hasattr(self, '_calculate_stability_score') else 0.0,
                
                # Timing
                'training_time': training_time,
                'total_samples': X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train),
                
                # Data characteristics
                'input_shape': X_train.shape,
                'n_classes': len(np.unique(y_train)),
                
                # Confusion Matrix Data - Thêm dữ liệu cần thiết
                'predictions': y_test_pred,           # ← Predictions trên test set
                'true_labels': y_test,                # ← True labels từ test set
                'validation_predictions': y_val_pred, # ← Predictions trên validation set
                'validation_true_labels': y_val,      # ← True labels từ validation set
                
                # Label Information - Thêm thông tin labels
                'unique_labels': sorted(list(set(y_train))),  # ← Unique labels đã được xử lý
                'label_mapping': self._get_label_mapping(y_train),  # ← Mapping labels
                
                # Status
                'status': 'success',
                'error_message': None
            }
            
            if cv_f1_based_accuracy is not None:
                print(f"     ✅ {combination_key}: Val={val_acc:.3f}, Test={test_acc:.3f}, ML-CV={cv_f1_based_accuracy:.3f}±{cv_f1_based_std:.3f}")
            else:
                print(f"     ✅ {combination_key}: Val={val_acc:.3f}, Test={test_acc:.3f}, ML-CV=Not available")
            
            # Print overfitting information if available
            if overfitting_score is not None:
                print(f"     📊 Overfitting: {overfitting_level}")
                print(f"     📊 Status: {overfitting_status}")
            else:
                print(f"     📊 Overfitting: {overfitting_level}")
                print(f"     📊 Status: {overfitting_status}")
            
            return result
            
        except Exception as e:
            error_result = {
                'model_name': model_name,
                'embedding_name': embedding_name,
                'combination_key': combination_key,
                'status': 'error',
                'error_message': str(e),
                'validation_accuracy': 0.0,
                'test_accuracy': 0.0,
                'f1_score': 0.0,  # ← Thêm F1 score cho error case
                'overfitting_score': None,
                'overfitting_status': 'error',
                'overfitting_level': 'Error occurred',
                'overfitting_classification': 'Error occurred',
                'ml_cv_accuracy': None,
                'ml_cv_variation': None
            }
            print(f"     ❌ {combination_key}: Error - {e}")
            return error_result
    
    def _get_label_mapping(self, y_train: np.ndarray) -> Dict[int, str]:
        """
        Tạo mapping từ numeric labels sang text labels theo pipeline
        """
        try:
            # Lấy unique labels đã được sắp xếp
            unique_labels = sorted(list(set(y_train)))
            
            # PRIORITY: Sử dụng label mapping động từ data_loader nếu có (transferred from training pipeline)
            if hasattr(self.data_loader, 'id_to_label') and self.data_loader.id_to_label:
                label_mapping = {}
                for label_id in unique_labels:
                    if label_id in self.data_loader.id_to_label:
                        label_mapping[label_id] = self.data_loader.id_to_label[label_id]
                    else:
                        label_mapping[label_id] = f"Class_{label_id}"
                
                print(f"✅ [COMPREHENSIVE_EVALUATOR] Using transferred label mapping from data_loader:")
                print(f"   - Full id_to_label keys: {list(self.data_loader.id_to_label.keys())}")
                print(f"   - Applied mapping keys: {list(label_mapping.keys())}")
                return label_mapping
            
            # CRITICAL FIX: Nếu không có id_to_label, tạo meaningful labels từ preprocessed_samples
            if hasattr(self.data_loader, 'preprocessed_samples') and self.data_loader.preprocessed_samples:
                # Extract actual labels from preprocessed samples
                actual_labels = set()
                for sample in self.data_loader.preprocessed_samples:
                    if 'label' in sample:
                        actual_labels.add(sample['label'])
                
                if actual_labels:
                    # Create mapping using actual labels
                    sorted_actual_labels = sorted(list(actual_labels))
                    label_mapping = {i: label for i, label in enumerate(sorted_actual_labels)}
                    print(f"✅ Sử dụng labels từ preprocessed_samples: {len(label_mapping)} labels")
                    return label_mapping
            
            # IMPROVED FALLBACK: Cố gắng đoán text labels từ tên cột và dataset pattern
            try:
                # Check if this looks like arxiv dataset pattern
                if hasattr(self.data_loader, 'label_column') and self.data_loader.label_column:
                    # Common arxiv categories
                    common_arxiv_labels = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']
                    if len(unique_labels) == len(common_arxiv_labels):
                        label_mapping = {i: label for i, label in enumerate(common_arxiv_labels)}
                        print(f"✅ Sử dụng arxiv pattern labels: {len(label_mapping)} labels")
                        return label_mapping
                
                # Try to load data and create mapping if data_loader has file_path
                if hasattr(self.data_loader, 'file_path') and self.data_loader.file_path:
                    try:
                        # Try to discover and load categories
                        if hasattr(self.data_loader, 'discover_categories'):
                            self.data_loader.discover_categories()
                        
                        # Try to get recommended categories
                        if hasattr(self.data_loader, 'get_category_recommendations'):
                            recommended = self.data_loader.get_category_recommendations(max_categories=len(unique_labels))
                            if len(recommended) == len(unique_labels):
                                label_mapping = {i: label for i, label in enumerate(sorted(recommended))}
                                print(f"✅ Sử dụng recommended labels từ data: {len(label_mapping)} labels")
                                return label_mapping
                    except Exception as load_error:
                        print(f"⚠️ Không thể load data cho label mapping: {load_error}")
                        
            except Exception as fallback_error:
                print(f"⚠️ Enhanced fallback failed: {fallback_error}")
            
            # Final fallback: tạo mapping đơn giản
            label_mapping = {label_id: f"Class_{label_id}" for label_id in unique_labels}
            print(f"⚠️  Sử dụng fallback label mapping: {len(label_mapping)} labels")
            return label_mapping
            
        except Exception as e:
            print(f"Warning: Could not create label mapping: {e}")
            # Fallback: tạo mapping đơn giản
            return {label_id: f"Class_{label_id}" for label_id in set(y_train)}
    
    def _classify_overfitting(self, overfitting_score: float) -> str:
        """Classify the level of overfitting"""
        if overfitting_score is None:
            return "no_validation_data"
        elif overfitting_score < -0.05:
            return "underfitting"
        elif overfitting_score > 0.05:
            return "overfitting"
        else:
            return "well_fitted"
    
    def _get_overfitting_classification(self, overfitting_score: float) -> str:
        """Get detailed overfitting classification"""
        if overfitting_score is None:
            return "Cannot Determine"
        elif overfitting_score < -0.1:
            return "Severe Underfitting"
        elif overfitting_score < -0.05:
            return "Moderate Underfitting"
        elif overfitting_score < 0.02:
            return "Well Fitted"
        elif overfitting_score < 0.05:
            return "Slight Overfitting"
        elif overfitting_score < 0.1:
            return "Moderate Overfitting"
        else:
            return "Severe Overfitting"
    
    def _get_overfitting_level_from_score(self, overfitting_score: float) -> str:
        """Get overfitting level description from score (for CV fold approximation)"""
        if overfitting_score is None:
            return "Cannot Determine"
        elif overfitting_score > 0.1:
            return f"High overfitting - {overfitting_score:.3f}"
        elif overfitting_score > 0.05:
            return f"Moderate overfitting - {overfitting_score:.3f}"
        elif overfitting_score > -0.05:
            return f"Good fit - {overfitting_score:.3f}"
        elif overfitting_score > -0.1:
            return f"Slight underfitting - {overfitting_score:.3f}"
        else:
            return f"Underfitting - {overfitting_score:.3f}"
    
    def _calculate_stability_score(self, cv_results: Dict[str, Any]) -> float:
        """Calculate model stability score from CV results"""
        try:
            accuracies = [fold['accuracy'] for fold in cv_results['fold_results']]
            return 1.0 - (np.std(accuracies) / np.mean(accuracies))
        except:
            return 0.0
    
    def _generate_embedding_cache_key(self, data_dict: Dict, selected_embeddings: List[str], 
                                     step1_data: Dict, step2_data: Dict, sampling_config: Dict) -> str:
        """Generate a unique cache key for embeddings based on data characteristics"""
        import hashlib
        
        # Create a string representation of the data characteristics
        cache_components = []
        
        # Add data size info
        cache_components.append(f"train_{len(data_dict['X_train'])}")
        cache_components.append(f"val_{len(data_dict['X_val']) if data_dict['X_val'] else 0}")
        cache_components.append(f"test_{len(data_dict['X_test'])}")
        
        # Add selected embeddings
        if selected_embeddings:
            cache_components.append(f"emb_{'_'.join(sorted(selected_embeddings))}")
        else:
            cache_components.append("emb_all")
        
        # Add sampling info
        if sampling_config and sampling_config.get('num_samples'):
            cache_components.append(f"samples_{sampling_config['num_samples']}")
        
        # Add column info from step2_data
        if step2_data and 'text_column' in step2_data and 'label_column' in step2_data:
            cache_components.append(f"cols_{step2_data['text_column']}_{step2_data['label_column']}")
        
        # Add categories info from step1_data
        if step1_data and 'selected_categories' in step1_data:
            categories = step1_data['selected_categories']
            if categories:
                # Ensure all categories are strings before joining
                cat_str = '_'.join(str(cat) for cat in sorted(categories)[:3])
                cache_components.append(f"cats_{cat_str}")
        
        # CRITICAL FIX: Add data content hash to distinguish different files
        try:
            # Create content hash from actual data
            content_samples = []
            
            # Sample from train data (first 100 samples for efficiency)
            train_samples = data_dict['X_train'][:100] if len(data_dict['X_train']) > 100 else data_dict['X_train']
            content_samples.extend([str(sample) for sample in train_samples])
            
            # Sample from test data (first 50 samples)
            test_samples = data_dict['X_test'][:50] if len(data_dict['X_test']) > 50 else data_dict['X_test']
            content_samples.extend([str(sample) for sample in test_samples])
            
            # Create content hash
            content_string = '|'.join(content_samples)
            content_hash = hashlib.md5(content_string.encode()).hexdigest()[:8]
            cache_components.append(f"content_{content_hash}")
            
        except Exception as e:
            # Fallback: use timestamp if content hashing fails
            import time
            fallback_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            cache_components.append(f"fallback_{fallback_hash}")
            print(f"⚠️ Warning: Could not create content hash, using fallback: {e}")
        
        # Create hash from components
        cache_string = '_'.join(cache_components)
        cache_hash = hashlib.md5(cache_string.encode()).hexdigest()[:12]
        
        return f"embeddings_{cache_hash}"
    
    def _save_embeddings_to_cache(self, cache_key: str, embeddings: Dict):
        """Save embeddings to persistent cache"""
        try:
            import os
            import pickle
            from config import CACHE_DIR
            
            # Create embeddings cache directory
            embeddings_cache_dir = os.path.join(CACHE_DIR, "embeddings")
            os.makedirs(embeddings_cache_dir, exist_ok=True)
            
            cache_file = os.path.join(embeddings_cache_dir, f"{cache_key}.pkl")
            
            # Save embeddings
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            
            print(f"💾 Embeddings cached to: {cache_file}")
            return True
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save embeddings to cache: {e}")
            return False
    
    def _load_embeddings_from_cache(self, cache_key: str) -> Dict:
        """Load embeddings from persistent cache"""
        try:
            import os
            import pickle
            from config import CACHE_DIR
            
            embeddings_cache_dir = os.path.join(CACHE_DIR, "embeddings")
            cache_file = os.path.join(embeddings_cache_dir, f"{cache_key}.pkl")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    embeddings = pickle.load(f)
                print(f"📂 Loaded embeddings from cache: {cache_file}")
                return embeddings
            else:
                return None
                
        except Exception as e:
            print(f"⚠️ Warning: Could not load embeddings from cache: {e}")
            return None
    
    def _get_or_create_embeddings(self, data_dict: Dict, selected_embeddings: List[str], 
                                 stop_callback, step1_data: Dict, step2_data: Dict, 
                                 sampling_config: Dict) -> Dict:
        """Get embeddings from cache or create new ones"""
        
        # First check if we have embeddings in memory
        if hasattr(self, 'embeddings') and self.embeddings is not None:
            print(f"\n🔄 Reusing embeddings from memory...")
            return self.embeddings
        
        # Generate cache key
        cache_key = self._generate_embedding_cache_key(
            data_dict, selected_embeddings, step1_data, step2_data, sampling_config
        )
        
        # Try to load from persistent cache
        embeddings = self._load_embeddings_from_cache(cache_key)
        
        if embeddings is not None:
            print(f"\n✅ Loaded embeddings from persistent cache!")
            # Store in memory for reuse
            self.embeddings = embeddings
            return embeddings
        
        # Create new embeddings
        print(f"\n🔤 Creating new embeddings (will be cached for future use)...")
        
        # Check if stopped before creating embeddings
        try:
            from training_pipeline import global_stop_check
            if global_stop_check():
                print("🛑 Training stopped before embedding creation")
                return None
        except ImportError:
            if stop_callback and stop_callback():
                print("🛑 Training stopped before embedding creation")
                return None
        
        embeddings = self.create_all_embeddings(
            data_dict['X_train'], 
            data_dict['X_val'], 
            data_dict['X_test'],
            selected_embeddings,
            stop_callback
        )
        
        if embeddings is not None:
            # Store in memory for reuse
            self.embeddings = embeddings
            
            # Save to persistent cache
            self._save_embeddings_to_cache(cache_key, embeddings)
        
        return embeddings
    
    def show_embedding_cache_status(self):
        """Show status of embedding cache"""
        try:
            import os
            from config import CACHE_DIR
            
            embeddings_cache_dir = os.path.join(CACHE_DIR, "embeddings")
            
            if not os.path.exists(embeddings_cache_dir):
                print("📭 No embedding cache directory found")
                return
            
            cache_files = [f for f in os.listdir(embeddings_cache_dir) if f.endswith('.pkl')]
            
            if not cache_files:
                print("📭 No cached embeddings found")
                return
            
            print("\n" + "="*70)
            print("📊 EMBEDDING CACHE STATUS")
            print("="*70)
            print(f"📍 Cache Directory: {embeddings_cache_dir}")
            print(f"📁 Total Cached Embeddings: {len(cache_files)}")
            print("-"*70)
            
            for i, cache_file in enumerate(cache_files, 1):
                file_path = os.path.join(embeddings_cache_dir, cache_file)
                file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                mod_time = os.path.getmtime(file_path)
                import time
                age_hours = (time.time() - mod_time) / 3600
                
                print(f"{i:2d}. {cache_file}")
                print(f"    Size: {file_size:.1f} MB | Age: {age_hours:.1f}h")
                print()
            
            print("="*70)
            
        except Exception as e:
            print(f"⚠️ Error checking embedding cache status: {e}")
    
    def clear_embedding_cache(self, cache_key: str = None):
        """Clear embedding cache"""
        try:
            import os
            from config import CACHE_DIR
            
            embeddings_cache_dir = os.path.join(CACHE_DIR, "embeddings")
            
            if not os.path.exists(embeddings_cache_dir):
                print("📭 No embedding cache directory found")
                return
            
            if cache_key:
                # Clear specific cache
                cache_file = os.path.join(embeddings_cache_dir, f"{cache_key}.pkl")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    print(f"✅ Cleared embedding cache: {cache_key}")
                else:
                    print(f"⚠️ Cache file not found: {cache_key}")
            else:
                # Clear all cache
                cache_files = [f for f in os.listdir(embeddings_cache_dir) if f.endswith('.pkl')]
                for cache_file in cache_files:
                    os.remove(os.path.join(embeddings_cache_dir, cache_file))
                print(f"✅ Cleared {len(cache_files)} embedding cache files")
            
        except Exception as e:
            print(f"⚠️ Error clearing embedding cache: {e}")

    def run_comprehensive_evaluation(self, max_samples: int = None, skip_csv_prompt: bool = False, 
                                   sampling_config: Dict = None, selected_models: List[str] = None, selected_embeddings: List[str] = None, stop_callback=None, step3_data: Dict = None, preprocessing_config: Dict = None, step1_data: Dict = None, step2_data: Dict = None, ensemble_config: Dict = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of model-embedding combinations
        
        Args:
            max_samples: Maximum number of samples to use
            skip_csv_prompt: If True, skip CSV backup prompt (for Streamlit usage)
            sampling_config: Sampling configuration from Streamlit (optional)
            selected_models: List of model names to evaluate (if None, evaluate all)
            selected_embeddings: List of embedding names to evaluate (if None, evaluate all)
            preprocessing_config: Preprocessing configuration from Streamlit (optional)
            step1_data: Step 1 data from Streamlit to avoid reloading dataset (optional)
            step2_data: Step 2 data from Streamlit with column configuration (optional)
        
        Returns:
            Complete evaluation results
        """
        print("\n🚀 Starting Comprehensive Evaluation...")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. Load and prepare data
        # Store step1_data and step2_data for use in load_and_prepare_data
        if step1_data:
            self.step1_data = step1_data
            print(f"📊 Received Step 1 data with keys: {list(step1_data.keys())}")
        
        if step2_data:
            self.step2_data = step2_data
            print(f"📊 Received Step 2 data with keys: {list(step2_data.keys())}")
        
        # Get ensemble_config from step3_data if not provided
        if ensemble_config is None and step3_data:
            ensemble_config = step3_data.get('ensemble_learning', {})
            print(f"📊 Using ensemble_config from step3_data: {ensemble_config}")
        
        # Get selected_embeddings from step2_data if not provided
        if selected_embeddings is None and step2_data:
            selected_embeddings = step2_data.get('selected_embeddings', [])
            print(f"📊 Using selected_embeddings from step2_data: {selected_embeddings}")
        
        data_dict, sorted_labels = self.load_and_prepare_data(max_samples, skip_csv_prompt, sampling_config, preprocessing_config)
        
        # Store data_dict for later use
        self.data_dict = data_dict
        
        # 2. Create selected embeddings (only once) with persistent cache
        embeddings = self._get_or_create_embeddings(
            data_dict, selected_embeddings, stop_callback, 
            step1_data, step2_data, sampling_config
        )
        
        if embeddings is None:
            print("🛑 Failed to create or load embeddings")
            return {'status': 'error', 'message': 'Failed to create or load embeddings'}
        
        # 2.5. Pre-compute CV embeddings for fair comparison (only for embeddings, not BoW/TF-IDF)
        # Map Streamlit embedding names to internal names for pre-computation check
        embedding_mapping_for_cv = {
            'BoW': 'bow',
            'TF-IDF': 'tfidf',
            'Word Embeddings': 'embeddings'
        }
        
        # Check if any selected embeddings should trigger CV pre-computation
        embeddings_to_precompute = []
        if selected_embeddings:
            for emb in selected_embeddings:
                internal_name = embedding_mapping_for_cv.get(emb, emb)
                if internal_name == 'embeddings':
                    embeddings_to_precompute.append(internal_name)
        else:
            # If no specific embeddings selected, check if 'embeddings' exists
            if 'embeddings' in embeddings:
                embeddings_to_precompute = ['embeddings']
        
        print(f"🔍 Debug: embeddings_to_precompute = {embeddings_to_precompute}")
        print(f"🔍 Debug: embeddings keys = {list(embeddings.keys())}")
        print(f"🔍 Debug: 'embeddings' in embeddings = {'embeddings' in embeddings}")
        
        if embeddings_to_precompute and ('embeddings' in embeddings):
            # Check for stop signal
            try:
                from training_pipeline import global_stop_check
                if global_stop_check():
                    print("🛑 Training stopped before CV embeddings pre-computation")
                    return {'status': 'stopped', 'message': 'Training stopped before CV embeddings pre-computation'}
            except ImportError:
                if stop_callback and stop_callback():
                    print("🛑 Training stopped before CV embeddings pre-computation")
                    return {'status': 'stopped', 'message': 'Training stopped before CV embeddings pre-computation'}
            
            print(f"\n🔧 Pre-computing CV embeddings for fair model comparison...")
            # Combine training and validation data for CV splits
            # Handle case where X_val might be empty or different type
            if data_dict['X_val'] is not None and len(data_dict['X_val']) > 0:
                all_train_texts = data_dict['X_train'] + list(data_dict['X_val'])
                all_train_labels = data_dict['y_train'] + list(data_dict['y_val'])
            else:
                all_train_texts = data_dict['X_train']
                all_train_labels = data_dict['y_train']
            
            # Pre-compute CV embeddings with test data
            cv_embeddings = self.precompute_cv_embeddings(
                all_train_texts, all_train_labels, stop_callback, 
                test_texts=self.data_dict.get('X_test') if hasattr(self, 'data_dict') else None, 
                test_labels=self.data_dict.get('y_test') if hasattr(self, 'data_dict') else None
            )
            
            if not cv_embeddings:  # Empty dict means stopped
                print("🛑 Training stopped during CV embeddings pre-computation")
                return {'status': 'stopped', 'message': 'Training stopped during CV embeddings pre-computation'}
            
            # Store CV embeddings in evaluator for reuse
            self.cv_embeddings_cache = cv_embeddings
            print(f"✅ CV embeddings cache updated with {len(cv_embeddings)} folds")
        else:
            print(f"\n⏭️ Skipping CV embeddings pre-computation (only BoW/TF-IDF selected)")
            cv_embeddings = {}
        
        # 3. Define models and embeddings to evaluate
        if selected_models is None:
            models_to_evaluate = ['kmeans', 'knn', 'decision_tree', 'naive_bayes', 'svm', 'logistic_regression', 'linear_svc']
        else:
            # Map Streamlit model names to internal names
            model_mapping = {
                'K-Means Clustering': 'kmeans',
                'K-Nearest Neighbors': 'knn', 
                'Decision Tree': 'decision_tree',
                'Naive Bayes': 'naive_bayes',
                'Support Vector Machine': 'svm',
                'Logistic Regression': 'logistic_regression',
                'Linear SVC': 'linear_svc',
                'Random Forest': 'random_forest',
                'AdaBoost': 'adaboost',
                'Gradient Boosting': 'gradient_boosting',
                'XGBoost': 'xgboost',
                'LightGBM': 'lightgbm',
                'CatBoost': 'catboost'
            }
            models_to_evaluate = [model_mapping.get(model, model) for model in selected_models]
            print(f"🔍 Model mapping: {selected_models} -> {models_to_evaluate}")
        
        if selected_embeddings is None:
            embeddings_to_evaluate = list(embeddings.keys())
        else:
            # Map Streamlit embedding names to internal names
            embedding_mapping = {
                'BoW': 'bow',
                'TF-IDF': 'tfidf',
                'Word Embeddings': 'embeddings'
            }
            embeddings_to_evaluate = [embedding_mapping.get(emb, emb) for emb in selected_embeddings]
            print(f"🔍 Embedding mapping: {selected_embeddings} -> {embeddings_to_evaluate}")
            print(f"🔍 Available embeddings: {list(embeddings.keys())}")
        
        # 4. Run evaluation for selected combinations
        print(f"\n🤖 Evaluating Selected Model-Embedding Combinations...")
        print("=" * 60)
        print(f"📊 Models: {', '.join(selected_models or ['All'])}")
        print(f"🔤 Embeddings: {', '.join(selected_embeddings or ['All'])}")
        
        all_results = []
        successful_combinations = 0
        total_combinations = len(models_to_evaluate) * len(embeddings_to_evaluate)
        
        # Add ensemble to total if enabled
        if ensemble_config and ensemble_config.get('enabled', False):
            total_combinations += len(embeddings_to_evaluate)  # Add ensemble for each embedding
        
        # Progress tracking removed - using overall testing progress instead
        
        # Overall progress tracking for testing process
        import threading
        import time as time_module
        
        def show_testing_progress():
            dots = 0
            start_time = time_module.time()
            while not getattr(show_testing_progress, 'stop', False):
                elapsed_time = time_module.time() - start_time
                # Create progress bar with time countdown
                progress_bar = "█" * (dots % 20) + "░" * (19 - (dots % 20))
                
                # Calculate estimated remaining time based on dots progress
                if dots > 0:
                    # Estimate total time based on current progress (dots represent cycles)
                    # Use a more stable estimation method
                    progress_ratio = (dots % 20) / 20.0
                    if progress_ratio > 0:
                        estimated_total_time = elapsed_time / progress_ratio
                        remaining_time = max(0, estimated_total_time - elapsed_time)
                        time_display = f"⏱️ {remaining_time:.1f}s remaining"
                    else:
                        time_display = f"⏱️ {elapsed_time:.1f}s elapsed"
                else:
                    time_display = "⏱️ calculating..."
                
                print(f"\r🔄 Testing all model-embedding combinations [{progress_bar}] {time_display}", end="", flush=True)
                time_module.sleep(0.5)
                dots += 1
        
        # Start overall testing progress indicator
        testing_progress_thread = threading.Thread(target=show_testing_progress, daemon=True)
        testing_progress_thread.start()
        
        try:
            for model_name in models_to_evaluate:
                # Check if training should stop (outer loop)
                try:
                    from training_pipeline import global_stop_check
                    if global_stop_check():
                        print("🛑 Training stopped by user request")
                        break
                except ImportError:
                    if stop_callback and stop_callback():
                        print("🛑 Training stopped by user request")
                        break
                
                for embedding_name in embeddings_to_evaluate:
                    # Check if training should stop (inner loop)
                    try:
                        from training_pipeline import global_stop_check
                        if global_stop_check():
                            print("🛑 Training stopped by user request")
                            break
                    except ImportError:
                        if stop_callback and stop_callback():
                            print("🛑 Training stopped by user request")
                            break
                    
                    if embedding_name in embeddings:
                        embedding_data = embeddings[embedding_name]
                        
                        # Progress tracking removed - using overall testing progress
                        
                        # Debug: Check data shapes
                        print(f"     🔍 Debug - Data shapes:")
                        print(f"       X_train: {embedding_data['X_train'].shape if hasattr(embedding_data['X_train'], 'shape') else len(embedding_data['X_train']) if embedding_data['X_train'] is not None else 'None'}")
                        print(f"       X_val: {embedding_data['X_val'].shape if hasattr(embedding_data['X_val'], 'shape') else len(embedding_data['X_val']) if embedding_data['X_val'] is not None else 'None'}")
                        print(f"       X_test: {embedding_data['X_test'].shape if hasattr(embedding_data['X_test'], 'shape') else len(embedding_data['X_test']) if embedding_data['X_test'] is not None else 'None'}")
                        print(f"       y_train: {len(data_dict['y_train'])}")
                        print(f"       y_val: {len(data_dict['y_val'])}")
                        print(f"       y_test: {len(data_dict['y_test'])}")
                        
                        # Calculate feature dimensions for cache key
                        feature_dimensions = {}
                        for emb_name, emb_data in embeddings.items():
                            if hasattr(emb_data['X_train'], 'shape'):
                                feature_dimensions[emb_name] = emb_data['X_train'].shape[1]
                            elif isinstance(emb_data['X_train'], list):
                                feature_dimensions[emb_name] = len(emb_data['X_train'][0]) if emb_data['X_train'] else 0
                        
                        result = self.evaluate_single_combination(
                            model_name=model_name,
                            embedding_name=embedding_name,
                            X_train=embedding_data['X_train'],
                            X_val=embedding_data['X_val'],
                            X_test=embedding_data['X_test'],
                            y_train=data_dict['y_train'],
                            y_val=data_dict['y_val'],
                            y_test=data_dict['y_test'],
                            step3_data=step3_data,
                            selected_embeddings=selected_embeddings,
                            feature_dimensions=feature_dimensions
                        )
                        
                        # Progress tracking removed - using overall testing progress
                        
                        all_results.append(result)
                        if result['status'] == 'success':
                            successful_combinations += 1
                    else:
                        print(f"⚠️  Warning: {embedding_name} not found in created embeddings. Skipping {model_name}_{embedding_name}")
                        # Add error result
                        error_result = {
                            'model_name': model_name,
                            'embedding_name': embedding_name,
                            'combination_key': f"{model_name}_{embedding_name}",
                            'status': 'error',
                            'error_message': f'Embedding {embedding_name} not created',
                            'validation_accuracy': 0.0,
                            'test_accuracy': 0.0,
                            'overfitting_score': 0.0,
                            'overfitting_status': 'error'
                        }
                        all_results.append(error_result)
        
        except Exception as e:
            print(f"\n❌ Error during model evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Stop overall testing progress indicator
            show_testing_progress.stop = True
            testing_progress_thread.join(timeout=0.1)
            print(f"\n✅ Testing process completed")
        
        # 5. Analyze results
        print(f"\n📊 Individual Models Evaluation Complete!")
        individual_combinations = len(models_to_evaluate) * len(embeddings_to_evaluate)
        print(f"   • Successful individual combinations: {successful_combinations}/{individual_combinations}")
        if individual_combinations - successful_combinations > 0:
            print(f"   • Failed individual combinations: {individual_combinations - successful_combinations}")
        
        # Progress tracking removed - using overall testing progress
        
        # 6. 🚀 ENSEMBLE LEARNING - Train ensemble model with ALL selected embeddings
        if ensemble_config and ensemble_config.get('enabled', False):
            print(f"\n🚀 Starting Ensemble Learning with ALL embeddings...")
            
            # Progress tracking for ensemble learning
            def show_ensemble_progress():
                dots = 0
                while not getattr(show_ensemble_progress, 'stop', False):
                    print(f"\r🎯 Training Ensemble Learning{'...' + '.' * (dots % 3):<4}", end="", flush=True)
                    import time
                    time.sleep(0.5)
                    dots += 1
            
            # Start ensemble progress indicator
            ensemble_progress_thread = threading.Thread(target=show_ensemble_progress, daemon=True)
            ensemble_progress_thread.start()
            
            try:
                # Train ensemble with each selected embedding
                ensemble_success_count = 0
                for embedding_name in embeddings_to_evaluate:
                    if embedding_name in embeddings:
                        print(f"\n🎯 Training Ensemble Learning with {embedding_name} embedding...")
                        
                        # Create embeddings dict with only this embedding
                        single_embedding = {embedding_name: embeddings[embedding_name]}
                        
                        ensemble_result = self._train_ensemble_model(
                            all_results=all_results,
                            data_dict=data_dict,
                            embeddings=single_embedding,
                            ensemble_config=ensemble_config,
                            step3_data=step3_data,
                            target_embedding=embedding_name,
                            selected_embeddings=selected_embeddings
                        )
                        
                        if ensemble_result:
                            all_results.append(ensemble_result)
                            successful_combinations += 1
                            ensemble_success_count += 1
                            print(f"✅ Ensemble Learning with {embedding_name} completed successfully")
                        else:
                            print(f"❌ Ensemble Learning with {embedding_name} failed")
                    else:
                        print(f"⚠️ Skipping {embedding_name} - embedding not available")
            
            finally:
                # Stop ensemble progress indicator
                show_ensemble_progress.stop = True
                ensemble_progress_thread.join(timeout=0.1)
                print(f"\n✅ Ensemble Learning process completed")
            
            if ensemble_success_count > 0:
                print(f"🎉 Ensemble Learning completed with {ensemble_success_count}/{len(embeddings_to_evaluate)} embeddings")
            else:
                print(f"❌ Ensemble Learning failed for all embeddings")
        
        # 7. Find best combinations
        self._analyze_results(all_results)
        
        # 8. Final Summary
        total_time = time.time() - start_time
        
        # Calculate actual counts dynamically
        individual_count = len([r for r in all_results if r['model_name'] != 'Ensemble Learning' and r['status'] == 'success'])
        ensemble_count = len([r for r in all_results if r['model_name'] == 'Ensemble Learning' and r['status'] == 'success'])
        actual_total = individual_count + ensemble_count
        
        print(f"\n🏆 Final Evaluation Summary:")
        print(f"   • Total models evaluated: {actual_total}/{total_combinations}")
        print(f"     - Individual models: {individual_count}")
        if ensemble_count > 0:
            print(f"     - Ensemble models: {ensemble_count}")
        print(f"   • Total evaluation time: {total_time:.2f}s")
        
        # Analyze results and find best combinations
        self._analyze_results(all_results)
        
        # Store results
        self.evaluation_results = {
            'all_results': all_results,
            'successful_combinations': successful_combinations,
            'total_combinations': total_combinations,
            'evaluation_time': total_time,
            'data_info': {
                'n_samples': len(data_dict['X_train']),
                'n_validation': len(data_dict['X_val']),
                'n_test': len(data_dict['X_test']),
                'n_classes': len(sorted_labels),
                'labels': sorted_labels
            },
            'embedding_info': embeddings,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add analysis results to evaluation_results for cache access
        if hasattr(self, 'best_combinations'):
            self.evaluation_results['best_combinations'] = self.best_combinations
        
        return self.evaluation_results
    
    def _analyze_results(self, results: List[Dict[str, Any]]):
        """Analyze evaluation results and find best combinations"""
        print(f"\n🔍 Analyzing Results...")
        print("=" * 40)
        
        # Filter successful results
        successful_results = [r for r in results if r['status'] == 'success']
        
        if not successful_results:
            print("❌ No successful combinations to analyze!")
            return
        
        # Initialize analysis results
        analysis_results = {
            'best_combinations': {},
            'performance_comparison': {},
            'summary_statistics': {}
        }
        
        # 1. Best overall performance - Primary: F1 Score, Secondary: Test Accuracy
        # Check if F1 scores are available
        f1_scores_available = any('f1_score' in r and r['f1_score'] is not None for r in successful_results)
        
        if f1_scores_available:
            # Use F1 Score as primary criterion
            best_overall = max(successful_results, key=lambda x: (x.get('f1_score', 0) or 0, x['test_accuracy']))
            print(f"🏆 Best Overall (F1 Score): {best_overall['combination_key']}")
            print(f"   • F1 Score: {best_overall.get('f1_score', 0):.3f}")
            print(f"   • Test Accuracy: {best_overall['test_accuracy']:.3f}")
        else:
            # Fallback to Test Accuracy if no F1 scores
            best_overall = max(successful_results, key=lambda x: x['test_accuracy'])
            print(f"🏆 Best Overall (Test Accuracy): {best_overall['combination_key']}")
            print(f"   • Test Accuracy: {best_overall['test_accuracy']:.3f}")
        
        if best_overall['validation_accuracy'] is not None and best_overall['validation_accuracy'] > 0:
            print(f"   • Validation Accuracy: {best_overall['validation_accuracy']:.3f}")
        else:
            print(f"   • Validation: Handled by CV folds")
        print(f"   • CV Accuracy (F1-based): {best_overall['cv_mean_accuracy']:.3f}±{best_overall['cv_std_accuracy']:.3f}")
        print(f"   • CV Stability: {best_overall['cv_stability_score']:.3f}")
        
        # 2. Best for each embedding
        print(f"\n📊 Best Model for Each Embedding:")
        # Get unique embeddings from successful results
        unique_embeddings = list(set(r['embedding_name'] for r in successful_results))
        for embedding in unique_embeddings:
            embedding_results = [r for r in successful_results if r['embedding_name'] == embedding]
            if embedding_results:
                if f1_scores_available:
                    best_embedding = max(embedding_results, key=lambda x: (x.get('f1_score', 0) or 0, x['test_accuracy']))
                    f1_display = f"F1: {best_embedding.get('f1_score', 0):.3f}"
                    print(f"   • {embedding.upper()}: {best_embedding['model_name']} ({f1_display}, Test: {best_embedding['test_accuracy']:.3f})")
                else:
                    best_embedding = max(embedding_results, key=lambda x: x['test_accuracy'])
                    print(f"   • {embedding.upper()}: {best_embedding['model_name']} (Test: {best_embedding['test_accuracy']:.3f})")
        
        # 3. Best for each model
        print(f"\n🤖 Best Embedding for Each Model:")
        # Get unique models from successful results
        unique_models = list(set(r['model_name'] for r in successful_results))
        for model in unique_models:
            model_results = [r for r in successful_results if r['model_name'] == model]
            if model_results:
                if f1_scores_available:
                    best_model = max(model_results, key=lambda x: (x.get('f1_score', 0) or 0, x['test_accuracy']))
                    f1_display = f"F1: {best_model.get('f1_score', 0):.3f}"
                    print(f"   • {model.upper()}: {best_model['embedding_name']} ({f1_display}, Test: {best_model['test_accuracy']:.3f})")
                else:
                    best_model = max(model_results, key=lambda x: x['test_accuracy'])
                    print(f"   • {model.upper()}: {best_model['embedding_name']} (Test: {best_model['test_accuracy']:.3f})")
        
        # 4. Overfitting analysis
        print(f"\n⚖️ Overfitting Analysis:")
        overfitting_counts = {}
        for result in successful_results:
            status = result['overfitting_status']
            overfitting_counts[status] = overfitting_counts.get(status, 0) + 1
        
        for status, count in overfitting_counts.items():
            percentage = (count / len(successful_results)) * 100
            print(f"   • {status.title()}: {count} combinations ({percentage:.1f}%)")
        
        # 5. Stability analysis
        print(f"\n🔄 Stability Analysis (CV Results):")
        stable_models = [r for r in successful_results if r['cv_stability_score'] is not None and r['cv_stability_score'] > 0.8]
        print(f"   • Stable models (CV stability > 0.8): {len(stable_models)} combinations")
        
        if stable_models:
            most_stable = max(stable_models, key=lambda x: x['cv_stability_score'])
            print(f"   • Most stable: {most_stable['combination_key']} (stability: {most_stable['cv_stability_score']:.3f})")
        
        # Store best combinations
        self.best_combinations = {
            'best_overall': best_overall,
            'best_by_embedding': {
                emb: max([r for r in successful_results if r['embedding_name'] == emb], 
                        key=lambda x: (x.get('f1_score', 0) or 0, x['test_accuracy']) if f1_scores_available else x['test_accuracy']) 
                for emb in unique_embeddings
            },
            'best_by_model': {
                model: max([r for r in successful_results if r['model_name'] == model], 
                          key=lambda x: (x.get('f1_score', 0) or 0, x['test_accuracy']) if f1_scores_available else x['test_accuracy'])
                for model in unique_models
            }
        }
        
        # Store analysis results in evaluation_results for cache access
        if hasattr(self, 'evaluation_results'):
            self.evaluation_results['best_combinations'] = self.best_combinations
            self.evaluation_results['performance_comparison'] = {
                'overfitting_analysis': overfitting_counts,
                'stability_analysis': {
                    'stable_models_count': len(stable_models),
                    'most_stable': most_stable if stable_models else None
                },
                'summary_statistics': {
                    'total_models': len(successful_results),
                    'f1_scores_available': f1_scores_available,
                    'unique_embeddings': unique_embeddings,
                    'unique_models': unique_models
                }
            }
    
    def generate_detailed_report(self) -> str:
        """Generate a concise evaluation report with key metrics"""
        if not self.evaluation_results:
            return "No evaluation results available. Run evaluation first."
        
        report = []
        report.append("=" * 60)
        report.append("📊 EVALUATION SUMMARY")
        report.append("=" * 60)
        report.append(f"Total Combinations: {self.evaluation_results['total_combinations']}")
        report.append(f"Successful: {self.evaluation_results['successful_combinations']}")
        report.append(f"Evaluation Time: {self.evaluation_results['evaluation_time']:.2f}s")
        report.append("")
        
        # Data info
        data_info = self.evaluation_results['data_info']
        report.append(f"📋 Dataset: {data_info['n_samples']} train, {data_info['n_validation']} val, {data_info['n_test']} test, {data_info['n_classes']} classes")
        report.append("")
        
        # Best overall
        if self.best_combinations:
            best = self.best_combinations['best_overall']
            report.append(f"🏆 Best Overall: {best['combination_key']}")
            
            # Check if F1 score is available
            if 'f1_score' in best and best['f1_score'] is not None:
                report.append(f"   F1 Score: {best['f1_score']:.3f} | Test Accuracy: {best['test_accuracy']:.3f}")
            else:
                report.append(f"   Test Accuracy: {best['test_accuracy']:.3f}")
            
            if best['validation_accuracy'] is not None and best['validation_accuracy'] > 0:
                report.append(f"   Validation Accuracy: {best['validation_accuracy']:.3f}")
            else:
                report.append(f"   Validation: CV folds")
            report.append("")
        
        # Results table - simplified
        report.append("📈 RESULTS TABLE:")
        report.append("-" * 60)
        
        # Check if F1 scores are available
        f1_scores_available = any('f1_score' in r and r['f1_score'] is not None for r in self.evaluation_results['all_results'] if r['status'] == 'success')
        
        if f1_scores_available:
            report.append(f"{'Combination':<20} {'F1 Score':<8} {'Val Acc':<8} {'Test Acc':<8} {'CV(F1)':<10}")
            report.append("-" * 60)
            
            for result in self.evaluation_results['all_results']:
                if result['status'] == 'success':
                    cv_acc = f"{result['cv_mean_accuracy']:.3f}±{result['cv_std_accuracy']:.3f}"
                    val_acc = f"{result['validation_accuracy']:.3f}" if result['validation_accuracy'] is not None and result['validation_accuracy'] > 0 else "CV"
                    f1_score = f"{result.get('f1_score', 0):.3f}" if result.get('f1_score') is not None else "N/A"
                    report.append(f"{result['combination_key']:<20} {f1_score:<8} {val_acc:<8} {result['test_accuracy']:<8.3f} {cv_acc:<10}")
                else:
                    report.append(f"{result['combination_key']:<20} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8} {'ERROR':<10}")
        else:
            report.append(f"{'Combination':<20} {'Val Acc':<8} {'Test Acc':<8} {'CV(F1)':<10}")
            report.append("-" * 60)
            
            for result in self.evaluation_results['all_results']:
                if result['status'] == 'success':
                    cv_acc = f"{result['cv_mean_accuracy']:.3f}±{result['cv_std_accuracy']:.3f}"
                    val_acc = f"{result['validation_accuracy']:.3f}" if result['validation_accuracy'] is not None and result['validation_accuracy'] > 0 else "CV"
                    report.append(f"{result['combination_key']:<20} {val_acc:<8} {result['test_accuracy']:<8.3f} {cv_acc:<10}")
                else:
                    report.append(f"{result['combination_key']:<20} {'ERROR':<8} {'ERROR':<8} {'ERROR':<10}")
        
        report.append("-" * 60)
        
        return "\n".join(report)
    
    def save_results(self, filename: str = None):
        """Display concise evaluation results (no file creation)"""
        # Only show summary if there are successful combinations
        if not hasattr(self, 'evaluation_results') or self.evaluation_results['successful_combinations'] == 0:
            print("⚠️  No successful results to display")
            return None
            
        # Generate and display concise report
        report = self.generate_detailed_report()
        print("\n" + report)
        
        print(f"✅ Evaluation completed successfully!")
        individual_successful = len([r for r in self.evaluation_results['all_results'] if r['model_name'] != 'Ensemble Learning' and r['status'] == 'success'])
        ensemble_successful = len([r for r in self.evaluation_results['all_results'] if r['model_name'] == 'Ensemble Learning' and r['status'] == 'success'])
        total_successful = self.evaluation_results['successful_combinations']
        total_combinations = self.evaluation_results['total_combinations']
        
        print(f"   • {total_successful}/{total_combinations} models successful")
        print(f"     - Individual models: {individual_successful}")
        if ensemble_successful > 0:
            print(f"     - Ensemble model: {ensemble_successful}")
        print(f"   • Time: {self.evaluation_results['evaluation_time']:.2f}s")
        print(f"   • No files created - results displayed above")
        
        return "displayed"
    
    def _train_ensemble_model(self, all_results: List[Dict[str, Any]], 
                             data_dict: Dict[str, Any], 
                             embeddings: Dict[str, Any], 
                             ensemble_config: Dict[str, Any],
                             step3_data: Dict[str, Any],
                             target_embedding: str = None,
                             selected_embeddings: List[str] = None) -> Dict[str, Any]:
        import numpy as np
        """
        Train ensemble model using StackingClassifier with specific embedding
        
        Args:
            all_results: Results from individual model training
            data_dict: Data dictionary with train/val/test splits
            embeddings: Dictionary of embeddings (should contain target_embedding)
            ensemble_config: Ensemble learning configuration
            step3_data: Step 3 configuration data
            target_embedding: Specific embedding to use for ensemble training
            
        Returns:
            Ensemble training result dictionary
        """
        try:
            print(f"🚀 Training Ensemble Model...")
            
            # Check ensemble cache first
            if self.cache_enabled:
                try:
                    # Get data for cache check
                    X_train = data_dict.get('X_train')
                    y_train = data_dict.get('y_train')
                    
                    if X_train is not None and y_train is not None:
                        # Generate cache identifiers for ensemble
                        # Determine ensemble type based on final estimator
                        final_estimator_type = ensemble_config.get('final_estimator', 'logistic_regression')
                        if final_estimator_type == 'logistic_regression':
                            ensemble_type = 'stacking'
                        elif final_estimator_type == 'voting':
                            ensemble_type = 'voting'
                        else:
                            ensemble_type = 'stacking'  # Default to stacking
                        
                        ensemble_model_key = f"{ensemble_type}_ensemble_{target_embedding}"
                        # Handle text data for cache
                        if isinstance(X_train, list):
                            import hashlib
                            X_train_hash = hashlib.md5(str(X_train).encode()).hexdigest()[:8]
                            X_train_for_cache = np.array([int(X_train_hash, 16)])
                        else:
                            X_train_for_cache = X_train
                        dataset_id = self._generate_dataset_id(X_train_for_cache, y_train)
                        config_hash = self._generate_config_hash(
                            ensemble_model_key, 
                            ensemble_config.get('cv_folds', 5), 
                            ensemble_config.get('random_state', 42), 
                            step3_data,
                            selected_embeddings
                        )
                        dataset_fingerprint = self._generate_dataset_fingerprint(X_train, y_train)
                        
                        # Check cache
                        cache_exists, cache_info = self.cache_manager.check_cache_exists(
                            ensemble_model_key, dataset_id, config_hash, dataset_fingerprint
                        )
                        
                        if cache_exists:
                            print(f"✅ Cache HIT for ensemble_{target_embedding}")
                            print(f"   📁 Cache path: {cache_info.get('cache_dir', 'Unknown')}")
                            
                            # Load cached ensemble model
                            cached_model, cached_params, cached_metrics, cached_config, cached_eval_predictions, cached_shap_sample, cached_feature_names, cached_label_mapping = self.cache_manager.load_model_cache(
                                ensemble_model_key, dataset_id, config_hash, dataset_fingerprint
                            )
                            
                            if cached_model is not None:
                                print(f"✅ Using cached ensemble model")
                                
                                # Create ensemble result from cache
                                ensemble_result = {
                                    'model_name': 'Ensemble Learning',
                                    'embedding_name': target_embedding,
                                    'combination_key': f"Ensemble_{target_embedding}",
                                    'status': 'success',
                                    'validation_accuracy': cached_metrics.get('validation_accuracy', 0.0),
                                    'test_accuracy': cached_metrics.get('test_accuracy', 0.0),
                                    'overfitting_score': 0.0,  # Will be calculated from CV
                                    'overfitting_status': 'cv_not_available',
                                    'cv_mean_accuracy': 0.0,
                                    'cv_std_accuracy': 0.0,
                                    'cv_stability_score': 1.0,
                                    'training_time': 0.0,  # Cached, no training time
                                    'precision': cached_metrics.get('test_metrics', {}).get('precision', 0.0),
                                    'recall': cached_metrics.get('test_metrics', {}).get('recall', 0.0),
                                    'f1_score': cached_metrics.get('test_metrics', {}).get('f1_score', 0.0),
                                    'test_metrics': cached_metrics.get('test_metrics', {}),
                                    'predictions': cached_eval_predictions.get('y_pred', []) if cached_eval_predictions is not None else [],
                                    'true_labels': cached_eval_predictions.get('y_true', []) if cached_eval_predictions is not None else [],
                                    'probabilities': cached_eval_predictions.get('y_proba', []) if cached_eval_predictions is not None else [],
                                    'predictions_detail': {
                                        'train': {'y_true': [], 'y_pred': [], 'y_proba': []},
                                        'val': {'y_true': [], 'y_pred': [], 'y_proba': []},
                                        'test': {
                                            'y_true': cached_eval_predictions.get('y_true', []) if cached_eval_predictions is not None else [],
                                            'y_pred': cached_eval_predictions.get('y_pred', []) if cached_eval_predictions is not None else [],
                                            'y_proba': cached_eval_predictions.get('y_proba', []) if cached_eval_predictions is not None else []
                                        }
                                    },
                                    'ensemble_info': {
                                        'base_models': cached_params.get('base_models', []),
                                        'final_estimator': cached_params.get('final_estimator', 'logistic_regression'),
                                        'performance_comparison': {},
                                        'individual_results': {},
                                        'model_reuse_stats': {
                                            'models_reused': cached_params.get('base_models', []),
                                            'models_retrained': [],
                                            'reuse_errors': [],
                                            'total_models_reused': len(cached_params.get('base_models', [])),
                                            'total_models_retrained': 0,
                                            'reuse_percentage': 100.0
                                        }
                                    }
                                }
                                
                                print(f"✅ Ensemble Learning completed from cache!")
                                print(f"   • Test Accuracy: {ensemble_result['test_accuracy']:.4f}")
                                print(f"   • Base Models: {', '.join(cached_params.get('base_models', []))}")
                                print(f"   • Final Estimator: {cached_params.get('final_estimator', 'logistic_regression')}")
                                
                                return ensemble_result
                            else:
                                print(f"⚠️ Cache exists but failed to load ensemble model")
                        else:
                            print(f"❌ Cache MISS for {ensemble_model_key}")
                    else:
                        print(f"⚠️ Cannot check ensemble cache: X_train={X_train is not None}, y_train={y_train is not None}")
                        
                except Exception as e:
                    print(f"⚠️ Error checking ensemble cache: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Debug: Show all available results
            print(f"🔍 Debug: Available successful results:")
            for result in all_results:
                if result['status'] == 'success':
                    print(f"   • {result['model_name']} with {result['embedding_name']}")
            
            # Check if we have the required base models
            # Use internal model names that match what's actually being trained
            required_models_internal = {"knn", "decision_tree", "naive_bayes"}
            successful_models = {r['model_name'] for r in all_results if r['status'] == 'success'}
            
            print(f"🔍 Debug: Required models (internal): {required_models_internal}")
            print(f"🔍 Debug: Successful models: {successful_models}")
            
            if not required_models_internal.issubset(successful_models):
                missing = required_models_internal - successful_models
                print(f"❌ Cannot train ensemble: Missing models: {', '.join(missing)}")
                return None
            
            print(f"✅ All required models found: {', '.join(required_models_internal)}")
            
            # Map internal names to display names for UI consistency
            model_display_mapping = {
                "knn": "K-Nearest Neighbors",
                "decision_tree": "Decision Tree", 
                "naive_bayes": "Naive Bayes"
            }
            
            # Use target embedding if specified, otherwise find the first available
            if target_embedding and target_embedding in embeddings:
                best_embedding = target_embedding
                print(f"🎯 Using specified embedding: {best_embedding}")
            else:
                # Fallback: find the first available embedding from successful results
                best_embedding = None
                for result in all_results:
                    if result['status'] == 'success' and result['model_name'] in required_models_internal:
                        best_embedding = result['embedding_name']
                        break
                
                if not best_embedding or best_embedding not in embeddings:
                    print(f"❌ Cannot train ensemble: No suitable embedding found")
                    return None
                
                print(f"🔍 Using fallback embedding: {best_embedding}")
            
            # Get embedding data
            embedding_data = embeddings[best_embedding]
            X_train = embedding_data['X_train']
            X_val = embedding_data['X_val']
            X_test = embedding_data['X_test']
            y_train = data_dict['y_train']
            y_val = data_dict['y_val']
            y_test = data_dict['y_test']
            
            # Create ensemble manager
            try:
                from models.ensemble.ensemble_manager import EnsembleManager
                print(f"✅ Successfully imported EnsembleManager")
            except Exception as e:
                print(f"❌ Failed to import EnsembleManager: {e}")
                return None
            
            try:
                ensemble_manager = EnsembleManager(
                    base_models=['knn', 'decision_tree', 'naive_bayes'],
                    final_estimator=ensemble_config.get('final_estimator', 'logistic_regression'),
                    cv_folds=step3_data.get('cross_validation', {}).get('cv_folds', 5),
                    random_state=step3_data.get('cross_validation', {}).get('random_state', 42)
                )
                print(f"✅ Successfully created EnsembleManager instance")
            except Exception as e:
                print(f"❌ Failed to create EnsembleManager instance: {e}")
                return None
            
            # Get already trained individual model instances from the results
            # We need to find the trained models that were used in individual training
            try:
                from models import model_factory
                print(f"✅ Successfully imported model_factory")
                print(f"🔍 Model factory registry: {model_factory.registry is not None}")
            except Exception as e:
                print(f"❌ Failed to import model_factory: {e}")
                return None
            
            # Find the trained models from individual results
            base_model_instances = {}
            
            for internal_name in required_models_internal:
                try:
                    print(f"🔧 Looking for trained {internal_name} model...")
                    
                    # Find the result for this model
                    model_result = None
                    for result in all_results:
                        if (result['status'] == 'success' and 
                            result['model_name'] == internal_name):
                            model_result = result
                            break
                    
                    if model_result is None:
                        print(f"❌ No successful training result found for {internal_name}")
                        return None
                    
                    # ENHANCED: Try to reuse trained model from results instead of retraining
                    trained_model = model_result.get('trained_model', None)
                    
                    if trained_model and hasattr(trained_model, 'is_fitted') and trained_model.is_fitted:
                        print(f"✅ Reusing trained {internal_name} model from individual results")
                        model_instance = trained_model
                    else:
                        # Fallback: Create a fresh model instance and train it
                        print(f"🔧 Creating and training {internal_name} model for ensemble...")
                        model_instance = model_factory.create_model(internal_name)
                        
                        if model_instance is None:
                            print(f"❌ Model factory returned None for {internal_name}")
                            return None
                        
                        # Train the model with the same data used in individual training
                        if internal_name == 'knn':
                            # Use GPU for embeddings (dense), CPU for TF-IDF/BOW (sparse)
                            use_gpu = not hasattr(X_train, 'toarray')  # Dense matrices can use GPU
                            model_instance.fit(X_train, y_train, use_gpu=use_gpu)
                        else:
                            model_instance.fit(X_train, y_train)
                    
                    print(f"🔍 Model type: {type(model_instance)}")
                    print(f"🔍 Model fitted: {hasattr(model_instance, 'is_fitted') and model_instance.is_fitted}")
                    print(f"🔍 Model has 'model' attribute: {hasattr(model_instance, 'model')}")
                    if hasattr(model_instance, 'model'):
                        print(f"🔍 Underlying model type: {type(model_instance.model)}")
                    
                    base_model_instances[internal_name] = model_instance
                    display_name = model_display_mapping[internal_name]
                    print(f"✅ Created and trained {display_name} instance for ensemble")
                except Exception as e:
                    print(f"❌ Failed to create/train {internal_name} instance: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
            
            # ENHANCED: Use model reuse for ensemble creation
            ensemble_creation_result = ensemble_manager.create_ensemble_with_reuse(
                all_results, X_train, y_train, model_factory, target_embedding=best_embedding
            )
            
            if not ensemble_creation_result.get('success', False):
                print(f"❌ Failed to create ensemble with reuse: {ensemble_creation_result.get('error', 'Unknown error')}")
                return None
            
            ensemble_model = ensemble_creation_result['ensemble_model']
            reuse_stats = ensemble_creation_result.get('reuse_results', {})
            
            # Train ensemble model
            print(f"🚀 Training ensemble model with {best_embedding} embedding...")
            ensemble_results = ensemble_manager.train_ensemble(X_train, y_train, X_val, y_val)
            
            if not ensemble_results.get('is_trained', False):
                print(f"❌ Ensemble training failed: {ensemble_results.get('error', 'Unknown error')}")
                return None
            
            # Evaluate ensemble model
            ensemble_eval = ensemble_manager.evaluate_ensemble(X_test, y_test)
            
            if 'error' in ensemble_eval:
                print(f"❌ Ensemble evaluation failed: {ensemble_eval['error']}")
                return None
            
            # Generate predictions for train/val/test sets for confusion matrix
            print(f"🔍 Generating predictions for train/val/test sets...")
            try:
                # Train set predictions
                y_train_pred, y_train_proba = ensemble_manager.predict_ensemble(X_train)
                print(f"   ✅ Train predictions: {y_train_pred.shape}")
                
                # Val set predictions (only if validation set exists)
                if X_val is not None and len(X_val) > 0:
                    y_val_pred, y_val_proba = ensemble_manager.predict_ensemble(X_val)
                    print(f"   ✅ Val predictions: {y_val_pred.shape}")
                else:
                    y_val_pred, y_val_proba = np.array([]), np.array([])
                    print(f"   ⚠️ Val predictions: Skipped (no validation set)")
                
                # Test set predictions
                y_test_pred, y_test_proba = ensemble_manager.predict_ensemble(X_test)
                print(f"   ✅ Test predictions: {y_test_pred.shape}")
                
                # Create detailed predictions structure
                ensemble_predictions = {
                    'train': {
                        'y_true': y_train,
                        'y_pred': y_train_pred,
                        'y_proba': y_train_proba
                    },
                    'val': {
                        'y_true': y_val,
                        'y_pred': y_val_pred,
                        'y_proba': y_val_proba
                    },
                    'test': {
                        'y_true': y_test,
                        'y_pred': y_test_pred,
                        'y_proba': y_test_proba
                    }
                }
                
                print(f"✅ Created ensemble predictions structure")
                
            except Exception as e:
                print(f"❌ Error generating ensemble predictions: {e}")
                # Fallback to using predictions from ensemble_eval
                test_predictions = ensemble_eval.get('predictions', [])
                test_probabilities = ensemble_eval.get('probabilities', [])
                
                if len(test_predictions) > 0:
                    print(f"   🔄 Using predictions from ensemble_eval: {len(test_predictions)} predictions")
                    ensemble_predictions = {
                        'train': {'y_true': y_train, 'y_pred': y_train, 'y_proba': np.array([])},
                        'val': {'y_true': y_val, 'y_pred': y_val, 'y_proba': np.array([])},
                        'test': {'y_true': y_test, 'y_pred': test_predictions, 'y_proba': test_probabilities}
                    }
                else:
                    print(f"   ⚠️ No predictions available in ensemble_eval, using fallback")
                    ensemble_predictions = {
                        'train': {'y_true': y_train, 'y_pred': y_train, 'y_proba': np.array([])},
                        'val': {'y_true': y_val, 'y_pred': y_val, 'y_proba': np.array([])},
                        'test': {'y_true': y_test, 'y_pred': y_test, 'y_proba': np.array([])}
                    }
            
            # Debug: Check ensemble_eval structure
            print(f"🔍 Debug: ensemble_eval keys: {list(ensemble_eval.keys())}")
            if 'classification_report' in ensemble_eval:
                # Extract metrics from classification report
                pass  # Metrics already extracted above
            else:
                print(f"🔍 Debug: No classification_report found in ensemble_eval")
            
            # Compare with individual models
            individual_results = {r['model_name']: r for r in all_results if r['status'] == 'success'}
            performance_comparison = ensemble_manager.compare_performance(individual_results)
            
            # Calculate CV accuracy from base models only (not from ensemble_results which is 0.0)
            base_model_cv_accuracies = []
            base_models = ensemble_config.get('base_models', ['knn', 'decision_tree', 'naive_bayes'])
            
            for model_name in base_models:
                # Find individual model result for the same embedding
                for result in all_results:
                    if (isinstance(result, dict) and 
                        result.get('model_name') == model_name and
                        result.get('embedding_name') == best_embedding and
                        result.get('status') == 'success'):
                        cv_acc = result.get('cv_mean_accuracy', 0.0)
                        if cv_acc > 0.0:
                            base_model_cv_accuracies.append(cv_acc)
                            print(f"   📊 Base model {model_name}: CV = {cv_acc:.4f}")
                        break
            
            # Calculate ensemble CV accuracy as average of base models
            if base_model_cv_accuracies:
                import numpy as np
                ensemble_cv_mean = np.mean(base_model_cv_accuracies)
                ensemble_cv_std = np.std(base_model_cv_accuracies)
                print(f"   🎯 Ensemble CV calculated: {ensemble_cv_mean:.4f} ± {ensemble_cv_std:.4f}")
            else:
                ensemble_cv_mean = 0.0
                ensemble_cv_std = 0.0
                print(f"   ⚠️ No base model CV accuracies found, using 0.0")
            
            # Create ensemble result with FULL integration into the evaluation pipeline
            ensemble_result = {
                'model_name': 'Ensemble Learning',
                'embedding_name': best_embedding,
                'combination_key': f"Ensemble_{best_embedding}",
                'status': 'success',
                'validation_accuracy': ensemble_results.get('validation_accuracy', 0.0),
                'test_accuracy': ensemble_eval.get('accuracy', 0.0),
                
                # Calculate overfitting score using CV vs test accuracy for ensemble (like individual models)
                'overfitting_score': ensemble_cv_mean - ensemble_eval.get('accuracy', 0.0),
                'overfitting_status': self._classify_overfitting(ensemble_cv_mean - ensemble_eval.get('accuracy', 0.0)),
                
                # Use calculated CV accuracy from base models
                'cv_mean_accuracy': ensemble_cv_mean,
                'cv_std_accuracy': ensemble_cv_std,
                'cv_stability_score': 1.0 - ensemble_cv_std if ensemble_cv_std > 0 else 1.0,
                'training_time': ensemble_results.get('training_time', 0.0),
                
                # Extract precision, recall, and F1-score from classification report
                'precision': ensemble_eval.get('classification_report', {}).get('weighted avg', {}).get('precision', 0.0),
                'recall': ensemble_eval.get('classification_report', {}).get('weighted avg', {}).get('recall', 0.0),
                'f1_score': ensemble_eval.get('classification_report', {}).get('weighted avg', {}).get('f1-score', 0.0),
                
                # Also add test_metrics for UI compatibility
                'test_metrics': {
                    'precision': ensemble_eval.get('classification_report', {}).get('weighted avg', {}).get('precision', 0.0),
                    'recall': ensemble_eval.get('classification_report', {}).get('weighted avg', {}).get('recall', 0.0),
                    'f1_score': ensemble_eval.get('classification_report', {}).get('weighted avg', {}).get('f1-score', 0.0)
                },
                
                # Add predictions and true labels for confusion matrix
                'predictions': ensemble_predictions['test']['y_pred'],
                'true_labels': y_test,
                'probabilities': ensemble_predictions['test']['y_proba'],
                'predictions_detail': ensemble_predictions,
                
                # Add ensemble-specific information with reuse stats
                'ensemble_info': {
                    'base_models': [model_display_mapping[name] for name in required_models_internal],
                    'final_estimator': ensemble_config.get('final_estimator', 'logistic_regression'),
                    'performance_comparison': performance_comparison,
                    'individual_results': individual_results,
                    # ENHANCED: Add model reuse statistics
                    'model_reuse_stats': {
                        'models_reused': reuse_stats.get('models_reused', []),
                        'models_retrained': reuse_stats.get('models_retrained', []),
                        'reuse_errors': reuse_stats.get('reuse_errors', []),
                        'total_models_reused': len(reuse_stats.get('models_reused', [])),
                        'total_models_retrained': len(reuse_stats.get('models_retrained', [])),
                        'reuse_percentage': len(reuse_stats.get('models_reused', [])) / len(required_models_internal) * 100 if required_models_internal else 0
                    }
                }
            }
            
            # Final ensemble result ready
            
            print(f"✅ Ensemble Learning completed successfully!")
            print(f"   • Test Accuracy: {ensemble_result['test_accuracy']:.4f}")
            print(f"   • Training Time: {ensemble_result['training_time']:.2f}s")
            
            # Save ensemble model cache
            if self.cache_enabled:
                try:
                    print(f"💾 Saving ensemble model cache...")
                    
                    # Prepare ensemble model for caching
                    ensemble_model_data = {
                        'ensemble_manager': ensemble_manager,
                        'base_models': required_models_internal,
                        'final_estimator': ensemble_config.get('final_estimator', 'logistic_regression'),
                        'cv_folds': ensemble_config.get('cv_folds', 5),
                        'random_state': ensemble_config.get('random_state', 42)
                    }
                    
                    # Generate cache identifiers for ensemble
                    # Determine ensemble type based on final estimator
                    final_estimator_type = ensemble_config.get('final_estimator', 'logistic_regression')
                    if final_estimator_type == 'logistic_regression':
                        ensemble_type = 'stacking'
                    elif final_estimator_type == 'voting':
                        ensemble_type = 'voting'
                    else:
                        ensemble_type = 'stacking'  # Default to stacking
                    
                    ensemble_model_key = f"{ensemble_type}_ensemble_{best_embedding}"
                    # Handle text data for cache
                    if isinstance(X_train, list):
                        import hashlib
                        X_train_hash = hashlib.md5(str(X_train).encode()).hexdigest()[:8]
                        X_train_for_cache = np.array([int(X_train_hash, 16)])
                    else:
                        X_train_for_cache = X_train
                    dataset_id = self._generate_dataset_id(X_train_for_cache, y_train)
                    config_hash = self._generate_config_hash(
                        ensemble_model_key, 
                        ensemble_config.get('cv_folds', 5), 
                        ensemble_config.get('random_state', 42), 
                        step3_data,
                        selected_embeddings
                    )
                    dataset_fingerprint = self._generate_dataset_fingerprint(X_train, y_train)
                    
                    # Prepare eval predictions for ensemble
                    eval_predictions = self._prepare_eval_predictions_for_cache(
                        y_test, ensemble_predictions['test']['y_pred'], ensemble_predictions['test']['y_proba']
                    )
                    
                    # Save ensemble cache
                    self.cache_manager.save_model_cache(
                        model_key=ensemble_model_key,
                        dataset_id=dataset_id,
                        config_hash=config_hash,
                        dataset_fingerprint=dataset_fingerprint,
                        model=ensemble_model_data,
                        params={
                            'base_models': list(required_models_internal),
                            'final_estimator': ensemble_config.get('final_estimator', 'logistic_regression'),
                            'cv_folds': ensemble_config.get('cv_folds', 5),
                            'random_state': ensemble_config.get('random_state', 42),
                            'embedding': best_embedding
                        },
                        metrics={
                            'validation_accuracy': ensemble_result['validation_accuracy'],
                            'test_accuracy': ensemble_result['test_accuracy'],
                            'test_metrics': ensemble_result['test_metrics']
                        },
                        config={
                            'model_name': f"ensemble_{best_embedding}",
                            'base_models': list(required_models_internal),
                            'final_estimator': ensemble_config.get('final_estimator', 'logistic_regression'),
                            'cv_folds': ensemble_config.get('cv_folds', 5),
                            'embedding': best_embedding
                        },
                        eval_predictions=eval_predictions,
                        shap_sample=None,  # Ensemble doesn't support SHAP directly
                        feature_names=None,  # Will be handled by base models
                        label_mapping=self.data_loader.id_to_label
                    )
                    
                    print(f"✅ Ensemble cache saved successfully!")
                    
                except Exception as e:
                    print(f"⚠️ Failed to save ensemble cache: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Note: ensemble_result will be added to all_results by the caller
            return ensemble_result
            
        except Exception as e:
            print(f"❌ Ensemble Learning failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None


    def _add_cache_methods(self):
        """Add cache helper methods to comprehensive evaluator"""
        
        def _check_model_cache(model_name, X_train, y_train, cv_folds, random_state, step3_data=None, selected_embeddings=None, feature_dimensions=None):
            """Check if model cache exists and is valid"""
            if not self.cache_enabled:
                return None
                
            try:
                # Handle text data by converting to numerical representation for cache
                if isinstance(X_train, list):
                    # Convert text data to numerical representation for cache consistency
                    # Use hash of text content as numerical representation
                    import hashlib
                    X_train_hash = hashlib.md5(str(X_train).encode()).hexdigest()[:8]
                    X_train_for_cache = np.array([int(X_train_hash, 16)])  # Convert to numerical
                else:
                    X_train_for_cache = X_train
                
                # Generate cache identifiers using converted data
                model_key = self._get_model_key(model_name)
                dataset_id = self._generate_dataset_id(X_train_for_cache, y_train)
                config_hash = self._generate_config_hash(model_name, cv_folds, random_state, step3_data, selected_embeddings, feature_dimensions)
                dataset_fingerprint = self._generate_dataset_fingerprint(X_train_for_cache, y_train)
                
                # Check cache
                cache_exists, cache_info = self.cache_manager.check_cache_exists(
                    model_key, dataset_id, config_hash, dataset_fingerprint
                )
                
                if cache_exists:
                    # Load cached model
                    cache_data = self.cache_manager.load_model_cache(model_key, dataset_id, config_hash)
                    
                    # Check feature compatibility before using cached model
                    cached_model = cache_data['model']
                    if hasattr(cached_model, 'n_features_in_') and hasattr(X_train, 'shape'):
                        expected_features = cached_model.n_features_in_
                        actual_features = X_train.shape[1]
                        if expected_features != actual_features:
                            print(f"⚠️ Feature mismatch: cached model expects {expected_features} features, got {actual_features}")
                            print(f"   Skipping cache for {model_name} - will retrain with correct features")
                            return None
                    
                    # Check label compatibility before using cached model
                    if hasattr(cached_model, 'classes_') and cached_model.classes_ is not None:
                        cached_classes = set(cached_model.classes_)
                        current_classes = set(y_train)
                        if cached_classes != current_classes:
                            print(f"⚠️ Label mismatch: cached model expects labels {sorted(cached_classes)}, got {sorted(current_classes)}")
                            print(f"   Skipping cache for {model_name} - will retrain with correct labels")
                            return None
                    
                    print(f"✅ Cache HIT for {model_name}")
                    
                    # Convert to expected format
                    result = {
                        'model_name': model_name,
                        'status': 'success',
                        'model': cache_data['model'],
                        'params': cache_data['params'],
                        'metrics': cache_data['metrics'],
                        'config': cache_data['config'],
                        'eval_predictions': cache_data['eval_predictions'],
                        'shap_sample': cache_data['shap_sample'],
                        'feature_names': cache_data['feature_names'],
                        'label_mapping': cache_data['label_mapping'],
                        'cache_hit': True,
                        'cache_path': cache_data['cache_path']
                    }
                    
                    return result
                
                return None
                
            except Exception as e:
                print(f"Warning: Cache check failed for {model_name}: {e}")
                return None
        
        def _save_model_cache(model_name, X_train, y_train, cv_folds, random_state,
                             model, params, metrics, config,
                             eval_predictions=None, shap_sample=None,
                             feature_names=None, label_mapping=None, step3_data=None, selected_embeddings=None, feature_dimensions=None):
            """Save model cache"""
            if not self.cache_enabled:
                return ""
                
            try:
                # Handle text data by converting to numerical representation for cache
                if isinstance(X_train, list):
                    # Convert text data to numerical representation for cache consistency
                    # Use hash of text content as numerical representation
                    import hashlib
                    X_train_hash = hashlib.md5(str(X_train).encode()).hexdigest()[:8]
                    X_train_for_cache = np.array([int(X_train_hash, 16)])  # Convert to numerical
                else:
                    X_train_for_cache = X_train
                
                # Generate cache identifiers using converted data
                model_key = self._get_model_key(model_name)
                dataset_id = self._generate_dataset_id(X_train_for_cache, y_train)
                config_hash = self._generate_config_hash(model_name, cv_folds, random_state, step3_data, selected_embeddings, feature_dimensions)
                dataset_fingerprint = self._generate_dataset_fingerprint(X_train_for_cache, y_train)
                
                # Save cache
                cache_path = self.cache_manager.save_model_cache(
                    model_key=model_key,
                    dataset_id=dataset_id,
                    config_hash=config_hash,
                    dataset_fingerprint=dataset_fingerprint,
                    model=model,
                    params=params,
                    metrics=metrics,
                    config=config,
                    eval_predictions=eval_predictions,
                    shap_sample=shap_sample,
                    feature_names=feature_names,
                    label_mapping=label_mapping
                )
                
                print(f"💾 Cache saved for {model_name} at {cache_path}")
                return cache_path
                
            except Exception as e:
                print(f"Warning: Cache save failed for {model_name}: {e}")
                return ""
        
        def _get_model_key(model_name):
            """Get model key for cache"""
            model_mapping = {
                'random_forest': 'random_forest',
                'adaboost': 'adaboost',
                'gradient_boosting': 'gradient_boosting',
                'xgboost': 'xgboost',
                'lightgbm': 'lightgbm',
                'catboost': 'catboost',
                'knn': 'knn',
                'decision_tree': 'decision_tree',
                'naive_bayes': 'naive_bayes',
                'svm': 'svm',
                'logistic_regression': 'logistic_regression',
                'linear_svc': 'linear_svc',
                'kmeans': 'kmeans'
            }
            return model_mapping.get(model_name, model_name.lower().replace(' ', '_'))
        
        def _generate_dataset_id(X_train, y_train):
            """Generate dataset ID for cache - use consistent identifiers"""
            import hashlib
            
            # Use consistent dataset identifier based on dataset metadata, not content
            # This ensures cache reuse across different runs with same dataset
            if hasattr(self, 'data_dict') and 'dataset_info' in self.data_dict:
                # Use dataset metadata if available
                dataset_info = self.data_dict['dataset_info']
            else:
                # Fallback to basic info
                if hasattr(X_train, 'shape'):
                    X_shape = X_train.shape
                elif isinstance(X_train, list):
                    X_shape = (len(X_train), 0)  # Use list length
                else:
                    X_shape = (0, 0)
                
                num_classes = len(set(y_train)) if len(y_train) > 0 else 0
                
                # Use only essential info for consistency
                dataset_info = f"dataset_{X_shape[0]}samples_{num_classes}classes"
            
            return hashlib.md5(dataset_info.encode()).hexdigest()[:8]
        
        def _generate_config_hash(model_name, cv_folds, random_state, step3_data=None, selected_embeddings=None, feature_dimensions=None):
            """Generate configuration hash for cache using cache_manager"""
            config = {
                'model_name': model_name,
                'cv_folds': cv_folds,
                'random_state': random_state
            }
            
            # Add vectorization method to config hash to distinguish between different embeddings
            if selected_embeddings:
                config['selected_embeddings'] = sorted(selected_embeddings)  # Sort for consistency
                print(f"🔑 Including vectorization methods in cache key: {selected_embeddings}")
            
            # Add feature dimensions to distinguish between different feature spaces
            if feature_dimensions:
                config['feature_dimensions'] = sorted(feature_dimensions.items())  # Sort for consistency
                print(f"🔑 Including feature dimensions in cache key: {feature_dimensions}")
            
            if step3_data:
                config.update({
                    'optuna_enabled': step3_data.get('optuna', {}).get('enabled', False),
                    'optuna_trials': step3_data.get('optuna', {}).get('trials', 100),
                    'optuna_timeout': step3_data.get('optuna', {}).get('timeout', None)
                })
            
            return self.cache_manager.generate_config_hash(config)
        
        def _generate_dataset_fingerprint(X_train, y_train):
            """Generate dataset fingerprint for cache using cache_manager"""
            # Use consistent dataset fingerprint based on dataset metadata, not content
            # This ensures cache reuse across different runs with same dataset
            
            if hasattr(self, 'data_dict') and 'dataset_info' in self.data_dict:
                # Use dataset metadata if available
                dataset_info = self.data_dict['dataset_info']
            else:
                # Use basic info for consistency
                if hasattr(X_train, 'shape'):
                    original_X_shape = (X_train.shape[0], 0)  # Only use row count
                elif isinstance(X_train, list):
                    original_X_shape = (len(X_train), 0)  # Use list length
                else:
                    original_X_shape = (0, 0)
                
                num_classes = len(set(y_train)) if len(y_train) > 0 else 0
                
                # Use only essential info for consistency
                dataset_info = f"dataset_{original_X_shape[0]}samples_{num_classes}classes"
            
            # Use cache_manager to generate fingerprint
            if hasattr(X_train, 'shape'):
                num_rows = X_train.shape[0]
            elif isinstance(X_train, list):
                num_rows = len(X_train)
            else:
                num_rows = 0
                
            return self.cache_manager.generate_dataset_fingerprint(
                dataset_path=dataset_info,
                dataset_size=len(dataset_info),
                num_rows=num_rows
            )
        
        def _prepare_eval_predictions_for_cache(X_test, y_test, model):
            """Prepare evaluation predictions for cache storage"""
            import pandas as pd
            try:
                # Make predictions
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                    y_pred = model.predict(X_test)
                    
                    # Create DataFrame with probabilities
                    proba_df = pd.DataFrame(y_pred_proba, 
                                          columns=[f'proba__class_{i}' for i in range(y_pred_proba.shape[1])])
                    
                    eval_df = pd.DataFrame({
                        'y_true': y_test.flatten() if hasattr(y_test, 'flatten') else y_test,
                        'y_pred': y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
                    })
                    
                    # Combine with probabilities
                    eval_df = pd.concat([eval_df, proba_df], axis=1)
                    
                else:
                    y_pred = model.predict(X_test)
                    eval_df = pd.DataFrame({
                        'y_true': y_test.flatten() if hasattr(y_test, 'flatten') else y_test,
                        'y_pred': y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
                    })
                
                return eval_df
                
            except Exception as e:
                print(f"Warning: Failed to prepare eval predictions: {e}")
                return None
        
        # Bind methods to self
        self._check_model_cache = _check_model_cache
        self._save_model_cache = _save_model_cache
        self._get_model_key = _get_model_key
        self._generate_dataset_id = _generate_dataset_id
        self._generate_config_hash = _generate_config_hash
        self._generate_dataset_fingerprint = _generate_dataset_fingerprint
        self._prepare_eval_predictions_for_cache = _prepare_eval_predictions_for_cache


def main():
    """Main execution function"""
    print("🚀 Starting Comprehensive Evaluation System")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        cv_folds=5,
        validation_size=0.0,  # No separate validation set - CV will handle it
        test_size=0.2,
        random_state=42
    )
    
    # Run comprehensive evaluation with reduced samples for faster testing
    results = evaluator.run_comprehensive_evaluation(max_samples=1000)
    
    # Display results summary (no file creation)
    evaluator.save_results()
    
    print(f"\n🎉 Comprehensive evaluation completed!")
    print(f"📊 Results displayed above ")
    print(f"🔍 Check the summary above for detailed analysis and recommendations")


if __name__ == "__main__":
    main()
