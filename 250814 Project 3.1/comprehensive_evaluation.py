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
import pandas as pd
from typing import Dict, List, Tuple, Any, Union
from scipy import sparse
import time
from datetime import datetime

# Import project modules
from data_loader import DataLoader
from text_encoders import TextVectorizer
from models import NewModelTrainer, validation_manager, ModelMetrics

# Import visualization
from visualization import create_output_directories


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation system for all embedding-model combinations
    """
    
    def __init__(self, 
                 cv_folds: int = 5,
                 validation_size: float = 0.2,
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize the comprehensive evaluator
        
        Args:
            cv_folds: Number of cross-validation folds
            validation_size: Size of validation set (for overfitting detection)
            test_size: Size of test set
            random_state: Random seed for reproducibility
        """
        self.cv_folds = cv_folds
        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize components
        self.data_loader = DataLoader()
        self.text_vectorizer = TextVectorizer()
        
        # Import model components
        from models import model_factory, validation_manager as vm
        
        self.model_trainer = NewModelTrainer(
            cv_folds=cv_folds,
            validation_size=0.0,  # No separate validation set - CV will handle it
            test_size=test_size,
            model_factory=model_factory,
            validation_manager=vm
        )
        
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
        print("   • Note: Using reduced samples (1000) for faster testing")

    def precompute_cv_embeddings(self, texts: List[str], labels: List[str], stop_callback=None) -> Dict[str, Any]:
        """Pre-compute embeddings for all CV folds to ensure fair comparison across models
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            stop_callback: Optional callback to check for stop signal
            
        Returns:
            Dictionary containing pre-computed embeddings for each fold
        """
        print("🔧 Pre-computing CV embeddings for all folds...")
        
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
            
            # Store fold data
            cv_embeddings[f'fold_{fold}'] = {
                'X_train': X_train_emb,
                'X_val': X_val_emb, 
                'y_train': y_train_fold,
                'y_val': y_val_fold,
                'train_idx': train_idx,
                'val_idx': val_idx
            }
        
        if stop_callback and stop_callback():
            return {}
            
        print(f"✅ Pre-computed embeddings for {self.cv_folds} folds")
        self.cv_embeddings_cache = cv_embeddings  # Cache for reuse
        return cv_embeddings
    
    def create_cv_folds_for_sparse_embeddings(self, X_train: Union[np.ndarray, sparse.csr_matrix], 
                                             y_train: np.ndarray, 
                                             embedding_type: str) -> Dict[str, Any]:
        """
        Create CV folds for BoW/TF-IDF (sparse matrices) - REUSES existing logic
        
        Args:
            X_train: Pre-computed sparse matrix (BoW/TF-IDF)
            y_train: Training labels
            embedding_type: 'bow' or 'tfidf'
        
        Returns:
            Dictionary with CV folds data compatible with existing structure
        """
        print(f"🔄 Creating CV folds for {embedding_type.upper()}...")
        
        # Import sklearn for CV splitting
        from sklearn.model_selection import StratifiedKFold
        
        # Create CV splitter
        kf = StratifiedKFold(
            n_splits=self.cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        cv_folds = {}
        
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
            
            # Store fold data - compatible with existing structure
            cv_folds[f'fold_{fold}'] = {
                'X_train': X_train_fold,
                'X_val': X_val_fold,
                'y_train': y_train_fold,
                'y_val': y_val_fold,
                'train_idx': train_idx,
                'val_idx': val_idx
            }
        
        print(f"✅ Created CV folds for {embedding_type.upper()}: {self.cv_folds} folds")
        return cv_folds
    
    def load_and_prepare_data(self, max_samples: int = None, skip_csv_prompt: bool = False, sampling_config: Dict = None) -> Tuple[Dict[str, Any], List[str]]:
        """
        Load and prepare all data formats for evaluation
        
        Args:
            max_samples: Maximum number of samples to use
            skip_csv_prompt: If True, skip CSV backup prompt (for Streamlit usage)
            sampling_config: Sampling configuration from Streamlit (optional)
        
        Returns:
            Tuple of (data_dict, sorted_labels)
        """
        print("\n📊 Loading and Preparing Data...")
        print("=" * 50)
        
        # Load dataset
        self.data_loader.load_dataset(skip_csv_prompt=skip_csv_prompt)
        
        # Select samples - prioritize sampling_config over max_samples
        if skip_csv_prompt:
            print("🚀 Streamlit mode: Using existing data configuration...")
            
        # Use sampling_config if available, otherwise fall back to max_samples
        if sampling_config and sampling_config.get('num_samples'):
            actual_max_samples = sampling_config['num_samples']
            print(f"📊 Using sampling config: {actual_max_samples:,} samples")
        else:
            actual_max_samples = max_samples
            print(f"📊 Using max_samples parameter: {actual_max_samples:,} samples" if actual_max_samples else "📊 No sample limit specified")
        
        self.data_loader.select_samples(actual_max_samples)
        
        self.data_loader.preprocess_samples()
        self.data_loader.create_label_mappings()
        
        # Prepare train/test data (no separate validation set)
        X_train, X_test, y_train, y_test = self.data_loader.prepare_train_test_data()
        sorted_labels = self.data_loader.get_sorted_labels()
        
        # Use train/test split directly (validation handled by CV)
        X_train_full, y_train_full = X_train, y_train
        X_val, y_val = np.array([]), np.array([])  # Empty validation set
        
        # Verify split consistency
        print(f"🔍 Split verification:")
        print(f"   • Total: {len(X_train_full) + len(X_test)}")
        print(f"   • Train: {len(X_train_full)} | Test: {len(X_test)}")
        
        print(f"✅ Data prepared:")
        print(f"   • Training: {len(X_train_full)} samples (for CV)")
        print(f"   • Validation: Handled by CV folds")
        print(f"   • Test: {len(X_test)} samples")
        print(f"   • Labels: {len(sorted_labels)} classes")
        
        # Store data
        data_dict = {
            'X_train': X_train_full,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train_full,
            'y_val': y_val,
            'y_test': y_test,
            'labels': sorted_labels
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
            
            # FIXED: Fit embedding model on TRAINING DATA ONLY to prevent data leakage
            print(f"🔧 Fitting embedding model on {len(X_train):,} training samples...")
            # Import global stop check if available
            try:
                from training_pipeline import global_stop_check
                actual_stop_callback = global_stop_check
            except ImportError:
                actual_stop_callback = stop_callback
                
            X_train_emb = self.text_vectorizer.fit_transform_embeddings(X_train, stop_callback=actual_stop_callback)
            
            # Check if stopped during training embeddings
            if actual_stop_callback and actual_stop_callback():
                print("🛑 Embedding creation stopped by user request")
                return {}
            
            # Transform test data using fitted model (no data leakage)
            print(f"🔧 Transforming {len(X_test):,} test samples using fitted model...")
            X_test_emb = self.text_vectorizer.transform_embeddings(X_test, stop_callback=actual_stop_callback)
            
            # Check if stopped during test embeddings
            if actual_stop_callback and actual_stop_callback():
                print("🛑 Embedding creation stopped by user request")
                return {}
            
            # Transform validation data if exists
            X_val_emb = None
            if len(X_val) > 0:
                print(f"🔧 Transforming {len(X_val):,} validation samples...")
                X_val_emb = self.text_vectorizer.transform_embeddings(X_val, stop_callback=actual_stop_callback)
                
                # Check if stopped during validation embeddings
                if actual_stop_callback and actual_stop_callback():
                    print("🛑 Embedding creation stopped by user request")
                    return {}
            
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
                                  X_train: Union[np.ndarray, sparse.csr_matrix],
                                  X_val: Union[np.ndarray, sparse.csr_matrix],
                                  X_test: Union[np.ndarray, sparse.csr_matrix],
                                  y_train: np.ndarray,
                                  y_val: np.ndarray,
                                  y_test: np.ndarray,
                                  step3_data: Dict = None) -> Dict[str, Any]:
        """
        Evaluate a single model-embedding combination
        
        Returns:
            Dictionary with evaluation results
        """
        combination_key = f"{model_name}_{embedding_name}"
        print(f"   🔍 Evaluating {combination_key}...")
        
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
            
            # Training
            start_time = time.time()
            y_test_pred, y_val_pred, y_test, val_acc, test_acc, test_metrics = \
                self.model_trainer.train_validate_test_model(
                    model_name, X_train, y_train, 
                    X_val, y_val, X_test, y_test, step3_data
                )
            training_time = time.time() - start_time
            
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
                        temp_model.fit(X_train, y_train, use_gpu=False)
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
                    if overfitting_score > 0.1:
                        overfitting_level = "High overfitting - model memorizing training data"
                    elif overfitting_score > 0.05:
                        overfitting_level = "Moderate overfitting - model overfitting to training data"
                    elif overfitting_score > -0.05:
                        overfitting_level = "Good fit - model generalizing well"
                    elif overfitting_score > -0.1:
                        overfitting_level = "Slight underfitting - model could learn more"
                    else:
                        overfitting_level = "Underfitting - model not learning enough"
                    
                    print(f"     📊 ML Standard Overfitting: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}, Score={overfitting_score:+.3f}")
                    print(f"     📊 Overfitting Level: {overfitting_level}")
                    
                else:
                    # No validation set: Use Training vs Test accuracy for overfitting detection
                    # This is still ML standard but using test set as validation proxy
                    overfitting_score = train_acc - test_acc  # Training Acc - Test Acc
                    overfitting_status = self._classify_overfitting(overfitting_score)
                    
                    # Classify overfitting level using training vs test
                    if overfitting_score > 0.1:
                        overfitting_level = "High overfitting - model memorizing training data"
                    elif overfitting_score > 0.05:
                        overfitting_level = "Moderate overfitting - model overfitting to training data"
                    elif overfitting_score > -0.05:
                        overfitting_level = "Good fit - model generalizing well"
                    elif overfitting_score > -0.1:
                        overfitting_level = "Slight underfitting - model could learn more"
                    else:
                        overfitting_level = "Underfitting - model not learning enough"
                    
                    print(f"     📊 ML Standard Overfitting (Train vs Test): Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}, Score={overfitting_score:+.3f}")
                    print(f"     📊 Overfitting Level: {overfitting_level}")
                    print(f"     ⚠️ Note: Using test set as validation proxy (no separate validation set)")
                    
            except Exception as e:
                # Fallback to F1-based overfitting if ML standard fails
                if val_metrics and 'f1_score' in val_metrics and test_metrics and 'f1_score' in test_metrics:
                    val_f1 = val_metrics['f1_score']
                    test_f1 = test_metrics['f1_score']
                    overfitting_score = val_f1 - test_f1
                    overfitting_status = self._classify_overfitting(overfitting_score)
                    print(f"     📊 F1-based Overfitting (fallback): Val F1={val_f1:.3f}, Test F1={test_f1:.3f}, Score={overfitting_score:+.3f}")
                else:
                    overfitting_score = val_acc - test_acc
                    overfitting_status = self._classify_overfitting(overfitting_score)
                    print(f"     📊 Accuracy-based Overfitting (fallback): Val Acc={val_acc:.3f}, Test Acc={test_acc:.3f}, Score={overfitting_score:+.3f}")
                print(f"     ⚠️ ML Standard overfitting failed, using fallback: {e}")
            
            # Cross-validation - ENHANCED: Use optimized CV for sparse embeddings (BoW/TF-IDF) or fitted embeddings
            if embedding_name in ['bow', 'tfidf']:
                # For BoW and TF-IDF, use optimized CV with sparse matrix handling
                print(f"     🔧 CV using optimized sparse {embedding_name} data for {model_name} (no data leakage)")
                
                # Create CV folds specifically for sparse embeddings
                cv_folds = self.create_cv_folds_for_sparse_embeddings(X_train, y_train, embedding_name)
                
                # Use CV folds for evaluation
                cv_results = self.model_trainer.cross_validate_with_precomputed_embeddings(
                    model_name, cv_folds, ['accuracy', 'precision', 'recall', 'f1']
                )
            else:
                # For embeddings, use pre-computed CV embeddings for fair comparison
                print(f"     🔧 CV using pre-computed {embedding_name} embeddings for {model_name} (fair comparison)")
                
                print(f"     🔍 Debug: cv_embeddings_cache exists = {hasattr(self, 'cv_embeddings_cache')}")
                print(f"     🔍 Debug: cv_embeddings_cache content = {bool(self.cv_embeddings_cache)}")
                print(f"     🔍 Debug: cv_embeddings_cache keys = {list(self.cv_embeddings_cache.keys()) if hasattr(self, 'cv_embeddings_cache') else 'N/A'}")
                
                if hasattr(self, 'cv_embeddings_cache') and self.cv_embeddings_cache:
                    # Use cached pre-computed embeddings
                    print(f"     ✅ Using pre-computed CV embeddings for {model_name}")
                    cv_results = self.model_trainer.cross_validate_with_precomputed_embeddings(
                        model_name, self.cv_embeddings_cache, ['accuracy', 'precision', 'recall', 'f1']
                    )
                else:
                    # Fallback to old method if cache not available
                    print(f"     ⚠️  Fallback: CV embeddings cache not found, using standard CV")
                    cv_results = self.model_trainer.cross_validate_model(
                        model_name, X_train, y_train, ['accuracy', 'precision', 'recall', 'f1']
                    )
            
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
                    # No validation set: Use training vs test accuracy for CV calculation
                    cv_f1_based_accuracy = (train_acc + test_acc) / 2.0  # ML standard CV accuracy (train vs test)
                    cv_f1_based_std = abs(train_acc - test_acc) / 2.0   # ML standard variation (train vs test)
                    print(f"     📊 ML Standard CV (Train vs Test): Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}, CV={cv_f1_based_accuracy:.3f}±{cv_f1_based_std:.3f}")
                
            except Exception as e:
                # Fallback to F1-based calculation if ML standard fails
                if val_metrics and 'f1_score' in val_metrics and test_metrics and 'f1_score' in test_metrics:
                    val_f1 = val_metrics['f1_score']
                    test_f1 = test_metrics['f1_score']
                    cv_f1_based_accuracy = (val_f1 + test_f1) / 2.0
                    cv_f1_based_std = abs(val_f1 - test_f1) / 2.0
                else:
                    # Safe fallback if all methods fail
                    cv_f1_based_accuracy = test_acc
                    cv_f1_based_std = 0.0
                print(f"     ⚠️ ML Standard CV failed, using fallback: {e}")
            
            # Store results
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
                
                # Overfitting analysis
                'overfitting_score': overfitting_score,
                'overfitting_status': overfitting_status,
                'overfitting_classification': self._get_overfitting_classification(overfitting_score),
                
                # Cross-validation results - Traditional CV accuracy from folds
                'cv_mean_accuracy': cv_results.get('overall_results', {}).get('accuracy_mean', val_acc),  # Traditional CV accuracy (from folds)
                'cv_std_accuracy': cv_results.get('overall_results', {}).get('accuracy_std', 0.0),       # Traditional CV accuracy std (from folds)
                'cv_mean_f1': cv_results.get('overall_results', {}).get('f1_mean', 0.0),                # Traditional CV F1 (from folds)
                'cv_std_f1': cv_results.get('overall_results', {}).get('f1_std', 0.0),                  # Traditional CV F1 std (from folds)
                
                # ML Standard CV Score - NEW: Based on training vs validation accuracy for overfitting detection
                'ml_cv_accuracy': cv_f1_based_accuracy,   # ← ML standard CV accuracy (avg of train_acc + val_acc)
                'ml_cv_variation': cv_f1_based_std,       # ← ML standard variation (variation between train_acc and val_acc)
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
            
            print(f"     ✅ {combination_key}: Val={val_acc:.3f}, Test={test_acc:.3f}, ML-CV={cv_f1_based_accuracy:.3f}±{cv_f1_based_std:.3f}")
            
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
                'overfitting_score': 0.0,
                'overfitting_status': 'error'
            }
            print(f"     ❌ {combination_key}: Error - {e}")
            return error_result
    
    def _get_label_mapping(self, y_train: np.ndarray) -> Dict[int, str]:
        """
        Tạo mapping từ numeric labels sang text labels theo pipeline
        Sử dụng cùng logic như main.py để đảm bảo consistency
        """
        try:
            # Lấy unique labels đã được sắp xếp
            unique_labels = sorted(list(set(y_train)))
            
            # Tạo mapping theo logic của main.py
            label_mapping = {}
            for i, label_id in enumerate(unique_labels):
                if label_id == 0:
                    label_mapping[label_id] = "astro-ph"
                elif label_id == 1:
                    label_mapping[label_id] = "cond-mat"
                elif label_id == 2:
                    label_mapping[label_id] = "cs"
                elif label_id == 3:
                    label_mapping[label_id] = "math"
                elif label_id == 4:
                    label_mapping[label_id] = "physics"
                else:
                    label_mapping[label_id] = f"Class_{label_id}"
            
            return label_mapping
            
        except Exception as e:
            print(f"Warning: Could not create label mapping: {e}")
            # Fallback: tạo mapping đơn giản
            return {label_id: f"Class_{label_id}" for label_id in set(y_train)}
    
    def _classify_overfitting(self, overfitting_score: float) -> str:
        """Classify the level of overfitting"""
        if overfitting_score < -0.05:
            return "underfitting"
        elif overfitting_score > 0.05:
            return "overfitting"
        else:
            return "well_fitted"
    
    def _get_overfitting_classification(self, overfitting_score: float) -> str:
        """Get detailed overfitting classification"""
        if overfitting_score < -0.1:
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
    
    def _calculate_stability_score(self, cv_results: Dict[str, Any]) -> float:
        """Calculate model stability score from CV results"""
        try:
            accuracies = [fold['accuracy'] for fold in cv_results['fold_results']]
            return 1.0 - (np.std(accuracies) / np.mean(accuracies))
        except:
            return 0.0
    
    def run_comprehensive_evaluation(self, max_samples: int = None, skip_csv_prompt: bool = False, 
                                   sampling_config: Dict = None, selected_models: List[str] = None, selected_embeddings: List[str] = None, stop_callback=None, step3_data: Dict = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of model-embedding combinations
        
        Args:
            max_samples: Maximum number of samples to use
            skip_csv_prompt: If True, skip CSV backup prompt (for Streamlit usage)
            sampling_config: Sampling configuration from Streamlit (optional)
            selected_models: List of model names to evaluate (if None, evaluate all)
            selected_embeddings: List of embedding names to evaluate (if None, evaluate all)
        
        Returns:
            Complete evaluation results
        """
        print("\n🚀 Starting Comprehensive Evaluation...")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. Load and prepare data
        data_dict, sorted_labels = self.load_and_prepare_data(max_samples, skip_csv_prompt, sampling_config)
        
        # 2. Create selected embeddings (only once)
        if not hasattr(self, 'embeddings') or self.embeddings is None:
            print(f"\n🔤 Creating Embeddings for Data...")
            
            # Check if stopped before creating embeddings
            try:
                from training_pipeline import global_stop_check
                if global_stop_check():
                    print("🛑 Training stopped before embedding creation")
                    return {'status': 'stopped', 'message': 'Training stopped before embedding creation'}
            except ImportError:
                if stop_callback and stop_callback():
                    print("🛑 Training stopped before embedding creation")
                    return {'status': 'stopped', 'message': 'Training stopped before embedding creation'}
            
            embeddings = self.create_all_embeddings(
                data_dict['X_train'], 
                data_dict['X_val'], 
                data_dict['X_test'],
                selected_embeddings,
                stop_callback
            )
            # Store embeddings in the evaluator for reuse
            self.embeddings = embeddings
        else:
            print(f"\n🔄 Reusing cached embeddings...")
            embeddings = self.embeddings
        
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
            
            # Pre-compute CV embeddings
            cv_embeddings = self.precompute_cv_embeddings(all_train_texts, all_train_labels, stop_callback)
            
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
                'Linear SVC': 'linear_svc'
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
                    
                    print(f"🚀 Training {model_name} with {embedding_name}...")
                    
                    result = self.evaluate_single_combination(
                        model_name=model_name,
                        embedding_name=embedding_name,
                        X_train=embedding_data['X_train'],
                        X_val=embedding_data['X_val'],
                        X_test=embedding_data['X_test'],
                        y_train=data_dict['y_train'],
                        y_val=data_dict['y_val'],
                        y_test=data_dict['y_test'],
                        step3_data=step3_data
                    )
                    
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
        
        # 5. Analyze results
        print(f"\n📊 Evaluation Complete!")
        print(f"   • Successful combinations: {successful_combinations}/{total_combinations}")
        print(f"   • Failed combinations: {total_combinations - successful_combinations}")
        
        # 6. Find best combinations
        self._analyze_results(all_results)
        
        # 7. Generate comprehensive report
        total_time = time.time() - start_time
        print(f"   • Total evaluation time: {total_time:.2f}s")
        
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
        
        if best_overall['validation_accuracy'] > 0:
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
        stable_models = [r for r in successful_results if r['cv_stability_score'] > 0.8]
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
            
            if best['validation_accuracy'] > 0:
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
                    val_acc = f"{result['validation_accuracy']:.3f}" if result['validation_accuracy'] > 0 else "CV"
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
                    val_acc = f"{result['validation_accuracy']:.3f}" if result['validation_accuracy'] > 0 else "CV"
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
        print(f"   • {self.evaluation_results['successful_combinations']}/{self.evaluation_results['total_combinations']} combinations successful")
        print(f"   • Time: {self.evaluation_results['evaluation_time']:.2f}s")
        print(f"   • No files created - results displayed above")
        
        return "displayed"


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
