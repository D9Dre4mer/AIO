"""
Training Pipeline for Streamlit Wizard UI
Executes comprehensive training evaluation like main.py
Integrates with existing project modules and comprehensive_evaluation.py
"""

import warnings
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Any

# Suppress warnings
warnings.filterwarnings("ignore")

# Import project modules
try:
    from data_loader import DataLoader
    from text_encoders import TextVectorizer
    from models import NewModelTrainer, validation_manager, model_factory
    from visualization import (
        plot_confusion_matrix,
        create_output_directories,
        plot_model_comparison,
        print_model_results
    )
    from comprehensive_evaluation import ComprehensiveEvaluator
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    # Create dummy classes for fallback
    class DataLoader:
        pass
    class TextVectorizer:
        pass
    class NewModelTrainer:
        pass
    class validation_manager:
        pass
    class model_factory:
        pass
    class ComprehensiveEvaluator:
        pass


class StreamlitTrainingPipeline:
    """Training pipeline specifically designed for Streamlit Wizard UI"""
    
    def __init__(self):
        """Initialize the training pipeline"""
        self.results = {}
        self.training_status = "idle"
        self.current_model = None
        self.current_phase = "initializing"
        self.models_completed = 0
        self.total_models = 0
        self.start_time = None
        self.elapsed_time = 0
        
    def initialize_pipeline(self, df: pd.DataFrame, step1_data: Dict, 
                          step2_data: Dict, step3_data: Dict) -> Dict:
        """Initialize the training pipeline with configuration from previous steps"""
        
        try:
            self.current_phase = "initializing"
            
            # Extract configuration from previous steps
            sampling_config = step1_data.get('sampling_config', {})
            text_column = step2_data.get('text_column')
            label_column = step2_data.get('label_column')
            preprocessing_config = {
                'text_cleaning': step2_data.get('text_cleaning', True),
                'category_mapping': step2_data.get('category_mapping', True),
                'data_validation': step2_data.get('data_validation', True),
                'memory_optimization': step2_data.get('memory_optimization', True)
            }
            
            model_config = step3_data.get('data_split', {})
            selected_models = step3_data.get('selected_models', [])
            selected_vectorization = step3_data.get('selected_vectorization', [])
            cv_config = step3_data.get('cross_validation', {})
            
            # Calculate total models to train
            self.total_models = len(selected_models) * len(selected_vectorization)
            
            # Create output directories
            try:
                create_output_directories()
            except:
                pass  # Directory might already exist
            
            return {
                'status': 'success',
                'message': 'Pipeline initialized successfully',
                'total_models': self.total_models,
                'config': {
                    'sampling': sampling_config,
                    'preprocessing': preprocessing_config,
                    'model': model_config,
                    'vectorization': selected_vectorization,
                    'cv': cv_config
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to initialize pipeline: {str(e)}',
                'error': str(e)
            }
    
    def execute_training(self, df: pd.DataFrame, step1_data: Dict,
                         step2_data: Dict, step3_data: Dict,
                         progress_callback=None) -> Dict:
        """Execute comprehensive training evaluation like main.py"""

        try:
            self.start_time = time.time()
            self.training_status = "training"
            self.models_completed = 0

            # Initialize pipeline
            self.current_phase = "initialization"
            if progress_callback:
                progress_callback(self.current_phase, "Initializing comprehensive evaluation...", 0.05)

            # Extract configuration
            text_column = step2_data.get('text_column')
            label_column = step2_data.get('label_column')
            cv_config = step3_data.get('cross_validation', {})
            cv_folds = cv_config.get('cv_folds', 5)
            data_split = step3_data.get('data_split', {})
            
            # Calculate validation and test sizes from step 3 configuration
            test_size = data_split.get('test', 20) / 100.0
            validation_size = data_split.get('validation', 10) / 100.0

            # Prepare data for comprehensive evaluation
            self.current_phase = "data_preparation"
            if progress_callback:
                progress_callback(self.current_phase, "Preparing data for evaluation...", 0.1)

            # Apply sampling if configured
            if step1_data.get('sampling_config'):
                df = self._apply_sampling(df, step1_data['sampling_config'])

            # Apply preprocessing
            df = self._apply_preprocessing(df, step2_data)

            # Save processed data to temporary CSV for DataLoader
            import tempfile
            import os
            
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            df.to_csv(temp_file.name, index=False)
            temp_file.close()

            # Initialize Comprehensive Evaluator
            self.current_phase = "evaluator_setup"
            if progress_callback:
                progress_callback(self.current_phase, "Setting up comprehensive evaluator...", 0.2)

            try:
                # Create evaluator with proper model factory and validation manager
                evaluator = ComprehensiveEvaluator(
                    cv_folds=cv_folds,
                    validation_size=validation_size,
                    test_size=test_size,
                    random_state=cv_config.get('random_state', 42)
                )
                
                # Ensure evaluator has access to model factory and validation manager
                if hasattr(evaluator, 'model_trainer') and evaluator.model_trainer:
                    evaluator.model_trainer.model_factory = model_factory
                    evaluator.model_trainer.validation_manager = validation_manager

                # Temporarily override DataLoader's file path
                original_file_path = getattr(evaluator.data_loader, 'file_path', None)
                evaluator.data_loader.file_path = temp_file.name
                
                # Set text and label columns
                evaluator.data_loader.text_column = text_column
                evaluator.data_loader.label_column = label_column

                # Run comprehensive evaluation
                self.current_phase = "comprehensive_evaluation"
                if progress_callback:
                    progress_callback(self.current_phase, "Running comprehensive evaluation...", 0.3)

                # Get max samples from step 1 configuration
                max_samples = step1_data.get('sampling_config', {}).get('num_samples', None)
                
                # Run comprehensive evaluation with skip_csv_prompt=True for Streamlit usage
                evaluation_results = evaluator.run_comprehensive_evaluation(
                    max_samples=max_samples, 
                    skip_csv_prompt=True
                )
                
                # Update progress based on evaluation progress
                self.current_phase = "analysis"
                if progress_callback:
                    progress_callback(self.current_phase, "Analyzing results...", 0.8)

                # Get comprehensive results
                comprehensive_results = evaluation_results.get('all_results', [])
                successful_results = [r for r in comprehensive_results if r['status'] == 'success']
                
                self.models_completed = len(successful_results)
                self.total_models = evaluation_results.get('total_combinations', 0)

                # Generate summary report
                self.current_phase = "report_generation"
                if progress_callback:
                    progress_callback(self.current_phase, "Generating comprehensive report...", 0.9)

                # Display results (this will print the comprehensive report)
                evaluator.save_results()

                # Clean up temp file
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

                # Restore original file path
                if original_file_path:
                    evaluator.data_loader.file_path = original_file_path

                # Finalize results
                self.current_phase = "completed"
                self.training_status = "completed"
                if progress_callback:
                    progress_callback(self.current_phase, "Comprehensive evaluation completed!", 1.0)

                return {
                    'status': 'success',
                    'message': 'Comprehensive evaluation completed successfully',
                    'results': evaluation_results,
                    'comprehensive_results': comprehensive_results,
                    'successful_combinations': evaluation_results.get('successful_combinations', 0),
                    'total_combinations': evaluation_results.get('total_combinations', 0),
                    'best_combinations': evaluator.best_combinations if hasattr(evaluator, 'best_combinations') else {},
                    'total_models': self.total_models,
                    'models_completed': self.models_completed,
                    'elapsed_time': time.time() - self.start_time,
                    'evaluation_time': evaluation_results.get('evaluation_time', 0),
                    'data_info': evaluation_results.get('data_info', {}),
                    'embedding_info': evaluation_results.get('embedding_info', {})
                }

            except Exception as e:
                # Clean up temp file on error
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
                raise e

        except Exception as e:
            self.training_status = "error"
            return {
                'status': 'error',
                'message': f'Comprehensive evaluation failed: {str(e)}',
                'error': str(e)
            }
    
    def _apply_sampling(self, df: pd.DataFrame, sampling_config: Dict) -> pd.DataFrame:
        """Apply sampling configuration to dataset"""
        try:
            num_samples = sampling_config.get('num_samples', len(df))
            strategy = sampling_config.get('sampling_strategy', 'Random')
            
            if num_samples >= len(df):
                return df
            
            if 'Stratified' in strategy and 'label_column' in sampling_config:
                # Stratified sampling
                from sklearn.model_selection import train_test_split
                df_sample, _ = train_test_split(
                    df, 
                    train_size=num_samples, 
                    stratify=df[sampling_config['label_column']],
                    random_state=42
                )
                return df_sample
            else:
                # Random sampling
                return df.sample(n=num_samples, random_state=42)
                
        except Exception as e:
            print(f"Warning: Sampling failed, using full dataset: {e}")
            return df
    
    def _apply_preprocessing(self, df: pd.DataFrame, step2_data: Dict) -> pd.DataFrame:
        """Apply preprocessing options to dataset"""
        try:
            text_column = step2_data.get('text_column')
            label_column = step2_data.get('label_column')
            
            # Text cleaning
            if step2_data.get('text_cleaning', True):
                df[text_column] = df[text_column].astype(str).str.replace(
                    r'[^\w\s]', '', regex=True
                ).str.strip()
            
            # Data validation (remove nulls)
            if step2_data.get('data_validation', True):
                df = df.dropna(subset=[text_column, label_column])
            
            # Category mapping
            if step2_data.get('category_mapping', True):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[label_column] = le.fit_transform(df[label_column])
            
            # Memory optimization
            if step2_data.get('memory_optimization', True):
                df[text_column] = df[text_column].astype('category')
            
            return df
            
        except Exception as e:
            print(f"Warning: Preprocessing failed: {e}")
            return df
    
    def _create_data_splits(self, X: np.ndarray, y: np.ndarray, 
                           data_split: Dict) -> Tuple:
        """Create train/validation/test splits"""
        try:
            # Use validation manager if available
            if hasattr(validation_manager, 'split_data'):
                return validation_manager.split_data(X, y)
            else:
                # Fallback to sklearn
                from sklearn.model_selection import train_test_split
                
                # First split: separate test set
                test_size = data_split.get('test', 0.2)
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Second split: separate validation set
                val_size = data_split.get('validation', 0.1)
                if val_size > 0:
                    val_ratio = val_size / (1 - test_size)
                    X_train_full, X_val, y_train_full, y_val = train_test_split(
                        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
                    )
                else:
                    X_train_full, y_train_full = X_temp, y_temp
                    X_val, y_val = np.array([]), np.array([])
                
                return X_train_full, X_val, X_test, y_train_full, y_val, y_test
                
        except Exception as e:
            print(f"Warning: Data splitting failed, using simple split: {e}")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, np.array([]), X_test, y_train, np.array([]), y_test
    
    def _prepare_vectorized_data(self, X_train: np.ndarray, X_val: np.ndarray, 
                                X_test: np.ndarray, y_train: np.ndarray, 
                                y_val: np.ndarray, y_test: np.ndarray,
                                vectorization_methods: List[str]) -> Dict:
        """Prepare vectorized data for all selected methods"""
        
        vectorized_data = {}
        
        try:
            # Initialize text vectorizer
            if hasattr(TextVectorizer, '__init__'):
                text_vectorizer = TextVectorizer()
            else:
                # Fallback vectorizer
                text_vectorizer = None
            
            for method in vectorization_methods:
                if method == 'Bag of Words (BoW)':
                    vectorized_data['bow'] = self._vectorize_bow(
                        text_vectorizer, X_train, X_val, X_test
                    )
                elif method == 'TF-IDF':
                    vectorized_data['tfidf'] = self._vectorize_tfidf(
                        text_vectorizer, X_train, X_val, X_test
                    )
                elif method == 'Word Embeddings':
                    vectorized_data['embeddings'] = self._vectorize_embeddings(
                        text_vectorizer, X_train, X_val, X_test
                    )
            
            # Add labels separately
            vectorized_data['labels'] = {
                'train': y_train,
                'val': y_val,
                'test': y_test
            }
            
        except Exception as e:
            print(f"Warning: Vectorization failed: {e}")
            # Create simple fallback vectorization
            vectorized_data = self._create_fallback_vectorization(
                X_train, X_val, X_test, y_train, y_val, y_test
            )
        
        return vectorized_data
    
    def _create_fallback_vectorization(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Create fallback vectorization when main methods fail"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Simple TF-IDF fallback
            tfidf = TfidfVectorizer(max_features=1000)
            X_train_vec = tfidf.fit_transform(X_train)
            X_val_vec = tfidf.transform(X_val) if len(X_val) > 0 else None
            X_test_vec = tfidf.transform(X_test)
            
            return {
                'bow': {
                    'train': X_train_vec,
                    'val': X_val_vec,
                    'test': X_test_vec
                },
                'tfidf': {
                    'train': X_train_vec,
                    'val': X_val_vec,
                    'test': X_test_vec
                },
                'embeddings': {
                    'train': X_train_vec,
                    'val': X_val_vec,
                    'test': X_test_vec
                },
                'labels': {
                    'train': y_train,
                    'val': y_val,
                    'test': y_test
                }
            }
        except Exception as e:
            print(f"Warning: Fallback vectorization failed: {e}")
            return {
                'labels': {
                    'train': y_train,
                    'val': y_val,
                    'test': y_test
                }
            }
    
    def _vectorize_bow(self, vectorizer, X_train, X_val, X_test):
        """Vectorize data using Bag of Words"""
        try:
            if hasattr(vectorizer, 'fit_transform_bow'):
                X_train_bow = vectorizer.fit_transform_bow(X_train)
                X_val_bow = vectorizer.transform_bow(X_val) if len(X_val) > 0 else None
                X_test_bow = vectorizer.transform_bow(X_test)
            else:
                # Fallback to sklearn
                from sklearn.feature_extraction.text import CountVectorizer
                cv = CountVectorizer(max_features=1000)
                X_train_bow = cv.fit_transform(X_train)
                X_val_bow = cv.transform(X_val) if len(X_val) > 0 else None
                X_test_bow = cv.transform(X_test)
            
            return {
                'train': X_train_bow,
                'val': X_val_bow,
                'test': X_test_bow
            }
        except Exception as e:
            print(f"Warning: BoW vectorization failed: {e}")
            return None
    
    def _vectorize_tfidf(self, vectorizer, X_train, X_val, X_test):
        """Vectorize data using TF-IDF"""
        try:
            if hasattr(vectorizer, 'fit_transform_tfidf'):
                X_train_tfidf = vectorizer.fit_transform_tfidf(X_train)
                X_val_tfidf = vectorizer.transform_tfidf(X_val) if len(X_val) > 0 else None
                X_test_tfidf = vectorizer.transform_tfidf(X_test)
            else:
                # Fallback to sklearn
                from sklearn.feature_extraction.text import TfidfVectorizer
                tfidf = TfidfVectorizer(max_features=1000)
                X_train_tfidf = tfidf.fit_transform(X_train)
                X_val_tfidf = tfidf.transform(X_val) if len(X_val) > 0 else None
                X_test_tfidf = tfidf.transform(X_test)
            
            return {
                'train': X_train_tfidf,
                'val': X_val_tfidf,
                'test': X_test_tfidf
            }
        except Exception as e:
            print(f"Warning: TF-IDF vectorization failed: {e}")
            return None
    
    def _vectorize_embeddings(self, vectorizer, X_train, X_val, X_test):
        """Vectorize data using Word Embeddings"""
        try:
            if hasattr(vectorizer, 'transform_embeddings'):
                X_train_emb = vectorizer.transform_embeddings(X_train)
                X_val_emb = vectorizer.transform_embeddings(X_val) if len(X_val) > 0 else None
                X_test_emb = vectorizer.transform_embeddings(X_test)
            else:
                # Fallback to simple embeddings
                X_train_emb = self._create_simple_embeddings(X_train)
                X_val_emb = self._create_simple_embeddings(X_val) if len(X_val) > 0 else None
                X_test_emb = self._create_simple_embeddings(X_test)
            
            return {
                'train': X_train_emb,
                'val': X_val_emb,
                'test': X_test_emb
            }
        except Exception as e:
            print(f"Warning: Embeddings vectorization failed: {e}")
            return None
    
    def _create_simple_embeddings(self, X):
        """Create simple embeddings as fallback"""
        try:
            # Simple character-level encoding
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(max_features=100, analyzer='char')
            return tfidf.fit_transform(X)
        except:
            # Last resort: random embeddings
            return np.random.rand(len(X), 100)
    
    def _train_all_models(self, vectorized_data: Dict, selected_models: List[str], 
                          cv_config: Dict, progress_callback=None) -> Dict:
        """Train all selected models with all vectorization methods"""
        
        results = {}
        cv_folds = cv_config.get('cv_folds', 5)
        random_state = cv_config.get('random_state', 42)
        
        # Get labels from vectorized_data
        labels_dict = vectorized_data.get('labels', {})
        if not labels_dict:
            print("Warning: No labels found in vectorized data")
            return results
        
        for model_name in selected_models:
            for vec_method, vec_data in vectorized_data.items():
                if vec_method == 'labels' or vec_data is None:
                    continue
                
                if progress_callback:
                    progress = 0.4 + (0.4 * (self.models_completed / self.total_models))
                    progress_callback(
                        "model_training", 
                        f"Training {model_name} with {vec_method}...", 
                        progress
                    )
                
                try:
                    # Train model with labels
                    model_result = self._train_single_model(
                        model_name, vec_method, vec_data, labels_dict, cv_folds, random_state
                    )
                    
                    if model_result:
                        key = f"{model_name}_{vec_method}"
                        results[key] = model_result
                        self.models_completed += 1
                        
                        if progress_callback:
                            progress = 0.4 + (0.4 * (self.models_completed / self.total_models))
                            progress_callback(
                                "model_training", 
                                f"Completed {model_name} with {vec_method}", 
                                progress
                            )
                
                except Exception as e:
                    print(f"Warning: Failed to train {model_name} with {vec_method}: {e}")
                    continue
        
        return results
    
    def _train_single_model(self, model_name: str, vec_method: str, 
                           vec_data: Dict, labels_dict: Dict, cv_folds: int, random_state: int) -> Dict:
        """Train a single model with specific vectorization method"""
        
        try:
            # Check if vec_data has the required structure
            if not isinstance(vec_data, dict) or 'train' not in vec_data:
                print(f"Warning: Invalid vectorization data structure for {model_name} with {vec_method}")
                return None
            
            # Check if labels_dict has the required structure
            if not isinstance(labels_dict, dict) or 'train' not in labels_dict:
                print(f"Warning: Invalid labels structure for {model_name} with {vec_method}")
                return None
            
            # Use NewModelTrainer if available and properly configured
            if hasattr(NewModelTrainer, '__init__'):
                try:
                    # Try to create NewModelTrainer with proper arguments
                    if hasattr(NewModelTrainer, 'train_validate_test_model'):
                        # Check if we can create an instance without arguments first
                        try:
                            model_trainer = NewModelTrainer()
                        except TypeError as e:
                            # If constructor requires arguments, try with defaults
                            try:
                                model_trainer = NewModelTrainer(
                                    cv_folds=cv_folds,
                                    validation_size=0.2,
                                    test_size=0.2
                                )
                            except Exception as e2:
                                print(f"Warning: NewModelTrainer constructor failed: {e2}")
                                # Skip to sklearn fallback
                                return self._train_sklearn_fallback(
                                    model_name, vec_method, vec_data, labels_dict, cv_folds, random_state
                                )
                        
                        # Map model names to trainer method names
                        model_mapping = {
                            'K-Nearest Neighbors': 'knn',
                            'Decision Tree': 'decision_tree',
                            'Naive Bayes': 'naive_bayes',
                            'K-Means Clustering': 'kmeans',
                            'Support Vector Machine (SVM)': 'svm'
                        }
                        
                        trainer_method = model_mapping.get(model_name, model_name.lower())
                        
                        try:
                            labels, _, _, _, accuracy, report = model_trainer.train_validate_test_model(
                                trainer_method,
                                vec_data['train'], labels_dict['train'],
                                vec_data['val'], labels_dict['val'],
                                vec_data['test'], labels_dict['test']
                            )
                            
                            return {
                                'labels': labels,
                                'accuracy': accuracy,
                                'report': report,
                                'vectorization': vec_method
                            }
                        except Exception as e3:
                            print(f"Warning: NewModelTrainer training failed for {model_name}: {e3}")
                            # Fall through to sklearn fallback
                            
                except Exception as e:
                    print(f"Warning: NewModelTrainer failed for {model_name}: {e}")
                    # Fall through to sklearn fallback
            
            # Fallback to sklearn models
            return self._train_sklearn_fallback(
                model_name, vec_method, vec_data, labels_dict, cv_folds, random_state
            )
            
        except Exception as e:
            print(f"Warning: Model training failed for {model_name}: {e}")
            return None
    
    def _train_sklearn_fallback(self, model_name: str, vec_method: str, 
                               vec_data: Dict, labels_dict: Dict, cv_folds: int, random_state: int) -> Dict:
        """Fallback training using sklearn models"""
        
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import accuracy_score, classification_report
            
            # Create model
            model = self._create_sklearn_model(model_name, random_state)
            
            # Train model
            model.fit(vec_data['train'], labels_dict['train'])
            
            # Make predictions
            y_pred = model.predict(vec_data['test'])
            
            # Calculate accuracy
            accuracy = accuracy_score(labels_dict['test'], y_pred)
            
            # Generate report
            report = classification_report(
                labels_dict['test'], y_pred, output_dict=True
            )
            
            return {
                'labels': y_pred,
                'accuracy': accuracy,
                'report': report,
                'vectorization': vec_method,
                'model': model
            }
            
        except Exception as e:
            print(f"Warning: Sklearn fallback failed: {e}")
            return None
    
    def _create_sklearn_model(self, model_name: str, random_state: int):
        """Create sklearn model instance"""
        
        if 'K-Nearest Neighbors' in model_name:
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(n_neighbors=5)
        
        elif 'Decision Tree' in model_name:
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(random_state=random_state)
        
        elif 'Naive Bayes' in model_name:
            from sklearn.naive_bayes import MultinomialNB
            return MultinomialNB()
        
        elif 'K-Means' in model_name:
            from sklearn.cluster import KMeans
            return KMeans(n_clusters=5, random_state=random_state)
        
        elif 'SVM' in model_name:
            from sklearn.svm import SVC
            return SVC(random_state=random_state)
        
        else:
            # Default to Decision Tree
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(random_state=random_state)
    
    def _generate_visualizations(self, training_results: Dict, y_test: np.ndarray):
        """Generate confusion matrices and other visualizations"""
        
        try:
            # Get unique labels for confusion matrix
            unique_labels = sorted(list(set(y_test)))
            
            for result_key, result_data in training_results.items():
                if 'labels' not in result_data:
                    continue
                
                # Extract model name and vectorization method
                parts = result_key.split('_')
                if len(parts) >= 2:
                    model_name = parts[0]
                    vec_method = '_'.join(parts[1:])
                else:
                    model_name = result_key
                    vec_method = 'unknown'
                
                # Generate confusion matrix
                try:
                    if hasattr(plot_confusion_matrix, '__call__'):
                        plot_confusion_matrix(
                            y_test, result_data['labels'], unique_labels,
                            f"{model_name} Confusion Matrix ({vec_method})",
                            f"pdf/Figures/{model_name.lower()}_{vec_method}_confusion_matrix.pdf"
                        )
                    else:
                        # Fallback confusion matrix
                        self._create_fallback_confusion_matrix(
                            y_test, result_data['labels'], unique_labels,
                            model_name, vec_method
                        )
                except Exception as e:
                    print(f"Warning: Failed to create confusion matrix for {result_key}: {e}")
                    continue
            
            # Create model comparison plot
            try:
                if hasattr(plot_model_comparison, '__call__'):
                    # Prepare results for comparison
                    comparison_results = {}
                    for key, data in training_results.items():
                        if 'accuracy' in data:
                            comparison_results[key] = data['accuracy']
                    
                    plot_model_comparison(
                        comparison_results,
                        "pdf/Figures/model_comparison.pdf"
                    )
                else:
                    # Fallback model comparison
                    self._create_fallback_model_comparison(training_results)
            except Exception as e:
                print(f"Warning: Failed to create model comparison: {e}")
                # Try fallback
                try:
                    self._create_fallback_model_comparison(training_results)
                except Exception as e2:
                    print(f"Warning: Fallback model comparison also failed: {e2}")
                
        except Exception as e:
            print(f"Warning: Visualization generation failed: {e}")
    
    def _create_fallback_confusion_matrix(self, y_true, y_pred, labels, 
                                        model_name: str, vec_method: str):
        """Create confusion matrix using matplotlib as fallback"""
        
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            # Ensure y_true and y_pred are numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Handle empty arrays
            if len(y_true) == 0 or len(y_pred) == 0:
                print(f"Warning: Empty arrays for confusion matrix - {model_name} with {vec_method}")
                return
            
            # Ensure labels is a list and not empty
            if not isinstance(labels, (list, np.ndarray)) or len(labels) == 0:
                # Create labels from unique values in y_true and y_pred
                all_labels = np.concatenate([y_true, y_pred])
                labels = sorted(list(set(all_labels)))
                if len(labels) == 0:
                    print(f"Warning: No valid labels found for confusion matrix - {model_name} with {vec_method}")
                    return
            
            # Create confusion matrix
            try:
                cm = confusion_matrix(y_true, y_pred, labels=labels)
            except Exception as e:
                print(f"Warning: Confusion matrix calculation failed: {e}")
                # Try without labels
                cm = confusion_matrix(y_true, y_pred)
                # Update labels based on actual values
                labels = sorted(list(set(np.concatenate([y_true, y_pred]))))
            
            # Create plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
            plt.title(f'{model_name} Confusion Matrix ({vec_method})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save plot
            import os
            os.makedirs('pdf/Figures', exist_ok=True)
            plt.savefig(f'pdf/Figures/{model_name.lower()}_{vec_method}_confusion_matrix.pdf')
            plt.close()
            
            print(f"✅ Confusion matrix created for {model_name} with {vec_method}")
            
        except Exception as e:
            print(f"Warning: Fallback confusion matrix failed for {model_name} with {vec_method}: {e}")
    
    def _create_fallback_model_comparison(self, training_results: Dict):
        """Create model comparison using matplotlib as fallback"""
        try:
            import matplotlib.pyplot as plt
            
            # Prepare results for comparison
            comparison_results = {}
            for key, data in training_results.items():
                if 'accuracy' in data:
                    comparison_results[key] = data['accuracy']
            
            if not comparison_results:
                print("Warning: No accuracy results to compare")
                return
            
            # Create comparison plot
            plt.figure(figsize=(10, 6))
            models = list(comparison_results.keys())
            accuracies = list(comparison_results.values())
            
            plt.bar(range(len(models)), accuracies, color='skyblue', edgecolor='navy')
            plt.xlabel('Models')
            plt.ylabel('Accuracy')
            plt.title('Model Performance Comparison')
            plt.xticks(range(len(models)), models, rotation=45, ha='right')
            plt.ylim(0, 1)
            
            # Add accuracy values on bars
            for i, v in enumerate(accuracies):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            import os
            os.makedirs('pdf/Figures', exist_ok=True)
            plt.savefig('pdf/Figures/model_comparison_fallback.pdf')
            plt.close()
            
            print("✅ Fallback model comparison created successfully")
            
        except Exception as e:
            print(f"Warning: Fallback model comparison failed: {e}")
    
    def get_training_status(self) -> Dict:
        """Get current training status"""
        return {
            'status': self.training_status,
            'phase': self.current_phase,
            'current_model': self.current_model,
            'models_completed': self.models_completed,
            'total_models': self.total_models,
            'elapsed_time': self.elapsed_time,
            'start_time': self.start_time
        }
    
    def stop_training(self):
        """Stop training process"""
        self.training_status = "stopped"
        self.current_phase = "stopped"
    
    def reset_pipeline(self):
        """Reset pipeline to initial state"""
        self.results = {}
        self.training_status = "idle"
        self.current_model = None
        self.current_phase = "initializing"
        self.models_completed = 0
        self.total_models = 0
        self.start_time = None
        self.elapsed_time = 0


# Global pipeline instance for Streamlit
training_pipeline = StreamlitTrainingPipeline()


def execute_streamlit_training(df: pd.DataFrame, step1_data: Dict, 
                             step2_data: Dict, step3_data: Dict,
                             progress_callback=None) -> Dict:
    """Main function to execute training from Streamlit"""
    
    global training_pipeline
    
    # Reset pipeline if needed
    if training_pipeline.training_status == "training":
        training_pipeline.stop_training()
    
    training_pipeline.reset_pipeline()
    
    # Execute training
    result = training_pipeline.execute_training(
        df, step1_data, step2_data, step3_data, progress_callback
    )
    
    return result


def get_training_status() -> Dict:
    """Get current training status for Streamlit"""
    global training_pipeline
    return training_pipeline.get_training_status()


def stop_training():
    """Stop training process from Streamlit"""
    global training_pipeline
    training_pipeline.stop_training()


def reset_training():
    """Reset training pipeline from Streamlit"""
    global training_pipeline
    training_pipeline.reset_pipeline()
