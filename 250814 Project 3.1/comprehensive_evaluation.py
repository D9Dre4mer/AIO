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
        self.model_trainer = NewModelTrainer(
            cv_folds=cv_folds,
            validation_size=validation_size,
            test_size=test_size
        )
        
        # Results storage
        self.evaluation_results = {}
        self.overfitting_analysis = {}
        self.best_combinations = {}
        
        # Performance tracking
        self.training_times = {}
        self.prediction_times = {}
        
        print(f"🚀 Comprehensive Evaluator initialized with:")
        print(f"   • CV Folds: {cv_folds}")
        print(f"   • Validation Size: {validation_size:.1%}")
        print(f"   • Test Size: {test_size:.1%}")
        print(f"   • Random State: {random_state}")
        print(f"   • Note: Using reduced samples (1000) for faster testing")
    
    def load_and_prepare_data(self, max_samples: int = None, skip_csv_prompt: bool = False) -> Tuple[Dict[str, Any], List[str]]:
        """
        Load and prepare all data formats for evaluation
        
        Args:
            max_samples: Maximum number of samples to use
            skip_csv_prompt: If True, skip CSV backup prompt (for Streamlit usage)
        
        Returns:
            Tuple of (data_dict, sorted_labels)
        """
        print("\n📊 Loading and Preparing Data...")
        print("=" * 50)
        
        # Load dataset
        self.data_loader.load_dataset(skip_csv_prompt=skip_csv_prompt)
        
        # Select samples
        if skip_csv_prompt:
            print("🚀 Streamlit mode: Using existing data configuration...")
        
        self.data_loader.select_samples(max_samples)
        
        self.data_loader.preprocess_samples()
        self.data_loader.create_label_mappings()
        
        # Prepare train/val/test data
        X_train, X_test, y_train, y_test = self.data_loader.prepare_train_test_data()
        sorted_labels = self.data_loader.get_sorted_labels()
        
        # Create 3-way split using validation manager
        X_train_full, X_val, X_test, y_train_full, y_val, y_test = \
            validation_manager.split_data(
                np.concatenate([X_train, X_test]), 
                np.concatenate([y_train, y_test])
            )
        
        # Verify split consistency
        print(f"🔍 Split verification:")
        print(f"   • Total: {len(X_train_full) + len(X_val) + len(X_test)}")
        print(f"   • Train: {len(X_train_full)} | Val: {len(X_val)} | Test: {len(X_test)}")
        
        print(f"✅ Data prepared:")
        print(f"   • Training: {len(X_train_full)} samples")
        print(f"   • Validation: {len(X_val)} samples")
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
        
        # Create embeddings for immediate use
        print(f"\n🔤 Creating Embeddings for Data...")
        embeddings = self.create_all_embeddings(
            data_dict['X_train'], 
            data_dict['X_val'], 
            data_dict['X_test']
        )
        
        # Store embeddings in the evaluator
        self.embeddings = embeddings
        
        return data_dict, sorted_labels
    
    def create_all_embeddings(self, X_train: List[str], X_val: List[str], X_test: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Create all embedding representations for the data
        
        Returns:
            Dictionary of embeddings for each method
        """
        print("\n🔤 Creating All Embedding Representations...")
        print("=" * 50)
        
        embeddings = {}
        
        # 1. Bag of Words (BoW)
        print("📦 Processing Bag of Words...")
        start_time = time.time()
        X_train_bow = self.text_vectorizer.fit_transform_bow(X_train)
        X_val_bow = self.text_vectorizer.transform_bow(X_val)
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
        print("📊 Processing TF-IDF...")
        start_time = time.time()
        X_train_tfidf = self.text_vectorizer.fit_transform_tfidf(X_train)
        X_val_tfidf = self.text_vectorizer.transform_tfidf(X_val)
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
        print("🧠 Processing Word Embeddings...")
        start_time = time.time()
        X_train_emb = self.text_vectorizer.transform_embeddings(X_train)
        X_val_emb = self.text_vectorizer.transform_embeddings(X_val)
        X_test_emb = self.text_vectorizer.transform_embeddings(X_test)
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
        total_time = sum(emb['processing_time'] for emb in embeddings.values())
        print(f"\n📊 Embedding Summary:")
        print(f"   • Total processing time: {total_time:.2f}s")
        print(f"   • Memory efficient: {sum(1 for emb in embeddings.values() if emb['sparse'])}/3 methods use sparse matrices")
        
        return embeddings
    
    def evaluate_single_combination(self, 
                                  model_name: str, 
                                  embedding_name: str,
                                  X_train: Union[np.ndarray, sparse.csr_matrix],
                                  X_val: Union[np.ndarray, sparse.csr_matrix],
                                  X_test: Union[np.ndarray, sparse.csr_matrix],
                                  y_train: np.ndarray,
                                  y_val: np.ndarray,
                                  y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a single model-embedding combination
        
        Returns:
            Dictionary with evaluation results
        """
        combination_key = f"{model_name}_{embedding_name}"
        print(f"   🔍 Evaluating {combination_key}...")
        
        try:
            # Training
            start_time = time.time()
            y_test_pred, y_val_pred, y_test_true, val_acc, test_acc, test_metrics = \
                self.model_trainer.train_validate_test_model(
                    model_name, X_train, y_train, 
                    X_val, y_val, X_test, y_test
                )
            training_time = time.time() - start_time
            
            # Validation metrics
            val_metrics = ModelMetrics.compute_classification_metrics(y_val, y_val_pred)
            
            # Test metrics
            test_metrics = ModelMetrics.compute_classification_metrics(y_test, y_test_pred)
            
            # Overfitting analysis
            overfitting_score = val_acc - test_acc
            overfitting_status = self._classify_overfitting(overfitting_score)
            
            # Cross-validation
            cv_results = self.model_trainer.cross_validate_model(
                model_name, X_train, y_train, ['accuracy', 'precision', 'recall', 'f1']
            )
            
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
                
                # Overfitting analysis
                'overfitting_score': overfitting_score,
                'overfitting_status': overfitting_status,
                'overfitting_classification': self._get_overfitting_classification(overfitting_score),
                
                # Cross-validation results
                'cv_mean_accuracy': cv_results['overall_results']['accuracy_mean'],
                'cv_std_accuracy': cv_results['overall_results']['accuracy_std'],
                'cv_mean_f1': cv_results['overall_results']['f1_mean'],
                'cv_std_f1': cv_results['overall_results']['f1_std'],
                'cv_stability_score': self._calculate_stability_score(cv_results),
                
                # Timing
                'training_time': training_time,
                'total_samples': X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train),
                
                # Data characteristics
                'input_shape': X_train.shape,
                'n_classes': len(np.unique(y_train)),
                
                # Status
                'status': 'success',
                'error_message': None
            }
            
            print(f"     ✅ {combination_key}: Val={val_acc:.3f}, Test={test_acc:.3f}, CV={cv_results['overall_results']['accuracy_mean']:.3f}±{cv_results['overall_results']['accuracy_std']:.3f}")
            
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
                'overfitting_score': 0.0,
                'overfitting_status': 'error'
            }
            print(f"     ❌ {combination_key}: Error - {e}")
            return error_result
    
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
    
    def run_comprehensive_evaluation(self, max_samples: int = None, skip_csv_prompt: bool = False) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of all model-embedding combinations
        
        Args:
            max_samples: Maximum number of samples to use
            skip_csv_prompt: If True, skip CSV backup prompt (for Streamlit usage)
        
        Returns:
            Complete evaluation results
        """
        print("\n🚀 Starting Comprehensive Evaluation...")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. Load and prepare data
        data_dict, sorted_labels = self.load_and_prepare_data(max_samples, skip_csv_prompt)
        
        # 2. Create all embeddings
        embeddings = self.create_all_embeddings(
            data_dict['X_train'], 
            data_dict['X_val'], 
            data_dict['X_test']
        )
        
        # 3. Define all models to evaluate
        models_to_evaluate = [
            'kmeans', 'knn', 'decision_tree', 'naive_bayes', 'svm'
        ]
        
        # 4. Run evaluation for all combinations
        print(f"\n🤖 Evaluating All Model-Embedding Combinations...")
        print("=" * 60)
        
        all_results = []
        successful_combinations = 0
        total_combinations = len(models_to_evaluate) * len(embeddings)
        
        for model_name in models_to_evaluate:
            for embedding_name, embedding_data in embeddings.items():
                result = self.evaluate_single_combination(
                    model_name=model_name,
                    embedding_name=embedding_name,
                    X_train=embedding_data['X_train'],
                    X_val=embedding_data['X_val'],
                    X_test=embedding_data['X_test'],
                    y_train=data_dict['y_train'],
                    y_val=data_dict['y_val'],
                    y_test=data_dict['y_test']
                )
                
                all_results.append(result)
                if result['status'] == 'success':
                    successful_combinations += 1
        
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
        
        # 1. Best overall performance
        best_overall = max(successful_results, key=lambda x: x['test_accuracy'])
        print(f"🏆 Best Overall: {best_overall['combination_key']}")
        print(f"   • Test Accuracy: {best_overall['test_accuracy']:.3f}")
        print(f"   • Validation Accuracy: {best_overall['validation_accuracy']:.3f}")
        print(f"   • CV Stability: {best_overall['cv_stability_score']:.3f}")
        
        # 2. Best for each embedding
        print(f"\n📊 Best Model for Each Embedding:")
        for embedding in ['bow', 'tfidf', 'embeddings']:
            embedding_results = [r for r in successful_results if r['embedding_name'] == embedding]
            if embedding_results:
                best_embedding = max(embedding_results, key=lambda x: x['test_accuracy'])
                print(f"   • {embedding.upper()}: {best_embedding['model_name']} (Test: {best_embedding['test_accuracy']:.3f})")
        
        # 3. Best for each model
        print(f"\n🤖 Best Embedding for Each Model:")
        for model in ['kmeans', 'knn', 'decision_tree', 'naive_bayes', 'svm']:
            model_results = [r for r in successful_results if r['model_name'] == model]
            if model_results:
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
                        key=lambda x: x['test_accuracy']) 
                for emb in ['bow', 'tfidf', 'embeddings']
            },
            'best_by_model': {
                model: max([r for r in successful_results if r['model_name'] == model], 
                          key=lambda x: x['test_accuracy'])
                for model in ['kmeans', 'knn', 'decision_tree', 'naive_bayes', 'svm']
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
            report.append(f"   Test Accuracy: {best['test_accuracy']:.3f} | Val Accuracy: {best['validation_accuracy']:.3f}")
            report.append("")
        
        # Results table - simplified
        report.append("📈 RESULTS TABLE:")
        report.append("-" * 60)
        report.append(f"{'Combination':<20} {'Val Acc':<8} {'Test Acc':<8} {'CV Acc':<10}")
        report.append("-" * 60)
        
        for result in self.evaluation_results['all_results']:
            if result['status'] == 'success':
                cv_acc = f"{result['cv_mean_accuracy']:.3f}±{result['cv_std_accuracy']:.3f}"
                report.append(f"{result['combination_key']:<20} {result['validation_accuracy']:<8.3f} {result['test_accuracy']:<8.3f} {cv_acc:<10}")
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
        validation_size=0.2,
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
