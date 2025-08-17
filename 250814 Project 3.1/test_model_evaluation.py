#!/usr/bin/env python3
"""
Test script for Model Evaluation functionality
Demonstrates cross-validation, hyperparameter tuning, and performance analysis
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.classifier import AcademicPaperClassifier
from models.model_evaluator import ModelEvaluator, evaluate_all_models
from utils.text_processor import TextProcessor
from utils.cache_manager import load_cached_dataset


def create_sample_classification_data(n_samples: int = 1000) -> tuple:
    """
    Create sample data for classification testing.
    
    Args:
        n_samples: Number of samples to create
        
    Returns:
        Tuple of (X, y) for classification
    """
    print(f"🔧 Creating sample classification data with {n_samples} samples...")
    
    # Create sample text data (simulating academic paper abstracts)
    sample_texts = [
        "Machine learning algorithms for healthcare applications",
        "Quantum computing in cryptography and optimization",
        "Economic impact of climate change on global markets",
        "Neural networks for image recognition and classification",
        "Statistical analysis of social media data",
        "Biomedical engineering advances in prosthetics",
        "Chemical synthesis of novel pharmaceutical compounds",
        "Mathematical modeling of population dynamics",
        "Computer vision applications in autonomous vehicles",
        "Environmental science and sustainability research"
    ]
    
    # Create more diverse samples by repeating and modifying
    X = []
    y = []
    
    # Define some sample categories
    categories = ['Computer Science', 'Physics', 'Economics', 'Biology', 'Chemistry']
    
    for i in range(n_samples):
        # Select base text and modify slightly
        base_text = sample_texts[i % len(sample_texts)]
        modified_text = base_text + f" - Research {i} - " + " ".join([f"term{j}" for j in range(i % 5 + 1)])
        X.append(modified_text)
        
        # Assign category based on text content
        if 'machine learning' in modified_text.lower() or 'neural' in modified_text.lower():
            y.append('Computer Science')
        elif 'quantum' in modified_text.lower() or 'physics' in modified_text.lower():
            y.append('Physics')
        elif 'economic' in modified_text.lower() or 'market' in modified_text.lower():
            y.append('Economics')
        elif 'biomedical' in modified_text.lower() or 'biology' in modified_text.lower():
            y.append('Biology')
        elif 'chemical' in modified_text.lower() or 'synthesis' in modified_text.lower():
            y.append('Chemistry')
        else:
            y.append('Computer Science')  # Default
    
    print(f"✅ Created {len(X)} text samples with {len(set(y))} categories")
    print(f"📊 Category distribution: {pd.Series(y).value_counts().to_dict()}")
    
    return X, y


def test_model_evaluation():
    """Test the complete model evaluation pipeline."""
    
    print("🚀 Testing Model Evaluation Pipeline")
    print("=" * 60)
    
    # 1. Create sample data
    X, y = create_sample_classification_data(n_samples=500)
    
    # 2. Preprocess text data
    print("\n🔧 Preprocessing text data...")
    text_processor = TextProcessor()
    
    # Convert text to TF-IDF features
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_vectorized = vectorizer.fit_transform(X)
    
    print(f"✅ Text vectorized: {X_vectorized.shape}")
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    # 4. Initialize models
    print("\n🤖 Initializing classification models...")
    
    models = {}
    
    # SVM
    from sklearn.svm import SVC
    models['SVM'] = SVC(probability=True, random_state=42)
    
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Naive Bayes
    from sklearn.naive_bayes import MultinomialNB
    models['Naive Bayes'] = MultinomialNB()
    
    # Neural Network
    from sklearn.neural_network import MLPClassifier
    models['Neural Network'] = MLPClassifier(
        hidden_layer_sizes=(100, 50), 
        random_state=42, 
        max_iter=300
    )
    
    print(f"✅ Initialized {len(models)} models: {list(models.keys())}")
    
    # 5. Create model evaluator
    print("\n📊 Creating Model Evaluator...")
    evaluator = ModelEvaluator()
    
    # 6. Evaluate all models with cross-validation
    print("\n🔍 Evaluating models with cross-validation...")
    
    for model_name, model in models.items():
        print(f"\n📈 Evaluating {model_name}...")
        
        # Train and evaluate
        evaluator.evaluate_model_performance(
            model_name, model, X_train, y_train, X_test, y_test
        )
        
        # Cross-validation
        cv_results = evaluator.cross_validate_model(model_name, model, X_train, y_train)
        print(f"   CV Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
        print(f"   CV F1-Score: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
    
    # 7. Hyperparameter tuning for best model
    print("\n🎯 Performing hyperparameter tuning...")
    
    # Define parameter grids
    param_grids = {
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        },
        'Naive Bayes': {
            'alpha': [0.1, 0.5, 1.0, 2.0]
        }
    }
    
    # Perform tuning for SVM (as example)
    if 'SVM' in models:
        print("🔧 Tuning SVM hyperparameters...")
        tuning_results = evaluator.hyperparameter_tuning(
            'SVM', models['SVM'], param_grids['SVM'], X_train, y_train, method='grid'
        )
        print(f"   Best parameters: {tuning_results['best_params']}")
        print(f"   Best CV score: {tuning_results['best_score']:.4f}")
    
    # 8. Generate performance report
    print("\n📋 Generating performance report...")
    performance_report = evaluator.generate_performance_report()
    
    print("\n" + "="*80)
    print("📊 PERFORMANCE REPORT")
    print("="*80)
    print(performance_report.to_string(index=False))
    
    # 9. Create visualization plots
    print("\n📈 Creating performance visualization...")
    evaluator.plot_model_comparison()
    
    # 10. Get best model
    print("\n🏆 Best Model Analysis:")
    best_model_name, best_model = evaluator.get_best_model(metric='f1_macro')
    print(f"   Best model by F1-Score: {best_model_name}")
    
    best_model_name_acc, _ = evaluator.get_best_model(metric='accuracy')
    print(f"   Best model by Accuracy: {best_model_name_acc}")
    
    # 11. Save results
    print("\n💾 Saving evaluation results...")
    results_file = evaluator.save_evaluation_results()
    print(f"   Results saved to: {results_file}")
    
    print("\n🎉 Model evaluation pipeline completed successfully!")
    
    return evaluator


def test_with_real_data():
    """Test with real academic paper data from cache."""
    
    print("\n" + "="*60)
    print("📚 Testing with Real Academic Paper Data")
    print("="*60)
    
    try:
        # Load real dataset
        dataset_name = "UniverseTBD/arxiv-abstracts-large"
        dataset = load_cached_dataset(dataset_name, split="train")
        
        if dataset is not None:
            print(f"✅ Loaded dataset: {len(dataset)} papers")
            
            # Take a smaller sample for testing
            sample_size = min(1000, len(dataset))
            sample_data = dataset.select(range(sample_size))
            
            # Extract text and categories
            texts = [item.get('abstract', '') for item in sample_data]
            categories = [item.get('categories', 'Unknown') for item in sample_data]
            
            # Filter out empty texts and limit categories
            valid_data = [(text, cat) for text, cat in zip(texts, categories) if text.strip()]
            texts, categories = zip(*valid_data[:500])  # Limit to 500 samples
            
            print(f"📊 Using {len(texts)} valid samples")
            print(f"🏷️ Categories: {pd.Series(categories).value_counts().head().to_dict()}")
            
            # Convert to classification format
            # For demo, we'll use the first category from each paper
            y = [cat.split()[0] if cat and ' ' in cat else cat for cat in categories]
            
            # Preprocess and vectorize
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.model_selection import train_test_split
            
            vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            X_vectorized = vectorizer.fit_transform(texts)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_vectorized, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"🔧 Data prepared: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
            
            # Test with a few models
            models = {
                'SVM': SVC(probability=True, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Naive Bayes': MultinomialNB()
            }
            
            # Quick evaluation
            evaluator = ModelEvaluator()
            
            for model_name, model in models.items():
                print(f"\n📈 Quick evaluation of {model_name}...")
                evaluator.evaluate_model_performance(
                    model_name, model, X_train, y_train, X_test, y_test
                )
            
            # Generate report
            performance_report = evaluator.generate_performance_report()
            print("\n📊 Real Data Performance Report:")
            print(performance_report.to_string(index=False))
            
        else:
            print("⚠️ Could not load real dataset")
            
    except Exception as e:
        print(f"❌ Error testing with real data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test with sample data
    evaluator = test_model_evaluation()
    
    # Test with real data if available
    test_with_real_data()
