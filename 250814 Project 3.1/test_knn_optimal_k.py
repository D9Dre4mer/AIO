#!/usr/bin/env python3
"""
Test script for KNN Optimal K Determination
Demonstrates the new features integrated into the KNN model
"""

import sys
import os
import numpy as np

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.classification.knn_model import KNNModel
from data_loader import DataLoader
from text_encoders import TextEncoder


def test_knn_optimal_k():
    """Test the new optimal K determination features"""
    
    print("🚀 Testing KNN Optimal K Determination Features")
    print("=" * 60)
    
    # Load sample data
    print("\n📊 Loading sample dataset...")
    try:
        data_loader = DataLoader()
        # Use a small sample for testing
        df = data_loader.load_sample_dataset(max_samples=1000)
        
        if df is None or df.empty:
            print("⚠️  Could not load sample dataset, creating synthetic data...")
            # Create synthetic data for testing
            np.random.seed(42)
            n_samples = 500
            n_features = 100
            
            # Create synthetic text-like features
            X = np.random.rand(n_samples, n_features)
            # Create synthetic labels (3 classes)
            y = np.random.randint(0, 3, n_samples)
            
            print(f"✅ Created synthetic data: {X.shape}, {y.shape}")
        else:
            print(f"✅ Loaded dataset: {df.shape}")
            # Use first text column and first categorical column
            text_col = df.select_dtypes(include=['object']).columns[0]
            if len(df.select_dtypes(include=['object']).columns) > 1:
                label_col = df.select_dtypes(include=['object']).columns[1]
            else:
                label_col = df.columns[0]
            
            # Prepare text data
            text_encoder = TextEncoder()
            X = text_encoder.encode_text(df[text_col].fillna(''), method='tfidf')
            y = df[label_col].astype('category').cat.codes
            
            print(f"✅ Encoded text data: {X.shape}, labels: {y.shape}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Data split: Train {X_train.shape}, Test {X_test.shape}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Creating minimal synthetic data for testing...")
        
        # Minimal synthetic data
        np.random.seed(42)
        X = np.random.rand(100, 50)
        y = np.random.randint(0, 2, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Test 1: Basic KNN model
    print("\n🧪 Test 1: Basic KNN Model")
    print("-" * 40)
    
    knn_basic = KNNModel(n_neighbors=5)
    knn_basic.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn_basic.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"✅ Basic KNN accuracy: {accuracy:.4f}")
    
    # Test 2: Hyperparameter tuning
    print("\n🧪 Test 2: Hyperparameter Tuning")
    print("-" * 40)
    
    knn_tune = KNNModel()
    tuning_results = knn_tune.tune_hyperparameters(
        X_train, y_train, cv_folds=3, verbose=0
    )
    
    print(f"✅ Tuning completed:")
    print(f"   Best params: {tuning_results['best_params']}")
    print(f"   Best score: {tuning_results['best_score']:.4f}")
    
    # Test 3: Optimal K determination (NEW FEATURE!)
    print("\n🧪 Test 3: Optimal K Determination (NEW!)")
    print("-" * 40)
    
    knn_optimal = KNNModel()
    optimal_results = knn_optimal.determine_optimal_k(
        X_train, y_train, cv_folds=3, verbose=0, plot_results=True
    )
    
    print(f"✅ Optimal K determination completed:")
    print(f"   Optimal K: {optimal_results['optimal_k']}")
    print(f"   Best score: {optimal_results['best_score']:.4f}")
    print(f"   Best params: {optimal_results['best_params']}")
    
    # Test 4: Get optimal K summary
    print("\n🧪 Test 4: Optimal K Summary")
    print("-" * 40)
    
    summary = knn_optimal.get_optimal_k_summary()
    print(f"✅ Summary retrieved:")
    for key, value in summary.items():
        if key != 'cv_results':  # Skip large cv_results
            print(f"   {key}: {value}")
    
    # Test 5: Print optimal K summary
    print("\n🧪 Test 5: Print Optimal K Summary")
    print("-" * 40)
    
    knn_optimal.print_optimal_k_summary()
    
    # Test 6: Benchmark results
    print("\n🧪 Test 6: Benchmark Results")
    print("-" * 40)
    
    benchmark = knn_optimal.get_benchmark_results()
    if benchmark:
        print(f"✅ Benchmark results available for {len(benchmark)} K values")
        knn_optimal.print_benchmark_summary()
    else:
        print("ℹ️  No benchmark results (run tune_hyperparameters first)")
    
    # Test 7: Compare models
    print("\n🧪 Test 7: Model Comparison")
    print("-" * 40)
    
    # Test basic model
    y_pred_basic = knn_basic.predict(X_test)
    acc_basic = (y_pred_basic == y_test).mean()
    
    # Test tuned model
    y_pred_tuned = knn_tune.predict(X_test)
    acc_tuned = (y_pred_tuned == y_test).mean()
    
    # Test optimal K model
    y_pred_optimal = knn_optimal.predict(X_test)
    acc_optimal = (y_pred_optimal == y_test).mean()
    
    print(f"📊 Model Performance Comparison:")
    print(f"   Basic KNN (K=5):     {acc_basic:.4f}")
    print(f"   Tuned KNN:           {acc_tuned:.4f}")
    print(f"   Optimal K KNN:       {acc_optimal:.4f}")
    
    # Find best model
    models = [
        ("Basic KNN (K=5)", acc_basic),
        ("Tuned KNN", acc_tuned),
        ("Optimal K KNN", acc_optimal)
    ]
    best_model_name, best_acc = max(models, key=lambda x: x[1])
    
    print(f"\n🏆 Best Model: {best_model_name} with accuracy: {best_acc:.4f}")
    
    print("\n✅ All tests completed successfully!")
    print("🎯 New features integrated:")
    print("   - determine_optimal_k(): Find optimal K with cosine metric")
    print("   - _plot_k_benchmark(): Visualize K performance")
    print("   - get_optimal_k_summary(): Get optimal K results")
    print("   - print_optimal_k_summary(): Print formatted summary")
    
    return True


if __name__ == "__main__":
    try:
        success = test_knn_optimal_k()
        if success:
            print("\n🎉 KNN Optimal K features test completed successfully!")
        else:
            print("\n❌ Some tests failed")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
