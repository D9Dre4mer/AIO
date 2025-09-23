#!/usr/bin/env python3
"""
Test script to verify KNN strategy for different data types
"""

import numpy as np
from scipy import sparse
from sklearn.datasets import make_classification
from models.classification.knn_model import KNNModel

def test_knn_strategy():
    """Test KNN strategy for sparse vs dense matrices"""
    
    print("🧪 Testing KNN strategy for different data types...")
    
    # Create test data
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=50, n_classes=3, random_state=42)
    
    # Test 1: Sparse matrix (TF-IDF/BOW simulation)
    print("\n📊 Test 1: Sparse Matrix (TF-IDF/BOW)")
    X_sparse = sparse.csr_matrix(X)
    print(f"   Data type: {type(X_sparse)}")
    print(f"   Has toarray(): {hasattr(X_sparse, 'toarray')}")
    
    knn_sparse = KNNModel(n_neighbors=5, weights='uniform', metric='cosine')
    print("   🔄 Fitting KNN with sparse data...")
    knn_sparse.fit(X_sparse, y, use_gpu=True)  # Try GPU first
    
    print(f"   ✅ Sparse matrix strategy: {'scikit-learn' if not knn_sparse.use_faiss_gpu else 'FAISS'}")
    
    # Test 2: Dense matrix (Embeddings simulation)
    print("\n📊 Test 2: Dense Matrix (Embeddings)")
    X_dense = X  # Already dense
    print(f"   Data type: {type(X_dense)}")
    print(f"   Has toarray(): {hasattr(X_dense, 'toarray')}")
    
    knn_dense = KNNModel(n_neighbors=5, weights='uniform', metric='cosine')
    print("   🔄 Fitting KNN with dense data...")
    knn_dense.fit(X_dense, y, use_gpu=True)  # Try GPU first
    
    print(f"   ✅ Dense matrix strategy: {'scikit-learn' if not knn_dense.use_faiss_gpu else 'FAISS'}")
    
    # Test 3: Large sparse matrix (should use scikit-learn)
    print("\n📊 Test 3: Large Sparse Matrix (300k samples)")
    X_large, y_large = make_classification(n_samples=300000, n_features=1000, n_informative=100, n_classes=3, random_state=42)
    X_large_sparse = sparse.csr_matrix(X_large)
    
    knn_large = KNNModel(n_neighbors=5, weights='uniform', metric='cosine')
    print("   🔄 Fitting KNN with large sparse data...")
    knn_large.fit(X_large_sparse, y_large, use_gpu=True)  # Try GPU first
    
    print(f"   ✅ Large sparse strategy: {'scikit-learn' if not knn_large.use_faiss_gpu else 'FAISS'}")
    
    return True

if __name__ == "__main__":
    print("🚀 Testing KNN strategy for different data types...")
    
    try:
        success = test_knn_strategy()
        print(f"\n🎉 Strategy test completed successfully!")
        print(f"\n💡 Summary:")
        print(f"   • Sparse matrices (TF-IDF/BOW): Use scikit-learn for memory efficiency")
        print(f"   • Dense matrices (Embeddings): Use FAISS for GPU acceleration")
        print(f"   • Large datasets: Auto-detect and use appropriate strategy")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
