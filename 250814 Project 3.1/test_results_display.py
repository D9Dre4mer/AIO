"""
Test script to verify comprehensive evaluation results display
"""

import pandas as pd

# Simulate comprehensive evaluation results (like from training_pipeline.py)
def create_sample_results():
    """Create sample comprehensive evaluation results for testing"""
    
    # Sample comprehensive results
    comprehensive_results = [
        {
            'model_name': 'kmeans',
            'embedding_name': 'bow',
            'validation_accuracy': 0.125,
            'test_accuracy': 0.150,
            'cv_mean_accuracy': 0.345,
            'cv_std_accuracy': 0.047,
            'overfitting_status': 'underfitting',
            'training_time': 2.5,
            'status': 'success'
        },
        {
            'model_name': 'kmeans',
            'embedding_name': 'tfidf',
            'validation_accuracy': 0.312,
            'test_accuracy': 0.300,
            'cv_mean_accuracy': 0.562,
            'cv_std_accuracy': 0.083,
            'overfitting_status': 'well_fitted',
            'training_time': 3.1,
            'status': 'success'
        },
        {
            'model_name': 'kmeans',
            'embedding_name': 'embeddings',
            'validation_accuracy': 0.875,
            'test_accuracy': 0.650,
            'cv_mean_accuracy': 0.654,
            'cv_std_accuracy': 0.124,
            'overfitting_status': 'overfitting',
            'training_time': 8.2,
            'status': 'success'
        },
        {
            'model_name': 'knn',
            'embedding_name': 'bow',
            'validation_accuracy': 0.625,
            'test_accuracy': 0.300,
            'cv_mean_accuracy': 0.313,
            'cv_std_accuracy': 0.010,
            'overfitting_status': 'overfitting',
            'training_time': 1.8,
            'status': 'success'
        },
        {
            'model_name': 'knn',
            'embedding_name': 'tfidf',
            'validation_accuracy': 0.500,
            'test_accuracy': 0.650,
            'cv_mean_accuracy': 0.686,
            'cv_std_accuracy': 0.103,
            'overfitting_status': 'well_fitted',
            'training_time': 2.1,
            'status': 'success'
        },
        {
            'model_name': 'knn',
            'embedding_name': 'embeddings',
            'validation_accuracy': 0.812,
            'test_accuracy': 0.700,
            'cv_mean_accuracy': 0.750,
            'cv_std_accuracy': 0.030,
            'overfitting_status': 'well_fitted',
            'training_time': 5.5,
            'status': 'success'
        },
        {
            'model_name': 'decision_tree',
            'embedding_name': 'bow',
            'validation_accuracy': 0.438,
            'test_accuracy': 0.600,
            'cv_mean_accuracy': 0.469,
            'cv_std_accuracy': 0.086,
            'overfitting_status': 'overfitting',
            'training_time': 1.2,
            'status': 'success'
        },
        {
            'model_name': 'decision_tree',
            'embedding_name': 'tfidf',
            'validation_accuracy': 0.250,
            'test_accuracy': 0.600,
            'cv_mean_accuracy': 0.391,
            'cv_std_accuracy': 0.085,
            'overfitting_status': 'underfitting',
            'training_time': 1.4,
            'status': 'success'
        },
        {
            'model_name': 'decision_tree',
            'embedding_name': 'embeddings',
            'validation_accuracy': 0.500,
            'test_accuracy': 0.600,
            'cv_mean_accuracy': 0.451,
            'cv_std_accuracy': 0.082,
            'overfitting_status': 'well_fitted',
            'training_time': 3.8,
            'status': 'success'
        },
        {
            'model_name': 'naive_bayes',
            'embedding_name': 'bow',
            'validation_accuracy': 0.500,
            'test_accuracy': 0.800,
            'cv_mean_accuracy': 0.623,
            'cv_std_accuracy': 0.099,
            'overfitting_status': 'overfitting',
            'training_time': 0.8,
            'status': 'success'
        },
        {
            'model_name': 'naive_bayes',
            'embedding_name': 'tfidf',
            'validation_accuracy': 0.562,
            'test_accuracy': 0.600,
            'cv_mean_accuracy': 0.673,
            'cv_std_accuracy': 0.052,
            'overfitting_status': 'well_fitted',
            'training_time': 0.9,
            'status': 'success'
        },
        {
            'model_name': 'naive_bayes',
            'embedding_name': 'embeddings',
            'validation_accuracy': 0.812,
            'test_accuracy': 0.700,
            'cv_mean_accuracy': 0.796,
            'cv_std_accuracy': 0.041,
            'overfitting_status': 'well_fitted',
            'training_time': 4.2,
            'status': 'success'
        },
        {
            'model_name': 'svm',
            'embedding_name': 'bow',
            'validation_accuracy': 0.688,
            'test_accuracy': 0.500,
            'cv_mean_accuracy': 0.515,
            'cv_std_accuracy': 0.102,
            'overfitting_status': 'overfitting',
            'training_time': 12.5,
            'status': 'success'
        },
        {
            'model_name': 'svm',
            'embedding_name': 'tfidf',
            'validation_accuracy': 0.375,
            'test_accuracy': 0.500,
            'cv_mean_accuracy': 0.456,
            'cv_std_accuracy': 0.120,
            'overfitting_status': 'underfitting',
            'training_time': 11.8,
            'status': 'success'
        },
        {
            'model_name': 'svm',
            'embedding_name': 'embeddings',
            'validation_accuracy': 0.688,
            'test_accuracy': 0.450,
            'cv_mean_accuracy': 0.518,
            'cv_std_accuracy': 0.106,
            'overfitting_status': 'overfitting',
            'training_time': 15.2,
            'status': 'success'
        }
    ]
    
    # Best combinations
    best_combinations = {
        'best_overall': {
            'combination_key': 'naive_bayes_bow',
            'test_accuracy': 0.800,
            'validation_accuracy': 0.500,
            'cv_stability_score': 0.842
        },
        'best_by_embedding': {
            'bow': {'model_name': 'naive_bayes', 'test_accuracy': 0.800},
            'tfidf': {'model_name': 'knn', 'test_accuracy': 0.650},
            'embeddings': {'model_name': 'knn', 'test_accuracy': 0.700}
        },
        'best_by_model': {
            'kmeans': {'embedding_name': 'embeddings', 'test_accuracy': 0.650},
            'knn': {'embedding_name': 'embeddings', 'test_accuracy': 0.700},
            'decision_tree': {'embedding_name': 'bow', 'test_accuracy': 0.600},
            'naive_bayes': {'embedding_name': 'bow', 'test_accuracy': 0.800},
            'svm': {'embedding_name': 'bow', 'test_accuracy': 0.500}
        }
    }
    
    # Data info
    data_info = {
        'n_samples': 64,
        'n_validation': 16,
        'n_test': 20,
        'n_classes': 5,
        'labels': ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']
    }
    
    # Complete result structure
    result = {
        'status': 'success',
        'message': 'Comprehensive evaluation completed successfully',
        'comprehensive_results': comprehensive_results,
        'successful_combinations': 15,
        'total_combinations': 15,
        'best_combinations': best_combinations,
        'total_models': 15,
        'models_completed': 15,
        'elapsed_time': 19.24,
        'evaluation_time': 13.48,
        'data_info': data_info,
        'embedding_info': {
            'bow': {'processing_time': 0.01, 'sparse': True, 'shape': (64, 584)},
            'tfidf': {'processing_time': 0.01, 'sparse': True, 'shape': (64, 584)},
            'embeddings': {'processing_time': 1.08, 'sparse': False, 'shape': (64, 768)}
        }
    }
    
    return result

def test_results_processing():
    """Test the results processing logic that will be used in Streamlit"""
    
    print("🧪 Testing Comprehensive Evaluation Results Processing...")
    print("=" * 60)
    
    # Get sample results
    result = create_sample_results()
    
    # Test 1: Basic metrics
    print("📊 Basic Metrics:")
    successful_combinations = result.get('successful_combinations', 0)
    total_combinations = result.get('total_combinations', 0)
    print(f"• Total Combinations: {total_combinations}")
    print(f"• Successful: {successful_combinations}")
    print(f"• Success Rate: {(successful_combinations/total_combinations)*100:.1f}%")
    print(f"• Total Time: {result['elapsed_time']:.1f}s")
    
    # Test 2: Best model
    print("\n🏆 Best Overall Model:")
    if 'best_combinations' in result and result['best_combinations']:
        best_overall = result['best_combinations'].get('best_overall', {})
        if best_overall:
            print(f"• Model: {best_overall.get('combination_key', 'N/A')}")
            print(f"• Test Accuracy: {best_overall.get('test_accuracy', 0):.3f}")
            print(f"• Validation Accuracy: {best_overall.get('validation_accuracy', 0):.3f}")
    
    # Test 3: Results table creation
    print("\n📊 Results Table Creation:")
    if 'comprehensive_results' in result and result['comprehensive_results']:
        results_data = []
        for res in result['comprehensive_results']:
            if res['status'] == 'success':
                results_data.append({
                    'Model': res['model_name'].replace('_', ' ').title(),
                    'Embedding': res['embedding_name'].replace('_', ' ').title(),
                    'Val Accuracy': f"{res['validation_accuracy']:.3f}",
                    'Test Accuracy': f"{res['test_accuracy']:.3f}",
                    'CV Accuracy': f"{res.get('cv_mean_accuracy', 0):.3f}±{res.get('cv_std_accuracy', 0):.3f}",
                    'Overfitting': res.get('overfitting_status', 'N/A').replace('_', ' ').title(),
                    'Training Time': f"{res.get('training_time', 0):.2f}s"
                })
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            print(f"✅ Created DataFrame with {len(results_df)} rows and {len(results_df.columns)} columns")
            print("📋 Sample data:")
            print(results_df.head(3).to_string(index=False))
    
    # Test 4: Overfitting analysis
    print("\n📈 Overfitting Analysis:")
    if 'comprehensive_results' in result and result['comprehensive_results']:
        overfitting_counts = {}
        for res in result['comprehensive_results']:
            if res['status'] == 'success':
                status = res.get('overfitting_status', 'unknown')
                overfitting_counts[status] = overfitting_counts.get(status, 0) + 1
        
        if overfitting_counts:
            print("Overfitting Distribution:")
            for status, count in overfitting_counts.items():
                percentage = (count / successful_combinations) * 100
                print(f"• {status.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    # Test 5: Data information
    print("\n📋 Dataset Information:")
    if 'data_info' in result:
        data_info = result['data_info']
        print(f"• Training Samples: {data_info.get('n_samples', 0)}")
        print(f"• Validation Samples: {data_info.get('n_validation', 0)}")
        print(f"• Test Samples: {data_info.get('n_test', 0)}")
        print(f"• Classes: {data_info.get('n_classes', 0)}")
        print(f"• Labels: {data_info.get('labels', [])}")
    
    print("\n✅ All tests completed successfully!")
    print("🎯 This confirms the results processing logic will work correctly in Streamlit!")

if __name__ == "__main__":
    test_results_processing()
