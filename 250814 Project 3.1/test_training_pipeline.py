"""
Test script for training pipeline
"""

import pandas as pd
import numpy as np
from training_pipeline import execute_streamlit_training

def test_training_pipeline():
    """Test the training pipeline with sample data"""
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 100
    
    # Generate sample text data
    sample_texts = [
        "This is a positive review about the product",
        "I love this service, it's amazing",
        "Great experience with the company",
        "This is a negative review, very bad",
        "Terrible service, would not recommend",
        "Poor quality product, disappointed",
        "Excellent customer support",
        "Bad experience overall",
        "Wonderful product, highly recommend",
        "Awful service, waste of money"
    ]
    
    # Create text column
    texts = np.random.choice(sample_texts, n_samples)
    
    # Create labels (0: negative, 1: positive)
    labels = np.random.choice([0, 1], n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    print(f"Created test dataset with {len(df)} samples")
    print(f"Text column: {df['text'].iloc[0]}")
    print(f"Label column: {df['label'].iloc[0]}")
    
    # Create step configurations
    step1_data = {
        'sampling_config': {
            'num_samples': n_samples,
            'sampling_strategy': 'Random'
        }
    }
    
    step2_data = {
        'text_column': 'text',
        'label_column': 'label',
        'text_cleaning': True,
        'category_mapping': True,
        'data_validation': True,
        'memory_optimization': True,
        'completed': True
    }
    
    step3_data = {
        'data_split': {
            'training': 70,
            'validation': 10,
            'test': 20
        },
        'cross_validation': {
            'cv_folds': 5,
            'random_state': 42
        },
        'selected_models': ['Decision Tree', 'Naive Bayes'],
        'selected_vectorization': ['Bag of Words (BoW)', 'TF-IDF'],
        'completed': True
    }
    
    print("\nStep configurations created:")
    print(f"Step 1: Sampling config with {step1_data['sampling_config']['num_samples']} samples")
    print(f"Step 2: Text column '{step2_data['text_column']}', Label column '{step2_data['label_column']}'")
    print(f"Step 3: {len(step3_data['selected_models'])} models, {len(step3_data['selected_vectorization'])} vectorization methods")
    
    # Test training pipeline
    print("\n🚀 Starting training pipeline test...")
    
    try:
        result = execute_streamlit_training(
            df, step1_data, step2_data, step3_data
        )
        
        if result['status'] == 'success':
            print("✅ Comprehensive evaluation test successful!")
            print(f"Successful combinations: {result.get('successful_combinations', 0)}/{result.get('total_combinations', 0)}")
            print(f"Total evaluation time: {result['elapsed_time']:.2f} seconds")
            print(f"Evaluation time: {result.get('evaluation_time', 0):.2f} seconds")
            
            # Display best combinations if available
            if 'best_combinations' in result and result['best_combinations']:
                best_overall = result['best_combinations'].get('best_overall', {})
                if best_overall:
                    print(f"🏆 Best overall: {best_overall.get('combination_key', 'N/A')} (Test: {best_overall.get('test_accuracy', 0):.3f})")
            
            # Display data info
            if 'data_info' in result:
                data_info = result['data_info']
                print(f"📊 Dataset: {data_info.get('n_samples', 0)} train, {data_info.get('n_validation', 0)} val, {data_info.get('n_test', 0)} test, {data_info.get('n_classes', 0)} classes")
        else:
            print(f"❌ Comprehensive evaluation test failed: {result['message']}")
            if 'error' in result:
                print(f"Error details: {result['error']}")
                
    except Exception as e:
        print(f"❌ Exception during training pipeline test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training_pipeline()
