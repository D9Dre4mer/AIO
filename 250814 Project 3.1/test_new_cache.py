"""
Test New Cache with Confusion Matrix Data
Chạy training để tạo cache mới với dữ liệu confusion matrix
"""

import os
import sys
import pandas as pd
import numpy as np

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_new_cache():
    """Test tạo cache mới với dữ liệu confusion matrix"""
    print("🚀 Test tạo cache mới với dữ liệu confusion matrix")
    print("=" * 60)
    
    try:
        from training_pipeline import StreamlitTrainingPipeline
        
        # Tạo pipeline
        pipeline = StreamlitTrainingPipeline()
        
        # Tạo dữ liệu test đơn giản
        print("📊 Tạo dữ liệu test...")
        np.random.seed(42)
        
        # Tạo 100 samples với 5 classes
        n_samples = 100
        n_classes = 5
        
        # Tạo text data
        texts = [f"Sample text {i} for class {i % n_classes}" for i in range(n_samples)]
        
        # Tạo labels
        labels = [i % n_classes for i in range(n_samples)]
        
        # Tạo DataFrame
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
        
        print(f"✅ Tạo DataFrame: {df.shape}")
        print(f"   Text column: {df['text'].iloc[0][:50]}...")
        print(f"   Labels: {sorted(df['label'].unique())}")
        
        # Tạo step data
        step1_data = {
            'sampling_config': {
                'num_samples': 100,
                'sampling_strategy': 'random'
            }
        }
        
        step2_data = {
            'text_column': 'text',
            'label_column': 'label',
            'text_cleaning': True,
            'category_mapping': True,
            'data_validation': True,
            'memory_optimization': True
        }
        
        step3_data = {
            'data_split': {
                'test': 20,
                'validation': 10
            },
            'cross_validation': {
                'cv_folds': 3,
                'random_state': 42
            },
            'selected_models': ['K-Nearest Neighbors', 'Decision Tree'],
            'selected_vectorization': ['BoW', 'TF-IDF']
        }
        
        print("⚙️  Step data:")
        print(f"   Models: {step3_data['selected_models']}")
        print(f"   Vectorization: {step3_data['selected_vectorization']}")
        
        # Chạy training
        print("\n🚀 Bắt đầu training...")
        result = pipeline.execute_training(df, step1_data, step2_data, step3_data)
        
        if result.get('status') == 'success':
            print("✅ Training thành công!")
            print(f"   Successful combinations: {result.get('successful_combinations', 0)}/{result.get('total_combinations', 0)}")
            print(f"   From cache: {result.get('from_cache', False)}")
            
            # Kiểm tra cache mới
            print("\n🔍 Kiểm tra cache mới...")
            cached_results = pipeline.get_cached_results(step1_data, step2_data, step3_data)
            
            if cached_results:
                print("✅ Cache mới được tạo!")
                
                # Kiểm tra dữ liệu confusion matrix
                if 'comprehensive_results' in cached_results:
                    comp_results = cached_results['comprehensive_results']
                    print(f"   Comprehensive results: {len(comp_results)} items")
                    
                    # Kiểm tra từng result
                    for result_item in comp_results:
                        if result_item.get('status') == 'success':
                            model_name = result_item.get('model_name', 'N/A')
                            embedding_name = result_item.get('embedding_name', 'N/A')
                            
                            # Kiểm tra dữ liệu confusion matrix
                            confusion_matrix_data = []
                            if 'predictions' in result_item:
                                confusion_matrix_data.append("predictions")
                            if 'true_labels' in result_item:
                                confusion_matrix_data.append("true_labels")
                            if 'label_mapping' in result_item:
                                confusion_matrix_data.append("label_mapping")
                            
                            if confusion_matrix_data:
                                print(f"      ✅ {model_name} + {embedding_name}: {', '.join(confusion_matrix_data)}")
                            else:
                                print(f"      ❌ {model_name} + {embedding_name}: Thiếu dữ liệu")
                
                # Test vẽ confusion matrix từ cache
                print("\n🎨 Test vẽ confusion matrix từ cache...")
                success = pipeline.plot_confusion_matrices_from_cache(cached_results)
                
                if success:
                    print("✅ Vẽ confusion matrix từ cache thành công!")
                    return True
                else:
                    print("❌ Vẽ confusion matrix từ cache thất bại!")
                    return False
            else:
                print("❌ Không tìm thấy cache mới")
                return False
        else:
            print(f"❌ Training thất bại: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_new_cache()
    
    if success:
        print("\n🎉 TEST THÀNH CÔNG!")
        print("✅ Cache mới có đủ dữ liệu confusion matrix")
        print("✅ Có thể vẽ confusion matrix từ cache")
    else:
        print("\n💥 TEST THẤT BẠI!")
        print("❌ Cache không có đủ dữ liệu")
