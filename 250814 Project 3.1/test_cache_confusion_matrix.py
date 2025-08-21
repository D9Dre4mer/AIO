"""
Test Cache Confusion Matrix Data
Kiểm tra xem cache có đủ dữ liệu để vẽ confusion matrix không
"""

import os
import sys
import pickle
import json

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_cache_confusion_matrix_data():
    """Kiểm tra dữ liệu confusion matrix trong cache"""
    print("🔍 Kiểm tra dữ liệu confusion matrix trong cache")
    print("=" * 60)
    
    try:
        # Kiểm tra cache directory
        cache_dir = "cache/training_results"
        metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        
        if not os.path.exists(metadata_file):
            print("❌ Không tìm thấy cache metadata")
            return False
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ Tìm thấy {len(metadata)} cached results")
        
        # Kiểm tra từng cache file
        for cache_key, cache_info in metadata.items():
            print(f"\n🔑 Cache Key: {cache_key[:12]}...")
            print(f"   📁 File: {cache_info['file_path']}")
            
            # Load cached results
            cache_file = cache_info['file_path']
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cached_results = pickle.load(f)
                
                print(f"   📊 Status: {cached_results.get('status', 'N/A')}")
                print(f"   🎯 From Cache: {cached_results.get('from_cache', False)}")
                
                # Kiểm tra comprehensive_results
                if 'comprehensive_results' in cached_results:
                    comp_results = cached_results['comprehensive_results']
                    print(f"   📋 Comprehensive Results: {len(comp_results)} items")
                    
                    # Kiểm tra từng result
                    for i, result in enumerate(comp_results[:3]):  # Chỉ xem 3 đầu
                        print(f"      {i+1}. {result.get('model_name', 'N/A')} + {result.get('embedding_name', 'N/A')}")
                        print(f"         Status: {result.get('status', 'N/A')}")
                        print(f"         Test Accuracy: {result.get('test_accuracy', 'N/A')}")
                        
                        # Kiểm tra dữ liệu cho confusion matrix
                        confusion_matrix_data = []
                        if 'predictions' in result:
                            confusion_matrix_data.append("predictions")
                        if 'true_labels' in result:
                            confusion_matrix_data.append("true_labels")
                        if 'label_mapping' in result:
                            confusion_matrix_data.append("label_mapping")
                        if 'unique_labels' in result:
                            confusion_matrix_data.append("unique_labels")
                        
                        if confusion_matrix_data:
                            print(f"         📊 Confusion Matrix Data: {', '.join(confusion_matrix_data)}")
                            
                            # Hiển thị chi tiết
                            if 'predictions' in result:
                                pred = result['predictions']
                                print(f"            Predictions: {type(pred)}, shape: {getattr(pred, 'shape', len(pred))}")
                            
                            if 'true_labels' in result:
                                true = result['true_labels']
                                print(f"            True Labels: {type(true)}, shape: {getattr(true, 'shape', len(true))}")
                            
                            if 'label_mapping' in result:
                                mapping = result['label_mapping']
                                print(f"            Label Mapping: {mapping}")
                        else:
                            print(f"         ⚠️  Không có dữ liệu confusion matrix")
                
                print(f"   📈 Summary: {cached_results.get('successful_combinations', 0)}/{cached_results.get('total_combinations', 0)} successful")
                
            else:
                print(f"   ❌ Cache file không tồn tại")
        
        print("\n🎯 Kết luận về khả năng vẽ confusion matrix:")
        
        # Đánh giá khả năng vẽ confusion matrix
        can_plot = True
        missing_data = []
        
        for cache_key, cache_info in metadata.items():
            cache_file = cache_info['file_path']
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cached_results = pickle.load(f)
                
                if 'comprehensive_results' in cached_results:
                    for result in cached_results['comprehensive_results']:
                        if result.get('status') == 'success':
                            # Kiểm tra dữ liệu cần thiết
                            if 'predictions' not in result or 'true_labels' not in result:
                                can_plot = False
                                missing_data.append(f"{result.get('model_name', 'N/A')} + {result.get('embedding_name', 'N/A')}")
        
        if can_plot:
            print("✅ CÓ THỂ vẽ confusion matrix từ cache!")
            print("   - Có đủ dữ liệu predictions và true_labels")
            print("   - Có label mapping để hiển thị text labels")
        else:
            print("❌ KHÔNG THỂ vẽ confusion matrix từ cache")
            print(f"   - Thiếu dữ liệu cho: {', '.join(missing_data)}")
        
        return can_plot
        
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra cache: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plot_confusion_matrix_from_cache():
    """Test function plot_confusion_matrices_from_cache"""
    print("\n🎨 Test function plot_confusion_matrices_from_cache")
    print("=" * 60)
    
    try:
        from training_pipeline import StreamlitTrainingPipeline
        
        # Tạo pipeline instance
        pipeline = StreamlitTrainingPipeline()
        
        # Load cache data
        cache_dir = "cache/training_results"
        metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        
        if not os.path.exists(metadata_file):
            print("❌ Không tìm thấy cache metadata")
            return False
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Lấy cache đầu tiên để test
        if metadata:
            first_cache_key = list(metadata.keys())[0]
            cache_file = metadata[first_cache_key]['file_path']
            
            with open(cache_file, 'rb') as f:
                cached_results = pickle.load(f)
            
            print(f"✅ Loaded cache data: {first_cache_key[:12]}...")
            
            # Test function
            success = pipeline.plot_confusion_matrices_from_cache(cached_results)
            
            if success:
                print("✅ Function plot_confusion_matrices_from_cache hoạt động!")
                return True
            else:
                print("❌ Function plot_confusion_matrices_from_cache thất bại!")
                return False
        else:
            print("❌ Không có cache data để test")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi khi test function: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Bắt đầu test cache confusion matrix data...")
    
    # Test 1: Kiểm tra dữ liệu trong cache
    print("\n" + "="*60)
    print("TEST 1: Kiểm tra dữ liệu trong cache")
    print("="*60)
    success1 = test_cache_confusion_matrix_data()
    
    # Test 2: Test function plot_confusion_matrices_from_cache
    print("\n" + "="*60)
    print("TEST 2: Test function plot_confusion_matrices_from_cache")
    print("="*60)
    success2 = test_plot_confusion_matrix_from_cache()
    
    # Kết luận
    print("\n" + "="*60)
    print("KẾT QUẢ TEST")
    print("="*60)
    
    if success1 and success2:
        print("🎉 TẤT CẢ TEST THÀNH CÔNG!")
        print("✅ Cache có đủ dữ liệu để vẽ confusion matrix")
        print("✅ Function plot_confusion_matrices_from_cache hoạt động")
        print("🚀 Bạn có thể sử dụng cache để vẽ confusion matrix!")
    else:
        print("💥 CÓ TEST THẤT BẠI!")
        if not success1:
            print("❌ Cache không có đủ dữ liệu")
        if not success2:
            print("❌ Function plot_confusion_matrices_from_cache không hoạt động")
