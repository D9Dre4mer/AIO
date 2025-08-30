#!/usr/bin/env python3
"""
Script để điều tra vấn đề label mapping và tìm cách khôi phục
"""

import pickle
import os
import numpy as np
import pandas as pd

def investigate_label_issue():
    """Điều tra vấn đề label mapping"""
    
    print("🔍 ĐIỀU TRA VẤN ĐỀ LABEL MAPPING")
    print("="*80)
    
    # Đường dẫn cache
    cache_dir = "cache/training_results"
    
    # Tìm file cache mới nhất
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
    
    if not cache_files:
        print("❌ Không tìm thấy file cache nào")
        return
    
    # Lấy file cache mới nhất
    latest_cache = max(cache_files, key=lambda x: os.path.getmtime(os.path.join(cache_dir, x)))
    cache_path = os.path.join(cache_dir, latest_cache)
    
    print(f"📁 Cache file: {latest_cache}")
    print("="*80)
    
    try:
        # Đọc cache file
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Phân tích vấn đề
        print("🔍 PHÂN TÍCH VẤN ĐỀ:")
        
        # 1. Kiểm tra step1_data
        print("\n1️⃣ KIỂM TRA STEP1_DATA:")
        if 'step1_data' in cache_data:
            step1 = cache_data['step1_data']
            print(f"   - Keys: {list(step1.keys())}")
            
            # Kiểm tra uploaded_file
            if 'uploaded_file' in step1:
                print(f"   ✅ Có uploaded_file")
            else:
                print(f"   ❌ THIẾU uploaded_file → cache key hiển thị 'unknown_dataset'")
            
            # Kiểm tra selected_categories
            if 'selected_categories' in step1:
                print(f"   ✅ Có selected_categories: {step1['selected_categories']}")
            else:
                print(f"   ❌ THIẾU selected_categories → cache key hiển thị 'no_cats'")
            
            # Kiểm tra dataframe
            if 'dataframe' in step1 and step1['dataframe'] is not None:
                df = step1['dataframe']
                print(f"   ✅ Có dataframe gốc")
                print(f"   - Shape: {df.shape}")
                print(f"   - Columns: {list(df.columns)}")
                
                # Tìm cột label
                label_col = None
                for col in ['label', 'category', 'class', 'target', 'y']:
                    if col in df.columns:
                        label_col = col
                        break
                
                if label_col:
                    print(f"   - Label column: {label_col}")
                    
                    # Lấy labels gốc từ dataframe
                    original_labels = df[label_col].unique()
                    print(f"   - Labels gốc từ dataframe: {original_labels}")
                    
                    # Kiểm tra xem labels có phải là số không
                    numeric_labels = [str(v) for v in original_labels if str(v).isdigit()]
                    if len(numeric_labels) == len(original_labels):
                        print(f"   ⚠️  Labels gốc đã là số → có thể đã bị encode trước đó")
                    else:
                        print(f"   ✅ Labels gốc có ý nghĩa → có thể khôi phục")
                else:
                    print(f"   ❌ Không tìm thấy cột label")
            else:
                print(f"   ❌ Không có dataframe")
        else:
            print(f"   ❌ Không có step1_data")
        
        # 2. Kiểm tra label_mapping hiện tại
        print("\n2️⃣ KIỂM TRA LABEL_MAPPING HIỆN TẠI:")
        if 'label_mapping' in cache_data:
            current_mapping = cache_data['label_mapping']
            print(f"   - Label mapping: {current_mapping}")
            print(f"   - Type: {type(current_mapping)}")
            
            # Kiểm tra chất lượng mapping
            if isinstance(current_mapping, dict):
                numeric_keys = [k for k in current_mapping.keys() if isinstance(k, (int, np.integer))]
                meaningful_values = [v for v in current_mapping.values() if not str(v).isdigit()]
                
                print(f"   - Keys số: {len(numeric_keys)} - {numeric_keys}")
                print(f"   - Values có nghĩa: {len(meaningful_values)} - {meaningful_values}")
                
                if len(meaningful_values) == 0:
                    print(f"   ❌ VẤN ĐỀ: Tất cả values đều là số → cần khôi phục labels gốc")
                else:
                    print(f"   ✅ Labels có ý nghĩa")
        else:
            print(f"   ❌ Không có label_mapping")
        
        # 3. Kiểm tra comprehensive_results
        print("\n3️⃣ KIỂM TRA COMPREHENSIVE_RESULTS:")
        if 'comprehensive_results' in cache_data:
            comp_results = cache_data['comprehensive_results']
            print(f"   - Số lượng kết quả: {len(comp_results)}")
            
            if comp_results:
                first_result = comp_results[0]
                print(f"   - Keys trong kết quả đầu tiên: {list(first_result.keys())}")
                
                # Kiểm tra label_mapping trong kết quả
                if 'label_mapping' in first_result:
                    result_mapping = first_result['label_mapping']
                    print(f"   - Label mapping trong kết quả: {result_mapping}")
                else:
                    print(f"   ❌ Không có label_mapping trong kết quả")
                
                # Kiểm tra predictions và true_labels
                if 'predictions' in first_result and 'true_labels' in first_result:
                    predictions = first_result['predictions']
                    true_labels = first_result['true_labels']
                    print(f"   ✅ Có predictions và true_labels")
                    print(f"   - Predictions type: {type(predictions)}, length: {len(predictions) if hasattr(predictions, '__len__') else 'N/A'}")
                    print(f"   - True labels type: {type(true_labels)}, length: {len(true_labels) if hasattr(true_labels, '__len__') else 'N/A'}")
                    
                    if hasattr(true_labels, '__len__') and len(true_labels) > 0:
                        unique_true = set(true_labels)
                        unique_pred = set(predictions)
                        print(f"   - Unique true_labels: {sorted(unique_true)}")
                        print(f"   - Unique predictions: {sorted(unique_pred)}")
                else:
                    print(f"   ❌ Thiếu predictions hoặc true_labels")
        else:
            print(f"   ❌ Không có comprehensive_results")
        
        # 4. Phân tích nguyên nhân
        print("\n4️⃣ PHÂN TÍCH NGUYÊN NHÂN:")
        
        # Kiểm tra cache key
        cache_key = cache_data.get('cache_key', '')
        print(f"   - Cache key: {cache_key}")
        
        if 'unknown_dataset' in cache_key:
            print(f"   ❌ NGUYÊN NHÂN 1: step1_data thiếu 'uploaded_file'")
            print(f"      → Không thể xác định tên dataset")
        
        if 'no_cats' in cache_key:
            print(f"   ❌ NGUYÊN NHÂN 2: step1_data thiếu 'selected_categories'")
            print(f"      → Không thể tạo label mapping gốc")
        
        # 5. Đề xuất giải pháp
        print("\n5️⃣ ĐỀ XUẤT GIẢI PHÁP:")
        
        if 'step1_data' in cache_data:
            step1 = cache_data['step1_data']
            
            if 'dataframe' in step1 and step1['dataframe'] is not None:
                df = step1['dataframe']
                print(f"   ✅ GIẢI PHÁP 1: Khôi phục labels từ dataframe gốc")
                print(f"      - DataFrame có sẵn trong cache")
                print(f"      - Có thể tìm cột label và tạo mapping gốc")
                
                # Tìm cột label
                label_col = None
                for col in ['label', 'category', 'class', 'target', 'y']:
                    if col in df.columns:
                        label_col = col
                        break
                
                if label_col:
                    original_labels = df[label_col].unique()
                    print(f"      - Tìm thấy cột label: {label_col}")
                    print(f"      - Labels gốc: {original_labels}")
                    
                    # Tạo mapping gốc
                    original_mapping = {i: label for i, label in enumerate(sorted(original_labels))}
                    print(f"      - Mapping gốc: {original_mapping}")
                    
                    print(f"   ✅ CÓ THỂ KHÔI PHỤC!")
                else:
                    print(f"   ❌ Không tìm thấy cột label trong dataframe")
            
            elif 'file_path' in step1:
                file_path = step1['file_path']
                print(f"   ✅ GIẢI PHÁP 2: Đọc lại file gốc")
                print(f"      - File path: {file_path}")
                print(f"      - Có thể đọc file và tạo mapping gốc")
                
                if os.path.exists(file_path):
                    print(f"      - File tồn tại")
                    print(f"   ✅ CÓ THỂ KHÔI PHỤC!")
                else:
                    print(f"      - File không tồn tại")
                    print(f"   ❌ KHÔNG THỂ KHÔI PHỤC!")
            else:
                print(f"   ❌ KHÔNG CÓ GIẢI PHÁP: Không có dataframe hoặc file_path")
        else:
            print(f"   ❌ KHÔNG CÓ GIẢI PHÁP: Không có step1_data")
        
        print("\n" + "="*80)
        print("📋 TÓM TẮT VẤN ĐỀ:")
        print("   • Cache key hiển thị 'unknown_dataset' và 'no_cats'")
        print("   • step1_data thiếu 'uploaded_file' và 'selected_categories'")
        print("   • LabelEncoder chuyển đổi labels thành số nhưng không lưu mapping gốc")
        print("   • Labels trong cache là số (0, 1) thay vì labels có nghĩa")
        print("   • Cần khôi phục labels gốc từ dataset để tạo confusion matrix đúng")
        
    except Exception as e:
        print(f"❌ Lỗi khi điều tra: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    investigate_label_issue()
