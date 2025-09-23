#!/usr/bin/env python3
"""
GPU Configuration Manager
Quản lý cấu hình GPU optimization cho Topic Modeling Auto Classifier
"""

import os
import sys
from pathlib import Path

def update_config(enable_gpu=False, force_dense=False):
    """Cập nhật cấu hình GPU trong config.py"""
    
    config_path = Path("config.py")
    if not config_path.exists():
        print("❌ Không tìm thấy file config.py")
        return False
    
    # Đọc file config hiện tại
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Cập nhật các giá trị
    content = content.replace(
        f"ENABLE_GPU_OPTIMIZATION = {not enable_gpu}", 
        f"ENABLE_GPU_OPTIMIZATION = {enable_gpu}"
    )
    content = content.replace(
        f"FORCE_DENSE_CONVERSION = {not force_dense}", 
        f"FORCE_DENSE_CONVERSION = {force_dense}"
    )
    
    # Ghi lại file
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Cấu hình đã được cập nhật:")
    print(f"   • ENABLE_GPU_OPTIMIZATION = {enable_gpu}")
    print(f"   • FORCE_DENSE_CONVERSION = {force_dense}")
    
    return True

def show_current_config():
    """Hiển thị cấu hình hiện tại"""
    try:
        from config import ENABLE_GPU_OPTIMIZATION, FORCE_DENSE_CONVERSION
        
        print("🔧 Cấu hình GPU hiện tại:")
        print(f"   • ENABLE_GPU_OPTIMIZATION = {ENABLE_GPU_OPTIMIZATION}")
        print(f"   • FORCE_DENSE_CONVERSION = {FORCE_DENSE_CONVERSION}")
        
        if ENABLE_GPU_OPTIMIZATION or FORCE_DENSE_CONVERSION:
            print("\n🚀 Chế độ: GPU OPTIMIZATION")
            print("   • Sử dụng dense matrices")
            print("   • Kích hoạt GPU acceleration")
            print("   • Tốn nhiều memory hơn")
        else:
            print("\n💾 Chế độ: MEMORY OPTIMIZATION")
            print("   • Sử dụng sparse matrices")
            print("   • Tắt GPU acceleration")
            print("   • Tiết kiệm memory")
            
    except ImportError as e:
        print(f"❌ Không thể đọc cấu hình: {e}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("🔧 GPU Configuration Manager")
        print("\nCách sử dụng:")
        print("  python gpu_config_manager.py show          # Hiển thị cấu hình hiện tại")
        print("  python gpu_config_manager.py gpu           # Bật GPU optimization")
        print("  python gpu_config_manager.py memory        # Bật memory optimization")
        print("  python gpu_config_manager.py dense         # Bật dense conversion")
        print("  python gpu_config_manager.py sparse        # Tắt dense conversion")
        return
    
    command = sys.argv[1].lower()
    
    if command == "show":
        show_current_config()
        
    elif command == "gpu":
        update_config(enable_gpu=True, force_dense=False)
        print("\n🚀 Đã bật GPU optimization!")
        print("   • Models sẽ sử dụng GPU acceleration")
        print("   • BoW và TF-IDF sẽ chuyển sang dense matrices")
        
    elif command == "memory":
        update_config(enable_gpu=False, force_dense=False)
        print("\n💾 Đã bật Memory optimization!")
        print("   • Models sẽ sử dụng CPU với sparse matrices")
        print("   • Tiết kiệm memory đáng kể")
        print("   • Phù hợp cho dataset lớn")
        
    elif command == "dense":
        update_config(enable_gpu=False, force_dense=True)
        print("\n🔄 Đã bật Dense conversion!")
        print("   • BoW và TF-IDF sẽ chuyển sang dense matrices")
        print("   • Không sử dụng GPU acceleration")
        print("   • Tốn memory nhưng có thể nhanh hơn cho một số models")
        
    elif command == "sparse":
        update_config(enable_gpu=False, force_dense=False)
        print("\n📊 Đã tắt Dense conversion!")
        print("   • BoW và TF-IDF sẽ giữ sparse matrices")
        print("   • Tiết kiệm memory tối đa")
        print("   • Phù hợp cho dataset rất lớn")
        
    else:
        print(f"❌ Lệnh không hợp lệ: {command}")
        print("Sử dụng: python gpu_config_manager.py show")

if __name__ == "__main__":
    main()
