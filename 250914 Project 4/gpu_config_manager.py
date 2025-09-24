#!/usr/bin/env python3
"""
GPU Configuration Manager
Quản lý cấu hình GPU optimization cho Topic Modeling Auto Classifier
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_gpu_availability() -> Tuple[bool, str]:
    """
    Detect GPU availability and return status with details
    
    Returns:
        Tuple[bool, str]: (is_gpu_available, device_info)
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return True, f"GPU: {gpu_name} ({gpu_count} devices, {gpu_memory:.1f}GB)"
        else:
            return False, "No CUDA GPU detected"
    except ImportError:
        return False, "PyTorch not available"
    except Exception as e:
        return False, f"GPU detection error: {str(e)}"

def get_device_policy_config() -> Dict[str, Any]:
    """
    Get device policy configuration from config.py
    
    Returns:
        Dict containing device policy settings
    """
    try:
        from config import DEVICE_POLICY, N_JOBS, SEED
        return {
            "policy": DEVICE_POLICY,
            "n_jobs": N_JOBS,
            "seed": SEED
        }
    except ImportError as e:
        logger.warning(f"Could not import config: {e}")
        return {
            "policy": "gpu_first",
            "n_jobs": -1,
            "seed": 42
        }

def configure_model_device(model_name: str, device_policy: str = "gpu_first") -> Dict[str, Any]:
    """
    Configure device settings for specific ML models
    
    Args:
        model_name: Name of the model (xgboost, lightgbm, catboost, etc.)
        device_policy: Device policy ("gpu_first" or "cpu_only")
    
    Returns:
        Dict containing model-specific device configuration
    """
    is_gpu_available, gpu_info = detect_gpu_availability()
    
    config = {
        "model_name": model_name,
        "device_policy": device_policy,
        "gpu_available": is_gpu_available,
        "gpu_info": gpu_info,
        "use_gpu": False,
        "device_params": {}
    }
    
    if device_policy == "cpu_only":
        config["use_gpu"] = False
        logger.info(f"CPU-only mode for {model_name}")
    elif device_policy == "gpu_first":
        if is_gpu_available:
            config["use_gpu"] = True
            logger.info(f"GPU mode enabled for {model_name}: {gpu_info}")
        else:
            config["use_gpu"] = False
            logger.warning(f"GPU not available, falling back to CPU for {model_name}")
    
    # Model-specific GPU configuration
    if config["use_gpu"]:
        if model_name == "xgboost":
            config["device_params"] = {
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor"
            }
        elif model_name == "lightgbm":
            config["device_params"] = {
                "device_type": "gpu"
            }
        elif model_name == "catboost":
            config["device_params"] = {
                "task_type": "GPU"
            }
        else:
            # Sklearn models (RF, AdaBoost, GradientBoosting) don't support GPU
            config["use_gpu"] = False
            config["device_params"] = {}
            logger.info(f"{model_name} doesn't support GPU, using CPU")
    
    return config

def log_device_usage(model_name: str, config: Dict[str, Any]):
    """
    Log device usage information
    
    Args:
        model_name: Name of the model
        config: Device configuration dict
    """
    device_type = "GPU" if config["use_gpu"] else "CPU"
    logger.info(f"Training {model_name} on {device_type}")
    
    if config["use_gpu"]:
        logger.info(f"GPU Info: {config['gpu_info']}")
        if config["device_params"]:
            logger.info(f"GPU Parameters: {config['device_params']}")
    else:
        logger.info("Using CPU with multithreading optimization")

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
        print("  python gpu_config_manager.py detect        # Kiểm tra GPU availability")
        print("  python gpu_config_manager.py gpu           # Bật GPU optimization")
        print("  python gpu_config_manager.py memory        # Bật memory optimization")
        print("  python gpu_config_manager.py dense         # Bật dense conversion")
        print("  python gpu_config_manager.py sparse        # Tắt dense conversion")
        print("  python gpu_config_manager.py test <model>  # Test device config cho model")
        return
    
    command = sys.argv[1].lower()
    
    if command == "show":
        show_current_config()
        
    elif command == "detect":
        is_gpu_available, gpu_info = detect_gpu_availability()
        print(f"🔍 GPU Detection Results:")
        print(f"   • Available: {'✅ Yes' if is_gpu_available else '❌ No'}")
        print(f"   • Info: {gpu_info}")
        
        # Show device policy config
        policy_config = get_device_policy_config()
        print(f"\n⚙️ Device Policy Configuration:")
        print(f"   • Policy: {policy_config['policy']}")
        print(f"   • N_JOBS: {policy_config['n_jobs']}")
        print(f"   • SEED: {policy_config['seed']}")
        
    elif command == "test":
        if len(sys.argv) < 3:
            print("❌ Vui lòng chỉ định model name")
            print("Ví dụ: python gpu_config_manager.py test xgboost")
            return
        
        model_name = sys.argv[2].lower()
        policy_config = get_device_policy_config()
        device_config = configure_model_device(model_name, policy_config["policy"])
        
        print(f"🧪 Testing device configuration for {model_name}:")
        print(f"   • Device Policy: {device_config['device_policy']}")
        print(f"   • GPU Available: {'✅ Yes' if device_config['gpu_available'] else '❌ No'}")
        print(f"   • Will Use GPU: {'✅ Yes' if device_config['use_gpu'] else '❌ No'}")
        print(f"   • GPU Info: {device_config['gpu_info']}")
        
        if device_config['device_params']:
            print(f"   • Device Parameters: {device_config['device_params']}")
        else:
            print(f"   • Device Parameters: None (CPU mode)")
        
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
