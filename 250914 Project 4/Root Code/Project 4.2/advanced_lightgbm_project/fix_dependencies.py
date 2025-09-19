"""
Script để khắc phục các vấn đề dependency và cài đặt lại packages
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Chạy command và hiển thị kết quả"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} thành công!")
            if result.stdout:
                print(f"Output: {result.stdout}")
        else:
            print(f"❌ {description} thất bại!")
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Lỗi khi {description}: {e}")
        return False

def check_package(package_name):
    """Kiểm tra package có cài đặt không"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    print("🚀 Khắc phục Dependencies cho Advanced LightGBM Project")
    print("=" * 60)
    
    # 1. Kiểm tra Python version
    print(f"\n🐍 Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Cần Python 3.8 trở lên!")
        return False
    
    # 2. Upgrade pip
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrade pip")
    
    # 3. Cài đặt lại NumPy với version tương thích
    print("\n📦 Cài đặt lại NumPy với version tương thích...")
    run_command(f"{sys.executable} -m pip uninstall numpy -y", "Uninstall NumPy")
    run_command(f"{sys.executable} -m pip install numpy==1.24.3", "Install NumPy 1.24.3")
    
    # 4. Cài đặt lại scikit-learn
    print("\n📦 Cài đặt lại scikit-learn...")
    run_command(f"{sys.executable} -m pip uninstall scikit-learn -y", "Uninstall scikit-learn")
    run_command(f"{sys.executable} -m pip install scikit-learn==1.3.0", "Install scikit-learn 1.3.0")
    
    # 5. Cài đặt các packages khác
    packages = [
        "pandas==2.0.3",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "plotly==5.15.0",
        "lightgbm==4.0.0",
        "optuna==3.0.0",
        "shap==0.42.0",
        "gdown==4.7.0",
        "tqdm==4.65.0",
        "scipy==1.10.1"
    ]
    
    for package in packages:
        run_command(f"{sys.executable} -m pip install {package}", f"Install {package}")
    
    # 6. Kiểm tra lại các packages
    print("\n🔍 Kiểm tra lại các packages...")
    
    test_packages = [
        'numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn', 
        'plotly', 'lightgbm', 'optuna', 'shap', 'gdown'
    ]
    
    all_good = True
    for package in test_packages:
        if check_package(package):
            print(f"✅ {package}")
        else:
            print(f"❌ {package}")
            all_good = False
    
    if all_good:
        print("\n🎉 Tất cả packages đã được cài đặt thành công!")
        
        # 7. Test import các modules chính
        print("\n🧪 Test import các modules chính...")
        try:
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            import lightgbm as lgb
            import optuna
            import shap
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            print("✅ Tất cả modules import thành công!")
            
            # Test basic functionality
            print("\n🧪 Test basic functionality...")
            X = np.random.randn(100, 5)
            y = np.random.randint(0, 2, 100)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            print(f"✅ Basic functionality test passed! Accuracy: {acc:.3f}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            all_good = False
    
    if all_good:
        print("\n🎉 Dependencies đã được khắc phục thành công!")
        print("\n📚 Bây giờ bạn có thể chạy:")
        print("   python demo.py")
        print("   python run_optimization.py --quick")
        print("   python main.py")
    else:
        print("\n❌ Vẫn còn lỗi. Hãy thử các giải pháp khác:")
        print("1. Tạo virtual environment mới")
        print("2. Sử dụng conda thay vì pip")
        print("3. Cài đặt từng package một cách thủ công")
    
    return all_good

if __name__ == "__main__":
    main()
