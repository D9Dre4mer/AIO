"""
Debug Toàn Bộ Hệ Thống - Kiểm tra tính ổn định và hoạt động
"""

import traceback
from sklearn.datasets import make_classification


def debug_imports():
    """Debug việc import các modules"""
    print("🔍 Debug Imports...")
    
    try:
        # Test import kiến trúc mới
        print("  📦 Testing New Architecture Imports...")
        from models.register_models import register_all_models
        print("    ✅ models.register_models")
        
        from models.new_model_trainer import NewModelTrainer
        print("    ✅ models.new_model_trainer")
        
        from models.utils.validation_manager import validation_manager
        print("    ✅ models.utils.validation_manager")
        
        from models.utils.model_factory import model_factory
        print("    ✅ models.utils.model_factory")
        
        from models.base.metrics import ModelMetrics
        print("    ✅ models.base.metrics")
        
        print("  ✅ New Architecture imports successful")
        
    except Exception as e:
        print(f"  ❌ New Architecture import error: {e}")
        traceback.print_exc()
        return False
    
    try:
        # Test import kiến trúc cũ
        print("  📦 Testing Legacy Architecture Imports...")
        from models import ModelTrainer
        print("    ✅ models.ModelTrainer")
        print("  ✅ Legacy Architecture imports successful")
        
    except Exception as e:
        print(f"  ❌ Legacy Architecture import error: {e}")
        traceback.print_exc()
        return False
    
    return True


def debug_model_registration():
    """Debug việc đăng ký models"""
    print("\n🔍 Debug Model Registration...")
    
    try:
        from models.register_models import register_all_models
        register_all_models()
        print("  ✅ Models registered successfully")
        
        from models.utils.model_factory import model_factory
        available_models = model_factory.get_available_models()
        print(f"  📊 Available models: {available_models}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model registration error: {e}")
        traceback.print_exc()
        return False


def debug_data_splitting():
    """Debug việc chia dữ liệu"""
    print("\n🔍 Debug Data Splitting...")
    
    try:
        # Tạo dữ liệu mẫu
        X, y = make_classification(n_samples=200, n_features=10, 
                                 n_classes=2, random_state=42)
        print(f"  📊 Data created: {X.shape}, {y.shape}")
        
        # Test validation manager
        from models.utils.validation_manager import validation_manager
        
        # Test split data
        X_train, X_val, X_test, y_train, y_val, y_test = \
            validation_manager.split_data(X, y)
        
        print(f"  ✂️ Split successful:")
        print(f"    Train: {X_train.shape}")
        print(f"    Val:   {X_val.shape}")
        print(f"    Test:  {X_test.shape}")
        
        # Test split info
        split_info = validation_manager.get_split_info(X, y)
        print(f"  📈 Split info: {split_info}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data splitting error: {e}")
        traceback.print_exc()
        return False


def debug_single_model():
    """Debug việc train single model"""
    print("\n🔍 Debug Single Model Training...")
    
    try:
        # Tạo dữ liệu
        X, y = make_classification(n_samples=100, n_features=5, 
                                 n_classes=2, random_state=42)
        
        # Test với kiến trúc mới
        from models.new_model_trainer import NewModelTrainer
        trainer = NewModelTrainer(cv_folds=3)
        
        # Test KNN model
        result = trainer.train_validate_test_model('knn', X, y)
        print(f"  ✅ KNN training successful: {len(result)} values returned")
        
        # Test cross-validation
        cv_result = trainer.cross_validate_model('knn', X, y, ['accuracy'])
        print(f"  ✅ KNN CV successful: {len(cv_result['fold_results'])} folds")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Single model error: {e}")
        traceback.print_exc()
        return False


def debug_all_models():
    """Debug việc train tất cả models"""
    print("\n🔍 Debug All Models Training...")
    
    try:
        # Tạo dữ liệu
        X, y = make_classification(n_samples=150, n_features=8, 
                                 n_classes=2, random_state=42)
        
        # Test với kiến trúc mới
        from models.new_model_trainer import NewModelTrainer
        trainer = NewModelTrainer(cv_folds=3)
        
        # Test tất cả models
        results = trainer.train_validate_test_all_models(X, y)
        print(f"  ✅ All models training successful: {len(results)} models")
        
        # Test cross-validation cho tất cả
        cv_results = trainer.cross_validate_all_models(X, y, ['accuracy'])
        print(f"  ✅ All models CV successful: {len(cv_results)} models")
        
        # Test comparison
        trainer.print_cv_comparison(cv_results)
        
        return True
        
    except Exception as e:
        print(f"  ❌ All models error: {e}")
        traceback.print_exc()
        return False


def debug_legacy_compatibility():
    """Debug tính tương thích với kiến trúc cũ"""
    print("\n🔍 Debug Legacy Compatibility...")
    
    try:
        # Tạo dữ liệu
        X, y = make_classification(n_samples=100, n_features=5, 
                                 n_classes=2, random_state=42)
        
        # Test với kiến trúc cũ
        from models import ModelTrainer
        legacy_trainer = ModelTrainer()
        
        # Test single model
        y_pred, accuracy, metrics = legacy_trainer.train_and_test_model('knn', X, y)
        print(f"  ✅ Legacy KNN successful: Acc={accuracy:.4f}")
        
        # Test all models
        results = legacy_trainer.train_and_test_all_models(X, y)
        print(f"  ✅ Legacy all models successful: {len(results)} models")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Legacy compatibility error: {e}")
        traceback.print_exc()
        return False


def debug_unified_system():
    """Debug hệ thống thống nhất"""
    print("\n🔍 Debug Unified System...")
    
    try:
        # Import unified system
        from unified_system import UnifiedSystem
        
        # Tạo dữ liệu
        X, y = make_classification(n_samples=100, n_features=5, 
                                 n_classes=2, random_state=42)
        
        # Test kiến trúc mới
        print("  🚀 Testing New Architecture...")
        system = UnifiedSystem(use_new_architecture=True, cv_folds=3)
        
        # Test single model
        result = system.train_and_test_model('knn', X, y)
        print(f"    ✅ Single model: {len(result)} values")
        
        # Test cross-validation
        cv_result = system.cross_validate_model('knn', X, y, ['accuracy'])
        print(f"    ✅ CV: {len(cv_result['fold_results'])} folds")
        
        # Test kiến trúc cũ
        print("  🔧 Testing Legacy Architecture...")
        system.switch_architecture(False)
        
        # Test single model
        result = system.train_and_test_model('knn', X, y)
        print(f"    ✅ Single model: {len(result)} values")
        
        # Test all models
        results = system.train_and_test_all_models(X, y)
        print(f"    ✅ All models: {len(results)} models")
        
        # Test system status
        status = system.get_system_status()
        print(f"  📊 System status: {status}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Unified system error: {e}")
        traceback.print_exc()
        return False


def main():
    """Main debug function"""
    print("🐛 DEBUG TOÀN BỘ HỆ THỐNG")
    print("=" * 50)
    
    tests = [
        ("Imports", debug_imports),
        ("Model Registration", debug_model_registration),
        ("Data Splitting", debug_data_splitting),
        ("Single Model", debug_single_model),
        ("All Models", debug_all_models),
        ("Legacy Compatibility", debug_legacy_compatibility),
        ("Unified System", debug_unified_system)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test {test_name} failed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DEBUG SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:20s}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Tất cả tests đều PASS! Hệ thống hoạt động ổn định.")
    else:
        print("⚠️ Một số tests FAIL. Cần kiểm tra và sửa lỗi.")
    
    return passed == total


if __name__ == "__main__":
    main()
