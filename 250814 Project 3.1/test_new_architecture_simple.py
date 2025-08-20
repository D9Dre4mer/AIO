"""
Test Đơn Giản Kiến Trúc Mới - Kiểm tra tính ổn định
"""

from sklearn.datasets import make_classification


def test_new_architecture():
    """Test kiến trúc mới một cách đơn giản"""
    print("🧪 Test Kiến Trúc Mới - Đơn Giản")
    print("=" * 40)
    
    try:
        # 1. Import và đăng ký models
        print("📦 Importing modules...")
        from models.register_models import register_all_models
        register_all_models()
        print("✅ Models registered")
        
        # 2. Tạo dữ liệu
        print("📊 Creating sample data...")
        X, y = make_classification(n_samples=100, n_features=5, 
                                 n_classes=2, random_state=42)
        print(f"✅ Data: {X.shape}, {y.shape}")
        
        # 3. Test validation manager
        print("✂️ Testing data splitting...")
        from models.utils.validation_manager import validation_manager
        
        X_train, X_val, X_test, y_train, y_val, y_test = \
            validation_manager.split_data(X, y)
        
        print(f"✅ Split: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
        
        # 4. Test single model training
        print("🚀 Testing single model training...")
        from models.new_model_trainer import NewModelTrainer
        
        trainer = NewModelTrainer(cv_folds=3)
        result = trainer.train_validate_test_model('knn', X, y)
        
        print(f"✅ Training completed: {len(result)} values returned")
        
        # 5. Test cross-validation
        print("🔄 Testing cross-validation...")
        cv_result = trainer.cross_validate_model('knn', X, y, ['accuracy'])
        
        print(f"✅ CV completed: {len(cv_result['fold_results'])} folds")
        
        # 6. Test all models
        print("🚀 Testing all models...")
        results = trainer.train_validate_test_all_models(X, y)
        
        print(f"✅ All models completed: {len(results)} models")
        
        print("\n🎉 Tất cả tests PASS! Kiến trúc mới hoạt động ổn định.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_new_architecture()
