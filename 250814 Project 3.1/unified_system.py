"""
Unified System - Merge kiến trúc cũ và mới để hoạt động đồng nhất
"""

from typing import Dict, Any, List

# Import kiến trúc mới
from models.new_model_trainer import NewModelTrainer

# Import kiến trúc cũ (legacy) - từ file gốc
from models import ModelTrainer as LegacyModelTrainer


class UnifiedSystem:
    """Hệ thống thống nhất kết hợp kiến trúc cũ và mới"""
    
    def __init__(self, use_new_architecture: bool = True, 
                 test_size: float = 0.2, validation_size: float = 0.2,
                 cv_folds: int = 5, cv_stratified: bool = True):
        """Khởi tạo hệ thống thống nhất
        
        Args:
            use_new_architecture: Sử dụng kiến trúc mới (True) hay cũ (False)
            test_size: Tỷ lệ dữ liệu test
            validation_size: Tỷ lệ dữ liệu validation
            cv_folds: Số fold cho cross-validation
            cv_stratified: Có sử dụng stratified sampling không
        """
        self.use_new_architecture = use_new_architecture
        self.test_size = test_size
        self.validation_size = validation_size
        self.cv_folds = cv_folds
        self.cv_stratified = cv_stratified
        
        # Khởi tạo trainer dựa trên kiến trúc được chọn
        if use_new_architecture:
            self.trainer = NewModelTrainer(
                test_size=test_size,
                validation_size=validation_size,
                cv_folds=cv_folds,
                cv_stratified=cv_stratified
            )
            print("🚀 Sử dụng kiến trúc mới (modular)")
        else:
            self.trainer = LegacyModelTrainer()
            print("🔧 Sử dụng kiến trúc cũ (legacy)")
        
        # Đăng ký models
        self._register_models()
    
    def _register_models(self):
        """Đăng ký tất cả models"""
        try:
            from models.register_models import register_all_models
            register_all_models()
            print("✅ Models đã được đăng ký")
        except Exception as e:
            print(f"⚠️ Không thể đăng ký models: {e}")
    
    def list_available_models(self) -> List[str]:
        """Liệt kê tất cả models có sẵn"""
        try:
            if self.use_new_architecture:
                return self.trainer.list_available_models()
            else:
                # Legacy system không có method này
                return ['kmeans', 'knn', 'decision_tree', 'naive_bayes']
        except Exception as e:
            print(f"⚠️ Lỗi khi liệt kê models: {e}")
            return []
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Lấy thông tin về model cụ thể"""
        try:
            if self.use_new_architecture:
                return self.trainer.get_model_info(model_name)
            else:
                # Legacy system không có method này
                return {'name': model_name, 'type': 'legacy'}
        except Exception as e:
            print(f"⚠️ Lỗi khi lấy thông tin model: {e}")
            return {}
    
    def train_and_test_model(self, model_name: str, X, y, **kwargs):
        """Train và test model (interface thống nhất)"""
        try:
            if self.use_new_architecture:
                # Sử dụng kiến trúc mới với validation
                return self.trainer.train_validate_test_model(model_name, X, y, **kwargs)
            else:
                # Sử dụng kiến trúc cũ
                return self.trainer.train_and_test_model(model_name, X, y, **kwargs)
        except Exception as e:
            print(f"❌ Lỗi khi train/test model {model_name}: {e}")
            return None
    
    def train_and_test_all_models(self, X, y):
        """Train và test tất cả models (interface thống nhất)"""
        try:
            if self.use_new_architecture:
                # Sử dụng kiến trúc mới với validation
                return self.trainer.train_validate_test_all_models(X, y)
            else:
                # Sử dụng kiến trúc cũ
                return self.trainer.train_and_test_all_models(X, y)
        except Exception as e:
            print(f"❌ Lỗi khi train/test tất cả models: {e}")
            return {}
    
    def cross_validate_model(self, model_name: str, X, y, metrics: List[str] = None, **kwargs):
        """Cross-validate model (chỉ có trong kiến trúc mới)"""
        if not self.use_new_architecture:
            print("⚠️ Cross-validation chỉ có trong kiến trúc mới")
            return None
        
        try:
            return self.trainer.cross_validate_model(model_name, X, y, metrics, **kwargs)
        except Exception as e:
            print(f"❌ Lỗi khi cross-validate model {model_name}: {e}")
            return None
    
    def cross_validate_all_models(self, X, y, metrics: List[str] = None):
        """Cross-validate tất cả models (chỉ có trong kiến trúc mới)"""
        if not self.use_new_architecture:
            print("⚠️ Cross-validation chỉ có trong kiến trúc mới")
            return None
        
        try:
            return self.trainer.cross_validate_all_models(X, y, metrics)
        except Exception as e:
            print(f"❌ Lỗi khi cross-validate tất cả models: {e}")
            return None
    
    def get_model_comparison(self, results):
        """So sánh performance của models (interface thống nhất)"""
        try:
            if self.use_new_architecture:
                return self.trainer.get_model_comparison_with_validation(results)
            else:
                return self.trainer.get_model_comparison(results)
        except Exception as e:
            print(f"⚠️ Lỗi khi so sánh models: {e}")
            return {}
    
    def print_summary(self, results):
        """In summary (interface thống nhất)"""
        try:
            if self.use_new_architecture:
                self.trainer.print_validation_summary(results)
            else:
                # Legacy system không có method này
                print("📊 Results Summary:")
                for model_name, result in results.items():
                    if 'accuracy' in result:
                        print(f"  {model_name}: {result['accuracy']:.4f}")
        except Exception as e:
            print(f"⚠️ Lỗi khi in summary: {e}")
    
    def switch_architecture(self, use_new: bool):
        """Chuyển đổi giữa kiến trúc cũ và mới"""
        self.use_new_architecture = use_new
        
        if use_new:
            self.trainer = NewModelTrainer(
                test_size=self.test_size,
                validation_size=self.validation_size,
                cv_folds=self.cv_folds,
                cv_stratified=self.cv_stratified
            )
            print("🔄 Đã chuyển sang kiến trúc mới")
        else:
            self.trainer = LegacyModelTrainer()
            print("🔄 Đã chuyển sang kiến trúc cũ")
        
        self._register_models()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Lấy trạng thái hệ thống"""
        return {
            'architecture': 'new' if self.use_new_architecture else 'legacy',
            'test_size': self.test_size,
            'validation_size': self.validation_size,
            'cv_folds': self.cv_folds,
            'cv_stratified': self.cv_stratified,
            'available_models': self.list_available_models(),
            'trainer_type': type(self.trainer).__name__
        }


# Global instance
unified_system = UnifiedSystem()


def demo_unified_system():
    """Demo hệ thống thống nhất"""
    print("🎯 DEMO HỆ THỐNG THỐNG NHẤT")
    print("=" * 50)
    
    # Tạo dữ liệu mẫu
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    print(f"📊 Data: {X.shape}, {y.shape}")
    
    # Test kiến trúc mới
    print("\n🚀 Testing Kiến Trúc Mới:")
    unified_system.switch_architecture(True)
    
    # Test single model
    print("\n📈 Test Single Model (KNN):")
    result = unified_system.train_and_test_model('knn', X, y)
    if result:
        print(f"✅ Kết quả: {len(result)} values returned")
    
    # Test cross-validation
    print("\n🔄 Test Cross-Validation:")
    cv_result = unified_system.cross_validate_model('knn', X, y, ['accuracy'])
    if cv_result:
        print(f"✅ CV completed: {len(cv_result['fold_results'])} folds")
    
    # Test kiến trúc cũ
    print("\n🔧 Testing Kiến Trúc Cũ:")
    unified_system.switch_architecture(False)
    
    # Test single model
    print("\n📈 Test Single Model (KNN) - Legacy:")
    result = unified_system.train_and_test_model('knn', X, y)
    if result:
        print(f"✅ Kết quả: {len(result)} values returned")
    
    # Test all models
    print("\n🚀 Test All Models - Legacy:")
    results = unified_system.train_and_test_all_models(X, y)
    if results:
        print(f"✅ Completed: {len(results)} models")
        unified_system.print_summary(results)
    
    print("\n🎉 Demo hoàn thành!")


if __name__ == "__main__":
    demo_unified_system()
