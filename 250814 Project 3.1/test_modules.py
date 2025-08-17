"""
Test script for Topic Modeling Project
Verifies that all modules can be imported and basic functionality works
"""

def test_imports():
    """Test that all modules can be imported successfully"""
    try:
        print("Testing imports...")
        
        # Test config
        from config import CACHE_DIR, CATEGORIES_TO_SELECT
        print("✅ config.py imported successfully")
        
        # Test data_loader
        from data_loader import DataLoader
        print("✅ data_loader.py imported successfully")
        
        # Test text_encoders
        from text_encoders import TextVectorizer, EmbeddingVectorizer
        print("✅ text_encoders.py imported successfully")
        
        # Test models
        from models import ModelTrainer, get_model_descriptions
        print("✅ models.py imported successfully")
        
        # Test visualization
        from visualization import plot_confusion_matrix, create_output_directories
        print("✅ visualization.py imported successfully")
        
        print("\n🎉 All modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of key components"""
    try:
        print("\nTesting basic functionality...")
        
        # Test config values
        from config import CACHE_DIR, CATEGORIES_TO_SELECT
        assert CACHE_DIR == "./cache"
        assert len(CATEGORIES_TO_SELECT) == 5
        print("✅ Config values are correct")
        
        # Test DataLoader initialization
        from data_loader import DataLoader
        dl = DataLoader()
        assert dl.cache_dir == "./cache"
        print("✅ DataLoader initialized correctly")
        
        # Test TextVectorizer initialization
        from text_encoders import TextVectorizer
        tv = TextVectorizer()
        print("✅ TextVectorizer initialized correctly")
        
        # Test ModelTrainer initialization
        from models import ModelTrainer
        mt = ModelTrainer()
        print("✅ ModelTrainer initialized correctly")
        
        # Test model descriptions
        from models import get_model_descriptions
        descriptions = get_model_descriptions()
        assert 'kmeans' in descriptions
        assert 'knn' in descriptions
        print("✅ Model descriptions loaded correctly")
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 Testing Topic Modeling Project Modules")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n🎉 All tests passed! The project is ready to run.")
        else:
            print("\n❌ Some functionality tests failed.")
    else:
        print("\n❌ Import tests failed. Please check dependencies.")


if __name__ == "__main__":
    main()
