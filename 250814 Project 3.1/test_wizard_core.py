"""
Test Wizard Core Functionality

Simple test script to verify that the wizard core components work correctly.
Run this to test the basic wizard infrastructure.

Author: AI Assistant
Created: 2025-01-27
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_wizard_imports():
    """Test that all wizard components can be imported"""
    try:
        from wizard_ui import WizardManager, SessionManager, StepValidator, NavigationController
        print("✅ All wizard components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_wizard_manager():
    """Test WizardManager functionality"""
    try:
        from wizard_ui import WizardManager
        
        # Create wizard manager
        wizard = WizardManager(total_steps=7)
        print("✅ WizardManager created successfully")
        
        # Test basic functionality
        assert wizard.get_current_step() == 1
        assert wizard.total_steps == 7
        assert wizard.get_step_info(1) is not None
        
        print("✅ WizardManager basic functionality working")
        return True
        
    except Exception as e:
        print(f"❌ WizardManager test failed: {e}")
        return False

def test_session_manager():
    """Test SessionManager functionality"""
    try:
        from wizard_ui import SessionManager
        
        # Create session manager
        session = SessionManager()
        print("✅ SessionManager created successfully")
        
        # Test basic functionality
        assert session.get_step_data(1) == {}
        session.set_step_data(1, {'test': 'data'})
        assert session.get_step_data(1)['test'] == 'data'
        
        print("✅ SessionManager basic functionality working")
        return True
        
    except Exception as e:
        print(f"❌ SessionManager test failed: {e}")
        return False

def test_step_validator():
    """Test StepValidator functionality"""
    try:
        from wizard_ui import StepValidator
        
        # Create step validator
        validator = StepValidator()
        print("✅ StepValidator created successfully")
        
        # Test basic functionality
        validation_rules = validator.validation_rules
        assert len(validation_rules) == 7  # 7 steps
        
        print("✅ StepValidator basic functionality working")
        return True
        
    except Exception as e:
        print(f"❌ StepValidator test failed: {e}")
        return False

def test_navigation_controller():
    """Test NavigationController functionality"""
    try:
        from wizard_ui import NavigationController, WizardManager, SessionManager
        
        # Create dependencies
        wizard = WizardManager()
        session = SessionManager()
        
        # Create navigation controller
        nav = NavigationController(wizard, session)
        print("✅ NavigationController created successfully")
        
        # Test basic functionality
        nav_summary = nav.get_navigation_summary()
        assert 'current_step' in nav_summary
        assert 'total_steps' in nav_summary
        
        print("✅ NavigationController basic functionality working")
        return True
        
    except Exception as e:
        print(f"❌ NavigationController test failed: {e}")
        return False

def test_package_structure():
    """Test that package structure is correct"""
    try:
        # Check if all required directories exist
        required_dirs = [
            'wizard_ui',
            'wizard_ui/steps',
            'wizard_ui/components', 
            'wizard_ui/windows',
            'wizard_ui/responsive'
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                print(f"❌ Required directory missing: {dir_path}")
                return False
        
        # Check if all required files exist
        required_files = [
            'wizard_ui/__init__.py',
            'wizard_ui/core.py',
            'wizard_ui/session_manager.py',
            'wizard_ui/validation.py',
            'wizard_ui/navigation.py'
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"❌ Required file missing: {file_path}")
                return False
        
        print("✅ Package structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ Package structure test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Wizard Core Functionality")
    print("=" * 50)
    
    tests = [
        ("Package Structure", test_package_structure),
        ("Wizard Imports", test_wizard_imports),
        ("Wizard Manager", test_wizard_manager),
        ("Session Manager", test_session_manager),
        ("Step Validator", test_step_validator),
        ("Navigation Controller", test_navigation_controller)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Wizard core is ready for use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
