#!/usr/bin/env python3
"""
MLflow Integration for Current Project
Configures MLflow to work with the existing project structure
"""

import os
import sys
import mlflow
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def setup_mlflow_for_project():
    """Setup MLflow for the current project"""
    print("Setting up MLflow for current project...")
    
    # MLflow server configuration
    MLFLOW_URL = "http://localhost:5000"
    
    # Test connection
    try:
        response = requests.get(f"{MLFLOW_URL}/health", timeout=10)
        if response.status_code != 200:
            print(f"❌ MLflow server not responding: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to MLflow server: {e}")
        print("💡 Please start MLflow server first:")
        print("   cd mlflow-server && make start")
        return False
    
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_URL)
    print(f"✅ MLflow tracking URI set to: {MLFLOW_URL}")
    
    # Create project-specific experiment
    experiment_name = "enhanced_ml_pipeline"
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Check if experiment exists
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(experiment_name)
            print(f"✅ Created experiment: {experiment_name} (ID: {experiment_id})")
        else:
            print(f"✅ Found existing experiment: {experiment_name}")
        
        # Set as active experiment
        mlflow.set_experiment(experiment_name)
        
    except Exception as e:
        print(f"⚠️ Could not setup experiment: {e}")
    
    # Update project configuration
    update_project_config(MLFLOW_URL, experiment_name)
    
    print("🎉 MLflow setup completed successfully!")
    return True

def update_project_config(mlflow_url, experiment_name):
    """Update project configuration files"""
    
    # Update src/mlflow_integration.py
    mlflow_integration_path = project_root / "src" / "mlflow_integration.py"
    
    if mlflow_integration_path.exists():
        print("📝 Updating mlflow_integration.py...")
        
        # Read current content
        with open(mlflow_integration_path, 'r') as f:
            content = f.read()
        
        # Update tracking URI if needed
        if "http://localhost:5000" not in content:
            content = content.replace(
                'tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")',
                'tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")'
            )
            
            with open(mlflow_integration_path, 'w') as f:
                f.write(content)
            
            print("✅ Updated mlflow_integration.py")
    
    # Create environment file
    env_file_path = project_root / ".env"
    env_content = f"""# MLflow Configuration
MLFLOW_TRACKING_URI={mlflow_url}
MLFLOW_EXPERIMENT_NAME={experiment_name}

# Optional: Authentication
# MLFLOW_AUTH_USERNAME=admin
# MLFLOW_AUTH_PASSWORD=password
"""
    
    with open(env_file_path, 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file with MLflow configuration")

def test_integration():
    """Test MLflow integration with project"""
    print("Testing MLflow integration...")
    
    try:
        # Test basic MLflow operations
        with mlflow.start_run(run_name="integration_test"):
            mlflow.log_param("test_param", "integration_test")
            mlflow.log_metric("test_metric", 0.95)
            mlflow.log_text("MLflow integration test successful", "test.txt")
            
        print("✅ MLflow integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ MLflow integration test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("MLflow Project Integration Setup")
    print("=" * 40)
    
    # Setup MLflow
    if not setup_mlflow_for_project():
        print("❌ Setup failed")
        return False
    
    # Test integration
    if not test_integration():
        print("❌ Integration test failed")
        return False
    
    print("\n🎯 Next steps:")
    print("1. MLflow server is running on http://localhost:5000")
    print("2. Your project is configured to use MLflow")
    print("3. Run your Streamlit app: streamlit run app.py")
    print("4. Check MLflow UI for logged experiments")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
