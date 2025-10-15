#!/usr/bin/env python3
"""
MLflow Client Configuration Script
Configures MLflow client to connect to the server
"""

import os
import mlflow
import requests
from datetime import datetime

def test_connection(mlflow_url):
    """Test connection to MLflow server"""
    try:
        response = requests.get(f"{mlflow_url}/health", timeout=10)
        if response.status_code == 200:
            print(f"✅ Successfully connected to MLflow server at {mlflow_url}")
            return True
        else:
            print(f"❌ Failed to connect to MLflow server: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to MLflow server: {e}")
        return False

def configure_mlflow_client(mlflow_url="http://localhost:5000"):
    """Configure MLflow client"""
    print("🔧 Configuring MLflow client...")
    
    # Test connection first
    if not test_connection(mlflow_url):
        print("❌ Cannot connect to MLflow server. Please check if server is running.")
        return False
    
    # Set tracking URI
    mlflow.set_tracking_uri(mlflow_url)
    print(f"📡 MLflow tracking URI set to: {mlflow_url}")
    
    # Test MLflow client
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        print(f"📊 Found {len(experiments)} experiments")
        
        # Create test experiment
        experiment_name = f"test_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            experiment_id = client.create_experiment(experiment_name)
            print(f"✅ Created test experiment: {experiment_name} (ID: {experiment_id})")
            
            # Test run creation
            with mlflow.start_run(experiment_name=experiment_name, run_name="test_run"):
                mlflow.log_param("test_param", "test_value")
                mlflow.log_metric("test_metric", 0.95)
                print("✅ Successfully created test run with params and metrics")
            
            # Clean up test experiment
            client.delete_experiment(experiment_id)
            print("🧹 Cleaned up test experiment")
            
        except Exception as e:
            print(f"⚠️ Could not create test experiment: {e}")
        
        print("🎉 MLflow client configuration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ MLflow client configuration failed: {e}")
        return False

def create_client_config():
    """Create client configuration file"""
    config_content = f"""# MLflow Client Configuration
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Optional: Set experiment name
# mlflow.set_experiment("your_experiment_name")

# Optional: Set tags
# mlflow.set_tag("environment", "production")
# mlflow.set_tag("team", "ml_team")
"""
    
    with open("mlflow_client_config.py", "w") as f:
        f.write(config_content)
    
    print("📄 Created mlflow_client_config.py")

if __name__ == "__main__":
    print("MLflow Client Configuration Script")
    print("=" * 40)
    
    # Get MLflow URL from environment or use default
    mlflow_url = os.getenv("MLFLOW_URL", "http://localhost:5000")
    
    # Configure client
    success = configure_mlflow_client(mlflow_url)
    
    if success:
        create_client_config()
        print("\n🎯 Next steps:")
        print("1. Import mlflow_client_config.py in your Python scripts")
        print("2. Start logging experiments and models")
        print("3. Access MLflow UI at http://localhost:5000")
    else:
        print("\n❌ Configuration failed. Please check MLflow server status.")
