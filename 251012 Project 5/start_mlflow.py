#!/usr/bin/env python3
"""
MLflow Server Startup Script
Automatically starts MLflow server with proper configuration
"""

import subprocess
import sys
import os
import time
import requests
from pathlib import Path

def check_mlflow_server():
    """Check if MLflow server is already running"""
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_mlflow_server():
    """Start MLflow server with proper configuration"""
    print("Starting MLflow Server...")
    
    # Check if already running
    if check_mlflow_server():
        print("MLflow server is already running on http://localhost:5000")
        return True
    
    # Create mlruns directory if it doesn't exist
    mlruns_dir = Path("./mlruns")
    mlruns_dir.mkdir(exist_ok=True)
    
    # MLflow server command
    cmd = [
        "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", "5000",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlruns"
    ]
    
    try:
        print("Starting MLflow server with command:")
        print(" ".join(cmd))
        
        # Start server in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for server to start
        print("Waiting for server to start...")
        time.sleep(3)
        
        # Check if server is running
        if check_mlflow_server():
            print("MLflow server started successfully!")
            print("MLflow UI: http://localhost:5000")
            print("Tracking URI: http://localhost:5000")
            return True
        else:
            print("Failed to start MLflow server")
            return False
            
    except Exception as e:
        print(f"Error starting MLflow server: {e}")
        return False

def main():
    """Main function"""
    print("MLflow Server Startup Script")
    print("=" * 40)
    
    # Check if mlflow is installed
    try:
        import mlflow
        print(f"MLflow version: {mlflow.__version__}")
    except ImportError:
        print("MLflow not installed. Please install with: pip install mlflow")
        return False
    
    # Start server
    success = start_mlflow_server()
    
    if success:
        print("\nMLflow is ready!")
        print("You can now run your Streamlit app and MLflow will be connected.")
    else:
        print("\nFailed to start MLflow server")
        print("Please check the error messages above and try again.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
