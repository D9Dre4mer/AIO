#!/usr/bin/env python3
"""
Migration Verification Script

This script verifies that all migration requirements have been successfully implemented.
"""

import os
import sys
import json
import subprocess
import requests
import time
from pathlib import Path
from typing import Dict, Any, List


class MigrationVerifier:
    """Verify migration implementation"""
    
    def __init__(self):
        self.results = {}
        self.project_root = Path.cwd()
    
    def verify_file_structure(self) -> Dict[str, Any]:
        """Verify required file structure exists"""
        print("🔍 Verifying file structure...")
        
        required_files = [
            "dvc.yaml",
            "params.yaml",
            "environment.yml",
            "Dockerfile",
            "Makefile",
            "README.md",
            "CHANGELOG.md",
            "src/mlflow_integration.py",
            "src/optuna_integration.py",
            "src/prefect_flows.py",
            "src/drift_monitoring.py",
            "src/monitoring_config.py",
            "src/cd_pipeline.py",
            "src/serve/app.py",
            "src/train/main.py",
            "src/data_validation.py",
            ".github/workflows/ci-cd.yml",
            "infra/docker-compose.dev.yml",
            "monitoring/grafana/dashboards/ml-monitoring-dashboard.json",
            "monitoring/prometheus/prometheus.yml"
        ]
        
        required_dirs = [
            "src/ingest",
            "src/fe", 
            "src/train",
            "src/eval",
            "src/serve",
            "tests",
            "monitoring",
            "infra"
        ]
        
        missing_files = []
        missing_dirs = []
        
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        for dir_path in required_dirs:
            if not (self.project_root / dir_path).is_dir():
                missing_dirs.append(dir_path)
        
        success = len(missing_files) == 0 and len(missing_dirs) == 0
        
        return {
            "status": "success" if success else "failed",
            "missing_files": missing_files,
            "missing_dirs": missing_dirs,
            "total_required": len(required_files) + len(required_dirs),
            "found": len(required_files) + len(required_dirs) - len(missing_files) - len(missing_dirs)
        }
    
    def verify_dependencies(self) -> Dict[str, Any]:
        """Verify required dependencies are installed"""
        print("🔍 Verifying dependencies...")
        
        required_packages = [
            "mlflow",
            "dvc",
            "fastapi",
            "uvicorn",
            "pydantic",
            "psycopg2",
            "boto3",
            "prefect",
            "great_expectations",
            "pandera",
            "evidently",
            "prometheus_client",
            "optuna",
            "pytest",
            "flake8",
            "black",
            "isort"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        success = len(missing_packages) == 0
        
        return {
            "status": "success" if success else "failed",
            "missing_packages": missing_packages,
            "total_required": len(required_packages),
            "found": len(required_packages) - len(missing_packages)
        }
    
    def verify_dvc_setup(self) -> Dict[str, Any]:
        """Verify DVC is properly initialized"""
        print("🔍 Verifying DVC setup...")
        
        try:
            # Check if DVC is initialized
            result = subprocess.run(["dvc", "status"], capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "message": "DVC properly initialized"
                }
            else:
                return {
                    "status": "failed",
                    "error": result.stderr
                }
        except FileNotFoundError:
            return {
                "status": "failed",
                "error": "DVC not found in PATH"
            }
    
    def verify_git_setup(self) -> Dict[str, Any]:
        """Verify Git is properly initialized"""
        print("🔍 Verifying Git setup...")
        
        try:
            # Check if Git is initialized
            result = subprocess.run(["git", "status"], capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "message": "Git properly initialized"
                }
            else:
                return {
                    "status": "failed",
                    "error": result.stderr
                }
        except FileNotFoundError:
            return {
                "status": "failed",
                "error": "Git not found in PATH"
            }
    
    def verify_config_files(self) -> Dict[str, Any]:
        """Verify configuration files are valid"""
        print("🔍 Verifying configuration files...")
        
        config_files = {
            "dvc.yaml": "yaml",
            "params.yaml": "yaml",
            "environment.yml": "yaml"
        }
        
        results = {}
        
        for file_path, file_type in config_files.items():
            try:
                with open(file_path, 'r') as f:
                    if file_type == "yaml":
                        import yaml
                        yaml.safe_load(f)
                results[file_path] = {"status": "success"}
            except Exception as e:
                results[file_path] = {"status": "failed", "error": str(e)}
        
        success = all(r["status"] == "success" for r in results.values())
        
        return {
            "status": "success" if success else "failed",
            "file_results": results
        }
    
    def verify_fastapi_app(self) -> Dict[str, Any]:
        """Verify FastAPI app can be imported"""
        print("🔍 Verifying FastAPI app...")
        
        try:
            # Add src to path
            sys.path.insert(0, str(self.project_root / "src"))
            
            # Try to import the app
            from serve.app import app
            
            return {
                "status": "success",
                "message": "FastAPI app imported successfully"
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def verify_mlflow_integration(self) -> Dict[str, Any]:
        """Verify MLflow integration can be imported"""
        print("🔍 Verifying MLflow integration...")
        
        try:
            # Add src to path
            sys.path.insert(0, str(self.project_root / "src"))
            
            # Try to import MLflow integration
            from mlflow_integration import MLflowTracker, ModelRegistry
            
            return {
                "status": "success",
                "message": "MLflow integration imported successfully"
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def verify_tests(self) -> Dict[str, Any]:
        """Verify tests can be run"""
        print("🔍 Verifying tests...")
        
        try:
            # Run a simple test to check if pytest works
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "--collect-only", "-q"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "message": "Tests can be collected successfully"
                }
            else:
                return {
                    "status": "failed",
                    "error": result.stderr
                }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def verify_monitoring_config(self) -> Dict[str, Any]:
        """Verify monitoring configurations exist"""
        print("🔍 Verifying monitoring configurations...")
        
        monitoring_files = [
            "monitoring/grafana/dashboards/ml-monitoring-dashboard.json",
            "monitoring/prometheus/prometheus.yml",
            "monitoring/prometheus/rules/ml-alerts.yml",
            "monitoring/grafana/datasources/prometheus.json",
            "monitoring/docker-compose.monitoring.yml"
        ]
        
        missing_files = []
        
        for file_path in monitoring_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        success = len(missing_files) == 0
        
        return {
            "status": "success" if success else "failed",
            "missing_files": missing_files,
            "total_required": len(monitoring_files),
            "found": len(monitoring_files) - len(missing_files)
        }
    
    def run_all_verifications(self) -> Dict[str, Any]:
        """Run all verification checks"""
        print("🚀 Starting migration verification...")
        print("=" * 60)
        
        verifications = [
            ("File Structure", self.verify_file_structure),
            ("Dependencies", self.verify_dependencies),
            ("DVC Setup", self.verify_dvc_setup),
            ("Git Setup", self.verify_git_setup),
            ("Config Files", self.verify_config_files),
            ("FastAPI App", self.verify_fastapi_app),
            ("MLflow Integration", self.verify_mlflow_integration),
            ("Tests", self.verify_tests),
            ("Monitoring Config", self.verify_monitoring_config)
        ]
        
        results = {}
        passed = 0
        failed = 0
        
        for name, verification_func in verifications:
            try:
                result = verification_func()
                results[name] = result
                
                if result["status"] == "success":
                    print(f"✅ {name}: PASSED")
                    passed += 1
                else:
                    print(f"❌ {name}: FAILED")
                    if "error" in result:
                        print(f"   Error: {result['error']}")
                    failed += 1
                    
            except Exception as e:
                results[name] = {"status": "failed", "error": str(e)}
                print(f"❌ {name}: FAILED - {e}")
                failed += 1
        
        print("=" * 60)
        print(f"📊 Verification Summary:")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
        
        overall_success = failed == 0
        
        if overall_success:
            print("🎉 Migration verification PASSED!")
            print("   All requirements have been successfully implemented.")
        else:
            print("⚠️  Migration verification FAILED!")
            print("   Some requirements need attention.")
        
        return {
            "overall_status": "success" if overall_success else "failed",
            "passed": passed,
            "failed": failed,
            "success_rate": passed/(passed+failed)*100,
            "results": results
        }
    
    def save_verification_report(self, results: Dict[str, Any]):
        """Save verification report to file"""
        report_path = "artifacts/migration_verification_report.json"
        
        # Create artifacts directory if it doesn't exist
        os.makedirs("artifacts", exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Verification report saved to: {report_path}")


def main():
    """Main verification function"""
    verifier = MigrationVerifier()
    
    # Run all verifications
    results = verifier.run_all_verifications()
    
    # Save report
    verifier.save_verification_report(results)
    
    # Exit with appropriate code
    if results["overall_status"] == "success":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
