@echo off
REM MLflow Server Startup Script for Windows

setlocal enabledelayedexpansion

REM Configuration
if not defined MLFLOW_HOST set MLFLOW_HOST=0.0.0.0
if not defined MLFLOW_PORT set MLFLOW_PORT=5000
if not defined MLFLOW_BACKEND_STORE_URI set MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
if not defined MLFLOW_ARTIFACT_ROOT set MLFLOW_ARTIFACT_ROOT=./mlruns
if not defined MLFLOW_WORKERS set MLFLOW_WORKERS=4
if not defined MLFLOW_TIMEOUT set MLFLOW_TIMEOUT=120

echo ========================================
echo     MLflow Server Startup Script
echo ========================================

REM Check if MLflow is installed
echo [INFO] Checking MLflow installation...
mlflow --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] MLflow is not installed!
    echo [INFO] Installing MLflow...
    pip install -r requirements.txt
) else (
    for /f "tokens=*" %%i in ('mlflow --version') do echo [INFO] MLflow is installed: %%i
)

REM Create necessary directories
echo [INFO] Creating necessary directories...
if not exist mlruns mkdir mlruns
if not exist logs mkdir logs
if not exist config mkdir config
echo [INFO] Directories created successfully

REM Check if port is available
echo [INFO] Checking if port %MLFLOW_PORT% is available...
netstat -an | findstr :%MLFLOW_PORT% >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Port %MLFLOW_PORT% is already in use!
    echo [INFO] Attempting to kill existing process...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%MLFLOW_PORT%') do (
        taskkill /PID %%a /F >nul 2>&1
    )
    timeout /t 2 >nul
)
echo [INFO] Port %MLFLOW_PORT% is available

REM Start MLflow server
echo [INFO] Starting MLflow server...
echo [INFO] Configuration:
echo   Host: %MLFLOW_HOST%
echo   Port: %MLFLOW_PORT%
echo   Backend Store: %MLFLOW_BACKEND_STORE_URI%
echo   Artifact Root: %MLFLOW_ARTIFACT_ROOT%
echo   Workers: %MLFLOW_WORKERS%
echo   Timeout: %MLFLOW_TIMEOUT%

REM Start server
mlflow server ^
    --host %MLFLOW_HOST% ^
    --port %MLFLOW_PORT% ^
    --backend-store-uri %MLFLOW_BACKEND_STORE_URI% ^
    --default-artifact-root %MLFLOW_ARTIFACT_ROOT% ^
    --workers %MLFLOW_WORKERS%

pause
