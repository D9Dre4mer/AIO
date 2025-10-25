@echo off
REM Simple script to run Streamlit - can be run from anywhere
REM This script will find the project directory and run Streamlit

echo Starting AIO Classifier Streamlit App...
echo.

REM Find the project directory (where this script is located)
cd /d "%~dp0"

REM Set UTF-8 encoding
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Check if wizard_ui/main.py exists
if not exist "wizard_ui\main.py" (
    echo ERROR: wizard_ui\main.py not found!
    echo Current directory: %CD%
    echo Please make sure you're in the correct project directory.
    pause
    exit /b 1
)

echo Project directory: %CD%
echo Starting Streamlit on http://localhost:8501
echo.

REM Run Streamlit
python -m streamlit run wizard_ui/main.py --server.port 8501

pause
