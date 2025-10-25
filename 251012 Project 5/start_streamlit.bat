@echo off
REM Batch script to run Streamlit with UTF-8 encoding
REM Run this script from the project root directory

echo Starting Streamlit with UTF-8 encoding...
echo Project directory: %CD%
echo Application will be available at: http://localhost:8501
echo.

REM Set UTF-8 encoding for Python
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Change to project directory (in case script is run from elsewhere)
cd /d "%~dp0"

REM Check if wizard_ui/main.py exists
if not exist "wizard_ui\main.py" (
    echo ERROR: wizard_ui\main.py not found!
    echo Please run this script from the project root directory.
    pause
    exit /b 1
)

REM Run Streamlit
echo Starting Streamlit...
python -m streamlit run wizard_ui/main.py --server.port 8501

pause
