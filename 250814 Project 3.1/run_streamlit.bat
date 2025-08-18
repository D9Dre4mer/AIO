@echo off
echo ========================================
echo Starting Streamlit Topic Modeling App...
echo ========================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo.
echo Installing/Updating requirements...
pip install -r requirements_streamlit.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo Checking Streamlit installation...
streamlit --version
if %errorlevel% neq 0 (
    echo ERROR: Streamlit not found. Installing now...
    pip install streamlit
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install Streamlit
        pause
        exit /b 1
    )
)

echo.
echo Starting Streamlit app...
echo The app will open in your default web browser
echo Press Ctrl+C to stop the app
echo.
streamlit run streamlit_app.py

echo.
echo App stopped. Press any key to exit...
pause >nul
