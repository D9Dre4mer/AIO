@echo off
echo ========================================
echo    Topic Modeling Project Runner
echo ========================================
echo.

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Testing modules...
python test_modules.py

echo.
echo Running the complete project...
python main.py

echo.
echo Project execution completed!
pause
