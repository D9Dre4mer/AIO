# PowerShell script to run Streamlit Topic Modeling App
Write-Host "Starting Streamlit Topic Modeling App..." -ForegroundColor Green
Write-Host ""

Write-Host "Installing requirements..." -ForegroundColor Yellow
pip install -r requirements_streamlit.txt

Write-Host ""
Write-Host "Starting Streamlit app..." -ForegroundColor Yellow
streamlit run streamlit_app.py

Read-Host "Press Enter to exit"
