# Topic Modeling Project Runner (PowerShell)
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Topic Modeling Project Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "Running the complete project..." -ForegroundColor Yellow
python main.py

Write-Host ""
Write-Host "Project execution completed!" -ForegroundColor Green
Read-Host "Press Enter to continue"
