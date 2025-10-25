# PowerShell script to run Streamlit with UTF-8 encoding
# Run this script from the project root directory

# Set UTF-8 encoding for Python
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "Starting Streamlit with UTF-8 encoding..." -ForegroundColor Green
Write-Host "Project directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host "Application will be available at: http://localhost:8501" -ForegroundColor Yellow
Write-Host ""

# Change to script directory (in case script is run from elsewhere)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if wizard_ui/main.py exists
if (-not (Test-Path "wizard_ui\main.py")) {
    Write-Host "ERROR: wizard_ui\main.py not found!" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Run Streamlit
Write-Host "Starting Streamlit..." -ForegroundColor Green
python -m streamlit run wizard_ui/main.py --server.port 8501
