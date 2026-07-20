param(
    [int]$Port = 8502
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Virtual environment missing. Run .\setup_windows.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Starting MedClaim Sentinel on http://localhost:$Port" -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m streamlit run app.py --server.port $Port
