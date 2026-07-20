param(
    [switch]$SkipLocalAI,
    [string]$LocalAIModel = "qwen2.5vl:7b"
)

$ErrorActionPreference = "Stop"
Write-Host "=== MedClaim Sentinel v6 setup ===" -ForegroundColor Cyan

$pythonCommand = $null
$pythonArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.12 --version *> $null
    if ($LASTEXITCODE -eq 0) {
        $pythonCommand = "py"
        $pythonArgs = @("-3.12")
    } else {
        & py -3.11 --version *> $null
        if ($LASTEXITCODE -eq 0) {
            $pythonCommand = "py"
            $pythonArgs = @("-3.11")
        }
    }
}
if (-not $pythonCommand -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $pythonCommand = "python"
}
if (-not $pythonCommand) {
    Write-Host "Python 3.11/3.12 was not found." -ForegroundColor Red
    Write-Host "Install it with: winget install Python.Python.3.12"
    exit 1
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $pythonCommand @pythonArgs -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

$tessDefault = "C:\Program Files\Tesseract-OCR\tesseract.exe"
$tess = Get-Command tesseract -ErrorAction SilentlyContinue
if (-not $tess -and -not (Test-Path $tessDefault)) {
    Write-Host "Tesseract not found. Installing through winget..." -ForegroundColor Yellow
    try {
        winget install --id UB-Mannheim.TesseractOCR --exact --accept-package-agreements --accept-source-agreements
    } catch {
        Write-Host "Tesseract installation failed. Basic local vision extraction can still work, but conventional OCR will be limited." -ForegroundColor Yellow
    }
}

if (-not $SkipLocalAI) {
    & .\install_local_ai.ps1 -Model $LocalAIModel
}

Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run: .\run_windows.ps1"
