param(
    [string]$Model = "qwen2.5vl:7b"
)

$ErrorActionPreference = "Stop"
Write-Host "=== MedClaim Sentinel local AI setup ===" -ForegroundColor Cyan
Write-Host "This installs/uses Ollama locally. Claim images are not sent to a paid cloud API." -ForegroundColor White

function Find-Ollama {
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
        "$env:ProgramFiles\Ollama\ollama.exe",
        "$env:ProgramFiles\Ollama\ollama app.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

$ollama = Find-Ollama
if (-not $ollama) {
    Write-Host "Ollama was not found. Installing through winget..." -ForegroundColor Yellow
    winget install --id Ollama.Ollama --exact --accept-package-agreements --accept-source-agreements
    Start-Sleep -Seconds 4
    $ollama = Find-Ollama
}

if (-not $ollama) {
    Write-Host "Ollama installation was not detected. Restart PowerShell and run this script again." -ForegroundColor Red
    exit 1
}

# Start the local server only when it is not already answering.
$serverReady = $false
try {
    Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null
    $serverReady = $true
} catch {}

if (-not $serverReady) {
    Write-Host "Starting the Ollama local server..." -ForegroundColor Yellow
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
    foreach ($attempt in 1..20) {
        Start-Sleep -Seconds 1
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2 | Out-Null
            $serverReady = $true
            break
        } catch {}
    }
}

if (-not $serverReady) {
    Write-Host "Ollama did not start. Open the Ollama desktop application, then rerun this script." -ForegroundColor Red
    exit 1
}

Write-Host "Pulling local vision model $Model. The first download is several GB..." -ForegroundColor Yellow
& $ollama pull $Model
if ($LASTEXITCODE -ne 0) {
    Write-Host "Model download failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

$config = @{
    enabled = $true
    ollama_url = "http://127.0.0.1:11434"
    vision_model = $Model
    clinical_model = $Model
    timeout_seconds = 420
    max_image_dimension = 1800
    minimum_evidence_confidence = 0.55
}
$config | ConvertTo-Json | Set-Content -Path "local_ai_config.json" -Encoding UTF8

Write-Host "Local AI is ready: $Model" -ForegroundColor Green
Write-Host "Start the application with: .\run_windows.ps1"
