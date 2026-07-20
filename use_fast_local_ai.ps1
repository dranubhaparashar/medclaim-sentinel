param(
    [string]$Model = "qwen2.5vl:3b"
)

$ErrorActionPreference = "Stop"
Write-Host "Switching MedClaim Sentinel to the faster local vision model: $Model" -ForegroundColor Cyan

$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    throw "Ollama is not available in PATH. Start/install Ollama first."
}

ollama pull $Model
if ($LASTEXITCODE -ne 0) {
    throw "Could not download $Model."
}

$configPath = Join-Path $PSScriptRoot "local_ai_config.json"
if (Test-Path $configPath) {
    $config = Get-Content $configPath -Raw | ConvertFrom-Json
} else {
    $config = [PSCustomObject]@{}
}

$config | Add-Member -NotePropertyName enabled -NotePropertyValue $true -Force
$config | Add-Member -NotePropertyName ollama_url -NotePropertyValue "http://127.0.0.1:11434" -Force
$config | Add-Member -NotePropertyName vision_model -NotePropertyValue $Model -Force
$config | Add-Member -NotePropertyName clinical_model -NotePropertyValue $Model -Force
$config | Add-Member -NotePropertyName timeout_seconds -NotePropertyValue 150 -Force
$config | Add-Member -NotePropertyName max_image_dimension -NotePropertyValue 1280 -Force
$config | Add-Member -NotePropertyName minimum_evidence_confidence -NotePropertyValue 0.55 -Force
$config | Add-Member -NotePropertyName num_ctx -NotePropertyValue 8192 -Force
$config | Add-Member -NotePropertyName num_predict -NotePropertyValue 1800 -Force
$config | Add-Member -NotePropertyName clinical_attach_images -NotePropertyValue $false -Force
$config | Add-Member -NotePropertyName skip_existing_document_ai -NotePropertyValue $true -Force

$config | ConvertTo-Json -Depth 10 | Set-Content $configPath -Encoding UTF8
Write-Host "Done. Restart Streamlit, then use Resume local AI." -ForegroundColor Green
Write-Host "Run 'ollama ps' during processing to confirm whether GPU is being used." -ForegroundColor Yellow
