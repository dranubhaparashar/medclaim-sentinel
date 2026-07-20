$ErrorActionPreference = "Continue"
Write-Host "=== Ollama version ===" -ForegroundColor Cyan
ollama --version
Write-Host "`n=== Installed models ===" -ForegroundColor Cyan
ollama list
Write-Host "`n=== Currently loaded model / processor ===" -ForegroundColor Cyan
ollama ps
Write-Host "`nPROCESSOR should show GPU or a GPU/CPU split. 100% CPU will be much slower." -ForegroundColor Yellow
