# Run CatVTON Backend using the CUDA-enabled venv from D:\TRY_ON_MODEL
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Starting AI Virtual Try-On Backend (NVIDIA GPU Enabled)" -ForegroundColor Green
Write-Host " Using venv: D:\TRY_ON_MODEL\venv" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

Set-Location $PSScriptRoot
& "D:\TRY_ON_MODEL\venv\Scripts\python.exe" app.py
