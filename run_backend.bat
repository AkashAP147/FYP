@echo off
title AI Virtual Try-On Backend (GPU)
echo ==========================================================
echo  Starting AI Virtual Try-On Backend (NVIDIA GPU Enabled)
echo  Using venv: D:\TRY_ON_MODEL\venv
echo ==========================================================
cd /d "%~dp0"
"D:\TRY_ON_MODEL\venv\Scripts\python.exe" app.py
pause
