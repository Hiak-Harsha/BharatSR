@echo off
title BharatSR — NTRO Satellite Super-Resolution
echo ================================================================
echo  BharatSR: Deep Learning Super-Resolution Mapping System
echo  National Technical Research Organisation (NTRO) - SIH26142
echo ================================================================
echo.

cd /d "%~dp0"

REM Check if Python venv exists
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .\venv
    echo Please install dependencies first:
    echo   python -m venv venv
    echo   .\venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Run unified launcher
.\venv\Scripts\python.exe run.py

pause
