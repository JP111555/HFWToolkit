@echo off
title HFWToolkit — One-Click Setup & System Doctor
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [HFWToolkit] Creating Python virtual environment (.venv)...
    python -m venv .venv
    if errorlevel 1 (
        echo [!] Could not find Python 3.10+ on PATH. Please install Python 3.10+ from python.org.
        pause
        exit /b 1
    )
    echo [HFWToolkit] Installing HFWToolkit package and dependencies...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e .
)

echo.
".venv\Scripts\python.exe" -m hfw_toolkit.cli setup %*
echo.
".venv\Scripts\python.exe" -m hfw_toolkit.cli doctor
echo.
pause
