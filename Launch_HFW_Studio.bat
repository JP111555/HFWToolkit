@echo off
title HFWToolkit — Material, Rig & Shader Studio
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [HFWToolkit] First-time launch detected — creating local Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [!] Could not find Python 3.10+ on PATH. Please install Python 3.10+ from python.org.
        pause
        exit /b 1
    )
    echo [HFWToolkit] Installing dependencies...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e .
)

echo.
echo ========================================================================
echo  Starting HFWToolkit Studio (http://127.0.0.1:8420)
echo  Keep this window open while using the Web Studio. Press Ctrl+C to stop.
echo ========================================================================
echo.
".venv\Scripts\python.exe" -m hfw_toolkit.cli web %*
pause
