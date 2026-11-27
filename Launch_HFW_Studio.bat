@echo off
setlocal EnableDelayedExpansion
title HFWToolkit - Material, Rig ^& Shader Studio
cd /d "%~dp0"

set "UPGRADE_PY=0"
set "DO_UPDATE=0"
set "PASSTHROUGH="
for %%A in (%*) do (
    if /I "%%~A"=="--upgrade-python" (
        set "UPGRADE_PY=1"
    ) else if /I "%%~A"=="--update" (
        set "DO_UPDATE=1"
    ) else (
        set "PASSTHROUGH=!PASSTHROUGH! %%A"
    )
)

call :FIND_NEWEST_PYTHON

if "%UPGRADE_PY%"=="1" (
    if exist ".venv" (
        echo [HFWToolkit] Rebuilding .venv using newest system Python !BEST_PY_VER! - !BEST_PY!
        rmdir /s /q ".venv"
    )
)

if not exist ".venv\Scripts\python.exe" (
    if not defined BEST_PY (
        echo [!] Could not find Python 3.10+ on this PC.
        echo     Please install Python 3.12+ from https://www.python.org/downloads/
        pause
        exit /b 1
    )
    echo [HFWToolkit] Creating virtual environment with newest system Python !BEST_PY_VER!:
    echo              !BEST_PY!
    "!BEST_PY!" -m venv .venv
    if errorlevel 1 (
        echo [!] Failed to create .venv using !BEST_PY!.
        pause
        exit /b 1
    )
    echo [HFWToolkit] Installing dependencies...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e .
) else (
    for /f "delims=" %%V in ('".venv\Scripts\python.exe" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')" 2^>nul') do set "VENV_PY_VER=%%V"
    if defined BEST_PY_VER if defined VENV_PY_VER (
        ".venv\Scripts\python.exe" -c "import sys; b=tuple(map(int, '!BEST_PY_VER!'.split('.'))); sys.exit(0 if b > sys.version_info[:3] else 1)" 2>nul
        if not errorlevel 1 (
            echo [HFWToolkit] Note: Local .venv uses Python !VENV_PY_VER!, while Python !BEST_PY_VER! is installed on your system.
            echo              Run "Launch_HFW_Studio.bat --upgrade-python" anytime to upgrade .venv cleanly.
        )
    )
)

if "%DO_UPDATE%"=="1" (
    ".venv\Scripts\python.exe" -m hfw_toolkit.cli update --apply
)

echo.
echo ========================================================================
echo  Starting HFWToolkit Studio ^(http://127.0.0.1:8420^)
echo  Keep this window open while using the Web Studio. Press Ctrl+C to stop.
echo ========================================================================
echo.
".venv\Scripts\python.exe" -m hfw_toolkit.cli web !PASSTHROUGH!
set "RC=%errorlevel%"
if not "%RC%"=="0" (
    echo.
    echo [!] HFWToolkit Studio exited with code %RC%.
)
pause
exit /b %RC%

:FIND_NEWEST_PYTHON
set "BEST_PY="
set "BEST_PY_VER="
set "BEST_SCORE=0"

REM 1. Probe Windows Python Launcher from newest to oldest (3.16 .. 3.10)
where py >nul 2>nul
if not errorlevel 1 (
    for %%M in (3.16 3.15 3.14 3.13 3.12 3.11 3.10) do (
        if not defined BEST_PY (
            for /f "delims=" %%P in ('py -%%M -c "import sys; print(sys.executable)" 2^>nul') do (
                call :CONSIDER_PYTHON "%%P"
            )
        )
    )
)

REM 2. Probe standard Windows install folders
for %%R in ("%LOCALAPPDATA%\Programs\Python" "C:\Program Files" "C:") do (
    if exist "%%~R" (
        for /d %%D in ("%%~R\Python3*") do (
            if exist "%%~D\python.exe" call :CONSIDER_PYTHON "%%~D\python.exe"
        )
    )
)

REM 3. Probe PATH (skipping WindowsApps Microsoft Store stubs)
for /f "delims=" %%P in ('where python 2^>nul') do (
    call :CONSIDER_PYTHON "%%P"
)
goto :eof

:CONSIDER_PYTHON
set "CAND=%~1"
if not exist "%CAND%" goto :eof
echo "%CAND%" | findstr /I /C:"WindowsApps" >nul && goto :eof
for /f "tokens=1,2 delims=," %%A in ('"%CAND%" -c "import sys; v=sys.version_info; print(f'{v[0]*10000+v[1]*100+v[2]},{v[0]}.{v[1]}.{v[2]}')" 2^>nul') do (
    set "SC=%%A"
    set "VR=%%B"
    if !SC! GEQ 31000 (
        if !SC! GTR !BEST_SCORE! (
            set "BEST_SCORE=!SC!"
            set "BEST_PY=%CAND%"
            set "BEST_PY_VER=!VR!"
        )
    )
)
goto :eof
