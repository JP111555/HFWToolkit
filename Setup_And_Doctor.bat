@echo off
setlocal EnableDelayedExpansion
title HFWToolkit - One-Click Setup ^& System Doctor
cd /d "%~dp0"

set "UPGRADE_PY=0"
set "PASSTHROUGH="
for %%A in (%*) do (
    if /I "%%~A"=="--upgrade-python" (
        set "UPGRADE_PY=1"
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
    echo [HFWToolkit] Installing HFWToolkit package and dependencies...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e .
) else (
    for /f "delims=" %%V in ('".venv\Scripts\python.exe" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')" 2^>nul') do set "VENV_PY_VER=%%V"
    if defined BEST_PY_VER if defined VENV_PY_VER (
        ".venv\Scripts\python.exe" -c "import sys; b=tuple(map(int, '!BEST_PY_VER!'.split('.'))); sys.exit(0 if b > sys.version_info[:3] else 1)" 2>nul
        if not errorlevel 1 (
            echo [HFWToolkit] Newer system Python detected: !BEST_PY_VER! - current .venv uses !VENV_PY_VER!.
            echo              Run "Setup_And_Doctor.bat --upgrade-python" to rebuild .venv on Python !BEST_PY_VER!.
        )
    )
)

echo.
".venv\Scripts\python.exe" -m hfw_toolkit.cli setup !PASSTHROUGH!
echo.
".venv\Scripts\python.exe" -m hfw_toolkit.cli doctor
echo.
pause
exit /b %errorlevel%

:FIND_NEWEST_PYTHON
set "BEST_PY="
set "BEST_PY_VER="
set "BEST_SCORE=0"

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

for %%R in ("%LOCALAPPDATA%\Programs\Python" "C:\Program Files" "C:") do (
    if exist "%%~R" (
        for /d %%D in ("%%~R\Python3*") do (
            if exist "%%~D\python.exe" call :CONSIDER_PYTHON "%%~D\python.exe"
        )
    )
)

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
