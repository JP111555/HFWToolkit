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
        if defined BEST_PY (
            echo [HFWToolkit] Rebuilding .venv using newest system Python !BEST_PY_VER! - !BEST_PY!
            rmdir /s /q ".venv"
        ) else (
            echo [!] Cannot rebuild .venv: no system Python 3.10+ found. Keeping existing .venv.
        )
    )
)

if not exist ".venv\Scripts\python.exe" (
    if not defined BEST_PY (
        echo [!] Could not find Python 3.10+ on this PC.
        echo     Please install Python 3.12+ from https://www.python.org/downloads/
        echo     Tip: In the installer, check "Add python.exe to PATH" and "Install py launcher".
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
    for /f "delims=" %%V in ('call ".venv\Scripts\python.exe" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')" 2^>nul') do set "VENV_PY_VER=%%V"
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

REM 1. Probe Windows Python Launcher (py.exe: classic + PEP 773 Python Install Manager)
set "PY_LAUNCHER="
for /f "delims=" %%L in ('where py 2^>nul') do if not defined PY_LAUNCHER set "PY_LAUNCHER=%%L"
if not defined PY_LAUNCHER if exist "%SystemRoot%\py.exe" set "PY_LAUNCHER=%SystemRoot%\py.exe"
if not defined PY_LAUNCHER if exist "%LOCALAPPDATA%\Python\bin\py.exe" set "PY_LAUNCHER=%LOCALAPPDATA%\Python\bin\py.exe"
if defined PY_LAUNCHER (
    for /f "tokens=1,*" %%A in ('call "!PY_LAUNCHER!" -0p 2^>nul') do (
        set "RAW_P=%%B"
        if defined RAW_P (
            if "!RAW_P:~0,1!"=="*" set "RAW_P=!RAW_P:~1!"
            for /f "tokens=*" %%Q in ("!RAW_P!") do call :CONSIDER_PYTHON "%%Q"
        )
    )
    for /f "delims=" %%P in ('call "!PY_LAUNCHER!" -3 -c "import sys; print(sys.executable)" 2^>nul') do (
        call :CONSIDER_PYTHON "%%P"
    )
)

REM 2. Probe Windows Registry (HKCU / HKLM PythonCore - catches custom install paths)
for %%H in ("HKCU\Software\Python\PythonCore" "HKLM\Software\Python\PythonCore" "HKLM\Software\WOW6432Node\Python\PythonCore") do (
    for /f "tokens=2,*" %%A in ('reg query %%H /s /v ExecutablePath 2^>nul ^| findstr /I "ExecutablePath"') do (
        call :CONSIDER_PYTHON "%%B"
    )
    for /f "tokens=2,*" %%A in ('reg query %%H /s /ve 2^>nul ^| findstr /I "REG_SZ"') do (
        if exist "%%B\python.exe" call :CONSIDER_PYTHON "%%B\python.exe"
    )
)

REM 3. Probe PEP 773 Python Install Manager (%LOCALAPPDATA%\Python\pythoncore-*) and classic folders
if exist "%LOCALAPPDATA%\Python" (
    if exist "%LOCALAPPDATA%\Python\bin\python.exe" call :CONSIDER_PYTHON "%LOCALAPPDATA%\Python\bin\python.exe"
    for /d %%D in ("%LOCALAPPDATA%\Python\pythoncore-3*" "%LOCALAPPDATA%\Python\python-3*") do (
        if exist "%%~D\python.exe" call :CONSIDER_PYTHON "%%~D\python.exe"
    )
)
for %%R in ("%LOCALAPPDATA%\Programs\Python" "%ProgramFiles%" "%ProgramFiles(x86)%" "C:") do (
    if exist "%%~R" (
        for /d %%D in ("%%~R\Python3*") do (
            if exist "%%~D\python.exe" call :CONSIDER_PYTHON "%%~D\python.exe"
        )
    )
)
for %%E in ("%USERPROFILE%\scoop\apps\python\current\python.exe" "%USERPROFILE%\miniconda3\python.exe" "%USERPROFILE%\anaconda3\python.exe" "C:\ProgramData\miniconda3\python.exe") do (
    if exist "%%~E" call :CONSIDER_PYTHON "%%~E"
)

REM 4. Probe PATH (python & python3; real Store/MSIX Python succeeds, fake Store stubs return nothing)
for /f "delims=" %%P in ('where python 2^>nul') do call :CONSIDER_PYTHON "%%P"
for /f "delims=" %%P in ('where python3 2^>nul') do call :CONSIDER_PYTHON "%%P"
goto :eof

:CONSIDER_PYTHON
set "CAND=%~1"
if not defined CAND goto :eof
if not exist "%CAND%" goto :eof
REM NOTE: Using `call "%CAND%"` avoids cmd.exe's quote-stripping quirk inside `for /f ('"..." ...')`!
for /f "tokens=1,2,3,4 delims=|" %%A in ('call "%CAND%" -c "import sys; v=sys.version_info; print(f'HFWPY|{v[0]*10000+v[1]*100+v[2]}|{v[0]}.{v[1]}.{v[2]}|{sys.executable}')" 2^>nul') do (
    if "%%A"=="HFWPY" (
        set "SC=%%B"
        set "VR=%%C"
        set "EX=%%D"
        if not defined EX set "EX=!CAND!"
        if !SC! GEQ 31000 (
            if !SC! GTR !BEST_SCORE! (
                set "BEST_SCORE=!SC!"
                set "BEST_PY=!EX!"
                set "BEST_PY_VER=!VR!"
            )
        )
    )
)
goto :eof
