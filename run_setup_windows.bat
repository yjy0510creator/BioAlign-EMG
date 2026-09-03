@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 (
  set "PY=py"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or 3.11 and enable Add to PATH.
    pause
    exit /b 1
  )
  set "PY=python"
)

%PY% --version
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv
if errorlevel 1 goto fail

".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto fail

echo.
echo Environment setup completed.
echo Next: run_smoke_test.bat
pause
exit /b 0

:fail
echo.
echo Setup failed. Keep this window open and capture the full error.
pause
exit /b 1
