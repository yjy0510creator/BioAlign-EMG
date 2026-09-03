@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run run_setup_windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\smoke_test.py
pause
