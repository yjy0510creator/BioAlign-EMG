@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run run_setup_windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\run_bioalign_final_evidence_pack.py --subjects h0-h29 --seeds 42,2026,3407 --epochs 20 --batch-size 256
pause
