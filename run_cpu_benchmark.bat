@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run run_setup_windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\benchmark_bioalign_final_cpu.py --subject h0 --seed 2026 --epochs 20 --threads 1 --warmup 200 --runs 1500
pause
