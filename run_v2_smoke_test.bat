@echo off
setlocal
cd /d "%~dp0"
python -m pip install -r requirements_v2.txt
python -m pytest tests_v2 -q
python -m mechanism_validation.synthetic_validation --output smoke_results --epochs 3
if errorlevel 1 (
  echo [FAILED] See the error above.
  exit /b 1
)
echo [OK] Core operators, gradient flow, model variants, and a synthetic end-to-end run passed.
endlocal
