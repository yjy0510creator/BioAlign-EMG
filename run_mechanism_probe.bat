@echo off
setlocal
cd /d "%~dp0"
if "%~3"=="" (
  echo Usage: run_mechanism_probe.bat SUBJECT_NPZ CHECKPOINT OUTPUT_DIR [VARIANT]
  exit /b 2
)
set VARIANT=%~4
if "%VARIANT%"=="" set VARIANT=continuous
python -m mechanism_validation.validate_mechanism --data "%~1" --checkpoint "%~2" --output "%~3" --variant "%VARIANT%"
endlocal
