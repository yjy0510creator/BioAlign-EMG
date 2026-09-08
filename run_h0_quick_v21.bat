@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

REM Usage: edit PROJECT_ROOT if your real SeNic data is not under D:\BioSelect_EMG
set PROJECT_ROOT=D:\BioSelect_EMG
set CODE_ROOT=%~dp0

python -m pip install -r requirements_v2.txt

python -m mechanism_validation.build_senic_npz ^
  --project-root "%PROJECT_ROOT%" ^
  --subjects h0 ^
  --rebuild

python -m mechanism_validation.run_experiment ^
  --data "%PROJECT_ROOT%\data\processed_v21\h0.npz" ^
  --output "%PROJECT_ROOT%\results_v21\h0_quick_5epoch" ^
  --classical ^
  --epochs 5 ^
  --seeds 42 ^
  --models all ^
  --correction-baselines all ^
  --force

python -m mechanism_validation.validate_mechanism ^
  --data "%PROJECT_ROOT%\data\processed_v21\h0.npz" ^
  --checkpoint "%PROJECT_ROOT%\results_v21\h0_quick_5epoch\checkpoints\continuous_seed42.pt" ^
  --variant continuous ^
  --output "%PROJECT_ROOT%\results_v21\h0_quick_5epoch\mechanism_continuous"

echo.
echo Done. Main table:
echo %PROJECT_ROOT%\results_v21\h0_quick_5epoch\metrics_all.csv
pause
