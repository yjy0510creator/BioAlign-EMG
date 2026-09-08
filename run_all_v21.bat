@echo off
setlocal enabledelayedexpansion
cd /d %~dp0
set PROJECT_ROOT=D:\BioSelect_EMG
set CODE_ROOT=%~dp0

python -m pip install -r requirements_v2.txt

python -m mechanism_validation.build_senic_npz ^
  --project-root "%PROJECT_ROOT%" ^
  --subjects h0-h29

python -m mechanism_validation.aggregate_results ^
  --mode run_collect ^
  --data-dir "%PROJECT_ROOT%\data\processed_v21" ^
  --output-root "%PROJECT_ROOT%\results_v21\full_30subjects" ^
  --subjects h0-h29 ^
  --epochs 30 ^
  --seeds 42 2026 3407 ^
  --models all ^
  --correction-baselines all ^
  --classical

pause
