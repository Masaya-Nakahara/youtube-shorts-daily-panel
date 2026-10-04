@echo off
REM Daily YouTube Shorts collection (search corpus + panel time series + channel deepening).
REM Pinned to the project venv. Logs to logs\shorts_daily_<ts>.log.
REM Quota budget (YouTube API daily cap = 10,000 units):
REM   collect 7000 + panel seed 300 + panel resample 1200 + deepen 800 = 9300 max.
REM The panel is the irreplaceable time series (views cannot be re-measured for a past day).
REM Corpus rebuild / template-stability are NOT run here: they are reconstructible
REM from raw at any time and would write a ~256MB corpus every day.
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
if not exist logs mkdir logs

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "!PY!" (
    echo FATAL: venv python not found at !PY!
    exit /b 1
)

for /f %%i in (
'
powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"
'
) do set TS=%%i
set LOG=logs\shorts_daily_!TS!.log
echo shorts daily start !TS!  > "!LOG!"

echo [1/4] search collection (JP/US, relevance+date+viewCount) >> "!LOG!"
"!PY!" scripts\01_collect_pilot.py --orders relevance date viewCount --quota-limit 7000 --out "data\raw\daily_!TS!.jsonl" --log "logs\daily_!TS!.log" --meta-out "logs\daily_!TS!_meta.json" >> "!LOG!" 2>&1
if errorlevel 1 echo   WARNING: collection step failed >> "!LOG!"

echo [2/4] panel seed (add fresh videos to cohort) >> "!LOG!"
"!PY!" scripts\07_panel_resample.py seed --quota-limit 300 >> "!LOG!" 2>&1
if errorlevel 1 echo   WARNING: panel seed failed >> "!LOG!"

echo [3/4] panel resample (record todays views - THE MOAT) >> "!LOG!"
"!PY!" scripts\07_panel_resample.py resample --quota-limit 1200 >> "!LOG!" 2>&1
if errorlevel 1 echo   WARNING: panel resample failed >> "!LOG!"

echo [4/4] deepen channels >> "!LOG!"
"!PY!" scripts\13_deepen_channels.py --quota-limit 800 >> "!LOG!" 2>&1
if errorlevel 1 echo   WARNING: deepen failed (non-fatal) >> "!LOG!"

echo [status] cohort >> "!LOG!"
"!PY!" scripts\07_panel_resample.py status >> "!LOG!" 2>&1

echo shorts daily DONE !TS! >> "!LOG!"
endlocal
exit /b 0
