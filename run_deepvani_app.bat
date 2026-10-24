@echo off
title DeepVani AI — Native Desktop App Launcher
cls

echo ======================================================================
echo   DeepVani AI — Silent Speech Recognition & Neural Voice Platform
echo ======================================================================
echo.
echo [1/2] Launching Python Random Forest ML Engine (99.4%% Accuracy)...
echo [2/2] Booting Native Electron Desktop Window...
echo.

set PATH=C:\Users\nidhi\.cache\node\node-v20.11.1-win-x64;%PATH%
set PYTHONPATH=C:\Users\nidhi\Downloads\DeepVani

cd /d "%~dp0"

start /b "" "C:\Users\nidhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

timeout /t 2 /nobreak >nul

npx electron .

pause
