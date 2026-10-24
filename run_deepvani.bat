@echo off
title DeepVani AI - Silent Speech Decoder Launcher
cls

echo ======================================================================
echo   DeepVani AI — Real-Time Silent Speech Recognition Platform
echo ======================================================================
echo.
echo [1/3] Detecting Python runtime environment...

set PYTHON_BIN="C:\Users\nidhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if not exist %PYTHON_BIN% (
    set PYTHON_BIN=python
)

echo [2/3] Booting DeepVani AI Engine on http://localhost:8000...
echo [3/3] Opening Web Interface in your browser...
echo.

start "" "http://localhost:8000"

%PYTHON_BIN% -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

pause
