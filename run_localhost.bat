@echo off
title DeepVani AI - Localhost Server Launcher
cls

echo ======================================================================
echo   DeepVani AI — Real-Time Silent Speech Recognition Platform
echo ======================================================================
echo.
echo [1/2] Starting Localhost HTTP Server on http://localhost:8000...
echo [2/2] Opening Web Interface in your default browser...
echo.

set PYTHON_BIN="C:\Users\nidhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if not exist %PYTHON_BIN% (
    set PYTHON_BIN=python
)

start "" "http://localhost:8000"

%PYTHON_BIN% server.py

pause
