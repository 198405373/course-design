@echo off
rem ============================================================
rem  One-click demo launcher for the defect detection system.
rem  1) find python (prefer conda env cv_tutorial, fallback to python)
rem  2) start Flask backend in a new window
rem  3) open browser at http://127.0.0.1:5000
rem ============================================================
cd /d "%~dp0"

set "PY=D:\ANACONDA\envs\cv_tutorial\python.exe"
if not exist "%PY%" set "PY=python"

echo [1/3] Using Python: %PY%
"%PY%" --version >nul 2>&1 || (echo ERROR: python not found. Please edit run_demo.cmd PY path. & pause & exit /b 1)

echo [2/3] Starting backend (keep this window). Port: 5000
start "defect-backend" cmd /k ""%PY%" backend\app.py"

echo [3/3] Waiting for service...
timeout /t 4 /nobreak >nul
start http://127.0.0.1:5000
echo Done. If the page does not open, visit http://127.0.0.1:5000 manually.
pause
