@echo off
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Local environment not found. Run setup.ps1 first.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0supervisor.py"
pause
