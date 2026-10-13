@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing. Please check the normal setup.
  pause
  exit /b 1
)
echo Starting Phase 9 isolated acceptance. Regular data will not be changed.
echo Test messages use the same LLM API as your normal app.
".venv\Scripts\python.exe" -m scripts.phase9_acceptance_app
set "PHASE9_EXIT=%ERRORLEVEL%"
if not "%PHASE9_EXIT%"=="0" pause
exit /b %PHASE9_EXIT%
