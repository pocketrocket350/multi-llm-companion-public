@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "MSG_SETUP_INCOMPLETE=[エラー] 初回セットアップが終わっていません。"
set "MSG_RUN_SETUP=setup_windows.bat を先に実行してください。"
set "MSG_PRESS_CLOSE=何かキーを押すと閉じます。"
set "MSG_APP_CONFIG_MISSING=[エラー] local\app.yaml がありません。"
set "MSG_CREATE_CONFIG=setup_windows.bat を実行して見本設定を作成してください。"
if not exist ".venv\Scripts\pythonw.exe" echo %MSG_SETUP_INCOMPLETE%
if not exist ".venv\Scripts\pythonw.exe" echo %MSG_RUN_SETUP%
if not exist ".venv\Scripts\pythonw.exe" echo.
if not exist ".venv\Scripts\pythonw.exe" echo %MSG_PRESS_CLOSE%
if not exist ".venv\Scripts\pythonw.exe" pause >nul
if not exist ".venv\Scripts\pythonw.exe" exit /b 1

if not exist "local\app.yaml" echo %MSG_APP_CONFIG_MISSING%
if not exist "local\app.yaml" echo %MSG_CREATE_CONFIG%
if not exist "local\app.yaml" echo.
if not exist "local\app.yaml" echo %MSG_PRESS_CLOSE%
if not exist "local\app.yaml" pause >nul
if not exist "local\app.yaml" exit /b 1

start "" ".venv\Scripts\pythonw.exe" "scripts\launch_windowed.py"
exit /b %ERRORLEVEL%
