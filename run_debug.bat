@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "MSG_SETUP_INCOMPLETE=[エラー] 初回セットアップが終わっていません。"
set "MSG_RUN_SETUP=setup_windows.bat を先に実行してください。"
set "MSG_PRESS_CLOSE=何かキーを押すと閉じます。"
set "MSG_APP_CONFIG_MISSING=[エラー] local\app.yaml がありません。"
set "MSG_CREATE_CONFIG=setup_windows.bat を実行して見本設定を作成してください。"
set "MSG_APP_FAILED=[エラー] アプリが正常に終了しませんでした。"
set "MSG_EXIT_CODE=が終了コードです。"
set "MSG_KEEP_ERROR=表示されたエラー内容を控えてください。"

if not exist ".venv\Scripts\python.exe" echo %MSG_SETUP_INCOMPLETE%
if not exist ".venv\Scripts\python.exe" echo %MSG_RUN_SETUP%
if not exist ".venv\Scripts\python.exe" echo.
if not exist ".venv\Scripts\python.exe" echo %MSG_PRESS_CLOSE%
if not exist ".venv\Scripts\python.exe" pause >nul
if not exist ".venv\Scripts\python.exe" exit /b 1

if not exist "local\app.yaml" echo %MSG_APP_CONFIG_MISSING%
if not exist "local\app.yaml" echo %MSG_CREATE_CONFIG%
if not exist "local\app.yaml" echo.
if not exist "local\app.yaml" echo %MSG_PRESS_CLOSE%
if not exist "local\app.yaml" pause >nul
if not exist "local\app.yaml" exit /b 1

echo multi-llm-companion を起動します...
".venv\Scripts\python.exe" -m multi_llm_companion
set "APP_EXIT_CODE=%ERRORLEVEL%"

if not "%APP_EXIT_CODE%"=="0" echo.
if not "%APP_EXIT_CODE%"=="0" echo %MSG_APP_FAILED%
if not "%APP_EXIT_CODE%"=="0" echo %APP_EXIT_CODE% %MSG_EXIT_CODE%
if not "%APP_EXIT_CODE%"=="0" echo %MSG_KEEP_ERROR%
if not "%APP_EXIT_CODE%"=="0" echo %MSG_PRESS_CLOSE%
if not "%APP_EXIT_CODE%"=="0" pause >nul

exit /b %APP_EXIT_CODE%
