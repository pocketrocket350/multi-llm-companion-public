@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "MSG_SETUP_INCOMPLETE=[エラー] 初回セットアップが終わっていません。"
set "MSG_RUN_SETUP=setup_windows.bat を先に実行してください。"
set "MSG_PRESS_CLOSE=何かキーを押すと閉じます。"
set "MSG_TEST_FAILED=[エラー] pytestの自動テストに失敗しました。"
set "MSG_RUFF_FAILED=[エラー] ruffがコードの問題を見つけました。"
set "MSG_FORMAT_FAILED=[エラー] ruffが書式のずれを見つけました。"
set "MSG_FIX=表示された内容を確認して修正してください。"
set "MSG_RERUN_TEST=test.bat をもう一度実行してください。"

if not exist ".venv\Scripts\python.exe" echo %MSG_SETUP_INCOMPLETE%
if not exist ".venv\Scripts\python.exe" echo %MSG_RUN_SETUP%
if not exist ".venv\Scripts\python.exe" echo.
if not exist ".venv\Scripts\python.exe" echo %MSG_PRESS_CLOSE%
if not exist ".venv\Scripts\python.exe" pause >nul
if not exist ".venv\Scripts\python.exe" exit /b 1

echo [1/3] pytestで自動テストを実行します...
".venv\Scripts\python.exe" -m pytest --basetemp "%CD%\.test-tmp"
set "TEST_RESULT=%ERRORLEVEL%"
if not "%TEST_RESULT%"=="0" echo.
if not "%TEST_RESULT%"=="0" echo %MSG_TEST_FAILED%
if not "%TEST_RESULT%"=="0" echo %MSG_FIX%
if not "%TEST_RESULT%"=="0" echo %MSG_RERUN_TEST%
if not "%TEST_RESULT%"=="0" echo %MSG_PRESS_CLOSE%
if not "%TEST_RESULT%"=="0" pause >nul
if not "%TEST_RESULT%"=="0" exit /b 1

echo.
echo [2/3] ruffでコードの問題を確認します...
".venv\Scripts\python.exe" -m ruff check .
set "RUFF_RESULT=%ERRORLEVEL%"
if not "%RUFF_RESULT%"=="0" echo.
if not "%RUFF_RESULT%"=="0" echo %MSG_RUFF_FAILED%
if not "%RUFF_RESULT%"=="0" echo %MSG_FIX%
if not "%RUFF_RESULT%"=="0" echo %MSG_RERUN_TEST%
if not "%RUFF_RESULT%"=="0" echo %MSG_PRESS_CLOSE%
if not "%RUFF_RESULT%"=="0" pause >nul
if not "%RUFF_RESULT%"=="0" exit /b 1

echo.
echo [3/3] ruffで書式を確認します...
".venv\Scripts\python.exe" -m ruff format --check .
set "FORMAT_RESULT=%ERRORLEVEL%"
if not "%FORMAT_RESULT%"=="0" echo.
if not "%FORMAT_RESULT%"=="0" echo %MSG_FORMAT_FAILED%
if not "%FORMAT_RESULT%"=="0" echo %MSG_FIX%
if not "%FORMAT_RESULT%"=="0" echo %MSG_RERUN_TEST%
if not "%FORMAT_RESULT%"=="0" echo %MSG_PRESS_CLOSE%
if not "%FORMAT_RESULT%"=="0" pause >nul
if not "%FORMAT_RESULT%"=="0" exit /b 1

echo.
echo ================================================
echo [完了] すべての確認に成功しました。
echo ================================================
echo 何かキーを押すと閉じます。
pause >nul
exit /b 0
