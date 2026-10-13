@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ================================================
echo Whisper日本語認識モデルの準備
echo ================================================
echo.

if not exist ".venv\Scripts\python.exe" echo [エラー] 先に setup_windows.bat を実行してください。
if not exist ".venv\Scripts\python.exe" pause
if not exist ".venv\Scripts\python.exe" exit /b 1

if not exist "local\models\faster-whisper-small" mkdir "local\models\faster-whisper-small"

echo Whisper smallモデルを取得して読み込み確認します。
echo 数百MBあるため、回線によっては時間がかかります。
echo.

".venv\Scripts\python.exe" "scripts\download_whisper_model.py"
if errorlevel 1 echo.
if errorlevel 1 echo [エラー] モデルを準備できませんでした。ネット接続と空き容量を確認してください。
if errorlevel 1 pause
if errorlevel 1 exit /b 1

echo.
echo モデルファイルの読み込み確認を行います...
".venv\Scripts\python.exe" -c "from faster_whisper import WhisperModel; WhisperModel('local/models/faster-whisper-small', device='cpu', compute_type='int8'); print('[成功] モデルを正常に読み込めました。')"
if errorlevel 1 echo [エラー] モデルを読み込めませんでした。
if errorlevel 1 pause
if errorlevel 1 exit /b 1
echo 次に run.bat を実行し、設定画面でWhisperを選んでください。
pause
exit /b 0
