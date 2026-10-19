@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "MSG_PYTHON_MISSING=[エラー] Python 3.12 x64 が見つかりませんでした。"
set "MSG_PYTHON_INSTALL=Python 3.12 x64 を公式サイトからインストールし、もう一度実行してください。"
set "MSG_PRESS_CLOSE=何かキーを押すと閉じます。"
set "MSG_VENV_VERSION=[エラー] 既存の仮想環境は必要なバージョンではありません。"
set "MSG_DELETE_VENV=アプリ終了後、.venv を .venv-before-recovery へ名前変更してください。"
set "MSG_RERUN_SETUP=setup_windows.bat をもう一度実行してください。"
set "MSG_USE_VENV=[1/5] .venv を使用します。"
set "MSG_CREATE_VENV=[1/5] Python 3.12 x64 の仮想環境を作成しています..."
set "MSG_VENV_FAILED=[エラー] 仮想環境を作成できませんでした。"
set "MSG_PIP_FAILED=[エラー] pipの更新に失敗しました。"
set "MSG_INSTALL_FAILED=[エラー] パッケージのインストールに失敗しました。"
set "MSG_NETWORK=インターネット接続を確認して、もう一度実行してください。"
set "MSG_LOCAL_FAILED=[エラー] local フォルダを作成できませんでした。"
set "MSG_DATA_FAILED=[エラー] data フォルダを作成できませんでした。"
set "MSG_APP_CONFIG_FAILED=[エラー] local\app.yaml を作成できませんでした。"
set "MSG_PROVIDERS_CONFIG_FAILED=[エラー] local\providers.yaml を作成できませんでした。"
set "MSG_PERSONA_CONFIG_FAILED=[エラー] local\persona.yaml を作成できませんでした。"
set "MSG_AUDIO_CONFIG_FAILED=[エラー] local\audio.yaml を作成できませんでした。"
set "MSG_MEMORY_CONFIG_FAILED=[エラー] local\memory.yaml を作成できませんでした。"
set "MSG_AVATAR_CONFIG_FAILED=[エラー] local\avatar.yaml を作成できませんでした。"
set "MSG_MMD_FOLDER_FAILED=[エラー] local\models\mmd フォルダを作成できませんでした。"
set "MSG_ENV_FAILED=[エラー] .env を作成できませんでした。"
set "MSG_MOTIONS_FAILED=[エラー] 同梱モーションの配置に失敗しました。既存ファイルは上書きしません。"

echo ================================================
echo multi-llm-companion 初回セットアップ
echo ================================================
echo.

set "PYTHON_CMD="
py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 and __import__('platform').machine().upper() == 'AMD64' else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.12"
if not defined PYTHON_CMD python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 and __import__('platform').machine().upper() == 'AMD64' else 1)" >nul 2>&1
if not defined PYTHON_CMD if not errorlevel 1 set "PYTHON_CMD=python"

if not defined PYTHON_CMD echo.
if not defined PYTHON_CMD echo %MSG_PYTHON_MISSING%
if not defined PYTHON_CMD echo %MSG_PYTHON_INSTALL%
if not defined PYTHON_CMD echo.
if not defined PYTHON_CMD echo %MSG_PRESS_CLOSE%
if not defined PYTHON_CMD pause >nul
if not defined PYTHON_CMD exit /b 1

for /f "delims=" %%V in ('%PYTHON_CMD% --version 2^>^&1') do set "PYTHON_VERSION=%%V"
echo [確認] %PYTHON_VERSION% が見つかりました。

set "VENV_EXISTS=0"
if exist ".venv\Scripts\python.exe" set "VENV_EXISTS=1"
set "VENV_CHECK_RESULT=0"
if "%VENV_EXISTS%"=="1" ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 and __import__('platform').machine().upper() == 'AMD64' else 1)" >nul 2>&1
if "%VENV_EXISTS%"=="1" set "VENV_CHECK_RESULT=%ERRORLEVEL%"
if not "%VENV_CHECK_RESULT%"=="0" echo.
if not "%VENV_CHECK_RESULT%"=="0" echo %MSG_VENV_VERSION%
if not "%VENV_CHECK_RESULT%"=="0" echo %MSG_DELETE_VENV%
if not "%VENV_CHECK_RESULT%"=="0" echo %MSG_RERUN_SETUP%
if not "%VENV_CHECK_RESULT%"=="0" echo %MSG_PRESS_CLOSE%
if not "%VENV_CHECK_RESULT%"=="0" pause >nul
if not "%VENV_CHECK_RESULT%"=="0" exit /b 1

if "%VENV_EXISTS%"=="1" echo %MSG_USE_VENV%
if "%VENV_EXISTS%"=="0" echo %MSG_CREATE_VENV%
set "VENV_CREATE_RESULT=0"
if "%VENV_EXISTS%"=="0" %PYTHON_CMD% -m venv .venv
if "%VENV_EXISTS%"=="0" set "VENV_CREATE_RESULT=%ERRORLEVEL%"
if not "%VENV_CREATE_RESULT%"=="0" echo.
if not "%VENV_CREATE_RESULT%"=="0" echo %MSG_VENV_FAILED%
if not "%VENV_CREATE_RESULT%"=="0" echo %MSG_PRESS_CLOSE%
if not "%VENV_CREATE_RESULT%"=="0" pause >nul
if not "%VENV_CREATE_RESULT%"=="0" exit /b 1

echo [2/5] 検証済みの版のpipを準備しています...
".venv\Scripts\python.exe" -m pip install --upgrade pip --constraint "constraints-windows-py312.txt"
set "PIP_UPGRADE_RESULT=%ERRORLEVEL%"
if not "%PIP_UPGRADE_RESULT%"=="0" echo.
if not "%PIP_UPGRADE_RESULT%"=="0" echo %MSG_PIP_FAILED%
if not "%PIP_UPGRADE_RESULT%"=="0" echo %MSG_NETWORK%
if not "%PIP_UPGRADE_RESULT%"=="0" echo %MSG_PRESS_CLOSE%
if not "%PIP_UPGRADE_RESULT%"=="0" pause >nul
if not "%PIP_UPGRADE_RESULT%"=="0" exit /b 1

echo [3/5] アプリと必要なパッケージをインストールしています...
".venv\Scripts\python.exe" -m pip install -e . --constraint "constraints-windows-py312.txt" --build-constraint "constraints-windows-py312.txt"
set "INSTALL_RESULT=%ERRORLEVEL%"
if not "%INSTALL_RESULT%"=="0" echo.
if not "%INSTALL_RESULT%"=="0" echo %MSG_INSTALL_FAILED%
if not "%INSTALL_RESULT%"=="0" echo %MSG_NETWORK%
if not "%INSTALL_RESULT%"=="0" echo %MSG_PRESS_CLOSE%
if not "%INSTALL_RESULT%"=="0" pause >nul
if not "%INSTALL_RESULT%"=="0" exit /b 1

echo [4/5] ローカル用フォルダを用意しています...
if not exist "local" mkdir "local"
if not exist "local" echo %MSG_LOCAL_FAILED%
if not exist "local" echo %MSG_PRESS_CLOSE%
if not exist "local" pause >nul
if not exist "local" exit /b 1
if not exist "data" mkdir "data"
if not exist "data" echo %MSG_DATA_FAILED%
if not exist "data" echo %MSG_PRESS_CLOSE%
if not exist "data" pause >nul
if not exist "data" exit /b 1
if not exist "local\models\mmd" mkdir "local\models\mmd"
if not exist "local\models\mmd" echo %MSG_MMD_FOLDER_FAILED%
if not exist "local\models\mmd" echo %MSG_PRESS_CLOSE%
if not exist "local\models\mmd" pause >nul
if not exist "local\models\mmd" exit /b 1

echo [5/5] 見本設定をコピーしています...
if not exist "local\app.yaml" copy /Y "config\app.example.yaml" "local\app.yaml" >nul
if not exist "local\app.yaml" echo %MSG_APP_CONFIG_FAILED%
if not exist "local\app.yaml" echo %MSG_PRESS_CLOSE%
if not exist "local\app.yaml" pause >nul
if not exist "local\app.yaml" exit /b 1
if not exist "local\providers.yaml" copy /Y "config\providers.example.yaml" "local\providers.yaml" >nul
if not exist "local\providers.yaml" echo %MSG_PROVIDERS_CONFIG_FAILED%
if not exist "local\providers.yaml" echo %MSG_PRESS_CLOSE%
if not exist "local\providers.yaml" pause >nul
if not exist "local\providers.yaml" exit /b 1
if not exist "local\persona.yaml" copy /Y "config\persona.example.yaml" "local\persona.yaml" >nul
if not exist "local\persona.yaml" echo %MSG_PERSONA_CONFIG_FAILED%
if not exist "local\persona.yaml" echo %MSG_PRESS_CLOSE%
if not exist "local\persona.yaml" pause >nul
if not exist "local\persona.yaml" exit /b 1
if not exist "local\audio.yaml" copy /Y "config\audio.example.yaml" "local\audio.yaml" >nul
if not exist "local\audio.yaml" echo %MSG_AUDIO_CONFIG_FAILED%
if not exist "local\audio.yaml" echo %MSG_PRESS_CLOSE%
if not exist "local\audio.yaml" pause >nul
if not exist "local\audio.yaml" exit /b 1
if not exist "local\memory.yaml" copy /Y "config\memory.example.yaml" "local\memory.yaml" >nul
if not exist "local\memory.yaml" echo %MSG_MEMORY_CONFIG_FAILED%
if not exist "local\memory.yaml" echo %MSG_PRESS_CLOSE%
if not exist "local\memory.yaml" pause >nul
if not exist "local\memory.yaml" exit /b 1
if not exist "local\avatar.yaml" copy /Y "config\avatar.example.yaml" "local\avatar.yaml" >nul
if not exist "local\avatar.yaml" echo %MSG_AVATAR_CONFIG_FAILED%
if not exist "local\avatar.yaml" echo %MSG_PRESS_CLOSE%
if not exist "local\avatar.yaml" pause >nul
if not exist "local\avatar.yaml" exit /b 1
if not exist ".env" copy /Y ".env.example" ".env" >nul
if not exist ".env" echo %MSG_ENV_FAILED%
if not exist ".env" echo %MSG_PRESS_CLOSE%
if not exist ".env" pause >nul
if not exist ".env" exit /b 1

".venv\Scripts\python.exe" "scripts\install_bundled_motions.py"
set "MOTIONS_RESULT=%ERRORLEVEL%"
if not "%MOTIONS_RESULT%"=="0" echo %MSG_MOTIONS_FAILED%
if not "%MOTIONS_RESULT%"=="0" echo %MSG_PRESS_CLOSE%
if not "%MOTIONS_RESULT%"=="0" pause >nul
if not "%MOTIONS_RESULT%"=="0" exit /b 1

".venv\Scripts\python.exe" "scripts\create_windows_shortcut.py"
if errorlevel 1 echo [注意] ショートカットを作れませんでした。run.bat を使用してください。

echo.
echo ================================================
echo [完了] セットアップが正常に終わりました。
echo ================================================
echo 1. Mock を使えば、設定を変更せずに動作確認できます。
echo 2. 「起動」ショートカットを開いてください。診断時は run_debug.bat を使います。
echo 3. APIキーはアプリの「設定 → 環境設定」から入力できます。
echo    従来のWindows環境変数による設定も使えます。
echo 4. 音声機能の準備は docs\audio_ja.md を参照してください。
echo 5. Whisperを使う場合は download_whisper_model.bat を実行してください。
echo 6. 記憶は既定で自動承認します。「設定」から「記憶を管理」で確認・変更できます。
echo 7. アバターの表示名と色は local\avatar.yaml で変更できます。
echo 8. 自分で利用許諾を確認したPMXモデルは local\models\mmd に置いてください。
echo 9. 同梱VMDの出典・規約と導入方法は docs\bundled_motions_ja.md を参照してください。
echo.
echo 何かキーを押すと閉じます。
pause >nul
exit /b 0
