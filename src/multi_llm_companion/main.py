"""デスクトップアプリの起動処理。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from multi_llm_companion.application import ApplicationBackend, ApplicationError
from multi_llm_companion.config import ConfigError, EnvironmentSetupRequired, load_runtime_config
from multi_llm_companion.environment import EnvironmentSettingsError
from multi_llm_companion.ui.chat_window import ChatWindow
from multi_llm_companion.ui.environment_dialog import EnvironmentDialog
from multi_llm_companion.ui.render_quality import configure_default_surface_format


def _project_root() -> Path:
    configured = os.getenv("MULTI_LLM_COMPANION_HOME", "").strip()
    return Path(configured).expanduser() if configured else Path.cwd()


def main() -> int:
    app = QApplication.instance()
    backend: ApplicationBackend | None = None
    try:
        runtime = load_runtime_config(_project_root())
    except EnvironmentSetupRequired as exc:
        app = app or QApplication(sys.argv)
        try:
            dialog = EnvironmentDialog(_project_root())
        except (ConfigError, EnvironmentSettingsError) as setup_error:
            QMessageBox.critical(None, "環境設定を開けません", str(setup_error))
            return 1
        dialog.error_label.setText(str(exc))
        accepted = dialog.exec() == EnvironmentDialog.DialogCode.Accepted
        dialog.deleteLater()
        if accepted:
            QMessageBox.information(
                None,
                "環境設定を保存しました",
                "アプリをもう一度起動してください。保存した方式で接続します。",
            )
        return 0 if accepted else 1
    except ConfigError as exc:
        app = app or QApplication(sys.argv)
        QMessageBox.critical(
            None,
            "起動できません",
            f"{exc}\n\n設定を編集した場合は、直前の変更を戻してください。\n"
            "詳しい手順は docs/backup_and_recovery_ja.md を開いてください。\n"
            "初回セットアップが未完了の場合だけ setup_windows.bat を実行してください。",
        )
        return 1
    except Exception as exc:
        app = app or QApplication(sys.argv)
        QMessageBox.critical(
            None,
            "起動できません",
            f"設定の読み込み中に予期しないエラーが発生しました（{type(exc).__name__}）。\n"
            "run_debug.bat で起動し直してください。\n"
            "解決しない場合は、この表示を控えて報告してください。",
        )
        return 1

    # WindowsのQOpenGLWidgetは、QApplication生成前に共有形式を決める必要がある。
    if app is None:
        configure_default_surface_format(runtime.avatar.three_d.render_quality)
        app = QApplication(sys.argv)

    try:
        backend = ApplicationBackend(runtime)
    except ApplicationError as exc:
        QMessageBox.critical(
            None,
            "起動できません",
            f"{exc}\n\n設定と素材の配置を確認してください。\n"
            "詳しい手順は docs/backup_and_recovery_ja.md を開いてください。\n"
            "調査用の画面を表示するには run_debug.bat で起動し直してください。",
        )
        return 1
    except Exception as exc:
        QMessageBox.critical(
            None,
            "起動できません",
            f"会話機能の準備中に予期しないエラーが発生しました（{type(exc).__name__}）。\n"
            "run_debug.bat で起動し直してください。\n"
            "解決しない場合は、この表示を控えて報告してください。",
        )
        return 1

    try:
        window = ChatWindow(backend)
    except ApplicationError as exc:
        backend.close()
        QMessageBox.critical(None, "起動できません", str(exc))
        return 1
    except Exception as exc:
        backend.close()
        QMessageBox.critical(
            None,
            "起動できません",
            f"画面の準備中に予期しないエラーが発生しました（{type(exc).__name__}）。\n"
            "run_debug.bat で起動し直してください。\n"
            "解決しない場合は、この表示を控えて報告してください。",
        )
        return 1
    window.show()
    try:
        return app.exec()
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
