"""デスクトップアプリの起動処理。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from multi_llm_companion.application import ApplicationBackend, ApplicationError
from multi_llm_companion.config import ConfigError, load_runtime_config
from multi_llm_companion.ui.chat_window import ChatWindow
from multi_llm_companion.ui.render_quality import configure_default_surface_format


def _project_root() -> Path:
    configured = os.getenv("MULTI_LLM_COMPANION_HOME", "").strip()
    return Path(configured).expanduser() if configured else Path.cwd()


def main() -> int:
    app = QApplication.instance()
    backend: ApplicationBackend | None = None
    try:
        runtime = load_runtime_config(_project_root())
    except ConfigError as exc:
        app = app or QApplication(sys.argv)
        QMessageBox.critical(
            None,
            "起動できません",
            f"{exc}\n\n初回は setup_windows.bat を実行してください。",
        )
        return 1
    except Exception:
        app = app or QApplication(sys.argv)
        QMessageBox.critical(
            None,
            "起動できません",
            "予期しない起動エラーが発生しました。test.batを実行してください。",
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
            f"{exc}\n\n初回は setup_windows.bat を実行してください。",
        )
        return 1
    except Exception:
        QMessageBox.critical(
            None,
            "起動できません",
            "予期しない起動エラーが発生しました。test.batを実行してください。",
        )
        return 1

    try:
        window = ChatWindow(backend)
    except ApplicationError as exc:
        backend.close()
        QMessageBox.critical(None, "起動できません", str(exc))
        return 1
    except Exception:
        backend.close()
        QMessageBox.critical(
            None,
            "起動できません",
            "画面の準備中に予期しないエラーが発生しました。test.batを実行してください。",
        )
        return 1
    window.show()
    try:
        return app.exec()
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
