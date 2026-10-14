"""pythonwから通常起動し、読込失敗も画面で案内する。"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


def _show_error() -> None:
    ctypes.windll.user32.MessageBoxW(
        None,
        "アプリを起動できませんでした。\n\n"
        "run_debug.bat を開き、表示されるエラーを確認してください。\n"
        "初回は setup_windows.bat を実行してください。",
        "起動できません",
        0x10,
    )


def main() -> int:
    try:
        os.chdir(Path(__file__).resolve().parents[1])
        # Qt自体を読み込めない場合も、Windowsのダイアログで案内する。
        from multi_llm_companion.main import main as run_app

        return run_app()
    except Exception:
        _show_error()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
