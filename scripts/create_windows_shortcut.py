"""インストール先のpythonwへ直接つながる起動ショートカットを作る。"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def create_shortcut(root: Path) -> None:
    root = root.resolve()
    python = root / ".venv/Scripts/pythonw.exe"
    launcher = root / "scripts/launch_windowed.py"
    if not python.is_file() or not launcher.is_file():
        raise FileNotFoundError("起動用ファイルがありません。セットアップを確認してください。")
    arguments = f'"{launcher}"'
    command = (
        "$ErrorActionPreference = 'Stop'; "
        "$shell = New-Object -ComObject WScript.Shell; "
        f"$link = $shell.CreateShortcut({_literal(str(root / '起動.lnk'))}); "
        f"$link.TargetPath = {_literal(str(python))}; "
        f"$link.Arguments = {_literal(arguments)}; "
        f"$link.WorkingDirectory = {_literal(str(root))}; "
        "$link.Description = 'multi-llm-companion'; $link.Save()"
    )
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        check=True,
        capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
        timeout=30,
    )


if __name__ == "__main__":
    try:
        create_shortcut(Path(__file__).resolve().parents[1])
    except (OSError, subprocess.SubprocessError):
        print("[エラー] 起動ショートカットを作成できませんでした。")
        print("フォルダーの書込み権限を確認してください。run.bat からも起動できます。")
        raise SystemExit(1) from None
