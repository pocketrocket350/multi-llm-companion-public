"""窓なし起動のエラー案内とショートカットの参照先を確認する。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import launch_windowed
from scripts.create_windows_shortcut import _literal, create_shortcut


@pytest.mark.parametrize("exit_code", [0, 1])
def test_windowed_launch_preserves_app_exit_and_home(monkeypatch, tmp_path, exit_code):
    root = tmp_path / "アプリ"
    (root / "scripts").mkdir(parents=True)
    monkeypatch.setattr(launch_windowed, "__file__", str(root / "scripts/launch_windowed.py"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MULTI_LLM_COMPANION_HOME", str(tmp_path / "別の保存先"))

    def run():
        assert Path.cwd() == root
        assert os.environ["MULTI_LLM_COMPANION_HOME"] == str(tmp_path / "別の保存先")
        return exit_code

    monkeypatch.setitem(sys.modules, "multi_llm_companion.main", SimpleNamespace(main=run))
    monkeypatch.setattr(launch_windowed, "_show_error", lambda: pytest.fail("重複したエラー案内"))
    assert launch_windowed.main() == exit_code


@pytest.mark.parametrize("failure", ["import", "startup"])
def test_windowed_launch_failure_gives_safe_guidance(monkeypatch, failure):
    monkeypatch.chdir(Path.cwd())
    messages = []

    def fail():
        raise RuntimeError("artificial-secret-never-display")

    module = None if failure == "import" else SimpleNamespace(main=fail)
    monkeypatch.setitem(sys.modules, "multi_llm_companion.main", module)
    user32 = SimpleNamespace(MessageBoxW=lambda *args: messages.append(args))
    monkeypatch.setattr(
        launch_windowed.ctypes, "windll", SimpleNamespace(user32=user32), raising=False
    )
    assert launch_windowed.main() == 1
    assert len(messages) == 1
    assert "run_debug.bat" in messages[0][1]
    assert "setup_windows.bat" in messages[0][1]
    assert "artificial-secret" not in str(messages)


def test_shortcut_requires_installed_launcher(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_shortcut(tmp_path)
    assert not (tmp_path / "起動.lnk").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windowsのショートカットを検証する")
def test_shortcut_resolves_unicode_spaces_and_quotes(tmp_path):
    root = tmp_path / "日本語 app ' & $test"
    (root / ".venv/Scripts").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / ".venv/Scripts/pythonw.exe").touch()
    (root / "scripts/launch_windowed.py").touch()
    create_shortcut(root)
    shortcut = root / "起動.lnk"
    assert shortcut.is_file()
    # コピー先での再セットアップは、そのフォルダーへ参照先を更新する。
    moved = tmp_path / "移動後"
    shutil.copytree(root, moved)
    create_shortcut(moved)
    command = (
        "$shell = New-Object -ComObject WScript.Shell; "
        f"$link = $shell.CreateShortcut({_literal(str(moved / '起動.lnk'))}); "
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
        "@{target=$link.TargetPath; arguments=$link.Arguments; cwd=$link.WorkingDirectory} "
        "| ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        check=True,
        capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
        timeout=30,
    )
    link = json.loads(result.stdout.decode("utf-8-sig"))
    assert Path(link["target"]) == moved / ".venv/Scripts/pythonw.exe"
    assert link["arguments"] == f'"{moved / "scripts/launch_windowed.py"}"'
    assert Path(link["cwd"]) == moved
