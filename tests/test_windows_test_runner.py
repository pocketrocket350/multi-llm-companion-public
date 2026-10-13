"""Windowsバッチの改行と検証起動の文字コードを確認する。"""

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "filename", ["test.bat", "run_phase9_acceptance.bat", "setup_windows.bat", "run.bat"]
)
def test_windows_batch_uses_consistent_line_endings(filename: str) -> None:
    content = (Path(__file__).resolve().parents[1] / filename).read_bytes()
    assert b"\r\n" in content
    assert b"\n" not in content.replace(b"\r\n", b"")
    assert content.decode("utf-8").splitlines()[1] == "chcp 65001 >nul"
    if filename == "run_phase9_acceptance.bat":
        assert content.isascii()
