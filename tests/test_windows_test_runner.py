"""Windowsバッチの改行と検証起動の文字コードを確認する。"""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_BATCHES = {
    "test.bat",
    "setup_windows.bat",
    "run.bat",
    "run_debug.bat",
    "download_whisper_model.bat",
}
# 開発用の検証バッチは、同梱される環境で従来の条件を確認する。
_DEVELOPMENT_BATCHES = {path.name for path in _ROOT.glob("run_phase9_acceptance.bat")}


@pytest.mark.parametrize(
    "filename",
    sorted(_REQUIRED_BATCHES | _DEVELOPMENT_BATCHES),
)
def test_windows_batch_uses_consistent_line_endings(filename: str) -> None:
    content = (_ROOT / filename).read_bytes()
    assert b"\r\n" in content
    assert b"\n" not in content.replace(b"\r\n", b"")
    assert content.decode("utf-8").splitlines()[1] == "chcp 65001 >nul"
    if filename == "run_phase9_acceptance.bat":
        assert content.isascii()
