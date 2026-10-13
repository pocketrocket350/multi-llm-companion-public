from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from multi_llm_companion.ui.lip_sync_monitor_widget import LipSyncMonitorWidget


def test_lip_sync_monitor_draws_a_large_vowel_shape() -> None:
    app = QApplication.instance() or QApplication([])
    widget = LipSyncMonitorWidget()
    widget.resize(200, 100)

    widget.set_lip_sync("o")
    app.processEvents()

    assert widget.mouth_shape == "o"
    assert not widget.grab().isNull()
    widget.set_lip_sync("unknown")
    assert widget.mouth_shape == "neutral"
    widget.close()
