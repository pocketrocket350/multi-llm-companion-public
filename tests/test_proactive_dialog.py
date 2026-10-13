from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from multi_llm_companion.services.proactive import ProactiveSettings
from multi_llm_companion.ui.proactive_dialog import ProactiveSettingsDialog


def test_dialog_exposes_opt_in_quiet_hours_and_rate_limits() -> None:
    QApplication.instance() or QApplication([])
    dialog = ProactiveSettingsDialog(
        ProactiveSettings(
            enabled=True,
            read_aloud=True,
            quiet_start_hour=21,
            quiet_end_hour=7,
            minimum_idle_minutes=45,
            minimum_interval_minutes=180,
        )
    )

    assert dialog.selected_settings() == ProactiveSettings(
        enabled=True,
        read_aloud=True,
        quiet_start_hour=21,
        quiet_end_hour=7,
        minimum_idle_minutes=45,
        minimum_interval_minutes=180,
    )
    dialog.close()
