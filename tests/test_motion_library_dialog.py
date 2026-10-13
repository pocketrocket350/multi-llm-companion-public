from __future__ import annotations

import os
from dataclasses import replace
from typing import cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from multi_llm_companion.ui.models import (
    UiMotionLibraryItem,
    UiVmdBoneFrame,
    UiVmdMotion,
)
from multi_llm_companion.ui.motion_library_dialog import MotionLibraryDialog


class _MotionBackend:
    def __init__(self) -> None:
        types = ("idle", "talk", "reaction", "dance")
        energies = ("low", "medium", "high")
        self.items = [
            UiMotionLibraryItem(
                motion_id=f"motion-{index:02d}",
                motion_type=types[index % len(types)],
                action=f"action-{index % 5}",
                moods=("calm" if index % 2 == 0 else "happy",),
                energy=energies[index % len(energies)],
                enabled=index != 19,
                available=index != 18,
                status_text="読込失敗" if index == 18 else "利用可能",
                audio_required=index % 4 == 1,
                frame_count=30 + index,
            )
            for index in range(20)
        ]
        self.preview_ids: list[str] = []
        self.enabled_changes: list[tuple[str, bool]] = []

    def list_motion_library(self) -> list[UiMotionLibraryItem]:
        return list(self.items)

    def load_motion_preview(self, motion_id: str) -> UiVmdMotion:
        self.preview_ids.append(motion_id)
        return UiVmdMotion(
            motion_id,
            30,
            (UiVmdBoneFrame("センター", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),),
            loop=False,
        )

    def set_motion_enabled(self, motion_id: str, enabled: bool) -> None:
        self.enabled_changes.append((motion_id, enabled))
        self.items = [
            replace(item, enabled=enabled) if item.motion_id == motion_id else item
            for item in self.items
        ]


def test_motion_library_lists_twenty_and_filters_type_mood_energy_and_text() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = _MotionBackend()
    dialog = MotionLibraryDialog(backend, lambda _motion: None, lambda: None)

    assert dialog.motion_list.count() == 20

    dialog.search_edit.setText("motion-03")
    assert dialog.motion_list.count() == 1
    dialog.search_edit.clear()

    dialog.type_combo.setCurrentIndex(dialog.type_combo.findData("dance"))
    assert dialog.motion_list.count() == 5
    dialog.type_combo.setCurrentIndex(0)

    dialog.mood_combo.setCurrentIndex(dialog.mood_combo.findData("calm"))
    assert dialog.motion_list.count() == 10
    dialog.mood_combo.setCurrentIndex(0)

    dialog.energy_combo.setCurrentIndex(dialog.energy_combo.findData("high"))
    assert dialog.motion_list.count() == 6

    app.processEvents()
    dialog.close()


def test_motion_library_preview_toggle_and_unavailable_reason_are_separate() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = _MotionBackend()
    previews: list[UiVmdMotion] = []
    stop_count = 0

    def stop_preview() -> None:
        nonlocal stop_count
        stop_count += 1

    dialog = MotionLibraryDialog(backend, previews.append, stop_preview)
    dialog.show()
    app.processEvents()
    assert backend.preview_ids == ["motion-00"]
    dialog.motion_list.setCurrentRow(3)
    assert backend.preview_ids == ["motion-00", "motion-03"]
    QTest.mouseClick(dialog.preview_button, Qt.MouseButton.LeftButton)

    assert backend.preview_ids == ["motion-00", "motion-03", "motion-03"]
    assert [motion.name for motion in previews] == backend.preview_ids

    dialog.enabled_check.setChecked(False)
    assert backend.enabled_changes == [("motion-03", False)]
    assert dialog.catalog_changed is True
    assert stop_count == 1
    assert len(previews) == 3

    dialog.search_edit.setText("motion-18")
    assert dialog.motion_list.count() == 1
    assert dialog.status_label.text() == "読込失敗"
    assert dialog.preview_button.isEnabled() is False
    assert stop_count == 2

    app.processEvents()
    dialog.close()


def test_selection_previews_disabled_motion_once_without_changing_catalog():
    app = QApplication.instance() or QApplication([])
    backend = _MotionBackend()
    original = backend.load_motion_preview
    backend.load_motion_preview = lambda name: replace(original(name), loop=True)
    previews = []
    dialog = MotionLibraryDialog(backend, previews.append, lambda: None)
    dialog.show()
    app.processEvents()
    dialog.motion_list.setCurrentRow(19)
    item = dialog.motion_list.item(19)
    assert item.foreground().color().name() == "#718396"
    assert "使用しない" in item.text()
    assert item.flags() & Qt.ItemFlag.ItemIsEnabled
    assert [motion.name for motion in previews] == ["motion-00", "motion-19"]
    assert all(not motion.loop for motion in previews)
    assert not backend.enabled_changes
    dialog.motion_list.setCurrentRow(19)
    assert len(previews) == 2
    dialog.stop_button.click()
    dialog.hide()
    dialog.show()
    app.processEvents()
    assert len(previews) == 2
    dialog.close()


def test_original_japanese_name_is_visible_and_searchable_without_sample_prefix() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = _MotionBackend()
    original_name = "ot0510_待機_両手後ろ_まばたき付き.vmd"
    backend.items = [
        replace(backend.items[0], motion_id="sample_otokam_048", source_name=original_name)
    ]
    dialog = MotionLibraryDialog(backend, lambda _: None, lambda: None)
    dialog.search_edit.setText("両手後ろ")
    assert dialog.motion_list.count() == 1
    assert original_name in dialog.motion_list.item(0).text()
    assert "sample_" not in dialog.motion_list.item(0).text()
    assert dialog.source_name_label.text() == original_name
    assert dialog.id_label.text() == "sample_otokam_048"
    dialog.search_edit.setText("sample_otokam_048")
    assert dialog.motion_list.count() == 1
    app.processEvents()
    dialog.close()
