from __future__ import annotations

import os
from typing import cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from multi_llm_companion.ui.expression_library_dialog import (
    ExpressionLibraryDialog,
    mouth_shape_for_morph,
)
from multi_llm_companion.ui.models import UiAvatarModelPreview, UiExpressionPreset


def test_expression_library_filters_real_morph_kinds_and_previews_without_saving() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    preview = UiAvatarModelPreview(
        model_name="表情カタログPMX",
        vertex_count=1,
        face_count=0,
        points=((0.0, 0.0, 0.0),),
        mouth_morphs={"a": ((0, 0.0, 0.2, 0.0),)},
        morph_names=("あ", "笑い", "ベレーOFF", "未対応"),
        named_vertex_morphs={
            "あ": ((0, 0.0, 0.2, 0.0),),
            "笑い": ((0, 0.1, 0.0, 0.0),),
        },
        named_material_morphs={"ベレーOFF": ()},
    )
    previews: list[str] = []
    stops: list[bool] = []
    dialog = ExpressionLibraryDialog(preview, previews.append, lambda: stops.append(True))

    assert dialog.morph_list.count() == 4
    dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData("mouth"))
    assert dialog.morph_list.count() == 1
    assert "あ" in dialog.morph_list.currentItem().text()
    QTest.mouseClick(dialog.preview_button, Qt.MouseButton.LeftButton)
    assert previews == ["あ"]

    dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData("material"))
    assert dialog.morph_list.count() == 1
    assert "ベレーOFF" in dialog.detail_label.text()
    assert "GPU描画" in dialog.detail_label.text()

    dialog.reject()
    assert stops == [True]
    assert mouth_shape_for_morph(preview, "あ") == "a"
    assert mouth_shape_for_morph(preview, "笑い") == ""
    app.processEvents()


def test_expression_library_saves_weighted_combination_for_one_emotion(
    monkeypatch,
) -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    preview = UiAvatarModelPreview(
        model_name="表情保存PMX",
        vertex_count=1,
        face_count=0,
        points=((0.0, 0.0, 0.0),),
        morph_names=("笑い", "照れ", "あ"),
        mouth_morphs={"a": ((0, 0.0, 0.2, 0.0),)},
        named_vertex_morphs={
            "笑い": ((0, 0.1, 0.0, 0.0),),
            "照れ": ((0, 0.0, 0.1, 0.0),),
            "あ": ((0, 0.0, 0.2, 0.0),),
        },
    )
    saved: list[tuple[str, str, dict[str, float]]] = []
    monkeypatch.setattr(QMessageBox, "information", lambda *_args: None)
    dialog = ExpressionLibraryDialog(
        preview,
        lambda _name: None,
        lambda: None,
        save_preset=lambda name, emotion, morphs: saved.append((name, emotion, morphs)),
    )

    dialog.search_edit.setText("笑い")
    QTest.mouseClick(dialog.add_button, Qt.MouseButton.LeftButton)
    dialog.strength_spin.setValue(0.65)
    dialog.search_edit.setText("照れ")
    QTest.mouseClick(dialog.add_button, Qt.MouseButton.LeftButton)
    dialog.preset_name_edit.setText("happy_rich")
    dialog.emotion_combo.setCurrentIndex(dialog.emotion_combo.findData("happy"))
    QTest.mouseClick(dialog.save_button, Qt.MouseButton.LeftButton)

    assert saved == [("happy_rich", "happy", {"笑い": 0.65, "照れ": 1.0})]
    dialog.search_edit.setText("あ")
    assert dialog.add_button.isEnabled() is False
    dialog.reject()
    app.processEvents()


def test_expression_library_previews_current_weighted_combination() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    preview = UiAvatarModelPreview(
        model_name="組合せ確認PMX",
        vertex_count=1,
        face_count=0,
        points=((0.0, 0.0, 0.0),),
        morph_names=("笑い", "照れ"),
        named_vertex_morphs={
            "笑い": ((0, 0.1, 0.0, 0.0),),
            "照れ": ((0, 0.0, 0.1, 0.0),),
        },
    )
    previews: list[dict[str, float]] = []
    dialog = ExpressionLibraryDialog(
        preview,
        lambda _name: None,
        lambda: None,
        start_combination_preview=previews.append,
    )

    QTest.mouseClick(dialog.add_button, Qt.MouseButton.LeftButton)
    dialog.strength_spin.setValue(0.5)
    dialog.search_edit.setText("照れ")
    QTest.mouseClick(dialog.add_button, Qt.MouseButton.LeftButton)
    dialog.strength_spin.setValue(0.2)
    QTest.mouseClick(dialog.combination_preview_button, Qt.MouseButton.LeftButton)

    assert previews == [{"笑い": 0.5, "照れ": 0.2}]
    dialog.reject()
    app.processEvents()


def test_expression_library_loads_duplicates_updates_and_deletes_saved_presets(
    monkeypatch,
) -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    preview = UiAvatarModelPreview(
        model_name="保存済み表情PMX",
        vertex_count=1,
        face_count=0,
        points=((0.0, 0.0, 0.0),),
        morph_names=("笑い", "照れ", "困る"),
        named_vertex_morphs={
            "笑い": ((0, 0.1, 0.0, 0.0),),
            "照れ": ((0, 0.0, 0.1, 0.0),),
            "困る": ((0, -0.1, 0.0, 0.0),),
        },
    )
    presets = [
        UiExpressionPreset("happy_soft", (("笑い", 0.7), ("照れ", 0.2)), ("happy",)),
        UiExpressionPreset("sad_soft", (("困る", 0.8),), ("sad",)),
    ]
    saved: list[tuple[str, str, dict[str, float]]] = []
    deleted: list[str] = []

    def save(name: str, emotion: str, weights: dict[str, float]) -> None:
        normalized = name.strip().lower()
        saved.append((normalized, emotion, weights))
        presets[:] = [preset for preset in presets if preset.name != normalized]
        presets.append(
            UiExpressionPreset(normalized, tuple(weights.items()), (emotion,) if emotion else ())
        )

    def delete(name: str) -> None:
        deleted.append(name)
        presets[:] = [preset for preset in presets if preset.name != name]

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *_args: None)
    dialog = ExpressionLibraryDialog(
        preview,
        lambda _name: None,
        lambda: None,
        save_preset=save,
        load_presets=lambda: list(presets),
        delete_preset=delete,
    )

    assert dialog.saved_preset_list.count() == 2
    QTest.mouseClick(dialog.load_preset_button, Qt.MouseButton.LeftButton)
    assert dialog.preset_name_edit.text() == "happy_soft"
    assert dialog.emotion_combo.currentData() == "happy"
    assert dialog.save_button.text() == "同名のプリセットを上書き"
    assert dialog._combination == {"笑い": 0.7, "照れ": 0.2}

    QTest.mouseClick(dialog.duplicate_preset_button, Qt.MouseButton.LeftButton)
    assert dialog.preset_name_edit.text() == "happy_soft_copy"
    assert dialog.save_button.text() == "新しいプリセットとして保存"
    QTest.mouseClick(dialog.save_button, Qt.MouseButton.LeftButton)
    assert saved == [("happy_soft_copy", "happy", {"笑い": 0.7, "照れ": 0.2})]

    dialog.saved_preset_list.setCurrentRow(1)
    assert "sad_soft" in dialog.saved_preset_list.currentItem().text()
    QTest.mouseClick(dialog.delete_preset_button, Qt.MouseButton.LeftButton)
    assert deleted == ["sad_soft"]
    assert dialog.saved_preset_list.count() == 2
    dialog.reject()
    app.processEvents()


def test_expression_library_requires_explicit_choice_for_legacy_multi_emotion_preset(
    monkeypatch,
) -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    preview = UiAvatarModelPreview(
        model_name="複数分類の表情PMX",
        vertex_count=1,
        face_count=0,
        points=((0.0, 0.0, 0.0),),
        morph_names=("笑い",),
        named_vertex_morphs={"笑い": ((0, 0.1, 0.0, 0.0),)},
    )
    preset = UiExpressionPreset(
        "shared_smile",
        (("笑い", 0.6),),
        ("neutral", "happy"),
    )
    saved: list[tuple[str, str, dict[str, float]]] = []
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(message),
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *_args: None)
    dialog = ExpressionLibraryDialog(
        preview,
        lambda _name: None,
        lambda: None,
        save_preset=lambda name, emotion, morphs: saved.append((name, emotion, morphs)),
        load_presets=lambda: [preset],
    )

    QTest.mouseClick(dialog.load_preset_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(dialog.save_button, Qt.MouseButton.LeftButton)
    assert saved == []
    assert warnings == [
        "このプリセットは複数の大分類に登録されていました。"
        "保存前に大分類を一つ選ぶか、保存のみを選んでください。"
    ]

    dialog.emotion_combo.activated.emit(0)
    QTest.mouseClick(dialog.save_button, Qt.MouseButton.LeftButton)
    assert saved == [("shared_smile", "", {"笑い": 0.6})]
    dialog.reject()
    app.processEvents()
