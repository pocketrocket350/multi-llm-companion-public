from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from multi_llm_companion.ui.avatar_widget import AvatarWidget


def test_avatar_widget_accepts_common_response_values() -> None:
    app = QApplication.instance() or QApplication([])
    widget = AvatarWidget()
    widget.resize(200, 260)
    widget.set_avatar_state("happy", 0.7, "wave", "left")
    app.processEvents()

    assert widget.emotion == "happy"
    assert widget.intensity == 0.7
    assert widget.gesture == "wave"
    assert widget.gaze == "left"
    assert widget._timer.isActive() is True
    assert not widget.grab().isNull()
    widget.close()


def test_avatar_widget_returns_to_neutral_for_unknown_emotion() -> None:
    widget = AvatarWidget()

    widget.set_avatar_state("unknown", 5.0, "", "")

    assert widget.emotion == "neutral"
    assert widget.intensity == 1.0


def test_avatar_widget_can_start_and_stop_motion() -> None:
    widget = AvatarWidget()
    widget.set_avatar_state("neutral", 0.0, "nod", "right")
    widget.set_speaking(True)

    assert widget.is_speaking is True
    assert widget._timer.isActive() is True
    widget._advance_frame()
    assert widget._gaze_x > 0

    widget.set_motion_enabled(False)

    assert widget.motion_enabled is False
    assert widget._timer.isActive() is False
    assert widget.is_speaking is True
    widget.close()


def test_avatar_widget_has_idle_motion_and_blinks() -> None:
    widget = AvatarWidget()

    assert widget._timer.isActive() is True
    widget._idle_phase = 8.0
    assert widget._eye_open_ratio() < 0.3
    widget._advance_frame()
    assert widget._idle_phase > 8.0
    assert widget._breath_offset(1.0) != 0
    widget.close()


def test_avatar_widget_accepts_estimated_vowel_mouth_shape() -> None:
    widget = AvatarWidget()

    widget.set_lip_sync(True, "u")

    assert widget.is_speaking is True
    assert widget._mouth_shape == "u"
    widget.set_lip_sync(False, "unknown")
    assert widget.is_speaking is False
    assert widget._mouth_shape == "neutral"
    widget.close()
