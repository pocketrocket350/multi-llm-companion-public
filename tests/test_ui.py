from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event
from typing import cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from multi_llm_companion.domain import ConversationState
from multi_llm_companion.services.proactive import ProactiveSettings
from multi_llm_companion.ui.chat_window import (
    ChatWindow,
    _format_emotion_morph,
    _format_mmd_command,
)
from multi_llm_companion.ui.memory_dialog import MemoryDialog
from multi_llm_companion.ui.models import (
    BackendStatus,
    MemorySettings,
    SendOutcome,
    SpeechSettings,
    UiAudioDevice,
    UiAvatar3DStatus,
    UiAvatarAction,
    UiAvatarAppearance,
    UiAvatarBone,
    UiAvatarModelPreview,
    UiConversation,
    UiEmotionState,
    UiLipSyncState,
    UiMemory,
    UiMessage,
    UiMmdAvatarCommand,
    UiTokenUsageSummary,
)
from multi_llm_companion.ui.settings_dialog import SettingsDialog
from multi_llm_companion.ui.workers import ProactiveWorker, VoiceStartWorker


@pytest.fixture(autouse=True)
def isolate_window_settings(tmp_path, monkeypatch) -> None:
    """画面テストのQSettingsを利用者のWindows設定から分離する。"""

    settings_path = tmp_path / "desktop-companion.ini"

    def create_test_settings(_organization: str, _application: str) -> QSettings:
        return QSettings(str(settings_path), QSettings.Format.IniFormat)

    monkeypatch.setattr(
        "multi_llm_companion.ui.chat_window.QSettings",
        create_test_settings,
    )


class FakeBackend:
    def __init__(self) -> None:
        self.messages: list[UiMessage] = []
        self.has_conversation = False
        self.listening = False
        self.memories: list[UiMemory] = []
        self.memory_enabled = True
        self.memory_auto_approve_candidates = True
        self.created_titles: list[str] = []
        self.conversation_title = "画面テスト"
        self.voice_speaking = False
        self.voice_stop_count = 0
        self.finish_recognition_count = 0
        self.spoken_texts: list[str] = []
        self.proactive_spoken_texts: list[str] = []
        self.proactive_conversation_ids: list[str] = []
        self.proactive_cancel_count = 0
        self.proactive_outcome = SendOutcome(
            speech="少し静かな時間だね。",
            input_tokens=2,
            output_tokens=3,
            gesture="",
            gaze="forward",
        )
        self.lip_sync_state = UiLipSyncState(False)
        self.saved_expression_presets: list[tuple[str, str, dict[str, float]]] = []

    def status(self) -> BackendStatus:
        return BackendStatus("mock", "mock-model", "API不要")

    def provider_names(self) -> list[str]:
        return ["mock"]

    def current_provider_name(self) -> str:
        return "mock"

    def list_conversations(self) -> list[UiConversation]:
        return [UiConversation("test-id", self.conversation_title)] if self.has_conversation else []

    def load_messages(self, conversation_id: str) -> list[UiMessage]:
        assert conversation_id == "test-id"
        return list(self.messages)

    def create_conversation(self, title: str) -> str:
        self.created_titles.append(title)
        self.conversation_title = title
        self.has_conversation = True
        return "test-id"

    def delete_conversation(self, conversation_id: str) -> None:
        assert conversation_id == "test-id"

    def generate_proactive_turn(self, conversation_id: str) -> SendOutcome:
        assert conversation_id == "test-id"
        self.proactive_conversation_ids.append(conversation_id)
        self.messages.append(UiMessage("assistant", self.proactive_outcome.speech))
        return self.proactive_outcome

    def cancel_proactive_turn(self) -> None:
        self.proactive_cancel_count += 1

    def memory_settings(self) -> MemorySettings:
        return MemorySettings(self.memory_enabled, self.memory_auto_approve_candidates)

    def conversation_context_status(self) -> str:
        return "直近 2 件"

    def set_memory_enabled(self, enabled: bool) -> None:
        self.memory_enabled = enabled

    def set_memory_auto_approve_candidates(self, enabled: bool) -> None:
        self.memory_auto_approve_candidates = enabled

    def list_memories(self, status: str | None = None) -> list[UiMemory]:
        return [item for item in self.memories if status is None or item.status == status]

    def approve_memory(self, memory_id: int, content: str, category: str) -> UiMemory:
        updated = UiMemory(memory_id, content, category, "approved")
        self.memories = [updated if item.id == memory_id else item for item in self.memories]
        return updated

    def update_memory(self, memory_id: int, content: str, category: str) -> UiMemory:
        current = next(item for item in self.memories if item.id == memory_id)
        updated = UiMemory(
            memory_id,
            content,
            category,
            current.status,
            current.created_at,
            current.updated_at,
            current.source_conversation_id,
        )
        self.memories = [updated if item.id == memory_id else item for item in self.memories]
        return updated

    def delete_memory(self, memory_id: int) -> None:
        self.memories = [item for item in self.memories if item.id != memory_id]

    def delete_all_memories(self) -> int:
        count = len(self.memories)
        self.memories = []
        return count

    def emotion_state(self, conversation_id: str) -> UiEmotionState | None:
        assert conversation_id == "test-id"
        return None

    def avatar_action(self, conversation_id: str) -> UiAvatarAction:
        assert conversation_id == "test-id"
        return UiAvatarAction("", "")

    def avatar_appearance(self) -> UiAvatarAppearance:
        return UiAvatarAppearance("テスト案内役", "#684765", "#b87aa4", "#d99ac1")

    def avatar_3d_status(self) -> UiAvatar3DStatus:
        return UiAvatar3DStatus("3D連携Mock", "接続済み（モデル未読み込み）", True)

    def avatar_model_preview(self) -> UiAvatarModelPreview | None:
        return None

    def mmd_avatar_command(self) -> UiMmdAvatarCommand | None:
        return None

    def save_expression_preset(
        self,
        preset_name: str,
        emotion: str,
        morph_weights: dict[str, float],
    ) -> None:
        self.saved_expression_presets.append((preset_name, emotion, morph_weights))

    def token_usage_summary(self, conversation_id: str | None = None) -> UiTokenUsageSummary:
        if conversation_id is None:
            return UiTokenUsageSummary(12, 8, 20, 2)
        assert conversation_id == "test-id"
        return UiTokenUsageSummary(6, 4, 10, 1)

    def change_provider(self, provider_name: str) -> BackendStatus:
        assert provider_name == "mock"
        return self.status()

    def start_speech_recognition(
        self,
        on_recognized: Callable[[str], None],
        on_error: Callable[[str], None],
        on_partial: Callable[[str], None],
        on_level: Callable[[int], None],
    ) -> None:
        del on_error
        self.listening = True
        on_level(42)
        on_partial("音声")
        on_recognized("音声入力")
        self.listening = False

    def stop_speech_recognition(self) -> None:
        self.listening = False

    def finish_speech_recognition(self) -> None:
        self.finish_recognition_count += 1
        self.listening = False

    def is_voice_speaking(self) -> bool:
        return self.voice_speaking

    def voice_lip_sync_state(self) -> UiLipSyncState:
        return self.lip_sync_state

    def stop_voice_output(self) -> None:
        self.voice_stop_count += 1
        self.voice_speaking = False
        self.lip_sync_state = UiLipSyncState(False)

    def start_voice_output(self, text: str) -> str | None:
        self.spoken_texts.append(text)
        return None

    def start_proactive_voice_output(self, text: str) -> str | None:
        self.spoken_texts.append(text)
        self.proactive_spoken_texts.append(text)
        return None

    def is_speech_recognition_available(self) -> bool:
        return True

    def is_listening(self) -> bool:
        return self.listening

    def speech_settings(self) -> SpeechSettings:
        return SpeechSettings("whisper", None, [UiAudioDevice(1, "テストマイク")])

    def change_speech_settings(
        self,
        engine: str,
        input_device: int | str | None,
        silence_threshold: float,
        silence_duration: float,
        max_recording_duration: float,
        partial_interval: float,
    ) -> BackendStatus:
        assert engine in {"whisper", "vosk"}
        assert input_device is None or isinstance(input_device, int)
        assert 0.001 <= silence_threshold <= 1.0
        assert 0.3 <= silence_duration <= 10.0
        assert 3.0 <= max_recording_duration <= 120.0
        assert 1.0 <= partial_interval <= 15.0
        return self.status()

    def stream_message(
        self,
        conversation_id: str,
        user_text: str,
        on_chunk: Callable[[str], None],
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> SendOutcome:
        assert conversation_id == "test-id"
        self.messages.append(UiMessage("user", user_text))
        speech = "モック回答"
        for chunk in ("モック", "回答"):
            on_chunk(chunk)
        self.messages.append(UiMessage("assistant", speech))
        return SendOutcome(
            speech=speech,
            input_tokens=2,
            output_tokens=3,
            gesture="wave",
            gaze="left",
            emotion_state=UiEmotionState("happy", 0.6),
        )


def test_chat_window_settings_are_isolated_from_the_user_scope(tmp_path) -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    window = ChatWindow(FakeBackend())

    assert Path(window._settings.fileName()).resolve().is_relative_to(tmp_path.resolve())

    app.processEvents()
    window.close()


def test_conversation_state_updates_existing_ui_and_safe_trace(monkeypatch) -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    records: list[tuple[str, str, dict[str, object]]] = []
    monkeypatch.setattr(
        "multi_llm_companion.ui.chat_window.write_lip_sync_trace",
        lambda source, event, **fields: records.append((source, event, fields)),
    )
    window = ChatWindow(FakeBackend())

    for state in (
        ConversationState.LISTENING,
        ConversationState.THINKING,
        ConversationState.REACTING,
        ConversationState.SPEAKING,
        ConversationState.IDLE,
        ConversationState.INTERRUPTED,
    ):
        window._set_conversation_state(state)
        assert window.conversation_state_label.text() == f"会話状態: {state.display_name}"
        assert window._conversation_state == state

    transitions = [record for record in records if record[1] == "conversation_state_transition"]
    assert len(transitions) == 6
    for source, _event, fields in transitions:
        assert source == "companion"
        assert set(fields) == {"generation", "previous_state", "state", "duration_ms"}
        assert fields["generation"] == 0
        assert isinstance(fields["duration_ms"], int)
    app.processEvents()
    window.close()


def test_speaking_state_waits_for_actual_voice_playback() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    window = ChatWindow(backend)

    window._pending_voice_text = "確認用"
    window._sync_voice_animation()
    assert window._conversation_state == ConversationState.REACTING

    backend.voice_speaking = True
    backend.lip_sync_state = UiLipSyncState(True, "a")
    window._sync_voice_animation()
    assert window._conversation_state == ConversationState.SPEAKING

    backend.voice_speaking = False
    backend.lip_sync_state = UiLipSyncState(False)
    window._pending_voice_text = None
    window._sync_voice_animation()
    assert window._conversation_state == ConversationState.IDLE
    app.processEvents()
    window.close()


def test_open_motion_manager_invalidates_voice_and_stops_preview(monkeypatch) -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    backend.voice_speaking = True
    window = ChatWindow(backend)
    window.pmx_preview_widget.set_voice_active(True)
    preview_stops: list[bool] = []
    monkeypatch.setattr(
        window.pmx_preview_widget,
        "stop_motion_preview",
        lambda: preview_stops.append(True),
    )

    class FakeMotionDialog:
        catalog_changed = False

        def __init__(self, *_args: object) -> None:
            pass

        def exec(self) -> int:
            assert backend.voice_speaking is False
            return 0

    monkeypatch.setattr(
        "multi_llm_companion.ui.chat_window.MotionLibraryDialog",
        FakeMotionDialog,
    )
    monkeypatch.setattr(window, "_exec_avatar_manager", lambda dialog: dialog.exec())

    window._open_motion_manager()

    assert backend.voice_stop_count == 1
    assert preview_stops == [True]
    assert window.motion_library_button.text() == "モーションを管理"
    app.processEvents()
    window.close()


def test_stream_invalidation_clears_old_memory_feedback() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    window = ChatWindow(FakeBackend())
    window._set_used_memories([UiMemory(1, "古い記憶", "profile", "approved")])
    window.memory_status_label.setText("保存した記憶: 古い記憶")
    window._busy = True
    window._messages.append(UiMessage("assistant", "古い応答"))

    window._invalidate_active_stream()

    assert window._used_memories == []
    assert window.memory_details_button.isVisible() is False
    assert window.memory_status_label.text() == "保存した記憶: 使用中"
    window._set_busy(False)
    app.processEvents()
    window.close()


def test_new_conversation_clears_old_mmd_and_memory_feedback() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    window = ChatWindow(FakeBackend())
    command = UiMmdAvatarCommand("happy", 0.6, "smile", "left", "head", "wave", "hand")
    window._current_mmd_command = command
    window.pmx_preview_widget.set_mmd_command(command)
    window._set_used_memories([UiMemory(1, "前の会話の記憶", "profile", "approved")])

    window._start_new_conversation()

    assert window._current_mmd_command is None
    assert window._used_memories == []
    assert window.memory_details_button.isVisible() is False
    app.processEvents()
    window.close()


def test_chat_window_sends_without_allowing_double_send() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    window = ChatWindow(backend)
    window.input_edit.setPlainText("こんにちは")

    QTest.keyClick(window.input_edit, Qt.Key.Key_Return)

    assert window.busy_label.text() == "処理中…"
    assert window.send_button.isEnabled() is False
    deadline = time.monotonic() + 3
    while window.busy_label.text() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert window.busy_label.text() == ""
    assert "あなた\nこんにちは" in window.chat_view.toPlainText()
    assert f"{window._avatar_display_name}\nモック回答" in window.chat_view.toPlainText()
    assert window.usage_label.text() == (
        "トークン使用量\n"
        "今回: 入 2 / 出 3\n"
        "会話: 計 10（入 6 / 出 4）\n"
        "保存済み: 計 20（入 12 / 出 8）"
    )
    assert window.emotion_label.text() == "感情: 喜び（60%）"
    assert window.avatar_expression_label.text() == "表情: 喜び（60%）"
    assert window.avatar_gesture_label.text() == "動作: wave"
    assert window.avatar_gaze_label.text() == "視線: left"
    assert window.avatar_widget.emotion == "happy"
    assert window.avatar_widget.gesture == "wave"
    assert window.avatar_widget.gaze == "left"
    assert window.avatar_name_label.text() == "アバター: テスト案内役"
    assert (
        window.avatar_3d_status_label.text() == "3D連携: 3D連携Mock（接続済み（モデル未読み込み））"
    )
    assert window.avatar_motion_label.text() == "動き: 常時動作中"
    assert window.avatar_widget.is_speaking is False
    assert backend.spoken_texts == ["モック回答"]
    assert window.avatar_widget.motion_enabled is True
    assert not hasattr(window, "avatar_motion_checkbox")
    assert backend.created_titles == ["こんにちは"]
    window.close()


def test_draft_and_microphone_clear_an_old_proactive_conversation_status(monkeypatch) -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    window = ChatWindow(backend)
    monkeypatch.setattr(
        window,
        "_proactive_settings",
        lambda: ProactiveSettings(enabled=True, quiet_start_hour=0, quiet_end_hour=0),
    )
    window.proactive_status_label.setText("能動会話: 発話済み")

    window.input_edit.setPlainText("下書き")

    assert window._proactive.state.last_user_activity_at is not None
    assert window.proactive_status_label.text() == "能動会話: 有効（文字のみ）"

    window.proactive_status_label.setText("能動会話: 発話済み")
    window.input_edit.clear()
    window._toggle_microphone()

    assert window.proactive_status_label.text() == "能動会話: 有効（文字のみ）"
    assert window._proactive.state.last_user_activity_at is not None
    app.processEvents()
    window.close()


def test_enabling_proactive_conversation_starts_a_fresh_wait_period() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    window = ChatWindow(FakeBackend())
    settings = ProactiveSettings(
        enabled=True,
        quiet_start_hour=0,
        quiet_end_hour=0,
        minimum_idle_minutes=5,
        minimum_interval_minutes=15,
    )

    window._proactive.note_user_activity(datetime.now() - timedelta(minutes=10))
    window._apply_proactive_settings(settings)
    started_at = window._proactive.state.last_user_activity_at

    assert started_at is not None
    assert (
        window._proactive.consider(
            settings,
            now=started_at + timedelta(minutes=4, seconds=59),
            is_busy=False,
            is_listening=False,
            is_voice_speaking=False,
            has_draft=False,
        )
        is None
    )
    assert (
        window._proactive.consider(
            settings,
            now=started_at + timedelta(minutes=5),
            is_busy=False,
            is_listening=False,
            is_voice_speaking=False,
            has_draft=False,
        )
        is not None
    )
    app.processEvents()
    window.close()


def test_proactive_turn_is_rendered_as_a_normal_assistant_message(monkeypatch) -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    backend.has_conversation = True
    window = ChatWindow(backend)
    window._current_conversation_id = "test-id"
    monkeypatch.setattr(
        window,
        "_proactive_settings",
        lambda: ProactiveSettings(enabled=True, quiet_start_hour=0, quiet_end_hour=0),
    )
    window._proactive_turn_generation = 1
    window._proactive_turn_active = True
    outcome = SendOutcome(
        speech="少し静かな時間だね。",
        input_tokens=2,
        output_tokens=3,
        gesture="",
        gaze="forward",
    )
    backend.messages.append(UiMessage("assistant", outcome.speech))

    window._proactive_turn_succeeded(1, outcome)

    assert "少し静かな時間だね。" in window.chat_view.toPlainText()
    assert window.proactive_status_label.text() == "能動会話: 発話済み"
    app.processEvents()
    window.close()


def test_proactive_turn_only_requests_voice_when_read_aloud_is_enabled(monkeypatch) -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    backend.has_conversation = True
    window = ChatWindow(backend)
    window._current_conversation_id = "test-id"
    requested: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        window,
        "_proactive_settings",
        lambda: ProactiveSettings(
            enabled=True,
            read_aloud=True,
            quiet_start_hour=0,
            quiet_end_hour=0,
        ),
    )
    monkeypatch.setattr(
        window,
        "_prepare_voice_output",
        lambda text, *, proactive=False: requested.append((text, proactive)),
    )
    window._proactive_turn_generation = 1
    window._proactive_turn_active = True

    window._proactive_turn_succeeded(
        1,
        SendOutcome(speech="少し静かな時間だね."),
    )

    assert requested == [("少し静かな時間だね.", True)]
    app.processEvents()
    window.close()


def test_proactive_worker_uses_the_llm_generation_path() -> None:
    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    completed: list[object] = []
    worker = ProactiveWorker(backend, "test-id", 3)
    worker.succeeded.connect(lambda generation, outcome: completed.append((generation, outcome)))

    worker.run()

    assert backend.proactive_conversation_ids == ["test-id"]
    assert completed == [(3, backend.proactive_outcome)]


def test_cancelled_proactive_result_is_not_rendered_or_read_aloud(monkeypatch) -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    backend.has_conversation = True
    window = ChatWindow(backend)
    window._current_conversation_id = "test-id"
    requested: list[str] = []
    monkeypatch.setattr(
        window,
        "_prepare_voice_output",
        lambda text, *, proactive=False: requested.append(text),
    )
    window._proactive_turn_generation = 1
    window._proactive_turn_active = True

    window._cancel_pending_proactive_turn()
    window._proactive_turn_succeeded(1, SendOutcome(speech="古い能動発話"))

    assert backend.proactive_cancel_count == 1
    assert "古い能動発話" not in window.chat_view.toPlainText()
    assert requested == []
    app.processEvents()
    window.close()


def test_draft_cancels_pending_proactive_voice(monkeypatch) -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    window = ChatWindow(backend)
    monkeypatch.setattr(
        window,
        "_proactive_settings",
        lambda: ProactiveSettings(enabled=True, read_aloud=True),
    )
    window._utterance_generation = 4
    window._proactive_voice_generation = 4
    window._pending_voice_text = "古い能動発話"
    window._pending_utterance_generation = 4
    window._proactive_turn_active = True

    window.input_edit.setPlainText("下書き")

    assert window._proactive_voice_generation is None
    assert window._pending_voice_text is None
    assert backend.voice_stop_count == 1
    assert backend.proactive_cancel_count == 1
    app.processEvents()
    window.close()


def test_proactive_voice_worker_uses_the_dedicated_voice_path() -> None:
    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    completed: list[object] = []
    worker = VoiceStartWorker(backend, "雨衣からの一言", 3, proactive=True)
    worker.completed.connect(completed.append)

    worker.run()

    assert backend.proactive_spoken_texts == ["雨衣からの一言"]
    assert completed == [(3, None)]


def test_chat_window_queues_completed_sentences_for_streaming_voice() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    started: list[str] = []
    enqueued: list[str] = []
    finished: list[bool] = []

    def start_stream(text: str) -> str | None:
        started.append(text)
        return None

    def enqueue_stream(text: str) -> str | None:
        enqueued.append(text)
        return None

    def finish_stream() -> None:
        finished.append(True)

    backend.start_stream_voice_output = start_stream  # type: ignore[attr-defined]
    backend.enqueue_stream_voice_output = enqueue_stream  # type: ignore[attr-defined]
    backend.finish_stream_voice_output = finish_stream  # type: ignore[attr-defined]
    window = ChatWindow(backend)
    window.input_edit.setPlainText("こんにちは")

    QTest.keyClick(window.input_edit, Qt.Key.Key_Return)

    deadline = time.monotonic() + 3
    while window.busy_label.text() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert started == ["モック回答"]
    assert enqueued == []
    assert finished == [True]
    window.close()


def test_final_stream_command_starts_selected_motion_without_restarting_voice() -> None:
    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    backend.mmd_avatar_command = lambda: UiMmdAvatarCommand(
        "neutral",
        0.0,
        "",
        "forward",
        "",
        "hair_flip",
        "",
        gesture_motion_id="hair_flip",
    )
    window = ChatWindow(backend)
    started: list[bool] = []
    window.pmx_preview_widget.begin_conversation_motion = lambda: started.append(True) or 12
    window._stream_voice_started = True

    window._message_succeeded(SendOutcome("短い返事", gesture="hair_flip"))

    assert started == [True]
    assert backend.spoken_texts == []
    window.close()


def test_voice_waits_for_the_matching_conversation_pose() -> None:
    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    window = ChatWindow(backend)
    window.pmx_preview_widget.begin_conversation_motion = lambda: 12

    window._message_succeeded(SendOutcome("姿勢を準備して話します", gesture="wave", gaze="left"))

    assert backend.spoken_texts == []
    window._conversation_pose_ready(11)
    assert backend.spoken_texts == []
    window._conversation_pose_ready(12)
    deadline = time.monotonic() + 2
    while not backend.spoken_texts and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    assert backend.spoken_texts == ["姿勢を準備して話します"]
    while window._voice_thread is not None and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    window.close()


def test_voice_state_starts_and_stops_avatar_lip_sync() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    window = ChatWindow(backend)

    backend.voice_speaking = True
    backend.lip_sync_state = UiLipSyncState(True, "a")
    window._sync_voice_animation()
    app.processEvents()

    assert window.avatar_widget.is_speaking is True
    assert window.pmx_preview_widget._voice_active is True
    assert window.avatar_motion_label.text() == "動き: 音声再生中"
    assert window.avatar_lip_sync_label.text() == "口形: 推定 あ"
    assert window.lip_sync_monitor_widget.mouth_shape == "a"
    assert window.stop_voice_button.isEnabled() is True
    window.stop_voice_button.click()
    app.processEvents()

    assert window.avatar_widget.is_speaking is False
    assert window.pmx_preview_widget._voice_active is False
    assert window.stop_voice_button.isEnabled() is False
    assert window.lip_sync_monitor_widget.mouth_shape == "neutral"
    window.close()


def test_chat_window_keeps_lip_sync_and_expression_in_diagnostics() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    preview = UiAvatarModelPreview(
        model_name="表示テスト",
        vertex_count=3,
        face_count=1,
        points=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        triangles=((0, 1, 2),),
        named_vertex_morphs={"笑い": ((0, 0.1, 0.0, 0.0),)},
    )
    backend.avatar_model_preview = lambda: preview
    backend.mmd_avatar_command = lambda: UiMmdAvatarCommand(
        "happy",
        0.8,
        "笑い",
        "forward",
        "",
        "none",
        "",
        gesture_motion_id="hair_flip",
        expression_preset="gentle_smile",
        expression_morphs=(("笑い", 0.28),),
    )
    backend.avatar_3d_status = lambda: UiAvatar3DStatus(
        "MMDランタイム基盤",
        "本格MMD描画は未実装のため、PMX簡易プレビューで表示中です。",
        True,
    )
    window = ChatWindow(backend)
    window.show()
    app.processEvents()
    window._set_avatar_lip_sync(UiLipSyncState(True, "a"), True)

    assert window.avatar_name_label.text() == (
        "アバター: テスト案内役　｜　口形: 推定 あ　｜　"
        "表情: gentle_smile（笑い 28%）　｜　MMD動作: hair_flip"
    )
    assert window.options_dialog.isAncestorOf(window.avatar_name_label)
    assert not window.avatar_name_label.isVisible()
    window.options_button.click()
    window.options_tabs.setCurrentIndex(2)
    app.processEvents()
    assert window.avatar_name_label.isVisible()
    window.close()


def test_mmd_header_and_details_use_the_loaded_motion_name() -> None:
    from multi_llm_companion.ui.models import UiVmdMotion

    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    backend.avatar_model_preview = lambda: UiAvatarModelPreview("表示テスト", 0, 0, ())
    backend.mmd_avatar_command = lambda: UiMmdAvatarCommand(
        "happy",
        0.7,
        "",
        "forward",
        "",
        "delight",
        "",
        gesture_motion_id="sample_otokam_007",
        gesture_motion=UiVmdMotion(name="ot0510_喜びジャンプ", frame_count=30, bone_frames=()),
    )
    window = ChatWindow(backend)
    for label in (window.avatar_name_label, window.mmd_command_label):
        assert "ot0510_喜びジャンプ" in label.text()
        assert "sample_otokam" not in label.text()
    window.close()


def test_voice_state_marks_api_phoneme_timing_as_audio_sync() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    window = ChatWindow(backend)

    backend.voice_speaking = True
    backend.lip_sync_state = UiLipSyncState(True, "o", is_estimated=False)
    window._sync_voice_animation()
    app.processEvents()

    assert window.avatar_lip_sync_label.text() == "口形: 音声同期 お"
    window.close()


def test_voice_state_shows_closed_mouth_during_phoneme_timeline() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    window = ChatWindow(backend)

    backend.voice_speaking = True
    backend.lip_sync_state = UiLipSyncState(
        False,
        "neutral",
        is_estimated=False,
        is_timeline_active=True,
    )
    window._sync_voice_animation()
    app.processEvents()

    assert window.avatar_lip_sync_label.text() == "口形: 音声同期 閉じる"
    window.close()


def test_conversation_state_is_forwarded_to_natural_motion() -> None:
    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    window = ChatWindow(backend)

    window._set_busy(True)
    assert window.pmx_preview_widget._natural_motion.state == "thinking"

    backend.listening = True
    window._set_busy(False)
    assert window.pmx_preview_widget._natural_motion.state == "listening"

    backend.listening = False
    backend.voice_speaking = True
    window._update_conversation_state()
    assert window.pmx_preview_widget._natural_motion.state == "speaking"

    backend.voice_speaking = False
    window._update_conversation_state()
    assert window.pmx_preview_widget._natural_motion.state == "idle"
    window.close()


def test_pose_preparation_uses_reacting_state_before_voice_starts() -> None:
    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    window = ChatWindow(backend)
    window.pmx_preview_widget.begin_conversation_motion = lambda: 123

    window._prepare_voice_output("短い発話")

    assert window._conversation_state == "reacting"
    assert window.conversation_state_label.text() == "会話状態: 姿勢を準備中"
    assert window.pmx_preview_widget._natural_motion.state == "reacting"
    window._pending_voice_text = None
    window._pending_pose_revision = None
    window._voice_pose_timeout.stop()
    window.close()


def test_speech_pause_requests_one_neck_nod() -> None:
    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    window = ChatWindow(backend)
    nods: list[bool] = []
    window.pmx_preview_widget._natural_motion.set_state("speaking")
    window.pmx_preview_widget._natural_motion.phrase_boundary = lambda: nods.append(True)

    window._set_avatar_lip_sync(
        UiLipSyncState(True, "a", is_timeline_active=True),
        True,
    )
    window._set_avatar_lip_sync(
        UiLipSyncState(False, "neutral", is_timeline_active=True),
        True,
    )
    window._set_avatar_lip_sync(
        UiLipSyncState(False, "neutral", is_timeline_active=True),
        True,
    )

    assert nods == [True]
    window.close()


def test_new_user_input_interrupts_voice_output_before_sending() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    backend.voice_speaking = True
    window = ChatWindow(backend)
    window.input_edit.setPlainText("次の質問です")
    motion_interrupts: list[bool] = []
    window.pmx_preview_widget.interrupt_motion = lambda: motion_interrupts.append(True)

    window._send_message()

    assert backend.voice_stop_count == 1
    assert motion_interrupts == [True]
    deadline = time.monotonic() + 3
    while window.busy_label.text() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    window.close()


def test_interrupt_before_voice_start_invalidates_pending_pose_and_all_outputs() -> None:
    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    window = ChatWindow(backend)
    window.pmx_preview_widget.begin_conversation_motion = lambda: 42
    window._set_avatar_action("wave", "left", 0.5, 0.4, "適用中")
    window._set_emotion_state(UiEmotionState("happy", 0.8), 0.6)

    window._prepare_voice_output("まだ開始していない音声")
    prepared_generation = window._utterance_generation
    window._stop_voice_output()
    window._conversation_pose_ready(42)

    assert window._utterance_generation == prepared_generation + 1
    assert window._pending_voice_text is None
    assert window._pending_pose_revision is None
    assert backend.spoken_texts == []
    assert backend.voice_stop_count == 1
    assert window.avatar_widget.emotion == "neutral"
    assert window.avatar_widget.gesture == ""
    assert window.avatar_widget.gaze == ""
    assert window.pmx_preview_widget._command is None
    assert window.pmx_preview_widget._target_expression_weights == {}
    window.close()


def test_interrupt_during_playback_stops_mouth_expression_gaze_motion_and_pose() -> None:
    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    backend.voice_speaking = True
    backend.lip_sync_state = UiLipSyncState(True, "a")
    window = ChatWindow(backend)
    command = UiMmdAvatarCommand(
        "happy",
        0.6,
        "笑顔",
        "left",
        "両目",
        "wave",
        "手を振る",
        expression_morphs=(("笑顔", 0.6),),
        expression_status_text="適用中",
    )
    window.pmx_preview_widget.set_mmd_command(command)
    window.pmx_preview_widget.begin_conversation_motion()
    pose_revision = window.pmx_preview_widget._dense_view._motion_mixer.revision
    window._sync_voice_animation()

    window._stop_voice_output()

    assert backend.voice_speaking is False
    assert window.avatar_widget.is_speaking is False
    assert window.pmx_preview_widget._voice_active is False
    assert window.pmx_preview_widget._command is None
    assert window.pmx_preview_widget._target_emotion == "neutral"
    assert window.pmx_preview_widget._target_expression_weights == {}
    assert window.pmx_preview_widget._dense_view._command is None
    assert window.pmx_preview_widget._dense_view._motion_mixer.revision > pose_revision
    window.close()


def test_stale_voice_worker_completion_cannot_restart_interrupted_generation() -> None:
    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    window = ChatWindow(backend)
    window._utterance_generation = 7
    backend.voice_speaking = True
    backend.lip_sync_state = UiLipSyncState(True, "o")

    window._voice_output_started((6, None))

    assert backend.voice_stop_count == 1
    assert backend.voice_speaking is False
    assert window.avatar_widget.is_speaking is False
    assert window.pmx_preview_widget._command is None
    window.close()


def test_new_answer_after_previous_end_ignores_old_pose_completion() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    window = ChatWindow(backend)
    revisions = iter((10, 11))
    window.pmx_preview_widget.begin_conversation_motion = lambda: next(revisions)

    window._prepare_voice_output("終了した発話")
    window._prepare_voice_output("新しい発話")
    window._conversation_pose_ready(10)
    assert backend.spoken_texts == []
    window._conversation_pose_ready(11)
    deadline = time.monotonic() + 2
    while not backend.spoken_texts and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert backend.spoken_texts == ["新しい発話"]
    while window._voice_thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    window.close()


def test_microphone_start_interrupts_voice_and_companion_controls_switch() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    backend.voice_speaking = True
    window = ChatWindow(backend)
    window._voice_state_timer.stop()

    window._toggle_microphone()
    window.show()
    window.content_splitter.setSizes([600, 400])
    app.processEvents()
    window._open_options()
    window.options_tabs.setCurrentIndex(2)

    assert backend.voice_stop_count == 1
    assert window.content_splitter.sizes()[0] > window.content_splitter.sizes()[1]
    assert window._camera_view == "bust"
    assert window.status_panel.isVisible() is True
    window.close()


def test_async_voice_failure_is_visible_and_does_not_start_continuous_recording() -> None:
    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    window = ChatWindow(backend)
    window.continuous_voice_mode_checkbox.setChecked(True)
    window._continuous_voice_waiting_generation = window._utterance_generation
    window._continuous_voice_playback_observed = True
    backend.status = lambda: BackendStatus(
        "mock", "mock-model", "API不要", voice_output_text="エラー"
    )
    window._sync_voice_animation()
    assert "読み上げに失敗" in window.error_label.text()
    assert window._continuous_voice_waiting_generation is None
    assert not backend.is_listening()
    window._clear_error()
    window._sync_voice_animation()
    assert window.error_label.text() == ""
    backend.status = lambda: BackendStatus("mock", "mock-model", "API不要")
    window._sync_voice_animation()
    assert not window._voice_error_reported
    window.close()


def test_microphone_start_during_stream_ignores_old_worker_result() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    started = Event()
    release = Event()

    def delayed_stream(
        conversation_id: str,
        user_text: str,
        on_chunk: Callable[[str], None],
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> SendOutcome:
        assert conversation_id == "test-id"
        backend.messages.append(UiMessage("user", user_text))
        on_chunk("古い結果")
        started.set()
        assert release.wait(2)
        backend.messages.append(UiMessage("assistant", "古い結果"))
        return SendOutcome(speech="古い結果")

    backend.stream_message = delayed_stream  # type: ignore[method-assign]
    window = ChatWindow(backend)
    window.input_edit.setPlainText("最初の発話")
    window._send_message()
    deadline = time.monotonic() + 2
    while not started.is_set() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    window._toggle_microphone()
    release.set()
    while window.busy_label.text() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert window.input_edit.toPlainText() == "音声入力"
    assert "古い結果" not in window.chat_view.toPlainText()
    window.close()


def test_pmx_preview_replaces_2d_widget_when_backend_has_a_model() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    backend.avatar_model_preview = lambda: UiAvatarModelPreview(
        "テストPMX",
        3,
        1,
        ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    window = ChatWindow(backend)

    assert window._camera_view in {"full", "bust"}
    window.camera_view_button.click()
    assert window._camera_view == "full"
    window.camera_view_button.click()
    assert window._camera_view == "bust"
    app.processEvents()

    assert window.pmx_preview_widget.isHidden() is False
    assert window.avatar_widget.isHidden() is True
    assert window.expression_policy_label.isHidden() is False
    assert window.expression_policy_label.text() == "表現統合: 従来表示"
    assert window.mouth_morph_label.text() == "PMX口モーフ: 頂点モーフなし"
    window.close()


def test_mmd_command_is_shown_when_backend_maps_a_model() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    backend.mmd_avatar_command = lambda: UiMmdAvatarCommand(
        "happy", 0.6, "笑顔", "left", "両目", "wave", "手を振る"
    )
    window = ChatWindow(backend)
    app.processEvents()

    assert (
        window.mmd_command_label.text()
        == "MMD表情: 笑顔（60%）\nMMD視線: 両目\nMMD姿勢: 未設定\nMMD動作: 手を振る"
    )
    assert window.emotion_morph_label.text() == "PMX表情モーフ: 未使用"
    assert window.pose_bone_label.text() == "PMXボーン: 未使用"
    window.close()


def test_available_pmx_emotion_morph_is_shown_as_active() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    backend.avatar_model_preview = lambda: UiAvatarModelPreview(
        model_name="テストPMX",
        vertex_count=3,
        face_count=1,
        points=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        emotion_morphs={"happy": ((0, 0.0, 0.5, 0.0),)},
    )
    backend.mmd_avatar_command = lambda: UiMmdAvatarCommand(
        "happy", 0.6, "笑顔", "left", "両目", "wave", "手を振る"
    )

    window = ChatWindow(backend)
    app.processEvents()

    assert window.emotion_morph_label.text() == "PMX表情モーフ: 適用中 笑顔（60%）"
    window.close()


def test_expression_preset_status_shows_applied_and_missing_model_morphs() -> None:
    preview = UiAvatarModelPreview(
        model_name="表情プリセットテスト",
        vertex_count=1,
        face_count=0,
        points=((0.0, 0.0, 0.0),),
        named_vertex_morphs={"笑い": ((0, 0.1, 0.0, 0.0),)},
    )
    command = UiMmdAvatarCommand(
        "happy",
        0.8,
        "笑い",
        "forward",
        "",
        "none",
        "",
        expression_preset="gentle_smile",
        expression_morphs=(("笑い", 0.28), ("頬染め", 0.03)),
    )

    assert _format_emotion_morph(preview, command) == (
        "表情プリセット: gentle_smile\nPMX表情モーフ: 適用中 笑い（28%）\n未検出: 頬染め"
    )
    assert "gentle_smile / 笑い 28%、頬染め 3%" in _format_mmd_command(command)


def test_expression_preview_uses_a_non_persistent_pmx_command() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    preview = UiAvatarModelPreview(
        model_name="表情確認PMX",
        vertex_count=1,
        face_count=0,
        points=((0.0, 0.0, 0.0),),
        named_vertex_morphs={"笑い": ((0, 0.1, 0.0, 0.0),)},
    )
    command = UiMmdAvatarCommand(
        "happy",
        0.82,
        "笑い",
        "forward",
        "",
        "",
        "",
        expression_preset="gentle_smile",
        expression_morphs=(("笑い", 0.82),),
    )
    backend.avatar_model_preview = lambda: preview
    backend.expression_preview_command = lambda emotion: command if emotion == "happy" else None

    window = ChatWindow(backend)
    window._start_expression_combination_preview(dict(command.expression_morphs))
    app.processEvents()

    assert window.pmx_preview_widget.is_expression_preview_active is True
    assert window.pmx_preview_widget._command.expression_morphs == command.expression_morphs
    assert backend.messages == []
    assert "笑い 82%" in window.avatar_name_label.text()

    window._stop_expression_preview()

    assert window.pmx_preview_widget.is_expression_preview_active is False
    assert window.pmx_preview_widget._command is None
    window.close()


def test_gaze_renderer_keeps_smooth_motion_and_interrupt_without_test_controls() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    preview = UiAvatarModelPreview(
        model_name="視線確認PMX",
        vertex_count=3,
        face_count=0,
        points=((0.0, 0.0, 0.0), (0.2, 1.0, 0.0), (-0.2, 1.0, 0.0)),
        bones=(
            UiAvatarBone("両目", -1, (0.0, 1.0, 0.0)),
            UiAvatarBone("左目", 0, (0.2, 1.0, 0.0)),
            UiAvatarBone("右目", 0, (-0.2, 1.0, 0.0)),
        ),
        vertex_weights=(((1, 1.0),), ((1, 1.0),), ((2, 1.0),)),
    )
    base_command = UiMmdAvatarCommand(
        "neutral",
        0.0,
        "",
        "forward",
        "",
        "none",
        "",
    )
    backend.avatar_model_preview = lambda: preview
    backend.mmd_avatar_command = lambda: base_command

    window = ChatWindow(backend)
    assert not hasattr(window, "gaze_preview_combo")
    window.pmx_preview_widget.start_gaze_preview(
        replace(base_command, gaze="left", gaze_intensity=0.50)
    )
    app.processEvents()

    assert window.pmx_preview_widget.is_gaze_preview_active is True
    assert window.pmx_preview_widget.is_conversation_motion_active is True
    assert window.pmx_preview_widget._command is not None
    assert window.pmx_preview_widget._command.gaze == "left"
    assert window.pmx_preview_widget._command.gaze_intensity == 0.50
    renderer = window.pmx_preview_widget._dense_view
    assert renderer._next_mix_delta_seconds == pytest.approx(0.06)
    rotations, _ik, _translations, _morphs = renderer._pose_settings()
    expected_yaw = 0.22 * 0.50
    assert 0.0 < rotations[1][1] < expected_yaw
    assert 0.0 < rotations[2][1] < expected_yaw
    for _ in range(12):
        rotations, _ik, _translations, _morphs = renderer._pose_settings()
    assert rotations[1][1] == pytest.approx(expected_yaw, abs=0.005)
    assert rotations[2][1] == pytest.approx(expected_yaw, abs=0.005)
    assert backend.messages == []
    assert backend.spoken_texts == []

    window.pmx_preview_widget.start_gaze_preview(
        replace(base_command, gaze="down", gaze_intensity=1.0)
    )
    assert window.pmx_preview_widget._command is not None
    assert window.pmx_preview_widget._command.gaze == "down"
    assert window.pmx_preview_widget._command.gaze_intensity == 1.0
    rotations, _ik, _translations, _morphs = renderer._pose_settings()
    assert -0.12 < rotations[1][0] < 0.0
    assert -0.12 < rotations[2][0] < 0.0
    assert 0.0 < rotations[1][1] < expected_yaw
    assert 0.0 < rotations[2][1] < expected_yaw
    for _ in range(12):
        rotations, _ik, _translations, _morphs = renderer._pose_settings()
    assert rotations[1][0] == pytest.approx(-0.12, abs=0.005)
    assert rotations[2][0] == pytest.approx(-0.12, abs=0.005)

    window.pmx_preview_widget.start_gaze_preview(base_command)
    assert window.pmx_preview_widget._command is not None
    assert window.pmx_preview_widget._command.gaze == "forward"
    assert window.pmx_preview_widget.is_conversation_motion_active is False
    assert 1 <= window.pmx_preview_widget._dense_view._blend_transition_steps <= 6
    rotations, _ik, _translations, _morphs = renderer._pose_settings()
    assert -0.12 < rotations[1][0] < 0.0
    assert -0.12 < rotations[2][0] < 0.0

    window.pmx_preview_widget.stop_gaze_preview()
    assert window.pmx_preview_widget.is_gaze_preview_active is False
    assert window.pmx_preview_widget._command == base_command
    for direction in ("forward", "left", "down"):
        window.pmx_preview_widget.start_gaze_preview(replace(base_command, gaze=direction))
        assert window.pmx_preview_widget._command.gaze == direction
    window._invalidate_utterance(interrupted=True)
    assert window.pmx_preview_widget.is_gaze_preview_active is False
    assert window.pmx_preview_widget.is_conversation_motion_active is False
    assert window.pmx_preview_widget._command is None
    window.close()


def test_expression_catalog_mouth_preview_uses_the_existing_lip_sync_path() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    preview = UiAvatarModelPreview(
        model_name="表情カタログPMX",
        vertex_count=1,
        face_count=0,
        points=((0.0, 0.0, 0.0),),
        mouth_morphs={"a": ((0, 0.0, 0.1, 0.0),)},
        morph_names=("あ", "笑い"),
        named_vertex_morphs={
            "あ": ((0, 0.0, 0.1, 0.0),),
            "笑い": ((0, 0.1, 0.0, 0.0),),
        },
    )
    backend.avatar_model_preview = lambda: preview
    window = ChatWindow(backend)

    window._start_expression_morph_preview("あ")
    assert window.pmx_preview_widget._mouth_shape == "a"
    assert window.pmx_preview_widget._command is not None
    assert window.pmx_preview_widget._command.expression_morphs == ()

    window._stop_expression_preview()
    assert window.pmx_preview_widget._mouth_shape == "neutral"

    window._start_expression_morph_preview("笑い")
    assert window.pmx_preview_widget._command is not None
    assert window.pmx_preview_widget._command.expression_morphs == (("笑い", 1.0),)
    window._stop_expression_preview()
    app.processEvents()
    window.close()


def test_expression_catalog_forwards_weighted_preset_save_to_backend() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    window = ChatWindow(backend)

    window._save_expression_preset("happy_rich", "happy", {"笑い": 0.7, "照れ": 0.2})

    assert backend.saved_expression_presets == [("happy_rich", "happy", {"笑い": 0.7, "照れ": 0.2})]
    app.processEvents()
    window.close()


def test_expression_catalog_previews_weighted_combination() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    window = ChatWindow(backend)

    window._start_expression_combination_preview({"笑い": 0.5, "照れ": 0.2})

    command = window.pmx_preview_widget._command
    assert command is not None
    assert command.expression_preset == "モーフ組合せ（未保存）"
    assert command.expression_morphs == (("笑い", 0.5), ("照れ", 0.2))
    window._stop_expression_preview()
    app.processEvents()
    window.close()


def test_available_pmx_bones_are_shown_as_active() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    backend.avatar_model_preview = lambda: UiAvatarModelPreview(
        model_name="テストPMX",
        vertex_count=1,
        face_count=0,
        points=((0.0, 0.0, 0.0),),
        bones=(
            UiAvatarBone("両目", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("頭", -1, (0.0, 1.0, 0.0)),
        ),
        vertex_weights=(((0, 1.0),),),
    )
    backend.mmd_avatar_command = lambda: UiMmdAvatarCommand(
        "neutral", 0.0, "", "left", "両目", "nod", "うなずき", "頭", "上半身"
    )

    window = ChatWindow(backend)
    app.processEvents()

    assert window.pose_bone_label.text() == "PMXボーン: 設定名「上半身」が見つかりません"
    window.close()


def test_shift_enter_adds_a_line_break_without_sending() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    window = ChatWindow(backend)
    window.input_edit.setPlainText("1行目")
    window.input_edit.moveCursor(QTextCursor.MoveOperation.End)

    QTest.keyClick(window.input_edit, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    app.processEvents()

    assert window.input_edit.toPlainText() == "1行目\n"
    assert backend.created_titles == []
    window.close()


def test_settings_do_not_generate_test_vmd_or_expose_development_controls() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    generated = []
    backend.generate_confirmation_vmd = lambda: generated.append(True)
    window = ChatWindow(backend)
    window._open_options()
    for index in range(window.options_tabs.count()):
        window.options_tabs.setCurrentIndex(index)
        app.processEvents()
    assert generated == []
    for name in (
        "generate_vmd_button",
        "gaze_preview_combo",
        "expression_preview_combo",
        "status_toggle_button",
        "avatar_motion_checkbox",
    ):
        assert not hasattr(window, name)
    assert window._camera_view in {"full", "bust"}
    window.close()


def test_new_conversation_is_created_and_selected_in_history() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    backend.has_conversation = True
    window = ChatWindow(backend)

    window.new_button.click()
    app.processEvents()

    current = window.conversation_list.currentItem()
    assert backend.created_titles == ["新しい会話"]
    assert window._current_conversation_id == "test-id"
    assert current is not None
    assert current.data(Qt.ItemDataRole.UserRole) == "test-id"
    assert current.text() == "新しい会話"
    assert "新しい会話です" in window.chat_view.toPlainText()
    window.close()


def test_microphone_result_is_put_in_input_box() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    window = ChatWindow(FakeBackend())

    window.microphone_button.click()
    app.processEvents()

    assert window.input_edit.toPlainText() == "音声入力"
    assert window.microphone_level.value() == 0
    assert window.partial_label.text() == ""
    assert window._ptt_phase == "認識済み"
    assert window.ptt_phase_label.text() == "マイク: 認識済み"
    window.close()


def test_auto_send_disabled_keeps_recognition_in_input_box() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    window = ChatWindow(backend)

    window._toggle_microphone()
    app.processEvents()

    assert window.auto_send_recognition_checkbox.isChecked() is False
    assert window.input_edit.toPlainText() == "音声入力"
    assert backend.messages == []
    window.close()


def test_settings_dialog_exposes_existing_speech_turn_settings() -> None:
    QApplication.instance() or QApplication([])
    dialog = SettingsDialog(
        ["mock"],
        "mock",
        SpeechSettings(
            "whisper",
            None,
            [],
            silence_threshold=0.02,
            silence_duration=1.5,
            max_recording_duration=45.0,
            partial_interval=3.0,
        ),
    )

    assert dialog.selected_speech_turn_settings() == (0.02, 1.5, 45.0, 3.0)
    dialog.close()


def test_hold_to_talk_finishes_recording_without_discarding_final_result() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    callbacks: list[Callable[[str], None]] = []

    def start(
        on_recognized: Callable[[str], None],
        _on_error: Callable[[str], None],
        _on_partial: Callable[[str], None],
        _on_level: Callable[[int], None],
    ) -> None:
        backend.listening = True
        callbacks.append(on_recognized)

    backend.start_speech_recognition = start  # type: ignore[method-assign]
    window = ChatWindow(backend)

    window._start_hold_to_talk()
    assert backend.listening is True
    window._finish_hold_to_talk()

    assert backend.finish_recognition_count == 1
    assert window._recognition_generation == 1
    callbacks[0]("押している間の発話")
    app.processEvents()

    assert window.input_edit.toPlainText() == "押している間の発話"
    assert window.microphone_button.isEnabled() is True
    window.close()


def test_hold_to_talk_uses_space_only_outside_the_text_input() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    callbacks: list[Callable[[str], None]] = []

    def start(
        on_recognized: Callable[[str], None],
        _on_error: Callable[[str], None],
        _on_partial: Callable[[str], None],
        _on_level: Callable[[int], None],
    ) -> None:
        backend.listening = True
        callbacks.append(on_recognized)

    backend.start_speech_recognition = start  # type: ignore[method-assign]
    window = ChatWindow(backend)
    window.hold_to_talk_checkbox.setChecked(True)
    window.show()
    window.activateWindow()
    window.setFocus()
    app.processEvents()

    QTest.keyPress(window, Qt.Key.Key_Space)
    assert backend.listening is True
    QTest.keyRelease(window, Qt.Key.Key_Space)
    assert backend.finish_recognition_count == 1
    callbacks[0]("Space PTT")
    app.processEvents()
    assert window.input_edit.toPlainText() == "Space PTT"

    window.input_edit.setFocus()
    QTest.mouseClick(window.conversation_state_label, Qt.MouseButton.LeftButton)
    assert window.input_edit.hasFocus() is False
    QTest.keyPress(window, Qt.Key.Key_Space)
    assert backend.listening is True
    QTest.keyRelease(window, Qt.Key.Key_Space)
    assert backend.finish_recognition_count == 2
    window.close()


def test_auto_send_submits_one_nonempty_final_recognition_per_generation() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    callbacks: list[Callable[[str], None]] = []
    partials: list[Callable[[str], None]] = []
    errors: list[Callable[[str], None]] = []

    def start(
        on_recognized: Callable[[str], None],
        on_error: Callable[[str], None],
        on_partial: Callable[[str], None],
        _on_level: Callable[[int], None],
    ) -> None:
        backend.listening = True
        callbacks.append(on_recognized)
        errors.append(on_error)
        partials.append(on_partial)

    backend.start_speech_recognition = start  # type: ignore[method-assign]
    window = ChatWindow(backend)
    window.auto_send_recognition_checkbox.setChecked(True)
    window._toggle_microphone()
    partials[0]("途中")
    callbacks[0](" ")
    errors[0]("認識エラー")
    callbacks[0]("確定結果")
    assert backend.messages == []

    window._toggle_microphone()
    window._toggle_microphone()
    callbacks[1]("確定結果")

    deadline = time.monotonic() + 3
    while window.busy_label.text() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert [message for message in backend.messages if message.role == "user"] == [
        UiMessage("user", "確定結果")
    ]
    assert window.input_edit.toPlainText() == ""
    window.close()


def test_stopped_microphone_result_is_not_put_in_input_box() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    callbacks: list[Callable[[str], None]] = []

    def start(
        on_recognized: Callable[[str], None],
        _on_error: Callable[[str], None],
        _on_partial: Callable[[str], None],
        _on_level: Callable[[int], None],
    ) -> None:
        backend.listening = True
        callbacks.append(on_recognized)

    backend.start_speech_recognition = start  # type: ignore[method-assign]
    window = ChatWindow(backend)

    window._toggle_microphone()
    window._toggle_microphone()
    callbacks[0]("古い認識結果")
    app.processEvents()

    assert window.input_edit.toPlainText() == ""
    assert window._ptt_phase == "待機"
    window.close()


def test_auto_send_invalidates_old_stream_without_provider_cancellation() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    started = Event()
    release = Event()
    callbacks: list[Callable[[str], None]] = []

    def delayed_stream(
        conversation_id: str,
        user_text: str,
        on_chunk: Callable[[str], None],
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> SendOutcome:
        assert conversation_id == "test-id"
        backend.messages.append(UiMessage("user", user_text))
        if user_text == "最初の発話":
            started.set()
            assert release.wait(2)
            backend.messages.append(UiMessage("assistant", "古い応答"))
            on_chunk("古い応答")
            return SendOutcome(speech="古い応答")
        backend.messages.append(UiMessage("assistant", "新しい応答"))
        on_chunk("新しい応答")
        return SendOutcome(speech="新しい応答")

    def start(
        on_recognized: Callable[[str], None],
        _on_error: Callable[[str], None],
        _on_partial: Callable[[str], None],
        _on_level: Callable[[int], None],
    ) -> None:
        backend.listening = True
        callbacks.append(on_recognized)

    backend.stream_message = delayed_stream  # type: ignore[method-assign]
    backend.start_speech_recognition = start  # type: ignore[method-assign]
    window = ChatWindow(backend)
    window.input_edit.setPlainText("最初の発話")
    window._send_message()
    deadline = time.monotonic() + 2
    while not started.is_set() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    window.auto_send_recognition_checkbox.setChecked(True)
    window._toggle_microphone()
    callbacks[0]("次の発話")
    app.processEvents()
    release.set()
    deadline = time.monotonic() + 3
    while window.busy_label.text() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert "古い応答" not in window.chat_view.toPlainText()
    assert "新しい応答" in window.chat_view.toPlainText()
    assert backend.spoken_texts == ["新しい応答"]
    assert not hasattr(backend, "cancel_provider")
    window.close()


def test_microphone_trace_records_ptt_and_voice_interruption(monkeypatch) -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    backend.voice_speaking = True
    events: list[str] = []
    monkeypatch.setattr(
        "multi_llm_companion.ui.chat_window.write_lip_sync_trace",
        lambda _component, event, **_fields: events.append(event),
    )
    window = ChatWindow(backend)
    window._voice_state_timer.stop()

    window._toggle_microphone()
    app.processEvents()

    assert [
        event
        for event in events
        if event not in {"lip_sync_state_applied", "conversation_state_transition"}
    ] == [
        "ptt_pressed",
        "voice_stop_requested",
        "voice_stopped",
        "recognition_started",
        "recognition_finished",
    ]
    window.close()
    assert window._voice_state_timer.isActive() is False


def test_continuous_voice_mode_is_disabled_by_default() -> None:
    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    window = ChatWindow(backend)
    started: list[bool] = []
    window._start_continuous_voice_recording = lambda: started.append(True)  # type: ignore[method-assign]
    window._utterance_generation = 3
    window._arm_continuous_voice_restart(3)
    backend.voice_speaking = True
    window._sync_voice_animation()
    backend.voice_speaking = False
    window._sync_voice_animation()

    assert window.continuous_voice_mode_checkbox.isChecked() is False
    assert started == []
    window.close()


def test_continuous_voice_mode_restarts_once_after_actual_playback_end() -> None:
    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    window = ChatWindow(backend)
    started: list[bool] = []
    window._start_continuous_voice_recording = lambda: started.append(True)  # type: ignore[method-assign]
    window.continuous_voice_mode_checkbox.setChecked(True)
    window._utterance_generation = 4
    window._arm_continuous_voice_restart(4)

    backend.voice_speaking = True
    window._sync_voice_animation()
    backend.voice_speaking = False
    window._sync_voice_animation()
    window._sync_voice_animation()

    assert started == [True]
    window.close()


def test_continuous_voice_mode_ignores_stale_or_manually_stopped_playback() -> None:
    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    window = ChatWindow(backend)
    started: list[bool] = []
    window._start_continuous_voice_recording = lambda: started.append(True)  # type: ignore[method-assign]
    window.continuous_voice_mode_checkbox.setChecked(True)
    window._utterance_generation = 5
    window._arm_continuous_voice_restart(4)
    backend.voice_speaking = True
    window._sync_voice_animation()
    backend.voice_speaking = False
    window._sync_voice_animation()

    window._arm_continuous_voice_restart(5)
    backend.voice_speaking = True
    window._sync_voice_animation()
    window._stop_voice_output()
    backend.voice_speaking = False
    window._sync_voice_animation()

    assert started == []
    window.close()


def test_continuous_voice_mode_does_not_restart_on_empty_or_failed_recognition() -> None:
    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    window = ChatWindow(backend)
    window.continuous_voice_mode_checkbox.setChecked(True)
    window._speech_recognized("  ")
    window._speech_recognition_failed("認識エラー")

    assert backend.messages == []
    assert window._continuous_voice_waiting_generation is None
    window.close()


def test_continuous_voice_trace_records_no_recognition_or_conversation_text(monkeypatch) -> None:
    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    records: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        "multi_llm_companion.ui.chat_window.write_lip_sync_trace",
        lambda _component, event, **fields: records.append((event, fields)),
    )
    window = ChatWindow(backend)
    window._start_continuous_voice_recording = lambda: None  # type: ignore[method-assign]
    window.continuous_voice_mode_checkbox.setChecked(True)
    window._utterance_generation = 6
    window._arm_continuous_voice_restart(6)
    backend.voice_speaking = True
    window._sync_voice_animation()
    backend.voice_speaking = False
    window._sync_voice_animation()

    assert ("continuous_voice_playback_finished", {}) in records
    assert all("認識" not in str(fields) and "会話" not in str(fields) for _, fields in records)
    window.close()


def test_memory_manager_edits_and_deletes_saved_memory() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    backend.memories = [UiMemory(1, "ユーザーは紅茶が好き", "preference", "approved")]
    dialog = MemoryDialog(backend)

    dialog.memory_list.setCurrentRow(0)
    dialog.content_edit.setPlainText("ユーザーは緑茶が好き")
    dialog.save_button.click()
    app.processEvents()

    assert backend.memories[0].content == "ユーザーは緑茶が好き"
    dialog.delete_button.click()
    app.processEvents()
    assert backend.memories == []
    dialog.close()


def test_memory_manager_approves_candidate_and_shows_its_evidence() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    backend.has_conversation = True
    backend.memories = [
        UiMemory(
            1,
            "ユーザーは紅茶が好き",
            "preference",
            "candidate",
            "2026-08-14 10:00",
            "2026-08-14 10:00",
            "test-id",
        )
    ]
    dialog = MemoryDialog(backend)

    dialog.memory_list.setCurrentRow(0)
    assert dialog.status_label.text() == "確認待ち"
    assert "画面テスト" in dialog.source_label.text()
    assert dialog.created_label.text() == "2026-08-14 10:00"
    dialog.approve_button.click()
    app.processEvents()

    assert backend.memories[0].status == "approved"
    dialog.close()


def test_memory_manager_toggles_auto_approval_setting() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    backend = FakeBackend()
    dialog = MemoryDialog(backend)

    assert dialog.auto_approve_check.isChecked() is True
    dialog.auto_approve_check.setChecked(False)
    app.processEvents()

    assert backend.memory_auto_approve_candidates is False
    dialog.close()
