"""日本語の文字チャット画面。"""

from __future__ import annotations

import html
import math
import re
import time
from dataclasses import replace
from datetime import datetime
from functools import partial

from PySide6.QtCore import QByteArray, QEvent, QSettings, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFocusEvent,
    QInputMethodEvent,
    QKeyEvent,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from multi_llm_companion.config import ConfigError
from multi_llm_companion.domain import ConversationState
from multi_llm_companion.environment import EnvironmentSettingsError
from multi_llm_companion.integrations.lip_sync_diagnostic import write_lip_sync_trace
from multi_llm_companion.services.image_sharing import PreparedImage
from multi_llm_companion.services.proactive import (
    ProactiveConversationService,
    ProactiveSettings,
    ProactiveState,
)
from multi_llm_companion.services.web_search import (
    _needs_weather_region,
    _search_query,
    _weather_period,
    _weather_region,
    _weather_regions,
    is_search_request,
    is_weather_request,
    safe_source_url,
)
from multi_llm_companion.streaming import SpeechSentenceBuffer
from multi_llm_companion.ui._dialog_layout import _AvatarPreview, _fit_dialog
from multi_llm_companion.ui._window_chrome import (
    _application_icon,
    _apply_title_bar_palette,
    _set_control_icon,
)
from multi_llm_companion.ui._windows_ime import _enable_windows_ime
from multi_llm_companion.ui.avatar_widget import AvatarWidget
from multi_llm_companion.ui.conversation_view import ConversationView
from multi_llm_companion.ui.environment_dialog import EnvironmentDialog
from multi_llm_companion.ui.expression_library_dialog import (
    ExpressionLibraryDialog,
    mouth_shape_for_morph,
)
from multi_llm_companion.ui.floating_notice import NOTICE_DEFAULT_SECONDS, FloatingNotice
from multi_llm_companion.ui.image_attachment import (
    _image_drop_source,
    _ImageDrop,
    _ImageDropWorker,
    prepare_image,
)
from multi_llm_companion.ui.lip_sync_monitor_widget import LipSyncMonitorWidget
from multi_llm_companion.ui.memory_dialog import MemoryDialog
from multi_llm_companion.ui.models import (
    ChatBackend,
    SendOutcome,
    UiAvatarModelPreview,
    UiEmotionState,
    UiExpressionPreset,
    UiLipSyncState,
    UiMemory,
    UiMessage,
    UiMmdAvatarCommand,
    UiTokenUsageSummary,
)
from multi_llm_companion.ui.motion_library_dialog import MotionLibraryDialog
from multi_llm_companion.ui.persona_dialog import PersonaDialog
from multi_llm_companion.ui.pmx_preview_widget import PmxPreviewWidget
from multi_llm_companion.ui.proactive_dialog import ProactiveSettingsDialog
from multi_llm_companion.ui.settings_dialog import SettingsDialog
from multi_llm_companion.ui.workers import ProactiveWorker, StreamWorker, VoiceStartWorker


class MessageInput(QTextEdit):
    """Enterで送信し、Shift+Enterで改行する入力欄。"""

    send_requested = Signal()
    image_dropped = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._composing = False

    def focusInEvent(self, event: QFocusEvent) -> None:  # noqa: N802
        super().focusInEvent(event)
        # Qtが入力先を切り替えた後に、現在の入力欄だけへ適用する。
        QTimer.singleShot(0, self, self._open_ime)

    def _open_ime(self) -> None:
        if self.hasFocus() and self.isEnabled() and not self.isReadOnly():
            _enable_windows_ime()

    def inputMethodEvent(self, event: QInputMethodEvent) -> None:  # noqa: N802
        self._composing = bool(event.preeditString())
        super().inputMethodEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if (
            event.mimeData().hasUrls()
            or event.mimeData().hasImage()
            or _image_drop_source(event.mimeData())
        ):
            if (
                self.acceptDrops()
                and event.possibleActions() & Qt.DropAction.CopyAction
                and _image_drop_source(event.mimeData())
            ):
                event.setDropAction(Qt.DropAction.CopyAction)
                event.accept()
            else:
                event.ignore()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802
        if (
            event.mimeData().hasUrls()
            or event.mimeData().hasImage()
            or _image_drop_source(event.mimeData())
        ):
            self.dragEnterEvent(event)
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        source = _image_drop_source(event.mimeData())
        if event.mimeData().hasUrls() or event.mimeData().hasImage() or source:
            if self.acceptDrops() and source and event.possibleActions() & Qt.DropAction.CopyAction:
                self.image_dropped.emit(source)
                event.setDropAction(Qt.DropAction.CopyAction)
                event.accept()
            else:
                event.ignore()
            return
        super().dropEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if self._composing and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.send_requested.emit()
            return
        super().keyPressEvent(event)


class ChatWindow(QMainWindow):
    speech_recognized = Signal(str)
    speech_recognition_failed = Signal(str)
    speech_partial = Signal(str)
    speech_level = Signal(int)

    def __init__(self, backend: ChatBackend) -> None:
        super().__init__()
        self._backend = backend
        self._current_conversation_id: str | None = None
        self._messages: list[UiMessage] = []
        self._busy = False
        self._conversation_state = ConversationState.IDLE
        self._conversation_state_started_at = time.monotonic()
        self._thread: QThread | None = None
        self._worker: StreamWorker | None = None
        self._used_memories: list[UiMemory] = []
        self._stream_generation = 0
        self._web_turn_active = False
        self._image_turn_active = False
        self._attached_image: PreparedImage | None = None
        self._image_loading = False
        self._image_fetch_generation = 0
        self._image_fetches = {}
        self._close_after_image_fetch = False
        self._recognition_generation = 0
        self._auto_sent_recognition_generation: int | None = None
        self._pending_auto_send_text: str | None = None
        self._continuous_voice_waiting_generation: int | None = None
        self._continuous_voice_playback_observed = False
        self._voice_error_reported = False
        self._hold_to_talk_active = False
        self._ptt_phase = "待機"
        self._settings = QSettings("multi-llm-companion", "desktop-companion")
        self._proactive = ProactiveConversationService(self._load_proactive_state())
        self._proactive.note_user_activity(datetime.now())
        self._proactive_turn_generation = 0
        self._proactive_turn_active = False
        self._proactive_thread: QThread | None = None
        self._proactive_worker: ProactiveWorker | None = None
        self._proactive_timer = QTimer(self)
        self._proactive_timer.setInterval(30_000)
        self._voice_state_timer = QTimer(self)
        # 母音の短い切替も拾えるよう、口形状態を約16回/秒で反映する。
        self._voice_state_timer.setInterval(60)
        self._last_voice_sync_at = time.perf_counter()
        self._pending_voice_text: str | None = None
        self._pending_pose_revision: int | None = None
        self._pending_utterance_generation: int | None = None
        self._utterance_generation = 0
        self._proactive_voice_generation: int | None = None
        self._voice_pose_timeout = QTimer(self)
        self._voice_pose_timeout.setSingleShot(True)
        self._voice_pose_timeout.setInterval(3000)
        self._voice_thread: QThread | None = None
        self._close_after_voice = False
        self._voice_worker: VoiceStartWorker | None = None
        self._stream_sentence_buffer = SpeechSentenceBuffer()
        self._stream_voice_started = False
        self._stream_voice_preparing = False
        self._stream_voice_finished = False
        self._stream_pending_sentences: list[str] = []
        self._stream_pose_revision: int | None = None
        self._stream_sentence_index = 0
        self._stream_timing_started_at: float | None = None
        self._stream_first_chunk_logged = False
        self._current_avatar_preview: UiAvatarModelPreview | None = None
        self._current_mmd_command: UiMmdAvatarCommand | None = None
        self._expression_preview_active = False
        self._camera_view = "bust"

        self._avatar_display_name = ""

        self.setWindowTitle("雨衣 — Multi LLM Companion")
        self.setWindowIcon(_application_icon())
        self.resize(1120, 780)
        self.setMinimumSize(760, 540)
        self.setAcceptDrops(True)
        self._build_ui()
        self._connect_events()
        QApplication.instance().installEventFilter(self)
        self._restore_window_preferences()
        self._voice_state_timer.start()
        self._proactive_timer.start()
        self._update_status_labels()
        self._reload_conversations()
        self._set_busy(False)

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("companionRoot")
        self.setCentralWidget(central)
        shell = QVBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        header = QHBoxLayout()
        header.setSpacing(12)
        header.setContentsMargins(20, 10, 20, 10)
        self.history_button = QPushButton("履歴")
        _set_control_icon(self.history_button, "history", "履歴")
        self.history_button.setFixedSize(40, 40)
        self.history_button.setCheckable(True)
        header.addWidget(self.history_button)
        appearance = self._backend.avatar_appearance()
        self._avatar_display_name = appearance.display_name
        title = QLabel(self._avatar_display_name)
        title.setStyleSheet("font-size: 20px; font-weight: 600; color: #1A267C;")
        header.addWidget(title)
        header.addStretch()
        self.options_button = QPushButton("設定")
        _set_control_icon(self.options_button, "settings", "設定")
        self.options_button.setFixedSize(40, 40)
        header.addWidget(self.options_button)
        shell.addLayout(header)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(8)
        shell.addWidget(self.main_splitter, 1)
        self.history_panel = QWidget()
        self.history_panel.setObjectName("historyPanel")
        self.history_panel.setMinimumWidth(170)
        history = QVBoxLayout(self.history_panel)
        history.setContentsMargins(14, 16, 14, 14)
        history.addWidget(QLabel("会話履歴"))
        self.conversation_list = QListWidget()
        self.conversation_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        history.addWidget(self.conversation_list, 1)
        self.new_button = QPushButton("新しい会話")
        self.delete_button = QPushButton("この履歴を削除")
        history.addWidget(self.new_button)
        history.addWidget(self.delete_button)
        self.main_splitter.addWidget(self.history_panel)
        self.history_panel.hide()
        self.history_button.toggled.connect(self.history_panel.setVisible)

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.setHandleWidth(8)
        self.main_splitter.addWidget(self.content_splitter)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([190, 930])
        self.avatar_panel = QWidget()
        self.avatar_panel.setMinimumWidth(180)
        avatar_grid = QGridLayout(self.avatar_panel)
        avatar_grid.setContentsMargins(0, 0, 0, 0)
        avatar_surface = QWidget()
        avatar_grid.addWidget(avatar_surface, 0, 0)
        avatar_layout = QVBoxLayout(avatar_surface)
        avatar_layout.setContentsMargins(0, 0, 0, 0)
        avatar_layout.setSpacing(0)
        self.avatar_widget = AvatarWidget()
        self.avatar_widget.set_appearance(
            appearance.hair_color, appearance.outfit_color, appearance.accent_color
        )
        avatar_layout.addWidget(self.avatar_widget, 1)
        self.pmx_preview_widget = PmxPreviewWidget(
            render_quality=self._backend.avatar_3d_status().render_quality
        )
        self.pmx_preview_widget.setMinimumHeight(160)
        self.pmx_preview_widget.setVisible(False)
        avatar_layout.addWidget(self.pmx_preview_widget, 1)
        self.pmx_preview_widget.set_view_mode(self._camera_view)
        self.camera_view_button = QPushButton()
        self.camera_view_button.setFixedSize(40, 40)
        self._update_camera_view_icon()
        camera_controls = QWidget()
        controls_layout = QHBoxLayout(camera_controls)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.addWidget(self.camera_view_button)
        avatar_grid.addWidget(
            camera_controls, 0, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight
        )
        self.content_splitter.addWidget(self.avatar_panel)

        self._left_panel = QWidget()
        self._left_panel.setMinimumWidth(340)
        chat_layout = QVBoxLayout(self._left_panel)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)
        self.chat_view = ConversationView()
        chat_layout.addWidget(self.chat_view, 1)
        self.notice = FloatingNotice(self._left_panel)
        self.notice.action_requested.connect(self._open_memory_manager)
        composer = QWidget()
        composer.setObjectName("composer")
        compose = QVBoxLayout(composer)
        compose.setContentsMargins(16, 10, 16, 14)
        compose.setSpacing(8)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #87505A; font-size: 13px;")
        self.error_label.setTextFormat(Qt.TextFormat.PlainText)
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self.dismiss_error_button = QPushButton("×")
        self.dismiss_error_button.setAccessibleName("エラー表示を閉じる")
        self.dismiss_error_button.setFixedSize(24, 24)
        self.dismiss_error_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.dismiss_error_button.clicked.connect(self._clear_error)
        self.dismiss_error_button.hide()
        error_row = QHBoxLayout()
        error_row.addWidget(self.error_label, 1)
        error_row.addWidget(self.dismiss_error_button, 0, Qt.AlignmentFlag.AlignTop)
        compose.addLayout(error_row)

        self.activity_panel = QWidget()
        activity = QVBoxLayout(self.activity_panel)
        activity.setContentsMargins(0, 0, 0, 0)
        activity.setSpacing(4)
        external = QHBoxLayout()
        self.web_activity_label = QLabel("Web検索・返答のためOpenAIへ送信中")
        self.web_activity_label.setWordWrap(True)
        self.web_cancel_button = QPushButton("検索・返答を取消")
        self.web_cancel_button.clicked.connect(self._cancel_web_turn)
        external.addWidget(self.web_activity_label, 1)
        external.addWidget(self.web_cancel_button)
        self.web_activity_label.hide()
        self.web_cancel_button.hide()
        activity.addLayout(external)
        current = QHBoxLayout()
        self.conversation_state_label = QLabel("")
        self.conversation_state_label.setWordWrap(True)
        for label in (self.conversation_state_label, self.web_activity_label):
            label.setStyleSheet("color: #576A7D; font-size: 12px;")
        self.stop_voice_button = QPushButton("音声・動作を停止")
        self.stop_voice_button.setEnabled(False)
        self.stop_voice_button.hide()
        current.addWidget(self.conversation_state_label, 1)
        current.addWidget(self.stop_voice_button)
        activity.addLayout(current)
        self.recognition_panel = QWidget()
        recognition = QVBoxLayout(self.recognition_panel)
        recognition.setContentsMargins(0, 0, 0, 0)
        recording_status = QHBoxLayout()
        self.microphone_level = QProgressBar()
        self.microphone_level.setRange(0, 100)
        self.microphone_level.setValue(0)
        self.microphone_level.setTextVisible(False)
        self.microphone_level.setFixedWidth(70)
        self.microphone_level.setAccessibleName("マイク入力音量")
        self.ptt_phase_label = QLabel("PTT: 待機")
        self.partial_label = QLineEdit()
        self.partial_label.setReadOnly(True)
        self.partial_label.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.partial_label.setAccessibleName("認識途中の文章")
        recording_status.addWidget(self.microphone_level)
        recording_status.addWidget(self.ptt_phase_label)
        recording_status.addStretch()
        recognition.addLayout(recording_status)
        recognition.addWidget(self.partial_label)
        activity.addWidget(self.recognition_panel)
        compose.addWidget(self.activity_panel)

        self.attachment_panel = QWidget()
        self.attachment_panel.setFixedSize(104, 84)
        self.attachment_preview = QLabel(self.attachment_panel)
        self.attachment_preview.setGeometry(0, 4, 100, 80)
        self.attachment_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.attachment_preview.setAccessibleName("添付画像")
        self.attachment_preview.setStyleSheet(
            "background: #F6FCFD; border: 1px solid #B6E3F2; border-radius: 8px;"
        )
        self.remove_attachment_button = QPushButton(self.attachment_panel)
        _set_control_icon(self.remove_attachment_button, "remove", "画像を外す")
        self.remove_attachment_button.setIconSize(QSize(16, 16))
        self.remove_attachment_button.setGeometry(78, 0, 26, 26)
        self.remove_attachment_button.setStyleSheet(
            "QPushButton { padding: 2px; border-radius: 13px; background: #F6FCFD; }"
            "QPushButton:hover { background: #B2F0ED; }"
        )
        self.remove_attachment_button.clicked.connect(self._clear_attachment)
        self.attachment_panel.hide()

        input_row = QHBoxLayout()
        self.input_edit = MessageInput()
        self.input_edit.setTabChangesFocus(True)
        self.input_edit.setPlaceholderText(f"{self._avatar_display_name}に話しかける…")
        self.input_edit.setMinimumHeight(64)
        self.input_edit.setMaximumHeight(96)
        self.input_edit.setAccessibleName("会話の入力")
        self.input_edit.setToolTip("Enterで送信、Shift+Enterで改行")
        input_row.addWidget(self.input_edit, 1)
        self.microphone_button = QPushButton("マイク")
        _set_control_icon(self.microphone_button, "microphone", "マイク")
        self.microphone_button.setObjectName("microphoneButton")
        self.microphone_button.setFixedSize(52, 52)
        input_row.addWidget(self.microphone_button)
        self.send_button = QPushButton("送信")
        _set_control_icon(self.send_button, "send", "送信")
        self.send_button.setObjectName("sendButton")
        self.send_button.setDefault(True)
        self.send_button.setFixedSize(52, 52)
        input_row.addWidget(self.send_button)
        compose.addLayout(input_row)
        tools_row = QHBoxLayout()
        self.attach_image_button = QPushButton("画像を添付")
        _set_control_icon(self.attach_image_button, "attach", "画像を添付")
        self.attach_image_button.setFixedSize(40, 40)
        self.attach_image_button.clicked.connect(self._select_image)
        tools_row.addWidget(self.attach_image_button)
        tools_row.addWidget(self.attachment_panel)
        tools_row.addStretch()
        compose.addLayout(tools_row)
        chat_layout.addWidget(composer)
        self.content_splitter.addWidget(self._left_panel)
        self.content_splitter.setSizes([470, 530])

        self.options_dialog = QDialog(self)
        self.options_dialog.setWindowTitle("設定")
        options_layout = QVBoxLayout(self.options_dialog)
        tabs = QTabWidget()
        self.options_tabs = tabs
        options_layout.addWidget(tabs)
        self.options_button.clicked.connect(self._open_options)
        pages = []
        for name, description in (
            (
                "会話",
                "雨衣のふるまい、記憶、外部への送信を設定します。チェックの変更はすぐ保存されます。",
            ),
            ("音声", "マイク入力と音声会話の動作を設定します。チェックの変更はすぐ保存されます。"),
            ("表示", "ウィンドウと雨衣の見え方を設定します。変更はすぐ反映・保存されます。"),
            (
                "環境設定",
                "AIと読み上げ音声の接続をまとめて設定します。保存した接続設定は再起動後に反映されます。",
            ),
            (
                "診断情報",
                "接続や音声に問題があるときの確認用情報です。設定を閉じると会話に戻ります。",
            ),
        ):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(18, 18, 18, 18)
            layout.setSpacing(12)
            scroll.setWidget(page)
            tabs.addTab(scroll, name)
            note = QLabel(description)
            note.setWordWrap(True)
            layout.addWidget(note)
            pages.append(layout)
        general, audio_settings, avatar_settings, environment_settings, diagnostics = pages
        self.settings_button = QPushButton("マイクの設定")
        self.environment_button = QPushButton("LLM・VoiSona Talkの接続設定")
        self.persona_button = QPushButton("人格を編集")
        self.memory_button = QPushButton("記憶を管理")
        self.proactive_button = QPushButton("能動的な話しかけ")
        character = _settings_section(
            general,
            "人格と記憶",
            "性格や会話方針、保存した記憶を管理します。各画面で保存・管理します。",
        )
        character.addWidget(self.persona_button)
        character.addWidget(self.memory_button)
        proactive = _settings_section(
            general, "話しかけ", "会話が止まったときに雨衣から話しかけるかを設定します。"
        )
        proactive.addWidget(self.proactive_button)
        microphone = _settings_section(
            audio_settings,
            "マイク入力",
            "音声認識の方式・機器・録音の調整は、保存するとすぐ反映されます。",
        )
        microphone.addWidget(self.settings_button)
        voice_behavior = _settings_section(
            audio_settings,
            "音声会話の動作",
            "読み上げの接続先やボイス名は「環境設定」で変更できます。",
        )
        self.auto_send_recognition_checkbox = QCheckBox("認識した音声を自動で送信する")
        self.continuous_voice_mode_checkbox = QCheckBox("返答の読み上げ後、次の録音を始める")
        self.hold_to_talk_checkbox = QCheckBox("Spaceを押している間だけ録音")
        microphone.addWidget(self.auto_send_recognition_checkbox)
        microphone.addWidget(self.hold_to_talk_checkbox)
        voice_behavior.addWidget(self.continuous_voice_mode_checkbox)
        self.hold_to_talk_checkbox.setToolTip("入力欄にカーソルがない時だけSpaceで録音します。")
        sharing = _settings_section(
            general,
            "Web検索・画像共有",
            "対応する接続先で、必要な情報を外部サービスへ送ることを許可します。",
        )
        self.web_permission_checkbox = QCheckBox("Web検索を許可")
        self.web_permission_checkbox.setToolTip(
            "依頼した検索文と現在日時をOpenAIへ送ります（有料）。"
            "検索側へ会話全文・保存記憶は送りません。現在地は取得しません。\n"
            "検索結果と通常の会話文脈から返答し、出典を履歴に残します。"
            "実行中は入力欄の近くで取り消せます。送信後の取消でも料金が発生します。"
        )
        self.image_permission_checkbox = QCheckBox("画像の送信を許可")
        self.web_permission_checkbox.clicked.connect(self._toggle_web_permission)
        self.image_permission_checkbox.clicked.connect(self._toggle_image_permission)
        sharing.addWidget(self.web_permission_checkbox)
        sharing.addWidget(self.image_permission_checkbox)
        permission_note = QLabel(
            "検索を頼んで送信した時、または画像を添付して送信した時だけ外部へ送ります。"
            "設定の変更はすぐ保存され、次の起動時も引き継ぎます。"
            "接続先・モデルが未対応の機能は使えません。"
        )
        permission_note.setWordWrap(True)
        sharing.addWidget(permission_note)
        self._refresh_web_permission()
        general.addStretch()
        audio_settings.addStretch()

        connections = _settings_section(
            environment_settings,
            "接続先と認証情報",
            "使用するLLM、APIキー・URL・モデル名、VoiSona Talkの認証情報を一つの画面で管理します。",
        )
        self.connection_source_label = QLabel()
        self.connection_source_label.setWordWrap(True)
        connections.addWidget(self.connection_source_label)
        connections.addWidget(self.environment_button)
        storage_note = QLabel(
            "APIキーとパスワードはWindowsで暗号化して保管します。環境変数や .env は書き換えません。"
        )
        storage_note.setWordWrap(True)
        connections.addWidget(storage_note)
        environment_settings.addStretch()

        window_settings = _settings_section(
            avatar_settings, "ウィンドウ", "ほかのアプリと並べて使うときの表示を設定します。"
        )
        notices = _settings_section(
            avatar_settings,
            "フローティング通知",
            "記憶の保存・確認待ちと、音声停止・返答取消の案内を別々に設定できます。\n"
            "変更は自動保存されます。チェックを外すと表示中の通知も閉じます。"
            "表示時間は次の通知から使います。エラーや処理中の表示は対象外です。",
        )
        self._notice_controls: dict[str, tuple[QCheckBox, QDoubleSpinBox]] = {}
        for kind, name in (("memory", "記憶の通知"), ("info", "通常の案内")):
            row = QHBoxLayout()
            enabled = QCheckBox(f"{name}を表示")
            enabled.setChecked(True)
            seconds = QDoubleSpinBox()
            seconds.setRange(1.0, 30.0)
            seconds.setDecimals(1)
            seconds.setSingleStep(0.5)
            seconds.setSuffix(" 秒")
            seconds.setValue(NOTICE_DEFAULT_SECONDS[kind])
            seconds.setKeyboardTracking(False)
            seconds.setAccessibleName(f"{name}の表示時間")
            seconds.setToolTip("1～30秒で設定します。入力後にEnterを押すと反映します。")
            duration_label = QLabel("表示時間")
            duration_label.setBuddy(seconds)
            row.addWidget(enabled, 1)
            row.addWidget(duration_label)
            row.addWidget(seconds)
            notices.addLayout(row)
            self._notice_controls[kind] = (enabled, seconds)
        avatar_view = _settings_section(
            avatar_settings,
            "雨衣の表示と動き",
            "表示方向を調整し、登録済みの動きや表情を管理します。",
        )
        self.rotation_360_checkbox = QCheckBox("モデルを横方向へ続けて回転できるようにする")
        self.rotation_360_checkbox.setToolTip(
            "オフは正面から左右180度まで。オンは境界を越えて何周でも回転できます。"
        )
        avatar_view.addWidget(self.rotation_360_checkbox)
        self.always_on_top_checkbox = QCheckBox("ほかのウィンドウより手前に表示する")
        window_settings.addWidget(self.always_on_top_checkbox)
        self.reset_camera_button = QPushButton("雨衣を正面に戻す")
        self.motion_library_button = QPushButton("モーションを管理")
        self.expression_library_button = QPushButton("表情を管理")
        for button in (
            self.reset_camera_button,
            self.motion_library_button,
            self.expression_library_button,
        ):
            avatar_view.addWidget(button)
        avatar_settings.addStretch()

        self.status_panel = QWidget()
        status = QVBoxLayout(self.status_panel)
        status.setContentsMargins(0, 0, 0, 0)
        for name in (
            "provider_label",
            "model_label",
            "connection_label",
            "emotion_label",
            "usage_label",
            "busy_label",
            "speech_status_label",
            "voice_status_label",
            "memory_status_label",
            "context_status_label",
            "proactive_status_label",
            "avatar_name_label",
            "avatar_3d_status_label",
            "mmd_command_label",
            "mouth_morph_label",
            "emotion_morph_label",
            "pose_bone_label",
            "texture_status_label",
            "avatar_expression_label",
            "avatar_gesture_label",
            "avatar_gaze_label",
            "avatar_motion_label",
            "avatar_lip_sync_label",
            "expression_policy_label",
        ):
            label = QLabel()
            label.setWordWrap(True)
            setattr(self, name, label)
            status.addWidget(label)
        self.memory_details_button = QPushButton("使用した記憶")
        self.memory_details_button.hide()
        status.addWidget(self.memory_details_button)
        self.lip_sync_monitor_widget = LipSyncMonitorWidget()
        status.addWidget(self.lip_sync_monitor_widget)
        self._compact_when_pmx = (
            self.avatar_3d_status_label,
            self.mmd_command_label,
            self.mouth_morph_label,
            self.emotion_morph_label,
            self.pose_bone_label,
            self.avatar_expression_label,
            self.avatar_gesture_label,
            self.avatar_gaze_label,
            self.avatar_motion_label,
            self.avatar_lip_sync_label,
        )
        diagnostics.addWidget(self.status_panel)
        diagnostics.addStretch()
        close_options = QPushButton("閉じる")
        close_options.clicked.connect(self.options_dialog.close)
        options_layout.addWidget(close_options, alignment=Qt.AlignmentFlag.AlignRight)
        _fit_dialog(self.options_dialog, 680, 650)
        self.avatar_name_label.setText(f"アバター: {self._avatar_display_name}")
        self.setStyleSheet(
            "QMainWindow, QDialog, QWidget#companionRoot, QWidget#transcriptContent {background: #E8F5F9;}"
            "QWidget {color: #182436; font-family: 'Yu Gothic UI'; font-size: 14px;}"
            "QWidget#composer, QWidget#historyPanel {background: #E8F5F9; border-top: 1px solid #D8EEF6;}"
            "QPushButton {background: #F5FCFE; color: #1A267C; border: 1px solid #C7E4F0; border-radius: 8px; padding: 8px 12px;}"
            "QPushButton[iconOnly=true] {padding: 6px;}"
            "QPushButton:hover {background: #D8EEF6;} QPushButton:pressed {background: #C5E3EF;}"
            "QPushButton:disabled {color: #718396; background: #E5EEF2;}"
            "QPushButton:focus, QTextEdit:focus {border: 2px solid #3D65C6;}"
            "QPushButton#sendButton {background: #A1C9F0; font-weight: 600;}"
            "QPushButton#microphoneButton {background: #B4F0EB;}"
            "QPushButton#sendButton:disabled, QPushButton#microphoneButton:disabled {background: #E5EEF2;}"
            "QPushButton[iconOnly=true]:checked {background: #D8EEF6; border-color: #3D65C6;}"
            "QTextEdit, QListWidget, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {background: #FCFEFF; border: 1px solid #C7E4F0; border-radius: 8px; padding: 8px;}"
            "QGroupBox {border: 1px solid #C7E4F0; border-radius: 8px; margin-top: 12px; padding-top: 12px;}"
            "QGroupBox::title {subcontrol-origin: margin; left: 12px; padding: 0 4px;}"
            "QScrollArea#conversationView {background: #E8F5F9; border: none;}"
            "QSplitter::handle {background: #D8EEF6;}"
            "QSplitter::handle:hover {background: #A1C9F0;}"
            "QScrollBar:vertical {width: 9px; background: transparent;}"
            "QScrollBar::handle:vertical {background: #BCD8E5; border-radius: 4px; min-height: 30px;}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {height: 0px;}"
        )
        self.setTabOrder(self.input_edit, self.send_button)
        self.setTabOrder(self.send_button, self.microphone_button)
        self.setTabOrder(self.microphone_button, self.attach_image_button)
        self.setTabOrder(self.attach_image_button, self.remove_attachment_button)

    def _open_options(self) -> None:
        self._refresh_web_permission()
        self._refresh_connection_source()
        self.options_dialog.show()
        self.options_dialog.raise_()
        self.options_dialog.activateWindow()

    def _refresh_connection_source(self) -> None:
        _, source = self._backend.environment_settings_context()
        self.connection_source_label.setText(
            f"現在使用中: {self._backend.current_provider_name()} / {source}\n"
            "接続先の選択と入力値は、保存してアプリを起動し直すと切り替わります。"
        )

    def _sync_activity_controls(self) -> None:
        external = self._web_turn_active and self._thread is not None
        recording = self._backend.is_listening() or self._ptt_phase in {"録音中", "認識中"}
        self.persona_button.setEnabled(
            not self._busy and not recording and self._proactive_thread is None
        )
        self.stop_voice_button.setEnabled(
            self._busy
            or self._proactive_turn_active
            or self._voice_thread is not None
            or self._pending_voice_text is not None
            or self._backend.is_voice_speaking()
            or self.pmx_preview_widget.is_conversation_motion_active
        )
        self.recognition_panel.setVisible(recording)
        self.conversation_state_label.setVisible(
            self._conversation_state != ConversationState.IDLE and not external
        )
        self.stop_voice_button.setVisible(self.stop_voice_button.isEnabled() and not external)
        self.activity_panel.setVisible(
            recording
            or not self.conversation_state_label.isHidden()
            or not self.stop_voice_button.isHidden()
            or not self.web_activity_label.isHidden()
            or not self.web_cancel_button.isHidden()
        )

    def _connect_events(self) -> None:
        self.send_button.clicked.connect(self._send_message)
        self.input_edit.send_requested.connect(self._send_message)
        self.input_edit.image_dropped.connect(self._attach_image)
        self.input_edit.textChanged.connect(self._input_text_changed)
        self.new_button.clicked.connect(self._start_new_conversation)
        self.delete_button.clicked.connect(self._delete_current_conversation)
        self.settings_button.clicked.connect(self._open_settings)
        self.environment_button.clicked.connect(self._open_environment_settings)
        self.persona_button.clicked.connect(self._open_persona_settings)
        self.memory_button.clicked.connect(self._open_memory_manager)
        self.proactive_button.clicked.connect(self._open_proactive_settings)
        self.motion_library_button.clicked.connect(self._open_motion_manager)
        self.expression_library_button.clicked.connect(self._open_expression_manager)
        self.memory_details_button.clicked.connect(self._open_used_memory_details)
        self.conversation_list.currentItemChanged.connect(self._conversation_selection_changed)
        self.microphone_button.clicked.connect(self._toggle_microphone)
        self.continuous_voice_mode_checkbox.toggled.connect(self._continuous_voice_mode_changed)
        for checkbox, key in (
            (self.auto_send_recognition_checkbox, "voice/auto_send_recognition"),
            (self.continuous_voice_mode_checkbox, "voice/continuous"),
            (self.hold_to_talk_checkbox, "voice/hold_to_talk"),
        ):
            checkbox.toggled.connect(partial(self._save_preference, key))
        self.speech_recognized.connect(self._speech_recognized)
        self.speech_recognition_failed.connect(self._speech_recognition_failed)
        self.speech_partial.connect(self._speech_partial_updated)
        self.speech_level.connect(self.microphone_level.setValue)
        self.reset_camera_button.clicked.connect(self.pmx_preview_widget.reset_camera)
        self.camera_view_button.clicked.connect(self._toggle_camera_view)
        self.rotation_360_checkbox.toggled.connect(self._change_rotation_range)
        self.main_splitter.splitterMoved.connect(self._save_splitter_preferences)
        self.content_splitter.splitterMoved.connect(self._save_splitter_preferences)
        self.pmx_preview_widget.renderer_status_changed.connect(
            self._update_texture_renderer_status
        )
        self.pmx_preview_widget.conversation_pose_ready.connect(self._conversation_pose_ready)
        self._voice_pose_timeout.timeout.connect(self._start_pose_ready_voice_output)
        self.stop_voice_button.clicked.connect(self._stop_voice_output)
        self.always_on_top_checkbox.toggled.connect(self._set_always_on_top)
        for kind, (enabled, seconds) in self._notice_controls.items():
            enabled.toggled.connect(
                lambda _checked, kind=kind: self._update_notice_preferences(kind)
            )
            seconds.valueChanged.connect(
                lambda _seconds, kind=kind: self._update_notice_preferences(kind)
            )
        self._voice_state_timer.timeout.connect(self._on_voice_state_timer)
        self._proactive_timer.timeout.connect(self._on_proactive_timer)

    def _update_status_labels(self) -> None:
        status = self._backend.status()
        self.provider_label.setText(f"プロバイダー: {status.provider_name}")
        self.model_label.setText(f"モデル: {status.model_name}")
        self.connection_label.setText(f"接続状態: {status.connection_text}")
        self.speech_status_label.setText(f"音声認識: {status.speech_recognition_text}")
        self.voice_status_label.setText(f"音声出力: {status.voice_output_text}")
        memory = self._backend.memory_settings()
        self.memory_status_label.setText(
            "保存した記憶: 使用中" if memory.enabled else "保存した記憶: 停止中"
        )
        self.context_status_label.setText(
            f"会話文脈: {self._backend.conversation_context_status()}"
        )
        self._refresh_proactive_status()
        avatar_3d = self._backend.avatar_3d_status()
        self.avatar_3d_status_label.setText(
            f"3D連携: {avatar_3d.controller_name}（{avatar_3d.status_text}）"
        )
        preview = self._backend.avatar_model_preview()
        command = self._backend.mmd_avatar_command()
        self._current_avatar_preview = preview
        self._current_mmd_command = command
        self.pmx_preview_widget.set_preview(preview)
        create_expression = getattr(self._backend, "expression_preview_command", None)
        rest = (
            create_expression("neutral")
            if preview is not None and callable(create_expression)
            else None
        )
        self.pmx_preview_widget._set_rest_expression(dict(rest.expression_morphs) if rest else {})
        self.pmx_preview_widget.set_mmd_command(command)
        self.pmx_preview_widget.setVisible(preview is not None)
        self.camera_view_button.setVisible(preview is not None)
        self.reset_camera_button.setVisible(preview is not None)
        self.avatar_widget.setVisible(preview is None)
        for widget in self._compact_when_pmx:
            widget.setVisible(preview is None)
        self.lip_sync_monitor_widget.setVisible(preview is None)
        self.mmd_command_label.setText(_format_mmd_command(command))
        self.mouth_morph_label.setText(_format_mouth_morphs(preview))
        self.emotion_morph_label.setText(_format_emotion_morph(preview, command))
        self.pose_bone_label.setText(_format_pose_bones(preview, command))
        self._update_texture_renderer_status()
        self._update_avatar_header()

    def _update_texture_renderer_status(self) -> None:
        self.texture_status_label.setText(self.pmx_preview_widget.renderer_status_text)

    def _update_avatar_header(self) -> None:
        """口形と表情を、モデルへ重ねず既存ヘッダーへ表示する。"""

        parts = [f"アバター: {self._avatar_display_name}"]
        if self._current_avatar_preview is not None:
            parts.append(self.avatar_lip_sync_label.text())
            parts.append(
                _format_avatar_expression_summary(
                    self._current_avatar_preview,
                    self._current_mmd_command,
                )
            )
            command = self._current_mmd_command
            motion_name = ""
            if command is not None:
                motion_name = _mmd_motion_display_name(command)
            parts.append(f"MMD動作: {motion_name or '未設定'}")
        self.avatar_name_label.setText("　｜　".join(parts))

    def _update_camera_view_icon(self) -> None:
        full = self._camera_view == "bust"
        _set_control_icon(
            self.camera_view_button,
            "full_body" if full else "bust",
            "全身表示に切り替える" if full else "バストアップに切り替える",
        )

    def _toggle_camera_view(self) -> None:
        self._camera_view = "full" if self._camera_view == "bust" else "bust"
        self.pmx_preview_widget.set_view_mode(self._camera_view)
        self._update_camera_view_icon()
        self._save_preference("camera_view", self._camera_view)

    def _change_rotation_range(self, enabled: bool) -> None:
        self.pmx_preview_widget.set_rotation_360(enabled)
        self._save_preference("camera/rotation_360", enabled)

    def _save_splitter_preferences(self, *_args) -> None:
        self._settings.setValue("layout/main_splitter", self.main_splitter.saveState())
        self._settings.setValue("layout/content_splitter", self.content_splitter.saveState())
        self._settings.sync()

    def _stop_expression_preview(self) -> None:
        """表情確認を終了し、古い会話命令を再開せず通常の目元へ戻す。"""

        if not self._expression_preview_active:
            return
        self.pmx_preview_widget.stop_expression_preview()
        self.pmx_preview_widget.set_lip_sync("neutral")
        self._expression_preview_active = False
        self._current_mmd_command = None
        self._update_avatar_header()

    def _set_always_on_top(self, enabled: bool) -> None:
        was_visible = self.isVisible()
        options_visible = self.options_dialog.isVisible()
        options_focus = self.options_dialog.focusWidget()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        # Windowsでは親だけ最前面にすると、開いている設定が背面へ回る。
        self.options_dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        if was_visible:
            self.show()
        if options_visible:
            self._open_options()
            if options_focus is not None:
                options_focus.setFocus()
        self._save_preference("always_on_top", enabled)

    def _set_emotion_state(
        self,
        state: UiEmotionState | None,
        expression_intensity: float | None = None,
    ) -> None:
        applied_intensity = (
            max(0.0, min(1.0, expression_intensity))
            if expression_intensity is not None
            else (state.intensity if state is not None else 0.0)
        )
        if state is None or state.emotion == "neutral" or state.intensity <= 0.0:
            self.emotion_label.setText("感情: 通常")
            self.avatar_expression_label.setText("表情: 通常")
            self.avatar_widget.set_avatar_state(
                gesture=self.avatar_widget.gesture,
                gaze=self.avatar_widget.gaze,
                gesture_intensity=self.avatar_widget.gesture_intensity,
                gaze_intensity=self.avatar_widget.gaze_intensity,
            )
            return
        labels = {
            "happy": "喜び",
            "sad": "悲しみ",
            "angry": "怒り",
            "embarrassed": "照れ",
            "surprised": "驚き",
        }
        name = labels.get(state.emotion, "通常")
        self.emotion_label.setText(f"感情: {name}（{round(state.intensity * 100)}%）")
        self.avatar_expression_label.setText(f"表情: {name}（{round(applied_intensity * 100)}%）")
        self.avatar_widget.set_avatar_state(
            state.emotion,
            applied_intensity,
            self.avatar_widget.gesture,
            self.avatar_widget.gaze,
            self.avatar_widget.gesture_intensity,
            self.avatar_widget.gaze_intensity,
        )

    def _set_avatar_action(
        self,
        gesture: str = "",
        gaze: str = "",
        gesture_intensity: float = 1.0,
        gaze_intensity: float = 1.0,
        status_text: str = "",
    ) -> None:
        """回答の動作・視線を、表示用のアバターへ渡す。"""

        normalized_gesture = gesture.strip()
        normalized_gaze = gaze.strip()
        gesture_text = f"動作: {_avatar_text(normalized_gesture, 'なし')}"
        gaze_text = f"視線: {_avatar_text(normalized_gaze, '正面')}"
        if status_text:
            gesture_text += f"（{round(max(0.0, min(1.0, gesture_intensity)) * 100)}%）"
            gaze_text += f"（{round(max(0.0, min(1.0, gaze_intensity)) * 100)}%）"
        self.avatar_gesture_label.setText(gesture_text)
        self.avatar_gaze_label.setText(gaze_text)
        self.expression_policy_label.setText(f"表現統合: {status_text or '従来表示'}")
        self.avatar_widget.set_avatar_state(
            self.avatar_widget.emotion,
            self.avatar_widget.intensity,
            normalized_gesture,
            normalized_gaze,
            gesture_intensity,
            gaze_intensity,
        )

    def _set_avatar_lip_sync(self, state: UiLipSyncState, voice_is_active: bool) -> None:
        """VoiSonaの状態と推定口形を、2D・PMX表示へ反映する。"""

        self.avatar_widget.set_lip_sync(state.is_active, state.mouth_shape)
        self.pmx_preview_widget.set_voice_active(voice_is_active)
        self.pmx_preview_widget.set_speech_activity(
            state.is_active,
            state.is_timeline_active,
        )
        current_trace_state = (
            state.is_active,
            state.mouth_shape,
            state.is_estimated,
            state.is_timeline_active,
            voice_is_active,
        )
        if current_trace_state != getattr(self, "_last_lip_sync_trace_state", None):
            self._last_lip_sync_trace_state = current_trace_state
            write_lip_sync_trace(
                "ui",
                "lip_sync_state_applied",
                active=state.is_active,
                mouth=state.mouth_shape,
                estimated=state.is_estimated,
                timeline_active=state.is_timeline_active,
                voice_active=voice_is_active,
            )
        mouth_shape = state.mouth_shape if state.is_active else "neutral"
        self.pmx_preview_widget.set_lip_sync(mouth_shape)
        self.lip_sync_monitor_widget.set_lip_sync(mouth_shape)
        self.stop_voice_button.setEnabled(
            voice_is_active
            or self.pmx_preview_widget.is_conversation_motion_active
            or self._busy
            or self._voice_thread is not None
            or self._pending_voice_text is not None
        )
        if not self.avatar_widget.motion_enabled:
            self.avatar_motion_label.setText("動き: 停止中")
        elif voice_is_active and not state.is_active:
            self.avatar_motion_label.setText("動き: 音声開始待ち")
        else:
            self.avatar_motion_label.setText(
                "動き: 音声再生中" if state.is_active else "動き: 常時動作中"
            )
        if state.is_timeline_active or state.is_active:
            mouth = {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お"}.get(
                state.mouth_shape,
                "閉じる",
            )
            source = "推定" if state.is_estimated else "音声同期"
            label = f"口形: {source} {mouth}"
            self.avatar_lip_sync_label.setText(label)
        elif voice_is_active:
            self.avatar_lip_sync_label.setText("口形: 音声開始を待機中")
        else:
            self.avatar_lip_sync_label.setText("口形: 待機中")
        self._update_avatar_header()

    def _sync_voice_animation(self) -> None:
        voice_is_speaking = self._backend.is_voice_speaking()
        voice_failed = (
            not voice_is_speaking and self._backend.status().voice_output_text == "エラー"
        )
        if voice_failed:
            self._disarm_continuous_voice_restart()
            if not self._voice_error_reported:
                self._show_error("読み上げに失敗しました。返答の本文は会話欄で確認できます。")
        self._voice_error_reported = voice_failed
        self._set_avatar_lip_sync(
            self._backend.voice_lip_sync_state(),
            voice_is_speaking,
        )
        self._continue_after_voice_playback(voice_is_speaking)
        if (
            not voice_is_speaking
            and self._thread is None
            and self._voice_thread is None
            and self._pending_voice_text is None
            and not self._stream_voice_preparing
            and not self._voice_pose_timeout.isActive()
        ):
            self._web_turn_active = False
            self._image_turn_active = False
        if (
            not voice_is_speaking
            and self._proactive_voice_generation == self._utterance_generation
            and self._pending_voice_text is None
            and self._voice_thread is None
        ):
            self._proactive_voice_generation = None
        self._update_conversation_state()

    def _on_voice_state_timer(self) -> None:
        """口形同期タイマーの周期超過だけを診断時に記録する。"""

        if (
            self._web_turn_active
            and not self._image_turn_active
            and getattr(self._backend, "_uses_research_flow", False)
        ):
            phase = getattr(self._backend, "_research_phase", "")
            text = phase + "（OpenAIへ送信中）"
            if phase and self.web_activity_label.text() != text:
                self.web_activity_label.setText(text)

        started_at = time.perf_counter()
        interval_ms = round((started_at - self._last_voice_sync_at) * 1000)
        self._last_voice_sync_at = started_at
        self._sync_voice_animation()
        callback_duration_ms = round((time.perf_counter() - started_at) * 1000)
        period_ms = self._voice_state_timer.interval()
        write_lip_sync_trace(
            "ui",
            "ui_voice_sync_tick",
            period_ms=period_ms,
            interval_ms=interval_ms,
            overrun_ms=max(0, interval_ms - period_ms),
            callback_duration_ms=callback_duration_ms,
        )

    def _on_proactive_timer(self) -> None:
        """会話を割り込ませず、許可された一言だけを非同期で生成する。"""

        settings = self._proactive_settings()
        if (
            not settings.enabled
            or self._current_conversation_id is None
            or self._proactive_turn_active
            or self._proactive_thread is not None
        ):
            return
        if (
            self._proactive.consider(
                settings,
                now=datetime.now(),
                is_busy=self._busy,
                is_listening=self._backend.is_listening(),
                is_voice_speaking=self._backend.is_voice_speaking(),
                has_draft=bool(self.input_edit.toPlainText().strip()),
            )
            is None
        ):
            return
        self._save_proactive_state()
        self._proactive_turn_generation += 1
        generation = self._proactive_turn_generation
        conversation_id = self._current_conversation_id
        self._proactive_turn_active = True
        self.proactive_status_label.setText("能動会話: 考え中")
        thread = QThread(self)
        worker = ProactiveWorker(self._backend, conversation_id, generation)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(self._proactive_turn_succeeded)
        worker.failed.connect(self._proactive_turn_failed)
        worker.cancelled.connect(self._proactive_turn_cancelled)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._proactive_turn_finished)
        self._proactive_thread = thread
        self._proactive_worker = worker
        thread.start()

    def _proactive_turn_succeeded(self, generation: int, value: object) -> None:
        """取消されていない能動発話だけを、通常の会話として反映する。"""

        if generation != self._proactive_turn_generation or not self._proactive_turn_active:
            return
        self._proactive_turn_active = False
        outcome = value if isinstance(value, SendOutcome) else None
        if outcome is None:
            self._show_error("能動発話の結果を読み取れませんでした。")
            self._refresh_proactive_status()
            return
        self._clear_turn_memory_feedback()
        self._messages.append(
            UiMessage("assistant", outcome.speech, created_at=datetime.now().astimezone())
        )
        self._set_emotion_state(
            outcome.emotion_state,
            outcome.expression_intensity if outcome.expression_status_text else None,
        )
        self._set_avatar_action(
            outcome.gesture,
            outcome.gaze,
            outcome.gesture_intensity,
            outcome.gaze_intensity,
            outcome.expression_status_text,
        )
        self._set_used_memories([])
        self._render_messages()
        self._reload_conversations(self._current_conversation_id)
        self._update_usage_label(outcome.input_tokens, outcome.output_tokens)
        self.proactive_status_label.setText("能動会話: 発話済み")
        if self._proactive_settings().read_aloud:
            self._prepare_voice_output(outcome.speech, proactive=True)

    def _proactive_turn_failed(self, generation: int, message: str) -> None:
        """能動発話の失敗は、通常会話を止めずに状態だけ戻す。"""

        if generation != self._proactive_turn_generation or not self._proactive_turn_active:
            return
        self._proactive_turn_active = False
        self._refresh_proactive_status()
        self._show_error(message or "能動発話を開始できませんでした。")

    def _proactive_turn_cancelled(self, generation: int) -> None:
        """利用者操作で無効化した生成は、画面にも履歴にも残さない。"""

        if generation != self._proactive_turn_generation:
            return
        self._proactive_turn_active = False
        self._refresh_proactive_status()

    def _proactive_turn_finished(self) -> None:
        """新しい通常要求とは独立して、旧能動スレッドを最後まで回収する。"""

        self._proactive_thread.wait()
        self._proactive_worker.deleteLater()
        self._proactive_thread.deleteLater()
        self._proactive_thread = None
        self._proactive_worker = None

    def _cancel_pending_proactive_turn(self) -> None:
        """利用者の新しい操作を優先し、返却待ちの結果を無効化する。"""

        if self._proactive_thread is None and not self._proactive_turn_active:
            return
        self._proactive_turn_active = False
        self._proactive_turn_generation += 1
        if self._proactive_worker is not None:
            self._proactive_worker.cancel()
        self._backend.cancel_proactive_turn()

    def _proactive_settings(self) -> ProactiveSettings:
        """既定オフの画面設定を、会話DBと分けて読み出す。"""

        return ProactiveSettings(
            enabled=self._settings.value("proactive/enabled", False, type=bool),
            read_aloud=self._settings.value("proactive/read_aloud", False, type=bool),
            quiet_start_hour=self._settings.value("proactive/quiet_start_hour", 22, type=int),
            quiet_end_hour=self._settings.value("proactive/quiet_end_hour", 8, type=int),
            minimum_idle_minutes=self._settings.value(
                "proactive/minimum_idle_minutes", 30, type=int
            ),
            minimum_interval_minutes=self._settings.value(
                "proactive/minimum_interval_minutes", 120, type=int
            ),
        )

    def _input_text_changed(self) -> None:
        """下書き開始を利用者活動として扱い、通知表示を残さない。"""

        self._update_send_button()
        if self._web_turn_active:
            self._cancel_web_turn()
        if not self.input_edit.toPlainText().strip():
            return
        self._proactive.note_user_activity(datetime.now())
        self._cancel_pending_proactive_turn()
        self._cancel_proactive_voice()
        self._refresh_proactive_status()

    def _refresh_proactive_status(self) -> None:
        """会話を始めたら、能動会話の待機状態を短く表示する。"""

        settings = self._proactive_settings()
        if not settings.enabled:
            self.proactive_status_label.setText("能動会話: 停止中")
        elif self._proactive_turn_active:
            self.proactive_status_label.setText("能動会話: 考え中")
        elif settings.read_aloud:
            self.proactive_status_label.setText("能動会話: 有効（音声あり）")
        else:
            self.proactive_status_label.setText("能動会話: 有効（文字のみ）")

    def _load_proactive_state(self) -> ProactiveState:
        raw_timestamp = str(self._settings.value("proactive/last_delivered_at", ""))
        try:
            last_delivered_at = datetime.fromisoformat(raw_timestamp) if raw_timestamp else None
        except ValueError:
            last_delivered_at = None
        return ProactiveState(last_delivered_at=last_delivered_at)

    def _save_proactive_state(self) -> None:
        state = self._proactive.state
        self._settings.setValue(
            "proactive/last_delivered_at",
            state.last_delivered_at.isoformat() if state.last_delivered_at else "",
        )

    def _continuous_voice_mode_changed(self, enabled: bool) -> None:
        if not enabled:
            self._disarm_continuous_voice_restart()

    def _arm_continuous_voice_restart(self, generation: int) -> None:
        if not self.continuous_voice_mode_checkbox.isChecked():
            return
        if generation != self._utterance_generation:
            return
        self._continuous_voice_waiting_generation = generation
        self._continuous_voice_playback_observed = False

    def _disarm_continuous_voice_restart(self) -> None:
        self._continuous_voice_waiting_generation = None
        self._continuous_voice_playback_observed = False

    def _continue_after_voice_playback(self, voice_is_speaking: bool) -> None:
        generation = self._continuous_voice_waiting_generation
        if generation is None:
            return
        if (
            not self.continuous_voice_mode_checkbox.isChecked()
            or generation != self._utterance_generation
        ):
            self._disarm_continuous_voice_restart()
            return
        if voice_is_speaking:
            self._continuous_voice_playback_observed = True
            return
        if not self._continuous_voice_playback_observed:
            return
        self._disarm_continuous_voice_restart()
        if self._busy or self._backend.is_listening():
            return
        write_lip_sync_trace("companion", "continuous_voice_playback_finished")
        self._start_continuous_voice_recording()

    def _start_continuous_voice_recording(self) -> None:
        if not self.continuous_voice_mode_checkbox.isChecked():
            return
        write_lip_sync_trace("companion", "continuous_recognition_started")
        self._toggle_microphone()

    def _stop_voice_output(self) -> None:
        if self._image_loading:
            self._clear_attachment()
        if self._web_turn_active:
            self._cancel_web_turn()
        elif self._busy:
            self._invalidate_active_stream()
            self.web_activity_label.setText("停止済み（処理の終了待ち）")
            self.web_activity_label.setVisible(self._thread is not None)
        self._cancel_pending_proactive_turn()
        self._proactive_voice_generation = None
        self._invalidate_utterance(interrupted=True)
        self._sync_voice_animation()

    def _interrupt_voice_output(self) -> None:
        """利用者入力を優先し、一つ前の発話世代を全出力で無効化する。"""

        self._proactive_voice_generation = None
        self._invalidate_utterance(interrupted=True)

    def _cancel_proactive_voice(self) -> None:
        """下書きや無効化で、能動発話だけを古い世代にする。"""

        if self._proactive_voice_generation != self._utterance_generation:
            return
        self._proactive_voice_generation = None
        self._invalidate_utterance(interrupted=True)

    def _invalidate_utterance(self, *, interrupted: bool) -> int:
        """音声・口・表情・視線・動作・非同期姿勢を同じ親世代で止める。"""

        had_active_utterance = (
            self._pending_voice_text is not None
            or self._voice_pose_timeout.isActive()
            or self._voice_thread is not None
            or self._stream_voice_started
            or self._stream_voice_preparing
            or self._backend.is_voice_speaking()
        )
        if self._proactive_voice_generation == self._utterance_generation:
            self._proactive_voice_generation = None
        self._utterance_generation += 1
        if self._voice_worker is not None:
            self._voice_worker.cancel()
        self._disarm_continuous_voice_restart()
        self._reset_stream_voice_state()
        self._pending_voice_text = None
        self._pending_pose_revision = None
        self._pending_utterance_generation = None
        self._voice_pose_timeout.stop()
        if self._expression_preview_active:
            self.pmx_preview_widget.stop_expression_preview()
            self._expression_preview_active = False
        self.pmx_preview_widget.interrupt_motion()
        self._current_mmd_command = None
        self._backend.stop_voice_output()
        self._set_avatar_action("", "", 0.0, 0.0, "割り込みにより無効化")
        self._set_emotion_state(None, 0.0)
        if interrupted and had_active_utterance:
            self._set_conversation_state(ConversationState.INTERRUPTED)
        self._sync_voice_animation()
        return self._utterance_generation

    def _update_conversation_state(self) -> None:
        if self._backend.is_voice_speaking():
            state = ConversationState.SPEAKING
        elif self._pending_voice_text is not None or self._voice_pose_timeout.isActive():
            state = ConversationState.REACTING
        elif self._busy or self._proactive_turn_active:
            state = ConversationState.THINKING
        elif self._backend.is_listening():
            state = ConversationState.LISTENING
        else:
            state = ConversationState.IDLE
        self._set_conversation_state(state)

    def _set_conversation_state(self, state: ConversationState) -> None:
        """画面表示と3D自然動作へ、同じ正式状態を一度だけ渡す。"""

        previous = self._conversation_state
        now = time.monotonic()
        self._conversation_state = state
        self.conversation_state_label.setText(f"会話状態: {state.display_name}")
        self.pmx_preview_widget.set_conversation_state(state)
        self._sync_activity_controls()
        if state == previous:
            return
        if state == ConversationState.INTERRUPTED:
            self.notice.show_notice("音声を停止しました。")
        write_lip_sync_trace(
            "companion",
            "conversation_state_transition",
            generation=self._stream_generation,
            previous_state=previous.value,
            state=state.value,
            duration_ms=round((now - self._conversation_state_started_at) * 1000),
        )
        self._conversation_state_started_at = now

    def _update_usage_label(
        self,
        latest_input_tokens: int | None = None,
        latest_output_tokens: int | None = None,
    ) -> None:
        """直近の応答と、SQLiteへ保存済みの合計を表示する。"""

        conversation_summary = self._backend.token_usage_summary(self._current_conversation_id)
        all_summary = self._backend.token_usage_summary()
        if latest_input_tokens is None and latest_output_tokens is None:
            latest_text = "未取得"
        else:
            input_text = "不明" if latest_input_tokens is None else f"{latest_input_tokens:,}"
            output_text = "不明" if latest_output_tokens is None else f"{latest_output_tokens:,}"
            latest_text = f"入 {input_text} / 出 {output_text}"
        self.usage_label.setText(
            "トークン使用量\n"
            f"今回: {latest_text}\n"
            f"会話: {_format_usage_summary(conversation_summary)}\n"
            f"保存済み: {_format_usage_summary(all_summary)}"
        )

    def _toggle_microphone(self) -> None:
        if self._image_loading:
            self._clear_attachment()
        if self._backend.is_listening():
            self._disarm_continuous_voice_restart()
            self._recognition_generation += 1
            self._set_ptt_phase("停止中")
            write_lip_sync_trace("companion", "recognition_stop_requested")
            self._backend.stop_speech_recognition()
            self._set_ptt_phase("待機")
            _set_control_icon(self.microphone_button, "microphone", "マイク")
            self.settings_button.setEnabled(True)
            self.motion_library_button.setEnabled(True)
            self.expression_library_button.setEnabled(True)
            self.partial_label.clear()
            self.microphone_level.setValue(0)
            self._update_status_labels()
            self._update_conversation_state()
            return
        self._disarm_continuous_voice_restart()
        self._proactive.note_user_activity(datetime.now())
        self._cancel_pending_proactive_turn()
        self._refresh_proactive_status()
        write_lip_sync_trace("companion", "ptt_pressed")
        self._invalidate_active_stream()
        was_voice_active = self._backend.is_voice_speaking()
        if was_voice_active:
            write_lip_sync_trace("companion", "voice_stop_requested")
        self._interrupt_voice_output()
        if was_voice_active and not self._backend.is_voice_speaking():
            write_lip_sync_trace("companion", "voice_stopped")
        self._recognition_generation += 1
        generation = self._recognition_generation
        self._set_ptt_phase("録音中")
        self._clear_error()
        write_lip_sync_trace("companion", "recognition_started")
        self._backend.start_speech_recognition(
            lambda text: self._emit_recognized(generation, text),
            lambda message: self._emit_recognition_failed(generation, message),
            lambda text: self._emit_speech_partial(generation, text),
            lambda level: self._emit_speech_level(generation, level),
        )
        if self._backend.is_listening():
            _set_control_icon(self.microphone_button, "stop", "録音停止")
            self.settings_button.setEnabled(False)
            self.motion_library_button.setEnabled(False)
            self.expression_library_button.setEnabled(False)
        self._update_status_labels()
        self._update_conversation_state()

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # noqa: N802
        """タイトルバーの配色と、入力を妨げないアプリ内PTT。"""

        if (
            event.type() in (QEvent.Type.Show, QEvent.Type.WinIdChange)
            and isinstance(watched, QWidget)
            and watched.isWindow()
        ):
            owner = watched
            while owner is not None and owner is not self:
                owner = owner.parentWidget()
            if owner is self:
                _apply_title_bar_palette(watched)

        if (
            event.type() == QEvent.Type.MouseButtonPress
            and self.input_edit.hasFocus()
            and isinstance(watched, QWidget)
            and watched is not self.input_edit
            and not self.input_edit.isAncestorOf(watched)
            and watched is not self.notice
            and not self.notice.isAncestorOf(watched)
            and watched is not self.dismiss_error_button
        ):
            self.input_edit.clearFocus()
        if (
            not self.isActiveWindow()
            or not self.hold_to_talk_checkbox.isChecked()
            or self.input_edit.hasFocus()
            or event.type() not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease)
            or not isinstance(event, QKeyEvent)
            or event.key() != Qt.Key.Key_Space
            or event.modifiers() != Qt.KeyboardModifier.NoModifier
            or event.isAutoRepeat()
        ):
            return super().eventFilter(watched, event)
        if event.type() == QEvent.Type.KeyPress:
            self._start_hold_to_talk()
        else:
            self._finish_hold_to_talk()
        return True

    def _start_hold_to_talk(self) -> None:
        if self._hold_to_talk_active or self._backend.is_listening():
            return
        self._hold_to_talk_active = True
        self._toggle_microphone()
        if not self._backend.is_listening():
            self._hold_to_talk_active = False

    def _finish_hold_to_talk(self) -> None:
        if not self._hold_to_talk_active:
            return
        self._hold_to_talk_active = False
        if not self._backend.is_listening():
            return
        self._set_ptt_phase("認識中")
        write_lip_sync_trace("companion", "ptt_released")
        _set_control_icon(self.microphone_button, "microphone", "認識中")
        self.microphone_button.setEnabled(False)
        self._backend.finish_speech_recognition()

    def _emit_recognized(self, generation: int, text: str) -> None:
        if generation == self._recognition_generation:
            self.speech_recognized.emit(text)

    def _emit_recognition_failed(self, generation: int, message: str) -> None:
        if generation == self._recognition_generation:
            self._recognition_generation += 1
            self.speech_recognition_failed.emit(message)

    def _emit_speech_partial(self, generation: int, text: str) -> None:
        if generation == self._recognition_generation:
            self.speech_partial.emit(text)

    def _emit_speech_level(self, generation: int, level: int) -> None:
        if generation == self._recognition_generation:
            self.speech_level.emit(level)

    def _speech_recognized(self, text: str) -> None:
        write_lip_sync_trace("companion", "recognition_finished")
        self._set_ptt_phase("認識済み")
        recognized = text.strip()
        _set_control_icon(self.microphone_button, "microphone", "マイク")
        self.microphone_button.setEnabled(self._backend.is_speech_recognition_available())
        self.settings_button.setEnabled(not self._busy)
        self.motion_library_button.setEnabled(not self._busy)
        self.expression_library_button.setEnabled(not self._busy)
        self.partial_label.clear()
        self.microphone_level.setValue(0)
        self._update_status_labels()
        if (
            self.auto_send_recognition_checkbox.isChecked()
            or self.continuous_voice_mode_checkbox.isChecked()
        ):
            if self._auto_sent_recognition_generation == self._recognition_generation:
                return
            if not recognized:
                return
            self._auto_sent_recognition_generation = self._recognition_generation
            self.input_edit.setPlainText(recognized)
            if self._busy:
                self._pending_auto_send_text = recognized
                return
            self._send_message()
            return
        current = self.input_edit.toPlainText().strip()
        self.input_edit.setPlainText(f"{current} {text}".strip())
        self.input_edit.setFocus()

    def _speech_recognition_failed(self, message: str) -> None:
        self._disarm_continuous_voice_restart()
        write_lip_sync_trace("companion", "recognition_finished")
        self._set_ptt_phase("待機")
        _set_control_icon(self.microphone_button, "microphone", "マイク")
        self.settings_button.setEnabled(not self._busy)
        self.motion_library_button.setEnabled(not self._busy)
        self.expression_library_button.setEnabled(not self._busy)
        self._show_error(message)
        self.partial_label.clear()
        self.microphone_level.setValue(0)
        self._update_status_labels()

    def _speech_partial_updated(self, text: str) -> None:
        self.partial_label.setText(f"認識途中: {text}" if text else "")

    def _set_ptt_phase(self, phase: str) -> None:
        """PTTの録音と認識確定待ちを画面上で区別する。"""

        self._ptt_phase = phase
        self.ptt_phase_label.setText(f"マイク: {phase}")
        self._sync_activity_controls()

    def _reload_conversations(self, select_id: str | None = None) -> None:
        if select_id is None:
            select_id = self._current_conversation_id
        self.conversation_list.blockSignals(True)
        self.conversation_list.clear()
        selected_item: QListWidgetItem | None = None
        for conversation in self._backend.list_conversations():
            item = QListWidgetItem(conversation.title)
            item.setData(Qt.ItemDataRole.UserRole, conversation.id)
            self.conversation_list.addItem(item)
            if conversation.id == select_id:
                selected_item = item
        self.conversation_list.blockSignals(False)

        if selected_item is not None:
            self.conversation_list.setCurrentItem(selected_item)
        elif self.conversation_list.count() > 0:
            self.conversation_list.setCurrentRow(0)
        else:
            self._current_conversation_id = None
            self._messages = []
            self._set_emotion_state(None)
            self._set_avatar_action()
            self._render_messages()
            self._update_usage_label()
        self._update_action_buttons()

    def _conversation_selection_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None or self._busy or self._proactive_thread is not None:
            return
        conversation_id = str(current.data(Qt.ItemDataRole.UserRole))
        if conversation_id != self._current_conversation_id:
            self.chat_view._follow_latest()
        self._current_conversation_id = conversation_id
        self._messages = self._backend.load_messages(conversation_id)
        action = self._backend.avatar_action(conversation_id)
        self._set_emotion_state(
            self._backend.emotion_state(conversation_id),
            action.expression_intensity if action.status_text else None,
        )
        self._set_avatar_action(
            action.gesture,
            action.gaze,
            action.gesture_intensity,
            action.gaze_intensity,
            action.status_text,
        )
        self._clear_previous_conversation_feedback()
        self._update_usage_label()
        self._clear_error()
        self._render_messages()
        self._update_action_buttons()

    def _render_messages(self) -> None:
        if not self._messages:
            self.chat_view.set_messages([])
            return

        parts: list[tuple[str, str, str]] = []
        timestamps: list[datetime | None] = []
        images: list[str | None] = []
        for index, message in enumerate(self._messages):
            if not message.content and not message.sources and not message.image_attachment:
                continue
            is_user = message.role == "user"
            css_class = "user" if is_user else "assistant"
            speaker = "あなた" if is_user else self._avatar_display_name
            content = html.escape(message.content).replace("\n", "<br>")
            source_lines = []
            for source_index, source in enumerate(message.sources, 1):
                if not safe_source_url(source.url):
                    continue
                url, title = html.escape(source.url, quote=True), html.escape(source.title)
                content = content.replace(
                    f"[S{source_index}]", f'<a href="{url}">[S{source_index}]</a>'
                )
                published = source.published_at.isoformat() if source.published_at else "不明"
                source_lines.append(
                    f'<a href="{url}">[S{source_index}] {title}</a><br>'
                    f"取得: {html.escape(source.retrieved_at.isoformat(timespec='seconds'))} / "
                    f"公開日時: {html.escape(published)}"
                )
            if source_lines:
                content += "<br><small>出典<br>" + "<br>".join(source_lines) + "</small>"
            info = message.image_attachment
            images.append(info.preview_jpeg if info else None)
            has_user_preview = (
                not is_user
                and index > 0
                and self._messages[index - 1].role == "user"
                and self._messages[index - 1].image_attachment is not None
                and self._messages[index - 1].image_attachment.preview_jpeg is not None
            )
            if info and not info.preview_jpeg and not has_user_preview:
                info = message.image_attachment
                content += (
                    "<br><small>添付画像: "
                    + html.escape(info.name)
                    + f" / 解析範囲 {info.width}×{info.height}px"
                    + " / 原本は保存していません。再確認には再添付してください。</small>"
                )
            parts.append((css_class, speaker, content))
            timestamps.append(message.created_at)
        self.chat_view.set_messages(parts, timestamps=timestamps, images=images)

    def _send_message(self) -> None:
        text = self.input_edit.toPlainText().strip()
        if self._busy or self._image_loading or not text:
            return
        if self._proactive_thread is not None:
            self._cancel_pending_proactive_turn()

        attached = self._attached_image
        if attached is not None:
            if not getattr(self._backend, "image_sharing_available", lambda: False)():
                self._show_error("画像共有にはOpenAI公式APIへの接続が必要です。")
                return
            if not self._backend.image_sharing_enabled():
                self._show_error("画像送信はオフです。設定の「会話」から許可できます。")
                return
        research_turn = attached is None and getattr(self._backend, "_uses_research_flow", False)
        query = self._web_query(text) if attached is None and not research_turn else ""
        search_turn = (
            attached is None
            and not research_turn
            and (is_search_request(text) or is_search_request(query))
            and not _needs_weather_region(query)
        )
        if search_turn and not self._prepare_web_search(text):
            return

        self._proactive.note_user_activity(datetime.now())
        self._refresh_proactive_status()
        self._interrupt_voice_output()
        self._set_ptt_phase("待機")
        self._stream_generation += 1
        generation = self._stream_generation
        self._reset_stream_voice_state()
        self._stream_timing_started_at = time.monotonic()
        self._stream_first_chunk_logged = False
        status = self._backend.status()
        write_lip_sync_trace(
            "companion",
            "input_finished",
            provider_name=status.provider_name,
            model_name=status.model_name,
        )
        self._clear_turn_memory_feedback()
        self._clear_error()
        try:
            if self._current_conversation_id is None:
                title = text.replace("\n", " ")[:30]
                self._current_conversation_id = self._backend.create_conversation(title)
            conversation_id = self._current_conversation_id
        except Exception as exc:
            self._show_error(str(exc) or "新しい会話を作成できませんでした。")
            return

        self._messages.append(
            UiMessage(
                role="user",
                content=text,
                created_at=datetime.now().astimezone(),
                image_attachment=attached.history_info() if attached else None,
            )
        )
        self._messages.append(
            UiMessage(
                role="assistant",
                content="調べているよ。結果が分かったらここで伝えるね。" if search_turn else "",
            )
        )
        self.input_edit.clear()
        self._clear_attachment()
        self._image_turn_active = attached is not None
        # 画像も既存の外部処理の停止・音声割り込み経路へ接続する。
        self._web_turn_active = research_turn or search_turn or self._image_turn_active
        self.web_activity_label.setText(
            "画像解析・返答のためOpenAIへ送信中"
            if attached
            else "返事を考えています（OpenAIへ送信中）"
            if research_turn
            else "Web検索・返答のためOpenAIへ送信中"
        )
        self.web_cancel_button.setText(
            "画像・返答を取消"
            if attached
            else "返答を取消"
            if research_turn
            else "検索・返答を取消"
        )
        self.web_activity_label.setVisible(self._web_turn_active)
        self.web_cancel_button.setVisible(self._web_turn_active)
        self.chat_view._follow_latest()
        self._render_messages()
        self._set_busy(True)

        thread = QThread(self)
        worker = StreamWorker(self._backend, conversation_id, text, generation, image=attached)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.chunk_received.connect(self._append_assistant_chunk)
        worker.succeeded.connect(self._message_succeeded)
        worker.failed.connect(self._message_failed)
        worker.cancelled.connect(self._message_cancelled)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._worker_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _refresh_web_permission(self) -> None:
        enabled = getattr(self._backend, "web_search_enabled", lambda: False)()
        self.web_permission_checkbox.setChecked(enabled)
        self.web_permission_checkbox.setEnabled(
            getattr(self._backend, "web_search_available", lambda: False)()
        )
        self.image_permission_checkbox.setChecked(
            getattr(self._backend, "image_sharing_enabled", lambda: False)()
        )
        self.image_permission_checkbox.setEnabled(
            getattr(self._backend, "image_sharing_available", lambda: False)()
        )

    def _select_image(self) -> None:
        if self._busy:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "会話に添付する画像1枚を選択", "", "静止画像 (*.png *.jpg *.jpeg *.webp)"
        )
        if not path:
            return
        self._attach_image(path)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if (
            not self._busy
            and event.possibleActions() & Qt.DropAction.CopyAction
            and _image_drop_source(event.mimeData())
        ):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802
        self.dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        source = _image_drop_source(event.mimeData())
        if not self._busy and source and event.possibleActions() & Qt.DropAction.CopyAction:
            self._attach_image(source)
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def _attach_image(self, path: str | _ImageDrop) -> None:
        if self._busy:
            return
        if isinstance(path, _ImageDrop):
            self._cancel_image_fetch()
            self._image_loading = True
            self._clear_error()
            self.attachment_preview.clear()
            self.attachment_preview.setText("画像を取得中…")
            _set_control_icon(self.remove_attachment_button, "remove", "画像の取得を取消")
            self.attachment_panel.show()
            self._update_send_button()
            generation = self._image_fetch_generation
            thread = QThread(self)
            thread.setProperty("image_generation", generation)
            worker = _ImageDropWorker(path, generation)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.succeeded.connect(self._image_fetch_succeeded)
            worker.failed.connect(self._image_fetch_failed)
            worker.finished.connect(thread.quit)
            thread.finished.connect(self._image_fetch_finished)
            self._image_fetches[generation] = (thread, worker)
            self._proactive.note_user_activity(datetime.now())
            self._cancel_pending_proactive_turn()
            self._cancel_proactive_voice()
            thread.start()
            return
        try:
            image = prepare_image(path)
        except RuntimeError as exc:
            self._show_error(str(exc))
            return
        self._cancel_image_fetch()
        self._set_attached_image(image)

    def _set_attached_image(self, image: PreparedImage) -> None:
        self._attached_image = image
        self._image_loading = False
        _set_control_icon(self.remove_attachment_button, "remove", "画像を外す")
        preview = QPixmap()
        preview.loadFromData(image.jpeg, "JPEG")
        self.attachment_preview.setPixmap(
            preview.scaled(
                92,
                72,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.attachment_panel.show()
        self._proactive.note_user_activity(datetime.now())
        self._cancel_pending_proactive_turn()
        self._cancel_proactive_voice()
        self.input_edit.setFocus()
        self._update_send_button()

    def _cancel_image_fetch(self) -> None:
        self._image_fetch_generation += 1
        self._image_loading = False
        for _thread, worker in self._image_fetches.values():
            worker.cancel()

    def _image_fetch_succeeded(self, generation, image) -> None:
        if generation == self._image_fetch_generation:
            self._set_attached_image(image)

    def _image_fetch_failed(self, generation, message) -> None:
        if generation != self._image_fetch_generation:
            return
        self._image_loading = False
        if self._attached_image is not None:
            self._set_attached_image(self._attached_image)
        else:
            self.attachment_panel.hide()
            self._update_send_button()
        self._show_error(message)

    def _image_fetch_finished(self) -> None:
        thread = self.sender()
        thread.wait()
        _thread, worker = self._image_fetches.pop(thread.property("image_generation"))
        worker.deleteLater()
        thread.deleteLater()
        if self._close_after_image_fetch and not self._image_fetches:
            QTimer.singleShot(0, self, self.close)

    def _clear_attachment(self) -> None:
        self._cancel_image_fetch()
        self._attached_image = None
        self.attachment_preview.clear()
        self.attachment_panel.hide()
        self._update_send_button()

    def _web_query(self, text: str) -> str:
        previous = ""
        proposal = ""
        # 中断・失敗して返答がない依頼を、地域だけの次の入力で再開しない。
        if (
            len(self._messages) >= 2
            and self._messages[-2].role == "user"
            and self._messages[-1].role == "assistant"
            and self._messages[-1].content
        ):
            previous = self._messages[-2].content
            proposal = self._messages[-1].content
        query = _search_query(text, previous, proposal)
        if _needs_weather_region(text):
            # 同じ天気の話が続く間だけ地域を参照する。新しい依頼なしでは再開しない。
            region = _weather_region(query)
            for index in range(len(self._messages) - 1, 0, -1):
                message = self._messages[index]
                if message.role != "assistant" or not message.content:
                    if message.role == "user" and not (
                        is_weather_request(message.content)
                        or re.match(r"^(?:それ|その件|その内容|その条件)を?調べて", message.content)
                    ):
                        break
                    continue
                prior = self._messages[index - 1]
                if prior.role != "user" or not (
                    is_weather_request(prior.content) or is_weather_request(message.content)
                ):
                    break
                candidate = _search_query(text, prior.content, message.content)
                if candidate != text:
                    candidate_region = _weather_region(candidate)
                    if region and candidate_region != region:
                        break
                    query, region = candidate, candidate_region
                    if _weather_period(text, (prior.content, message.content)):
                        return query
                elif _weather_regions(prior.content) or _weather_regions(message.content):
                    break
                # 地域を言い直さない最終返答の手前にある、同じ話題だけを見る。
        return query

    def _toggle_image_permission(self, enabled: bool) -> None:
        if enabled and not getattr(self._backend, "image_sharing_available", lambda: False)():
            self.image_permission_checkbox.setChecked(False)
            self._show_error("画像共有にはOpenAI公式APIへの接続が必要です。")
            return
        if not enabled:
            if self._image_turn_active:
                self._cancel_web_turn()
            self._clear_attachment()
        setter = getattr(self._backend, "set_image_sharing_enabled", None)
        if callable(setter):
            setter(enabled)
        self.image_permission_checkbox.setChecked(enabled)
        self._save_preference("image_sharing/enabled", enabled)

    def _toggle_web_permission(self, enabled: bool) -> None:
        if enabled and not getattr(self._backend, "web_search_available", lambda: False)():
            self.web_permission_checkbox.setChecked(False)
            self._show_error("Web検索にはOpenAI公式APIへの接続が必要です。")
            return
        if not enabled and not self._image_turn_active:
            self._cancel_web_turn()
        setter = getattr(self._backend, "set_web_search_enabled", None)
        if callable(setter):
            setter(enabled)
        self.web_permission_checkbox.setChecked(enabled)
        self._save_preference("web_search/enabled", enabled)

    def _prepare_web_search(self, text: str) -> bool:
        """許可設定を尊重し、今回送信する検索文だけを登録する。"""
        if not getattr(self._backend, "web_search_available", lambda: False)():
            self._show_error(
                "現在の接続先・モデルではWeb検索に対応していません。検索はしていません。"
            )
            return False
        if not self._backend.web_search_enabled():
            self._show_error("Web検索はオフです。設定の「会話」から許可できます。")
            return False
        query = self._web_query(text)
        if not query:
            self._show_error("検索したい内容も入力してください（例：明日の千代田区の天気は？）。")
            return False
        try:
            self._backend.approve_web_search(
                text,
                query,
                "詳しく" in text or "詳しい" in text,
                "",
            )
        except RuntimeError as exc:
            self._show_error(str(exc))
            return False
        self._refresh_web_permission()
        return True

    def _cancel_web_turn(self) -> None:
        """外部結果と、その後の音声・表現を同じ操作で無効化する。"""

        if not self._web_turn_active:
            return
        if self._worker is not None:
            self._worker.cancel()
        self._web_turn_active = False
        self._image_turn_active = False
        if self._thread is not None:
            self._invalidate_active_stream()
        self._invalidate_utterance(interrupted=True)
        self.web_activity_label.setText("取消済み（通信の終了待ち）")
        self.web_activity_label.setVisible(self._thread is not None)
        self.web_cancel_button.hide()

        self.notice.show_notice("検索・返答を取り消しました。")

    def _append_assistant_chunk(self, generation: int, chunk: str) -> None:
        if generation != self._stream_generation or not chunk or not self._messages:
            return
        # 検索中の仮本文は保存・読み上げせず、最初の実結果で置き換える。
        replace_search_acknowledgement = (
            self._web_turn_active
            and not self._image_turn_active
            and not self._stream_first_chunk_logged
        )
        if not self._stream_first_chunk_logged:
            self._stream_first_chunk_logged = True
            write_lip_sync_trace(
                "companion",
                "llm_first_chunk",
                input_to_llm_first_chunk_ms=self._elapsed_stream_ms(),
            )
        current = self._messages[-1]
        self._messages[-1] = replace(
            current,
            content=("" if replace_search_acknowledgement else current.content) + chunk,
            created_at=current.created_at or datetime.now().astimezone(),
        )
        self._render_messages()
        for sentence in self._stream_sentence_buffer.feed(chunk):
            self._queue_stream_sentence(sentence)

    def _message_succeeded(self, generation: int | object, value: object | None = None) -> None:
        if value is None:
            value = generation
            generation = self._stream_generation
        if generation != self._stream_generation:
            return
        outcome = value if isinstance(value, SendOutcome) else None
        if outcome is None:
            self._show_error("LLMからの結果を読み取れませんでした。")
            return
        if self._messages:
            self._messages[-1] = UiMessage(
                role="assistant",
                content=outcome.speech,
                sources=outcome.sources,
                image_attachment=outcome.image_attachment,
                created_at=self._messages[-1].created_at or datetime.now().astimezone(),
            )
        self.connection_label.setText("接続状態: 接続済み")
        self._update_status_labels()
        if self._stream_voice_started or self._stream_voice_preparing:
            revision = self.pmx_preview_widget.begin_conversation_motion()
            if self._stream_voice_preparing:
                self._stream_pose_revision = revision
        self._set_emotion_state(
            outcome.emotion_state,
            outcome.expression_intensity if outcome.expression_status_text else None,
        )
        self._set_avatar_action(
            outcome.gesture,
            outcome.gaze,
            outcome.gesture_intensity,
            outcome.gaze_intensity,
            outcome.expression_status_text,
        )
        update_voice_expression = getattr(self._backend, "_update_stream_voice_expression", None)
        if callable(update_voice_expression):
            update_voice_expression()
        remaining = self._stream_sentence_buffer.flush()
        if remaining:
            self._queue_stream_sentence(remaining)
        self._finish_stream_voice()
        if not self._stream_voice_started and not self._stream_voice_preparing:
            self._prepare_voice_output(outcome.speech)
        if outcome.memory_candidates:
            self._show_saved_memories(outcome.memory_candidates)
        self._set_used_memories(outcome.used_memories)
        self._render_messages()
        self._reload_conversations(self._current_conversation_id)
        self._update_usage_label(outcome.input_tokens, outcome.output_tokens)

    def _elapsed_stream_ms(self) -> int | None:
        """入力終了からの経過時間を診断記録用に整数ミリ秒で返す。"""

        if self._stream_timing_started_at is None:
            return None
        return round((time.monotonic() - self._stream_timing_started_at) * 1000)

    def _queue_stream_sentence(self, sentence: str) -> None:
        """確定した一文を、姿勢準備後の音声キューへ渡す。"""

        if not sentence.strip():
            return
        start_stream = getattr(self._backend, "start_stream_voice_output", None)
        enqueue_stream = getattr(self._backend, "enqueue_stream_voice_output", None)
        if not callable(start_stream) or not callable(enqueue_stream):
            return
        self._stream_sentence_index += 1
        write_lip_sync_trace(
            "companion",
            "llm_sentence_ready",
            sentence_index=self._stream_sentence_index,
            input_to_sentence_ms=self._elapsed_stream_ms(),
        )
        if self._stream_voice_started:
            enqueue_stream(sentence)
            return
        self._stream_pending_sentences.append(sentence)
        if self._stream_voice_preparing:
            return
        self._stream_voice_preparing = True
        self._utterance_generation += 1
        self._set_conversation_state(ConversationState.REACTING)
        revision = self.pmx_preview_widget.begin_conversation_motion()
        self._stream_pose_revision = revision
        if revision is None:
            self._start_stream_voice_after_pose()
        else:
            self._voice_pose_timeout.start()

    def _start_stream_voice_after_pose(self) -> None:
        """姿勢準備が終わったら、保留文を順番に先読みへ渡す。"""

        if not self._stream_voice_preparing:
            return
        start_stream = getattr(self._backend, "start_stream_voice_output", None)
        enqueue_stream = getattr(self._backend, "enqueue_stream_voice_output", None)
        if not callable(start_stream) or not callable(enqueue_stream):
            self._stream_voice_preparing = False
            self._stream_pending_sentences.clear()
            return
        self._stream_voice_preparing = False
        self._stream_pose_revision = None
        self._voice_pose_timeout.stop()
        pending = self._stream_pending_sentences
        self._stream_pending_sentences = []
        if not pending:
            return
        error = start_stream(pending[0])
        if isinstance(error, str) and error:
            self._show_error(error)
            return
        self._stream_voice_started = True
        self._arm_continuous_voice_restart(self._utterance_generation)
        write_lip_sync_trace(
            "companion",
            "voisona_stream_started",
            input_to_voice_request_ms=self._elapsed_stream_ms(),
        )
        for sentence in pending[1:]:
            enqueue_stream(sentence)
        if self._stream_voice_finished:
            self._finish_stream_voice()
        if self._thread is None:
            self._set_busy(False)

    def _finish_stream_voice(self) -> None:
        """回答終了を音声キューへ通知し、残りの文を再生し切る。"""

        self._stream_voice_finished = True
        if not self._stream_voice_started:
            return
        finish_stream = getattr(self._backend, "finish_stream_voice_output", None)
        if callable(finish_stream):
            finish_stream()
        write_lip_sync_trace(
            "companion",
            "voisona_stream_finished",
            input_to_voice_finish_ms=self._elapsed_stream_ms(),
        )

    def _reset_stream_voice_state(self) -> None:
        """次の回答へ切り替える際に、先読み状態と計測を初期化する。"""

        self._stream_sentence_buffer.reset()
        self._stream_voice_started = False
        self._stream_voice_preparing = False
        self._stream_voice_finished = False
        self._stream_pending_sentences.clear()
        self._stream_pose_revision = None
        self._stream_sentence_index = 0
        self._stream_timing_started_at = None
        self._stream_first_chunk_logged = False
        self._voice_pose_timeout.stop()

    def _start_pose_ready_voice_output(self) -> None:
        """姿勢タイムアウト時も、通常発話と先読み発話を同じ入口で処理する。"""

        if self._stream_voice_preparing:
            self._start_stream_voice_after_pose()
        else:
            self._start_pending_voice_output()

    def _prepare_voice_output(self, speech: str, *, proactive: bool = False) -> None:
        """会話姿勢を先に用意し、準備完了後に一度だけ音声を開始する。"""

        self._utterance_generation += 1
        generation = self._utterance_generation
        self._proactive_voice_generation = generation if proactive else None
        self._pending_voice_text = speech
        self._pending_utterance_generation = generation
        self._set_conversation_state(ConversationState.REACTING)
        revision = self.pmx_preview_widget.begin_conversation_motion()
        self._pending_pose_revision = revision
        if revision is None:
            self._start_pending_voice_output()
            return
        self._voice_pose_timeout.start()

    def _conversation_pose_ready(self, revision: int) -> None:
        if self._stream_voice_preparing and revision == self._stream_pose_revision:
            self._start_stream_voice_after_pose()
            return
        if revision != self._pending_pose_revision:
            return
        self._start_pending_voice_output()

    def _start_pending_voice_output(self) -> None:
        text = self._pending_voice_text
        generation = self._pending_utterance_generation
        if text is None or generation is None or generation != self._utterance_generation:
            return
        self._pending_pose_revision = None
        self._voice_pose_timeout.stop()
        # 前の受付処理を保持し、新しい世代はその終了後に一度だけ開始する。
        if self._voice_thread is not None:
            return
        self._pending_voice_text = None
        self._pending_utterance_generation = None
        thread = QThread(self)
        worker = VoiceStartWorker(
            self._backend,
            text,
            generation,
            proactive=generation == self._proactive_voice_generation,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._voice_output_started)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._voice_worker_finished)
        self._voice_thread = thread
        self._voice_worker = worker
        thread.start()

    def _voice_output_started(self, value: object) -> None:
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], int):
            generation = value[0]
            result = value[1]
        else:
            generation = self._utterance_generation
            result = value
        if generation != self._utterance_generation:
            return
        error = result if isinstance(result, str) and result else None
        if error:
            self._show_error(error)
        elif generation != self._proactive_voice_generation:
            self._arm_continuous_voice_restart(generation)
        self._sync_voice_animation()
        if not self._backend.is_voice_speaking():
            self.pmx_preview_widget.end_conversation_motion()
            if generation == self._proactive_voice_generation:
                self._proactive_voice_generation = None

    def _voice_worker_finished(self) -> None:
        self._voice_thread.wait()
        self._voice_worker.deleteLater()
        self._voice_thread.deleteLater()
        self._voice_thread = None
        self._voice_worker = None
        if self._close_after_voice:
            QTimer.singleShot(0, self, self.close)
        elif self._pending_voice_text is not None and self._pending_pose_revision is None:
            self._start_pending_voice_output()
        if (
            self._thread is None
            and self._voice_thread is None
            and self._pending_voice_text is None
            and not self._stream_voice_preparing
        ):
            self._set_busy(False)

    def _message_cancelled(self, generation: int) -> None:
        if generation != self._stream_generation:
            return
        if self._messages and self._messages[-1].role == "assistant":
            self._messages.pop()
        self._reset_stream_voice_state()
        self._render_messages()
        self.notice.show_notice("返答を取り消しました。")

    def _message_failed(self, generation: int, message: str) -> None:
        if generation != self._stream_generation:
            return
        self._web_turn_active = False
        self._image_turn_active = False
        if self._stream_voice_started or self._stream_voice_preparing:
            self._backend.stop_voice_output()
            self.pmx_preview_widget.interrupt_motion()
        self._reset_stream_voice_state()
        if self._messages and self._messages[-1].role == "assistant":
            self._messages.pop()
        self.connection_label.setText("接続状態: エラー")
        self._show_error(message)
        self._render_messages()

    def _invalidate_active_stream(self) -> None:
        """認識開始時に古い生成の画面反映と保存を無効にする。"""

        if not self._busy:
            return
        if self._web_turn_active:
            self._web_turn_active = False
            self._image_turn_active = False
            if self._worker is not None:
                self._worker.cancel()
            self.web_activity_label.setText("取消済み（通信の終了待ち）")
            self.web_cancel_button.hide()
        self._stream_generation += 1
        invalidate = getattr(self._backend, "_invalidate_active_stream", None)
        if callable(invalidate):
            invalidate()
        if self._messages and self._messages[-1].role == "assistant":
            self._messages.pop()
            self._render_messages()
        self._clear_turn_memory_feedback()

    def _worker_finished(self) -> None:
        # finished直後もQObjectの破棄が続くため、完了前にPython参照を落とさない。
        self._thread.wait()
        self._worker.deleteLater()
        self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self.web_activity_label.hide()
        self.web_cancel_button.hide()
        if (
            self._pending_voice_text is None
            and self._voice_thread is None
            and not self._voice_pose_timeout.isActive()
            and not self._stream_voice_preparing
        ):
            self._set_busy(False)
        pending_auto_send = self._pending_auto_send_text
        self._pending_auto_send_text = None
        if pending_auto_send is not None:
            self.input_edit.setPlainText(pending_auto_send)
            self._send_message()

    def _start_new_conversation(self) -> None:
        if self._busy or self._proactive_thread is not None:
            return
        try:
            conversation_id = self._backend.create_conversation("新しい会話")
        except Exception as exc:
            self._show_error(str(exc) or "新しい会話を作成できませんでした。")
            return
        self._current_conversation_id = conversation_id
        self._messages = []
        self._web_turn_active = False
        self._set_emotion_state(None)
        self._set_avatar_action()
        self._clear_previous_conversation_feedback()
        self._clear_error()
        self._reload_conversations(conversation_id)
        self.input_edit.setFocus()
        self._update_action_buttons()

    def _delete_current_conversation(self) -> None:
        if (
            self._busy
            or self._proactive_thread is not None
            or self._current_conversation_id is None
        ):
            return
        answer = QMessageBox.question(
            self,
            "会話履歴の削除",
            "表示中の会話履歴を削除します。元には戻せません。よろしいですか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._backend.delete_conversation(self._current_conversation_id)
        except Exception as exc:
            self._show_error(str(exc) or "会話履歴を削除できませんでした。")
            return
        self._current_conversation_id = None
        self._messages = []
        self._clear_attachment()
        self._reload_conversations()
        self._clear_error()

    def _open_persona_settings(self) -> None:
        if (
            self._busy
            or self._proactive_thread is not None
            or self._backend.is_listening()
            or self._ptt_phase == "認識中"
        ):
            return
        self.options_dialog.hide()
        timer_active = self._proactive_timer.isActive()
        self._proactive_timer.stop()
        self._interrupt_voice_output()
        try:
            dialog = PersonaDialog(
                self._backend.persona_settings(), self._backend.change_persona_settings, self
            )
            dialog.exec()
            dialog.deleteLater()
        finally:
            self._proactive.note_user_activity(datetime.now())
            if timer_active:
                self._proactive_timer.start()
            self.options_dialog.show()

    def _open_environment_settings(self) -> None:
        if self._busy or self._proactive_thread is not None:
            return
        self.options_dialog.hide()
        timer_active = self._proactive_timer.isActive()
        self._proactive_timer.stop()
        try:
            root, source = self._backend.environment_settings_context()
            source = f"{self._backend.current_provider_name()} / {source}"
            dialog = EnvironmentDialog(
                root,
                self,
                running_source=source,
                app_config_path=self._backend.environment_app_config_path(),
            )
            if dialog.exec() == EnvironmentDialog.DialogCode.Accepted:
                QMessageBox.information(
                    self,
                    "環境設定を保存しました",
                    "アプリを終了し、もう一度起動してください。再起動までは現在の接続設定を使います。",
                )
            dialog.deleteLater()
        except (ConfigError, EnvironmentSettingsError) as exc:
            self._show_error(str(exc))
        finally:
            self._proactive.note_user_activity(datetime.now())
            if timer_active:
                self._proactive_timer.start()
            self.options_dialog.show()

    def _open_settings(self) -> None:
        if self._busy or self._proactive_thread is not None:
            return
        self.options_dialog.hide()
        dialog = SettingsDialog(
            self._backend.speech_settings(),
            self,
        )
        accepted = dialog.exec() == SettingsDialog.DialogCode.Accepted
        dialog.deleteLater()
        self.options_dialog.show()
        if not accepted:
            return
        try:
            current_audio = self._backend.speech_settings()
            selected_engine = dialog.selected_speech_engine()
            selected_microphone = dialog.selected_microphone()
            selected_turn_settings = dialog.selected_speech_turn_settings()
            if (
                selected_engine != current_audio.engine
                or selected_microphone != current_audio.input_device
                or selected_turn_settings
                != (
                    current_audio.silence_threshold,
                    current_audio.silence_duration,
                    current_audio.max_recording_duration,
                    current_audio.partial_interval,
                )
            ):
                self._backend.change_speech_settings(
                    selected_engine,
                    selected_microphone,
                    *selected_turn_settings,
                )
        except Exception as exc:
            self._show_error(str(exc) or "マイクの設定を変更できませんでした。")
            return
        self._clear_error()
        self._update_usage_label()
        self._update_status_labels()
        self._set_busy(False)

    def _open_proactive_settings(self) -> None:
        if self._busy:
            return
        self.options_dialog.hide()
        dialog = ProactiveSettingsDialog(self._proactive_settings(), self)
        accepted = dialog.exec() == ProactiveSettingsDialog.DialogCode.Accepted
        dialog.deleteLater()
        self.options_dialog.show()
        if not accepted:
            return
        self._apply_proactive_settings(dialog.selected_settings())

    def _apply_proactive_settings(self, settings: ProactiveSettings) -> None:
        """能動会話の設定を保存し、有効化直後は新しい待機を始める。"""

        was_enabled = self._proactive_settings().enabled
        self._settings.setValue("proactive/enabled", settings.enabled)
        self._settings.setValue("proactive/read_aloud", settings.read_aloud)
        self._settings.setValue("proactive/quiet_start_hour", settings.quiet_start_hour)
        self._settings.setValue("proactive/quiet_end_hour", settings.quiet_end_hour)
        self._settings.setValue("proactive/minimum_idle_minutes", settings.minimum_idle_minutes)
        self._settings.setValue(
            "proactive/minimum_interval_minutes", settings.minimum_interval_minutes
        )
        if not settings.enabled:
            self._cancel_pending_proactive_turn()
            self._cancel_proactive_voice()
            self._proactive.disable()
            self._save_proactive_state()
            self.proactive_status_label.setText("能動会話: 停止中")
            return
        if not was_enabled:
            self._proactive.note_user_activity(datetime.now())
        self._refresh_proactive_status()

    def _open_memory_manager(self) -> None:
        if self._busy:
            return
        restore_options = self.options_dialog.isVisible()
        self.notice.dismiss()
        self.options_dialog.hide()
        dialog = MemoryDialog(self._backend, self)
        dialog.exec()
        dialog.deleteLater()
        self._update_status_labels()
        if restore_options:
            self.options_dialog.show()

    def _open_motion_manager(self) -> None:
        """会話出力を止め、本番状態と分離したVMD管理画面を開く。"""

        if self._busy or self._backend.is_listening():
            return
        self._invalidate_utterance(interrupted=True)
        dialog = MotionLibraryDialog(
            self._backend,
            self.pmx_preview_widget.start_motion_preview,
            self.pmx_preview_widget.stop_motion_preview,
            self,
        )
        try:
            with self.pmx_preview_widget._temporary_preview_camera("full"):
                self._exec_avatar_manager(dialog)
        finally:
            self.pmx_preview_widget.stop_motion_preview()
        if dialog.catalog_changed:
            self._update_status_labels()

    def _open_expression_manager(self) -> None:
        """会話を止め、PMXに登録済みのモーフだけを試す。"""

        if self._busy or self._backend.is_listening() or self._current_avatar_preview is None:
            return
        self._invalidate_utterance(interrupted=True)
        dialog = ExpressionLibraryDialog(
            self._current_avatar_preview,
            self._start_expression_morph_preview,
            self._stop_expression_preview,
            self,
            save_preset=self._save_expression_preset,
            start_combination_preview=self._start_expression_combination_preview,
            load_presets=self._list_expression_presets,
            delete_preset=self._delete_expression_preset,
        )
        self._exec_avatar_manager(dialog)
        self.pmx_preview_widget.set_mmd_command(None)
        self._current_mmd_command = None
        self._update_avatar_header()

    def _exec_avatar_manager(self, dialog: QDialog) -> None:
        """現在の雨衣を管理画面内にも表示する。素材と描画器は共有する。"""
        self.options_dialog.hide()
        preview = _AvatarPreview(self.pmx_preview_widget, dialog)
        dialog._body_layout.insertWidget(0, preview, 2)
        dialog._body_layout.setStretch(1, 3)
        _fit_dialog(dialog, 940, 660)
        dialog.exec()
        preview._timer.stop()
        dialog.deleteLater()

    def _start_expression_morph_preview(self, morph_name: str) -> None:
        preview = self._current_avatar_preview
        mouth_shape = mouth_shape_for_morph(preview, morph_name) if preview is not None else ""
        command = UiMmdAvatarCommand(
            emotion="preview",
            emotion_intensity=1.0,
            emotion_morph="" if mouth_shape else morph_name,
            gaze="forward",
            gaze_bone="",
            gesture="none",
            gesture_target="",
            expression_preset="モーフカタログ",
            expression_morphs=() if mouth_shape else ((morph_name, 1.0),),
            expression_status_text="モーフカタログ確認（会話へは保存しません）",
        )
        self.pmx_preview_widget.start_expression_preview(command)
        if mouth_shape:
            self.pmx_preview_widget.set_lip_sync(mouth_shape)
        self._expression_preview_active = True
        self._current_mmd_command = command
        self._update_avatar_header()

    def _start_expression_combination_preview(self, morph_weights: dict[str, float]) -> None:
        """保存前の強さを含め、カタログ組合せとして表示する。"""

        if not morph_weights:
            return
        command = UiMmdAvatarCommand(
            emotion="preview",
            emotion_intensity=1.0,
            emotion_morph="",
            gaze="forward",
            gaze_bone="",
            gesture="none",
            gesture_target="",
            expression_preset="モーフ組合せ（未保存）",
            expression_morphs=tuple(morph_weights.items()),
            expression_status_text="モーフ組合せ確認（会話へは保存しません）",
        )
        self.pmx_preview_widget.start_expression_preview(command)
        self._expression_preview_active = True
        self._current_mmd_command = command
        self._update_avatar_header()

    def _save_expression_preset(
        self,
        preset_name: str,
        emotion: str,
        morph_weights: dict[str, float],
    ) -> None:
        """カタログの組合せを保存し、再構築したPMXへ表示を戻す。"""

        self._stop_expression_preview()
        save = getattr(self._backend, "save_expression_preset", None)
        if not callable(save):
            raise RuntimeError("このアバターでは表情を保存できません。")
        save(preset_name, emotion, morph_weights)
        self._update_status_labels()

    def _list_expression_presets(self) -> list[UiExpressionPreset]:
        """保存済み表情がある実装だけ、その一覧を画面へ渡す。"""

        load = getattr(self._backend, "list_expression_presets", None)
        return load() if callable(load) else []

    def _delete_expression_preset(self, preset_name: str) -> None:
        """選択した表情を削除し、再構築後のPMXへ表示を戻す。"""

        self._stop_expression_preview()
        delete = getattr(self._backend, "delete_expression_preset", None)
        if not callable(delete):
            raise RuntimeError("このアバターでは表情を削除できません。")
        delete(preset_name)
        self._update_status_labels()

    def _show_saved_memories(self, memories: list[UiMemory]) -> None:
        """会話本文や読み上げとは別に、保存結果を短く知らせる。"""

        approved = sum(memory.status == "approved" for memory in memories)
        candidates = sum(memory.status == "candidate" for memory in memories)
        parts = []
        if approved:
            parts.append(f"記憶を{approved}件保存しました。")
        if candidates:
            parts.append(f"記憶の候補が{candidates}件あります（確認待ち）。")
        if parts:
            # 個人情報の本文を、画面上の通知へ転載しない。
            self.notice.show_notice("\n".join(parts), kind="memory")
            self.notice.action_button.setEnabled(not self._busy)

    def _clear_turn_memory_feedback(self) -> None:
        """古い世代の記憶表示を、次の会話ターンへ持ち込まない。"""

        self._set_used_memories([])
        self.notice.dismiss()
        memory = self._backend.memory_settings()
        self.memory_status_label.setText(
            "保存した記憶: 使用中" if memory.enabled else "保存した記憶: 停止中"
        )

    def _clear_previous_conversation_feedback(self) -> None:
        """会話切替後に前会話の記憶やMMD指示を残さない。"""

        self._clear_attachment()
        self._web_turn_active = False
        self._image_turn_active = False
        self._clear_turn_memory_feedback()
        self._current_mmd_command = None
        self.pmx_preview_widget.interrupt_motion()
        self.pmx_preview_widget.set_mmd_command(None)
        self._update_avatar_header()

    def _set_used_memories(self, memories: list[UiMemory]) -> None:
        self._used_memories = list(memories)
        self.memory_details_button.setVisible(bool(memories))

    def _open_used_memory_details(self) -> None:
        if not getattr(self, "_used_memories", []):
            return
        details = "\n\n".join(
            f"{memory.content}\n根拠の会話: {memory.source_conversation_id or '記録なし'}\n"
            f"作成日時: {memory.created_at or '記録なし'}"
            for memory in self._used_memories
        )
        QMessageBox.information(self, "この応答で使用した記憶", details)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.busy_label.setText("処理中…" if busy else "")
        self.input_edit.setReadOnly(busy and not self._web_turn_active)
        self.input_edit.setAcceptDrops(not busy)
        self.new_button.setEnabled(not busy)
        self.attach_image_button.setEnabled(not busy)
        self.settings_button.setEnabled(not busy)
        self.memory_button.setEnabled(not busy)
        self.notice.action_button.setEnabled(not busy)
        self.proactive_button.setEnabled(not busy)
        self.motion_library_button.setEnabled(not busy and not self._backend.is_listening())
        self.expression_library_button.setEnabled(not busy and not self._backend.is_listening())
        self.conversation_list.setEnabled(not busy)
        self.microphone_button.setEnabled(self._backend.is_speech_recognition_available())
        self._update_send_button()
        self._update_action_buttons()
        self._update_conversation_state()

    def _update_send_button(self) -> None:
        self.send_button.setEnabled(
            not self._busy
            and not self._image_loading
            and bool(self.input_edit.toPlainText().strip())
        )

    def _update_action_buttons(self) -> None:
        self.delete_button.setEnabled(not self._busy and self._current_conversation_id is not None)

    def _show_error(self, message: str) -> None:
        self.error_label.setText(f"エラー: {message}")
        self.error_label.setVisible(True)
        self.dismiss_error_button.show()

    def _clear_error(self) -> None:
        self.error_label.clear()
        self.error_label.setVisible(False)
        self.dismiss_error_button.hide()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._image_fetches:
            self._clear_attachment()
            self._close_after_image_fetch = True
            event.ignore()
            return
        if self._busy:
            self._cancel_web_turn()
            QMessageBox.information(
                self,
                "処理中です",
                "LLMからの回答が終わってから画面を閉じてください。",
            )
            event.ignore()
            return
        if self._proactive_thread is not None:
            self._cancel_pending_proactive_turn()
            QMessageBox.information(
                self,
                "能動発話を停止しました",
                "返却待ちの能動発話を取り消しました。処理が終わってから、もう一度閉じてください。",
            )
            event.ignore()
            return
        if self._voice_thread is not None:
            self._close_after_voice = True
            self._invalidate_utterance(interrupted=True)
            event.ignore()
            return
        self._backend.stop_speech_recognition()
        self.notice.dismiss()
        self.options_dialog.close()
        self._clear_attachment()
        QApplication.instance().removeEventFilter(self)
        self._voice_state_timer.stop()
        self._proactive_timer.stop()
        self._voice_pose_timeout.stop()
        self._backend.stop_voice_output()
        self.pmx_preview_widget.close()
        self._save_window_preferences()
        event.accept()

    def _restore_window_preferences(self) -> None:
        """表示・音声操作・送信許可を復元し、録音や送信は開始しない。"""

        geometry = self._settings.value("window_geometry")
        if isinstance(geometry, (QByteArray, bytes)):
            self.restoreGeometry(geometry)
        mode = str(self._settings.value("companion_view", "chat"))
        # 新しい幅の記録がない場合だけ、以前の表示幅を引き継ぐ。
        self.content_splitter.setSizes([600, 400] if mode in {"face", "avatar"} else [470, 530])
        for splitter, key in (
            (self.main_splitter, "layout/main_splitter"),
            (self.content_splitter, "layout/content_splitter"),
        ):
            state = self._settings.value(key)
            if isinstance(state, (QByteArray, bytes)):
                splitter.restoreState(state)
        self.status_panel.show()
        self.always_on_top_checkbox.setChecked(
            self._settings.value("always_on_top", False, type=bool)
        )
        camera_mode = str(self._settings.value("camera_view", "bust"))
        self._camera_view = camera_mode if camera_mode in {"full", "bust"} else "bust"
        self.pmx_preview_widget.set_view_mode(self._camera_view)
        self._update_camera_view_icon()
        self.rotation_360_checkbox.setChecked(
            self._settings.value("camera/rotation_360", False, type=bool)
        )
        for checkbox, key in (
            (self.auto_send_recognition_checkbox, "voice/auto_send_recognition"),
            (self.continuous_voice_mode_checkbox, "voice/continuous"),
            (self.hold_to_talk_checkbox, "voice/hold_to_talk"),
        ):
            checkbox.setChecked(self._settings.value(key, False, type=bool))
        # 未対応の接続先でも、利用者が選んだ許可設定は保持する。
        for key, method in (
            ("web_search/enabled", "set_web_search_enabled"),
            ("image_sharing/enabled", "set_image_sharing_enabled"),
        ):
            setter = getattr(self._backend, method, None)
            if callable(setter) and self._settings.contains(key):
                setter(self._settings.value(key, True, type=bool))
        self._refresh_web_permission()
        for kind, (enabled, seconds) in self._notice_controls.items():
            key = f"notifications/{kind}"
            default = NOTICE_DEFAULT_SECONDS[kind]
            try:
                duration = float(self._settings.value(f"{key}/seconds", default))
            except (TypeError, ValueError, OverflowError):
                duration = default
            if not math.isfinite(duration):
                duration = default
            enabled.blockSignals(True)
            seconds.blockSignals(True)
            enabled.setChecked(self._settings.value(f"{key}/enabled", True, type=bool))
            seconds.setValue(duration)
            enabled.blockSignals(False)
            seconds.blockSignals(False)
            self._update_notice_preferences(kind, save=False)

    def _update_notice_preferences(self, kind: str, *, save: bool = True) -> None:
        enabled, seconds = self._notice_controls[kind]
        seconds.setEnabled(enabled.isChecked())
        self.notice.configure(kind, enabled=enabled.isChecked(), seconds=seconds.value())
        if save:
            self._settings.setValue(f"notifications/{kind}/enabled", enabled.isChecked())
            self._settings.setValue(f"notifications/{kind}/seconds", seconds.value())
            self._settings.sync()

    def _save_preference(self, key: str, value: bool | str) -> None:
        self._settings.setValue(key, value)
        self._settings.sync()

    def _save_window_preferences(self) -> None:
        self._settings.setValue("window_geometry", self.saveGeometry())
        self._save_splitter_preferences()
        self._settings.setValue("always_on_top", self.always_on_top_checkbox.isChecked())
        self._settings.sync()


def _settings_section(parent: QVBoxLayout, title: str, description: str) -> QVBoxLayout:
    group = QGroupBox(title)
    layout = QVBoxLayout(group)
    layout.setContentsMargins(14, 18, 14, 14)
    layout.setSpacing(10)
    note = QLabel(description)
    note.setWordWrap(True)
    layout.addWidget(note)
    parent.addWidget(group)
    return layout


def _format_usage_summary(summary: UiTokenUsageSummary) -> str:
    """取得済みの応答だけを、読みやすい日本語へ整える。"""

    if summary.response_count == 0:
        return "未取得"
    return (
        f"計 {summary.total_tokens:,}（入 {summary.input_tokens:,} / 出 {summary.output_tokens:,}）"
    )


def _avatar_text(value: str, fallback: str) -> str:
    """画面を極端に広げないよう、自由文の状態名を短くする。"""

    if not value or value.lower() in {"none", "forward"}:
        return fallback
    return value.replace("\n", " ")[:30]


def _mmd_motion_display_name(command: UiMmdAvatarCommand) -> str:
    """管理画面と同じ配布モーション名を表示する。"""

    if command.gesture_motion is not None and command.gesture_motion.name:
        return command.gesture_motion.name
    return command.gesture_motion_id or command.gesture_target


def _format_mmd_command(command: UiMmdAvatarCommand | None) -> str:
    if command is None:
        return "MMD命令: 未使用"
    intensity = round(command.emotion_intensity * 100)
    expression_text = command.emotion_morph or "未設定"
    if command.expression_morphs:
        weights = "、".join(
            f"{name} {round(weight * 100)}%" for name, weight in command.expression_morphs
        )
        expression_text = f"{command.expression_preset or '名称なし'} / {weights}"
    motion_name = _mmd_motion_display_name(command) or "未設定"
    if not command.expression_status_text:
        return (
            f"MMD表情: {expression_text}"
            f"{'（' + str(intensity) + '%）' if not command.expression_morphs else ''}\n"
            f"MMD視線: {command.gaze_bone or '未設定'}\n"
            f"MMD姿勢: {command.head_bone or '未設定'}\n"
            f"MMD動作: {motion_name}"
        )
    return (
        f"MMD表情: {expression_text}"
        f"{'（' + str(intensity) + '%）' if not command.expression_morphs else ''}\n"
        f"MMD視線: {command.gaze_bone or '未設定'}"
        f"（{round(command.gaze_intensity * 100)}%）\n"
        f"MMD姿勢: {command.head_bone or '未設定'}\n"
        f"MMD動作: {motion_name}（{round(command.gesture_intensity * 100)}%）\n"
        f"表現統合: {command.expression_status_text}"
    )


def _format_mouth_morphs(preview: UiAvatarModelPreview | None) -> str:
    if preview is None:
        return "PMX口モーフ: 未使用"
    if preview.mouth_morphs:
        shapes = "、".join(_mouth_label(shape) for shape in preview.mouth_morphs)
        return f"PMX口モーフ: 設定済み（{shapes}）"
    if preview.vertex_morph_names:
        names = "、".join(preview.vertex_morph_names[:5])
        suffix = "…" if len(preview.vertex_morph_names) > 5 else ""
        return f"PMX口モーフ候補: {names}{suffix}"
    return "PMX口モーフ: 頂点モーフなし"


def _format_emotion_morph(
    preview: UiAvatarModelPreview | None,
    command: UiMmdAvatarCommand | None,
) -> str:
    if preview is None:
        return "PMX表情モーフ: 未使用"
    available_morphs = set(preview.morph_names) | set(preview.named_vertex_morphs)
    if command is not None and command.expression_morphs:
        found = [
            (name, weight) for name, weight in command.expression_morphs if name in available_morphs
        ]
        missing = [
            name for name, _weight in command.expression_morphs if name not in available_morphs
        ]
        lines = [f"表情プリセット: {command.expression_preset or '名称なし'}"]
        if found:
            details = "、".join(f"{name}（{round(weight * 100)}%）" for name, weight in found)
            lines.append(f"PMX表情モーフ: 適用中 {details}")
        if missing:
            lines.append(f"未検出: {'、'.join(missing)}")
        return "\n".join(lines)
    if command is None or not command.emotion_morph:
        return "PMX表情モーフ: 未設定"
    if command.emotion in preview.emotion_morphs:
        intensity = round(command.emotion_intensity * 100)
        return f"PMX表情モーフ: 適用中 {command.emotion_morph}（{intensity}%）"
    return f"PMX表情モーフ: 設定名「{command.emotion_morph}」が見つかりません"


def _format_avatar_expression_summary(
    preview: UiAvatarModelPreview,
    command: UiMmdAvatarCommand | None,
) -> str:
    """アバターヘッダーへ収まる表情状態を作る。"""

    if command is None:
        return "表情: 未設定"
    available_morphs = set(preview.morph_names) | set(preview.named_vertex_morphs)
    if command.expression_morphs:
        applied = [
            f"{name} {round(weight * 100)}%"
            for name, weight in command.expression_morphs
            if name in available_morphs
        ]
        preset = command.expression_preset or "名称なし"
        details = "・".join(applied) if applied else "対応モーフなし"
        return f"表情: {preset}（{details}）"
    if command.emotion_morph:
        return f"表情: {command.emotion_morph} {round(command.emotion_intensity * 100)}%"
    return "表情: 通常"


def _format_pose_bones(
    preview: UiAvatarModelPreview | None,
    command: UiMmdAvatarCommand | None,
) -> str:
    if preview is None:
        return "PMXボーン: 未使用"
    if command is None:
        return "PMXボーン: 命令なし"
    names = {bone.name for bone in preview.bones}
    configured = [
        ("視線", command.gaze_bone),
        ("姿勢", command.head_bone),
        ("上半身", command.upper_body_bone),
    ]
    active = [f"{label} {name}" for label, name in configured if name and name in names]
    missing = [name for _label, name in configured if name and name not in names]
    if missing:
        return f"PMXボーン: 設定名「{'、'.join(missing)}」が見つかりません"
    if active:
        return f"PMXボーン: 適用中（{' / '.join(active)}）"
    return "PMXボーン: 未設定"


def _format_texture_status(preview: UiAvatarModelPreview | None) -> str:
    if preview is None:
        return "PMXテクスチャ: 未使用"
    if preview.textured_triangles:
        return f"PMXテクスチャ: 適用候補 {len(preview.textured_triangles):,}面"
    if preview.triangles:
        return "PMXテクスチャ: なし（単色メッシュ表示）"
    return "PMXテクスチャ: 大きなモデルのため簡易表示"


def _mouth_label(shape: str) -> str:
    return {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お"}.get(shape, shape)


def run_window(backend: ChatBackend) -> int:
    app = QApplication.instance() or QApplication([])
    window = ChatWindow(backend)
    window.show()
    return app.exec()
