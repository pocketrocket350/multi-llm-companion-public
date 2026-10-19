"""初心者向けの接続情報入力画面。"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event, Thread
from uuid import uuid4

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from multi_llm_companion.config import (
    ProviderConfig,
    load_environment_editor_config,
    resolve_provider,
    save_environment_config,
)
from multi_llm_companion.environment import (
    ConnectionValues,
    EnvironmentSettings,
    EnvironmentSettingsError,
    VoiceValues,
    environment_store,
    validate_connection_url,
    validate_voice,
)
from multi_llm_companion.services.connection_test import ConnectionTestResult, probe_llm_connection
from multi_llm_companion.services.voice_connection_test import (
    VoiceListResult,
    fetch_voisona_voices,
    probe_voisona_connection,
)
from multi_llm_companion.ui._dialog_layout import _fit_dialog


def _run_connection_test(config, generation, results) -> None:
    try:
        result = probe_llm_connection(config)
    except Exception:
        result = ConnectionTestResult(
            False, "接続テストを完了できませんでした。入力内容と通信環境を確認してください。"
        )
    results.put((generation, result))


def _run_voice_test(values, options, cancel, generation, results) -> None:
    try:
        result = probe_voisona_connection(
            values,
            **options,
            cancel=cancel,
            report=lambda message: results.put((generation, message)),
        )
    except Exception:
        result = ConnectionTestResult(
            False,
            "VoiSona Talkの試聴を完了できませんでした。接続設定と音声出力を確認してください。",
        )
    results.put((generation, result))


def _run_voice_list(values, timeout, cancel, generation, results) -> None:
    try:
        result = fetch_voisona_voices(values, timeout=timeout, cancel=cancel)
    except Exception:
        result = VoiceListResult(
            False, "ボイス一覧を取得できませんでした。接続設定を確認してください。"
        )
    results.put((generation, result))


class EnvironmentDialog(QDialog):
    def __init__(
        self,
        root: Path,
        parent: QWidget | None = None,
        *,
        running_source: str | None = None,
        app_config_path: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._root = root
        self._app_config_path = app_config_path
        self.setWindowTitle("環境設定")
        self._providers, self._audio, self._active = load_environment_editor_config(
            root, app_config_path
        )
        self._original_providers = self._providers.model_copy(deep=True)
        self._fallback_provider = self._active
        self._store = environment_store(root)
        self._stored_settings: EnvironmentSettings | None = None
        self._deletion_pending = False
        load_error = ""
        try:
            self._settings = self._store.load()
            self._stored_settings = self._settings
            if not self._store.path.exists():
                self._settings = replace(
                    self._settings,
                    voice=VoiceValues(
                        enabled=self._audio.voice_output.enabled,
                        base_url=self._audio.voice_output.base_url,
                    ),
                )
        except EnvironmentSettingsError as exc:
            self._settings = EnvironmentSettings()
            load_error = str(exc)
        self._drafts = dict(self._settings.providers)
        self._saved_values = dict(self._settings.providers)
        self._editing = ""
        self._test_generation = 0
        self._test_running = False
        self._test_results = SimpleQueue()
        self._test_timer = QTimer(self)
        self._test_timer.setInterval(50)
        self._test_timer.timeout.connect(self._poll_connection_test)
        self._voice_test_generation = 0
        self._voice_test_running = False
        self._voice_test_cancel = Event()
        self._voice_test_results = SimpleQueue()
        self._voice_test_timer = QTimer(self)
        self._voice_test_timer.setInterval(50)
        self._voice_test_timer.timeout.connect(self._poll_voice_test)
        self._voice_list_generation = 0
        self._voice_list_running = False
        self._voice_list_cancel = Event()
        self._voice_list_results = SimpleQueue()
        self._voice_list_timer = QTimer(self)
        self._voice_list_timer.setInterval(50)
        self._voice_list_timer.timeout.connect(self._poll_voice_list)
        self._voice_picker: QInputDialog | None = None

        layout = QVBoxLayout(self)
        source = QLabel(
            f"現在使用中: {running_source}。保存した変更は再起動後に使います。"
            if running_source
            else "接続情報がなくても、この画面で準備できます。"
        )
        source.setWordWrap(True)
        layout.addWidget(source)
        self.environment_check = QCheckBox("環境変数で設定")
        self.environment_check.setChecked(self._settings.use_environment)
        layout.addWidget(self.environment_check)
        self.mode_label = QLabel()
        self.mode_label.setWordWrap(True)
        layout.addWidget(self.mode_label)

        provider_form = QFormLayout()
        provider_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(list(self._providers.providers))
        self.provider_combo.setCurrentText(self._active)
        provider_form.addRow("使用するLLM接続先（再起動後）", self.provider_combo)
        layout.addLayout(provider_form)
        provider_actions = QHBoxLayout()
        self.add_provider_button = QPushButton("接続先を追加")
        self.rename_provider_button = QPushButton("名前を変更")
        self.delete_provider_button = QPushButton("接続先を削除")
        self.test_connection_button = QPushButton("接続テスト")
        self.add_provider_button.clicked.connect(self._add_provider)
        self.rename_provider_button.clicked.connect(self._rename_provider)
        self.delete_provider_button.clicked.connect(self._delete_provider)
        self.test_connection_button.clicked.connect(self._test_connection)
        provider_actions.addWidget(self.add_provider_button)
        provider_actions.addWidget(self.rename_provider_button)
        provider_actions.addWidget(self.delete_provider_button)
        provider_actions.addWidget(self.test_connection_button)
        provider_actions.addStretch()
        layout.addLayout(provider_actions)
        provider_note = QLabel(
            "選んだ接続先の情報を下で入力します。接続先の変更も保存後の再起動で反映されます。\n"
            "画面入力方式では、OpenAI互換APIの接続先を追加できます。名前は自分で決める呼び名です。\n"
            "openai・qwen・deepseek は設定例、mock は外部通信しない動作確認用です。"
        )
        provider_note.setWordWrap(True)
        layout.addWidget(provider_note)
        self.test_status = QLabel()
        self.test_status.setWordWrap(True)
        self.test_status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.test_status)
        self.env_names_label = QLabel()
        self.env_names_label.setWordWrap(True)
        self.env_names_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        page = QWidget()
        scroll.setWidget(page)
        layout.addWidget(scroll, 1)
        content = QVBoxLayout(page)
        content.addWidget(self.env_names_label)
        test_note = QLabel(
            "「接続テスト」は短い固定文を1回送信します。料金が発生する場合があります。"
            "会話履歴・人格・音声情報は送らず、設定も保存しません。"
            "画面を閉じても送信済みの要求は取り消せない場合があります。"
        )
        test_note.setWordWrap(True)
        content.addWidget(test_note)
        self.manual_page = QWidget()
        content.addWidget(self.manual_page)
        fields = QVBoxLayout(self.manual_page)
        fields.setContentsMargins(0, 0, 0, 0)
        llm = QGroupBox("LLMの接続情報")
        form = QFormLayout(llm)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("サービスの案内にあるAPIのURL（https://…/v1 など）")
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("サービスで発行した、AIへ接続するための秘密の文字列")
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("サービスが指定するモデル名を正確に入力")
        form.addRow("接続先URL", self.url_edit)
        form.addRow("APIキー", self.key_edit)
        form.addRow("モデル名", self.model_edit)
        fields.addWidget(llm)

        voice = QGroupBox("VoiSona Talk（読み上げ音声）")
        voice_form = QFormLayout(voice)
        voice_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.voice_check = QCheckBox("VoiSona Talkで読み上げる")
        self.voice_check.setChecked(self._settings.voice.enabled)
        voice_form.addRow(self.voice_check)
        self.voice_url_edit = QLineEdit(self._settings.voice.base_url)
        self.username_edit = QLineEdit(self._settings.voice.username)
        self.username_edit.setPlaceholderText("VoiSona TalkのREST API設定で決めたユーザー名")
        self.password_edit = QLineEdit(self._settings.voice.password)
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText("VoiSona TalkのREST API設定で決めたパスワード")
        self.voice_name_edit = QLineEdit(self._settings.voice.voice_name)
        self.voice_name_edit.setPlaceholderText(
            "「ボイス一覧から選ぶ」で選択できます。従来の手入力も可能"
        )
        self.voice_url_edit.setToolTip(
            "通常は初期値のまま使います。VoiSona Talk側と合わせてください。"
        )
        for label, edit in (
            ("接続先URL", self.voice_url_edit),
            ("ユーザー名", self.username_edit),
            ("パスワード", self.password_edit),
            ("ボイス名", self.voice_name_edit),
        ):
            voice_form.addRow(label, edit)
        voice_note = QLabel(
            "VoiSona Talk本体とボイスの導入、REST APIの有効化が必要です。"
            "購入用アカウントのパスワードは入力しません。\n"
            "チェックを外した場合、音声の項目は空欄でも保存できます。"
        )
        voice_note.setWordWrap(True)
        voice_form.addRow(voice_note)
        fields.addWidget(voice)
        voice_actions = QHBoxLayout()
        self.voice_list_button = QPushButton("ボイス一覧から選ぶ")
        self.voice_list_button.setToolTip(
            "画面入力方式のURL・ユーザー名・パスワードで一覧を取得します。合成・再生・保存はしません。"
        )
        self.voice_list_button.clicked.connect(self._fetch_voice_list)
        self.voice_test_button = QPushButton("VoiSona Talkの接続テスト・試聴")
        self.voice_stop_button = QPushButton("試聴を停止")
        self.voice_stop_button.setEnabled(False)
        self.voice_test_button.clicked.connect(self._test_voice)
        self.voice_stop_button.clicked.connect(self._stop_voice_test)
        voice_actions.addWidget(self.voice_list_button)
        voice_actions.addWidget(self.voice_test_button)
        voice_actions.addWidget(self.voice_stop_button)
        voice_actions.addStretch()
        content.addLayout(voice_actions)
        self.voice_list_status = QLabel()
        self.voice_list_status.setWordWrap(True)
        self.voice_list_status.setTextFormat(Qt.TextFormat.PlainText)
        content.addWidget(self.voice_list_status)
        voice_test_note = QLabel(
            "選択した方式の接続情報でボイス名を確認し、「こんにちは。音声のテストです。」を1回合成して"
            "このPCの既定の音声出力で再生します。VoiSona Talkは同じPCで起動してください。"
            "会話や設定は保存しません。停止・画面終了後も送信済みの合成は取り消せない場合があります。"
        )
        voice_test_note.setWordWrap(True)
        content.addWidget(voice_test_note)
        self.voice_test_status = QLabel()
        self.voice_test_status.setWordWrap(True)
        self.voice_test_status.setTextFormat(Qt.TextFormat.PlainText)
        content.addWidget(self.voice_test_status)
        content.addStretch()
        self._connection_scroll = scroll

        self.error_label = QLabel(load_error)
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        self.deletion_note = QLabel(
            "接続先と保存済みAPIキーの削除を予約しています。"
            "「保存」で確定し、「キャンセル」で取り消します。接続の切替は再起動後です。"
        )
        self.deletion_note.setWordWrap(True)
        self.deletion_note.hide()
        layout.addWidget(self.deletion_note)
        notice = QLabel(
            "保存しても接続テストや送信は行いません。アプリを終了して開き直すと反映されます。\n"
            "入力内容はWindowsで暗号化して個人用領域へ保存します。"
            "アプリのバックアップには含まれません。別PCや別フォルダーでは再入力してください。\n"
            "環境変数や .env は書き換えません。キャンセルでは変更を保存しません。"
        )
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存（再起動後に反映）")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.environment_check.toggled.connect(self._update_mode)
        self.voice_check.toggled.connect(self._update_voice)
        self.provider_combo.currentTextChanged.connect(self._select_provider)
        for edit in (self.url_edit, self.key_edit, self.model_edit):
            edit.textChanged.connect(self._invalidate_connection_test)
        for edit in (
            self.voice_url_edit,
            self.username_edit,
            self.password_edit,
            self.voice_name_edit,
        ):
            edit.textChanged.connect(self._invalidate_voice_test)
        for edit in (self.voice_url_edit, self.username_edit, self.password_edit):
            edit.textChanged.connect(self._invalidate_voice_list)
        self._select_provider(self.provider_combo.currentText())
        self._update_mode()
        _fit_dialog(self, 650, 760)

    def _remember_provider(self) -> None:
        if self._editing:
            self._drafts[self._editing] = ConnectionValues(
                self.url_edit.text().strip(),
                self.key_edit.text().strip(),
                self.model_edit.text().strip(),
            )

    def _select_provider(self, name: str) -> None:
        self._invalidate_connection_test()
        self._remember_provider()
        self._editing = name
        values = self._drafts.get(name, ConnectionValues())
        self.url_edit.setText(values.base_url)
        self.key_edit.setText(values.api_key)
        self.model_edit.setText(values.model)
        enabled = self._providers.providers[name].provider.lower() != "mock"
        for edit in (self.url_edit, self.key_edit, self.model_edit):
            edit.setEnabled(enabled)
        self._update_provider_actions()

    def _update_provider_actions(self) -> None:
        use_environment = self.environment_check.isChecked()
        configured = self._providers.providers[self.provider_combo.currentText()]
        self.add_provider_button.setEnabled(not use_environment)
        self.test_connection_button.setEnabled(
            not self._test_running and configured.provider.lower() == "openai_compatible"
        )
        self.test_connection_button.setToolTip(
            "選択した方式のURL・モデル名・APIキーで確認します。mockは実接続テストの対象外です。"
        )
        self.rename_provider_button.setEnabled(
            not use_environment and configured.provider.lower() == "openai_compatible"
        )
        is_mock = configured.provider.lower() == "mock"
        can_delete = (
            not is_mock and len(self._providers.providers) > 1 and self._stored_settings is not None
        )
        self.delete_provider_button.setEnabled(can_delete)
        self.delete_provider_button.setToolTip(
            "mockは外部通信しない復旧用です。削除できません。"
            if is_mock
            else "保存済み接続情報を読み出せないため削除できません。復旧して画面を開き直してください。"
            if self._stored_settings is None
            else "最後の接続先は削除できません。先に別の接続先を追加してください。"
            if not can_delete
            else "確認後に削除を予約します。最後の保存でAPIキーも保護ストアから削除します。"
        )
        self.env_names_label.setVisible(use_environment)
        self.env_names_label.setText(
            "この接続先は環境変数不要です。"
            if configured.provider.lower() == "mock"
            else "使用する環境変数名（値は表示しません）:\n"
            f"URL: {configured.base_url_env}\n"
            f"APIキー: {configured.api_key_env}\nモデル: {configured.model_env}"
        )

    def _provider_name(self, *, current: str = "") -> str | None:
        dialog = QInputDialog(self)
        dialog.setWindowTitle("接続先の名前を変更" if current else "OpenAI互換の接続先を追加")
        dialog.setLabelText("一覧に表示する名前（例: 仕事用AI）。APIキーは入力しません。")
        dialog.setTextValue(current)
        dialog.setOkButtonText("決定")
        dialog.setCancelButtonText("キャンセル")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        name = dialog.textValue().strip()
        if not name or len(name) > 80 or any(not char.isprintable() for char in name):
            self.error_label.setText(
                "接続先の名前は1〜80文字で入力してください。改行は使えません。"
            )
            return None
        existing = set(self._providers.providers) | set(self._drafts)
        if any(other != current and other.casefold() == name.casefold() for other in existing):
            self.error_label.setText("同じ名前の接続先があります。別の名前を入力してください。")
            return None
        self.error_label.clear()
        return name

    def _add_provider(self) -> None:
        if self.environment_check.isChecked():
            return
        name = self._provider_name()
        if name is None:
            return
        # 既存の参照名を変えず、新しい接続先には衝突しない参照名を割り当てる。
        prefix = f"COMPANION_{uuid4().hex.upper()}"
        self._providers.providers[name] = ProviderConfig(
            provider="openai_compatible",
            base_url_env=f"{prefix}_BASE_URL",
            api_key_env=f"{prefix}_API_KEY",
            model_env=f"{prefix}_MODEL",
        )
        self.provider_combo.addItem(name)
        self.provider_combo.setCurrentText(name)

    def _rename_provider(self) -> None:
        current = self.provider_combo.currentText()
        if (
            self.environment_check.isChecked()
            or self._providers.providers[current].provider.lower() != "openai_compatible"
        ):
            return
        name = self._provider_name(current=current)
        if name is None or name == current:
            return
        self._remember_provider()
        self._providers.providers = {
            name if key == current else key: value
            for key, value in self._providers.providers.items()
        }
        for values in (self._drafts, self._saved_values):
            if current in values:
                values[name] = values.pop(current)
        if self._fallback_provider == current:
            self._fallback_provider = name
        with QSignalBlocker(self.provider_combo):
            self.provider_combo.setItemText(self.provider_combo.currentIndex(), name)
        self._editing = ""
        self._select_provider(name)

    def _delete_provider(self) -> None:
        current = self.provider_combo.currentText()
        if (
            self._providers.providers[current].provider.lower() == "mock"
            or len(self._providers.providers) <= 1
            or self._stored_settings is None
        ):
            return
        remaining = [name for name in self._providers.providers if name != current]
        fallback = next(
            (
                name
                for name in remaining
                if self._providers.providers[name].provider.lower() == "mock"
            ),
            remaining[0],
        )
        if self._fallback_provider in remaining:
            fallback = self._fallback_provider
        confirmation = QMessageBox(self)
        confirmation.setWindowTitle("接続先の削除を確認")
        confirmation.setIcon(QMessageBox.Icon.Warning)
        confirmation.setTextFormat(Qt.TextFormat.PlainText)
        confirmation.setText(f"「{current}」を削除しますか？")
        confirmation.setInformativeText(
            "最後の「保存」で、接続先と対応する保存済みAPIキーをアプリの保護ストアから削除します。\n"
            f"削除後の接続先は「{fallback}」になります。保存前に一覧で変更できます。\n"
            "起動中の接続設定は再起動まで変わりません。編集画面の「キャンセル」で取り消せます。\n"
            "Windowsの環境変数・.envや、提供元のAPIキー自体は削除・無効化しません。"
        )
        delete = confirmation.addButton("削除を予約", QMessageBox.ButtonRole.DestructiveRole)
        cancel = confirmation.addButton("キャンセル", QMessageBox.ButtonRole.RejectRole)
        confirmation.setDefaultButton(cancel)
        confirmation.setEscapeButton(cancel)
        confirmation.exec()
        if confirmation.clickedButton() is not delete:
            return
        # 選択変更時に、削除した入力値をドラフトへ戻さない。
        self._editing = ""
        self._providers.providers.pop(current)
        self._drafts.pop(current, None)
        self._saved_values.pop(current, None)
        self._fallback_provider = fallback
        self._deletion_pending = True
        with QSignalBlocker(self.provider_combo):
            self.provider_combo.removeItem(self.provider_combo.currentIndex())
            self.provider_combo.setCurrentText(fallback)
        self._select_provider(fallback)
        self.error_label.clear()
        self.deletion_note.show()

    def _update_mode(self) -> None:
        self._invalidate_connection_test()
        self._invalidate_voice_test()
        self._invalidate_voice_list()
        use_environment = self.environment_check.isChecked()
        self.manual_page.setEnabled(not use_environment)
        self.mode_label.setText(
            "保存する方式: 環境変数。従来の設定を使い、この画面の入力値は使いません。"
            if use_environment
            else "保存する方式: 画面入力。ここで保存した値だけを使い、環境変数の接続情報は使いません。"
        )
        self._update_provider_actions()
        self._update_voice()

    def _invalidate_connection_test(self) -> None:
        self._test_generation += 1
        if self.test_status.text():
            self.test_status.setText(
                "入力が変わりました。実行中の結果は使いません。終了後にもう一度テストしてください。"
                if self._test_running
                else "入力が変わりました。もう一度接続テストしてください。"
            )

    def _test_connection(self) -> None:
        from multi_llm_companion.config import ConfigError

        name = self.provider_combo.currentText()
        configured = self._providers.providers[name]
        if self._test_running or configured.provider.lower() != "openai_compatible":
            return
        try:
            values = ConnectionValues(
                self.url_edit.text().strip(),
                self.key_edit.text().strip(),
                self.model_edit.text().strip(),
            )
            selected = resolve_provider(
                name,
                configured,
                EnvironmentSettings(self.environment_check.isChecked(), {name: values}),
            )
            validate_connection_url(selected.base_url, "接続先URL")
        except (ConfigError, EnvironmentSettingsError) as exc:
            self.test_status.setText(str(exc))
            return
        self._test_generation += 1
        self._test_running = True
        self.test_status.setText("接続テスト中です。設定は保存していません。")
        self._update_provider_actions()
        # 通信側はQtオブジェクトを持たず、画面終了時もスレッド破棄で停止しない。
        try:
            Thread(
                target=_run_connection_test,
                args=(selected, self._test_generation, self._test_results),
                daemon=True,
            ).start()
        except RuntimeError:
            self._test_running = False
            self.test_status.setText(
                "接続テストを開始できませんでした。時間を置いて再試行してください。"
            )
            self._update_provider_actions()
            return
        self._test_timer.start()

    def _poll_connection_test(self) -> None:
        try:
            generation, result = self._test_results.get_nowait()
        except Empty:
            return
        self._test_running = False
        self._test_timer.stop()
        if generation == self._test_generation:
            self.test_status.setText(result.message)
        else:
            self.test_status.setText(
                "入力が変わったため結果を使いません。もう一度接続テストしてください。"
            )
        self._update_provider_actions()

    def done(self, result: int) -> None:
        self._test_generation += 1
        self._test_timer.stop()
        self._voice_test_generation += 1
        self._voice_test_cancel.set()
        self._voice_test_timer.stop()
        self._invalidate_voice_list()
        self._voice_list_timer.stop()
        super().done(result)

    def _update_voice_list_button(self) -> None:
        self.voice_list_button.setEnabled(
            not self.environment_check.isChecked()
            and not self._voice_list_running
            and self._voice_picker is None
        )

    def _invalidate_voice_list(self) -> None:
        self._voice_list_generation += 1
        self._voice_list_cancel.set()
        if self._voice_picker is not None:
            self._voice_picker.reject()
        if self.voice_list_status.text():
            self.voice_list_status.setText(
                "設定が変わったため一覧を使いません。画面入力方式で、もう一度取得してください。"
            )
        self._update_voice_list_button()

    def _fetch_voice_list(self) -> None:
        if (
            self.environment_check.isChecked()
            or self._voice_list_running
            or self._voice_picker is not None
        ):
            return
        values = VoiceValues(
            True,
            self.voice_url_edit.text().strip(),
            self.username_edit.text().strip(),
            self.password_edit.text(),
            "",
        )
        try:
            # 一覧を取得してからボイス名を選ぶので、この欄の入力は不要。
            validate_voice(replace(values, voice_name="一覧取得"))
        except EnvironmentSettingsError as exc:
            self.voice_list_status.setText(str(exc))
            self._connection_scroll.ensureWidgetVisible(self.voice_list_status)
            return
        self._voice_list_generation += 1
        self._voice_list_cancel = Event()
        self._voice_list_running = True
        self._update_voice_list_button()
        self.voice_list_status.setText(
            "入力中の接続情報でボイス一覧を取得しています。合成・保存はしません。"
        )
        self._connection_scroll.ensureWidgetVisible(self.voice_list_status)
        try:
            Thread(
                target=_run_voice_list,
                args=(
                    values,
                    self._audio.voice_output.timeout,
                    self._voice_list_cancel,
                    self._voice_list_generation,
                    self._voice_list_results,
                ),
                daemon=True,
            ).start()
        except RuntimeError:
            self._voice_list_running = False
            self._update_voice_list_button()
            self.voice_list_status.setText(
                "ボイス一覧の取得を開始できませんでした。時間を置いて再試行してください。"
            )
            return
        self._voice_list_timer.start()

    def _poll_voice_list(self) -> None:
        try:
            generation, result = self._voice_list_results.get_nowait()
        except Empty:
            return
        self._voice_list_running = False
        self._voice_list_timer.stop()
        self._update_voice_list_button()
        if generation != self._voice_list_generation:
            self.voice_list_status.setText(
                "設定が変わったため取得した一覧を使いません。もう一度取得してください。"
            )
            return
        self.voice_list_status.setText(result.message)
        self._connection_scroll.ensureWidgetVisible(self.voice_list_status)
        if not result.success or not result.voices:
            return
        picker = QInputDialog(self)
        self._voice_picker = picker
        self._update_voice_list_button()
        picker.setWindowTitle("VoiSona Talkのボイスを選択")
        picker.setLabelText(
            "使用するボイスを選び「選択」を押してください。設定はまだ保存しません。"
        )
        picker.setComboBoxItems(list(result.voices))
        picker.setComboBoxEditable(False)
        current = self.voice_name_edit.text()
        if current in result.voices:
            picker.setTextValue(current)
        picker.setOkButtonText("選択")
        picker.setCancelButtonText("キャンセル")
        combo = picker.findChild(QComboBox)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(12)

        def choose() -> None:
            if generation == self._voice_list_generation and picker.textValue() in result.voices:
                self.voice_name_edit.setText(picker.textValue())
                self.voice_list_status.setText(
                    "選んだボイス名を入力欄に反映しました。「保存」で確定します。"
                )
                self._connection_scroll.ensureWidgetVisible(self.voice_list_status)

        def finished() -> None:
            self._voice_picker = None
            self._update_voice_list_button()
            picker.deleteLater()

        picker.accepted.connect(choose)
        picker.finished.connect(finished)
        _fit_dialog(picker, 500, 220)
        picker.open()

    def _invalidate_voice_test(self) -> None:
        self._voice_test_generation += 1
        self._voice_test_cancel.set()
        if self.voice_test_status.text():
            self.voice_test_status.setText(
                "入力が変わったため試聴を停止しています。終了後にもう一度テストしてください。"
                if self._voice_test_running
                else "入力が変わりました。もう一度VoiSona Talkの接続テスト・試聴を行ってください。"
            )

    def _test_voice(self) -> None:
        if self._voice_test_running:
            return
        settings = self._audio.voice_output
        if self.environment_check.isChecked():
            values = VoiceValues(
                True,
                settings.base_url,
                os.getenv(settings.username_env, "").strip(),
                os.getenv(settings.password_env, "").strip(),
                os.getenv(settings.voice_name_env, "").strip(),
            )
        else:
            values = VoiceValues(
                True,
                self.voice_url_edit.text().strip(),
                self.username_edit.text().strip(),
                self.password_edit.text(),
                self.voice_name_edit.text().strip(),
            )
        try:
            validate_voice(values)
        except EnvironmentSettingsError as exc:
            self.voice_test_status.setText(str(exc))
            self._connection_scroll.ensureWidgetVisible(self.voice_test_status)
            return
        self._voice_test_generation += 1
        self._voice_test_cancel = Event()
        self._voice_test_running = True
        self.voice_test_button.setEnabled(False)
        self.voice_stop_button.setEnabled(True)
        self.voice_test_status.setText(
            "VoiSona Talkの接続テスト・試聴を開始しています。設定は保存していません。"
        )
        self._connection_scroll.ensureWidgetVisible(self.voice_test_status)
        options = {
            "language": settings.language,
            "timeout": settings.timeout,
            "playback_preroll": settings.playback_preroll,
        }
        try:
            Thread(
                target=_run_voice_test,
                args=(
                    values,
                    options,
                    self._voice_test_cancel,
                    self._voice_test_generation,
                    self._voice_test_results,
                ),
                daemon=True,
            ).start()
        except RuntimeError:
            self._voice_test_running = False
            self.voice_test_button.setEnabled(True)
            self.voice_stop_button.setEnabled(False)
            self.voice_test_status.setText(
                "試聴を開始できませんでした。時間を置いて再試行してください。"
            )
            return
        self._voice_test_timer.start()

    def _stop_voice_test(self) -> None:
        self._voice_test_generation += 1
        self._voice_test_cancel.set()
        self.voice_stop_button.setEnabled(False)
        self.voice_test_status.setText("試聴を停止しています。設定は保存していません。")

    def _poll_voice_test(self) -> None:
        while True:
            try:
                generation, result = self._voice_test_results.get_nowait()
            except Empty:
                return
            if isinstance(result, ConnectionTestResult):
                self._voice_test_running = False
                self._voice_test_timer.stop()
                self.voice_test_button.setEnabled(True)
                self.voice_stop_button.setEnabled(False)
                self.voice_test_status.setText(
                    result.message
                    if generation == self._voice_test_generation
                    else "入力変更または停止操作により試聴を停止しました。結果は使いません。設定は保存していません。"
                )
                self._connection_scroll.ensureWidgetVisible(self.voice_test_status)
                return
            if generation == self._voice_test_generation:
                self.voice_test_status.setText(result)

    def _update_voice(self) -> None:
        self._invalidate_voice_test()
        self._invalidate_voice_list()
        for edit in (
            self.voice_url_edit,
            self.username_edit,
            self.password_edit,
            self.voice_name_edit,
        ):
            edit.setEnabled(self.voice_check.isChecked())

    def _save(self) -> None:
        from multi_llm_companion.config import ConfigError

        self._remember_provider()
        settings = EnvironmentSettings(
            self.environment_check.isChecked(),
            dict(self._drafts),
            VoiceValues(
                self.voice_check.isChecked(),
                self.voice_url_edit.text().strip(),
                self.username_edit.text().strip(),
                self.password_edit.text(),
                self.voice_name_edit.text().strip(),
            ),
        )
        selected_provider = self.provider_combo.currentText()
        try:
            if self._deletion_pending and self._store.load() != self._stored_settings:
                raise ConfigError(
                    "保存済みの接続情報が別の場所で変更されました。"
                    "削除を保存せず、この画面を開き直してから確認してください。"
                )
            if not settings.use_environment:
                for name, configured in self._providers.providers.items():
                    values = settings.providers.get(name, ConnectionValues())
                    if name == selected_provider or any(
                        (values.base_url, values.api_key, values.model)
                    ):
                        resolve_provider(name, configured, settings)
                validate_voice(settings.voice)
            # 環境変数方式へ戻す場合、無効な入力途中の値は保存しない。
            if settings.use_environment:
                settings = replace(
                    self._settings, use_environment=True, providers=dict(self._saved_values)
                )
                if selected_provider != self._active:
                    resolve_provider(
                        selected_provider, self._providers.providers[selected_provider], settings
                    )
            save_environment_config(
                self._root,
                settings,
                selected_provider,
                app_config_path=self._app_config_path,
                providers=self._providers,
                expected_providers=self._original_providers,
            )
        except (ConfigError, EnvironmentSettingsError) as exc:
            self.error_label.setText(str(exc))
            return
        self.accept()
