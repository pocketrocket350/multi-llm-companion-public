"""設定の保存と再起動を通常の利用者設定から分離して検査する。"""

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from test_application import _runtime
from test_ui import isolate_window_settings as isolate_window_settings

from multi_llm_companion.application import ApplicationBackend
from multi_llm_companion.ui.chat_window import ChatWindow


@pytest.fixture
def windows(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    runtime = _runtime(tmp_path)
    opened = []

    def create(*, available=True):
        backend = ApplicationBackend(runtime)
        monkeypatch.setattr(backend, "web_search_available", lambda: available)
        monkeypatch.setattr(backend, "image_sharing_available", lambda: available)

        def forbidden(*_args, **_kwargs):
            pytest.fail("設定復元だけで録音・音声・送信を開始してはいけません")

        for method in ("start_speech_recognition", "start_voice_output", "stream_message"):
            monkeypatch.setattr(backend, method, forbidden)
        window = ChatWindow(backend)
        opened.append((window, backend))
        app.processEvents()
        return window, backend

    yield create
    for window, backend in reversed(opened):
        window.close()
        window.deleteLater()
        backend.close()
    app.processEvents()


def fresh_settings(window):
    settings = QSettings(window._settings.fileName(), QSettings.Format.IniFormat)
    settings.sync()
    return settings


def test_topmost_switch_keeps_settings_visible_focused_and_editable(windows):
    app = QApplication.instance()
    window, _ = windows()
    window.show()
    window.options_button.click()
    window.options_tabs.setCurrentIndex(1)
    app.processEvents()
    dialog = window.options_dialog

    for enabled in (True, False, True):
        window.always_on_top_checkbox.setFocus()
        QTest.keyClick(window.always_on_top_checkbox, Qt.Key.Key_Space)
        app.processEvents()
        assert window.always_on_top_checkbox.isChecked() is enabled
        assert bool(window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint) is enabled
        assert bool(dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint) is enabled
        assert fresh_settings(window).value("always_on_top", type=bool) is enabled
        assert dialog.isVisible()
        assert app.activeWindow() is dialog
        assert dialog.focusWidget() is window.always_on_top_checkbox
        assert window.options_tabs.currentIndex() == 1
        # 設定を開き直さず、キーボードでも次の操作へ進める。
        window.rotation_360_checkbox.setFocus()
        QTest.keyClick(window.rotation_360_checkbox, Qt.Key.Key_Space)
        assert window.rotation_360_checkbox.isChecked()
        QTest.keyClick(window.rotation_360_checkbox, Qt.Key.Key_Space)
        assert not window.rotation_360_checkbox.isChecked()

    dialog.close()
    window.options_button.click()
    app.processEvents()
    assert dialog.isVisible()
    assert app.activeWindow() is dialog


def test_restoring_topmost_does_not_open_main_or_settings_early(windows):
    first, backend = windows()
    first.always_on_top_checkbox.setChecked(True)
    first.close()
    backend.close()
    restored, _ = windows()
    assert restored.always_on_top_checkbox.isChecked()
    assert restored.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    assert restored.options_dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    assert not restored.isVisible()
    assert not restored.options_dialog.isVisible()
    restored.show()
    assert restored.isVisible()
    assert not restored.options_dialog.isVisible()


def test_new_install_keeps_defaults_without_starting_activity(windows):
    window, backend = windows()
    assert window._camera_view == "bust"
    assert backend.web_search_enabled() and backend.image_sharing_enabled()
    assert window.web_permission_checkbox.isChecked()
    assert window.image_permission_checkbox.isChecked()
    assert not window.auto_send_recognition_checkbox.isChecked()
    assert not window.continuous_voice_mode_checkbox.isChecked()
    assert not window.hold_to_talk_checkbox.isChecked()
    assert not backend.is_listening()
    assert window._continuous_voice_waiting_generation is None
    assert window._thread is None


@pytest.mark.parametrize("web,image", [(False, False), (False, True), (True, False), (True, True)])
def test_settings_survive_restart_and_can_be_changed_back(windows, web, image):
    first, backend = windows()
    first.auto_send_recognition_checkbox.click()
    first.continuous_voice_mode_checkbox.click()
    first.hold_to_talk_checkbox.click()
    first.camera_view_button.click()
    first.show()
    first.content_splitter.setSizes([600, 400])
    first.always_on_top_checkbox.click()
    # オンへの戻しも保存されることを確認する。
    for checkbox, value in (
        (first.web_permission_checkbox, web),
        (first.image_permission_checkbox, image),
    ):
        checkbox.click()
        if value:
            checkbox.click()
    saved = fresh_settings(first)
    assert saved.value("camera_view") == "full"
    assert saved.value("voice/continuous", type=bool)
    assert saved.value("web_search/enabled", type=bool) is web
    assert saved.value("image_sharing/enabled", type=bool) is image
    first.close()
    backend.close()

    second, restored = windows()
    second.show()
    QApplication.processEvents()
    assert second._camera_view == "full"
    assert second.pmx_preview_widget._dense_view._view_mode == "full"
    assert second.content_splitter.sizes()[0] > second.content_splitter.sizes()[1]
    assert second.always_on_top_checkbox.isChecked()
    assert second.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    assert second.auto_send_recognition_checkbox.isChecked()
    assert second.continuous_voice_mode_checkbox.isChecked()
    assert second.hold_to_talk_checkbox.isChecked()
    assert restored.web_search_enabled() is web
    assert restored.image_sharing_enabled() is image
    assert second.web_permission_checkbox.isChecked() is web
    assert second.image_permission_checkbox.isChecked() is image
    assert not restored.is_listening()
    assert second._continuous_voice_waiting_generation is None
    assert second._thread is None

    second.auto_send_recognition_checkbox.click()
    second.continuous_voice_mode_checkbox.click()
    second.hold_to_talk_checkbox.click()
    second.camera_view_button.click()
    second.show()
    second.content_splitter.setSizes([470, 530])
    second.always_on_top_checkbox.click()
    second.close()
    restored.close()
    third, _ = windows()
    third.show()
    QApplication.processEvents()
    assert third._camera_view == "bust"
    assert third.content_splitter.sizes()[0] < third.content_splitter.sizes()[1]
    assert not third.always_on_top_checkbox.isChecked()
    assert not third.auto_send_recognition_checkbox.isChecked()
    assert not third.continuous_voice_mode_checkbox.isChecked()
    assert not third.hold_to_talk_checkbox.isChecked()


@pytest.mark.parametrize("enabled", [False, True])
def test_unsupported_provider_does_not_erase_saved_permissions(windows, enabled):
    first, backend = windows()
    first._toggle_web_permission(enabled)
    first._toggle_image_permission(enabled)
    first.close()
    backend.close()
    second, unsupported = windows(available=False)
    assert not second.web_permission_checkbox.isEnabled()
    assert not second.image_permission_checkbox.isEnabled()
    assert second.web_permission_checkbox.isChecked() is enabled
    assert second.image_permission_checkbox.isChecked() is enabled
    second.auto_send_recognition_checkbox.click()
    second.close()
    unsupported.close()
    third, supported = windows()
    assert supported.web_search_enabled() is enabled
    assert supported.image_sharing_enabled() is enabled
    assert third.web_permission_checkbox.isChecked() is enabled
    assert third.image_permission_checkbox.isChecked() is enabled


def test_preview_camera_never_overwrites_normal_view(windows):
    window, backend = windows()
    window.camera_view_button.click()
    window.camera_view_button.click()
    with window.pmx_preview_widget._temporary_preview_camera("full"):
        assert window.pmx_preview_widget._dense_view._view_mode == "full"
        window.auto_send_recognition_checkbox.click()
        window._save_window_preferences()
        assert fresh_settings(window).value("camera_view") == "bust"
    assert window.pmx_preview_widget._dense_view._view_mode == "bust"
    window.close()
    backend.close()
    restarted, _ = windows()
    assert restarted._camera_view == "bust"


def test_existing_preferences_remain_and_unknown_camera_falls_back(windows):
    window, backend = windows()
    window._settings.setValue("proactive/read_aloud", True)
    window._settings.setValue("proactive/minimum_interval_minutes", 73)
    window._settings.setValue("future/setting", "保持する値")
    window._settings.setValue("camera_view", "unknown")
    window.close()
    backend.close()
    restarted, _ = windows()
    assert restarted._camera_view == "bust"
    restarted.hold_to_talk_checkbox.click()
    saved = fresh_settings(restarted)
    assert saved.value("proactive/read_aloud", type=bool)
    assert saved.value("proactive/minimum_interval_minutes", type=int) == 73
    assert saved.value("future/setting") == "保持する値"
