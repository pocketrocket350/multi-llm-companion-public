"""Qt画面を次のテストへ持ち越さないための後始末。"""

import gc

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QDialog, QMainWindow


@pytest.fixture(autouse=True)
def dispose_test_windows():
    yield
    app = QApplication.instance()
    if app is None:
        return
    # closeだけでは残る親なし画面を、Qtのイベント経路で破棄してから参照を回収する。
    windows = app.topLevelWidgets()
    for window in windows:
        if isinstance(window, (QDialog, QMainWindow)) and window.parent() is None:
            assert window.close(), "テスト終了後も処理中の画面が残っています。"
            window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    del windows
    gc.collect()
