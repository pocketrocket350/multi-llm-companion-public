"""既存の配色で操作アイコンとWindowsのタイトルバーを描く。"""

import ctypes
import sys
from functools import cache

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QFont, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

_SHAPES = {
    "history": '<path d="M4 6h16M4 12h16M4 18h16"/>',
    "settings": (
        '<path d="M9.5 3h5l.5 3 2 1 2.8-1 2.5 4.3-2.3 2 .1 2.2 2.2 2'
        "-2.5 4.3-2.8-1-2 1-.5 3h-5l-.5-3-2-1-2.8 1-2.5-4.3 2.3-2-.1-2.2"
        '-2.2-2L4.2 6 7 7l2-1z" transform="translate(1.2 -.8) scale(.9)"/>'
        '<circle cx="12" cy="12" r="3"/>'
    ),
    "microphone": (
        '<rect x="9" y="3" width="6" height="12" rx="3"/>'
        '<path d="M6 11v1a6 6 0 0 0 12 0v-1M12 18v3M9 21h6"/>'
    ),
    "send": '<path d="m5 4 15 8-15 8 3-8z" fill="currentColor"/><path d="M8 12h6" stroke="#A1C9F0"/>',
    "attach": '<path d="M12 4v16M4 12h16"/>',
    "remove": '<path d="m6 6 12 12M18 6 6 18"/>',
    "stop": '<rect x="5" y="5" width="14" height="14" rx="2" fill="currentColor"/>',
    "full_body": ('<circle cx="12" cy="5" r="3"/><path d="M6 11h12M12 8v7m0 0-4 7m4-7 4 7"/>'),
    "bust": '<circle cx="12" cy="8" r="4"/><path d="M4 21v-2a8 6 0 0 1 16 0v2"/>',
}


@cache
def _application_icon() -> QIcon:
    """OS標準の閉じた傘を、タスクバーとショートカットに使う。"""

    icon = QIcon()
    for size in (16, 20, 24, 32, 40, 48, 64, 128, 256):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        font = QFont("Segoe UI Emoji")
        font.setPixelSize(round(size * 0.85))
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "🌂")
        painter.end()
        icon.addPixmap(pixmap)
    return icon


@cache
def _control_icon(name: str) -> QIcon:
    icon = QIcon()
    for mode, color in ((QIcon.Mode.Normal, "#1A267C"), (QIcon.Mode.Disabled, "#718396")):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
            f'color="{color}" fill="none" stroke="currentColor" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round">{_SHAPES[name]}</svg>'
        )
        renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
        # 高DPIでも文字フォントに依存せず、輪郭を保つ。
        for size in (24, 48, 72):
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            icon.addPixmap(pixmap, mode)
    return icon


def _set_control_icon(button: QPushButton, name: str, label: str) -> None:
    button.setProperty("iconOnly", True)
    button.setText("")
    button.setIcon(_control_icon(name))
    button.setIconSize(QSize(24, 24))
    button.setToolTip(label)
    button.setAccessibleName(label)


def _apply_title_bar_palette(window: QWidget) -> None:
    if sys.platform != "win32" or QApplication.platformName() != "windows":
        return
    handle = int(window.internalWinId())
    if not handle:
        return
    # 標準枠を保ち、未対応のWindowsではシステム既定色のままにする。
    setter = ctypes.WinDLL("dwmapi").DwmSetWindowAttribute
    setter.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
    setter.restype = ctypes.c_long
    for attribute, color in ((35, 0xF9F5E8), (36, 0x7C261A)):
        value = ctypes.c_uint(color)
        setter(handle, attribute, ctypes.byref(value), ctypes.sizeof(value))
