"""会話入力へフォーカスした時だけ、選択中のWindows IMEを開く。"""

import ctypes
import sys

from PySide6.QtWidgets import QApplication


def _enable_windows_ime() -> None:
    if sys.platform != "win32" or QApplication.platformName() != "windows":
        return
    user = ctypes.WinDLL("user32")
    user.GetFocus.argtypes = []
    user.GetFocus.restype = ctypes.c_void_p
    handle = user.GetFocus()
    if not handle:
        return
    imm = ctypes.WinDLL("imm32")
    imm.ImmGetContext.argtypes = [ctypes.c_void_p]
    imm.ImmGetContext.restype = ctypes.c_void_p
    imm.ImmSetOpenStatus.argtypes = [ctypes.c_void_p, ctypes.c_int]
    imm.ImmSetOpenStatus.restype = ctypes.c_int
    imm.ImmReleaseContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    imm.ImmReleaseContext.restype = ctypes.c_int
    context = imm.ImmGetContext(handle)
    if not context:
        return
    try:
        imm.ImmSetOpenStatus(context, True)
    finally:
        imm.ImmReleaseContext(handle, context)
