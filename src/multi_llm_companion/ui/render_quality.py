"""OpenGL画面を作る前に、3D描画品質を安全に設定する。"""

from __future__ import annotations

from PySide6.QtGui import QSurfaceFormat


def normalized_render_quality(value: str) -> str:
    """設定値を、対応している3段階のいずれかへ揃える。"""

    normalized = value.strip().lower()
    return normalized if normalized in {"low", "standard", "high"} else "standard"


def requested_sample_count(render_quality: str) -> int:
    """輪郭のギザギザを減らすために要求するサンプル数を返す。"""

    return {"low": 0, "standard": 4, "high": 8}[normalized_render_quality(render_quality)]


def surface_format(render_quality: str) -> QSurfaceFormat:
    """既定のGPU互換性を保ちながら、品質別のOpenGL形式を作る。"""

    surface = QSurfaceFormat(QSurfaceFormat.defaultFormat())
    surface.setSamples(requested_sample_count(render_quality))
    surface.setDepthBufferSize(24)
    surface.setStencilBufferSize(8)
    # 更新周期はQtタイマーで制御し、画面合成の同期待ちでUIを止めない。
    surface.setSwapInterval(0)
    return surface


def configure_default_surface_format(render_quality: str) -> QSurfaceFormat:
    """QApplicationより先に、全3D画面が共有する形式を設定する。"""

    surface = surface_format(render_quality)
    QSurfaceFormat.setDefaultFormat(surface)
    return surface
