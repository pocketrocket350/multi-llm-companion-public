"""再配布しない利用者提供背景を、縦横比を保って表示する。"""

from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage


@lru_cache(maxsize=1)
def _local_background() -> QImage:
    path = (
        Path(__file__).resolve().parents[3]
        / "local/design/phase11-20260924/background-with-floor.png"
    )
    return QImage(str(path))


def _cover_rect(image_width: int, image_height: int, width: int, height: int) -> QRectF:
    """中央を基準に、表示欄を埋める原本の範囲を返す。"""
    scale = max(width / max(1, image_width), height / max(1, image_height), 0.001)
    crop_width, crop_height = width / scale, height / scale
    return QRectF(
        (image_width - crop_width) / 2,
        (image_height - crop_height) / 2,
        crop_width,
        crop_height,
    )


def _standing_background_rect(
    image_width: int, image_height: int, feet: QPointF, body_height: float
) -> QRectF:
    """原本の同じ位置に立ち、背景の家具と身長の比率を保つ。"""
    height = body_height * 1.25
    width = height * image_width / max(1, image_height)
    # 本棚の右側を立ち位置とし、足元の下にも原本の余白を残す。
    return QRectF(feet.x() - width * 0.60, feet.y() - height * 0.95, width, height)
