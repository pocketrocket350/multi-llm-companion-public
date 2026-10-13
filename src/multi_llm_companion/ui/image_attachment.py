"""画像をローカルで検査し、送信・プレビュー共通のJPEGを作る。"""

import base64
import binascii
import threading
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import httpx
from PySide6.QtCore import (
    QBuffer,
    QByteArray,
    QIODevice,
    QMimeData,
    QObject,
    QSize,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QImage, QImageReader, QPainter, QPixmap

from multi_llm_companion.services.image_sharing import ImageSharingError, PreparedImage
from multi_llm_companion.services.web_search import safe_source_url


def _image_path_from_drop(mime: QMimeData) -> str | None:
    urls = mime.urls()
    if len(urls) != 1 or not urls[0].isLocalFile():
        return None
    path = urls[0].toLocalFile()
    if Path(path).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        return None
    return path


def prepare_image(path: str) -> PreparedImage:
    source = Path(path)
    if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ImageSharingError("添付できるのはPNG・JPEG・WebPの静止画像1枚です。")
    try:
        with source.open("rb") as stream:
            raw = stream.read(15_000_001)
    except OSError as exc:
        raise ImageSharingError("画像を開けませんでした。別の画像を選んでください。") from exc
    return _prepare_image_bytes(raw, source.name)


def _prepare_image_bytes(raw: bytes, name: str) -> PreparedImage:
    if not raw or len(raw) > 15_000_000:
        raise ImageSharingError("画像は15MB以内にしてください。")
    buffer = QBuffer()
    buffer.setData(QByteArray(raw))
    buffer.open(QIODevice.OpenModeFlag.ReadOnly)
    reader = QImageReader(buffer)
    reader.setAutoTransform(True)
    size = reader.size()
    if bytes(reader.format()).lower() not in {b"png", b"jpeg", b"jpg", b"webp"}:
        raise ImageSharingError("画像の形式を確認できませんでした。")
    if not size.isValid() or size.width() * size.height() > 24_000_000:
        raise ImageSharingError("画像は2400万画素以内にしてください。")
    if reader.supportsAnimation() and reader.imageCount() != 1:
        raise ImageSharingError("アニメーション画像は送信できません。静止画像を選んでください。")
    reader.setScaledSize(
        size.scaled(QSize(1536, 1536), Qt.AspectRatioMode.KeepAspectRatio)
        if max(size.width(), size.height()) > 1536
        else size
    )
    decoded = reader.read()
    if decoded.isNull():
        raise ImageSharingError(
            "画像を読み取れませんでした。ファイルが壊れている可能性があります。"
        )
    return _prepare_pixels(decoded, name)


def _prepare_pixels(decoded: QImage, name: str) -> PreparedImage:
    if decoded.isNull() or decoded.width() * decoded.height() > 24_000_000:
        raise ImageSharingError("画像は2400万画素以内の静止画像を選んでください。")
    if max(decoded.width(), decoded.height()) > 1536:
        decoded = decoded.scaled(1536, 1536, Qt.AspectRatioMode.KeepAspectRatio)
    # 新しい画素面に描画し、EXIF・位置・コメントなどを引き継がない。
    clean = QImage(decoded.size(), QImage.Format.Format_RGB32)
    clean.fill(Qt.GlobalColor.white)
    painter = QPainter(clean)
    painter.drawImage(0, 0, decoded)
    painter.end()
    output = QBuffer()
    output.open(QIODevice.OpenModeFlag.WriteOnly)
    if not clean.save(output, "JPEG", 90):
        raise ImageSharingError("送信用の画像へ変換できませんでした。")
    jpeg = bytes(output.data())
    if len(jpeg) > 2_000_000:
        raise ImageSharingError("変換後の画像が2MBを超えました。小さい画像を選んでください。")
    preview = (
        clean.scaled(
            320, 320, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        if max(clean.width(), clean.height()) > 320
        else clean
    )
    preview_output = QBuffer()
    preview_output.open(QIODevice.OpenModeFlag.WriteOnly)
    if not preview.save(preview_output, "JPEG", 75) or preview_output.size() > 120_000:
        raise ImageSharingError("履歴用の小さな画像を作成できませんでした。")
    name = "".join(c for c in name if c.isprintable())[:120] or "画像"
    return PreparedImage(
        name,
        clean.width(),
        clean.height(),
        jpeg,
        base64.b64encode(bytes(preview_output.data())).decode("ascii"),
    )


@dataclass(frozen=True)
class _ImageDrop:
    content: str | bytes | QImage


class _ImageFragment(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "img":
            self.sources.append(dict(attrs).get("src", ""))


def _image_drop_source(mime: QMimeData) -> str | _ImageDrop | None:
    """イベントの寿命内に画像を複製する。URLの取得はまだ行わない。"""
    if len(mime.urls()) > 1:
        return None
    local = _image_path_from_drop(mime)
    if local:
        return local
    if mime.urls() and mime.urls()[0].isLocalFile():
        return None
    for kind in ("image/png", "image/jpeg", "image/webp"):
        if mime.hasFormat(kind):
            raw = bytes(mime.data(kind))
            return _ImageDrop(raw) if len(raw) <= 15_000_000 else None
    if mime.hasImage():
        pixels = mime.imageData()
        if isinstance(pixels, QPixmap):
            pixels = pixels.toImage()
        if isinstance(pixels, QImage) and not pixels.isNull():
            if pixels.width() * pixels.height() > 24_000_000:
                return None
            return _ImageDrop(QImage(pixels))
    fragment = _ImageFragment()
    if mime.hasHtml():
        fragment.feed(mime.html()[:20_100_000])
    if len(fragment.sources) > 1:
        return None
    url = fragment.sources[0] if fragment.sources else ""
    if mime.urls():
        url = urljoin(mime.urls()[0].toString(), url) if url else mime.urls()[0].toString()
    if url and (safe_source_url(url) or url.startswith("data:image/")):
        return _ImageDrop(url)
    return None


def _prepare_drop(source: _ImageDrop, cancelled, *, transport=None) -> PreparedImage:
    content = source.content
    if cancelled():
        raise ImageSharingError("画像の取得を取り消しました。")
    if isinstance(content, QImage):
        return _prepare_pixels(content, "ブラウザの画像")
    if isinstance(content, bytes):
        return _prepare_image_bytes(content, "ブラウザの画像")
    if content.startswith("data:"):
        header, _, encoded = content.partition(",")
        if (
            header.lower()
            not in {"data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"}
            or len(encoded) > 20_000_000
        ):
            raise ImageSharingError("添付できるのは15MB以内のPNG・JPEG・WebPです。")
        try:
            return _prepare_image_bytes(base64.b64decode(encoded, validate=True), "ブラウザの画像")
        except (ValueError, binascii.Error) as exc:
            raise ImageSharingError("ブラウザから渡された画像を読み取れませんでした。") from exc
    started = time.monotonic()
    url = content
    try:
        with httpx.Client(
            timeout=httpx.Timeout(5.0), follow_redirects=False, trust_env=False, transport=transport
        ) as client:
            for _ in range(4):
                if cancelled() or time.monotonic() - started > 20:
                    raise ImageSharingError(
                        "画像の取得を取り消したか、時間内に取得できませんでした。"
                    )
                if not safe_source_url(url):
                    raise ImageSharingError(
                        "この画像の場所からは取得できません。画像を保存して添付してください。"
                    )
                with client.stream(
                    "GET", url, headers={"Accept": "image/png,image/jpeg,image/webp"}
                ) as response:
                    if response.is_redirect:
                        url = urljoin(url, response.headers.get("location", ""))
                        continue
                    if response.status_code != 200:
                        raise ImageSharingError(
                            "ブラウザの画像を取得できませんでした。画像を保存して添付してください。"
                        )
                    raw = bytearray()
                    for chunk in response.iter_bytes(64 * 1024):
                        if cancelled() or time.monotonic() - started > 20:
                            raise ImageSharingError(
                                "画像の取得を取り消したか、時間内に取得できませんでした。"
                            )
                        raw.extend(chunk)
                        if len(raw) > 15_000_000:
                            raise ImageSharingError("画像は15MB以内にしてください。")
                    return _prepare_image_bytes(
                        bytes(raw), unquote(Path(urlsplit(url).path).name) or "ブラウザの画像"
                    )
            raise ImageSharingError(
                "画像の転送先が多すぎるため取得を止めました。画像を保存して添付してください。"
            )
    except httpx.TimeoutException as exc:
        raise ImageSharingError(
            "画像を時間内に取得できませんでした。画像を保存して添付してください。"
        ) from exc
    except (httpx.RequestError, httpx.InvalidURL) as exc:
        raise ImageSharingError(
            "画像の取得先に接続できませんでした。画像を保存して添付してください。"
        ) from exc


class _ImageDropWorker(QObject):
    succeeded = Signal(int, object)
    failed = Signal(int, str)
    finished = Signal()

    def __init__(self, source: _ImageDrop, generation: int):
        super().__init__()
        self._owner_thread = self.thread()
        self.source, self.generation = source, generation
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    @Slot()
    def run(self):
        try:
            image = _prepare_drop(self.source, self._cancel.is_set)
            if not self._cancel.is_set():
                self.succeeded.emit(self.generation, image)
        except ImageSharingError as exc:
            if not self._cancel.is_set():
                self.failed.emit(self.generation, str(exc))
        except Exception:
            if not self._cancel.is_set():
                self.failed.emit(
                    self.generation,
                    "ブラウザの画像を読み取れませんでした。画像を保存して添付してください。",
                )
        finally:
            self.source = None
            # 通信スレッドではQtの接続解除を行わず、作成元へ返す。
            self.moveToThread(self._owner_thread)
            self.finished.emit()
