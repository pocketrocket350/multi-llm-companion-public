"""Whisper smallモデルを途中再開しながら安全に取得する。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

REPOSITORY = "Systran/faster-whisper-small"
FILES = {
    "config.json": (2370, None),
    "tokenizer.json": (2_203_239, None),
    "vocabulary.txt": (459_861, None),
    "model.bin": (
        483_546_902,
        "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671",
    ),
}


def main() -> int:
    destination = Path("local/models/faster-whisper-small")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(60.0, connect=30.0),
        ) as client:
            for filename, (expected_size, expected_hash) in FILES.items():
                _download(client, destination, filename, expected_size)
                if expected_hash is not None:
                    _verify_hash(destination / filename, expected_hash)
    except (httpx.HTTPError, OSError, RuntimeError) as exc:
        print(f"[エラー] {exc}")
        print("ネット接続を確認し、同じバッチを再実行してください。途中から再開します。")
        return 1

    print("[成功] Whisper smallモデルを利用できます。")
    return 0


def _download(
    client: httpx.Client,
    destination: Path,
    filename: str,
    expected_size: int,
) -> None:
    target = destination / filename
    if target.exists() and target.stat().st_size == expected_size:
        print(f"[確認済み] {filename}")
        return

    partial = destination / f"{filename}.part"
    downloaded = partial.stat().st_size if partial.exists() else 0
    if downloaded > expected_size:
        partial.unlink()
        downloaded = 0

    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    url = f"https://huggingface.co/{REPOSITORY}/resolve/main/{filename}"
    print(f"[取得] {filename} ({_megabytes(downloaded)}/{_megabytes(expected_size)} MB)")
    with client.stream("GET", url, headers=headers) as response:
        if response.status_code not in {200, 206}:
            raise RuntimeError(f"{filename} の取得に失敗しました (HTTP {response.status_code})")
        append = downloaded > 0 and response.status_code == 206
        if not append:
            downloaded = 0
        mode = "ab" if append else "wb"
        last_percent = -1
        with partial.open(mode) as output:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                percent = min(100, int(downloaded * 100 / expected_size))
                if percent != last_percent:
                    print(f"  {percent:3d}%  {_megabytes(downloaded)} MB", flush=True)
                    last_percent = percent

    if downloaded != expected_size:
        raise RuntimeError(
            f"{filename} のサイズが一致しません ({downloaded}/{expected_size}バイト)"
        )
    partial.replace(target)


def _verify_hash(path: Path, expected_hash: str) -> None:
    print("[検証] model.binのSHA-256を確認しています。")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        path.unlink(missing_ok=True)
        raise RuntimeError("model.binの検証に失敗しました。再取得が必要です")


def _megabytes(size: int) -> str:
    return f"{size / 1024 / 1024:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
