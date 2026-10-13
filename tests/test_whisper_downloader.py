from pathlib import Path

import httpx

from scripts.download_whisper_model import _download


def test_whisper_download_resumes_partial_file(tmp_path: Path) -> None:
    partial = tmp_path / "model.bin.part"
    partial.write_bytes(b"abc")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Range"] == "bytes=3-"
        return httpx.Response(206, content=b"def")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        _download(client, tmp_path, "model.bin", 6)

    assert (tmp_path / "model.bin").read_bytes() == b"abcdef"
    assert not partial.exists()
