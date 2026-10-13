import base64
import sqlite3

import httpx
import pytest
from PySide6.QtGui import QImage
from test_application import _runtime
from test_image_sharing import image_client, make_image_service
from test_image_sharing import picture as picture

from multi_llm_companion.application import ApplicationBackend
from multi_llm_companion.domain import AssistantResponse
from multi_llm_companion.domain.models import ImageAttachmentInfo
from multi_llm_companion.providers import MockProvider
from multi_llm_companion.services import ChatService
from multi_llm_companion.services.image_sharing import ImageSharingSession
from multi_llm_companion.storage import SQLiteConversationRepository
from multi_llm_companion.ui.image_attachment import prepare_image


def test_preview_is_small_metadata_free_and_never_upscaled(picture, tmp_path):
    for path, dimensions in ((picture, (80, 60)), (tmp_path / "large.png", (320, 213))):
        if path != picture:
            original = QImage(3000, 2000, QImage.Format.Format_RGB32)
            original.fill(0xFF0088FF)
            original.setText("private", "secret-location")
            assert original.save(str(path))
        image = prepare_image(str(path))
        raw = base64.b64decode(image.preview_jpeg)
        preview = QImage.fromData(raw, "JPEG")
        assert (preview.width(), preview.height()) == dimensions
        assert len(raw) <= 120_000 and not preview.textKeys()
        assert b"secret-location" not in raw
        assert image.preview_jpeg not in repr(image)


@pytest.mark.parametrize("delete_all", [False, True])
def test_sent_preview_survives_reopen_and_is_deleted_with_conversation(
    picture, tmp_path, delete_all
):
    path = tmp_path / "history.db"
    repository = SQLiteConversationRepository(path)
    service = ChatService(provider=MockProvider(), repository=repository, system_prompt="検査")
    service.image_sharing = ImageSharingSession(image_client(lambda _: httpx.Response(500)))
    image = prepare_image(str(picture))
    try:
        with pytest.raises(RuntimeError, match="画像解析"):
            service.send_message("この画像を見て", image=image)
        cid = service.current_conversation_id
    finally:
        service.close()
    repository = SQLiteConversationRepository(path)
    try:
        sent = repository.list_messages(cid)
        assert len(sent) == 1 and sent[0].role.value == "user"
        assert sent[0].response.image_attachment.preview_jpeg == image.preview_jpeg
        assert sent[0].response.image_attachment.analyzed_at is None
        assert str(picture.parent) not in sent[0].model_dump_json()
        if delete_all:
            assert repository.delete_all_conversations() == 1
        else:
            assert repository.delete_conversation(cid)
        assert repository.list_messages(cid) == []
    finally:
        repository.close()
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM messages").fetchone()[0] == 0
    assert picture.is_file()


@pytest.mark.parametrize("stream", [True, False])
def test_saved_image_never_enters_later_llm_history(picture, stream):
    prompts = []
    service, repository = make_image_service(
        responder=lambda items: prompts.append(items) or '{"speech":"見えたよ。"}'
    )
    image = prepare_image(str(picture))
    try:
        if stream:
            list(service.stream_message("見て", image=image))
        else:
            service.send_message("見て", image=image)
        saved = repository.list_messages(service.current_conversation_id)
        assert saved[0].response.image_attachment.preview_jpeg == image.preview_jpeg
        assert saved[-1].response.image_attachment.preview_jpeg is None
        service.send_message("続きも話そう")
        assert len(prompts) == 2
        text = "\n".join(item.content for items in prompts for item in items)
        assert image.preview_jpeg not in text
        assert base64.b64encode(image.jpeg).decode() not in text
        assert "再送していない" in text
    finally:
        service.close()


def test_old_metadata_only_history_remains_readable():
    legacy = AssistantResponse.model_validate(
        {
            "speech": "赤い画像だね。",
            "image_attachment": {
                "name": "old.png",
                "width": 80,
                "height": 60,
                "analyzed_at": "2026-09-22T12:00:00+09:00",
            },
        }
    )
    assert legacy.image_attachment.preview_jpeg is None
    assert legacy.image_attachment.analyzed_at.year == 2026


def test_failed_image_does_not_change_normal_memory_capture(picture):
    service, repository = make_image_service(
        lambda _: httpx.Response(500),
        lambda _: '{"speech":"青が好きなんだね。","memory_candidates":["ユーザーは青が好き"]}',
    )
    try:
        with pytest.raises(RuntimeError, match="画像解析"):
            service.send_message("画像を見て", image=prepare_image(str(picture)))
        service.send_message("青が好きなんだ")
        assert repository.list_memories()
    finally:
        service.close()


def test_user_attachment_does_not_replace_restored_avatar_response(tmp_path):
    backend = ApplicationBackend(_runtime(tmp_path))
    try:
        cid = backend.create_conversation("画像の取消後")
        backend._repository.add_message(
            cid, "assistant", "うん。", response=AssistantResponse(speech="うん。", gaze="left")
        )
        expected = backend.avatar_action(cid)
        backend._repository.add_message(
            cid,
            "user",
            "見て",
            response=AssistantResponse(
                speech="見て",
                image_attachment=ImageAttachmentInfo(name="image.jpg", width=80, height=60),
            ),
        )
        assert backend.avatar_action(cid) == expected
        assert backend.load_messages(cid)[-1].image_attachment.name == "image.jpg"
    finally:
        backend.close()
