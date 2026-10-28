"""Inline images are captured from agent replies and scoped to their saved chat."""

import base64
from types import SimpleNamespace

from aiohttp.test_utils import TestClient, TestServer

from openassistant.live import app as live
from openassistant.live.history import Conversations
from openassistant.live.images import MAX_IMAGE, capture_images

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
)


def test_capture_persists_bytes_and_rejects_untrusted_files(tmp_path):
    history = Conversations(tmp_path / "history", tmp_path)
    chat = history.create()["id"]
    photo = tmp_path / "photo.png"
    photo.write_bytes(PNG)
    fake = tmp_path / "fake.png"
    fake.write_text("not an image")
    big = tmp_path / "big.png"
    big.write_bytes(PNG + b"x" * MAX_IMAGE)
    link = tmp_path / "link.png"
    link.symlink_to(photo)
    images = capture_images(
        history, chat, "a", f"![Photo]({photo}) ![Fake]({fake}) ![Big]({big}) ![Link]({link})"
    )
    assert len(images) == 1
    photo.unlink()
    reopened = Conversations(tmp_path / "history", tmp_path)
    assert reopened.image_list(chat, "a") == images
    assert reopened.image(chat, images[0]["id"])["data"] == PNG
    other = reopened.create()["id"]
    assert reopened.image(other, images[0]["id"]) is None


async def test_websocket_image_request_is_conversation_scoped(tmp_path):
    config = SimpleNamespace(
        origin="http://localhost",
        access={},
        tailnet_only=True,
        workspace=tmp_path,
        history_dir=tmp_path / "history",
    )
    async with TestClient(TestServer(live.create_app(config))) as client:
        ws = await client.ws_connect("/ws", headers={"Origin": "http://localhost"})
        await ws.send_json({"token": ""})
        assert (await ws.receive_json())["type"] == "ready"
        await ws.send_json({"type": "new_conversation"})
        chat = await ws.receive_json()
        assert chat["type"] == "conversation"
        history = Conversations(config.history_dir, tmp_path)
        photo = tmp_path / "photo.png"
        photo.write_bytes(PNG)
        image = capture_images(history, chat["id"], "item", f"![Photo]({photo})")[0]
        await ws.send_json({"type": "get_image", "id": image["id"]})
        response = await ws.receive_json()
        assert response["type"] == "image_data"
        assert response["mime"] == "image/png"
        assert base64.b64decode(response["data"]) == PNG
        await ws.send_json({"type": "new_conversation"})
        assert (await ws.receive_json())["type"] == "conversation"
        await ws.send_json({"type": "get_image", "id": image["id"]})
        await ws.send_json({"type": "conversations"})
        assert (await ws.receive_json())["type"] == "conversations"
        await ws.close()


async def test_agent_message_captures_image_before_delivery(tmp_path):
    class Socket:
        closed = False

        async def send_json(self, data):
            self.message = data

    history = Conversations(tmp_path / "history", tmp_path)
    chat = history.create()["id"]
    photo = tmp_path / "photo.png"
    photo.write_bytes(PNG)
    socket = Socket()
    call = live.Call(socket, SimpleNamespace())
    call.history, call.conversation = history, chat
    await call.send("message", item="answer", text=f"![Photo]({photo})")
    assert socket.message["images"] == history.image_list(chat, "answer")
    assert call.snapshot["responses"][0]["images"] == socket.message["images"]


def test_restored_native_message_includes_captured_image(tmp_path):
    import json

    history = Conversations(tmp_path / "history", tmp_path)
    chat = history.create()["id"]
    photo = tmp_path / "photo.png"
    photo.write_bytes(PNG)
    text = f"![Photo]({photo})"
    images = capture_images(history, chat, "answer", text)
    photo.unlink()
    directory = history.path.parent / "pi-sessions"
    directory.mkdir()
    session = directory / "chat.jsonl"
    session.write_text(json.dumps({"type": "message", "id": "answer", "message": {
        "role": "assistant", "content": [{"type": "text", "text": text}]
    }}) + "\n")
    history.saved(chat, str(session), "Show image")
    restored = history.transcript(chat)["messages"][0]
    assert restored["images"] == images
    assert history.image(chat, images[0]["id"])["data"] == PNG
