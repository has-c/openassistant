"""Model, request, attachment and plan contracts at the application boundary."""

import asyncio
import base64
import json
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest
from aiohttp.test_utils import TestClient, TestServer

from openassistant.live import app as live
from openassistant.live.history import Conversations
from openassistant.live.images import save_uploads, validate_upload
from openassistant.models import model_key, model_options
from openassistant.plan import plan_steps

TEXT = '["one","same-id"]'
VISION = '["two","same-id"]'
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"


async def event(ws, kind):
    async with asyncio.timeout(3):
        while True:
            data = await ws.receive_json()
            if data["type"] == kind:
                return data


async def connect(client, owner=None, token=""):
    ws = await client.ws_connect("/ws", headers={"Origin": "http://localhost"})
    await ws.send_json({"token": token, "client_id": owner or str(uuid4())})
    await event(ws, "ready")
    return ws


async def open_chat(ws, identifier=None):
    await ws.send_json(
        {"type": "open_conversation" if identifier else "new_conversation", "id": identifier}
    )
    return (await event(ws, "conversation"))["id"]


@pytest.fixture
def config(tmp_path, monkeypatch):
    async def options(config, selected=None):
        if selected not in (None, TEXT, VISION):
            raise ValueError("Unknown model")
        return {
            "selected": selected,
            "default": TEXT,
            "effective": selected or TEXT,
            "models": [{"id": TEXT, "label": "Text"}, {"id": VISION, "label": "Vision"}],
            "provider": "two" if selected == VISION else "one",
            "model": "same-id",
            "levels": ["off", "high"] if selected == VISION else ["off"],
            "effort": "off",
            "images": selected == VISION,
        }

    monkeypatch.setattr(live, "model_options", options)
    return SimpleNamespace(
        origin="http://localhost",
        access={},
        tailnet_only=True,
        workspace=tmp_path,
        history_dir=tmp_path / "history",
        model=None,
    )


async def choose(ws, chat, model):
    await ws.send_json({"type": "model_select", "conversation": chat, "model": model})
    return await event(ws, "model_options")


async def test_settings_are_atomic_shared_and_apply_to_next_turn(config, monkeypatch):
    turns, finish = [], asyncio.Event()

    async def run(self, prompt, recording):
        turns.append(deepcopy(self.turn_selection))
        await finish.wait()
        await self.send("task", state="finished", text="Done")

    monkeypatch.setattr(live.Call, "run", run)
    async with TestClient(TestServer(live.create_app(config))) as client:
        ws = await connect(client)
        chat = await open_chat(ws)
        peer = await connect(client)
        await open_chat(peer, chat)
        await choose(ws, chat, VISION)
        assert (await event(peer, "model_options"))["selected"] == VISION
        await ws.send_json(
            {"type": "effort_select", "conversation": chat, "model": VISION, "effort": "high"}
        )
        assert (await event(ws, "model_options"))["effort"] == "high"
        await ws.send_json({"type": "prompt", "text": "First"})
        await event(ws, "start")
        await choose(ws, chat, TEXT)
        assert turns[0]["selected"] == VISION and turns[0]["effort"] == "high"
        await ws.send_json(
            {"type": "effort_select", "conversation": chat, "model": VISION, "effort": "high"}
        )
        await event(ws, "model_error")
        await ws.send_json({"type": "model_select", "conversation": chat, "model": "bad"})
        await event(ws, "model_error")
        history = Conversations(config.history_dir, config.workspace)
        assert history.model(chat) == TEXT
        finish.set()
        await event(ws, "turn_ready")
        await ws.send_json({"type": "prompt", "text": "Queued next"})
        await event(ws, "turn_ready")
        assert turns[1]["selected"] == TEXT
        assert (await choose(ws, chat, VISION))["effort"] == "high"
        assert (await choose(ws, chat, None))["effective"] == TEXT
        assert history.model(chat) is None
        # Other chats and reopening the SQLite database keep independent preferences.
        other = await open_chat(ws)
        await choose(ws, other, VISION)
        assert Conversations(config.history_dir, config.workspace).model(other) == VISION
        assert history.model(chat) is None


async def test_inbox_multiple_requests_survive_navigation_and_reject_observers(config, monkeypatch):
    answers = []

    async def run(self, prompt, recording):
        answers.extend(
            await asyncio.gather(
                self.request_ui({"method": "confirm", "title": "Confirm"}),
                self.request_ui({"method": "select", "title": "Choose", "options": ["A", "B"]}),
            )
        )
        await self.send("task", state="finished", text="Done")

    monkeypatch.setattr(live.Call, "run", run)
    owner = str(uuid4())
    async with TestClient(TestServer(live.create_app(config))) as client:
        ws = await connect(client, owner)
        chat = await open_chat(ws)
        await ws.send_json({"type": "prompt", "text": "Questions"})
        while True:
            inbox = await event(ws, "pending_inbox")
            if len(inbox["items"]) == 2:
                break
        await open_chat(ws)
        stranger = await connect(client)
        await open_chat(stranger, chat)
        item = inbox["items"][0]
        forged = {
            "type": "global_answer",
            "conversation": chat,
            "turn": item["turn"],
            "id": item["pending"]["id"],
            "value": True,
        }
        await stranger.send_json(forged)
        await event(stranger, "notice")
        assert not answers
        await ws.close()
        ws = await connect(client, owner)
        restored = await event(ws, "pending_inbox")
        assert len(restored["items"]) == 2
        for item in restored["items"]:
            value = True if item["pending"]["method"] == "confirm" else "B"
            await ws.send_json({**forged, "id": item["pending"]["id"], "value": value})
        async with asyncio.timeout(3):
            while not answers:
                await asyncio.sleep(0.01)
        assert answers == [True, "B"]
        await ws.send_json(forged)
        await event(ws, "notice")


async def test_image_submission_rejects_text_model_and_accepts_image_only(config, monkeypatch):
    received = []

    async def run(self, prompt, recording):
        received.append((prompt, self.attachments))
        await self.send("task", state="finished", text="Done")

    monkeypatch.setattr(live.Call, "run", run)
    async with TestClient(TestServer(live.create_app(config))) as client:
        ws = await connect(client)
        chat = await open_chat(ws)
        await ws.send_json({"type": "prompt", "text": "", "images": [PNG]})
        assert "cannot accept images" in (await event(ws, "upload_error"))["text"]
        assert not received
        await choose(ws, chat, VISION)
        await ws.send_json({"type": "prompt", "text": "", "images": [PNG, "broken"]})
        await event(ws, "upload_error")
        assert not (config.history_dir / "uploads").exists()
        await ws.send_json({"type": "prompt", "text": "", "images": [PNG]})
        await event(ws, "turn_ready")
        assert received[0][0] == "" and len(received[0][1]) == 1
        identifier = received[0][1][0]["id"]
        history = Conversations(config.history_dir, config.workspace)
        assert history.image(chat, identifier) is not None
        assert history.image(history.create()["id"], identifier) is None


def test_old_database_grouping_pagination_and_plan_persistence(tmp_path):
    import sqlite3

    directory = tmp_path / "history"
    directory.mkdir()
    with sqlite3.connect(directory / "conversations.sqlite") as db:
        db.execute(
            "CREATE TABLE conversations (id TEXT PRIMARY KEY, thread TEXT UNIQUE, "
            "title TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL)"
        )
        db.execute("INSERT INTO conversations VALUES ('old', NULL, 'Old chat', 1, 1)")
    history = Conversations(directory, tmp_path)
    history.update_task("old", "working", "Working", 1)
    for _ in range(55):
        history.create()
    assert history.listing()["chats"][0]["id"] == "old"
    assert len(history.listing(limit=100)["chats"]) == 56
    history.save_plan("old", [{"step": "Work", "status": "in_progress"}])
    history.interrupt_tasks()
    assert history.task_status("old")["state"] == "interrupted"
    assert history.task_status("old")["steps"][0]["step"] == "Work"


def test_uploaded_images_restore_from_native_history(tmp_path):
    history = Conversations(tmp_path / "history", tmp_path)
    chat = history.create()["id"]
    image = save_uploads(history, chat, [PNG])[0]
    folder = history.path.parent / "pi-sessions"
    folder.mkdir()
    thread = folder / "session.jsonl"
    thread.write_text(
        json.dumps(
            {
                "type": "message",
                "id": "user-1",
                "message": {
                    "role": "user",
                    "content": [{"type": "image", "data": PNG, "mimeType": "image/png"}],
                },
            }
        )
        + "\n"
    )
    history.saved(chat, str(thread), "")
    assert history.transcript(chat)["messages"][0]["images"][0]["id"] == image["id"]


def test_corrupt_image_batch_creates_no_files(tmp_path):
    history = Conversations(tmp_path / "history", tmp_path)
    chat = history.create()["id"]
    corrupt = base64.b64encode(b"\x89PNG\r\n\x1a\nnot pixels").decode()
    with pytest.raises(ValueError, match="damaged"):
        save_uploads(history, chat, [PNG, corrupt])
    assert not (history.path.parent / "uploads").exists()
    with pytest.raises(ValueError, match="large"):
        validate_upload("a" * 5_592_409)


@pytest.mark.parametrize(
    "steps",
    [
        None,
        [{"step": "Bad", "status": "done"}],
        [{"step": "A", "status": "in_progress"}, {"step": "B", "status": "in_progress"}],
    ],
)
def test_plan_validation(steps):
    with pytest.raises(ValueError):
        plan_steps(steps)


async def test_real_adapter_uses_provider_ids_and_runtime_levels():
    models = [{"provider": p, "id": "same", "name": p, "input": ["text"]} for p in ("a", "b")]
    clients = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["no_session"] and "--no-extensions" in kwargs["extra_args"]
            self.model = models[0]
            clients.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get_state(self):
            return {"model": self.model, "thinkingLevel": "off"}

        async def get_available_models(self):
            return models

        async def set_model(self, provider, model):
            self.model = next(m for m in models if m["provider"] == provider)

        async def get_available_thinking_levels(self):
            return ["off", "high"] if self.model["provider"] == "b" else ["off"]

    result = await model_options(SimpleNamespace(workspace="/tmp"), model_key(models[1]), Client)
    assert len({m["id"] for m in result["models"]}) == 2
    assert result["provider"] == "b" and result["levels"] == ["off", "high"]
    assert result["default"] == model_key(models[0])
