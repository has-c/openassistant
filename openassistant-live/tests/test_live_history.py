"""Saved chats persist Pi history and coordinate writers across live sockets."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from openassistant.live import app as live
from openassistant.live.history import Conversations


@pytest.fixture
def history(tmp_path):
    return Conversations(tmp_path / "index", tmp_path)


def test_index_survives_restart_without_copying_transcript(history):
    chat = history.create()
    history.saved(chat["id"], "native-thread", "First question")
    history.saved(chat["id"], "native-thread", "Follow-up must not replace title")
    restored = Conversations(history.path.parent, history.workspace)
    assert restored.get(chat["id"])["thread"] == "native-thread"
    assert restored.listing()["chats"][0]["title"] == "First question"
    assert restored.transcript(restored.create()["id"]) == {"messages": [], "truncated": False}
    with restored.connect() as db:
        assert [row[1] for row in db.execute("PRAGMA table_info(conversations)")] == [
            "id",
            "thread",
            "title",
            "created",
            "updated",
        ]
    with pytest.raises(ValueError):
        restored.get("unregistered-native-thread")


def test_hide_chat_survives_restart_without_deleting_or_stopping(history):
    chat = history.create()
    history.saved(chat["id"], "native-thread", "Keep this conversation")
    history.update_task(chat["id"], "working", "Working", 123)
    history.hide(chat["id"])
    restored = Conversations(history.path.parent, history.workspace)
    assert restored.listing()["chats"] == []
    assert restored.get(chat["id"])["thread"] == "native-thread"
    assert restored.task_status(chat["id"])["state"] == "working"
    restored.saved(chat["id"], "native-thread", "Later update")
    assert restored.listing()["chats"] == []
    with pytest.raises(ValueError):
        restored.hide("unknown-chat")


def test_stale_title_cannot_replace_newer_conversation(history):
    chat = history.create()["id"]
    first = history.completed_exchange(chat)
    second = history.completed_exchange(chat)
    assert not history.set_title_if_current(chat, first, "Outdated topic")
    assert history.set_title_if_current(chat, second, "Current topic")
    assert history.get(chat)["title"] == "Current topic"


def test_unregistered_sessions_are_not_listed(history):
    directory = history.path.parent / "pi-sessions"
    directory.mkdir()
    (directory / "unregistered.jsonl").write_text("")
    assert history.listing()["chats"] == []


def test_pi_transcript_filters_non_text_messages(history):
    directory = history.path.parent / "pi-sessions"
    directory.mkdir()
    path = directory / "chat.jsonl"
    items = [
        {"type": "message", "id": "system", "message": {"role": "system", "content": "setup"}},
        {
            "type": "message",
            "id": "u",
            "message": {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        },
        {
            "type": "message",
            "id": "tool",
            "message": {"role": "toolResult", "content": [{"type": "text", "text": "private"}]},
        },
        {
            "type": "message",
            "id": "a",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
        },
    ]
    path.write_text("\n".join(json.dumps(item) for item in items) + "\n")
    chat = history.create()
    history.saved(chat["id"], str(path), "Hi")
    assert history.transcript(chat["id"]) == {
        "messages": [
            {"id": "u", "who": "You", "text": "Hi"},
            {"id": "a", "who": "OpenAssistant", "text": "Hello"},
        ],
        "truncated": False,
    }


async def event(ws, kind):
    async with asyncio.timeout(5):
        while True:
            data = await ws.receive_json()
            if data["type"] == kind:
                return data


async def connect(client):
    ws = await client.ws_connect("/ws", headers={"Origin": "http://localhost"})
    await ws.send_json({"token": ""})
    await event(ws, "ready")
    return ws


async def test_hide_is_broadcast_and_keeps_current_chat(tmp_path):
    config = SimpleNamespace(
        origin="http://localhost",
        access={},
        tailnet_only=True,
        workspace=tmp_path,
        history_dir=tmp_path / "history",
    )
    async with TestClient(TestServer(live.create_app(config))) as client:
        first, second = await connect(client), await connect(client)
        await first.send_json({"type": "new_conversation"})
        chat = await event(first, "conversation")
        await second.send_json({"type": "open_conversation", "id": chat["id"]})
        await event(second, "conversation")
        await first.send_json({"type": "hide_conversation", "id": chat["id"]})
        assert (await event(first, "conversation_hidden"))["id"] == chat["id"]
        assert (await event(second, "conversation_hidden"))["id"] == chat["id"]
        await first.send_json({"type": "conversations"})
        assert (await event(first, "conversations"))["chats"] == []
        await second.send_json({"type": "open_conversation", "id": chat["id"]})
        assert (await event(second, "conversation"))["id"] == chat["id"]
        await first.close()
        await second.close()


async def test_shared_chat_has_one_writer_and_resumes_latest_binding(tmp_path, monkeypatch):
    config = SimpleNamespace(
        origin="http://localhost",
        access={},
        tailnet_only=True,
        workspace=tmp_path,
        model=None,
        history_dir=tmp_path / "history",
    )
    prompts, running = [], asyncio.Event()

    async def run(self, prompt, recording):
        prompts.append((self.thread, prompt))
        thread = "independent-thread" if prompt == "Independent" else "native-thread"
        self.saved_thread(thread, prompt)
        running.set()
        await self.send("user", text=prompt)
        await asyncio.Event().wait()

    monkeypatch.setattr(live.Call, "run", run)
    monkeypatch.setattr(
        Conversations, "transcript", lambda self, identifier: {"messages": [], "truncated": False}
    )
    async with TestClient(TestServer(live.create_app(config))) as client:
        first, second = await connect(client), await connect(client)
        await first.send_json({"type": "new_conversation"})
        chat = await event(first, "conversation")
        await second.send_json({"type": "open_conversation", "id": chat["id"]})
        assert not (await event(second, "conversation"))["busy"]
        await first.send_json({"type": "prompt", "text": "Start"})
        await running.wait()
        assert (await event(second, "start"))["observer"]
        await second.send_json({"type": "prompt", "text": "Duplicate"})
        await event(second, "conversation_busy")
        await second.send_json({"type": "stop"})
        assert "another open call" in (await event(second, "notice"))["text"]
        await second.send_json({"type": "new_conversation"})
        other = await event(second, "conversation")
        assert other["id"] != chat["id"]
        await second.send_json({"type": "prompt", "text": "Independent"})
        await event(second, "start")
        await first.send_json({"type": "new_conversation"})
        fresh = await event(first, "conversation")
        assert fresh["id"] != chat["id"] and not fresh["busy"]
        await first.close()
        await second.send_json({"type": "stop"})
        await event(second, "stopped")
        await second.send_json({"type": "open_conversation", "id": chat["id"]})
        restored = await event(second, "conversation")
        assert restored["busy"] and restored["observer"]
        await second.send_json({"type": "prompt", "text": "Continue"})
        await event(second, "conversation_busy")
        assert prompts == [(None, "Start"), (None, "Independent")]
        await second.close()
    # A new process can reopen the indexed conversation; arbitrary native IDs stay inaccessible.
    async with TestClient(TestServer(live.create_app(config))) as client:
        ws = await connect(client)
        await ws.send_json({"type": "open_conversation", "id": "native-thread"})
        await event(ws, "conversation_error")
        await ws.send_json({"type": "open_conversation", "id": chat["id"]})
        assert (await event(ws, "conversation"))["id"] == chat["id"]
        await ws.close()
