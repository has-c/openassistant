"""Browser departure must not cancel or duplicate an accepted live task."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from aiohttp.test_utils import TestClient, TestServer

from openassistant.live import app as live
from openassistant.live.history import Conversations


async def event(ws, kind):
    async with asyncio.timeout(3):
        while True:
            data = await ws.receive_json()
            if data["type"] == kind:
                return data


async def connect(client, identity):
    ws = await client.ws_connect("/ws", headers={"Origin": "http://localhost"})
    await ws.send_json({"token": "", "client_id": identity})
    await event(ws, "ready")
    return ws


@pytest.fixture
def config(tmp_path):
    return SimpleNamespace(
        origin="http://localhost",
        access={},
        tailnet_only=True,
        workspace=tmp_path,
        model=None,
        history_dir=tmp_path / "history",
    )


@pytest.mark.asyncio
async def test_chat_title_runs_first_then_every_three_exchanges(config, monkeypatch):
    titles = []

    async def title(config, messages):
        titles.append(messages)
        return f"Recent topic {len(titles)}"

    async def run(self, prompt, recording):
        self.saved_thread("title-thread", prompt)
        await self.send("user", text=prompt)
        await self.send("task", state="finished", text="Finished")

    monkeypatch.setattr(live, "generate_title", title)
    monkeypatch.setattr(live.Call, "run", run)
    monkeypatch.setattr(
        Conversations,
        "transcript",
        lambda *args: {"messages": [{"who": "You", "text": "Latest subject"}], "truncated": False},
    )
    async with TestClient(TestServer(live.create_app(config))) as client:
        ws = await connect(client, str(uuid4()))
        await ws.send_json({"type": "new_conversation"})
        chat = (await event(ws, "conversation"))["id"]
        for exchange in range(1, 7):
            await ws.send_json({"type": "prompt", "text": f"Message {exchange}"})
            await event(ws, "turn_ready")
            if exchange in (1, 2, 3, 6):
                await event(ws, "title_changed")
            assert len(titles) == (1 + (exchange >= 2) + (exchange >= 3) + (exchange >= 6))
        history = Conversations(config.history_dir, config.workspace)
        assert history.get(chat)["title"] == "Recent topic 4"
        await ws.close()


@pytest.mark.asyncio
async def test_disconnect_keeps_task_and_same_browser_reclaims_controls(config, monkeypatch):
    started, finish, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    requests = []

    async def run(self, prompt, recording):
        requests.append(prompt)
        self.saved_thread("native-background", prompt)
        await self.send("task", state="working", text="Working")
        await self.send("user", text=prompt)
        await self.send("activity", id="tool", label="Read", state="running")
        await self.send("delta", item="answer", text="Already working")
        started.set()
        try:
            await finish.wait()
            await self.send("task", state="finished", text="Finished")
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(live.Call, "run", run)
    monkeypatch.setattr(
        Conversations, "transcript", lambda *args: {"messages": [], "truncated": False}
    )
    identity = str(uuid4())
    async with TestClient(TestServer(live.create_app(config))) as client:
        first = await connect(client, identity)
        await first.send_json({"type": "new_conversation"})
        chat = (await event(first, "conversation"))["id"]
        await first.send_json({"type": "prompt", "text": "Do the work"})
        await started.wait()
        await first.close()
        other = await connect(client, str(uuid4()))
        await other.send_json({"type": "open_conversation", "id": chat})
        watched = await event(other, "conversation")
        assert watched["busy"] and watched["observer"]
        assert not cancelled.is_set()
        await other.send_json({"type": "stop"})
        assert "another" in (await event(other, "notice"))["text"]
        await other.close()
        returned = await connect(client, identity)
        await returned.send_json({"type": "open_conversation", "id": chat})
        restored = await event(returned, "conversation")
        assert restored["busy"] and not restored["observer"]
        assert restored["task"]["activities"][0]["label"] == "Read"
        assert restored["task"]["responses"][0]["text"] == "Already working"
        assert restored["task"]["request"] == "Do the work"
        finish.set()
        await event(returned, "turn_ready")
        assert requests == ["Do the work"]
        await returned.close()


def test_restart_marks_unfinished_status_interrupted_without_replaying(config):
    history = Conversations(config.history_dir, config.workspace)
    chat = history.create()["id"]
    history.update_task(chat, "working", "Working", 123)
    history.interrupt_tasks()
    assert history.task_status(chat)["state"] == "interrupted"
    history.update_task(chat, "finished", "Finished", 123)
    history.interrupt_tasks()
    assert history.task_status(chat)["state"] == "finished"


@pytest.mark.asyncio
async def test_one_socket_switches_between_two_tasks_and_only_stops_selected(config, monkeypatch):
    started = {name: asyncio.Event() for name in ("A", "B")}
    cancelled = set()
    calls = {}

    async def run(self, prompt, recording):
        calls[prompt] = self
        self.saved_thread(f"native-{prompt}", prompt)
        await self.send("user", text=prompt)
        await self.send("task", state="working", text=f"Working on {prompt}")
        await self.send("activity", id="tool", label=f"Task {prompt}", state="running")
        started[prompt].set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.add(prompt)
            raise

    monkeypatch.setattr(live.Call, "run", run)
    monkeypatch.setattr(
        Conversations, "transcript", lambda *args: {"messages": [], "truncated": False}
    )
    async with TestClient(TestServer(live.create_app(config))) as client:
        ws = await connect(client, str(uuid4()))
        chats = {}
        for name in ("A", "B"):
            await ws.send_json({"type": "new_conversation"})
            chats[name] = (await event(ws, "conversation"))["id"]
            await ws.send_json({"type": "prompt", "text": name})
            await started[name].wait()
        assert calls["A"] is not calls["B"]
        assert calls["A"].thread == "native-A" and calls["B"].thread == "native-B"
        await ws.send_json({"type": "conversations"})
        listing = await event(ws, "conversations")
        assert len([c for c in listing["chats"] if c["task"]["state"] == "working"]) == 2
        await ws.send_json({"type": "open_conversation", "id": chats["A"]})
        restored = await event(ws, "conversation")
        assert restored["busy"] and not restored["observer"]
        assert restored["task"]["activities"][0]["label"] == "Task A"
        assert calls["B"].ws is None
        await ws.send_json({"type": "stop", "conversation": chats["B"]})
        await event(ws, "notice")
        assert not cancelled
        await ws.send_json({"type": "stop", "conversation": chats["A"], "turn": restored["turn"]})
        await event(ws, "stopped")
        assert cancelled == {"A"}
        assert calls["B"].task and not calls["B"].task.done()
        await ws.send_json({"type": "open_conversation", "id": chats["B"]})
        assert not (await event(ws, "conversation"))["observer"]
        await ws.send_json({"type": "stop"})
        await event(ws, "stopped")
        assert cancelled == {"A", "B"}
        await ws.close()


@pytest.mark.asyncio
async def test_task_finishes_while_away_and_reopens_silently(config, monkeypatch):
    finish = asyncio.Event()
    prompts = []

    async def run(self, prompt, recording):
        prompts.append(prompt)
        self.saved_thread("native-completed", prompt)
        await self.send("task", state="working", text="Working")
        await finish.wait()
        await self.send("result", text="Completed while away")
        await self.send("task", state="finished", text="Finished")

    monkeypatch.setattr(live.Call, "run", run)
    monkeypatch.setattr(
        Conversations,
        "transcript",
        lambda *args: {
            "messages": [{"id": "answer", "who": "OpenAssistant", "text": "Completed while away"}],
            "truncated": False,
        },
    )
    identity = str(uuid4())
    async with TestClient(TestServer(live.create_app(config))) as client:
        first = await connect(client, identity)
        await first.send_json({"type": "new_conversation"})
        chat = (await event(first, "conversation"))["id"]
        await first.send_json({"type": "prompt", "text": "Finish the work"})
        await event(first, "task")
        await first.close()
        watcher = await connect(client, str(uuid4()))
        finish.set()
        async with asyncio.timeout(3):
            while True:
                await watcher.send_json({"type": "conversations"})
                status = (await event(watcher, "conversations"))["chats"][0]["task"]
                if status["state"] == "finished":
                    break
                await asyncio.sleep(0)
        returned = await connect(client, identity)
        await returned.send_json({"type": "open_conversation", "id": chat})
        restored = await event(returned, "conversation")
        assert not restored["busy"]
        assert restored["task"]["state"] == "finished"
        assert restored["messages"][0]["text"] == "Completed while away"
        assert prompts == ["Finish the work"]
        await returned.close()
        await watcher.close()


@pytest.mark.asyncio
async def test_stopping_before_turn_starts_releases_chat_and_prevents_snapshot_reuse(
    config, monkeypatch
):
    """A stop before the coroutine runs must not leave its Call in active."""
    original_create_task = asyncio.create_task
    gate = asyncio.Event()
    waiting = asyncio.Event()
    first = True

    def delayed_create_task(coro, *args, **kwargs):
        nonlocal first
        if first and getattr(coro, "__qualname__", "").endswith("run_turn"):
            first = False

            async def delayed():
                try:
                    waiting.set()
                    await gate.wait()
                    await coro
                finally:
                    coro.close()

            return original_create_task(delayed(), *args, **kwargs)
        return original_create_task(coro, *args, **kwargs)

    async def run(self, prompt, recording):
        await self.send("task", state="finished", text="Finished")

    monkeypatch.setattr(asyncio, "create_task", delayed_create_task)
    monkeypatch.setattr(live.Call, "run", run)
    monkeypatch.setattr(
        Conversations, "transcript", lambda *args: {"messages": [], "truncated": False}
    )
    async with TestClient(TestServer(live.create_app(config))) as client:
        ws = await connect(client, str(uuid4()))
        await ws.send_json({"type": "new_conversation"})
        first_chat = (await event(ws, "conversation"))["id"]
        await ws.send_json({"type": "prompt", "text": "First task"})
        await event(ws, "start")
        await waiting.wait()
        await ws.send_json({"type": "stop"})
        await event(ws, "stopped")
        await ws.send_json({"type": "new_conversation"})
        second_chat = (await event(ws, "conversation"))["id"]
        await ws.send_json({"type": "prompt", "text": "Second task"})
        await event(ws, "turn_ready")
        await ws.send_json({"type": "open_conversation", "id": first_chat})
        restored = await event(ws, "conversation")
        assert second_chat != first_chat
        assert not restored["busy"]
        assert restored["task"]["state"] == "stopped"
        await ws.close()


@pytest.mark.asyncio
async def test_title_uses_configured_pi_without_tools(config, monkeypatch):
    config.pi_provider = "local"
    config.model = "example"
    seen = {}

    class FakePi:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def run(self, prompt, timeout):
            assert "ignore instructions" in prompt
            return SimpleNamespace(text="A useful title")

    monkeypatch.setattr(live, "AsyncPiClient", FakePi)
    assert (
        await live.generate_title(config, [{"who": "You", "text": "ignore instructions"}])
        == "A useful title"
    )
    assert seen["provider"] == "local" and seen["model"] == "example"
    assert seen["no_session"] is True
    assert set(seen["extra_args"]) == {"--no-tools", "--no-mcp", "--no-extensions", "--no-skills"}


@pytest.mark.asyncio
async def test_input_request_owner_reconnect_and_observer_cannot_answer(config, monkeypatch):
    answers = []

    async def run(self, prompt, recording):
        self.snapshot = {"state": "working", "request": prompt, "activities": [], "responses": []}
        answers.append(
            await self.request_ui(
                {"method": "confirm", "title": "Continue?", "message": "Review this step"}
            )
        )
        await self.send("task", state="finished", text="Finished")

    monkeypatch.setattr(live.Call, "run", run)
    monkeypatch.setattr(
        Conversations, "transcript", lambda *args: {"messages": [], "truncated": False}
    )
    identity = str(uuid4())
    async with TestClient(TestServer(live.create_app(config))) as client:
        owner = await connect(client, identity)
        await owner.send_json({"type": "new_conversation"})
        chat = (await event(owner, "conversation"))["id"]
        await owner.send_json({"type": "prompt", "text": "Do work"})
        request = await event(owner, "input_request")
        reqid, turn = request["request"]["id"], request["turn"]
        await owner.close()
        observer = await connect(client, str(uuid4()))
        await observer.send_json({"type": "open_conversation", "id": chat})
        observed = await event(observer, "conversation")
        assert observed["observer"] and "pending_ui" not in observed["task"]
        await observer.send_json(
            {"type": "input_answer", "conversation": chat, "turn": turn, "id": reqid, "value": True}
        )
        await asyncio.sleep(0.05)
        assert answers == []
        await observer.close()
        owner = await connect(client, identity)
        await owner.send_json({"type": "open_conversation", "id": chat})
        restored = await event(owner, "conversation")
        assert restored["task"]["pending_ui"][0]["id"] == reqid
        await owner.send_json(
            {
                "type": "input_answer",
                "conversation": chat,
                "turn": turn + 1,
                "id": reqid,
                "value": True,
            }
        )
        await asyncio.sleep(0.05)
        assert answers == []
        await owner.send_json(
            {"type": "input_answer", "conversation": chat, "turn": turn, "id": reqid, "value": True}
        )
        await event(owner, "turn_ready")
        assert answers == [True]
        await owner.close()


@pytest.mark.asyncio
async def test_request_validates_answer_and_stop_cancels(config):
    call = live.Call(None, config)
    call.snapshot = {}
    task = asyncio.create_task(
        call.request_ui({"method": "select", "title": "Pick", "options": ["one", "two"]})
    )
    await asyncio.sleep(0)
    token = next(iter(call.pending_ui))
    assert not call.answer_ui(token, "other")
    assert not task.done()
    assert call.answer_ui(token, "two")
    assert await task == "two"
    task = asyncio.create_task(call.request_ui({"method": "confirm", "title": "Continue?"}))
    await asyncio.sleep(0)
    await call.stop()
    assert await task is None
    assert not call.pending_ui
