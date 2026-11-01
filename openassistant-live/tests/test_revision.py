"""Regression coverage for the PR review's preparation and lifecycle failures."""

import asyncio
import json
import os
import sys
import threading
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer
from test_alignment import connect, event, open_chat
from test_pi_session import Client, Stream

from openassistant.live import app as live
from openassistant.live import call as runtime
from openassistant.live.history import Conversations
from openassistant.models import ModelCatalogue, model_options
from openassistant.pi_session import Session, TurnRejected


async def test_discovery_failure_does_not_block_concurrent_turns(tmp_path, monkeypatch):
    async def unavailable(*args):
        raise TimeoutError("extension startup failed")

    monkeypatch.setattr(live, "model_options", unavailable)
    started = []
    release = asyncio.Event()

    async def run(self, prompt, recording):
        started.append(prompt)
        await self.send("start")
        await release.wait()

    monkeypatch.setattr(live.Call, "run", run)
    config = SimpleNamespace(
        origin="http://localhost",
        access={},
        tailnet_only=True,
        history_dir=tmp_path,
        workspace=tmp_path,
    )
    async with TestClient(TestServer(live.create_app(config))) as client:
        sockets = [await connect(client), await connect(client)]
        for index, ws in enumerate(sockets):
            chat = await open_chat(ws)
            await ws.send_json({"type": "model_list", "conversation": chat})
            await event(ws, "model_error")
            await ws.send_json({"type": "prompt", "text": str(index)})
        for ws in sockets:
            await event(ws, "start")
        assert sorted(started) == ["0", "1"]
        release.set()


async def test_cache_deduplicates_expires_and_accepts_session_refresh():
    calls = []

    async def discover(config, selected):
        calls.append(selected)
        await asyncio.sleep(0)
        return {"selected": selected, "value": len(calls)}

    cache = ModelCatalogue(None, discover, ttl=300)
    results = await asyncio.gather(*(cache.options(None) for _ in range(5)))
    assert len(calls) == 1 and all(r["value"] == 1 for r in results)
    cache.entries[None] = (cache.entries[None][0] - cache.ttl - 1, results[0])
    assert (await cache.options(None))["value"] == 2
    cache.offer({"selected": None, "value": 9})
    assert (await cache.options(None))["value"] == 9
    assert (await cache.options(None, refresh=True))["value"] == 3


async def test_removed_model_still_returns_catalogue():
    class Discovery:
        def __init__(self, **kwargs):
            assert "--no-extensions" not in kwargs["extra_args"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get_state(self):
            return {"model": {"provider": "ext", "id": "ok", "name": "Extension"}}

        async def get_available_models(self):
            return [(await self.get_state())["model"]]

    result = await model_options(SimpleNamespace(workspace="/tmp"), '["ext","gone"]', Discovery)
    assert result["unavailable"]
    assert result["models"][0]["id"] == '["ext","ok"]'
    assert result["selected"] == '["ext","gone"]'


def test_session_resolves_current_default_before_resume_and_ignores_catalogue_error(tmp_path):
    selected, efforts, accepted, saved = [], [], [], []

    class Native(Client):
        def switch_session(self, path):
            self.model = {"provider": "old", "id": "old"}
            return super().switch_session(path)

        def set_model(self, provider, model):
            selected.append((provider, model))
            super().set_model(provider, model)

        def set_thinking_level(self, effort):
            efforts.append(effort)

        def get_available_models(self, **kwargs):
            raise TimeoutError("catalogue broken")

    folder = tmp_path / "pi-sessions"
    folder.mkdir()
    thread = folder / "resume.jsonl"
    thread.write_text("")
    config = SimpleNamespace(
        history_dir=tmp_path, workspace=tmp_path, turn_selection={"selected": None, "effort": None}
    )
    Session(
        config,
        saved.append,
        lambda _: None,
        client_factory=Native,
        ready=lambda: accepted.append(True),
        catalogue=lambda _: None,
    ).run(str(thread), "Hi")
    assert selected == [("local", "default")]
    assert efforts == ["off"] and accepted == [True] and saved == [str(thread)]


def test_preparation_rejection_never_accepts_or_binds(tmp_path):
    class Missing(Client):
        def set_model(self, *args):
            raise RuntimeError("missing provider")

    accepted, saved = [], []
    config = SimpleNamespace(history_dir=tmp_path, workspace=tmp_path)
    with pytest.raises(TurnRejected):
        Session(
            config,
            saved.append,
            lambda _: None,
            client_factory=Missing,
            ready=lambda: accepted.append(True),
        ).run(None, "Hi")
    assert not accepted and not saved


async def test_reading_picker_does_not_persist_effort_and_reset_removes_override(tmp_path):
    config = SimpleNamespace(
        origin="http://localhost",
        access={},
        tailnet_only=True,
        history_dir=tmp_path,
        workspace=tmp_path,
    )
    history = Conversations(tmp_path, tmp_path)
    async with TestClient(TestServer(live.create_app(config))) as client:
        ws = await connect(client)
        chat = await open_chat(ws)
        await ws.send_json({"type": "model_list", "conversation": chat})
        options = await event(ws, "model_options")
        assert history.effort(chat, "") is None and options["inherited"]
        for value in ("high", None):
            await ws.send_json(
                {
                    "type": "effort_select",
                    "conversation": chat,
                    "model": options["effective"],
                    "effort": value,
                }
            )
            options = await event(ws, "model_options")
            assert history.effort(chat, "") == value
        assert options["inherited"]


async def test_optional_inbox_timeout_keeps_socket_usable(tmp_path, monkeypatch):
    from aiohttp import web

    original = web.WebSocketResponse.send_json

    async def send(self, data, **kwargs):
        if data.get("type") == "pending_inbox":
            raise TimeoutError("slow inbox")
        return await original(self, data, **kwargs)

    monkeypatch.setattr(web.WebSocketResponse, "send_json", send)
    config = SimpleNamespace(
        origin="http://localhost",
        access={},
        tailnet_only=True,
        history_dir=tmp_path,
        workspace=tmp_path,
    )
    async with TestClient(TestServer(live.create_app(config))) as client:
        ws = await connect(client)
        assert await open_chat(ws)
        await ws.send_json({"type": "pending_inbox"})
        assert await open_chat(ws)


async def test_watchdog_kills_unresponsive_pi_and_joins_run_worker(tmp_path, monkeypatch):
    """Exercise the actual SDK process transport, including SIGTERM resistance."""
    fake = tmp_path / "pi.py"
    fake.write_text("""import sys, json, os, signal, time
from pathlib import Path
if "--version" in sys.argv:
    print("1.3.0"); sys.exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
root = Path(sys.argv[sys.argv.index("--session-dir")+1])
(root / "pid").write_text(str(os.getpid()))
model = {"provider":"test","id":"test","name":"Test","input":["text"]}
for line in sys.stdin:
    request = json.loads(line); kind = request["type"]
    if kind == "prompt":
        print(json.dumps({"type":"response","id":request["id"],
                          "command":kind,"success":True}), flush=True)
        while True: time.sleep(1)
    data = {"model":model,"thinkingLevel":"off","sessionId":"test",
            "sessionFile":str(root/"chat.jsonl"), "isStreaming":False,
            "isCompacting":False,"pendingMessageCount":0}
    if kind == "get_available_thinking_levels": data = {"levels":["off"]}
    if kind == "get_available_models": data = {"models":[model]}
    print(json.dumps({"type":"response","id":request["id"],"command":kind,"success":True,"data":data}),flush=True)
while True: time.sleep(1)
""")
    finished = threading.Event()
    clients = []
    from pi_agent import PiClient

    class ObservedSession(Session):
        def run(self, *args, **kwargs):
            try:
                return super().run(*args, **kwargs)
            finally:
                finished.set()

    def factory(**kwargs):
        client = PiClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(
        runtime, "Session", lambda *a, **k: ObservedSession(*a, client_factory=factory, **k)
    )
    config = SimpleNamespace(
        history_dir=tmp_path,
        workspace=tmp_path,
        pi_executable=[sys.executable, str(fake)],
        inactivity_timeout=0.2,
    )
    call = runtime.Call(None, config)
    async with asyncio.timeout(10):
        await call.run("Silent tool")
    pid = int((tmp_path / "pi-sessions/pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert finished.is_set()
    assert not clients[0]._thread.is_alive()
    assert call.snapshot["state"] == "failed"
    assert "no Pi progress" in call.snapshot["detail"]


async def test_prompt_and_stop_are_processed_while_picker_is_still_loading(tmp_path, monkeypatch):
    loading, release = asyncio.Event(), asyncio.Event()

    async def slow(*args):
        loading.set()
        await release.wait()
        raise TimeoutError("catalogue unavailable")

    async def run(self, prompt, recording):
        await self.send("start")
        await asyncio.Event().wait()

    monkeypatch.setattr(live, "model_options", slow)
    monkeypatch.setattr(live.Call, "run", run)
    config = SimpleNamespace(
        origin="http://localhost",
        access={},
        tailnet_only=True,
        history_dir=tmp_path,
        workspace=tmp_path,
    )
    async with TestClient(TestServer(live.create_app(config))) as client:
        ws = await connect(client)
        chat = await open_chat(ws)
        await ws.send_json({"type": "model_list", "conversation": chat})
        await asyncio.wait_for(loading.wait(), 1)
        await ws.send_json({"type": "prompt", "text": "Do not wait for models"})
        await event(ws, "start")
        assert not release.is_set()
        await ws.send_json({"type": "stop"})
        await event(ws, "stopped")
        release.set()


async def test_default_reset_is_broadcast_even_if_discovery_fails(tmp_path, monkeypatch):
    async def fail(*args):
        raise RuntimeError("extension setup throws")

    monkeypatch.setattr(live, "model_options", fail)
    config = SimpleNamespace(
        origin="http://localhost",
        access={},
        tailnet_only=True,
        history_dir=tmp_path,
        workspace=tmp_path,
    )
    async with TestClient(TestServer(live.create_app(config))) as client:
        ws, peer = await connect(client), await connect(client)
        chat = await open_chat(ws)
        await open_chat(peer, chat)
        history = Conversations(tmp_path, tmp_path)
        history.set_model(chat, '["removed","model"]')
        await ws.send_json({"type": "model_select", "conversation": chat, "model": None})
        assert (await event(ws, "model_options"))["selected"] is None
        assert (await event(peer, "model_options"))["selected"] is None
        assert history.model(chat) is None


@pytest.mark.parametrize("mode", ["progress", "pending", "disabled"])
async def test_watchdog_keeps_live_or_waiting_tasks(mode, tmp_path, monkeypatch):
    from threading import Event

    running, finish = Event(), Event()

    class Working:
        def __init__(self, *args, **kwargs):
            self.ready, self.activity = kwargs["ready"], kwargs["activity"]
            self.ui = kwargs["ui_handler"]

        def run(self, *args):
            self.ready()
            running.set()
            if mode == "pending":
                self.ui({"method": "confirm", "title": "Wait for me"})
            while not finish.wait(0.01):
                if mode == "progress":
                    self.activity()
            return "Finished"

        def stop(self):
            finish.set()

    monkeypatch.setattr(runtime, "Session", Working)
    call = runtime.Call(None, SimpleNamespace(inactivity_timeout=0 if mode == "disabled" else 0.05))
    task = asyncio.create_task(call.run("Work"))
    try:
        assert await asyncio.to_thread(running.wait, 2)
        await asyncio.sleep(0.2)
        assert not task.done()
        finish.set()
        for token in tuple(call.pending_ui):
            call.answer_ui(token, True)
        await asyncio.wait_for(task, 2)
        assert call.snapshot["state"] == "finished"
    finally:
        finish.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def test_legacy_user_image_paths_are_not_returned(tmp_path):
    from test_alignment import PNG

    from openassistant.live.images import save_uploads

    history = Conversations(tmp_path, tmp_path)
    chat = history.create()["id"]
    image = save_uploads(history, chat, [PNG])[0]
    with history.connect() as db:
        db.execute("UPDATE images SET path='/private/server/upload.png'")
    folder = tmp_path / "pi-sessions"
    folder.mkdir()
    path = folder / "history.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "message",
                "id": "user",
                "message": {"role": "user", "content": [{"type": "image", "data": PNG}]},
            }
        )
        + "\n"
    )
    history.saved(chat, str(path), "Image")
    restored = history.transcript(chat)["messages"][0]["images"][0]
    assert restored["id"] == image["id"] and restored["path"].startswith("/upload/")


async def test_stop_joins_worker_blocked_by_event_backpressure(monkeypatch):
    produced, finished = threading.Event(), threading.Event()
    blocked = asyncio.Event()

    class Flood:
        def __init__(self, *args, **kwargs):
            self.ready, self.emit = kwargs["ready"], kwargs["message_event"]

        def run(self, *args):
            try:
                self.ready()
                for index in range(300):
                    if index == 130:
                        produced.set()
                    self.emit("delta", {"itemId": "answer", "delta": "x"})
            finally:
                finished.set()

        def stop(self):
            pass

    monkeypatch.setattr(runtime, "Session", Flood)
    call = runtime.Call(None, SimpleNamespace(inactivity_timeout=0))
    original = call.send

    async def send(kind, **data):
        if kind == "delta":
            blocked.set()
            await asyncio.Event().wait()
        await original(kind, **data)

    call.send = send
    call.task = asyncio.create_task(call.run("Flood"))
    await asyncio.wait_for(blocked.wait(), 2)
    # Let the producer fill the 128-event queue while the browser consumer stalls.
    await asyncio.sleep(0.05)
    await asyncio.wait_for(call.stop(), 2)
    assert finished.is_set()


def test_tool_streaming_updates_count_as_progress(tmp_path):
    ticks = []

    class Updates(Stream):
        def __iter__(self):
            return iter(
                [
                    SimpleNamespace(
                        text_delta=None, raw={"type": "tool_execution_update", "toolName": "bash"}
                    )
                    for _ in range(3)
                ]
            )

    class Native(Client):
        def stream(self, text):
            return Updates()

    Session(
        SimpleNamespace(history_dir=tmp_path, workspace=tmp_path),
        lambda _: None,
        lambda _: None,
        client_factory=Native,
        activity=lambda: ticks.append(True),
    ).run(None, "Hi")
    assert len(ticks) == 4  # prepared session plus every raw streaming event
