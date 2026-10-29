import asyncio
from types import SimpleNamespace

import pytest
from aiohttp import WSMsgType, WSServerHandshakeError
from aiohttp.test_utils import TestClient, TestServer
from pi_agent.errors import PiProcessError

from openassistant.live import app as live
from openassistant.live import call as live_call


@pytest.mark.parametrize("inactive", [False, True])
async def test_closed_agent_transport_does_not_end_call(client, monkeypatch, inactive):
    from threading import Event, Lock

    from pi_agent.errors import PiProcessError

    class ClosedClient:
        def abort(self):
            raise PiProcessError("Pi process is not running")

        def close(self):
            pass

    class FailedSession:
        stop = live_call.Session.stop

        def __init__(self, *args, **kwargs):
            self.stopped = Event()
            self.lock = Lock()
            self.client = ClosedClient()
            self.turn = SimpleNamespace(id="turn1")
            self.thread_id = "thread1"

        def run(self, *args):
            raise PiProcessError("Pi process is not running")

    monkeypatch.setattr(live_call, "Session", FailedSession)
    ws = await connect(client)
    assert (await ws.receive_json())["type"] == "ready"
    await ws.send_json({"type": "prompt", "text": "Hello"})
    async with asyncio.timeout(3):
        events = []
        while (event := await ws.receive_json())["type"] != "done":
            events.append(event)
        assert any(e["type"] == "task" and e["state"] == "failed" for e in events)
        await ws.send_json({"type": "stop"})
        event = await ws.receive_json()
        if event["type"] == "turn_ready":
            event = await ws.receive_json()
        assert event["type"] == "stopped"
        await ws.send_json({"type": "prompt", "text": "Try again"})
        assert (await ws.receive_json())["type"] == "start"
    await ws.close()


@pytest.fixture
async def client(tmp_path):
    config = SimpleNamespace(
        origin="http://localhost", access={"User": "x" * 40}, workspace=tmp_path, model=None
    )
    async with TestClient(TestServer(live.create_app(config))) as client:
        yield client


async def connect(client, token="x" * 40):
    ws = await client.ws_connect("/ws", headers={"Origin": "http://localhost"})
    await ws.send_json({"token": token})
    return ws


@pytest.mark.parametrize("failure", [PiProcessError, RuntimeError])
async def test_intentional_stop_is_not_reported_as_task_error(client, monkeypatch, failure):
    from threading import Event
    from time import sleep

    running, stopped = Event(), Event()

    class InterruptedSession:

        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args):
            running.set()
            assert stopped.wait(3)
            raise failure("Agent interrupted during shutdown")

        def stop(self):
            stopped.set()
            # The reader observes a closed transport before shutdown returns.
            sleep(0.05)

    monkeypatch.setattr(live_call, "Session", InterruptedSession)
    ws = await connect(client)
    await ws.receive_json()
    await ws.send_json({"type": "prompt", "text": "Work until interrupted"})
    assert await asyncio.to_thread(running.wait, 3)
    await ws.send_json({"type": "stop"})
    events = []
    async with asyncio.timeout(3):
        while True:
            event = await ws.receive_json()
            events.append(event)
            if event["type"] == "stopped":
                break
    assert not any(e["type"] == "task" and e["state"] == "failed" for e in events)
    assert not any(e["type"] == "notice" and "reference" in e["text"] for e in events)
    await ws.send_json({"type": "prompt", "text": "Continue"})
    assert (await ws.receive_json())["type"] == "start"
    await ws.close()


async def test_origin_auth_exclusive_connection_and_release(client):
    with pytest.raises(WSServerHandshakeError):
        await client.ws_connect("/ws", headers={"Origin": "https://evil.example"})
    bad = await connect(client, "bad")
    assert (await bad.receive_json())["type"] == "notice"
    assert (await bad.receive()).type == WSMsgType.CLOSE
    first = await connect(client)
    assert (await first.receive_json())["type"] == "ready"
    second = await connect(client)
    assert (await second.receive_json())["type"] == "notice"
    await first.close()
    for _ in range(20):
        third = await connect(client)
        event = await third.receive_json()
        if event["type"] == "ready":
            await third.close()
            break
        await third.close()
        await asyncio.sleep(0.01)
    else:
        pytest.fail("Call slot was not released")


async def test_stream_audio_and_separate_session_history(client, monkeypatch):
    threads = []
    spoken = []

    class FakeSession:
        def __init__(self, config, saved, progress, message_event, **kwargs):
            self.saved, self.events = saved, message_event

        def run(self, thread, prompt):
            threads.append(thread)
            self.saved("private-thread")
            self.events("activity", {"id": "tool1", "label": "Calendar search", "state": "running"})
            self.events(
                "activity", {"id": "tool1", "label": "Calendar search", "state": "returned"}
            )
            self.events("delta", {"itemId": "answer", "delta": "\n\n"})
            self.events("delta", {"itemId": "answer", "delta": "Hello there. "})
            self.events("item/completed", {"id": "answer", "text": "Hello there. "})
            return "Hello there. "

        def stop(self):
            pass

    async def speech(self, path, text):
        spoken.append(text)
        path.write_bytes(b"OggS-test")

    monkeypatch.setattr(live_call, "Session", FakeSession)
    monkeypatch.setattr(live_call.Speech, "fetch", speech)
    ws = await connect(client)
    await ws.receive_json()
    for _ in range(2):
        await ws.send_json({"type": "prompt", "text": "Hello", "thread": "foreign-thread"})
        events = []
        async with asyncio.timeout(5):
            while True:
                event = await ws.receive_json()
                events.append(event)
                if event["type"] == "done":
                    break
        assert any(event["type"] == "delta" for event in events)
        assert len([event for event in events if event["type"] == "audio"]) == 1
        assert [event["state"] for event in events if event["type"] == "activity"] == [
            "running",
            "returned",
        ]
        assert any(event["type"] == "task" and event["state"] == "finished" for event in events)
        async with asyncio.timeout(5):
            while (await ws.receive_json())["type"] != "turn_ready":
                pass
    assert spoken == ["Hello there. ", "Hello there. "]
    assert threads == [None, "private-thread"]
    await ws.close()






async def test_security_headers_and_static_assets(client):
    response = await client.get("/")
    assert response.status == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert (await client.get("/static/openassistant-mark.svg")).status == 200


async def test_tailnet_mode_no_key_and_separate_calls(tmp_path):
    config = SimpleNamespace(
        origin="https://private.test", access={}, tailnet_only=True, workspace=tmp_path, model=None
    )
    async with TestClient(TestServer(live.create_app(config))) as client:
        assert await (await client.get("/client-config")).json() == {"access_required": False}
        with pytest.raises(WSServerHandshakeError):
            await client.ws_connect("/ws", headers={"Origin": "https://foreign.test"})
        sockets = []
        for _ in range(2):
            ws = await client.ws_connect("/ws", headers={"Origin": config.origin})
            await ws.send_json({})
            assert (await ws.receive_json())["type"] == "ready"
            sockets.append(ws)
        for ws in sockets:
            await ws.close()


async def test_guidance_does_not_interrupt_and_ready_follows_cleanup(client, monkeypatch):
    from threading import Event

    entered, release, cleaned = Event(), Event(), Event()
    guidance = []

    class WorkingSession:

        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args):
            entered.set()
            release.wait(3)
            return "Finished"

        def steer(self, text):
            guidance.append(text)
            return True

        def stop(self):
            cleaned.set()
            release.set()

    monkeypatch.setattr(live_call, "Session", WorkingSession)
    ws = await connect(client)
    await ws.receive_json()
    await ws.send_json({"type": "prompt", "text": "Start"})
    async with asyncio.timeout(5):
        while not entered.is_set():
            await asyncio.sleep(0.01)
        await ws.send_json({"type": "steer", "id": "one", "text": "Also check tests"})
        while (event := await ws.receive_json())["type"] != "steer_result":
            pass
        assert event["status"] == "accepted"
        assert guidance == ["Also check tests"]
        assert not cleaned.is_set()
        release.set()
        while (await ws.receive_json())["type"] != "turn_ready":
            pass
        assert cleaned.is_set()
        await ws.send_json({"type": "steer", "id": "two", "text": "Late guidance"})
        assert (await ws.receive_json())["status"] == "not_sent"
    await ws.close()


def test_session_steer_pins_turn_and_does_not_interrupt():
    from threading import Event, Lock
    from unittest.mock import Mock

    session = live_call.Session.__new__(live_call.Session)
    session.turn = SimpleNamespace(id="active")
    session.thread_id = "thread"
    session.stopped = Event()
    session.lock = Lock()
    session.client = Mock()
    assert session.steer("More context")
    session.client.steer.assert_called_once_with("More context")
    session.client.abort.assert_not_called()
    session.client = None
    assert not session.steer("Late")
