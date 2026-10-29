import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from openassistant.live.app import create_app
from openassistant.live.asr import LiveTranscriber


@pytest.fixture
def fake_worker(monkeypatch):
    original = asyncio.create_subprocess_exec
    processes = []
    code = """
import json, sys, time
print(json.dumps({'ready': True}), flush=True)
for line in sys.stdin:
    value = json.loads(line)
    if value == 'slow':
        time.sleep(30)
    if value == 'blank':
        print(json.dumps({'error': 'transcription_failed'}), flush=True)
        continue
    print(json.dumps({'text': value}), flush=True)
"""

    async def spawn(*args, **kwargs):
        process = await original(sys.executable, "-u", "-c", code, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    return processes


async def test_worker_reused_and_closed(fake_worker):
    asr = LiveTranscriber()
    try:
        assert await asr.transcribe(Path("one")) == "one"
        assert await asr.transcribe(Path("two")) == "two"
        assert len(fake_worker) == 1
    finally:
        await asr.close()
    assert fake_worker[0].returncode is not None


async def test_prewarm_starts_one_worker_and_reuses_it(fake_worker):
    asr = LiveTranscriber()
    try:
        asr.warm_in_background()
        await asr.warmup_task
        assert len(fake_worker) == 1
        assert await asr.transcribe(Path("warm")) == "warm"
        assert len(fake_worker) == 1
    finally:
        await asr.close()


async def test_app_startup_warms_and_shutdown_stops_worker(fake_worker, tmp_path):
    config = SimpleNamespace(origin="http://localhost", access={}, workspace=tmp_path)
    async with TestClient(TestServer(create_app(config))):
        async with asyncio.timeout(3):
            while not fake_worker:
                await asyncio.sleep(0.01)
        assert fake_worker[0].returncode is None
    assert fake_worker[0].returncode is not None


async def test_no_speech_keeps_warm_worker(fake_worker):
    asr = LiveTranscriber()
    try:
        asr.warm_in_background()
        await asr.warmup_task
        with pytest.raises(RuntimeError, match="No speech"):
            await asr.transcribe(Path("blank"))
        assert await asr.transcribe(Path("next")) == "next"
        assert len(fake_worker) == 1
    finally:
        await asr.close()


async def test_failed_warmup_retries_on_next_recording(fake_worker, monkeypatch):
    normal_spawn = asyncio.create_subprocess_exec
    original = asyncio.subprocess.create_subprocess_exec
    calls = 0

    async def fail_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return await original(sys.executable, "-c", "pass", **kwargs)
        return await normal_spawn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_first)
    asr = LiveTranscriber()
    try:
        asr.warm_in_background()
        await asr.warmup_task
        assert asr.process is None
        assert await asr.transcribe(Path("retry")) == "retry"
        assert calls == 2
    finally:
        await asr.close()


async def test_cancel_discards_worker_and_stale_response(fake_worker):
    asr = LiveTranscriber()
    pending = asyncio.create_task(asr.transcribe(Path("slow")))
    while not fake_worker:
        await asyncio.sleep(0.01)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert fake_worker[0].returncode is not None
    try:
        await asr.warmup_task
        assert await asr.transcribe(Path("fresh")) == "fresh"
        assert len(fake_worker) == 2
    finally:
        await asr.close()
