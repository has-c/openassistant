import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import ClientResponseError
from aiohttp.test_utils import TestClient, TestServer

from openassistant.live.asr import RemoteTranscriber

SERVER = Path(__file__).resolve().parents[2] / "speech/whisper/server.py"
spec = importlib.util.spec_from_file_location("whisper_service", SERVER)
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


class FakeModel:
    def transcribe(self, path, **kwargs):
        assert Path(path).read_bytes() == b"audio"
        return [SimpleNamespace(text=" hello ")], SimpleNamespace(duration=1)


async def test_remote_transcription_and_health(tmp_path):
    async with TestClient(TestServer(service.create_app(FakeModel()))) as client:
        assert (await client.get("/health")).status == 200
        audio = tmp_path / "audio"
        audio.write_bytes(b"audio")
        remote = RemoteTranscriber(str(client.make_url("/transcribe")))
        assert await remote.transcribe(audio) == "hello"


async def test_rejects_empty_and_oversized_audio():
    async with TestClient(TestServer(service.create_app(FakeModel()))) as client:
        assert (await client.post("/transcribe", data=b"")).status == 400
        response = await client.post("/transcribe", data=b"x" * (service.MAX_AUDIO_BYTES + 1))
        assert response.status == 413


async def test_client_reports_service_error(tmp_path):
    async with TestClient(TestServer(service.create_app(FakeModel()))) as client:
        audio = tmp_path / "audio"
        audio.write_bytes(b"")
        remote = RemoteTranscriber(str(client.make_url("/transcribe")))
        with pytest.raises(ClientResponseError):
            await remote.transcribe(audio)
