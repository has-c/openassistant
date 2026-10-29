"""Private HTTP interface for a single, warm faster-whisper model."""

import asyncio
import os
import tempfile
from pathlib import Path

from aiohttp import web

MAX_AUDIO_BYTES = 10 * 1024 * 1024


def create_app(model):
    app = web.Application(client_max_size=MAX_AUDIO_BYTES)
    lock = asyncio.Lock()

    async def health(_request):
        return web.json_response({"ready": True})

    def run_transcription(path):
        segments, info = model.transcribe(path, language="en", vad_filter=True)
        if info.duration > 65:
            raise ValueError("Utterance exceeds 65 seconds")
        text = " ".join(segment.text.strip() for segment in segments)
        if len(text) > 8000:
            raise ValueError("Transcript exceeds 8000 characters")
        return text

    async def transcribe(request):
        audio = await request.read()
        if not audio:
            raise web.HTTPBadRequest(text="Empty audio")
        name = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as audio_file:
                name = audio_file.name
                audio_file.write(audio)
            async with lock:
                task = asyncio.create_task(asyncio.to_thread(run_transcription, name))
                try:
                    text = await asyncio.shield(task)
                except asyncio.CancelledError:
                    await task
                    raise
            return web.json_response({"text": text})
        except ValueError:
            raise web.HTTPBadRequest(text="Audio or transcript exceeds limits") from None
        finally:
            if name:
                Path(name).unlink(missing_ok=True)

    app.router.add_get("/health", health)
    app.router.add_post("/transcribe", transcribe)
    return app


def main():
    from faster_whisper import WhisperModel
    from faster_whisper.vad import get_vad_model

    model = WhisperModel(
        os.environ.get("WHISPER_MODEL", "base.en"),
        device="cpu", compute_type="int8", cpu_threads=2,
    )
    get_vad_model()
    web.run_app(create_app(model), host="0.0.0.0", port=9000, access_log=None)


if __name__ == "__main__":
    main()
