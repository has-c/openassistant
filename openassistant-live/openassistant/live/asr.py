"""One bounded, reusable local Whisper worker for live calls."""

import asyncio
import json
import logging
import sys

import aiohttp

log = logging.getLogger(__name__)


class RemoteTranscriber:
    """Send an audio file to a separately managed Whisper service."""

    def __init__(self, endpoint):
        self.endpoint = endpoint

    def warm_in_background(self):
        # The service owns its model lifecycle.
        pass

    async def close(self):
        pass

    async def transcribe(self, path):
        timeout = aiohttp.ClientTimeout(total=90)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            with open(path, "rb") as audio:
                async with session.post(
                    self.endpoint, data=audio, headers={"Content-Type": "application/octet-stream"}
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
        if not result.get("text"):
            raise RuntimeError("No speech recognised")
        return result["text"]


class LiveTranscriber:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.process = None
        self.warmup_task = None
        self.closed = False

    async def close(self):
        self.closed = True
        if self.warmup_task and not self.warmup_task.done():
            self.warmup_task.cancel()
            await asyncio.gather(self.warmup_task, return_exceptions=True)
        await self._kill()

    async def _kill(self):
        process, self.process = self.process, None
        if process and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()

    def warm_in_background(self):
        if self.closed or (self.warmup_task and not self.warmup_task.done()):
            return
        self.warmup_task = asyncio.create_task(self._warm())

    async def _warm(self):
        try:
            async with asyncio.timeout(90):
                async with self.lock:
                    try:
                        await self._start()
                    except BaseException:
                        await self._kill()
                        raise
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning("Whisper warmup failed; the next recording will retry")

    async def _start(self):
        if self.process is not None and self.process.returncode is None:
            return
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "openassistant.live.asr",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=65536,
        )
        ready = json.loads(await self.process.stdout.readline())
        if ready != {"ready": True}:
            raise RuntimeError("Whisper worker failed to start")

    async def transcribe(self, path):
        async with asyncio.timeout(90):
            async with self.lock:
                try:
                    await self._start()
                    self.process.stdin.write((json.dumps(str(path)) + "\n").encode())
                    await self.process.stdin.drain()
                    result = json.loads(await self.process.stdout.readline())
                except BaseException:
                    # A cancelled request must not leave a response for the next caller.
                    await self._kill()
                    self.warm_in_background()
                    raise
                if not result.get("text"):
                    raise RuntimeError("No speech recognised")
                return result["text"]


def worker():
    from faster_whisper import WhisperModel
    from faster_whisper.vad import get_vad_model

    model = WhisperModel(
        "base.en", device="cpu", compute_type="int8", cpu_threads=2, local_files_only=True
    )
    get_vad_model()
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            segments, info = model.transcribe(
                json.loads(line),
                language="en",
                vad_filter=True,
                initial_prompt="OpenAssistant.",
            )
            if info.duration > 65:
                raise ValueError("Utterance exceeds live-call limit")
            text = " ".join(segment.text.strip() for segment in segments)
            if len(text) > 8000:
                raise ValueError("Transcript exceeds live-call limit")
            print(json.dumps({"text": text}), flush=True)
        except Exception:
            print(json.dumps({"error": "transcription_failed"}), flush=True)


if __name__ == "__main__":
    worker()
