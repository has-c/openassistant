"""Private, single-person voice calls using the existing OpenAssistant session engine."""

import asyncio
import base64
import contextlib
import logging
import re
import tempfile
import time
import uuid
from pathlib import Path

from aiohttp.client_exceptions import ClientError

from ..pi_session import Session
from ..speech import Speech
from .images import capture_images

log = logging.getLogger(__name__)


class Call:
    def __init__(self, ws, config, transcriber=None):
        self.ws, self.config = ws, config
        self.transcriber = transcriber
        self.thread = None
        self.session = None
        self.task = None
        self.serial = 0
        self.conversation = None
        self.history = None
        self.peers = None
        self.owner = None
        self.principal = None
        self.snapshot = {}
        self.pending_ui = {}

    def remember(self, kind, data):
        if kind == "start":
            self.snapshot = {
                "state": "working",
                "detail": "Working",
                "started": time.time(),
                "request": "",
                "responses": [],
                "activities": [],
            }
            if self.history and self.conversation:
                self.history.update_task(
                    self.conversation, "working", "Working", self.snapshot["started"]
                )
        elif kind == "task":
            self.snapshot.update(state=data["state"], detail=data.get("text", ""))
            if self.history and self.conversation:
                self.history.update_task(
                    self.conversation,
                    data["state"],
                    data.get("text", ""),
                    self.snapshot.get("started", time.time()),
                )
        elif kind == "user":
            self.snapshot["request"] = data.get("text", "")
        elif kind == "activity":
            rows = self.snapshot.setdefault("activities", [])
            rows[:] = [r for r in rows if not data.get("id") or r.get("id") != data["id"]]
            rows.append(data)
            del rows[:-20]
        elif kind in ("delta", "message", "result"):
            rows = self.snapshot.setdefault("responses", [])
            item = data.get("item", "result")
            if kind == "result" and rows:
                return
            row = next((r for r in rows if r["item"] == item), None)
            if row is None:
                row = {"item": item, "text": ""}
                rows.append(row)
            row["text"] = ((row["text"] if kind == "delta" else "") + data.get("text", ""))[-32000:]
            if data.get("images"):
                row["images"] = data["images"]
            del rows[:-20]

    async def send(self, kind, **data):
        if kind in ("message", "result") and data.get("text"):
            item = data.get("item", "result")
            images = await asyncio.to_thread(
                capture_images, self.history, self.conversation, item, data["text"]
            )
            if images:
                data["images"] = images
        self.remember(kind, data)
        payload = {"type": kind, "turn": self.serial, "conversation": self.conversation, **data}
        if self.ws and not getattr(self.ws, "closed", False):
            try:
                async with asyncio.timeout(10):
                    await self.ws.send_json(payload)
            except (ClientError, ConnectionError, TimeoutError):
                pass
        await self.broadcast(kind, **data)
        if kind in ("start", "task") and self.peers and self.conversation:

            async def status(peer):
                if peer.ws and not peer.ws.closed:
                    async with asyncio.timeout(2):
                        await peer.ws.send_json(
                            {
                                "type": "chat_status",
                                "id": self.conversation,
                                "task": {
                                    key: self.snapshot.get(key)
                                    for key in ("state", "detail", "started")
                                },
                            }
                        )

            await asyncio.gather(
                *(status(peer) for peer in tuple(self.peers)), return_exceptions=True
            )

    async def broadcast(self, kind, **data):
        payload = {"type": kind, "turn": self.serial, "conversation": self.conversation, **data}
        if (
            self.peers
            and self.conversation
            and kind
            in {
                "start",
                "user",
                "task",
                "delta",
                "message",
                "result",
                "activity",
                "done",
                "stopped",
                "turn_ready",
            }
        ):

            async def forward(peer):
                if peer is not self and peer.conversation == self.conversation:
                    async with asyncio.timeout(2):
                        await peer.ws.send_json({**payload, "observer": True})

            await asyncio.gather(
                *(forward(peer) for peer in tuple(self.peers)), return_exceptions=True
            )

    async def request_ui(self, request):
        method = request.get("method")
        if method not in {"confirm", "input", "select", "editor"}:
            return None
        if len(self.pending_ui) >= 4:
            return None
        token = uuid.uuid4().hex
        options = request.get("options", [])
        if method == "select" and (
            not isinstance(options, list)
            or len(options) > 50
            or any(not isinstance(x, str) or len(x) > 500 for x in options)
        ):
            return None
        item = {
            "id": token,
            "method": method,
            "title": str(request.get("title", "Input requested"))[:500],
            "message": str(request.get("message", ""))[:4000],
            "options": options if method == "select" else [],
            "placeholder": str(request.get("placeholder", ""))[:500],
            "prefill": str(request.get("prefill", ""))[:8000],
        }
        future = asyncio.get_running_loop().create_future()
        self.pending_ui[token] = (future, item)
        self.snapshot["pending_ui"] = [entry[1] for entry in self.pending_ui.values()]
        try:
            await self.send("input_request", request=item)
            timeout = request.get("timeout")
            seconds = (
                min(120, max(0.1, timeout / 1000)) if isinstance(timeout, (int, float)) else 120
            )
            async with asyncio.timeout(seconds):
                return await future
        except TimeoutError:
            return None
        finally:
            self.pending_ui.pop(token, None)
            self.snapshot["pending_ui"] = [entry[1] for entry in self.pending_ui.values()]
            await self.send("input_closed", id=token)

    def answer_ui(self, token, value):
        entry = self.pending_ui.get(token) if isinstance(token, str) else None
        if not entry or entry[0].done():
            return False
        future, item = entry
        method = item["method"]
        if value is not None:
            if method == "confirm" and not isinstance(value, bool):
                return False
            if method in {"input", "editor", "select"} and (
                not isinstance(value, str) or len(value) > 8000
            ):
                return False
            if method == "select" and value not in item["options"]:
                return False
        future.set_result(value)
        return True

    def saved_thread(self, thread, prompt):
        self.thread = thread
        if self.history and self.conversation:
            self.history.saved(self.conversation, thread, prompt)

    async def stop(self):
        for future, _ in list(self.pending_ui.values()):
            if not future.done():
                future.set_result(None)
        if self.task and not self.task.done():
            # Cancel the consumer before its finally block closes the agent.
            # Otherwise the reader can report the expected shutdown as a failure.
            # run() owns session cleanup; do not stop the same session twice.
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        elif self.session:
            await asyncio.to_thread(self.session.stop)
        self.session = self.task = None

    async def run(self, prompt=None, recording=None):
        loop = asyncio.get_running_loop()
        events = asyncio.Queue(maxsize=128)
        voice = asyncio.Queue(maxsize=16)
        worker = speaker = None
        active = True
        started = loop.time()
        first_audio = True
        stage = "starting"

        def bridge(coro):
            if not active:
                coro.close()
                raise RuntimeError("Call ended.")
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            try:
                return future.result(timeout=130)
            except BaseException:
                future.cancel()
                raise

        async def speak():
            nonlocal first_audio
            while (text := await voice.get()) is not None:
                if not self.ws or self.ws.closed:
                    continue
                listener = self.ws
                try:
                    with tempfile.TemporaryDirectory(prefix="openassistant-voice-") as folder:
                        path = Path(folder) / "reply.ogg"
                        await Speech(self.config).fetch(path, text)
                        if self.ws is not listener or listener.closed:
                            continue
                        await self.send("audio", data=base64.b64encode(path.read_bytes()).decode())
                        if first_audio:
                            first_audio = False
                            await self.send(
                                "timing",
                                stage="first_audio",
                                milliseconds=round((loop.time() - started) * 1000),
                            )
                except (TimeoutError, OSError, RuntimeError, ClientError):
                    log.warning("Speech synthesis unavailable")

        try:
            async with asyncio.timeout(900):
                if recording is not None:
                    stage = "transcription"
                    await self.send("status", text="Transcribing")
                    with tempfile.TemporaryDirectory(prefix="openassistant-mic-") as folder:
                        path = Path(folder) / "recording.webm"
                        path.write_bytes(recording)
                        if not self.transcriber:
                            raise RuntimeError("Speech transcription is not configured")
                        prompt = await self.transcriber.transcribe(path)
                    await self.send(
                        "timing",
                        stage="transcription",
                        milliseconds=round((loop.time() - started) * 1000),
                    )
                await self.send("user", text=prompt)
                await self.send("task", state="working", text="Working on your request")
                stage = "agent"
                self.session = Session(
                    self.config,
                    lambda thread: self.saved_thread(thread, prompt),
                    lambda value: bridge(events.put(("progress", value))),
                    message_event=lambda kind, value: bridge(events.put((kind, value))),
                    ui_handler=lambda request: bridge(self.request_ui(request)),
                    extra_instructions=(
                        "\nThis turn is in the OpenAssistant voice app. "
                        "Replies appear here and are spoken automatically. Use short natural "
                        "sentences; keep code and links in captions. "
                        "Do not claim to hear continuous audio."
                        " To show a local image in this app, include it in your final reply "
                        "as Markdown image syntax with an absolute path, such as "
                        "![Description](/tmp/example.png). Use PNG, JPEG, GIF or WebP."
                    ),
                )

                async def generate():
                    try:
                        result = await asyncio.to_thread(self.session.run, self.thread, prompt)
                        await events.put(("result", result))
                    except Exception as error:
                        await events.put(("failed", error))

                worker = asyncio.create_task(generate())
                speaker = asyncio.create_task(speak())
                buffers, size = {}, 0
                while True:
                    kind, value = await events.get()
                    if kind == "failed":
                        raise value
                    if kind == "result":
                        await self.send("task", state="finished", text="Agent turn finished")
                        await self.send("result", text=value)
                        break
                    if kind == "progress":
                        if isinstance(value, dict):
                            await self.send("status", text=value.get("activity", "Working"))
                        elif isinstance(value, str):
                            await self.send("activity", label=value[:240], state="update")
                        continue
                    if kind == "activity":
                        await self.send(kind, **value)
                        continue
                    item = value.get("itemId") or value.get("id")
                    if kind == "delta":
                        delta = value.get("delta", "")
                        size += len(delta)
                        if size > 32000:
                            raise RuntimeError("Reply limit reached")
                        await self.send("delta", item=item, text=delta)
                        buffers[item] = buffers.get(item, "") + delta
                        while match := re.search(r"[.!?]\s|\n|.{450}\s", buffers[item]):
                            end = match.end()
                            await voice.put(buffers[item][:end])
                            buffers[item] = buffers[item][end:]
                    elif kind == "item/completed":
                        await self.send("message", item=item, text=value.get("text", ""))
                        remaining = buffers.pop(item, value.get("text", ""))
                        for start in range(0, len(remaining), 800):
                            await voice.put(remaining[start : start + 800])
                await voice.put(None)
                await speaker
                await self.send("done")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            reference = uuid.uuid4().hex[:10]
            log.warning(
                "Live turn failed ref=%s type=%s stage=%s",
                reference,
                type(error).__name__,
                stage,
            )
            # A disconnected browser cannot receive an error notification.
            with contextlib.suppress(ClientError, ConnectionError):
                await self.send(
                    "task", state="failed", text=f"Task stopped. Reference {reference}."
                )
                await self.send(
                    "notice",
                    text=f"Task stopped (reference {reference}). "
                    "The call is still open. Check completed actions before asking me to continue.",
                )
                await self.send("done")
        finally:
            active = False
            if self.session:
                await asyncio.to_thread(self.session.stop)
            for task in (worker, speaker):
                if task:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            self.session = None
