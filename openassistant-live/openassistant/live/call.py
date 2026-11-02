"""Private, single-person voice calls using the existing OpenAssistant session engine."""

import asyncio
import base64
import contextlib
import copy
import logging
import re
import tempfile
import time
import uuid
from pathlib import Path
from threading import Lock

from aiohttp.client_exceptions import ClientError

from ..pi_session import Session, TurnRejected
from ..speech import Speech
from .images import capture_images, retain_uploads

log = logging.getLogger(__name__)


class InactivityTimeout(TimeoutError):
    """The turn watchdog observed no Pi progress within its configured limit."""


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
        self.pending_changed = None
        self.turn_selection = None
        self.attachments = []
        self.uploads = []
        self.catalogue = None
        self.last_activity = 0
        self.accepted = False

    def remember(self, kind, data):
        if kind == "start":
            self.snapshot = {
                "state": "working",
                "detail": "Working",
                "started": time.time(),
                "request": "",
                "responses": [],
                "activities": [],
                "steps": [],
            }
            if self.history and self.conversation:
                self.history.save_plan(self.conversation, [])
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
        elif kind == "plan":
            self.snapshot["steps"] = data["steps"]
            if self.history and self.conversation:
                self.history.save_plan(self.conversation, data["steps"])
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
                "plan",
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
            await self.refresh_pending()
            timeout = request.get("timeout")
            seconds = (
                min(120, max(0.1, timeout / 1000)) if isinstance(timeout, (int, float)) else 120
            )
            async with asyncio.timeout(seconds):
                return await future
        except TimeoutError:
            return None
        finally:
            self.last_activity = asyncio.get_running_loop().time()
            self.pending_ui.pop(token, None)
            self.snapshot["pending_ui"] = [entry[1] for entry in self.pending_ui.values()]
            await self.send("input_closed", id=token)
            await self.refresh_pending()

    async def refresh_pending(self):
        if self.pending_changed:
            await self.pending_changed()
        if self.snapshot.get("state") not in {"finished", "failed", "stopped", "interrupted"}:
            state = (
                "approval"
                if any(item["method"] == "confirm" for _, item in self.pending_ui.values())
                else "input"
                if self.pending_ui
                else "working"
            )
            await self.send(
                "task",
                state=state,
                text={
                    "approval": "Needs approval",
                    "input": "Needs your answer",
                    "working": "Working",
                }[state],
            )

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
        worker = speaker = watchdog = None
        active = True
        started = loop.time()
        self.last_activity = started
        self.accepted = False
        first_audio = True
        stage = "starting"
        bridge_lock = Lock()
        bridge_waiters = set()

        def bridge(coro):
            with bridge_lock:
                if not active:
                    coro.close()
                    raise RuntimeError("Call ended.")
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                bridge_waiters.add(future)
            try:
                return future.result(timeout=130)
            except BaseException:
                future.cancel()
                raise
            finally:
                with bridge_lock:
                    bridge_waiters.discard(future)

        async def accept():
            if self.uploads:
                self.attachments = await asyncio.to_thread(
                    retain_uploads, self.history, self.conversation, self.uploads
                )
            else:
                self.attachments = []
            self.accepted = True
            await self.send("start")
            await self.send("user", text=prompt, images=self.attachments)
            await self.send("task", state="working", text="Working on your request")

        def activity():
            loop.call_soon_threadsafe(touch)

        def touch():
            self.last_activity = loop.time()

        async def watch():
            seconds = getattr(self.config, "inactivity_timeout", 1800)
            if not seconds:
                return
            while True:
                await asyncio.sleep(min(seconds, 1))
                if self.pending_ui:
                    self.last_activity = loop.time()
                elif loop.time() - self.last_activity >= seconds:
                    await events.put(("failed", InactivityTimeout("Pi inactivity timeout")))
                    return

        async def speak():
            nonlocal first_audio
            while (text := await voice.get()) is not None:
                if not text.strip():
                    continue
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
            async with asyncio.timeout(None):
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
                stage = "agent"
                turn_config = copy.copy(self.config)
                turn_config.turn_selection = self.turn_selection
                self.session = Session(
                    turn_config,
                    lambda thread: self.saved_thread(thread, prompt),
                    lambda value: bridge(events.put(("progress", value))),
                    message_event=lambda kind, value: bridge(events.put((kind, value))),
                    ui_handler=lambda request: bridge(self.request_ui(request)),
                    ready=lambda: bridge(accept()),
                    activity=activity,
                    catalogue=(
                        lambda options: loop.call_soon_threadsafe(self.catalogue.offer, options)
                    )
                    if self.catalogue
                    else None,
                    extra_instructions=(
                        "\nThis turn is in the OpenAssistant voice app. "
                        "Replies appear here and are spoken automatically. Use short natural "
                        "sentences; keep code and links in captions. "
                        "Do not claim to hear continuous audio."
                        " To show a local image in this app, include it in your final reply "
                        "as Markdown image syntax with an absolute path, such as "
                        "![Description](/tmp/example.png). Use PNG, JPEG, GIF or WebP."
                        " For work with multiple steps, use update_plan with a steps array "
                        "to publish and maintain a concise plan. Skip plans for simple replies."
                    ),
                )

                async def generate():
                    try:
                        images = []
                        for data, mime in self.uploads:
                            images.append(
                                {
                                    "type": "image",
                                    "mimeType": mime,
                                    "data": base64.b64encode(data).decode("ascii"),
                                }
                            )
                        result = await asyncio.to_thread(
                            self.session.run,
                            self.thread,
                            prompt,
                            **({"images": images} if images else {}),
                        )
                        if active:
                            await events.put(("result", result))
                    except Exception as error:
                        if active:
                            await events.put(("failed", error))

                worker = asyncio.create_task(generate())
                speaker = asyncio.create_task(speak())
                watchdog = asyncio.create_task(watch())
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
                    if kind in ("activity", "plan"):
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
            if not self.accepted:
                if stage == "transcription":
                    message = (
                        "Could not transcribe this recording. Try recording again or send text."
                    )
                elif isinstance(error, TurnRejected):
                    message = str(error)
                else:
                    message = "Could not prepare this message. Check the model and retry."
                await self.send("upload_error", text=f"{message} Reference {reference}.")
                return
            # A disconnected browser cannot receive an error notification.
            with contextlib.suppress(ClientError, ConnectionError):
                await self.send(
                    "task",
                    state="failed",
                    text=(
                        "Task stopped after no Pi progress. "
                        "Adjust --inactivity-timeout for silent tools."
                        if isinstance(error, InactivityTimeout)
                        else f"Task stopped. Reference {reference}."
                    ),
                )
                await self.send(
                    "notice",
                    text=f"Task stopped (reference {reference}). "
                    "The call is still open. Check completed actions before asking me to continue.",
                )
                await self.send("done")
        finally:
            with bridge_lock:
                active = False
                for future in tuple(bridge_waiters):
                    future.cancel()
            # Unblock a producer already waiting to enqueue its terminal result.
            while not events.empty():
                events.get_nowait()
            for future, _ in list(self.pending_ui.values()):
                if not future.done():
                    future.set_result(None)
            if self.session:
                await asyncio.to_thread(self.session.stop)
            # Session.close has reaped the child. Join the actual run worker
            # before releasing capacity; cancelling to_thread would leave it alive.
            if worker:
                await worker
            for task in (speaker, watchdog):
                if task:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            self.session = None
            self.uploads = []
