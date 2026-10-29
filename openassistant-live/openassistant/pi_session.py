"""Pi SDK adapter for the browser conversation runtime."""

import uuid
from pathlib import Path
from threading import Event, Lock

from pi_agent import PiClient
from pi_agent.errors import PiError


class Session:
    def __init__(
        self,
        config,
        saved_thread,
        progress,
        client_factory=PiClient,
        message_event=None,
        extra_instructions="",
        ui_handler=None,
    ):
        self.config = config
        self.saved_thread = saved_thread
        self.progress = progress
        self.message_event = message_event
        self.extra_instructions = extra_instructions
        self.ui_handler = ui_handler
        self.client_factory = client_factory
        self.client = None
        self.lock = Lock()
        self.stopped = Event()

    def run(self, thread_id, text):
        if not isinstance(text, str):
            raise ValueError("Pi currently accepts text prompts only")
        directory = Path(self.config.history_dir).resolve() / "pi-sessions"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if thread_id:
            path = Path(thread_id).resolve()
            if path.parent != directory or path.suffix != ".jsonl" or not path.is_file():
                raise ValueError("Unknown Pi session")
        else:
            path = None
        client = self.client_factory(
            executable=getattr(self.config, "pi_executable", "pi"),
            cwd=str(self.config.workspace),
            provider=getattr(self.config, "pi_provider", None),
            model=getattr(self.config, "model", None),
            session=str(path) if path else None,
            session_dir=str(directory),
            ui_handler=self.ui_handler,
            extra_args=[
                "--no-skills",
                "--append-system-prompt",
                self.extra_instructions or "You are OpenAssistant.",
            ],
        )
        with self.lock:
            if self.stopped.is_set():
                raise RuntimeError("Task stopped")
            self.client = client
        try:
            with client:
                if self.stopped.is_set():
                    raise RuntimeError("Task stopped")
                session_file = client.session.session_file
                if not session_file or Path(session_file).resolve().parent != directory:
                    raise RuntimeError("Pi did not create a session in the expected directory")
                self.progress({"activity": "Working"})
                item_id = str(uuid.uuid4())
                saved = False
                with client.stream(text) as stream:
                    for event in stream:
                        if not saved and Path(session_file).is_file():
                            self.saved_thread(session_file)
                            saved = True
                        if self.stopped.is_set():
                            raise RuntimeError("Task stopped")
                        raw = getattr(event, "raw", {}) or {}
                        if self.message_event and raw.get("type") in (
                            "tool_execution_start",
                            "tool_execution_end",
                        ):
                            self.message_event(
                                "activity",
                                {
                                    "id": str(raw.get("toolCallId", ""))[:200],
                                    "label": str(raw.get("toolName", "Tool"))[:120],
                                    "state": (
                                        "running"
                                        if raw["type"] == "tool_execution_start"
                                        else "failed"
                                        if raw.get("isError")
                                        else "returned"
                                    ),
                                },
                            )
                        if event.text_delta and self.message_event:
                            self.message_event(
                                "delta", {"itemId": item_id, "delta": event.text_delta}
                            )
                    result = stream.result()
                if not saved:
                    if not Path(session_file).is_file():
                        raise RuntimeError("Pi session was not persisted")
                    self.saved_thread(session_file)
                if self.message_event:
                    self.message_event("item/completed", {"itemId": item_id, "text": result.text})
                return result.text
        finally:
            with self.lock:
                self.client = None

    def stop(self):
        self.stopped.set()
        with self.lock:
            client = self.client
        if client:
            try:
                client.abort()
            except PiError:
                pass

    def steer(self, text):
        with self.lock:
            client = self.client
        if not client or self.stopped.is_set():
            return False
        client.steer(text)
        return True
