"""Pi SDK adapter for the browser conversation runtime."""

import json
import logging
import uuid
from pathlib import Path
from threading import Event, Lock

from pi_agent import PiClient
from pi_agent.errors import PiError
from pi_agent.types import Limits

from .models import describe
from .plan import plan_steps

log = logging.getLogger(__name__)


class TurnRejected(ValueError):
    """Preparation failed before any prompt was submitted."""


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
        ready=None,
        activity=None,
        catalogue=None,
    ):
        self.ready = ready
        self.activity = activity or (lambda: None)
        self.catalogue = catalogue
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

    def run(self, thread_id, text, images=None):
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
            limits=Limits(cleanup_timeout=1),
            session_dir=str(directory),
            ui_handler=self.ui_handler,
            extra_args=[
                "--no-skills",
                "--extension",
                str(Path(__file__).with_name("extensions") / "plan.ts"),
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
                try:
                    # Capture Pi defaults before switching to a saved session, whose
                    # persisted settings may belong to a different model.
                    defaults = client.get_state()
                    default = defaults.get("model")
                    if not default:
                        raise TurnRejected("Configure a default model in Pi first.")
                    if path:
                        switched = client.switch_session(str(path))
                        if switched.get("cancelled"):
                            raise TurnRejected("An extension cancelled opening this session.")
                    selection = getattr(self.config, "turn_selection", None) or {}
                    selected = selection.get("selected")
                    provider, model = (
                        json.loads(selected) if selected else (default["provider"], default["id"])
                    )
                    client.set_model(provider, model)
                    levels = client.get_available_thinking_levels()
                    effort = selection.get("effort")
                    if effort is None:
                        effort = defaults.get("thinkingLevel", "off")
                        if effort not in levels:
                            effort = levels[0]
                    if effort not in levels:
                        raise TurnRejected(
                            "Saved thinking effort is unavailable. Reset it in the picker."
                        )
                    client.set_thinking_level(effort)
                    effective = client.get_state().get("model")
                    if images and (not effective or "image" not in effective.get("input", [])):
                        raise TurnRejected(
                            "This model cannot accept images. Choose an image model."
                        )
                except TurnRejected:
                    raise
                except Exception as error:
                    raise TurnRejected(
                        "Could not prepare this model. Retry or choose another model or Default."
                    ) from error
                self.activity()
                # Refresh using the already-running process. Failure is non-fatal;
                # no catalogue is needed to submit this turn.
                if self.catalogue:
                    try:
                        models = client.get_available_models(timeout=1)
                        options = describe(default, models, selected, levels, effort)
                        # Cache the inherited value, not this chat's explicit override.
                        options["effort"] = defaults.get("thinkingLevel", "off")
                        if options["effort"] not in levels:
                            options["effort"] = levels[0]
                        self.catalogue(options)
                    except Exception:
                        log.debug("Session catalogue refresh unavailable")
                if self.stopped.is_set():
                    raise RuntimeError("Task stopped")
                session_file = client.session.session_file
                if not session_file or Path(session_file).resolve().parent != directory:
                    raise RuntimeError("Pi did not create a session in the expected directory")
                if self.ready:
                    self.ready()
                self.progress({"activity": "Working"})
                item_id = str(uuid.uuid4())
                saved = False
                with client.stream(text, **({"images": images} if images else {})) as stream:
                    for event in stream:
                        self.activity()
                        if not saved and Path(session_file).is_file():
                            self.saved_thread(session_file)
                            saved = True
                        if self.stopped.is_set():
                            raise RuntimeError("Task stopped")
                        raw = getattr(event, "raw", {}) or {}
                        if (
                            self.message_event
                            and raw.get("type") == "tool_execution_end"
                            and raw.get("toolName") == "update_plan"
                            and not raw.get("isError")
                        ):
                            try:
                                steps = plan_steps(raw["result"]["details"]["steps"])
                            except (KeyError, TypeError, ValueError):
                                pass
                            else:
                                self.message_event("plan", {"steps": steps})
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
                client.abort(timeout=1)
            except PiError:
                pass
            finally:
                # Public SDK close escalates EOF -> terminate -> kill, reaps Pi,
                # and joins its reader loop. Cancelling an asyncio consumer cannot.
                client.close()

    def steer(self, text):
        with self.lock:
            client = self.client
        if not client or self.stopped.is_set():
            return False
        client.steer(text)
        return True
