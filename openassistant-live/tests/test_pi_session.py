from pathlib import Path
from types import SimpleNamespace

import pytest

from openassistant.pi_session import Session


class Stream:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        return iter([SimpleNamespace(text_delta="Hello")])

    def result(self):
        return SimpleNamespace(text="Hello")


class Client:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.session = SimpleNamespace(session_file=str(Path(kwargs["session_dir"]) / "chat.jsonl"))
        Path(self.session.session_file).write_text("")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get_state(self):
        return {
            "model": getattr(
                self, "model", {"provider": "local", "id": "default", "input": ["text", "image"]}
            ),
            "thinkingLevel": "off",
        }

    def set_model(self, provider, model):
        self.model = {"provider": provider, "id": model, "input": ["text", "image"]}

    def set_thinking_level(self, effort):
        pass

    def get_available_thinking_levels(self):
        return ["off", "high"]

    def switch_session(self, path):
        self.session.session_file = path
        return {"cancelled": False}

    def stream(self, text):
        return Stream()

    def abort(self):
        pass


def test_pi_stream_saves_session_and_enables_tools(tmp_path):
    config = SimpleNamespace(history_dir=tmp_path / "history", workspace=tmp_path, model=None)
    saved, events, clients = [], [], []

    def factory(**kwargs):
        client = Client(**kwargs)
        clients.append(client)
        return client

    session = Session(
        config,
        saved.append,
        lambda _: None,
        client_factory=factory,
        message_event=lambda kind, data: events.append((kind, data)),
    )
    assert session.run(None, "Hi") == "Hello"
    assert Path(saved[0]).parent == config.history_dir / "pi-sessions"
    assert "--no-tools" not in clients[0].options["extra_args"]
    assert "--no-mcp" not in clients[0].options["extra_args"]
    assert [kind for kind, _ in events] == ["delta", "item/completed"]
    assert events[0][1]["itemId"] == events[1][1]["itemId"]


def test_pi_rejects_session_outside_history(tmp_path):
    config = SimpleNamespace(history_dir=tmp_path / "history", workspace=tmp_path, model=None)
    foreign = tmp_path / "foreign.jsonl"
    foreign.write_text("")
    session = Session(config, lambda _: None, lambda _: None, client_factory=Client)
    with pytest.raises(ValueError, match="Unknown Pi session"):
        session.run(str(foreign), "Hi")


def test_tool_activity_does_not_expose_arguments(tmp_path):
    config = SimpleNamespace(history_dir=tmp_path / "history", workspace=tmp_path, model=None)
    events = []

    class ToolStream(Stream):
        def __iter__(self):
            return iter(
                [
                    SimpleNamespace(
                        text_delta=None,
                        raw={
                            "type": "tool_execution_start",
                            "toolCallId": "1",
                            "toolName": "read",
                            "args": {"secret": "private"},
                        },
                    ),
                    SimpleNamespace(
                        text_delta=None,
                        raw={
                            "type": "tool_execution_end",
                            "toolCallId": "1",
                            "toolName": "read",
                            "result": "private",
                            "isError": False,
                        },
                    ),
                ]
            )

    class ToolClient(Client):
        def stream(self, text):
            return ToolStream()

    session = Session(
        config,
        lambda _: None,
        lambda _: None,
        client_factory=ToolClient,
        message_event=lambda kind, data: events.append((kind, data)),
    )
    session.run(None, "Hi")
    assert [data["state"] for kind, data in events if kind == "activity"] == ["running", "returned"]
    assert "private" not in repr(events)


def test_resume_applies_effective_selection_and_native_images(tmp_path):
    calls = []

    class NativeClient(Client):
        def set_model(self, provider, model):
            calls.append(("model", provider, model))

        def set_thinking_level(self, effort):
            calls.append(("effort", effort))

        def stream(self, text, images=None):
            calls.append(("prompt", text, images))
            return Stream()

    config = SimpleNamespace(
        history_dir=tmp_path / "history",
        workspace=tmp_path,
        model=None,
        turn_selection={"selected": '["local","default"]', "effort": "off"},
    )
    folder = config.history_dir / "pi-sessions"
    folder.mkdir(parents=True)
    thread = folder / "chat.jsonl"
    thread.write_text("")
    images = [{"type": "image", "mimeType": "image/png", "data": "fixture"}]
    session = Session(config, lambda _: None, lambda _: None, client_factory=NativeClient)
    session.run(str(thread), "Look", images=images)
    assert calls == [("model", "local", "default"), ("effort", "off"), ("prompt", "Look", images)]


def test_plan_tool_results_are_validated(tmp_path):
    class PlanStream(Stream):
        def __iter__(self):
            plans = [
                [{"step": "Working", "status": "in_progress"}],
                [{"step": "Invalid", "status": "finished"}],
            ]
            return iter(
                SimpleNamespace(
                    text_delta=None,
                    raw={
                        "type": "tool_execution_end",
                        "toolName": "update_plan",
                        "isError": False,
                        "result": {"details": {"steps": steps}},
                    },
                )
                for steps in plans
            )

    class PlanClient(Client):
        def stream(self, text):
            return PlanStream()

    events = []
    config = SimpleNamespace(history_dir=tmp_path, workspace=tmp_path, model=None)
    Session(
        config,
        lambda _: None,
        lambda _: None,
        client_factory=PlanClient,
        message_event=lambda kind, value: events.append((kind, value)),
    ).run(None, "Work")
    assert [value for kind, value in events if kind == "plan"] == [
        {"steps": [{"step": "Working", "status": "in_progress"}]}
    ]
