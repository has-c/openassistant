"""Unit tests never consult the host's Pi model configuration."""

import pytest


@pytest.fixture(autouse=True)
def isolated_models(monkeypatch):
    from openassistant.live import app

    async def options(config, selected=None):
        return {
            "default": '["test","text"]',
            "effective": selected or '["test","text"]',
            "selected": selected,
            "models": [{"id": '["test","text"]', "label": "Test"}],
            "provider": "test",
            "model": "text",
            "levels": ["off", "high"],
            "effort": "off",
            "images": False,
        }

    monkeypatch.setattr(app, "model_options", options)
