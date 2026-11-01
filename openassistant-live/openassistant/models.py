"""Pi capability discovery, isolated from saved sessions and model turns."""

import asyncio
import json
import time

from pi_agent import AsyncPiClient


def model_key(model):
    return json.dumps([model["provider"], model["id"]], separators=(",", ":"))


async def model_options(config, selected=None, client_factory=AsyncPiClient):
    """Resolve the app default and selected model without touching chat history."""
    async with asyncio.timeout(20):
        async with client_factory(
            executable=getattr(config, "pi_executable", "pi"),
            cwd=str(config.workspace),
            provider=getattr(config, "pi_provider", None),
            model=getattr(config, "model", None),
            no_session=True,
            extra_args=["--no-tools", "--no-mcp", "--no-skills"],
        ) as client:
            state = await client.get_state()
            default = state.get("model")
            if not default:
                raise ValueError("Configure a default model in Pi first.")
            models = await client.get_available_models()
            catalogue = {model_key(model): model for model in models}
            key = selected or model_key(default)
            if key not in catalogue:
                return describe(default, models, selected, [], None, unavailable=True)
            model = catalogue[key]
            await client.set_model(model["provider"], model["id"])
            levels = await client.get_available_thinking_levels()
            state = await client.get_state()
            return describe(default, models, selected, levels, state.get("thinkingLevel", "off"))


def describe(default, models, selected, levels, effort, unavailable=False):
    key = selected or model_key(default)
    model = next((m for m in models if model_key(m) == key), {})
    return {
        "default": model_key(default),
        "models": [{"id": model_key(m), "label": f"{m['name']} · {m['provider']}"} for m in models],
        "selected": selected,
        "effective": key,
        "provider": model.get("provider"),
        "model": model.get("id"),
        "levels": levels,
        "effort": effort,
        "images": "image" in model.get("input", []),
        "unavailable": unavailable,
    }


class ModelCatalogue:
    """Picker-only cache. Turn execution never waits for discovery."""

    def __init__(self, config, discover=model_options, ttl=300):
        self.config, self.discover, self.ttl = config, discover, ttl
        self.entries = {}
        self.lock = asyncio.Lock()

    def offer(self, options):
        self.entries[options["selected"]] = (time.monotonic(), dict(options))
        if len(self.entries) > 128:
            del self.entries[min(self.entries, key=lambda key: self.entries[key][0])]

    async def options(self, selected, refresh=False):
        async with self.lock:
            entry = self.entries.get(selected)
            if not refresh and entry and time.monotonic() - entry[0] < self.ttl:
                return dict(entry[1])
            options = await self.discover(self.config, selected)
            self.offer(options)
            return dict(options)
