"""Pi capability discovery, isolated from saved sessions and model turns."""

import asyncio
import json

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
            extra_args=["--no-tools", "--no-mcp", "--no-extensions", "--no-skills"],
        ) as client:
            state = await client.get_state()
            default = state.get("model")
            if not default:
                raise ValueError("Configure a default model in Pi first.")
            models = await client.get_available_models()
            catalogue = {model_key(model): model for model in models}
            key = selected or model_key(default)
            if key not in catalogue:
                raise ValueError("Selected model is unavailable. Choose another model or Default.")
            model = catalogue[key]
            await client.set_model(model["provider"], model["id"])
            levels = await client.get_available_thinking_levels()
            state = await client.get_state()
            return {
                "default": model_key(default),
                "models": [
                    {"id": k, "label": f"{m['name']} · {m['provider']}"}
                    for k, m in catalogue.items()
                ],
                "selected": selected,
                "effective": key,
                "provider": model["provider"],
                "model": model["id"],
                "levels": levels,
                "effort": state.get("thinkingLevel", "off"),
                "images": "image" in model.get("input", []),
            }
