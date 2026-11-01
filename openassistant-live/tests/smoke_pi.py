"""Opt-in real Pi smoke test. Uses the caller's configured model, no speech services.

Run with PI_CODING_AGENT_DIR pointing at a dedicated test configuration. This
generates two replies and calls only the bundled update_plan tool as requested.
"""

import asyncio
import base64
import struct
import tempfile
import zlib
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from aiohttp.test_utils import TestClient, TestServer

from openassistant.live import app as live
from openassistant.live.call import Speech
from openassistant.live.history import Conversations


def red_png():
    def chunk(kind, data):
        return (
            struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!2I5B", 64, 64, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\x00" + b"\xff\x00\x00" * 64) * 64))
        + chunk(b"IEND", b"")
    )


class NoSpeech:
    def warm_in_background(self):
        pass

    async def close(self):
        pass


async def no_audio(*args):
    raise OSError("Speech deliberately disabled in Pi smoke test")


async def event(ws, kind):
    async with asyncio.timeout(120):
        while True:
            data = await ws.receive_json()
            if data["type"] == "task" and data["state"] == "failed":
                raise AssertionError(data["text"])
            if data["type"] in {"upload_error", "model_error"}:
                raise AssertionError(data["text"])
            if data["type"] == kind:
                return data


async def main():
    live.LiveTranscriber = NoSpeech
    Speech.fetch = no_audio

    # Title generation is independently covered; avoid incidental extra model calls.
    async def no_title(*args):
        return None

    live.generate_title = no_title
    # Keep synthetic evidence available for inspection after the smoke run.
    with tempfile.TemporaryDirectory(prefix="openassistant-smoke-", delete=False) as directory:
        root = Path(directory)
        config = SimpleNamespace(
            origin="http://localhost",
            access={},
            tailnet_only=True,
            workspace=root,
            history_dir=root / "history",
            model=None,
        )
        async with TestClient(TestServer(live.create_app(config))) as client:
            ws = await client.ws_connect("/ws", headers={"Origin": config.origin})
            await ws.send_json({"token": "", "client_id": str(uuid4())})
            await event(ws, "ready")
            await ws.send_json({"type": "new_conversation"})
            chat = (await event(ws, "conversation"))["id"]
            await ws.send_json({"type": "model_list", "conversation": chat})
            options = await event(ws, "model_options")
            assert options["images"], "Use an image-capable test model"
            await ws.send_json(
                {"type": "model_select", "conversation": chat, "model": options["effective"]}
            )
            await event(ws, "model_options")
            await ws.send_json(
                {
                    "type": "prompt",
                    "text": 'Use update_plan with one completed step named "Verify image". '
                    "Then state the image colour. Use no other tools.",
                    "images": [base64.b64encode(red_png()).decode()],
                }
            )
            plan = await event(ws, "plan")
            assert plan["steps"][0]["status"] == "completed"
            reply = await event(ws, "result")
            assert "red" in reply["text"].lower(), reply["text"]
            await event(ws, "turn_ready")
            history = Conversations(config.history_dir, config.workspace)
            thread = history.get(chat)["thread"]
            transcript = history.transcript(chat)
            assert transcript["messages"][0]["images"]
            # Reset Default and resume the same native Pi session.
            await ws.send_json({"type": "model_select", "conversation": chat, "model": None})
            await event(ws, "model_options")
            await ws.send_json(
                {
                    "type": "prompt",
                    "text": "What colour was the image in my last message? "
                    "Reply with the colour only. No tools.",
                }
            )
            assert "red" in (await event(ws, "result"))["text"].lower()
            await event(ws, "turn_ready")
            assert history.get(chat)["thread"] == thread
            await ws.close()
    print(
        "PASS real Pi: discovery, selection, native image, plan tool, history, Default and resume"
    )


if __name__ == "__main__":
    asyncio.run(main())
