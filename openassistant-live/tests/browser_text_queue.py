"""Browser queue/steering protocol checks; no agent or external actions."""

import asyncio
import json
from pathlib import Path

from browser_saved_conversations import FakeHistory
from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[1] / "openassistant/live/static"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True)
        page = await browser.new_page()
        errors, sockets, sent = [], [], []
        history = FakeHistory()
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.route(
            "**/static/*", lambda r: r.fulfill(path=str(ROOT / r.request.url.split("/static/")[-1]))
        )
        await page.route("**/client-config", lambda r: r.fulfill(json={"access_required": False}))
        await page.route(
            "http://localhost:8768/", lambda r: r.fulfill(path=str(ROOT / "index.html"))
        )

        def socket(ws):
            sockets.append(ws)

            def receive(raw):
                data = json.loads(raw)
                sent.append(data)
                if history.reply(ws, data):
                    return
                if "token" in data:
                    ws.send(json.dumps({"type": "ready", "name": "You"}))

            ws.on_message(receive)

        await page.route_web_socket("**/ws", socket)
        await page.goto("http://localhost:8768/")
        await expect(page.locator("#connection")).to_have_text("Connected")
        await page.locator("#toggle-panel").click()

        async def submit(text):
            await page.locator("#prompt").fill(text)
            await page.locator("#send").click()

        async def emit(data):
            sockets[0].send(json.dumps(data))
            await page.wait_for_timeout(50)

        await submit("First task")
        await emit({"type": "start", "turn": 1})
        await submit("A correction")
        await submit("Next task")
        assert len([d for d in sent if d.get("type") == "prompt"]) == 1
        await expect(page.locator("#text-queue li")).to_have_count(2)
        await page.locator("#text-queue li").first.get_by_text("Add now", exact=True).click()
        await page.wait_for_timeout(50)
        steer = next(d for d in sent if d.get("type") == "steer")
        await emit({"type": "steer_result", "id": steer["id"], "status": "accepted"})
        await expect(page.locator("#text-queue li")).to_have_count(1)
        assert not any(d.get("type") == "stop" for d in sent)
        await emit({"type": "task", "turn": 1, "state": "finished"})
        await emit({"type": "done", "turn": 1})
        assert len([d for d in sent if d.get("type") == "prompt"]) == 1
        await emit({"type": "turn_ready", "turn": 1})
        assert [d["text"] for d in sent if d.get("type") == "prompt"] == ["First task", "Next task"]
        await emit({"type": "start", "turn": 2})
        await submit("Keep for later")
        await page.locator("#interrupt").click()
        await emit({"type": "stopped", "turn": 2})
        await expect(page.locator("#text-queue")).to_contain_text("Paused")
        assert len([d for d in sent if d.get("type") == "prompt"]) == 2
        assert not errors, errors
        print("PASS: queue, explicit guidance, cleanup acknowledgement, FIFO and stop pauses queue")
        await browser.close()


asyncio.run(main())
