"""Audio notices clear on recovery without hiding unrelated errors."""

import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[1] / "openassistant/live/static"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True)
        page = await browser.new_page()
        await page.add_init_script("""window.Audio = class {
          constructor() { window.testAudio = this; this.paused = false; }
          async play() { this.paused = false; if (this.onplaying) this.onplaying(); }
          pause() { this.paused = true; }
        };""")
        await page.route(
            "**/static/*",
            lambda route: route.fulfill(path=str(ROOT / route.request.url.split("/static/")[-1])),
        )
        await page.route(
            "**/client-config", lambda route: route.fulfill(json={"access_required": False})
        )
        await page.route(
            "http://localhost:8768/", lambda route: route.fulfill(path=str(ROOT / "index.html"))
        )
        sockets = []

        def socket(ws):
            sockets.append(ws)

            def receive(raw):
                data = json.loads(raw)
                if "token" in data:
                    ws.send(json.dumps({"type": "ready", "name": "You"}))
                elif data.get("type") == "new_conversation":
                    ws.send(
                        json.dumps(
                            {
                                "type": "conversation",
                                "id": "example",
                                "messages": [],
                                "turn": 0,
                                "busy": False,
                                "observer": False,
                                "truncated": False,
                            }
                        )
                    )
                elif data.get("type") == "conversations":
                    ws.send(
                        json.dumps(
                            {
                                "type": "conversations",
                                "chats": [],
                                "next": None,
                                "offset": 0,
                                "query": "",
                            }
                        )
                    )

            ws.on_message(receive)

        await page.route_web_socket("**/ws", socket)
        await page.goto("http://localhost:8768/")
        await expect(page.locator("#sound")).to_be_enabled()
        sockets[0].send(json.dumps({"type": "audio", "data": "AA=="}))
        await page.wait_for_function("() => !!window.testAudio")
        await page.evaluate("() => window.testAudio.onerror()")
        await expect(page.locator("#notice")).to_have_text("Audio unavailable.")
        await page.locator("#sound").click()
        await page.locator("#sound").click()
        await expect(page.locator("#notice")).to_be_empty()
        sockets[0].send(json.dumps({"type": "notice", "text": "A different error"}))
        await expect(page.locator("#notice")).to_have_text("A different error")
        await page.locator("#sound").click()
        await page.locator("#sound").click()
        await expect(page.locator("#notice")).to_have_text("A different error")
        print("Audio notice recovery passed")
        await browser.close()


asyncio.run(main())
