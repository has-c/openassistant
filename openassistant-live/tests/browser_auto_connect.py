import asyncio
import json
from pathlib import Path

from browser_saved_conversations import FakeHistory
from playwright.async_api import async_playwright

root = Path(__file__).resolve().parents[1] / "openassistant/live/static"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True)
        page = await browser.new_page()
        errors = []
        sockets = []
        history = FakeHistory()
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.add_init_script(
            """window.tones=0;
            const original=AudioContext.prototype.createOscillator;
            AudioContext.prototype.createOscillator=function(){
              window.tones++; return original.call(this);
            };"""
        )
        await page.route(
            "**/static/*", lambda r: r.fulfill(path=str(root / r.request.url.split("/static/")[-1]))
        )
        await page.route("**/client-config", lambda r: r.fulfill(json={"access_required": False}))
        await page.route(
            "http://localhost:8768/", lambda r: r.fulfill(path=str(root / "index.html"))
        )

        def socket(ws):
            sockets.append(ws)

            def receive(raw):
                data = json.loads(raw)
                if not history.reply(ws, data) and "token" in data:
                    ws.send(json.dumps({"type": "ready", "name": "You"}))

            ws.on_message(receive)

        await page.route_web_socket("**/ws", socket)
        await page.goto("http://localhost:8768/")
        await page.wait_for_function("document.body.classList.contains('connected')")
        assert not await page.locator("#login").is_visible()
        assert await page.locator("#mic-label").inner_text() == "Mic off"
        await page.locator("#toggle-panel").click()
        await page.wait_for_function("document.body.classList.contains('connected')")
        await sockets[0].close(code=1011, reason="test disconnect")
        await page.wait_for_function("document.body.classList.contains('connected')")
        for _ in range(40):
            if len(sockets) == 2:
                break
            await asyncio.sleep(0.1)
        assert len(sockets) == 2
        await page.locator("#leave").click()
        await page.wait_for_timeout(1500)
        assert len(sockets) == 2, "Explicit disconnect reconnected"
        assert await page.locator("#login").is_visible()
        await page.locator("#join").click()
        await page.wait_for_function("document.body.classList.contains('connected')")
        assert len(sockets) == 3
        await page.route("**/client-config", lambda r: r.fulfill(json={"access_required": True}))
        await page.reload()
        await page.locator("#join").wait_for(state="visible")
        await page.wait_for_timeout(200)
        assert len(sockets) == 3, "Protected mode must wait for credentials"
        await page.locator("#token").fill("test-key")
        await page.locator("#join").click()
        await page.wait_for_function("document.body.classList.contains('connected')")
        assert len(sockets) == 4
        assert not errors, errors
        print(
            "PASS: auto-connect, mic off, reconnect, explicit disconnect, "
            "manual resume and protected-mode authentication"
        )
        await browser.close()


asyncio.run(main())
