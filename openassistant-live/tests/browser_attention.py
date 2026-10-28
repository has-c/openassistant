"""Verify attention requests, audio modes and replies in a browser."""

import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[1] / "openassistant/live/static"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True)
        for width in (1440, 768, 390):
            page = await browser.new_page(viewport={"width": width, "height": 900})
            await page.add_init_script(
                """window.bells=0;
                window.AudioContext=class {constructor(){this.state='running';
                this.currentTime=0; this.destination={};}
                resume(){return Promise.resolve()}
                createOscillator(){window.bells++;return {frequency:{value:0},
                connect(node){return node},disconnect(){},start(){},stop(){}}} 
                createGain(){return {gain:{setValueAtTime(){},linearRampToValueAtTime(){},
                exponentialRampToValueAtTime(){}},connect(){},disconnect(){}}}}"""
            )
            await page.route(
                "**/static/*",
                lambda r: r.fulfill(path=str(ROOT / r.request.url.split("/static/")[-1])),
            )
            await page.route(
                "**/client-config", lambda r: r.fulfill(json={"access_required": False})
            )
            await page.route(
                "http://localhost:8768/", lambda r: r.fulfill(path=str(ROOT / "index.html"))
            )
            sockets, sent = [], []

            def socket(ws):
                sockets.append(ws)

                def receive(raw):
                    d = json.loads(raw)
                    sent.append(d)
                    if "token" in d:
                        ws.send(json.dumps({"type": "ready", "name": "You"}))
                    if d.get("type") in ("new_conversation", "open_conversation"):
                        ws.send(
                            json.dumps(
                                {
                                    "type": "conversation",
                                    "id": "chat",
                                    "title": "Chat",
                                    "messages": [],
                                    "turn": 0,
                                    "busy": False,
                                    "observer": False,
                                    "truncated": False,
                                }
                            )
                        )
                    if d.get("type") == "conversations":
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
            page.on("pageerror", lambda e: print("PAGE ERROR", str(e), flush=True))
            await page.goto("http://localhost:8768/")
            await expect(page.locator("#prompt")).to_be_enabled()
            await page.wait_for_function("() => conversationId === 'chat' && historyReady")
            await page.locator("#sound").click()
            await page.locator('[data-sound="alerts"]').click()
            sockets[0].send(json.dumps({"type": "start", "turn": 1, "conversation": "chat"}))
            req = {
                "id": "abc",
                "method": "confirm",
                "title": "Review request",
                "message": "Continue with this change?",
                "options": [],
            }
            sockets[0].send(
                json.dumps(
                    {"type": "input_request", "turn": 1, "conversation": "chat", "request": req}
                )
            )
            await expect(page.locator("#input-requests")).to_be_visible()
            assert await page.evaluate("window.bells") == 2
            await page.screenshot(path=f"/tmp/openassistant-attention-{width}.png")
            await page.get_by_role("button", name="Yes", exact=True).click()
            await page.wait_for_timeout(100)
            assert any(
                x.get("type") == "input_answer" and x.get("value") is True and x.get("id") == "abc"
                for x in sent
            )
            sockets[0].send(
                json.dumps({"type": "input_closed", "turn": 1, "conversation": "chat", "id": "abc"})
            )
            await expect(page.locator("#input-requests")).to_be_hidden()
            await page.close()
        await browser.close()
    print("PASS attention requests and sound controls at desktop, tablet and mobile widths")


asyncio.run(main())
