"""Exercise chat-sidebar sizing with synthetic history and no real agent turns."""

import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[1] / "openassistant/live/static"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True)
        for width in (1440, 1024, 768, 390):
            page = await browser.new_page(viewport={"width": width, "height": 900})
            await page.route(
                "**/static/*",
                lambda route: route.fulfill(
                    path=str(ROOT / route.request.url.split("/static/")[-1])
                ),
            )
            await page.route(
                "**/client-config", lambda route: route.fulfill(json={"access_required": False})
            )
            await page.route(
                "http://localhost:8768/", lambda route: route.fulfill(path=str(ROOT / "index.html"))
            )

            def socket(ws):
                def receive(raw):
                    data = json.loads(raw)
                    if "token" in data:
                        ws.send(json.dumps({"type": "ready", "name": "You"}))
                    if data.get("type") in ("new_conversation", "open_conversation"):
                        ws.send(
                            json.dumps(
                                {
                                    "type": "conversation",
                                    "id": "test",
                                    "title": "Example",
                                    "messages": [],
                                    "turn": 0,
                                    "busy": False,
                                    "observer": False,
                                    "truncated": False,
                                }
                            )
                        )
                    if data.get("type") == "conversations":
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
            await expect(page.locator("#prompt")).to_be_enabled()
            await page.locator("#toggle-history").click()
            sidebar, handle = page.locator("#chat-history"), page.locator("#resize-history")
            search = page.locator("#search-chats")
            new_chat = page.get_by_role("button", name="New chat")
            search_box, new_box = await search.bounding_box(), await new_chat.bounding_box()
            assert abs(search_box["y"] - new_box["y"]) < 2
            assert search_box["x"] + search_box["width"] <= new_box["x"]
            assert new_box["width"] >= 44 and search_box["width"] >= 80
            initial = (await sidebar.bounding_box())["width"]
            if width > 650:
                assert 195 <= initial <= max(201, width * 0.26), (width, initial)
            else:
                assert 200 <= initial < width * 0.9, (width, initial)
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            rect = await handle.bounding_box()
            await page.mouse.move(rect["x"] + rect["width"] / 2, rect["y"] + rect["height"] / 2)
            await page.mouse.down()
            await page.mouse.move(
                rect["x"] + rect["width"] / 2 + 40, rect["y"] + rect["height"] / 2, steps=4
            )
            await page.mouse.up()
            dragged = (await sidebar.bounding_box())["width"]
            assert dragged > initial + 20, (width, initial, dragged)
            await handle.focus()
            await page.keyboard.press("ArrowLeft")
            adjusted = (await sidebar.bounding_box())["width"]
            assert abs(adjusted - (dragged - 16)) < 2, (width, dragged, adjusted)
            await page.reload()
            await expect(page.locator("#prompt")).to_be_enabled()
            await page.locator("#toggle-history").click()
            assert abs((await sidebar.bounding_box())["width"] - adjusted) < 2
            await page.screenshot(path=f"/tmp/openassistant-sidebar-after-{width}.png")
            print(f"{width}px: default {initial:.0f}px, drag and keyboard resize, reload passed")
            await page.close()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
