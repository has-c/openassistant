"""Exercise keyboard navigation and stable panels without a real agent."""

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
            sockets = []
            answers = []

            def socket(ws):
                sockets.append(ws)

                def receive(raw):
                    data = json.loads(raw)
                    if data.get("type") in ("answer", "global_answer"):
                        answers.append(data)
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
            prompt = page.locator("#prompt")
            await expect(prompt).to_be_enabled()
            assert await prompt.evaluate("el => el.tagName === 'TEXTAREA'")
            short_height = (await prompt.bounding_box())["height"]
            await prompt.fill("A long message " * 120)
            assert (await prompt.bounding_box())["height"] > short_height
            assert await prompt.evaluate("el => el.scrollHeight > el.clientHeight")
            await prompt.fill("")
            await prompt.press("Shift+Enter")
            await expect(prompt).to_have_value("\n")
            await prompt.fill("")
            await prompt.fill("first message")
            await prompt.press("Enter")
            await prompt.fill("draft")
            await prompt.press("ArrowUp")
            await expect(prompt).to_have_value("first message")
            await prompt.press("ArrowDown")
            await expect(prompt).to_have_value("draft")
            canvas = page.locator(".stage")
            before = await canvas.bounding_box()
            await page.keyboard.press("Alt+h")
            await expect(page.locator("#chat-history")).to_be_visible()
            await expect(page.locator("#search-chats")).to_be_focused()
            after = await canvas.bounding_box()
            if width > 650:
                assert after["width"] < before["width"]
                sidebar_box = await page.locator("#chat-history").bounding_box()
                footer_box = await page.locator("footer").bounding_box()
                assert (
                    sidebar_box["y"] + sidebar_box["height"]
                    >= footer_box["y"] + footer_box["height"]
                )
                composer_box = await page.locator("#composer").bounding_box()
                controls_box = await page.locator(".controls").bounding_box()
                assert (
                    abs(
                        (composer_box["x"] + composer_box["width"] / 2)
                        - (controls_box["x"] + controls_box["width"] / 2)
                    )
                    < 2
                )
                toggle_box = await page.locator("#toggle-panel").bounding_box()
                assert controls_box["x"] + controls_box["width"] <= toggle_box["x"]
            await page.screenshot(path=f"/tmp/openassistant-chats-open-{width}.png")
            await page.keyboard.press("Alt+c")
            await expect(page.locator("#conversation")).to_be_visible()
            panel = page.locator("#conversation")
            before = await panel.bounding_box()
            await page.locator("#transcript").evaluate(
                "el => { for (let i=0; i<80; i++) { const p = document.createElement('p'); "
                "p.textContent = 'A long conversation'; el.append(p); } }"
            )
            after = await panel.bounding_box()
            assert abs(before["height"] - after["height"]) < 1
            assert await page.locator("#transcript").evaluate(
                "el => el.scrollHeight > el.clientHeight"
            )
            await page.screenshot(path=f"/tmp/openassistant-keyboard-{width}.png")
            print(f"{width}px: history, shortcuts, stable canvas and scrolling passed")
            await page.close()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
