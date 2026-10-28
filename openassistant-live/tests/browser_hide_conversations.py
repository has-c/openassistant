import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[1] / "openassistant/live/static"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True)
        for width, height, mobile in [(1440, 900, False), (390, 844, True)]:
            page = await browser.new_page(
                viewport={"width": width, "height": height}, has_touch=mobile
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
            chats = [
                {
                    "id": "chat-1",
                    "title": "Review PR 182",
                    "updated": 1793173000,
                    "task": {"state": "working"},
                }
            ]

            def socket(ws):
                def receive(raw):
                    d = json.loads(raw)
                    k = d.get("type")
                    if "token" in d:
                        ws.send(json.dumps({"type": "ready", "name": "You"}))
                    elif k == "conversations":
                        ws.send(
                            json.dumps(
                                {"type": k, "chats": chats, "next": None, "offset": 0, "query": ""}
                            )
                        )
                    elif k in ("new_conversation", "open_conversation"):
                        ws.send(
                            json.dumps(
                                {
                                    "type": "conversation",
                                    "id": "chat-1",
                                    "title": "Review PR 182",
                                    "messages": [],
                                    "busy": False,
                                    "observer": False,
                                    "truncated": False,
                                    "turn": 0,
                                }
                            )
                        )
                    elif k == "hide_conversation":
                        chats.clear()
                        ws.send(json.dumps({"type": "conversation_hidden", "id": d["id"]}))

                ws.on_message(receive)

            await page.route_web_socket("**/ws", socket)
            await page.goto("http://localhost:8768/")
            await expect(page.locator("#prompt")).to_be_enabled()
            await page.locator("#toggle-history").click()
            row = page.locator("#saved-chats li").first
            close = page.get_by_role("button", name="Remove Review PR 182 from recent chats")
            await expect(close).to_have_count(1)
            if not mobile:
                assert await close.evaluate("el => getComputedStyle(el).opacity") == "0"
                await row.hover()
            await expect(close).to_have_css("opacity", "1")
            await page.screenshot(path=f"/tmp/openassistant-hide-{width}.png")
            await close.click()
            await expect(page.locator("#saved-chats")).to_contain_text("No chats yet")
            assert await page.evaluate("conversationId") == "chat-1"
            await page.close()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
