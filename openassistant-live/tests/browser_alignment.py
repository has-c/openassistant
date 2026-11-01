"""Exercise the alignment controls with an in-browser protocol fixture."""

import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[1] / "openassistant/live/static"
TEXT, IMAGE = '["local","text"]', '["local","vision"]'
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True)
        for width in (1440, 768, 390, 320):
            page = await browser.new_page(viewport={"width": width, "height": 900})
            errors, sockets, sent = [], [], []
            page.on("pageerror", lambda error: errors.append(str(error)))
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
            selected, effort = None, "off"

            def socket(ws):
                sockets.append(ws)

                def receive(raw):
                    nonlocal selected, effort
                    data = json.loads(raw)
                    sent.append(data)
                    kind = data.get("type")
                    if "token" in data:
                        ws.send(json.dumps({"type": "ready", "name": "You"}))
                    if kind in ("new_conversation", "open_conversation"):
                        ws.send(
                            json.dumps(
                                {
                                    "type": "conversation",
                                    "id": "chat",
                                    "messages": [],
                                    "busy": False,
                                    "turn": 0,
                                    "observer": False,
                                }
                            )
                        )
                    if kind == "conversations":
                        ws.send(
                            json.dumps(
                                {
                                    "type": "conversations",
                                    "query": data["query"],
                                    "request_id": data["request_id"],
                                    "offset": 0,
                                    "next": 50 if data["limit"] == 50 else None,
                                    "chats": [
                                        {
                                            "id": "chat",
                                            "title": "Local model review",
                                            "task": {"state": "working"},
                                        },
                                        {
                                            "id": "old",
                                            "title": "Previous conversation",
                                            "task": {"state": "finished"},
                                        },
                                    ],
                                }
                            )
                        )
                    if kind in ("model_list", "model_select", "effort_select"):
                        if kind == "model_select":
                            selected = data["model"]
                        if kind == "effort_select":
                            effort = data["effort"]
                        ws.send(
                            json.dumps(
                                {
                                    "type": "model_options",
                                    "conversation": "chat",
                                    "models": [
                                        {"id": TEXT, "label": "Local text · provider"},
                                        {"id": IMAGE, "label": "Vision model · provider"},
                                    ],
                                    "default": TEXT,
                                    "selected": selected,
                                    "effective": selected or TEXT,
                                    "levels": ["off", "low", "high"],
                                    "effort": effort,
                                }
                            )
                        )

                ws.on_message(receive)

            await page.route_web_socket("**/ws", socket)
            await page.goto("http://localhost:8768/")
            await expect(page.locator("#chat-model")).to_be_enabled()
            await page.locator("#chat-model").click()
            await page.get_by_role("button", name="Vision model · provider", exact=True).click()
            await expect(page.locator("#model-current")).to_have_text("Vision model · provider")
            await page.locator("#effort-trigger").click()
            await page.locator("#effort-slider").focus()
            await page.keyboard.press("End")
            await expect(page.locator("#effort-current")).to_have_text("high")
            await page.keyboard.press("Escape")
            assert any(d.get("model") == IMAGE and d.get("effort") == "high" for d in sent)
            # A stale model response from another chat must not replace the header.
            sockets[0].send(
                json.dumps({"type": "model_error", "conversation": "elsewhere", "text": "Stale"})
            )
            await expect(page.locator("#model-current")).to_have_text("Vision model · provider")
            # Queue an image, then preserve it if the server rejects the model capability.
            import base64

            await page.locator("#image-file").set_input_files(
                {"name": "fixture.png", "mimeType": "image/png", "buffer": base64.b64decode(PNG)}
            )
            await expect(page.locator("#image-draft img")).to_have_count(1)
            await page.locator("#send").click()
            await page.wait_for_timeout(50)
            assert any(d.get("type") == "prompt" and len(d.get("images", [])) == 1 for d in sent)
            sockets[0].send(
                json.dumps(
                    {
                        "type": "upload_error",
                        "conversation": "chat",
                        "text": "Choose an image model",
                    }
                )
            )
            await expect(page.locator("#text-queue")).to_contain_text("fixture.png")
            await expect(page.locator("#text-queue")).to_contain_text("Paused")
            await page.get_by_role("button", name="Remove", exact=True).click()
            # Plans and requests render independently of the currently selected chat.
            sockets[0].send(json.dumps({"type": "start", "turn": 1, "conversation": "chat"}))
            sockets[0].send(
                json.dumps(
                    {
                        "type": "plan",
                        "turn": 1,
                        "conversation": "chat",
                        "steps": [
                            {"step": "Compare model settings", "status": "completed"},
                            {"step": "Verify saved conversation", "status": "in_progress"},
                        ],
                    }
                )
            )
            await expect(page.locator('#task-list [data-state="in_progress"]')).to_have_count(1)
            await page.locator("#prompt").fill("Keep my draft")
            await page.locator("#prompt").focus()
            request = {
                "conversation": "background",
                "turn": 7,
                "title": "Background review",
                "pending": {"id": "confirm", "method": "confirm", "title": "Continue?"},
            }
            sockets[0].send(json.dumps({"type": "pending_inbox", "items": [request]}))
            await expect(page.locator("#pending-inbox")).to_be_visible()
            await expect(page.locator("#prompt")).to_be_focused()
            await (
                page.locator("#inbox-list")
                .get_by_role("button", name="Confirm", exact=True)
                .focus()
            )
            await page.keyboard.press("Control+Enter")
            await page.wait_for_timeout(50)
            assert any(
                d.get("type") == "global_answer"
                and d["conversation"] == "background"
                and d["turn"] == 7
                and d["value"] is True
                for d in sent
            )
            sockets[0].send(json.dumps({"type": "pending_inbox", "items": []}))
            await expect(page.locator("#pending-inbox")).to_be_hidden()
            await page.locator("#toggle-history").click()
            await expect(page.locator("#saved-chats")).to_contain_text("Active")
            await page.locator("#older-chats").click()
            await expect(page.locator("#older-chats")).to_be_hidden()
            sockets[0].send(json.dumps({"type": "turn_ready", "turn": 1, "conversation": "chat"}))
            await page.wait_for_timeout(50)
            assert [d for d in sent if d.get("type") == "conversations"][-1]["limit"] == 100
            await page.locator("#close-history").click()
            # Native upload paths are structured data, not Markdown fragments.
            sockets[0].send(
                json.dumps(
                    {
                        "type": "user",
                        "turn": 1,
                        "conversation": "chat",
                        "text": "An attachment",
                        "images": [
                            {
                                "id": "image-fixture",
                                "path": "/tmp/history (test)/photo.png",
                                "alt": "Uploaded photo",
                            }
                        ],
                    }
                )
            )
            await expect(page.locator("#live-caption .message-image")).to_have_count(1)
            # Reopened terminal tasks cannot look as if their unfinished steps are running.
            sockets[0].send(
                json.dumps(
                    {
                        "type": "conversation",
                        "id": "chat",
                        "turn": 1,
                        "busy": False,
                        "messages": [],
                        "task": {
                            "state": "stopped",
                            "steps": [{"step": "Still unfinished", "status": "in_progress"}],
                        },
                    }
                )
            )
            await expect(page.locator('#task-list [data-state="paused"]')).to_have_count(1)
            await page.wait_for_timeout(50)
            assert [d for d in sent if d.get("type") == "conversations"][-1]["limit"] == 100
            # Exercise failure recovery without stranding the picker in a disabled state.
            sockets[0].send(
                json.dumps({"type": "model_error", "conversation": "chat", "text": "Unavailable"})
            )
            await expect(page.locator("#model-current")).to_have_text("Retry models")
            await page.locator("#chat-model").click()
            await page.get_by_role("button", name="Retry", exact=True).click()
            await expect(page.locator("#model-current")).to_have_text("Vision model · provider")
            await page.keyboard.press("Escape")
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth"), width
            assert not errors, errors
            await page.screenshot(path=f"/tmp/openassistant-alignment-{width}.png")
            await page.close()
        await browser.close()
    print("PASS model/effort, images, plans, requests, retry, focus and responsive layouts")


asyncio.run(main())
