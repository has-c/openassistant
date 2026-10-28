"""Browser checks for shared saved chats. No agent turns or external actions."""

import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[1] / "openassistant/live/static"


class FakeHistory:
    def __init__(self):
        self.chats = {}

    def reply(self, ws, data):
        kind = data.get("type")
        if kind == "conversations":
            ws.send(
                json.dumps(
                    {
                        "type": kind,
                        "chats": [
                            {
                                "id": key,
                                "title": value["title"],
                                "updated": 1793156400,
                                "task": value.get("task"),
                            }
                            for key, value in self.chats.items()
                            if data.get("query", "").lower() in value["title"].lower()
                        ],
                        "next": None,
                        "offset": 0,
                        "query": data.get("query", ""),
                    }
                )
            )
        elif kind in ("new_conversation", "open_conversation"):
            identifier = data.get("id") or f"chat-{len(self.chats) + 1}"
            self.chats.setdefault(identifier, {"title": "New conversation", "messages": []})
            ws.send(
                json.dumps(
                    {
                        "type": "conversation",
                        "id": identifier,
                        **self.chats[identifier],
                        "turn": self.chats[identifier].get("turn", 0),
                        "busy": self.chats[identifier].get("busy", False),
                        "observer": self.chats[identifier].get("observer", False),
                        "truncated": False,
                    }
                )
            )
        else:
            return False
        return True


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True)
        page = await browser.new_page()
        errors, sockets, sent = [], [], []
        history = FakeHistory()
        page.on("pageerror", lambda error: errors.append(str(error)))
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

        def socket(ws):
            sockets.append(ws)

            def receive(raw):
                data = json.loads(raw)
                sent.append(data)
                if history.reply(ws, data):
                    return
                if "token" in data:
                    ws.send(json.dumps({"type": "ready", "name": "You"}))
                    items = []
                    for identifier, chat in history.chats.items():
                        task = chat.get("task") or {}
                        if chat.get("busy") and task.get("pending") and not chat.get("observer"):
                            items.append(
                                {
                                    "conversation": identifier,
                                    "title": chat["title"],
                                    "turn": chat.get("turn", 0),
                                    "pending": task["pending"],
                                }
                            )
                    ws.send(json.dumps({"type": "pending_inbox", "items": items}))

            ws.on_message(receive)

        await page.route_web_socket("**/ws", socket)
        await page.goto("http://localhost:8768/")
        await expect(page.locator("#prompt")).to_be_enabled()
        await page.locator("#toggle-panel").click()
        assert await page.evaluate("conversationId") == "chat-1"
        history.chats["chat-1"] = {
            "title": "Saved tax discussion",
            "messages": [
                {"id": "u", "who": "You", "text": "Continue the tax discussion"},
                {
                    "id": "a",
                    "who": "OpenAssistant",
                    "text": "Saved reply <img src=x onerror=alert(1)>",
                },
            ],
        }
        await page.reload()
        await expect(page.locator("#prompt")).to_be_enabled()
        await page.locator("#toggle-panel").click()
        await expect(page.locator("#transcript .message")).to_have_count(2)
        await expect(page.locator("#transcript")).to_contain_text("Saved reply <img")
        assert await page.locator("#transcript img").count() == 0
        assert await page.evaluate("clips.length === 0 && !player && !busy")
        assert not any(item.get("type") in ("prompt", "stop") for item in sent)
        if not await page.locator("#chat-history").is_visible():
            await page.locator("#toggle-history").click()
        await page.locator("#new-chat").click()
        await expect(page.locator("#transcript .message")).to_have_count(0)
        await expect(page.locator('#saved-chats button[aria-current="page"]')).to_have_text(
            "New conversation"
        )
        await page.locator("#search-chats").fill("tax")
        await expect(page.locator("#saved-chats button[aria-current]")).to_have_count(1)
        await page.locator("#saved-chats").get_by_text("Saved tax discussion", exact=True).click()
        await page.locator("#search-chats").fill("")
        await expect(page.locator("#transcript .message")).to_have_count(2)
        await page.locator("#prompt").fill("A follow-up")
        await page.locator("#send").click()
        sockets[-1].send(json.dumps({"type": "start", "turn": 1, "conversation": "chat-1"}))
        await expect(page.locator("#task-state")).to_have_text("Working")
        history.chats["chat-1"].update(
            busy=True,
            turn=1,
            task={"state": "working", "request": "A follow-up", "responses": [], "steps": []},
        )
        await page.locator("#prompt").fill("Only for the first chat")
        await page.locator("#send").click()
        await expect(page.locator("#text-queue li")).to_have_count(1)
        await page.locator("#prompt").fill("Draft for the first chat")
        if not await page.locator("#chat-history").is_visible():
            await page.locator("#toggle-history").click()
        await page.locator("#new-chat").click()
        await page.wait_for_function("conversationId === 'chat-3' && historyReady")
        await expect(page.locator("#text-queue li")).to_have_count(0)
        await expect(page.locator("#status")).to_have_text("Ready")
        await expect(page.locator("#prompt")).to_have_value("")
        history.chats["chat-3"].update(
            title="Second running task",
            busy=True,
            turn=2,
            task={"state": "working", "request": "Work in the second chat", "responses": []},
        )
        await page.locator("#prompt").fill("Work in the second chat")
        await page.locator("#send").click()
        sockets[-1].send(json.dumps({"type": "start", "turn": 2, "conversation": "chat-3"}))
        await expect(page.locator("#task-state")).to_have_text("Working")
        await page.evaluate("refreshHistory()")
        await expect(page.locator('#saved-chats [data-task="working"]')).to_have_count(2)
        # Background content stays out of the selected chat and cannot drain its queue.
        sockets[-1].send(
            json.dumps(
                {
                    "type": "message",
                    "turn": 1,
                    "conversation": "chat-1",
                    "item": "hidden",
                    "text": "Background-only reply",
                }
            )
        )
        sockets[-1].send(json.dumps({"type": "turn_ready", "turn": 1, "conversation": "chat-1"}))
        await expect(page.locator("#transcript")).not_to_contain_text("Background-only reply")
        assert await page.evaluate("busy && conversationId === 'chat-3'")
        if not await page.locator("#chat-history").is_visible():
            await page.locator("#toggle-history").click()
        await page.locator("#saved-chats").get_by_text("Saved tax discussion", exact=True).click()
        await expect(page.locator("#text-queue li")).to_have_count(1)
        await expect(page.locator("#text-queue")).to_contain_text("Paused: Only for the first chat")
        await expect(page.locator("#prompt")).to_have_value("Draft for the first chat")
        assert [item["text"] for item in sent if item.get("type") == "prompt"] == [
            "A follow-up",
            "Work in the second chat",
        ]
        sockets[-1].send(json.dumps({"type": "stopped", "turn": 1, "conversation": "chat-1"}))
        await expect(page.locator("#task-state")).to_have_text("Task stopped")
        await page.locator("#text-queue").get_by_text("Remove", exact=True).click()
        # Another call's active work is read-only here; it cannot be stopped or steered.
        sockets[-1].send(
            json.dumps({"type": "start", "turn": 2, "conversation": "chat-1", "observer": True})
        )
        await expect(page.locator("#interrupt")).to_be_disabled()
        await expect(page.locator("#mic")).to_be_disabled()
        await page.locator("#prompt").fill("Keep for later")
        await page.locator("#send").click()
        await expect(page.locator("#text-queue li")).to_have_count(1)
        await expect(page.locator("#text-queue button").first).to_be_disabled()
        assert len([item for item in sent if item.get("type") == "prompt"]) == 2
        await page.locator("#text-queue").get_by_text("Remove", exact=True).click()
        await page.locator("#saved-chats").get_by_text("New conversation", exact=True).click()
        await expect(page.locator('#saved-chats button[aria-current="page"]')).to_have_text(
            "New conversation"
        )
        await page.locator("#saved-chats").get_by_text("Saved tax discussion", exact=True).click()
        # Use readable synthetic content for review screenshots after the escaping check.
        history.chats["chat-1"]["messages"][1]["text"] = (
            "We saved the discussion here. You can reopen this chat and continue where we left off."
        )
        history.chats["chat-1"].update(
            busy=True,
            turn=73,
            task={
                "state": "working",
                "detail": "Checking the reconnect behaviour",
                "request": "Keep this task running when I leave",
                "started": 1793156400,
                "steps": [
                    {"step": "Check reconnect behaviour", "status": "inProgress"},
                    {"step": "Review the sidebar", "status": "pending"},
                ],
                "responses": [{"item": "a", "text": "The task kept running while you were away."}],
            },
        )
        await page.reload()
        await expect(page.locator("#prompt")).to_be_enabled()
        await page.locator("#toggle-panel").click()
        await expect(page.locator("#task-state")).to_have_text("Working")
        await expect(page.locator("#transcript .message")).to_have_count(2)
        await expect(page.locator("#interrupt")).to_be_enabled()
        await expect(page.locator('#saved-chats [data-task="working"] .chat-task')).to_have_count(2)
        assert await page.evaluate("clips.length === 0 && !player && busy")
        for width in (1440, 768, 390):
            await page.set_viewport_size({"width": width, "height": 900})
            if await page.locator("#chat-history").is_visible():
                await page.locator("#close-history").click()
            await page.locator("#toggle-history").click()
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            await page.screenshot(path=f"/tmp/openassistant-saved-chats-{width}.png")
            await page.screenshot(path=f"/tmp/openassistant-background-{width}.png")
            await page.screenshot(path=f"/tmp/openassistant-parallel-{width}.png")
        assert not errors, errors
        await browser.close()
        print("PASS: reload, switching, silent history, safe text, writer ownership and layouts")


if __name__ == "__main__":
    asyncio.run(main())
