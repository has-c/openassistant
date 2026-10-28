import asyncio
import base64
import io
import json
import math
import struct
import wave
from pathlib import Path

from browser_saved_conversations import FakeHistory
from playwright.async_api import async_playwright, expect

root = Path(__file__).resolve().parents[1] / "openassistant/live/static"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        errors = []
        uploads = []
        stops = []
        sockets = []
        history = FakeHistory()
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(
                b"".join(struct.pack("<h", int(1000 * math.sin(i / 20))) for i in range(16000 * 8))
            )
        audio = base64.b64encode(output.getvalue()).decode()
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.add_init_script("""
          navigator.mediaDevices.getUserMedia=async()=>{
            const context=new AudioContext(); await context.resume();
            const oscillator=context.createOscillator();
            const gain=context.createGain(); gain.gain.value=0;
            const dest=context.createMediaStreamDestination();
            oscillator.connect(gain).connect(dest); oscillator.start();
            window.testGain=gain; window.testStream=dest.stream;
            return dest.stream;
          };
        """)

        async def static(route):
            name = route.request.url.split("/static/")[-1]
            await route.fulfill(path=str(root / name))

        await page.route("**/static/*", static)
        await page.route(
            "**/client-config", lambda route: route.fulfill(json={"access_required": False})
        )
        await page.route(
            "http://localhost:8768/", lambda route: route.fulfill(path=str(root / "index.html"))
        )

        async def socket(ws):
            sockets.append(ws)

            def receive(message):
                if isinstance(message, bytes):
                    uploads.append(message)
                    turn = len(uploads)
                    ws.send(json.dumps({"type": "start", "turn": turn}))
                    if turn == 3:
                        ws.send(
                            json.dumps(
                                {
                                    "type": "message",
                                    "turn": turn,
                                    "item": "reply",
                                    "text": "I found the calendar. I’m checking tonight’s "
                                    "availability before preparing the invite.",
                                }
                            )
                        )
                        ws.send(
                            json.dumps(
                                {
                                    "type": "task",
                                    "turn": turn,
                                    "state": "working",
                                    "text": "Working on your request",
                                }
                            )
                        )
                        ws.send(
                            json.dumps(
                                {
                                    "type": "activity",
                                    "turn": turn,
                                    "id": "calendar",
                                    "label": "Calendar search",
                                    "state": "running",
                                }
                            )
                        )
                        ws.send(json.dumps({"type": "audio", "turn": turn, "data": audio}))
                    else:
                        ws.send(json.dumps({"type": "task", "turn": turn, "state": "finished"}))
                        ws.send(json.dumps({"type": "done", "turn": turn}))
                        ws.send(json.dumps({"type": "turn_ready", "turn": turn}))
                else:
                    data = json.loads(message)
                    if history.reply(ws, data):
                        return
                    if "token" in data:
                        ws.send(json.dumps({"type": "ready", "name": "You"}))
                    if data.get("type") == "stop":
                        stops.append(True)

                        async def acknowledge():
                            await asyncio.sleep(1.8)
                            ws.send(
                                json.dumps(
                                    {
                                        "type": "delta",
                                        "turn": 3,
                                        "item": "stale",
                                        "text": "STALE REPLY",
                                    }
                                )
                            )
                            ws.send(json.dumps({"type": "stopped", "turn": 3}))

                        asyncio.create_task(acknowledge())

            ws.on_message(receive)

        await page.route_web_socket("**/ws", socket)
        await page.goto("http://localhost:8768/")
        await expect(page.locator("#prompt")).to_be_enabled()
        await expect(page.locator("#connection")).to_have_text("Connected")
        await page.locator("#mic").click()
        await expect(page.locator("#status")).to_have_text("Listening")
        await page.wait_for_timeout(1500)
        assert len(uploads) == 0, "Idle silence uploaded"
        for n in (1, 2):
            await page.evaluate("testGain.gain.value=0.15")
            await page.wait_for_timeout(700)
            await page.evaluate("testGain.gain.value=0")
            await page.wait_for_timeout(1900)
            assert len(uploads) == n, f"Expected {n} utterances, received {len(uploads)}"
            await expect(page.locator("#status")).to_have_text("Listening")
            await expect(page.locator("#mic-label")).to_have_text("Mic on")
        # A real audio clip plays while the mic captures a new utterance.
        await page.evaluate("testGain.gain.value=0.15")
        await page.wait_for_timeout(500)
        await page.evaluate("testGain.gain.value=0")
        await expect(page.locator("#status")).to_have_text("Speaking", timeout=5000)
        await expect(page.locator("#task-state")).to_have_text("Working")
        await page.locator("#task-activity summary").click()
        await expect(page.locator("#activity-list")).to_contain_text("Calendar search — running")
        await page.locator("#task-activity summary").click()
        for width in (1440, 768, 390):
            await page.set_viewport_size({"width": width, "height": 900})
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            await page.screenshot(path=f"/tmp/openassistant-task-{width}.png")
            await page.locator("#toggle-panel").click()
            await expect(page.locator("#conversation")).to_be_visible()
            await page.screenshot(path=f"/tmp/openassistant-conversation-{width}.png")
            await page.locator("#close-panel").click()
        await page.set_viewport_size({"width": 390, "height": 640})
        await page.screenshot(path="/tmp/openassistant-task-compact.png")
        await expect(page.locator("#leave")).to_be_in_viewport()
        await page.set_viewport_size({"width": 1280, "height": 900})
        # Sound off must not stop the agent's task.
        await page.locator("#sound").click()
        assert not stops
        await expect(page.locator("#task-state")).to_have_text("Working")
        await page.locator("#sound").click()
        await page.evaluate("testGain.gain.value=0.15")
        await page.wait_for_timeout(400)
        assert len(stops) == 1, "Speaking did not interrupt playback"
        await expect(page.locator("#task-state")).to_have_text("Stopping task…")
        assert await page.evaluate("player === null"), "Old audio is still playing"
        await page.evaluate("testGain.gain.value=0")
        await page.wait_for_timeout(1000)
        assert len(uploads) == 3, "New audio sent before stop acknowledgement"
        await page.wait_for_timeout(1000)
        assert len(uploads) == 4, "Interrupted utterance was lost"
        assert "STALE REPLY" not in await page.locator("#transcript").inner_text()
        await page.evaluate("testGain.gain.value=0.15")
        await page.wait_for_timeout(400)
        await page.locator("#mic").click()
        await page.wait_for_timeout(1400)
        assert len(uploads) == 4, "Mute submitted unfinished audio"
        assert await page.evaluate('testStream.getTracks().every(t=>t.readyState==="ended")')
        await expect(page.locator("#mic-label")).to_have_text("Mic off")
        await page.locator("#leave").click()
        await expect(page.locator(".portrait h2")).to_have_text("Call ended")
        await page.locator("#join").click()
        await expect(page.locator("#connection")).to_have_text("Connected")
        await sockets[-1].close(code=1011, reason="Simulated service failure")
        await expect(page.locator(".portrait h2")).to_have_text("Connection lost")
        await expect(page.locator("#connection")).to_have_text("Connected")
        await expect(page.locator("#mic-label")).to_have_text("Mic off")
        assert not errors, errors
        print(
            "Real Web Audio/MediaRecorder: silence ignored; two automatic turns; "
            "barge-in stops real playback; delayed acknowledgement preserves the next utterance; "
            "stale reply ignored; mute drops unfinished audio and stops tracks."
        )
        await browser.close()


asyncio.run(main())
