from types import SimpleNamespace as NS

import pytest
from aiohttp import web

from openassistant.speech import Speech


@pytest.mark.parametrize("mode", ["ok", "redirect", "invalid", "oversize", "unavailable"])
async def test_endpoint_contract(tmp_path, mode):
    async def respond(request):
        payload = await request.json()
        assert payload["voice"] == "af_sky" and payload["response_format"] == "opus"
        assert payload["input"] == "Private answer"
        if mode == "redirect":
            return web.Response(status=302, headers={"Location": "/unexpected"})
        if mode == "unavailable":
            return web.Response(status=503)
        body = b"OggS-test" if mode == "ok" else b"invalid"
        if mode == "oversize":
            body = b"OggS" + b"x" * 5_000_000
        return web.Response(body=body)

    server = web.Application()
    server.router.add_post("/v1/audio/speech", respond)
    runner = web.AppRunner(server)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    config = NS(speech_endpoint=f"http://127.0.0.1:{port}/v1/audio/speech", speech_voice="af_sky")
    output = tmp_path / "reply.ogg"
    try:
        if mode == "ok":
            await Speech(config).fetch(output, "Private answer")
            assert output.read_bytes() == b"OggS-test"
        else:
            with pytest.raises(RuntimeError):
                await Speech(config).fetch(output, "Private answer")
    finally:
        await runner.cleanup()
