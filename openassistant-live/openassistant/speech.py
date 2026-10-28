"""Bounded speech synthesis through a configured endpoint."""

import aiohttp


class Speech:
    def __init__(self, config):
        self.config = config

    async def fetch(self, path, text):
        """Bound endpoint responses and never forward reply text through redirects."""
        timeout = aiohttp.ClientTimeout(total=90)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as client:
            async with client.post(
                self.config.speech_endpoint,
                json={
                    "model": "kokoro",
                    "input": text,
                    "voice": self.config.speech_voice,
                    "response_format": "opus",
                    "speed": 1.0,
                },
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise RuntimeError("Speech endpoint unavailable.")
                size = 0
                with path.open("wb") as output:
                    async for chunk in response.content.iter_chunked(65536):
                        size += len(chunk)
                        if size > 5_000_000:
                            raise RuntimeError("Speech endpoint output too large.")
                        output.write(chunk)
                with path.open("rb") as output:
                    if output.read(4) != b"OggS":
                        raise RuntimeError("Speech endpoint returned invalid Opus audio.")
