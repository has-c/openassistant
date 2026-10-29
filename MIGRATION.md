# OpenAssistant preview

The browser app uses the Pi CLI through `pi-agent-python-sdk`, local Whisper transcription and a Kokoro speech endpoint. It is an experimental preview.

Install Python 3.12, Node.js and a compatible Pi CLI, then install the app:

```sh
python -m pip install ./openassistant-live
pi --version
```

Configure a provider and model in Pi. For local vLLM, the Pi provider must use the OpenAI-compatible Chat Completions endpoint, a served model name and the appropriate model-specific settings. Supply the chosen provider with `--pi-provider` and model with `--model`; omit them to use Pi's defaults.

Start Kokoro with `docker compose -f speech/compose.yaml up -d`. Start Whisper with `docker compose -f speech/whisper/compose.yaml up -d --build`. The first start downloads the Whisper model into a Docker volume. Both services bind to loopback and have no authentication.

Create a private JSON file mapping user names to random access keys of at least 32 characters, with mode 0600. Then start the app:

```sh
python -m openassistant.live --access-file /path/to/access.json \
  --workspace /path/to/workspace --pi-provider PROVIDER --model MODEL \
  --speech-endpoint http://127.0.0.1:8880/v1/audio/speech \
  --whisper-endpoint http://127.0.0.1:9000/transcribe
```

The app listens on localhost port 8766. Remote microphone use needs a private HTTPS proxy and the corresponding `--origin`. Access to the app grants access to the configured agent and shared conversations. Back up the history directory as a unit: it contains both the SQLite index and Pi session files. Existing Codex sessions are not imported into Pi history.

Pi’s default read, bash, edit and write tools and configured MCP servers are enabled. Use `pi mcp list` and `pi mcp add` to manage servers; no MCP server is bundled. Pi extensions are enabled because Pi uses them to load MCP; installed extensions may also run. Skills are disabled. Tool actions run with the app user’s operating system permissions, without a per-action approval prompt or an additional sandbox. Anyone with access to the app can request those actions, so use a dedicated account and workspace and restrict access accordingly.

Conversation titles use the same Pi provider and model as the conversation, in a separate temporary session with all tools and MCP disabled.

Before public release, finish a clean-install test, a secret scan of the final file set and a licensing review. The project license and redistribution rights for copied code still need confirmation.
