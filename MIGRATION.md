# OpenAssistant preview

The browser app uses the Pi CLI through `pi-agent-python-sdk`, local Whisper transcription and a Kokoro speech endpoint. It is an experimental preview.

Install Python 3.12, Node.js and a compatible Pi CLI, then install the app:

```sh
python -m pip install ./openassistant-live
pi --version
```

Configure a provider and model in Pi. For local vLLM, the Pi provider must use the OpenAI-compatible Chat Completions endpoint, a served model name and the appropriate model-specific settings. Supply the chosen provider with `--pi-provider` and model with `--model`; omit them to use Pi's defaults.

Start Kokoro with `docker compose -f speech/kokoro/compose.yaml up -d`. Start Whisper with `docker compose -f speech/whisper/compose.yaml up -d --build`. The first start downloads the Whisper model into a Docker volume. Both services bind to loopback and have no authentication.

Create a private JSON file mapping user names to random access keys of at least 32 characters, with mode 0600. Then start the app:

```sh
python -m openassistant.live --access-file /path/to/access.json \
  --workspace /path/to/workspace --pi-provider PROVIDER --model MODEL \
  --speech-endpoint http://127.0.0.1:8880/v1/audio/speech \
  --whisper-endpoint http://127.0.0.1:9000/transcribe
```

The app listens on localhost port 8766. Remote microphone use needs a private HTTPS proxy and the corresponding `--origin`. Access to the app grants access to the configured agent and shared conversations. Back up the history directory as a unit: it contains both the SQLite index and Pi session files. Existing Codex sessions are not imported into Pi history.

## Chat controls and recovery

Use Pi CLI 1.3 or later. The Python SDK remains pinned at 0.2.6. Model discovery
uses a separate temporary, tool-free Pi process and never changes a saved session.
Models are identified by provider and model ID. Model selection is shared by
everyone with access to that conversation; thinking effort is remembered per
conversation/model. Unsupported saved effort falls back to Pi's supported value.
Unavailable models leave the selection unchanged and expose Retry / Use Default.
The selected settings are applied explicitly when resuming the native session.
Titles use the conversation's selected provider/model with tools and extensions disabled.

The SQLite upgrade is additive and preserves existing chats. Back up the history
directory before upgrading. New tables hold model preferences and task plans;
older chats use the configured default until changed. Active tasks are marked
Interrupted after a server restart and are never replayed automatically.

The Requests panel contains Pi extension confirm/select/input/editor requests
across chats, bound to the originating browser ID and access identity. Reconnecting
that browser restores pending requests. Other browsers can watch the shared chat
but cannot answer its requests. Requests expire within two minutes; cancellation,
disconnect and expiry never grant consent. Pi's ordinary tools retain their existing
permissions: this panel does not add a command approval sandbox.

Images may be PNG, JPEG, GIF or WebP, up to five per message and 4 MiB each.
Input batches are validated before storage and submitted as native Pi image inputs.
Pi/provider image limits still apply. If a queued message's model no longer accepts
images, the message stays paused for review. Accepted images are retained privately
in history (including Pi's resized version) and remain accessible to everyone
authorised for the shared app. Hiding a chat does not delete its images. Drafts and
unsent attachments stay in the browser tab and clear on refresh.

Work can continue beyond fifteen minutes. Stop, server shutdown and runtime errors
still end the task; transcription, speech, discovery and request waits remain
bounded. The bundled `update_plan` Pi extension publishes up to twenty structured
steps; plans restore when reopening a chat. Completed work is never inferred from
prose or from a disconnected browser.

## Validation

From `openassistant-live`, install `.[test]`, then run `python -m ruff check .`
and `python -m pytest -q`. Install Chrome with
`python -m playwright install --with-deps chrome` and run the `tests/browser_*.py`
scripts; CI runs these same interaction checks. They use protocol fixtures and do
not call your model or external tools.

For an opt-in end-to-end check, configure a dedicated image-capable Pi provider
and run `PI_CODING_AGENT_DIR=/path/to/test-pi python tests/smoke_pi.py`.
This makes two real model calls, verifies the bundled plan tool and image input,
and resumes the same native session after resetting Default. It uses a temporary
workspace, leaves synthetic evidence under `/tmp/openassistant-smoke-*`, and
disables speech and title calls. This check is separate from CI and needs a live model.

Pi’s default read, bash, edit and write tools and configured MCP servers are enabled. Use `pi mcp list` and `pi mcp add` to manage servers; no MCP server is bundled. Pi extensions are enabled because Pi uses them to load MCP; installed extensions may also run. Skills are disabled. Tool actions run with the app user’s operating system permissions, without a per-action approval prompt or an additional sandbox. Anyone with access to the app can request those actions, so use a dedicated account and workspace and restrict access accordingly.

Conversation titles use the same Pi provider and model as the conversation, in a separate temporary session with all tools and MCP disabled.

Before public release, finish a clean-install test, a secret scan of the final file set and a licensing review. The project license and redistribution rights for copied code still need confirmation.
