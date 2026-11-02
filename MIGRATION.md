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
caches its catalogue for five minutes. Normal turns refresh it through their existing
Pi connection; a separate sessionless process is used for a cold cache or explicit
refresh. Discovery enables configured extensions (including their setup code), but
disables tools, MCP and skills. Picker failures never prevent message submission or
Stop. Models are identified by provider and model ID. Selections are shared across
browsers and apply after the picker acknowledges them; running turns keep their snapshot.
Only explicit thinking choices are saved per conversation/model choice. Use default
effort clears that override; an inherited effort follows Pi's current default, clamped
to the model's supported levels. An unsupported explicit effort requires a reset.
Unavailable explicit models never silently switch provider. Choose an alternative or
Default; preparation failures preserve queued text/images and do not bind a new session.
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
permissions: this panel does not add a command approval sandbox. In tailnet mode,
all admitted peers share an access identity. Browser IDs are client-supplied, so these
ownership checks are not strong per-person authentication; trust admitted peers.

Images may be PNG, JPEG, GIF or WebP, up to five per message, 4 MiB and 25 million
pixels each. Pillow verifies and decodes uploads to reject damaged image files.
Input batches are validated before storage and submitted as native Pi image inputs.
Pi/provider image limits still apply. If a queued message's model no longer accepts
images, the message stays paused for review. Accepted images are retained privately
in SQLite, without a redundant upload file. Pi also retains native image data in its
JSONL history; resized versions can add another SQLite rendition. There is no automatic
retention cleanup. Browser upload references use opaque IDs. Images remain accessible to everyone
authorised for the shared app. Hiding a chat does not delete its images. Drafts and
unsent attachments stay in the browser tab and clear on refresh.

Work can continue beyond fifteen minutes. Stop, server shutdown and runtime errors
still end the task. By default, 1,800 seconds without Pi progress stops the task.
Every streamed Pi event, including tool output, resets this inactivity watchdog;
waiting for an extension answer pauses it. Use `--inactivity-timeout SECONDS` to change
it or `--inactivity-timeout 0` to disable it. A legitimately silent tool can hit this
limit; browser heartbeats do not extend it. Shutdown aborts Pi, then uses the SDK's
bounded terminate/kill/reap path and joins the run worker before releasing capacity.
This does not undo completed actions or guarantee cancellation of external jobs.
Transcription, speech, discovery and request waits remain bounded. The bundled `update_plan` Pi extension publishes up to twenty structured
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

Set `OPENASSISTANT_SMOKE_NO_DISCOVERY=1` for the same live check with picker
discovery deliberately unavailable. Use a provider registered by a test extension
in the dedicated Pi configuration to check extension-provider parity.
