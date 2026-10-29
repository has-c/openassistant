"""Private, single-person voice calls using the existing OpenAssistant session engine."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web

from .app import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    auth = parser.add_mutually_exclusive_group(required=True)
    auth.add_argument("--access-file", type=Path)
    auth.add_argument(
        "--tailnet-only",
        action="store_true",
        help="Trust the Tailnet-restricted reverse proxy; no application access key.",
    )
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--origin", default="http://localhost:8766")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--speech-endpoint", required=True)
    parser.add_argument(
        "--whisper-endpoint", default=None,
        help="Optional URL of the separate Whisper transcription service.",
    )
    parser.add_argument("--speech-voice", default="af_sky")
    parser.add_argument("--model", default=None)
    parser.add_argument("--pi-provider", default=None, help="Provider configured in Pi.")
    parser.add_argument("--pi-executable", default="pi", help="Path to the Pi CLI executable.")
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path.home() / ".local/share/openassistant/pi-history",
        help="Persistent conversation index and Pi session files.",
    )
    args = parser.parse_args()
    if args.access_file and args.access_file.stat().st_mode & 0o077:
        parser.error("Access file must have mode 0600.")
    access = json.loads(args.access_file.read_text()) if args.access_file else {}
    if (
        not isinstance(access, dict)
        or (not access and not args.tailnet_only)
        or any(not isinstance(token, str) or len(token) < 32 for token in access.values())
    ):
        parser.error("Access file must map names to random tokens of at least 32 characters.")
    config = SimpleNamespace(**vars(args), access=access)
    web.run_app(create_app(config), host="127.0.0.1", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
