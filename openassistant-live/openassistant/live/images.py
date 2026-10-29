"""Capture image references from assistant replies for authenticated live chats."""

import hashlib
import os
import re
import stat
from pathlib import Path

MAX_IMAGE = 4 * 1024 * 1024
MAX_IMAGES = 6
# Only absolute local paths explicitly included in an assistant message are considered.
IMAGE_LINK = re.compile(r"!?\[([^\]\n]{0,300})\]\((?:<([^>\n]+)>|(/[^)\n]+))\)")


def image_type(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def capture_images(history, conversation, item, text):
    if not history or not conversation or not isinstance(item, str):
        return []
    found = []
    for match in IMAGE_LINK.finditer(text):
        label, quoted, plain = match.groups()
        path = quoted or plain
        if (
            not path
            or not path.startswith("/")
            or Path(path).suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        ):
            continue
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_IMAGE:
                    continue
                data = stream.read(MAX_IMAGE + 1)
        except (OSError, ValueError):
            continue
        mime = image_type(data)
        if not mime or len(data) > MAX_IMAGE:
            continue
        identifier = hashlib.sha256(data).hexdigest()
        history.save_image(conversation, item, identifier, path, label or "Image", mime, data)
        found.append({"id": identifier, "path": path, "alt": label or "Image"})
        if len(found) >= MAX_IMAGES:
            break
    return found
