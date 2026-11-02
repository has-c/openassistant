"""Capture image references from assistant replies for authenticated live chats."""

import base64
import binascii
import hashlib
import os
import re
import stat
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

MAX_IMAGE = 4 * 1024 * 1024
MAX_IMAGES = 6
MAX_PIXELS = 25_000_000
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


def validate_upload(encoded):
    """Decode a bounded base64 upload and return its bytes and detected MIME type."""
    if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_IMAGE + 2) // 3):
        raise ValueError("Image is too large or unavailable.")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("Invalid image data.") from None
    mime = image_type(data)
    if (
        not data
        or len(data) > MAX_IMAGE
        or mime not in {"image/png", "image/jpeg", "image/gif", "image/webp"}
    ):
        raise ValueError("Choose a PNG, JPEG, GIF or WebP image under 4 MB.")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Images must contain at most 25 million pixels.")
            image.verify()
        # verify() alone does not decode JPEG pixels or catch truncated streams.
        with Image.open(BytesIO(data)) as image:
            image.load()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, SyntaxError) as error:
        raise ValueError("The image is damaged or cannot be decoded.") from error
    return data, mime


def prepare_uploads(encoded_images):
    """Fully validate a batch off-loop without persisting rejected input."""
    if not isinstance(encoded_images, list) or not 1 <= len(encoded_images) <= 5:
        raise ValueError("Attach between one and five images.")
    return [validate_upload(encoded) for encoded in encoded_images]


def retain_uploads(history, conversation, uploads):
    """Persist an already validated, accepted batch in one transaction."""
    if not history or not conversation:
        raise ValueError("Image history is unavailable.")
    history.get(conversation)
    attachments = []
    with history.connect() as db:
        for data, mime in uploads:
            identifier = hashlib.sha256(data).hexdigest()
            path = f"/upload/{identifier}"
            db.execute(
                "INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conversation, "uploads", identifier, path, "Attached image", mime, data),
            )
            attachments.append({"id": identifier, "path": path, "alt": "Attached image"})
    return attachments


def save_upload(history, conversation, encoded):
    return save_uploads(history, conversation, [encoded])[0]


def save_uploads(history, conversation, encoded_images):
    return retain_uploads(history, conversation, prepare_uploads(encoded_images))


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
