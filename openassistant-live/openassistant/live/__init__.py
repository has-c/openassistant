"""OpenAssistant Live voice app."""

from .app import create_app
from .call import Call

__all__ = ["Call", "create_app"]
