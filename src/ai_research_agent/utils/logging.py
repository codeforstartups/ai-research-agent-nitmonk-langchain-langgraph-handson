"""Rich-powered logging and a shared console.

Using a single configured :class:`rich.console.Console` keeps terminal output
consistent across the CLI, the graph, and the demos — important for a project
whose output is recorded on camera.
"""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler


def _force_utf8_streams() -> None:
    """Make stdout/stderr UTF-8 so Rich's box-drawing glyphs don't crash on
    legacy Windows consoles (cp1252 can't encode ``│``, ``●``, etc.).

    Recorded on Windows, so this matters: without it ``research-agent doctor``
    raises ``UnicodeEncodeError``. Safe no-op on platforms that are already
    UTF-8 or on streams that don't support ``reconfigure``.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):  # pragma: no cover - platform dependent
                pass


_force_utf8_streams()

console = Console()

_CONFIGURED = False


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Return a logger that renders through Rich.

    Configuration is applied once, lazily, so importing this module has no
    side effects until a logger is actually requested.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=level or "INFO",
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
        )
        _CONFIGURED = True
    return logging.getLogger(name)
