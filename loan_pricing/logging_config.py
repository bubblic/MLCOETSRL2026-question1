"""Centralised logging factory for the term loan pricing model.

Every module obtains its logger through :func:`get_logger` rather than
calling :func:`logging.getLogger` directly.  This guarantees a uniform
format and handler configuration across the entire package.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_DEFAULT_LEVEL = logging.INFO

_HANDLER_INSTALLED = False


def _ensure_root_handler() -> None:
    """Attach a stderr handler to the ``loan_pricing`` root logger once."""
    global _HANDLER_INSTALLED  # noqa: PLW0603
    if _HANDLER_INSTALLED:
        return

    root = logging.getLogger("loan_pricing")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(_DEFAULT_LEVEL)
    _HANDLER_INSTALLED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger that inherits the shared ``loan_pricing`` handler.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` with the package-wide format and level.
    """
    _ensure_root_handler()
    return logging.getLogger(name)
