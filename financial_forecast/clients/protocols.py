"""Structural interface for LLM clients.

Defines :class:`LLMClient`, a :class:`~typing.Protocol` that any LLM
client implementation must satisfy.  Stage modules depend on this
protocol rather than on concrete classes such as
:class:`AzureLLMClient`, satisfying the Dependency Inversion Principle.

``AzureLLMClient`` already conforms to this protocol without
modification — no inheritance or registration is needed.
"""

from __future__ import annotations

from typing import Any, Dict

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Structural interface for LLM clients.

    Any object with ``ask_json`` and ``ask_text`` methods matching
    these signatures is a valid ``LLMClient``.  This enables
    dependency inversion: high-level pipeline stages depend on this
    abstraction, not on a concrete class.
    """

    def ask_json(
        self,
        message: str,
        prompt: str,
        parameters: Dict[str, Any],
        reasoning: bool = True,
    ) -> Dict[str, Any]:
        """Send a prompt and return the parsed JSON response."""
        ...

    def ask_text(
        self,
        message: str,
        prompt: str,
        parameters: Dict[str, Any],
        reasoning: bool = True,
    ) -> str:
        """Send a prompt and return the response as plain text."""
        ...
