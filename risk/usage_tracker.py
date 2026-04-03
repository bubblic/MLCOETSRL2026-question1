"""Lightweight LLM usage tracking per pipeline stage.

The Azure endpoint does not return token counts, so this module tracks
call counts and character volumes as a best-effort approximation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class StageUsage:
    """Accumulated usage for a single pipeline stage."""

    stage: str
    call_count: int = 0
    chars_in: int = 0
    chars_out: int = 0


class UsageTracker:
    """Thread-safe tracker for LLM call volume per pipeline stage.

    Safe to share across ``ThreadPoolExecutor`` workers.  All mutations
    to internal state are protected by a ``threading.Lock``.
    """

    def __init__(self) -> None:
        self._stages: Dict[str, StageUsage] = {}
        self._lock = threading.Lock()

    def record(self, stage: str, prompt: str, response: str) -> None:
        """Record one LLM call's character volumes.

        Thread-safe: acquires an internal lock before mutating state.
        """
        with self._lock:
            if stage not in self._stages:
                self._stages[stage] = StageUsage(stage=stage)
            usage = self._stages[stage]
            usage.call_count += 1
            usage.chars_in += len(prompt)
            usage.chars_out += len(response)

    def summary(self) -> Dict[str, Any]:
        """Return per-stage and total usage summary."""
        stages = {}
        total_calls = 0
        total_in = 0
        total_out = 0
        for name, usage in self._stages.items():
            stages[name] = {
                "call_count": usage.call_count,
                "chars_in": usage.chars_in,
                "chars_out": usage.chars_out,
            }
            total_calls += usage.call_count
            total_in += usage.chars_in
            total_out += usage.chars_out
        return {
            "stages": stages,
            "total_calls": total_calls,
            "total_chars_in": total_in,
            "total_chars_out": total_out,
        }

    def print_summary(self) -> None:
        """Print a human-readable usage summary to stdout."""
        s = self.summary()
        print("\n=== LLM Usage Summary ===")
        for name, data in s["stages"].items():
            print(
                f"  {name}: {data['call_count']} calls, "
                f"{data['chars_in']:,} chars in, "
                f"{data['chars_out']:,} chars out"
            )
        print(
            f"  TOTAL: {s['total_calls']} calls, "
            f"{s['total_chars_in']:,} chars in, "
            f"{s['total_chars_out']:,} chars out"
        )
