"""Stage 3: Risk memo synthesis.

Combines all category-level findings into a unified risk memo
using an LLM call.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from financial_forecast.clients.protocols import LLMClient
from risk.prompts import SYNTHESIS_PROMPT
from risk.usage_tracker import UsageTracker


def synthesise_risk_memo(
    all_findings: Dict[str, Dict[str, Any]],
    llm_client: LLMClient,
    parameters: Dict[str, Any],
    tracker: UsageTracker,
) -> Dict[str, Any]:
    """Combine all category findings into a risk memo.

    Returns:
        Dict with ``risk_memo`` (free-text memo string).
    """
    findings_json = json.dumps(all_findings, indent=2, ensure_ascii=False)
    prompt = SYNTHESIS_PROMPT.format(findings=findings_json)

    memo_text = llm_client.ask_text(
        message="gen-ai-response",
        prompt=prompt,
        parameters=parameters,
        reasoning=True,
    )
    tracker.record("synthesis", prompt, memo_text)

    return {"risk_memo": memo_text}
