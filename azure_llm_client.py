"""
Azure-hosted LLM with reasoning and non-reasoning models.
Reasoning model uses DeepSeek-V3.2 while the non-reasoning model uses gpt-5-nano.
It posts a payload with `message` and `body`.
"""

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class AzureLLMClient:
    """Generic Azure LLM client for JSON and text prompts."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.endpoint = (endpoint or os.getenv("AZURE_DEEPSEEK_ENDPOINT", "")).strip()
        self.timeout_seconds = timeout_seconds

        if not self.endpoint:
            raise ValueError("Missing Azure config. Set AZURE_DEEPSEEK_ENDPOINT.")

    def ask_json(
        self,
        message: str,
        prompt: str,
        parameters: Dict[str, Any],
        reasoning: bool = True,
    ) -> Dict[str, Any]:
        payload = {
            "message": message,
            "body": {
                "prompt": prompt,
                "parameters": parameters,
                "reasoning": reasoning,
            },
        }
        response_text = self._post_json(payload)
        return self._extract_json(response_text)

    def ask_text(
        self,
        message: str,
        prompt: str,
        parameters: Dict[str, Any],
        reasoning: bool = True,
    ) -> str:
        payload = {
            "message": message,
            "body": {
                "prompt": prompt,
                "parameters": parameters,
                "reasoning": reasoning,
            },
        }
        response_text = self._post_json(payload)
        extracted = self._extract_json(response_text)
        if isinstance(extracted, dict) and "raw_response" in extracted:
            return str(extracted["raw_response"])
        return json.dumps(extracted, ensure_ascii=False)

    def _post_json(self, payload: Dict[str, Any]) -> str:
        data = json.dumps(payload).encode("utf-8")
        max_attempts = 100
        backoff_seconds = 1.0
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=data,
                headers={
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as resp:
                    return resp.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                last_exc = exc
                print(
                    f"[Attempt {attempt}/{max_attempts}] HTTP {exc.code}: {exc.reason}"
                )
                if exc.code == 503:
                    print(
                        "503: High traffic on Azure API — server-side timeout, retrying..."
                    )
                elif exc.code not in (429, 500, 502, 503, 504):
                    raise
            except Exception as exc:
                last_exc = exc
                print(
                    f"[Attempt {attempt}/{max_attempts}] Azure API request failed: {exc}"
                )
            if attempt == max_attempts:
                break
            print(f"Sleeping for {backoff_seconds} seconds before next attempt...")
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 10.0)

        raise RuntimeError(f"Azure API request failed: {last_exc}") from last_exc

    def _extract_json(self, response_text: str) -> Dict[str, Any]:
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"API did not return JSON: {exc}") from exc

        if isinstance(payload, dict) and "data" in payload:
            data = payload["data"]
            if isinstance(data, dict) and "response" in data:
                response_text = data["response"]
                if isinstance(response_text, str):
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError:
                        extracted = self._extract_json_from_text(response_text)
                        if extracted is not None:
                            return extracted
                        return {"raw_response": response_text}
            if isinstance(data, dict):
                return data

        if isinstance(payload, dict):
            return payload
        raise ValueError("Unexpected API response shape.")

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
