"""
Balance Sheet Prediction Model using an Azure API.

This module calls an Azure-hosted API to predict balance sheet elements
from structured historical inputs. It posts a payload with `message`
and `body`, mirroring a Flutter HTTP request pattern.
"""

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class BalanceSheetInputs:
    """Structured inputs for balance sheet prediction."""

    company_name: str
    ticker: str
    fiscal_year: int
    currency: str
    historical_facts: Dict[str, Any]


@dataclass(frozen=True)
class BalanceSheetPrediction:
    """Predicted balance sheet line items."""

    current_assets: float
    non_current_assets: float
    total_assets: float
    current_liabilities: float
    non_current_liabilities: float
    total_liabilities: float
    total_equity: float
    cash_and_equivalents: float
    accounts_receivable: float
    inventory: float
    property_plant_equipment: float
    goodwill_intangibles: float


class AzureBalanceSheetPredictor:
    """Calls an Azure API endpoint to predict balance sheet elements."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.endpoint = (
            endpoint or os.getenv("AZURE_BALANCE_SHEET_ENDPOINT", "")
        ).strip()
        self.timeout_seconds = timeout_seconds

        if not self.endpoint:
            raise ValueError("Missing Azure config. Set AZURE_BALANCE_SHEET_ENDPOINT.")

    def predict(
        self,
        inputs: BalanceSheetInputs,
        message: str,
        parameters: Dict[str, Any],
    ) -> BalanceSheetPrediction:
        """Predict balance sheet elements with the Azure API."""
        payload = self._build_payload(inputs, message, parameters)
        response_text = self._post_json(payload)
        data = self._extract_json(response_text)
        return self._parse_prediction(data)

    def _build_payload(
        self,
        inputs: BalanceSheetInputs,
        message: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = {
            "task": (
                "Predict balance sheet elements for next fiscal year. "
                "Return ONLY a single JSON object with numeric values."
            ),
            "company": {
                "name": inputs.company_name,
                "ticker": inputs.ticker,
                "fiscal_year": inputs.fiscal_year,
                "currency": inputs.currency,
            },
            "historical_facts": inputs.historical_facts,
            "output_schema": {
                "current_assets": "float",
                "non_current_assets": "float",
                "total_assets": "float",
                "current_liabilities": "float",
                "non_current_liabilities": "float",
                "total_liabilities": "float",
                "total_equity": "float",
                "cash_and_equivalents": "float",
                "accounts_receivable": "float",
                "inventory": "float",
                "property_plant_equipment": "float",
                "goodwill_intangibles": "float",
            },
            "constraints": [
                "total_assets = current_assets + non_current_assets",
                "total_liabilities = current_liabilities + non_current_liabilities",
                "total_assets = total_liabilities + total_equity",
            ],
        }
        payload: Dict[str, Any] = {
            "message": message,
            "body": {"prompt": json.dumps(prompt), "parameters": parameters},
        }

        return payload

    def _post_json(self, payload: Dict[str, Any]) -> str:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                return resp.read().decode("utf-8")
        except Exception as exc:
            raise RuntimeError(f"Azure API request failed: {exc}") from exc

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

    def _parse_prediction(self, data: Dict[str, Any]) -> BalanceSheetPrediction:
        if "raw_response" in data:
            raise ValueError(
                "Model response was not JSON. Raw response: " f"{data['raw_response']}"
            )

        def safe_float(value: Any, field: str) -> float:
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid value for {field}: {value}") from exc

        return BalanceSheetPrediction(
            current_assets=safe_float(data.get("current_assets"), "current_assets"),
            non_current_assets=safe_float(
                data.get("non_current_assets"), "non_current_assets"
            ),
            total_assets=safe_float(data.get("total_assets"), "total_assets"),
            current_liabilities=safe_float(
                data.get("current_liabilities"), "current_liabilities"
            ),
            non_current_liabilities=safe_float(
                data.get("non_current_liabilities"), "non_current_liabilities"
            ),
            total_liabilities=safe_float(
                data.get("total_liabilities"), "total_liabilities"
            ),
            total_equity=safe_float(data.get("total_equity"), "total_equity"),
            cash_and_equivalents=safe_float(
                data.get("cash_and_equivalents"), "cash_and_equivalents"
            ),
            accounts_receivable=safe_float(
                data.get("accounts_receivable"), "accounts_receivable"
            ),
            inventory=safe_float(data.get("inventory"), "inventory"),
            property_plant_equipment=safe_float(
                data.get("property_plant_equipment"), "property_plant_equipment"
            ),
            goodwill_intangibles=safe_float(
                data.get("goodwill_intangibles"), "goodwill_intangibles"
            ),
        )


def run_azure_balance_sheet_example() -> None:
    """Example invocation using placeholder historical data."""
    predictor = AzureBalanceSheetPredictor()

    inputs = BalanceSheetInputs(
        company_name="Apple Inc.",
        ticker="AAPL",
        fiscal_year=2025,
        currency="USD",
        historical_facts={
            "2023": {
                "total_assets": 3.52755e11,
                "total_liabilities": 2.86083e11,
                "total_equity": 6.6672e10,
                "current_assets": 1.43566e11,
                "non_current_assets": 2.09189e11,
                "current_liabilities": 1.57456e11,
                "non_current_liabilities": 1.28627e11,
                "cash_and_equivalents": 2.3646e10,
                "accounts_receivable": 6.0932e10,
                "inventory": 4.946e9,
                "property_plant_equipment": 4.38e10,
                "goodwill_intangibles": 5.96e10,
            }
        },
    )

    parameters = {"temperature": 0, "max_tokens": 10000, "top_p": 1}

    prediction = predictor.predict(
        inputs, message="gen-ai-response", parameters=parameters
    )
    print("Balance Sheet Prediction:", prediction)


if __name__ == "__main__":
    run_azure_balance_sheet_example()
