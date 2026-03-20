"""LLM-only balance-sheet forecasting with accounting identity enforcement.

This module provides:

* ``ForecastInputs`` -- a frozen dataclass encapsulating inputs for multi-year
  balance-sheet forecasting.
* ``AzureReasoningBalanceSheetForecaster`` -- a forecaster that calls an Azure
  reasoning LLM, parses the structured response, and enforces the accounting
  identity exactly for every forecast year.
* Helper functions for loading historical data, plotting results, and running
  the full end-to-end forecast pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np  # Only used for plotting
import tensorflow as tf
from dotenv import load_dotenv

from financial_forecast.clients.azure_llm_client import AzureLLMClient
from financial_forecast.data.aapl.financial_statements import get_financial_statements

load_dotenv()

BALANCE_SHEET_KEYS = [
    "inventory",
    "nca",
    "accounts_receivable",
    "cash",
    "investment_in_market_securities",
    "advance_payments_purchases",
    "accounts_payable",
    "advance_payments_sales",
    "current_liabilities",
    "non_current_liabilities",
    "equity",
]

ADDITIONAL_FORECAST_KEYS = [
    "dividends",
    "net_income",
    "sales",
    "cogs",
    "depreciation",
    "opex",
    "tax",
    "stock_buyback",
]

ELEMENT_KEYS = BALANCE_SHEET_KEYS + ADDITIONAL_FORECAST_KEYS
LLM_AMOUNT_SCALE = 1e12


@dataclass(frozen=True)
class ForecastInputs:
    """Inputs for multi-year balance-sheet forecasting.

    Attributes
    ----------
    historical_values : dict
        Mapping of element keys to TensorFlow tensors of historical values.
    historical_years : tf.Tensor
        Tensor of historical year labels.
    forecast_horizon : int
        Number of future years to forecast.
    blind_mode : bool
        If ``True``, hide calendar years and company identity from the LLM.
    company_name : str or None
        Company name (used in non-blind mode).
    ticker : str or None
        Stock ticker symbol (used in non-blind mode).
    currency : str
        Currency of the financial data.
    """

    historical_values: Dict[str, tf.Tensor]
    historical_years: tf.Tensor
    forecast_horizon: int
    blind_mode: bool = True
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    currency: str = "USD"


class AzureReasoningBalanceSheetForecaster:
    """Uses an Azure reasoning LLM to forecast balance-sheet elements.

    The forecaster builds a structured prompt from historical financial data,
    sends it to an Azure-hosted LLM, parses the multi-year response, and
    enforces the accounting identity by adjusting equity as the residual.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout_seconds: int = 900,
    ) -> None:
        """Initialise the forecaster.

        Parameters
        ----------
        endpoint : str, optional
            Azure LLM endpoint URL.
        timeout_seconds : int
            HTTP timeout for the LLM request.
        """
        self.client = AzureLLMClient(endpoint=endpoint, timeout_seconds=timeout_seconds)

    def forecast(
        self,
        inputs: ForecastInputs,
        message: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, tf.Tensor]:
        """Generate forecast arrays for each element key.

        Parameters
        ----------
        inputs : ForecastInputs
            Historical data and configuration for the forecast.
        message : str
            A label or identifier forwarded to the LLM endpoint.
        parameters : dict, optional
            Model parameters (defaults to temperature=0, max_tokens=12000,
            top_k=1).

        Returns
        -------
        dict
            Mapping of element keys to forecast tensors in original USD scale.
        """
        params = parameters or {"temperature": 0, "max_tokens": 12000, "top_k": 1}
        prompt = self._build_prompt(inputs)
        print("Waiting for LLM response...")
        data = self.client.ask_json(
            message=message,
            prompt=json.dumps(prompt),
            parameters=params,
            reasoning=True,
        )
        # Parse + enforce in scaled (trillion) units for numeric stability.
        forecast_scaled = self._parse_multi_year_response(data, inputs.forecast_horizon)
        self._enforce_identity_inplace(forecast_scaled)
        self._validate_identity(forecast_scaled)
        # Return values back in original USD scale.
        return self._rescale_forecast(forecast_scaled, LLM_AMOUNT_SCALE)

    def _build_prompt(self, inputs: ForecastInputs) -> Dict[str, Any]:
        """Build the structured JSON prompt for the LLM.

        Parameters
        ----------
        inputs : ForecastInputs
            Historical data and configuration.

        Returns
        -------
        dict
            The prompt dictionary to be serialised as JSON.
        """
        historical_rows: Dict[str, Dict[str, float]] = {}
        n_hist = len(inputs.historical_values[ELEMENT_KEYS[0]])
        for idx in range(n_hist):
            row_key = (
                f"t{idx + 1}"
                if inputs.blind_mode
                else str(int(inputs.historical_years[idx]))
            )
            historical_rows[row_key] = {
                key: float(inputs.historical_values[key][idx] / LLM_AMOUNT_SCALE)
                for key in ELEMENT_KEYS
            }

        prompt: Dict[str, Any] = {
            "task": "Forecast all required financial elements for future years. Return ONLY valid JSON (no markdown).",
            "historical_facts": historical_rows,
            "value_units": "trillions_of_usd",
            "value_scale_to_usd": LLM_AMOUNT_SCALE,
            "required_elements": ELEMENT_KEYS,
            "accounting_identity": (
                "inventory + nca + accounts_receivable + cash + "
                "investment_in_market_securities + advance_payments_purchases = "
                "accounts_payable + advance_payments_sales + current_liabilities + "
                "non_current_liabilities + equity"
            ),
            "output_schema": {
                "forecast": [
                    {
                        "inventory": "float",
                        "nca": "float",
                        "accounts_receivable": "float",
                        "cash": "float",
                        "investment_in_market_securities": "float",
                        "advance_payments_purchases": "float",
                        "accounts_payable": "float",
                        "advance_payments_sales": "float",
                        "current_liabilities": "float",
                        "non_current_liabilities": "float",
                        "equity": "float",
                        "dividends": "float",
                        "net_income": "float",
                        "sales": "float",
                        "cogs": "float",
                        "depreciation": "float",
                        "opex": "float",
                        "tax": "float",
                        "stock_buyback": "float",
                    }
                ]
            },
        }
        if inputs.blind_mode:
            prompt["task"] = (
                "Forecast all required financial elements for future years. "
                "Do not infer company identity; extrapolate only from provided series. "
                "Return ONLY valid JSON (no markdown)."
            )
            prompt["time_axis_note"] = (
                "Historical rows are generic time steps (t1..tn), not calendar years."
            )
            prompt["forecast_horizon_steps"] = inputs.forecast_horizon
        else:
            start_year = int(inputs.historical_years[-1]) + 1
            end_year = start_year + inputs.forecast_horizon - 1
            prompt["company"] = {
                "name": inputs.company_name or "Unknown",
                "ticker": inputs.ticker or "N/A",
                "currency": inputs.currency,
            }
            prompt["forecast_horizon_years"] = inputs.forecast_horizon
            prompt["forecast_year_range"] = [start_year, end_year]
            # Optional year in schema for easier non-blind parsing by the model.
            prompt["output_schema"]["forecast"][0]["year"] = "int"

        return prompt

    def _parse_multi_year_response(
        self, data: Dict[str, Any], horizon: int
    ) -> Dict[str, tf.Tensor]:
        """Parse the LLM JSON response into per-element tensors.

        Parameters
        ----------
        data : dict
            The parsed JSON response from the LLM.
        horizon : int
            Expected number of forecast years.

        Returns
        -------
        dict
            Mapping of element keys to ``tf.Tensor`` values (scaled units).

        Raises
        ------
        ValueError
            If the response shape is unsupported or has insufficient data.
        """
        if "raw_response" in data:
            raise ValueError(
                f"Please try again. Model response was not JSON: {data['raw_response']}"
            )

        # Preferred shape: {"forecast": [{year, ...elements...}, ...]}
        if isinstance(data.get("forecast"), list):
            rows = data["forecast"]
            if len(rows) < horizon:
                raise ValueError(
                    f"Forecast length {len(rows)} is shorter than required horizon {horizon}."
                )
            parsed = {
                key: tf.constant(
                    [self._safe_float(row.get(key), key) for row in rows[:horizon]],
                    dtype=tf.float64,
                )
                for key in ELEMENT_KEYS
            }
            return parsed

        # Alternate shape: top-level arrays keyed by element name
        if all(key in data for key in ELEMENT_KEYS):
            parsed = {}
            for key in ELEMENT_KEYS:
                values = data[key]
                if not isinstance(values, list) or len(values) < horizon:
                    raise ValueError(
                        f"Invalid list for '{key}'. Need at least {horizon} values."
                    )
                parsed[key] = tf.constant(
                    [self._safe_float(v, key) for v in values[:horizon]],
                    dtype=tf.float64,
                )
            return parsed

        raise ValueError(
            "Unsupported forecast JSON shape. Expected {'forecast': [...]} "
            "or top-level arrays for all required elements."
        )

    @staticmethod
    def _safe_float(value: Any, field: str) -> float:
        """Convert a value to float, raising a clear error on failure.

        Parameters
        ----------
        value : Any
            The value to convert.
        field : str
            The field name (used in the error message).

        Returns
        -------
        float
            The converted value.
        """
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for '{field}': {value}") from exc

    @staticmethod
    def _enforce_identity_inplace(forecast: Dict[str, tf.Tensor]) -> None:
        """Enforce the accounting identity by solving equity as the residual.

        Modifies *forecast* in place so that total assets equal total
        liabilities plus equity for every forecast year.

        Parameters
        ----------
        forecast : dict
            Mutable mapping of element keys to forecast tensors.
        """
        lhs = (
            forecast["inventory"]
            + forecast["nca"]
            + forecast["accounts_receivable"]
            + forecast["cash"]
            + forecast["investment_in_market_securities"]
            + forecast["advance_payments_purchases"]
        )
        rhs_without_equity = (
            forecast["accounts_payable"]
            + forecast["advance_payments_sales"]
            + forecast["current_liabilities"]
            + forecast["non_current_liabilities"]
        )
        # Force exact identity by solving equity as residual.
        forecast["equity"] = lhs - rhs_without_equity

    @staticmethod
    def _validate_identity(forecast: Dict[str, tf.Tensor]) -> None:
        """Validate that the accounting identity holds within tolerance.

        Parameters
        ----------
        forecast : dict
            Mapping of element keys to forecast tensors.

        Raises
        ------
        ValueError
            If the identity ``|LHS - RHS| >= 1e-6`` for any forecast year.
        """
        lhs = (
            forecast["inventory"]
            + forecast["nca"]
            + forecast["accounts_receivable"]
            + forecast["cash"]
            + forecast["investment_in_market_securities"]
            + forecast["advance_payments_purchases"]
        )
        rhs = (
            forecast["accounts_payable"]
            + forecast["advance_payments_sales"]
            + forecast["current_liabilities"]
            + forecast["non_current_liabilities"]
            + forecast["equity"]
        )
        if not tf.reduce_all(tf.abs(lhs - rhs) < 1e-6):
            raise ValueError("Accounting identity validation failed after enforcement.")

    @staticmethod
    def _rescale_forecast(
        forecast: Dict[str, tf.Tensor], factor: float
    ) -> Dict[str, tf.Tensor]:
        """Multiply all forecast tensors by *factor* to restore original scale.

        Parameters
        ----------
        forecast : dict
            Mapping of element keys to forecast tensors in scaled units.
        factor : float
            The scale factor to apply.

        Returns
        -------
        dict
            Rescaled forecast tensors.
        """
        return {key: values * factor for key, values in forecast.items()}


def load_historical_balance_sheet() -> Dict[str, tf.Tensor]:
    """Load and map historical Apple financial fields.

    Returns
    -------
    dict
        Mapping of element keys (plus ``"years"``) to TensorFlow tensors
        containing the historical Apple balance-sheet data.
    """
    data = get_financial_statements()
    mapped = {
        "inventory": data["inventory"],
        "nca": data["nca"],
        "accounts_receivable": data["accounts_receivable"],
        "cash": data["cash"],
        "investment_in_market_securities": data["ims"],
        "advance_payments_purchases": data["advance_payments_purchases"],
        "accounts_payable": data["accounts_payable"],
        "advance_payments_sales": data["advance_payments_sales"],
        "current_liabilities": data["current_liabilities"],
        "non_current_liabilities": data["non_current_liabilities"],
        "equity": data["equity"],
        "dividends": data["dividends"],
        "net_income": data["net_income"],
        "sales": data["sales"],
        "cogs": data["cogs"],
        "depreciation": data["depreciation"],
        "opex": data["opex"],
        "tax": data["tax"],
        "stock_buyback": data["stock_buyback"],
    }
    mapped["years"] = data["years"]
    return mapped


def _get_output_path(file_name: str) -> str:
    """Return a path inside the ``training_results`` directory.

    Creates the directory if it does not already exist.

    Parameters
    ----------
    file_name : str
        The file name to place inside the output directory.

    Returns
    -------
    str
        The full path to the output file.
    """
    output_dir = "training_results"
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, file_name)


def plot_forecast_elements(
    historical_years: tf.Tensor,
    historical_data: Dict[str, tf.Tensor],
    forecast_years: tf.Tensor,
    forecast_data: Dict[str, tf.Tensor],
    holdout_year: Optional[int] = None,
    holdout_actual: Optional[Dict[str, float]] = None,
    mode_label: str = "blind",
    show_plot: bool = False,
) -> None:
    """Plot all forecast elements in a multi-panel chart.

    Each element is shown with its historical values, LLM forecast, and
    (optionally) a holdout actual data point for backtesting.

    Parameters
    ----------
    historical_years : tf.Tensor
        Tensor of historical year labels.
    historical_data : dict
        Historical values keyed by element name.
    forecast_years : tf.Tensor
        Tensor of forecast year labels.
    forecast_data : dict
        Forecast values keyed by element name.
    holdout_year : int, optional
        Year of the holdout data point.
    holdout_actual : dict, optional
        Actual values for the holdout year, keyed by element name.
    mode_label : str
        Label for the forecast mode (e.g. ``"blind"``).
    show_plot : bool
        If ``True``, display the plot interactively.
    """
    elements = list(forecast_data.keys())
    n_elements = len(elements)
    ncols = 3
    nrows = (n_elements + ncols - 1) // ncols

    display_names = {
        "inventory": "Inventory",
        "nca": "Non-Current Assets",
        "accounts_receivable": "Accounts Receivable",
        "cash": "Cash",
        "investment_in_market_securities": "Investment in Market Securities",
        "advance_payments_purchases": "Advance Payments (Purchases)",
        "accounts_payable": "Accounts Payable",
        "advance_payments_sales": "Advance Payments (Sales)",
        "current_liabilities": "Current Liabilities",
        "non_current_liabilities": "Non-Current Liabilities",
        "equity": "Stockholders' Equity",
        "dividends": "Dividends",
        "net_income": "Net Income",
        "sales": "Sales",
        "cogs": "Cost of Goods Sold",
        "depreciation": "Depreciation",
        "opex": "Operating Expenses",
        "tax": "Tax",
        "stock_buyback": "Stock Buyback",
    }

    fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
    axs = np.array(axs).reshape(-1)

    for idx, key in enumerate(elements):
        ax = axs[idx]
        ax.plot(
            historical_years,
            historical_data[key],
            "ko-",
            label="Historical",
            markersize=5,
            linewidth=1.5,
        )
        ax.plot(
            forecast_years,
            forecast_data[key],
            "s-",
            color="tab:blue",
            label="Forecast",
            markersize=5,
            linewidth=1.5,
        )
        if holdout_year is not None and holdout_actual is not None:
            ax.plot(
                [holdout_year],
                [holdout_actual[key]],
                "x",
                color="tab:red",
                label="Holdout Actual",
                markersize=7,
                markeredgewidth=2,
            )
        ax.set_title(display_names.get(key, key), fontsize=11, fontweight="bold")
        ax.set_ylabel("USD")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        ax.tick_params(axis="x", rotation=45)

    for idx in range(n_elements, len(axs)):
        axs[idx].set_visible(False)

    fig.suptitle(
        f"LLM Financial Forecast ({mode_label}, Identity-Constrained)",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = _get_output_path(
        f"llm_financial_forecast_{mode_label}_{timestamp}.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {output_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()


def run_llm_balance_sheet_forecast(
    horizon_years: int = 10,
    message: str = "gen-ai-response",
    blind_mode: bool = True,
    show_plot: bool = False,
) -> Dict[str, tf.Tensor]:
    """Run the end-to-end LLM balance-sheet forecast pipeline.

    Steps performed:

    1. Load historical Apple balance-sheet data.
    2. Hold out the final observed year for backtesting.
    3. Call the Azure reasoning LLM to generate a multi-year forecast.
    4. Print the forecast and holdout backtest errors.
    5. Plot all elements and save the figure.

    Parameters
    ----------
    horizon_years : int
        Number of years to forecast.
    message : str
        Label forwarded to the LLM endpoint.
    blind_mode : bool
        Whether to hide calendar years and company identity from the LLM.
    show_plot : bool
        If ``True``, display the plot interactively.

    Returns
    -------
    dict
        Mapping of element keys to forecast tensors in USD.
    """
    hist = load_historical_balance_sheet()
    all_years = tf.cast(hist["years"], tf.int32)
    all_values = {k: hist[k] for k in ELEMENT_KEYS}

    # Hold out the final observed year from LLM inputs for a backtest point.
    holdout_year = int(all_years[-1])
    model_historical_years = all_years[:-1]
    model_historical_values = {k: all_values[k][:-1] for k in ELEMENT_KEYS}
    holdout_actual = {k: float(all_values[k][-1]) for k in ELEMENT_KEYS}

    inputs = ForecastInputs(
        historical_values=model_historical_values,
        historical_years=model_historical_years,
        forecast_horizon=horizon_years,
        blind_mode=blind_mode,
        company_name="Apple Inc.",
        ticker="AAPL",
        currency="USD",
    )

    mode_label = "blind" if blind_mode else "non_blind"
    print(f"\nRunning mode: {mode_label}")

    forecaster = AzureReasoningBalanceSheetForecaster()
    forecast = forecaster.forecast(inputs=inputs, message=message)

    forecast_years = tf.range(
        holdout_year,
        holdout_year + horizon_years,
        dtype=tf.int32,
    )

    print("\n10-year LLM forecast (USD):")
    for i, year in enumerate(forecast_years):
        row = {key: float(forecast[key][i]) for key in ELEMENT_KEYS}
        print(f"{year}: {json.dumps(row)}")

    print(f"\nHoldout backtest ({holdout_year}) absolute percentage errors:")
    for key in ELEMENT_KEYS:
        actual = holdout_actual[key]
        pred = float(forecast[key][0])
        ape = abs(pred - actual) / max(abs(actual), 1.0)
        print(f"{key}: {ape:.2%} (pred={pred:.3e}, actual={actual:.3e})")

    plot_forecast_elements(
        historical_years=model_historical_years,
        historical_data=model_historical_values,
        forecast_years=forecast_years,
        forecast_data=forecast,
        holdout_year=holdout_year,
        holdout_actual=holdout_actual,
        mode_label=mode_label,
        show_plot=show_plot,
    )

    return forecast


if __name__ == "__main__":
    run_llm_balance_sheet_forecast(horizon_years=10, show_plot=False, blind_mode=True)
