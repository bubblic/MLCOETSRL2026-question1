"""Tests for llm-only-balance-sheet-forecast.py.


Run the script by:
python -m pytest -q test_llm_only_balance_sheet_forecast.py
"""

import importlib.util
from pathlib import Path
import sys

import pytest
import tensorflow as tf


@pytest.fixture
def llm_bs_module():
    """Load the hyphenated module via importlib."""
    module_path = Path(__file__).resolve().parent / "llm-only-balance-sheet-forecast.py"
    spec = importlib.util.spec_from_file_location(
        "llm_only_balance_sheet_forecast", module_path
    )
    module_name = "llm_only_balance_sheet_forecast"
    loaded = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    # Dataclasses with postponed annotations expect the module to be present.
    sys.modules[module_name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_safe_float_rejects_invalid_value(llm_bs_module):
    with pytest.raises(ValueError, match="Invalid value"):
        llm_bs_module.AzureReasoningBalanceSheetForecaster._safe_float("bad", "equity")


def test_parse_multi_year_response_forecast_shape(llm_bs_module):
    forecaster = llm_bs_module.AzureReasoningBalanceSheetForecaster(
        endpoint="http://dummy"
    )
    row = {k: 1.0 for k in llm_bs_module.ELEMENT_KEYS}
    parsed = forecaster._parse_multi_year_response({"forecast": [row, row]}, horizon=2)
    assert set(parsed.keys()) == set(llm_bs_module.ELEMENT_KEYS)
    assert parsed["equity"].shape == (2,)


def test_parse_multi_year_response_top_level_arrays(llm_bs_module):
    forecaster = llm_bs_module.AzureReasoningBalanceSheetForecaster(
        endpoint="http://dummy"
    )
    payload = {k: [1.0, 2.0, 3.0] for k in llm_bs_module.ELEMENT_KEYS}
    parsed = forecaster._parse_multi_year_response(payload, horizon=2)
    assert parsed["sales"].numpy().tolist() == [1.0, 2.0]


def test_parse_multi_year_response_raw_response_raises(llm_bs_module):
    forecaster = llm_bs_module.AzureReasoningBalanceSheetForecaster(
        endpoint="http://dummy"
    )
    with pytest.raises(ValueError, match="not JSON"):
        forecaster._parse_multi_year_response({"raw_response": "text"}, horizon=1)


def test_identity_enforcement_and_validation(llm_bs_module):
    forecast = {
        k: tf.constant([1.0, 2.0], dtype=tf.float64) for k in llm_bs_module.ELEMENT_KEYS
    }
    # Break equity then enforce.
    forecast["equity"] = tf.constant([999.0, 999.0], dtype=tf.float64)
    llm_bs_module.AzureReasoningBalanceSheetForecaster._enforce_identity_inplace(
        forecast
    )
    llm_bs_module.AzureReasoningBalanceSheetForecaster._validate_identity(forecast)
    assert tf.reduce_all(tf.math.is_finite(forecast["equity"]))


def test_load_historical_balance_sheet_mapping(llm_bs_module, monkeypatch):
    fake = {
        "inventory": tf.constant([1.0]),
        "nca": tf.constant([2.0]),
        "accounts_receivable": tf.constant([3.0]),
        "cash": tf.constant([4.0]),
        "ims": tf.constant([5.0]),
        "advance_payments_purchases": tf.constant([6.0]),
        "accounts_payable": tf.constant([7.0]),
        "advance_payments_sales": tf.constant([8.0]),
        "current_liabilities": tf.constant([9.0]),
        "non_current_liabilities": tf.constant([10.0]),
        "equity": tf.constant([11.0]),
        "dividends": tf.constant([12.0]),
        "net_income": tf.constant([13.0]),
        "sales": tf.constant([14.0]),
        "cogs": tf.constant([15.0]),
        "depreciation": tf.constant([16.0]),
        "opex": tf.constant([17.0]),
        "tax": tf.constant([18.0]),
        "stock_buyback": tf.constant([19.0]),
        "years": tf.constant([2024]),
    }
    monkeypatch.setattr(llm_bs_module, "get_apple_historical_data", lambda: fake)
    mapped = llm_bs_module.load_historical_balance_sheet()
    assert mapped["investment_in_market_securities"][0] == 5.0
    assert mapped["years"][0] == 2024


def test_run_llm_balance_sheet_forecast_orchestrates(monkeypatch, llm_bs_module):
    hist = {
        k: tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)
        for k in llm_bs_module.ELEMENT_KEYS
    }
    hist["years"] = tf.constant([2022, 2023, 2024], dtype=tf.float64)

    monkeypatch.setattr(llm_bs_module, "load_historical_balance_sheet", lambda: hist)
    dummy_forecast = {
        k: tf.constant([10.0, 11.0], dtype=tf.float64)
        for k in llm_bs_module.ELEMENT_KEYS
    }

    class DummyForecaster:
        def forecast(self, inputs, message):
            assert inputs.forecast_horizon == 2
            return dummy_forecast

    monkeypatch.setattr(
        llm_bs_module, "AzureReasoningBalanceSheetForecaster", lambda: DummyForecaster()
    )
    plot_calls = []
    monkeypatch.setattr(
        llm_bs_module,
        "plot_forecast_elements",
        lambda **kwargs: plot_calls.append(kwargs),
    )

    out = llm_bs_module.run_llm_balance_sheet_forecast(
        horizon_years=2, show_plot=False, blind_mode=True
    )
    assert out["equity"].numpy().tolist() == [10.0, 11.0]
    assert len(plot_calls) == 1
