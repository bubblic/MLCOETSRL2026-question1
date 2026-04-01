"""Type contracts and boundary validation for the financial forecast system.

Provides :class:`RecurrentState` and :class:`ForecastInputs` TypedDicts
that define the dict-key contracts used by ``forecast_step``,
``TrajectorySimulator``, and ``state_index`` utilities.

Also provides :func:`validate_recurrent_state` and
:func:`validate_forecast_inputs` for runtime key-checking at system
boundaries (model preparation, simulator entry) — never inside
``@tf.function`` hot paths.

Why TypedDict instead of dataclass?
    ``@tf.function`` traces Python dicts natively into TensorFlow graphs.
    TypedDicts add static type safety (mypy/pyright) with zero runtime
    overhead, complemented by explicit boundary validators for runtime
    correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


# ------------------------------------------------------------------
# TypedDict contracts
# ------------------------------------------------------------------


class RecurrentState(TypedDict):
    """Dict contract for the 14-field recurrent state vector.

    Keys match ``RECURRENT_KEYS`` in
    :mod:`financial_forecast.inference.state_index`.
    """

    nca: tf.Tensor
    advance_payments_purchases: tf.Tensor
    accounts_receivable: tf.Tensor
    inventory: tf.Tensor
    cash: tf.Tensor
    investment_in_market_securities: tf.Tensor
    accounts_payable: tf.Tensor
    advance_payments_sales: tf.Tensor
    effective_st_debt: tf.Tensor
    current_lt_debt: tf.Tensor
    non_current_liabilities: tf.Tensor
    equity: tf.Tensor
    net_income: tf.Tensor
    dividends: tf.Tensor


class ForecastInputs(TypedDict):
    """Dict contract for single-period forecast inputs."""

    sales_t: tf.Tensor
    year: tf.Tensor
    cum_inflation: tf.Tensor


# ------------------------------------------------------------------
# Training data contract
# ------------------------------------------------------------------


@dataclass(frozen=True)
class HistoricalTrainingData:
    """All historical time series needed for training, already scaled.

    Consolidates the 20+ positional tensor arguments that
    ``PolicyTrainer`` and ``StructuralTrainer`` previously accepted
    via ``**kwargs`` into a single typed container.
    """

    sales: tf.Tensor
    purchases: tf.Tensor
    cogs: tf.Tensor
    nca: tf.Tensor
    depreciation: tf.Tensor
    advance_payments_sales: tf.Tensor
    advance_payments_purchases: tf.Tensor
    accounts_receivable: tf.Tensor
    accounts_payable: tf.Tensor
    inventory: tf.Tensor
    cash: tf.Tensor
    ims: tf.Tensor
    net_income: tf.Tensor
    dividends: tf.Tensor
    stock_buyback: tf.Tensor
    opex: tf.Tensor
    tax: tf.Tensor
    effective_st_debt: tf.Tensor
    current_lt_debt: tf.Tensor
    non_current_liabilities: tf.Tensor
    interest_payment: tf.Tensor
    ms_return: tf.Tensor
    equity: tf.Tensor
    inflation: tf.Tensor
    years: tf.Tensor


# ------------------------------------------------------------------
# Expected key sets (derived from TypedDict annotations)
# ------------------------------------------------------------------

_RECURRENT_STATE_KEYS = frozenset(RecurrentState.__annotations__)
_FORECAST_INPUTS_KEYS = frozenset(ForecastInputs.__annotations__)


# ------------------------------------------------------------------
# Boundary validators
# ------------------------------------------------------------------


def validate_recurrent_state(state: dict, context: str = "") -> None:
    """Check that *state* has exactly the 14 expected recurrent-state keys.

    Call once at system boundaries (``prepare()``, simulator entry),
    **never** inside ``@tf.function`` compiled paths.

    Args:
        state: Dict to validate.
        context: Optional label included in error messages (e.g.
            ``"prepare"`` or ``"TrajectorySimulator.run"``).

    Raises:
        KeyError: If keys are missing or unexpected keys are present.
    """
    actual = frozenset(state.keys())
    prefix = f"[{context}] " if context else ""

    missing = _RECURRENT_STATE_KEYS - actual
    if missing:
        raise KeyError(f"{prefix}RecurrentState missing keys: {sorted(missing)}")

    extra = actual - _RECURRENT_STATE_KEYS
    if extra:
        raise KeyError(f"{prefix}RecurrentState has unexpected keys: {sorted(extra)}")


def validate_forecast_inputs(inputs: dict, context: str = "") -> None:
    """Check that *inputs* has exactly the 3 expected forecast-input keys.

    Args:
        inputs: Dict to validate.
        context: Optional label included in error messages.

    Raises:
        KeyError: If keys are missing or unexpected keys are present.
    """
    actual = frozenset(inputs.keys())
    prefix = f"[{context}] " if context else ""

    missing = _FORECAST_INPUTS_KEYS - actual
    if missing:
        raise KeyError(f"{prefix}ForecastInputs missing keys: {sorted(missing)}")

    extra = actual - _FORECAST_INPUTS_KEYS
    if extra:
        raise KeyError(f"{prefix}ForecastInputs has unexpected keys: {sorted(extra)}")
