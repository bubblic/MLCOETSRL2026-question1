"""Integer indices for packed recurrent-state and diagnostic tensors.

These constants are the single source of truth for the tensor layout used by
:meth:`TrainableFinancialModel.forecast_step_compiled` and
:func:`run_monte_carlo_forecast`.
"""

from typing import Dict

import tensorflow as tf

from financial_forecast.types import RecurrentState

# ---- Recurrent state: [n_samples, 14] ----
R_NCA = 0
R_ADV_PP = 1
R_AR = 2
R_INV = 3
R_CASH = 4
R_IMS = 5
R_AP = 6
R_ADV_PS = 7
R_EFF_ST_DEBT = 8
R_CUR_LT_DEBT = 9
R_NCL = 10
R_EQUITY = 11
R_NET_INCOME = 12
R_DIVIDENDS = 13

N_RECURRENT = 14

RECURRENT_KEYS = (
    "nca",
    "advance_payments_purchases",
    "accounts_receivable",
    "inventory",
    "cash",
    "investment_in_market_securities",
    "accounts_payable",
    "advance_payments_sales",
    "effective_st_debt",
    "current_lt_debt",
    "non_current_liabilities",
    "equity",
    "net_income",
    "dividends",
)

# ---- Diagnostic output: [n_samples, 25] ----
D_TOTAL_ASSETS = 0
D_NCA = 1
D_ADV_PP = 2
D_AR = 3
D_INV = 4
D_CASH = 5
D_IMS = 6
D_AP = 7
D_ADV_PS = 8
D_EFF_ST_DEBT = 9
D_CUR_LT_DEBT = 10
D_NCL = 11
D_EQUITY = 12
D_NET_INCOME = 13
D_DEPRECIATION = 14
D_COGS = 15
D_OPEX = 16
D_TAX = 17
D_MS_RETURN = 18
D_INTEREST_PAYMENT = 19
D_DIVIDENDS = 20
D_STOCK_BUYBACK = 21
D_NEW_LT_LOAN = 22
D_EQUITY_FINANCING = 23
D_LIQUIDITY_DEFICIT_ST = 24
D_LIQUIDITY_CHECK = 25
D_CHECK = 26

N_DIAGNOSTIC = 27

DIAGNOSTIC_KEYS = (
    "total_assets",
    "nca",
    "advance_payments_purchases",
    "accounts_receivable",
    "inventory",
    "cash",
    "investment_in_market_securities",
    "accounts_payable",
    "advance_payments_sales",
    "effective_st_debt",
    "current_lt_debt",
    "non_current_liabilities",
    "equity",
    "net_income",
    "depreciation",
    "cogs",
    "opex",
    "tax",
    "ms_return",
    "interest_payment",
    "dividends",
    "stock_buyback",
    "new_long_term_loan",
    "equity_financing",
    "liquidity_deficit_st",
    "liquidity_check",
    "check",
)


def state_dict_to_tensor(
    state_dict: RecurrentState,
) -> tf.Tensor:
    """Convert a scalar state dict to a 1-D tensor of shape ``[14]``.

    Args:
        state_dict: Dict mapping :data:`RECURRENT_KEYS` to scalar tensors.

    Returns:
        1-D float64 tensor of shape ``[14]``.
    """
    return tf.stack([tf.cast(state_dict[k], tf.float64) for k in RECURRENT_KEYS])


def initial_state_to_batched(
    state_dict: RecurrentState,
    n_samples: int,
) -> tf.Tensor:
    """Broadcast a scalar state dict to a batched tensor ``[n_samples, 14]``.

    Args:
        state_dict: Dict mapping :data:`RECURRENT_KEYS` to scalar tensors.
        n_samples: Number of Monte Carlo samples.

    Returns:
        float64 tensor of shape ``[n_samples, 14]``.
    """
    single = state_dict_to_tensor(state_dict)
    return tf.tile(tf.expand_dims(single, 0), [n_samples, 1])
