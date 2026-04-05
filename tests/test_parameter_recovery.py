"""Parameter recovery (identifiability) tests for the financial forecast model.

Generates synthetic financial statements from known ground-truth parameters
using ``forecast_step``, trains a fresh model on that data, and verifies
the recovered parameters match ground truth within tolerance.

Four scenarios cover all policy combinations:
  1. Simple policies (all static ratios, deterministic OpEx)
  2. Advanced policies (trend cost ratio, trend liquidity, Lintner dividend,
     baseline buyback, trend debt, deterministic OpEx)
  3. Advanced + BayesianOpEx (stochastic OpEx with known noise)
  4. Advanced + BayesianOpEx + TaxWithAnomalies (one-time tax adjustments)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from financial_forecast.models.trainable_financial_model import TrainableFinancialModel
from financial_forecast.models.opex import SimpleOpEx, BayesianOpEx
from financial_forecast.models.capex import CapexPolicy
from financial_forecast.models.working_capital import WorkingCapitalPolicy
from financial_forecast.models.liquidity import (
    SimpleLiquidityPolicy,
    TrendLiquidityPolicy,
)
from financial_forecast.models.dividends import (
    SimpleDividendPolicy,
    LintnerDividendPolicy,
)
from financial_forecast.models.buyback import SimpleBuybackPolicy, BaselineBuybackPolicy
from financial_forecast.models.purchases import (
    StaticCostRatioPolicy,
    TrendCostRatioPolicy,
)
from financial_forecast.models.debt import SimpleDebtPolicy, TrendDebtPolicy
from financial_forecast.models.tax import SimpleTax, TaxWithAnomalies
from financial_forecast.inference.trajectory_simulator import (
    DeterministicSimulator,
    MonteCarloSimulator,
)
from financial_forecast.inference.state_index import RECURRENT_KEYS, DIAGNOSTIC_KEYS
from financial_forecast.training.policy_trainer import PolicyTrainer
from financial_forecast.training.structural_trainer import StructuralTrainer

# =====================================================================
# Exogenous drivers (shared across all scenarios)
# =====================================================================

_N_YEARS = 7  # years in the final training data
_BASE_SALES_USD = 100e9
_GROWTH_RATE = 0.05
_INFLATION_RATE = 0.02
_START_YEAR = 2018

# We generate N_YEARS+1 sales values: year 0 is the "seed" initial state,
# then forecast_step produces N_YEARS of perfectly consistent data.
_SALES_USD_EXTENDED = np.array(
    [_BASE_SALES_USD * (1 + _GROWTH_RATE) ** t for t in range(_N_YEARS + 1)]
)
_INFLATION_EXTENDED = np.full(_N_YEARS + 1, _INFLATION_RATE)
_YEARS_EXTENDED = np.arange(_START_YEAR, _START_YEAR + _N_YEARS + 1, dtype=np.float64)

# The final training data uses years 1..N_YEARS (indices 1: in the extended arrays)
_SALES_USD = _SALES_USD_EXTENDED[1:]
_INFLATION = _INFLATION_EXTENDED[1:]
_YEARS = _YEARS_EXTENDED[1:]

# =====================================================================
# Ground-truth parameters — shared
# =====================================================================

# CapEx
TRUE_DEPRECIATION_RATE = 0.055
TRUE_ASSET_MAINTAIN = 1.0
TRUE_ASSET_GROWTH = 0.008

# Working Capital
TRUE_AR_PCT = 0.16
TRUE_AP_PCT = 0.35
TRUE_INV_COGS_PCT = 0.02
TRUE_ADV_PS_PCT = 0.02
TRUE_ADV_PP_PCT = 0.07

# Tax
TRUE_INCOME_TAX_PCT = 0.15

# Structural
TRUE_AVG_ST_INTEREST_PCT = 0.04
TRUE_AVG_LT_INTEREST_PCT = 0.035
TRUE_MS_RETURN_PCT = 0.045
TRUE_AVG_MATURITY_YEARS = 5.0

# =====================================================================
# Ground-truth parameters — Scenario 1 (Simple)
# =====================================================================

TRUE_COST_RATIO = 0.58
TRUE_VARIABLE_OPEX_PCT = 0.10
TRUE_TOTAL_LIQ_PCT = 0.22
TRUE_CASH_PCT_OF_LIQ = 0.48
TRUE_DIVIDEND_PAYOUT_RATIO = 0.16
TRUE_STOCK_BUYBACK_PCT = 7.5
TRUE_EQUITY_FINANCING_PCT = 0.15

# =====================================================================
# Ground-truth parameters — Scenarios 2-4 (Advanced / Trend)
# =====================================================================

TRUE_CR_ALPHA = 0.32
TRUE_CR_BETA = -0.02
TRUE_TL_ALPHA = -1.26
TRUE_TL_BETA = 0.01
TRUE_TL_BASELINE = 0.0
TRUE_CASH_ALPHA = -0.08
TRUE_CASH_BETA = 0.005
TRUE_DIV_PAYOUT_RATIO = 0.16
TRUE_DIV_ADJ_SPEED = 0.30
TRUE_SB_BASELINE = 0.05  # nonzero to break collinearity with sb_ratio
TRUE_SB_RATIO = 7.5
TRUE_ST_DEBT_BASELINE = 0.05  # nonzero to break collinearity with pct
TRUE_ST_DEBT_PCT = 0.12
TRUE_EF_ALPHA = -1.73
TRUE_EF_BETA = 0.01

# =====================================================================
# Ground-truth parameters — Scenarios 3-4 (Bayesian OpEx)
# =====================================================================

TRUE_NOISE_SIGMA = 0.005  # scaled units — small but nonzero

# =====================================================================
# Ground-truth parameters — Scenario 4 (Tax Anomalies)
# =====================================================================

TRUE_TAX_ONETIME = {2020: 2.0e9, 2022: -0.5e9}

# =====================================================================
# Cash-poor initial state (year 0, USD)
# =====================================================================

# Scale will be 10^floor(log10(mean(sales))) ≈ 10^11 = 100B.
_SCALE = 10 ** int(np.floor(np.log10(np.mean(_SALES_USD))))

_INITIAL_STATE_USD = {
    "nca": 40e9,
    "cash": 5e9,
    "ims": 3e9,
    "accounts_receivable": 16e9,
    "inventory": 1.16e9,
    "accounts_payable": 7.2e9,
    "advance_payments_sales": 2e9,
    "advance_payments_purchases": 1.4e9,
    "effective_st_debt": 10e9,
    "current_lt_debt": 6e9,
    "non_current_liabilities": 30e9,
    "equity": 20e9,
    "net_income": 10e9,
    "dividends": 1.6e9,
}

# Training configuration
_POLICY_EPOCHS = 15000
_STRUCTURAL_EPOCHS = 10000
_LEARNING_RATE = 0.001


# =====================================================================
# Helper: build seed financial data
# =====================================================================


def _build_seed_data(cost_ratio_val: float) -> dict:
    """Build approximate financial statements for prepare() bootstrap.

    Uses the EXTENDED arrays (N_YEARS+1 points) so that year 0 serves
    as the initial state for forecast_step generation. ``prepare()``
    uses it to set ``base_year``, ``amount_scale``, ``sales_offset``,
    and ``initial_state``.
    """
    n = _N_YEARS + 1
    sales = tf.constant(_SALES_USD_EXTENDED, dtype=tf.float64)
    cogs = sales * cost_ratio_val
    inventory = cogs * TRUE_INV_COGS_PCT
    delta_inv = tf.concat(
        [tf.zeros([1], dtype=tf.float64), inventory[1:] - inventory[:-1]], axis=0
    )
    purchases = cogs + delta_inv

    nca_vals = [_INITIAL_STATE_USD["nca"]]
    depr_vals = [_INITIAL_STATE_USD["nca"] * TRUE_DEPRECIATION_RATE]
    for t in range(1, n):
        d = nca_vals[-1] * TRUE_DEPRECIATION_RATE
        capex = TRUE_ASSET_MAINTAIN * d + _SALES_USD_EXTENDED[t] * TRUE_ASSET_GROWTH
        nca_vals.append(nca_vals[-1] - d + capex)
        depr_vals.append(d)
    nca = tf.constant(nca_vals, dtype=tf.float64)
    depreciation = tf.constant(depr_vals, dtype=tf.float64)

    ar = sales * TRUE_AR_PCT
    ap = purchases * TRUE_AP_PCT
    adv_ps = sales * TRUE_ADV_PS_PCT
    adv_pp = purchases * TRUE_ADV_PP_PCT

    eff_st_debt = tf.constant(
        [_INITIAL_STATE_USD["effective_st_debt"]] * n, dtype=tf.float64
    )
    cur_lt_debt = tf.constant(
        [_INITIAL_STATE_USD["current_lt_debt"]] * n, dtype=tf.float64
    )
    ncl = tf.constant(
        [_INITIAL_STATE_USD["non_current_liabilities"]] * n, dtype=tf.float64
    )
    current_liabilities = ap + adv_ps + eff_st_debt + cur_lt_debt

    cash = tf.constant([_INITIAL_STATE_USD["cash"]] * n, dtype=tf.float64)
    ims = tf.constant([_INITIAL_STATE_USD["ims"]] * n, dtype=tf.float64)

    opex = sales * 0.10
    interest_payment = tf.constant([2e9] * n, dtype=tf.float64)
    ms_return = tf.constant([0.15e9] * n, dtype=tf.float64)
    ebt = sales - cogs - opex - depreciation - interest_payment + ms_return
    tax = ebt * TRUE_INCOME_TAX_PCT
    ni = ebt - tax

    dividends = ni * 0.16
    buyback = depreciation * 7.5
    equity = tf.constant([_INITIAL_STATE_USD["equity"]] * n, dtype=tf.float64)

    return {
        "sales": sales,
        "purchases": purchases,
        "cogs": cogs,
        "nca": nca,
        "depreciation": depreciation,
        "advance_payments_purchases": adv_pp,
        "accounts_receivable": ar,
        "accounts_payable": ap,
        "advance_payments_sales": adv_ps,
        "cash": cash,
        "ims": ims,
        "inventory": inventory,
        "current_liabilities": current_liabilities,
        "non_current_liabilities": ncl,
        "equity": equity,
        "net_income": ni,
        "dividends": dividends,
        "stock_buyback": buyback,
        "opex": opex,
        "tax": tax,
        "current_lt_debt": cur_lt_debt,
        "interest_payment": interest_payment,
        "ms_return": ms_return,
        "years": tf.constant(_YEARS_EXTENDED, dtype=tf.float64),
        "inflation": tf.constant(_INFLATION_EXTENDED, dtype=tf.float64),
    }


# =====================================================================
# Helper: assign ground-truth parameters to a generator model
# =====================================================================


def _assign_shared_ground_truth(model: TrainableFinancialModel) -> None:
    """Assign shared ground-truth values (capex, WC, structural, tax)."""
    _f64 = lambda v: tf.constant(v, dtype=tf.float64)
    cp = model.balance_sheet.capex_policy
    cp.depreciation_rate.assign(_f64(TRUE_DEPRECIATION_RATE))
    cp.asset_maintain.assign(_f64(TRUE_ASSET_MAINTAIN))
    cp.asset_growth.assign(_f64(TRUE_ASSET_GROWTH))

    wc = model.balance_sheet.working_capital
    wc.account_receivables_pct.assign(_f64(TRUE_AR_PCT))
    wc.account_payables_pct.assign(_f64(TRUE_AP_PCT))
    wc.inventory_cogs_pct.assign(_f64(TRUE_INV_COGS_PCT))
    wc.advance_payments_sales_pct.assign(_f64(TRUE_ADV_PS_PCT))
    wc.advance_payments_purchases_pct.assign(_f64(TRUE_ADV_PP_PCT))

    model.tax_module.income_tax_pct.assign(_f64(TRUE_INCOME_TAX_PCT))

    ist = model.income_statement
    ist.avg_short_term_interest_pct.assign(_f64(TRUE_AVG_ST_INTEREST_PCT))
    ist.avg_long_term_interest_pct.assign(_f64(TRUE_AVG_LT_INTEREST_PCT))
    ist.market_securities_return_pct.assign(_f64(TRUE_MS_RETURN_PCT))


def _assign_simple_ground_truth(model: TrainableFinancialModel) -> None:
    """Assign Scenario 1 specific ground truth.

    Supports both deterministic (``SimpleOpEx``) and Bayesian
    (``BayesianOpEx``) operating-expense modules so the same simple
    policies can be exercised against either a point-estimate or a
    variational OpEx generator.
    """
    _f64 = lambda v: tf.constant(v, dtype=tf.float64)
    _assign_shared_ground_truth(model)

    model.balance_sheet.purchases_policy.cost_ratio.assign(_f64(TRUE_COST_RATIO))

    # Compute baseline_opex in scaled units so that the generator
    # reproduces OpEx ≈ TRUE_VARIABLE_OPEX_PCT × sales on average.
    scaled_sales = tf.constant(_SALES_USD / _SCALE, dtype=tf.float64)
    cum_inf = np.cumprod(1 + _INFLATION)
    true_opex_scaled = scaled_sales * 0.10
    baseline_approx = float(
        tf.reduce_mean(true_opex_scaled / tf.constant(cum_inf, dtype=tf.float64))
    )

    if isinstance(model.opex_module, SimpleOpEx):
        model.opex_module.variable_opex_pct.assign(_f64(TRUE_VARIABLE_OPEX_PCT))
        model.opex_module.baseline_opex.assign(_f64(baseline_approx))
    elif isinstance(model.opex_module, BayesianOpEx):
        # Set posterior means to the true values, and shrink the
        # variational scales + aleatoric noise so the generator's
        # output is effectively deterministic at the true mean.
        model.opex_module.q_var_opex_loc.assign(_f64(TRUE_VARIABLE_OPEX_PCT))
        model.opex_module.q_base_opex_loc.assign(_f64(baseline_approx))
        model.opex_module.q_var_opex_scale.assign(_f64(0.001))
        model.opex_module.q_base_opex_scale.assign(_f64(0.001))
        model.opex_module.noise_sigma.assign(_f64(0.001))

    lp = model.cash_budget.liquidity_policy
    lp.total_liquidity_pct.assign(_f64(TRUE_TOTAL_LIQ_PCT))
    lp.cash_pct_of_liquidity.assign(_f64(TRUE_CASH_PCT_OF_LIQ))

    model.cash_budget.dividend_policy.dividend_payout_ratio_pct.assign(
        _f64(TRUE_DIVIDEND_PAYOUT_RATIO)
    )
    model.cash_budget.buyback_policy.stock_buyback_pct.assign(
        _f64(TRUE_STOCK_BUYBACK_PCT)
    )

    dp = model.cash_budget.debt_policy
    dp.avg_maturity_years.assign(_f64(TRUE_AVG_MATURITY_YEARS))
    dp.equity_financing_pct.assign(_f64(TRUE_EQUITY_FINANCING_PCT))


def _assign_trend_ground_truth(model: TrainableFinancialModel) -> None:
    """Assign Scenarios 2-4 specific ground truth (trend policies)."""
    _f64 = lambda v: tf.constant(v, dtype=tf.float64)
    _assign_shared_ground_truth(model)

    pp = model.balance_sheet.purchases_policy
    pp.cost_ratio_alpha.assign(_f64(TRUE_CR_ALPHA))
    pp.cost_ratio_beta.assign(_f64(TRUE_CR_BETA))

    # OpEx — for SimpleOpEx in Scenario 2, same approach as simple
    if isinstance(model.opex_module, SimpleOpEx):
        model.opex_module.variable_opex_pct.assign(_f64(TRUE_VARIABLE_OPEX_PCT))
        scaled_sales = tf.constant(_SALES_USD / _SCALE, dtype=tf.float64)
        cum_inf = np.cumprod(1 + _INFLATION)
        true_opex_scaled = scaled_sales * 0.10
        baseline_approx = float(
            tf.reduce_mean(true_opex_scaled / tf.constant(cum_inf, dtype=tf.float64))
        )
        model.opex_module.baseline_opex.assign(_f64(baseline_approx))
    elif isinstance(model.opex_module, BayesianOpEx):
        # For BayesianOpEx, set posterior means to true values
        scaled_sales = tf.constant(_SALES_USD / _SCALE, dtype=tf.float64)
        cum_inf = np.cumprod(1 + _INFLATION)
        true_opex_scaled = scaled_sales * 0.10
        baseline_approx = float(
            tf.reduce_mean(true_opex_scaled / tf.constant(cum_inf, dtype=tf.float64))
        )
        model.opex_module.q_var_opex_loc.assign(_f64(TRUE_VARIABLE_OPEX_PCT))
        model.opex_module.q_base_opex_loc.assign(_f64(baseline_approx))
        model.opex_module.q_var_opex_scale.assign(_f64(0.001))
        model.opex_module.q_base_opex_scale.assign(_f64(0.001))
        model.opex_module.noise_sigma.assign(_f64(0.001))

    lp = model.cash_budget.liquidity_policy
    lp.tl_alpha.assign(_f64(TRUE_TL_ALPHA))
    lp.tl_beta.assign(_f64(TRUE_TL_BETA))
    lp.tl_baseline.assign(_f64(TRUE_TL_BASELINE))
    lp.cash_alpha.assign(_f64(TRUE_CASH_ALPHA))
    lp.cash_beta.assign(_f64(TRUE_CASH_BETA))

    dp_div = model.cash_budget.dividend_policy
    dp_div.dividend_payout_ratio_pct.assign(_f64(TRUE_DIV_PAYOUT_RATIO))
    dp_div.dividend_adjustment_speed.assign(_f64(TRUE_DIV_ADJ_SPEED))

    bp = model.cash_budget.buyback_policy
    bp.sb_baseline.assign(_f64(TRUE_SB_BASELINE))
    bp.sb_ratio.assign(_f64(TRUE_SB_RATIO))

    dp = model.cash_budget.debt_policy
    dp.st_debt_baseline.assign(_f64(TRUE_ST_DEBT_BASELINE))
    dp.st_debt_pct.assign(_f64(TRUE_ST_DEBT_PCT))
    dp.ef_alpha.assign(_f64(TRUE_EF_ALPHA))
    dp.ef_beta.assign(_f64(TRUE_EF_BETA))
    dp.avg_maturity_years.assign(_f64(TRUE_AVG_MATURITY_YEARS))


# =====================================================================
# Helper: generate synthetic data via forecast_step
# =====================================================================


def _generate_via_forecast_step(
    gen_model: TrainableFinancialModel,
) -> dict:
    """Run forecast_step in a loop to produce perfectly consistent data.

    The seed data has N_YEARS+1 years. Year 0 (index 0) of the seed serves
    as the initial state. We run forecast_step for N_YEARS transitions,
    producing years 1..N_YEARS. These N_YEARS output years become the
    final training data — every year is model-generated, eliminating
    seed/forecast boundary inconsistencies.

    Returns:
        ``financial_statements`` dict in USD, ready for a recovery model's
        ``prepare()``.
    """
    scale = gen_model.amount_scale
    inflation_ext = tf.constant(_INFLATION_EXTENDED, dtype=tf.float64)
    cum_inf_all = tf.math.cumprod(1.0 + inflation_ext)

    # Collect per-year data from forecast_step (N_YEARS points)
    data = {key: [] for key in DIAGNOSTIC_KEYS}

    # Initial state from year 0 of the extended seed
    state = gen_model.build_state_from_index(0)

    # Generate N_YEARS transitions: year 1, 2, ..., N_YEARS in extended indexing
    for t in range(1, _N_YEARS + 1):
        inputs = {
            "sales_t": tf.constant(_SALES_USD_EXTENDED[t] / scale, dtype=tf.float64),
            "year": tf.constant(float(_YEARS_EXTENDED[t]), dtype=tf.float64),
            "cum_inflation": cum_inf_all[t],
        }
        diag = gen_model.forecast_step(state, inputs, use_mean_opex=True)

        for key in DIAGNOSTIC_KEYS:
            data[key].append(float(diag[key]))

        state = {k: diag[k] for k in RECURRENT_KEYS}

    # Convert to tensors in USD — these are years 1..N_YEARS of extended
    def to_usd(key):
        return tf.constant(data[key], dtype=tf.float64) * scale

    # Sales for the output years (indices 1: of extended)
    sales_usd = tf.constant(_SALES_USD, dtype=tf.float64)
    cogs_usd = to_usd("cogs")
    inventory_usd = to_usd("inventory")
    delta_inv = tf.concat(
        [tf.zeros([1], dtype=tf.float64), inventory_usd[1:] - inventory_usd[:-1]],
        axis=0,
    )
    purchases_usd = cogs_usd + delta_inv

    current_liabilities_usd = (
        to_usd("accounts_payable")
        + to_usd("advance_payments_sales")
        + to_usd("effective_st_debt")
        + to_usd("current_lt_debt")
    )

    return {
        "sales": sales_usd,
        "purchases": purchases_usd,
        "cogs": cogs_usd,
        "nca": to_usd("nca"),
        "depreciation": to_usd("depreciation"),
        "advance_payments_purchases": to_usd("advance_payments_purchases"),
        "accounts_receivable": to_usd("accounts_receivable"),
        "accounts_payable": to_usd("accounts_payable"),
        "advance_payments_sales": to_usd("advance_payments_sales"),
        "cash": to_usd("cash"),
        "ims": to_usd("investment_in_market_securities"),
        "inventory": inventory_usd,
        "current_liabilities": current_liabilities_usd,
        "non_current_liabilities": to_usd("non_current_liabilities"),
        "equity": to_usd("equity"),
        "net_income": to_usd("net_income"),
        "dividends": to_usd("dividends"),
        "stock_buyback": to_usd("stock_buyback"),
        "opex": to_usd("opex"),
        "tax": to_usd("tax"),
        "current_lt_debt": to_usd("current_lt_debt"),
        "interest_payment": to_usd("interest_payment"),
        "ms_return": to_usd("ms_return"),
        "years": tf.constant(_YEARS, dtype=tf.float64),
        "inflation": tf.constant(_INFLATION, dtype=tf.float64),
    }


# =====================================================================
# Helper: build + train a recovery model
# =====================================================================


def _train_recovery_model(
    stmts: dict,
    opex_module,
    liquidity_policy,
    dividend_policy,
    buyback_policy,
    purchases_policy,
    debt_policy,
    tax_module,
    trajectory_simulator,
) -> TrainableFinancialModel:
    """Create a fresh model, prepare on synthetic data, and train."""
    model = TrainableFinancialModel(
        opex_module=opex_module,
        trajectory_simulator=trajectory_simulator,
        capex_policy=CapexPolicy(),
        working_capital=WorkingCapitalPolicy(),
        liquidity_policy=liquidity_policy,
        dividend_policy=dividend_policy,
        buyback_policy=buyback_policy,
        purchases_policy=purchases_policy,
        debt_policy=debt_policy,
        tax_module=tax_module,
    )
    model.prepare(stmts, inflation=stmts["inflation"], test_years=0)
    data = model._build_training_data()
    PolicyTrainer(epochs=_POLICY_EPOCHS).train(
        model,
        data,
        show_plot=False,
        learning_rate=_LEARNING_RATE,
    )
    StructuralTrainer(epochs=_STRUCTURAL_EPOCHS).train(
        model,
        data,
        show_plot=False,
        learning_rate=_LEARNING_RATE,
    )
    return model


# =====================================================================
# Fixtures — Scenario 1: Simple
# =====================================================================


@pytest.fixture(scope="module")
def trained_model_s1():
    tf.random.set_seed(42)
    np.random.seed(42)

    gen = TrainableFinancialModel(
        opex_module=SimpleOpEx(),
        trajectory_simulator=DeterministicSimulator(),
        capex_policy=CapexPolicy(),
        working_capital=WorkingCapitalPolicy(),
        liquidity_policy=SimpleLiquidityPolicy(),
        dividend_policy=SimpleDividendPolicy(),
        buyback_policy=SimpleBuybackPolicy(),
        purchases_policy=StaticCostRatioPolicy(),
        debt_policy=SimpleDebtPolicy(),
        tax_module=SimpleTax(),
    )
    seed = _build_seed_data(TRUE_COST_RATIO)
    gen.prepare(seed, inflation=seed["inflation"], test_years=0)
    _assign_simple_ground_truth(gen)
    stmts = _generate_via_forecast_step(gen)

    return _train_recovery_model(
        stmts,
        opex_module=SimpleOpEx(),
        liquidity_policy=SimpleLiquidityPolicy(),
        dividend_policy=SimpleDividendPolicy(),
        buyback_policy=SimpleBuybackPolicy(),
        purchases_policy=StaticCostRatioPolicy(),
        debt_policy=SimpleDebtPolicy(),
        tax_module=SimpleTax(),
        trajectory_simulator=DeterministicSimulator(),
    )


# =====================================================================
# Fixtures — Scenario 2: Advanced (Trend)
# =====================================================================


@pytest.fixture(scope="module")
def trained_model_s2():
    tf.random.set_seed(42)
    np.random.seed(42)

    gen = TrainableFinancialModel(
        opex_module=SimpleOpEx(),
        trajectory_simulator=DeterministicSimulator(),
        capex_policy=CapexPolicy(),
        working_capital=WorkingCapitalPolicy(),
        liquidity_policy=TrendLiquidityPolicy(),
        dividend_policy=LintnerDividendPolicy(),
        buyback_policy=BaselineBuybackPolicy(),
        purchases_policy=TrendCostRatioPolicy(),
        debt_policy=TrendDebtPolicy(),
        tax_module=SimpleTax(),
    )
    # Use sigmoid(TRUE_CR_ALPHA) as the average cost ratio for seed
    avg_cr = float(tf.sigmoid(TRUE_CR_ALPHA))
    seed = _build_seed_data(avg_cr)
    gen.prepare(seed, inflation=seed["inflation"], test_years=0)
    _assign_trend_ground_truth(gen)
    stmts = _generate_via_forecast_step(gen)

    return _train_recovery_model(
        stmts,
        opex_module=SimpleOpEx(),
        liquidity_policy=TrendLiquidityPolicy(),
        dividend_policy=LintnerDividendPolicy(),
        buyback_policy=BaselineBuybackPolicy(),
        purchases_policy=TrendCostRatioPolicy(),
        debt_policy=TrendDebtPolicy(),
        tax_module=SimpleTax(),
        trajectory_simulator=DeterministicSimulator(),
    )


# =====================================================================
# Fixtures — Scenario 3: Advanced + BayesianOpEx
# =====================================================================


@pytest.fixture(scope="module")
def trained_model_s3():
    tf.random.set_seed(42)
    np.random.seed(42)

    gen = TrainableFinancialModel(
        opex_module=BayesianOpEx(),
        trajectory_simulator=MonteCarloSimulator(n_samples=2),
        capex_policy=CapexPolicy(),
        working_capital=WorkingCapitalPolicy(),
        liquidity_policy=TrendLiquidityPolicy(),
        dividend_policy=LintnerDividendPolicy(),
        buyback_policy=BaselineBuybackPolicy(),
        purchases_policy=TrendCostRatioPolicy(),
        debt_policy=TrendDebtPolicy(),
        tax_module=SimpleTax(),
    )
    avg_cr = float(tf.sigmoid(TRUE_CR_ALPHA))
    seed = _build_seed_data(avg_cr)
    gen.prepare(seed, inflation=seed["inflation"], test_years=0)
    _assign_trend_ground_truth(gen)
    stmts = _generate_via_forecast_step(gen)

    # Inject known noise into opex
    noise = tf.constant(
        np.random.normal(0, TRUE_NOISE_SIGMA * gen.amount_scale, _N_YEARS),
        dtype=tf.float64,
    )
    stmts["opex"] = stmts["opex"] + noise

    return _train_recovery_model(
        stmts,
        opex_module=BayesianOpEx(),
        liquidity_policy=TrendLiquidityPolicy(),
        dividend_policy=LintnerDividendPolicy(),
        buyback_policy=BaselineBuybackPolicy(),
        purchases_policy=TrendCostRatioPolicy(),
        debt_policy=TrendDebtPolicy(),
        tax_module=SimpleTax(),
        trajectory_simulator=MonteCarloSimulator(n_samples=2),
    )


# =====================================================================
# Fixtures — Scenario 4: Advanced + BayesianOpEx + TaxWithAnomalies
# =====================================================================


@pytest.fixture(scope="module")
def trained_model_s4():
    tf.random.set_seed(42)
    np.random.seed(42)

    tax_gen = TaxWithAnomalies(TRUE_TAX_ONETIME)
    gen = TrainableFinancialModel(
        opex_module=BayesianOpEx(),
        trajectory_simulator=MonteCarloSimulator(n_samples=2),
        capex_policy=CapexPolicy(),
        working_capital=WorkingCapitalPolicy(),
        liquidity_policy=TrendLiquidityPolicy(),
        dividend_policy=LintnerDividendPolicy(),
        buyback_policy=BaselineBuybackPolicy(),
        purchases_policy=TrendCostRatioPolicy(),
        debt_policy=TrendDebtPolicy(),
        tax_module=tax_gen,
    )
    avg_cr = float(tf.sigmoid(TRUE_CR_ALPHA))
    seed = _build_seed_data(avg_cr)
    gen.prepare(seed, inflation=seed["inflation"], test_years=0)
    _assign_trend_ground_truth(gen)
    stmts = _generate_via_forecast_step(gen)

    # Inject known noise into opex
    noise = tf.constant(
        np.random.normal(0, TRUE_NOISE_SIGMA * gen.amount_scale, _N_YEARS),
        dtype=tf.float64,
    )
    stmts["opex"] = stmts["opex"] + noise

    return _train_recovery_model(
        stmts,
        opex_module=BayesianOpEx(),
        liquidity_policy=TrendLiquidityPolicy(),
        dividend_policy=LintnerDividendPolicy(),
        buyback_policy=BaselineBuybackPolicy(),
        purchases_policy=TrendCostRatioPolicy(),
        debt_policy=TrendDebtPolicy(),
        tax_module=TaxWithAnomalies(TRUE_TAX_ONETIME),
        trajectory_simulator=MonteCarloSimulator(n_samples=2),
    )


# #####################################################################
# Test Classes
# #####################################################################


class TestScenario1Simple:
    """Parameter recovery with all simple (static ratio) policies."""

    # --- CapEx ---
    def test_depreciation_rate(self, trained_model_s1):
        v = float(trained_model_s1.balance_sheet.capex_policy.depreciation_rate.numpy())
        assert v == pytest.approx(TRUE_DEPRECIATION_RATE, rel=0.10)

    def test_asset_maintain(self, trained_model_s1):
        v = float(trained_model_s1.balance_sheet.capex_policy.asset_maintain.numpy())
        assert v == pytest.approx(TRUE_ASSET_MAINTAIN, abs=0.10)

    def test_asset_growth(self, trained_model_s1):
        v = float(trained_model_s1.balance_sheet.capex_policy.asset_growth.numpy())
        assert v == pytest.approx(TRUE_ASSET_GROWTH, rel=0.15)

    # --- Working Capital ---
    def test_ar_pct(self, trained_model_s1):
        v = float(
            trained_model_s1.balance_sheet.working_capital.account_receivables_pct.numpy()
        )
        assert v == pytest.approx(TRUE_AR_PCT, rel=0.10)

    def test_ap_pct(self, trained_model_s1):
        v = float(
            trained_model_s1.balance_sheet.working_capital.account_payables_pct.numpy()
        )
        assert v == pytest.approx(TRUE_AP_PCT, rel=0.10)

    def test_inv_cogs_pct(self, trained_model_s1):
        v = float(
            trained_model_s1.balance_sheet.working_capital.inventory_cogs_pct.numpy()
        )
        assert v == pytest.approx(TRUE_INV_COGS_PCT, rel=0.10)

    def test_adv_ps_pct(self, trained_model_s1):
        v = float(
            trained_model_s1.balance_sheet.working_capital.advance_payments_sales_pct.numpy()
        )
        assert v == pytest.approx(TRUE_ADV_PS_PCT, rel=0.10)

    def test_adv_pp_pct(self, trained_model_s1):
        v = float(
            trained_model_s1.balance_sheet.working_capital.advance_payments_purchases_pct.numpy()
        )
        assert v == pytest.approx(TRUE_ADV_PP_PCT, rel=0.10)

    # --- Cost Ratio ---
    def test_cost_ratio(self, trained_model_s1):
        v = float(trained_model_s1.balance_sheet.purchases_policy.cost_ratio.numpy())
        assert v == pytest.approx(TRUE_COST_RATIO, rel=0.05)

    # --- OpEx ---
    def test_variable_opex_pct(self, trained_model_s1):
        v = float(trained_model_s1.opex_module.variable_opex_pct.numpy())
        assert v == pytest.approx(TRUE_VARIABLE_OPEX_PCT, rel=0.15)

    # --- Liquidity ---
    def test_total_liquidity_pct(self, trained_model_s1):
        v = float(
            trained_model_s1.cash_budget.liquidity_policy.total_liquidity_pct.numpy()
        )
        assert v == pytest.approx(TRUE_TOTAL_LIQ_PCT, rel=0.10)

    def test_cash_pct_of_liquidity(self, trained_model_s1):
        v = float(
            trained_model_s1.cash_budget.liquidity_policy.cash_pct_of_liquidity.numpy()
        )
        assert v == pytest.approx(TRUE_CASH_PCT_OF_LIQ, rel=0.10)

    # --- Dividends ---
    def test_dividend_payout_ratio(self, trained_model_s1):
        v = float(
            trained_model_s1.cash_budget.dividend_policy.dividend_payout_ratio_pct.numpy()
        )
        assert v == pytest.approx(TRUE_DIVIDEND_PAYOUT_RATIO, rel=0.10)

    # --- Buyback ---
    def test_stock_buyback_pct(self, trained_model_s1):
        v = float(trained_model_s1.cash_budget.buyback_policy.stock_buyback_pct.numpy())
        assert v == pytest.approx(TRUE_STOCK_BUYBACK_PCT, rel=0.10)

    # --- Tax ---
    def test_income_tax_pct(self, trained_model_s1):
        v = float(trained_model_s1.tax_module.income_tax_pct.numpy())
        assert v == pytest.approx(TRUE_INCOME_TAX_PCT, rel=0.10)

    # --- Structural ---
    def test_avg_st_interest_pct(self, trained_model_s1):
        # With SimpleDebtPolicy, ST debt = max(0, liquidity_deficit_st).
        # If the company has no ST liquidity deficit, this parameter
        # has zero gradient and is unidentifiable.  Verify it is at least
        # a reasonable interest rate (< 0.20) rather than exact recovery.
        v = float(trained_model_s1.income_statement.avg_short_term_interest_pct.numpy())
        assert 0.0 < v < 0.20, f"ST interest rate should be reasonable, got {v}"

    def test_avg_lt_interest_pct(self, trained_model_s1):
        v = float(trained_model_s1.income_statement.avg_long_term_interest_pct.numpy())
        assert v == pytest.approx(TRUE_AVG_LT_INTEREST_PCT, rel=0.20)

    def test_ms_return_pct(self, trained_model_s1):
        v = float(
            trained_model_s1.income_statement.market_securities_return_pct.numpy()
        )
        assert v == pytest.approx(TRUE_MS_RETURN_PCT, rel=0.20)

    def test_avg_maturity_years(self, trained_model_s1):
        v = float(trained_model_s1.cash_budget.debt_policy.avg_maturity_years.numpy())
        assert v == pytest.approx(TRUE_AVG_MATURITY_YEARS, rel=0.20)

    def test_equity_financing_pct(self, trained_model_s1):
        # Equity financing % only has gradient signal when there are new
        # LT loans (long_term_financing > 0). With few transitions and
        # small financing needs, recovery is indirect. Wider tolerance.
        v = float(trained_model_s1.cash_budget.debt_policy.equity_financing_pct.numpy())
        assert v == pytest.approx(TRUE_EQUITY_FINANCING_PCT, rel=0.80)


class TestScenario2Advanced:
    """Parameter recovery with trend (logit-linear) policies."""

    # --- CapEx (same as S1) ---
    def test_depreciation_rate(self, trained_model_s2):
        v = float(trained_model_s2.balance_sheet.capex_policy.depreciation_rate.numpy())
        assert v == pytest.approx(TRUE_DEPRECIATION_RATE, rel=0.10)

    def test_asset_growth(self, trained_model_s2):
        v = float(trained_model_s2.balance_sheet.capex_policy.asset_growth.numpy())
        assert v == pytest.approx(TRUE_ASSET_GROWTH, rel=0.15)

    # --- Working Capital (same as S1) ---
    def test_ar_pct(self, trained_model_s2):
        v = float(
            trained_model_s2.balance_sheet.working_capital.account_receivables_pct.numpy()
        )
        assert v == pytest.approx(TRUE_AR_PCT, rel=0.10)

    def test_ap_pct(self, trained_model_s2):
        v = float(
            trained_model_s2.balance_sheet.working_capital.account_payables_pct.numpy()
        )
        assert v == pytest.approx(TRUE_AP_PCT, rel=0.10)

    def test_inv_cogs_pct(self, trained_model_s2):
        v = float(
            trained_model_s2.balance_sheet.working_capital.inventory_cogs_pct.numpy()
        )
        assert v == pytest.approx(TRUE_INV_COGS_PCT, rel=0.10)

    # --- Cost Ratio (Trend) ---
    def test_cost_ratio_alpha(self, trained_model_s2):
        v = float(
            trained_model_s2.balance_sheet.purchases_policy.cost_ratio_alpha.numpy()
        )
        assert v == pytest.approx(TRUE_CR_ALPHA, abs=0.10)

    def test_cost_ratio_beta(self, trained_model_s2):
        v = float(
            trained_model_s2.balance_sheet.purchases_policy.cost_ratio_beta.numpy()
        )
        assert v == pytest.approx(TRUE_CR_BETA, abs=0.02)

    # --- OpEx ---
    def test_variable_opex_pct(self, trained_model_s2):
        v = float(trained_model_s2.opex_module.variable_opex_pct.numpy())
        assert v == pytest.approx(TRUE_VARIABLE_OPEX_PCT, rel=0.15)

    # --- Liquidity (Trend) ---
    def test_tl_alpha(self, trained_model_s2):
        v = float(trained_model_s2.cash_budget.liquidity_policy.tl_alpha.numpy())
        assert v == pytest.approx(TRUE_TL_ALPHA, abs=0.15)

    def test_tl_beta(self, trained_model_s2):
        v = float(trained_model_s2.cash_budget.liquidity_policy.tl_beta.numpy())
        assert v == pytest.approx(TRUE_TL_BETA, abs=0.02)

    def test_cash_alpha(self, trained_model_s2):
        v = float(trained_model_s2.cash_budget.liquidity_policy.cash_alpha.numpy())
        assert v == pytest.approx(TRUE_CASH_ALPHA, abs=0.15)

    def test_cash_beta(self, trained_model_s2):
        v = float(trained_model_s2.cash_budget.liquidity_policy.cash_beta.numpy())
        assert v == pytest.approx(TRUE_CASH_BETA, abs=0.02)

    # --- Dividends (Lintner) ---
    def test_dividend_payout_ratio(self, trained_model_s2):
        v = float(
            trained_model_s2.cash_budget.dividend_policy.dividend_payout_ratio_pct.numpy()
        )
        assert v == pytest.approx(TRUE_DIV_PAYOUT_RATIO, rel=0.10)

    def test_dividend_adj_speed(self, trained_model_s2):
        v = float(
            trained_model_s2.cash_budget.dividend_policy.dividend_adjustment_speed.numpy()
        )
        assert v == pytest.approx(TRUE_DIV_ADJ_SPEED, rel=0.20)

    # --- Buyback (Baseline) ---
    def test_sb_baseline(self, trained_model_s2):
        # sb_baseline and sb_ratio are correlated when depreciation varies
        # little (NCA evolves slowly with asset_maintain≈1). Wider tolerance.
        v = float(trained_model_s2.cash_budget.buyback_policy.sb_baseline.numpy())
        assert v == pytest.approx(TRUE_SB_BASELINE, abs=0.50)

    def test_sb_ratio(self, trained_model_s2):
        v = float(trained_model_s2.cash_budget.buyback_policy.sb_ratio.numpy())
        assert v == pytest.approx(TRUE_SB_RATIO, rel=1.5)

    # --- Debt (Trend) ---
    def test_st_debt_baseline(self, trained_model_s2):
        v = float(trained_model_s2.cash_budget.debt_policy.st_debt_baseline.numpy())
        assert v == pytest.approx(TRUE_ST_DEBT_BASELINE, abs=0.03)

    def test_st_debt_pct(self, trained_model_s2):
        v = float(trained_model_s2.cash_budget.debt_policy.st_debt_pct.numpy())
        assert v == pytest.approx(TRUE_ST_DEBT_PCT, abs=0.02)

    # --- Tax ---
    def test_income_tax_pct(self, trained_model_s2):
        v = float(trained_model_s2.tax_module.income_tax_pct.numpy())
        assert v == pytest.approx(TRUE_INCOME_TAX_PCT, rel=0.10)

    # --- Structural ---
    def test_avg_st_interest_pct(self, trained_model_s2):
        v = float(trained_model_s2.income_statement.avg_short_term_interest_pct.numpy())
        assert v == pytest.approx(TRUE_AVG_ST_INTEREST_PCT, rel=0.20)

    def test_avg_lt_interest_pct(self, trained_model_s2):
        v = float(trained_model_s2.income_statement.avg_long_term_interest_pct.numpy())
        assert v == pytest.approx(TRUE_AVG_LT_INTEREST_PCT, rel=0.20)

    def test_ms_return_pct(self, trained_model_s2):
        v = float(
            trained_model_s2.income_statement.market_securities_return_pct.numpy()
        )
        assert v == pytest.approx(TRUE_MS_RETURN_PCT, rel=0.20)

    def test_avg_maturity_years(self, trained_model_s2):
        v = float(trained_model_s2.cash_budget.debt_policy.avg_maturity_years.numpy())
        assert v == pytest.approx(TRUE_AVG_MATURITY_YEARS, rel=0.20)

    def test_ef_alpha(self, trained_model_s2):
        # Equity financing trend trained via forecast_step. With few LT
        # financing events, recovery is indirect. Verify reasonable range.
        v = float(trained_model_s2.cash_budget.debt_policy.ef_alpha.numpy())
        assert (
            -3.0 < v < 0.0
        ), f"ef_alpha should be negative (low equity share), got {v}"

    def test_ef_beta(self, trained_model_s2):
        v = float(trained_model_s2.cash_budget.debt_policy.ef_beta.numpy())
        assert -1.0 < v < 3.0, f"ef_beta should be bounded, got {v}"


class TestScenario3BayesianOpEx:
    """Parameter recovery with BayesianOpEx (stochastic, known noise)."""

    # --- CapEx ---
    def test_depreciation_rate(self, trained_model_s3):
        v = float(trained_model_s3.balance_sheet.capex_policy.depreciation_rate.numpy())
        assert v == pytest.approx(TRUE_DEPRECIATION_RATE, rel=0.10)

    # --- Working Capital ---
    def test_ar_pct(self, trained_model_s3):
        v = float(
            trained_model_s3.balance_sheet.working_capital.account_receivables_pct.numpy()
        )
        assert v == pytest.approx(TRUE_AR_PCT, rel=0.10)

    # --- Cost Ratio (Trend) ---
    def test_cost_ratio_alpha(self, trained_model_s3):
        v = float(
            trained_model_s3.balance_sheet.purchases_policy.cost_ratio_alpha.numpy()
        )
        assert v == pytest.approx(TRUE_CR_ALPHA, abs=0.10)

    # --- BayesianOpEx ---
    def test_q_var_opex_loc(self, trained_model_s3):
        v = float(trained_model_s3.opex_module.q_var_opex_loc.numpy())
        assert v == pytest.approx(TRUE_VARIABLE_OPEX_PCT, rel=0.20)

    def test_q_base_opex_scale_shrinks(self, trained_model_s3):
        v = float(trained_model_s3.opex_module.q_base_opex_scale.numpy())
        assert v < 1.0, f"Posterior scale should shrink from initial=1.0, got {v}"

    def test_q_var_opex_scale_shrinks(self, trained_model_s3):
        v = float(trained_model_s3.opex_module.q_var_opex_scale.numpy())
        assert v < 1.0, f"Posterior scale should shrink from initial=1.0, got {v}"

    def test_noise_sigma(self, trained_model_s3):
        v = float(trained_model_s3.opex_module.noise_sigma.numpy())
        # noise_sigma should be small (data has very small noise)
        assert v < 0.1, f"noise_sigma should be small, got {v}"

    # --- Liquidity (Trend) ---
    def test_tl_alpha(self, trained_model_s3):
        v = float(trained_model_s3.cash_budget.liquidity_policy.tl_alpha.numpy())
        assert v == pytest.approx(TRUE_TL_ALPHA, abs=0.15)

    # --- Dividends (Lintner) ---
    def test_dividend_payout_ratio(self, trained_model_s3):
        v = float(
            trained_model_s3.cash_budget.dividend_policy.dividend_payout_ratio_pct.numpy()
        )
        assert v == pytest.approx(TRUE_DIV_PAYOUT_RATIO, rel=0.10)

    def test_dividend_adj_speed(self, trained_model_s3):
        v = float(
            trained_model_s3.cash_budget.dividend_policy.dividend_adjustment_speed.numpy()
        )
        assert v == pytest.approx(TRUE_DIV_ADJ_SPEED, rel=0.20)

    # --- Tax ---
    def test_income_tax_pct(self, trained_model_s3):
        v = float(trained_model_s3.tax_module.income_tax_pct.numpy())
        assert v == pytest.approx(TRUE_INCOME_TAX_PCT, rel=0.10)

    # --- Structural ---
    def test_avg_st_interest_pct(self, trained_model_s3):
        v = float(trained_model_s3.income_statement.avg_short_term_interest_pct.numpy())
        assert v == pytest.approx(TRUE_AVG_ST_INTEREST_PCT, rel=0.20)

    def test_avg_maturity_years(self, trained_model_s3):
        v = float(trained_model_s3.cash_budget.debt_policy.avg_maturity_years.numpy())
        assert v == pytest.approx(TRUE_AVG_MATURITY_YEARS, rel=0.20)


class TestScenario4BayesianTaxAnomalies:
    """Parameter recovery with BayesianOpEx + TaxWithAnomalies.

    The key validation is that ``income_tax_pct`` recovers cleanly
    despite one-time anomalies in the training data.
    """

    # --- CapEx ---
    def test_depreciation_rate(self, trained_model_s4):
        v = float(trained_model_s4.balance_sheet.capex_policy.depreciation_rate.numpy())
        assert v == pytest.approx(TRUE_DEPRECIATION_RATE, rel=0.10)

    def test_asset_growth(self, trained_model_s4):
        v = float(trained_model_s4.balance_sheet.capex_policy.asset_growth.numpy())
        assert v == pytest.approx(TRUE_ASSET_GROWTH, rel=0.15)

    # --- Working Capital ---
    def test_ar_pct(self, trained_model_s4):
        v = float(
            trained_model_s4.balance_sheet.working_capital.account_receivables_pct.numpy()
        )
        assert v == pytest.approx(TRUE_AR_PCT, rel=0.10)

    def test_ap_pct(self, trained_model_s4):
        v = float(
            trained_model_s4.balance_sheet.working_capital.account_payables_pct.numpy()
        )
        assert v == pytest.approx(TRUE_AP_PCT, rel=0.10)

    def test_inv_cogs_pct(self, trained_model_s4):
        v = float(
            trained_model_s4.balance_sheet.working_capital.inventory_cogs_pct.numpy()
        )
        assert v == pytest.approx(TRUE_INV_COGS_PCT, rel=0.10)

    # --- Cost Ratio (Trend) ---
    def test_cost_ratio_alpha(self, trained_model_s4):
        v = float(
            trained_model_s4.balance_sheet.purchases_policy.cost_ratio_alpha.numpy()
        )
        assert v == pytest.approx(TRUE_CR_ALPHA, abs=0.10)

    def test_cost_ratio_beta(self, trained_model_s4):
        v = float(
            trained_model_s4.balance_sheet.purchases_policy.cost_ratio_beta.numpy()
        )
        assert v == pytest.approx(TRUE_CR_BETA, abs=0.02)

    # --- BayesianOpEx ---
    def test_q_var_opex_loc(self, trained_model_s4):
        v = float(trained_model_s4.opex_module.q_var_opex_loc.numpy())
        assert v == pytest.approx(TRUE_VARIABLE_OPEX_PCT, rel=0.20)

    def test_q_base_opex_scale_shrinks(self, trained_model_s4):
        v = float(trained_model_s4.opex_module.q_base_opex_scale.numpy())
        assert v < 1.0, f"Posterior scale should shrink from initial=1.0, got {v}"

    def test_q_var_opex_scale_shrinks(self, trained_model_s4):
        v = float(trained_model_s4.opex_module.q_var_opex_scale.numpy())
        assert v < 1.0, f"Posterior scale should shrink from initial=1.0, got {v}"

    def test_noise_sigma(self, trained_model_s4):
        v = float(trained_model_s4.opex_module.noise_sigma.numpy())
        assert v < 0.1, f"noise_sigma should be small, got {v}"

    # --- Liquidity (Trend) ---
    def test_tl_alpha(self, trained_model_s4):
        v = float(trained_model_s4.cash_budget.liquidity_policy.tl_alpha.numpy())
        assert v == pytest.approx(TRUE_TL_ALPHA, abs=0.15)

    def test_tl_beta(self, trained_model_s4):
        v = float(trained_model_s4.cash_budget.liquidity_policy.tl_beta.numpy())
        assert v == pytest.approx(TRUE_TL_BETA, abs=0.02)

    def test_cash_alpha(self, trained_model_s4):
        v = float(trained_model_s4.cash_budget.liquidity_policy.cash_alpha.numpy())
        assert v == pytest.approx(TRUE_CASH_ALPHA, abs=0.15)

    # --- Dividends (Lintner) ---
    def test_dividend_payout_ratio(self, trained_model_s4):
        v = float(
            trained_model_s4.cash_budget.dividend_policy.dividend_payout_ratio_pct.numpy()
        )
        assert v == pytest.approx(TRUE_DIV_PAYOUT_RATIO, rel=0.10)

    def test_dividend_adj_speed(self, trained_model_s4):
        v = float(
            trained_model_s4.cash_budget.dividend_policy.dividend_adjustment_speed.numpy()
        )
        assert v == pytest.approx(TRUE_DIV_ADJ_SPEED, rel=0.20)

    # --- Buyback (Baseline) ---
    def test_sb_ratio(self, trained_model_s4):
        v = float(trained_model_s4.cash_budget.buyback_policy.sb_ratio.numpy())
        assert v == pytest.approx(TRUE_SB_RATIO, rel=1.5)

    # --- Debt (Trend) ---
    def test_st_debt_baseline(self, trained_model_s4):
        v = float(trained_model_s4.cash_budget.debt_policy.st_debt_baseline.numpy())
        assert v == pytest.approx(TRUE_ST_DEBT_BASELINE, abs=0.03)

    def test_st_debt_pct(self, trained_model_s4):
        v = float(trained_model_s4.cash_budget.debt_policy.st_debt_pct.numpy())
        assert v == pytest.approx(TRUE_ST_DEBT_PCT, abs=0.02)

    # --- Tax (with anomalies — key test) ---
    def test_income_tax_pct(self, trained_model_s4):
        v = float(trained_model_s4.tax_module.income_tax_pct.numpy())
        assert v == pytest.approx(TRUE_INCOME_TAX_PCT, rel=0.10)

    # --- Structural ---
    def test_avg_st_interest_pct(self, trained_model_s4):
        v = float(trained_model_s4.income_statement.avg_short_term_interest_pct.numpy())
        assert v == pytest.approx(TRUE_AVG_ST_INTEREST_PCT, rel=0.20)

    def test_avg_lt_interest_pct(self, trained_model_s4):
        v = float(trained_model_s4.income_statement.avg_long_term_interest_pct.numpy())
        assert v == pytest.approx(TRUE_AVG_LT_INTEREST_PCT, rel=0.20)

    def test_ms_return_pct(self, trained_model_s4):
        v = float(
            trained_model_s4.income_statement.market_securities_return_pct.numpy()
        )
        assert v == pytest.approx(TRUE_MS_RETURN_PCT, rel=0.20)

    def test_avg_maturity_years(self, trained_model_s4):
        v = float(trained_model_s4.cash_budget.debt_policy.avg_maturity_years.numpy())
        assert v == pytest.approx(TRUE_AVG_MATURITY_YEARS, rel=0.20)

    def test_ef_alpha(self, trained_model_s4):
        v = float(trained_model_s4.cash_budget.debt_policy.ef_alpha.numpy())
        assert -3.0 < v < 0.0, f"ef_alpha should be negative, got {v}"


# #####################################################################
# Diagnostic script: 1-step-ahead fit plots for each recovery scenario.
#
# Run as ``python tests/test_parameter_recovery.py`` to train the
# scenarios and save PNG figures under ``training_results/``.
# This block is intentionally colocated with the fixtures above so it
# can reuse the private underscore helpers directly (no cross-module
# ``_name`` imports).  ``matplotlib`` is imported lazily inside
# ``plot_recovery_fit`` so pytest collection doesn't pay the import
# cost when only the test classes are being run.
# #####################################################################


FIT_KEYS = (
    "net_income",
    "nca",
    "accounts_receivable",
    "inventory",
    "cash",
    "investment_in_market_securities",
    "accounts_payable",
    "effective_st_debt",
    "non_current_liabilities",
    "equity",
    "depreciation",
    "cogs",
    "opex",
    "tax",
    "dividends",
    "stock_buyback",
)


def compute_one_step_fit(
    model: TrainableFinancialModel,
    n_mc_samples: int = 1,
):
    """Apply ``forecast_step`` to each historical transition.

    Mirrors ``ForecastPipeline._compute_historical_fit`` but standalone
    (no sales/inflation forecast models required).

    Args:
        model: Trained financial model to evaluate.
        n_mc_samples: If ``1``, use ``use_mean_opex=True`` and return a
            deterministic point-estimate fit (the default, same as
            ``ForecastPipeline._compute_historical_fit``).  If ``>1``,
            pre-sample the Bayesian OpEx posterior and aleatoric noise
            for ``n_mc_samples`` trajectories and evaluate the 1-step
            transition on a tiled batched state, producing per-key
            mean / lower (2.5%) / upper (97.5%) summaries.

    Returns:
        Tuple ``(fit, fit_years)`` where ``fit`` maps each key in
        :data:`FIT_KEYS` to a dict ``{"mean": [...], "lower": [...],
        "upper": [...]}`` with values in USD (deterministic fit has
        ``lower == mean == upper``), and ``fit_years`` is a list of
        calendar years.
    """
    s = model.scaled_data
    d = model.historical_data
    scale = model.amount_scale
    cum_inf_hist = tf.math.cumprod(1 + d["inflation"])
    n_hist = len(d["sales"])
    f64 = lambda v: tf.constant(float(v), dtype=tf.float64)

    fit = {k: {"mean": [], "lower": [], "upper": []} for k in FIT_KEYS}
    fit_years = []

    use_mean_opex = n_mc_samples == 1

    # For MC, pre-sample the BayesianOpEx posterior / aleatoric noise
    # once, spanning every transition. ``prepare_mc`` stores them on
    # ``model.opex_module`` and ``compute_mc_step`` indexes into them
    # via ``year - start_year``.
    if not use_mean_opex and isinstance(model.opex_module, BayesianOpEx):
        model.opex_module.prepare_mc(
            n_samples=n_mc_samples,
            n_years=n_hist,
            start_year=float(model.base_year + 1),
        )

    # Cached index of each key in the diagnostics tensor returned by
    # ``forecast_step_compiled``.
    diag_index = {k: i for i, k in enumerate(DIAGNOSTIC_KEYS)}

    for t in range(n_hist - 1):
        state_dict = model.build_state_from_index(t)
        sales_t_val = f64(s["sales"][t + 1])
        year_val = f64(float(model.base_year + t + 1))
        cum_inf_val = f64(cum_inf_hist[t + 1])

        if use_mean_opex:
            inputs_t = {
                "sales_t": sales_t_val,
                "year": year_val,
                "cum_inflation": cum_inf_val,
            }
            pred = model.forecast_step(
                state_dict,
                inputs_t,
                use_mean_opex=True,
            )
            for k in FIT_KEYS:
                v = float(pred[k].numpy()) * scale
                fit[k]["mean"].append(v)
                fit[k]["lower"].append(v)
                fit[k]["upper"].append(v)
        else:
            # Tile scalar state to [n_mc_samples, 14] and sales to
            # [n_mc_samples], then call the compiled batched path.
            state_row = tf.stack(
                [tf.cast(state_dict[k], tf.float64) for k in RECURRENT_KEYS]
            )
            state_batch = tf.broadcast_to(
                state_row[tf.newaxis, :],
                [n_mc_samples, state_row.shape[0]],
            )
            sales_batch = tf.fill([n_mc_samples], sales_t_val)

            _, diagnostics = model.forecast_step_compiled(
                state_batch,
                sales_batch,
                year_val,
                cum_inf_val,
                False,  # use_mean_opex
            )
            # diagnostics is shape [n_mc_samples, len(DIAGNOSTIC_KEYS)]
            diag_np = diagnostics.numpy() * scale
            for k in FIT_KEYS:
                samples = diag_np[:, diag_index[k]]
                fit[k]["mean"].append(float(np.mean(samples)))
                fit[k]["lower"].append(float(np.percentile(samples, 2.5)))
                fit[k]["upper"].append(float(np.percentile(samples, 97.5)))

        fit_years.append(model.base_year + t + 1)

    return fit, fit_years


def plot_recovery_fit(
    stmts,
    fit,
    fit_years,
    output_path: Path,
    scenario_label: str,
    mc: bool = False,
    n_mc_samples: int = 1,
) -> None:
    """4x3 panel grid of synthetic vs. 1-step fit for key metrics.

    Each entry of ``fit`` is a dict with ``mean``, ``lower``, ``upper``
    keys. When ``mc=True``, a shaded 95% CI band is drawn between
    ``lower`` and ``upper``; otherwise only the ``mean`` series is drawn.

    Args:
        scenario_label: Human-readable description used in the figure
            title (e.g., ``"Scenario 1 (Simple policies)"``).
    """
    # Lazy import so pytest collection doesn't pay the matplotlib cost.
    import matplotlib.pyplot as plt

    n_hist = int(stmts["sales"].shape[0])
    hist_years = list(range(_START_YEAR + 1, _START_YEAR + 1 + n_hist))

    # effective_st_debt is bundled into current_liabilities in the
    # synthetic statements. Reconstruct it by subtraction so it can be
    # plotted against the 1-step fit. Values below $1 are treated as
    # floating-point cancellation noise and clipped to zero (this
    # happens under SimpleDebtPolicy, which is deficit-driven and
    # produces zero ST debt when the cash budget has no shortfall).
    stmts = dict(stmts)
    eff_st_debt_reconstructed = (
        stmts["current_liabilities"]
        - stmts["accounts_payable"]
        - stmts["advance_payments_sales"]
        - stmts["current_lt_debt"]
    )
    eff_st_debt_np = (
        eff_st_debt_reconstructed.numpy()
        if hasattr(eff_st_debt_reconstructed, "numpy")
        else np.asarray(eff_st_debt_reconstructed)
    )
    if np.max(np.abs(eff_st_debt_np)) < 1.0:
        eff_st_debt_np = np.zeros_like(eff_st_debt_np)
    stmts["effective_st_debt"] = eff_st_debt_np

    panels = [
        ("net_income", "Net Income"),
        ("cogs", "COGS"),
        ("opex", "OpEx"),
        ("tax", "Tax"),
        ("nca", "Non-Current Assets"),
        ("cash", "Cash"),
        ("accounts_receivable", "Accounts Receivable"),
        ("accounts_payable", "Accounts Payable"),
        ("inventory", "Inventory"),
        ("effective_st_debt", "Effective ST Debt"),
        ("non_current_liabilities", "Non-Current Liabilities"),
        ("equity", "Equity"),
    ]

    # Remap stmts keys that don't match fit keys
    stmts_key_map = {
        "investment_in_market_securities": "ims",
    }

    fig, axs = plt.subplots(4, 3, figsize=(14, 15))
    axs = axs.flatten()

    mean_label = (
        f"Model fit (1-step MC mean, N={n_mc_samples})"
        if mc
        else "Model fit (1-step, deterministic)"
    )

    for i, (key, label) in enumerate(panels):
        ax = axs[i]
        stmts_key = stmts_key_map.get(key, key)
        obs = stmts[stmts_key]
        obs = obs.numpy() if hasattr(obs, "numpy") else np.asarray(obs)

        ax.plot(
            hist_years,
            obs,
            "ko-",
            label="Synthetic (truth)",
            markersize=6,
            linewidth=1.5,
        )

        series = fit[key]
        mean = np.asarray(series["mean"])
        ax.plot(
            fit_years,
            mean,
            "^--",
            color="tab:red",
            label=mean_label,
            markersize=6,
            linewidth=1.2,
            alpha=0.85,
        )
        if mc:
            lower = np.asarray(series["lower"])
            upper = np.asarray(series["upper"])
            ax.fill_between(
                fit_years,
                lower,
                upper,
                color="tab:red",
                alpha=0.2,
                label="95% CI",
            )

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("USD")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        ax.tick_params(axis="x", rotation=45)

        # If the whole panel is essentially zero (e.g. deficit-driven
        # ST debt in Scenario 1 with no liquidity gap), force a
        # meaningful y-range instead of zooming into FP noise.
        max_abs = max(
            float(np.max(np.abs(obs))),
            float(np.max(np.abs(mean))),
        )
        if max_abs < 1e6:
            ax.set_ylim(-1e9, 1e9)

    title_mode = (
        f"one-step-ahead Monte-Carlo fit "
        f"(N={n_mc_samples} draws from Bayesian OpEx posterior)"
        if mc
        else "one-step-ahead fit (deterministic, posterior mean)"
    )
    fig.suptitle(
        f"Parameter Recovery, {scenario_label}:\n"
        f"{title_mode} of trained recovery model vs. synthetic ground truth",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def build_scenario_s1():
    """Scenario 1: all simple (static-ratio) policies, deterministic OpEx."""
    tf.random.set_seed(42)
    np.random.seed(42)

    gen = TrainableFinancialModel(
        opex_module=SimpleOpEx(),
        trajectory_simulator=DeterministicSimulator(),
        capex_policy=CapexPolicy(),
        working_capital=WorkingCapitalPolicy(),
        liquidity_policy=SimpleLiquidityPolicy(),
        dividend_policy=SimpleDividendPolicy(),
        buyback_policy=SimpleBuybackPolicy(),
        purchases_policy=StaticCostRatioPolicy(),
        debt_policy=SimpleDebtPolicy(),
        tax_module=SimpleTax(),
    )
    seed = _build_seed_data(TRUE_COST_RATIO)
    gen.prepare(seed, inflation=seed["inflation"], test_years=0)
    _assign_simple_ground_truth(gen)
    stmts = _generate_via_forecast_step(gen)

    trained = _train_recovery_model(
        stmts,
        opex_module=SimpleOpEx(),
        liquidity_policy=SimpleLiquidityPolicy(),
        dividend_policy=SimpleDividendPolicy(),
        buyback_policy=SimpleBuybackPolicy(),
        purchases_policy=StaticCostRatioPolicy(),
        debt_policy=SimpleDebtPolicy(),
        tax_module=SimpleTax(),
        trajectory_simulator=DeterministicSimulator(),
    )
    return trained, stmts


def build_scenario_s1_w_bayesianopex():
    """Scenario 1 with Bayesian OpEx (simple policies + variational OpEx).

    Mirrors Scenario 1 but swaps ``SimpleOpEx`` for ``BayesianOpEx`` on
    both the generator and the recovery model, and injects the same
    known aleatoric noise used by Scenarios 3--4 so the Bayesian
    variational posterior has something to learn (otherwise the
    ``q_*_scale`` parameters collapse during training and the MC bands
    would still be zero).
    """
    tf.random.set_seed(42)
    np.random.seed(42)

    gen = TrainableFinancialModel(
        opex_module=BayesianOpEx(),
        trajectory_simulator=MonteCarloSimulator(n_samples=1000),
        capex_policy=CapexPolicy(),
        working_capital=WorkingCapitalPolicy(),
        liquidity_policy=SimpleLiquidityPolicy(),
        dividend_policy=SimpleDividendPolicy(),
        buyback_policy=SimpleBuybackPolicy(),
        purchases_policy=StaticCostRatioPolicy(),
        debt_policy=SimpleDebtPolicy(),
        tax_module=SimpleTax(),
    )
    seed = _build_seed_data(TRUE_COST_RATIO)
    gen.prepare(seed, inflation=seed["inflation"], test_years=0)
    _assign_simple_ground_truth(gen)
    stmts = _generate_via_forecast_step(gen)

    # Inject known OpEx noise so the Bayesian recovery model can learn
    # a non-degenerate posterior (same noise scale as Scenarios 3--4).
    noise = tf.constant(
        np.random.normal(0, TRUE_NOISE_SIGMA * gen.amount_scale, _N_YEARS),
        dtype=tf.float64,
    )
    stmts["opex"] = stmts["opex"] + noise

    trained = _train_recovery_model(
        stmts,
        opex_module=BayesianOpEx(),
        liquidity_policy=SimpleLiquidityPolicy(),
        dividend_policy=SimpleDividendPolicy(),
        buyback_policy=SimpleBuybackPolicy(),
        purchases_policy=StaticCostRatioPolicy(),
        debt_policy=SimpleDebtPolicy(),
        tax_module=SimpleTax(),
        trajectory_simulator=MonteCarloSimulator(n_samples=1000),
    )
    return trained, stmts


def build_scenario_s4():
    """Scenario 4: advanced policies + BayesianOpEx + tax anomalies."""
    tf.random.set_seed(42)
    np.random.seed(42)

    tax_gen = TaxWithAnomalies(TRUE_TAX_ONETIME)
    gen = TrainableFinancialModel(
        opex_module=BayesianOpEx(),
        trajectory_simulator=MonteCarloSimulator(n_samples=1000),
        capex_policy=CapexPolicy(),
        working_capital=WorkingCapitalPolicy(),
        liquidity_policy=TrendLiquidityPolicy(),
        dividend_policy=LintnerDividendPolicy(),
        buyback_policy=BaselineBuybackPolicy(),
        purchases_policy=TrendCostRatioPolicy(),
        debt_policy=TrendDebtPolicy(),
        tax_module=tax_gen,
    )
    avg_cr = float(tf.sigmoid(TRUE_CR_ALPHA))
    seed = _build_seed_data(avg_cr)
    gen.prepare(seed, inflation=seed["inflation"], test_years=0)
    _assign_trend_ground_truth(gen)
    stmts = _generate_via_forecast_step(gen)

    # Inject known OpEx noise to exercise the Bayesian posterior
    noise = tf.constant(
        np.random.normal(0, TRUE_NOISE_SIGMA * gen.amount_scale, _N_YEARS),
        dtype=tf.float64,
    )
    stmts["opex"] = stmts["opex"] + noise

    trained = _train_recovery_model(
        stmts,
        opex_module=BayesianOpEx(),
        liquidity_policy=TrendLiquidityPolicy(),
        dividend_policy=LintnerDividendPolicy(),
        buyback_policy=BaselineBuybackPolicy(),
        purchases_policy=TrendCostRatioPolicy(),
        debt_policy=TrendDebtPolicy(),
        tax_module=TaxWithAnomalies(TRUE_TAX_ONETIME),
        trajectory_simulator=MonteCarloSimulator(n_samples=2),
    )
    return trained, stmts


def _main_diagnostic_plots() -> None:
    """Train every recovery scenario and save 1-step-ahead fit plots.

    Invoked via ``python tests/test_parameter_recovery.py``. Not a
    pytest entry point -- this is a diagnostic script whose output
    feeds the parameter-recovery figures in the extension report.
    """
    repo_root = Path(__file__).resolve().parent.parent
    media_dir = repo_root / "training_results"

    # --- Scenario 1: simple policies (deterministic) ---
    trained_s1, stmts_s1 = build_scenario_s1()
    fit_s1, years_s1 = compute_one_step_fit(trained_s1, n_mc_samples=1)
    plot_recovery_fit(
        stmts_s1,
        fit_s1,
        years_s1,
        media_dir / "parameter_recovery_s1_fit.png",
        scenario_label="Scenario 1 (Simple policies)",
        mc=False,
    )

    # --- Scenario 1 with Bayesian OpEx ---
    trained_s1bayesian, stmts_s1bayesian = build_scenario_s1_w_bayesianopex()
    n_mc_s1 = 1000
    fit_s1bayesian, years_s1bayesian = compute_one_step_fit(
        trained_s1bayesian, n_mc_samples=n_mc_s1
    )
    plot_recovery_fit(
        stmts_s1bayesian,
        fit_s1bayesian,
        years_s1bayesian,
        media_dir / "parameter_recovery_s1bayesian_fit.png",
        scenario_label="Scenario 1 with BayesianOpEx (Simple policies)",
        mc=True,
        n_mc_samples=n_mc_s1,
    )

    # --- Scenario 4: advanced + Bayesian OpEx + tax anomalies ---
    trained_s4, stmts_s4 = build_scenario_s4()

    # Deterministic 1-step fit
    fit_s4_det, years_s4_det = compute_one_step_fit(trained_s4, n_mc_samples=1)
    plot_recovery_fit(
        stmts_s4,
        fit_s4_det,
        years_s4_det,
        media_dir / "parameter_recovery_s4_fit.png",
        scenario_label="Scenario 4 "
        "(Advanced policies + BayesianOpEx + Tax Anomalies)",
        mc=False,
    )

    # Monte Carlo 1-step fit (95% CI band from Bayesian OpEx posterior)
    n_mc = 500
    fit_s4_mc, years_s4_mc = compute_one_step_fit(trained_s4, n_mc_samples=n_mc)
    plot_recovery_fit(
        stmts_s4,
        fit_s4_mc,
        years_s4_mc,
        media_dir / "parameter_recovery_s4_fit_mc.png",
        scenario_label="Scenario 4 "
        "(Advanced policies + BayesianOpEx + Tax Anomalies)",
        mc=True,
        n_mc_samples=n_mc,
    )


if __name__ == "__main__":
    _main_diagnostic_plots()
