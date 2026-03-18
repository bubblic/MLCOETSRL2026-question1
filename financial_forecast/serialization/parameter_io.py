"""Save and load Bayesian financial model parameters to/from .npz files.

All serialization logic is concentrated here so the model class stays
focused on forecasting.  The functions operate on any object that exposes
the expected parameter attributes (duck-typed), keeping the dependency
one-directional: ``serialization → models/base``, never upward.
"""

import os

import numpy as np


def save_parameters(model, path: str) -> None:
    """Serialize every learnable parameter to a NumPy ``.npz`` archive.

    Args:
        model: A trained model instance whose attributes will be read.
        path: Filesystem path for the output ``.npz`` file.
    """
    params = {
        # Policy parameters
        "asset_growth": float(model.asset_growth.numpy()),
        "asset_maintain": float(model.asset_maintain.numpy()),
        "depreciation_rate": float(model.depreciation_rate.numpy()),
        "advance_payments_sales_pct": float(model.advance_payments_sales_pct.numpy()),
        "advance_payments_purchases_pct": float(
            model.advance_payments_purchases_pct.numpy()
        ),
        "account_receivables_pct": float(model.account_receivables_pct.numpy()),
        "account_payables_pct": float(model.account_payables_pct.numpy()),
        "inventory_pct": float(model.inventory_pct.numpy()),
        "tl_alpha": float(model.tl_alpha.numpy()),
        "tl_beta": float(model.tl_beta.numpy()),
        "tl_baseline": float(model.tl_baseline.numpy()),
        "cash_alpha": float(model.cash_alpha.numpy()),
        "cash_beta": float(model.cash_beta.numpy()),
        "income_tax_pct": float(model.income_tax_pct.numpy()),
        "dividend_payout_ratio_pct": float(model.dividend_payout_ratio_pct.numpy()),
        "dividend_adjustment_speed": float(model.dividend_adjustment_speed.numpy()),
        "sb_baseline": float(model.sb_baseline.numpy()),
        "sb_ratio": float(model.sb_ratio.numpy()),
        "cost_ratio_alpha": float(model.cost_ratio_alpha.numpy()),
        "cost_ratio_beta": float(model.cost_ratio_beta.numpy()),
        "st_debt_alpha": float(model.st_debt_alpha.numpy()),
        "st_debt_beta": float(model.st_debt_beta.numpy()),
        # Bayesian OpEx parameters
        "q_var_opex_loc": float(model.q_var_opex_loc.numpy()),
        "q_var_opex_scale": float(model.q_var_opex_scale.numpy()),
        "q_base_opex_loc": float(model.q_base_opex_loc.numpy()),
        "q_base_opex_scale": float(model.q_base_opex_scale.numpy()),
        "noise_sigma": float(model.noise_sigma.numpy()),
        "sales_offset": float(model.sales_offset.numpy()),
        # Structural parameters
        "avg_short_term_interest_pct": float(
            model.avg_short_term_interest_pct.numpy()
        ),
        "avg_long_term_interest_pct": float(
            model.avg_long_term_interest_pct.numpy()
        ),
        "avg_maturity_years": float(model.avg_maturity_years.numpy()),
        "market_securities_return_pct": float(
            model.market_securities_return_pct.numpy()
        ),
        "ef_alpha": float(model.ef_alpha.numpy()),
        "ef_beta": float(model.ef_beta.numpy()),
        # Metadata
        "base_year": model.base_year,
    }
    np.savez(path, **params)


def load_parameters(model, path: str) -> None:
    """Restore model parameters from a previously saved ``.npz`` archive.

    Missing keys are assigned sensible defaults for backward compatibility
    with older parameter files.

    Args:
        model: A model instance whose attributes will be updated in-place.
        path: Filesystem path to the ``.npz`` parameter file.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parameter file not found: {path}")
    data = np.load(path)

    # Policy parameters (Layer 1)
    model.asset_growth.assign(data["asset_growth"])
    model.asset_maintain.assign(data["asset_maintain"])
    model.depreciation_rate.assign(data["depreciation_rate"])
    model.advance_payments_sales_pct.assign(data["advance_payments_sales_pct"])
    model.advance_payments_purchases_pct.assign(
        data["advance_payments_purchases_pct"]
    )
    model.account_receivables_pct.assign(data["account_receivables_pct"])
    model.account_payables_pct.assign(data["account_payables_pct"])
    model.inventory_pct.assign(data["inventory_pct"])
    model.tl_alpha.assign(data["tl_alpha"])
    model.tl_beta.assign(data["tl_beta"])
    # .get() with defaults provides backward compat with older parameter files
    # that predate these parameters.
    model.tl_baseline.assign(data.get("tl_baseline", 0.0))
    model.cash_alpha.assign(data["cash_alpha"])
    model.cash_beta.assign(data["cash_beta"])
    model.income_tax_pct.assign(data["income_tax_pct"])
    model.dividend_payout_ratio_pct.assign(data["dividend_payout_ratio_pct"])
    model.dividend_adjustment_speed.assign(
        data.get("dividend_adjustment_speed", 1.0)
    )
    model.sb_baseline.assign(data.get("sb_baseline", 0.0))
    model.sb_ratio.assign(data.get("sb_ratio", 1.0))
    model.cost_ratio_alpha.assign(data["cost_ratio_alpha"])
    model.cost_ratio_beta.assign(data["cost_ratio_beta"])
    model.st_debt_alpha.assign(data.get("st_debt_alpha", -1.59))
    model.st_debt_beta.assign(data.get("st_debt_beta", 0.0))

    # Bayesian OpEx parameters
    model.q_var_opex_loc.assign(data["q_var_opex_loc"])
    model.q_var_opex_scale.assign(data["q_var_opex_scale"])
    model.q_base_opex_loc.assign(data["q_base_opex_loc"])
    model.q_base_opex_scale.assign(data["q_base_opex_scale"])
    model.noise_sigma.assign(data["noise_sigma"])
    model.sales_offset.assign(data["sales_offset"])

    # Structural parameters
    model.avg_short_term_interest_pct.assign(data["avg_short_term_interest_pct"])
    model.avg_long_term_interest_pct.assign(data["avg_long_term_interest_pct"])
    model.avg_maturity_years.assign(data["avg_maturity_years"])
    model.market_securities_return_pct.assign(data["market_securities_return_pct"])
    model.ef_alpha.assign(data["ef_alpha"])
    model.ef_beta.assign(data["ef_beta"])

    # Metadata
    if "base_year" in data:
        model.base_year = int(data["base_year"])
