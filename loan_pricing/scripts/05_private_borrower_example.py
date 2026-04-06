"""Generate private borrower pricing example outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from loan_pricing.logging_config import get_logger
from loan_pricing.models.spread_model import LoanPricer, LoanPricingInput
from loan_pricing.scripts._pipeline import TABLES_DIR, run_pipeline

logger = get_logger(__name__)


def _altman_z_double_prime(
    working_capital_to_ta: float,
    retained_earnings_to_ta: float,
    ebit_to_ta: float,
    book_equity_to_tl: float,
) -> float:
    """Compute Altman Z''-score for private firms."""
    return (
        6.56 * working_capital_to_ta
        + 3.26 * retained_earnings_to_ta
        + 6.72 * ebit_to_ta
        + 1.05 * book_equity_to_tl
    )


def main() -> None:
    """Generate the private borrower worked example."""
    p = run_pipeline()

    # Define a synthetic private borrower.
    borrower = {
        "company": "Acme Manufacturing LLC (Private)",
        "revenue_mm": 250.0,
        "ebitda_mm": 37.5,
        "total_assets_mm": 400.0,
        "total_liabilities_mm": 280.0,
        "long_term_debt_mm": 150.0,
        "cash_mm": 20.0,
        "retained_earnings_mm": 45.0,
        "working_capital_mm": 50.0,
        "interest_expense_mm": 12.0,
        "book_equity_mm": 120.0,
    }

    # Compute financial ratios.
    debt_to_ebitda = borrower["long_term_debt_mm"] / borrower["ebitda_mm"]
    interest_coverage = borrower["ebitda_mm"] / borrower["interest_expense_mm"]
    net_debt_to_equity = (borrower["long_term_debt_mm"] - borrower["cash_mm"]) / borrower["book_equity_mm"]
    ebitda_margin = borrower["ebitda_mm"] / borrower["revenue_mm"]

    # Z''-score.
    z_pp = _altman_z_double_prime(
        working_capital_to_ta=borrower["working_capital_mm"] / borrower["total_assets_mm"],
        retained_earnings_to_ta=borrower["retained_earnings_mm"] / borrower["total_assets_mm"],
        ebit_to_ta=(borrower["ebitda_mm"] * 0.85) / borrower["total_assets_mm"],
        book_equity_to_tl=borrower["book_equity_mm"] / borrower["total_liabilities_mm"],
    )

    # Price the loan.
    loan_input = LoanPricingInput(
        financial_ratios={
            "debt_to_ebitda": debt_to_ebitda,
            "interest_coverage_ratio": interest_coverage,
            "net_debt_to_equity": net_debt_to_equity,
            "ebitda_margin": ebitda_margin,
            "altman_z_double_prime": z_pp,
        },
        loan_maturity_years=5.0,
        loan_size_mm=75.0,
        is_secured=True,
        industry_naics="3329",
        treasury_yield_pct=4.25,
    )

    pricer = LoanPricer(
        pd_model=p.pd_model,
        spread_model=p.spread_model,
        quantile_models=(p.quantile_lower, p.quantile_upper),
        feature_engineer=p.feature_engineer,
    )
    result = pricer.price(loan_input)

    # Add illiquidity premium for private company.
    illiquidity_premium_bps = 150.0
    adjusted_spread = result.credit_spread_bps + illiquidity_premium_bps
    adjusted_all_in = loan_input.treasury_yield_pct + adjusted_spread / 100.0

    # Save as JSON for the LaTeX report to reference.
    example = {
        "borrower": borrower,
        "ratios": {
            "debt_to_ebitda": round(debt_to_ebitda, 2),
            "interest_coverage_ratio": round(interest_coverage, 2),
            "net_debt_to_equity": round(net_debt_to_equity, 2),
            "ebitda_margin": round(ebitda_margin, 3),
        },
        "altman_z_double_prime": round(z_pp, 2),
        "internal_rating": result.internal_rating,
        "estimated_pd": round(result.estimated_pd, 4),
        "base_spread_bps": round(result.credit_spread_bps, 1),
        "illiquidity_premium_bps": illiquidity_premium_bps,
        "adjusted_spread_bps": round(adjusted_spread, 1),
        "treasury_yield_pct": loan_input.treasury_yield_pct,
        "all_in_rate_pct": round(adjusted_all_in, 2),
    }

    out_path = TABLES_DIR / "tbl_05_private_borrower.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(example, indent=2))
    logger.info("Wrote %s", out_path)
    logger.info("05_private_borrower_example complete")


if __name__ == "__main__":
    main()
