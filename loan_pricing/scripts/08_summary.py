"""Generate summary comparison figures and tables."""

from __future__ import annotations

import numpy as np

from loan_pricing.logging_config import get_logger
from loan_pricing.reporting.tables import write_model_comparison_table
from loan_pricing.scripts._pipeline import TABLES_DIR, run_pipeline

logger = get_logger(__name__)


def main() -> None:
    """Generate all summary outputs."""
    p = run_pipeline()

    # OLS baseline: simple linear regression on the same features.
    from numpy.linalg import lstsq

    X_aug = np.column_stack([p.X_train, np.ones(len(p.X_train))])
    coeffs, _, _, _ = lstsq(X_aug, p.y_spread_train, rcond=None)
    X_test_aug = np.column_stack([p.X_test, np.ones(len(p.X_test))])
    ols_pred = X_test_aug @ coeffs

    ols_rmse = float(np.sqrt(np.mean((p.y_spread_test - ols_pred) ** 2)))
    ols_mae = float(np.mean(np.abs(p.y_spread_test - ols_pred)))
    ss_res = float(np.sum((p.y_spread_test - ols_pred) ** 2))
    ss_tot = float(np.sum((p.y_spread_test - p.y_spread_test.mean()) ** 2))
    ols_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    # 95% CI coverage from conformal.
    cov_95 = p.coverage_results.get("95%", {})

    results = [
        {
            "model_name": "OLS Baseline",
            "rmse_bps": round(ols_rmse, 1),
            "mae_bps": round(ols_mae, 1),
            "r_squared": round(ols_r2, 4),
            "ci_95_coverage": "—",
            "ci_avg_width_bps": "—",
        },
        {
            "model_name": "TF Neural Network (ours)",
            "rmse_bps": round(p.spread_eval.rmse_bps, 1),
            "mae_bps": round(p.spread_eval.mae_bps, 1),
            "r_squared": round(p.spread_eval.r_squared, 4),
            "ci_95_coverage": round(cov_95.get("actual", 0), 3),
            "ci_avg_width_bps": round(cov_95.get("avg_width_bps", 0), 1),
        },
    ]

    write_model_comparison_table(results, TABLES_DIR / "tbl_08_comparison.csv")

    logger.info("08_summary complete")


if __name__ == "__main__":
    main()
