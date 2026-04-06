"""Generate spread model evaluation figures and tables."""

from __future__ import annotations

import numpy as np

from loan_pricing.logging_config import get_logger
from loan_pricing.reporting.figures import (
    plot_predicted_vs_actual,
    plot_residuals,
    plot_shap_beeswarm,
)
from loan_pricing.reporting.tables import write_spread_eval_metrics
from loan_pricing.scripts._pipeline import FIGURES_DIR, TABLES_DIR, run_pipeline

logger = get_logger(__name__)


def main() -> None:
    """Generate all spread model evaluation outputs."""
    p = run_pipeline()

    # Metrics table.
    write_spread_eval_metrics(p.spread_eval, TABLES_DIR / "tbl_03_spread_metrics.csv")

    # Predicted vs actual + residuals.
    y_pred = p.spread_model.predict(p.X_test)
    plot_predicted_vs_actual(y_pred, p.y_spread_test, FIGURES_DIR / "fig_03_pred_vs_actual.png")
    plot_residuals(y_pred, p.y_spread_test, FIGURES_DIR / "fig_03_residuals.png")

    # SHAP beeswarm (permutation-based approximation).
    n_features = p.X_test.shape[1]
    shap_approx = np.zeros((len(p.X_test), n_features))
    base_pred = p.spread_model.predict(p.X_test)
    for j in range(n_features):
        X_permuted = p.X_test.copy()
        X_permuted[:, j] = np.random.default_rng(j).permutation(X_permuted[:, j])
        perm_pred = p.spread_model.predict(X_permuted)
        shap_approx[:, j] = base_pred - perm_pred

    plot_shap_beeswarm(shap_approx, p.feature_names, FIGURES_DIR / "fig_03_shap_beeswarm.png")

    logger.info("03_spread_model_eval complete")


if __name__ == "__main__":
    main()
