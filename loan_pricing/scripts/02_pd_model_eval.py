"""Generate PD model evaluation figures and tables."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from loan_pricing.logging_config import get_logger
from loan_pricing.reporting.figures import (
    plot_calibration_curve,
    plot_roc_and_pr_curves,
    plot_shap_bar,
)
from loan_pricing.reporting.tables import write_pd_eval_metrics
from loan_pricing.scripts._pipeline import FIGURES_DIR, TABLES_DIR, run_pipeline

logger = get_logger(__name__)


def main() -> None:
    """Generate all PD model evaluation outputs."""
    p = run_pipeline()

    # Metrics table.
    write_pd_eval_metrics(p.pd_eval, TABLES_DIR / "tbl_02_pd_metrics.csv")

    # ROC + PR curves.
    y_prob = p.pd_model.predict_proba(p.X_test)
    fpr, tpr, _ = roc_curve(p.y_default_test, y_prob)
    precision, recall, _ = precision_recall_curve(p.y_default_test, y_prob)

    plot_roc_and_pr_curves(
        fpr, tpr, precision, recall,
        p.pd_eval.roc_auc, p.pd_eval.pr_auc,
        FIGURES_DIR / "fig_02_roc_pr.png",
    )

    # Calibration curve.
    n_bins = min(10, max(2, int(np.sum(p.y_default_test))))
    frac_pos, mean_pred = calibration_curve(p.y_default_test, y_prob, n_bins=n_bins)
    plot_calibration_curve(frac_pos, mean_pred, FIGURES_DIR / "fig_02_calibration.png")

    # SHAP bar (approximate with absolute prediction differences).
    n_features = p.X_test.shape[1]
    shap_approx = np.zeros((len(p.X_test), n_features))
    base_pred = p.pd_model.predict_proba(p.X_test)
    for j in range(n_features):
        X_permuted = p.X_test.copy()
        X_permuted[:, j] = np.random.default_rng(j).permutation(X_permuted[:, j])
        perm_pred = p.pd_model.predict_proba(X_permuted)
        shap_approx[:, j] = base_pred - perm_pred

    plot_shap_bar(shap_approx, p.feature_names, FIGURES_DIR / "fig_02_shap_bar.png")

    logger.info("02_pd_model_eval complete")


if __name__ == "__main__":
    main()
