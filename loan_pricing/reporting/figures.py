"""Figure-generation functions for the term loan pricing report.

Every public function in this module is a **pure function**: it
accepts data arguments and an ``output_path``, writes exactly one
``.png`` file atomically, and returns the path it wrote.

All functions use the non-interactive ``Agg`` backend so they run
cleanly in headless environments.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from loan_pricing.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

FIGURE_PALETTE: tuple[str, ...] = (
    "#003f5c",  # navy
    "#58508d",  # purple
    "#bc5090",  # pink
    "#ff6361",  # coral
    "#ffa600",  # amber
    "#374c80",  # slate
    "#7a5195",  # mauve
    "#ef5675",  # rose
    "#2f4b7c",  # dark blue
    "#665191",  # violet
)

_DPI = 300
_FACECOLOR = "white"


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    """Atomically save a figure and close it.

    Args:
        fig: Matplotlib figure to save.
        output_path: Target ``.png`` path.

    Returns:
        The *output_path* that was written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.parent / (output_path.stem + "_tmp.png")
    fig.savefig(str(tmp), dpi=_DPI, facecolor=_FACECOLOR, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(output_path)
    logger.info("Wrote %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------


def plot_spread_distribution(
    spreads: np.ndarray,
    output_path: Path,
) -> Path:
    """Histogram of observed credit spreads.

    Args:
        spreads: 1-D array of spread values in bps.
        output_path: Destination ``.png``.

    Returns:
        Path written.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(spreads, bins=40, color=FIGURE_PALETTE[0], edgecolor="white", alpha=0.85)
    ax.set_xlabel("Credit Spread (bps)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Observed Credit Spreads")
    return _save_figure(fig, output_path)


def plot_roc_and_pr_curves(
    fpr: np.ndarray,
    tpr: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
    roc_auc: float,
    pr_auc: float,
    output_path: Path,
) -> Path:
    """Side-by-side ROC and Precision-Recall curves.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        precision: Precision values.
        recall: Recall values.
        roc_auc: Area under the ROC curve.
        pr_auc: Area under the PR curve.
        output_path: Destination ``.png``.

    Returns:
        Path written.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(fpr, tpr, color=FIGURE_PALETTE[0], lw=2, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()

    ax2.plot(recall, precision, color=FIGURE_PALETTE[2], lw=2, label=f"AP = {pr_auc:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()

    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_calibration_curve(
    fraction_of_positives: np.ndarray,
    mean_predicted_value: np.ndarray,
    output_path: Path,
) -> Path:
    """Calibration (reliability) diagram.

    Args:
        fraction_of_positives: Observed positive fraction per bin.
        mean_predicted_value: Mean predicted probability per bin.
        output_path: Destination ``.png``.

    Returns:
        Path written.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", color=FIGURE_PALETTE[1], lw=2)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    return _save_figure(fig, output_path)


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    top_n: int = 15,
) -> Path:
    """Horizontal bar chart of mean absolute SHAP values.

    Args:
        shap_values: SHAP matrix ``(n_samples, n_features)``.
        feature_names: Feature names.
        output_path: Destination ``.png``.
        top_n: Number of top features to show.

    Returns:
        Path written.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in reversed(order)],
        mean_abs[order][::-1],
        color=FIGURE_PALETTE[3],
    )
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Feature Importance (SHAP)")
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_predicted_vs_actual(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    output_path: Path,
) -> Path:
    """Scatter plot of predicted vs actual spread.

    Args:
        y_pred: Predicted spreads (bps).
        y_true: Actual spreads (bps).
        output_path: Destination ``.png``.

    Returns:
        Path written.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.5, s=15, color=FIGURE_PALETTE[0])
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "--", color="grey", lw=1)
    ax.set_xlabel("Actual Spread (bps)")
    ax.set_ylabel("Predicted Spread (bps)")
    ax.set_title("Predicted vs Actual")
    return _save_figure(fig, output_path)


def plot_residuals(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    output_path: Path,
) -> Path:
    """Residual plot (predicted on x, residual on y).

    Args:
        y_pred: Predicted spreads (bps).
        y_true: Actual spreads (bps).
        output_path: Destination ``.png``.

    Returns:
        Path written.
    """
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.5, s=15, color=FIGURE_PALETTE[1])
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_xlabel("Predicted Spread (bps)")
    ax.set_ylabel("Residual (bps)")
    ax.set_title("Residual Plot")
    return _save_figure(fig, output_path)


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    top_n: int = 15,
) -> Path:
    """Beeswarm plot of SHAP values (simplified strip plot).

    Args:
        shap_values: SHAP matrix ``(n_samples, n_features)``.
        feature_names: Feature names.
        output_path: Destination ``.png``.
        top_n: Number of features to display.

    Returns:
        Path written.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    n_show = min(top_n, len(feature_names))
    order = np.argsort(mean_abs)[::-1][:n_show]

    fig, ax = plt.subplots(figsize=(8, 6))
    for rank, idx in enumerate(reversed(order)):
        vals = shap_values[:, idx]
        jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(vals))
        ax.scatter(vals, rank + jitter, alpha=0.3, s=5, color=FIGURE_PALETTE[4])

    ax.set_yticks(range(n_show))
    ax.set_yticklabels([feature_names[i] for i in reversed(order)])
    ax.set_xlabel("SHAP Value")
    ax.set_title("SHAP Beeswarm")
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_optuna_history(
    trial_values: list[float],
    output_path: Path,
) -> Path:
    """Optuna optimisation history (objective value per trial).

    Args:
        trial_values: Objective value for each completed trial.
        output_path: Destination ``.png``.

    Returns:
        Path written.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    trials = range(1, len(trial_values) + 1)
    ax.plot(list(trials), trial_values, "o-", ms=4, color=FIGURE_PALETTE[5])
    best_so_far = np.minimum.accumulate(trial_values)
    ax.plot(list(trials), best_so_far, "--", color=FIGURE_PALETTE[3], lw=2, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective Value")
    ax.set_title("Hyperparameter Tuning History")
    ax.legend()
    return _save_figure(fig, output_path)


def plot_mc_price_histogram(
    simulated_spreads: np.ndarray,
    mean_spread: float,
    ci_lower: float,
    ci_upper: float,
    output_path: Path,
) -> Path:
    """Histogram of Monte Carlo simulated terminal spreads.

    Args:
        simulated_spreads: 1-D array of terminal spread samples.
        mean_spread: Mean of the distribution.
        ci_lower: Lower CI bound.
        ci_upper: Upper CI bound.
        output_path: Destination ``.png``.

    Returns:
        Path written.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(simulated_spreads, bins=60, color=FIGURE_PALETTE[0], edgecolor="white", alpha=0.8)
    ax.axvline(mean_spread, color=FIGURE_PALETTE[3], ls="-", lw=2, label=f"Mean = {mean_spread:.1f}")
    ax.axvline(ci_lower, color=FIGURE_PALETTE[2], ls="--", lw=1.5, label=f"CI lower = {ci_lower:.1f}")
    ax.axvline(ci_upper, color=FIGURE_PALETTE[2], ls="--", lw=1.5, label=f"CI upper = {ci_upper:.1f}")
    ax.set_xlabel("Simulated Spread (bps)")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo Spread Distribution")
    ax.legend()
    return _save_figure(fig, output_path)


def plot_mc_fan_chart(
    percentile_paths: dict[str, np.ndarray],
    time_grid: np.ndarray,
    output_path: Path,
) -> Path:
    """Fan chart of Monte Carlo spread percentile paths.

    Args:
        percentile_paths: Mapping from label (e.g. ``"p10"``,
            ``"p50"``, ``"p90"``) to 1-D arrays of spread values
            over time.
        time_grid: 1-D array of time points (years).
        output_path: Destination ``.png``.

    Returns:
        Path written.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = sorted(percentile_paths.keys())
    if "p50" in percentile_paths:
        ax.plot(time_grid, percentile_paths["p50"], color=FIGURE_PALETTE[0], lw=2, label="Median")

    if "p10" in percentile_paths and "p90" in percentile_paths:
        ax.fill_between(
            time_grid,
            percentile_paths["p10"],
            percentile_paths["p90"],
            alpha=0.25, color=FIGURE_PALETTE[1], label="80% CI",
        )
    if "p2.5" in percentile_paths and "p97.5" in percentile_paths:
        ax.fill_between(
            time_grid,
            percentile_paths["p2.5"],
            percentile_paths["p97.5"],
            alpha=0.12, color=FIGURE_PALETTE[2], label="95% CI",
        )

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Credit Spread (bps)")
    ax.set_title("Monte Carlo Fan Chart")
    ax.legend()
    return _save_figure(fig, output_path)
