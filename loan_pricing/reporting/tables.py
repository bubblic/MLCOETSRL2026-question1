"""Table-generation functions for the term loan pricing report.

Every public function writes a single ``.csv`` file atomically and
returns the path it wrote.  The CSV can be consumed directly by
``pgfplotstable`` in LaTeX or post-processed into a ``tabular``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from loan_pricing.logging_config import get_logger
from loan_pricing.models.ou_calibration import OUParameters
from loan_pricing.models.pd_model import PDEvaluationResult
from loan_pricing.models.spread_model import SpreadEvaluationResult

logger = get_logger(__name__)


def _write_csv_atomic(df: pd.DataFrame, output_path: Path) -> Path:
    """Atomically write a DataFrame to CSV.

    Args:
        df: Data to write.
        output_path: Target ``.csv`` path.

    Returns:
        The *output_path* that was written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.parent / (output_path.stem + "_tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(output_path)
    logger.info("Wrote %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Table functions
# ---------------------------------------------------------------------------


def write_dataset_summary(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Summary statistics for the modelling dataset.

    Args:
        df: Full (or training) dataset.
        output_path: Destination ``.csv``.

    Returns:
        Path written.
    """
    numeric = df.select_dtypes(include="number")
    summary = numeric.describe().T
    summary.index.name = "feature"
    summary = summary.reset_index()
    return _write_csv_atomic(summary, output_path)


def write_pd_eval_metrics(
    eval_result: PDEvaluationResult,
    output_path: Path,
) -> Path:
    """PD model evaluation metrics as a single-row CSV.

    Args:
        eval_result: Output of
            :func:`~loan_pricing.models.pd_model.evaluate_pd_model`.
        output_path: Destination ``.csv``.

    Returns:
        Path written.
    """
    row = {
        "roc_auc": eval_result.roc_auc,
        "pr_auc": eval_result.pr_auc,
        "brier_score": eval_result.brier_score,
        "ks_statistic": eval_result.ks_statistic,
        "gini_coefficient": eval_result.gini_coefficient,
    }
    df = pd.DataFrame([row])
    return _write_csv_atomic(df, output_path)


def write_spread_eval_metrics(
    eval_result: SpreadEvaluationResult,
    output_path: Path,
) -> Path:
    """Spread model evaluation metrics as a single-row CSV.

    Args:
        eval_result: Output of
            :func:`~loan_pricing.models.spread_model.evaluate_spread_model`.
        output_path: Destination ``.csv``.

    Returns:
        Path written.
    """
    row = {
        "rmse_bps": eval_result.rmse_bps,
        "mae_bps": eval_result.mae_bps,
        "r_squared": eval_result.r_squared,
        "mape": eval_result.mape,
        "median_absolute_error_bps": eval_result.median_absolute_error_bps,
    }
    df = pd.DataFrame([row])
    return _write_csv_atomic(df, output_path)


def write_hyperparameter_table(
    best_params: dict[str, object],
    all_trials: list[dict[str, object]],
    output_path: Path,
) -> Path:
    """Hyperparameter tuning results.

    Args:
        best_params: Best hyperparameter set found.
        all_trials: List of trial dicts with at least ``"value"``
            and parameter keys.
        output_path: Destination ``.csv``.

    Returns:
        Path written.
    """
    df = pd.DataFrame(all_trials)
    return _write_csv_atomic(df, output_path)


def write_ou_calibration_table(
    ou_params: OUParameters,
    output_path: Path,
) -> Path:
    """OU calibration results: point estimates and standard errors.

    Args:
        ou_params: Fitted OU parameters.
        output_path: Destination ``.csv``.

    Returns:
        Path written.
    """
    rows = [
        {"parameter": "kappa", "estimate": ou_params.kappa, "std_error": ou_params.se_kappa},
        {"parameter": "theta", "estimate": ou_params.theta, "std_error": ou_params.se_theta},
        {"parameter": "sigma", "estimate": ou_params.sigma, "std_error": ou_params.se_sigma},
    ]
    df = pd.DataFrame(rows)
    return _write_csv_atomic(df, output_path)


def write_coverage_table(
    coverage_results: dict[str, float],
    output_path: Path,
) -> Path:
    """Conformal prediction interval coverage results.

    Args:
        coverage_results: Mapping from label (e.g.
            ``"95% nominal"``) to empirical coverage fraction.
        output_path: Destination ``.csv``.

    Returns:
        Path written.
    """
    rows = [
        {"interval": label, "empirical_coverage": cov}
        for label, cov in coverage_results.items()
    ]
    df = pd.DataFrame(rows)
    return _write_csv_atomic(df, output_path)


def write_model_comparison_table(
    results: list[dict[str, object]],
    output_path: Path,
) -> Path:
    """Side-by-side comparison of different model configurations.

    Args:
        results: List of dicts, each representing one model's metrics
            (must include at least ``"model_name"``).
        output_path: Destination ``.csv``.

    Returns:
        Path written.
    """
    df = pd.DataFrame(results)
    return _write_csv_atomic(df, output_path)
