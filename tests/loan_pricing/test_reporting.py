"""Tests for the reporting layer (figures and tables)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from loan_pricing.models.ou_calibration import OUParameters
from loan_pricing.models.pd_model import PDEvaluationResult
from loan_pricing.models.spread_model import SpreadEvaluationResult
from loan_pricing.reporting.figures import (
    plot_calibration_curve,
    plot_mc_fan_chart,
    plot_mc_price_histogram,
    plot_optuna_history,
    plot_predicted_vs_actual,
    plot_residuals,
    plot_roc_and_pr_curves,
    plot_shap_bar,
    plot_shap_beeswarm,
    plot_spread_distribution,
)
from loan_pricing.reporting.tables import (
    write_coverage_table,
    write_dataset_summary,
    write_hyperparameter_table,
    write_model_comparison_table,
    write_ou_calibration_table,
    write_pd_eval_metrics,
    write_spread_eval_metrics,
)

_PNG_MAGIC = b"\x89PNG"


def _is_valid_png(path: Path) -> bool:
    """Check that a file starts with the PNG magic bytes."""
    with open(path, "rb") as f:
        return f.read(4) == _PNG_MAGIC


# ------------------------------------------------------------------
# Synthetic data helpers
# ------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_N = 100
_D = 5


# ------------------------------------------------------------------
# Figure tests
# ------------------------------------------------------------------


class TestPlotSpreadDistribution:
    def test_creates_valid_png(self, tmp_path: Path) -> None:
        out = tmp_path / "spread_dist.png"
        plot_spread_distribution(_RNG.normal(200, 50, _N), out)
        assert out.exists()
        assert out.stat().st_size > 0
        assert _is_valid_png(out)


class TestPlotRocAndPr:
    def test_creates_valid_png(self, tmp_path: Path) -> None:
        out = tmp_path / "roc_pr.png"
        fpr = np.linspace(0, 1, 50)
        tpr = np.sqrt(fpr)
        precision = 1.0 - fpr
        recall = fpr
        plot_roc_and_pr_curves(fpr, tpr, precision, recall, 0.85, 0.70, out)
        assert out.exists() and _is_valid_png(out)


class TestPlotCalibrationCurve:
    def test_creates_valid_png(self, tmp_path: Path) -> None:
        out = tmp_path / "cal.png"
        bins = np.linspace(0, 1, 10)
        plot_calibration_curve(bins, bins, out)
        assert out.exists() and _is_valid_png(out)


class TestPlotShapBar:
    def test_creates_valid_png(self, tmp_path: Path) -> None:
        out = tmp_path / "shap_bar.png"
        shap = _RNG.normal(0, 1, (_N, _D))
        names = [f"feat_{i}" for i in range(_D)]
        plot_shap_bar(shap, names, out)
        assert out.exists() and _is_valid_png(out)


class TestPlotPredictedVsActual:
    def test_creates_valid_png(self, tmp_path: Path) -> None:
        out = tmp_path / "pred_vs_actual.png"
        y = _RNG.normal(200, 50, _N)
        plot_predicted_vs_actual(y + _RNG.normal(0, 10, _N), y, out)
        assert out.exists() and _is_valid_png(out)


class TestPlotResiduals:
    def test_creates_valid_png(self, tmp_path: Path) -> None:
        out = tmp_path / "resid.png"
        y = _RNG.normal(200, 50, _N)
        plot_residuals(y + _RNG.normal(0, 10, _N), y, out)
        assert out.exists() and _is_valid_png(out)


class TestPlotShapBeeswarm:
    def test_creates_valid_png(self, tmp_path: Path) -> None:
        out = tmp_path / "beeswarm.png"
        shap = _RNG.normal(0, 1, (_N, _D))
        names = [f"feat_{i}" for i in range(_D)]
        plot_shap_beeswarm(shap, names, out)
        assert out.exists() and _is_valid_png(out)


class TestPlotOptunaHistory:
    def test_creates_valid_png(self, tmp_path: Path) -> None:
        out = tmp_path / "optuna.png"
        vals = list(_RNG.uniform(0.5, 1.0, 20))
        plot_optuna_history(vals, out)
        assert out.exists() and _is_valid_png(out)


class TestPlotMcHistogram:
    def test_creates_valid_png(self, tmp_path: Path) -> None:
        out = tmp_path / "mc_hist.png"
        sims = _RNG.normal(200, 30, 5000)
        plot_mc_price_histogram(sims, 200.0, 140.0, 260.0, out)
        assert out.exists() and _is_valid_png(out)


class TestPlotMcFanChart:
    def test_creates_valid_png(self, tmp_path: Path) -> None:
        out = tmp_path / "fan.png"
        t = np.linspace(0, 5, 50)
        paths = {
            "p2.5": 180 - 10 * t,
            "p10": 190 - 5 * t,
            "p50": 200 + 0 * t,
            "p90": 210 + 5 * t,
            "p97.5": 220 + 10 * t,
        }
        plot_mc_fan_chart(paths, t, out)
        assert out.exists() and _is_valid_png(out)


# ------------------------------------------------------------------
# Table tests
# ------------------------------------------------------------------


class TestWriteDatasetSummary:
    def test_parseable_csv(self, tmp_path: Path, synthetic_loan_df: pd.DataFrame) -> None:
        out = tmp_path / "summary.csv"
        write_dataset_summary(synthetic_loan_df, out)
        df = pd.read_csv(out)
        assert "feature" in df.columns
        assert len(df) > 0


class TestWritePdEvalMetrics:
    def test_parseable_csv(self, tmp_path: Path) -> None:
        out = tmp_path / "pd_metrics.csv"
        result = PDEvaluationResult(
            roc_auc=0.85, pr_auc=0.70, brier_score=0.15,
            ks_statistic=0.50, gini_coefficient=0.70,
        )
        write_pd_eval_metrics(result, out)
        df = pd.read_csv(out)
        assert "roc_auc" in df.columns
        assert len(df) == 1


class TestWriteSpreadEvalMetrics:
    def test_parseable_csv(self, tmp_path: Path) -> None:
        out = tmp_path / "spread_metrics.csv"
        result = SpreadEvaluationResult(
            rmse_bps=25.0, mae_bps=18.0, r_squared=0.85,
            mape=0.10, median_absolute_error_bps=15.0,
        )
        write_spread_eval_metrics(result, out)
        df = pd.read_csv(out)
        assert "rmse_bps" in df.columns


class TestWriteHyperparameterTable:
    def test_parseable_csv(self, tmp_path: Path) -> None:
        out = tmp_path / "hp.csv"
        trials = [
            {"trial": 1, "lr": 0.01, "value": 0.8},
            {"trial": 2, "lr": 0.001, "value": 0.75},
        ]
        write_hyperparameter_table({"lr": 0.001}, trials, out)
        df = pd.read_csv(out)
        assert len(df) == 2


class TestWriteOuCalibrationTable:
    def test_parseable_csv(self, tmp_path: Path) -> None:
        out = tmp_path / "ou.csv"
        params = OUParameters(
            kappa=0.5, theta=200.0, sigma=15.0,
            log_likelihood=-500.0,
            se_kappa=0.05, se_theta=2.0, se_sigma=1.0,
        )
        write_ou_calibration_table(params, out)
        df = pd.read_csv(out)
        assert "parameter" in df.columns
        assert "estimate" in df.columns
        assert "std_error" in df.columns
        assert len(df) == 3


class TestWriteCoverageTable:
    def test_parseable_csv(self, tmp_path: Path) -> None:
        out = tmp_path / "cov.csv"
        write_coverage_table({"95% nominal": 0.96, "80% nominal": 0.83}, out)
        df = pd.read_csv(out)
        assert "interval" in df.columns
        assert "empirical_coverage" in df.columns


class TestWriteModelComparisonTable:
    def test_parseable_csv(self, tmp_path: Path) -> None:
        out = tmp_path / "comp.csv"
        results = [
            {"model_name": "baseline", "rmse": 30.0},
            {"model_name": "tuned", "rmse": 22.0},
        ]
        write_model_comparison_table(results, out)
        df = pd.read_csv(out)
        assert "model_name" in df.columns
        assert len(df) == 2
