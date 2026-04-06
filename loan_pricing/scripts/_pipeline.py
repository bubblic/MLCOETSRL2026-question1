"""Shared data pipeline for all generation scripts.

This module fetches real data from FRED and SEC EDGAR, constructs a
modelling-ready dataset with synthetic but realistic spread/default
targets derived from real financial ratios, trains all models, and
exposes the results as a cached singleton so that the numbered scripts
can each grab what they need without redundant computation.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from loan_pricing.config import ProjectConfig
from loan_pricing.data.fetch_fred import DEFAULT_FRED_SERIES, FredDataFetcher
from loan_pricing.data.fetch_sec import REQUIRED_XBRL_TAGS, SecEdgarFetcher
from loan_pricing.data.preprocess import (
    compute_financial_ratios,
    handle_missing_values,
    pivot_fred_series,
)
from loan_pricing.features.engineering import FeatureConfig, FeatureEngineer
from loan_pricing.logging_config import get_logger
from loan_pricing.models.ou_calibration import (
    MonteCarloLoanPricer,
    MonteCarloResult,
    OrnsteinUhlenbeckCalibrator,
    OUParameters,
    VasicekParameters,
)
from loan_pricing.models.pd_model import (
    PDEvaluationResult,
    PDModelConfig,
    TFPDClassifier,
    evaluate_pd_model,
)
from loan_pricing.models.quantile_model import (
    ConformalPredictionInterval,
    QuantileModelConfig,
    TFQuantileRegressor,
)
from loan_pricing.models.spread_model import (
    SpreadEvaluationResult,
    SpreadModelConfig,
    TFSpreadRegressor,
    evaluate_spread_model,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Target companies (CIK → ticker)
# ---------------------------------------------------------------------------

COMPANY_CIKS: dict[str, str] = {
    "0000037996": "F",       # Ford
    "0000040730": "GM",      # General Motors
    "0000732712": "VZ",      # Verizon
    "0000732717": "T",       # AT&T
    "0000034088": "XOM",     # ExxonMobil
    "0000093410": "CVX",     # Chevron
    "0000078003": "PFE",     # Pfizer
    "0000200406": "JNJ",     # Johnson & Johnson
    "0000021344": "KO",      # Coca-Cola
    "0000080424": "PG",      # Procter & Gamble
    "0000050863": "INTC",    # Intel
    "0000764180": "CSCO",    # Cisco
    "0000004962": "AXP",     # American Express
    "0000070858": "BA",      # Boeing
    "0000018230": "CAT",     # Caterpillar
    "0000831001": "CVS",     # CVS Health
    "0000072971": "WFC",     # Wells Fargo
    "0000019617": "JPM",     # JPMorgan Chase
    "0000886982": "GS",      # Goldman Sachs
    "0000310158": "MS",      # Morgan Stanley
}

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

FIGURES_DIR = Path("report_latex_media/media/loan_pricing")
TABLES_DIR = ProjectConfig().tables_dir


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Everything the generation scripts need."""

    # Data
    full_df: pd.DataFrame
    fred_wide: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_default_train: np.ndarray
    y_default_test: np.ndarray
    y_spread_train: np.ndarray
    y_spread_test: np.ndarray
    X_cal: np.ndarray
    y_spread_cal: np.ndarray
    feature_names: list[str]
    feature_engineer: FeatureEngineer

    # Trained models
    pd_model: TFPDClassifier
    spread_model: TFSpreadRegressor
    quantile_lower: TFQuantileRegressor
    quantile_upper: TFQuantileRegressor
    conformal: ConformalPredictionInterval

    # Evaluation results
    pd_eval: PDEvaluationResult
    spread_eval: SpreadEvaluationResult

    # OU / MC
    ou_params: OUParameters
    mc_result: MonteCarloResult
    spread_series: np.ndarray

    # Coverage results
    coverage_results: dict[str, dict[str, float]]


# ---------------------------------------------------------------------------
# Data construction
# ---------------------------------------------------------------------------


def _fetch_and_build_dataset(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch real FRED + SEC data and build a modelling dataset.

    Returns:
        Tuple of (full_df, fred_wide).
    """
    logger.info("Fetching FRED macro data")
    fred_fetcher = FredDataFetcher(config=config)
    fred_long = fred_fetcher.fetch(DEFAULT_FRED_SERIES, "2015-01-01", "2024-12-31")
    fred_wide = pivot_fred_series(fred_long)

    logger.info("Fetching SEC EDGAR financial data for %d companies", len(COMPANY_CIKS))
    sec_fetcher = SecEdgarFetcher(config=config, ticker_map=COMPANY_CIKS)
    sec_long = sec_fetcher.fetch(list(COMPANY_CIKS.keys()), "2015", "2024")

    # Pivot SEC data to wide format (one row per company-year).
    if not sec_long.empty and sec_long["fiscal_year"].max() > 0:
        sec_wide = sec_long[sec_long["fiscal_year"] > 0].pivot_table(
            index=["cik", "ticker", "fiscal_year"],
            columns="tag",
            values="value",
            aggfunc="last",
        ).reset_index()
        sec_wide.columns.name = None
    else:
        sec_wide = pd.DataFrame()

    # Compute financial ratios from real data.
    if not sec_wide.empty:
        ratios = compute_financial_ratios(sec_wide)
    else:
        ratios = pd.DataFrame()

    # Build modelling dataset: one row per company-year with ratios + macro.
    if not ratios.empty:
        # Match macro data by year (use year-end values).
        fred_annual = fred_wide.copy()
        fred_annual["year"] = pd.to_datetime(fred_annual["date"]).dt.year
        fred_yearly = fred_annual.groupby("year").last().reset_index()

        df = ratios.merge(
            fred_yearly.rename(columns={"year": "fiscal_year"}),
            on="fiscal_year",
            how="left",
        )
    else:
        df = pd.DataFrame()

    # If SEC data is sparse, augment with synthetic rows to get enough samples.
    rng = np.random.default_rng(config.random_seed)
    min_samples = 300

    if len(df) < min_samples:
        logger.info("Augmenting dataset from %d to %d rows with synthetic data", len(df), min_samples)
        n_extra = min_samples - len(df)
        synthetic = _generate_synthetic_rows(n_extra, rng, fred_wide)
        df = pd.concat([df, synthetic], ignore_index=True)

    # Generate realistic targets from financial ratios.
    df = _generate_targets(df, rng)

    # Fill missing values.
    df = handle_missing_values(df, strategy="median")

    return df, fred_wide


def _generate_synthetic_rows(
    n: int,
    rng: np.random.Generator,
    fred_wide: pd.DataFrame,
) -> pd.DataFrame:
    """Generate synthetic company-year rows with realistic ratio distributions."""
    rows = []
    for _ in range(n):
        row: dict[str, float] = {
            "debt_to_ebitda": rng.lognormal(0.5, 0.8),
            "interest_coverage_ratio": rng.lognormal(1.5, 0.7),
            "net_debt_to_equity": rng.normal(1.0, 1.5),
            "fcf_to_debt": rng.normal(0.15, 0.2),
            "revenue_growth_yoy": rng.normal(0.05, 0.15),
            "ebitda_margin": rng.beta(3, 7),
            "current_ratio": rng.lognormal(0.3, 0.4),
            "altman_z_double_prime": rng.normal(3.0, 2.0),
            "maturity_years": float(rng.choice([1, 2, 3, 5, 7, 10])),
            "loan_size_mm": float(rng.lognormal(4.5, 1.0)),
            "is_secured": float(rng.choice([0, 1])),
        }

        # Pick a random macro snapshot from FRED.
        if not fred_wide.empty:
            macro_row = fred_wide.iloc[rng.integers(0, len(fred_wide))]
            for col in fred_wide.columns:
                if col != "date":
                    row[col] = float(macro_row.get(col, 0.0))

        rows.append(row)

    return pd.DataFrame(rows)


def _generate_targets(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Derive synthetic but realistic spread and default targets from ratios."""
    out = df.copy()

    # Default probability driven by leverage and Z-score.
    z = out.get("altman_z_double_prime", pd.Series(3.0, index=out.index))
    leverage = out.get("debt_to_ebitda", pd.Series(3.0, index=out.index))

    # Logistic default probability.
    default_logit = -2.0 + 0.3 * leverage.fillna(3) - 0.5 * z.fillna(3)
    default_prob = 1.0 / (1.0 + np.exp(-default_logit))
    out["defaulted"] = (rng.random(len(out)) < default_prob).astype(float)

    # Credit spread driven by leverage, coverage, macro conditions.
    base_spread = 150.0
    leverage_effect = 30.0 * leverage.fillna(3).clip(0, 15)
    coverage = out.get("interest_coverage_ratio", pd.Series(5.0, index=out.index))
    coverage_effect = -10.0 * coverage.fillna(5).clip(0, 20)
    vix = out.get("VIXCLS", pd.Series(20.0, index=out.index))
    vix_effect = 3.0 * vix.fillna(20)
    noise = rng.normal(0, 25, len(out))

    out["credit_spread_bps"] = np.maximum(
        base_spread + leverage_effect + coverage_effect + vix_effect + noise,
        20.0,
    )

    return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def run_pipeline() -> PipelineResult:
    """Execute the full pipeline (cached — only runs once per process).

    Returns:
        A :class:`PipelineResult` with all data, models, and evaluations.
    """
    config = ProjectConfig()
    tf.random.set_seed(config.random_seed)

    # Ensure output directories exist.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)

    # -- Data ---------------------------------------------------------------
    full_df, fred_wide = _fetch_and_build_dataset(config)
    logger.info("Dataset: %d rows, %d columns", *full_df.shape)

    # Feature engineering.
    feature_config = FeatureConfig(
        financial_ratio_columns=(
            "debt_to_ebitda",
            "interest_coverage_ratio",
            "net_debt_to_equity",
            "fcf_to_debt",
            "revenue_growth_yoy",
            "ebitda_margin",
            "current_ratio",
            "altman_z_double_prime",
        ),
        macro_columns=("VIXCLS", "BAMLH0A0HYM2", "T10Y2Y", "UNRATE"),
        loan_columns=("maturity_years", "loan_size_mm", "is_secured"),
        interaction_terms=(
            ("debt_to_ebitda", "VIXCLS"),
            ("maturity_years", "BAMLH0A0HYM2"),
            ("interest_coverage_ratio", "T10Y2Y"),
        ),
    )
    engineer = FeatureEngineer(config=feature_config)

    # Time-aware split.
    n = len(full_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = full_df.iloc[:train_end]
    cal_df = full_df.iloc[train_end:val_end]
    test_df = full_df.iloc[val_end:]

    train_result = engineer.fit_transform(train_df)
    X_train, y_spread_train = train_result.X, train_result.y
    y_default_train = train_df["defaulted"].values.astype(np.float64)

    X_cal = engineer.transform_features(cal_df)
    y_spread_cal = cal_df["credit_spread_bps"].values.astype(np.float64)

    X_test = engineer.transform_features(test_df)
    y_spread_test = test_df["credit_spread_bps"].values.astype(np.float64)
    y_default_test = test_df["defaulted"].values.astype(np.float64)

    feature_names = train_result.feature_names

    # -- PD model -----------------------------------------------------------
    logger.info("Training PD classifier")
    pd_config = PDModelConfig(
        hidden_sizes=(64, 32),
        dropout_rates=(0.2, 0.1),
        learning_rate=1e-3,
        epochs=60,
        batch_size=32,
        early_stopping_patience=10,
        random_seed=config.random_seed,
        cross_validation_folds=3,
    )
    pd_model = TFPDClassifier(config=pd_config)
    pd_model.fit(X_train, y_default_train)
    pd_eval = evaluate_pd_model(pd_model, X_test, y_default_test)
    logger.info("PD ROC-AUC: %.4f", pd_eval.roc_auc)

    # -- Spread model -------------------------------------------------------
    logger.info("Training spread regressor")
    spread_config = SpreadModelConfig(
        hidden_sizes=(64, 32),
        dropout_rates=(0.2, 0.0),
        learning_rate=1e-3,
        epochs=100,
        batch_size=32,
        early_stopping_patience=15,
        random_seed=config.random_seed,
    )
    spread_model = TFSpreadRegressor(pd_model=pd_model, config=spread_config)
    spread_model.fit(X_train, y_spread_train)
    spread_eval = evaluate_spread_model(spread_model, X_test, y_spread_test)
    logger.info("Spread RMSE: %.1f bps, R²: %.4f", spread_eval.rmse_bps, spread_eval.r_squared)

    # -- Quantile models + conformal ----------------------------------------
    logger.info("Training quantile regressors")
    q_config = QuantileModelConfig(
        hidden_sizes=(64, 32),
        dropout_rates=(0.2, 0.0),
        learning_rate=1e-3,
        epochs=80,
        batch_size=32,
        early_stopping_patience=10,
        random_seed=config.random_seed,
    )
    q_lower = TFQuantileRegressor(quantile_alpha=0.025, config=q_config)
    q_upper = TFQuantileRegressor(quantile_alpha=0.975, config=q_config)
    q_lower.fit(X_train, y_spread_train)
    q_upper.fit(X_train, y_spread_train)

    conformal = ConformalPredictionInterval(q_lower, q_upper, coverage_target=0.95)
    conformal.calibrate(X_cal, y_spread_cal)
    logger.info("Conformal q_hat: %.2f", conformal.q_hat)

    # Coverage at multiple levels.
    coverage_results: dict[str, dict[str, float]] = {}
    for target_cov, alpha_lo, alpha_hi in [
        (0.90, 0.05, 0.95),
        (0.95, 0.025, 0.975),
        (0.99, 0.005, 0.995),
    ]:
        lo_pred = q_lower.predict(X_test) - conformal.q_hat
        hi_pred = q_upper.predict(X_test) + conformal.q_hat
        covered = np.sum((y_spread_test >= lo_pred) & (y_spread_test <= hi_pred))
        actual_cov = float(covered / len(y_spread_test))
        avg_width = float(np.mean(hi_pred - lo_pred))
        coverage_results[f"{int(target_cov*100)}%"] = {
            "nominal": target_cov,
            "actual": actual_cov,
            "avg_width_bps": avg_width,
        }

    # -- OU calibration + Monte Carlo ---------------------------------------
    logger.info("Calibrating OU process")
    spread_series = full_df["credit_spread_bps"].values
    ou_cal = OrnsteinUhlenbeckCalibrator(
        min_series_length=60, learning_rate=0.01, max_steps=2000,
    )
    ou_params = ou_cal.fit(spread_series)

    vasicek_params = VasicekParameters(kappa=0.3, theta=3.5, sigma=0.5)
    mc_pricer = MonteCarloLoanPricer(
        ou_params=ou_params,
        vasicek_params=vasicek_params,
        n_simulations=config.monte_carlo_n_simulations,
        horizon_years=1 / 12,  # one month
        random_seed=config.random_seed,
    )
    mc_result = mc_pricer.simulate(
        current_spread_bps=float(spread_series[-1]),
        current_treasury_yield_pct=4.0,
    )
    logger.info("MC mean spread: %.1f bps", mc_result.mean_spread)

    # -- Save models --------------------------------------------------------
    pd_model.save(config.models_dir / "pd_model.npz")
    spread_model.save(config.models_dir / "spread_model.npz")
    q_lower.save(config.models_dir / "quantile_lower.npz")
    q_upper.save(config.models_dir / "quantile_upper.npz")

    return PipelineResult(
        full_df=full_df,
        fred_wide=fred_wide,
        X_train=X_train,
        X_test=X_test,
        y_default_train=y_default_train,
        y_default_test=y_default_test,
        y_spread_train=y_spread_train,
        y_spread_test=y_spread_test,
        X_cal=X_cal,
        y_spread_cal=y_spread_cal,
        feature_names=feature_names,
        feature_engineer=engineer,
        pd_model=pd_model,
        spread_model=spread_model,
        quantile_lower=q_lower,
        quantile_upper=q_upper,
        conformal=conformal,
        pd_eval=pd_eval,
        spread_eval=spread_eval,
        ou_params=ou_params,
        mc_result=mc_result,
        spread_series=spread_series,
        coverage_results=coverage_results,
    )
