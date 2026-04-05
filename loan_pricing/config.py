"""Centralised project configuration for the term loan pricing model.

Every configurable constant lives here as a field on the frozen
``ProjectConfig`` dataclass.  No other module may hardcode directory
paths, random seeds, or numeric constants that belong to the project
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProjectConfig:
    """Immutable project-wide configuration.

    Attributes:
        package_root: Absolute path to the ``loan_pricing/`` package directory.
        artifacts_dir: Root of all data artefacts (raw downloads, processed
            outputs, serialised models).
        raw_fred_dir: Cached FRED CSV files.
        raw_sec_dir: Cached SEC EDGAR JSON files.
        figures_dir: Generated ``.png`` figure outputs.
        tables_dir: Generated ``.csv`` table outputs.
        models_dir: Serialised model artefacts (joblib).
        fred_api_key_env_var: Name of the environment variable holding the
            FRED API key.
        random_seed: Global seed for reproducibility across all random
            operations (train/test splits, model init, Monte Carlo).
        confidence_levels: Tuple of confidence levels used for prediction
            intervals and Monte Carlo summaries.
        monte_carlo_n_simulations: Number of Monte Carlo paths to draw.
        min_ou_series_length: Minimum number of observations required by the
            Ornstein-Uhlenbeck MLE calibrator.
        cross_validation_folds: Number of folds for stratified k-fold CV.
        test_fraction: Fraction of data reserved for the held-out test set.
        validation_fraction: Fraction of data reserved for validation.
        figure_dpi: Resolution for all saved figures.
        default_missing_strategy: Default imputation strategy used by the
            preprocessor when none is specified by the caller.
    """

    # --- directory paths ---------------------------------------------------
    package_root: Path = field(default=_PACKAGE_ROOT)
    artifacts_dir: Path = field(default=_PACKAGE_ROOT / "artifacts")
    raw_fred_dir: Path = field(default=_PACKAGE_ROOT / "artifacts" / "raw" / "fred")
    raw_sec_dir: Path = field(default=_PACKAGE_ROOT / "artifacts" / "raw" / "sec")
    figures_dir: Path = field(
        default=_PACKAGE_ROOT / "artifacts" / "processed" / "figures",
    )
    tables_dir: Path = field(
        default=_PACKAGE_ROOT / "artifacts" / "processed" / "tables",
    )
    models_dir: Path = field(
        default=_PACKAGE_ROOT / "artifacts" / "processed" / "models",
    )

    # --- API keys ----------------------------------------------------------
    fred_api_key_env_var: str = "FRED_API_KEY"

    # --- reproducibility ---------------------------------------------------
    random_seed: int = 42

    # --- modelling constants -----------------------------------------------
    confidence_levels: tuple[float, ...] = (0.80, 0.95)
    monte_carlo_n_simulations: int = 10_000
    min_ou_series_length: int = 60
    cross_validation_folds: int = 5
    test_fraction: float = 0.15
    validation_fraction: float = 0.15

    # --- reporting ---------------------------------------------------------
    figure_dpi: int = 300
    default_missing_strategy: str = "median"
