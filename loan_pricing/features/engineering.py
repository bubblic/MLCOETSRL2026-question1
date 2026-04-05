"""Feature engineering for the term loan pricing model.

The :class:`FeatureEngineer` transforms a preprocessed loan-level
DataFrame into model-ready feature matrices ``X`` and target vectors
``y``, including interaction terms that capture cross-variable dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from loan_pricing.exceptions import DataLeakageError
from loan_pricing.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureConfig:
    """Immutable specification of which features to include.

    Attributes:
        financial_ratio_columns: Column names for borrower financial
            ratios.
        macro_columns: Column names for macroeconomic variables.
        loan_columns: Column names for loan-level characteristics.
        interaction_terms: Pairs of columns whose product forms an
            interaction feature.
        target_column: Name of the regression target (credit spread).
        default_column: Name of the binary default indicator (PD target).
    """

    financial_ratio_columns: tuple[str, ...] = (
        "debt_to_ebitda",
        "interest_coverage_ratio",
        "net_debt_to_equity",
        "fcf_to_debt",
        "revenue_growth_yoy",
        "ebitda_margin",
        "current_ratio",
        "altman_z_double_prime",
    )
    macro_columns: tuple[str, ...] = (
        "treasury_yield",
        "ig_oas",
        "hy_oas",
        "vix",
        "yield_curve_slope",
        "unemployment_rate",
        "gdp_growth",
    )
    loan_columns: tuple[str, ...] = (
        "maturity_years",
        "loan_size_mm",
        "is_secured",
    )
    interaction_terms: tuple[tuple[str, str], ...] = (
        ("debt_to_ebitda", "vix"),
        ("maturity_years", "hy_oas"),
        ("interest_coverage_ratio", "treasury_yield"),
    )
    target_column: str = "credit_spread_bps"
    default_column: str = "defaulted"


# ---------------------------------------------------------------------------
# Transform result
# ---------------------------------------------------------------------------


class TransformResult(NamedTuple):
    """Arrays returned by :meth:`FeatureEngineer.transform`.

    Attributes:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: Target vector of shape ``(n_samples,)``.
        feature_names: Ordered list of feature column names matching
            the columns of *X*.
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]


# ---------------------------------------------------------------------------
# Feature engineer
# ---------------------------------------------------------------------------


class FeatureEngineer:
    """Stateful feature transformer with fit / transform semantics.

    The :meth:`fit` method learns column means and standard deviations
    from the training set.  :meth:`transform` applies the learned
    normalisation to any split.  Calling :meth:`fit` a second time with
    a *different* DataFrame raises :class:`DataLeakageError`.

    Args:
        config: Feature specification.  Defaults to the standard
            :class:`FeatureConfig`.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self._config = config or FeatureConfig()
        self._means: pd.Series | None = None
        self._stds: pd.Series | None = None
        self._fitted_id: int | None = None
        self._feature_names: list[str] | None = None

    # -- public API ---------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._means is not None

    @property
    def feature_names(self) -> list[str]:
        """Feature names determined during :meth:`fit`.

        Raises:
            RuntimeError: If accessed before fitting.
        """
        if self._feature_names is None:
            msg = "FeatureEngineer has not been fitted yet"
            raise RuntimeError(msg)
        return list(self._feature_names)

    def fit(self, df: pd.DataFrame) -> FeatureEngineer:
        """Learn normalisation statistics from the training set.

        Args:
            df: Training-set DataFrame.

        Returns:
            ``self`` for method chaining.

        Raises:
            DataLeakageError: If :meth:`fit` is called again with a
                DataFrame that has a different identity (i.e. a
                different object), suggesting the caller is fitting on
                validation or test data.
        """
        df_id = id(df)
        if self._fitted_id is not None and self._fitted_id != df_id:
            raise DataLeakageError(
                "FeatureEngineer.fit() was already called on a different "
                "DataFrame.  Re-fitting on new data risks data leakage."
            )

        raw_features = self._select_and_engineer(df)
        self._means = raw_features.mean()
        self._stds = raw_features.std().replace(0, 1.0)
        self._fitted_id = df_id
        self._feature_names = list(raw_features.columns)

        logger.info(
            "FeatureEngineer fitted on %d rows, %d features",
            len(df),
            len(self._feature_names),
        )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        target: str | None = None,
    ) -> TransformResult:
        """Apply learned normalisation and return arrays.

        Args:
            df: Any split (train, validation, or test).
            target: Column name for the target vector.  Defaults to
                ``config.target_column`` (credit spread).

        Returns:
            A :class:`TransformResult` containing ``X``, ``y``, and
            ``feature_names``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._means is None or self._stds is None:
            msg = "Call fit() before transform()"
            raise RuntimeError(msg)

        target_col = target or self._config.target_column
        raw = self._select_and_engineer(df)

        # Align columns in case some are missing in this split.
        raw = raw.reindex(columns=self._feature_names, fill_value=0.0)

        normalised = (raw - self._means) / self._stds
        X = normalised.values.astype(np.float64)
        y = df[target_col].values.astype(np.float64)

        return TransformResult(X=X, y=y, feature_names=list(self._feature_names))

    def fit_transform(
        self,
        df: pd.DataFrame,
        target: str | None = None,
    ) -> TransformResult:
        """Convenience: :meth:`fit` then :meth:`transform` in one call.

        Args:
            df: Training-set DataFrame.
            target: Passed through to :meth:`transform`.

        Returns:
            A :class:`TransformResult`.
        """
        self.fit(df)
        return self.transform(df, target=target)

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Normalise features without extracting a target column.

        This is the inference-time entry point used by
        :class:`~loan_pricing.models.spread_model.LoanPricer` when
        pricing a single loan that has no target value.

        Args:
            df: DataFrame with feature columns (target column is not
                required).

        Returns:
            Normalised feature matrix ``(n_rows, n_features)``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._means is None or self._stds is None:
            msg = "Call fit() before transform_features()"
            raise RuntimeError(msg)

        raw = self._select_and_engineer(df)
        raw = raw.reindex(columns=self._feature_names, fill_value=0.0)
        normalised = (raw - self._means) / self._stds
        return normalised.values.astype(np.float64)

    # -- internals ----------------------------------------------------------

    def _select_and_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select configured columns and build interaction terms.

        Args:
            df: Source DataFrame.

        Returns:
            A new DataFrame containing only the feature columns (raw
            + interactions), with no target column.
        """
        cfg = self._config
        base_cols = list(cfg.financial_ratio_columns + cfg.macro_columns + cfg.loan_columns)

        # Only include columns that are actually present.
        present = [c for c in base_cols if c in df.columns]
        features = df[present].copy()

        # Cast boolean-like columns to float.
        for col in features.columns:
            if features[col].dtype == bool or features[col].dtype == object:
                features[col] = features[col].astype(np.float64)

        # Interaction terms.
        for col_a, col_b in cfg.interaction_terms:
            if col_a in features.columns and col_b in features.columns:
                interaction_name = f"{col_a}_x_{col_b}"
                features[interaction_name] = features[col_a] * features[col_b]

        return features.astype(np.float64)
