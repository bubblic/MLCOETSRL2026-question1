"""SHAP-based explainability for the structured financial tower.

Uses :class:`shap.GradientExplainer` (a TensorFlow-compatible SHAP
backend) to attribute the model's output to each of the 25 financial
ratio features.  The resulting SHAP values are returned as a
feature-name-to-value mapping and can optionally be rendered as a
matplotlib waterfall chart.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf

from credit_rating.domain.features import FEATURE_NAMES, FinancialRatios
from credit_rating.models.structured_tower import StructuredTower

logger = logging.getLogger(__name__)

# Number of background samples drawn from the reference distribution
# to estimate the expected model output.
_DEFAULT_NUM_BACKGROUND_SAMPLES: int = 100

# DPI resolution for saved matplotlib figures.
_FIGURE_DPI: int = 150


def _import_shap():
    """Lazily import shap, returning the module or ``None``.

    Returns:
        The ``shap`` module if available, otherwise ``None``.
    """
    try:
        import shap  # type: ignore[import-untyped]
        return shap
    except ImportError:
        logger.warning(
            "The 'shap' package is not installed.  "
            "Install it with `pip install shap` to enable SHAP explanations."
        )
        return None


class StructuredTowerExplainer:
    """SHAP gradient explainer for :class:`StructuredTower`.

    Wraps ``shap.GradientExplainer`` so that callers can obtain
    per-feature attribution values for any single observation
    represented as a :class:`FinancialRatios` instance.

    Args:
        model: A trained :class:`StructuredTower` instance.
        background_data: A rank-2 NumPy array of shape
            ``(n_samples, 25)`` used as the reference distribution.
            If ``None``, explanations cannot be computed until
            :meth:`set_background` is called.
    """

    def __init__(
        self,
        model: StructuredTower,
        background_data: Optional[np.ndarray] = None,
    ) -> None:
        self._model = model
        self._explainer: Optional[object] = None
        self._shap_module = _import_shap()

        if background_data is not None:
            self.set_background(background_data)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_background(self, background_data: np.ndarray) -> None:
        """Set (or replace) the reference distribution.

        Args:
            background_data: Array of shape ``(n_samples, 25)``
                representing typical financial-ratio observations.

        Raises:
            RuntimeError: If the ``shap`` package is not available.
            ValueError: If *background_data* has the wrong number of
                features.
        """
        if self._shap_module is None:
            raise RuntimeError(
                "Cannot initialise the SHAP explainer because the "
                "'shap' package is not installed."
            )

        num_features = len(FEATURE_NAMES)
        if background_data.shape[1] != num_features:
            raise ValueError(
                f"background_data must have {num_features} columns, "
                f"got {background_data.shape[1]}"
            )

        background_sample = background_data
        if background_data.shape[0] > _DEFAULT_NUM_BACKGROUND_SAMPLES:
            indices = np.random.default_rng(seed=42).choice(
                background_data.shape[0],
                size=_DEFAULT_NUM_BACKGROUND_SAMPLES,
                replace=False,
            )
            background_sample = background_data[indices]

        background_tensor = tf.constant(
            background_sample, dtype=tf.float32,
        )
        self._explainer = self._shap_module.GradientExplainer(
            self._model, background_tensor,
        )
        logger.info(
            "SHAP GradientExplainer initialised with %d background samples",
            background_sample.shape[0],
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def explain(self, ratios: FinancialRatios) -> dict[str, float]:
        """Compute per-feature SHAP values for a single observation.

        Args:
            ratios: The financial ratios to explain.

        Returns:
            A mapping from each feature name in
            :data:`~credit_rating.domain.features.FEATURE_NAMES` to its
            SHAP attribution value.

        Raises:
            RuntimeError: If no background data has been set or if
                ``shap`` is not installed.
        """
        if self._explainer is None:
            raise RuntimeError(
                "Explainer is not ready.  Call set_background() first "
                "or pass background_data to the constructor."
            )

        input_tensor = tf.expand_dims(ratios.to_tensor(), axis=0)
        raw_shap_values = self._explainer.shap_values(input_tensor)

        # GradientExplainer may return a list (one array per output
        # class) or a single array.  Collapse to the first output.
        if isinstance(raw_shap_values, list):
            values_array: np.ndarray = np.asarray(raw_shap_values[0])
        else:
            values_array = np.asarray(raw_shap_values)

        # Remove the batch dimension.
        values_1d: np.ndarray = values_array.squeeze(axis=0)

        return dict(zip(FEATURE_NAMES, values_1d.tolist()))

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_waterfall(
        self,
        shap_values: dict[str, float],
        output_path: Path,
        title: Optional[str] = None,
    ) -> Path:
        """Save a waterfall chart of SHAP attributions.

        Bars are sorted by absolute magnitude so the most influential
        features appear at the top of the chart.

        Args:
            shap_values: Feature-name-to-SHAP-value mapping, as
                returned by :meth:`explain`.
            output_path: Destination file path for the saved figure
                (e.g. ``Path("outputs/shap_waterfall.png")``).
            title: Optional chart title.  Defaults to
                ``"SHAP Feature Attribution — Structured Tower"``.

        Returns:
            The resolved *output_path* where the figure was written.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names: List[str] = list(shap_values.keys())
        values: List[float] = list(shap_values.values())

        # Sort features by absolute SHAP magnitude (largest first).
        sorted_pairs = sorted(
            zip(names, values),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )
        sorted_names = [p[0] for p in sorted_pairs]
        sorted_values = [p[1] for p in sorted_pairs]

        colours = [
            "#d73027" if v > 0 else "#4575b4" for v in sorted_values
        ]

        fig, ax = plt.subplots(figsize=(8, max(6, len(sorted_names) * 0.35)))
        y_positions = range(len(sorted_names))

        ax.barh(y_positions, sorted_values, color=colours, edgecolor="none")
        ax.set_yticks(list(y_positions))
        ax.set_yticklabels(sorted_names)
        ax.invert_yaxis()
        ax.set_xlabel("SHAP Value")
        ax.set_title(
            title or "SHAP Feature Attribution \u2014 Structured Tower",
        )
        ax.axvline(x=0, color="black", linewidth=0.8)
        fig.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=_FIGURE_DPI)
        plt.close(fig)

        logger.info("SHAP waterfall chart saved to %s", output_path)
        return output_path
