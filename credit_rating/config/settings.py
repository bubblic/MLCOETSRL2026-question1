"""Centralised configuration for the credit rating system.

All coefficients, thresholds, hyperparameters, and paths live here.
Nothing is hardcoded elsewhere in the package.  Override any value via
environment variables or a ``.env`` file.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Tuple

from pydantic import Field
from pydantic_settings import BaseSettings


# ------------------------------------------------------------------
# Rating taxonomy
# ------------------------------------------------------------------


class RatingClass(enum.IntEnum):
    """Seven-bucket ordinal credit rating classification.

    Ordinal values preserve the natural ordering from highest credit
    quality (0) to default (6).
    """

    AAA_AA = 0
    A = 1
    BBB = 2
    BB = 3
    B = 4
    CCC_CC = 5
    D = 6

    @classmethod
    def from_sp_string(cls, label: str) -> "RatingClass":
        """Map an S&P-style rating string to a :class:`RatingClass`.

        Args:
            label: An S&P rating such as ``"AA+"``, ``"BBB-"``, or ``"D"``.

        Returns:
            The corresponding :class:`RatingClass` bucket.

        Raises:
            ValueError: If *label* cannot be mapped.
        """
        mapping = _build_sp_mapping()
        normalised = label.strip().upper()
        if normalised not in mapping:
            raise ValueError(
                f"Unknown S&P rating string: {label!r}. "
                f"Expected one of {sorted(mapping)}"
            )
        return mapping[normalised]

    @property
    def is_investment_grade(self) -> bool:
        """Return ``True`` for AAA/AA, A, and BBB buckets."""
        return self.value <= self.BBB


def _build_sp_mapping() -> dict[str, RatingClass]:
    """Build a lookup from every S&P notch string to a RatingClass."""
    rc = RatingClass
    pairs: list[tuple[list[str], RatingClass]] = [
        (["AAA", "AA+", "AA", "AA-"], rc.AAA_AA),
        (["A+", "A", "A-"], rc.A),
        (["BBB+", "BBB", "BBB-"], rc.BBB),
        (["BB+", "BB", "BB-"], rc.BB),
        (["B+", "B", "B-"], rc.B),
        (["CCC+", "CCC", "CCC-", "CC", "C"], rc.CCC_CC),
        (["D", "SD"], rc.D),
    ]
    result: dict[str, RatingClass] = {}
    for labels, bucket in pairs:
        for lbl in labels:
            result[lbl] = bucket
    return result


# ------------------------------------------------------------------
# Risk level for shenanigans reports
# ------------------------------------------------------------------


class RiskLevel(enum.IntEnum):
    """Overall risk level emitted by the shenanigans report builder."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


# ------------------------------------------------------------------
# Altman Z-Score zone labels
# ------------------------------------------------------------------


class AltmanZone(str, enum.Enum):
    """Classification zones for Altman Z-Score."""

    SAFE = "Safe"
    GREY = "Grey Zone"
    DISTRESS = "Distress"


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------


class CreditRatingSettings(BaseSettings):
    """Single source of truth for every tuneable value in the system.

    All fields are overridable via environment variables (prefix
    ``CR_``) or a ``.env`` file.
    """

    model_config = {"env_prefix": "CR_", "env_file": ".env", "extra": "ignore"}

    # -- Data paths ------------------------------------------------

    raw_data_dir: Path = Field(
        default=Path("data/raw"),
        description="Directory for raw input data (CSVs, PDFs, filings).",
    )
    processed_data_dir: Path = Field(
        default=Path("data/processed"),
        description="Directory for processed / intermediate artefacts.",
    )
    checkpoint_dir: Path = Field(
        default=Path("checkpoints"),
        description="Directory for saved model checkpoints.",
    )
    output_dir: Path = Field(
        default=Path("outputs"),
        description="Directory for final reports and predictions.",
    )

    # -- Model hyperparameters -------------------------------------

    structured_hidden_dims: Tuple[int, ...] = (256, 256, 128)
    text_projection_dim: int = 128
    fusion_hidden_dim: int = 256
    num_rating_classes: int = 7
    dropout_rate: float = 0.3
    learning_rate: float = 1e-3
    bert_learning_rate: float = 2e-5
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10
    lambda_ordinal: float = 0.1

    # -- FinBERT ---------------------------------------------------

    finbert_model_name: str = "ProsusAI/finbert"
    text_chunk_size: int = 512
    text_chunk_stride: int = 128

    # -- Altman Z-Score coefficients (original manufacturing) ------

    altman_x1_coefficient: float = 1.2
    altman_x2_coefficient: float = 1.4
    altman_x3_coefficient: float = 3.3
    altman_x4_coefficient: float = 0.6
    altman_x5_coefficient: float = 1.0

    altman_safe_threshold: float = 2.99
    altman_distress_threshold: float = 1.81

    # -- Altman Z''-Score coefficients (non-manufacturing) ---------

    altman_zpp_x1_coefficient: float = 6.56
    altman_zpp_x2_coefficient: float = 3.26
    altman_zpp_x3_coefficient: float = 6.72
    altman_zpp_x4_coefficient: float = 1.05
    altman_zpp_constant: float = 3.25

    altman_zpp_safe_threshold: float = 2.60
    altman_zpp_distress_threshold: float = 1.10

    # -- Beneish M-Score coefficients ------------------------------

    beneish_constant: float = -4.840
    beneish_dsri_coeff: float = 0.920
    beneish_gmi_coeff: float = 0.528
    beneish_aqi_coeff: float = 0.404
    beneish_sgi_coeff: float = 0.892
    beneish_depi_coeff: float = 0.115
    beneish_sgai_coeff: float = -0.172
    beneish_lvgi_coeff: float = 4.679
    beneish_tata_coeff: float = -0.327

    beneish_manipulation_threshold: float = -2.22

    # -- Shenanigan detection thresholds ---------------------------

    fog_index_cutoff: float = 18.0
    cosine_similarity_cutoff: float = 0.90
    tone_shift_delta_cutoff: float = 0.15

    # -- SEC EDGAR -------------------------------------------------

    edgar_base_url: str = "https://www.sec.gov/cgi-bin/browse-edgar"
    edgar_full_text_search_url: str = "https://efts.sec.gov/LATEST/search-index"
    edgar_request_delay_seconds: float = 0.11
    edgar_user_agent: str = "CreditRatingResearch research@example.com"

    # -- Anonymization ---------------------------------------------

    anonymize_text_for_llm: bool = True
    anonymize_use_ner: bool = True
    anonymize_years: bool = True

    # -- Reproducibility -------------------------------------------

    random_seed: int = 42
