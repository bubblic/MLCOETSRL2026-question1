"""Apple Inc. one-time tax anomaly data (FY2018-FY2025).

Extracted from ``extracted_text/apple_YYYY.tax-anomalies-contingencies.llm.json``.
This data is kept separate from core historical financials so the pipeline
can demonstrate model improvement when one-time tax effects are incorporated.
"""

import tensorflow as tf


def get_tax_onetime_payments():
    """Return Apple's one-time tax anomaly amounts (FY2018-FY2025).

    Source values are in USD; null/absent years are mapped to 0.0.

    Returns:
        1-D float64 tensor of length 8.
    """
    return tf.constant(
        [
            1.5e9,      # 2018
            0.0,        # 2019 (null)
            -0.582e9,   # 2020
            0.0,        # 2021 (null)
            0.0,        # 2022 (null)
            0.0,        # 2023 (null)
            10.2e9,     # 2024
            0.0,        # 2025 (null)
        ],
        dtype=tf.float64,
    )
