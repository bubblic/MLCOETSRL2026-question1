"""US CPI inflation rates (2018-2025).

Macro-level data kept separate from company-specific financials so
the pipeline can demonstrate model behavior with and without inflation.
"""

import tensorflow as tf


def get_us_inflation() -> tf.Tensor:
    """Return annual US CPI inflation rates (2018-2025).

    Returns:
        1-D float64 tensor of length 8.
    """
    return tf.constant(
        [0.024, 0.018, 0.012, 0.047, 0.08, 0.041, 0.029, 0.027],
        dtype=tf.float64,
    )
