"""I/O helpers for financial model artifacts.

This module provides path-resolution utilities for storing and retrieving
training result files (plots, parameter snapshots, etc.) in a dedicated
output directory.
"""

import os

TRAINING_RESULTS_DIR = "training_results"


def get_training_results_path(filename: str) -> str:
    """Return the full path for a file under the training results directory.

    Creates the directory if it does not already exist.

    Args:
        filename: Base name of the file to store (e.g. ``"params.npz"``).

    Returns:
        Absolute or relative path joining ``TRAINING_RESULTS_DIR`` with
        *filename*.
    """
    os.makedirs(TRAINING_RESULTS_DIR, exist_ok=True)
    return os.path.join(TRAINING_RESULTS_DIR, filename)
