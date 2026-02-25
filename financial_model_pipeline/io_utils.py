"""I/O helpers for financial model artifacts."""

import os

TRAINING_RESULTS_DIR = "training_results"


def _get_training_results_path(filename: str) -> str:
    """Return path under training results directory, creating it if needed."""
    os.makedirs(TRAINING_RESULTS_DIR, exist_ok=True)
    return os.path.join(TRAINING_RESULTS_DIR, filename)
