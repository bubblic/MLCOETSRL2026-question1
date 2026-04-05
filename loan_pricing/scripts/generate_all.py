"""Master runner that regenerates every figure and table in order.

Usage::

    python -m loan_pricing.scripts.generate_all
    python loan_pricing/scripts/generate_all.py
"""

from __future__ import annotations

from loan_pricing.logging_config import get_logger

logger = get_logger(__name__)

# Each script module must expose a ``main()`` function with no required
# arguments.  Imports are deferred to :func:`main` so that this module
# can be imported cheaply (e.g. for testing) without triggering heavy
# model loads.

_SCRIPT_MODULES: tuple[str, ...] = (
    "loan_pricing.scripts.01_eda",
    "loan_pricing.scripts.02_pd_model_eval",
    "loan_pricing.scripts.03_spread_model_eval",
    "loan_pricing.scripts.04_hyperparameter_tuning",
    "loan_pricing.scripts.05_private_borrower_example",
    "loan_pricing.scripts.06_monte_carlo_forecast",
    "loan_pricing.scripts.07_uncertainty",
    "loan_pricing.scripts.08_summary",
)


def main() -> None:
    """Import and run every generation script in sequence."""
    import importlib

    for module_name in _SCRIPT_MODULES:
        logger.info("Running %s …", module_name)
        module = importlib.import_module(module_name)
        module.main()  # type: ignore[attr-defined]
        logger.info("Finished %s", module_name)

    logger.info("All outputs regenerated successfully.")


if __name__ == "__main__":
    main()
