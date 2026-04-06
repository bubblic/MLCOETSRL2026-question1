"""Generate uncertainty quantification figures and tables."""

from __future__ import annotations

from loan_pricing.logging_config import get_logger
from loan_pricing.reporting.tables import write_coverage_table
from loan_pricing.scripts._pipeline import TABLES_DIR, run_pipeline

logger = get_logger(__name__)


def main() -> None:
    """Generate all uncertainty quantification outputs."""
    p = run_pipeline()

    # Flatten coverage results for the table writer.
    flat: dict[str, float] = {}
    for label, metrics in p.coverage_results.items():
        flat[f"{label} nominal"] = metrics["nominal"]
        flat[f"{label} actual"] = metrics["actual"]
        flat[f"{label} avg_width_bps"] = metrics["avg_width_bps"]

    write_coverage_table(flat, TABLES_DIR / "tbl_07_coverage.csv")

    logger.info("07_uncertainty complete")


if __name__ == "__main__":
    main()
