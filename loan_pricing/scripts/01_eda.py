"""Generate exploratory data analysis figures and tables."""

from __future__ import annotations

from loan_pricing.logging_config import get_logger
from loan_pricing.reporting.figures import plot_spread_distribution
from loan_pricing.reporting.tables import write_dataset_summary
from loan_pricing.scripts._pipeline import FIGURES_DIR, TABLES_DIR, run_pipeline

logger = get_logger(__name__)


def main() -> None:
    """Generate all EDA outputs."""
    p = run_pipeline()

    write_dataset_summary(p.full_df, TABLES_DIR / "tbl_01_dataset_summary.csv")
    plot_spread_distribution(
        p.full_df["credit_spread_bps"].values,
        FIGURES_DIR / "fig_01_spread_dist.png",
    )

    logger.info("01_eda complete")


if __name__ == "__main__":
    main()
