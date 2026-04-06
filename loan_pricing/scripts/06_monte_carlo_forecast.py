"""Generate Monte Carlo forecast figures and tables."""

from __future__ import annotations

import numpy as np

from loan_pricing.logging_config import get_logger
from loan_pricing.models.ou_calibration import MonteCarloLoanPricer, VasicekParameters
from loan_pricing.reporting.figures import plot_mc_fan_chart, plot_mc_price_histogram
from loan_pricing.reporting.tables import write_ou_calibration_table
from loan_pricing.scripts._pipeline import FIGURES_DIR, TABLES_DIR, run_pipeline

logger = get_logger(__name__)


def main() -> None:
    """Generate all Monte Carlo forecast outputs."""
    p = run_pipeline()

    # OU calibration table.
    write_ou_calibration_table(p.ou_params, TABLES_DIR / "tbl_06_ou_params.csv")

    # MC histogram.
    plot_mc_price_histogram(
        p.mc_result.simulated_spreads,
        p.mc_result.mean_spread,
        p.mc_result.ci_lower_95,
        p.mc_result.ci_upper_95,
        FIGURES_DIR / "fig_06_mc_histogram.png",
    )

    # Fan chart: simulate multiple time steps and collect percentile paths.
    n_steps = 22  # ~1 month of trading days
    steps_per_year = 252
    vasicek = VasicekParameters(kappa=0.3, theta=3.5, sigma=0.5)
    current_spread = float(p.spread_series[-1])

    percentile_spreads: dict[str, list[float]] = {
        "p2.5": [current_spread],
        "p10": [current_spread],
        "p50": [current_spread],
        "p90": [current_spread],
        "p97.5": [current_spread],
    }

    for day in range(1, n_steps + 1):
        horizon = day / steps_per_year
        mc = MonteCarloLoanPricer(
            ou_params=p.ou_params,
            vasicek_params=vasicek,
            n_simulations=5000,
            horizon_years=horizon,
            random_seed=42 + day,
            steps_per_year=steps_per_year,
        )
        r = mc.simulate(current_spread, 4.0)
        percentile_spreads["p2.5"].append(float(np.percentile(r.simulated_spreads, 2.5)))
        percentile_spreads["p10"].append(float(np.percentile(r.simulated_spreads, 10)))
        percentile_spreads["p50"].append(float(np.percentile(r.simulated_spreads, 50)))
        percentile_spreads["p90"].append(float(np.percentile(r.simulated_spreads, 90)))
        percentile_spreads["p97.5"].append(float(np.percentile(r.simulated_spreads, 97.5)))

    time_grid = np.array([i / steps_per_year for i in range(n_steps + 1)])
    paths_arrays = {k: np.array(v) for k, v in percentile_spreads.items()}

    plot_mc_fan_chart(paths_arrays, time_grid, FIGURES_DIR / "fig_06_fan_chart.png")

    logger.info("06_monte_carlo_forecast complete")


if __name__ == "__main__":
    main()
