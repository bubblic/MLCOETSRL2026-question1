"""Tests for TrajectorySimulator interface consistency.

Verifies that all simulator implementations expose the attributes
that downstream consumers (ForecastPipeline) depend on.
"""

import pytest

from financial_forecast.inference.trajectory_simulator import (
    DeterministicSimulator,
    MonteCarloSimulator,
)


class TestSimulatorNSamplesAttribute:
    """Both simulators must expose n_samples for ForecastPipeline._export_report_json."""

    def test_deterministic_simulator_has_n_samples(self):
        sim = DeterministicSimulator()
        assert sim.n_samples == 1

    def test_monte_carlo_simulator_has_n_samples(self):
        sim = MonteCarloSimulator(n_samples=500)
        assert sim.n_samples == 500

    def test_deterministic_n_samples_is_int(self):
        sim = DeterministicSimulator()
        assert isinstance(sim.n_samples, int)

    def test_monte_carlo_default_n_samples(self):
        sim = MonteCarloSimulator()
        assert sim.n_samples == 1000
