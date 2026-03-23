"""Run the financial model with all simple policies.

Uses static liquidity, simple dividends, simple buybacks,
static cost ratio, deficit-driven debt, and deterministic OpEx.
This is the simplest configuration — no time trends, no Bayesian
inference, no Monte Carlo sampling.

Usage:
    python run_trainable_model_forecast_simple.py
"""

from financial_forecast.data.loader import HistoricalDataLoader
from financial_forecast.models.trainable_financial_model import TrainableFinancialModel
from financial_forecast.models.opex import SimpleOpEx
from financial_forecast.inference.trajectory_simulator import DeterministicSimulator
from financial_forecast.models.liquidity import SimpleLiquidityPolicy
from financial_forecast.models.dividends import SimpleDividendPolicy
from financial_forecast.models.buyback import SimpleBuybackPolicy
from financial_forecast.models.purchases import StaticCostRatioPolicy
from financial_forecast.models.debt import SimpleDebtPolicy
from financial_forecast.models.capex import CapexPolicy
from financial_forecast.models.working_capital import WorkingCapitalPolicy
from financial_forecast.training.policy_trainer import PolicyTrainer
from financial_forecast.training.structural_trainer import StructuralTrainer
from financial_forecast.training.pipeline import ForecastPipeline

if __name__ == "__main__":
    historical_data = HistoricalDataLoader(
        "aapl",
        include_inflation=True,
    )

    model = TrainableFinancialModel(
        opex_module=SimpleOpEx(),
        trajectory_simulator=DeterministicSimulator(),
        capex_policy=CapexPolicy(),
        working_capital=WorkingCapitalPolicy(),
        liquidity_policy=SimpleLiquidityPolicy(),
        dividend_policy=SimpleDividendPolicy(),
        buyback_policy=SimpleBuybackPolicy(),
        purchases_policy=StaticCostRatioPolicy(),
        debt_policy=SimpleDebtPolicy(),
    )

    ForecastPipeline(
        model=model,
        trainers=[PolicyTrainer(epochs=25000), StructuralTrainer(epochs=20000)],
        financial_statements=historical_data.financial_statements,
        inflation=historical_data.inflation,
        test_years=1,  # hold out last N years for testing
        forecast_years=10,  # includes test_years
        sales_forecast_usd=None,  # default: use linear extrapolation of the historical average annual delta
        inflation_forecast=None,  # default: 3% inflation
        parameters_save_path="trained_parameters_simple.npz",
        use_trained_parameters=False,
    ).run()
