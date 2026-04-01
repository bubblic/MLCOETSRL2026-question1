"""Trainable financial model — extends base with training and serialization.

Inherits all forecast logic from :class:`BaseFinancialModel` and adds:

- :meth:`train` — gradient-based parameter fitting.
- :meth:`save_parameters` / :meth:`load_parameters` — ``.npz`` persistence.
"""

from __future__ import annotations

from typing import Mapping, Optional, TYPE_CHECKING

import tensorflow as tf

from financial_forecast.models.base import BaseFinancialModel
from financial_forecast.types import HistoricalTrainingData
from financial_forecast.serialization.parameter_io import (
    save_parameters as _save_parameters,
    load_parameters as _load_parameters,
)

if TYPE_CHECKING:
    from financial_forecast.training.base_trainer import BaseTrainer


class TrainableFinancialModel(BaseFinancialModel):
    """Financial model with training and parameter serialization support.

    All forecast logic (policy modules, ``forecast_step``,
    ``forecast_step_compiled``) is inherited from
    :class:`BaseFinancialModel`.  This subclass adds gradient-based
    parameter fitting and ``.npz`` save/load.
    """

    def prepare(
        self,
        financial_statements: Mapping[str, tf.Tensor],
        inflation: Optional[tf.Tensor] = None,
        test_years: int = 1,
    ) -> None:
        """Prepare model and configure sub-modules for training.

        Extends :meth:`BaseFinancialModel.prepare` by also calling
        ``prepare_for_training`` on the OpEx and tax modules.
        """
        super().prepare(financial_statements, inflation, test_years)

        s = self._scaled_data
        d = self._historical_data
        t = len(s["sales"]) - self._test_years
        train_years = tf.cast(
            tf.range(self.base_year, self.base_year + t),
            dtype=tf.float64,
        )
        self.opex_module.prepare_for_training(
            self.amount_scale,
            s["sales"][:t],
            s["opex"][:t],
            d["inflation"][:t],
        )
        self.tax_module.prepare_for_training(self.amount_scale, train_years)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _build_training_data(self) -> HistoricalTrainingData:
        """Build a :class:`HistoricalTrainingData` from prepared model state."""
        s = self._scaled_data
        d = self._historical_data
        t = len(s["sales"]) - self._test_years
        train_years = tf.cast(
            tf.range(self.base_year, self.base_year + t),
            dtype=tf.float64,
        )
        return HistoricalTrainingData(
            sales=s["sales"][:t],
            purchases=s["purchases"][:t],
            cogs=s["cogs"][:t],
            nca=s["nca"][:t],
            depreciation=s["depreciation"][:t],
            advance_payments_sales=s["advance_payments_sales"][:t],
            advance_payments_purchases=s["advance_payments_purchases"][:t],
            accounts_receivable=s["accounts_receivable"][:t],
            accounts_payable=s["accounts_payable"][:t],
            inventory=s["inventory"][:t],
            cash=s["cash"][:t],
            ims=s["ims"][:t],
            net_income=s["net_income"][:t],
            dividends=s["dividends"][:t],
            stock_buyback=s["stock_buyback"][:t],
            opex=s["opex"][:t],
            tax=s["tax"][:t],
            effective_st_debt=s["effective_st_debt"][:t],
            current_lt_debt=s["current_lt_debt"][:t],
            non_current_liabilities=s["non_current_liabilities"][:t],
            interest_payment=s["interest_payment"][:t],
            ms_return=s["ms_return"][:t],
            equity=s["equity"][:t],
            inflation=d["inflation"][:t],
            years=train_years,
        )

    def train(
        self,
        policy_trainer: BaseTrainer,
        structural_trainer: BaseTrainer,
        parameters_save_path: str = "trained_parameters.npz",
        use_trained_parameters: bool = False,
    ) -> None:
        """Train model parameters or load from disk.

        Must call :meth:`prepare` first.

        Args:
            policy_trainer: Trainer for policy-level and OpEx parameters.
            structural_trainer: Trainer for interest rates, debt maturity,
                and equity financing mix.
            parameters_save_path: Path for saving/loading ``.npz`` file.
            use_trained_parameters: If ``True``, load from disk instead
                of training.
        """
        self.parameters_path = parameters_save_path

        if use_trained_parameters:
            self.load_parameters(parameters_save_path)
            return

        data = self._build_training_data()

        policy_trainer.train(self, data, loss_scale_mode="std")
        structural_trainer.train(self, data, loss_scale_mode="std")

        self.save_parameters(parameters_save_path)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_parameters(self, path: str) -> None:
        """Save all model parameters to an .npz file."""
        _save_parameters(self, path)

    def load_parameters(self, path: str) -> None:
        """Load model parameters from an .npz file."""
        _load_parameters(self, path)
