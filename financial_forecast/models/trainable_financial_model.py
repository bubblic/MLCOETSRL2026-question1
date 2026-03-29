"""Trainable financial model — extends base with training and serialization.

Inherits all forecast logic from :class:`BaseFinancialModel` and adds:

- :meth:`train` — gradient-based parameter fitting.
- :meth:`save_parameters` / :meth:`load_parameters` — ``.npz`` persistence.
"""

from typing import Mapping, Optional

import tensorflow as tf

from financial_forecast.models.base import BaseFinancialModel
from financial_forecast.serialization.parameter_io import (
    save_parameters as _save_parameters,
    load_parameters as _load_parameters,
)


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

        s = self._s
        d = self._d
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

    def train(
        self,
        trainers: list,
        parameters_save_path: str = "trained_parameters.npz",
        use_trained_parameters: bool = False,
    ) -> None:
        """Train model parameters or load from disk.

        Must call :meth:`prepare` first.

        Args:
            trainers: Ordered list of trainer instances (e.g.
                ``[PolicyTrainer, StructuralTrainer]``).
            parameters_save_path: Path for saving/loading ``.npz`` file.
            use_trained_parameters: If ``True``, load from disk instead
                of training.
        """
        self.parameters_path = parameters_save_path

        if use_trained_parameters:
            self.load_parameters(parameters_save_path)
            return

        s = self._s
        d = self._d
        inflation = d["inflation"]
        t = len(s["sales"]) - self._test_years
        train_years = tf.cast(
            tf.range(self.base_year, self.base_year + t),
            dtype=tf.float64,
        )

        # Policy + OpEx trainer
        trainers[0].train(
            self,
            historical_sales=s["sales"][:t],
            historical_purchases=s["purchases"][:t],
            historical_cogs=s["cogs"][:t],
            historical_nca=s["nca"][:t],
            historical_depreciation=s["depreciation"][:t],
            historical_adv_pay_sales=s["advance_payments_sales"][:t],
            historical_adv_pay_purch=s["advance_payments_purchases"][:t],
            historical_ar=s["accounts_receivable"][:t],
            historical_ap=s["accounts_payable"][:t],
            historical_inventory=s["inventory"][:t],
            historical_cash=s["cash"][:t],
            historical_ims=s["ims"][:t],
            historical_net_income=s["net_income"][:t],
            historical_dividends=s["dividends"][:t],
            historical_stock_buyback=s["stock_buyback"][:t],
            historical_opex=s["opex"][:t],
            historical_tax=s["tax"][:t],
            historical_eff_st_debt=s["effective_st_debt"][:t],
            historical_inflation=inflation[:t],
            historical_years=train_years,
            show_plot=False,
            loss_scale_mode="std",
        )

        # Structural trainer
        trainers[1].train(
            self,
            historical_sales=s["sales"][:t],
            historical_nca=s["nca"][:t],
            historical_adv_pay_sales=s["advance_payments_sales"][:t],
            historical_adv_pay_purch=s["advance_payments_purchases"][:t],
            historical_ar=s["accounts_receivable"][:t],
            historical_ap=s["accounts_payable"][:t],
            historical_inventory=s["inventory"][:t],
            historical_cash=s["cash"][:t],
            historical_ims=s["ims"][:t],
            historical_net_income=s["net_income"][:t],
            historical_dividends=s["dividends"][:t],
            historical_stock_buyback=s["stock_buyback"][:t],
            historical_opex=s["opex"][:t],
            historical_tax=s["tax"][:t],
            historical_effective_st_debt=s["effective_st_debt"][:t],
            historical_current_lt_debt=s["current_lt_debt"][:t],
            historical_non_current_liabilities=s["non_current_liabilities"][:t],
            historical_interest_payment=s["interest_payment"][:t],
            historical_ms_return=s["ms_return"][:t],
            historical_equity=s["equity"][:t],
            historical_inflation=inflation[:t],
            historical_years=train_years,
            loss_scale_mode="std",
        )
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
