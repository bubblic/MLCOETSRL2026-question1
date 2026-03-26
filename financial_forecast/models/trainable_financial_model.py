"""Trainable financial model — extends base with training and serialization.

Inherits all forecast logic from :class:`BaseFinancialModel` and adds:

- :meth:`train` — gradient-based parameter fitting.
- :meth:`save_parameters` / :meth:`load_parameters` — ``.npz`` persistence.
"""

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

    def prepare_for_training(self, years, amount_scale):
        """Configure data-derived model settings.

        Convenience method called by legacy code.  Prefer
        :meth:`prepare` for new code.

        Args:
            years: 1-D tensor of fiscal year labels.
            amount_scale: USD-to-scaled-units conversion factor.
        """
        self.base_year = int(years[0])
        self.amount_scale = amount_scale

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        trainers,
        parameters_save_path="trained_parameters.npz",
        use_trained_parameters=False,
    ):
        """Train model parameters or load from disk.

        Must call :meth:`prepare` first.

        Args:
            trainers: Ordered list of trainer instances (e.g.
                ``[PolicyTrainer, StructuralTrainer]``).
            parameters_save_path: Path for saving/loading ``.npz`` file.
            use_trained_parameters: If ``True``, load from disk instead
                of training.
        """
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

    def save_parameters(self, path):
        """Save all model parameters to an .npz file."""
        _save_parameters(self, path)

    def load_parameters(self, path):
        """Load model parameters from an .npz file."""
        _load_parameters(self, path)
