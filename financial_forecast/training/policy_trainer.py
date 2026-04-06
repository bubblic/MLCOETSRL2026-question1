"""Policy parameter trainer with joint Bayesian OpEx variational inference.

The trainer receives a model instance and updates its policy-level and
Bayesian OpEx parameters in-place via gradient descent.  Each policy
parameter (e.g. %AR, %AP, %Inv) is fit against its own independent
historical ratio target — there is no coupling between parameters across
loss terms.  The Bayesian OpEx block is the exception: ``var_opex``,
``base_opex``, and ``noise_sigma`` share a joint ELBO loss
(negative log-likelihood + KL divergence).

Dependency flow::

    policy_trainer  ->  base_trainer  (ABC)
                    ->  diagnostics   (plotting)
                    ->  models/base   (model interface, read-only except assigns)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import tensorflow as tf
import tensorflow_probability as tfp

from financial_forecast.training.base_trainer import BaseTrainer
from financial_forecast.training.diagnostics import (
    plot_simple_policy_diagnostics,
)

if TYPE_CHECKING:
    from financial_forecast.models.base import BaseFinancialModel
    from financial_forecast.types import HistoricalTrainingData

tfd = tfp.distributions


def _as_float64_tensor(value: tf.Tensor) -> tf.Tensor:
    """Cast *value* to a ``tf.float64`` tensor."""
    return tf.convert_to_tensor(value, dtype=tf.float64)


@dataclass(frozen=True)
class PolicyLossScales:
    """Normalization factors for each policy loss term."""

    growth: tf.Tensor
    depr: tf.Tensor
    adv_ps: tf.Tensor
    adv_pp: tf.Tensor
    ar: tf.Tensor
    ap: tf.Tensor
    inv: tf.Tensor
    tl: tf.Tensor
    cash: tf.Tensor
    tax: tf.Tensor
    div: tf.Tensor
    bb: tf.Tensor
    cost_ratio: tf.Tensor
    eff_st: tf.Tensor
    opex: tf.Tensor


class PolicyTrainer(BaseTrainer):
    """Trains deterministic policy parameters and Bayesian OpEx jointly.

    Optimizes all policy-level parameters (asset growth, depreciation rate,
    working capital ratios, cost ratio trend, dividend smoothing, etc.)
    alongside the variational OpEx parameters in a single Adam loop.

    Args:
        epochs: Default number of training iterations.  Can be overridden
            per-call via the *epochs* argument of :meth:`train`.
    """

    def __init__(self, epochs: int = 25000):
        self.epochs = epochs

    def train(
        self,
        model: BaseFinancialModel,
        data: HistoricalTrainingData,
        loss_scale_mode: str = "std",
        show_plot: bool = False,
        learning_rate: float = 0.001,
        epochs: Optional[int] = None,
        plot_every: int = 1000,
    ) -> None:
        """Train deterministic policy parameters and Bayesian OpEx jointly."""
        if epochs is None:
            epochs = self.epochs

        # Extract tensors from data contract
        sales_tensor = _as_float64_tensor(data.sales)
        purchases_tensor = _as_float64_tensor(data.purchases)
        cogs_tensor = _as_float64_tensor(data.cogs)
        nca_tensor = _as_float64_tensor(data.nca)
        depr_tensor = _as_float64_tensor(data.depreciation)
        adv_ps_tensor = _as_float64_tensor(data.advance_payments_sales)
        adv_pp_tensor = _as_float64_tensor(data.advance_payments_purchases)
        ar_tensor = _as_float64_tensor(data.accounts_receivable)
        ap_tensor = _as_float64_tensor(data.accounts_payable)
        inv_tensor = _as_float64_tensor(data.inventory)
        cash_tensor = _as_float64_tensor(data.cash)
        ims_tensor = _as_float64_tensor(data.ims)
        ni_tensor = _as_float64_tensor(data.net_income)
        div_tensor = _as_float64_tensor(data.dividends)
        bb_tensor = _as_float64_tensor(data.stock_buyback)
        opex_tensor = _as_float64_tensor(data.opex)
        tax_tensor = _as_float64_tensor(data.tax)
        eff_st_debt_tensor = _as_float64_tensor(data.effective_st_debt)

        inf_tensor = _as_float64_tensor(data.inflation)
        cum_inf_tensor = tf.math.cumprod(1 + inf_tensor)

        cost_ratio_hist = cogs_tensor / sales_tensor
        logit_cr_hist = tf.math.log(cost_ratio_hist / (1.0 - cost_ratio_hist))

        time_indices = tf.cast(data.years, dtype=tf.float64) - tf.constant(
            float(model.base_year), dtype=tf.float64
        )

        # Growth/depreciation alignment: delta_NCA_t = NCA_t - NCA_{t-1}
        delta_nca_true = nca_tensor[1:] - nca_tensor[:-1]
        sales_aligned_growth = sales_tensor[1:]
        depr_true = depr_tensor[1:]
        nca_prev_aligned = nca_tensor[:-1]

        # Lintner dividend smoothing alignment
        div_true = div_tensor[1:]
        ni_prev_aligned = ni_tensor[:-1]
        div_prev_aligned = div_tensor[:-1]

        scales = self._compute_loss_scales(
            loss_scale_mode,
            delta_nca_true=delta_nca_true,
            depr_true=depr_true,
            adv_ps_true=adv_ps_tensor,
            adv_pp_true=adv_pp_tensor,
            ar=ar_tensor,
            ap=ap_tensor,
            inv=inv_tensor,
            cash=cash_tensor,
            ims=ims_tensor,
            tax=tax_tensor,
            div_true=div_true,
            bb=bb_tensor,
            logit_cr_hist=logit_cr_hist,
            eff_st_debt=eff_st_debt_tensor,
            opex=opex_tensor,
        )

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        n_years = len(data.sales)
        print(f"Training on {n_years} years of historical data...")

        vars_to_train = model.policy_trainable_variables

        simple_history = {
            "epochs": [],
            "loss_total": [],
            "loss_growth": [],
            "loss_depr": [],
            "loss_adv_ps": [],
            "loss_adv_pp": [],
            "loss_ar": [],
            "loss_ap": [],
            "loss_inv": [],
            "loss_tl": [],
            "loss_cash": [],
            "loss_tax": [],
            "loss_div": [],
            "loss_bb": [],
            "loss_cost_ratio": [],
            "loss_eff_st_debt": [],
            "loss_opex": [],
            "loss_prior_am": [],
        }

        compiled_step = self._build_compiled_step(
            model,
            optimizer,
            vars_to_train,
            sales_tensor,
            purchases_tensor,
            cogs_tensor,
            nca_prev_aligned,
            delta_nca_true,
            depr_true,
            depr_tensor,
            sales_aligned_growth,
            adv_ps_tensor,
            adv_pp_tensor,
            ar_tensor,
            ap_tensor,
            inv_tensor,
            cash_tensor,
            ims_tensor,
            ni_tensor,
            ni_prev_aligned,
            div_true,
            div_prev_aligned,
            bb_tensor,
            opex_tensor,
            tax_tensor,
            eff_st_debt_tensor,
            cum_inf_tensor,
            time_indices,
            logit_cr_hist,
            scales,
        )

        self._run_training_loop(
            compiled_step,
            epochs,
            plot_every,
            model,
            simple_history,
        )

        print("-" * 50)
        print("Training Complete.")
        model.print_policy_summary(n_years)
        print("\nFinal policy losses:")
        for key in simple_history:
            if key.startswith("loss_") and simple_history[key]:
                label = key.replace("loss_", "")
                print(f"  {label:15s} {simple_history[key][-1]:.4e}")
        print("-" * 50)

        self._plot_diagnostics(
            simple_history,
            model,
            time_indices,
            logit_cr_hist,
            data.years,
            n_years,
            show_plot,
        )

    # ------------------------------------------------------------------
    # Decomposed sub-methods
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_loss_scales(
        mode: str,
        *,
        delta_nca_true: tf.Tensor,
        depr_true: tf.Tensor,
        adv_ps_true: tf.Tensor,
        adv_pp_true: tf.Tensor,
        ar: tf.Tensor,
        ap: tf.Tensor,
        inv: tf.Tensor,
        cash: tf.Tensor,
        ims: tf.Tensor,
        tax: tf.Tensor,
        div_true: tf.Tensor,
        bb: tf.Tensor,
        logit_cr_hist: tf.Tensor,
        eff_st_debt: tf.Tensor,
        opex: tf.Tensor,
    ) -> PolicyLossScales:
        """Compute per-loss normalization factors from historical std devs."""
        eps = tf.constant(1e-12, dtype=tf.float64)
        if mode == "std":
            return PolicyLossScales(
                growth=tf.math.reduce_std(delta_nca_true) + eps,
                depr=tf.math.reduce_std(depr_true) + eps,
                adv_ps=tf.math.reduce_std(adv_ps_true) + eps,
                adv_pp=tf.math.reduce_std(adv_pp_true) + eps,
                ar=tf.math.reduce_std(ar) + eps,
                ap=tf.math.reduce_std(ap) + eps,
                inv=tf.math.reduce_std(inv) + eps,
                tl=tf.math.reduce_std(cash + ims) + eps,
                cash=tf.math.reduce_std(cash) + eps,
                tax=tf.math.reduce_std(tax) + eps,
                div=tf.math.reduce_std(div_true) + eps,
                bb=tf.math.reduce_std(bb) + eps,
                cost_ratio=tf.math.reduce_std(logit_cr_hist) + eps,
                eff_st=tf.math.reduce_std(eff_st_debt) + eps,
                opex=tf.math.reduce_std(opex) + eps,
            )
        elif mode == "none":
            one = tf.constant(1.0, dtype=tf.float64)
            return PolicyLossScales(
                growth=one,
                depr=one,
                adv_ps=one,
                adv_pp=one,
                ar=one,
                ap=one,
                inv=one,
                tl=one,
                cash=one,
                tax=one,
                div=one,
                bb=one,
                cost_ratio=one,
                eff_st=one,
                opex=one,
            )
        else:
            raise ValueError(
                f"Unsupported loss_scale_mode='{mode}'. Use 'std' or 'none'."
            )

    @staticmethod
    def _build_compiled_step(
        model,
        optimizer,
        vars_to_train,
        sales_tensor,
        purchases_tensor,
        cogs_tensor,
        nca_prev_aligned,
        delta_nca_true,
        depr_true,
        depr_tensor,
        sales_aligned_growth,
        adv_ps_true,
        adv_pp_true,
        ar_tensor,
        ap_tensor,
        inv_tensor,
        cash_tensor,
        ims_tensor,
        ni_tensor,
        ni_prev_aligned,
        div_true,
        div_prev_aligned,
        bb_tensor,
        opex_tensor,
        tax_tensor,
        eff_st_debt_tensor,
        cum_inf_tensor,
        time_indices,
        logit_cr_hist,
        scales,
    ):
        """Build and return the @tf.function compiled training step."""

        @tf.function
        def _compiled_train_step():
            with tf.GradientTape() as tape:
                loss_growth, loss_depr, prior_loss_am = (
                    model.balance_sheet.capex_policy.loss(
                        delta_nca_true,
                        depr_true,
                        sales_aligned_growth,
                        nca_prev_aligned,
                        scales.growth,
                        scales.depr,
                    )
                )
                loss_adv_ps, loss_adv_pp, loss_ar, loss_ap, loss_inv = (
                    model.balance_sheet.working_capital.loss(
                        sales_tensor,
                        purchases_tensor,
                        cogs_tensor,
                        adv_ps_true,
                        adv_pp_true,
                        ar_tensor,
                        ap_tensor,
                        inv_tensor,
                        scales.adv_ps,
                        scales.adv_pp,
                        scales.ar,
                        scales.ap,
                        scales.inv,
                    )
                )
                loss_tl, loss_cash = model.cash_budget.liquidity_policy.loss(
                    sales_tensor,
                    cash_tensor,
                    ims_tensor,
                    time_indices,
                    scales.tl,
                    scales.cash,
                )
                loss_tax = model.tax_module.loss(
                    tax_tensor,
                    ni_tensor,
                    scales.tax,
                )
                loss_div = model.cash_budget.dividend_policy.loss(
                    ni_prev_aligned,
                    div_true,
                    div_prev_aligned,
                    scales.div,
                )
                loss_bb = model.cash_budget.buyback_policy.loss(
                    bb_tensor,
                    depr_tensor,
                    scales.bb,
                )
                loss_cost_ratio = model.balance_sheet.purchases_policy.loss(
                    sales_tensor,
                    cogs_tensor,
                    inv_tensor,
                    time_indices,
                    scales.cost_ratio,
                )
                loss_eff_st_debt = model.cash_budget.debt_policy.loss_st_debt(
                    eff_st_debt_tensor,
                    sales_tensor,
                    time_indices,
                    scales.eff_st,
                )
                loss_opex = model.opex_module.loss(
                    opex_tensor,
                    sales_tensor,
                    cum_inf_tensor,
                    scales.opex,
                )

                total_loss = (
                    loss_growth
                    + loss_depr
                    + loss_adv_ps
                    + loss_adv_pp
                    + loss_ar
                    + loss_ap
                    + loss_inv
                    + loss_tl
                    + loss_cash
                    + loss_tax
                    + loss_div
                    + loss_bb
                    + loss_cost_ratio
                    + loss_eff_st_debt
                    + loss_opex
                    + prior_loss_am
                )

            grads = tape.gradient(total_loss, vars_to_train)
            optimizer.apply_gradients(zip(grads, vars_to_train))

            return tf.stack(
                [
                    total_loss,
                    loss_growth,
                    loss_depr,
                    loss_adv_ps,
                    loss_adv_pp,
                    loss_ar,
                    loss_ap,
                    loss_inv,
                    loss_tl,
                    loss_cash,
                    loss_tax,
                    loss_div,
                    loss_bb,
                    loss_cost_ratio,
                    loss_eff_st_debt,
                    loss_opex,
                    prior_loss_am,
                ]
            )

        return _compiled_train_step

    @staticmethod
    def _run_training_loop(
        compiled_step,
        epochs: int,
        plot_every: int,
        model: tf.Module,
        history: dict,
    ) -> None:
        """Execute the epoch loop and populate training history."""
        _L_TOTAL, _L_GROWTH, _L_DEPR = 0, 1, 2
        _L_ADV_PS, _L_ADV_PP, _L_AR, _L_AP, _L_INV = 3, 4, 5, 6, 7
        _L_TL, _L_CASH, _L_TAX, _L_DIV, _L_BB = 8, 9, 10, 11, 12
        _L_CR, _L_EFF_ST, _L_OPEX, _L_PRIOR_AM = 13, 14, 15, 16

        for i in range(epochs):
            loss_stack = compiled_step()

            if i % plot_every == 0:
                v = loss_stack.numpy()

                if model.opex_module.is_stochastic:
                    model.opex_module.record_step(i, v[_L_OPEX])
                else:
                    history["loss_opex"].append(v[_L_OPEX])

                history["epochs"].append(i)
                history["loss_total"].append(v[_L_TOTAL])
                history["loss_growth"].append(v[_L_GROWTH])
                history["loss_depr"].append(v[_L_DEPR])
                history["loss_adv_ps"].append(v[_L_ADV_PS])
                history["loss_adv_pp"].append(v[_L_ADV_PP])
                history["loss_ar"].append(v[_L_AR])
                history["loss_ap"].append(v[_L_AP])
                history["loss_inv"].append(v[_L_INV])
                history["loss_tl"].append(v[_L_TL])
                history["loss_cash"].append(v[_L_CASH])
                history["loss_tax"].append(v[_L_TAX])
                history["loss_div"].append(v[_L_DIV])
                history["loss_bb"].append(v[_L_BB])
                history["loss_cost_ratio"].append(v[_L_CR])
                history["loss_eff_st_debt"].append(v[_L_EFF_ST])
                history["loss_prior_am"].append(v[_L_PRIOR_AM])

                noise_str = (
                    f"OpEx Noise={(model.opex_module.noise_sigma.numpy() * model.amount_scale):.2e} | "
                    if model.opex_module.is_stochastic
                    else ""
                )
                print(
                    f"Epoch {i}: Loss={v[_L_TOTAL]:.4e} | "
                    f"OpEx Loss={v[_L_OPEX]:.4e} | "
                    f"{noise_str}"
                    f"AM={model.balance_sheet.capex_policy.asset_maintain.numpy():.4f} "
                    f"AG={model.balance_sheet.capex_policy.asset_growth.numpy():.6f} "
                    f"Prior_AM={v[_L_PRIOR_AM]:.4e}"
                )

    @staticmethod
    def _plot_diagnostics(
        history: dict,
        model: tf.Module,
        time_indices: tf.Tensor,
        logit_cr_hist: tf.Tensor,
        historical_years: Optional[tf.Tensor],
        n_years: int,
        show_plot: bool,
    ) -> None:
        """Generate diagnostic plots for training history."""
        plot_simple_policy_diagnostics(
            history,
            model,
            time_indices,
            logit_cr_hist,
            historical_years,
            n_years,
            show_plot,
        )
        model.opex_module.plot_diagnostics(show_plot)
