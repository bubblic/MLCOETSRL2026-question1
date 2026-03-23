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

import tensorflow as tf
import tensorflow_probability as tfp

from financial_forecast.training.base_trainer import BaseTrainer
from financial_forecast.training.diagnostics import (
    plot_simple_policy_diagnostics,
)

tfd = tfp.distributions


def _as_float64_tensor(value):
    """Cast *value* to a ``tf.float64`` tensor."""
    return tf.convert_to_tensor(value, dtype=tf.float64)


class PolicyTrainer(BaseTrainer):
    """Trains deterministic policy parameters and Bayesian OpEx jointly.

    Optimizes all policy-level parameters (asset growth, depreciation rate,
    working capital ratios, cost ratio trend, dividend smoothing, etc.)
    alongside the variational OpEx parameters in a single Adam loop.

    Args:
        epochs: Default number of training iterations.  Can be overridden
            per-call via the *epochs* argument of :meth:`train`.
    """

    def __init__(self, epochs=25000):
        self.epochs = epochs

    def train(
        self,
        model,
        historical_sales,
        historical_purchases,
        historical_cogs,
        historical_nca,
        historical_depreciation,
        historical_adv_pay_sales,
        historical_adv_pay_purch,
        historical_ar,
        historical_ap,
        historical_inventory,
        historical_cash,
        historical_ims,
        historical_net_income,
        historical_dividends,
        historical_stock_buyback,
        historical_opex,
        historical_tax,
        historical_eff_st_debt,
        historical_inflation=None,
        historical_years=None,
        learning_rate=0.001,
        epochs=None,
        plot_every=1000,
        show_plot=False,
        loss_scale_mode="std",
    ):
        """Train deterministic policy parameters and Bayesian OpEx jointly.

        Each loss component is normalized by its historical standard deviation
        (when ``loss_scale_mode="std"``) to ensure balanced gradients across
        heterogeneous magnitudes.

        Args:
            model: ``TrainableFinancialModel`` whose parameters are updated
                in-place.
            historical_sales: 1-D array-like of annual sales figures.
            historical_purchases: 1-D array-like of annual purchase figures.
            historical_cogs: 1-D array-like of annual cost of goods sold.
            historical_nca: 1-D array-like of annual non-current assets.
            historical_depreciation: 1-D array-like of annual depreciation.
            historical_adv_pay_sales: 1-D array-like of advance payments on
                sales.
            historical_adv_pay_purch: 1-D array-like of advance payments on
                purchases.
            historical_ar: 1-D array-like of accounts receivable.
            historical_ap: 1-D array-like of accounts payable.
            historical_inventory: 1-D array-like of inventory values.
            historical_cash: 1-D array-like of cash balances.
            historical_ims: 1-D array-like of investment in market securities.
            historical_net_income: 1-D array-like of net income.
            historical_dividends: 1-D array-like of dividend payments.
            historical_stock_buyback: 1-D array-like of stock buyback amounts.
            historical_opex: 1-D array-like of operating expenses.
            historical_tax: 1-D array-like of tax payments.
            historical_eff_st_debt: 1-D array-like of effective short-term
                debt.
            historical_inflation: Optional 1-D array-like of annual inflation
                rates.
            historical_years: Optional 1-D array-like of fiscal years.
            learning_rate: Adam optimizer learning rate.
            epochs: Number of training iterations.
            plot_every: Logging and history recording interval.
            show_plot: Whether to display plots interactively.
            loss_scale_mode: ``"std"`` or ``"none"``.
        """
        if epochs is None:
            epochs = self.epochs

        # --- Convert inputs to tensors ---
        sales_tensor = _as_float64_tensor(historical_sales)
        purchases_tensor = _as_float64_tensor(historical_purchases)
        cogs_tensor = _as_float64_tensor(historical_cogs)
        nca_tensor = _as_float64_tensor(historical_nca)
        depr_tensor = _as_float64_tensor(historical_depreciation)
        adv_pay_sales_tensor = _as_float64_tensor(historical_adv_pay_sales)
        adv_pay_purch_tensor = _as_float64_tensor(historical_adv_pay_purch)
        ar_tensor = _as_float64_tensor(historical_ar)
        ap_tensor = _as_float64_tensor(historical_ap)
        inv_tensor = _as_float64_tensor(historical_inventory)
        cash_tensor = _as_float64_tensor(historical_cash)
        ims_tensor = _as_float64_tensor(historical_ims)
        ni_tensor = _as_float64_tensor(historical_net_income)
        div_tensor = _as_float64_tensor(historical_dividends)
        bb_tensor = _as_float64_tensor(historical_stock_buyback)
        opex_tensor = _as_float64_tensor(historical_opex)
        tax_tensor = _as_float64_tensor(historical_tax)
        eff_st_debt_tensor = _as_float64_tensor(historical_eff_st_debt)

        if historical_inflation is None:
            historical_inflation = tf.zeros_like(sales_tensor)
        inf_tensor = _as_float64_tensor(historical_inflation)
        cum_inf_tensor = tf.math.cumprod(1 + inf_tensor)

        # --- Cost Ratio Training Data ---
        cost_ratio_hist = cogs_tensor / sales_tensor
        logit_cr_hist = tf.math.log(cost_ratio_hist / (1.0 - cost_ratio_hist))

        if historical_years is not None:
            time_indices = tf.cast(historical_years, dtype=tf.float64) - tf.constant(
                float(model.base_year), dtype=tf.float64
            )
        else:
            time_indices = tf.cast(tf.range(len(historical_sales)), dtype=tf.float64)

        # --- Prepare Training Data & Alignment ---
        # Growth and depreciation losses use [1:] vs [:-1] alignment because
        # delta_NCA_t = NCA_t - NCA_{t-1} and depr_t = NCA_{t-1} * depr_rate.
        delta_nca_true = nca_tensor[1:] - nca_tensor[:-1]
        sales_aligned_growth = sales_tensor[1:]
        depr_true = depr_tensor[1:]
        nca_prev_aligned = nca_tensor[:-1]
        adv_ps_true = adv_pay_sales_tensor
        adv_pp_true = adv_pay_purch_tensor
        purchases_aligned_adv_pp = purchases_tensor
        # Dividend smoothing (Lintner): D_t depends on NI_{t-1} and D_{t-1}
        div_true = div_tensor[1:]
        ni_prev_aligned = ni_tensor[:-1]
        div_prev_aligned = div_tensor[:-1]

        # --- Loss Scaling ---
        # Normalize each loss by historical std to balance gradients across
        # quantities with different magnitudes (e.g. NCA in billions vs ratios).
        eps = tf.constant(1e-12, dtype=tf.float64)
        if loss_scale_mode == "std":
            scale_growth = tf.math.reduce_std(delta_nca_true) + eps
            scale_depr = tf.math.reduce_std(depr_true) + eps
            scale_adv_ps = tf.math.reduce_std(adv_ps_true) + eps
            scale_adv_pp = tf.math.reduce_std(adv_pp_true) + eps
            scale_ar = tf.math.reduce_std(ar_tensor) + eps
            scale_ap = tf.math.reduce_std(ap_tensor) + eps
            scale_inv = tf.math.reduce_std(inv_tensor) + eps
            scale_tl = tf.math.reduce_std(cash_tensor + ims_tensor) + eps
            scale_cash = tf.math.reduce_std(cash_tensor) + eps
            scale_tax = tf.math.reduce_std(tax_tensor) + eps
            scale_div = tf.math.reduce_std(div_true) + eps
            scale_bb = tf.math.reduce_std(bb_tensor) + eps
            scale_cost_ratio = tf.math.reduce_std(logit_cr_hist) + eps
            scale_eff_st = tf.math.reduce_std(eff_st_debt_tensor) + eps
            scale_opex = tf.math.reduce_std(opex_tensor) + eps
        elif loss_scale_mode == "none":
            one = tf.constant(1.0, dtype=tf.float64)
            scale_growth = scale_depr = scale_adv_ps = scale_adv_pp = one
            scale_ar = scale_ap = scale_inv = scale_tl = scale_cash = one
            scale_tax = scale_div = scale_bb = scale_cost_ratio = one
            scale_eff_st = scale_opex = one
        else:
            raise ValueError(
                f"Unsupported loss_scale_mode='{loss_scale_mode}'. "
                "Use 'std' or 'none'."
            )

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        print(f"Training on {len(historical_sales)} years of historical data...")

        # --- Collect trainable variables from model ---
        # TransformedVariable wraps an unconstrained variable and applies a
        # bijector on read (e.g. Softplus for non-negative).  We train the
        # underlying unconstrained variable via .trainable_variables[0].
        vars_to_train = [
            *model.balance_sheet.trainable_variables,
            *model.tax_module.trainable_variables,
            *model.cash_budget.debt_policy.trainable_variables,
            *model.opex_module.trainable_variables,
        ]

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

        _zero = tf.constant(0.0, dtype=tf.float64)

        # --- Compiled training step ---
        # @tf.function compiles the loss computation + gradient update into
        # an optimized TF graph, eliminating Python overhead per epoch.
        # All losses are returned as a stacked tensor so the Python loop
        # can log them with a single GPU-to-CPU transfer.
        @tf.function
        def _compiled_train_step():
            with tf.GradientTape() as tape:
                # Deterministic Losses (MSE on historical ratio targets)
                loss_growth, loss_depr, prior_loss_am = (
                    model.balance_sheet.capex_policy.loss(
                        delta_nca_true,
                        depr_true,
                        sales_aligned_growth,
                        nca_prev_aligned,
                        scale_growth,
                        scale_depr,
                    )
                )
                loss_adv_ps, loss_adv_pp, loss_ar, loss_ap, loss_inv = (
                    model.balance_sheet.working_capital.loss(
                        sales_tensor,
                        purchases_tensor,
                        adv_ps_true,
                        adv_pp_true,
                        ar_tensor,
                        ap_tensor,
                        inv_tensor,
                        scale_adv_ps,
                        scale_adv_pp,
                        scale_ar,
                        scale_ap,
                        scale_inv,
                    )
                )
                loss_tl, loss_cash = model.balance_sheet.liquidity_policy.loss(
                    sales_tensor,
                    cash_tensor,
                    ims_tensor,
                    time_indices,
                    scale_tl,
                    scale_cash,
                )
                loss_tax = model.tax_module.loss(
                    tax_tensor,
                    ni_tensor,
                    scale_tax,
                )
                loss_div = model.cash_budget.dividend_policy.loss(
                    ni_prev_aligned,
                    div_true,
                    div_prev_aligned,
                    scale_div,
                )
                loss_bb = model.cash_budget.buyback_policy.loss(
                    bb_tensor,
                    depr_tensor,
                    scale_bb,
                )

                # Cost Ratio
                loss_cost_ratio = model.balance_sheet.purchases_policy.loss(
                    sales_tensor,
                    cogs_tensor,
                    inv_tensor,
                    time_indices,
                    scale_cost_ratio,
                )

                # ST Debt (Logit-Linear)
                loss_eff_st_debt = model.cash_budget.debt_policy.loss_st_debt(
                    eff_st_debt_tensor,
                    sales_tensor,
                    time_indices,
                    scale_eff_st,
                )

                # OpEx Loss (could be simple MSE or Bayesian with ELBO = NLL + KL)
                loss_opex = model.opex_module.loss(
                    opex_tensor,
                    sales_tensor,
                    cum_inf_tensor,
                    scale_opex,
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

            # Return all losses as a single stacked tensor for efficient
            # GPU-to-CPU transfer during logging.
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

        # Index mapping for the stacked loss tensor
        _L_TOTAL, _L_GROWTH, _L_DEPR = 0, 1, 2
        _L_ADV_PS, _L_ADV_PP, _L_AR, _L_AP, _L_INV = 3, 4, 5, 6, 7
        _L_TL, _L_CASH, _L_TAX, _L_DIV, _L_BB = 8, 9, 10, 11, 12
        _L_CR, _L_EFF_ST, _L_OPEX, _L_PRIOR_AM = 13, 14, 15, 16

        for i in range(epochs):
            loss_stack = _compiled_train_step()

            if i % plot_every == 0:
                v = loss_stack.numpy()

                if model.opex_module.is_stochastic == True:
                    model.opex_module.record_step(i, v[_L_OPEX])
                else:
                    simple_history["loss_opex"].append(v[_L_OPEX])

                simple_history["epochs"].append(i)
                simple_history["loss_total"].append(v[_L_TOTAL])
                simple_history["loss_growth"].append(v[_L_GROWTH])
                simple_history["loss_depr"].append(v[_L_DEPR])
                simple_history["loss_adv_ps"].append(v[_L_ADV_PS])
                simple_history["loss_adv_pp"].append(v[_L_ADV_PP])
                simple_history["loss_ar"].append(v[_L_AR])
                simple_history["loss_ap"].append(v[_L_AP])
                simple_history["loss_inv"].append(v[_L_INV])
                simple_history["loss_tl"].append(v[_L_TL])
                simple_history["loss_cash"].append(v[_L_CASH])
                simple_history["loss_tax"].append(v[_L_TAX])
                simple_history["loss_div"].append(v[_L_DIV])
                simple_history["loss_bb"].append(v[_L_BB])
                simple_history["loss_cost_ratio"].append(v[_L_CR])
                simple_history["loss_eff_st_debt"].append(v[_L_EFF_ST])
                simple_history["loss_prior_am"].append(v[_L_PRIOR_AM])

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

        # --- Print final parameter values ---
        print("-" * 50)
        print("Training Complete.")
        model.balance_sheet.capex_policy.print_summary()
        model.balance_sheet.working_capital.print_summary()
        n_years = len(historical_sales)
        model.balance_sheet.liquidity_policy.print_summary(n_years)
        model.tax_module.print_summary()
        model.cash_budget.dividend_policy.print_summary()
        model.cash_budget.buyback_policy.print_summary()
        model.cash_budget.debt_policy.print_summary(n_years)
        model.balance_sheet.purchases_policy.print_summary(n_years)
        model.opex_module.print_summary()
        print("-" * 50)

        # --- Diagnostic Plots ---
        plot_simple_policy_diagnostics(
            simple_history,
            model,
            time_indices,
            logit_cr_hist,
            historical_years,
            n_years,
            show_plot,
        )
        model.opex_module.plot_diagnostics(show_plot)
