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
    plot_vi_diagnostics,
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
        historical_tax_onetime_payments=None,
        historical_inflation=None,
        historical_years=None,
        learning_rate=0.001,
        epochs=None,
        plot_vi=True,
        plot_every=1000,
        show_plot=False,
        prior_strength_asset_maintain=1.0,
        loss_scale_mode="std",
    ):
        """Train deterministic policy parameters and Bayesian OpEx jointly.

        Each loss component is normalized by its historical standard deviation
        (when ``loss_scale_mode="std"``) to ensure balanced gradients across
        heterogeneous magnitudes.

        Args:
            model: ``BayesianFinancialModel`` whose parameters are updated
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
            historical_tax_onetime_payments: Optional 1-D array-like of
                one-time tax payments.
            historical_inflation: Optional 1-D array-like of annual inflation
                rates.
            historical_years: Optional 1-D array-like of fiscal years.
            learning_rate: Adam optimizer learning rate.
            epochs: Number of training iterations.
            plot_vi: Whether to save diagnostic plots.
            plot_every: Logging and history recording interval.
            show_plot: Whether to display plots interactively.
            prior_strength_asset_maintain: Strength of the quadratic prior
                pulling ``asset_maintain`` toward 1.0.
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
        if historical_tax_onetime_payments is None:
            tax_onetime_tensor = tf.zeros_like(tax_tensor)
        else:
            tax_onetime_tensor = _as_float64_tensor(historical_tax_onetime_payments)
        eff_st_debt_tensor = _as_float64_tensor(historical_eff_st_debt)

        if historical_inflation is None:
            historical_inflation = tf.zeros_like(sales_tensor)
        inf_tensor = _as_float64_tensor(historical_inflation)
        cum_inf_tensor = tf.math.cumprod(1 + inf_tensor)

        # --- Calculate and store sales offset for OpEx training ---
        sales_offset_value = tf.reduce_mean(sales_tensor)
        model.sales_offset.assign(sales_offset_value)
        sales_tensor_centered = sales_tensor - sales_offset_value

        print(f"Sales offset for OpEx training: {sales_offset_value.numpy():.4e}")
        print(
            f"Sales range before centering: "
            f"[{tf.reduce_min(sales_tensor).numpy():.4e}, "
            f"{tf.reduce_max(sales_tensor).numpy():.4e}]"
        )
        print(
            f"Sales range after centering: "
            f"[{tf.reduce_min(sales_tensor_centered).numpy():.4e}, "
            f"{tf.reduce_max(sales_tensor_centered).numpy():.4e}]"
        )

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
        num_opex_obs = tf.cast(tf.size(opex_tensor), tf.float64)

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        print(f"Training on {len(historical_sales)} years of historical data...")

        # --- Collect trainable variables from model ---
        # TransformedVariable wraps an unconstrained variable and applies a
        # bijector on read (e.g. Softplus for non-negative).  We train the
        # underlying unconstrained variable via .trainable_variables[0].
        vars_to_train = [
            model.asset_growth.trainable_variables[0],
            model.asset_maintain.trainable_variables[0],
            model.depreciation_rate.trainable_variables[0],
            model.advance_payments_sales_pct.trainable_variables[0],
            model.advance_payments_purchases_pct.trainable_variables[0],
            model.account_receivables_pct.trainable_variables[0],
            model.account_payables_pct.trainable_variables[0],
            model.inventory_pct.trainable_variables[0],
            model.tl_alpha,
            model.tl_beta,
            model.tl_baseline,
            model.cash_alpha,
            model.cash_beta,
            model.income_tax_pct.trainable_variables[0],
            model.dividend_payout_ratio_pct.trainable_variables[0],
            model.dividend_adjustment_speed.trainable_variables[0],
            model.sb_baseline,
            model.sb_ratio,
            model.st_debt_alpha,
            model.st_debt_beta,
            model.cost_ratio_alpha,
            model.cost_ratio_beta,
            model.q_var_opex_loc,
            model.q_var_opex_scale.trainable_variables[0],
            model.q_base_opex_loc,
            model.q_base_opex_scale.trainable_variables[0],
            model.noise_sigma.trainable_variables[0],
        ]

        vi_history = {
            "epochs": [],
            "loss_vi": [],
            "q_var_opex_loc": [],
            "q_var_opex_scale": [],
            "q_base_opex_loc": [],
            "q_base_opex_scale": [],
            "noise_sigma": [],
        }
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
            "loss_prior_am": [],
        }

        for i in range(epochs):
            with tf.GradientTape() as tape:
                # --- Deterministic Losses (MSE on historical ratio targets) ---
                # Asset growth: delta_NCA = (AM-1)*Depr + AG*Sales
                loss_growth = tf.reduce_mean(
                    tf.square(
                        (
                            delta_nca_true
                            - (
                                (model.asset_maintain - 1) * depr_true
                                + sales_aligned_growth * model.asset_growth
                            )
                        )
                        / scale_growth
                    )
                )
                loss_depr = tf.reduce_mean(
                    tf.square(
                        (depr_true - nca_prev_aligned * model.depreciation_rate)
                        / scale_depr
                    )
                )
                loss_adv_ps = tf.reduce_mean(
                    tf.square(
                        (adv_ps_true - sales_tensor * model.advance_payments_sales_pct)
                        / scale_adv_ps
                    )
                )
                loss_adv_pp = tf.reduce_mean(
                    tf.square(
                        (
                            adv_pp_true
                            - purchases_aligned_adv_pp
                            * model.advance_payments_purchases_pct
                        )
                        / scale_adv_pp
                    )
                )
                loss_ar = tf.reduce_mean(
                    tf.square(
                        (ar_tensor - sales_tensor * model.account_receivables_pct)
                        / scale_ar
                    )
                )
                loss_ap = tf.reduce_mean(
                    tf.square(
                        (ap_tensor - purchases_tensor * model.account_payables_pct)
                        / scale_ap
                    )
                )
                loss_inv = tf.reduce_mean(
                    tf.square(
                        (inv_tensor - sales_tensor * model.inventory_pct) / scale_inv
                    )
                )
                tl_pct_t_logit = model.tl_alpha + model.tl_beta * time_indices
                tl_pct_t = tf.sigmoid(tl_pct_t_logit)
                loss_tl = tf.reduce_mean(
                    tf.square(
                        (
                            (cash_tensor + ims_tensor)
                            - (model.tl_baseline + sales_tensor * tl_pct_t)
                        )
                        / scale_tl
                    )
                )
                cash_pct_t_logit = model.cash_alpha + model.cash_beta * time_indices
                cash_pct_t = tf.sigmoid(cash_pct_t_logit)
                loss_cash = tf.reduce_mean(
                    tf.square(
                        (cash_tensor - (cash_tensor + ims_tensor) * cash_pct_t)
                        / scale_cash
                    )
                )
                tax_pred_total = (
                    ni_tensor / (1 / model.income_tax_pct - 1) + tax_onetime_tensor
                )
                loss_tax = tf.reduce_mean(
                    tf.square((tax_tensor - tax_pred_total) / scale_tax)
                )
                div_target = ni_prev_aligned * model.dividend_payout_ratio_pct
                div_pred = (
                    model.dividend_adjustment_speed * div_target
                    + (1.0 - model.dividend_adjustment_speed) * div_prev_aligned
                )
                loss_div = tf.reduce_mean(tf.square((div_true - div_pred) / scale_div))
                bb_pred = model.sb_baseline + model.sb_ratio * depr_tensor
                loss_bb = tf.reduce_mean(tf.square((bb_tensor - bb_pred) / scale_bb))

                # Cost Ratio (Logit-Linear)
                logit_cr_pred = (
                    model.cost_ratio_alpha + model.cost_ratio_beta * time_indices
                )
                loss_cost_ratio = tf.reduce_mean(
                    tf.square((logit_cr_hist - logit_cr_pred) / scale_cost_ratio)
                )

                # ST Debt (Logit-Linear)
                st_debt_pct_pred = tf.sigmoid(
                    model.st_debt_alpha + model.st_debt_beta * time_indices
                )
                loss_eff_st_debt = tf.reduce_mean(
                    tf.square(
                        (eff_st_debt_tensor - sales_tensor * st_debt_pct_pred)
                        / scale_eff_st
                    )
                )

                # --- Bayesian OpEx Loss (ELBO = NLL + KL) ---
                # Sample from variational posterior via reparameterization trick,
                # then compute negative log-likelihood + KL divergence against
                # wide Gaussian priors (approximately uniform).
                var_opex_sample, base_opex_sample = model.sample_opex_params()
                pred_opex_raw = (base_opex_sample * cum_inf_tensor) + (
                    var_opex_sample * sales_tensor_centered
                )
                residuals = (opex_tensor - pred_opex_raw) / scale_opex
                likelihood_dist = tfd.Normal(
                    loc=0.0, scale=(model.noise_sigma / scale_opex)
                )
                neg_log_likelihood = -tf.reduce_sum(likelihood_dist.log_prob(residuals))
                kl = model.get_opex_kl_divergence()
                loss_opex_bayes = (neg_log_likelihood + kl) / num_opex_obs

                # Quadratic prior on asset_maintain centered at 1.0:
                # AM ≈ 1.0 means capex fully replaces depreciation (maintenance).
                # Without this prior, the optimizer can collapse AM → 0 and absorb
                # everything into asset_growth, which is economically implausible.
                prior_loss_am = prior_strength_asset_maintain * tf.square(
                    model.asset_maintain - 1.0
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
                    + loss_opex_bayes
                    + prior_loss_am
                )

            grads = tape.gradient(total_loss, vars_to_train)
            optimizer.apply_gradients(zip(grads, vars_to_train))

            if i % plot_every == 0:
                vi_history["epochs"].append(i)
                vi_history["loss_vi"].append(loss_opex_bayes.numpy())
                vi_history["q_var_opex_loc"].append(model.q_var_opex_loc.numpy())
                vi_history["q_var_opex_scale"].append(model.q_var_opex_scale.numpy())
                vi_history["q_base_opex_loc"].append(model.q_base_opex_loc.numpy())
                vi_history["q_base_opex_scale"].append(model.q_base_opex_scale.numpy())
                vi_history["noise_sigma"].append(model.noise_sigma.numpy())

                simple_history["epochs"].append(i)
                simple_history["loss_total"].append(total_loss.numpy())
                simple_history["loss_growth"].append(loss_growth.numpy())
                simple_history["loss_depr"].append(loss_depr.numpy())
                simple_history["loss_adv_ps"].append(loss_adv_ps.numpy())
                simple_history["loss_adv_pp"].append(loss_adv_pp.numpy())
                simple_history["loss_ar"].append(loss_ar.numpy())
                simple_history["loss_ap"].append(loss_ap.numpy())
                simple_history["loss_inv"].append(loss_inv.numpy())
                simple_history["loss_tl"].append(loss_tl.numpy())
                simple_history["loss_cash"].append(loss_cash.numpy())
                simple_history["loss_tax"].append(loss_tax.numpy())
                simple_history["loss_div"].append(loss_div.numpy())
                simple_history["loss_bb"].append(loss_bb.numpy())
                simple_history["loss_cost_ratio"].append(loss_cost_ratio.numpy())
                simple_history["loss_eff_st_debt"].append(loss_eff_st_debt.numpy())
                simple_history["loss_prior_am"].append(prior_loss_am.numpy())

                print(
                    f"Epoch {i}: Loss={total_loss.numpy():.4e} | "
                    f"OpEx VI Loss={loss_opex_bayes.numpy():.4e} | "
                    f"OpEx Noise={(model.noise_sigma.numpy() * model.amount_scale):.2e} | "
                    f"AM={model.asset_maintain.numpy():.4f} "
                    f"AG={model.asset_growth.numpy():.6f} "
                    f"Prior_AM={prior_loss_am.numpy():.4e}"
                )

        # --- Print final parameter values ---
        print("-" * 50)
        print("Training Complete.")
        print(f"Final %AG: {model.asset_growth.numpy():.5f}")
        print(f"Final %AM: {model.asset_maintain.numpy():.5f}")
        print(f"Final %Depr: {model.depreciation_rate.numpy():.5f}")
        print(f"Final %AdvPS: {model.advance_payments_sales_pct.numpy():.5f}")
        print(f"Final %AdvPP: {model.advance_payments_purchases_pct.numpy():.5f}")
        print(f"Final %AR: {model.account_receivables_pct.numpy():.5f}")
        print(f"Final %AP: {model.account_payables_pct.numpy():.5f}")
        print(f"Final %Inv: {model.inventory_pct.numpy():.5f}")
        n_years = len(historical_sales)
        print(
            f"Total Liquidity (baseline + logit-linear): "
            f"baseline={model.tl_baseline.numpy():.4f}, "
            f"alpha={model.tl_alpha.numpy():.4f}, "
            f"beta={model.tl_beta.numpy():.6f}"
        )
        print(
            f"  => %TL at t=0: {tf.sigmoid(model.tl_alpha).numpy():.4f}, "
            f"%TL at t={n_years-1}: "
            f"{tf.sigmoid(model.tl_alpha + model.tl_beta * (n_years-1)).numpy():.4f}"
        )
        print(
            f"Cash % of Liquidity (logit-linear): "
            f"alpha={model.cash_alpha.numpy():.4f}, "
            f"beta={model.cash_beta.numpy():.6f}"
        )
        print(
            f"  => %Cash at t=0: {tf.sigmoid(model.cash_alpha).numpy():.4f}, "
            f"%Cash at t={n_years-1}: "
            f"{tf.sigmoid(model.cash_alpha + model.cash_beta * (n_years-1)).numpy():.4f}"
        )
        print(f"Final %IT: {model.income_tax_pct.numpy():.5f}")
        print(f"Final %PR: {model.dividend_payout_ratio_pct.numpy():.5f}")
        print(f"Final DivAdjSpeed: {model.dividend_adjustment_speed.numpy():.5f}")
        print(
            f"Stock Buyback (baseline + ratio*depr): "
            f"baseline={model.sb_baseline.numpy():.4f}, "
            f"ratio={model.sb_ratio.numpy():.6f}"
        )
        print(
            f"Effective ST Debt % of Sales (logit-linear): "
            f"alpha={model.st_debt_alpha.numpy():.4f}, "
            f"beta={model.st_debt_beta.numpy():.6f}"
        )
        print(
            f"  => %EffSTDebt at t=0: "
            f"{tf.sigmoid(model.st_debt_alpha).numpy():.4f}, "
            f"%EffSTDebt at t={n_years-1}: "
            f"{tf.sigmoid(model.st_debt_alpha + model.st_debt_beta * (n_years-1)).numpy():.4f}"
        )
        print(
            f"Cost Ratio (logit-linear): "
            f"alpha={model.cost_ratio_alpha.numpy():.4f}, "
            f"beta={model.cost_ratio_beta.numpy():.4f}"
        )
        print(
            f"  => CR at t=0: "
            f"{tf.sigmoid(model.cost_ratio_alpha).numpy():.4f}, "
            f"CR at t={n_years-1}: "
            f"{tf.sigmoid(model.cost_ratio_alpha + model.cost_ratio_beta * (n_years-1)).numpy():.4f}"
        )
        print(
            f"Bayesian OpEx Variable %: "
            f"Mean={model.q_var_opex_loc.numpy():.4f}, "
            f"Std={model.q_var_opex_scale.numpy():.4f}"
        )
        print(
            f"Bayesian OpEx Baseline (USD):   "
            f"Mean={(model.q_base_opex_loc.numpy() * model.amount_scale):.2e}, "
            f"Std={(model.q_base_opex_scale.numpy() * model.amount_scale):.2e}"
        )
        print(
            f"OpEx aleatoric uncertainty (USD): "
            f"{(model.noise_sigma.numpy() * model.amount_scale):.2e}"
        )
        print("-" * 50)

        # --- Diagnostic Plots ---
        if plot_vi:
            plot_vi_diagnostics(vi_history, model.amount_scale, show_plot)
            plot_simple_policy_diagnostics(
                simple_history,
                model,
                time_indices,
                logit_cr_hist,
                historical_years,
                n_years,
                show_plot,
            )
