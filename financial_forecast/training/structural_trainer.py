"""Structural parameter trainer using historical state transitions.

The trainer receives a model instance and updates its structural
parameters (interest rates, debt maturity, market-securities return,
equity-financing mix) via gradient descent.  Unlike the policy trainer
where each parameter has an independent loss, structural parameters are
**coupled through ``forecast_step``**: a single forward pass produces
predictions for net income, debt, equity, and interest, and every
structural parameter's gradient flows through that shared computation.

Dependency flow::

    structural_trainer  ->  base_trainer  (ABC)
                        ->  diagnostics   (plotting)
                        ->  models/base   (model interface)
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import tensorflow as tf

from financial_forecast.training.base_trainer import BaseTrainer
from financial_forecast.training.diagnostics import plot_structural_diagnostics

if TYPE_CHECKING:
    from financial_forecast.models.base import BaseFinancialModel
    from financial_forecast.types import HistoricalTrainingData


def _as_float64_tensor(value: tf.Tensor) -> tf.Tensor:
    """Cast *value* to a ``tf.float64`` tensor."""
    return tf.convert_to_tensor(value, dtype=tf.float64)


class StructuralTrainer(BaseTrainer):
    """Trains structural parameters on state-transition prediction errors.

    Optimizes interest rates, average debt maturity, market securities
    return, and the equity financing mix by minimizing state-transition
    prediction errors over consecutive historical years.

    Args:
        epochs: Default number of training iterations.  Can be overridden
            per-call via the *epochs* argument of :meth:`train`.
    """

    def __init__(self, epochs: int = 20000):
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
        gradient_clip_norm: Optional[float] = 5.0,
    ) -> None:
        """Train structural parameters using historical state transitions."""
        if epochs is None:
            epochs = self.epochs

        sales_t = _as_float64_tensor(data.sales)
        nca_t = _as_float64_tensor(data.nca)
        adv_ps_t = _as_float64_tensor(data.advance_payments_sales)
        adv_pp_t = _as_float64_tensor(data.advance_payments_purchases)
        ar_t = _as_float64_tensor(data.accounts_receivable)
        ap_t = _as_float64_tensor(data.accounts_payable)
        inv_t = _as_float64_tensor(data.inventory)
        cash_t = _as_float64_tensor(data.cash)
        ims_t = _as_float64_tensor(data.ims)
        ni_t = _as_float64_tensor(data.net_income)
        div_t = _as_float64_tensor(data.dividends)
        eff_st_t = _as_float64_tensor(data.effective_st_debt)
        curr_lt_t = _as_float64_tensor(data.current_lt_debt)
        ncl_t = _as_float64_tensor(data.non_current_liabilities)
        interest_t = _as_float64_tensor(data.interest_payment)
        ms_return_t = _as_float64_tensor(data.ms_return)
        equity_t = _as_float64_tensor(data.equity)
        inf_t = _as_float64_tensor(data.inflation)
        cum_inf_t = tf.math.cumprod(1 + inf_t)
        years_t = _as_float64_tensor(data.years)

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        eps = tf.constant(1e-12, dtype=tf.float64)

        def finite_std(values):
            """Compute std over finite values only; fall back to 1.0 if insufficient data."""
            finite_values = tf.boolean_mask(values, tf.math.is_finite(values))
            return tf.cond(
                tf.size(finite_values) > 1,
                lambda: tf.math.reduce_std(finite_values) + eps,
                lambda: tf.constant(1.0, dtype=tf.float64),
            )

        if loss_scale_mode == "std":
            scale_ni = tf.math.reduce_std(ni_t[1:]) + eps
            scale_eff_st = tf.math.reduce_std(eff_st_t[1:]) + eps
            scale_curr_lt = tf.math.reduce_std(curr_lt_t[1:]) + eps
            scale_ncl = tf.math.reduce_std(ncl_t[1:]) + eps
            scale_equity = tf.math.reduce_std(equity_t[1:]) + eps
            scale_interest = finite_std(interest_t[1:])
            scale_ms_return = finite_std(ms_return_t[1:])
        elif loss_scale_mode == "none":
            one = tf.constant(1.0, dtype=tf.float64)
            scale_ni = scale_eff_st = scale_curr_lt = scale_ncl = one
            scale_equity = scale_interest = scale_ms_return = one
        else:
            raise ValueError(
                f"Unsupported loss_scale_mode='{loss_scale_mode}'. "
                "Use 'std' or 'none'."
            )

        vars_to_train = model.structural_trainable_variables

        structural_history = {
            "epochs": [],
            "loss_total": [],
            "loss_ni": [],
            "loss_interest": [],
            "loss_ms_return": [],
            "loss_curr_lt": [],
            "loss_ncl": [],
            "loss_equity": [],
        }

        print("Training structural parameters...")
        num_transitions = len(data.sales) - 1

        # Cast gradient clip norm to tensor for graph-mode compatibility
        clip_norm = (
            tf.constant(gradient_clip_norm, dtype=tf.float64)
            if gradient_clip_norm is not None and gradient_clip_norm > 0
            else None
        )
        _zero = tf.constant(0.0, dtype=tf.float64)

        # --- Compiled training step ---
        # @tf.function compiles the full transition loop + gradient update
        # into a TF graph.  The inner Python loop over transitions is
        # unrolled at trace time (typically ~6 iterations, trivial).
        @tf.function
        def _compiled_train_step():
            with tf.GradientTape() as tape:
                total_loss = _zero
                total_loss_ni = _zero
                total_loss_interest = _zero
                total_loss_ms_return = _zero
                total_loss_curr_lt = _zero
                total_loss_ncl = _zero
                total_loss_equity = _zero

                for t in range(num_transitions):
                    state_prev = {
                        "nca": nca_t[t],
                        "advance_payments_purchases": adv_pp_t[t],
                        "accounts_receivable": ar_t[t],
                        "inventory": inv_t[t],
                        "cash": cash_t[t],
                        "investment_in_market_securities": ims_t[t],
                        "accounts_payable": ap_t[t],
                        "advance_payments_sales": adv_ps_t[t],
                        "effective_st_debt": eff_st_t[t],
                        "current_lt_debt": curr_lt_t[t],
                        "non_current_liabilities": ncl_t[t],
                        "equity": equity_t[t],
                        "net_income": ni_t[t],
                        "dividends": div_t[t],
                    }
                    inputs_curr = {
                        "sales_t": sales_t[t + 1],
                        "year": years_t[t + 1],
                        "cum_inflation": cum_inf_t[t + 1],
                    }

                    # Deterministic mean OpEx for structural training
                    state_pred = model.forecast_step(
                        state_prev,
                        inputs_curr,
                        use_mean_opex=True,
                    )

                    loss_ni = tf.square(
                        (state_pred["net_income"] - ni_t[t + 1]) / scale_ni
                    )
                    loss_eff_st = tf.square(
                        (state_pred["effective_st_debt"] - eff_st_t[t + 1])
                        / scale_eff_st
                    )
                    loss_curr_lt = tf.square(
                        (state_pred["current_lt_debt"] - curr_lt_t[t + 1])
                        / scale_curr_lt
                    )
                    loss_ncl = tf.square(
                        (state_pred["non_current_liabilities"] - ncl_t[t + 1])
                        / scale_ncl
                    )
                    loss_equity = tf.square(
                        (state_pred["equity"] - equity_t[t + 1]) / scale_equity
                    )
                    # Interest may be NaN/Inf for missing observations
                    valid_interest = tf.cast(
                        tf.math.is_finite(interest_t[t + 1]), tf.float64
                    )
                    interest_target = tf.where(
                        tf.math.is_finite(interest_t[t + 1]),
                        interest_t[t + 1],
                        state_pred["interest_payment"],
                    )
                    loss_interest = valid_interest * tf.square(
                        (state_pred["interest_payment"] - interest_target)
                        / scale_interest
                    )
                    # MS return may be NaN for missing observations
                    valid_ms_return = tf.cast(
                        tf.math.is_finite(ms_return_t[t + 1]), tf.float64
                    )
                    ms_return_target = tf.where(
                        tf.math.is_finite(ms_return_t[t + 1]),
                        ms_return_t[t + 1],
                        state_pred["ms_return"],
                    )
                    loss_ms_return = valid_ms_return * tf.square(
                        (state_pred["ms_return"] - ms_return_target) / scale_ms_return
                    )

                    total_loss += (
                        loss_ni
                        + loss_eff_st
                        + loss_curr_lt
                        + loss_ncl
                        + loss_equity
                        + loss_interest
                        + loss_ms_return
                    )
                    total_loss_ni += loss_ni
                    total_loss_curr_lt += loss_curr_lt
                    total_loss_ncl += loss_ncl
                    total_loss_equity += loss_equity
                    total_loss_interest += loss_interest
                    total_loss_ms_return += loss_ms_return

            grads = tape.gradient(total_loss, vars_to_train)
            if clip_norm is not None:
                grads = [
                    None if g is None else tf.clip_by_norm(g, clip_norm) for g in grads
                ]
            optimizer.apply_gradients(zip(grads, vars_to_train))

            return tf.stack(
                [
                    total_loss,
                    total_loss_ni,
                    total_loss_interest,
                    total_loss_ms_return,
                    total_loss_curr_lt,
                    total_loss_ncl,
                    total_loss_equity,
                ]
            )

        # Index mapping for the stacked loss tensor
        _L_TOTAL, _L_NI, _L_INT, _L_MSR, _L_CLT, _L_NCL, _L_EQ = range(7)

        for i in range(epochs):
            loss_stack = _compiled_train_step()

            if i % plot_every == 0:
                v = loss_stack.numpy()
                structural_history["epochs"].append(i)
                structural_history["loss_total"].append(v[_L_TOTAL])
                structural_history["loss_ni"].append(v[_L_NI])
                structural_history["loss_interest"].append(v[_L_INT])
                structural_history["loss_ms_return"].append(v[_L_MSR])
                structural_history["loss_curr_lt"].append(v[_L_CLT])
                structural_history["loss_ncl"].append(v[_L_NCL])
                structural_history["loss_equity"].append(v[_L_EQ])

            if i % 1000 == 0:
                print(f"Epoch {i}: Structural Loss={loss_stack[_L_TOTAL].numpy():.4e}")

        print("Structural Training Complete.")
        model.print_structural_summary(num_transitions + 1)
        print("\nFinal structural losses:")
        for key in structural_history:
            if key.startswith("loss_") and structural_history[key]:
                label = key.replace("loss_", "")
                print(f"  {label:15s} {structural_history[key][-1]:.4e}")
        print("-" * 50)

        plot_structural_diagnostics(structural_history, show_plot)
