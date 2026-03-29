"""Training diagnostics and plotting utilities.

All functions accept pre-computed history dicts and are model-agnostic.
"""

from datetime import datetime
from typing import Dict, Optional

import matplotlib.pyplot as plt
import tensorflow as tf

from financial_forecast.training.io_utils import get_training_results_path


def plot_vi_diagnostics(
    vi_history: Dict, amount_scale: float, show_plot: bool = False
) -> None:
    """Plot variational inference parameter convergence over training epochs.

    Args:
        vi_history: Dict with keys ``epochs``, ``loss_vi``,
            ``q_var_opex_loc``, ``q_var_opex_scale``, ``q_base_opex_loc``,
            ``q_base_opex_scale``, ``noise_sigma``.
        amount_scale: Multiplier to convert scaled values to USD.
        show_plot: Whether to display the plot interactively.
    """
    if not vi_history["epochs"]:
        return

    epochs_hist = tf.constant(vi_history["epochs"], dtype=tf.float64)
    fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    axs[0].plot(epochs_hist, vi_history["q_var_opex_loc"], label="q_var_opex_loc")
    axs[0].plot(epochs_hist, vi_history["q_var_opex_scale"], label="q_var_opex_scale")
    axs[0].set_ylabel("Variable OpEx %")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(
        epochs_hist,
        tf.constant(vi_history["q_base_opex_loc"], dtype=tf.float64) * amount_scale,
        label="q_base_opex_loc (USD)",
    )
    axs[1].plot(
        epochs_hist,
        tf.constant(vi_history["q_base_opex_scale"], dtype=tf.float64) * amount_scale,
        label="q_base_opex_scale (USD)",
    )
    axs[1].set_ylabel("Baseline OpEx (USD)")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(
        epochs_hist,
        tf.constant(vi_history["noise_sigma"], dtype=tf.float64) * amount_scale,
        label="noise_sigma (USD)",
    )
    axs[2].set_ylabel("Noise Sigma (USD)")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(epochs_hist, vi_history["loss_vi"], label="Loss_VI")
    axs[3].set_xlabel("Epoch")
    axs[3].set_ylabel("Loss_VI")
    axs[3].legend()
    axs[3].grid(True, alpha=0.3)

    fig.suptitle("Variational Inference Parameters and Loss Over Epochs")
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = get_training_results_path(f"vi_training_diagnostics_{timestamp}.png")
    plt.savefig(plot_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_simple_policy_diagnostics(
    simple_history: Dict,
    model: tf.Module,
    time_indices: tf.Tensor,
    logit_cr_hist: tf.Tensor,
    historical_years: Optional[tf.Tensor],
    n_years: int,
    show_plot: bool = False,
) -> None:
    """Plot simple policy parameter training diagnostics.

    Args:
        simple_history: Dict of per-epoch loss values for each policy
            component.
        model: The trained model (used to read final cost-ratio params).
        time_indices: Time index tensor (``year - base_year``).
        logit_cr_hist: Historical logit cost-ratio targets.
        historical_years: Year tensor for x-axis labelling.
        n_years: Number of historical years.
        show_plot: Whether to display the plot interactively.
    """
    if not simple_history["epochs"]:
        return

    epochs_hist = tf.constant(simple_history["epochs"], dtype=tf.float64)
    fig, axs = plt.subplots(4, 1, figsize=(10, 15))

    # Panel 1: Total loss
    axs[0].plot(
        epochs_hist,
        simple_history["loss_total"],
        label="Total Loss",
        color="black",
        linewidth=2,
    )
    axs[0].set_ylabel("Total Loss")
    axs[0].set_yscale("log")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Panel 2: Individual ratio losses
    ratio_losses = [
        ("loss_ar", "%AR"),
        ("loss_ap", "%AP"),
        ("loss_inv", "%Inv"),
        ("loss_tl", "%TL"),
        ("loss_cash", "%Cash"),
        ("loss_adv_ps", "%AdvPS"),
        ("loss_adv_pp", "%AdvPP"),
        ("loss_tax", "%IT"),
        ("loss_div", "%PR"),
        ("loss_bb", "%BB"),
        ("loss_eff_st_debt", "%EffSTDebt"),
    ]
    if model.opex_module.is_stochastic != True:
        ratio_losses.append(("loss_opex", "OpEx"))

    for key, label in ratio_losses:
        axs[1].plot(epochs_hist, simple_history[key], label=label)
    axs[1].set_ylabel("Loss (MSE)")
    axs[1].set_yscale("log")
    axs[1].legend(ncol=3, fontsize=8)
    axs[1].grid(True, alpha=0.3)

    # Panel 3: Asset-related losses + cost ratio + prior
    structural_losses = [
        ("loss_growth", "%AG (Growth)"),
        ("loss_depr", "%Depr"),
        ("loss_cost_ratio", "Cost Ratio"),
        ("loss_prior_am", "Prior AM"),
    ]
    for key, label in structural_losses:
        axs[2].plot(epochs_hist, simple_history[key], label=label)
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Loss (MSE)")
    axs[2].set_yscale("log")
    axs[2].legend(fontsize=8)
    axs[2].grid(True, alpha=0.3)

    # Panel 4: Fitted logit(CR) vs historical logit(CR)
    if historical_years is not None:
        cr_x = tf.constant(historical_years, dtype=tf.float64)
        axs[3].set_xlabel("Year")
    else:
        cr_x = tf.cast(tf.range(n_years), dtype=tf.float64)
        axs[3].set_xlabel("Time Index")
    pp = model.balance_sheet.purchases_policy
    if hasattr(pp, "cost_ratio_alpha"):
        final_logit_cr_pred = pp.cost_ratio_alpha + pp.cost_ratio_beta * time_indices
    else:
        final_logit_cr_pred = tf.zeros_like(time_indices)
    axs[3].plot(cr_x, logit_cr_hist.numpy(), marker="o", label="logit_cr_hist")
    axs[3].plot(
        cr_x,
        final_logit_cr_pred.numpy(),
        marker="x",
        linestyle="--",
        label="logit_cr_pred",
    )
    axs[3].set_ylabel("logit(CR)")
    axs[3].legend(fontsize=8)
    axs[3].grid(True, alpha=0.3)

    fig.suptitle("Simple Parameters Training Diagnostics")
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = get_training_results_path(
        f"simple_training_diagnostics_{timestamp}.png"
    )
    plt.savefig(plot_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_structural_diagnostics(
    structural_history: Dict, show_plot: bool = False
) -> None:
    """Plot structural parameter training diagnostics.

    Args:
        structural_history: Dict of per-epoch loss values for each
            structural component.
        show_plot: Whether to display the plot interactively.
    """
    if not structural_history["epochs"]:
        return

    epochs_hist = tf.constant(structural_history["epochs"], dtype=tf.float64)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel 1: Total loss
    axs[0].plot(
        epochs_hist,
        structural_history["loss_total"],
        label="Total Loss",
        color="black",
        linewidth=2,
    )
    axs[0].set_ylabel("Total Loss")
    axs[0].set_yscale("log")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Panel 2: Component losses
    component_losses = [
        ("loss_ni", "Net Income"),
        ("loss_interest", "Interest Payment"),
        ("loss_ms_return", "Return on Market Securities Investment"),
        ("loss_curr_lt", "Current LT Debt"),
        ("loss_ncl", "Non-Current Liabilities"),
        ("loss_equity", "Equity"),
    ]
    for key, label in component_losses:
        axs[1].plot(epochs_hist, structural_history[key], label=label)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss (SSE)")
    axs[1].set_yscale("log")
    axs[1].legend(fontsize=9)
    axs[1].grid(True, alpha=0.3)

    fig.suptitle("Structural Parameters Training Diagnostics")
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = get_training_results_path(
        f"structural_training_diagnostics_{timestamp}.png"
    )
    plt.savefig(plot_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close()
