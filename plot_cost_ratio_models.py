"""Compare cost ratio trend models: Linear vs Logit-Linear on Apple FY2018-FY2025 data."""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

# --- Data ---
years = np.arange(2018, 2026)
t = np.arange(len(years), dtype=np.float64)  # 0, 1, ..., 7

sales_hist = np.array(
    [2.65595e11, 2.60174e11, 2.74515e11, 3.65817e11,
     3.94328e11, 3.83285e11, 3.91035e11, 4.16161e11],
    dtype=np.float64,
)
cogs_hist = np.array(
    [1.52853e11, 1.49235e11, 1.58503e11, 2.01697e11,
     2.12442e11, 2.02618e11, 1.98907e11, 2.09262e11],
    dtype=np.float64,
)
cost_ratio = cogs_hist / sales_hist

# =============================================================================
# Model 1: Naive Linear Trend  (cost_ratio_t = α + β·t)
# =============================================================================
coeffs_lin = np.polyfit(t, cost_ratio, 1)
lin_fit = np.polyval(coeffs_lin, t)
lin_resid = cost_ratio - lin_fit
lin_sigma = np.std(lin_resid, ddof=2)
lin_ss_res = np.sum(lin_resid**2)
lin_ss_tot = np.sum((cost_ratio - cost_ratio.mean())**2)
lin_r2 = 1 - lin_ss_res / lin_ss_tot

print("=" * 60)
print("Model 1: Naive Linear Trend  CR_t = α + β·t")
print(f"  α = {coeffs_lin[1]:.6f}  ({coeffs_lin[1]*100:.2f}%)")
print(f"  β = {coeffs_lin[0]:+.6f}  ({coeffs_lin[0]*100:+.2f}pp/yr)")
print(f"  σ_resid = {lin_sigma:.4f}  ({lin_sigma*100:.2f}pp)")
print(f"  R² = {lin_r2:.4f}")

# =============================================================================
# Model 2: Logit-Linear Trend  logit(CR_t) = α + β·t  ⟹  CR_t = σ(α + β·t)
# =============================================================================
logit_cr = np.log(cost_ratio / (1 - cost_ratio))
coeffs_logit = np.polyfit(t, logit_cr, 1)
logit_fit = np.polyval(coeffs_logit, t)
logit_resid = logit_cr - logit_fit
logit_sigma = np.std(logit_resid, ddof=2)
logit_ss_res = np.sum(logit_resid**2)
logit_ss_tot = np.sum((logit_cr - logit_cr.mean())**2)
logit_r2 = 1 - logit_ss_res / logit_ss_tot

# Back-transform to cost ratio space
sigmoid = lambda z: 1 / (1 + np.exp(-z))
logit_fit_cr = sigmoid(logit_fit)
logit_resid_cr = cost_ratio - logit_fit_cr

print(f"\nModel 2: Logit-Linear Trend  logit(CR_t) = α + β·t")
print(f"  α = {coeffs_logit[1]:.6f}")
print(f"  β = {coeffs_logit[0]:+.6f}")
print(f"  σ_resid (logit space) = {logit_sigma:.4f}")
print(f"  σ_resid (CR space)    = {np.std(logit_resid_cr, ddof=2):.4f}  ({np.std(logit_resid_cr, ddof=2)*100:.2f}pp)")
print(f"  R² (logit space)      = {logit_r2:.4f}")
print(f"  R² (CR space)         = {1 - np.sum(logit_resid_cr**2)/lin_ss_tot:.4f}")

# --- Print logit-transformed data ---
print(f"\n{'FY':<6} {'CR':>8} {'logit(CR)':>10} {'Fitted logit':>13} {'Fitted CR':>10} {'Resid (logit)':>14}")
print("-" * 65)
for i, y in enumerate(years):
    print(f"{y:<6} {cost_ratio[i]:>8.4f} {logit_cr[i]:>10.4f} {logit_fit[i]:>13.4f} {logit_fit_cr[i]:>10.4f} {logit_resid[i]:>14.4f}")

# =============================================================================
# Forecast 30 years ahead
# =============================================================================
n_fwd = 30
t_forecast = np.arange(len(years) + n_fwd, dtype=np.float64)
years_forecast = np.arange(2018, 2018 + len(t_forecast))

# Linear forecast
lin_forecast = np.polyval(coeffs_lin, t_forecast)
horizon = np.maximum(t_forecast - t[-1], 0)
lin_1sigma = lin_sigma * np.sqrt(1 + horizon)

# Logit-linear forecast with uncertainty in logit space
logit_forecast_z = np.polyval(coeffs_logit, t_forecast)
logit_1sigma_z = logit_sigma * np.sqrt(1 + horizon)
logit_forecast_cr = sigmoid(logit_forecast_z)
logit_upper_cr = sigmoid(logit_forecast_z + logit_1sigma_z)
logit_lower_cr = sigmoid(logit_forecast_z - logit_1sigma_z)
logit_upper2_cr = sigmoid(logit_forecast_z + 2 * logit_1sigma_z)
logit_lower2_cr = sigmoid(logit_forecast_z - 2 * logit_1sigma_z)

print("\n" + "=" * 60)
print("Forecast comparison:")
print(f"{'Year':<6} {'Linear':>10} {'Logit-Lin':>10} {'Logit ±1σ':>18}")
print("-" * 50)
for i, y in enumerate(years_forecast):
    if y >= 2025 and (y <= 2035 or y % 5 == 0):
        print(f"{y:<6} {lin_forecast[i]*100:>9.2f}% {logit_forecast_cr[i]*100:>9.2f}%  [{logit_lower_cr[i]*100:.1f}%, {logit_upper_cr[i]*100:.1f}%]")

# Check: when does naive linear go to zero?
t_zero = -coeffs_lin[1] / coeffs_lin[0]
year_zero = 2018 + t_zero
print(f"\n⚠  Naive linear hits 0% at t={t_zero:.1f} → FY{year_zero:.0f}")
print(f"   Logit-linear at same point: {sigmoid(np.polyval(coeffs_logit, t_zero))*100:.2f}%")

# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# --- Top-left: Logit-transformed data + fit ---
ax1 = axes[0, 0]
ax1.scatter(years, logit_cr, s=100, c="black", zorder=5, label="logit(CR) data")
ax1.plot(years, logit_fit, "o-", color="tab:red", linewidth=2, markersize=6,
         label=f"OLS fit: {coeffs_logit[1]:.3f} {coeffs_logit[0]:+.4f}·t\nR² = {logit_r2:.4f}")
# Show ±1σ band on historical
ax1.fill_between(years, logit_fit - logit_sigma, logit_fit + logit_sigma,
                 color="tab:red", alpha=0.15, label=f"±1σ (σ={logit_sigma:.3f})")
ax1.set_xlabel("Fiscal Year", fontsize=11)
ax1.set_ylabel("logit(Cost Ratio) = log(CR / (1−CR))", fontsize=11)
ax1.set_title("Logit-Transformed Cost Ratio", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(years)
for i, y in enumerate(years):
    ax1.annotate(f"{logit_cr[i]:.3f}", (y, logit_cr[i]),
                 textcoords="offset points", xytext=(6, 8), fontsize=8)

# --- Top-right: Both fits on historical data (CR space) ---
ax2 = axes[0, 1]
ax2.scatter(years, cost_ratio * 100, s=100, c="black", zorder=5, label="Historical")
ax2.plot(years, lin_fit * 100, "o--", color="tab:blue", linewidth=2, markersize=5,
         label=f"Linear (R²={lin_r2:.3f})")
ax2.plot(years, logit_fit_cr * 100, "s-", color="tab:red", linewidth=2, markersize=5,
         label=f"Logit-Linear (R²={1 - np.sum(logit_resid_cr**2)/lin_ss_tot:.3f})")
ax2.set_xlabel("Fiscal Year", fontsize=11)
ax2.set_ylabel("Cost Ratio = COGS/Sales (%)", fontsize=11)
ax2.set_title("Model Fits Compared (CR Space)", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(years)

# --- Bottom-left: Long-range forecast showing the key difference ---
ax3 = axes[1, 0]
ax3.scatter(years, cost_ratio * 100, s=80, c="black", zorder=5, label="Historical")

# Linear
ax3.plot(years_forecast, lin_forecast * 100, "-", color="tab:blue", linewidth=2, label="Linear", alpha=0.8)
ax3.fill_between(years_forecast, (lin_forecast - lin_1sigma) * 100,
                 (lin_forecast + lin_1sigma) * 100, color="tab:blue", alpha=0.1)

# Logit-linear
ax3.plot(years_forecast, logit_forecast_cr * 100, "-", color="tab:red", linewidth=2.5, label="Logit-Linear")
ax3.fill_between(years_forecast, logit_lower_cr * 100, logit_upper_cr * 100,
                 color="tab:red", alpha=0.15, label="±1σ")
ax3.fill_between(years_forecast, logit_lower2_cr * 100, logit_upper2_cr * 100,
                 color="tab:red", alpha=0.07, label="±2σ")

ax3.axhline(0, color="black", linewidth=1, linestyle="-")
ax3.axhline(100, color="black", linewidth=1, linestyle="-")
ax3.axvline(2025.5, color="gray", linewidth=1, linestyle="--", alpha=0.5)
ax3.text(2026, 90, "Forecast →", fontsize=10, color="gray")
ax3.set_xlabel("Fiscal Year", fontsize=11)
ax3.set_ylabel("Cost Ratio = COGS/Sales (%)", fontsize=11)
ax3.set_title("30-Year Forecast: Linear Goes Negative!", fontsize=13, fontweight="bold")
ax3.legend(fontsize=9, loc="upper right")
ax3.grid(True, alpha=0.3)
ax3.set_ylim([-15, 105])

# Annotate the crossing
ax3.annotate(f"Linear hits 0%\naround FY{year_zero:.0f}",
             xy=(year_zero, 0), xytext=(year_zero - 5, -10),
             fontsize=9, color="tab:blue", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:blue"))

# --- Bottom-right: Zoomed forecast (next 10 years) with gross margin ---
ax4 = axes[1, 1]
mask_10yr = years_forecast <= 2035
ax4.scatter(years, cost_ratio * 100, s=80, c="black", zorder=5, label="Historical CR")

ax4.plot(years_forecast[mask_10yr], logit_forecast_cr[mask_10yr] * 100, "-", color="tab:red",
         linewidth=2.5, label="Logit-Linear CR forecast")
ax4.fill_between(years_forecast[mask_10yr],
                 logit_lower_cr[mask_10yr] * 100,
                 logit_upper_cr[mask_10yr] * 100,
                 color="tab:red", alpha=0.15)
ax4.fill_between(years_forecast[mask_10yr],
                 logit_lower2_cr[mask_10yr] * 100,
                 logit_upper2_cr[mask_10yr] * 100,
                 color="tab:red", alpha=0.07)

# Also show implied Gross Margin on twin axis
ax4b = ax4.twinx()
ax4b.plot(years_forecast[mask_10yr], (1 - logit_forecast_cr[mask_10yr]) * 100, "-",
          color="tab:green", linewidth=2, label="Implied Gross Margin")
ax4b.scatter(years, (1 - cost_ratio) * 100, s=60, c="tab:green", edgecolors="black",
             zorder=5, marker="D")
ax4b.set_ylabel("Gross Margin (%)", fontsize=11, color="tab:green")
ax4b.tick_params(axis="y", labelcolor="tab:green")

ax4.axvline(2025.5, color="gray", linewidth=1, linestyle="--", alpha=0.5)
ax4.set_xlabel("Fiscal Year", fontsize=11)
ax4.set_ylabel("Cost Ratio (%)", fontsize=11, color="tab:red")
ax4.tick_params(axis="y", labelcolor="tab:red")
ax4.set_title("10-Year Logit-Linear Forecast", fontsize=13, fontweight="bold")
ax4.grid(True, alpha=0.3)

# Combined legend
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4b.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

for i, y in enumerate(years_forecast):
    if mask_10yr[i] and y >= 2025 and y % 2 == 1:
        ax4.annotate(f"{logit_forecast_cr[i]*100:.1f}%",
                     (y, logit_forecast_cr[i] * 100),
                     textcoords="offset points", xytext=(0, -14),
                     ha="center", fontsize=8, color="tab:red")

plt.tight_layout()
plt.savefig("cost_ratio_models.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to cost_ratio_models.png")
