"""Plot historical gross margin for Apple (FY2018-FY2025)."""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

# --- Historical Data from Apple ---
# Fiscal years (Apple FY ends in late September)
years = np.arange(2018, 2026)

# Revenues from Income Statement
sales_hist = np.array(
    [
        2.65595e11,
        2.60174e11,
        2.74515e11,
        3.65817e11,
        3.94328e11,
        3.83285e11,
        3.91035e11,
        4.16161e11,
    ],
    dtype=np.float64,
)


# Depreciation from Reconciled Depreciation in Income Statement
depr_hist = np.array(
    [
        10903000000,
        12547000000,
        11056000000,
        11284000000,
        11104000000,
        11519000000,
        11445000000,
        11698000000,
    ],
    dtype=np.float64,
)

# COGS from Cost of Revenue - Depreciation in Income Statement
cogs_hist = np.array(
    [
        1.52853e11,
        1.49235e11,
        1.58503e11,
        2.01697e11,
        2.12442e11,
        2.02618e11,
        1.98907e11,
        2.09262e11,
    ],
    dtype=np.float64,
)

# Full Cost of Revenue = COGS (excl. depr) + Depreciation
cost_of_revenue_hist = cogs_hist + depr_hist

# Gross Margin using model's COGS definition (excl. depreciation)
gross_margin_excl_depr = (sales_hist - cogs_hist) / sales_hist

# Gross Margin using full Cost of Revenue (incl. depreciation)
gross_margin_incl_depr = (sales_hist - cost_of_revenue_hist) / sales_hist

# Cost Ratio (COGS / Sales, excl. depreciation) — this is what the user wants to model
cost_ratio = cost_of_revenue_hist / sales_hist

# --- Print summary ---
print("=" * 70)
print(
    f"{'FY':<6} {'Sales ($B)':>12} {'COGS ($B)':>12} {'CoR ($B)':>12} {'GM (excl D)':>12} {'GM (incl D)':>12} {'Cost Ratio':>12}"
)
print("=" * 70)
for i, y in enumerate(years):
    print(
        f"{y:<6} {sales_hist[i]/1e9:>12.2f} {cogs_hist[i]/1e9:>12.2f} {cost_of_revenue_hist[i]/1e9:>12.2f} "
        f"{gross_margin_excl_depr[i]:>12.2%} {gross_margin_incl_depr[i]:>12.2%} {cost_ratio[i]:>12.4f}"
    )
print("=" * 70)

# Inflation History (annual CPI inflation rate)
inflation_hist = np.array(
    [0.024, 0.018, 0.012, 0.047, 0.08, 0.041, 0.029, 0.027],
    dtype=np.float64,
)

# Cumulative inflation (indexed to FY2018 = 1.0)
cum_inflation = np.cumprod(1 + inflation_hist)

# Year-over-year gross margin change
gm_change = np.diff(gross_margin_excl_depr)

# Correlation
corr_gm_inf = np.corrcoef(gross_margin_excl_depr, inflation_hist)[0, 1]
corr_gm_change_inf = np.corrcoef(gm_change, inflation_hist[1:])[0, 1]

print(f"\nCorrelation (GM level vs Inflation):          {corr_gm_inf:+.4f}")
print(f"Correlation (GM YoY change vs Inflation):     {corr_gm_change_inf:+.4f}")

# --- Sales stats ---
sales_growth = np.diff(sales_hist) / sales_hist[:-1]
print(f"\nSales YoY growth rates:")
for i in range(len(sales_growth)):
    print(f"  FY{years[i]}->{years[i+1]}: {sales_growth[i]:+.2%}")
print(f"  CAGR (FY2018-FY2025): {(sales_hist[-1]/sales_hist[0])**(1/7) - 1:.2%}")

# --- Plot ---
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.35)

# Row 0, col 0: Sales vs Year
ax0 = fig.add_subplot(gs[0, 0])
ax0.bar(
    years, sales_hist / 1e9, color="tab:green", edgecolor="black", alpha=0.85, width=0.6
)
ax0.plot(
    years,
    sales_hist / 1e9,
    "o-",
    color="darkgreen",
    linewidth=2,
    markersize=6,
    zorder=5,
)
ax0.set_xlabel("Fiscal Year", fontsize=11)
ax0.set_ylabel("Revenue ($B)", fontsize=11)
ax0.set_title("Apple Revenue (FY2018-FY2025)", fontsize=13, fontweight="bold")
ax0.set_xticks(years)
ax0.grid(True, alpha=0.3, axis="y")
for i, y in enumerate(years):
    ax0.annotate(
        f"${sales_hist[i]/1e9:.0f}B",
        (y, sales_hist[i] / 1e9),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=8,
        fontweight="bold",
        color="darkgreen",
    )

# Row 0, col 1: Sales YoY Growth
ax0b = fig.add_subplot(gs[0, 1])
colors_growth = ["tab:green" if g >= 0 else "tab:red" for g in sales_growth]
ax0b.bar(
    years[1:],
    sales_growth * 100,
    color=colors_growth,
    edgecolor="black",
    alpha=0.85,
    width=0.6,
)
ax0b.axhline(0, color="black", linewidth=0.8)
ax0b.set_xlabel("Fiscal Year", fontsize=11)
ax0b.set_ylabel("YoY Revenue Growth (%)", fontsize=11)
ax0b.set_title("Apple Revenue Growth (FY2019-FY2025)", fontsize=13, fontweight="bold")
ax0b.set_xticks(years[1:])
ax0b.grid(True, alpha=0.3, axis="y")
for i in range(len(sales_growth)):
    ax0b.annotate(
        f"{sales_growth[i]:+.1%}",
        (years[i + 1], sales_growth[i] * 100),
        textcoords="offset points",
        xytext=(0, 8 if sales_growth[i] >= 0 else -14),
        ha="center",
        fontsize=8,
        fontweight="bold",
    )

# Row 1, col 0: Gross Margin with Inflation overlay
ax1 = fig.add_subplot(gs[1, 0])
color_gm = "tab:blue"
ax1.plot(
    years,
    gross_margin_excl_depr * 100,
    "o-",
    color=color_gm,
    linewidth=2,
    markersize=8,
    label="Gross Margin (excl. depr.)",
)
ax1.set_xlabel("Fiscal Year", fontsize=11)
ax1.set_ylabel("Gross Margin (%)", fontsize=11, color=color_gm)
ax1.tick_params(axis="y", labelcolor=color_gm)
ax1.set_xticks(years)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([40, 52])
for i, y in enumerate(years):
    ax1.annotate(
        f"{gross_margin_excl_depr[i]*100:.1f}%",
        (y, gross_margin_excl_depr[i] * 100),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=8,
        color=color_gm,
    )

# Inflation on twin axis
ax1b = ax1.twinx()
color_inf = "tab:red"
ax1b.plot(
    years,
    inflation_hist * 100,
    "s--",
    color=color_inf,
    linewidth=2,
    markersize=7,
    label="Annual Inflation",
)
ax1b.set_ylabel("Inflation Rate (%)", fontsize=11, color=color_inf)
ax1b.tick_params(axis="y", labelcolor=color_inf)
ax1b.set_ylim([0, 10])
for i, y in enumerate(years):
    ax1b.annotate(
        f"{inflation_hist[i]*100:.1f}%",
        (y, inflation_hist[i] * 100),
        textcoords="offset points",
        xytext=(0, -14),
        ha="center",
        fontsize=8,
        color=color_inf,
    )

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
ax1.set_title(
    f"Gross Margin vs Inflation (corr = {corr_gm_inf:+.3f})",
    fontsize=13,
    fontweight="bold",
)

# Row 1, col 1: Cost Ratio with Inflation overlay
ax2 = fig.add_subplot(gs[1, 1])
color_cr = "tab:purple"
ax2.plot(
    years,
    cost_ratio * 100,
    "o-",
    color=color_cr,
    linewidth=2,
    markersize=8,
    label="Cost Ratio (CoR/Sales)",
)
ax2.set_xlabel("Fiscal Year", fontsize=11)
ax2.set_ylabel("Cost Ratio (%)", fontsize=11, color=color_cr)
ax2.tick_params(axis="y", labelcolor=color_cr)
ax2.set_xticks(years)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([55, 67])

ax2b = ax2.twinx()
ax2b.plot(
    years,
    inflation_hist * 100,
    "s--",
    color=color_inf,
    linewidth=2,
    markersize=7,
    label="Annual Inflation",
)
ax2b.set_ylabel("Inflation Rate (%)", fontsize=11, color=color_inf)
ax2b.tick_params(axis="y", labelcolor=color_inf)
ax2b.set_ylim([0, 10])

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")
corr_cr_inf = np.corrcoef(cost_ratio, inflation_hist)[0, 1]
ax2.set_title(
    f"Cost Ratio vs Inflation (corr = {corr_cr_inf:+.3f})",
    fontsize=13,
    fontweight="bold",
)

# Row 2, col 0: Scatter — GM vs Inflation
ax3 = fig.add_subplot(gs[2, 0])
ax3.scatter(
    inflation_hist * 100,
    gross_margin_excl_depr * 100,
    s=100,
    c=years,
    cmap="viridis",
    edgecolors="black",
    zorder=3,
)
for i, y in enumerate(years):
    ax3.annotate(
        str(y),
        (inflation_hist[i] * 100, gross_margin_excl_depr[i] * 100),
        textcoords="offset points",
        xytext=(8, 4),
        fontsize=9,
    )
# Trend line
z = np.polyfit(inflation_hist * 100, gross_margin_excl_depr * 100, 1)
p = np.poly1d(z)
x_fit = np.linspace(
    inflation_hist.min() * 100 - 0.5, inflation_hist.max() * 100 + 0.5, 50
)
ax3.plot(
    x_fit,
    p(x_fit),
    "--",
    color="gray",
    alpha=0.7,
    label=f"Linear fit (slope={z[0]:.2f})",
)
ax3.set_xlabel("Annual Inflation Rate (%)", fontsize=11)
ax3.set_ylabel("Gross Margin (%)", fontsize=11)
ax3.set_title("Scatter: Gross Margin vs Inflation", fontsize=13, fontweight="bold")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Row 2, col 1: Scatter — YoY GM change vs Inflation
ax4 = fig.add_subplot(gs[2, 1])
ax4.scatter(
    inflation_hist[1:] * 100,
    gm_change * 100,
    s=100,
    c=years[1:],
    cmap="viridis",
    edgecolors="black",
    zorder=3,
)
for i in range(len(gm_change)):
    ax4.annotate(
        f"{years[i]}->{years[i+1]}",
        (inflation_hist[i + 1] * 100, gm_change[i] * 100),
        textcoords="offset points",
        xytext=(8, 4),
        fontsize=8,
    )
ax4.axhline(0, color="black", linewidth=0.8, linestyle="-")
z2 = np.polyfit(inflation_hist[1:] * 100, gm_change * 100, 1)
p2 = np.poly1d(z2)
x_fit2 = np.linspace(
    inflation_hist[1:].min() * 100 - 0.5, inflation_hist[1:].max() * 100 + 0.5, 50
)
ax4.plot(
    x_fit2,
    p2(x_fit2),
    "--",
    color="gray",
    alpha=0.7,
    label=f"Linear fit (slope={z2[0]:.2f})",
)
ax4.set_xlabel("Annual Inflation Rate (%)", fontsize=11)
ax4.set_ylabel("YoY Gross Margin Change (pp)", fontsize=11)
ax4.set_title(
    f"Scatter: GM Change vs Inflation (corr = {corr_gm_change_inf:+.3f})",
    fontsize=13,
    fontweight="bold",
)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# plt.tight_layout()  # using gridspec spacing instead
plt.savefig("gross_margin_plot.png", dpi=150, bbox_inches="tight")
# plt.show()  # non-interactive
print("\nPlot saved to gross_margin_plot.png")
