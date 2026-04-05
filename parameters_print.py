path = "trained_parameters_dont_include_tax_anomalies.npz"
import os
import numpy as np  # Only used for loading parameters
import tensorflow as tf
from financial_forecast.data.aapl.financial_statements import get_financial_statements


def sigmoid(x):
    return 1.0 / (1.0 + tf.exp(-x))


if not os.path.exists(path):
    raise FileNotFoundError(f"Parameter file not found: {path}")
data = np.load(path)  # np.load for .npz file I/O

historical_data = get_financial_statements()
historical_sales = historical_data["sales"]

asset_growth = data["asset_growth"]
asset_maintain = data["asset_maintain"]
depreciation_rate = data["depreciation_rate"]
advance_payments_sales_pct = data["advance_payments_sales_pct"]
advance_payments_purchases_pct = data["advance_payments_purchases_pct"]
account_receivables_pct = data["account_receivables_pct"]
account_payables_pct = data["account_payables_pct"]
inventory_pct = data["inventory_pct"]
tl_baseline = data["tl_baseline"]
tl_alpha = data["tl_alpha"]
tl_beta = data["tl_beta"]
cash_alpha = data["cash_alpha"]
cash_beta = data["cash_beta"]
income_tax_pct = data["income_tax_pct"]
dividend_payout_ratio_pct = data["dividend_payout_ratio_pct"]
dividend_adjustment_speed = data.get("dividend_adjustment_speed", "N/A (old file)")
sb_baseline = data["sb_baseline"]
sb_ratio = data["sb_ratio"]
st_debt_baseline = data["st_debt_baseline"]
st_debt_pct = data["st_debt_pct"]
cost_ratio_alpha = data["cost_ratio_alpha"]
cost_ratio_beta = data["cost_ratio_beta"]
q_var_opex_loc = data["q_var_opex_loc"]
q_var_opex_scale = data["q_var_opex_scale"]
q_base_opex_loc = data["q_base_opex_loc"]
q_base_opex_scale = data["q_base_opex_scale"]
noise_sigma = data["noise_sigma"]
avg_short_term_interest_pct = data["avg_short_term_interest_pct"]
avg_long_term_interest_pct = data["avg_long_term_interest_pct"]
avg_maturity_years = data["avg_maturity_years"]
market_securities_return_pct = data["market_securities_return_pct"]
ef_alpha = data["ef_alpha"]
ef_beta = data["ef_beta"]
amount_scale = 1.0e11


print("-" * 50)
print("Simple parameters:")
print(f"Final %AG: {asset_growth:.5f}")
print(f"Final %AM: {asset_maintain:.5f}")
print(f"Final %Depr: {depreciation_rate:.5f}")
print(f"Final %AdvPS: {advance_payments_sales_pct:.5f}")
print(f"Final %AdvPP: {advance_payments_purchases_pct:.5f}")
print(f"Final %AR: {account_receivables_pct:.5f}")
print(f"Final %AP: {account_payables_pct:.5f}")
print(f"Final %Inv: {inventory_pct:.5f}")
print(
    f"Total Liquidity (baseline + logit-linear): baseline={tl_baseline:.4f}, "
    f"alpha={tl_alpha:.4f}, "
    f"beta={tl_beta:.6f}"
)
print(
    f"  => %TL at t=0: {sigmoid(tl_alpha):.4f}, "
    f"%TL at t={len(historical_sales)-1}: "
    f"{sigmoid(tl_alpha + tl_beta * (len(historical_sales)-1)):.4f}"
)
print(
    f"Cash % of Liquidity (logit-linear): alpha={cash_alpha:.4f}, "
    f"beta={cash_beta:.6f}"
)
print(
    f"  => %Cash at t=0: {sigmoid(cash_alpha):.4f}, "
    f"%Cash at t={len(historical_sales)-1}: "
    f"{sigmoid(cash_alpha + cash_beta * (len(historical_sales)-1)):.4f}"
)
print(f"Final %IT: {income_tax_pct:.5f}")
print(f"Final %PR: {dividend_payout_ratio_pct:.5f}")
print(f"Final DivAdjSpeed (α): {dividend_adjustment_speed:.5f}")
print(
    f"Stock Buyback (baseline + ratio*depr): baseline={sb_baseline:.4f}, "
    f"ratio={sb_ratio:.6f}"
)
print(
    f"Effective ST Debt (baseline + % of sales): "
    f"baseline={st_debt_baseline:.4f}, pct={st_debt_pct:.5f}"
)

print(
    f"Cost Ratio (logit-linear): alpha={cost_ratio_alpha:.4f}, "
    f"beta={cost_ratio_beta:.4f}"
)
print(
    f"  => CR at t=0: {sigmoid(cost_ratio_alpha):.4f}, "
    f"CR at t={len(historical_sales)-1}: "
    f"{sigmoid(cost_ratio_alpha + cost_ratio_beta * (len(historical_sales)-1)):.4f}"
)
print(
    f"Bayesian OpEx Variable %: Mean={q_var_opex_loc:.4f}, Std={q_var_opex_scale:.4f}"
)
print(
    "Bayesian OpEx Baseline (USD):   "
    f"Mean={(q_base_opex_loc * amount_scale):.2e}, "
    f"Std={(q_base_opex_scale * amount_scale):.2e}"
)
print("OpEx aleatoric uncertainty (USD): " f"{(noise_sigma * amount_scale):.2e}")

print("-" * 50)
print("Structural parameters:")
print(f"Final %AvgSTInt: {avg_short_term_interest_pct:.5f}")
print(f"Final %AvgLTInt: {avg_long_term_interest_pct:.5f}")
print(f"Final AvgM: {avg_maturity_years:.5f}")
print(f"Final %MSReturn: {market_securities_return_pct:.5f}")
print(
    f"Equity Financing % (logit-linear): alpha={ef_alpha:.4f}, " f"beta={ef_beta:.6f}"
)
print(
    f"  => %EF at t=0: {sigmoid(ef_alpha):.4f}, "
    f"%EF at t={len(historical_sales)-1}: "
    f"{sigmoid(ef_alpha + ef_beta * (len(historical_sales)-1)):.4f}"
)
print("-" * 50)
