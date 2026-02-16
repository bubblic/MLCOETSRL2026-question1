path = "trained_parameters.npz"
import os
import numpy as np

if not os.path.exists(path):
    raise FileNotFoundError(f"Parameter file not found: {path}")
data = np.load(path)
asset_growth = data["asset_growth"]

print("asset_growth: ", data["asset_growth"])
depreciation_rate = data["depreciation_rate"]
advance_payments_sales_pct = data["advance_payments_sales_pct"]
advance_payments_purchases_pct = data["advance_payments_purchases_pct"]
account_receivables_pct = data["account_receivables_pct"]
account_payables_pct = data["account_payables_pct"]
inventory_pct = data["inventory_pct"]
tl_alpha = data["tl_alpha"]
tl_beta = data["tl_beta"]
cash_alpha = data["cash_alpha"]
cash_beta = data["cash_beta"]
income_tax_pct = data["income_tax_pct"]
dividend_payout_ratio_pct = data["dividend_payout_ratio_pct"]
sb_alpha = data["sb_alpha"]
sb_beta = data["sb_beta"]
q_var_opex_loc = data["q_var_opex_loc"]
q_var_opex_scale = data["q_var_opex_scale"]
q_base_opex_loc = data["q_base_opex_loc"]
q_base_opex_scale = data["q_base_opex_scale"]
noise_sigma = data["noise_sigma"]
sales_offset = data["sales_offset"]
avg_short_term_interest_pct = data["avg_short_term_interest_pct"]
avg_long_term_interest_pct = data["avg_long_term_interest_pct"]
avg_maturity_years = data["avg_maturity_years"]
market_securities_return_pct = data["market_securities_return_pct"]
ef_alpha = data["ef_alpha"]
ef_beta = data["ef_beta"]
