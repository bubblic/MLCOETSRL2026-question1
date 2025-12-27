import pandas as pd
import yfinance as yf

# Specify the stock you want to analyze (e.g., Walmart)
# STOCKTICKER = yf.Ticker("WMT")
# STOCKTICKER = yf.Ticker("AAPL")
STOCKTICKER = yf.Ticker("GM")

# Get historical price data
hist = STOCKTICKER.history(period="max")

# Print the first few rows of the data
print(hist.head())

# Save the data to a CSV file
hist.to_csv(f"{STOCKTICKER.ticker}_historical_prices.csv")

# Get financial statements
income_stmt = STOCKTICKER.financials
balance_sheet = STOCKTICKER.balance_sheet
cash_flow = STOCKTICKER.cashflow

# Print the financial statements
print(income_stmt.head())
print(balance_sheet.head())
print(cash_flow.head())

# Save the financial statements to a CSV file
income_stmt.to_csv(f"{STOCKTICKER.ticker}_income_stmt.csv")
balance_sheet.to_csv(f"{STOCKTICKER.ticker}_balance_sheet.csv")
cash_flow.to_csv(f"{STOCKTICKER.ticker}_cash_flow.csv")
