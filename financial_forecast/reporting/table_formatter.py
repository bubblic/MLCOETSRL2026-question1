"""Abstract and concrete table formatters for forecast reporting.

Provides a base class :class:`TableFormatter` that defines the interface
for converting historical and forecast data into string tables, and a
concrete :class:`MarkdownTableFormatter` implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import tensorflow as tf

from financial_forecast.data.loader import HistoricalDataLoader


class TableFormatter(ABC):
    """Abstract base class for formatting financial data into tables.

    Subclasses implement :meth:`format_historical` and
    :meth:`format_forecast` to produce tables in a specific format
    (Markdown, HTML, LaTeX, etc.).
    """

    @abstractmethod
    def format_historical(self, data: HistoricalDataLoader) -> str:
        """Format historical financial data into a table string.

        Args:
            data: Loaded historical data for the company.

        Returns:
            Formatted table as a string.
        """

    @abstractmethod
    def format_forecast(
        self,
        trajectories: Dict[str, tf.Tensor],
        forecast_years: tf.Tensor,
        amount_scale: float,
    ) -> str:
        """Format Monte Carlo forecast trajectories into a table string.

        Args:
            trajectories: Dict mapping metric names to
                ``[n_samples, n_years]`` tensors.
            forecast_years: 1-D tensor of calendar years.
            amount_scale: Scalar to convert internal units to USD.

        Returns:
            Formatted table as a string.
        """


class MarkdownTableFormatter(TableFormatter):
    """Formats financial data as GitHub-flavored Markdown tables."""

    @staticmethod
    def _dollar(value: float) -> str:
        return f"${value:,.0f}"

    def format_historical(self, data: HistoricalDataLoader) -> str:
        fs = data.financial_statements
        years = [int(y) for y in fs["years"].numpy()]

        header = "| Line Item | " + " | ".join(f"FY{y}" for y in years) + " |"
        sep = "| --- | " + " | ".join("---:" for _ in years) + " |"

        def row(label, tensor):
            vals = " | ".join(self._dollar(float(v)) for v in tensor.numpy())
            return f"| {label} | {vals} |"

        lines = [
            "HISTORICAL FINANCIAL DATA (USD)",
            "",
            "Balance Sheet:",
            header,
            sep,
            row("Non-Current Assets", fs["nca"]),
            row("Advance Payments (Purchases)", fs["advance_payments_purchases"]),
            row("Accounts Receivable", fs["accounts_receivable"]),
            row("Inventory", fs["inventory"]),
            row("Cash", fs["cash"]),
            row("Investment in Market Securities", fs["ims"]),
            row("Accounts Payable", fs["accounts_payable"]),
            row("Advance Payments (Sales)", fs["advance_payments_sales"]),
            row("Non-Current Liabilities", fs["non_current_liabilities"]),
            row("Equity", fs["equity"]),
            "",
            "Income Statement:",
            header,
            sep,
            row("Sales (Revenue)", fs["sales"]),
            row("COGS", fs["cogs"]),
            row("Depreciation", fs["depreciation"]),
            row("OpEx", fs["opex"]),
            row("Income Tax", fs["tax"]),
            row("Net Income", fs["net_income"]),
            "",
            "Cash Flow:",
            header,
            sep,
            row("Dividends", fs["dividends"]),
            row("Stock Buyback", fs["stock_buyback"]),
        ]
        return "\n".join(lines)

    def format_forecast(
        self,
        trajectories: Dict[str, tf.Tensor],
        forecast_years: tf.Tensor,
        amount_scale: float,
    ) -> str:
        n_years = trajectories["total_assets"].shape[1]
        year_labels = [f"FY{int(forecast_years[i])}" for i in range(n_years)]
        scale = amount_scale

        header = "| Line Item | " + " | ".join(year_labels) + " |"
        sep = "| --- | " + " | ".join("---:" for _ in year_labels) + " |"
        blank_cells = " | ".join([""] * n_years)

        def row(label, key):
            mean = tf.reduce_mean(trajectories[key], axis=0)
            vals = " | ".join(self._dollar(float(v) * scale) for v in mean)
            return f"| {label} | {vals} |"

        def computed_row(label, tensor):
            vals = " | ".join(self._dollar(float(v) * scale) for v in tensor)
            return f"| {label} | {vals} |"

        mean_total_assets = tf.reduce_mean(trajectories["total_assets"], axis=0)
        mean_total_liab = (
            tf.reduce_mean(trajectories["accounts_payable"], axis=0)
            + tf.reduce_mean(trajectories["advance_payments_sales"], axis=0)
            + tf.reduce_mean(trajectories["effective_st_debt"], axis=0)
            + tf.reduce_mean(trajectories["current_lt_debt"], axis=0)
            + tf.reduce_mean(trajectories["non_current_liabilities"], axis=0)
        )
        mean_equity = tf.reduce_mean(trajectories["equity"], axis=0)
        mean_total_liab_equity = mean_total_liab + mean_equity

        lines = [
            f"FORECAST BALANCE SHEET — Mean across Monte Carlo samples (USD, {n_years}-year horizon)",
            header,
            sep,
            f"| **ASSETS** | {blank_cells} |",
            row("Non-Current Assets", "nca"),
            row("Advance Payments (Purchases)", "advance_payments_purchases"),
            row("Accounts Receivable", "accounts_receivable"),
            row("Inventory", "inventory"),
            row("Cash", "cash"),
            row("Investment in Market Securities", "investment_in_market_securities"),
            computed_row("**TOTAL ASSETS**", mean_total_assets),
            f"| | {blank_cells} |",
            f"| **LIABILITIES** | {blank_cells} |",
            row("Accounts Payable", "accounts_payable"),
            row("Advance Payments (Sales)", "advance_payments_sales"),
            row("Effective ST Debt", "effective_st_debt"),
            row("Current LT Debt", "current_lt_debt"),
            row("Non-Current Liabilities", "non_current_liabilities"),
            computed_row("**TOTAL LIABILITIES**", mean_total_liab),
            f"| | {blank_cells} |",
            computed_row("**EQUITY**", mean_equity),
            f"| | {blank_cells} |",
            computed_row("**TOTAL LIAB + EQUITY**", mean_total_liab_equity),
            "",
            "FORECAST INCOME STATEMENT — Mean across Monte Carlo samples (USD)",
            header,
            sep,
            row("COGS", "cogs"),
            row("OpEx", "opex"),
            row("Depreciation", "depreciation"),
            row("Interest Payments", "interest_payment"),
            row("ST Investment Returns", "ms_return"),
            row("Income Taxes", "tax"),
            row("**Net Income**", "net_income"),
            row("Dividends", "dividends"),
            row("Stock Buyback", "stock_buyback"),
        ]
        return "\n".join(lines)
