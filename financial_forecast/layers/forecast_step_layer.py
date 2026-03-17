"""Custom Keras layers implementing one-period forecast steps.

These layers encapsulate the asset evolution, income statement, and
liquidity/financing logic from the Pareja (2009) Cash Budget model as
reusable Keras layers suitable for gradient-based optimisation.
"""

import tensorflow as tf


class AssetEvolutionLayer(tf.keras.layers.Layer):
    """Evolves asset accounts for one forecast period.

    This layer encapsulates the asset evolution logic from the Pareja (2009)
    Cash Budget model as a reusable Keras layer.

    Args:
        asset_growth: Expected growth rate for non-current assets (capex proxy).
        depreciation_rate: Depreciation rate applied to beginning NCA.
        adv_pp_pct: Advance payments on purchases as a fraction of next-period purchases.
        ar_pct: Accounts receivable as a fraction of current-period sales.
        inv_pct: Inventory as a fraction of current-period sales.
        total_liq_pct: Total liquid assets as a fraction of current-period sales.
        cash_pct: Cash as a fraction of total liquid assets.
    """

    def __init__(
        self,
        asset_growth,
        depreciation_rate,
        adv_pp_pct,
        ar_pct,
        inv_pct,
        total_liq_pct,
        cash_pct,
        **kwargs,
    ):
        super().__init__(dtype=tf.float64, **kwargs)
        self.asset_growth = tf.cast(asset_growth, tf.float64)
        self.depreciation_rate = tf.cast(depreciation_rate, tf.float64)
        self.adv_pp_pct = tf.cast(adv_pp_pct, tf.float64)
        self.ar_pct = tf.cast(ar_pct, tf.float64)
        self.inv_pct = tf.cast(inv_pct, tf.float64)
        self.total_liq_pct = tf.cast(total_liq_pct, tf.float64)
        self.cash_pct = tf.cast(cash_pct, tf.float64)

    def call(self, nca_prev, sales_t, purchases_t_plus_1):
        """Compute evolved asset accounts for one period.

        Args:
            nca_prev: Non-current assets at the end of the previous period.
            sales_t: Sales revenue for the current period.
            purchases_t_plus_1: Purchases forecast for the next period (used for
                advance payments).

        Returns:
            A dictionary with the following keys:

            - ``depreciation``: Depreciation expense for the period.
            - ``capex``: Capital expenditure for the period.
            - ``nca``: Non-current assets at end of period.
            - ``accounts_receivable``: Accounts receivable at end of period.
            - ``inventory``: Inventory at end of period.
            - ``advance_payments_purchases``: Advance payments on purchases.
            - ``total_liquid_assets``: Total liquid assets at end of period.
            - ``cash``: Cash portion of liquid assets.
            - ``ims``: Investment in marketable securities (non-cash liquid assets).
        """
        depreciation = nca_prev * self.depreciation_rate
        capex = nca_prev * self.asset_growth
        nca = nca_prev - depreciation + capex

        accounts_receivable = sales_t * self.ar_pct
        inventory = sales_t * self.inv_pct
        advance_payments_purchases = purchases_t_plus_1 * self.adv_pp_pct

        total_liquid_assets = sales_t * self.total_liq_pct
        cash = total_liquid_assets * self.cash_pct
        ims = total_liquid_assets - cash

        return {
            "depreciation": depreciation,
            "capex": capex,
            "nca": nca,
            "accounts_receivable": accounts_receivable,
            "inventory": inventory,
            "advance_payments_purchases": advance_payments_purchases,
            "total_liquid_assets": total_liquid_assets,
            "cash": cash,
            "ims": ims,
        }


class IncomeStatementLayer(tf.keras.layers.Layer):
    """Computes income statement from state and economic inputs.

    Derives COGS, operating expenses, EBITDA, interest, taxes, and net income
    for a single forecast period.

    Args:
        baseline_opex: Fixed operating expense baseline (in nominal terms of
            the base year).
        variable_opex_pct: Variable operating expenses as a fraction of sales.
        avg_st_interest: Average short-term interest rate.
        avg_lt_interest: Average long-term interest rate.
        avg_maturity: Average debt maturity used to estimate current portion of
            long-term debt repayment.
        ms_return_pct: Expected return on marketable securities.
        income_tax_pct: Effective income tax rate.
    """

    def __init__(
        self,
        baseline_opex,
        variable_opex_pct,
        avg_st_interest,
        avg_lt_interest,
        avg_maturity,
        ms_return_pct,
        income_tax_pct,
        **kwargs,
    ):
        super().__init__(dtype=tf.float64, **kwargs)
        self.baseline_opex = tf.cast(baseline_opex, tf.float64)
        self.variable_opex_pct = tf.cast(variable_opex_pct, tf.float64)
        self.avg_st_interest = tf.cast(avg_st_interest, tf.float64)
        self.avg_lt_interest = tf.cast(avg_lt_interest, tf.float64)
        self.avg_maturity = tf.cast(avg_maturity, tf.float64)
        self.ms_return_pct = tf.cast(ms_return_pct, tf.float64)
        self.income_tax_pct = tf.cast(income_tax_pct, tf.float64)

    def call(self, state_dict, assets_dict, inputs_dict):
        """Compute the income statement for one forecast period.

        Args:
            state_dict: Dictionary containing prior-period state, including
                ``st_debt``, ``non_current_liabilities``, and ``ims``.
            assets_dict: Dictionary of evolved asset values from
                :class:`AssetEvolutionLayer`, including ``depreciation``.
            inputs_dict: Dictionary of exogenous inputs, including ``sales``,
                ``cogs``, and ``cum_inflation``.

        Returns:
            A dictionary with the following keys:

            - ``cogs``: Cost of goods sold.
            - ``opex``: Operating expenses.
            - ``ebitda``: Earnings before interest, taxes, depreciation, and
              amortisation.
            - ``ebit``: Earnings before interest and taxes.
            - ``interest_expense``: Total interest expense (short-term +
              long-term).
            - ``ms_return``: Return on marketable securities.
            - ``ebt``: Earnings before taxes.
            - ``tax``: Income tax provision.
            - ``net_income``: Net income after taxes.
            - ``current_lt_debt``: Current portion of long-term debt.
        """
        sales = inputs_dict["sales"]
        cogs = inputs_dict["cogs"]
        cum_inflation = inputs_dict["cum_inflation"]

        opex = self.baseline_opex * cum_inflation + sales * self.variable_opex_pct

        ebitda = sales - cogs - opex
        ebit = ebitda - assets_dict["depreciation"]

        st_interest = state_dict["st_debt"] * self.avg_st_interest
        lt_interest = state_dict["non_current_liabilities"] * self.avg_lt_interest
        interest_expense = st_interest + lt_interest

        ms_return = state_dict["ims"] * self.ms_return_pct

        ebt = ebit - interest_expense + ms_return
        tax = tf.maximum(ebt * self.income_tax_pct, tf.constant(0.0, dtype=tf.float64))
        net_income = ebt - tax

        current_lt_debt = state_dict["non_current_liabilities"] / self.avg_maturity

        return {
            "cogs": cogs,
            "opex": opex,
            "ebitda": ebitda,
            "ebit": ebit,
            "interest_expense": interest_expense,
            "ms_return": ms_return,
            "ebt": ebt,
            "tax": tax,
            "net_income": net_income,
            "current_lt_debt": current_lt_debt,
        }


class LiquidityFinancingLayer(tf.keras.layers.Layer):
    """Determines cash budget and new financing needs.

    Computes the operating net liquid balance, identifies financing deficits,
    and determines new short-term and long-term loans as well as equity
    distributions.

    Args:
        ar_pct: Accounts receivable as a fraction of sales (for collections).
        ap_pct: Accounts payable as a fraction of purchases (for payments).
        adv_ps_pct: Advance payments received from customers as a fraction of
            next-period sales.
        equity_fin_pct: Fraction of financing deficit covered by new equity.
        div_payout_pct: Dividend payout ratio (fraction of net income).
        buyback_pct: Share buyback ratio (fraction of net income or equity).
    """

    def __init__(
        self,
        ar_pct,
        ap_pct,
        adv_ps_pct,
        equity_fin_pct,
        div_payout_pct,
        buyback_pct,
        **kwargs,
    ):
        super().__init__(dtype=tf.float64, **kwargs)
        self.ar_pct = tf.cast(ar_pct, tf.float64)
        self.ap_pct = tf.cast(ap_pct, tf.float64)
        self.adv_ps_pct = tf.cast(adv_ps_pct, tf.float64)
        self.equity_fin_pct = tf.cast(equity_fin_pct, tf.float64)
        self.div_payout_pct = tf.cast(div_payout_pct, tf.float64)
        self.buyback_pct = tf.cast(buyback_pct, tf.float64)

    def call(self, state_dict, assets_dict, income_dict, inputs_dict):
        """Compute cash budget and financing decisions for one period.

        Args:
            state_dict: Dictionary of prior-period state, including
                ``accounts_receivable``, ``accounts_payable``,
                ``advance_payments_sales``, ``advance_payments_purchases``,
                ``cash``, ``st_debt``, ``non_current_liabilities``, and
                ``equity``.
            assets_dict: Dictionary of evolved asset values from
                :class:`AssetEvolutionLayer`, including ``capex``.
            income_dict: Dictionary of income statement results from
                :class:`IncomeStatementLayer`, including ``net_income``,
                ``tax``, ``interest_expense``, and ``current_lt_debt``.
            inputs_dict: Dictionary of exogenous inputs, including ``sales``,
                ``purchases``, and ``sales_next``.

        Returns:
            A dictionary with the following keys:

            - ``collections``: Cash collected from customers.
            - ``payments``: Cash paid to suppliers.
            - ``advance_payments_sales``: Advance payments received from
              customers.
            - ``accounts_payable``: Accounts payable at end of period.
            - ``operating_nlb``: Operating net liquid balance.
            - ``deficit``: Financing deficit (positive means funding needed).
            - ``new_st_loan``: New short-term borrowing.
            - ``new_lt_loan``: New long-term borrowing.
            - ``new_equity``: New equity issued.
            - ``dividends``: Dividends paid.
            - ``stock_buyback``: Share repurchases.
            - ``st_debt``: Short-term debt at end of period.
            - ``non_current_liabilities``: Long-term debt at end of period.
            - ``equity``: Equity at end of period.
        """
        sales = inputs_dict["sales"]
        purchases = inputs_dict["purchases"]
        sales_next = inputs_dict["sales_next"]

        # Cash collections: prior AR collected + cash sales
        collections = state_dict["accounts_receivable"] + sales * (1.0 - self.ar_pct)

        # Cash payments to suppliers
        payments = state_dict["accounts_payable"] + purchases * (1.0 - self.ap_pct)
        accounts_payable = purchases * self.ap_pct

        # Advance payments received from customers
        advance_payments_sales = sales_next * self.adv_ps_pct

        # Operating net liquid balance
        operating_nlb = (
            state_dict["cash"]
            + collections
            + state_dict["advance_payments_sales"]
            + advance_payments_sales
            - payments
            - state_dict["advance_payments_purchases"]
            - income_dict["tax"]
            - income_dict["interest_expense"]
            - assets_dict["capex"]
        )

        # Dividends and buybacks
        net_income = income_dict["net_income"]
        dividends = tf.maximum(net_income * self.div_payout_pct, tf.constant(0.0, dtype=tf.float64))
        stock_buyback = tf.maximum(net_income * self.buyback_pct, tf.constant(0.0, dtype=tf.float64))

        # Debt repayments
        current_lt_debt = income_dict["current_lt_debt"]
        st_debt_repayment = state_dict["st_debt"]

        # Available after distributions and repayments
        available = operating_nlb - dividends - stock_buyback - st_debt_repayment - current_lt_debt

        # Financing deficit
        deficit = tf.maximum(-available, tf.constant(0.0, dtype=tf.float64))

        # New financing
        new_equity = deficit * self.equity_fin_pct
        debt_financing = deficit - new_equity
        new_st_loan = debt_financing * tf.constant(0.5, dtype=tf.float64)
        new_lt_loan = debt_financing - new_st_loan

        # End-of-period balances
        st_debt = new_st_loan
        non_current_liabilities = state_dict["non_current_liabilities"] - current_lt_debt + new_lt_loan
        equity = state_dict["equity"] + new_equity + net_income - dividends - stock_buyback

        return {
            "collections": collections,
            "payments": payments,
            "advance_payments_sales": advance_payments_sales,
            "accounts_payable": accounts_payable,
            "operating_nlb": operating_nlb,
            "deficit": deficit,
            "new_st_loan": new_st_loan,
            "new_lt_loan": new_lt_loan,
            "new_equity": new_equity,
            "dividends": dividends,
            "stock_buyback": stock_buyback,
            "st_debt": st_debt,
            "non_current_liabilities": non_current_liabilities,
            "equity": equity,
        }
