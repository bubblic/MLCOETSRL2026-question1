"""Apple Inc. one-time tax anomaly data (FY2018-FY2025).

Extracted from ``extracted_text/apple_YYYY.tax-anomalies-contingencies.llm.json``.
This data is kept separate from core historical financials so the pipeline
can demonstrate model improvement when one-time tax effects are incorporated.
"""


def get_tax_onetime_payments():
    """Return Apple's one-time tax anomaly amounts by fiscal year.

    Source values are in USD; years with no anomaly are omitted.

    Returns:
        Dict mapping fiscal year (int) to one-time tax amount (float, USD).
    """
    return {
        2018: 1.5e9,
        2020: -0.582e9,
        2024: 10.2e9,
    }
