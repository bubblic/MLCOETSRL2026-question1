"""Compute per-company, per-field hallucination rates from extraction run values.

A value is hallucinated if it is not equal to the median value for that
(company_id, year, field) distribution.  This module builds a detailed
report broken down by company-field and company-field-year granularity.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def values_equal(lhs: Optional[float], rhs: Optional[float]) -> bool:
    """Compare values with numeric tolerance and exact None matching."""
    if lhs is None or rhs is None:
        return lhs is rhs
    return math.isclose(float(lhs), float(rhs), rel_tol=1e-9, abs_tol=1e-6)


def aggregate_entry(
    values_by_run: List[Optional[float]],
    median_value: Optional[float],
) -> Dict[str, Any]:
    """Build hallucination metrics for one (company, year, field) entry."""
    total_values = len(values_by_run)
    hallucinated_all = sum(
        1 for value in values_by_run if not values_equal(value, median_value)
    )

    non_null_values = [value for value in values_by_run if value is not None]
    evaluated_non_null = len(non_null_values)
    hallucinated_non_null = sum(
        1 for value in non_null_values if not values_equal(value, median_value)
    )

    rate_all = hallucinated_all / total_values if total_values else None
    rate_non_null = (
        hallucinated_non_null / evaluated_non_null if evaluated_non_null else None
    )

    return {
        "total_values": total_values,
        "hallucinated_count_all_values": hallucinated_all,
        "hallucination_rate_all_values": rate_all,
        "evaluated_non_null_values": evaluated_non_null,
        "hallucinated_count_non_null_values": hallucinated_non_null,
        "hallucination_rate_non_null_values": rate_non_null,
    }


def build_hallucination_report(data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute hallucination rates per company_id/field and company_id/field/year."""
    field_distributions = data.get("field_distributions", [])
    if not isinstance(field_distributions, list):
        raise ValueError("Invalid run values file: field_distributions must be a list.")

    per_year_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for entry in field_distributions:
        company_id = entry.get("company_id")
        year = entry.get("year")
        field = entry.get("field")
        values_by_run = entry.get("values_by_run", [])
        median_value = entry.get("median_value")

        if (
            not isinstance(company_id, str)
            or not isinstance(year, str)
            or not isinstance(field, str)
        ):
            continue
        if not isinstance(values_by_run, list):
            continue

        metrics = aggregate_entry(
            values_by_run=values_by_run, median_value=median_value
        )
        per_year_rows.append(
            {
                "company_id": company_id,
                "field": field,
                "year": year,
                "median_value": median_value,
                **metrics,
            }
        )

        group_key = (company_id, field)
        if group_key not in grouped:
            grouped[group_key] = {
                "company_id": company_id,
                "field": field,
                "total_values": 0,
                "hallucinated_count_all_values": 0,
                "evaluated_non_null_values": 0,
                "hallucinated_count_non_null_values": 0,
            }
        grouped[group_key]["total_values"] += metrics["total_values"]
        grouped[group_key]["hallucinated_count_all_values"] += metrics[
            "hallucinated_count_all_values"
        ]
        grouped[group_key]["evaluated_non_null_values"] += metrics[
            "evaluated_non_null_values"
        ]
        grouped[group_key]["hallucinated_count_non_null_values"] += metrics[
            "hallucinated_count_non_null_values"
        ]

    per_company_field_rows: List[Dict[str, Any]] = []
    for _, aggregate in grouped.items():
        total_values = aggregate["total_values"]
        evaluated_non_null_values = aggregate["evaluated_non_null_values"]
        aggregate["hallucination_rate_all_values"] = (
            aggregate["hallucinated_count_all_values"] / total_values
            if total_values
            else None
        )
        aggregate["hallucination_rate_non_null_values"] = (
            aggregate["hallucinated_count_non_null_values"] / evaluated_non_null_values
            if evaluated_non_null_values
            else None
        )
        per_company_field_rows.append(aggregate)

    per_year_rows.sort(key=lambda row: (row["company_id"], row["field"], row["year"]))
    per_company_field_rows.sort(key=lambda row: (row["company_id"], row["field"]))

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_schema_version": data.get("schema_version"),
        "num_runs": data.get("num_runs"),
        "definition": {
            "hallucinated": "value != median_value for same company_id/year/field",
            "notes": [
                "Two rate variants are reported.",
                "all_values includes null values.",
                "non_null_values excludes null values.",
            ],
        },
        "per_company_field": per_company_field_rows,
        "per_company_field_year": per_year_rows,
    }


def print_top_rows(rows: List[Dict[str, Any]], top_k: int) -> None:
    """Print top-k rows by non-null hallucination rate."""
    ranked = [
        row
        for row in rows
        if isinstance(row.get("hallucination_rate_non_null_values"), (int, float))
    ]
    ranked.sort(key=lambda row: row["hallucination_rate_non_null_values"], reverse=True)
    shown = min(top_k, len(ranked))
    print()
    print("=" * 72)
    print("Hallucination Rate Summary")
    print("-" * 72)
    print(
        "A value is considered hallucinated when it differs from the median"
    )
    print(
        "across extraction runs for the same company, year, and field."
    )
    print(
        "Rates below are computed over non-null values only, aggregated"
    )
    print("across all years per company-field pair.")
    print("=" * 72)
    print(f"Top {shown} company-field pairs by hallucination rate:")
    print()
    for row in ranked[:top_k]:
        print(
            f"- {row['company_id']} | {row['field']} | "
            f"rate_non_null={row['hallucination_rate_non_null_values']:.4f} "
            f"({row['hallucinated_count_non_null_values']}/{row['evaluated_non_null_values']})"
        )


def run_hallucination_analysis(
    run_values_file: Path,
    output_file: Path,
    top_k: int = 15,
) -> None:
    """Load run values, compute hallucination report, write output, and print summary."""
    if not run_values_file.exists():
        raise FileNotFoundError(f"Run values file not found: {run_values_file}")

    with run_values_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    report = build_hallucination_report(data)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"Hallucination report written to: {output_file}")
    print_top_rows(report["per_company_field"], top_k=top_k)
