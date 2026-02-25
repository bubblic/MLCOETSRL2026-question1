"""Financial ratio computation and run-level aggregation helpers."""

import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from .statement_config import (
    SCHEMA_VERSION,
    STATEMENT_CONFIGS,
    RunMap,
    all_statement_fields,
    parse_normalized_filename,
)
from .statement_normalization import sanitize_filename, statement_period_to_single_values


def safe_divide(
    numerator: Optional[float], denominator: Optional[float]
) -> Optional[float]:
    """Safely divide two values, returning None when division is not possible."""
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def sum_if_all_present(*values: Optional[float]) -> Optional[float]:
    """Return sum only when every input is present."""
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def calculate_financial_ratios(
    flat_values: Dict[str, Optional[float]],
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    """Compute derived metrics and financial ratios for one company-year period."""
    revenue = flat_values.get("total_revenue")
    operating_cost = flat_values.get("total_operating_cost")
    cash_and_cash_equivalents = flat_values.get("cash_and_cash_equivalents")
    short_term_market_securities = flat_values.get("short_term_market_securities")
    total_accounts_receivable = flat_values.get("total_accounts_receivable")
    total_current_liabilities = flat_values.get("total_current_liabilities")
    total_debt = flat_values.get("total_debt_short_term_and_long_term")
    total_equity = flat_values.get("total_equity")
    total_assets = flat_values.get("total_assets")
    net_income = flat_values.get("net_income")
    taxes = flat_values.get("taxes")
    interest_expenses = flat_values.get("interest_expenses")
    depreciation_and_amortization = flat_values.get("depreciation_and_amortization")

    ebit = sum_if_all_present(net_income, interest_expenses, taxes)
    ebitda = sum_if_all_present(ebit, depreciation_and_amortization)
    quick_assets = sum_if_all_present(
        cash_and_cash_equivalents,
        short_term_market_securities,
        total_accounts_receivable,
    )
    debt_plus_equity = sum_if_all_present(total_debt, total_equity)

    ratios = {
        "cost_to_income_ratio": safe_divide(operating_cost, revenue),
        "quick_ratio": safe_divide(quick_assets, total_current_liabilities),
        "debt_to_equity_ratio": safe_divide(total_debt, total_equity),
        "debt_to_assets_ratio": safe_divide(total_debt, total_assets),
        "debt_to_capital_ratio": safe_divide(total_debt, debt_plus_equity),
        "debt_to_ebitda_ratio": safe_divide(total_debt, ebitda),
        "interest_coverage_ratio": safe_divide(ebit, interest_expenses),
    }
    derived_values = {
        "ebit": ebit,
        "ebitda": ebitda,
        "quick_assets": quick_assets,
        "debt_plus_equity": debt_plus_equity,
    }
    return derived_values, ratios


def extract_single_values_from_normalized_files(
    input_dir: Path,
) -> RunMap:
    """Read normalized files and aggregate single values per company and year."""
    files = sorted(input_dir.glob("*.normalized.json"))
    if not files:
        raise ValueError(f"No normalized files found in: {input_dir}")

    company_periods: RunMap = {}
    for path in files:
        parsed = parse_normalized_filename(path.name)
        if not parsed:
            continue
        filename_company_id, statement_type = parsed
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        statement = payload.get("statement", {})
        if not isinstance(statement, dict):
            continue
        company_id = (
            str(statement.get("company_id", filename_company_id)).strip()
            or filename_company_id
        )
        periods = statement.get("periods", [])
        if not isinstance(periods, list):
            continue

        fields = payload.get("fields")
        if not isinstance(fields, list):
            fields = STATEMENT_CONFIGS[statement_type]["fields"]
        fields = [str(field) for field in fields]

        company_years = company_periods.setdefault(company_id, {})
        for period in periods:
            if not isinstance(period, dict):
                continue
            year = str(period.get("year", "")).strip()
            if not year:
                continue
            year_record = company_years.setdefault(
                year,
                {
                    "currency": "",
                    "field_values": {},
                    "present_fields": set(),
                },
            )
            currency = str(period.get("currency", "")).strip()
            if currency:
                year_record["currency"] = currency

            single_values = statement_period_to_single_values(period, fields=fields)
            year_record["field_values"].update(single_values)
            year_record["present_fields"].update(fields)

    normalized_company_periods: RunMap = {}
    for company_id in sorted(company_periods):
        normalized_company_periods[company_id] = {}
        for year in sorted(company_periods[company_id]):
            record = company_periods[company_id][year]
            present_fields = record.get("present_fields", set())
            if not isinstance(present_fields, set):
                present_fields = set()
            field_values: Dict[str, Optional[float]] = {}
            for field in all_statement_fields():
                if field in present_fields:
                    field_values[field] = float(record["field_values"].get(field, 0.0))
                else:
                    field_values[field] = None
            normalized_company_periods[company_id][year] = {
                "currency": record.get("currency", ""),
                "field_values": field_values,
            }
    return normalized_company_periods


def compute_ratios_from_normalized_files(input_dir: Path, output_file: Path) -> int:
    """Compute financial ratios directly from one normalized output directory."""
    company_periods = extract_single_values_from_normalized_files(input_dir)
    companies_output: List[Dict[str, Any]] = []
    total_periods = 0
    for company_id in sorted(company_periods):
        periods_out: List[Dict[str, Any]] = []
        for year in sorted(company_periods[company_id]):
            record = company_periods[company_id][year]
            field_values = record["field_values"]
            derived_values, ratios = calculate_financial_ratios(field_values)
            periods_out.append(
                {
                    "year": year,
                    "currency": record.get("currency", ""),
                    "field_values": field_values,
                    "derived_values": derived_values,
                    "ratios": ratios,
                }
            )
            total_periods += 1
        companies_output.append({"company_id": company_id, "periods": periods_out})

    output_payload = {
        "schema_version": SCHEMA_VERSION,
        "source_dir": str(input_dir),
        "companies": companies_output,
    }
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
    print(
        f"Finished ratios. Wrote {len(companies_output)} company file(s), {total_periods} period(s) to {output_file}"
    )
    return total_periods


def build_run_value_tracking(
    run_dirs: List[Path],
) -> Tuple[
    List[Dict[str, Any]],
    RunMap,
    List[Dict[str, Any]],
]:
    """Build per-run values, medians across runs, and value distributions."""
    run_entries: List[Dict[str, Any]] = []
    per_run_maps: List[RunMap] = []
    for run_idx, run_dir in enumerate(run_dirs, start=1):
        run_map = extract_single_values_from_normalized_files(run_dir)
        per_run_maps.append(run_map)
        companies_payload: List[Dict[str, Any]] = []
        for company_id in sorted(run_map):
            periods_payload: List[Dict[str, Any]] = []
            for year in sorted(run_map[company_id]):
                record = run_map[company_id][year]
                periods_payload.append(
                    {
                        "year": year,
                        "currency": record.get("currency", ""),
                        "field_values": record["field_values"],
                    }
                )
            companies_payload.append({"company_id": company_id, "periods": periods_payload})
        run_entries.append(
            {
                "run_id": f"run_{run_idx:02d}",
                "source_dir": str(run_dir),
                "companies": companies_payload,
            }
        )

    all_fields = all_statement_fields()
    all_keys: set = set()
    for run_map in per_run_maps:
        for company_id, year_map in run_map.items():
            for year in year_map:
                all_keys.add((company_id, year))

    median_map: RunMap = {}
    field_distributions: List[Dict[str, Any]] = []
    for company_id, year in sorted(all_keys):
        if company_id not in median_map:
            median_map[company_id] = {}
        currency = ""
        median_field_values: Dict[str, Optional[float]] = {}
        for field in all_fields:
            values_by_run: List[Optional[float]] = []
            valid_values: List[float] = []
            for run_map in per_run_maps:
                value: Optional[float] = None
                period_record = run_map.get(company_id, {}).get(year)
                if period_record:
                    run_currency = str(period_record.get("currency", "")).strip()
                    if run_currency and not currency:
                        currency = run_currency
                    raw_value = period_record.get("field_values", {}).get(field)
                    if isinstance(raw_value, (int, float)):
                        value = float(raw_value)
                values_by_run.append(value)
                if value is not None:
                    valid_values.append(value)
            median_value = median(valid_values) if valid_values else None
            median_field_values[field] = median_value
            field_distributions.append(
                {
                    "company_id": company_id,
                    "year": year,
                    "field": field,
                    "values_by_run": values_by_run,
                    "non_null_values": valid_values,
                    "median_value": median_value,
                }
            )
        median_map[company_id][year] = {
            "currency": currency,
            "field_values": median_field_values,
        }
    return run_entries, median_map, field_distributions


def write_distribution_plots(
    field_distributions: List[Dict[str, Any]],
    output_dir: Path,
) -> int:
    """Write per-company, per-year field distribution charts."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to plot distributions. Please install it (pip install matplotlib)."
        ) from exc

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for item in field_distributions:
        key = (str(item["company_id"]), str(item["year"]))
        grouped.setdefault(key, []).append(item)

    plots_written = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for (company_id, year), entries in grouped.items():
        entries_sorted = sorted(entries, key=lambda item: str(item["field"]))
        n = len(entries_sorted)
        cols = 3
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
        if rows == 1 and cols == 1:
            axes_list = [axes]
        elif rows == 1:
            axes_list = list(axes)
        else:
            axes_list = [axis for axis_row in axes for axis in axis_row]

        for idx, entry in enumerate(entries_sorted):
            ax = axes_list[idx]
            values_by_run = entry["values_by_run"]
            run_points = [
                (run_index + 1, value)
                for run_index, value in enumerate(values_by_run)
                if value is not None
            ]
            if run_points:
                xs = [point[0] for point in run_points]
                ys = [point[1] for point in run_points]
                ax.scatter(xs, ys, s=18)
                if len(ys) >= 2:
                    ax.plot(xs, ys, linewidth=0.7, alpha=0.4)
            median_value = entry.get("median_value")
            if isinstance(median_value, (int, float)):
                ax.axhline(float(median_value), linestyle="--", linewidth=1.0)
            ax.set_title(str(entry["field"]), fontsize=9)
            ax.set_xlabel("Run #", fontsize=8)
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(alpha=0.25)

        for idx in range(n, len(axes_list)):
            axes_list[idx].axis("off")

        fig.suptitle(f"{company_id} - {year} field distributions", fontsize=12)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        company_dir = output_dir / sanitize_filename(company_id)
        company_dir.mkdir(parents=True, exist_ok=True)
        plot_path = company_dir / f"{sanitize_filename(year)}.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        plots_written += 1
    return plots_written


def compute_ratios_from_median_runs(
    run_dirs: List[Path],
    ratios_output_file: Path,
    run_values_output_file: Path,
    plot_distributions: bool,
    plots_dir: Path,
) -> int:
    """Compute ratios from median values aggregated across multiple runs."""
    run_entries, median_map, field_distributions = build_run_value_tracking(run_dirs)

    run_tracking_payload = {
        "schema_version": SCHEMA_VERSION,
        "num_runs": len(run_dirs),
        "run_dirs": [str(path) for path in run_dirs],
        "fields": all_statement_fields(),
        "runs": run_entries,
        "field_distributions": field_distributions,
    }
    run_values_output_file.parent.mkdir(parents=True, exist_ok=True)
    with run_values_output_file.open("w", encoding="utf-8") as f:
        json.dump(run_tracking_payload, f, ensure_ascii=False, indent=2)

    companies_output: List[Dict[str, Any]] = []
    total_periods = 0
    for company_id in sorted(median_map):
        periods_out: List[Dict[str, Any]] = []
        for year in sorted(median_map[company_id]):
            record = median_map[company_id][year]
            field_values = record["field_values"]
            derived_values, ratios = calculate_financial_ratios(field_values)
            periods_out.append(
                {
                    "year": year,
                    "currency": record.get("currency", ""),
                    "field_values": field_values,
                    "derived_values": derived_values,
                    "ratios": ratios,
                }
            )
            total_periods += 1
        companies_output.append({"company_id": company_id, "periods": periods_out})

    output_payload = {
        "schema_version": SCHEMA_VERSION,
        "aggregation": "median_across_runs",
        "num_runs": len(run_dirs),
        "run_values_file": str(run_values_output_file),
        "companies": companies_output,
    }
    ratios_output_file.parent.mkdir(parents=True, exist_ok=True)
    with ratios_output_file.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    plots_written = 0
    if plot_distributions:
        plots_written = write_distribution_plots(
            field_distributions=field_distributions,
            output_dir=plots_dir,
        )
    print(
        f"Finished median ratios. Wrote {len(companies_output)} company file(s), {total_periods} period(s) to {ratios_output_file}"
    )
    print(f"Run value tracking written to {run_values_output_file}")
    if plot_distributions:
        print(f"Distribution plots written: {plots_written} to {plots_dir}")
    return total_periods
