"""Financial ratio computation from normalized statement data.

Provides :class:`RatioCalculator` for single-directory mode and
:class:`MedianRatioCalculator` for median-across-runs aggregation.
"""

import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from financial_forecast.extraction.statement_config import (
    SCHEMA_VERSION,
    STATEMENT_CONFIGS,
    RunMap,
    all_statement_fields,
    parse_normalized_filename,
)
from financial_forecast.extraction.statement_normalization import (
    sanitize_filename,
    statement_period_to_single_values,
)
from financial_forecast.extraction.utils import safe_divide, sum_if_all_present


class RatioCalculator:
    """Compute financial ratios from a single normalized directory.

    Args:
        input_dir: Directory containing ``*.normalized.json`` files.
        output_file: Path for the output ratios JSON file.
    """

    def __init__(
        self,
        input_dir: str,
        output_file: str = "financial_ratios.json",
    ):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        if not self.output_file.is_absolute():
            self.output_file = self.input_dir / self.output_file

    def run(self) -> int:
        """Compute ratios and write output.

        Returns:
            Number of company-year periods processed.
        """
        company_periods = self._extract_single_values(self.input_dir)
        companies_output, total_periods = self._build_ratios_output(
            company_periods,
        )

        output_payload = {
            "schema_version": SCHEMA_VERSION,
            "source_dir": str(self.input_dir),
            "companies": companies_output,
        }
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_file.open("w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)
        print(
            f"Finished ratios. Wrote {len(companies_output)} company(s), "
            f"{total_periods} period(s) to {self.output_file}"
        )
        return total_periods

    # ------------------------------------------------------------------
    # Private computation methods
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_ratios(
        flat_values: Dict[str, Optional[float]],
    ) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]]]:
        """Compute derived metrics and financial ratios for one period."""
        revenue = flat_values.get("total_revenue")
        operating_cost = flat_values.get("total_operating_cost")
        cash = flat_values.get("cash_and_cash_equivalents")
        securities = flat_values.get("short_term_market_securities")
        receivables = flat_values.get("net_accounts_receivable")
        current_liab = flat_values.get("total_current_liabilities")
        total_debt = flat_values.get("total_debt_short_term_and_long_term")
        total_equity = flat_values.get("total_equity")
        total_assets = flat_values.get("total_assets")
        net_income = flat_values.get("net_income")
        income_tax_expense = flat_values.get("income_tax_expense")
        interest = flat_values.get("interest_expenses")
        dep_amort = flat_values.get("depreciation_and_amortization")

        ebit = sum_if_all_present(net_income, interest, income_tax_expense)
        ebitda = sum_if_all_present(ebit, dep_amort)
        quick_assets = sum_if_all_present(cash, securities, receivables)
        debt_plus_equity = sum_if_all_present(total_debt, total_equity)

        ratios = {
            "cost_to_income_ratio": safe_divide(operating_cost, revenue),
            "quick_ratio": safe_divide(quick_assets, current_liab),
            "debt_to_equity_ratio": safe_divide(total_debt, total_equity),
            "debt_to_assets_ratio": safe_divide(total_debt, total_assets),
            "debt_to_capital_ratio": safe_divide(
                total_debt,
                debt_plus_equity,
            ),
            "debt_to_ebitda_ratio": safe_divide(total_debt, ebitda),
            "interest_coverage_ratio": safe_divide(ebit, interest),
        }
        derived_values = {
            "ebit": ebit,
            "ebitda": ebitda,
            "quick_assets": quick_assets,
            "debt_plus_equity": debt_plus_equity,
        }
        return derived_values, ratios

    @staticmethod
    def _extract_single_values(input_dir: Path) -> RunMap:
        """Read normalized files and aggregate single values per company/year."""
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
                single_values = statement_period_to_single_values(
                    period,
                    fields=fields,
                )
                year_record["field_values"].update(single_values)
                year_record["present_fields"].update(fields)

        normalized: RunMap = {}
        for company_id in sorted(company_periods):
            normalized[company_id] = {}
            for year in sorted(company_periods[company_id]):
                record = company_periods[company_id][year]
                present_fields = record.get("present_fields", set())
                if not isinstance(present_fields, set):
                    present_fields = set()
                field_values: Dict[str, Optional[float]] = {}
                for field in all_statement_fields():
                    if field in present_fields:
                        field_values[field] = float(
                            record["field_values"].get(field, 0.0)
                        )
                    else:
                        field_values[field] = None
                normalized[company_id][year] = {
                    "currency": record.get("currency", ""),
                    "field_values": field_values,
                }
        return normalized

    def _build_ratios_output(
        self,
        company_periods: RunMap,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Build the output payload from company periods."""
        companies_output: List[Dict[str, Any]] = []
        total_periods = 0
        for company_id in sorted(company_periods):
            periods_out: List[Dict[str, Any]] = []
            for year in sorted(company_periods[company_id]):
                record = company_periods[company_id][year]
                field_values = record["field_values"]
                derived_values, ratios = self._calculate_ratios(field_values)
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
            companies_output.append(
                {
                    "company_id": company_id,
                    "periods": periods_out,
                }
            )
        return companies_output, total_periods


class MedianRatioCalculator(RatioCalculator):
    """Compute ratios from median values aggregated across multiple runs.

    Args:
        runs_output_dir: Root directory containing ``run_01/``, etc.
        output_file: Path for the output ratios JSON file.
        plot_distributions: Whether to write field distribution plots.
        plots_dir: Directory for distribution plot images.
        hallucination_output_file: Optional path for hallucination report.
        hallucination_top_k: Print top-k hallucination rows to stdout.
    """

    def __init__(
        self,
        runs_output_dir: str,
        output_file: str = "financial_ratios.json",
        plot_distributions: bool = False,
        plots_dir: str = "field_value_distributions",
        hallucination_output_file: Optional[str] = None,
        hallucination_top_k: int = 15,
    ):
        self.runs_root = Path(runs_output_dir)
        self.output_file = Path(output_file)
        if not self.output_file.is_absolute():
            self.output_file = self.runs_root / self.output_file
        self.plot_distributions = plot_distributions
        self.plots_dir = Path(plots_dir)
        if not self.plots_dir.is_absolute():
            self.plots_dir = self.runs_root / self.plots_dir
        self.hallucination_output_file = (
            Path(hallucination_output_file) if hallucination_output_file else None
        )
        self.hallucination_top_k = hallucination_top_k
        # input_dir not used directly — overridden by runs
        super().__init__(
            input_dir=runs_output_dir,
            output_file=output_file,
        )

    def run(self) -> int:
        """Aggregate medians across runs, compute ratios."""
        from financial_forecast.extraction.statement_runs import (
            list_run_dirs,
        )

        run_dirs = list_run_dirs(self.runs_root)
        if not run_dirs:
            raise ValueError(f"No run directories found in {self.runs_root}")

        run_entries, median_map, field_distributions = self._build_run_value_tracking(
            run_dirs
        )

        # Write run value tracking
        run_values_file = self.runs_root / "extraction_run_values.json"
        run_tracking_payload = {
            "schema_version": SCHEMA_VERSION,
            "num_runs": len(run_dirs),
            "run_dirs": [str(p) for p in run_dirs],
            "fields": all_statement_fields(),
            "runs": run_entries,
            "field_distributions": field_distributions,
        }
        run_values_file.parent.mkdir(parents=True, exist_ok=True)
        with run_values_file.open("w", encoding="utf-8") as f:
            json.dump(run_tracking_payload, f, ensure_ascii=False, indent=2)

        # Compute ratios from medians
        companies_output, total_periods = self._build_ratios_output(
            median_map,
        )
        output_payload = {
            "schema_version": SCHEMA_VERSION,
            "aggregation": "median_across_runs",
            "num_runs": len(run_dirs),
            "run_values_file": str(run_values_file),
            "companies": companies_output,
        }
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_file.open("w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)

        if self.plot_distributions:
            plots_written = self._write_distribution_plots(
                field_distributions,
                self.plots_dir,
            )
            print(f"Distribution plots written: {plots_written}")

        if self.hallucination_output_file:
            from financial_forecast.extraction.statement_hallucination import (
                run_hallucination_analysis,
            )

            hall_file = self.hallucination_output_file
            if not hall_file.is_absolute():
                hall_file = self.runs_root / hall_file
            run_hallucination_analysis(
                run_values_file=run_values_file,
                output_file=hall_file,
                top_k=self.hallucination_top_k,
            )

        print(
            f"Finished median ratios. {len(companies_output)} company(s), "
            f"{total_periods} period(s) to {self.output_file}"
        )
        return total_periods

    def _build_run_value_tracking(
        self,
        run_dirs: List[Path],
    ) -> Tuple[List[Dict[str, Any]], RunMap, List[Dict[str, Any]]]:
        """Build per-run values, medians, and distributions."""
        run_entries: List[Dict[str, Any]] = []
        per_run_maps: List[RunMap] = []

        for run_idx, run_dir in enumerate(run_dirs, start=1):
            run_map = self._extract_single_values(run_dir)
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
                companies_payload.append(
                    {
                        "company_id": company_id,
                        "periods": periods_payload,
                    }
                )
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

    @staticmethod
    def _write_distribution_plots(
        field_distributions: List[Dict[str, Any]],
        output_dir: Path,
    ) -> int:
        """Write per-company, per-year field distribution charts."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required for distribution plots."
            ) from exc

        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for item in field_distributions:
            key = (str(item["company_id"]), str(item["year"]))
            grouped.setdefault(key, []).append(item)

        plots_written = 0
        output_dir.mkdir(parents=True, exist_ok=True)
        for (company_id, year), entries in grouped.items():
            entries_sorted = sorted(
                entries,
                key=lambda x: str(x["field"]),
            )
            n = len(entries_sorted)
            cols = 3
            rows = math.ceil(n / cols)
            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(cols * 5, rows * 3.5),
            )
            if rows == 1 and cols == 1:
                axes_list = [axes]
            elif rows == 1:
                axes_list = list(axes)
            else:
                axes_list = [ax for row in axes for ax in row]

            for idx, entry in enumerate(entries_sorted):
                ax = axes_list[idx]
                values_by_run = entry["values_by_run"]
                run_points = [
                    (i + 1, v) for i, v in enumerate(values_by_run) if v is not None
                ]
                if run_points:
                    xs = [p[0] for p in run_points]
                    ys = [p[1] for p in run_points]
                    ax.scatter(xs, ys, s=18)
                    if len(ys) >= 2:
                        ax.plot(xs, ys, linewidth=0.7, alpha=0.4)
                median_value = entry.get("median_value")
                if isinstance(median_value, (int, float)):
                    ax.axhline(
                        float(median_value),
                        linestyle="--",
                        linewidth=1.0,
                    )
                ax.set_title(str(entry["field"]), fontsize=9)
                ax.set_xlabel("Run #", fontsize=8)
                ax.tick_params(axis="both", labelsize=8)
                ax.grid(alpha=0.25)

            for idx in range(n, len(axes_list)):
                axes_list[idx].axis("off")

            fig.suptitle(
                f"{company_id} - {year} field distributions",
                fontsize=12,
            )
            fig.tight_layout(rect=[0, 0.02, 1, 0.96])
            company_dir = output_dir / sanitize_filename(company_id)
            company_dir.mkdir(parents=True, exist_ok=True)
            plot_path = company_dir / f"{sanitize_filename(year)}.png"
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
            plots_written += 1
        return plots_written
