"""Generate figures for the risk warnings report from extracted JSON.

Reads pipeline output JSON and creates matplotlib figures saved to
``report_latex_media/media/risk_warnings/``.

Usage:
    python generate_risk_report_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JP_BLUE = "#002D72"
JP_BLUE_LIGHT = "#4A7FC1"
CODE_GREEN = "#008C00"
CODE_RED = "#B40000"
CODE_GRAY = "#646464"

CATEGORY_LABELS = {
    "auditor_opinion": "Auditor Opinion",
    "going_concern": "Going Concern",
    "contingent_liabilities": "Contingent Liabilities",
    "debt_covenants": "Debt Covenants",
    "related_party": "Related Party",
    "accounting_policy": "Accounting Policy",
    "director_changes": "Director Changes",
    "cash_flow_warnings": "Cash Flow Warnings",
    "md_and_a_red_flags": "MD&A Red Flags",
}


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def generate_categories_flagged(data: dict, output_path: Path) -> None:
    """Horizontal bar chart of pages flagged per risk category."""
    flagged = data["flagged_pages"]
    categories = []
    counts = []
    for cat, pages in sorted(flagged.items(), key=lambda x: len(x[1])):
        categories.append(CATEGORY_LABELS.get(cat, cat))
        counts.append(len(pages))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(categories, counts, color=JP_BLUE, edgecolor="white", height=0.6)
    ax.set_xlabel("Pages Flagged", fontsize=11)
    ax.set_title("Risk Categories Detected — Evergrande 2022", fontsize=13,
                 color=JP_BLUE, fontweight="bold")
    ax.set_xlim(0, max(counts) + 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=10, color=JP_BLUE)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_pipeline_usage(data: dict, output_path: Path) -> None:
    """Grouped bar chart of LLM usage by pipeline stage."""
    usage = data["usage"]["stages"]
    stages = list(usage.keys())
    stage_labels = [s.capitalize() for s in stages]

    calls = [usage[s]["call_count"] for s in stages]
    chars_in_k = [usage[s]["chars_in"] / 1000 for s in stages]
    chars_out_k = [usage[s]["chars_out"] / 1000 for s in stages]

    x = np.arange(len(stages))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    bars1 = ax1.bar(x - width, calls, width, label="LLM Calls",
                    color=JP_BLUE, edgecolor="white")
    ax1.set_ylabel("Call Count", color=JP_BLUE, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=JP_BLUE)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x, chars_in_k, width, label="Input (K chars)",
                    color=JP_BLUE_LIGHT, edgecolor="white")
    bars3 = ax2.bar(x + width, chars_out_k, width, label="Output (K chars)",
                    color=CODE_GREEN, edgecolor="white")
    ax2.set_ylabel("Characters (thousands)", color=CODE_GRAY, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=CODE_GRAY)

    ax1.set_xticks(x)
    ax1.set_xticklabels(stage_labels, fontsize=11)
    ax1.set_title("LLM Usage by Pipeline Stage — Evergrande 2022", fontsize=13,
                  color=JP_BLUE, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               fontsize=9, framealpha=0.9)

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_extraction_summary(data: dict, output_path: Path) -> None:
    """Summary figure showing key extraction findings by category."""
    findings = data["category_results"]

    categories = []
    descriptions = []
    colors = []

    for cat_name, result in sorted(findings.items()):
        label = CATEGORY_LABELS.get(cat_name, cat_name)
        extraction = result.get("extraction")

        if extraction is None or extraction == []:
            desc = "No findings"
            color = CODE_GRAY
        elif isinstance(extraction, dict) and "raw_response" in extraction:
            raw = extraction["raw_response"]
            try:
                parsed = json.loads(raw.strip().strip("`").removeprefix("json\n"))
                n = len(parsed) if isinstance(parsed, list) else 1
                desc = f"{n} item(s) extracted"
                color = CODE_RED if n > 5 else JP_BLUE
            except (json.JSONDecodeError, AttributeError):
                desc = "Raw response (unparsed)"
                color = CODE_GRAY
        elif isinstance(extraction, list):
            n = len(extraction)
            desc = f"{n} item(s) extracted" if n > 0 else "No findings"
            color = CODE_RED if n > 5 else JP_BLUE
        elif isinstance(extraction, dict):
            non_null = sum(1 for v in extraction.values() if v is not None)
            total = len(extraction)
            desc = f"{non_null}/{total} fields populated"
            color = JP_BLUE if non_null > 0 else CODE_GRAY
        else:
            desc = "Unknown format"
            color = CODE_GRAY

        categories.append(label)
        descriptions.append(desc)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")

    col_labels = ["Category", "Stage 2 Extraction Result"]
    cell_text = list(zip(categories, descriptions))
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(JP_BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F5F5F5" if row % 2 == 0 else "white")
            if col == 1:
                cell.set_text_props(color=colors[row - 1])

    ax.set_title("Extraction Results Summary — Evergrande 2022",
                 fontsize=13, color=JP_BLUE, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_aggregate_comparison(all_data: dict, output_path: Path) -> None:
    """Grouped bar chart comparing pipeline results across all 5 companies."""
    companies = []
    pages_flagged = []
    cats_hit = []
    total_calls = []

    for company, data in all_data.items():
        companies.append(company.upper() if len(company) <= 4 else company.capitalize())
        flagged = data["flagged_pages"]
        pages_flagged.append(sum(len(v) for v in flagged.values()))
        cats_hit.append(len(flagged))
        total_calls.append(data["usage"]["total_calls"])

    x = np.arange(len(companies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, pages_flagged, width, label="Pages Flagged",
                   color=JP_BLUE, edgecolor="white")
    bars2 = ax.bar(x, cats_hit, width, label="Categories Hit",
                   color=JP_BLUE_LIGHT, edgecolor="white")
    bars3 = ax.bar(x + width, total_calls, width, label="LLM Calls",
                   color=CODE_GREEN, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(companies, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Pipeline Results Across All Five Test Companies",
                 fontsize=13, color=JP_BLUE, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                        str(int(h)), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_format_comparison(all_data: dict, output_path: Path) -> None:
    """Bar chart highlighting native PDF vs HTML-to-PDF performance."""
    labels = []
    pages = []
    colors = []
    formats = {"evergrande": "Native PDF", "wirecard": "Native PDF",
               "svb": "HTML-to-PDF", "bbby": "HTML-to-PDF", "lehman": "HTML-to-PDF"}

    for company, data in all_data.items():
        name = company.upper() if len(company) <= 4 else company.capitalize()
        fmt = formats.get(company, "Unknown")
        labels.append(f"{name}\n({fmt})")
        flagged = data["flagged_pages"]
        pages.append(sum(len(v) for v in flagged.values()))
        colors.append(JP_BLUE if fmt == "Native PDF" else CODE_RED)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, pages, color=colors, edgecolor="white", width=0.6)
    ax.set_ylabel("Pages Flagged", fontsize=11)
    ax.set_title("Document Format Impact on Pipeline Effectiveness",
                 fontsize=13, color=JP_BLUE, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, count in zip(bars, pages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(count), ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=JP_BLUE, label="Native PDF"),
                       Patch(facecolor=CODE_RED, label="HTML-to-PDF (SEC EDGAR)")]
    ax.legend(handles=legend_elements, fontsize=10, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    import glob

    output_dir = Path("report_latex_media/media/risk_warnings")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Evergrande data for per-company figures
    eg_path = Path("extracted_json/risk_warnings/evergrande/ar2022.risk-warnings.llm.json")
    eg_data = load_json(eg_path)

    generate_categories_flagged(eg_data, output_dir / "categories_flagged.png")
    generate_pipeline_usage(eg_data, output_dir / "pipeline_usage.png")
    generate_extraction_summary(eg_data, output_dir / "extraction_summary.png")

    # Load all company data for aggregate figures
    companies = ["evergrande", "svb", "bbby", "lehman", "wirecard"]
    all_data = {}
    for company in companies:
        files = glob.glob(f"extracted_json/risk_warnings/{company}/*.json")
        if files:
            all_data[company] = load_json(Path(files[0]))

    if len(all_data) > 1:
        generate_aggregate_comparison(all_data, output_dir / "aggregate_comparison.png")
        generate_format_comparison(all_data, output_dir / "format_comparison.png")

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
