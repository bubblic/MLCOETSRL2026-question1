"""CLI entry logic for extraction and ratio workflows."""

import argparse
from pathlib import Path

from .statement_extraction import run_extraction_pipeline
from .statement_hallucination import run_hallucination_analysis
from .statement_ratios import (
    compute_ratios_from_median_runs,
    compute_ratios_from_normalized_files,
)
from .statement_runs import list_run_dirs, resolve_output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract normalized financial-statement fields from extracted_text "
            "using Azure DeepSeek."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="extracted_text",
        help=(
            "Directory containing *.consolidated-balance-sheet.llm.json, "
            "*.consolidated-income-statement.llm.json, and "
            "*.consolidated-cash-flow-statement.llm.json files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="deepseek_financial_statements",
        help="Output directory to write one normalized JSON per statement file.",
    )
    parser.add_argument(
        "--num-extraction-runs",
        type=int,
        default=1,
        help="Number of repeated extraction runs to execute.",
    )
    parser.add_argument(
        "--runs-output-dir",
        default="deepseek_financial_statements_runs",
        help="Output root for multi-run extraction folders (run_01, run_02, ...).",
    )
    parser.add_argument(
        "--no-append-runs",
        action="store_true",
        help=(
            "Do not append to existing run folders. "
            "When set, multi-run extraction starts again from run_01."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help="Maximum tokens for model response.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--message",
        default="gen-ai-response",
        help="Message field sent to the Azure DeepSeek endpoint.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=9,
        help=(
            "Number of statement files to process in parallel. "
            "Set to 1 to run sequentially."
        ),
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip LLM extraction and only run ratio calculation (if enabled).",
    )
    parser.add_argument(
        "--compute-ratios",
        dest="compute_ratios",
        action="store_true",
        default=True,
        help="Compute single-value fields and financial ratios from *.normalized.json files.",
    )
    parser.add_argument(
        "--skip-ratios",
        dest="compute_ratios",
        action="store_false",
        help="Skip financial ratio calculation.",
    )
    parser.add_argument(
        "--ratios-input-dir",
        default=None,
        help=(
            "Input directory containing *.normalized.json files. "
            "Defaults to --output-dir."
        ),
    )
    parser.add_argument(
        "--ratios-output-file",
        default="financial_ratios.json",
        help="Output JSON file path for computed ratios.",
    )
    parser.add_argument(
        "--ratios-aggregation",
        choices=["single", "median"],
        default="single",
        help=(
            "How to aggregate field values before ratio calculation. "
            "'single' reads one normalized directory. "
            "'median' aggregates medians across multi-run outputs."
        ),
    )
    parser.add_argument(
        "--run-values-output-file",
        default="extraction_run_values.json",
        help="Output JSON file path for per-run field-value tracking.",
    )
    parser.add_argument(
        "--plot-distributions",
        action="store_true",
        help="Plot per-field value distributions across runs.",
    )
    parser.add_argument(
        "--plots-dir",
        default="field_value_distributions",
        help="Output directory for generated field-distribution plots.",
    )
    parser.add_argument(
        "--hallucination-report",
        dest="hallucination_report",
        action="store_true",
        default=True,
        help=(
            "Compute hallucination rates after multi-run median aggregation. "
            "Enabled by default when --ratios-aggregation=median."
        ),
    )
    parser.add_argument(
        "--skip-hallucination-report",
        dest="hallucination_report",
        action="store_false",
        help="Skip hallucination rate calculation.",
    )
    parser.add_argument(
        "--hallucination-output-file",
        default="hallucination_rates.json",
        help="Output JSON file path for the hallucination report.",
    )
    parser.add_argument(
        "--hallucination-top-k",
        type=int,
        default=15,
        help="Print top-k highest non-null hallucination-rate rows to stdout.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate key CLI constraints before running any workload."""
    if args.num_extraction_runs < 1:
        raise ValueError("--num-extraction-runs must be >= 1")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")


def run_ratio_pipeline(args: argparse.Namespace) -> None:
    """Run ratio computation in single-directory or median-across-runs mode."""
    if not args.compute_ratios:
        return

    ratios_output_file = Path(args.ratios_output_file)
    if args.ratios_aggregation == "median":
        runs_root = Path(args.runs_output_dir)
        run_dirs = list_run_dirs(runs_root)
        if not run_dirs:
            raise ValueError(
                "No run directories found for median aggregation. "
                "Run extraction with --num-extraction-runs > 1 or provide existing runs in --runs-output-dir."
            )
        ratios_output_file = resolve_output_path(runs_root, args.ratios_output_file)
        run_values_output_file = resolve_output_path(
            runs_root, args.run_values_output_file
        )
        plots_dir = resolve_output_path(runs_root, args.plots_dir)
        compute_ratios_from_median_runs(
            run_dirs=run_dirs,
            ratios_output_file=ratios_output_file,
            run_values_output_file=run_values_output_file,
            plot_distributions=args.plot_distributions,
            plots_dir=plots_dir,
        )
        return

    output_dir = Path(args.output_dir)
    ratios_input_dir = (
        Path(args.ratios_input_dir) if args.ratios_input_dir else output_dir
    )
    if not ratios_input_dir.exists() or not ratios_input_dir.is_dir():
        raise ValueError(f"Invalid ratios input dir: {ratios_input_dir}")
    if not ratios_output_file.is_absolute():
        ratios_output_file = ratios_input_dir / ratios_output_file
    ratios_output_file.parent.mkdir(parents=True, exist_ok=True)
    compute_ratios_from_normalized_files(
        input_dir=ratios_input_dir,
        output_file=ratios_output_file,
    )


def run_hallucination_pipeline(args: argparse.Namespace) -> None:
    """Run hallucination analysis when multi-run median aggregation was used."""
    if not args.hallucination_report:
        return
    if args.ratios_aggregation != "median":
        return

    runs_root = Path(args.runs_output_dir)
    run_values_file = resolve_output_path(runs_root, args.run_values_output_file)
    hallucination_output_file = resolve_output_path(
        runs_root, args.hallucination_output_file
    )

    run_hallucination_analysis(
        run_values_file=run_values_file,
        output_file=hallucination_output_file,
        top_k=args.hallucination_top_k,
    )


def main() -> None:
    """Entry point for extraction, ratio, and hallucination workflows."""
    args = parse_args()
    validate_args(args)
    run_extraction_pipeline(args)
    run_ratio_pipeline(args)
    run_hallucination_pipeline(args)
