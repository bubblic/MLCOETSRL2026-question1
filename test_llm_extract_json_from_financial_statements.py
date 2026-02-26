"""Tests for extraction entrypoint and financial statement CLI logic.

These tests are intentionally mock-heavy so they validate control flow and
argument handling without triggering real extraction workloads or network I/O.
"""

import argparse
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

import financial_statement_pipeline.statement_cli as statement_cli


@pytest.fixture
def valid_args(tmp_path):
    """Return a reusable Namespace with valid defaults for CLI routines."""
    normalized_input_dir = tmp_path / "normalized"
    normalized_input_dir.mkdir(parents=True, exist_ok=True)

    return argparse.Namespace(
        input_dir=str(tmp_path / "extracted_text"),
        output_dir=str(tmp_path / "deepseek_financial_statements"),
        num_extraction_runs=1,
        runs_output_dir=str(tmp_path / "deepseek_financial_statements_runs"),
        no_append_runs=False,
        temperature=0.0,
        max_tokens=8000,
        top_k=1,
        message="gen-ai-response",
        max_workers=3,
        skip_extraction=False,
        compute_ratios=True,
        ratios_input_dir=str(normalized_input_dir),
        ratios_output_file="financial_ratios.json",
        ratios_aggregation="single",
        run_values_output_file="extraction_run_values.json",
        plot_distributions=False,
        plots_dir="field_value_distributions",
    )


@pytest.fixture
def module_name():
    """Entry module name for script-style execution checks."""
    return "llm_extract_json_from_financial_statements"


def test_entrypoint_main_reexport():
    """Entrypoint should re-export CLI main for direct script invocation."""
    import llm_extract_json_from_financial_statements as entrypoint

    assert entrypoint.main is statement_cli.main


def test_entrypoint_script_execution_calls_main_once(monkeypatch, module_name):
    """Executing the entrypoint as __main__ should call CLI main exactly once."""
    mocked_main = Mock()
    monkeypatch.setattr(statement_cli, "main", mocked_main)

    runpy.run_module(module_name, run_name="__main__")

    mocked_main.assert_called_once_with()


def test_validate_args_accepts_valid_inputs(valid_args):
    """validate_args should not raise when key constraints are satisfied."""
    statement_cli.validate_args(valid_args)


def test_validate_args_rejects_invalid_num_runs(valid_args):
    """num_extraction_runs must be >= 1."""
    valid_args.num_extraction_runs = 0

    with pytest.raises(ValueError, match="--num-extraction-runs must be >= 1"):
        statement_cli.validate_args(valid_args)


def test_validate_args_rejects_invalid_max_workers(valid_args):
    """max_workers must be >= 1."""
    valid_args.max_workers = 0

    with pytest.raises(ValueError, match="--max-workers must be >= 1"):
        statement_cli.validate_args(valid_args)


def test_run_ratio_pipeline_returns_early_when_disabled(monkeypatch, valid_args):
    """When ratio computation is disabled, no ratio helper should be called."""
    valid_args.compute_ratios = False
    mocked_median = Mock()
    mocked_single = Mock()
    monkeypatch.setattr(statement_cli, "compute_ratios_from_median_runs", mocked_median)
    monkeypatch.setattr(
        statement_cli, "compute_ratios_from_normalized_files", mocked_single
    )

    statement_cli.run_ratio_pipeline(valid_args)

    mocked_median.assert_not_called()
    mocked_single.assert_not_called()


def test_run_ratio_pipeline_median_raises_without_run_dirs(monkeypatch, valid_args):
    """Median aggregation requires at least one run directory."""
    valid_args.ratios_aggregation = "median"
    monkeypatch.setattr(statement_cli, "list_run_dirs", Mock(return_value=[]))

    with pytest.raises(ValueError, match="No run directories found"):
        statement_cli.run_ratio_pipeline(valid_args)


def test_run_ratio_pipeline_median_calls_aggregation(monkeypatch, valid_args, tmp_path):
    """Median mode should call aggregator with resolved output paths."""
    valid_args.ratios_aggregation = "median"
    valid_args.runs_output_dir = str(tmp_path / "runs_root")
    runs_root = Path(valid_args.runs_output_dir)
    run_dirs = [runs_root / "run_01", runs_root / "run_02"]

    monkeypatch.setattr(statement_cli, "list_run_dirs", Mock(return_value=run_dirs))
    monkeypatch.setattr(
        statement_cli,
        "resolve_output_path",
        Mock(side_effect=lambda root, rel: Path(root) / rel),
    )
    mocked_compute = Mock()
    monkeypatch.setattr(statement_cli, "compute_ratios_from_median_runs", mocked_compute)

    statement_cli.run_ratio_pipeline(valid_args)

    mocked_compute.assert_called_once()
    kwargs = mocked_compute.call_args.kwargs
    assert kwargs["run_dirs"] == run_dirs
    assert kwargs["ratios_output_file"] == runs_root / valid_args.ratios_output_file
    assert kwargs["run_values_output_file"] == runs_root / valid_args.run_values_output_file
    assert kwargs["plots_dir"] == runs_root / valid_args.plots_dir
    assert kwargs["plot_distributions"] is False


def test_run_ratio_pipeline_single_invalid_input_dir_raises(valid_args):
    """Single aggregation should reject a missing ratios input directory."""
    valid_args.ratios_aggregation = "single"
    valid_args.ratios_input_dir = "does_not_exist_for_test"

    with pytest.raises(ValueError, match="Invalid ratios input dir"):
        statement_cli.run_ratio_pipeline(valid_args)


def test_run_ratio_pipeline_single_calls_normalized_compute(monkeypatch, valid_args):
    """Single aggregation should call normalized-file ratio computation."""
    valid_args.ratios_aggregation = "single"
    mocked_compute = Mock()
    monkeypatch.setattr(
        statement_cli, "compute_ratios_from_normalized_files", mocked_compute
    )

    statement_cli.run_ratio_pipeline(valid_args)

    mocked_compute.assert_called_once()
    kwargs = mocked_compute.call_args.kwargs
    expected_input = Path(valid_args.ratios_input_dir)
    expected_output = expected_input / valid_args.ratios_output_file
    assert kwargs["input_dir"] == expected_input
    assert kwargs["output_file"] == expected_output


def test_main_orchestrates_parse_validate_extract_and_ratios(monkeypatch, valid_args):
    """main should parse args and execute validate, extraction, then ratio pipeline."""
    calls = []

    monkeypatch.setattr(statement_cli, "parse_args", Mock(return_value=valid_args))

    def _validate(args):
        calls.append(("validate", args))

    def _extract(args):
        calls.append(("extract", args))

    def _ratios(args):
        calls.append(("ratios", args))

    monkeypatch.setattr(statement_cli, "validate_args", _validate)
    monkeypatch.setattr(statement_cli, "run_extraction_pipeline", _extract)
    monkeypatch.setattr(statement_cli, "run_ratio_pipeline", _ratios)

    statement_cli.main()

    assert calls == [
        ("validate", valid_args),
        ("extract", valid_args),
        ("ratios", valid_args),
    ]
