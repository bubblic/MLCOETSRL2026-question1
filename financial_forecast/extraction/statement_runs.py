"""Run-directory utilities used by extraction and ratio workflows.

This module provides helpers for discovering, sorting, and numbering the
``run_NN`` directories created during multi-run extraction, as well as a
small path-resolution utility.
"""

import re
from pathlib import Path
from typing import List, Tuple


def list_run_dirs(runs_root: Path) -> List[Path]:
    """Return run directories sorted by numeric run suffix."""
    run_paths: List[Tuple[int, Path]] = []
    for path in runs_root.glob("run_*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"run_(\d+)", path.name)
        if not match:
            continue
        run_paths.append((int(match.group(1)), path))
    return [path for _, path in sorted(run_paths, key=lambda item: item[0])]


def get_next_run_number(runs_root: Path, append_runs: bool) -> int:
    """Return the first run number to use for this extraction batch."""
    if not append_runs:
        return 1
    existing_run_dirs = list_run_dirs(runs_root)
    if not existing_run_dirs:
        return 1
    match = re.fullmatch(r"run_(\d+)", existing_run_dirs[-1].name)
    if not match:
        return 1
    return int(match.group(1)) + 1


def resolve_output_path(base_dir: Path, maybe_relative_path: str) -> Path:
    """Resolve output path, keeping absolute paths unchanged."""
    candidate = Path(maybe_relative_path)
    return candidate if candidate.is_absolute() else base_dir / candidate
