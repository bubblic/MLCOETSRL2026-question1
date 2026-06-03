# CLAUDE.md

Project + multi-repo context for this repository lives in
[`AGENTS.md`](AGENTS.md) (shared by all coding agents). Read it first.

@AGENTS.md

## Quick reminders (most-violated rules)

- **This repo** (`bubblic/MLCOETSRL2026-question1`) is the **implementation**
  home for the TensorFlow program + agentic workflow. The **proposal** is
  authored in the fork **`jaybhum/JPM_internship`** and is *not* checked in here.
- **Deep learning is TensorFlow only** — never PyTorch/JAX.
- **Run pipelines from the repo root** via top-level `run_*.py` entry points
  (prefer a new `run_*.py` over `python -m ...`).
- Install with `pip install -e ".[dev]"`; test with `python -m pytest tests/ -v`.
- `.env` (holds `AZURE_DEEPSEEK_ENDPOINT` etc.) is git-ignored — keep secrets out
  of source.
