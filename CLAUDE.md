# CLAUDE.md

Project + multi-repo context for this repository lives in
[`AGENTS.md`](AGENTS.md) (shared by all coding agents). Read it first.

@AGENTS.md

## Quick reminders (most-violated rules)

- **Two repos, one codebase.** This file is committed to both
  `bubblic/MLCOETSRL2026-question1` (**implementation**) and its fork
  `jaybhum/JPM_internship` (**proposal**). Don't assume which you're in — run
  `git remote -v` and check the origin URL (see [AGENTS.md](AGENTS.md) →
  *Step 0*). The TF/agentic code goes in the implementation repo; the proposal
  isn't checked in there.
- **Deep learning is TensorFlow only** — never PyTorch/JAX.
- **Run pipelines from the repo root** via top-level `run_*.py` entry points
  (prefer a new `run_*.py` over `python -m ...`).
- Install with `pip install -e ".[dev]"`; test with `python -m pytest tests/ -v`.
- `.env` (holds `AZURE_DEEPSEEK_ENDPOINT` etc.) is git-ignored — keep secrets out
  of source.
