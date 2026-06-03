# CLAUDE.md

Full project + multi-repo context lives in [`AGENTS.md`](AGENTS.md), which is
committed identically to **both** repositories of this project. Read it first —
especially **Step 0** (detect your context) and the **Terminology** note.

@AGENTS.md

## Quick reminders (most-violated rules)

- **Two repos, two modes, one codebase — detect, never assume.** This file is
  committed to both `bubblic/MLCOETSRL2026-question1` (**Code** repo) and its fork
  `jaybhum/JPM_internship` (**Proposal** repo). Run `git remote -v` for the repo
  (origin URL) and `git branch --show-current` for the mode (a
  `claude/jpm-internship-proposal-*` branch = Proposal mode). See AGENTS.md → *Step 0*.
- **Vocabulary is deliberately non-overlapping:** the Proposal workflow is
  **Research → Plan → Write** ("Write" = produce the proposal *document*);
  software work in the Code repo is **build**. Don't say or think "implement" —
  it used to mean both.
- **Building is gated on sign-off.** In the Code repo, net-new build work for the
  proposal's design waits until Research → Plan is signed off by Jaebum (check
  `proposal/RESEARCH.md` in the fork). Ordinary maintenance of existing code is
  fine anytime.
- **`financial_forecast/` is the validated core**; `loan_pricing/`,
  `credit_rating/`, `risk/` are exploratory sketches — don't treat them as production.
- **Deep learning is TensorFlow only** — never PyTorch/JAX.
- **Run pipelines from the repo root** via top-level `run_*.py` entry points
  (prefer a new `run_*.py` over `python -m ...`).
- Install with `pip install -e ".[dev]"`; test with `python -m pytest tests/ -v`.
- `.env` (holds `AZURE_DEEPSEEK_ENDPOINT` etc.) is git-ignored — keep secrets out
  of source.
