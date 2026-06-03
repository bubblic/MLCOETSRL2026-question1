# AGENTS.md

> Orientation for AI coding agents (Claude Code, Codex, Cursor, and any future
> agent) working in this repository. Human contributors should start with
> [`README.md`](README.md) for build/run instructions — this file exists to
> explain *context that is not obvious from the code or git history*.

## TL;DR — two repositories, one shared codebase

This codebase lives in **two GitHub repositories linked by a fork**, and this
`AGENTS.md` is committed to **both** of them. **Do not infer which one you are
in from this text — detect it at runtime (Step 0 below).**

| Repository | Role |
| --- | --- |
| `bubblic/MLCOETSRL2026-question1` | **Implementation** — the TensorFlow program and the agentic-workflow code. |
| `jaybhum/JPM_internship` *(a fork of the above)* | **Proposal** — where the written proposal is drafted and compiled. |

Author: Jaebum "Albert" Chung (JP Morgan MLCOE TSRL 2026 Internship, Question 1).
Work flows **proposal → implementation**: the fork describes intent; the
implementation repo realizes it.

### Step 0 — identify which repo you are in

Branch *names* alone won't tell you (both repos can have a `main`); the
**origin remote URL** is the reliable signal:

```
git remote -v          # or:  git config --get remote.origin.url
```

- origin contains **`bubblic/MLCOETSRL2026-question1`** → you are in the
  **implementation** repo. New TensorFlow modeling and agentic-workflow code
  belongs here, fitted into the existing `financial_forecast/` layout (below).
  The proposal is **not** checked in here — it lives in the fork; ask the user
  to paste any section you need rather than assuming its contents.
- origin contains **`jaybhum/JPM_internship`** → you are in the **proposal
  fork**. Authoring/proposal work lives here; treat the implementation repo as
  upstream and keep the two in sync when you merge.
- **anything else** (another fork, mirror, or fresh local clone) → don't guess;
  ask the user which role this checkout plays.

> **Why this matters on merge:** because the fork merges back into the
> implementation repo, shared files like this one must stay **identical and
> identity-neutral** in both. Hardcoding "this repo is X" makes the file wrong
> the moment it lands in the other repo — so describe both roles and detect,
> never assert.

## What already exists here

This is a mature codebase, not a blank slate. Before adding anything, check
whether a module already covers it. See [`README.md`](README.md) for the full
file-by-file tree; the high-level shape is:

- **`financial_forecast/`** — the core TensorFlow package. Bayesian financial
  forecasting for Apple (FY2018–FY2025 historicals), built around composable
  policy modules (dividends, buybacks, debt, OpEx, tax, working capital, etc.),
  a `training/` pipeline, and `inference/` simulators (deterministic +
  Monte Carlo). This is where the modeling work concentrates.
- **Agentic / LLM components already in place** (the foundation the proposal's
  workflow builds on):
  - `financial_forecast/clients/` — Azure-hosted LLM HTTP client + protocols.
  - `financial_forecast/extraction/` — multi-stage LLM pipeline that pulls
    financial statements and one-time tax anomalies out of 10-K PDFs
    (page identification → extraction → normalization → ratios).
  - `financial_forecast/reporting/advisor.py` — `DeepseekCEOAdvisor`, an
    LLM that turns a forecast report into a CEO-facing recommendation.
  - `financial_forecast/models/llm_forecaster.py` — LLM-based balance-sheet
    forecaster.
  - `risk/` — LLM-based risk-warning extraction/synthesis from filings.
  - `send_reasoning_prompt.py` — standalone reasoning-model prompt harness.
- **Bonus-question packages** — `credit_rating/`, `loan_pricing/`, `risk/`
  (each self-contained; see their dirs and `bonusquestionplans/`).
- **Reports** — LaTeX sources + compiled PDFs at the repo root
  (`*_report.tex` / `*.pdf`). These are deliverables; regenerate the PDF when
  you change the `.tex`.
- **`run_*.py`** — top-level entry-point scripts, each runnable from the repo
  root. **Prefer adding a new `run_*.py` entry point over `python -m ...`
  invocation** for new pipelines.
- **`tests/`** — pytest suite. Run and keep it green.

## Conventions (apply to all agents, not just Claude)

- **Deep learning is TensorFlow, always.** This project standardizes on
  TensorFlow / TensorFlow-Probability (see `pyproject.toml`). Do **not**
  introduce PyTorch, JAX, or another DL framework.
- **Entry points run from the repo root.** Scripts assume the root as the
  working directory and write reports to `training_results/` and `outputs/`.
  Add new user-facing pipelines as top-level `run_*.py` scripts.
- **Install editable:** `pip install -e .` (or `pip install -e ".[dev]"` for
  pytest). Optional extras exist for `loan-pricing`, `credit-rating`, `risk`,
  and `data` — see `pyproject.toml`.
- **Tests:** `python -m pytest tests/ -v`. Add tests alongside new modeling or
  workflow code.
- **Style:** match the surrounding code — this codebase favors clean, composable
  OOP (ABCs + small policy classes). Mirror existing naming and structure when
  extending a package.

## Secrets & environment

- LLM pipelines need `AZURE_DEEPSEEK_ENDPOINT` (and related credentials),
  supplied via a local `.env` (git-ignored) or the shell environment.
- **`.env` is not committed and must stay that way.** Never hard-code endpoints,
  keys, or tokens into source or into this file.

## Pointers

- [`README.md`](README.md) — install, every `run_*.py` script, full project tree,
  Windows/CUDA setup notes.
- [`pyproject.toml`](pyproject.toml) — dependencies, optional extras, tooling.
- `bonusquestionplans/` — design notes for the bonus-question packages.
- `final_interview_prep/` — study/interview notes (context, not code).

---
*If the two-repo arrangement, ownership, or workflow above changes, update this
file (in **both** repos) so the next agent inherits an accurate picture.*
