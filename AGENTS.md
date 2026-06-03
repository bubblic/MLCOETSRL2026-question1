# AGENTS.md

> Orientation for AI coding agents (Claude Code, Codex, Cursor, and any future
> agent) working in this repository. Human contributors should start with
> [`README.md`](README.md) for build/run instructions — this file exists to
> explain *context that is not obvious from the code or git history*.

## TL;DR — the two-repo setup

This repository is **`bubblic/MLCOETSRL2026-question1`** (the `origin` remote).
It is the **implementation home** for JP Morgan MLCOE TSRL 2026 Internship
Question 1 (author: Jaebum "Albert" Chung).

There is a companion repository, **`jaybhum/JPM_internship`**, which is a
**fork of this repo**. That fork is where the **written proposal** is drafted
and compiled. **This repo is where the TensorFlow program and the agentic
workflow described in that proposal actually get built.**

```
 jaybhum/JPM_internship  (fork)          bubblic/MLCOETSRL2026-question1  (this repo / origin)
 ──────────────────────────────          ───────────────────────────────────────────────────
 Proposal authoring & compiling   ──▶    Implementation: TensorFlow models + agentic workflow
 (the "what" and "why")                  (the "how" — runnable code, tests, LaTeX reports)
```

What this means for an agent working **here**:

- **This repo is the source of truth for code.** New TensorFlow modeling and
  agentic-workflow code belongs here, fitted into the existing
  `financial_forecast/` package layout (see below) — not bolted on as
  standalone scripts unless that matches an existing pattern.
- **The proposal is not checked into this repo.** It lives in the
  `jaybhum/JPM_internship` fork. When a task refers to "the proposal," ask the
  user to paste the relevant section if you need its details; do not assume its
  contents.
- Work flows **proposal → implementation**: the fork describes intent, this
  repo realizes it. If you change the design here in a way that diverges from
  the proposal, flag it so the proposal can be kept in sync.

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
file so the next agent inherits an accurate picture.*
