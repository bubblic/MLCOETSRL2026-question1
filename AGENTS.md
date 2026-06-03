# AGENTS.md

> Orientation for AI coding agents (Claude Code, Codex, Cursor, and any future
> agent). **This file is committed identically to both repositories of this
> project** (see below), so it must stay identity-neutral: **detect your
> context (Step 0) before acting — never assume it from this text.** Human
> contributors should start with [`README.md`](README.md) for build/run/test
> instructions.

## TL;DR — two repositories, two modes, one shared codebase

This project spans **two GitHub repositories linked by a fork**, and this
`AGENTS.md` lives in both:

| Repository | Role |
| --- | --- |
| `bubblic/MLCOETSRL2026-question1` *(upstream original)* | **Code** — the TensorFlow simulator and the agentic-workflow software the proposal describes; this is where code gets **built**. |
| `jaybhum/JPM_internship` *(a fork of the above)* | **Proposal** — where a six-month JPMorgan MLCOE internship proposal is researched and authored, on top of the simulator. |

Author: Jaebum "Albert" Chung (JP Morgan MLCOE TSRL 2026 Internship, Question 1).
The Proposal repo defines intent through a **Research → Plan → Write** workflow;
the Code repo **builds** that intent — but only **once it is signed off**.

### Terminology (read once — these are deliberately non-overlapping)

- **Proposal repo / Proposal mode** — authoring the internship proposal (in the
  fork). Its phases are **Research → Plan → Write**.
- **Code repo / Code mode** — the software (this simulator + the proposal's
  agentic workflow). The activity here is **build**.
- **"Write"** is a *proposal phase* (produce the **document**). **"Build"** is
  *software work*. They are different activities in different places — **do not
  conflate them.** We deliberately avoid the word *"implement,"* which used to
  mean both and caused confusion.

## Step 0 — detect your context

Two independent signals. Check **both**.

**(a) Which repository?** — the origin remote URL (branch *names* won't tell you
this; both repos can have a `main`):

```
git remote -v          # or:  git config --get remote.origin.url
```

- origin contains **`bubblic/MLCOETSRL2026-question1`** → **Code repo**.
- origin contains **`jaybhum/JPM_internship`** → **Proposal repo (fork)**.
- anything else (another fork, mirror, or fresh clone) → don't guess; ask the user.

**(b) Which mode?** — the current branch:

```
git branch --show-current
```

- branch matches **`claude/jpm-internship-proposal-*`** → **Proposal mode**
  (see *Proposal mode* below).
- otherwise → **Code mode** (see *The shared codebase* and *Code mode*).

> **Why both signals, and why identity-neutral:** because the fork merges back
> into the Code repo, shared files like this one must be identical in both.
> Hardcoding "this repo is X" or "we are in the Research phase" makes the file
> wrong the moment it lands in the other repo or the phase advances. Describe
> every role, detect at runtime, and assert nothing.

## The shared codebase (present in both repos)

A TensorFlow financial-statement simulator and related pipelines. Not a blank
slate — check for an existing module before adding anything. Full file-by-file
tree is in [`README.md`](README.md); the shape:

- **`financial_forecast/`** — the **real, validated core**. Bayesian financial
  forecasting for Apple (FY2018–FY2025 historicals), built from composable policy
  modules (dividends, buybacks, debt, OpEx, tax, working capital…), a `training/`
  pipeline, and `inference/` simulators (deterministic + Monte Carlo). It also
  contains the existing **LLM / agentic pieces** the proposal's workflow builds
  on: `clients/` (Azure LLM client), `extraction/` (multi-stage LLM pipeline that
  pulls financial statements and tax anomalies out of 10-K PDFs),
  `reporting/advisor.py` (`DeepseekCEOAdvisor`), and `models/llm_forecaster.py`.
- **`loan_pricing/`, `credit_rating/`, `risk/`** — **exploratory sketches**, not
  production. Do not treat them as validated; `risk/` also uses LLMs. Design notes
  live in `bonusquestionplans/`.
- **`run_*.py`** — top-level entry-point scripts, each runnable from the repo root.
- **Reports** — LaTeX sources + compiled PDFs at the root (`*_report.tex` / `*.pdf`);
  regenerate the PDF when you change the `.tex`.
- **`tests/`** — pytest suite; keep it green.

## Proposal mode  *(Proposal fork · `claude/jpm-internship-proposal-*` branches)*

If Step 0 put you here, you are helping develop the internship proposal — **read
this first**:

- **Single source of truth:** [`proposal/README.md`](proposal/README.md), then
  `proposal/RESEARCH.md`. That directory governs all proposal work.
- **Workflow:** Research → Plan → Write. Last known phase is **Research**
  — confirm the current phase via `proposal/README.md`.
- **Do not** write or overhaul a polished proposal document, and **do not** start
  building software, until research and planning are **explicitly signed off by
  Jaebum**. (An earlier polished draft was written prematurely and has been
  retired into `proposal/RESEARCH.md`.)
- The `proposal/` directory is **not** present in the Code repo; it lives only in
  the fork.

## Code mode  *(Code repo · code branches)*

If Step 0 put you here, this is where the proposal's TensorFlow program + agentic
workflow get **built** — **but building is gated**:

- **Net-new build work for the proposal's design is GATED on sign-off.** It waits
  until Research → Plan is explicitly signed off by Jaebum. Check
  `proposal/RESEARCH.md` in the fork (`jaybhum/JPM_internship`) for the current
  phase, and confirm with Jaebum before starting net-new feature work.
- **Ordinary maintenance of the existing codebase** — bug fixes, tests, refactors,
  report regeneration, docs — is fine anytime and is not gated.
- **Once building is greenlit:** fit new code into the existing
  `financial_forecast/` layout (mirror its composable-OOP style), expose pipelines
  as top-level `run_*.py` entry points, and add tests alongside.

## Conventions (all agents, both repos)

- **Deep learning is TensorFlow, always** (TensorFlow / TensorFlow-Probability —
  see `pyproject.toml`). Do **not** introduce PyTorch, JAX, or another DL framework.
- **Entry points run from the repo root** and write reports to `training_results/`
  and `outputs/`. Prefer a new top-level `run_*.py` over `python -m ...`.
- **Install editable:** `pip install -e .` (or `pip install -e ".[dev]"` for
  pytest). Optional extras: `loan-pricing`, `credit-rating`, `risk`, `data` —
  see `pyproject.toml`.
- **Tests:** `python -m pytest tests/ -v`.
- **Style:** match the surrounding code — clean, composable OOP (ABCs + small
  policy classes).

## Secrets & environment

- LLM pipelines need `AZURE_DEEPSEEK_ENDPOINT` (and related credentials), supplied
  via a local `.env` (git-ignored) or the shell environment.
- **`.env` is never committed.** Never hard-code endpoints, keys, or tokens into
  source or into this file.

## Pointers

- [`README.md`](README.md) — install, every `run_*.py` script, full tree, Windows/CUDA notes.
- [`pyproject.toml`](pyproject.toml) — dependencies, optional extras, tooling.
- `proposal/README.md` → `proposal/RESEARCH.md` — proposal source of truth *(fork only)*.
- `bonusquestionplans/` — design notes for the exploratory sketch packages.
- `final_interview_prep/` — study/interview notes (context, not code).

---
*This file is identity-neutral by design and lives in both repos. If the
two-repo arrangement, ownership, or workflow changes, update it in **both** repos
so the next agent inherits an accurate picture.*
