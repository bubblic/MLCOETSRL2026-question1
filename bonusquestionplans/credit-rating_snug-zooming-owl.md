# Credit Rating Assignment System — Implementation Plan

## Context

The JP Morgan MLCOE internship Q1 report covers financial forecasting (Part 1) and automated extraction pipelines (Part 2). This plan implements a **new, standalone credit rating assignment system** — a separate `credit_rating/` package that coexists with the existing `financial_forecast/` package. The system ingests corporate annual reports and produces credit ratings (AAA-D) plus a financial shenanigans detection report.

**Why a separate package:** The existing `financial_forecast/` handles time-series balance sheet forecasting. The credit rating system solves a different problem — classification from annual reports. Different domain types, different data flows, different model architectures. Separate packages keep each cohesive.

**What we reuse from existing code:** The TensorFlow framework, conventions (Google-style docstrings, type hints, Protocol-based interfaces, pathlib everywhere, `tf.Module`/`tf.keras.Model` patterns), and the existing `pdfplumber` PDF extraction approach. We build fresh domain types tuned for credit rating classification.

---

## IMPLEMENTATION PLAN

### Phase 1 — Foundation (no ML dependencies)
*Goal: Domain model, configuration, and data contracts that everything else builds on.*

| # | File | Est. Lines | Purpose |
|---|------|-----------|---------|
| 1 | `credit_rating/__init__.py` | 5 | Package marker with version |
| 2 | `credit_rating/config/__init__.py` | 1 | |
| 3 | `credit_rating/config/settings.py` | ~220 | `CreditRatingSettings(BaseSettings)` — all paths, hyperparams, coefficients, thresholds |
| 4 | `credit_rating/domain/__init__.py` | 1 | |
| 5 | `credit_rating/domain/financial_statements.py` | ~120 | `IncomeStatement`, `BalanceSheet`, `CashFlowStatement`, `FinancialStatements` frozen dataclasses |
| 6 | `credit_rating/domain/features.py` | ~220 | 5 ratio-group dataclasses + `FinancialRatios` composite with `__iter__`, `__len__`, `to_tensor()`, `__slots__` |
| 7 | `credit_rating/domain/report.py` | ~50 | `AnnualReport` dataclass |
| 8 | `credit_rating/domain/rating.py` | ~110 | `RatingClass` Enum (7 classes), `RatingPrediction` dataclass |
| 9 | `credit_rating/domain/shenanigans.py` | ~200 | Signal/score/report dataclasses, `RiskLevel` Enum |

**Rationale for ordering:** Settings first (everything references it for constants/Enums). Domain types second (every downstream module imports them). No ML dependencies in this phase — only stdlib + pydantic.

### Phase 2 — Data Ingestion & Feature Engineering
*Goal: Get data in, compute features, normalize them.*

| # | File | Est. Lines | Purpose |
|---|------|-----------|---------|
| 10 | `credit_rating/ingestion/__init__.py` | 1 | |
| 11 | `credit_rating/ingestion/protocols.py` | ~40 | `AnnualReportParser`, `FinancialDataSource` Protocols |
| 12 | `credit_rating/ingestion/kaggle_loader.py` | ~120 | CSV loader → `Iterator[(FinancialRatios, RatingClass)]` |
| 13 | `credit_rating/ingestion/sec_edgar.py` | ~220 | `EdgarDownloader` with `@throttle` decorator, XBRL + regex extraction |
| 14 | `credit_rating/ingestion/pdf_parser.py` | ~200 | `PdfAnnualReportParser` using pdfplumber + camelot fallback |
| 15 | `credit_rating/features/__init__.py` | 1 | |
| 16 | `credit_rating/features/ratio_calculator.py` | ~280 | 25 ratios across 5 dimensions, pure functions, `@lru_cache`-ready |
| 17 | `credit_rating/features/altman.py` | ~120 | Z-Score, Z''-Score, zone classification |
| 18 | `credit_rating/features/normalizer.py` | ~130 | `RobustScaler` wrapper with fit/transform/persist |

### Phase 3 — Classical Models
*Goal: Baseline models that don't require deep learning.*

| # | File | Est. Lines | Purpose |
|---|------|-----------|---------|
| 19 | `credit_rating/models/__init__.py` | 1 | |
| 20 | `credit_rating/models/protocols.py` | ~35 | `CreditRatingModel`, `Trainable` Protocols |
| 21 | `credit_rating/models/ordinal_logistic.py` | ~160 | Proportional-odds model via `mord`, sklearn-compatible |

### Phase 4 — Deep Learning Models
*Goal: Two-tower hybrid architecture with ordinal loss.*

| # | File | Est. Lines | Purpose |
|---|------|-----------|---------|
| 22 | `credit_rating/models/structured_tower.py` | ~90 | `tf.keras.Model` MLP: d→256→256→128, BatchNormalization + Dropout |
| 23 | `credit_rating/models/text_tower.py` | ~140 | `TFAutoModel` (FinBERT) + attention pooling + Dense 768→128 |
| 24 | `credit_rating/models/fusion_head.py` | ~70 | `tf.keras.Model`: Concat 256→256→7, Dropout before final Dense |
| 25 | `credit_rating/models/hybrid.py` | ~220 | `tf.keras.Model` composing towers + fusion, `tf.train.Checkpoint` I/O |
| 26 | `credit_rating/models/loss.py` | ~70 | `tf.keras.losses.Loss` subclass: ordinal CE + distance penalty (lambda=0.1) |

### Phase 5 — Training Pipeline
*Goal: Dataset, trainer with staged fine-tuning, metrics, temporal CV.*

| # | File | Est. Lines | Purpose |
|---|------|-----------|---------|
| 27 | `credit_rating/training/__init__.py` | 1 | |
| 28 | `credit_rating/training/dataset.py` | ~130 | `tf.data.Dataset` pipeline with `.map()`, `.batch()`, `.prefetch()` |
| 29 | `credit_rating/training/trainer.py` | ~270 | `ModelTrainer` — `tf.keras.optimizers.AdamW`, differential LR via `tf.GradientTape`, early stopping, staged training |
| 30 | `credit_rating/training/metrics.py` | ~110 | accuracy, F1, MAE, Spearman rho, IG binary accuracy, confusion matrix |
| 31 | `credit_rating/training/cross_validator.py` | ~90 | `TemporalCrossValidator` — walk-forward folds |

### Phase 6 — Shenanigans Toolkit
*Goal: Full financial fraud/manipulation detection suite.*

| # | File | Est. Lines | Purpose |
|---|------|-----------|---------|
| 32 | `credit_rating/shenanigans/__init__.py` | 1 | |
| 33 | `credit_rating/shenanigans/beneish.py` | ~220 | 8 index variables + M-Score + classification |
| 34 | `credit_rating/shenanigans/earnings_manipulation.py` | ~320 | EMS-1 through EMS-8, one method each |
| 35 | `credit_rating/shenanigans/cash_flow.py` | ~160 | CFS-1 through CFS-4 |
| 36 | `credit_rating/shenanigans/text_signals.py` | ~210 | Fog index, FinBERT tone shift, TF-IDF boilerplate |
| 37 | `credit_rating/shenanigans/report_builder.py` | ~130 | Builder pattern → `ShenanigansReport` |

### Phase 7 — Explainability
*Goal: SHAP for structured features, integrated gradients for text.*

| # | File | Est. Lines | Purpose |
|---|------|-----------|---------|
| 38 | `credit_rating/explainability/__init__.py` | 1 | |
| 39 | `credit_rating/explainability/shap_explainer.py` | ~110 | `shap.GradientExplainer` (TF-compatible) wrapper + waterfall plot |
| 40 | `credit_rating/explainability/attention_visualizer.py` | ~110 | `tf.GradientTape` integrated gradients on FinBERT embeddings → `TokenImportance` |

### Phase 8 — Evaluation & Case Studies
*Goal: Backtesting, bankruptcy validation, Evergrande & Enron analyses.*

| # | File | Est. Lines | Purpose |
|---|------|-----------|---------|
| 41 | `credit_rating/evaluation/__init__.py` | 1 | |
| 42 | `credit_rating/evaluation/backtester.py` | ~210 | Full pipeline backtesting + migration matrices + ROC-AUC |
| 43 | `credit_rating/evaluation/bankruptcy_validator.py` | ~160 | 3-year pre-bankruptcy signal timelines |
| 44 | `credit_rating/case_studies/__init__.py` | 1 | |
| 45 | `credit_rating/case_studies/evergrande.py` | ~160 | PDF → full pipeline → JSON + markdown report |
| 46 | `credit_rating/case_studies/enron.py` | ~160 | 10-K → full pipeline → JSON + markdown report |

### Phase 9 — API & CLI
*Goal: REST API and command-line interface.*

| # | File | Est. Lines | Purpose |
|---|------|-----------|---------|
| 47 | `credit_rating/api/__init__.py` | 1 | |
| 48 | `credit_rating/api/main.py` | ~65 | FastAPI app with lifespan context manager |
| 49 | `credit_rating/api/schemas.py` | ~110 | Pydantic request/response models |
| 50 | `credit_rating/api/dependencies.py` | ~55 | DI for model + detector singletons |
| 51 | `credit_rating/api/routers/__init__.py` | 1 | |
| 52 | `credit_rating/api/routers/rating.py` | ~55 | `POST /rate` |
| 53 | `credit_rating/api/routers/shenanigans.py` | ~55 | `POST /analyze` |
| 54 | `credit_rating/api/routers/explain.py` | ~55 | `POST /explain` |
| 55 | `credit_rating/cli/__init__.py` | 1 | |
| 56 | `credit_rating/cli/main.py` | ~160 | `typer` CLI: rate, analyze, train, backtest, case-study |

### Phase 10 — Tests
*Goal: >=85% coverage on all modules except api/ and cli/.*

| # | File | Est. Lines | Purpose |
|---|------|-----------|---------|
| 57 | `tests/credit_rating/__init__.py` | 0 | |
| 58 | `tests/credit_rating/conftest.py` | ~120 | Shared fixtures: synthetic statements, ratios, settings |
| 59 | `tests/credit_rating/unit/test_settings.py` | ~60 | Settings validation, env override |
| 60 | `tests/credit_rating/unit/test_domain.py` | ~150 | All dataclass construction, enum parsing, to_tensor |
| 61 | `tests/credit_rating/unit/test_ratio_calculator.py` | ~200 | Every ratio formula against hand-calculated values |
| 62 | `tests/credit_rating/unit/test_altman.py` | ~80 | Z, Z'', zone boundaries |
| 63 | `tests/credit_rating/unit/test_normalizer.py` | ~80 | Fit/transform/inverse roundtrip |
| 64 | `tests/credit_rating/unit/test_beneish.py` | ~150 | All 8 M-Score variables |
| 65 | `tests/credit_rating/unit/test_earnings_manipulation.py` | ~200 | EMS-1 through EMS-8 |
| 66 | `tests/credit_rating/unit/test_cash_flow_shenanigans.py` | ~120 | CFS-1 through CFS-4 |
| 67 | `tests/credit_rating/unit/test_text_signals.py` | ~100 | Fog index, tone shift, boilerplate |
| 68 | `tests/credit_rating/unit/test_loss.py` | ~80 | Ordinal CE loss value checks + `tf.GradientTape` gradient checks |
| 69 | `tests/credit_rating/unit/test_metrics.py` | ~100 | All metric computations |
| 70 | `tests/credit_rating/integration/test_feature_pipeline.py` | ~100 | statements → ratios → normalized |
| 71 | `tests/credit_rating/integration/test_model_pipeline.py` | ~120 | ratios → model → prediction |
| 72 | `tests/credit_rating/integration/test_shenanigans_pipeline.py` | ~100 | statements → all detectors → report |
| 73 | `tests/credit_rating/case_studies/test_evergrande.py` | ~60 | Smoke test: predicted D |
| 74 | `tests/credit_rating/case_studies/test_enron.py` | ~60 | Smoke test: M-Score > -2.22 |

**Total: ~74 files, ~5,800 lines production code, ~1,780 lines tests**

---

### Dependency Graph

```
                    config/settings
                         |
                    domain/*  (statements, features, rating, shenanigans, report)
                   /    |    \         \
             ingestion  features  shenanigans
              /    \       |        /   |   \
         kaggle  edgar  ratio_calc  beneish  ems  cfs  text_signals
           |     pdf     altman      \       |    /       |
           |            normalizer    report_builder
           |               \            /
           |            models/protocols
           |           /       |        \
           |    ordinal_log  structured  text_tower
           |                  tower     /
           |                   \       /
           |                  fusion_head
           |                      |
           |                   hybrid
           |                  /      \
           |            training    explainability
           |          /    |    \       |
           |     dataset trainer metrics  shap  attention
           |              |      |
           |        cross_validator
           |                |
           |           evaluation
           |          /         \
           |    backtester  bankruptcy_validator
           |         \         /
           |       case_studies
           |      /           \
           |  evergrande     enron
           |       \          /
            \    api/  +  cli/
             \  (FastAPI) (typer)
```

**All arrows point downward. No circular dependencies.**

---

### Key Design Decisions

1. **Separate `credit_rating/` package** — The existing `financial_forecast/` is TF-based forecasting with fundamentally different domain types. Merging would create coupling with no benefit. The two packages share a repo but are independent installable packages.

2. **TensorFlow for all deep learning** — Consistent with the existing `financial_forecast/` package. The structured tower and fusion head use `tf.keras.Model` with `tf.keras.layers` (Dense, BatchNormalization, Dropout). The text tower wraps FinBERT via `transformers.TFAutoModel`. The ordinal loss is a custom `tf.keras.losses.Loss` subclass. Training uses `tf.GradientTape` for fine-grained control (matching the pattern in `financial_forecast/training/`). `FinancialRatios.to_tensor()` returns `tf.Tensor`.

3. **Frozen dataclasses for domain types** — Financial statement data should never be mutated after construction. `@dataclass(frozen=True)` catches accidental mutation at runtime. `__slots__` on `FinancialRatios` for memory efficiency when processing thousands of companies.

4. **Protocol over ABC everywhere** — `AnnualReportParser`, `CreditRatingModel`, `Trainable`, `FinancialDataSource` are all `Protocol` classes. The SEC EDGAR parser and PDF parser satisfy `AnnualReportParser` without inheriting from it — structural subtyping per Fluent Python.

5. **Single `pydantic.BaseSettings` for all configuration** — Every coefficient (Altman, Beneish), threshold (Fog >18, M-Score <-2.22, boilerplate >0.90), hyperparameter (batch size, LR, dropout), and path comes from one validated settings object. Overridable via `.env`. No magic numbers anywhere in code.

6. **Ordinal cross-entropy loss with distance penalty** — Standard CE treats AAA→A and AAA→D as equally wrong. The ordinal distance penalty (lambda=0.1) penalizes predictions proportionally to their distance from the true rating notch. This is critical for credit rating quality.

7. **Staged fine-tuning for FinBERT** — Phase 1: freeze BERT backbone, train only the MLP structured tower and fusion head on Kaggle structured data. Phase 2: unfreeze BERT, train jointly with lower learning rate on BERT parameters. Prevents catastrophic forgetting of pre-trained financial language knowledge.

8. **Temporal cross-validation** — Financial data has strong temporal dependencies. Walk-forward validation (train on years ≤T, validate on T+1, test on T+2) prevents future data leakage. No random shuffling across time.

9. **Builder pattern for ShenanigansReport** — The report aggregates outputs from 4 independent detector modules (Beneish, EMS, CFS, text signals). A builder pattern lets callers chain detectors in any order and produce a composite report. Cleaner than a god-function that runs everything.

10. **API schemas separate from domain types** — `api/schemas.py` defines pydantic request/response models independent of `domain/` dataclasses. The API layer never leaks internal types. This lets the API evolve independently of the domain model.

11. **Kaggle dataset as primary training source** — The Kaggle Corporate Credit Ratings CSV (~2,029 US firms with S&P ratings + 30 financial features) is the structured data source. The 7-class mapping is: `{AAA,AA+,AA,AA-}→0, {A+,A,A-}→1, {BBB+,BBB,BBB-}→2, {BB+,BB,BB-}→3, {B+,B,B-}→4, {CCC+,CCC,CCC-,CC}→5, D→6`.

12. **`@throttle` decorator for SEC EDGAR** — EDGAR's rate limit is 10 requests/second. A decorator approach (not inline `time.sleep`) keeps the download logic clean and the rate-limiting concern separated.

---

### New Dependencies (to add to pyproject.toml)

```toml
[project.optional-dependencies]
credit-rating = [
    "tensorflow>=2.14",
    "transformers>=4.30",
    "mord>=0.7",
    "shap>=0.42",
    "fastapi>=0.100",
    "uvicorn>=0.23",
    "typer>=0.9",
    "wandb>=0.15",
    "scikit-learn>=1.3",
    "sec-edgar-downloader>=5.0",
    "camelot-py[cv]>=0.11",
    "pdfplumber>=0.10",
    "pandas>=2.0",
]
```

Note: `tensorflow` is already a core dependency. `transformers` provides `TFAutoModel` for FinBERT. `captum` removed (PyTorch-only) — integrated gradients implemented directly via `tf.GradientTape`.

---

### Verification Plan

1. **Unit tests first:** `pytest tests/credit_rating/unit/ -v` — every ratio, coefficient, loss function
2. **Integration tests:** `pytest tests/credit_rating/integration/ -v` — end-to-end pipelines with mocked I/O
3. **Coverage check:** `pytest --cov=credit_rating --cov-report=term-missing` — target >=85%
4. **Type checking:** `mypy credit_rating/ --strict`
5. **Smoke test the CLI:** `python -m credit_rating.cli.main rate --ticker AAPL --year 2023`
6. **Case study validation:** Run Evergrande analysis, verify D rating output; run Enron analysis, verify M-Score > -2.22
7. **API smoke test:** `uvicorn credit_rating.api.main:app` then `curl -X POST /rate` with test PDF

---

### Implementation Order Summary

**Phase 1** (Foundation): settings → domain types (7 files)
**Phase 2** (Ingestion + Features): protocols → kaggle → edgar → pdf → ratios → altman → normalizer (8 files)
**Phase 3** (Classical): model protocols → ordinal logistic (2 files)
**Phase 4** (Deep Learning): structured tower → text tower → fusion → hybrid → loss (5 files)
**Phase 5** (Training): dataset → trainer → metrics → cross-validator (4 files)
**Phase 6** (Shenanigans): beneish → EMS → CFS → text signals → report builder (5 files)
**Phase 7** (Explainability): SHAP → attention (2 files)
**Phase 8** (Evaluation): backtester → bankruptcy validator → case studies (4 files)
**Phase 9** (API + CLI): FastAPI app + routers + schemas + typer CLI (8 files)
**Phase 10** (Tests): All test files (18 files)

Each phase depends only on phases above it. Within a phase, files are ordered by dependency.
