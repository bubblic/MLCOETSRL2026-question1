# Term Loan Pricing Model — Implementation Plan

## Context

This is the **bonus question** for the JP Morgan MLCOE internship (Q1). It builds a quantitative
term loan pricing model that recommends the credit spread `s` on a new term loan via:

```
r_loan = r_f(T) + s
```

The work lives entirely under `loan_pricing/` as a sibling package to the existing `financial_forecast/`
(the main Q1 deliverable). No existing files are modified except `pyproject.toml` and `.gitignore`.

## Structure Mapping (plan → repo)

| Original plan path | Actual repo path |
|---|---|
| `src/` | `loan_pricing/` |
| `data/raw/`, `data/processed/` | `loan_pricing/artifacts/raw/`, `loan_pricing/artifacts/processed/` |
| `scripts/` | `loan_pricing/scripts/` |
| `tests/` | `tests/loan_pricing/` |

All imports: `from loan_pricing.config import ProjectConfig`, etc.

## Conventions (from existing codebase)

- `from __future__ import annotations` at top of every module
- Absolute imports only (`from loan_pricing.X import Y`)
- `@runtime_checkable` Protocol for interfaces (see `financial_forecast/clients/protocols.py`)
- Frozen dataclasses for immutable config/results
- Class-based test organization with fixtures
- Thin entry-point scripts (existing `run_*.py` pattern)

## Framework Decision: TensorFlow Throughout

All ML/statistical models use **TensorFlow + TensorFlow Probability**, consistent with
the existing `financial_forecast/` package. No XGBoost, no LightGBM, no raw scipy optimizers.

| Component | TF approach | Key TF APIs |
|---|---|---|
| PD classifier | `tf.keras.Model` subclass, custom training loop | `tf.GradientTape`, `tf.keras.layers.Dense`, `tf.keras.losses.BinaryCrossentropy` |
| Spread regressor | `tf.keras.Model` subclass, custom training loop | `tf.keras.losses.MeanSquaredError` |
| Quantile regressor | `tf.keras.Model` with pinball loss | Custom `pinball_loss(alpha)` via TF ops |
| OU calibration | `tf.GradientTape` MLE optimization | `tfp.distributions.Normal`, `tf.optimizers.Adam` |
| Monte Carlo | Vectorized TF simulation | `tf.random.stateless_normal`, `tf.while_loop` or tensor ops |
| Interpretability | SHAP `GradientExplainer` (supports TF natively) | `shap.GradientExplainer` |
| Serialization | `.npz` for parameters (matches existing pattern) | `np.savez` / `np.load` with `var.numpy()` |
| dtype | `tf.float64` everywhere (matches existing pattern) | — |

**Composition pattern**: Model classes keep the `BaseClassifier` / `BaseRegressor` ABC
interface. Internally they compose a `tf.keras.Model` — the class HAS a keras model,
not IS one. This follows the existing codebase's preference for composition and keeps
the public API framework-agnostic.

**What stays non-TF** (evaluation utilities, not models):
- `sklearn.model_selection.StratifiedKFold` — CV splitting utility
- `sklearn.metrics` — ROC-AUC, PR-AUC, Brier score, calibration curve
- `optuna` — hyperparameter tuning (framework-agnostic)
- `shap` — interpretability (has native TF support)

---

## Implementation Status

- **Phase 0**: DONE (scaffolding, config, exceptions, logging, conftest, scripts, pyproject)
- **Phase 1**: DONE (protocols, fetch_fred, fetch_sec, preprocess — 24 tests pass)
- **Phase 2–8**: Not started — Phases 3–6 revised below to use TensorFlow

**Files to update before Phase 2**: `requirements.txt` — remove xgboost/lightgbm, add
tensorflow/tensorflow-probability (the file was already written with the old deps).

---

## PHASE 0 — Project Scaffolding (DONE)

### 0.1 Directory tree & `__init__.py` files

Create (most already exist from prior scaffolding step):

```
loan_pricing/
├── __init__.py              # __version__ = "0.1.0"
├── config.py
├── exceptions.py
├── logging_config.py
├── data/
│   ├── __init__.py
│   ├── protocols.py
│   ├── fetch_fred.py
│   ├── fetch_sec.py
│   └── preprocess.py
├── features/
│   ├── __init__.py
│   └── engineering.py
├── models/
│   ├── __init__.py
│   ├── pd_model.py
│   ├── spread_model.py
│   ├── quantile_model.py
│   └── ou_calibration.py
├── reporting/
│   ├── __init__.py
│   ├── figures.py
│   └── tables.py
├── scripts/
│   ├── generate_all.py
│   ├── 01_eda.py
│   ├── 02_pd_model_eval.py
│   ├── 03_spread_model_eval.py
│   ├── 04_hyperparameter_tuning.py
│   ├── 05_private_borrower_example.py
│   ├── 06_monte_carlo_forecast.py
│   ├── 07_uncertainty.py
│   └── 08_summary.py
└── artifacts/
    ├── raw/
    │   ├── fred/
    │   └── sec/
    └── processed/
        ├── figures/
        ├── tables/
        └── models/

tests/loan_pricing/
├── __init__.py
├── conftest.py
├── test_preprocess.py
├── test_engineering.py
├── test_pd_model.py
├── test_spread_model.py
├── test_quantile_model.py
├── test_ou_calibration.py
├── test_fetch_fred.py
└── test_reporting.py
```

### 0.2 `requirements.txt`

New file at repo root with pinned versions:
`tensorflow`, `tensorflow-probability`, `pandas`, `numpy`, `scipy`,
`scikit-learn`, `shap`, `optuna`, `fredapi`, `sec-edgar-downloader`,
`pydantic`, `pytest`, `pytest-mock`, `pytest-cov`, `ruff`, `black`,
`mypy`, `matplotlib`, `seaborn`, `statsmodels`, `typing_extensions`, `joblib`.
No `jupyter`, `ipykernel`, `xgboost`, or `lightgbm`.

### 0.3 `loan_pricing/config.py`

Frozen dataclass `ProjectConfig` with all configurable constants:
- Directory paths (all under `loan_pricing/artifacts/`)
- `fred_api_key_env_var: str = "FRED_API_KEY"`
- `random_seed: int = 42`
- `confidence_levels: tuple[float, ...] = (0.80, 0.95)`
- `monte_carlo_n_simulations: int = 10_000`
- `min_ou_series_length: int = 60`
- `cross_validation_folds: int = 5`
- `test_fraction: float = 0.15`
- `validation_fraction: float = 0.15`

Uses `pathlib.Path` for all paths. Root derived from `Path(__file__).resolve().parent`.

### 0.4 `loan_pricing/exceptions.py`

Three custom exceptions, all inheriting from a base `LoanPricingError`:
- `DataFetchError` — API/network failures
- `InsufficientDataError` — too few observations
- `DataLeakageError` — fit called on non-training data

### 0.5 `loan_pricing/logging_config.py`

`get_logger(name: str) -> logging.Logger` factory with:
- StreamHandler to stderr
- Format: `"%(asctime)s | %(name)s | %(levelname)s | %(message)s"`
- Default level: `INFO`

Every module uses `logger = get_logger(__name__)` instead of `logging.getLogger`.

### 0.6 `loan_pricing/__init__.py`

Already created: exposes `__version__ = "0.1.0"`.

### 0.7 `tests/loan_pricing/conftest.py`

Shared fixtures:
- `synthetic_loan_df` — 50-row DataFrame with realistic columns
- `synthetic_fred_df` — long-format FRED data
- `synthetic_sec_df` — long-format SEC XBRL data
- `tmp_artifacts_dir` — temp directory mimicking artifacts/ structure
- `project_config` — ProjectConfig pointing at tmp dirs

### 0.8 `loan_pricing/scripts/generate_all.py`

Imports each script's `main()` and calls them in order (01–08).
Runnable as `python -m loan_pricing.scripts.generate_all` or directly.

### 0.9 Update `pyproject.toml`

Add `loan_pricing` to `[tool.setuptools.packages.find]`.
Add loan pricing deps to `[project.optional-dependencies.loan-pricing]`.
Add `tests/loan_pricing` to pytest testpaths.

### 0.10 Update `.gitignore`

Add:
```
loan_pricing/artifacts/raw/
loan_pricing/artifacts/processed/
```

---

## PHASE 1 — Data Ingestion (`loan_pricing/data/`)

### 1.1 `loan_pricing/data/protocols.py`

```python
@runtime_checkable
class DataFetcher(Protocol):
    def fetch(self, series_ids: Sequence[str], start_date: str, end_date: str) -> pd.DataFrame: ...
```

### 1.2 `loan_pricing/data/fetch_fred.py`

`FredDataFetcher` class satisfying `DataFetcher` protocol:
- Returns tidy DataFrame: `[date, series_id, value]`
- Fetches: DGS1, DGS2, DGS5, DGS10, DGS30, BAMLC0A0CM, BAMLH0A0HYM2, VIXCLS, T10Y2Y, UNRATE, A191RL1Q225SBEA
- Caches to `artifacts/raw/fred/` as CSV; 24h TTL
- 3 retries with exponential backoff; raises `DataFetchError` on failure
- Dates as `datetime.date`, never strings

### 1.3 `loan_pricing/data/fetch_sec.py`

`SecEdgarFetcher` satisfying `DataFetcher` protocol:
- Downloads from `https://data.sec.gov/api/xbrl/companyfacts/`
- Returns tidy DataFrame: `[cik, ticker, fiscal_year, tag, value]`
- XBRL tags: Revenues, GrossProfit, OperatingIncomeLoss, NetIncomeLoss, Assets, Liabilities, StockholdersEquity, RetainedEarningsAccumulatedDeficit, CashAndCashEquivalentsAtCarryingValue, LongTermDebt, InterestExpense, DepreciationAndAmortization
- Caches raw JSON to `artifacts/raw/sec/` by CIK
- Missing tags → NaN (no raise)
- Accepts list of CIKs

### 1.4 `loan_pricing/data/preprocess.py`

`LoanDataPreprocessor` with independently-testable functions:
1. `pivot_fred_series(df) -> pd.DataFrame` — long to wide
2. `compute_financial_ratios(df) -> pd.DataFrame` — debt_to_ebitda, interest_coverage_ratio, net_debt_to_equity, fcf_to_debt, revenue_growth_yoy, ebitda_margin, current_ratio, altman_z_double_prime
3. `merge_loan_features(loans_df, ratios_df, fred_df) -> pd.DataFrame` — left-join on borrower + date, match Treasury yield to maturity bucket
4. `handle_missing_values(df, strategy: Literal["median", "knn"]) -> pd.DataFrame`
5. `split_train_val_test(df, val_frac, test_frac, time_column) -> tuple[...]` — time-aware, no future leakage

### 1.5 Tests: `tests/loan_pricing/test_preprocess.py`, `tests/loan_pricing/test_fetch_fred.py`

- Preprocessing: each function with 50-row synthetic DataFrame; assert shapes, column names, no NaN leakage
- FRED: cache hit logic (mock HTTP); `DataFetchError` after 3 retries

---

## PHASE 2 — Feature Engineering (`loan_pricing/features/engineering.py`)

`FeatureEngineer` class with frozen `FeatureConfig` dataclass:
- `fit(df)` — learns normalisation stats on training set only
- `transform(df) -> tuple[np.ndarray, np.ndarray]` — applies learned transforms
- `fit_transform(df)` — convenience
- Returns feature names alongside arrays (for SHAP)
- Raises `DataLeakageError` if `fit` called again with different data

Interaction terms:
- `debt_to_ebitda * vix`
- `maturity_years * hy_oas`
- `interest_coverage_ratio * treasury_yield`

Tests: `tests/loan_pricing/test_engineering.py`
- fit on train only; transform on val gives same columns; interaction terms correct

---

## PHASE 3 — PD Model (`loan_pricing/models/pd_model.py`)

Abstract base class `BaseClassifier(abc.ABC)`:
- `fit(X, y) -> BaseClassifier`
- `predict_proba(X) -> np.ndarray`

Internal Keras network (private, built by `_build_network`):
```
Input → Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.2) → Dense(1, sigmoid)
```
All layers use `dtype=tf.float64`.

Concrete `TFPDClassifier(BaseClassifier)`:
- Composes a `tf.keras.Model` internally (not inherits)
- Accepts frozen `PDModelConfig` dataclass (hidden_sizes, dropout_rate, learning_rate,
  epochs, batch_size, early_stopping_patience)
- Custom training loop with `tf.GradientTape` + `tf.keras.losses.BinaryCrossentropy`
- `StratifiedKFold` CV during fit (sklearn utility for splitting, TF for training)
- `.feature_importances_` computed via `shap.GradientExplainer` (SHAP supports TF natively)
- `.calibration_curve_` → `CalibrationResult` named tuple (via `sklearn.calibration`)
- `save(path)` / `load(path)` — serialise weights to `.npz` (matching existing pattern
  in `financial_forecast/serialization/parameter_io.py`), architecture is rebuilt from config

`evaluate_pd_model(model, X_test, y_test) -> PDEvaluationResult`:
- Frozen dataclass: roc_auc, pr_auc, brier_score, ks_statistic, gini_coefficient
- Uses `sklearn.metrics` for evaluation (standard practice, not model code)

Tests: `tests/loan_pricing/test_pd_model.py`
- predict_proba output in [0,1]; ROC-AUC > 0.5 on synthetic data; save/load round-trip

---

## PHASE 4 — Spread Model (`loan_pricing/models/spread_model.py`)

`BaseRegressor(abc.ABC)`:
- `fit(X, y) -> BaseRegressor`
- `predict(X) -> np.ndarray`

Internal Keras network:
```
Input → Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dense(1, linear)
```

Concrete `TFSpreadRegressor(BaseRegressor)`:
- Composes a `tf.keras.Model` internally
- Takes `pd_model: BaseClassifier` via constructor (DI)
- During `fit` and `predict`, calls `pd_model.predict_proba(X)` and appends the
  estimated PD as an additional feature column — never re-trains the PD model
- Custom training loop: `tf.GradientTape` + `tf.keras.losses.MeanSquaredError`
- `.shap_values(X) -> np.ndarray` via `shap.GradientExplainer`
- `save(path)` / `load(path)` via `.npz`

`evaluate_spread_model(model, X_test, y_test) -> SpreadEvaluationResult`:
- rmse_bps, mae_bps, r_squared, mape, median_absolute_error_bps

`LoanPricer` — public API (unchanged):
- Constructor: pd_model, spread_model, quantile_models, feature_engineer
- `price(LoanPricingInput) -> LoanPricingOutput`
- `LoanPricingInput` / `LoanPricingOutput` — frozen dataclasses

Tests: `tests/loan_pricing/test_spread_model.py`
- Spreads positive; LoanPricer.price returns valid output; CI lower < point < CI upper

---

## PHASE 5 — Quantile Model (`loan_pricing/models/quantile_model.py`)

`TFQuantileRegressor(BaseRegressor)`:
- Composes a `tf.keras.Model` internally
- Accepts explicit `quantile_alpha: float` parameter
- Custom **pinball loss** implemented in pure TF ops:
  ```python
  def pinball_loss(y_true, y_pred, alpha):
      error = y_true - y_pred
      return tf.reduce_mean(tf.maximum(alpha * error, (alpha - 1.0) * error))
  ```
- Trained with `tf.GradientTape` + `tf.optimizers.Adam`
- `coverage_test(X_cal, y_cal) -> float` — empirical coverage on calibration set

`ConformalPredictionInterval` (unchanged logic):
- Accepts fitted lower/upper `TFQuantileRegressor` pair + calibration set
- `calibrate()` → conformal correction factor `q_hat`
- `predict_interval(X) -> tuple[np.ndarray, np.ndarray]` — corrected bounds

Tests: `tests/loan_pricing/test_quantile_model.py`
- Empirical coverage >= 0.93 on held-out synthetic data

---

## PHASE 6 — OU Calibration & Monte Carlo (`loan_pricing/models/ou_calibration.py`)

`OrnsteinUhlenbeckCalibrator`:
- `fit(spread_series) -> OUParameters`
- **TF-based MLE**: Constructs the discrete-time OU log-likelihood using
  `tfp.distributions.Normal` and optimises with `tf.GradientTape` + `tf.optimizers.Adam`
- Parameters stored as `tfp.util.TransformedVariable` with `tfp.bijectors.Softplus()`
  to enforce positivity on kappa and sigma (matching existing pattern in
  `financial_forecast/models/capex.py`)
- Standard errors computed via the TF Hessian (`tf.hessians` or finite differences)
- Returns frozen `OUParameters` dataclass (kappa, theta, sigma, log_likelihood,
  se_kappa, se_theta, se_sigma)
- Raises `InsufficientDataError` if < 60 observations

`MonteCarloLoanPricer`:
- Constructor: ou_params, vasicek_params, n_simulations, horizon_years, random_seed
- `simulate(current_spread_bps, current_treasury_yield_pct, loan) -> MonteCarloResult`
- **Vectorised TF simulation**: uses `tf.random.stateless_normal` for reproducible
  draws and tensor ops for the OU/Vasicek Euler-Maruyama update (no Python for-loop)
- `MonteCarloResult`: simulated_prices (tf.Tensor → .numpy()), mean_price, std_price,
  ci_lower_95, ci_upper_95, ci_lower_80, ci_upper_80

Tests: `tests/loan_pricing/test_ou_calibration.py`
- MLE recovers known params on synthetic OU path within 2 SEs
- `InsufficientDataError` on short series

---

## PHASE 7 — Full Test Suite

All test files listed above, plus verify:
- All tests runnable offline (mock external HTTP)
- Total runtime < 60s
- `pytest --cov=loan_pricing` achieves >= 80% line coverage

---

## PHASE 8 — Reporting Layer

### `loan_pricing/reporting/figures.py`

Module-level: `matplotlib.use("Agg")`, `FIGURE_PALETTE` constant, 300 dpi, white background.
Each function: pure, accepts data + `output_path: Path`, writes one .png atomically, returns Path.

| Function | Output file |
|---|---|
| `plot_spread_distribution` | `fig_01_spread_dist.png` |
| `plot_roc_and_pr_curves` | `fig_02_roc_pr.png` |
| `plot_calibration_curve` | `fig_02_calibration.png` |
| `plot_shap_bar` | `fig_02_shap_bar.png` |
| `plot_predicted_vs_actual` | `fig_03_pred_vs_actual.png` |
| `plot_residuals` | `fig_03_residuals.png` |
| `plot_shap_beeswarm` | `fig_03_shap_beeswarm.png` |
| `plot_optuna_history` | `fig_04_optuna_history.png` |
| `plot_mc_price_histogram` | `fig_06_mc_histogram.png` |
| `plot_mc_fan_chart` | `fig_06_fan_chart.png` |

### `loan_pricing/reporting/tables.py`

Each function: writes one .csv atomically, returns Path.

| Function | Output file |
|---|---|
| `write_dataset_summary` | `tbl_01_dataset_summary.csv` |
| `write_pd_eval_metrics` | `tbl_02_pd_metrics.csv` |
| `write_spread_eval_metrics` | `tbl_03_spread_metrics.csv` |
| `write_hyperparameter_table` | `tbl_04_hyperparams.csv` |
| `write_ou_calibration_table` | `tbl_06_ou_params.csv` |
| `write_coverage_table` | `tbl_07_coverage.csv` |
| `write_model_comparison_table` | `tbl_08_comparison.csv` |

### `loan_pricing/scripts/` — one per output group

Each script:
- Imports only from `loan_pricing.*` (never from other scripts)
- All logic inside `main()` (importable without side effects)
- Runnable standalone: `python loan_pricing/scripts/02_pd_model_eval.py`
- Logs every file written at INFO level

### Tests: `tests/loan_pricing/test_reporting.py`

- Figure functions: synthetic inputs → assert file exists, valid PNG magic bytes, non-empty
- Table functions: assert CSV parseable by `pd.read_csv`, expected column headers
- All output to `tmp_path` fixture

---

## Hard Requirements (cross-cutting)

1. **No `print()` in `loan_pricing/`** — use `logging` everywhere via `get_logger(__name__)`
2. **Google-style docstrings** on all public functions/classes (Args, Returns, Raises)
3. **`mypy --strict`** must pass on `loan_pricing/`
4. **`ruff check`** and **`black --check`** must pass with zero warnings
5. **Reproducibility** — every random op uses `random_seed` from `ProjectConfig`;
   TF seeded via `tf.random.set_seed()`; Monte Carlo uses `tf.random.stateless_*`
6. **No data leakage** — `FeatureEngineer.fit()` guarded by `DataLeakageError`
7. **Atomic file writes** — write to temp, then `Path.replace()`
8. **`pathlib.Path` everywhere** — no `os.path`
9. **`tf.float64` everywhere** — all TF tensors and keras layers use float64
   (matching `financial_forecast/` convention for financial precision)
10. **TF only for models** — no XGBoost, no LightGBM, no raw scipy optimizers.
    sklearn is permitted for evaluation metrics and CV splitting only.

## Verification

After each phase:
1. `pytest tests/loan_pricing/ -v` — all tests pass
2. `ruff check loan_pricing/` — zero warnings
3. `black --check loan_pricing/` — already formatted
4. `mypy --strict loan_pricing/` — no type errors
5. At the end: `pytest tests/loan_pricing/ --cov=loan_pricing` — >= 80% coverage
