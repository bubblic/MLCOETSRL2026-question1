# MLCOETSRL2026-question1

JP Morgan MLCOE TSRL 2026 Internship Question 1 by Jaebum (Albert) Chung

## Installation

```bash
# From the project root:
pip install -e .

# Or with dev dependencies (pytest):
pip install -e ".[dev]"
```

## Financial Forecast Models

Each script can be run directly from the project root. All forecast
pipelines export a JSON report to ``training_results/`` for downstream
use (e.g., LLM-based CEO recommendations).

### Simple Model (no training -- forward simulation only)

```bash
python run_simple_model_forecast.py
```

Demonstrates the Pareja (2009) Cash Budget construction as a pure forward
simulation. Uses simple policies with parameters set from historical
averages -- no gradient-based training. Produces a deterministic 10-year
balance sheet forecast and one-step-ahead historical fit.

### Trainable Model with Simple Policies

```bash
python run_trainable_model_forecast_simple_policies.py
```

Trains all simple policies (cash-target liquidity, simple dividends/buybacks,
static cost ratio, simple debt, deterministic OpEx) via gradient descent,
then runs a deterministic 10-year forecast.

### Trainable Model with Advanced Policies

```bash
python run_trainable_model_forecast_adv_policies.py
```

Uses trend-based liquidity, Lintner dividends, baseline buybacks, trend
cost ratio, and trend debt. Deterministic simulator with SimpleOpEx.

### Trainable Model with Bayesian OpEx

```bash
python run_trainable_model_forecast_adv_policies_w_bayesianopex.py
```

Advanced policies with Bayesian variational inference for OpEx and Monte
Carlo simulation (1,000 samples). Includes OpEx fit diagnostics.

### Trainable Model with Bayesian OpEx + Tax Anomalies

```bash
python run_trainable_model_forecast_adv_policies_w_bayesianopex_taxanomaly.py
```

The most complete model configuration. Adds LLM-extracted one-time tax
anomaly adjustments (from 10-K filings) on top of the Bayesian OpEx
configuration. Runs an 8-year Monte Carlo forecast and exports a JSON
report for the LLM recommendation pipeline.

#### Extraction Model Used for Tax Anomaly

```bash
python run_tax_anomaly_extraction.py
```

Extracts one-time tax charges and contingency amounts from 10-K PDFs
using an LLM. Outputs structured JSON files used by ``TaxWithAnomalies``.

### CEO Recommendation via LLM

```bash
python run_recommendation_to_ceo.py
```

Reads the forecast report JSON produced by the training pipeline and
sends the historical + forecast tables to the Azure DeepSeek reasoning
model for strategic capital-structure and capital-allocation analysis.
Uses greedy decoding (temperature=0, top_k=1) for minimal hallucination.

## Financial Statement Extraction Pipeline

The extraction pipeline has three stages, each with its own run script:

### Stage 1: Extract Financial Statements from PDFs

```bash
python run_statement_extraction.py
```

Uses an LLM to identify relevant pages in each 10-K PDF, then extracts
primary financial tables and supplementary disclosures.

### Stage 2: Normalize Extracted Statements

```bash
python run_statement_normalization.py
```

Reads raw ``*.llm.json`` files, sends them to an LLM for field
normalization, and writes structured ``*.normalized.json`` output.

### Stage 2b: Multi-Run Normalization (Hallucination Measurement)

```bash
python run_statement_normalization_multi.py
```

Runs the LLM normalization N times on the same input, then computes
median-aggregated ratios and hallucination rates across runs.

### Stage 3: Calculate Financial Ratios

```bash
python run_ratio_calculation.py
```

Reads ``*.normalized.json`` files and computes derived metrics and
financial ratios.

### Full Pipeline (PDF to Ratios)

```bash
python run_pdf_to_ratios_pipeline.py
```

Chains all three stages: PDF extraction, LLM normalization, and ratio
calculation in a single run.

## Running Tests

```bash
python -m pytest tests/ -v
```

87 tests across 8 test files.

## Project Structure

```
financial_forecast/               # Main package
  types.py                        # FinancialState, EconomicInputs dataclasses
  models/
    base.py                       # BaseFinancialModel (composable forecast logic)
    trainable_financial_model.py  # Extends base with training + serialization
    balance_sheet.py              # Asset evolution
    income_statement.py           # Income computation
    cash_budget.py                # Liquidity & financing
    opex.py                       # OpExModule ABC, SimpleOpEx, BayesianOpEx
    capex.py                      # Asset growth & depreciation policy
    working_capital.py            # Working capital ratios (AR, AP, Inv)
    liquidity.py                  # Cash/IMS allocation policies
    dividends.py                  # Dividend policies (Simple, Lintner)
    buyback.py                    # Buyback policies
    purchases.py                  # Cost ratio policies (Static, Trend)
    debt.py                       # Debt financing policies
    tax.py                        # SimpleTax, TaxWithAnomalies
    llm_forecaster.py             # LLM-based balance sheet forecaster
  training/
    pipeline.py                   # ForecastPipeline orchestrator + JSON export
    base_trainer.py               # BaseTrainer abstract class
    policy_trainer.py             # Policy + OpEx parameter training
    structural_trainer.py         # Structural parameter training (with grad clipping)
    diagnostics.py                # Training loss diagnostics
    io_utils.py                   # Training results I/O
  inference/
    trajectory_simulator.py       # DeterministicSimulator, MonteCarloSimulator
    state_index.py                # Tensor layout constants
    forecast_driver_models.py     # Sales/inflation forecast models
    plotting.py                   # Forecast visualization
  serialization/
    parameter_io.py               # .npz parameter save/load
  data/
    loader.py                     # HistoricalDataLoader
    inflation.py                  # US inflation data
    aapl/
      financial_statements.py     # Apple FY2018-FY2025 historical data
  extraction/
    base_pdf_extractor.py         # Abstract PDF extraction base class
    financial_statement_extractor.py  # Two-stage LLM financial statement extraction
    tax_anomaly_extractor.py      # Tax anomaly extraction from 10-K
    page_identifier.py            # LLM-based PDF page selection
    pdf_extractor.py              # pdfplumber wrapper
    statement_config.py           # Extraction configuration
    statement_normalizer.py       # LLM-based field normalization
    statement_normalization.py    # Normalization helpers
    statement_ratios.py           # Financial ratio computation
    statement_hallucination.py    # Hallucination analysis
    statement_runs.py             # Multi-run aggregation
    utils.py                      # Extraction utilities
  reporting/
    table_formatter.py            # TableFormatter ABC, MarkdownTableFormatter
    advisor.py                    # Advisor ABC, DeepseekCEOAdvisor
  clients/
    azure_llm_client.py           # Azure LLM HTTP client

run_*.py                          # Entry-point scripts (see above)
send_reasoning_prompt.py          # Standalone LLM prompt script

tests/                            # Test suite (87 tests)
  test_simple_model_forecast.py
  test_trainable_financial_model_simpleopex.py
  test_trainable_financial_model_bayesianopex.py
  test_extraction.py
  test_llm_forecast.py
  test_reasoning_prompt.py
  test_statement_normalization.py
  test_tax_anomalies.py
```

## Environment Variables

The LLM-based extraction and forecasting pipelines require:

- `AZURE_DEEPSEEK_ENDPOINT` -- Azure-hosted LLM endpoint URL

Set via `.env` file or shell environment.

## Setting It Up (Windows)

Setting up the environment was a little tricky even when following the instruction in TensorFlow in Action (Thushan Ganegedara 2022) textbook because some links to downloading NVIDIA drivers were broken and no executable setup file was available for cuDNN package, which required me to copy individual files to appropriate locations. The instruction had to be followed carefully, taking into account the exact versions mentioned by the book.

Despite that, there were still some files missing that kept crashing simple TensorFlow convolution operations in Ch02 examples. (Kept getting this error message: Could not locate cudnn_cnn_infer64_8.dll. Please make sure it is in your library path!)

I had to download additional files and copy them into C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin for it to work. Refer to https://github.com/SYSTRAN/faster-whisper/discussions/715 and https://github.com/Purfview/whisper-standalone-win/releases/tag/libs for the files needed.
