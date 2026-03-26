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

Each script can be run directly from the project root.

### Simple Model (no training -- forward simulation only)

```bash
python run_simple_model_forecast.py
```

Demonstrates the Pareja (2009) Cash Budget construction as a pure forward simulation. Uses simple policies with parameters set from historical averages -- no gradient-based training. Produces a deterministic 10-year balance sheet forecast and one-step-ahead historical fit.

### Trainable Model with Simple Policies

```bash
python run_trainable_model_forecast_simple_policies.py
```

Trains all simple policies (cash-target liquidity, simple dividends/buybacks, static cost ratio, simple debt, deterministic OpEx) via gradient descent, then forecasts.

### Trainable Model with Advanced Policies

```bash
python run_trainable_model_forecast_adv_policies.py
```

Uses trend-based liquidity, Lintner dividends, baseline buybacks, trend cost ratio, and trend debt. Deterministic simulator with SimpleOpEx.

### Trainable Model with Bayesian OpEx

```bash
python run_trainable_model_forecast_adv_policies_w_bayesianopex.py
```

Advanced policies with Bayesian variational inference for OpEx and Monte Carlo simulation (1000 samples).

### Trainable Model with Bayesian OpEx + Tax Anomalies

```bash
python run_trainable_model_forecast_adv_policies_w_bayesianopex_taxanomaly.py
```

Adds one-time tax anomaly adjustments on top of the Bayesian OpEx configuration.

## Financial Statement Extraction Pipeline

```bash
python -m scripts.run_extraction \
  --input-dir extracted_text \
  --num-extraction-runs 3 \
  --ratios-aggregation median \
  --runs-output-dir deepseek_financial_statements_runs \
  --max-workers 9 \
  --plot-distributions \
  --hallucination-output-file hallucination_rates.json \
  --hallucination-top-k 10
```

Reads extracted statement text files, calls the Azure DeepSeek endpoint to normalize field values, writes normalized JSON outputs, and computes financial ratios.

### PDF Extraction

```bash
# Single file
python -m financial_forecast.extraction.statement_extractor \
  --input-file ./annual_reports/alibaba_2025.pdf \
  --query "Consolidated Balance Sheet"

# Full directory
python -m financial_forecast.extraction.statement_extractor \
  --input-dir ./annual_reports \
  --query "Consolidated Balance Sheet"
```

### Tax Anomaly Extraction

```bash
python -m financial_forecast.extraction.tax_anomaly_extractor \
  --input-file ./annual_reports/google_2024.pdf
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
financial_forecast/               # Main package
  types.py                        # FinancialState, EconomicInputs dataclasses
  models/
    base.py                       # BaseFinancialModel (abstract, shared forecast logic)
    trainable_financial_model.py  # Composable model with pluggable policy modules
    balance_sheet.py              # Asset evolution
    income_statement.py           # Income computation
    cash_budget.py                # Liquidity & financing
    opex.py                       # SimpleOpEx, BayesianOpEx
    capex.py                      # Asset growth & depreciation policy
    working_capital.py            # Working capital ratios (AR, AP, Inv)
    liquidity.py                  # Cash/IMS allocation policies
    dividends.py                  # Dividend policies (Simple, Lintner)
    buyback.py                    # Buyback policies
    purchases.py                  # Cost ratio policies (Static, Trend)
    debt.py                       # Debt financing policies
    tax.py                        # Tax modules (Simple, with anomalies)
    llm_forecaster.py             # LLM-based balance sheet forecaster
  training/
    pipeline.py                   # ForecastPipeline orchestrator
    base_trainer.py               # BaseTrainer abstract class
    policy_trainer.py             # Policy + OpEx parameter training
    structural_trainer.py         # Structural parameter training
    diagnostics.py                # Training diagnostics
    io_utils.py                   # Training results I/O
  inference/
    trajectory_simulator.py       # DeterministicSimulator, MonteCarloSimulator
    state_index.py                # Tensor layout constants
    plotting.py                   # Forecast visualization
  serialization/
    parameter_io.py               # .npz parameter save/load
  data/
    loader.py                     # HistoricalDataLoader
    inflation.py                  # Inflation data
    aapl/
      financial_statements.py     # Apple FY2018-FY2025 historical data
      tax_onetime_payments.py     # Apple one-time tax payments
  extraction/
    statement_extractor.py        # Two-stage LLM financial statement extraction
    tax_anomaly_extractor.py      # Tax anomaly extraction from 10-K
    page_identifier.py            # LLM-based PDF page selection
    pdf_extractor.py              # pdfplumber wrapper
    statement_cli.py              # Batch extraction CLI
    statement_config.py           # Extraction configuration
    statement_normalization.py    # Field normalization
    statement_extraction.py       # Extraction logic
    statement_ratios.py           # Financial ratio computation
    statement_hallucination.py    # Hallucination analysis
    statement_runs.py             # Multi-run aggregation
  clients/
    azure_llm_client.py           # Azure LLM HTTP client

scripts/
  run_extraction.py               # Batch extraction pipeline entry point

tests/                            # Test suite (99 tests)
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
