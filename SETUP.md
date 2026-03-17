# Financial Forecast Package — Setup & Usage

## Installation

```bash
# From the project root:
pip install -e .

# Or with dev dependencies (pytest):
pip install -e ".[dev]"
```

This installs the `financial_forecast` package in editable mode so all imports resolve correctly.

## Running the Pipelines

Below is the mapping from old scripts to the new refactored commands.

### 1. Deterministic Balance-Sheet Forecast

**Old:** `python simple_financial_model.py`
**New:**
```bash
python -m scripts.run_forecast
```
Runs a 3-year deterministic forecast using `SimpleFinancialModel` with fixed `tf.constant` parameters and Apple FY2023 initial state.

### 2. Bayesian Model Training + Monte Carlo Forecast

**Old:** `python trainable_financial_model_taxanomaly_effstdebt_bayesian_vi_refactored.py`
**New:**
```bash
python -m scripts.run_bayesian_forecast
```
Trains the `BayesianFinancialModel` (policy params, structural params, variational OpEx inference) on Apple FY2018-FY2024 data, then runs a 1000-sample Monte Carlo 10-year forecast.

### 3. Trainable Model (Gradient-Based Parameter Fitting)

**Old:** `python trainable_financial_model.py`
**New:** Use the `TrainableFinancialModel` class directly:
```python
from financial_forecast.models.trainable_model import TrainableFinancialModel
from financial_forecast.data.historical_data import get_apple_historical_data

model = TrainableFinancialModel()
data = get_apple_historical_data()
model.train_simple_policies(data, epochs=5000)
model.train_structural_parameters(data, epochs=5000)
```

### 4. Financial Statement Extraction from PDFs

**Old:** `python llm_extract_financial_statement_from_pdf.py --input-file ./annual_reports/alibaba_2025.pdf --query "Consolidated Balance Sheet"`
**New:**
```bash
python -m financial_forecast.extraction.statement_extractor  --input-file ./annual_reports/alibaba_2025.pdf --query "Consolidated Balance Sheet" --query "Consolidated Cash Flow Statement" --query "Consolidated Income Statement"
```
Or for a full directory:
```bash
python -m financial_forecast.extraction.statement_extractor --input-dir ./annual_reports --query "Consolidated Balance Sheet"
```

### 5. Batch Extraction + Normalization + Ratios + Hallucination Analysis

**Old:** `python llm_extract_json_from_financial_statements.py --input-dir extracted_text --num-extraction-runs 3 ...`
**New:**
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
All the same CLI flags from the old script work identically.

### 6. Tax Anomaly Extraction from 10-K PDFs

**Old:** `python llm_extract_tax_anomalies_from_pdf.py --input-file ./annual_reports/google_2024.pdf`
**New:**
```bash
python -m financial_forecast.extraction.tax_anomaly_extractor --input-file ./annual_reports/google_2024.pdf
```
Or for a full directory:
```bash
python -m financial_forecast.extraction.tax_anomaly_extractor --input-dir ./annual_reports
```

### 7. LLM-Only Balance Sheet Forecast

**Old:** `python "llm-only-balance-sheet-forecast.py"`
**New:**
```python
from financial_forecast.models.llm_forecaster import (
    AzureReasoningBalanceSheetForecaster,
    load_historical_balance_sheet,
    run_llm_balance_sheet_forecast,
    plot_forecast_elements,
)

run_llm_balance_sheet_forecast()
```

### 8. LLM PDF Page Identifier (Library Use)

**Old:** `from llm_pdf_pages_identifier import select_pages_with_llm`
**New:**
```python
from financial_forecast.extraction.page_identifier import select_pages_with_llm
```
This module is used internally by the statement and tax extractors; you rarely need to call it directly.

## Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
financial_forecast/           # Main package
  types.py                    # FinancialState, EconomicInputs dataclasses
  models/
    base.py                   # BaseFinancialModel (abstract, shared forecast logic)
    simple_model.py           # Deterministic model (tf.constant params)
    trainable_model.py        # Gradient-trainable model (tf.Variable params)
    bayesian_model.py         # Bayesian VI model (TransformedVariable + OpEx VI)
    llm_forecaster.py         # LLM-based balance sheet forecaster
  layers/
    forecast_step_layer.py    # Keras layers: AssetEvolution, IncomeStatement, LiquidityFinancing
    opex_layer.py             # OpEx Keras layer
  training/
    pipeline.py               # End-to-end training + forecast orchestrator
    training.py               # Training entry points
    io_utils.py               # Training results I/O
  inference/
    forecast.py               # Monte Carlo simulation
    plotting.py               # Forecast and OpEx plots
  extraction/
    pdf_extractor.py          # pdfplumber wrapper
    page_identifier.py        # LLM-based PDF page selection
    statement_extractor.py    # Two-stage LLM financial statement extraction
    tax_anomaly_extractor.py  # Tax anomaly extraction from 10-K
    statement_cli.py          # Batch extraction CLI
    statement_config.py       # Extraction configuration
    statement_normalization.py
    statement_extraction.py
    statement_ratios.py
    statement_hallucination.py
    statement_runs.py
  clients/
    azure_llm_client.py       # Azure LLM HTTP client
  data/
    historical_data.py        # Apple FY2018-FY2025 data

scripts/                      # CLI entry points
  run_forecast.py             # Deterministic forecast
  run_bayesian_forecast.py    # Bayesian training + Monte Carlo
  run_extraction.py           # Batch extraction pipeline

tests/                        # All tests (104 tests)
```

## Environment Variables

The LLM-based extraction and forecasting pipelines require:

- `AZURE_DEEPSEEK_ENDPOINT` — Azure-hosted LLM endpoint URL

Set via `.env` file or shell environment.
