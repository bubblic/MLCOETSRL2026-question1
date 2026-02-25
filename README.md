# MLCOETSRL2026-question1

JP Morgan MLCOE TSRL 2026 Internship Question 1 by Jaebum (Albert) Chung

## Azure Balance Sheet Model

`azure_balance_sheet_model.py` adds a balance sheet prediction workflow that calls
an Azure API endpoint. Configure the following environment variable:

- `AZURE_DEEPSEEK_ENDPOINT`

Run:
`python azure_balance_sheet_model.py`

## Financial Statement Extraction Pipeline

`llm_extract_json_from_financial_statements.py` runs an end-to-end pipeline that:

1. reads extracted statement text files (`*.llm.json`),
2. calls the Azure DeepSeek endpoint to normalize field values,
3. writes normalized JSON outputs,
4. computes financial ratios from those outputs.

### Input Requirements

- Input directory (default: `extracted_text`) must contain files named like:
  - `<company_id>.consolidated-balance-sheet.llm.json`
  - `<company_id>.consolidated-income-statement.llm.json`
  - `<company_id>.consolidated-cash-flow-statement.llm.json`
- Azure endpoint configuration must be available in your environment (for example via your existing Azure client setup).

### Common Commands

Single extraction run + ratio computation:

```bash
python llm_extract_json_from_financial_statements.py \
  --input-dir extracted_text \
  --output-dir deepseek_financial_statements \
  --max-workers 9
```

Multi-run extraction (10 runs) + median aggregation + distribution plots:

```bash
python llm_extract_json_from_financial_statements.py \
  --input-dir extracted_text \
  --num-extraction-runs 10 \
  --runs-output-dir deepseek_financial_statements_runs \
  --max-workers 9 \
  --plot-distributions
```

Skip extraction and compute ratios only from existing outputs:

```bash
python llm_extract_json_from_financial_statements.py \
  --skip-extraction \
  --ratios-aggregation median \
  --runs-output-dir deepseek_financial_statements_runs
```

### Output Artifacts

Single-run mode (`--num-extraction-runs 1`):

- Normalized files in `--output-dir` (default `deepseek_financial_statements`)
- Ratio output JSON (default `financial_ratios.json` inside the ratios input directory)

Multi-run mode (`--num-extraction-runs > 1`):

- Per-run normalized files in `--runs-output-dir/run_XX/`
- Median-aggregated ratio output JSON (default under `--runs-output-dir`)
- Per-run value tracking JSON (default `extraction_run_values.json` under `--runs-output-dir`)
- Optional plot images in `field_value_distributions/` under `--runs-output-dir` when `--plot-distributions` is enabled

### Useful Flags

- `--max-workers`: parallelism across statement files.
- `--skip-extraction`: only run ratio computation.
- `--skip-ratios`: only run extraction.
- `--ratios-aggregation {single,median}`:
  - `single`: compute ratios from one normalized directory.
  - `median`: aggregate field medians across run folders before computing ratios.

## Setting It Up (Windows)

Setting up the environment was a little tricky even when following the instruction in TensorFlow in Action (Thushan Ganegedara 2022) textbook because some links to downloading NVIDIA drivers were broken and no executable setup file was available for cuDNN package, which required me to copy individual files to appropriate locations. The instruction had to be followed carefully, taking into account the exact versions mentioned by the book.

Despite that, there were still some files missing that kept crashing simple TensorFlow convolution operations in Ch02 examples. (Kept getting this error message: Could not locate cudnn_cnn_infer64_8.dll. Please make sure it is in your library path!)

I had to download additional files and copy them into C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin for it to work. Refer to https://github.com/SYSTRAN/faster-whisper/discussions/715 and https://github.com/Purfview/whisper-standalone-win/releases/tag/libs for the files needed.
