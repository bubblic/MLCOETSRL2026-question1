# MLCOETSRL2026-question1

JP Morgan MLCOE TSRL 2026 Internship Question 1 by Jaebum (Albert) Chung

## Azure Balance Sheet Model

`azure_balance_sheet_model.py` adds a balance sheet prediction workflow that calls
an Azure API endpoint. Configure the following environment variable:

- `AZURE_BALANCE_SHEET_ENDPOINT`

Run:
`python azure_balance_sheet_model.py`

## Setting It Up (Windows)

Setting up the environment was a little tricky even when following the instruction in TensorFlow in Action (Thushan Ganegedara 2022) textbook because some links to downloading NVIDIA drivers were broken and no executable setup file was available for cuDNN package, which required me to copy individual files to appropriate locations. The instruction had to be followed carefully, taking into account the exact versions mentioned by the book.

Despite that, there were still some files missing that kept crashing simple TensorFlow convolution operations in Ch02 examples. (Kept getting this error message: Could not locate cudnn_cnn_infer64_8.dll. Please make sure it is in your library path!)

I had to download additional files and copy them into C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin for it to work. Refer to https://github.com/SYSTRAN/faster-whisper/discussions/715 and https://github.com/Purfview/whisper-standalone-win/releases/tag/libs for the files needed.
