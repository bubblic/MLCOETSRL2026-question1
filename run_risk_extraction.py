"""Extract risk warnings from annual report PDFs.

Uses an LLM to flag risk-relevant pages, extract structured findings
per risk category, and synthesise a professional risk memo.

Usage:
    python run_risk_extraction.py
"""

import subprocess
import sys


def _ensure_spacy_ner() -> None:
    """Install spacy and the English NER model if not already present."""
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        print("Installing spacy and en_core_web_sm model...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "spacy>=3.5,<3.8"],
            stdout=subprocess.DEVNULL,
        )
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            stdout=subprocess.DEVNULL,
        )
        print("spacy NER ready.")


from risk.risk_extractor import RiskWarningsExtractor
from financial_forecast.clients.azure_llm_client import AzureLLMClient

if __name__ == "__main__":

    _ensure_spacy_ner()

    company = "evergrande"

    extractor = RiskWarningsExtractor(
        llm_client=AzureLLMClient(),
        known_entities=[
            {
                "type": "ORG",
                "names": [
                    "China Evergrande Group",
                    "Evergrande Group",
                    "Evergrande",
                    "Hengda Real Estate Group",
                ],
            },
        ],
        use_ner=True,
        anonymize_years=True,
    )

    extractor.run(
        input_path=f"./annual_reports/for_risk_warnings/{company}",
        output_dir=f"./extracted_json/risk_warnings/{company}",
    )
