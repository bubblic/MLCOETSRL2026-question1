"""Extract risk warnings from annual report PDFs.

Uses an LLM to flag risk-relevant pages, extract structured findings
per risk category, and synthesise a professional risk memo.

Usage:
    python run_risk_extraction.py --company evergrande
    python run_risk_extraction.py --company svb --no-ner
"""

import argparse
import subprocess
import sys


COMPANY_CONFIGS = {
    "evergrande": {
        "known_entities": [
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
    },
    "svb": {
        "known_entities": [
            {
                "type": "ORG",
                "names": [
                    "SVB Financial Group",
                    "Silicon Valley Bank",
                    "SVB",
                ],
            },
        ],
    },
    "bbby": {
        "known_entities": [
            {
                "type": "ORG",
                "names": [
                    "Bed Bath & Beyond Inc.",
                    "Bed Bath & Beyond",
                    "BBBY",
                ],
            },
        ],
    },
    "lehman": {
        "known_entities": [
            {
                "type": "ORG",
                "names": [
                    "Lehman Brothers Holdings Inc.",
                    "Lehman Brothers",
                ],
            },
        ],
    },
    "wirecard": {
        "known_entities": [
            {
                "type": "ORG",
                "names": [
                    "Wirecard AG",
                    "Wirecard",
                ],
            },
        ],
    },
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract risk warnings from annual report PDFs.",
    )
    parser.add_argument(
        "--company",
        required=True,
        choices=list(COMPANY_CONFIGS.keys()),
        help="Company to process (must have a config in COMPANY_CONFIGS).",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Input directory containing PDFs. "
             "Defaults to ./annual_reports/for_risk_warnings/{company}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for extracted JSON. "
             "Defaults to ./extracted_json/risk_warnings/{company}.",
    )
    parser.add_argument(
        "--no-ner",
        action="store_true",
        help="Disable spaCy NER (use only known entities).",
    )
    parser.add_argument(
        "--no-anonymize-years",
        action="store_true",
        help="Disable year anonymisation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    use_ner = not args.no_ner
    if use_ner:
        _ensure_spacy_ner()

    from risk.risk_extractor import RiskWarningsExtractor
    from financial_forecast.clients.azure_llm_client import AzureLLMClient

    config = COMPANY_CONFIGS[args.company]
    input_dir = args.input_dir or f"./annual_reports/for_risk_warnings/{args.company}"
    output_dir = args.output_dir or f"./extracted_json/risk_warnings/{args.company}"

    extractor = RiskWarningsExtractor(
        llm_client=AzureLLMClient(),
        known_entities=config["known_entities"],
        use_ner=use_ner,
        anonymize_years=not args.no_anonymize_years,
    )

    extractor.run(
        input_path=input_dir,
        output_dir=output_dir,
    )
