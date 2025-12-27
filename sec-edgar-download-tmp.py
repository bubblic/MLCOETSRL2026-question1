#!/usr/bin/env python3
import os
import sys
import re
import json
import shutil
import pathlib
from typing import List, Optional, Tuple, Dict

import pandas as pd
from sec_edgar_downloader import Downloader

# Arelle imports (from arelle-release)
from arelle import Cntlr, ModelManager, ModelXbrl

"""
CONFIG
Adjust these as needed.
"""
TICKER = "WMT"  # e.g., Walmart
FORMS = ["10-K", "10-Q"]  # which form types to download
AMOUNT_PER_FORM = 3  # how many of each form to download
AFTER = None  # e.g., "2022-01-01" or None
BEFORE = None  # e.g., "2025-12-31" or None

DOWNLOAD_DIR = pathlib.Path.cwd() / "sec-edgar-filings"
OUTPUT_DIR = pathlib.Path.cwd() / "xbrl-extracts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Common US-GAAP concepts to filter (add as needed)
COMMON_CONCEPTS = [
    # Income Statement
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "GrossProfit",
    "OperatingExpenses",
    "SellingGeneralAndAdministrativeExpense",
    "ResearchAndDevelopmentExpense",
    "OperatingIncomeLoss",
    "IncomeBeforeEquityMethodInvestments",
    "IncomeBeforeIncomeTaxes",
    "NetIncomeLoss",
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
    "WeightedAverageNumberOfSharesOutstandingBasic",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    # Balance Sheet
    "Assets",
    "AssetsCurrent",
    "CashAndCashEquivalentsAtCarryingValue",
    "InventoryNet",
    "PropertyPlantAndEquipmentNet",
    "Liabilities",
    "LiabilitiesCurrent",
    "LongTermDebtNoncurrent",
    "CommitmentsAndContingencies",
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "RetainedEarningsAccumulatedDeficit",
    # Cash Flow (optional but useful)
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInInvestingActivities",
    "NetCashProvidedByUsedInFinancingActivities",
    "CashAndCashEquivalentsPeriodIncreaseDecrease",
]

"""
HELPERS
"""


def download_filings(
    ticker: str,
    forms: List[str],
    amount_per_form: int = 3,
    after: Optional[str] = None,
    before: Optional[str] = None,
    download_dir: Optional[pathlib.Path] = None,
) -> None:
    dl = Downloader(
        company_name="Example Extractor",
        email_address="your-email@example.com",
        # You may set download_directory if you want custom path; default is ./sec-edgar-filings
        # download_directory=str(download_dir) if download_dir else None  # (library handles None differently)
    )
    # sec-edgar-downloader will default to ./sec-edgar-filings relative to CWD.
    # If you need a custom dir, set the env var "DOWNLOAD_DIRECTORY" or pass in init.
    for form in forms:
        kwargs = {}
        if after:
            kwargs["after"] = after
        if before:
            kwargs["before"] = before
        if amount_per_form:
            kwargs["limit"] = amount_per_form
        print(f"Downloading {form} for {ticker}...")
        count = dl.get(form, ticker, **kwargs)
        print(f"Downloaded {count} filings for {form}.")


def find_inline_xbrl_htmls(accession_dir: pathlib.Path) -> List[pathlib.Path]:
    """
    Return a list of HTML/HTM files in an accession folder that likely contain Inline XBRL.
    Heuristics:
    - Include .htm/.html files
    - Prioritize filenames referencing 'htm' / 'html' that look like main filing or contain 'xbrl'/'inline'
    - Exclude obvious exhibit-only htmls if desired (light heuristic)
    """
    htmls = sorted(
        list(accession_dir.glob("*.htm")) + list(accession_dir.glob("*.html"))
    )
    candidates = []
    for p in htmls:
        name = p.name.lower()
        if (
            any(
                key in name
                for key in ["10k", "10q", "index", "main", "form", "report", "htm"]
            )
            or "xbrl" in name
            or "inline" in name
        ):
            candidates.append(p)
    # Fallback to all HTMLs if we found none
    return candidates if candidates else htmls


def accession_dirs_for(
    ticker: str, forms: List[str], base_dir: pathlib.Path
) -> List[pathlib.Path]:
    """
    Returns accession directories for given ticker and forms.
    Path pattern: base_dir/<TICKER>/<FORM>/<ACCESSION>/
    """
    dirs = []
    for form in forms:
        form_dir = base_dir / ticker.upper() / form
        if not form_dir.exists():
            continue
        for acc in sorted(form_dir.iterdir()):
            if acc.is_dir():
                dirs.append(acc)
    return dirs


def arelle_load_inline_xbrl(
    cntlr: Cntlr.Cntlr, file_path: str
) -> Optional[ModelXbrl.ModelXbrl]:
    """
    Load an Inline XBRL document with Arelle.
    Returns a ModelXbrl or None if load failed.
    """
    try:
        model_manager = ModelManager.initialize(cntlr)
        # Arelle can handle Inline XBRL HTML directly
        model_xbrl = model_manager.load(file_path)
        return model_xbrl
    except Exception as e:
        print(f"[WARN] Failed to load XBRL for {file_path}: {e}")
        return None


def facts_to_records(model_xbrl: ModelXbrl.ModelXbrl) -> List[Dict]:
    """
    Convert all facts to flat records.
    Fields commonly useful for analysis.
    """
    records = []
    for fact in model_xbrl.factsInInstance:
        try:
            concept = fact.concept.qname.localName if fact.concept is not None else None
            ns = fact.concept.qname.namespaceURI if fact.concept is not None else None
            value = fact.value
            decimals = fact.decimals
            unit = (
                fact.unit.qname.localName
                if fact.unit is not None and fact.unit.qname is not None
                else (fact.unitID if getattr(fact, "unitID", None) else None)
            )
            # Period info
            period_type = None
            start = None
            end = None
            instant = None
            if fact.context is not None and fact.context.period is not None:
                p = fact.context.period
                if p.isInstant:
                    period_type = "instant"
                    instant = (
                        p.instantDatetime
                        if hasattr(p, "instantDatetime")
                        else p.instant
                    )
                elif p.isStartEnd:
                    period_type = "duration"
                    start = p.startDatetime if hasattr(p, "startDatetime") else p.start
                    end = p.endDatetime if hasattr(p, "endDatetime") else p.end
                else:
                    period_type = "forever"
            # Entity
            entity = (
                fact.context.entityIdentifier()[1] if fact.context is not None else None
            )
            # Dimensions (axis:member)
            dims = {}
            if fact.context is not None and fact.context.segDimValues:
                for dim, mem in fact.context.segDimValues.items():
                    try:
                        axis = dim.qname.localName
                        member = (
                            mem.memberQname.localName
                            if hasattr(mem, "memberQname")
                            else str(mem)
                        )
                        dims[axis] = member
                    except Exception:
                        pass

            rec = {
                "concept": concept,
                "namespace": ns,
                "value": value,
                "decimals": decimals,
                "unit": unit,
                "period_type": period_type,
                "start": str(start) if start else None,
                "end": str(end) if end else None,
                "instant": str(instant) if instant else None,
                "entity": entity,
                "dimensions_json": json.dumps(dims) if dims else None,
                "docinfo_document": (
                    model_xbrl.modelDocument.basename
                    if model_xbrl.modelDocument
                    else None
                ),
                "docinfo_sourceUrl": getattr(model_xbrl.modelDocument, "uri", None),
            }
            records.append(rec)
        except Exception as e:
            # Skip problematic facts but continue
            print(f"[WARN] Skipping fact due to error: {e}")
    return records


def write_csvs_for_accession(
    accession_dir: pathlib.Path, out_dir: pathlib.Path, ticker: str
) -> List[pathlib.Path]:
    """
    For a single accession directory, try each candidate Inline XBRL HTML and extract facts.
    Combines all facts from candidates into one CSV per accession.
    Returns list of created CSV file paths.
    """
    created = []
    candidates = find_inline_xbrl_htmls(accession_dir)
    if not candidates:
        print(f"[INFO] No HTML candidates found in {accession_dir}")
        return created

    cntlr = Cntlr.Cntlr(logFileName=None, logFileMode="w")
    all_records = []
    any_loaded = False

    for html in candidates:
        print(f"Parsing Inline XBRL: {html}")
        model_xbrl = arelle_load_inline_xbrl(cntlr, str(html))
        if model_xbrl is None:
            continue
        any_loaded = True
        recs = facts_to_records(model_xbrl)
        all_records.extend(recs)
        # Close model to release memory/resources
        model_xbrl.close()

    if not any_loaded or not all_records:
        print(f"[INFO] No facts extracted for {accession_dir}")
        return created

    # Write full facts CSV for this accession
    acc_id = accession_dir.name
    facts_csv = out_dir / f"{ticker.upper()}_{acc_id}_facts.csv"
    df = pd.DataFrame(all_records)

    # Normalize datatypes a bit
    # Keep value as string to avoid losing precision; user can convert numerics later.
    df.to_csv(facts_csv, index=False)
    created.append(facts_csv)

    # Also write a filtered CSV with common concepts
    filtered = df[df["concept"].isin(COMMON_CONCEPTS)].copy()
    filtered_csv = out_dir / f"{ticker.upper()}_{acc_id}_facts_filtered.csv"
    filtered.to_csv(filtered_csv, index=False)
    created.append(filtered_csv)

    print(f"Wrote: {facts_csv}")
    print(f"Wrote: {filtered_csv}")
    return created


def aggregate_filtered_csvs(
    out_dir: pathlib.Path, ticker: str, output_name: str = "aggregate_filtered.csv"
) -> pathlib.Path:
    """
    Aggregate all *_facts_filtered.csv for the ticker into one CSV.
    """
    files = sorted(out_dir.glob(f"{ticker.upper()}_*_facts_filtered.csv"))
    if not files:
        print("[INFO] No filtered CSVs found to aggregate.")
        return out_dir / output_name

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_accession"] = f.name.split(f"{ticker.upper()}_")[-1].replace(
            "_facts_filtered.csv", ""
        )
        dfs.append(df)
    agg = pd.concat(dfs, ignore_index=True)
    out = out_dir / output_name
    agg.to_csv(out, index=False)
    print(f"Wrote aggregated filtered CSV: {out}")
    return out


def main():
    print("Step 1: Download filings")
    download_filings(
        TICKER,
        FORMS,
        amount_per_form=AMOUNT_PER_FORM,
        after=AFTER,
        before=BEFORE,
        download_dir=DOWNLOAD_DIR,
    )

    print("Step 2: Find accession directories")
    acc_dirs = accession_dirs_for(TICKER, FORMS, DOWNLOAD_DIR)
    if not acc_dirs:
        print("No accession directories found. Exiting.")
        return

    print(f"Found {len(acc_dirs)} accession directories.")

    print("Step 3: Extract Inline XBRL facts to CSV")
    generated = []
    for acc_dir in acc_dirs:
        generated += write_csvs_for_accession(acc_dir, OUTPUT_DIR, TICKER)

    print(f"Generated {len(generated)} CSV files.")

    print("Step 4: Aggregate filtered CSVs")
    aggregate_filtered_csvs(OUTPUT_DIR, TICKER)


if __name__ == "__main__":
    # Quick dependency tips
    # pip install sec-edgar-downloader arelle-release pandas lxml
    # If you run into SSL or taxonomy fetching issues, Arelle may download taxonomies as needed.
    main()
