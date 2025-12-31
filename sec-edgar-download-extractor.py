import os
import glob
import shutil
import csv
import re
from typing import Optional

# Compute script directory early so we can configure Arelle before importing it.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _configure_arelle_app_dirs() -> None:
    """
    Ensure Arelle uses writable directories.

    In sandboxed environments (like Cursor's), writing to user-level app/config dirs
    can fail with 'Operation not permitted'. For local runs, this is harmless.
    """
    arelle_home = os.path.join(SCRIPT_DIR, ".arelle")
    os.makedirs(arelle_home, exist_ok=True)
    # Arelle may append its own "arelle/cache/http" under the user app dir.
    os.makedirs(os.path.join(arelle_home, "arelle", "cache", "http"), exist_ok=True)
    # Many libraries respect these. Arelle uses platform app dirs derived from HOME on macOS.
    os.environ.setdefault("HOME", SCRIPT_DIR)
    os.environ.setdefault("XDG_CONFIG_HOME", arelle_home)
    os.environ.setdefault("XDG_DATA_HOME", arelle_home)


# Must run before importing Arelle (it may compute app dirs at import time).
_configure_arelle_app_dirs()

from sec_edgar_downloader import Downloader
from arelle import Cntlr
from arelle import WebCache as _ArelleWebCache

# --- Configuration ---
# COMPANY_TICKER = "WMT"  # Walmart
COMPANY_TICKER = "0000320193"  # Apple
# COMPANY_TICKER = "KO"  # Coca Cola
# COMPANY_TICKER = "GM"  # General Motors
EMAIL_ADDRESS = (
    "your.email@example.com"  # <--- REQUIRED BY SEC: Change this to your actual email
)
ORG_NAME = "Personal Research"  # <--- REQUIRED BY SEC: Change this to your organization
DOWNLOAD_LIMIT = 30

# Use absolute path for reliability
CUSTOM_BASE_DOWNLOAD_NAME = "xbrl_download_root"
DOWNLOAD_FOLDER = os.path.join(SCRIPT_DIR, CUSTOM_BASE_DOWNLOAD_NAME)
DOWNLOADER_SUBFOLDER_NAME = "sec-edgar-filings"
OUTPUT_CSV_FILE = os.path.join(
    SCRIPT_DIR, f"{COMPANY_TICKER.lower()}_financial_facts.csv"
)

print(f"Target Download Root: {DOWNLOAD_FOLDER}")


def _patch_arelle_cache_lock_dir_creation() -> None:
    """
    Arelle uses file locks for its HTTP cache downloads. For some Arelle versions,
    the lock file is created before the cache subdirectories are created, which can
    raise FileNotFoundError for paths like:
      .../arelle/cache/http/xbrl.fasb.org/us-gaap/2016/elts/...xsd.lock

    This patch ensures the parent directory exists before Arelle attempts to lock.
    """
    try:
        original = _ArelleWebCache.WebCache._downloadFileWithLock
    except Exception:
        return

    def _wrapped_downloadFileWithLock(
        self, url, filepath, retrievingDueToRecheckInterval=False, retryCount=5
    ):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        except Exception:
            pass
        return original(
            self,
            url,
            filepath,
            retrievingDueToRecheckInterval=retrievingDueToRecheckInterval,
            retryCount=retryCount,
        )

    _ArelleWebCache.WebCache._downloadFileWithLock = _wrapped_downloadFileWithLock


def _read_file_head(path: str, max_bytes: int = 250_000) -> str:
    """Read first N bytes as text (best-effort) for quick content sniffing."""
    try:
        with open(path, "rb") as f:
            b = f.read(max_bytes)
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _looks_like_inline_xbrl_html(path: str) -> bool:
    """
    Heuristic: Inline XBRL HTML usually includes ix namespace / tags.
    We only need a cheap sniff, not full parsing.
    """
    head_lower = _read_file_head(path).lower()
    return ("xmlns:ix=" in head_lower) or ("<ix:" in head_lower)


def _extract_xbrl_package_from_full_submission(accession_path: str) -> Optional[str]:
    """
    For non-Inline XBRL filings (often pre-2018), `sec_edgar_downloader` may only save:
      - primary-document.html (10-K HTML, not iXBRL)
      - full-submission.txt (contains the XBRL instance + schema/linkbases embedded)

    This function materializes the XBRL package files into the accession folder
    so Arelle can load the instance and resolve relative schemaRef/linkbaseRefs.

    Returns: path to extracted XBRL instance file, or None if not found.
    """
    full_submission = os.path.join(accession_path, "full-submission.txt")
    if not os.path.exists(full_submission):
        return None

    try:
        with open(full_submission, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except Exception:
        return None

    # Split into SEC <DOCUMENT> blocks; case-insensitive and tolerant.
    blocks = re.split(r"(?i)<document>", txt)
    if len(blocks) <= 1:
        return None

    desired_types = {
        "EX-101.INS",
        "EX-101.SCH",
        "EX-101.CAL",
        "EX-101.DEF",
        "EX-101.LAB",
        "EX-101.PRE",
    }

    instance_path: Optional[str] = None

    for block in blocks[1:]:
        # SEC full-submission.txt commonly uses tag-lines like "<TYPE>EX-101.INS" (no closing tag).
        # But tolerate XML-style "<TYPE>...</TYPE>" too.
        m_type = re.search(r"(?is)<type>\s*([^<\r\n]+)\s*</type>", block)
        if not m_type:
            m_type = re.search(r"(?im)^<type>\s*([^\r\n<]+)", block)
        if not m_type:
            continue
        doc_type = m_type.group(1).strip().upper()
        if doc_type not in desired_types:
            continue

        m_fn = re.search(r"(?is)<filename>\s*([^<\r\n]+)\s*</filename>", block)
        if not m_fn:
            m_fn = re.search(r"(?im)^<filename>\s*([^\r\n<]+)", block)
        if not m_fn:
            continue
        filename = m_fn.group(1).strip()
        out_path = os.path.join(accession_path, filename)

        m_text = re.search(r"(?is)<text>\s*(.*?)\s*</text>", block)
        if not m_text:
            continue
        payload = m_text.group(1)

        # Some submissions wrap XBRL instance in <XBRL> ... </XBRL>
        m_xbrl = re.search(r"(?is)<xbrl>\s*(.*?)\s*</xbrl>", payload)
        if m_xbrl:
            payload = m_xbrl.group(1)

        # Avoid rewriting if file already exists and is non-trivial.
        if not (os.path.exists(out_path) and os.path.getsize(out_path) > 200):
            try:
                with open(out_path, "w", encoding="utf-8", newline="") as out:
                    out.write(payload)
            except Exception:
                continue

        if doc_type == "EX-101.INS":
            instance_path = out_path

    return instance_path


def clean_previous_downloads():
    """Deletes the download folder to force a fresh download, avoiding cache issues."""
    if os.path.exists(DOWNLOAD_FOLDER):
        print(f"Removing old '{DOWNLOAD_FOLDER}' to ensure fresh download...")
        shutil.rmtree(DOWNLOAD_FOLDER)


def download_10k():
    print(f"Downloading last {DOWNLOAD_LIMIT} 10-K filings for {COMPANY_TICKER}...")
    try:
        dl = Downloader(ORG_NAME, EMAIL_ADDRESS, DOWNLOAD_FOLDER)
        dl.get("10-K", COMPANY_TICKER, limit=DOWNLOAD_LIMIT, download_details=True)
        print("Download complete.")
    except Exception as e:
        if "Please provide a valid email" in str(e):
            print(
                "\n!!! ERROR: Please update EMAIL_ADDRESS and ORG_NAME in the script configuration. The SEC requires this information. !!!\n"
            )
        raise e


def find_all_filing_paths():
    """
    Finds the primary document path for ALL downloaded 10-K filings.
    Returns a list of tuples: (accession_number, file_path)
    """
    base_dir = os.path.join(
        DOWNLOAD_FOLDER, DOWNLOADER_SUBFOLDER_NAME, COMPANY_TICKER, "10-K"
    )

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base download directory not found: {base_dir}")

    accession_folders = [
        f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))
    ]

    if not accession_folders:
        raise FileNotFoundError(f"No accession number folders found inside: {base_dir}")

    filing_paths = []

    for accession_folder_name in accession_folders:
        accession_path = os.path.join(base_dir, accession_folder_name)
        primary_html = os.path.join(accession_path, "primary-document.html")

        # Prefer Inline XBRL HTML when present (newer filings).
        if os.path.exists(primary_html) and _looks_like_inline_xbrl_html(primary_html):
            filing_paths.append((accession_folder_name, primary_html))
            continue

        # For older filings, materialize XBRL instance + schema/linkbases from full-submission.txt
        instance = _extract_xbrl_package_from_full_submission(accession_path)
        if instance and os.path.exists(instance):
            filing_paths.append((accession_folder_name, instance))
            continue

        # Last-resort fallback: pick largest HTML/XML excluding full-submission.
        all_candidates = glob.glob(os.path.join(accession_path, "*"))
        candidates = [
            f
            for f in all_candidates
            if (f.endswith(".htm") or f.endswith(".xml") or f.endswith(".html"))
            and "full-submission.txt" not in f
        ]
        if candidates:
            primary_doc_path = max(candidates, key=os.path.getsize)
            filing_paths.append((accession_folder_name, primary_doc_path))

    return filing_paths


def get_dimensions_from_context(context):
    """
    Extracts all dimension (segment) information from an Arelle model context.

    Returns: A string representation of all dimensions and their members,
             or "None" if no explicit dimensions are found.
    """
    dimensions = []
    # Check for segment in the context
    if context.segment:
        for item in context.segment.iterchildren():
            # item can be a concept (explicit dimension) or typed dimension
            if item.qname.localName == "explicitMember":
                dim_qname = item.dimensionQname
                mem_qname = item.memberQname
                dimensions.append(f"{dim_qname.localName}:{mem_qname.localName}")
            elif item.qname.localName == "typedMember":
                # For typed dimensions, we include the dimension name and the typed value
                dim_qname = item.dimensionQname
                # The value is typically the text content of the typedMember element
                typed_value = " ".join(item.itertext())
                dimensions.append(f"{dim_qname.localName}:{typed_value}")

    if dimensions:
        return "; ".join(dimensions)
    else:
        return "None"


def extract_financial_data(filing_paths):
    all_extracted_facts = []

    # Initialize Arelle Controller once
    print("Initializing Arelle...")

    # --- FIX 2: DISABLE CACHE/VALIDATION FOR ROBUSTNESS ---
    # This tells Arelle to skip some steps that might be causing the cache lock error.
    _configure_arelle_app_dirs()
    _patch_arelle_cache_lock_dir_creation()
    # Also disable persistent config so Arelle doesn't try to write to user-level app dirs.
    ctrl = Cntlr.Cntlr(
        hasGui=False,
        logFileName=os.path.join(SCRIPT_DIR, "arelle.log"),
        logFileMode="w",
        disable_persistent_config=True,
    )

    # --- UPDATED: Comprehensive list of requested US-GAAP tags ---
    target_tags = {
        # --- Equity and Capitalization ---
        "Ordinary Shares Number": [
            "us-gaap:CommonStockSharesOutstanding",
            "us-gaap:SharesOutstanding",
        ],
        "Share Issued": ["us-gaap:CommonStockSharesIssued", "us-gaap:SharesIssued"],
        "Common Stock Equity": "us-gaap:StockholdersEquity",
        "Total Equity Gross Minority Interest": "us-gaap:PartnersCapitalIncludingNoncontrollingInterest",
        "Minority Interest": [
            "us-gaap:NoncontrollingInterest",
            "us-gaap:MinorityInterest",
        ],
        "Stockholders Equity": "us-gaap:StockholdersEquity",
        "Retained Earnings": "us-gaap:RetainedEarnings",
        "Additional Paid In Capital": "us-gaap:AdditionalPaidInCapital",
        "Capital Stock": "us-gaap:CapitalStock",
        "Common Stock": "us-gaap:CommonStockValue",
        "Foreign Currency Translation Adjustments": "us-gaap:AccumulatedOtherComprehensiveIncomeLossForeignCurrencyTranslationAdjustment",
        "Total Capitalization": [
            "us-gaap:TotalEquityAndNoncontrollingInterest",
            "us-gaap:DebtAndEquity",
        ],
        "Invested Capital": "us-gaap:TotalEquityAndLiabilities",  # Often calculated, but this is a broad proxy
        # --- Debt and Liabilities ---
        "Total Liabilities Net Minority Interest": "us-gaap:Liabilities",
        "Total Non Current Liabilities Net Minority Interest": "us-gaap:LiabilitiesNoncurrent",
        "Long Term Debt And Capital Lease Obligation": "us-gaap:LongTermDebtAndCapitalLeaseObligation",
        "Long Term Debt": "us-gaap:LongTermDebt",
        "Current Liabilities": "us-gaap:CurrentLiabilities",
        "Current Debt And Capital Lease Obligation": "us-gaap:CurrentDebtAndCapitalLeaseObligation",
        "Current Debt": "us-gaap:DebtCurrent",
        "Debt": "us-gaap:Debt",
        "CommercialPaper(ST debt)": "us-gaap:CommercialPaper",
        "LT Debt Current": "us-gaap:LongTermDebtCurrent",
        "LT Debt Non-current": "us-gaap:LongTermDebtNoncurrent",
        "Accounts Payable Current": "us-gaap:AccountsPayableCurrent",
        "Total Tax Payable": "us-gaap:TaxesPayable",
        "Income Tax Payable": "us-gaap:IncomeTaxesPayable",
        "Payables And Accrued Expenses": "us-gaap:AccountsPayableAndAccruedLiabilities",
        "Other Current Liabilities": "us-gaap:OtherLiabilitiesCurrent",
        "Current Accrued Income Taxes": "us-gaap:AccruedIncomeTaxesCurrent",
        "Current Operating Lease Liability": "us-gaap:OperatingLeaseLiabilityCurrent",
        "Current Finance Lease Liability": "us-gaap:FinanceLeaseLiabilityCurrent",
        "Deferred Revenue": "us-gaap:ContractWithCustomerLiability",
        "Current Deferred Revenue": "us-gaap:ContractWithCustomerLiabilityCurrent",
        "Other Non-Current Liabilities": "us-gaap:OtherLiabilitiesNoncurrent",
        # --- Debt Maturity Schedule ---
        "Debt Due Year 1": "us-gaap:LongTermDebtMaturityObligationsDueInNextTwelveMonths",
        "Debt Due Year 2": "us-gaap:LongTermDebtMaturityObligationsDueInSecondYear",
        "Debt Due Year 3": "us-gaap:LongTermDebtMaturityObligationsDueInThirdYear",
        "Debt Due Year 4": "us-gaap:LongTermDebtMaturityObligationsDueInFourthYear",
        "Debt Due Year 5": "us-gaap:LongTermDebtMaturityObligationsDueInFifthYear",
        "Debt Due After Year 5": "us-gaap:LongTermDebtMaturityObligationsDueAfterFifthYear",
        # --- Assets and Working Capital ---
        "Total Assets": "us-gaap:Assets",
        "Total Non Current Assets": "us-gaap:AssetsNoncurrent",
        "Goodwill And Other Intangible Assets": "us-gaap:IntangibleAssetsNetIncludingGoodwill",
        "Goodwill": "us-gaap:Goodwill",
        "Net PPE": "us-gaap:PropertyPlantAndEquipmentNet",
        "Current Assets": "us-gaap:AssetsCurrent",
        "Inventory": "us-gaap:InventoryNet",
        "Accounts Receivable": "us-gaap:AccountsReceivableNetCurrent",
        "Receivables": "us-gaap:ReceivablesNetCurrent",
        "Cash Cash Equivalents And Short Term Investments": "us-gaap:CashCashEquivalentsAndShortTermInvestments",
        "Cash And Cash Equivalents": "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        "Working Capital": "us-gaap:WorkingCapital",
        "Marketable Securities Current": "us-gaap:MarketableSecuritiesCurrent",
        "Vendor Non Trade Receivables": "aapl:VendorNonTradeReceivables",
        "Other Current Assets": "us-gaap:OtherAssetsCurrent",
        "Marketable Securities Non Current": "us-gaap:MarketableSecuritiesNoncurrent",
        "Other Non Current Assets": "us-gaap:OtherAssetsNoncurrent",
        # --- Revenue ---
        "Total Net Sales": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "Product Net Sales": "us-gaap:RevenueFromContractWithCustomerProduct",  # Or use segment-specific contexts
        "Service Net Sales": "us-gaap:RevenueFromContractWithCustomerService",
        "Total Revenue": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",  # Alias for total net sales
        # --- Income Statement and Operating Metrics ---
        "Total Revenue": "us-gaap:Revenues",
        "Gross Profit": "us-gaap:GrossProfit",
        "Selling General And Administration": "us-gaap:SellingGeneralAndAdministrativeExpense",
        "Operating Income": "us-gaap:OperatingIncomeLoss",
        "Pretax Income": "us-gaap:IncomeLossBeforeIncomeTax",
        "Tax Provision": "us-gaap:IncomeTaxExpenseBenefit",
        "Net Income": "us-gaap:NetIncomeLoss",
        "Diluted EPS": "us-gaap:EarningsPerShareDiluted",
        "Basic EPS": "us-gaap:EarningsPerShareBasic",
        "Operating Revenue": "us-gaap:SalesRevenueNet",
        "Cost Of Revenue": "us-gaap:CostOfRevenue",
        "Cost of Goods and Services sold": "us-gaap:CostOfGoodsAndServicesSold",
        "Cost of Goods sold": "us-gaap:CostOfGoodsSold",
        "Cost of Services": "us-gaap:CostOfServices",
        "Operating Expense": "us-gaap:OperatingExpenses",
        "Net Income Continuous Operations": "us-gaap:NetIncomeLossFromContinuingOperations",
        "Net Income Including Noncontrolling Interests": "us-gaap:NetIncomeLoss",
        "Net Income Common Stockholders": "us-gaap:NetIncomeLossAvailableToCommonStockholders",
        "EBIT": "us-gaap:OperatingIncomeLoss",
        "EBITDA": "us-gaap:EBITDA",
        "Diluted Average Shares": "us-gaap:WeightedAverageNumberOfSharesOutstandingDiluted",
        "Basic Average Shares": "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",
        "Share-based compensation expense": "us-gaap:AllocatedShareBasedCompensationExpense",
        "Share-based compensation": "us-gaap:ShareBasedCompensation",
        # --- Cash Flow Statement ---
        "Operating Cash Flow": "us-gaap:NetCashProvidedByUsedInOperatingActivities",
        "Investing Cash Flow": "us-gaap:NetCashProvidedByUsedInInvestingActivities",
        "Financing Cash Flow": "us-gaap:NetCashProvidedByUsedInFinancingActivities",
        "Changes In Cash": "us-gaap:CashAndCashEquivalentsPeriodIncreaseDecrease",
        "End Cash Position": "us-gaap:CashAndCashEquivalentsAtCarryingValue",  # Already included, but good proxy for ending
        "Beginning Cash Position": "us-gaap:CashAndCashEquivalentsAtCarryingValue",  # Use same tag, context date is start of period
        "Free Cash Flow": [
            "us-gaap:FreeCashFlow",
            "us-gaap:NetCashProvidedByUsedInOperatingActivities",
            "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
        ],  # FCF is derived: OpCF - CapEx
        "Capital Expenditure": [
            "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
            "us-gaap:CapitalExpenditures",
        ],
        "Repurchase Of Capital Stock": "us-gaap:PaymentsForRepurchaseOfCommonStock",
        "Repayment Of Debt": "us-gaap:PaymentsOfDebt",
        "Issuance Of Debt": "us-gaap:ProceedsFromIssuanceOfDebt",
        "Interest Paid Supplemental Data": "us-gaap:InterestPaid",
        "Income Tax Paid Supplemental Data": "us-gaap:IncomeTaxesPaid",
        "Effect Of Exchange Rate Changes": "us-gaap:EffectOfExchangeRateOnCashAndCashEquivalents",
        "Cash Flow From Continuing Financing Activities": "us-gaap:NetCashProvidedByUsedInFinancingActivitiesContinuingOperations",
        "Cash Dividends Paid": "us-gaap:PaymentsOfDividends",
        "Common Stock Dividend Paid": "us-gaap:CommonStockDividendsPaid",
        "Net Common Stock Issuance": "us-gaap:ProceedsFromIssuanceOfCommonStockNet",
        "Net Issuance Payments Of Debt": "us-gaap:NetIncreaseDecreaseInDebt",
        "Long Term Debt Payments": "us-gaap:PaymentsOfLongTermDebt",
        "Long Term Debt Issuance": "us-gaap:ProceedsFromIssuanceOfLongTermDebt",
        "Net Investment Purchase And Sale": "us-gaap:NetCashProvidedByUsedInPurchasesAndSalesOfInvestments",
        "Sale Of Investment": "us-gaap:ProceedsFromSaleOfInvestments",
        "Purchase Of PPE": "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
        "Sale Of PPE": "us-gaap:ProceedsFromSaleOfPropertyPlantAndEquipment",
        "Change In Working Capital": "us-gaap:IncreaseDecreaseInWorkingCapital",
        "Change In Payables And Accrued Expense": "us-gaap:IncreaseDecreaseInAccountsPayableAndAccruedLiabilities",
        "Change In Inventory": "us-gaap:IncreaseDecreaseInInventory",
        "Change In Receivables": "us-gaap:IncreaseDecreaseInReceivables",
        "Depreciation Amortization Depletion": "us-gaap:DepreciationDepletionAndAmortization",
        "Deferred Tax": "us-gaap:DeferredIncomeTaxExpenseBenefit",
        "Gain Loss On Sale Of PPE": "us-gaap:GainLossOnSaleOfPropertyPlantAndEquipment",
        "Net Income From Continuing Operations": "us-gaap:NetIncomeLossFromContinuingOperations",
        # Some less common/more specific tags:
        "Net Short Term Debt Issuance": "us-gaap:NetIncreaseDecreaseInShortTermDebt",
        "Other Non Cash Items": "us-gaap:OtherNoncashItemsAdjustmentNet",
        "Operating Gains Losses": "us-gaap:GainsLossesOnOperatingActivities",
    }

    for accession_number, file_path in filing_paths:
        print(f"\nParsing XBRL data from filing: {accession_number}")

        model_xbrl = None
        try:
            # Add validate=False to load function to skip schema validation
            model_xbrl = ctrl.modelManager.load(file_path, validate=False)

            if not model_xbrl:
                print(f"Skipping {accession_number}: Failed to load XBRL model.")
                continue

            for fact in model_xbrl.facts:
                if fact.value is None or not fact.context:
                    continue

                # Safely determine context type
                context_type = "Unknown"
                if hasattr(fact.context, "isInstant"):
                    context_type = "Instant" if fact.context.isInstant else "Duration"

                fact_date = (
                    str(fact.context.endDatetime).split()[0]
                    if fact.context.endDatetime
                    else None
                )
                if not fact_date:
                    continue

                full_tag = f"{fact.qname.prefix}:{fact.qname.localName}"

                # --- NEW: Extract Dimension Information ---
                dimensions_str = get_dimensions_from_context(fact.context)

                if dimensions_str != "None":
                    continue

                for label, search_tags in target_tags.items():
                    if isinstance(search_tags, str):
                        search_tags = [search_tags]

                    if full_tag in search_tags:
                        all_extracted_facts.append(
                            {
                                "Ticker": COMPANY_TICKER,
                                "Accession Number": accession_number,
                                "Fact Label": label,
                                "XBRL Tag": full_tag,
                                "Dimension(s)": dimensions_str,  # <-- NEW COLUMN
                                "Fact Value": fact.value,
                                "Fact Date": fact_date,
                                "Context Type": context_type,
                                "Unit": (
                                    fact.unit.qname.localName if fact.unit else "None"
                                ),
                            }
                        )

            # Close the model after processing
            if model_xbrl:
                ctrl.modelManager.close(model_xbrl)

        except Exception as e:
            print(f"Error processing {accession_number}: {e}")
            if model_xbrl:
                ctrl.modelManager.close(model_xbrl)

    return all_extracted_facts


def save_to_csv(data):
    """Saves the list of dictionaries to a CSV file."""
    if not data:
        print("\nNo financial facts were extracted to save.")
        return

    # Use data[0].keys() to dynamically get all field names, including the new one
    fieldnames = list(data[0].keys())

    print(f"\nSaving {len(data)} facts to {OUTPUT_CSV_FILE}...")

    with open(OUTPUT_CSV_FILE, "w", newline="", encoding="utf-8") as csvfile:
        # Use DictWriter to easily write dictionaries
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data)

    print("CSV saved successfully.")


if __name__ == "__main__":
    try:
        # 1. Clean up old data and download the new detailed files
        # NOTE: If you have already downloaded the files, you can comment out the next two lines
        # to save time, but it's safest to run them.
        # clean_previous_downloads()
        # download_10k()

        # 2. Locate all the primary documents
        filing_paths = find_all_filing_paths()
        print(f"Found {len(filing_paths)} filings to process.")

        # 3. Parse all files and extract facts
        extracted_facts = extract_financial_data(filing_paths)

        # 4. Save the results to CSV
        save_to_csv(extracted_facts)

    except Exception as e:
        import traceback

        print(f"\nCRITICAL ERROR: {e}")
        traceback.print_exc()
