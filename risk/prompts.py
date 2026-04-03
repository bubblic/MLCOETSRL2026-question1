"""Prompt templates for the risk warnings extraction pipeline.

All prompts are single strings (no separate system/user) because
:class:`AzureLLMClient` takes a single ``prompt`` parameter.  The
system context is embedded as the opening paragraph of each prompt.
"""

from __future__ import annotations

from risk.risk_categories import RiskCategory

# ---------------------------------------------------------------------------
# Stage 1: Page flagging
# ---------------------------------------------------------------------------

PAGE_FLAG_PROMPT = (
    "You are a financial risk analyst reviewing an annual report.\n"
    "Your job is to identify pages containing potential risk warnings.\n\n"
    "Review the following pages (pages {start} to {end}).\n\n"
    "For each page that contains content related to any risk warning "
    "category, return a JSON array like:\n"
    "[\n"
    '  {{"page": 12, "category": "going_concern", '
    '"reason": "brief explanation"}},\n'
    '  {{"page": 15, "category": "contingent_liabilities", '
    '"reason": "brief explanation"}}\n'
    "]\n\n"
    "Categories to detect: {categories}\n\n"
    "If no relevant pages found, return an empty array: []\n\n"
    "Respond ONLY with valid JSON.\n\n"
    "PAGES:\n{text}"
)

# ---------------------------------------------------------------------------
# Stage 2: Category-specific extraction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPTS = {
    RiskCategory.AUDITOR_OPINION: (
        "You are an expert financial analyst. Extract the following from "
        "this auditor's report section. Respond ONLY with valid JSON.\n\n"
        "Extract:\n"
        "- Type of opinion issued (unqualified/qualified/adverse/disclaimer)\n"
        "- Exact basis for qualification if any\n"
        "- Any emphasis of matter paragraphs\n"
        "- Any going concern language\n\n"
        "Return as JSON:\n"
        '{{"opinion_type": "...", "qualification_basis": "...", '
        '"emphasis_of_matter": "...", "going_concern_language": "..."}}\n\n'
        "Use null for fields not found in the text.\n\n"
        "CONTENT:\n{pages}"
    ),
    RiskCategory.GOING_CONCERN: (
        "You are an expert financial analyst. Extract all going concern "
        "disclosures. Respond ONLY with valid JSON.\n\n"
        "Extract:\n"
        "- Exact language used\n"
        "- Time period referenced\n"
        "- Mitigating actions mentioned by management\n\n"
        "Return as JSON:\n"
        '{{"disclosure_text": "...", "time_period": "...", '
        '"mitigating_actions": "..."}}\n\n'
        "Use null for fields not found in the text.\n\n"
        "CONTENT:\n{pages}"
    ),
    RiskCategory.CONTINGENT_LIABILITIES: (
        "You are an expert financial analyst. Extract all contingent "
        "liabilities. Respond ONLY with valid JSON.\n\n"
        "Extract for each liability:\n"
        "- Nature (lawsuit, regulatory, tax dispute etc)\n"
        "- Estimated financial exposure if stated\n"
        "- Likelihood assessment if stated\n"
        "- Expected resolution timeline if stated\n\n"
        "Return as JSON array:\n"
        '[{{"nature": "...", "exposure": "...", "likelihood": "...", '
        '"timeline": "..."}}]\n\n'
        "Use null for fields not found. Return [] if none found.\n\n"
        "CONTENT:\n{pages}"
    ),
    RiskCategory.DEBT_COVENANTS: (
        "You are an expert financial analyst. Extract all debt covenant "
        "information. Respond ONLY with valid JSON.\n\n"
        "Extract for each covenant:\n"
        "- Covenant conditions described\n"
        "- Current compliance status\n"
        "- Any breaches or waivers mentioned\n"
        "- Financial headroom if quantified\n\n"
        "Return as JSON array:\n"
        '[{{"covenant_type": "...", "condition": "...", '
        '"compliance_status": "...", "headroom": "..."}}]\n\n'
        "Use null for fields not found. Return [] if none found.\n\n"
        "CONTENT:\n{pages}"
    ),
    RiskCategory.RELATED_PARTY: (
        "You are an expert financial analyst. Extract all related-party "
        "transactions. Respond ONLY with valid JSON.\n\n"
        "Extract for each transaction:\n"
        "- Counterparty and their relationship to the company\n"
        "- Nature and value of transaction\n"
        "- Whether approved by independent directors\n\n"
        "Return as JSON array:\n"
        '[{{"counterparty": "...", "relationship": "...", '
        '"transaction_nature": "...", "value": "...", '
        '"independent_approval": "..."}}]\n\n'
        "Use null for fields not found. Return [] if none found.\n\n"
        "CONTENT:\n{pages}"
    ),
    RiskCategory.ACCOUNTING_POLICY: (
        "You are an expert financial analyst. Extract any changes in "
        "accounting policies. Respond ONLY with valid JSON.\n\n"
        "Extract for each change:\n"
        "- Policy that changed\n"
        "- Reason given for change\n"
        "- Financial impact of the change\n"
        "- Whether comparatives were restated\n\n"
        "Return as JSON array:\n"
        '[{{"policy_changed": "...", "reason": "...", '
        '"financial_impact": "...", "restatement": "..."}}]\n\n'
        "Use null for fields not found. Return [] if none found.\n\n"
        "CONTENT:\n{pages}"
    ),
    RiskCategory.DIRECTOR_CHANGES: (
        "You are an expert financial analyst. Extract any director or "
        "senior management changes. Respond ONLY with valid JSON.\n\n"
        "Extract for each change:\n"
        "- Name and role\n"
        "- Nature of change (resignation, appointment, share sale)\n"
        "- Timing\n"
        "- Reason given if any\n\n"
        "Return as JSON array:\n"
        '[{{"name": "...", "role": "...", "change_type": "...", '
        '"timing": "...", "reason": "..."}}]\n\n'
        "Use null for fields not found. Return [] if none found.\n\n"
        "CONTENT:\n{pages}"
    ),
    RiskCategory.CASH_FLOW_WARNINGS: (
        "You are an expert financial analyst. Extract cash flow warning "
        "signals. Respond ONLY with valid JSON.\n\n"
        "Extract:\n"
        "- Operating cash flow vs net income (are they diverging?)\n"
        "- Any negative operating cash flow\n"
        "- Large investing outflows\n"
        "- Financing activities suggesting cash stress "
        "(new debt, equity raises)\n\n"
        "Return as JSON:\n"
        '{{"operating_cf": "...", "net_income": "...", '
        '"divergence_flag": true/false, "key_observations": "..."}}\n\n'
        "Use null for fields not found.\n\n"
        "CONTENT:\n{pages}"
    ),
    RiskCategory.MD_AND_A_RED_FLAGS: (
        "You are an expert financial analyst. Analyse this MD&A section "
        "for red flags. Respond ONLY with valid JSON.\n\n"
        "Look for:\n"
        "- Vague or evasive language around underperformance\n"
        "- External blame (economy, FX, one-offs) used repeatedly\n"
        "- New risks not mentioned in prior disclosures\n"
        "- Contradiction between tone and financial figures\n\n"
        "Return as JSON:\n"
        '{{"red_flags": [{{"description": "...", '
        '"severity": "high/medium/low", "quoted_text": "..."}}]}}\n\n'
        "Return [] for red_flags if none found.\n\n"
        "CONTENT:\n{pages}"
    ),
}

# ---------------------------------------------------------------------------
# Stage 3: Synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = (
    "You are a senior credit risk analyst at an investment bank. "
    "Write a concise, professional risk memo.\n\n"
    "Based on the following extracted risk findings from an annual "
    "report, produce a structured risk memo:\n\n"
    "FINDINGS:\n{findings}\n\n"
    "Your memo should cover:\n"
    "1. Overall risk rating: HIGH / MEDIUM / LOW with one-line "
    "justification\n"
    "2. Top 3 most critical warnings ranked by severity, with "
    "explanation\n"
    "3. Any contradictions found between sections (e.g. optimistic "
    "MD&A but covenant stress in notes)\n"
    "4. Recommended next steps for a credit analyst\n\n"
    "Be direct and professional. Flag anything that would concern "
    "a lender or investor."
)
