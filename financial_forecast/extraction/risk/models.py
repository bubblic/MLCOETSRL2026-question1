"""Pydantic schemas for risk extraction pipeline stages.

Provides validation models for LLM responses at each pipeline stage:
Stage 1 (flagging), Stage 2 (category extraction), and Stage 3 (synthesis).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter

from financial_forecast.extraction.risk.risk_categories import RiskCategory


# ---------------------------------------------------------------------------
# Stage 1: Page flagging
# ---------------------------------------------------------------------------


class FlaggedPage(BaseModel):
    """A single page flagged as containing risk-relevant content."""

    page: int
    category: str
    reason: str = ""


# ---------------------------------------------------------------------------
# Stage 2: Category-specific extraction models
# ---------------------------------------------------------------------------


class AuditorOpinionResult(BaseModel):
    opinion_type: Optional[str] = None
    qualification_basis: Optional[str] = None
    emphasis_of_matter: Optional[str] = None
    going_concern_language: Optional[str] = None


class GoingConcernResult(BaseModel):
    disclosure_text: Optional[str] = None
    time_period: Optional[str] = None
    mitigating_actions: Optional[str] = None


class ContingentLiability(BaseModel):
    nature: Optional[str] = None
    exposure: Optional[str] = None
    likelihood: Optional[str] = None
    timeline: Optional[str] = None


class DebtCovenant(BaseModel):
    covenant_type: Optional[str] = None
    condition: Optional[str] = None
    compliance_status: Optional[str] = None
    headroom: Optional[str] = None


class RelatedPartyTransaction(BaseModel):
    counterparty: Optional[str] = None
    relationship: Optional[str] = None
    transaction_nature: Optional[str] = None
    value: Optional[str] = None
    independent_approval: Optional[str] = None


class AccountingPolicyChange(BaseModel):
    policy_changed: Optional[str] = None
    reason: Optional[str] = None
    financial_impact: Optional[str] = None
    restatement: Optional[str] = None


class DirectorChange(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    change_type: Optional[str] = None
    timing: Optional[str] = None
    reason: Optional[str] = None


class CashFlowWarning(BaseModel):
    operating_cf: Optional[str] = None
    net_income: Optional[str] = None
    divergence_flag: Optional[bool] = None
    key_observations: Optional[str] = None


class MdaRedFlag(BaseModel):
    description: Optional[str] = None
    severity: Optional[str] = None
    quoted_text: Optional[str] = None


class MdaRedFlagsResult(BaseModel):
    red_flags: List[MdaRedFlag] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 2: Category → TypeAdapter registry
# ---------------------------------------------------------------------------

CATEGORY_MODELS: Dict[RiskCategory, TypeAdapter] = {
    RiskCategory.AUDITOR_OPINION: TypeAdapter(AuditorOpinionResult),
    RiskCategory.GOING_CONCERN: TypeAdapter(GoingConcernResult),
    RiskCategory.CONTINGENT_LIABILITIES: TypeAdapter(List[ContingentLiability]),
    RiskCategory.DEBT_COVENANTS: TypeAdapter(List[DebtCovenant]),
    RiskCategory.RELATED_PARTY: TypeAdapter(List[RelatedPartyTransaction]),
    RiskCategory.ACCOUNTING_POLICY: TypeAdapter(List[AccountingPolicyChange]),
    RiskCategory.DIRECTOR_CHANGES: TypeAdapter(List[DirectorChange]),
    RiskCategory.CASH_FLOW_WARNINGS: TypeAdapter(CashFlowWarning),
    RiskCategory.MD_AND_A_RED_FLAGS: TypeAdapter(MdaRedFlagsResult),
}


# ---------------------------------------------------------------------------
# Stage 2: Envelope for any category result
# ---------------------------------------------------------------------------

#: The extraction payload after Pydantic validation: a model dict for
#: single-item categories, a list of model dicts for list categories,
#: or ``None`` when no relevant pages were found.
ExtractionResult = Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]


class CategoryResult(BaseModel):
    """Wrapper for a single category's extraction output.

    ``extraction`` is typed as the union of shapes that the pipeline
    actually produces, rather than ``Any``, so a type checker can
    enforce narrowing before field access.
    """

    category: str
    extraction: ExtractionResult = None


# ---------------------------------------------------------------------------
# Stage 3: Synthesis
# ---------------------------------------------------------------------------


class RiskMemoOutput(BaseModel):
    """Structured output from the synthesis stage."""

    risk_memo: str = ""
