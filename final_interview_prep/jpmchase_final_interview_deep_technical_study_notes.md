# JPMChase Final Interview Deep Technical Study Notes

Financial statement simulation, accounting standards, jurisdictions, industry modules, regulatory capital, graph effects, ML validation, and codebase mapping. Generated 2026-05-05.

Purpose: these notes are for interview preparation. They are not accounting, tax, legal, regulatory, investment, or valuation advice. The goal is to explain how a serious ML financial statement simulator should work when it must survive a senior technical interview with a director who cares about actual statements, covenants, jurisdictions, industries, regulatory capital, and marketable product design.

The prior accounting-standards conversation is folded into the framework here: recognition, measurement, presentation, disclosure, consolidation, FX, and normalization are treated as different layers of the system. The key shift from the earlier rough answer is that the application cannot merely read statements and map them to common economic metrics. It must simulate statements that look like legally reportable statements under the relevant standard, then separately compute normalized economic metrics for cross-company comparability.

## Director Concept Coverage Index

| Director concept | Where it is handled | Technical framing |
| --- | --- | --- |
| Covenants and accelerated repayment | Parts 1, 2, 3, 7, 9, 10 | Covenants bind to reported accounting books, not only normalized economic metrics. The system needs a CovenantEngine over reporting-book statements. |
| US parent owning European subsidiary | Part 3 | Separate local statutory books, reporting books, FX translation, consolidation, intercompany elimination, NCI, goodwill, and covenant books. |
| Standard-compliant simulation | Part 2 | AccountingPolicyEngine plus RegulatedStatementGenerator before normalization. |
| Apple/Ireland, Alibaba VIE, SoftBank holdings | Parts 4 and 5 | JurisdictionTaxEngine and EntityGraph with control, ownership, cash-flow rights, tax regimes, consolidation methods, and fair value/NAV. |
| Utilities, mining, oil, luxury, cosmetics, pharma | Part 6 | Industry modules with different state variables, drivers, templates, and validation. |
| Basel III, Solvency II, Tier 1, Tier 2, senior debt | Part 7 | CapitalRegulationEngine maps accounting statements into regulatory ratios and constrains dividends, buybacks, debt issuance, and equity issuance. |
| Apple supplier effects, global/national/sector interactions, Reliance | Part 8 | Macro-sector-company graph with supply chain, customer concentration, FX, rates, commodity, country, and conglomerate propagation. |
| Agentic workflow and product requirements | Parts 9 and 10 | Auditable agents coordinate extraction, assumptions, simulation, validation, scenario comparison, and human approvals. |

# Part 1: Core Thesis

## The Director's Main Correction

The right answer is not simply that ML should recover the underlying economic reality. That is necessary but incomplete. In finance, many hard contractual events are triggered by reported financial statements: leverage covenants, interest coverage covenants, net worth tests, debt incurrence baskets, restricted payment baskets, rating triggers, regulatory capital distributions, and auditor-qualified going-concern language. Therefore the simulator must produce two views at the same time: a regulated reported-statement view and a normalized economic view.

A normalized metric layer answers cross-company questions such as whether one firm is more asset efficient after adjusting for accounting standard differences. A regulated statement generator answers contract and reporting questions such as whether the borrower breaches a covenant under US GAAP, whether a European subsidiary can upstream dividends under local statutory books, whether a bank falls into capital buffer restrictions, or whether an insurance subsidiary has enough own funds under Solvency II.

## Two-Ledger Architecture

The system should maintain at least two synchronized ledgers. The ReportingLedger stores journal events and statement line items under a specified accounting standard, version, jurisdiction, consolidation scope, currency, and policy election. The EconomicLedger stores normalized measures intended to represent economics across standards. The two ledgers are not competitors. They answer different questions and must reconcile through explicit adjustment bridges.

| Layer | Question it answers | Example output | Director relevance |
| --- | --- | --- | --- |
| ReportingLedger | What would the financial statements look like under a specific reporting basis? | US GAAP consolidated income statement, IFRS statutory sub balance sheet, bank regulatory capital schedule | Covenants, debt acceleration, dividend blockers, regulatory capital, filings |
| EconomicLedger | What is the comparable economic performance after removing accounting-basis artifacts? | R&D treated consistently, standardized inventory cost, normalized impairment, operating lease comparability | ML features, peer comparison, valuation drivers, scenario analysis |
| Bridge | How do reported and normalized views differ? | IFRS development asset reversal, LIFO reserve adjustment, impairment reversal exclusion, tax bridge | Auditability, user trust, defensible interview answer |

```text
Event stream
  -> accounting policy selection
  -> journal recognition and measurement
  -> regulated statement generation
  -> covenant and regulatory capital calculations
  -> normalization bridge
  -> ML feature store and scenario valuation

Key invariant:
  reported_statement != normalized_economic_view
  but every normalized adjustment must be traceable to reported line items,
  disclosures, policy elections, and source documents.
```

## Covenants Are Statement-Dependent State Machines

Debt covenants are not abstract economics. They are formulas defined in a credit agreement. The agreement may define EBITDA, debt, interest expense, current assets, current liabilities, restricted payments, permitted acquisitions, and baskets differently from both US GAAP and IFRS. A forecast system that only predicts normalized EBITDA can miss an actual breach if the covenant uses reported EBITDA, excludes some addbacks, caps restructuring adjustments, or uses frozen GAAP rules from the debt issuance date.

| Covenant | Typical formula | Accounting trap | Simulation implication |
| --- | --- | --- | --- |
| Leverage ratio | Net debt / covenant EBITDA | Capitalized development cost, lease debt treatment, addback definitions, EBITDA caps | Generate covenant-book EBITDA and covenant-book debt separately from normalized EBITDA. |
| Interest coverage | EBITDA / cash interest | IFRS interest classification or PIK interest may differ from statement presentation | Debt instrument schedule must compute cash interest, accrued interest, and covenant interest. |
| Current ratio | Current assets / current liabilities | Debt acceleration can reclassify long-term debt into current liabilities | A breach can create a second-order breach through classification and liquidity pressure. |
| Minimum tangible net worth | Equity minus intangibles/goodwill | IFRS development assets and acquisition intangibles can inflate book equity but be excluded | The covenant engine needs line-item tags for tangible and intangible equity components. |
| Restricted payments | Dividends/buybacks limited by retained earnings, baskets, or leverage tests | Reported net income, OCI, and statutory distributable reserves affect capacity | Owner transaction policy must be conditional, not a fixed dividend rule. |

## Event-Sourced Simulation Instead Of One Direct Forecast Table

A robust simulator should not jump directly from historical fields to future fields. It should simulate business events and accounting events: sales contracts, purchases, R&D projects, asset construction, leases, impairment indicators, debt issuance, covenant tests, tax elections, intercompany royalties, dividends, and FX translation. Each event is then recognized and measured under each required reporting basis. This architecture lets the system answer why a line item moved.

```python
class BusinessEvent:
    entity_id: str
    event_date: date
    event_type: Literal[
        "sale", "inventory_purchase", "rd_project", "lease",
        "asset_impairment_indicator", "debt_issue", "tax_payment",
        "intercompany_royalty", "dividend", "fx_remeasurement"
    ]
    economics: dict
    source: Provenance

class AccountingPolicyContext:
    standard: Literal["US_GAAP", "IFRS", "LOCAL_STAT"]
    version: str
    jurisdiction: str
    functional_currency: str
    presentation_currency: str
    policy_elections: dict

def recognize(event, context) -> list[JournalEntry]:
    rule = accounting_policy_engine.lookup(event.event_type, context)
    return rule.apply(event.economics, context.policy_elections)
```

## What To Internalize

- There is no single correct forecasted statement. There is a correct statement for a reporting basis, entity scope, currency, and policy set.
- Normalization is a bridge, not a substitute for regulated reporting simulation.
- Covenants, regulatory capital, tax cash flows, and dividend capacity are downstream state machines that consume reported statements.
- A marketable ML product must show every adjustment, not hide accounting choices inside latent model weights.
- The current codebase is a good operating-company prototype, but it does not yet have the layers required for multi-standard, multi-entity, multi-industry simulation.

# Part 2: Accounting Standards And Reported Statement Simulation

## Recognition, Measurement, Presentation, Disclosure

Accounting standards affect four different layers. Recognition decides whether an item appears as an asset, liability, income, or expense. Measurement decides the amount: historical cost, fair value, amortized cost, recoverable amount, net realizable value, expected credit loss, or present value. Presentation decides where it appears: operating income, finance cost, OCI, investing cash flow, financing cash flow, current liability, noncurrent liability, and so on. Disclosure provides context that may be needed for normalization, such as useful lives, policy elections, LIFO reserve, lease maturity schedule, R&D capitalization policy, impairment assumptions, and risk exposures.

The ML trap is that a raw line item confounds all four layers. If one company reports a higher asset base because development expenditure is capitalized, and another expenses similar work, a model may learn false operational efficiency, false profitability, or false risk. If a covenant is based on reported figures, however, the reported difference is not noise. It can determine whether debt accelerates. That is why the simulator needs both reported and normalized views.

## Standard Metadata Needed On Every Statement

| Metadata field | Why it matters | Example |
| --- | --- | --- |
| standard | Selects recognition and measurement rules | US_GAAP, IFRS, Ind_AS, ASBE, local statutory GAAP |
| standard_version | Standards change over time | IFRS 17 effective from annual periods beginning 2023-01-01 |
| entity_id | Statements exist at legal-entity level before consolidation | AAPL_US_PARENT, AAPL_IE_SUB |
| book_type | Different books serve different purposes | local_statutory, group_reporting, covenant, tax, normalized |
| functional_currency | FX measurement currency under IAS 21 style logic | EUR for a European subsidiary |
| presentation_currency | Currency used for group reporting | USD for US parent consolidated financials |
| policy_elections | Many standards allow choices | Inventory cost formula, IFRS interest paid classification |
| provenance | Auditability and LLM-hallucination control | 10-K note, annual report page, XBRL tag, extraction confidence |

## R&D And Intangibles: IAS 38 Versus US GAAP

IAS 38 treats intangible assets as identifiable non-monetary assets without physical substance. It distinguishes research from development. Research expenditure is expensed, while development expenditure that meets specified criteria is capitalized and then amortized or tested for impairment depending on useful life. US GAAP generally expenses research and development as incurred under ASC 730, with specific exceptions such as certain software accounting rules. The modeling point is not merely that one number differs. The entire path of income, assets, amortization, impairment, tax, EBITDA, ROA, debt capacity, and covenants can differ.

| Mini case | US GAAP reported view | IFRS reported view | Normalized economic bridge |
| --- | --- | --- | --- |
| A software company spends 100 on R&D. Suppose 40 is research and 60 meets IFRS development criteria, amortized over 5 years starting next year. | Expense 100 in year 0. Assets unchanged except cash. Lower EBIT and net income now. No future amortization from this project. | Expense 40 in year 0. Capitalize 60 as development intangible. Year 0 income higher by 60 before tax. Future amortization of 12 per year. | For comparability, either expense all development or capitalize all qualifying development for both firms. Record a bridge: +60 IFRS asset, -60 IFRS expense deferral, future -12 amortization reversal if expensing convention is selected. |
| ML feature risk | Model sees lower income and lower assets for US GAAP firms. | Model sees higher income and higher intangible assets for IFRS firms. | Without standard tags, the model may learn that IFRS software companies are more profitable or more asset heavy even when economics are identical. |

For a reported-statement simulator, the R&D module must project R&D projects, split research from development, test capitalization criteria under the selected standard, schedule amortization, test impairment, and provide disclosure-derived assumptions. For a normalized metric layer, the same module must be able to reverse or restate the reported accounting policy into a common economic convention.

## Inventory Costing: FIFO, Weighted Average, LIFO, And Inflation

IAS 2 provides guidance on inventory cost and subsequent recognition of cost as expense, including write-down to net realizable value. IFRS does not permit LIFO. US GAAP permits LIFO. In an inflationary environment, LIFO produces higher COGS and lower ending inventory than FIFO, reducing reported profit and often taxes. If a model treats the result as operational efficiency, it will mistake an accounting policy election for supply chain performance.

| Assumption | FIFO | LIFO | ML/covenant implication |
| --- | --- | --- | --- |
| Units sold | 100 | 100 | Same physical business activity |
| Old cost layer | 100 units at 10 | 100 units at 10 | Historic layer exists in both systems |
| New cost layer | 100 units at 15 | 100 units at 15 | Inflation creates policy sensitivity |
| COGS for 100 units | 100 x 10 = 1,000 | 100 x 15 = 1,500 | LIFO firm appears lower margin |
| Ending inventory | 100 x 15 = 1,500 | 100 x 10 = 1,000 | LIFO firm appears less asset heavy |
| Tax and cash | Higher taxable income | Lower taxable income in inflation | Cash taxes differ; economic cash can differ due tax law |

## Impairment: IAS 36, US GAAP, Reversals, And Timing

IAS 36 requires that an asset not be carried above the amount recoverable through use or sale. If carrying amount exceeds recoverable amount, the asset is impaired. IFRS generally permits reversal of impairment losses for non-goodwill assets when estimates improve, subject to limits; goodwill impairment reversals are not permitted. US GAAP generally prohibits reversal of impairment losses for long-lived assets held and used and for goodwill. The nuance matters because an IFRS company can show a future income gain from reversal that a US GAAP company cannot report.

The simulation requirement is an impairment state machine. It should track carrying amount, recoverable amount or fair value estimates, cash-generating units, useful life, prior impairment, standard-specific reversibility, and whether the asset is goodwill, indefinite-lived intangible, finite-lived intangible, PPE, inventory, or financial asset. A generic depreciation model cannot handle this.

## Leases: Balance Sheet Similarity Does Not Mean Statement Similarity

Modern US GAAP and IFRS both put most leases on the lessee balance sheet, but presentation and expense pattern can still differ. IFRS 16 uses a single lessee model with depreciation of the right-of-use asset and interest on the lease liability, while US GAAP ASC 842 preserves operating and finance lease classification for lessees. That means EBITDA, operating income, interest, current liabilities, financing cash flow, and covenant debt may differ even when the underlying leased asset is identical.

- A covenant may define lease liabilities as debt or may exclude operating leases depending on negotiated language.
- A normalized metric layer may capitalize operating leases for comparability, but a covenant engine must use the covenant definition.
- Lease maturity disclosures are essential because the balance sheet alone does not provide a complete future cash obligation schedule.

## Cash Flow Classification: The Same Cash Can Move Categories

Cash flow statement classification matters because many credit metrics use operating cash flow, free cash flow, or cash interest. IFRS permits more classification choices for interest and dividends than US GAAP, subject to consistency and standard requirements. Under US GAAP, interest paid is typically operating, dividends paid financing, and dividends received operating for many entities. A model that uses CFO as a feature without standard and policy tags can treat presentation choices as operating quality.

## Foreign Private Issuer Filing Rules

A foreign private issuer listed in the United States can present financial statements using US GAAP, IFRS as issued by the IASB, or home-country accounting standards with reconciliation to US GAAP. This means a US-listed security is not enough to infer US GAAP. The ingestion pipeline must capture issuer status, filing form, reporting basis, reconciliation basis, and whether the IFRS statements are IFRS as issued by the IASB or a jurisdictional endorsement.

## Standard-Compliant Generation Mechanics

```python
def simulate_period(entity, opening_state, scenario, reporting_context):
    events = business_event_model.generate(entity, opening_state, scenario)
    entries = []
    for event in events:
        entries.extend(accounting_policy_engine.recognize(event, reporting_context))
    trial_balance = ledger.post(opening_state.trial_balance, entries)
    statements = regulated_statement_generator.present(
        trial_balance=trial_balance,
        standard=reporting_context.standard,
        statement_template=reporting_context.statement_template,
        currency=reporting_context.presentation_currency,
    )
    covenant_metrics = covenant_engine.compute(statements, reporting_context.covenants)
    normalized = normalized_metric_layer.bridge(statements, reporting_context)
    return statements, covenant_metrics, normalized
```

The important design choice is sequencing. Do not normalize first and then infer reported statements. Generate the reported statements first under the correct policy context. Then create a documented bridge to normalized economics. This is exactly the director's point: actual statements matter because contracts and regulation read actual statements.

# Part 3: Consolidation And Multi-Entity Simulation

## A US Parent Owning A European Subsidiary

The director's question about a US company owning a European company is a consolidation problem, not just a standards lookup. The European subsidiary may keep local statutory books under local GAAP for legal filing and distributable reserves, prepare IFRS reporting packages for group reporting, and then be adjusted to US GAAP if the US parent reports under US GAAP. The group then translates the subsidiary from functional currency to presentation currency, eliminates intercompany balances, recognizes non-controlling interest if ownership is less than 100 percent, and computes covenants on the relevant borrower group.

## EntityGraph Data Model

```python
class EntityNode:
    entity_id: str
    legal_name: str
    jurisdiction: str
    functional_currency: str
    local_statutory_standard: str
    group_reporting_standard: str
    tax_regime_id: str
    industry_module: str
    regulatory_regime_ids: list[str]

class OwnershipEdge:
    parent_id: str
    child_id: str
    legal_ownership_pct: float
    voting_rights_pct: float
    economic_interest_pct: float
    control_indicator: bool
    consolidation_method: Literal["full", "equity_method", "fair_value", "none"]

class IntercompanyFlow:
    from_entity: str
    to_entity: str
    flow_type: Literal["sale", "loan", "royalty", "dividend", "service_fee"]
    amount: float
    currency: str
    tax_transfer_pricing_policy: str
```

| Step | Simulation action | Why it matters |
| --- | --- | --- |
| 1. Local books | Simulate each legal entity under local statutory or local GAAP rules. | Local taxes, dividend capacity, regulatory filings, and legal restrictions depend on entity-level books. |
| 2. Reporting adjustments | Convert local books to group reporting basis: IFRS group or US GAAP group. | The parent consolidated statements must follow the parent reporting standard. |
| 3. FX measurement | Remeasure foreign-currency transactions and translate foreign operations. | Income, assets, liabilities, OCI, CTA, and covenant debt may move with FX. |
| 4. Consolidation | Apply control model, combine controlled entities, recognize NCI, eliminate intercompany balances. | Avoid double counting and create group-level statements. |
| 5. Covenant book | Apply debt-agreement definitions to the borrower group. | Covenants may exclude unrestricted subsidiaries or use frozen GAAP definitions. |
| 6. Normalized view | Bridge reported group statements to economic metrics. | Supports ML comparability and valuation analysis. |

## FX Translation Under IAS 21-Style Logic

IAS 21 focuses on foreign currency transactions, translation of foreign operations into the entity's functional currency, and translation into a presentation currency. The simulator must distinguish transaction currency, functional currency, and presentation currency. This distinction is crucial for a US parent with a EUR functional-currency subsidiary reporting consolidated USD statements.

- Foreign-currency transaction remeasurement affects profit or loss when a monetary item is denominated in a currency other than the entity's functional currency.
- Foreign operation translation usually translates assets and liabilities at closing rates and income statement items at average rates, with translation differences often recorded in OCI as CTA.
- A USD covenant may use translated debt. FX depreciation in the subsidiary currency can change leverage even if local operations are stable.
- A tax model must separately track realized FX gains/losses, taxable FX treatment, and accounting FX presentation.

## Acquisition Accounting, Goodwill, And NCI

When the US parent acquires the European subsidiary, the simulator needs purchase price allocation. Identifiable acquired assets and liabilities are recognized at fair value, separately identifiable intangibles may be created, deferred tax liabilities may arise from fair value step-ups, and goodwill is the residual. If the parent owns less than 100 percent, the consolidated group recognizes non-controlling interest. Future impairment, amortization, and FX translation affect the consolidated statements.

| Item | Technical treatment | Pitfall |
| --- | --- | --- |
| Goodwill | Residual after consideration, NCI, and fair value of net identifiable assets. | Do not amortize under US GAAP/IFRS for public companies; test impairment. Local statutory treatment can differ. |
| Customer relationships | Recognized as identifiable intangible if separable or contractual. | Creates amortization and deferred tax effects that were absent pre-acquisition. |
| NCI | Represents external shareholders' claim on controlled subsidiaries. | Consolidated revenue/assets are 100 percent, but equity and earnings are attributed between parent and NCI. |
| Intercompany sale | Eliminate parent/sub revenue, COGS, receivable/payable, and unrealized inventory profit. | Failure creates fake growth and fake working capital. |
| Intercompany loan | Eliminate group receivable/payable and interest income/expense. | Entity-level covenants and withholding taxes may still depend on the loan before elimination. |

## Consolidation Algorithm

```python
def consolidate_group(entity_graph, scenario, parent_reporting_context):
    standalone = {}
    for entity in topological_order(entity_graph):
        local = simulate_period(entity, entity.opening_state, scenario, entity.local_context)
        reporting = convert_to_group_basis(local, parent_reporting_context.standard)
        translated = translate_to_presentation_currency(reporting, parent_reporting_context.currency)
        standalone[entity.id] = translated

    consolidated = combine_controlled_entities(standalone, entity_graph)
    consolidated = eliminate_intercompany(consolidated, entity_graph.intercompany_flows)
    consolidated = recognize_nci(consolidated, entity_graph.ownership_edges)
    consolidated = apply_acquisition_accounting(consolidated, entity_graph.acquisitions)

    covenant_scope = select_borrower_group(consolidated, parent_reporting_context.debt_docs)
    covenant_metrics = covenant_engine.compute(covenant_scope)
    normalized = normalized_metric_layer.bridge(consolidated, parent_reporting_context)
    return consolidated, covenant_metrics, normalized
```

## Director-Relevant Takeaway

If asked directly, the answer is: simulate the two companies as legal entities first, not as one merged vector. Each entity has its own standard, jurisdiction, currency, tax rules, and industry drivers. Then convert to the parent reporting basis, translate currencies, eliminate intercompany items, recognize NCI and goodwill, and compute covenants on the exact covenant scope. Only after that should we build normalized economic features for ML.

# Part 4: Jurisdiction, Tax, Sanctions, And Legal Structures

## Jurisdiction Is Not A Dummy Variable

Jurisdiction affects tax, subsidies, capital controls, sanctions, dividend withholding, legal entity structure, reporting standard, labor costs, inflation, FX, local interest rates, insolvency regime, and sector-specific regulation. A model that uses a country flag without a jurisdiction engine can learn averages but cannot simulate the mechanics needed for a client-facing product.

## JurisdictionTaxEngine

```python
class TaxRegime:
    jurisdiction: str
    corporate_tax_rate: float
    minimum_tax_rules: dict
    withholding_tax_rates: dict[tuple[str, str], float]
    credit_rules: dict
    subsidy_rules: dict
    loss_carryforward_rules: dict
    deferred_tax_rules: dict
    transfer_pricing_rules: dict

def compute_entity_tax(entity_statement, tax_regime, intercompany_flows):
    taxable_income = tax_regime.adjust_book_to_tax(entity_statement)
    current_tax = apply_rates_and_minimum_tax(taxable_income, tax_regime)
    withholding = compute_withholding(intercompany_flows, tax_regime)
    credits = compute_credits(entity_statement, tax_regime)
    subsidies = compute_subsidies(entity_statement, tax_regime)
    deferred_tax = update_deferred_tax_assets_liabilities(entity_statement, tax_regime)
    cash_tax = current_tax + withholding - credits
    return TaxResult(cash_tax, current_tax, deferred_tax, subsidies)
```

## Apple/Ireland As A Modeling Archetype

Use Apple/Ireland carefully as an interview archetype rather than making a brittle claim about the exact current legal structure. The modeling issue is that valuable IP, cost-sharing, transfer pricing, non-US sales, Irish entities, US parent control, withholding taxes, US tax rules, OECD Pillar Two style minimum tax, and deferred taxes can interact. The consolidated financial statement may show one effective tax rate, while cash taxes, entity-level statutory profits, and tax exposures sit in different jurisdictions.

| Mechanic | Study note | ML/product pitfall |
| --- | --- | --- |
| Transfer pricing | Intercompany royalties or cost-sharing allocate profit among entities. | Model may attribute high margin to operating efficiency instead of IP/tax structure. |
| Deferred tax | Book-tax timing differences create DTAs/DTLs and future cash-tax effects. | Forecasting tax as EBT x rate misses reversal schedules. |
| Withholding tax | Cross-border dividends, royalties, and interest can trigger source-country withholding. | Consolidated tax rate may not capture cash leakage from repatriation. |
| Tax credits/subsidies | Credits can reduce cash tax or asset cost and may be conditional. | Asset efficiency and margin can be overstated without subsidy tags. |
| Minimum tax | Global minimum-tax regimes can cap benefits from low-tax structures. | Historical effective tax rate may not extrapolate. |

## Alibaba Cayman/VIE Structure

A variable interest entity structure separates legal ownership, contractual control, economics, and cash-flow rights. A Cayman listed holding company may not directly own restricted operating assets in China. Instead, it may have contractual arrangements with domestic operating companies. The accounting question is whether the reporting entity consolidates the VIE because it has power over the activities that most affect economic performance and exposure to variable returns. The legal risk question is different: contractual control can be less robust than direct equity ownership.

- The EntityGraph needs separate fields for legal ownership, voting rights, contractual control, economic exposure, and consolidation conclusion.
- Cash trapped in an operating jurisdiction is not the same as cash freely available to the offshore holding company.
- A valuation model should assign scenario probabilities to enforceability, regulatory intervention, cash repatriation, and delisting or listing venue risk.
- A statement simulator should not assume consolidated assets are legally owned by the listed parent.

## Subsidies And Tax Credits

Government support can appear through reduced asset cost, grant income, lower tax expense, refundable credits, subsidized loans, guaranteed offtake, or regulated returns. The accounting presentation can vary. A factory subsidy may reduce the carrying amount of PPE or be recognized as deferred income. A tax credit may reduce tax expense or be treated as government assistance depending on law and accounting policy. The economic issue is whether the support is recurring, conditional, clawback-prone, or politically exposed.

## Sanctions As Scenario Constraints

Sanctions are not just a revenue haircut. They can block customers, suppliers, banks, currencies, shipping routes, technology exports, insurance, working capital financing, and asset sales. A sanction scenario should update the supply chain graph, allowed counterparties, payment rails, FX convertibility, discount rate, default probability, and impairment indicators. It may also force inventory write-downs, receivable impairments, restructuring charges, or deconsolidation if control is lost.

| Shock | Statement effect | Modeling requirement |
| --- | --- | --- |
| Export ban | Lower sales, inventory build, possible write-down, lower receivables collections. | Customer-country and product restriction matrix. |
| Banking sanction | Higher financing cost, inability to refinance, trapped cash. | Counterparty and payment-rail constraints. |
| Technology sanction | Capex delays, lost suppliers, impairment of project assets. | Supplier graph and project-level asset tracking. |
| Currency controls | FX losses, inability to upstream dividends, valuation discount. | Functional/presentation currency and cash repatriation module. |

# Part 5: Cross-Holdings And Fair Value

## SoftBank-Style Holding Company Problem

A holding company with major investments across jurisdictions cannot be valued by a single operating company forecast. The system must decide for each investee whether it is fully consolidated, accounted for under the equity method, measured at fair value through profit or loss, measured at fair value through OCI, or excluded from consolidation. It must then compute a group view and a look-through NAV view without double counting debt, cash, minority interests, or cross-holdings.

| Investment type | Accounting view | Valuation view | Trap |
| --- | --- | --- | --- |
| Controlled subsidiary | Full consolidation with NCI if not 100 percent owned. | Enterprise value of sub less net debt, then allocate to parent and NCI. | Do not count subsidiary market cap plus consolidated assets. |
| Associate | Equity method: share of profit/loss and carrying value. | Market value if public, DCF/comps if private, adjusted for ownership. | Equity method carrying value can be stale relative to fair value. |
| Minority financial investment | Fair value through P&L or OCI depending classification. | Mark-to-market or scenario fair value. | OCI/P&L classification can distort earnings features. |
| Fund investment | May be fair value with fees/carry and liquidity constraints. | NAV, discount, lock-up, look-through exposure. | Private marks can lag market reality. |
| Cross-held investment | May create circular ownership. | Solve ownership matrix for effective economic exposure. | Naive aggregation double counts value. |

## NAV Mechanics

```text
Parent_NAV =
    parent_standalone_cash
  + sum(parent_ownership_i * fair_value_i)
  - parent_standalone_debt
  - holding_company_overheads_value
  - tax_leakage_on_disposals
  - liquidity_discount
  - governance_or_complexity_discount

Look-through_leverage =
    parent_standalone_debt
  + sum(parent_ownership_i * investee_net_debt_i)
  + guaranteed_or_recoursed_debt

Circular holdings:
    effective_ownership = inverse(I - cross_holding_matrix) * direct_ownership
```

The director's SoftBank question is asking whether the system can jointly simulate a network of holdings instead of independently forecasting each company. The answer should emphasize an entity and ownership graph, correlated scenarios, FX, country risk, financing links, liquidity needs at the parent, and a bridge between accounting carrying values and fair values.

## Correlation And Scenario Dependence

A holding company's portfolio value is not the sum of independent forecasts. Technology holdings may share interest-rate duration, venture funding cycles, semiconductor supply constraints, AI capex cycles, regulatory scrutiny, and exit-market liquidity. A macro scenario should shock investees jointly through common factors. A parent liquidity crisis can force asset sales at discounts, which feeds back into NAV and credit spreads.

| Risk factor | Investee-level effect | Holding-company effect |
| --- | --- | --- |
| USD/JPY or local FX | Translated earnings and fair value move. | NAV, leverage, and covenant headroom move. |
| Rates | Discount rates, debt cost, growth-stock multiples. | Portfolio markdowns and refinancing pressure. |
| IPO market closure | Private holdings cannot exit at model value. | Liquidity discount and parent debt refinancing risk rise. |
| Sector regulation | Specific investee revenue/margin shock. | Concentration risk and correlation in NAV. |
| Margin call/collateral | Listed holdings pledged against debt may trigger forced sale. | Debt policy must include collateral value and cure mechanics. |

# Part 6: Industry-Specific Financial Statements

## Why A Single Statement Template Fails

Different industries are not just different coefficient values in the same model. They often need different state variables, statement templates, accounting policies, drivers, regulatory constraints, and validation rules. The current operating-company state vector is suitable for a manufacturer or large technology hardware company style model with sales, purchases, inventory, receivables, payables, cash, market securities, debt, equity, net income, and dividends. It does not represent loan books, deposits, insurance liabilities, mineral reserves, rate base, drug pipelines, or brand portfolios.

| Industry | Core state variables | Key accounting issues | Forecast drivers |
| --- | --- | --- | --- |
| Tech/software | R&D projects, capitalized software, deferred revenue, cloud capex, stock comp, IP. | R&D capitalization differences, software revenue recognition, intangibles, share-based compensation. | Users, seats, pricing, cloud utilization, AI capex, renewal/churn. |
| Pharma/biotech | Pipeline by phase, patents, acquired IPR&D, inventory, milestones, royalty streams. | Research vs development, acquired in-process R&D, impairment, collaboration revenue. | Clinical success probabilities, patent cliffs, approvals, pricing regulation. |
| Oil/gas/mining | Reserves, wells/mines, depletion, ARO, commodity hedges, inventory, capex projects. | Successful efforts/full cost, reserve revisions, impairment, asset retirement obligations. | Commodity prices, production volumes, decline curves, capex, royalties. |
| Utilities | Rate base, regulated assets, allowed ROE, fuel pass-through, regulatory assets/liabilities. | Regulatory accounting, deferred costs, allowed returns, capex recovery. | Demand, weather, allowed ROE, capex plan, regulator decisions. |
| Luxury/cosmetics | Brand equity, inventory, boutiques, distribution, marketing assets. | Internally generated brand usually not recognized; acquired brands recognized. | Price/mix, region demand, brand heat, channel inventory, FX tourism. |
| Banks | Loans, deposits, securities, RWA, CET1, liquidity coverage, credit losses. | CECL/IFRS 9 ECL, fair value securities, regulatory capital. | Rates, loan growth, deposit beta, credit losses, RWA density. |
| Insurers | Premiums, reserves, CSM, risk adjustment, investment portfolio, SCR. | IFRS 17, actuarial liabilities, Solvency II capital. | Claims, lapse, mortality/morbidity, investment yield, capital ratio. |
| Conglomerates | Segment-level states and intersegment flows. | Mixed accounting and consolidation across industries. | Segment drivers plus capital allocation and holding-company debt. |

## Tech And Software

For technology companies, the most important assets may be unrecognized internally generated intangibles: code base, network effects, data, engineering organization, customer relationships, and brand. Under many accounting regimes, internally generated goodwill and many internally generated brands are not recognized as assets. That means book assets can severely understate economic assets. Conversely, acquired intangibles from M&A can make an acquisitive company look more asset heavy than an organically built company.

- A software simulator should track ARR, deferred revenue, RPO/backlog, churn, expansion, cloud infrastructure cost, sales commissions, capitalized contract costs, R&D projects, and stock compensation.
- The ML trap is comparing organic and acquisitive companies on book intangible intensity without controlling for acquisition history.
- For covenant purposes, adjusted EBITDA may add back stock compensation or restructuring expenses, but those addbacks are negotiated and capped.

## Pharma And Biotech

Pharma valuation depends on project-level probability distributions. A mature product faces patent cliff erosion; a pipeline asset moves through trial phases; an acquired asset may create identifiable intangibles or acquired in-process R&D; collaboration revenue may include milestones, royalties, and performance obligations. A generic OpEx ratio cannot represent this.

```text
drug_asset_value =
    probability_of_approval_by_phase
  * present_value(expected_peak_sales * margin * exclusivity_curve)
  - remaining_R_and_D_cost
  - commercialization_cost

statement_effects:
  trial failure -> impairment or R&D expense, lower future sales
  approval -> inventory build, launch costs, revenue ramp
  patent cliff -> price/volume erosion, possible impairment
```

## Oil, Gas, And Mining

Resource companies need reserve and production physics. The balance sheet includes PPE, exploration and evaluation assets, development assets, decommissioning or asset retirement obligations, inventory, environmental liabilities, and sometimes commodity hedges. The income statement depends on production volume, realized commodity price, royalties, lifting costs, depreciation/depletion/amortization, and impairment. Reserve revisions can change depletion rates and impairment conclusions.

## Utilities

Regulated utilities are driven by rate base and allowed return. Capex does not simply lower free cash flow; it can increase rate base and future allowed earnings if approved by regulators. Regulatory assets and liabilities can defer costs or refunds. The important state variables are allowed ROE, rate base, capital structure allowed by regulators, fuel/pass-through mechanisms, demand, weather, and regulatory lag.

## Luxury And Cosmetics

Luxury and cosmetics companies often rely on brand value, distribution control, pricing power, and marketing efficiency. Internally generated brand value is usually not recognized, while acquired brands may be. Inventory obsolescence, channel stuffing, gray market flows, tourism, China demand, FX, and advertising intensity can matter more than generic working capital ratios.

## Conglomerates

Conglomerates require segment simulators and a capital allocation layer. Reliance-style groups can span energy, petrochemicals, telecom, retail, digital platforms, financial services, and infrastructure. A parent-level statement hides segment economics. The simulator should model each segment with its own industry driver set, then consolidate with intersegment flows, group debt, minority interests, and holding-company capital allocation.

# Part 7: Banks, Insurers, And Regulatory Capital

## Banks Are Not Operating Companies With More Debt

A bank's liabilities, especially deposits and wholesale funding, are raw material for the business. Loans and securities are earning assets, not ordinary receivables or market securities. Net interest income, credit losses, RWA, liquidity coverage, deposit beta, duration risk, and capital ratios drive corporate policy. Treating deposits as generic debt or loans as generic assets destroys the economics.

| Operating-company model | Bank model |
| --- | --- |
| Revenue from selling goods/services | Interest income, fees, trading, asset management, payments, investment banking |
| Inventory and receivables are operating working capital | Loans are earning assets with credit risk, maturity, collateral, and risk weights |
| Debt is a financing decision | Deposits and wholesale funding are core funding inputs |
| Capex drives PPE | RWA, duration, liquidity buffer, branch/technology investment, securities portfolio |
| Tax = EBT x rate might be a rough approximation | Tax still matters, but regulatory capital and provisioning can dominate policy constraints |

## Basel III Capital Stack

The Basel Framework sets globally agreed standards for prudential regulation of internationally active banks and includes the definition of capital, risk-based capital, leverage ratio, liquidity coverage, net stable funding, large exposures, supervisory review, and disclosures. A bank simulator must project accounting statements and regulatory schedules together.

| Instrument/capital layer | Technical role | Corporate policy effect |
| --- | --- | --- |
| CET1 | Common equity tier 1 after regulatory deductions and adjustments. | Primary loss-absorbing capital; dividends and buybacks constrained when buffers are weak. |
| AT1 / Additional Tier 1 | Perpetual subordinated instruments with discretionary coupons and loss-absorption features. | Can support capital ratios, but coupons may be restricted; investors price call and trigger risk. |
| Tier 2 | Subordinated debt with loss-absorption capacity, subject to eligibility and maturity amortization rules. | Helps total capital, not CET1; refinancing and maturity schedule matter. |
| Senior debt | Ordinary senior funding; may count for TLAC/MREL-style loss absorbing capacity in some regimes but not regulatory capital in the same way as Tier 2. | Affects liquidity, funding cost, and resolution capacity; not a substitute for CET1. |
| RWA | Risk-weighted assets for credit, market, operational, and other risks. | Loan mix, trading book, credit migration, and operational risk change capital ratios even if total assets are stable. |
| Leverage exposure | Non-risk-weighted exposure denominator. | Constrains balance sheet growth even when risk weights look low. |

## Bank Corporate Policy State Machine

```python
def bank_policy_step(bank_state, scenario):
    nii = compute_net_interest_income(bank_state.assets, bank_state.funding, scenario.rates)
    fees = compute_fee_income(bank_state.business_mix, scenario.activity)
    provisions = expected_credit_loss_model(bank_state.loan_book, scenario.macro)
    accounting_income = nii + fees + trading_income - opex - provisions - tax

    rwa = compute_rwa(bank_state.loan_book, bank_state.trading_book, scenario.credit)
    cet1 = update_cet1(bank_state.cet1, accounting_income, dividends, deductions)
    tier1 = cet1 + eligible_at1
    total_capital = tier1 + eligible_tier2

    constraints = capital_regulation_engine.check(
        cet1_ratio=cet1 / rwa,
        tier1_ratio=tier1 / rwa,
        total_capital_ratio=total_capital / rwa,
        leverage_ratio=tier1 / leverage_exposure,
        buffers=bank_state.capital_buffers,
    )

    dividends, buybacks, at1_issue, tier2_issue, senior_issue, equity_issue =         choose_policy_actions(constraints, funding_plan, management_targets)
    return next_bank_state
```

This is the answer to the director's Tier 2, Tier 1, and senior debt question. These instruments are not interchangeable liabilities. They sit in a regulatory and contractual hierarchy. The model must choose issuance, redemption, coupon, call, and dividend actions subject to capital ratios, buffers, market access, maturity walls, rating implications, and management targets.

## Insurance Under IFRS 17 And Solvency II

Insurers also require separate simulators. IFRS 17 measures groups of insurance contracts using fulfilment cash flows plus or minus a contractual service margin, recognizes profit over the period services are provided, and separates insurance service result from insurance finance income or expense. Solvency II is the EU prudential regime for insurers and reinsurers, with risk-based capital, market-consistent valuation, governance, ORSA, group supervision, and public disclosure.

| Concept | Meaning | Simulation consequence |
| --- | --- | --- |
| Fulfilment cash flows | Risk-adjusted present value of expected future cash flows. | Requires actuarial projection of claims, premiums, expenses, discount rates, and risk adjustment. |
| CSM | Contractual service margin: unearned profit in a group of contracts. | Profit emerges as service is provided; new business strain and release patterns matter. |
| Loss component | Created when a group of contracts is onerous. | Loss recognized earlier, not spread as profitable CSM. |
| SCR | Solvency Capital Requirement. | Capital policy and asset allocation constrained by risk-based capital needs. |
| MCR | Minimum Capital Requirement. | Harder floor with severe supervisory consequences. |
| Own funds | Eligible capital resources under Solvency II. | Dividends and group remittances depend on local solvency position. |

## Insurance Corporate Policy

Insurance corporate policy is capital-driven. A profitable insurer may still be unable to dividend cash to the parent if the local solvency ratio is weak. Asset allocation affects solvency capital because market risk, credit risk, duration mismatch, and concentration risk drive capital charges. Reinsurance can reduce risk but introduces counterparty credit exposure and cost. The simulator therefore needs an asset-liability and solvency module, not merely an income statement module.

# Part 8: Macro, Sector, And Conglomerate Interaction

## Global Economy To Company Forecasts

The director's Apple supplier and multinational comments point to a graph, not a flat feature table. Global variables affect national economies; national economies affect sectors; sectors affect companies; companies affect suppliers and customers; ownership and financing links feed back into valuation and default risk. The model should propagate shocks along typed edges.

```text
GlobalFactors:
    rates_usd, rates_eur, oil_price, semiconductor_cycle, inflation, risk_premium

CountryFactors:
    GDP_growth, FX, wage_growth, tax_policy, sanctions, consumer_demand

SectorFactors:
    demand_index, input_cost_index, capacity_utilization, regulation_index

CompanyFactors:
    revenue_volume, price_mix, gross_margin, capex, working_capital, funding_cost

Edges:
    supplier_to_customer, customer_concentration, competitor, ownership,
    financing, commodity_input, currency_exposure, regulatory_exposure
```

## Apple Supplier Dependence

If Apple demand falls, suppliers with high customer concentration experience lower revenue, lower capacity utilization, inventory write-downs, receivable risk, delayed capex, covenant pressure, and possibly rating downgrades. Some suppliers may be in different jurisdictions and functional currencies, so the shock can combine customer demand, FX, working capital, and financing effects.

| Shock path | Supplier statement effect | Model requirement |
| --- | --- | --- |
| Apple unit demand down | Revenue and production volume decline. | Customer concentration edge with elasticity. |
| Component order cancellation | Inventory build and possible NRV write-down. | Inventory aging and product-specific obsolescence. |
| Payment terms extended | Receivables increase and cash conversion worsens. | Counterparty-specific working capital terms. |
| Factory utilization falls | Fixed costs spread over fewer units; margin compression. | Cost structure and capacity model. |
| Covenant headroom shrinks | Debt may become current or refinancing cost rises. | CovenantEngine and debt maturity schedule. |

## Reliance-Style Conglomerate

A conglomerate such as Reliance-style groups requires segment models. Energy and petrochemicals respond to commodity spreads and refining margins; telecom responds to subscribers, ARPU, spectrum costs, capex, and regulation; retail responds to same-store sales, inventory, logistics, and consumer demand; financial services respond to credit, rates, and capital. A parent capital allocation layer decides dividends, asset sales, debt issuance, equity issuance, and investment between segments.

## Graph Propagation Algorithm

```python
def propagate_scenario(graph, base_scenario):
    factors = initialize_global_country_sector_company_factors(base_scenario)
    for iteration in range(max_iterations):
        for edge in graph.edges:
            source_state = factors[edge.source]
            target_state = factors[edge.target]
            target_state.apply(edge.transmission_function(source_state))
        if converged(factors):
            break
    return factors

def company_statement_from_graph(company, factors):
    industry_driver = industry_driver_model(company.industry, factors)
    entity_statements = entity_graph_simulator(company.entities, industry_driver)
    return consolidate_group(entity_statements)
```

A marketable product should let users inspect the graph path behind a forecast. If a supplier's gross margin fell, the report should show whether the cause was Apple demand, FX, commodity input cost, sanctions, wage inflation, factory utilization, or a combination. This is where agentic UI becomes useful: the agent can trace drivers, ask for missing assumptions, run sensitivity cases, and produce a narrative with audit links.

# Part 9: ML Traps And Validation

## The Main ML Failure Modes

| Failure mode | Example | Mitigation |
| --- | --- | --- |
| Accounting-standard confounding | IFRS R&D capitalization appears as higher profitability. | Standard-aware features and normalization bridge. |
| Presentation leakage | CFO differs because interest paid classification differs. | Capture cash-flow classification policy and restate when needed. |
| Covenant blindness | Model predicts healthy normalized EBITDA but reported covenant EBITDA breaches. | CovenantEngine over reporting-book statements. |
| Industry-mix bias | Bank deposits treated as debt and loans treated as receivables. | Industry-specific state schemas and validators. |
| Jurisdiction omitted variable | Low tax rate interpreted as operating quality. | JurisdictionTaxEngine and transfer-pricing features. |
| Survivorship bias | Training excludes failed firms or delisted FPIs. | Include failure/delist/default outcomes and censored data handling. |
| Restatement drift | Historical line items change after restatements or standard adoption. | Versioned data store and filing-date snapshots. |
| Currency mismatch | Revenue in local currency but debt in USD. | Functional/presentation/transaction currency modeling. |
| Policy-election bias | Inventory method or impairment reversal policy becomes hidden signal. | Accounting policy tags and adjustment bridges. |
| LLM extraction hallucination | Normalizer invents a field value absent from the statement. | Provenance, confidence, source snippets, reconciliation checks, human review. |
| Regime shift | Tax reform, sanctions, new standard, Basel buffer change. | Scenario engine and rule-versioned policy modules. |

## Validation Suite

Validation should be layered. Accounting identity checks prove that statements articulate. Standard fixtures prove that accounting policy rules behave as expected. Consolidation tests prove that intercompany eliminations and NCI work. Covenant tests prove that reported statements trigger the right contractual outcomes. Regulatory tests prove that bank and insurer capital constraints work. Backtests prove historical plausibility, but backtests alone are not enough because they can pass for the wrong accounting reason.

| Test category | Example fixture | Expected result |
| --- | --- | --- |
| Accounting identity | Assets = liabilities + equity after R&D capitalization and amortization. | No imbalance; bridge ties to journal entries. |
| R&D standard fixture | 100 R&D, 60 qualifying IFRS development. | US GAAP expenses 100; IFRS capitalizes 60 and amortizes later. |
| Inventory fixture | Inflationary FIFO/LIFO case. | FIFO and LIFO produce different COGS/inventory/tax but same units. |
| Impairment fixture | Recoverable amount falls then recovers. | IFRS non-goodwill asset reversal allowed within cap; US GAAP no reversal. |
| FX fixture | EUR sub with USD parent. | Assets/liabilities translated at closing rate; CTA captured; P&L translated appropriately. |
| Consolidation fixture | Parent sells inventory to sub with unsold profit. | Intercompany sale and unrealized profit eliminated. |
| Covenant fixture | Leverage ratio above threshold after lease debt inclusion. | Debt acceleration or restricted payment block triggered. |
| Bank capital fixture | Credit losses reduce CET1 and RWA changes. | Dividend/buyback policy constrained by buffer. |
| Insurance fixture | Onerous insurance contract group. | Loss component recognized and CSM not created for loss. |
| Agent audit fixture | Missing tax jurisdiction assumption. | Agent asks for or flags assumption; does not silently proceed. |

## Agentic Workflow Requirements

The director's point about modern product design being more agentic means the application should not just expose a GUI form. It should orchestrate tasks: inspect filings, identify standards and entities, extract statements, build assumptions, ask targeted questions, run simulations, validate outputs, explain breaches, and produce audit trails. The agent should not hide model uncertainty. It should surface it and ask for human confirmation when assumptions are material.

```text
agent workflow:
  1. Identify company, filings, reporting basis, issuer status, and entities.
  2. Extract statements, notes, policy elections, debt covenants, and tax disclosures.
  3. Build entity graph, jurisdiction map, industry modules, and accounting contexts.
  4. Detect missing assumptions and ask targeted questions.
  5. Run base, downside, upside, and custom scenarios.
  6. Validate accounting identities, covenants, capital ratios, and graph propagation.
  7. Produce report with source provenance, assumptions, bridges, diagnostics, and sensitivity.
  8. Keep a reproducible audit trail of prompts, source documents, parameter versions, and overrides.
```

## Marketable Product Requirements

- Traceable provenance from every forecasted line item back to source statements, notes, assumptions, and model rules.
- Accounting standard and version coverage with explicit unsupported-standard warnings.
- Jurisdiction-aware tax, subsidy, sanctions, withholding, FX, and capital-control mechanics.
- Industry-specific model families with state schemas, templates, and validators.
- Consolidation and ownership graph with legal ownership, control, economics, and cash-flow rights.
- Covenant and regulatory capital engines for contract and prudential constraints.
- Scenario and uncertainty framework that handles macro, sector, supplier, customer, FX, rates, commodity, and regulation shocks.
- Human-in-the-loop review for low confidence extraction, missing disclosures, and material assumption choices.
- Model governance: versioned rules, model cards, validation reports, access controls, and review logs.

# Part 10: Codebase Mapping

## Current Shape Of The financial_forecast Codebase

The current implementation is a coherent operating-company financial forecast prototype. It packs a 14-field recurrent state, evolves non-current assets and working capital, computes COGS/OpEx/interest/tax/net income, manages liquidity, issues debt/equity based on simple policies, and exports historical and forecast tables. This is useful for Apple-like operating-company simulations, but it does not yet represent accounting standards, entity graphs, jurisdictions, industry-specific states, covenants, bank capital, insurance liabilities, or agentic audit trails.

| File | Current role | Limitation | Future technical direction |
| --- | --- | --- | --- |
| financial_forecast/types.py | Defines RecurrentState with 14 operating-company fields and HistoricalTrainingData. | One state schema cannot represent banks, insurers, holding companies, industry-specific assets, or standard-specific books. | Introduce domain-aware state schemas: OperatingCompanyState, BankState, InsurerState, HoldingCompanyState, EntityState, ReportingContext. |
| financial_forecast/inference/state_index.py | Hard-coded tensor indices for 14 recurrent fields and 27 diagnostics. | Packed tensor layout makes adding industry-specific state variables risky and global. | Create state adapters or model-family-specific index registries with validation and serialization metadata. |
| financial_forecast/extraction/statement_config.py | Defines three statement types and a narrow field list. | No accounting standard, version, jurisdiction, policy, provenance-rich line item taxonomy, or industry statement templates. | Add StandardAwareStatementConfig with reporting basis, currency, XBRL tag, source evidence, and required disclosure fields. |
| financial_forecast/extraction/statement_normalizer.py | Uses LLM to normalize extracted statement files into configured fields. | LLM output is not enough for auditability; no standard-aware adjustment bridge or confidence workflow. | Add provenance spans, confidence, extraction evidence, standard/policy metadata, and reconciliation checks. |
| financial_forecast/models/base.py | Composes BalanceSheetModel, IncomeStatementModel, CashBudgetModel, tax, debt, and policies. | Assumes one operating-company statement shape and direct forecast step. | Introduce simulator orchestration around entities, accounting contexts, policy engine, regulated statement generator, and normalization bridge. |
| financial_forecast/models/balance_sheet.py | Evolves NCA, inventory, AR, AP, advances, purchases. | No intangible R&D project state, leases, impairments, reserves, rate base, loans, deposits, insurance liabilities. | Split into industry-specific asset/liability modules and accounting-rule-driven measurement. |
| financial_forecast/models/income_statement.py | Computes COGS, OpEx, EBITDA, depreciation, interest, tax, net income. | No standard-specific R&D, lease presentation, impairment reversals, expected credit losses, insurance service result. | Use statement-generation modules that compute presentation under reporting context and bridge to normalized income. |
| financial_forecast/models/cash_budget.py | Computes operating, capex, financing, owner transactions, liquidity check. | No standard-specific cash-flow classification, trapped cash, withholding tax, covenant blocks, capital regulation. | Add CashFlowPresentationEngine, trapped cash/repatriation logic, covenant-constrained owner transactions. |
| financial_forecast/models/tax.py | SimpleTax and TaxWithAnomalies. | No jurisdiction-specific regimes, transfer pricing, deferred taxes, credits, subsidies, withholding, minimum tax. | Replace with JurisdictionTaxEngine operating on legal entity statements and intercompany flows. |
| financial_forecast/models/debt.py | Simple and trend debt policies for ST debt, LT financing mix, average maturity. | No instrument stack, covenants, revolvers, secured debt, AT1, Tier 2, senior debt, TLAC/MREL, refinancing market access. | Create DebtInstrumentStack and CovenantEngine; for banks integrate CapitalRegulationEngine. |
| financial_forecast/inference/pipeline.py | Runs trajectory simulation, plots, and exports forecast_report.json. | Report lacks assumptions, accounting standard, entity graph, jurisdiction, scenario diagnostics, audit trail. | Export full study/report package: source provenance, reporting contexts, normalized bridges, validation results, scenario comparisons. |

## Proposed State Schemas

```python
class ReportingContext(TypedDict):
    standard: str
    standard_version: str
    jurisdiction: str
    book_type: str              # local_statutory, group_reporting, covenant, tax
    functional_currency: str
    presentation_currency: str
    policy_elections: dict

class OperatingCompanyState(TypedDict):
    cash: Tensor
    receivables: Tensor
    inventory: Tensor
    ppe: Tensor
    right_of_use_assets: Tensor
    intangible_assets: Tensor
    capitalized_development: Tensor
    lease_liabilities: Tensor
    accounts_payable: Tensor
    debt_stack: DebtStackState
    deferred_tax_assets: Tensor
    deferred_tax_liabilities: Tensor
    equity: Tensor
    retained_earnings: Tensor

class BankState(TypedDict):
    cash_and_reserves: Tensor
    loan_book: LoanBookState
    securities_book: SecuritiesBookState
    deposits: FundingBookState
    wholesale_funding: FundingBookState
    allowance_for_credit_losses: Tensor
    cet1: Tensor
    at1: Tensor
    tier2: Tensor
    rwa: Tensor
    leverage_exposure: Tensor
    liquidity_buffer: Tensor

class InsurerState(TypedDict):
    investment_portfolio: AssetPortfolioState
    insurance_contract_liabilities: Tensor
    csm: Tensor
    risk_adjustment: Tensor
    loss_component: Tensor
    reinsurance_assets: Tensor
    own_funds: Tensor
    scr: Tensor
    mcr: Tensor

class HoldingCompanyState(TypedDict):
    standalone_cash: Tensor
    standalone_debt: DebtStackState
    investments: dict[str, InvestmentState]
    ownership_matrix: Tensor
    nav: Tensor
    pledged_collateral: Tensor
```

## Target Architecture

```text
FinancialForecastPlatform
  DataIngestion
    -> StatementExtractor
    -> StatementNormalizer
    -> ProvenanceStore
    -> AccountingPolicyDetector
  SimulationCore
    -> EntityGraphSimulator
    -> AccountingPolicyEngine
    -> RegulatedStatementGenerator
    -> ConsolidationEngine
    -> JurisdictionTaxEngine
    -> CovenantEngine
    -> CapitalRegulationEngine
    -> NormalizedMetricLayer
  ModelFamilies
    -> OperatingCompanySimulator
    -> BankSimulator
    -> InsurerSimulator
    -> HoldingCompanySimulator
    -> IndustrySpecificModules
  InferenceAndProduct
    -> ScenarioEngine
    -> MacroSectorGraph
    -> ValidationSuite
    -> AgenticWorkflow
    -> AuditReportExporter
```

## Minimal Evolution Path From Current Code

- First, add metadata and provenance to extraction outputs without changing forecast math: standard, version, currency, statement date, source file, and confidence.
- Second, add a normalized bridge table so the current operating-company model can show reported versus adjusted fields for R&D, inventory, leases, impairments, tax anomalies, and cash-flow classification.
- Third, wrap the current BaseFinancialModel as OperatingCompanySimulator rather than making it the universal simulator.
- Fourth, introduce EntityGraph and ConsolidationEngine outside the TensorFlow hot path so multiple entity-level forecasts can be combined and audited.
- Fifth, add CovenantEngine and scenario validation before adding more ML complexity; this directly addresses the director's concerns about actual statements.
- Sixth, build separate BankSimulator and InsurerSimulator instead of trying to stretch RecurrentState.
- Seventh, extend ForecastPipeline exports to include assumptions, validation results, source provenance, bridge adjustments, covenant results, and scenario diagnostics.

## Concrete Tests To Add Later

| Test file concept | Fixture | Assertion |
| --- | --- | --- |
| test_accounting_policy_engine.py | US GAAP vs IFRS R&D event. | Reported statements differ; normalized bridge reconciles. |
| test_inventory_policy.py | FIFO vs LIFO in inflation. | COGS, inventory, tax, and cash differences match expected mechanics. |
| test_impairment_rules.py | Recoverable value falls then recovers. | IFRS reversal for eligible non-goodwill assets; US GAAP no reversal. |
| test_consolidation_engine.py | US parent, EUR sub, intercompany sale. | FX translation, eliminations, NCI, and CTA are correct. |
| test_covenant_engine.py | Debt acceleration after leverage breach. | Debt reclassifies and liquidity/default flags update. |
| test_jurisdiction_tax_engine.py | Royalty from China sub to Irish entity to US parent. | Withholding, transfer pricing, current/deferred tax, and cash-tax bridge are traceable. |
| test_bank_capital_engine.py | Credit loss shock reduces CET1. | Dividend/buyback blocked when buffer is breached. |
| test_insurer_ifrs17_solvency.py | Onerous contract and market shock. | CSM/loss component and SCR/own funds respond correctly. |
| test_agentic_audit_report.py | Missing standard metadata. | Workflow asks for user confirmation or emits unsupported assumption warning. |

# Worked Technical Examples

This section is intentionally numerical and mechanical. The purpose is to make the abstract architecture feel concrete enough for a two-hour technical discussion. Each example shows the reported-statement view, the normalized-economic view, and the ML or product failure mode.

## Example 1: R&D Capitalization Changes Covenant Headroom

Assume a borrower spends 100 on R&D. Under the US GAAP reporting book, all 100 is expensed. Under an IFRS reporting book, 40 is research expense and 60 qualifies as development cost that is capitalized. Suppose EBITDA before R&D is 300, cash interest is 60, net debt is 900, and the credit agreement defines leverage as net debt divided by reported EBITDA before unusual addbacks.

| Metric | US GAAP reported book | IFRS reported book | Interpretation |
| --- | --- | --- | --- |
| R&D expense | 100 | 40 | The business spent the same cash, but reported expense differs. |
| Development asset | 0 | 60 | IFRS creates an asset that US GAAP does not. |
| EBITDA after R&D | 200 | 260 | Leverage denominator differs by 60. |
| Net debt / EBITDA | 900 / 200 = 4.5x | 900 / 260 = 3.46x | A 4.0x covenant is breached under US GAAP but not under IFRS. |
| Economic cash spent | 100 | 100 | Normalized cash economics are identical. |

The director's point is visible here. If the actual debt agreement binds to US GAAP, the company can face restricted payments, default, waiver fees, pricing step-ups, or acceleration even if a normalized economic model says the IFRS and US GAAP firms are economically identical. Therefore the system must compute both covenant-book EBITDA and normalized EBITDA.

## Example 2: Debt Acceleration Creates A Second-Order Balance Sheet Shock

Suppose a company has 500 of long-term debt, 80 of current liabilities, 160 of current assets, and a covenant breach that allows lenders to demand repayment within one year. If acceleration is probable or the debt is callable, classification can move the debt into current liabilities. That reclassification can itself break a current-ratio covenant or trigger liquidity concern disclosures.

| State | Current assets | Current liabilities | Current ratio | Technical note |
| --- | --- | --- | --- | --- |
| Before breach | 160 | 80 | 2.0x | Comfortably above a 1.5x current-ratio covenant. |
| After acceleration classification | 160 | 580 | 0.28x | Long-term debt becomes current, creating a liquidity shock. |
| After refinancing waiver | 160 | 95 plus fees | 1.68x | Waiver may cure classification but adds fees and higher interest. |

This is why a covenant engine cannot be an afterthought. It feeds back into the financial statements: debt classification, interest cost, refinancing cash flow, fees, going-concern risk, dividend policy, and equity issuance can all change after a breach.

## Example 3: US Parent With EUR Subsidiary

Assume a US parent owns 80 percent of a European subsidiary. The subsidiary's functional currency is EUR. It has revenue of EUR 1,000, net income of EUR 100, assets of EUR 1,500, liabilities of EUR 900, and an intercompany payable of EUR 100 to the US parent. Average EUR/USD is 1.10 and closing EUR/USD is 1.20. The group reports in USD.

| Item | Local EUR book | Translation rate | USD group input | Consolidation note |
| --- | --- | --- | --- | --- |
| Revenue | 1,000 | Average 1.10 | 1,100 | Income items often use average rate. |
| Net income | 100 | Average 1.10 | 110 | Attribution: 88 to parent, 22 to NCI before other adjustments. |
| Assets | 1,500 | Closing 1.20 | 1,800 | Balance sheet translated at closing rate. |
| Liabilities | 900 | Closing 1.20 | 1,080 | Includes intercompany payable before elimination. |
| Intercompany payable | 100 | Closing 1.20 | 120 | Eliminate against parent's receivable. |
| Net assets | 600 | Mixed | 720 before CTA | Translation difference goes to CTA/OCI rather than sales. |

The simulator should not collapse this into a single USD sales growth assumption. It needs an entity-level EUR statement, translation, ownership attribution, NCI, intercompany elimination, and then any US GAAP reporting adjustments required by the parent. Local statutory profits may also determine whether the European subsidiary can legally dividend cash to the US parent.

## Example 4: Intercompany Royalty And Tax Leakage

Assume a China operating entity pays a 60 royalty to an Irish IP entity. The China entity has book EBIT of 200 before royalty. The Irish entity has minimal operating expense. China applies 25 percent corporate tax and 10 percent withholding on royalties. Ireland applies 12.5 percent tax in this simplified example. Ignore treaty relief and minimum tax for the moment.

| Entity/cash flow | Amount | Tax/cash effect | Modeling issue |
| --- | --- | --- | --- |
| China EBIT before royalty | 200 | Book operating profit before transfer pricing. | Operating performance sits in China. |
| Royalty expense | -60 | China taxable income falls to 140. | Transfer pricing changes entity profit. |
| China corporate tax | 35 | 140 x 25%. | Lower current tax than without royalty. |
| Withholding tax | 6 | 60 x 10% cash leakage. | Consolidated tax model must capture cross-border payment tax. |
| Irish royalty income | 60 | Irish tax 7.5 at 12.5%. | Profit appears in low-tax jurisdiction. |
| Consolidation | Royalty revenue/expense eliminated | Withholding and tax remain real cash effects. | Intercompany P&L elimination does not erase tax cash flows. |

A naive consolidated tax rate may hide this. A product-grade simulator should compute legal-entity tax first, then consolidate. It should also support transfer-pricing sensitivity, withholding-tax treaties, uncertain tax positions, deferred taxes, and minimum tax scenarios.

## Example 5: VIE Structure Separates Control From Ownership

In a VIE-style structure, the listed offshore parent may consolidate an operating company because it has contractual power and exposure to returns, while not directly owning the restricted domestic equity. The accounting consolidation conclusion is not the same as legal asset ownership. A credit or valuation model should therefore store separate variables: legal ownership, voting rights, contractual control, economic exposure, cash remittance ability, and legal enforceability scenario.

| Dimension | Direct subsidiary | VIE-style controlled entity | Why the distinction matters |
| --- | --- | --- | --- |
| Legal equity ownership | Parent owns shares. | Parent may not own restricted operating shares. | Asset claim and enforcement differ. |
| Accounting consolidation | Usually consolidated if controlled. | May be consolidated if VIE control criteria are met. | Consolidated assets are not necessarily legally owned. |
| Cash remittance | Dividends subject to local law/tax. | Service fees, royalties, or contracts may transfer economics. | Cash trapping and regulatory risk differ. |
| Valuation risk | Corporate law and minority rights. | Contract enforceability and regulation. | Scenario discount may be required. |

## Example 6: SoftBank-Style NAV With Listed And Private Holdings

Assume a holding company has standalone cash of 20, standalone debt of 90, a 30 percent stake in a listed company worth 200, a 50 percent stake in a private company estimated at 100, and an 80 percent controlled subsidiary with enterprise value 150 and net debt 40. Ignore tax leakage first.

```text
listed stake value       = 30% * 200 = 60
private stake value      = 50% * 100 = 50
controlled equity value  = 80% * (150 - 40) = 88
gross investment value   = 60 + 50 + 88 = 198
NAV before discounts     = cash 20 + investments 198 - standalone debt 90 = 128

Then adjust for:
  tax leakage on disposals
  liquidity discount on private holdings
  holding-company overhead
  pledged collateral and margin-loan risk
  cross-holding double-counting
  FX translation
  correlation across tech/venture/rate factors
```

The accounting book may show the controlled subsidiary consolidated line by line, the associate under equity method, and financial investments at fair value. The valuation book may use market value or scenario fair value. The product should show a reconciliation between accounting carrying value, market/fair value, ownership-adjusted value, and parent NAV.

## Example 7: Utility Rate Base Turns Capex Into Future Earnings

For a regulated utility, capex can create future allowed revenue if it enters rate base. Suppose rate base is 1,000, the regulator allows 50 percent debt, 50 percent equity, cost of debt 5 percent, allowed ROE 9 percent, and operating cost recovery is separate. The allowed return component is 0.5 x 1,000 x 5% + 0.5 x 1,000 x 9% = 25 + 45 = 70. If approved capex raises rate base to 1,200, allowed return rises to 84, assuming the same capital structure and allowed returns.

A generic model may treat capex as a cash drain and depreciation as the only future effect. A utility model must track regulatory lag, prudence review, disallowed costs, fuel pass-through, deferred regulatory assets, storm costs, and allowed capital structure. This is why industry modules are not optional.

## Example 8: Mining Reserve Revision And Asset Retirement Obligation

A mining company has a mine carrying amount of 500, proven and probable reserves of 50 million tons, and expected asset retirement obligation of 80 present value. If reserves fall to 35 million tons after a geological revision, depletion per ton increases and the recoverable amount may fall below carrying amount. If environmental law changes, the ARO may increase, raising the liability and often the related asset retirement cost.

| Driver | Statement impact | Simulator state needed |
| --- | --- | --- |
| Reserve revision down | Higher depletion per unit, possible impairment. | Reserve volume, grade, extraction plan, commodity price curve. |
| Commodity price fall | Lower recoverable amount and cash flow. | Price scenario, hedge book, cost curve. |
| ARO estimate up | Higher liability and asset retirement cost. | Legal obligation, discount rate, expected remediation cost. |
| Mine closure acceleration | Shorter useful life and earlier cash outflow. | Production schedule and closure timing. |

## Example 9: Bank Capital Shock And Corporate Policy

Assume a bank has CET1 of 100, AT1 of 20, Tier 2 of 30, and RWA of 1,000. CET1 ratio is 10 percent, Tier 1 ratio is 12 percent, and total capital ratio is 15 percent. A credit shock creates 25 of after-tax losses and raises RWA to 1,080 because exposures migrate to higher risk weights. CET1 falls to 75 and CET1 ratio becomes 6.94 percent. If management target is 10.5 percent including buffers, dividends and buybacks should be blocked and the model should consider equity issuance, asset reduction, RWA optimization, credit risk transfer, or retained earnings rebuild.

Issuing Tier 2 debt may improve total capital but does not repair CET1. Issuing AT1 may improve Tier 1 but still does not repair CET1. Senior debt may help liquidity or resolution capacity but does not replace common equity capital. This is the precise reason the debt policy module needs instrument-specific treatment.

## Example 10: IFRS 17 Contractual Service Margin

Suppose an insurer writes a profitable group of contracts with expected premiums of 1,000, expected claims and expenses of 850, risk adjustment of 40, and discounting effects ignored for simplicity. The expected unearned profit is 110, which becomes CSM rather than day-one profit. If the coverage period is five years and services are provided evenly, roughly 22 of CSM is released each year, subject to experience adjustments and assumption changes. If the expected claims rise so fulfilment cash flows exceed premiums, the group is onerous and a loss is recognized rather than a positive CSM.

This is fundamentally different from the current operating-company model. Insurance profit emergence depends on service pattern, actuarial assumptions, risk adjustment, discount rates, and CSM release. Solvency capital may block dividends even if IFRS earnings are positive.

## Example 11: Apple Supplier Shock Through The Macro-Sector Graph

Assume a supplier gets 55 percent of revenue from Apple. Apple unit demand falls 10 percent. The supplier's Apple-related volume falls 12 percent because the component has above-average exposure to the weaker product line. Total revenue falls 6.6 percent before offsetting customers. Fixed manufacturing cost causes gross margin to fall from 24 percent to 20 percent. Inventory days rise from 50 to 75 because production was committed before orders were cut. The supplier's leverage ratio moves from 3.2x to 4.1x, tripping a 4.0x springing covenant.

A flat company-level sales model would miss the path. A graph model shows the path: Apple demand shock -> supplier volume shock -> factory utilization -> gross margin -> inventory build -> cash conversion -> covenant breach -> debt classification/refinancing. That is the level of explanation a senior client would expect.

## Example 12: Agentic Audit Trace

A useful agent should produce an audit trace, not just a forecast. For example: source statement says the company reports under IFRS; note says development costs are capitalized when criteria are met; extraction confidence for capitalized development is 0.71 because the PDF table is ambiguous; agent asks the user whether to treat the amount as development intangible or software asset; user confirms; model generates IFRS reported statements; normalized bridge expenses capitalized development for comparability; covenant engine uses reported EBITDA because the debt agreement references IFRS consolidated EBITDA.

```python
audit_trace = [
  {"step": "extract", "field": "capitalized_development", "source": "annual report note 12", "confidence": 0.71},
  {"step": "human_review", "question": "classify as development intangible or software?", "answer": "development intangible"},
  {"step": "reported_simulation", "standard": "IFRS", "rule": "IAS 38 development capitalization"},
  {"step": "normalization", "adjustment": "expense capitalized development for peer comparability"},
  {"step": "covenant", "metric": "reported EBITDA", "source": "credit agreement definition"},
]
```

# Background Knowledge Glossary For Director Concepts

This glossary is not a dictionary for memorization. It is a compact background layer for concepts the director mentioned or implied. The goal is to be able to connect each term to statement simulation, ML pitfalls, and codebase architecture.

### Actual reported financial statements

The legally or contractually relevant statements produced under a specific reporting basis. They can differ from economic reality but still determine debt covenants, regulatory capital, tax filings, dividend capacity, and investor reporting. In the system design, these are produced by the RegulatedStatementGenerator, not inferred from normalized metrics.

### Normalized economic metrics

Adjusted metrics intended to compare economics across standards, industries, and policy choices. They are useful for ML features and valuation, but they should be built as explicit bridges from reported statements. A normalized metric without a bridge is hard to audit.

### Covenant book

The statement basis and definitions specified by a debt agreement. It may use current GAAP, frozen GAAP, lender-adjusted EBITDA, restricted subsidiaries, permitted addbacks, lease treatment, and basket definitions. It is neither necessarily statutory GAAP nor pure normalized economics.

### Local statutory books

Legal-entity books prepared for a local jurisdiction. These can drive local tax, distributable reserves, regulatory filings, and legal dividend capacity. A subsidiary can be profitable on group reporting books but constrained on local statutory books.

### Group reporting books

Books converted to the parent group's reporting basis for consolidation. A European subsidiary may have local statutory accounts and a group reporting package adjusted to US GAAP or IFRS.

### Functional currency

The currency of the primary economic environment of an entity. It is not always the parent presentation currency. Functional currency determines remeasurement and translation behavior.

### Presentation currency

The currency in which group financial statements are presented. A USD parent can present a EUR subsidiary in USD, creating translation effects that do not equal local operating performance.

### CTA / cumulative translation adjustment

A component of equity/OCI used to accumulate translation differences for foreign operations. A model should not treat all FX movement as revenue or operating income.

### Non-controlling interest

The portion of equity and earnings in a controlled subsidiary attributable to outside owners. Full consolidation reports 100 percent of the subsidiary's assets and revenues, then allocates equity and income between parent and NCI.

### Goodwill

Residual acquisition value after identifiable net assets and NCI are measured. Goodwill is not a generic plug for all intangible value; it has impairment rules and interacts with cash-generating units or reporting units.

### Equity method

Accounting for significant influence rather than control. The investor recognizes its share of profit or loss and adjusts carrying value, but does not consolidate line-by-line. This matters for SoftBank-style holdings.

### Fair value through P&L or OCI

A measurement and presentation choice/classification for financial investments. Fair value changes may run through earnings or OCI. ML features based on net income can be distorted by classification and market volatility.

### VIE

A structure where consolidation can arise from power and exposure to variable returns even when legal equity ownership is limited or absent. The simulator must separate accounting control from legal ownership and cash remittance risk.

### Transfer pricing

Pricing of transactions among related entities. It moves profit across jurisdictions and affects tax, cash, and legal-entity statements. In consolidation, intercompany revenue/expense may eliminate, but tax and cash leakage can remain.

### Deferred tax

Tax effect of temporary differences between book and tax bases. A one-period effective tax rate cannot capture deferred tax reversal schedules, valuation allowances, or law changes.

### Withholding tax

Tax collected at source on cross-border payments such as dividends, royalties, or interest. It is often a cash leakage item in multinational structures and should be modeled by flow and treaty.

### Pillar Two / minimum tax

A global minimum-tax style regime can reduce the persistence of low effective tax rates. The interview answer should be cautious: do not extrapolate historical low tax rates without scenario rules.

### Sanctions

Restrictions that can affect customers, suppliers, banks, technology, shipping, insurance, FX, and asset sales. They are graph constraints and scenario rules, not just a scalar revenue penalty.

### Rate base

For regulated utilities, the asset base on which a regulator permits a return. Approved capex can increase future allowed revenue, which is the opposite of a simple capex-as-cash-drain view.

### Asset retirement obligation

A legal obligation to retire or remediate an asset, common in mining, oil/gas, utilities, and environmental contexts. It requires present value measurement and periodic accretion or remeasurement.

### Reserves

For mining and oil/gas, reserves drive production, depletion, impairment, valuation, and financing. Reserve revisions are operating facts that change accounting estimates.

### Internally generated brand

Often not recognized as an asset even when economically valuable. Acquired brands may be recognized, creating comparability issues between organic and acquisitive companies.

### Acquired IPR&D

In-process research and development acquired in a transaction can have different accounting from internally generated R&D. Pharma models need project-level pipeline accounting.

### CET1

Common Equity Tier 1 capital after regulatory deductions. It is the most important bank capital layer and cannot be replaced by Tier 2 or senior debt when the CET1 ratio is weak.

### AT1

Additional Tier 1 instruments are subordinated perpetual capital instruments with loss-absorbing features and discretionary coupons. They can support Tier 1 capital but are not common equity.

### Tier 2

Subordinated instruments that can count toward total capital subject to eligibility and maturity rules. Tier 2 can help total capital but not CET1.

### Senior debt

Ordinary senior funding. It can matter for liquidity and resolution capacity, but it is not the same as regulatory capital. A bank capital model must distinguish it from AT1 and Tier 2.

### RWA

Risk-weighted assets. A bank's denominator can rise when credit quality worsens, even if nominal assets do not grow. Capital ratios therefore depend on asset mix and risk migration.

### CSM

Contractual service margin under IFRS 17, representing unearned profit in insurance contracts. It is released over service, so insurance profit timing differs from simple premium minus claims.

### SCR and MCR

Solvency II capital requirements for insurers. SCR is risk-based capital needed for adverse events; MCR is a lower floor with severe supervisory implications. Dividends depend on solvency headroom.

### Customer concentration

Dependence on one large customer, such as an Apple supplier's exposure to Apple. Forecasting must propagate customer demand shocks into supplier revenue, margin, working capital, and covenants.

### Sector spillover

A shock in one sector affects another through supply chains, financing, commodity inputs, labor, technology restrictions, or demand. This motivates a macro-sector-company graph.

### Agentic workflow

A workflow where an agent plans data ingestion, identifies standards, asks for missing assumptions, runs scenarios, validates statements, explains failures, and produces an audit trail. It should increase control and transparency, not hide decisions.

# Appendices

## Appendix A: Accounting Policy Recipe Catalog

| Area | Recipe |
| --- | --- |
| R&D | Classify project costs into research, development, software, acquired IPR&D, and maintenance. Apply standard-specific capitalization, amortization, impairment, and disclosure rules. |
| Inventory | Track physical units, cost layers, cost formulas, NRV, write-downs, reversals, inflation, purchase commitments, and inventory obsolescence. |
| Impairment | Track carrying amount, recoverable amount/fair value, CGU or asset group, prior impairment, goodwill flag, reversal eligibility, and disclosure assumptions. |
| Leases | Track lease payments, discount rate, term, options, ROU asset, lease liability, operating/finance classification, covenant debt inclusion, and cash-flow presentation. |
| Debt | Track principal, coupon, floating-rate reset, maturity, security, seniority, covenant package, callability, amortization, PIK, revolver draw, and refinancing availability. |
| Tax | Track book-tax differences, NOLs, credits, withholding, transfer pricing, uncertain tax positions, minimum tax, deferred tax reversal schedules, and cash-tax timing. |
| Consolidation | Track control, ownership, economics, intercompany flows, NCI, equity method, fair value marks, goodwill, FX translation, and deconsolidation triggers. |
| Bank capital | Track CET1, AT1, Tier 2, deductions, RWA, leverage exposure, LCR, NSFR, buffers, stress capital, and distribution constraints. |
| Insurance | Track fulfilment cash flows, risk adjustment, CSM, loss component, reinsurance, own funds, SCR, MCR, duration matching, and asset risk charges. |

## Appendix B: Formula Notes

```text
Accounting identity:
  assets = liabilities + equity

Clean reported-to-normalized bridge:
  normalized_metric = reported_metric + sum(adjustments)
  every adjustment must have: source, rule, amount, sign, confidence, reviewer

Leverage covenant:
  covenant_leverage = covenant_net_debt / covenant_EBITDA
  breach if covenant_leverage > threshold

Bank capital:
  CET1_ratio = CET1 / RWA
  Tier1_ratio = (CET1 + AT1) / RWA
  Total_capital_ratio = (CET1 + AT1 + Tier2) / RWA
  Leverage_ratio = Tier1_capital / leverage_exposure

Insurance solvency:
  solvency_ratio = eligible_own_funds / SCR
  hard floor logic depends on MCR and local supervisor regime

Holding company NAV:
  NAV = cash + investments_fair_value - debt - tax_leakage - overhead_value - discounts
```

## Appendix C: Source Notes

The source URLs below are included so the notes can be refreshed later. The PDF uses only compact summaries and does not reproduce copyrighted standards text. Primary-source anchors checked include IFRS Foundation pages for IAS 38, IAS 2, IAS 36, IFRS 10, IAS 21, IFRS 17; SEC guidance for foreign private issuers; the Basel Framework PDF; and EIOPA's Solvency II overview.

| Source | URL |
| --- | --- |
| IAS 38 Intangible Assets | https://www.ifrs.org/issued-standards/list-of-standards/ias-38-intangible-assets/ |
| IAS 2 Inventories | https://www.ifrs.org/issued-standards/list-of-standards/ias-2-inventories/ |
| IAS 36 Impairment of Assets | https://www.ifrs.org/issued-standards/list-of-standards/ias-36-impairment-of-assets/ |
| IFRS 10 Consolidated Financial Statements | https://www.ifrs.org/issued-standards/list-of-standards/ifrs-10-consolidated-financial-statements/ |
| IAS 21 Effects of Changes in Foreign Exchange Rates | https://www.ifrs.org/issued-standards/list-of-standards/ias-21-the-effects-of-changes-in-foreign-exchange-rates/ |
| IFRS 17 Insurance Contracts | https://www.ifrs.org/issued-standards/list-of-standards/ifrs-17-insurance-contracts/ |
| SEC Foreign Private Issuers Overview | https://www.sec.gov/divisions/corpfin/internatl/foreign-private-issuers-overview.shtml |
| Basel Framework | https://www.bis.org/baselframework/BaselFramework.pdf |
| EIOPA Solvency II Overview | https://www.eiopa.europa.eu/browse/regulation-and-policy/solvency-ii_en |

## Appendix D: Final Mental Model

The concise technical answer to the whole interview is: build a multi-book, multi-entity, multi-standard simulation platform. Simulate legal entities first. Generate regulated financial statements under the correct accounting standard, version, jurisdiction, currency, and policy elections. Consolidate through an ownership/control graph. Compute covenants and regulatory capital from reported or covenant-defined books. Then build a normalized economic bridge for ML comparability. Validate every layer with accounting identities, standard-specific fixtures, consolidation tests, covenant tests, regulatory capital tests, and provenance-backed audit reports.
