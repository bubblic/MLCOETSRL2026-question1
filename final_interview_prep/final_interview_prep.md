# Final Interview Prep

## 1. What The Interviewer Is Really Testing

The four questions are all variants of one deeper question:

How would you take your current forecasting prototype and turn it into a real cross-country, cross-standard, cross-industry financial ML product without fooling yourself?

That means the interview is not mainly about memorizing accounting rules. It is about whether you can:

- identify where accounting differences change the data generating process
- explain how those differences would affect your conclusions
- redesign the ML system so it learns economics rather than filing artifacts
- be honest about what your current project does well and where it would break

The strongest overall framing is:

> My report is strongest for large operating companies with relatively standard three-statement dynamics. As soon as we move across accounting standards, jurisdictions, and industries, the main engineering problem becomes semantic normalization. If we do not normalize first, the model will learn reporting rules instead of business reality.

That is the sentence to come back to again and again.

## 2. The Main Story You Want To Tell

Your current model is a structural, no-plug, operating-company forecaster.

It works by rolling forward a recurrent state with items like:

- `nca`
- `accounts_receivable`
- `inventory`
- `cash`
- `investment_in_market_securities`
- `accounts_payable`
- `effective_st_debt`
- `current_lt_debt`
- `non_current_liabilities`
- `equity`
- `net_income`
- `dividends`

That comes directly from `financial_forecast/types.py`.

This is powerful because it makes the accounting identity structural rather than cosmetic.

But it also means the current model does **not** explicitly represent many items that become crucial across standards, countries, and industries, such as:

- goodwill
- acquired intangibles
- internally generated intangible assets
- deferred tax assets and liabilities
- right-of-use assets and lease liabilities
- expected credit loss allowances
- deposits and loan books
- insurance contract liabilities
- regulatory capital buckets

So your honest, strong answer is not:

> My current system already solves all of that.

It is:

> My current system gives the right structural starting point, but a marketable multi-country application needs a semantic layer above the simulator and sector-specific extensions below it.

## 3. Question 1

### Companies in different countries use different accounting standards. How do the conclusions in your report change as the accounting standard changes?

### Strong opening answer

> The conclusions in my report change whenever the accounting standard changes the recognition, timing, measurement, or classification of economically similar events. So the same company can look more profitable, more asset-light, or more levered depending on the standard, even if the underlying business is unchanged. In an ML application, that means raw line items are not portable labels. The model needs accounting-standard-aware normalization before training or inference.

### Concrete examples you can use

#### Example 1: development costs under IFRS vs US GAAP

Under IAS 38:

- research costs are expensed when incurred
- development costs are capitalized if specific criteria are met

Under the SEC's comparison of IFRS and US GAAP:

- research and development costs are generally expensed as incurred under US GAAP
- under IFRS, qualifying development costs can be capitalized

Why this matters:

- an IFRS software or pharma company can show higher assets and higher current-period earnings than a similar US GAAP company
- later periods then pick up amortization
- so ROA, EBITDA-style metrics, asset growth, and margin comparisons can all be distorted if you treat the statements as directly comparable

How to connect this to your project:

> My current model would treat that as a real asset and earnings difference unless I first map both companies onto a common economic representation. Otherwise, the model would learn accounting treatment of R&D rather than business productivity.

#### Example 2: inventory costing and LIFO

IAS 2 permits FIFO or weighted average cost for interchangeable inventories.

The SEC notes:

- IFRS does not allow LIFO
- US GAAP permits LIFO

Why this matters:

- in inflationary periods, LIFO usually gives higher COGS and lower ending inventory
- that changes gross margin, current ratio, inventory turnover, taxable income, and sometimes cash tax behavior
- the SEC also notes the US tax conformity issue around LIFO, so this is not just presentation noise

ML trap:

> If I train a model on mixed IFRS and US GAAP manufacturers using raw gross margin and inventory features, the model may learn the inventory accounting method instead of demand or operational efficiency.

#### Example 3: impairment reversals

IAS 36 allows reversals of impairment losses for assets other than goodwill, subject to limits.

The SEC comparison notes:

- IFRS models allow reversals of impairment losses in some cases
- US GAAP generally precludes reversals

IAS 36 is stricter for goodwill:

- goodwill impairment is not reversed

Why this matters:

- IFRS can show recovery of previously impaired non-goodwill assets
- US GAAP will often remain on the lower carrying amount
- that changes asset base, earnings path, and volatility

ML trap:

> A model trained on post-impairment recoveries under IFRS could misread those earnings rebounds as operating improvement rather than accounting reversal mechanics.

#### Example 4: statement of cash flows classification

Historically, IAS 7 allowed more classification flexibility for items like interest and dividends than US practice.

Also important:

- IFRS 18 was issued in April 2024
- it becomes effective for annual periods beginning on or after January 1, 2027
- it changes IAS 7 by introducing new requirements for classifying interest and dividend cash flows

Why this matters:

- a feature like CFO margin is not stable if classification rules move across standards or even across IFRS versions
- so the product must store the reporting standard **and** the standard version/date

### Main traps and pitfalls

- training on raw reported features from mixed standards
- assuming the same label means the same economics
- using ratios without standard metadata
- treating all differences as disclosure differences when some are measurement differences
- ignoring effective dates and amendments, such as IFRS 18 from January 1, 2027

### Best final line for Q1

> So the conclusion is not that the business changed; it is that the accounting mapping changed. A robust ML application has to normalize across recognition, measurement, and classification before it compares companies.

## 4. Question 2

### Different countries also have specific rules and standards. JP Morgan works with clients in many countries. How can we handle that problem in your application?

### Strong opening answer

> I would treat country and regulator as first-class metadata, not just background information. The application needs a jurisdiction layer that records the filing regulator, the accounting basis actually used, the taxonomy version, the language, the currency, and any local carve-outs. Without that layer, a global model will confuse country-specific reporting rules with company fundamentals.

### The key idea

Country is **not** the same as accounting standard.

The real keys are:

- issuer domicile
- listing venue
- regulator
- accounting basis actually used
- taxonomy version
- local endorsement or carve-outs
- filing format

### Concrete examples you can use

#### Example 1: US market rules for foreign private issuers

The SEC says:

- domestic issuers use US GAAP
- foreign private issuers may use IFRS as issued by the IASB
- or home-country GAAP with reconciliation to US GAAP

Why this matters:

- a company listed in the US is not necessarily on US GAAP
- a foreign company may appear in the same market data universe but use a different accounting basis

ML trap:

> If I group companies by exchange alone, I may accidentally mix domestic US GAAP issuers with IFRS filers in the same feature space.

#### Example 2: filing format is country and regulator specific

The SEC requires Inline XBRL for:

- domestic operating company filings like 10-K and 10-Q
- foreign private issuer filings like 20-F and 40-F

ESMA's ESEF requires:

- annual financial reports in XHTML
- IFRS consolidated financial statements marked up in Inline XBRL
- detailed tagging for primary statements
- notes block tagging since January 1, 2022

Why this matters:

- even when both are machine-readable, the tagging regimes and validation workflows differ
- extraction pipelines need regulator-specific parsers and validation rules

#### Example 3: one country can allow multiple frameworks

Japan's IFRS profile says domestic public companies may use one of four frameworks:

- IFRS
- Japanese GAAP
- Japan's Modified International Standards
- US GAAP

Why this matters:

- country alone does not identify the accounting basis
- the system must detect the actual framework from the filing metadata or auditor note

#### Example 4: "IFRS-like" is not the same as IFRS

India's IFRS profile says:

- Ind AS is based on and substantially converged with IFRS
- but India has not adopted IFRS for domestic companies
- Ind AS contains carve-outs and carve-ins

China's profile says:

- domestic listed companies use ASBE
- IFRS is not permitted for domestic companies in mainland China

Canada's profile says:

- most publicly accountable enterprises use IFRS
- but certain SEC issuers may use US GAAP

Why this matters:

- "country = IFRS" is often wrong
- the real world is endorsement, convergence, and carve-out driven

### Architecture answer for the product

I would build five layers:

1. `Ingestion layer`
   - regulator-specific parsers for SEC, ESEF, and local filing systems

2. `Jurisdiction registry`
   - country, regulator, filing form, accounting basis, taxonomy version, currency, scale, language

3. `Canonical financial ontology`
   - map raw tags and statement labels into normalized economic concepts

4. `Rule engine`
   - jurisdiction- and standard-specific adjustments before features are computed

5. `Model layer`
   - train either jurisdiction-aware models or a shared model with jurisdiction metadata and guardrails

### Main traps and pitfalls

- assuming domicile identifies accounting basis
- ignoring endorsed vs fully issued IFRS
- failing to version taxonomies and effective dates
- treating translation differences as semantic differences
- mixing units, currencies, and scale factors
- losing provenance between raw filing fact and normalized feature

### Best final line for Q2

> The global solution is not one giant parser and one giant model. It is a jurisdiction-aware ingestion stack feeding a common semantic layer, with every transformation traceable back to the original filing.

## 5. Question 3

### Different industries have different balance sheets and line items. Tech firms care a lot about intangibles. How do we handle accounting for different companies?

### Strong opening answer

> I would not force every company into one balance-sheet template. Instead, I would separate the system into a common core ontology plus industry-specific modules. My current model is strongest for non-financial operating companies because it explicitly models working capital, fixed assets, liquidity, debt, and payouts. For tech, pharma, banks, or insurers, I would need additional state variables and sector-specific accounting logic.

### Use your own model honestly

Your current recurrent state is great for:

- industrials
- consumer companies
- many mature tech firms with standard operating cash cycles

It is weaker for businesses where the key economics live in items your state does not explicitly model, especially:

- acquired and internally generated intangibles
- regulatory assets and liabilities
- loan-loss allowances
- deposits
- insurance liabilities

That is not a flaw to hide. It is exactly the right thing to say.

### Concrete examples you can use

#### Example 1: tech and internally generated intangibles

IAS 38 says:

- internally generated goodwill is not recognized as an asset
- research is expensed
- some development can be capitalized if criteria are met

Why this matters:

- many valuable software, data, brand, or ecosystem assets do not appear symmetrically on the balance sheet
- acquired intangibles may appear, internally generated value may not
- book value becomes a poor proxy for economic capital in intangible-heavy industries

ML trap:

> If I use book assets or equity naively across tech and traditional manufacturing, the model may systematically understate the asset base of intangible-heavy firms and learn misleading capital-efficiency patterns.

#### Example 2: pharma and biotech

Pharma companies depend on:

- R&D pipelines
- patents
- acquired IP
- milestone payments
- impairments

That means:

- accounting outcomes can change materially with standard choice
- business outcomes can shift abruptly with approvals, expiries, or litigation

This connects directly to your Pfizer result:

> Pfizer was already a regime-shift stress test in my project. For pharma, I would go further and make the architecture explicitly IP- and event-aware, rather than relying mainly on smooth historical policies.

#### Example 3: banks

Banks are fundamentally different from operating companies.

Under IFRS 9:

- financial assets are classified by business model and contractual cash flow characteristics
- impairment uses expected credit losses

Why this matters:

- for banks, the core state is not inventory, AP, and standard working capital
- it is loan books, deposits, securities, expected credit loss reserves, funding mix, and regulatory capital

So you can say:

> I would not reuse my current working-capital-driven simulator for banks. I would build a banking-specific simulator with loan, deposit, and credit-loss state variables.

#### Example 4: insurers

Under IFRS 17, effective January 1, 2023:

- insurance cash flows are measured on a current basis
- profit is recognized over service periods
- insurance service result is presented separately from insurance finance income or expense

That is a completely different economic structure from a normal operating company.

So:

> My current three-statement operating-company framework is not the right native model for insurers. I would treat insurance as a separate product line with a distinct state representation and accounting engine.

### Product design answer

I would use:

- a shared semantic core for concepts like liquidity, leverage, profitability, and capital return
- industry-specific ontologies for raw statement mapping
- industry-specific simulators or submodels
- an industry gate that routes each issuer to the right model family

Possible model families:

- operating corporates
- asset-light tech and SaaS
- pharma and biotech
- banks
- insurers
- real estate and REITs
- extractives and capital-intensive sectors

### Main traps and pitfalls

- one-size-fits-all schemas
- naive cross-industry ratio comparisons
- ignoring off-balance-sheet or weakly recognized intangible economics
- treating banks and insurers as if they were ordinary working-capital businesses
- training pooled models that mainly learn industry labels

### Best final line for Q3

> Different industries do not just have different feature values. They often have different accounting state spaces. So the right design is a common semantic layer with sector-specific accounting models beneath it.

## 6. Question 4

### If you were building this to sell to customers, what is missing for this to be marketable?

### Strong opening answer

> The current project is a strong research prototype, but not yet a marketable product. To become sellable, it needs a production-grade data normalization layer, governance, auditability, regulator-aware ingestion, sector coverage, monitoring, and customer-facing controls. In other words, the missing work is less about one more model and more about making the whole system trustworthy, explainable, and operable.

### Missing requirements and how to address them

#### 1. Standard and jurisdiction normalization

Missing:

- no full global semantic normalization layer

Needed:

- canonical chart of accounts
- accounting-basis metadata
- jurisdiction-aware transformations
- versioning by standard and taxonomy date

#### 2. Provenance and audit trail

Missing:

- not every normalized number is traceable to an original tagged filing fact or filing excerpt

Needed:

- fact-level lineage
- source document links
- transformation logs
- reproducible normalization outputs

Why customers care:

- finance users need to audit where a forecast input came from

#### 3. Human-in-the-loop review

Missing:

- no robust review workflow for low-confidence mappings, anomalies, or extraction disagreements

Needed:

- reviewer queue
- confidence scores
- exception handling
- approval and override logs

#### 4. Sector coverage

Missing:

- current model is mostly an operating-company simulator

Needed:

- separate model families for banks, insurers, and other sectors
- routing logic
- sector-specific validation rules

#### 5. Data quality and accounting controls

Missing:

- the repo has meaningful tests, but a customer product needs much broader production controls

Needed:

- cross-statement identity checks
- taxonomy validation
- unit and currency validation
- missing-tag diagnostics
- temporal consistency checks
- restatement handling

#### 6. Explainability and uncertainty

Missing:

- uncertainty exists in Bayesian OpEx, but not as a full product-level model risk layer

Needed:

- forecast intervals across more drivers
- scenario analysis
- feature contribution explanations
- reason codes for major forecast moves

#### 7. Regime shift handling

Missing:

- your Pfizer result already shows the limitation

Needed:

- regime detection
- scenario-conditioned forecasts
- event flags for M&A, tax disputes, product cycles, restructuring, and macro shocks

#### 8. Ongoing taxonomy and standards maintenance

Missing:

- no formal update process for IFRS, US GAAP taxonomy, ESEF, or SEC filing changes

Needed:

- standards monitoring
- regression suite against taxonomy changes
- controlled rollout by reporting season

This is especially important because:

- IFRS 18 becomes effective January 1, 2027
- ESMA published a 2025 ESEF taxonomy on April 21, 2026 with IAS 1 and IFRS 18 entry points for 2026 reporting preparation

#### 9. Security, governance, and compliance

Missing:

- no product-grade access control, customer isolation, or governance workflow

Needed:

- role-based access
- tenant separation
- data retention policy
- legal disclaimers
- model governance and approval process

#### 10. Customer workflow and UX

Missing:

- research outputs are not yet packaged as a clean customer workflow

Needed:

- dashboard
- scenario controls
- downloadable reports
- API access
- alerting
- override and commentary features

### The best honest sentence for Q4

> The missing pieces are mostly trust infrastructure: normalization, provenance, controls, monitoring, and sector coverage. A customer will not buy a model they cannot audit.

## 7. Likely Follow-Up Questions

### "Would you build one global model or separate models?"

> I would use a hybrid approach: one global semantic layer, then specialized model families by sector and sometimes by accounting regime. Shared learning should happen after normalization, not before.

### "Would you convert everything to one standard?"

> I would not fully restate everything into one accounting basis unless the use case really required it. I would first map filings into a canonical economic ontology, preserve provenance, and only do explicit adjustment layers where the accounting difference is material to the task.

### "What is the biggest product risk?"

> False comparability. The biggest risk is that the numbers look comparable but are not economically comparable because of accounting, jurisdiction, or industry differences.

### "What is the biggest weakness in your current project relative to these questions?"

> The current simulator is strongest for operating companies with standard working-capital and financing structure. It does not yet have explicit state for intangibles, banking balances, insurance liabilities, or full jurisdiction-aware normalization.

### "Why is that still a good project?"

> Because it solves the right core problem first: internal accounting consistency without plug variables. That gives a strong foundation to extend, as long as the extensions are semantic and sector-aware.

## 8. Five-Minute Break Prompts

If you use one of the five permitted breaks, these are good prompts to paste into an AI tool quickly.

### Break prompt 1

I am in an interview about a structural financial forecasting model. I need a crisp answer on how US GAAP vs IFRS changes ML conclusions for development costs, LIFO, impairment reversals, and cash flow classification. Give me a 90-second spoken answer plus 3 pitfalls.

### Break prompt 2

I need a sharp answer on how to build a jurisdiction-aware financial ML pipeline across SEC, ESEF, Japan, India, and China. Focus on metadata, taxonomies, carve-outs, and filing formats.

### Break prompt 3

I need a 90-second answer on why my current three-statement operating-company simulator should not be directly reused for banks or insurers. Mention IFRS 9, IFRS 17, and intangible-heavy tech.

### Break prompt 4

I need a marketable-product answer for a financial statement ML application. Give me the top 10 missing requirements beyond the prototype: provenance, governance, monitoring, human review, sector coverage, and explainability.

### Break prompt 5

Help me answer: "What are the biggest honest limitations of your current project, and why are they acceptable in a research prototype?"

## 9. One-Minute Closing Summary

If they ask you to summarize everything, say:

> My current report is strongest for standard operating companies, because the simulator explicitly models working capital, fixed assets, liquidity, debt, payouts, and equity in a no-plug way. But to make this a real global product, I would add a semantic normalization layer across standards and jurisdictions, preserve fact-level provenance, and route issuers into sector-specific model families. The main lesson is that accounting standard, jurisdiction, and industry are not nuisance variables. They are part of the data generating process.

## 10. Official Sources

- IAS 38 Intangible Assets: research expensed, development sometimes capitalized, internally generated goodwill not recognized  
  [IFRS IAS 38 PDF](https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ias-38-intangible-assets.pdf)

- IAS 2 Inventories: FIFO or weighted average for interchangeable inventories, inventory write-down reversals  
  [IFRS IAS 2 PDF](https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ias-2-inventories.pdf)

- IAS 36 Impairment of Assets: reversals permitted for non-goodwill, not for goodwill  
  [IFRS IAS 36 PDF](https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2023/issued/part-a/ias-36-impairment-of-assets.pdf?bypass=on)

- IAS 7 and IFRS 18 interaction: new cash flow classification requirements tied to IFRS 18 from January 1, 2027  
  [IFRS IAS 7 page](https://www.ifrs.org/issued-standards/list-of-standards/ias-7-statement-of-cash-flows.html/)

- SEC comparison of IFRS and US GAAP: LIFO, R&D, impairment, tax implications  
  [SEC Work Plan Final Staff Report](https://www.sec.gov/spotlight/globalaccountingstandards/ifrs-work-plan-final-report.pdf)

- SEC treatment of foreign private issuers  
  [SEC Foreign Private Issuer Overview](https://www.sec.gov/divisions/corpfin/internatl/foreign-private-issuers-overview.shtml)

- SEC Inline XBRL requirements  
  [SEC Inline XBRL](https://www.sec.gov/data-research/structured-data/inline-xbrl)

- ESMA ESEF requirements and taxonomy structure  
  [ESMA Electronic Reporting](https://www.esma.europa.eu/issuer-disclosure/electronic-reporting)

- ESMA 2026 taxonomy update and IFRS 18 entry points  
  [ESMA support ESEF implementation with updated taxonomy](https://www.esma.europa.eu/press-news/esma-news/esma-support-esef-implementation-updated-taxonomy)

- IFRS jurisdiction profiles used for country examples  
  [United States](https://www.ifrs.org/use-around-the-world/use-of-ifrs-standards-by-jurisdiction/view-jurisdiction/united-states/)  
  [Japan](https://www.ifrs.org/use-around-the-world/use-of-ifrs-standards-by-jurisdiction/view-jurisdiction/japan/)  
  [Canada](https://www.ifrs.org/use-around-the-world/use-of-ifrs-standards-by-jurisdiction/view-jurisdiction/canada/)  
  [China](https://www.ifrs.org/use-around-the-world/use-of-ifrs-standards-by-jurisdiction/view-jurisdiction/china/)  
  [India](https://www.ifrs.org/use-around-the-world/use-of-ifrs-standards-by-jurisdiction/view-jurisdiction/india/)

- Industry-specific standards you can mention for banks and insurers  
  [IFRS 9 Financial Instruments](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/)  
  [IFRS 17 Insurance Contracts](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-17-insurance-contracts/)
