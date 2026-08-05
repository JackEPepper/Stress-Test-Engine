# Credit Stress Engine

The Credit Stress Engine turns loan, financial, collateral, and consumer data
into repeatable stress-test results. It is designed for analysts who need an
auditable process without embedding portfolio assumptions in Python code.

Most day-to-day work happens in CSV files and the JSON files under
`examples/scenario/`. The engine code should not need to change when a source
system changes a column heading: update the corresponding `column_aliases`
value instead.

## What the engine does

The engine:

- loads CSV, XLSX, or XLSM source files;
- converts source headers to stable internal names;
- builds one borrower record from one or more loan records;
- assigns borrowers to CRE, C&I, Consumer, and overlay populations;
- applies the configured stress levels (the example uses S1 and S2);
- calculates stressed risk migration and pro forma CECL results;
- performs source reconciliation, tag tie-outs, and missing-data checks;
- writes detailed audit files and management summaries;
- runs sensitivity grids and paired scenarios defined in JSON; and
- compares a current run with a preserved prior scenario.

The engine does not approve a scenario or replace management review. A command
that finishes successfully can still contain material warnings, unavailable
results, or failed tie-outs. Review the control files described below before
using any result.

## Quick start

### 1. Requirements

You need:

- a command line (the examples below use Windows PowerShell);
- Python 3.9 or newer; and
- permission to install the project dependencies.

From the repository folder, create a private Python environment and install the
engine:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

If the `py` command is unavailable, use `python` in its place.

### 2. Run the included example

```powershell
.\.venv\Scripts\python.exe -m stress_engine examples/scenario.json --no-comparison
```

The example writes results to:

```text
examples/outputs/example_2026q2
```

### 3. Review the run

Open the output files in this order:

1. `exception_log.csv`
2. `source_reconciliation.csv`
3. tie-out rows in `tag_summary.csv`
4. `out_of_scope_summary.csv` and `out_of_scope_detail.csv`
5. `migration_summary.csv` and the module summaries
6. `cecl_basis_summary.csv`
7. `cecl_summary.csv`

Do not interpret a blank or `unavailable` result as zero.

## Recommended operating workflow

For each reporting period:

1. Copy or archive the prior period's scenario folder and input files.
2. Replace the current source files or point `inputs.json` to new files.
3. Update only the right side of `column_aliases` when headers changed.
   Extra source columns that the engine does not use can remain unmapped.
4. Update dates and stress assumptions in the scenario fragments.
5. Run the engine into a new, dated output directory.
6. Review and resolve controls in the order shown above.
7. Retain the scenario files, source files, outputs, and approval evidence
   together.

Reusing an output directory is allowed. Standard runs remove obsolete files
listed in `output_manifest.json`; batch runs use
`batch_output_manifest.json` to remove obsolete batch reports and child-run
files. User-created files are left alone. Use a new output directory when the
old run must be retained.

## Repository map

```text
examples/
├── data/                         Example source data
│   ├── loans.csv
│   ├── cecl_history.csv
│   ├── financials.csv
│   ├── collateral.csv
│   ├── consumer_scores_1.csv
│   ├── consumer_scores_2.csv
│   ├── fico_pd_lookup.csv
│   └── tag_tieouts.csv
├── scenario.json                 Manifest, migration cutoffs, CECL, and includes
├── scenario/
│   ├── inputs.json               Inputs and borrower construction settings
│   ├── modules/
│   │   ├── cre.json              CRE assumptions
│   │   ├── ci.json               C&I assumptions
│   │   └── consumer.json         Consumer assumptions
│   ├── tags/
│   │   ├── model_and_overlay.json
│   │   ├── cre_subsectors.json
│   │   └── ci_sectors.json
│   └── overlays.json             EF and BCC overlay definitions
└── scenario_batch.json           Editable batch-sensitivity configuration
```

The `stress_engine/` folder contains application code. A normal scenario
update should not require changes there.

## How the split scenario works

`examples/scenario.json` is the main manifest. Its `$include` list points to
smaller JSON files:

```json
{
  "$include": [
    "scenario/inputs.json",
    "scenario/tags/model_and_overlay.json",
    "scenario/tags/cre_subsectors.json",
    "scenario/tags/ci_sectors.json",
    "scenario/modules/cre.json",
    "scenario/modules/ci.json",
    "scenario/modules/consumer.json",
    "scenario/overlays.json"
  ],
  "scenario_id": "example_2026q2",
  "stress_levels": ["S1", "S2"],
  "cutoffs": {
    "dscr": {"special_mention": 1.15, "substandard": 1.0},
    "fccr": {"special_mention": 1.15, "substandard": 1.0},
    "ltv": {"special_mention": 0.75, "substandard": 0.9}
  },
  "cecl": {
    "reserve_field": "cecl_reserve",
    "portfolio_field": "cecl_portfolio",
    "reserve_basis": {
      "current_method": "in_place",
      "central_tendency": {
        "z_score_threshold": 2.0,
        "observation_grain": "borrower"
      },
      "historical": {
        "enabled": true,
        "source": "cecl_history",
        "tag_field": "cecl_tag",
        "period_field": "period",
        "bucket_field": "risk_bucket",
        "ratio_field": "historical_cecl_ratio",
        "current_period": {"name": "2026Q2", "weight": 0.3333333333333333},
        "periods": [
          {"name": "2026Q1", "weight": 0.3333333333333333},
          {"name": "2025Q4", "weight": 0.3333333333333333}
        ]
      }
    },
    "portfolios": {
      "Consumer": {"method": "expected_loss"}
    }
  },
  "run": {"cutoff_date": "2026-06-30"},
  "outputs": {"directory": "outputs/example_2026q2"},
  "module_order": ["CRE", "C&I", "Consumer"]
}
```

Included files are merged in the order listed. Values written directly in the
manifest override included values. If more than one top-level JSON file is
passed on the command line, later files override earlier files.

`scenario/inputs.json` contains both the top-level `inputs` definitions and the
top-level `borrower` construction settings. Global CECL field, portfolio, and
method settings live directly in the master manifest.

The master manifest is the single source for migration cutoffs. `cutoffs.dscr`
applies to both standard and refinance CRE DSCR, `cutoffs.fccr` applies to C&I,
and `cutoffs.ltv` applies to CRE LTV. Module fragments do not define their own
cutoff tables.

All relative input and output paths are anchored to the folder containing the
first top-level scenario file. In the example, `data/loans.csv` means
`examples/data/loans.csv`.

### JSON editing rules

JSON is strict:

- use double quotes around names and text;
- do not add comments;
- do not leave a comma after the final item;
- write percentages as decimals, so `0.05` means 5%; and
- distinguish a percentage from a factor: `1.25` means 1.25 times, not 1.25%.

Run the tests or a no-write run after editing JSON if you are unsure whether
the file is valid.

## Preparing source data

### Supported file types

Each input can be:

- CSV;
- XLSX; or
- XLSM.

For Excel files, set `type` to `xlsx` or `xlsm` and use `sheet_name` if
the required data is not on the first sheet.

Use `path` when a logical source has one file. Use `paths` with a JSON list
when the same source is delivered in multiple files. Every listed file must
contain the configured source columns and use the same aliases, date rules,
numeric rules, and aggregation. Files may contain other unrelated columns;
those columns do not need to match across files. The engine loads the files in
listed order, concatenates their rows, and records each physical file
separately in `metadata.json`.

```json
"paths": [
  "data/consumer_scores_1.csv",
  "data/consumer_scores_2.csv"
]
```

### Column aliases

List only the source columns that the engine should import in that source's
`column_aliases` block:

```json
"column_aliases": {
  "borrower_id": "Customer Number",
  "current_dscr": "Current DSCR",
  "noi": "Net Operating Income"
}
```

The left side is the canonical name used everywhere inside the scenario and
engine. The right side is the exact heading in the source file. Any source
column omitted from `column_aliases` is silently ignored: it is not profiled,
aggregated, tagged, reconciled, or written to engine outputs. This allows a
vendor or source system to add unrelated columns without breaking a run.

If `Customer Number` changes to `Borrower #`, make only this change:

```json
"borrower_id": "Borrower #"
```

Do not replace `borrower_id` elsewhere. Date lists, numeric lists,
aggregations, tag rules, and model settings all continue to use canonical
names. A configured source heading that is missing from the file, a missing
required canonical field, or one source heading mapped to multiple canonical
names stops the run with a clear error. Unmapped columns do not create errors
or exception-log entries.

Aliases solve header-only changes. A genuinely different data layout may also
require changes to keys, aggregation, tags, or model configuration.

### Borrower-keyed and account-keyed sources

Most supporting files use `borrower_id` directly. A source that has no
borrower/customer number can instead use an account number already present in
the identity file. The Consumer example uses:

```json
"key": "loan_id",
"identity_key": "loan_id",
"column_aliases": {
  "loan_id": "account_number"
}
```

Here, `account_number` is the physical Consumer-file heading and `loan_id` is
the stable engine name. `identity_key` instructs the engine to match that value
to `loan_id` in the master identity file and retrieve the associated
`borrower_id`. The source rows are then aggregated at borrower level.

Matching is exact and account fields are read as text, preserving leading
zeros. An unmatched source account is reported in source reconciliation and
does not enrich a borrower. If one master account maps to more than one
borrower, the run stops because choosing either borrower would be ambiguous.

### Dates, numbers, and blanks

`date_columns` lists canonical fields that should be read as dates.
`numeric_columns` lists canonical fields that should be read as numbers.
Numeric conversion understands commas, dollar signs, percentage signs, and
parentheses for negatives. Nonblank values that cannot be converted become
missing and are reported in the exception log.

Leave unavailable values blank. Do not use zero as a missing-value code unless
the relevant aggregation explicitly says `treat_zero_as_missing`. The one
exception is an intentionally unavailable CECL history ratio: enter `N/A` or
`#N/A` so that the engine can distinguish a skipped historical cell from
missing or malformed data. Recognition is case-insensitive and ignores
surrounding whitespace; `NA` without the slash is not a skip token.

### Identity file: loans

The identity file defines the population. A borrower found only in a supporting
source does not enter the results; it is reported as an orphan source key.

| Canonical field | Purpose |
|---|---|
| `loan_id` | Unique loan/account identifier, loan-count audit, and Consumer account crosswalk |
| `borrower_id` | Key used to combine loans and supporting sources |
| `borrower_name` | Display and audit name |
| `status` | Tag inclusion or exclusion, such as Paid Off |
| `subsector` | Model and sector tagging; may contain delimited tokens |
| `tag_hint` | Optional additional tag indicator |
| `tag_hint_2` | Second optional tag indicator |
| `risk_rating` | In-place commercial risk rating |
| `maturity_date` | Determines the CRE DSCR or refinance/LTV path |
| `outstanding_balance` | Exposure amount |
| `cecl_reserve` | Recorded base CECL reserve |

`loan_id`, `borrower_id`, `subsector`, `outstanding_balance`, and
`cecl_reserve` are required in the example. The other fields
become operationally required when a tag, module, or report uses them.

Tag definitions can reference any canonical identity field. The example checks
both `tag_hint` and `tag_hint_2`, so either source column can route a borrower.

### CECL history file: tag/bucket reserve ratios

`cecl_history.csv` is a long-form, non-merged input with one row per CECL-level
tag, historical period, and modeled risk bucket:

| Field | Purpose |
|---|---|
| `cecl_tag` | Exact scenario tag object key whose definition sets `cecl_level: true` |
| `period` | Period name referenced by `reserve_basis.historical.periods` |
| `risk_bucket` | `Pass`, `Special Mention`, or `Substandard` |
| `historical_cecl_ratio` | Historical CECL reserve ratio, written as a decimal, or `N/A`/`#N/A` to skip and reweight that cell (case-insensitive; surrounding whitespace ignored) |

Consumer and model-excluded tags such as ARR do not belong in this file.
Duplicate tag-period-bucket rows are rejected. The example contains the full
six-tag by two-period by three-bucket grid, or 36 rows.

### Financial file: C&I cash-flow inputs

The financial source contains the components used to calculate stressed FCCR:

| Canonical field | Purpose |
|---|---|
| `borrower_id` | Merge key |
| `ebitda` | Starting EBITDA |
| `cash_taxes` | Cash tax outflow |
| `cash_distribution` | Cash distribution outflow |
| `cash_dividends` | Total dividends |
| `discretionary_cash_dividends_distribution` | Discretionary dividend portion |
| `cash_management_fees` | Management-fee outflow |
| `unfinanced_capex` | Unfinanced capital spending |
| `global_total_outstanding` | Balance exposed to incremental rate stress |
| `interest_expense` | Total interest expense used by the optional ABL cash-interest calculation |
| `non_cash_interest_expense` | Non-cash interest removed by the optional ABL calculation |
| `cash_paid_for_interest` | Original cash-interest value and fallback |
| `principal_repayments_paid` | Principal measure used by Middle Market |
| `required_principal_paid_period` | Principal measure used by Sponsor/Specialty and ABL |

Source-reported FCCR is deliberately not imported. The engine calculates FCCR
from these components so the stressed result has one consistent definition.

### Commercial collateral file: appraisals, DSCR, and NOI

| Canonical field | Purpose |
|---|---|
| `borrower_id` | Merge key |
| `collateral_id` | Prevents duplicate commercial collateral from being summed twice |
| `current_appraisal_date` | Date of current commercial appraisal |
| `current_appraised_value_raw` | Current commercial appraisal candidate |
| `prior_appraisal_date` | Date of prior commercial appraisal |
| `prior_appraised_value` | Prior commercial appraisal candidate |
| `origination_appraised_value` | Origination commercial appraisal candidate |
| `current_dscr` | Current DSCR candidate |
| `current_dscr_date` | Date of current DSCR |
| `prior_dscr` | Prior DSCR candidate |
| `prior_dscr_date` | Date of prior DSCR |
| `origination_dscr` | Origination DSCR candidate |
| `noi` | Net operating income |

The example selects DSCR in this order: current, prior, origination. Blank and
zero candidates are skipped. Within a candidate, the newest dated row wins.

Commercial appraisals use the same fallback order separately for each
`collateral_id`, then sum the selected values by borrower. These values remain
separate from the Consumer appraisal total.

### Consumer files: FICO and appraisals

The example treats two physical CSVs as one logical `consumer_scores` source.
Each file carries both the FICO and appraisal fields for its accounts. It does
not contain a borrower or customer number.

| Canonical field | Purpose |
|---|---|
| `loan_id` | Canonical account key; supplied by the physical `account_number` column |
| `current_fico_score` | Current FICO candidate |
| `current_fico_date` | Current FICO date |
| `origination_fico_score` | Origination FICO fallback |
| `origination_fico_date` | Origination FICO date |
| `collateral_id` | Identifies a Consumer collateral item |
| `current_appraisal_date` | Date of current appraisal |
| `current_appraised_value_raw` | Current appraisal candidate |
| `prior_appraisal_date` | Date of prior appraisal |
| `prior_appraised_value` | Prior appraisal candidate |
| `origination_appraised_value` | Origination appraisal candidate |

Each account first maps through the identity file to a borrower. The source
aggregation then produces one canonical `fico_score` for that borrower, which
enters the Consumer calculation, and `fico_date`, which records the selected
score date for audit purposes. Appraisals use the same
current/prior/origination fallback chain separately for each `collateral_id`
across all of the borrower's accounts; the selected collateral values are then
summed into `consumer_appraised_value`. Repeated history rows for one collateral
ID therefore do not inflate the total.

The borrower audit also records `fico_source_record` and
`consumer_appraisal_source_record` as `full file path#row=N`. This identifies
the physical file and row that supplied a selected value. If two files report
different candidate values for the same borrower/entity and date, the engine
uses deterministic listed-file order and records a source reconciliation
warning for review. Listing the same resolved file twice is rejected.

### FICO-to-PD lookup

The lookup requires `min_score`, `max_score`, and `pd`. Score bands are
inclusive. Bands should not overlap, PD must be between zero and one, and the
range should not contain unintended gaps. Invalid, overlapping, or gapped
tables are reported.

### Tag tie-out file

The tie-out source contains:

- `tag`: the configured tag name; and
- `expected_balance`: the independently expected balance.

The engine compares tagged balance with expected balance using the tolerance in
the tag configuration.

## Building the borrower population

The engine begins with loan rows and creates one row per borrower:

- fields listed in `sum_fields`, such as balance, are summed;
- the configured current CECL reserve is always summed across a borrower's
  loans, even if it is omitted from `sum_fields`;
- loan IDs are collected into a unique list;
- other ordinary fields use the first nonblank value unless an explicit
  aggregation is configured; and
- rating, maturity, and every field used by tag conditions come from the
  borrower's largest loan.

Using one representative loan for rating, maturity, and tagging prevents a
borrower's module from being determined by one loan while its rating comes from
another. Conflicting loan-level values are still reported for review.

Rows with missing borrower IDs receive deterministic placeholder IDs so they
remain visible instead of being combined accidentally.

## Tags and module assignment

Tags determine which borrowers enter models and which sector labels are
assigned. A tag can:

- include borrowers that meet conditions;
- exclude borrowers that meet other conditions;
- assign derived fields such as `cre_subsector` or `ci_sector`;
- make a borrower eligible for a model;
- define a module-scoped CECL ratio group; and
- compare its balance with an external tie-out.

Supported condition operators are:

`eq`, `ne`, `in`, `not_in`, `gt`, `gte`, `lt`, `lte`,
`between`, `contains`, `has_token`, `has_any_token`,
`has_all_tokens`, `startswith`, `endswith`, `is_null`,
`not_null`, and `regex`.

Token operators split values on semicolons, commas, or pipes by default. This
allows one source field to contain values such as:

```text
Retail;Middle Market
```

Tags with `"model_eligible": false` are audit and classification tags: they do
not make a borrower eligible for a module, but they do not override a separate
model-eligible tag. Use `"exclude_from_model": true` when a matched borrower
must be excluded from all model calculations. Model-exclusion tags are
automatically non-model-eligible, must define a nonempty include condition, and
stop the run if any referenced condition field is absent.

A commercial tag can also define the scale for CECL reserve ratios:

```json
"CI_Sector_Middle_Market": {
  "model_eligible": false,
  "cecl_level": true,
  "cecl_module": "C&I",
  "include": [
    {"field": "subsector", "op": "has_token", "value": "Middle Market"}
  ]
}
```

`cecl_level: true` marks the tag as a CECL grouping key. `cecl_module` scopes
that key to a routed module. After model exclusions and module priority are
resolved, the engine considers only active CECL-level tags whose `cecl_module`
equals the borrower's `primary_module`. A commercial modeled borrower must
resolve to exactly one such tag. No match or multiple matches in the selected
module stop the run instead of depending on JSON order. Because overlay
migrations are calculated at public-portfolio grain, every modeled row in one
overlay CECL portfolio must resolve to the same single CECL-level tag.

This routing scope is important for overlaps. The example borrower that
matches both CRE and Middle Market is routed to CRE, so `CRE_Model` is its CECL
tag and `CI_Sector_Middle_Market` is ignored for CECL. Both raw tag flags and
the Middle Market tie-out remain visible. The example marks exactly six tags:
`CRE_Model`, the three `CI_Sector_*` tags, and the EF and BCC overlay tags.
`Consumer_Model`, `CI_Model`, every CRE subsector tag, and `ARR` are deliberately
unflagged.

The example scenario defines `ARR` as an excluded subset of Sponsor and
Specialty:

```json
"ARR": {
  "model_eligible": false,
  "exclude_from_model": true,
  "include": {
    "all": [
      {
        "field": "subsector",
        "op": "has_token",
        "value": "Sponsor and Specialty"
      },
      {
        "any": [
          {"field": "subsector", "op": "has_token", "value": "ARR"},
          {"field": "tag_hint", "op": "has_token", "value": "ARR"},
          {"field": "tag_hint_2", "op": "has_token", "value": "ARR"}
        ]
      }
    ]
  }
}
```

An ARR row therefore remains in
`tag_ci_sector_sponsor_and_specialty`, `ci_sector`, `all_tags`, Sponsor and
Specialty population totals, and Sponsor and Specialty tie-outs. It also
receives `model_excluded = true` and `model_exclusion_tags = ARR`. Its model
routing fields are cleared and it is omitted from stress modules, migration,
CECL, overlays, and targeted-stress selections. Because ARR is a subset, ARR
and Sponsor and Specialty balances overlap and should not be added together.

`module_order` is both the execution order and the priority order. In the
example, CRE has priority over C&I, which has priority over Consumer. A borrower
that matches both CRE and C&I remains visible in `eligible_modules`, but only
the selected `primary_module` applies stress. The overlap is logged.

## How the calculations work

### Base commercial risk bucket

The base bucket is derived from `risk_rating`:

| Rating | Base bucket |
|---|---|
| Less than 7 | Pass |
| Exactly 7 | Special Mention |
| Greater than 7 | Substandard |
| Missing or invalid | Unknown |

### CRE

CRE first compares maturity with the scenario cutoff date plus the configured
maturity threshold.

For longer-dated loans:

```text
stressed DSCR = selected DSCR × (1 − configured decline)
```

For loans inside the maturity threshold:

1. NOI is reduced using the configured decline.
2. A stressed interest rate is built from Treasury rate plus credit spread.
3. Annual debt payment is calculated from balance, stressed rate, and
   amortization term.
4. Refinance DSCR equals stressed NOI divided by annual debt payment.
5. Stressed LTV equals balance multiplied by stressed cap rate, divided by NOI.
6. Migration is the worse of the existing base bucket, refinance DSCR result,
   and LTV result; stress never improves an existing bucket.

Subsector assumptions can override the `default` assumptions. Use decimal
rates: a decline of `0.08` means 8%.

### C&I

C&I calculates stressed FCCR rather than importing it:

1. EBITDA is reduced according to sector, BRG, and stress level.
2. Taxes and distributions are re-scaled using their original ratios to
   EBITDA.
3. Sector rules determine whether non-discretionary dividends are subtracted.
4. Management fees and unfinanced capital spending are subtracted.
5. Debt service equals incremental stressed interest on global exposure, plus
   cash interest, plus the sector-specific principal field.
6. FCCR equals stressed available cash flow divided by debt service.
7. FCCR cutoffs determine migration.

The BRG comes from the configured `borrower.risk_rating_field`. Integral grades
1 through 7 select the matching string key in `ebitda_reduction`; every finite
numeric grade of 8 or higher selects key `"8"`. Missing, nonnumeric, below-1,
and fractional grades below 8 do not select an assumption. Every effective
sector table must be keyed by BRG and supply the applicable grade; bucket-keyed
EBITDA reduction tables are not supported.

Missing C&I components are treated as zero and logged. A row becomes out of
scope when available cash flow is zero or missing, or debt service is
nonpositive. Borrowers already rated Substandard still receive an FCCR
calculation, but their migration bucket remains Substandard.

Each configured sector can choose its principal field, whether to subtract
non-discretionary dividends, and whether to calculate cash interest from its
expense components. The canonical defaults are `principal_repayments_paid`,
`false`, and `false`, respectively. An unlisted sector first uses the optional
`sectors.default` object, then those canonical defaults; the fallback is
recorded in the exception log.

The example enables `use_calculated_cash_paid_for_interest` for Asset-Based
Lending. When enabled, debt service uses:

```text
calculated cash paid for interest = interest expense - non-cash interest expense
```

If both alternative fields are blank, or if their difference is zero, the
engine uses `cash_paid_for_interest` instead and records the reason. If only one
alternative component is blank, the normal C&I lenient rule treats that
component as zero and logs the substitution. A blank or zero fallback value is
still subject to the ordinary debt-service and out-of-scope rules.

The stressed borrower results expose the selected value in
`calculated_cash_paid_for_interest`, identify its source in
`calculated_cash_paid_for_interest_source`, and show any fallback reason in
`calculated_cash_paid_for_interest_fallback_reason`.

### Consumer

Consumer uses the aggregated FICO score and `consumer_appraised_value` from the
same logical multi-file source:

1. FICO maps to an unstressed PD through the lookup table.
2. Baseline collateral equals appraisal after the rushed-sale and closing-cost
   adjustments.
3. Baseline loss given default dollars equal balance less adjusted baseline
   collateral, floored at zero.
4. Stressed PD equals base PD times the scenario factor, capped at the
   configured maximum.
5. Stressed collateral applies the scenario collateral factor to appraisal in
   addition to the same rushed-sale and closing-cost adjustments.
6. Stressed expected loss equals stressed PD times stressed loss given default.

The stressed borrower output retains the gross appraisal in
`consumer_appraised_value` and exposes the adjusted baseline collateral in
`consumer_collateral_value_unstressed`.

### Overlays

An overlay has its own base Pass, Special Mention, Substandard, and Unknown
mix. Its stressed Special Mention and Substandard ratios grow in line with
weighted source populations.

Each source requires a nonblank `name`. It can also use:

- one or more `tags`, a portfolio, or include/exclude conditions to select
  borrowers;
- a `module` when its name is not the corresponding module name; and
- a `weight`, which defaults to 1.

Source weights are normalized across sources that have usable populations.
Tag-based source populations also respect `primary_module`, so an overlap
resolved to CRE does not accidentally enter the C&I source population.

### Targeted external shocks

`targeted_stress` is an opt-in loan-grain execution mode for sector or
externally supplied shocks. When it is absent, the engine retains the ordinary
borrower-grain behavior and output schemas.

The complete oil-and-tariff example is
`examples/targeted_stress.json`. It produces an implicit `baseline` plus named
variants:

- `unmatched_behavior: "baseline_stress"` layers a shock into the ordinary
  portfolio stress;
- `unmatched_behavior: "base"` leaves unmatched loans in their base buckets,
  isolating the targeted shock; and
- `primary_variant` selects the variant returned in `result["results"]`;
  `baseline` is the default.

A shock selects loans with nested `all`/`any` expressions. Atomic selectors
can use:

- `type: "naics_prefix"` with a configured loan field and one or more 2-6
  digit prefixes;
- `type: "external_list"` with a loaded input source, source field, exposure
  field, `exact` or `prefix` matching, and an optional tier field; or
- an ordinary tag-style field condition.

`exclude` uses the same selector syntax. External lists can match loan IDs,
borrower IDs, or industry codes. Account IDs that are missing or duplicated in
the identity input are not eligible for loan-ID list selection.

Each selected loan resolves to a configured tier. A tier contains a `modules`
object and approved assumption changes:

```json
{
  "high": {
    "modules": {
      "C&I": {
        "ebitda_reduction": {
          "operation": "add",
          "values": {"S1": 0.08, "S2": 0.15}
        },
        "interest_rate_stress": {
          "operation": "multiply",
          "values": {"S1": 1.5, "S2": 1.5}
        }
      }
    }
  }
}
```

Operations are `replace`, `add`, or `multiply`. If a variant lists multiple
shocks, the engine applies them in listed order. The assumption audit records
the baseline value, value before each operation, shock value, and final
effective value.

Supported targeted assumptions are:

| Module | Parameters |
|---|---|
| C&I | `ebitda_reduction`, `interest_rate_stress` |
| CRE | `dscr_decline`, `refinance_noi_decline`, `treasury_rate`, `credit_spread`, `amortization_years`, `cap_rate` |
| Consumer | `pd_increase_factor`, `collateral_value_factor`, `rushed_sale_discount`, `closing_costs` |

Targeted runs write `stressed_loan_results.csv`. Variant-aware summaries
contain `scenario_variant`, distinct `borrower_count`, and `loan_count`.
Additional controls are written to:

- `targeted_selection_detail.csv`;
- `targeted_assumption_audit.csv`;
- `targeted_stress_summary.csv`; and
- `variant_comparison.csv`.

`targeted_selection_detail.csv` distinguishes the raw selector result from the
final selection. In particular, `selection_reason = model_excluded` shows that
a shock selector matched an exposure but an ARR-style exclusion vetoed it.

### CECL

For the current-quarter `in_place` component, commercial CECL calculates one
base reserve ratio for each resolved CECL-level tag and base risk bucket:

```text
reserve ratio = sum of recorded CECL reserve / sum of balance
```

The final ratio applied to stressed bucket balances can also include a
current-quarter central-tendency component or weighted tag/bucket history, as
configured below.

`cecl.reserve_basis.current_method` selects the current-quarter commercial
basis:

- `in_place` uses the configured current loan-level `reserve_field`. This is
  the default when `reserve_basis` is omitted and preserves legacy behavior.
- `central_tendency` calculates each borrower's current reserve-to-balance
  ratio, calculates a one-pass population z-score within each resolved
  CECL-level tag/base bucket, inclusively retains observations whose absolute
  z-score is at or below `z_score_threshold`, and uses the arithmetic mean of
  the retained ratios. Singleton and zero-variance groups retain every
  observation.

`reserve_basis.historical.enabled` independently controls tag/bucket history.
When enabled, the scenario must define at least one `cecl_level: true` tag, and
`source` must identify a `merge: false` long-form input with one unique row per
configured CECL-level tag, historical period, and risk bucket. Period weights,
including `current_period.weight`, must be finite, greater than zero, and sum
to one. An arbitrary number of historical periods is supported. Explicit-tag
scenarios fail closed if the resolved `cecl_level_tag` field is absent; only
legacy scenarios with neither tag history nor CECL-level tags may fall back to
the public CECL portfolio grain.

Central tendency is applied only to the current-quarter loan population.
Historical CSV ratios are authoritative values and are never z-score trimmed
or divided by current balances. Duplicate configured tag-period-bucket rows are
rejected. A ratio written as `N/A` or `#N/A` (case-insensitive, after trimming
surrounding whitespace) skips only that historical tag/bucket/period cell. The
remaining current and historical configured weights for that tag/bucket are
normalized to sum to one. Missing rows, blank cells, `NA`, negative, malformed,
nonfinite, or greater-than-100% values remain errors: they are not skipped or
reweighted, and the affected tag/bucket basis is unavailable. The current
period cannot be skipped. Active-tag history rows must use `Pass`, `Special
Mention`, or `Substandard`; other bucket names are rejected. Extra tags and
unconfigured periods do not affect current calculations. The example blends
the current quarter and two prior tag/bucket ratios at one-third each before any
cell-local reweighting.

Each successfully reweighted CECL tag/bucket produces one `WARNING` with code
`CECL_HISTORY_RATIO_SKIPPED_REWEIGHTED`; its details list every skipped period
for that cell. This makes intentional skip-and-reweight decisions visible in
`exception_log.csv` without marking the resulting basis unavailable.

Within each tag and period, supplied ratios must not decrease from `Pass` to
`Special Mention` to `Substandard`. The final applied ratios are checked again
after blending; a decreasing commercial ladder is marked unavailable so an
increasing-stress migration cannot silently reduce reported CECL.

Historical ratios must represent the same CECL-level tag, risk-bucket
definition, and model-included commercial population used by the current run.
Do not supply Consumer or model-excluded subsets such as ARR. An ARR borrower
continues to appear in its parent Sponsor and Specialty tag and tie-out, but it
does not enter the current CECL population and is not represented in the
historical ratio.

Missing, invalid, or nonfinite values in the current loan-level
`cecl.reserve_field` are treated as zero and recorded in the exception report.

Negative, invalid, or nonfinite balance observations are never treated as
zero-balance CECL rows. Their count is carried through borrower aggregation,
every bucket basis for the affected CECL-level tag is marked unavailable, and
the condition is recorded in both the basis audit and exception report. Only a
finite balance whose absolute value is within `zero_balance_tolerance` is
eligible for zero-balance not-applicable treatment.

Each commercial tag/bucket cell blends the selected current-quarter ratio with
the historical ratios supplied for that exact tag and bucket:

```text
retained weight
  = sum(configured weights for the current period and non-skipped history)

effective period weight
  = configured period weight / retained weight

effective tag/bucket ratio
  = sum(effective period weight x available period ratio)
```

The effective ratio is applied to that tag's balance in the applicable Base or
stressed risk bucket. Historical balances, loans, and reserve dollars are not
reconstructed. If a tag/bucket has no current observation while the current
period has positive weight, its blended ratio is unavailable; history weights
are not renormalized around missing or invalid data. Explicit `N/A`/`#N/A`
historical cells are the sole reweighting exception. If every configured
historical cell for a tag/bucket is explicitly skipped, the available current
quarter receives an effective weight of one. A zero report balance for that
cell remains not-applicable and does not block otherwise valid CECL totals,
while a positive Base or stressed balance makes the affected result
unavailable.

`Unknown` is intentionally outside the supplied historical risk ladder. A
commercial row with an unknown base risk bucket keeps its current in-place
reserve and does not use central tendency or history; this preserves visibility
and current CECL while the missing rating remains separately reported.

The current `cecl.reserve_field` must use the same identity and borrower wiring
when `current_method` is `central_tendency`, because the estimator is
borrower-grain.
Current reserve and portfolio field names must be nonblank, distinct, and must
not reuse loader metadata, borrower keys, tag outputs, stress metrics, or other
engine-owned columns. This prevents a later pipeline stage from overwriting an
authoritative CECL input.

Central tendency uses borrower-grain observations within each resolved current
CECL-level tag/base bucket, including targeted runs. This avoids silently
changing the estimator merely because output grain changes from borrower to
loan when routing is the same. `cecl_basis_summary.csv` records each current and
historical component's CECL tag, risk bucket, source, method, period, configured
`weight`, normalized `effective_weight`, observation and current-period trimming
counts, period ratio, weighted component, final effective ratio, availability
status, and exception code. Explicitly skipped history rows have status
`skipped`, effective weight zero, and weighted component zero.
Each tag/bucket has one current-period audit row and one direct-ratio row for
each configured historical period. Any balance and implied reserve shown on a
history audit row are explanatory only; the supplied ratio is never derived by
allocating a portfolio reserve amount. Historical audit rows use
`period_method = tag_bucket_history`; a commercial in-place blend identifies
its `reserve_basis` as `in_place+tag_bucket_history`.
The resolved amount and method are also retained on stressed result rows as
`cecl_effective_reserve_base` and `cecl_reserve_basis_method`.

Consumer never uses CECL-level tags, central tendency, or tag/bucket history.
Consumer Base CECL always uses each record's current in-place
`cecl.reserve_field`, and its report rows identify `reserve_basis` as
`in_place`. No Consumer row is required in the history CSV, and an unavailable
commercial tag/bucket history component does not block Consumer CECL.
Stressed Consumer CECL is stressed quantitative expected loss plus the
qualitative amount implied by:

```text
current in-place reserve − unstressed quantitative expected loss
```

The unstressed quantitative expected loss in this residual uses baseline
collateral after rushed-sale and closing-cost adjustments.

Consumer reporting applies a borrower-level carry-forward waterfall in the
declared stress-level order. Base quantitative expected loss is the calculated
amount, or zero when the Base calculation is unavailable; Base qualitative
reserve is the residual needed to equal the current in-place reserve. For each
stressed level, quantitative expected loss is the greater of the prior
effective amount and the current calculated amount. If the current level is missing or out of
scope, the prior effective amount carries forward. Qualitative reserve is also
prevented from declining, including when a configured qualitative floor first
applies under stress. Pro forma CECL is then rebuilt as:

```text
effective quantitative expected loss + effective qualitative reserve
```

This guarantees `Base <= S1 <= S2` for Consumer CECL when the scenario declares
levels in increasing severity order, regardless of changes in the calculated
population. Raw borrower-level modeled values and scope flags remain unchanged
for audit, the full Consumer balance remains in the reported denominator, and
affected records remain visible in the dedicated out-of-scope reports. A
missing configured reserve field remains a CECL data error and makes Consumer
CECL unavailable.

The Aggregate row also remains monotonic when the commercial CECL portfolio
totals are non-decreasing. Commercial portfolios continue to use their separate
bucket-reserve-ratio method; Consumer carry-forward does not override that
method. A CECL portfolio must not mix Consumer and commercial rows because one
portfolio cannot use both expected-loss and bucket-reserve-ratio methods; the
run stops with a configuration error if such a mix is detected. Consumer CECL
portfolios must use `expected_loss`, and non-Consumer portfolios cannot use
that method.

`cecl_summary.csv` identifies the selected basis in `reserve_basis`, uses one
common schema across portfolios, and does not add Consumer-only in-scope or
out-of-scope balance columns. Those diagnostics
remain in `consumer_summary.csv` and the out-of-scope reports.

## Out-of-scope treatment

Commercial and Consumer treatment differs:

- A CRE or C&I borrower that cannot be stressed generally remains in its base
  commercial bucket. Its balance stays visible in migration and CECL reports,
  and the reason appears in out-of-scope reports.
- A Consumer borrower with a missing, out-of-scope, or lower stressed
  contribution carries forward its prior effective CECL components. Review
  `consumer_summary.csv` scope columns and out-of-scope detail for the affected
  balance.

This is why out-of-scope files must be reviewed alongside headline results.

## Running scenarios

### Standard run

```powershell
.\.venv\Scripts\python.exe -m stress_engine examples/scenario.json --no-comparison
```

### Override the output folder

```powershell
.\.venv\Scripts\python.exe -m stress_engine examples/scenario.json --output-dir outputs/2026q2_review --no-comparison
```

A relative override is still anchored to the first scenario file, so this
writes under `examples/outputs/2026q2_review`.

### Useful options

| Option | Meaning |
|---|---|
| `--output-dir PATH` | Override the configured output directory |
| `--no-comparison` | Skip prior-scenario comparisons |
| `--no-write` | Calculate in memory without creating output files |
| `--previous-scenario PATH` | Rerun and compare with a preserved prior scenario; repeatable |
| `--batch` | Run the attached batch configuration |
| `--batch-mode grid|paired` | Override the configured batch mode |
| `--max-scenarios N` | Limit the number of generated sensitivity runs |
| `--no-write-child-outputs` | Keep batch summaries but omit full child folders; previously engine-owned child outputs are cleaned up when reusing the directory |

## Control review

### 1. Exception log

`exception_log.csv` includes three severities:

| Severity | Meaning |
|---|---|
| ERROR | A calculation or control was unavailable or materially invalid |
| WARNING | The run continued, but data or reconciliation requires review |
| INFO | A documented default or fallback was used |

Start with ERROR, then WARNING. INFO rows are useful for explaining how defaults
were applied.

### 2. Source reconciliation

`source_reconciliation.csv` shows:

- null and duplicate source keys;
- source keys absent from the configured borrower or identity key;
- borrowers with and without source matches;
- matched and unmatched borrower balance;
- coercion issues; and
- conflicting values for repeated collateral/entity keys.

`key_field` identifies the canonical key read from the supporting source.
`identity_key_field` is populated when that source must be translated through
the master identity file before borrower-level aggregation.

Lookup and tie-out files have `merge_enabled = false`; borrower-match measures
are intentionally blank for those files.

### 3. Tag tie-outs

`tag_summary.csv` contains both population rows and tie-out rows. Filter to
rows where `tie_out_name` is populated, then review `expected`, `actual`,
`difference`, `tolerance`, and `passed`.

Raw `borrower_count`, `balance`, and tie-out `actual` remain inclusive so a
subset such as ARR stays in its Sponsor and Specialty parent control. The
`not_model_excluded_*` and `model_excluded_*` count and balance columns
reconcile that raw population to the global model veto. These columns describe
the veto split, not final module scope. Targeted runs additionally provide
corresponding loan counts.

### 4. Out-of-scope reports

`out_of_scope_summary.csv` counts issues by module, stress level, test, field,
and reason. `out_of_scope_detail.csv` identifies affected borrowers.

### 5. Results

Only after reviewing controls should you use migration, module, overlay, or CECL
summaries. `migration_summary.csv` covers commercial and overlay buckets; it
does not contain Consumer. `cecl_summary.csv` includes Consumer, so the two
files' aggregate balances are not expected to match.

## Output file guide

| File | What it contains |
|---|---|
| `borrower_audit_raw.csv` | Borrower records after aggregation, tagging, and enrichment but before stress |
| `stressed_borrower_results.csv` | Borrower-level metrics, stressed buckets, module applied, and scope flags |
| `input_summary.csv` | Counts, missing values, uniqueness, and numeric statistics for canonical input fields |
| `source_reconciliation.csv` | Key cardinality, coverage, unmatched balances, and source-data controls |
| `tag_summary.csv` | Tag populations, balances, and external tie-outs |
| `out_of_scope_summary.csv` | Aggregated missing/invalid calculation reasons |
| `out_of_scope_detail.csv` | Borrower-level out-of-scope reasons |
| `exception_log.csv` | INFO, WARNING, and ERROR events |
| `cre_summary.csv` | CRE DSCR/LTV and stressed-balance summary by subsector |
| `ci_summary.csv` | Calculated stressed FCCR and stressed-balance summary by sector |
| `consumer_summary.csv` | Consumer PD, LGD, EL, and reserve summary |
| `migration_summary.csv` | Commercial and overlay balance by portfolio, level, and bucket |
| `overlay_summary.csv` | Overlay sources, weights, source ratios, and final stressed ratios |
| `cecl_basis_summary.csv` | CECL configured/effective period weights, skipped history, central-tendency trimming, effective ratios, and availability controls |
| `cecl_summary.csv` | Base and stressed pro forma reserve dollars and ratios |
| `scenario_diff.csv` | Optional prior-scenario and marginal-variable comparison |
| `metadata.json` | Scenario/input hashes, runtime versions, exception counts, and report hashes |
| `scenario_used.json` | Merged configuration used for audit; not a portable rerun package |
| `output_manifest.json` | Files owned by the engine in this output directory |

The numeric reports are deterministic when configuration and input files are
unchanged. `run_timestamp_utc` changes on each run.

## Batch sensitivity analysis

### Edit the batch JSON

`examples/scenario_batch.json` is the editable source for batch sensitivity
analysis. Each entry in `variables` needs a dotted `path` to a scenario value
and exactly one value generator:

| JSON member | Meaning |
|---|---|
| `name` | Friendly label; defaults to the path when omitted |
| `path` | Canonical dotted JSON path, such as `modules.CRE.tests.dscr.decline.default.S1` |
| `values` | Explicit array of replacement values |
| `range` | Inclusive numeric sequence with `start`, `stop`, and `step` |
| `linspace` | Evenly spaced values with `start`, `stop`, and `count` |
| `multipliers` | Values multiplied by the base scenario value |
| `deltas` | Values added to the base scenario value |
| `precision` | Optional decimal rounding for generated values |
| `allow_create` | Normally omitted or false; allows `values`, `range`, or `linspace` to create a missing path. `multipliers` and `deltas` still require an existing numeric value |

The example exercises all five generators across CRE, C&I, and Consumer
assumptions. In `grid` mode its five variables produce 48 generated scenarios,
which is below the configured `max_scenarios` guardrail of 50.

### Choose a mode

- `grid` runs every combination of values.
- `paired` runs the first value from every variable together, then the second
  value from every variable together. Every variable must have the same number
  of values.

The engine always runs and reports the base scenario, then calculates deltas
for generated scenarios.

### Run the batch

```powershell
.\.venv\Scripts\python.exe -m stress_engine examples/scenario.json examples/scenario_batch.json --batch --no-comparison --no-write-child-outputs
```

Batch outputs are:

| File | Purpose |
|---|---|
| `batch_summary.csv` | Aggregate CECL results and deltas by run and stress level |
| `batch_variables.csv` | Variable values used in each generated run |
| `batch_cecl_summary.csv` | Full CECL rows with run and variable columns |
| `batch_exceptions.csv` | Exceptions from all included runs |
| `batch_metadata.json` | Scenario counts, hashes, per-run exception counts, and report hashes |
| `batch_output_manifest.json` | Batch reports and child directories owned by the engine for safe cleanup on reuse |

The maximum-scenario limit is checked before a grid or paired set is built, so
an accidental very large grid is rejected before child scenarios consume
memory.

## Comparing with a prior scenario

Preserve both the prior configuration and the prior source files. The engine
reruns the prior paths; it cannot reconstruct overwritten historical data from
hashes or old summaries.

A useful archive structure is:

```text
archive/2026q1/
├── scenario.json
├── scenario/
└── data/
```

Run the comparison:

```powershell
.\.venv\Scripts\python.exe -m stress_engine examples/scenario.json --previous-scenario archive/2026q1/scenario.json
```

`scenario_diff.csv` reports:

- changes in data profiles;
- nonzero or availability-related marginal CECL effects from rerunning one
  changed scenario variable at a time; and
- variable rerun errors or skipped reruns.

A changed scenario variable with exactly zero CECL impact does not produce its
own row.

Use the preserved prior manifest, not an output-folder `scenario_used.json`.
The audit copy retains relative input paths, which normally resolve incorrectly
when moved into an output directory.

## Troubleshooting

| Message or symptom | What to check |
|---|---|
| JSON parsing error | Double quotes, trailing commas, and accidental comments |
| Source file not found | Path relative to the first scenario manifest |
| Missing configured aliased columns | Right-side alias spelling versus the exact source header |
| Extra or unused source columns | No action is needed; unmapped columns are ignored |
| Missing required columns | Confirm the canonical left-side alias exists and is listed correctly |
| Source orphan keys | Borrower or account IDs present in a supporting source but absent from its configured identity key |
| Ambiguous account mapping | Ensure each master account number belongs to exactly one borrower |
| Cardinality warning | Duplicate borrower keys in a source expected to have one row per borrower |
| Failed tag tie-out | Tag logic, expected amount, source key, and tolerance |
| Consumer value carries forward in stressed CECL | Review missing FICO/appraisal/lookup values, invalid assumptions, and Consumer out-of-scope detail |
| CRE/C&I out of scope | Review borrower, field, test, and reason in out-of-scope detail |
| Batch exceeds maximum | Reduce values, use paired mode, or deliberately raise `max_scenarios` |
| Excel read error | Confirm XLSX/XLSM format, sheet name, and that the file is not corrupt |

## Glossary

| Term | Plain-language meaning |
|---|---|
| Base | The unstressed, in-place position |
| S1 / S2 | User-defined stress levels; S2 is typically more severe |
| Canonical field | Stable internal name used by configuration and calculations |
| Source header | Actual column heading in CSV or Excel |
| DSCR | Debt service coverage ratio |
| FCCR | Fixed charge coverage ratio |
| BRG | Borrower risk grade used to select C&I EBITDA stress |
| LTV | Loan-to-value ratio |
| NOI | Net operating income |
| PD | Probability of default |
| LGD | Loss given default |
| EL | Expected loss |
| CECL | Current expected credit losses reserve |
| Overlay | Portfolio stressed using migration patterns from modeled populations |
| Tie-out | Comparison with an independently expected amount |
| Out of scope | A calculation could not be completed for the stated reason |

## Tests and developer check

Run the complete automated test suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

The tests cover the sample run, split manifests, configured source aliases,
ignored extra source columns, borrower aggregation, missing-data behavior,
overlays, CECL, comparisons, batch generation, and deterministic reporting.
