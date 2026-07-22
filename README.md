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
- runs sensitivity grids built from a spreadsheet-friendly CSV; and
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
6. `cecl_summary.csv`

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
│   ├── financials.csv
│   ├── collateral.csv
│   ├── consumer_scores_1.csv
│   ├── consumer_scores_2.csv
│   ├── fico_pd_lookup.csv
│   └── tag_tieouts.csv
├── scenario.json                 Small manifest that includes the fragments
├── scenario/
│   ├── inputs.json               File locations, aliases, types, aggregation
│   ├── borrower.json             Borrower identity and balance fields
│   ├── modules/
│   │   ├── cre.json              CRE assumptions
│   │   ├── ci.json               C&I assumptions
│   │   └── consumer.json         Consumer assumptions
│   ├── tags/
│   │   ├── model_and_overlay.json
│   │   ├── cre_subsectors.json
│   │   └── ci_sectors.json
│   ├── overlays.json             EF and BCC overlay definitions
│   └── cecl.json                 CECL field and method settings
├── scenario_variables.csv        Editable batch-sensitivity variables
└── scenario_batch.json           Generated batch configuration
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
    "scenario/borrower.json",
    "scenario/tags/model_and_overlay.json",
    "scenario/tags/cre_subsectors.json",
    "scenario/tags/ci_sectors.json",
    "scenario/modules/cre.json",
    "scenario/modules/ci.json",
    "scenario/modules/consumer.json",
    "scenario/overlays.json",
    "scenario/cecl.json"
  ],
  "scenario_id": "example_2026q2",
  "stress_levels": ["S1", "S2"],
  "run": {"cutoff_date": "2026-06-30"},
  "outputs": {"directory": "outputs/example_2026q2"},
  "module_order": ["CRE", "C&I", "Consumer"]
}
```

Included files are merged in the order listed. Values written directly in the
manifest override included values. If more than one top-level JSON file is
passed on the command line, later files override earlier files.

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
the relevant aggregation explicitly says `treat_zero_as_missing`.

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

`loan_id`, `borrower_id`, `subsector`, and `outstanding_balance` are
required in the example. The other fields become operationally required when a
tag, module, or report uses them.

Tag definitions can reference any canonical identity field. The example checks
both `tag_hint` and `tag_hint_2`, so either source column can route a borrower.

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

- fields listed in `sum_fields`, such as balance and reserve, are summed;
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
- make a borrower eligible for a model; and
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

1. EBITDA is reduced according to sector, base bucket, and stress level.
2. Taxes and distributions are re-scaled using their original ratios to
   EBITDA.
3. Sector rules determine whether non-discretionary dividends are subtracted.
4. Management fees and unfinanced capital spending are subtracted.
5. Debt service equals incremental stressed interest on global exposure, plus
   cash interest, plus the sector-specific principal field.
6. FCCR equals stressed available cash flow divided by debt service.
7. FCCR cutoffs determine migration.

Missing C&I components are treated as zero and logged. A row becomes out of
scope when available cash flow is zero or missing, or debt service is
nonpositive. Borrowers already rated Substandard remain Substandard.

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
2. Unstressed loss given default dollars equal balance less collateral, floored
   at zero.
3. Stressed PD equals base PD times the scenario factor, capped at the
   configured maximum.
4. Stressed collateral equals appraisal times the collateral factor, rushed
   sale adjustment, and closing-cost adjustment.
5. Stressed expected loss equals stressed PD times stressed loss given default.

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

### CECL

Commercial CECL calculates one base reserve ratio for each CECL portfolio and
base risk bucket:

```text
reserve ratio = sum of recorded CECL reserve ÷ sum of balance
```

That portfolio-and-bucket ratio is then applied to stressed bucket balances.

Consumer Base CECL is the recorded loan reserve. Stressed Consumer CECL is
stressed quantitative expected loss plus the qualitative amount implied by:

```text
recorded base reserve − unstressed quantitative expected loss
```

If positive Consumer balance is out of scope, stressed Consumer CECL is marked
`unavailable`; it is not reported as zero.

## Out-of-scope treatment

Commercial and Consumer treatment differs:

- A CRE or C&I borrower that cannot be stressed generally remains in its base
  commercial bucket. Its balance stays visible in migration and CECL reports,
  and the reason appears in out-of-scope reports.
- Consumer stressed CECL becomes unavailable when positive Consumer balance is
  out of scope.

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
| `cecl_summary.csv` | Base and stressed pro forma reserve dollars and ratios |
| `scenario_diff.csv` | Optional prior-scenario and marginal-variable comparison |
| `metadata.json` | Scenario/input hashes, runtime versions, exception counts, and report hashes |
| `scenario_used.json` | Merged configuration used for audit; not a portable rerun package |
| `output_manifest.json` | Files owned by the engine in this output directory |

The numeric reports are deterministic when configuration and input files are
unchanged. `run_timestamp_utc` changes on each run.

## Batch sensitivity analysis

### Edit the variable CSV

`examples/scenario_variables.csv` is designed for spreadsheet editing. Each
row needs a `path` to a JSON value and exactly one method for generating
values.

| CSV columns | Meaning |
|---|---|
| `name` | Friendly label; defaults to the path when blank |
| `path` | Canonical dotted JSON path, such as `modules.CRE.tests.dscr.decline.default.S1` |
| `values` | JSON array such as `[1.25, 1.75]`, or a pipe-delimited list |
| `range_start`, `range_stop`, `range_step` | Inclusive numeric sequence |
| `range_inclusive` | Optional true/false control for the stopping value |
| `linspace_start`, `linspace_stop`, `linspace_count` | Evenly spaced values |
| `multipliers` | Values multiplied by the base scenario value |
| `deltas` | Values added to the base scenario value |
| `precision` | Decimal rounding for generated values |
| `allow_create` | Normally blank/false; permits `values`, `range`, or `linspace` to create a missing path. `multipliers` and `deltas` still require an existing numeric value |

### Convert CSV to JSON

```powershell
.\.venv\Scripts\python.exe -m stress_engine.config_tool batch-csv examples/scenario_variables.csv examples/scenario_batch.json --mode grid --output-directory outputs/example_batch --max-scenarios 20
```

Treat `scenario_variables.csv` as the editable source and regenerate
`scenario_batch.json` after changes.

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
| Consumer CECL unavailable | Missing FICO/appraisal/lookup values or invalid Consumer assumptions |
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
