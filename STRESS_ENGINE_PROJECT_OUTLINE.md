# Deterministic Credit Stress Engine Project Outline

## 1. Objective

Build a deterministic stress engine that ingests scenario variables and loan-level input data, tags and cleans exposures, selects one appropriate stress module per loan, aggregates stressed asset quality metrics, and produces sector-by-sector reports.

The project should use as much pure Python as possible. External dependencies should be limited to packages commonly available in a basic Anaconda distribution.

Recommended dependency set:

- Python standard library: `dataclasses`, `json`, `csv`, `pathlib`, `datetime`, `logging`, `decimal`, `typing`, `unittest`
- Anaconda core packages: `pandas`, `numpy`, `openpyxl`, `matplotlib`

Avoid unnecessary dependencies such as web frameworks, database engines, distributed compute frameworks, or probabilistic simulation packages unless a future requirement makes them necessary.

## 2. Design Principles

- Deterministic outputs: the same inputs, scenario, and configuration must always produce the same results.
- Modular stress logic: CRE, C&I, and consumer stress calculations should live in separate modules behind a shared interface.
- Transparent assumptions: every transformation, shock, and derived metric should be traceable to input data or scenario configuration.
- Data-contract driven: input schemas, required fields, and validation rules should be explicitly documented and checked.
- Simple local execution: the engine should run from the command line without requiring a server, cloud service, or database.
- Audit-ready outputs: each run should create loan-level results, aggregation reports, validation logs, and run metadata.
- Reconciled populations: tag-level populations should tie to configured external source values and report any differences.
- Comparable runs: the engine should explain aggregated changes between runs through scenario attribution and data attribution.

## 3. Proposed Repository Structure

```text
stress_engine/
  README.md
  pyproject.toml
  config/
    base_config.json
    data_dictionary.json
  scenarios/
    example_base_case.json
    example_adverse_case.json
    example_severe_case.json
    example_range_case.json
  data/
    raw/
    interim/
    processed/
  external_sources/
    tag_population_targets.csv
  reports/
  src/
    stress_engine/
      __init__.py
      cli.py
      run.py
      io/
        __init__.py
        loaders.py
        writers.py
      validation/
        __init__.py
        schemas.py
        checks.py
        rule_sets.py
      tagging/
        __init__.py
        loan_tags.py
        collateral_tags.py
        module_selection.py
      cleaning/
        __init__.py
        standardize.py
        missing_values.py
      scenarios/
        __init__.py
        scenario.py
        transforms.py
      stress/
        __init__.py
        base.py
        formula_selection.py
        cre.py
        ci.py
        consumer.py
      aggregation/
        __init__.py
        asset_quality.py
        sector_reports.py
        tag_tie_outs.py
        outcome_ranges.py
      attribution/
        __init__.py
        run_compare.py
        scenario_attribution.py
        data_attribution.py
      reporting/
        __init__.py
        excel_reports.py
        charts.py
      audit/
        __init__.py
        run_log.py
        lineage.py
  tests/
    test_validation.py
    test_tagging.py
    test_cre_stress.py
    test_ci_stress.py
    test_consumer_stress.py
    test_aggregation.py
```

For a smaller first version, this can be collapsed into fewer files, but the logical boundaries should remain.

## 4. Core Inputs

### 4.1 Scenario Variables

Scenario files should be stored as JSON to keep them readable, deterministic, and easy to version control.

Required scenario categories:

- CRE sector capitalization rates
- 5-year Treasury rate
- CRE sector interest rate spreads
- C&I loan interest rate shocks
- CRE sector amortization schedules
- CRE sector NOI shocks
- C&I EBITDA shocks
- C&I formula-specific shock variables
- Consumer EL shocks
- Consumer collateral or value loss shocks
- CRE sector capitalization rate shocks
- Near-term and longer-term formula settings for each module
- Scenario input ranges for sensitivity or range runs

Example shape:

```json
{
  "scenario_id": "severe_2026_v1",
  "as_of_date": "2026-06-30",
  "treasury_5y_rate": 0.0425,
  "cre": {
    "multifamily": {
      "base_cap_rate": 0.055,
      "cap_rate_shock": 0.015,
      "interest_spread": 0.025,
      "noi_shock": -0.12,
      "amortization_years": 30
    },
    "office": {
      "base_cap_rate": 0.075,
      "cap_rate_shock": 0.025,
      "interest_spread": 0.035,
      "noi_shock": -0.25,
      "amortization_years": 25
    }
  },
  "ci": {
    "formula_1": {
      "interest_rate_shock": 0.02,
      "ebitda_shock": -0.15
    },
    "formula_2": {
      "interest_rate_shock": 0.015,
      "ebitda_shock": -0.10,
      "revenue_shock": -0.08
    },
    "formula_3": {
      "interest_rate_shock": 0.025,
      "ebitda_shock": -0.20,
      "liquidity_haircut": -0.15
    }
  },
  "consumer": {
    "el_multiplier": 1.35,
    "value_loss_shock": -0.10
  }
}
```

Scenario files may also define ranges instead of a single point estimate for selected variables. The range runner should expand these definitions into deterministic scenario cases.

Example range shape:

```json
{
  "scenario_id": "cre_range_2026_v1",
  "as_of_date": "2026-06-30",
  "range_mode": "grid",
  "variables": {
    "treasury_5y_rate": [0.035, 0.045, 0.055],
    "cre.office.cap_rate_shock": [0.015, 0.025, 0.035],
    "cre.office.noi_shock": [-0.15, -0.25, -0.35]
  }
}
```

Range modes:

- `grid`: run every configured combination.
- `paired`: run values by position, such as low, base, and high cases.
- `one_way`: vary one input at a time from a base scenario.

### 4.2 Input Data Files

The first version should support CSV and Excel workbooks through `pandas`.

Required datasets:

- Loan identity data
- CRE collateral-level data
- C&I borrower financial data
- FICO-to-PD transformation table

Recommended canonical input files:

```text
loan_identity.csv
cre_collateral.csv
ci_financials.csv
fico_pd_table.csv
```

## 5. Data Contracts

The data contracts define canonical source fields and expected meanings. Not every listed analytical field must be populated for every loan. Loan-level required fields should be determined dynamically after tags, scope, selected module, selected formula, and maturity treatment are known.

### 5.1 Loan Identity Data

Canonical fields:

| Field | Description |
| --- | --- |
| `loan_id` | Unique loan identifier |
| `borrower_id` | Borrower or relationship identifier |
| `portfolio` | CRE, C&I, Consumer, or other portfolio label |
| `product_type` | Loan product type |
| `sector` | Sector or industry classification |
| `balance` | Current outstanding balance |
| `commitment` | Total committed exposure |
| `interest_rate` | Current contractual interest rate |
| `maturity_date` | Loan maturity date |
| `risk_rating` | Internal risk rating, if available |
| `fico` | Consumer FICO score, if applicable |
| `current_el_rate` | Current expected loss rate, if available |

### 5.2 CRE Collateral Data

Canonical fields:

| Field | Description |
| --- | --- |
| `loan_id` | Loan identifier |
| `collateral_id` | Collateral identifier |
| `cre_sector` | Office, retail, industrial, multifamily, hotel, etc. |
| `noi` | Net operating income |
| `appraised_value` | Current collateral value |
| `occupancy_rate` | Current occupancy |
| `debt_service` | Current annual debt service |
| `ltv` | Current loan-to-value, if available |
| `dscr` | Current debt service coverage ratio, if available |

### 5.3 C&I Financial Data

Canonical fields:

| Field | Description |
| --- | --- |
| `loan_id` | Loan identifier |
| `borrower_id` | Borrower identifier |
| `industry` | Borrower industry |
| `revenue` | Current revenue |
| `ebitda` | Current EBITDA |
| `debt_service` | Annual debt service |
| `total_debt` | Total borrower debt |
| `cash` | Cash and equivalents |
| `interest_expense` | Current interest expense |

### 5.4 FICO-to-PD Table

Canonical fields:

| Field | Description |
| --- | --- |
| `fico_min` | Lower bound of FICO band |
| `fico_max` | Upper bound of FICO band |
| `base_pd` | Base probability of default |

## 6. Run Flow

The deterministic engine should follow this sequence for a single point-in-time scenario:

1. Load configuration and scenario file.
2. Load raw input data.
3. Validate schemas and required fields.
4. Standardize data types, dates, numeric fields, and categorical labels.
5. Assign all deterministic tags.
6. Join collateral, financial, and FICO transformation data.
7. Select exactly one primary stress module for each in-scope loan.
8. Select the applicable formula path within that module, including near-term or longer-term treatment.
9. Dynamically validate the fields required by the selected module, tags, and formula path.
10. Log out-of-scope loans with reason codes.
11. Apply stress to in-scope loans.
12. Calculate stressed loan-level metrics.
13. Aggregate results by portfolio, sector, selected stress module, and risk bands.
14. Produce reports, charts, validation logs, and run metadata.

For a range run, the engine should first expand the range scenario into deterministic child scenarios, run the single-scenario flow for each child scenario, and then aggregate the child outputs into outcome ranges.

## 7. Tagging Logic

Tagging should be rules-based and deterministic. Tags are attributes used for scope, module selection, scenario variable selection, formula selection, and validation rule selection.

A loan may have many tags, but only one primary stress module should be applied to the loan. The selected module should be stored separately from the full tag list.

Suggested tags:

- `eligible_cre`: candidate for CRE collateral stress
- `eligible_ci`: candidate for C&I borrower financial stress
- `eligible_consumer`: candidate for consumer EL and value stress
- `has_cre_collateral`
- `has_ci_financials`
- `has_fico_pd`
- `in_scope`
- `out_of_scope`
- `near_term_maturity`
- `longer_term_maturity`
- `ci_formula_1`
- `ci_formula_2`
- `ci_formula_3`
- `missing_required_data`
- `manual_review_required`

Example rule hierarchy:

1. Assign all descriptive tags based on portfolio, product type, sector, collateral, borrower financials, FICO, maturity date, and data availability.
2. Assign `near_term_maturity` when the loan is within one year of maturity as of the scenario date; otherwise assign `longer_term_maturity`.
3. Determine whether the loan is in scope based on configured inclusion and exclusion tags.
4. Select exactly one primary module from the eligible module tags using a configured priority order and override rules.
5. Select the applicable formula path inside the primary module based on tags and configuration.
6. Run dynamic validation for only the fields required by the selected module, formula, and maturity treatment.
7. If validation fails, mark the loan `out_of_scope` and log the specific reason codes instead of applying stress.

Module selection should be explicit and auditable. For example, a loan may carry `has_cre_collateral`, `eligible_ci`, `near_term_maturity`, and `ci_formula_2`, while its selected module is still only `ci`.

## 8. Cleaning and Validation

Validation should have two layers:

1. File-level validation for inputs that are required to run the engine at all.
2. Loan-level dynamic validation based on the selected module, applicable tags, selected formula, and near-term or longer-term maturity treatment.

Loan-level validation failures should not stop the full run. Instead, failed loans should be marked out of scope, assigned reason codes, and included in the data quality report.

Blocking errors:

- Missing required file
- Missing file-level required column
- Duplicate `loan_id` in loan identity data
- Invalid scenario structure
- Invalid FICO ranges
- Scenario values outside configured bounds

Loan-level out-of-scope reason codes:

- Missing field required by selected module
- Missing field required by selected formula path
- Missing field required by near-term maturity formula
- Missing field required by longer-term maturity formula
- Negative or zero value where the applicable formula requires a positive value
- Unmapped sector
- Missing collateral for CRE-tagged exposure
- Missing C&I financial data for C&I-tagged exposure
- FICO score outside transformation table bands
- No eligible stress module selected

Warnings:

- Missing optional field
- Null risk rating
- Non-critical field ignored because it is not required by the selected formula

Cleaning rules should be centralized and documented. Examples:

- Trim whitespace in identifiers and categorical fields.
- Convert dates to ISO format.
- Convert percentage-like fields to decimal format.
- Standardize CRE sector names.
- Standardize portfolio names.
- Impute only when an explicit fallback rule exists in configuration.

Dynamic validation should be configured as a rule matrix. Example:

| Module | Formula | Maturity tag | Required fields |
| --- | --- | --- | --- |
| `cre` | `standard` | `near_term_maturity` | `loan_id`, `balance`, `maturity_date`, `cre_sector`, `noi`, `appraised_value` |
| `cre` | `standard` | `longer_term_maturity` | `loan_id`, `balance`, `maturity_date`, `cre_sector`, `noi`, `debt_service`, `amortization_years` |
| `ci` | `formula_1` | `near_term_maturity` | `loan_id`, `balance`, `maturity_date`, `ebitda`, `total_debt` |
| `ci` | `formula_2` | `longer_term_maturity` | `loan_id`, `balance`, `maturity_date`, `revenue`, `ebitda`, `debt_service` |
| `consumer` | `standard` | `all` | `loan_id`, `balance`, `fico` |

## 9. Stress Modules

All stress modules should expose a common interface. The engine should call only one module per in-scope loan.

```python
class StressModule:
    module_name = "base"

    def select_formula(self, loan_row, tags, scenario):
        raise NotImplementedError

    def required_fields(self, formula_name, tags, scenario):
        raise NotImplementedError

    def apply(self, loan_row, formula_name, scenario):
        raise NotImplementedError
```

The implementation may use pandas DataFrames for batch execution, but each module should have clear input fields, output fields, formula variants, maturity treatment, and validation requirements.

Every module should support:

- A near-term formula path for loans within one year of maturity.
- A longer-term formula path for loans outside one year of maturity.
- A required-field definition for each formula path.
- A deterministic reason code when the formula cannot be applied.

### 9.1 CRE Stress Module

Inputs:

- Loan balance
- CRE sector
- NOI
- Appraised value
- Debt service
- Current LTV
- Current DSCR
- Scenario cap rate
- Scenario cap rate shock
- Scenario NOI shock
- 5-year Treasury rate
- Sector spread
- Amortization schedule

Core calculations:

The CRE module should have at least two formula paths:

- Near-term maturity: applies the configured within-one-year formula.
- Longer-term maturity: applies the configured outside-one-year formula.

Illustrative longer-term calculations:

- `stressed_noi = noi * (1 + noi_shock)`
- `stressed_cap_rate = base_cap_rate + cap_rate_shock`
- `stressed_value = stressed_noi / stressed_cap_rate`
- `stressed_ltv = balance / stressed_value`
- `stressed_interest_rate = treasury_5y_rate + interest_spread`
- `stressed_debt_service = amortized_payment(balance, stressed_interest_rate, amortization_years)`
- `stressed_dscr = stressed_noi / stressed_debt_service`
The primary CRE output is stressed DSCR. Supporting metrics such as stressed value and stressed LTV should be retained for explanation.

Illustrative near-term calculations may place greater weight on refinance risk, current collateral value, stressed cap rate, and maturity exposure. The exact formula should be configured and documented separately from the longer-term amortization-based path.

Outputs:

- `stressed_noi`
- `stressed_value`
- `stressed_ltv`
- `stressed_dscr`
- `base_dscr`
- `stressed_debt_service`
- `dscr_change`
- `cre_stress_flag`

### 9.2 C&I Stress Module

Inputs:

- Balance
- EBITDA
- Revenue
- Debt service
- Total debt
- Interest expense
- Scenario EBITDA shock
- Scenario interest rate shock
- Formula-specific variables

Core calculations:

The C&I module should support three formula paths, each with modified required variables and calculations:

- `formula_1`: baseline EBITDA and leverage stress.
- `formula_2`: revenue-sensitive EBITDA and coverage stress.
- `formula_3`: liquidity-sensitive stress with cash or liquidity haircut variables.

Each formula should also have near-term and longer-term treatment. Near-term C&I loans may emphasize maturity/refinance pressure and current liquidity, while longer-term loans may emphasize sustained EBITDA, coverage, leverage, and interest-rate stress.

Illustrative formula 1 calculations:

- `stressed_ebitda = ebitda * (1 + ebitda_shock)`
- `stressed_interest_expense = interest_expense + total_debt * interest_rate_shock`
- `stressed_debt_service = debt_service + total_debt * interest_rate_shock`
- `stressed_debt_to_ebitda = total_debt / stressed_ebitda`
- `stressed_fixed_charge_coverage = stressed_ebitda / stressed_debt_service`
The primary C&I output is stressed fixed charge coverage ratio, or FCCR. Supporting metrics such as leverage, stressed EBITDA, interest expense, and liquidity gap should be retained for explanation.

Outputs:

- `stressed_ebitda`
- `stressed_interest_expense`
- `stressed_debt_to_ebitda`
- `stressed_fixed_charge_coverage`
- `selected_ci_formula`
- `maturity_formula`
- `base_fixed_charge_coverage`
- `fixed_charge_coverage_change`
- `liquidity_gap`
- `ci_stress_flag`

### 9.3 Consumer Stress Module

Inputs:

- Balance
- FICO
- Base PD from transformation table
- Current EL rate
- Scenario EL multiplier
- Scenario value loss shock

Core calculations:

The consumer module should also support near-term and longer-term formula paths when product type, maturity date, or remaining term changes the applicable stress treatment.

- Map FICO to base PD.
- Convert base PD to a base EL rate using configured consumer LGD, unless a current EL rate is supplied directly.
- Apply scenario EL multiplier.
- Apply value loss shock to consumer EL rate when relevant.
- Calculate stressed expected loss.

Outputs:

- `base_pd_from_fico`
- `base_el_rate`
- `stressed_el_rate`
- `stressed_expected_loss`
- `consumer_stress_flag`

## 10. Loan-Level Output Schema

The stressed loan-level output should include:

| Field | Description |
| --- | --- |
| `run_id` | Unique deterministic run identifier |
| `scenario_id` | Scenario identifier |
| `loan_id` | Loan identifier |
| `borrower_id` | Borrower identifier |
| `portfolio` | Portfolio |
| `sector` | Sector |
| `tags` | Pipe-delimited deterministic tags |
| `selected_stress_module` | The single module applied to the loan |
| `selected_formula` | Formula selected within the module |
| `maturity_formula` | Near-term or longer-term treatment |
| `scope_status` | In scope or out of scope |
| `out_of_scope_reasons` | Reason codes for loans excluded from stress |
| `balance` | Current balance |
| `base_dscr` | Base DSCR for CRE loans |
| `stressed_dscr` | Stressed DSCR for CRE loans |
| `base_fixed_charge_coverage` | Base FCCR for C&I loans |
| `stressed_fixed_charge_coverage` | Stressed FCCR for C&I loans |
| `base_el_rate` | Base expected loss rate for consumer loans |
| `base_expected_loss` | Base EL |
| `stressed_el_rate` | Stressed expected loss rate for consumer loans |
| `stressed_expected_loss` | Stressed EL |
| `expected_loss_change` | Stressed EL minus base EL |
| `data_quality_flags` | Validation, cleaning, and scope flags |

Module-specific fields should also be retained for auditability.

## 11. Aggregation and Reporting

Required aggregation views:

- Total portfolio
- Portfolio by sector
- Portfolio by selected stress module
- CRE by property sector
- C&I by industry
- Consumer by FICO band
- Risk rating migration, if current and stressed ratings are implemented
- Top contributors to expected loss increase
- Data quality exception summary
- Out-of-scope loans by reason code
- Tag populations compared to external source targets
- Scenario range summaries, when applicable
- Attribution summaries comparing two runs, when applicable

Required metrics:

- Exposure count
- Total balance
- Weighted average base DSCR for CRE
- Weighted average stressed DSCR for CRE
- Weighted average base FCCR for C&I
- Weighted average stressed FCCR for C&I
- Base expected loss
- Stressed expected loss
- Expected loss change
- Expected loss change as percent of balance
- Share of total stressed expected loss
- External source population
- Engine-calculated tag population
- Tag population difference
- Minimum, maximum, mean, and selected percentile outcomes for range runs
- Attribution amount by tag and attribution source

Recommended report outputs:

```text
reports/
  {run_id}/
    loan_level_results.csv
    portfolio_summary.csv
    sector_summary.csv
    cre_summary.csv
    ci_summary.csv
    consumer_summary.csv
    data_quality_summary.csv
    out_of_scope_summary.csv
    out_of_scope_loans.csv
    tag_population_tie_out.csv
    scenario_range_summary.csv
    run_attribution_summary.csv
    run_metadata.json
    stress_report.xlsx
```

Excel workbook tabs:

- `Run Summary`
- `Portfolio Summary`
- `Sector Summary`
- `CRE Detail`
- `C&I Detail`
- `Consumer Detail`
- `Top EL Contributors`
- `Data Quality`
- `Out of Scope`
- `Tag Tie-Outs`
- `Scenario Ranges`
- `Attribution`
- `Scenario Inputs`

## 12. External Source Tie-Outs

The engine should support external source values that tag populations must tie to. These values may come from regulatory reports, portfolio inventory files, general ledger extracts, management reporting, or other controlled sources.

External source tie-outs should operate at the tag population level, not the individual loan level.

Recommended input file:

```text
external_sources/tag_population_targets.csv
```

Recommended fields:

| Field | Description |
| --- | --- |
| `source_id` | Name or identifier of the external source |
| `as_of_date` | Source as-of date |
| `tag_name` | Tag expected to reconcile |
| `population_metric` | Metric to reconcile, such as count, balance, commitment, or exposure |
| `source_value` | External source value |
| `tolerance` | Acceptable absolute or percentage difference |
| `source_owner` | Owner or producer of the external value |

Engine output should include:

- Engine-calculated value by tag and metric.
- External source value by tag and metric.
- Absolute difference.
- Percentage difference.
- Tolerance.
- Tie-out status.
- Source identifier and as-of date.

Tie-out failures should not automatically stop a run unless configured as blocking. They should be visible in the run summary and reporting workbook.

## 13. Attribution Between Runs

The engine should support attribution comparing two completed runs. Attribution should explain aggregate differences at a tag level, not at a loan level.

Attribution should cover:

- Scenario input changes, such as rate shocks, cap rate shocks, NOI shocks, EBITDA shocks, EL multipliers, and value loss assumptions.
- Data changes, such as balance movement, population movement, sector changes, maturity changes, tag changes, collateral changes, financial statement changes, FICO changes, and out-of-scope changes.

Recommended attribution grain:

- Tag
- Selected stress module
- Selected formula
- Maturity formula
- Portfolio
- Sector
- Scenario variable group
- Data change group

Recommended attribution metrics:

- Balance change
- Stressed expected loss change
- Weighted average stressed DSCR change
- Weighted average stressed FCCR change
- Expected loss change
- In-scope population change
- Out-of-scope population change

The first version can use deterministic bridge attribution:

1. Summarize both runs to the same tag-level grain.
2. Separate population and balance changes from risk metric changes.
3. Attribute scenario changes by re-running or recalculating aggregate impacts using one scenario variable group changed at a time.
4. Attribute data changes by comparing grouped input populations and balances between runs.
5. Store any unexplained residual as `interaction_or_residual`.

Recommended output files:

```text
reports/
  attribution/
    {base_run_id}_to_{comparison_run_id}/
      attribution_summary.csv
      scenario_attribution.csv
      data_attribution.csv
      residual_attribution.csv
      attribution_metadata.json
```

Attribution metadata should include:

- Base run ID.
- Comparison run ID.
- Base and comparison scenario IDs.
- Base and comparison input file hashes.
- Attribution method.
- Attribution grain.
- Any variable groups included or excluded.

## 14. Scenario Range Runs

The engine should support running a deterministic range of scenario inputs and aggregating the resulting range of outcomes.

Range runs should be useful for sensitivity analysis, management overlays, and uncertainty bands. They should not rely on random sampling in the first version.

Supported range modes:

- `grid`: all combinations of selected inputs.
- `paired`: matching low/base/high sets.
- `one_way`: one variable changes at a time while others remain at base.

Recommended range outputs:

- One child run folder per expanded scenario case.
- A parent range-run metadata file.
- Aggregate range summaries by portfolio, sector, tag, selected module, formula, and maturity treatment.
- Minimum, maximum, mean, median, and selected percentile outcomes.
- Scenario case producing minimum and maximum stressed expected loss.

Recommended output files:

```text
reports/
  ranges/
    {range_run_id}/
      range_metadata.json
      range_case_index.csv
      range_portfolio_summary.csv
      range_sector_summary.csv
      range_tag_summary.csv
      range_outcome_distribution.csv
```

Range summary metrics:

- Minimum stressed expected loss.
- Maximum stressed expected loss.
- Mean stressed expected loss.
- Median stressed expected loss.
- 10th and 90th percentile stressed expected loss.
- Minimum and maximum weighted average stressed DSCR.
- Minimum and maximum weighted average stressed FCCR.
- Minimum and maximum expected loss change.

The range runner should preserve determinism by writing the exact child scenario file for each expanded case.

## 15. Deterministic Run Metadata

Each run should write a metadata file containing:

- `run_id`
- `scenario_id`
- `as_of_date`
- Input file names
- Input file hashes
- Scenario file hash
- Engine version
- Timestamp
- Row counts by input file
- Validation error count
- Validation warning count
- Out-of-scope loan count
- Out-of-scope balance
- Out-of-scope reason-code summary
- External source tie-out file hashes
- External source tie-out status
- Parent range run ID, if applicable
- Child scenario case ID, if applicable

The `run_id` can be generated as a hash of:

- Scenario file contents
- Input file hashes
- Engine version
- As-of date

This makes repeated identical runs easy to identify.

Range run metadata should include:

- `range_run_id`
- Parent range scenario file hash
- Expanded child scenario hashes
- Range expansion mode
- Variable names and values included
- Number of child runs
- Child run IDs

Attribution metadata should include the base and comparison run IDs, run hashes, scenario hashes, input hashes, attribution method, and attribution grain.

## 16. Configuration

Use JSON for configuration.

Configuration should include:

- Required input file names
- Required field lists
- Standard sector mappings
- Portfolio mappings
- Scope inclusion and exclusion tags
- Primary module selection priority and overrides
- Formula selection rules
- Near-term maturity threshold, defaulting to 365 days
- Dynamic validation rule matrix
- Validation thresholds by module, formula, and tag
- Fallback assumptions
- Consumer EL rate floors and caps
- External source tie-out requirements and tolerances
- Attribution grouping rules
- Scenario variable attribution groups
- Range run limits and expansion mode defaults
- Reporting options

Example:

```json
{
  "engine_version": "0.1.0",
  "primary_module_priority": ["cre", "ci", "consumer"],
  "near_term_maturity_days": 365,
  "el_rate_floor": 0.0,
  "el_rate_cap": 1.0,
  "allow_fallbacks": false,
  "external_tie_outs_required": true,
  "range_run_max_cases": 250
}
```

## 17. Testing Strategy

Use `unittest` from the Python standard library.

Minimum tests:

- Required columns are enforced.
- Duplicate loan IDs are rejected.
- Loans can carry multiple tags while receiving only one selected stress module.
- Dynamic validation requires different fields by module, tag, formula, and maturity treatment.
- Validation failures mark loans out of scope and write reason codes without stopping the run.
- FICO-to-PD mapping handles exact bounds correctly.
- CRE stressed value calculation is deterministic.
- CRE near-term and longer-term formula paths select correctly.
- CRE DSCR and LTV stress calculations handle zero and missing values safely.
- C&I formula 1, formula 2, and formula 3 select and calculate independently.
- C&I near-term and longer-term formula paths select correctly.
- C&I EBITDA shock handles positive, zero, and negative EBITDA.
- Consumer EL multiplier respects EL rate caps.
- Aggregations reconcile to loan-level totals.
- Tag population tie-outs calculate differences and tolerance status correctly.
- Attribution compares two runs at tag level without requiring loan-level output in the attribution report.
- Scenario attribution separates configured scenario variable groups.
- Data attribution separates balance, population, and tag changes.
- Range scenarios expand deterministically.
- Range outputs reconcile to child run outputs.
- Run ID is stable for unchanged inputs.

Test fixtures should be small CSV files checked into the repository.

## 18. Command-Line Interface

The first production interface should be a command-line runner.

Example:

```bash
python -m stress_engine.cli \
  --config config/base_config.json \
  --scenario scenarios/example_severe_case.json \
  --input-dir data/raw \
  --output-dir reports
```

Range run example:

```bash
python -m stress_engine.cli range \
  --config config/base_config.json \
  --range-scenario scenarios/example_range_case.json \
  --input-dir data/raw \
  --output-dir reports/ranges
```

Attribution example:

```bash
python -m stress_engine.cli attribute \
  --base-run reports/base_run_id/run_metadata.json \
  --comparison-run reports/comparison_run_id/run_metadata.json \
  --output-dir reports/attribution
```

CLI responsibilities:

- Parse paths.
- Start logging.
- Call the run orchestrator.
- Print final run status.
- Return non-zero exit code on file-level blocking validation errors.
- Complete the run when loan-level validation exclusions are logged successfully.
- Support single-scenario runs, range runs, and attribution runs.

## 19. Implementation Phases

### Phase 1: Skeleton and Data Contracts

- Create repository structure.
- Define canonical input schemas.
- Implement file loading.
- Implement schema validation.
- Add example scenario and input files.
- Add external source tie-out input contract.
- Add baseline unit tests.

### Phase 2: Cleaning and Tagging

- Implement standardization functions.
- Implement deterministic tagging rules.
- Implement scope, primary module selection, and formula selection.
- Add data quality flags.
- Implement tag population summaries.
- Implement external source tie-outs.
- Produce interim cleaned datasets.

### Phase 3: Stress Modules

- Implement CRE stress module.
- Implement C&I stress module with three formula paths.
- Implement consumer stress module.
- Implement near-term and longer-term treatment for each module.
- Implement dynamic required-field checks by module, tag, formula, and maturity treatment.
- Add module-level tests.
- Produce loan-level stressed outputs.

### Phase 4: Aggregation and Reports

- Implement aggregation tables.
- Implement Excel workbook output.
- Add charts using `matplotlib` if useful.
- Add run metadata and data quality summaries.
- Add tag population tie-out reports.

### Phase 5: Scenario Ranges and Attribution

- Implement deterministic scenario range expansion.
- Implement parent range-run metadata.
- Implement range aggregation outputs.
- Implement two-run attribution at tag level.
- Implement scenario input attribution groups.
- Implement data change attribution groups.

### Phase 6: Audit and Hardening

- Add input file hashing.
- Add deterministic run ID.
- Improve error messages.
- Add out-of-scope reason-code summaries.
- Add external source tie-out status to run summary.
- Add attribution metadata and residual checks.
- Add reconciliation checks.
- Add final documentation and usage examples.

## 20. Key Open Decisions

- Whether consumer base EL is supplied directly, inferred from FICO, or both.
- Whether future versions should convert DSCR/FCCR changes into rating grades or leave them as primary stressed metrics.
- Exact C&I formula 1, formula 2, and formula 3 definitions.
- Exact near-term versus longer-term formula definitions by module.
- Whether sector-level stress assumptions are regulatory, management-defined, or model-derived.
- Whether reporting should be CSV-only, Excel-first, or both.
- Whether manual overrides are allowed and, if so, how they are logged.
- Which external sources should be authoritative for each tag population.
- Whether external source tie-out failures are blocking or warning-only.
- Which attribution method should be used for interaction effects and residuals.
- Which scenario range mode should be the default for management reporting.
- Maximum allowed number of child scenarios in a range run.

## 21. Recommended First Milestone

The first milestone should produce a working deterministic vertical slice:

- Load one scenario JSON file.
- Load four CSV inputs.
- Validate file-level required columns.
- Assign multiple loan tags.
- Select one primary module and one formula path per in-scope loan.
- Log out-of-scope loans with deterministic reason codes.
- Apply simple but documented formulas.
- Produce one loan-level result CSV.
- Produce one sector summary CSV.
- Produce one tag population tie-out CSV.
- Produce one run metadata JSON file.
- Pass a small unit test suite.

This milestone keeps the project small enough to validate quickly while establishing the architecture needed for a full production-grade stress engine.
