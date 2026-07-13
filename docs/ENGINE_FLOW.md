# Stress Engine Execution Flow

This document is the master map for how the engine moves from JSON inputs to final reports. It focuses on formulas, data connections, and the purpose of each major function.

## Entry Points

`python -m stress_engine ...`
: Runs [stress_engine/cli.py](../stress_engine/cli.py). The CLI loads one or more scenario JSON files, optionally adds previous-scenario comparison paths, and calls `StressEngine.run`.

`run_scenario(...)`
: Convenience helper in [stress_engine/engine.py](../stress_engine/engine.py). It loads scenario JSON and executes the same `StressEngine.run` path used by the CLI.

`StressEngine.run(...)`
: The orchestrator. It is intentionally the only place where the whole workflow is sequenced.

`run_batch_scenarios(...)`
: Optional wrapper in [stress_engine/batch.py](../stress_engine/batch.py). It expands `scenario_batch` variables into child scenarios, runs each child through `StressEngine.run`, and writes batch-level summaries.

## Top-Level Pipeline

1. `load_inputs`
   - File: [stress_engine/io.py](../stress_engine/io.py)
   - Called by: `StressEngine.run`
   - Purpose: load the identity CSV/XLSX plus arbitrary source tables.
   - Output: `LoadedTable` objects containing DataFrames, file hashes, and input profiles.

2. `build_borrowers`
   - File: [stress_engine/borrower.py](../stress_engine/borrower.py)
   - Called by: `StressEngine.run`
   - Purpose: collapse loan-level identity rows to one borrower row.
   - Key rule: only configured `sum_fields` are summed. Everything else uses explicit or default aggregation.

3. `apply_tags`
   - File: [stress_engine/tagging.py](../stress_engine/tagging.py)
   - Called by: `StressEngine.run`
   - Purpose: evaluate JSON include/exclude rules, create tag columns, perform tag tie-outs, and apply `assign` fields.
   - Example derived fields: `cre_subsector`, `ci_sector`, `model_portfolio`, `model_module`.

4. `apply_module_priority`
   - File: [stress_engine/tagging.py](../stress_engine/tagging.py)
   - Called by: `StressEngine.run`
   - Purpose: resolve one `primary_module` when multiple model tags are active.
   - Example: a borrower tagged as both `CRE_Model` and `CI_Model` can still run only CRE when `module_priority` puts CRE first.
   - Also derives `cecl_portfolio`.

5. `enrich_borrowers`
   - File: [stress_engine/borrower.py](../stress_engine/borrower.py)
   - Called by: `StressEngine.run`
   - Purpose: aggregate borrower/collateral/lookup source tables and merge them into borrowers.
   - Guardrail: sources aggregate before merge to prevent many-to-many duplication.

6. `initialize_results`
   - File: [stress_engine/modules/base.py](../stress_engine/modules/base.py)
   - Called by: `StressEngine.run`
   - Purpose: create base bucket and stress-level bucket columns.
   - Base bucket rule: rating `< 7` is Pass, `7` is Special Mention, `> 7` is Substandard.

7. Stress modules
   - Files: [cre.py](../stress_engine/modules/cre.py), [ci.py](../stress_engine/modules/ci.py), [consumer.py](../stress_engine/modules/consumer.py)
   - Called by: `StressEngine.run` in `module_order`.
   - Shared selector: `module_population` enforces active model tags and `primary_module`.

8. `build_reports`
   - File: [stress_engine/reporting.py](../stress_engine/reporting.py)
   - Called by: `StressEngine.run`
   - Purpose: build migration, overlay, CECL, module, out-of-scope, input, tag, and exception reports.

9. Optional comparison
   - File: [stress_engine/comparison.py](../stress_engine/comparison.py)
   - Called by: `StressEngine.run` after current reports are built.
   - Purpose: rerun previous scenarios and changed variables to estimate CECL impact.

10. Output writing
   - File: [stress_engine/engine.py](../stress_engine/engine.py)
   - Called by: `StressEngine.run` when `write_outputs=True`.
   - Purpose: write CSV/JSON artifacts with stable sorting.

Batch wrapper:
- `expand_batch_scenarios` materializes child scenarios before any engine calculations.
- Each generated scenario is a deep copy of the base scenario with only configured JSON paths changed.
- Each child still flows through the same ten-step `StressEngine.run` pipeline above.
- Batch reports are built after child runs complete.

## Tagging And Routing

Tags are defined in scenario JSON and evaluated by `apply_tags`.

Tag condition structure:
- A list means all conditions must pass.
- `{"all": [...]}` means all child conditions must pass.
- `{"any": [...]}` means at least one child condition must pass.

Common operators:
- `eq`, `ne`, `in`, `not_in`
- `gt`, `gte`, `lt`, `lte`, `between`
- `contains`, `regex`
- `has_token`, `has_any_token`, `has_all_tokens`

Token operators split semicolon, comma, or pipe-delimited fields. In the example, `subsector` and `tag_hint` can drive tags without separate raw `portfolio` or `module` input fields.

`assign` is used to derive fields from tags. Example:

```json
"assign": {
  "model_portfolio": "CRE",
  "model_module": "CRE",
  "cre_subsector": "Retail"
}
```

Module routing:
- Tags can make a borrower eligible for multiple modules.
- `apply_module_priority` writes `eligible_modules` and one `primary_module`.
- Each stress module calls `module_population`, which runs only rows whose `primary_module` matches the module.

CECL routing:
- `cecl.portfolio_field` names the derived CECL grouping field, usually `cecl_portfolio`.
- Module config can set `cecl_portfolio_field` to use a derived field such as `cre_subsector`.
- CRE can set `cecl_portfolio_rollup` to a value like `CRE`; this rolls all CRE subsectors into one reserve-ratio pool.
- If `cecl_portfolio_rollup` is omitted, CRE remains at the subsector tag level.

## Borrower Assembly And Enrichment

Identity file:
- Loan-level.
- Defines the borrower universe.
- Multiple loans per borrower are grouped by `borrower_id_field`.
- Balances and reserves are summed, while rating, maturity, and every tag-driving field are inherited together from the largest loan.
- Conflicting multi-loan attributes are logged as `BORROWER_LOAN_ATTRIBUTE_CONFLICT`.
- Missing borrower IDs receive deterministic placeholders so unrelated rows cannot collapse together.

Aggregation:
- `sum_fields`: summed, usually balances and reserves.
- Other fields: first, max, min, latest, unique sum, or configured default.

Source enrichment:
- `aggregate_source` reduces source tables to one row per borrower key before merging.
- `latest` selects the latest dated row with deterministic source-row tie-breaking.
- `best_available` selects a canonical borrower field from an ordered list of raw candidate columns.
  - Blank values are skipped.
  - Zero values are skipped by default and can be allowed with `treat_zero_as_missing: false`.
  - `selection: "latest"` uses the newest valid candidate date first, then candidate order.
  - `selection: "precedence"` uses candidate order first, then the newest row within that candidate.
  - Optional `date_output`, `source_field_output`, and `candidate_output` columns make the selected date/source auditable in `borrower_audit_raw.csv`.
- `unique_sum` drops duplicate unique keys before summing, useful for collateral/appraisal values.
- `best_available_unique_sum` selects the best candidate separately for each collateral/entity key and then sums the selected values by borrower.
- Lower-priority candidate use is logged as `BEST_AVAILABLE_FALLBACK_USED`.

Source reconciliation:
- `source_reconciliation.csv` reports null/orphan keys, key cardinality, matched and unmatched borrower counts/balances, coercion issues, and conflicting entity/date values.
- Repeated borrower keys are informational unless a source sets `expect_unique_key: true`; this preserves valid collateral and historical source grains.

Example uses:
- `dscr` is derived from current, prior, and origination DSCR raw columns. Current zero/blank values fall back to prior/origination DSCR.
- `fico_score` is derived from current and origination FICO candidates, with `fico_date` and `fico_source_field` retained.
- `current_appraised_value` is derived from current, prior, and origination appraisal candidates, with `appraisal_date` and `appraisal_source_field` retained.

## CRE Formula Flow

File: [stress_engine/modules/cre.py](../stress_engine/modules/cre.py)

`run_cre`
: Called by `StressEngine.run`. Loops through stress levels and CRE rows.

Maturity split:
- `cutoff_date + maturity_threshold_months` determines the near-maturity boundary.
- If maturity is after the threshold, run DSCR test.
- If maturity is within the threshold, run refinance DSCR and LTV tests.

Base migration rule:
- If base bucket is Substandard, the loan stays Substandard.
- Otherwise, final stressed bucket is the worst of base bucket and applicable tests.

DSCR test:

```text
stressed_dscr = current_dscr * (1 - dscr_decline)
```

The result maps through lower-is-worse cutoffs:
- `<= substandard` maps to Substandard.
- `<= special_mention` maps to Special Mention.
- otherwise Pass.

Refinance DSCR:

```text
stressed_noi = noi * (1 - noi_decline)
stressed_rate = treasury_rate + credit_spread
annual_debt_payment = amortizing_payment(balance, stressed_rate, amortization_years)
stressed_dscr = stressed_noi / annual_debt_payment
```

LTV:

```text
stressed_ltv = balance * stressed_cap_rate / noi
```

The LTV result maps through higher-is-worse cutoffs:
- `>= substandard` maps to Substandard.
- `>= special_mention` maps to Special Mention.
- otherwise Pass.

CRE out-of-scope:
- Missing maturity prevents maturity split.
- DSCR path requires DSCR and subsector.
- Refinance/LTV path requires subsector, balance, and NOI.
- DSCR and NOI/balance sanity ranges are scenario controlled.

## C&I Formula Flow

File: [stress_engine/modules/ci.py](../stress_engine/modules/ci.py)

`run_ci`
: Called by `StressEngine.run`. Loops through stress levels and C&I rows.

Substandard rule:
- Base Substandard remains Substandard.

FCCR formula:

```text
stressed_ebitda = ebitda * (1 - ebitda_reduction)

cash_tax_ratio = cash_taxes / ebitda
distribution_ratio = cash_distribution / ebitda

stressed_taxes = cash_tax_ratio * stressed_ebitda
stressed_distribution = distribution_ratio * stressed_ebitda
```

For Middle Market and ABL:

```text
non_discretionary_dividends = cash_dividends - discretionary_cash_dividends_distribution
stressed_non_discretionary_dividends =
    (non_discretionary_dividends / ebitda) * stressed_ebitda
```

Available cash flow:

```text
available_cash_flow =
    stressed_ebitda
    - stressed_taxes
    - stressed_distribution
    - stressed_non_discretionary_dividends
    - cash_management_fees
    - unfinanced_capex
```

Debt service:

```text
debt_service =
    interest_rate_stress * global_total_outstanding
    + cash_paid_for_interest
    + sector_principal_field
```

FCCR:

```text
stressed_fccr = available_cash_flow / debt_service
```

Out-of-scope:
- C&I is intentionally lenient with missing numeric fields.
- Missing numeric fields are treated as zero and logged.
- The loan is out of scope only when available cash flow is zero/missing or debt service is zero/nonpositive.

## Consumer Formula Flow

File: [stress_engine/modules/consumer.py](../stress_engine/modules/consumer.py)

`run_consumer`
: Called by `StressEngine.run`. Writes PD, LGD, and EL metrics rather than commercial bucket migrations.

FICO:
- Use the enriched `fico_score` field, which can be selected by `best_available` from multiple raw FICO columns.
- Lookup base PD from exact score or score band.

Appraisal:
- Use the enriched `current_appraised_value` field, selected per collateral ID and summed across unique collateral.

Unstressed LGD:

```text
unstressed_lgd = max(balance - appraised_value, 0)
unstressed_el = base_pd * unstressed_lgd
```

Stressed values:

```text
stressed_pd = min(base_pd * pd_increase_factor, pd_cap)

stressed_collateral_value =
    appraised_value
    * collateral_value_factor
    * (1 - rushed_sale_discount)
    * (1 - closing_costs)

stressed_lgd = max(balance - stressed_collateral_value, 0)
stressed_el = stressed_pd * stressed_lgd

qualitative_reserve = recorded_base_cecl_reserve - unstressed_el
stressed_consumer_cecl = stressed_el + qualitative_reserve
```

Out-of-scope:
- Missing FICO, missing appraisal, missing balance, or missing PD lookup row.
- Base CECL still uses the recorded reserve; stressed Consumer CECL is unavailable when any positive balance is out of scope.

## Overlay Flow

File: [stress_engine/modules/overlay.py](../stress_engine/modules/overlay.py)

`apply_overlays`
: Called by `build_reports` after migration summary creation.

Purpose:
- EF/BCC-style portfolios may be in scope but lack financials.
- They borrow migration growth patterns from source portfolios.
- Overlay sources can be selected by broad portfolio labels, by tag columns, or by additional conditions.
- New weighted sources use borrower-level stressed results; legacy `source_portfolios: ["CRE", "C&I"]` remains balance-weighted from migration summary rows.

Weighted source configuration example:

```json
"overlays": {
  "BCC": {
    "enabled": true,
    "sources": [
      {"name": "CRE", "tags": ["CRE_Model"], "weight": 0.65},
      {"name": "C&I", "tags": ["CI_Model"], "weight": 0.35}
    ]
  }
}
```

Formula:

```text
weighted_source_ratio =
    sum(source_weight * source_bucket_ratio) / sum(source_weight)

source_growth = weighted_source_stressed_ratio / weighted_source_base_ratio
overlay_stressed_ratio = overlay_base_ratio * source_growth
```

If the source base ratio is zero or unavailable, the configured zero-base behavior is used and logged.

The resulting Special Mention and Substandard ratios are capped so their sum does not exceed 100%; Pass is the remainder.

`overlay_summary.csv` records `source_portfolios`, `source_weights`, `source_selection`, weighted source base/stressed ratios, and final stressed overlay ratios.

## CECL Flow

File: [stress_engine/reporting.py](../stress_engine/reporting.py)

CECL grouping:
- Migration reporting can use broad portfolios.
- CECL reporting uses `cecl_portfolio`, which may be a subsector/sector tag or a module-level rollup.

Reserve ratio derivation:

```text
reserve_ratio = sum(loan_cecl_reserve) / sum(loan_balance)
```

This is derived from loan data only. No external reserve-ratio source is used.

Missing individual loan CECL reserve:
- Treated as zero for aggregate derivation.
- Logged as `CECL_LOAN_RESERVE_MISSING_TREATED_AS_ZERO`.

Bucket CECL reserve:

```text
proforma_cecl_reserve = stressed_bucket_balance * derived_reserve_ratio
```

Portfolio total:

```text
portfolio_proforma_reserve = sum(bucket_proforma_cecl_reserve)
portfolio_proforma_ratio = portfolio_proforma_reserve / total_portfolio_balance
```

Unavailable state:
- If a positive-balance bucket has no derived reserve ratio, the bucket and portfolio total are marked unavailable.
- No fallback reserve ratio is inserted.

Consumer CECL:
- Base uses the recorded loan-level CECL reserve.
- Stressed levels use stressed quantitative EL plus the qualitative reserve derived from Base.
- Missing model inputs produce an unavailable stressed result, never a valid zero.

Missing ratings:
- Commercial and overlay balances remain in an explicit `Unknown` bucket and are logged as `RISK_RATING_MISSING`.
- Unknown balances remain in migration and CECL totals instead of disappearing or being assigned a false bucket.

## Exception Logging

File: [stress_engine/exceptions.py](../stress_engine/exceptions.py)

Exceptions are structured audit rows, not Python exceptions. They are collected during the run and written to `exception_log.csv`.

Examples:
- `CECL_LOAN_RESERVE_MISSING_TREATED_AS_ZERO`
- `CECL_RESERVE_RATIO_UNAVAILABLE`
- `OVERLAY_ZERO_SOURCE_BASE_RATIO`
- `CI_MISSING_FIELD_ZERO_SUBSTITUTION`
- `CI_SECTOR_DEFAULT_USED`
- `BEST_AVAILABLE_FALLBACK_USED`
- `SCENARIO_DEFAULT_PARAMETER_USED`
- `TAG_TIEOUT_DIFFERENCE`
- `SOURCE_ORPHAN_KEYS`

## Comparison Flow

File: [stress_engine/comparison.py](../stress_engine/comparison.py)

`build_comparison_report`
: Called by `StressEngine.run` only when previous scenario paths are configured and comparison is enabled.

Steps:
1. Load and rerun the previous scenario.
2. Compare input aggregate profiles.
3. Find changed JSON variables.
4. Rerun prior scenario with one changed variable at a time.
5. Report marginal CECL reserve and CECL ratio impact by portfolio and aggregate.

Data-profile changes are listed descriptively, while their combined CECL impact is reported once. CECL availability transitions and skipped variable reruns are retained explicitly.

## Batch Scenario Flow

File: [stress_engine/batch.py](../stress_engine/batch.py)

`scenario_batch`
: Optional JSON section used only when the CLI receives `--batch` or code calls `run_batch_scenarios`.

Supported modes:
- `grid`: Cartesian product of all variable values.
- `paired`: first value of each variable together, second value together, and so on.
- `named`: explicit named override sets.

Variable value sources:
- `values`: explicit list.
- `range`: numeric start/stop/step.
- `linspace`: numeric start/stop/count.
- `multipliers`: multiply the base value at the configured JSON path.
- `deltas`: add to the base value at the configured JSON path.

Example:

```json
"scenario_batch": {
  "mode": "grid",
  "variables": [
    {
      "name": "CRE DSCR S1 decline",
      "path": "modules.CRE.tests.dscr.decline.default.S1",
      "range": {"start": 0.03, "stop": 0.07, "step": 0.02}
    },
    {
      "name": "Consumer PD S2 factor",
      "path": "modules.Consumer.pd_increase_factor.S2",
      "values": [1.25, 1.50]
    }
  ]
}
```

Batch outputs:
- `batch_summary.csv`: aggregate CECL reserve and ratio by scenario/stress level, plus deltas from base.
- `batch_variables.csv`: scenario-variable audit rows.
- `batch_cecl_summary.csv`: full CECL summary for each scenario, with variable columns attached.
- `batch_exceptions.csv`: exception log rows across child runs.
- `batch_metadata.json`: batch hash, generated scenario count, scenario files, and report hashes.

Child outputs:
- By default, each child run writes a full engine output folder under `scenarios/<run_id>`.
- CLI `--no-write-child-outputs` writes only batch-level reports, which is useful for larger sensitivity grids.

## Main Outputs

`borrower_audit_raw.csv`
: Borrower-level data after aggregation, tagging, module priority, CECL portfolio assignment, and enrichment.

`stressed_borrower_results.csv`
: Borrower-level stress metrics and stressed buckets.

`tag_summary.csv`
: Tag counts, balances, and tie-outs.

`input_summary.csv`
: Field-level profiling of all loaded input files.

`source_reconciliation.csv`
: Source key cardinality, coverage, unmatched balance, coercion, and entity-value controls.

`out_of_scope_detail.csv`
: Borrower/test/field-level missing or invalid applicable variables.

`exception_log.csv`
: Fallback-like behavior and unavailable reserve states.

`migration_summary.csv`
: Broad portfolio bucket balances and counts, including overlays.

`cecl_summary.csv`
: CECL portfolio bucket reserves and total reserve ratios.

`metadata.json`
: Engine version, scenario hash, input file hashes, output hashes, and exception count.

`scenario_used.json`
: The merged scenario used for the run.

`output_manifest.json`
: Files owned by the current run. Engine-owned files absent from a later run are removed from reused output directories.
