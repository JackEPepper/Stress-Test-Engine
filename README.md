# Credit Stress Engine

Deterministic, JSON-driven stress testing engine for loan portfolios. The engine loads CSV/XLSX inputs, assembles a borrower-level universe, applies configurable tags and tie-outs, enriches borrowers from borrower/collateral sources, runs stress modules, applies overlay portfolios, and produces audit-ready CSV/JSON outputs.

The implementation uses only Python standard library plus common Anaconda packages: `pandas`, `numpy`, and `openpyxl`.

## Quick Start

From the repository root:

```powershell
C:\Users\jacke\anaconda3\condabin\conda.bat run -n base python -m stress_engine examples/scenario.json --output-dir outputs/example --no-comparison
```

Outputs are written relative to the scenario file unless `--output-dir` is absolute. The example command writes to `examples/outputs/example`.

Layer multiple JSON files by passing them in order:

```powershell
python -m stress_engine base_scenario.json q3_overlay.json
```

Later files override earlier files recursively.

Run a batch sensitivity grid by layering a batch JSON and passing `--batch`:

```powershell
C:\Users\jacke\anaconda3\condabin\conda.bat run -n base python -m stress_engine examples/scenario.json examples/scenario_batch.json --batch --batch-output-dir outputs/example_batch --no-comparison --no-write-child-outputs
```

The example batch file generates six child scenarios from two variables and writes batch-level CSVs without writing every child scenario's full output folder.

## Scenario Shape

The sample scenario in [examples/scenario.json](examples/scenario.json) is the best starting point. Core sections:

`inputs`
: Defines the identity file and any number of source files. Sources support CSV/XLSX, column renames, date/numeric coercion, and deterministic aggregation before merging. Set `"merge": false` for lookup or tie-out files.

Source aggregations can derive canonical fields from multiple raw candidates. The example uses `"method": "best_available"` for DSCR and FICO and `"best_available_unique_sum"` for appraisals. Candidates are listed in precedence order, blanks and zeros are skipped by default, and the newest row is used within a candidate. Appraisals are selected independently per collateral ID and then summed. Audit fields record selected columns/dates, while lower-priority selections are logged.

`borrower`
: Defines the borrower id, balance field, risk rating field, and loan-level aggregation rules. Only fields explicitly configured as `sum_fields` are summed. For borrowers with multiple loans, rating, maturity, and tag-driving fields are inherited together from the largest loan; conflicting loan attributes are logged.

`tags`
: Defines arbitrary include/exclude condition lists, model eligibility, internal-only tags, and optional external tie-outs.

Tags can also assign derived fields through `assign`. The example identity file intentionally has no `portfolio`, `module`, `cre_subsector`, or `ci_sector` columns; it carries a single tokenized `subsector` field instead. Subsector tags reconcile to external totals and assign downstream fields such as `cre_subsector`, `ci_sector`, `model_portfolio`, and `model_module`. Tokenized fields can be matched with `has_token`, `has_any_token`, and `has_all_tokens`, using semicolon, comma, or pipe delimiters by default.

When tags make a borrower eligible for more than one module, `module_priority` resolves one `primary_module`. The sample sets `["CRE", "C&I", "Consumer"]`, so a borrower tagged as both CRE and Middle Market C&I is stressed by CRE only. The example also includes a mostly blank `tag_hint` field to demonstrate auxiliary tag flags such as Middle Market without making that field part of the core borrower taxonomy.

`modules`
: Defines CRE, C&I, and Consumer assumptions, eligible tags, field names, cutoffs, stress levels, and sector/subsector parameters.

`overlays`
: Defines EF/BCC-style overlay portfolios that borrow Special Mention/Substandard growth rates from modeled source populations. Sources can be selected by tags and assigned explicit weights; legacy `source_portfolios` lists remain supported as balance-weighted portfolio groups.

`cecl`
: Defines the in-place reserve field, CECL portfolio grouping field, and portfolio methods. Consumer Base CECL always uses the recorded loan reserve. Stressed Consumer CECL equals stressed quantitative EL plus the qualitative reserve implied by Base reserve less unstressed quantitative EL.

`scenario_batch`
: Optional batch sensitivity configuration. Supports `grid`, `paired`, and `named` modes. Variables can provide explicit `values`, numeric `range`, `linspace`, `multipliers`, or `deltas`; each variable points to a JSON path such as `modules.CRE.tests.dscr.decline.default.S1`.

## Stress Modules

CRE applies the maturity split from the scenario cutoff date. Loans beyond the configured threshold run DSCR stress. Loans within the threshold run refinance DSCR and LTV stress. Missing variables and sanity-check failures are logged only when applicable to that maturity path.

C&I computes stressed fixed charge coverage by sector. Missing fields are treated leniently as zero; loans are out of scope only when stressed available cash flow is zero/missing or debt service is zero/nonpositive.

Consumer uses the enriched FICO and collateral-summed appraisal fields, validates FICO lookup bands, maps FICO to PD, and computes stressed PD/LGD/EL for every level. A stressed Consumer portfolio with out-of-scope balance is reported as unavailable rather than zero.

## Output Files

Each run writes:

- `borrower_audit_raw.csv`: borrower-level data after tags and enrichment, before stress.
- `stressed_borrower_results.csv`: borrower-level stress metrics, buckets, and out-of-scope flags.
- `input_summary.csv`: row counts, missing counts, unique counts, and numeric stats for every loaded field.
- `source_reconciliation.csv`: merge-key cardinality, null/orphan keys, borrower coverage, unmatched balances, coercion issues, and conflicting entity values.
- `tag_summary.csv`: tag populations, balances, and tie-out results.
- `out_of_scope_detail.csv` and `out_of_scope_summary.csv`: missing or invalid applicable variables.
- `exception_log.csv`: severe exceptions and fallback behavior used during processing.
- `cre_summary.csv`, `ci_summary.csv`, `consumer_summary.csv`: module reporting.
- `migration_summary.csv`: bucket balances by portfolio and stress level, including overlays.
- `cecl_summary.csv`: proforma reserve dollars and ratios by CECL portfolio, bucket, and level. CECL reserve ratios are always derived from loan data; missing individual loan reserves are treated as zero and logged. Missing commercial ratings remain visible in an `Unknown` bucket. CRE can either roll all subsectors into a broader `CRE` CECL portfolio using `cecl_portfolio_rollup`, or remain separate by omitting that option and using `cecl_portfolio_field`.
- `metadata.json`: engine version, scenario hash, input file hashes, package versions, and output hashes.
- `scenario_used.json`: the merged scenario used for the run.
- `output_manifest.json`: engine-owned files for the run; obsolete files from an earlier run in the same directory are removed safely.

Fallback-style behavior is logged in `exception_log.csv`, including ordered field fallbacks, configured default parameter use, overlay zero-base behavior, C&I missing-field substitutions, module/tag overlaps, and reconciliation discrepancies.

If previous scenarios are provided, `scenario_diff.csv` is also written:

```powershell
python -m stress_engine examples/scenario.json --previous-scenario path\to\previous_scenario.json
```

The comparison report reruns the previous scenario, reports the aggregate effect of all data changes once, lists descriptive profile changes separately, and reruns previous assumptions with one changed scenario variable at a time. Available-to-unavailable CECL transitions and reruns skipped by a configured limit are explicit rows.

Batch runs write:

- `batch_summary.csv`: aggregate CECL totals by generated scenario and stress level, including deltas from the base scenario.
- `batch_variables.csv`: one row per generated scenario variable value.
- `batch_cecl_summary.csv`: all CECL rows from all generated scenarios with variable columns attached.
- `batch_exceptions.csv`: exception logs across generated scenarios.
- `batch_metadata.json`: engine version, input hashes, batch hash, scenario count, per-run scenario/output hashes, variables, exceptions, and report hashes.

## Tests

```powershell
C:\Users\jacke\anaconda3\condabin\conda.bat run -n base python -m unittest discover -s tests
```
