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

## Scenario Shape

The sample scenario in [examples/scenario.json](examples/scenario.json) is the best starting point. Core sections:

`inputs`
: Defines the identity file and any number of source files. Sources support CSV/XLSX, column renames, date/numeric coercion, and deterministic aggregation before merging. Set `"merge": false` for lookup or tie-out files.

`borrower`
: Defines the borrower id, balance field, risk rating field, and loan-level aggregation rules. Only fields explicitly configured as `sum_fields` are summed.

`tags`
: Defines arbitrary include/exclude condition lists, model eligibility, internal-only tags, and optional external tie-outs.

`modules`
: Defines CRE, C&I, and Consumer assumptions, eligible tags, field names, cutoffs, stress levels, and sector/subsector parameters.

`overlays`
: Defines EF/BCC-style overlay portfolios that borrow Special Mention/Substandard growth rates from modeled portfolios.

`cecl`
: Defines the in-place reserve field, optional bucket reserve defaults, and portfolio methods. Consumer defaults to expected-loss output rather than risk-bucket migration.

## Stress Modules

CRE applies the maturity split from the scenario cutoff date. Loans beyond the configured threshold run DSCR stress. Loans within the threshold run refinance DSCR and LTV stress. Missing variables and sanity-check failures are logged only when applicable to that maturity path.

C&I computes stressed fixed charge coverage by sector. Missing fields are treated leniently as zero; loans are out of scope only when stressed available cash flow is zero/missing or debt service is zero/nonpositive.

Consumer selects the latest available FICO and appraisal, maps FICO to PD, computes stressed PD/LGD/EL for every level, and also computes unstressed quantitative EL.

## Output Files

Each run writes:

- `borrower_audit_raw.csv`: borrower-level data after tags and enrichment, before stress.
- `stressed_borrower_results.csv`: borrower-level stress metrics, buckets, and out-of-scope flags.
- `input_summary.csv`: row counts, missing counts, unique counts, and numeric stats for every loaded field.
- `tag_summary.csv`: tag populations, balances, and tie-out results.
- `out_of_scope_detail.csv` and `out_of_scope_summary.csv`: missing or invalid applicable variables.
- `cre_summary.csv`, `ci_summary.csv`, `consumer_summary.csv`: module reporting.
- `migration_summary.csv`: bucket balances by portfolio and stress level, including overlays.
- `cecl_summary.csv`: proforma reserve dollars and ratios by portfolio, bucket, and level.
- `metadata.json`: engine version, scenario hash, input file hashes, package versions, and output hashes.
- `scenario_used.json`: the merged scenario used for the run.

If previous scenarios are provided, `scenario_diff.csv` is also written:

```powershell
python -m stress_engine examples/scenario.json --previous-scenario path\to\previous_scenario.json
```

The comparison report reruns the previous scenario, measures current-vs-previous data aggregate differences, and reruns previous assumptions with one changed scenario variable at a time to estimate marginal CECL impact by portfolio and aggregate.

## Tests

```powershell
C:\Users\jacke\anaconda3\condabin\conda.bat run -n base python -m unittest discover -s tests
```

