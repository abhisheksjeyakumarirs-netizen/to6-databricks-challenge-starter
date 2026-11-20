# Case-selection dataset submission

## Inputs and outputs

**Inputs** (read-only, supplied by the training team, catalog `training.to6_challenge3_inputs`):

- `training.to6_challenge3_inputs.returns_initial` — initial synthetic delivery; one row = one
  return. **Rejected** (see Checks and decisions).
- `training.to6_challenge3_inputs.returns_corrected` — corrected synthetic delivery; one row = one
  return. **Accepted**; the source for the feature table.
- `training.to6_challenge3_inputs.source_lineage` — supplied lineage view; it maps both delivered
  tables to the same upstream table, `training.to6_challenge3_inputs.parsed_source`.

**Row meaning:** one row represents one tax return. `return_id` identifies the return and is the
uniqueness identifier; `taxpayer_id` identifies the filer (a filer may have returns in several
years). Required fields: `return_id`, `tax_year`, `income`, `deduction`. Amounts are nonnegative;
zero income is valid.

**Ratio definition:** `deduction_to_income = deduction / income`. A zero-income return is kept in
the output with a **null** ratio.

**Outputs** (personal output schema `training.to6_c3_9acf04df5c41`):

- `return_features` — exactly `return_id`, `tax_year`, `deduction_to_income`; one row per accepted
  return (12 rows). The filer identifier and raw amounts are excluded (need-to-know rule).
- `features_train` (tax_year 2022), `features_validation` (2023), `features_test` (2024) — 4 rows
  each, every accepted return in exactly one table in its assigned year.
- `initial_rejected_records` — the 3 affected initial-batch rows preserved with their rejection
  reasons.

## Checks and decisions

- **Source references:** Catalog comments on both input tables plus the supplied `source_lineage`
  view (upstream `parsed_source` for both deliveries); recorded in the notebook's Section 1.
- **Initial rejection:** `returns_initial` failed the completeness rule (`R02` missing required
  value: `income`) and the uniqueness rule (`R01` delivered twice). The whole batch was rejected
  **before any feature output was written** — no zero-filling of unknown amounts and no silent
  record drops. The 3 affected rows (both `R01` copies and `R02`) and their reasons are preserved
  in `training.to6_c3_9acf04df5c41.initial_rejected_records` and exported to
  `outputs/initial_rejected_records.csv`.
- **Replacement checks:** `returns_corrected` passed the identical rules: 12 rows, 12 distinct
  `return_id`s, no missing required values, no negative amounts (zero-income rows `R03` and `R09`
  are valid records).
- **Independently derived nonzero expected value:** for `R01` (income 100, deduction 20) the
  expected ratio 20/100 = 0.2 was recomputed in Python directly from the source row; the value
  read back from the saved `return_features` matched (0.2).
- **Saved-table readback (including zero income):** `return_features` was re-read from the catalog
  — exact three-column schema (no `taxpayer_id` or raw amounts), 12 rows = 12 distinct
  `return_id`s = every accepted return exactly once; zero-income returns `R03` and `R09` present
  with null `deduction_to_income`. The three split tables were read back: each holds only its
  assigned year, 4 unique returns each, and together cover every accepted return exactly once.

## Two successful runs

- **Run 1:** 2026-11-20T15:40:38 UTC —
  https://dbc-48a16403-f66f.cloud.databricks.com/editor/notebooks/127024866559307
  (interactive notebook run)
- **Run 2:** 2026-11-20T15:42:00 UTC — same notebook URL (interactive notebook run)
- **Comparison evidence:** `outputs/two_run_comparison.json`, derived from
  `outputs/run_evidence.json` (which retains both runs' full snapshots). Both runs produced the
  same table fingerprint
  `955d29b214bdb05600e66b3d893ed08d13000cdbc4888ea577110d384a43e8a3`, computed over every
  `return_id`, `tax_year` and `deduction_to_income` value in `return_features` and all three year
  tables, with identical row counts (12 / 4 / 4 / 4) — identical return IDs and values, no
  appended copies.
- **Year assignment and single-copy checks:** each split table is created with
  `CREATE OR REPLACE TABLE ... WHERE tax_year = <year>` from the saved `return_features`, so
  reruns replace rather than append. The notebook asserts each split holds only its assigned
  year, that the union of the three split tables equals the 12 accepted return IDs with no
  return ID in more than one table, and that no table contains duplicate return IDs.

## Final files and limitations

Exports under `outputs/` (small synthetic CSV/JSON):

- `return_features.csv`, `features_train.csv`, `features_validation.csv`, `features_test.csv` —
  the saved feature and split tables
- `initial_rejected_records.csv` — affected initial-batch records with rejection reasons
- `run_evidence.json`, `two_run_comparison.json` — two-run snapshots and comparison evidence
- `connection-practice.json` — ungraded connection practice file, not a task deliverable

Not verified / limitations:

- No model-leakage guarantee: these checks establish the data contract only. Later modelling must
  add suitable labels and assess which information is available at prediction time.
- The upstream `parsed_source` table was not independently validated (read-only input); the
  lineage view is the training team's supplied source map and does not claim a Catalog UI
  lineage check.
- Amount semantics beyond the stated contract (units, synthetic value generation) are taken from
  the task description and Catalog comments.
