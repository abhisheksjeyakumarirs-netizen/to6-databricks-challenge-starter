# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %sql
# MAGIC Select 1 AS connection_check

# COMMAND ----------

!git remote -v

# COMMAND ----------

# DBTITLE 1,Write connection practice output
from pathlib import Path
import json

repo_root = Path("/Workspace/Users/abhishek.s.jeyakumar.irs@catalyte.ai/to6-databricks-challenge-starter")
assert (repo_root / "README.md").is_file(), "Check the Git folder path"
practice_result = spark.sql("SELECT 1 AS connection_check").first().asDict()
assert practice_result == {"connection_check": 1}
(repo_root / "outputs").mkdir(exist_ok=True)
(repo_root / "outputs/connection-practice.json").write_text(
    json.dumps(practice_result) + "\n", encoding="utf-8"
)

# COMMAND ----------

import hashlib

INPUT_SCHEMA = "training.to6_challenge3_inputs"
INITIAL_TABLE = f"{INPUT_SCHEMA}.returns_initial"
CORRECTED_TABLE = f"{INPUT_SCHEMA}.returns_corrected"
LINEAGE_VIEW = f"{INPUT_SCHEMA}.source_lineage"
workspace_user = spark.sql("SELECT current_user() AS username").first()["username"]
OUTPUT_SCHEMA = "training.to6_c3_" + hashlib.sha256(workspace_user.encode("utf-8")).hexdigest()[:12]
print({"input_schema": INPUT_SCHEMA, "output_schema": OUTPUT_SCHEMA})

# COMMAND ----------

# DBTITLE 1,Section 1 — Understand the input
# MAGIC %md
# MAGIC ## 1. Understand the input
# MAGIC
# MAGIC **Row meaning:** one row represents one tax return. `return_id` identifies the return and is the
# MAGIC uniqueness identifier; `taxpayer_id` identifies the filer (a filer may have returns in several
# MAGIC years). Required fields: `return_id`, `tax_year`, `income`, `deduction`. Amounts are nonnegative;
# MAGIC zero income is valid.
# MAGIC
# MAGIC **Source reference** (Catalog comments on the input tables plus the supplied lineage view
# MAGIC `training.to6_challenge3_inputs.source_lineage`):
# MAGIC
# MAGIC - `training.to6_challenge3_inputs.returns_initial` — Catalog comment: "Initial synthetic delivery
# MAGIC   from parsed_source; known delivery defects. Validate before feature output."
# MAGIC - `training.to6_challenge3_inputs.returns_corrected` — Catalog comment: "Corrected synthetic
# MAGIC   delivery from parsed_source; one row per return. Validate required values and return_id
# MAGIC   uniqueness."
# MAGIC - The lineage view maps **both** delivered tables to the same upstream table:
# MAGIC   `training.to6_challenge3_inputs.parsed_source`.
# MAGIC
# MAGIC The cell below records the lineage view, sample rows from both deliveries, and a full-batch
# MAGIC profile (rows by year, missing required values, duplicate return IDs, zero-income records).

# COMMAND ----------

# DBTITLE 1,Section 1 — Investigation

REQUIRED_FIELDS = ["return_id", "tax_year", "income", "deduction"]

display(spark.table(LINEAGE_VIEW))
display(spark.table(INITIAL_TABLE).limit(5))
display(spark.table(CORRECTED_TABLE).limit(5))


def profile_batch(table_name, label):
    print(f"Profile — {label} ({table_name})")
    display(spark.sql(f"""
        SELECT tax_year,
               COUNT(*) AS rows_in_year,
               COUNT(DISTINCT return_id) AS distinct_return_ids_in_year
        FROM {table_name}
        GROUP BY tax_year
        ORDER BY tax_year
    """))
    display(spark.sql(f"""
        SELECT COUNT(*) AS total_rows,
               COUNT(DISTINCT return_id) AS distinct_return_ids,
               SUM(CASE WHEN return_id IS NULL THEN 1 ELSE 0 END) AS missing_return_id,
               SUM(CASE WHEN tax_year IS NULL THEN 1 ELSE 0 END) AS missing_tax_year,
               SUM(CASE WHEN income IS NULL THEN 1 ELSE 0 END) AS missing_income,
               SUM(CASE WHEN deduction IS NULL THEN 1 ELSE 0 END) AS missing_deduction,
               SUM(CASE WHEN income = 0 THEN 1 ELSE 0 END) AS zero_income_rows,
               SUM(CASE WHEN income < 0 OR deduction < 0 THEN 1 ELSE 0 END) AS negative_amount_rows
        FROM {table_name}
    """))
    display(spark.sql(f"""
        SELECT return_id, COUNT(*) AS occurrences
        FROM {table_name}
        GROUP BY return_id
        HAVING COUNT(*) > 1
        ORDER BY return_id
    """))

profile_batch(INITIAL_TABLE, "initial delivery")
profile_batch(CORRECTED_TABLE, "corrected delivery")

# COMMAND ----------

# DBTITLE 1,Section 1 — Findings
# MAGIC %md
# MAGIC **Findings (both batches) and what they affect.**
# MAGIC
# MAGIC - `returns_initial`: 13 rows but only 12 distinct `return_id`s — `R01` occurs twice. One row
# MAGIC   (`R02`) is missing `income`, a required field. Both violate the stated contract, so the initial
# MAGIC   batch is expected to fail the formal checks in the next section.
# MAGIC - `returns_corrected`: 12 rows, 12 distinct `return_id`s, no missing required values, no negative
# MAGIC   amounts, 4 returns in each of 2022–2024. Two returns (`R03`, `R09`) have zero income — valid
# MAGIC   records that must be kept in the output with a null ratio.
# MAGIC - Consequences for the next steps: the completeness and uniqueness checks in Section 2 must
# MAGIC   reject `returns_initial` before any feature output is written; the ratio computation must keep
# MAGIC   zero-income rows with a null ratio instead of dropping them or dividing by zero; the filer
# MAGIC   identifier and raw amounts must not reach the feature table.

# COMMAND ----------

# DBTITLE 1,Section 2 — Decide which batch is usable
# MAGIC %md
# MAGIC ## 2. Decide which batch is usable
# MAGIC
# MAGIC Checks applied identically to both deliveries (the task's data contract):
# MAGIC
# MAGIC - **Completeness (required-value):** every required field (`return_id`, `tax_year`, `income`,
# MAGIC   `deduction`) is non-null.
# MAGIC - **Uniqueness:** each `return_id` occurs exactly once (one row = one return).
# MAGIC - **Amount sanity** (contract observation): `income` and `deduction` are nonnegative; zero income
# MAGIC   is valid.
# MAGIC
# MAGIC `returns_initial` must be rejected before any feature output is written, with the affected records
# MAGIC and the reason for rejection preserved. `returns_corrected` is assessed with the same rules and,
# MAGIC only if it passes, used for the final run. Removing checks, replacing unknown amounts with zero or
# MAGIC silently discarding records is not an acceptable correction.

# COMMAND ----------

# DBTITLE 1,Section 2 — Checks and decision

from pyspark.sql import functions as F

def affected_records(df):
    """Rows that break the contract, each labelled with its rejection reason(s)."""
    duplicated_ids = (df.groupBy("return_id").count()
                      .where(F.col("count") > 1)
                      .select("return_id", F.lit(True).alias("_is_duplicated")))
    marked = (
        df.join(duplicated_ids, "return_id", "left")
          .withColumn("_missing", F.concat_ws("; ", *[
              F.when(F.col(c).isNull(), F.lit(f"missing required value: {c}"))
              for c in REQUIRED_FIELDS]))
          .withColumn("_duplicate",
              F.when(F.col("_is_duplicated"), F.lit("duplicate return_id (return not exactly once)")))
          .withColumn("_negative",
              F.when((F.col("income") < 0) | (F.col("deduction") < 0),
                     F.lit("negative amount (contract requires nonnegative)")))
          .withColumn("rejection_reasons",
              F.concat_ws("; ",
                  F.when(F.length(F.col("_missing")) > 0, F.col("_missing")),
                  F.when(F.length(F.col("_duplicate")) > 0, F.col("_duplicate")),
                  F.when(F.length(F.col("_negative")) > 0, F.col("_negative"))))
          .where(F.length(F.col("rejection_reasons")) > 0)
          .drop("_is_duplicated", "_missing", "_duplicate", "_negative")
    )
    return marked

def check_batch(table_name):
    """Run the contract checks; return a summary and the affected records."""
    df = spark.table(table_name)
    affected = affected_records(df)
    n_affected = affected.count()
    failed_rules = sorted({
        reason.strip()
        for row in affected.select("rejection_reasons").collect()
        for reason in row["rejection_reasons"].split(";")
        if reason.strip()
    }) if n_affected else []
    summary = {
        "table": table_name,
        "total_rows": df.count(),
        "distinct_return_ids": df.select("return_id").distinct().count(),
        "affected_rows": n_affected,
        "failed_rules": failed_rules,
        "passed": n_affected == 0,
    }
    return summary, affected

initial_summary, initial_affected = check_batch(INITIAL_TABLE)
corrected_summary, _corrected_affected = check_batch(CORRECTED_TABLE)

print("Initial batch check:", initial_summary)
print("Corrected batch check:", corrected_summary)
print("\nAffected records in returns_initial (preserved with reasons):")
display(initial_affected)


assert not initial_summary["passed"], "returns_initial was expected to fail the contract checks"
REJECTED_TABLE = f"{OUTPUT_SCHEMA}.initial_rejected_records"
initial_affected.write.mode("overwrite").saveAsTable(REJECTED_TABLE)
print(f"REJECTED {INITIAL_TABLE} — no feature output is written from it. Affected records and "
      f"reasons preserved in {REJECTED_TABLE} (CSV export added in Section 4).")

# The replacement must pass the identical rules before it is used for the final run.
assert corrected_summary["passed"], "returns_corrected must pass the same checks before use"
SOURCE_TABLE = CORRECTED_TABLE
print(f"ACCEPTED {SOURCE_TABLE} — same rules passed; it feeds the final feature run.")

# COMMAND ----------

# DBTITLE 1,Section 2 — Rejection explanation
# MAGIC %md
# MAGIC **Why `returns_initial` is rejected, and the next action.**
# MAGIC
# MAGIC The completeness rule fails on `R02` (missing required value: `income`) and the uniqueness rule
# MAGIC fails on `R01` (the same return delivered twice), so the batch as a whole cannot be trusted.
# MAGIC Zero-filling the unknown income or silently dropping the duplicated row would manufacture or
# MAGIC discard data, which the task forbids; the checks are therefore kept and the whole initial batch
# MAGIC is rejected **before any feature output is written**. The 3 affected rows (both `R01` copies and
# MAGIC `R02`) are preserved with their reasons in `initial_rejected_records` (personal output schema)
# MAGIC and exported to `outputs/initial_rejected_records.csv`.
# MAGIC
# MAGIC `returns_corrected` passes the identical rules (12 rows, 12 distinct `return_id`s, no missing
# MAGIC required values, no negative amounts), so it is the usable batch and feeds `return_features` in
# MAGIC the next section.

# COMMAND ----------

# DBTITLE 1,Section 3 — Deliver a checked feature table
# MAGIC %md
# MAGIC ## 3. Deliver a checked feature table
# MAGIC
# MAGIC `return_features` holds exactly `return_id`, `tax_year` and `deduction_to_income`, where
# MAGIC `deduction_to_income = deduction / income`. Zero-income returns stay in the table with a **null**
# MAGIC ratio (division by zero guarded — not dropped, not zero-filled). Every accepted return appears
# MAGIC exactly once; `taxpayer_id` and the raw amounts are excluded under the need-to-know rule.
# MAGIC Verification reads the saved table back from the catalog and checks: exact schema, row count and
# MAGIC uniqueness, a nonzero ratio against an expected value derived independently from the source row,
# MAGIC and the zero-income cases.

# COMMAND ----------

# DBTITLE 1,Section 3 — Feature table and readback verification
import math
from pyspark.sql import functions as F

FEATURES_TABLE = f"{OUTPUT_SCHEMA}.return_features"
spark.sql(f"""
    CREATE OR REPLACE TABLE {FEATURES_TABLE} AS
    SELECT return_id,
           tax_year,
           CASE WHEN income = 0 THEN CAST(NULL AS DOUBLE)
                ELSE CAST(deduction AS DOUBLE) / CAST(income AS DOUBLE)
           END AS deduction_to_income
    FROM {SOURCE_TABLE}
""")

features = spark.table(FEATURES_TABLE)
print("Saved table read back from the catalog:")
display(features)


assert features.schema.fieldNames() == ["return_id", "tax_year", "deduction_to_income"]


assert features.count() == corrected_summary["total_rows"], "row count differs from accepted batch"
assert features.select("return_id").distinct().count() == features.count(), "duplicate return_ids saved"



probe = spark.sql(f"""
    SELECT return_id, income, deduction
    FROM {SOURCE_TABLE}
    WHERE income > 0
    ORDER BY return_id
""").first()
expected_ratio = probe["deduction"] / probe["income"]
saved_ratio = features.where(F.col("return_id") == probe["return_id"]).first()["deduction_to_income"]
assert expected_ratio > 0, "probe row must give a nonzero expected ratio"
assert math.isclose(saved_ratio, expected_ratio, rel_tol=1e-12), "saved ratio differs from expected"
print({"probe_return_id": probe["return_id"], "income": probe["income"],
       "deduction": probe["deduction"], "expected_ratio": expected_ratio,
       "saved_ratio": saved_ratio})


saved_zero = sorted(r["return_id"] for r in features.where(F.col("deduction_to_income").isNull())
                    .select("return_id").collect())
source_zero = sorted(r["return_id"] for r in spark.table(SOURCE_TABLE).where(F.col("income") == 0)
                     .select("return_id").collect())
assert saved_zero == source_zero and len(saved_zero) > 0, "zero-income cases not preserved with null ratio"
print("Zero-income returns kept with null ratio:", saved_zero)
print("return_features verified: schema, uniqueness, count, nonzero expected value, zero-income nulls.")

# COMMAND ----------

# DBTITLE 1,Section 3 — Verification summary
# MAGIC %md
# MAGIC **Feature table verification (readback).** `return_features` was re-read from the catalog after
# MAGIC saving: exactly the three contract columns (no `taxpayer_id`, `income` or `deduction`), with
# MAGIC 12 rows = 12 distinct `return_id`s = every accepted return exactly once. The probe return's
# MAGIC nonzero ratio matched the value recomputed directly from its source `income` and `deduction`,
# MAGIC and the two zero-income returns (`R03`, `R09`) were present with a null `deduction_to_income`.

# COMMAND ----------

# DBTITLE 1,Section 4 — Make the handoff usable
# MAGIC %md
# MAGIC ## 4. Make the handoff usable
# MAGIC
# MAGIC From the saved `return_features`: `features_train` (tax_year 2022), `features_validation` (2023)
# MAGIC and `features_test` (2024). Each run fully replaces the split tables, so reruns cannot append
# MAGIC copies. Verification: each split holds only its assigned year, every accepted return lands in
# MAGIC exactly one table, and the union of the splits equals the feature table. Two complete successful
# MAGIC runs must leave the same return IDs and values: each run records a snapshot (with notebook URL
# MAGIC and execution timestamp) of all saved tables, and the run evidence compares the current run with
# MAGIC the previous one. Small CSV exports of every saved table, the rejected initial records and the
# MAGIC two-run comparison evidence land in `outputs/`, and `SUBMISSION.md` is completed from that
# MAGIC evidence.

# COMMAND ----------

# DBTITLE 1,Section 4 — Splits, rerun comparison, exports

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("/Workspace/Users/abhishek.s.jeyakumar.irs@catalyte.ai/to6-databricks-challenge-starter")
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

TRAIN_TABLE = f"{OUTPUT_SCHEMA}.features_train"
VALIDATION_TABLE = f"{OUTPUT_SCHEMA}.features_validation"
TEST_TABLE = f"{OUTPUT_SCHEMA}.features_test"


for table, year in [(TRAIN_TABLE, 2022), (VALIDATION_TABLE, 2023), (TEST_TABLE, 2024)]:
    spark.sql(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT return_id, tax_year, deduction_to_income
        FROM {FEATURES_TABLE}
        WHERE tax_year = {year}
    """)


split_tables = [(TRAIN_TABLE, 2022), (VALIDATION_TABLE, 2023), (TEST_TABLE, 2024)]
split_counts = {}
for table, year in split_tables:
    df = spark.table(table)
    years = sorted({r["tax_year"] for r in df.select("tax_year").distinct().collect()})
    assert years == [year], f"{table} holds years {years}, expected only {year}"
    assert df.count() == df.select("return_id").distinct().count(), f"{table} has duplicate return_ids"
    split_counts[table] = df.count()

split_id_rows = (spark.table(TRAIN_TABLE).select("return_id")
                 .union(spark.table(VALIDATION_TABLE).select("return_id"))
                 .union(spark.table(TEST_TABLE).select("return_id"))
                 .collect())
accepted_ids = {r["return_id"] for r in spark.table(FEATURES_TABLE).select("return_id").collect()}
assert len(split_id_rows) == len(accepted_ids), "accepted returns not split exactly once overall"
assert {r["return_id"] for r in split_id_rows} == accepted_ids, "split ids differ from accepted ids"
print("Split counts:", split_counts)
display(spark.sql(f"SELECT tax_year, COUNT(*) AS rows FROM {FEATURES_TABLE} GROUP BY tax_year ORDER BY tax_year"))


def table_snapshot(table):
    return [[r["return_id"], int(r["tax_year"]),
             None if r["deduction_to_income"] is None else float(r["deduction_to_income"])]
            for r in spark.table(table).orderBy("return_id").collect()]

snapshots = {FEATURES_TABLE: table_snapshot(FEATURES_TABLE)}
for table, _ in split_tables:
    snapshots[table] = table_snapshot(table)
fingerprint = hashlib.sha256(json.dumps(snapshots, sort_keys=True).encode("utf-8")).hexdigest()

try:
    workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
except Exception:
    workspace_host = "unknown-workspace-host"
notebook_url = f"https://{workspace_host}/editor/notebooks/127024866559307"

run_evidence = {
    "run_url": notebook_url,
    "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "row_counts": {**split_counts, FEATURES_TABLE: len(snapshots[FEATURES_TABLE])},
    "table_fingerprint": fingerprint,
    "snapshots": snapshots,
}
evidence_path = OUTPUTS_DIR / "run_evidence.json"
runs = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else []
previous = runs[-1] if runs else None
runs.append(run_evidence)
evidence_path.write_text(json.dumps(runs[-2:], indent=2), encoding="utf-8")

if previous is None:
    print("First successful run recorded at", run_evidence["execution_timestamp_utc"])
    print("Run this notebook once more — the second complete run writes the two-run comparison evidence.")
else:
    ids_values_match = (previous["table_fingerprint"] == fingerprint
                        and previous["snapshots"] == snapshots)
    comparison = {
        "run_1": {"run_url": previous["run_url"],
                  "execution_timestamp_utc": previous["execution_timestamp_utc"],
                  "row_counts": previous["row_counts"],
                  "table_fingerprint": previous["table_fingerprint"]},
        "run_2": {"run_url": run_evidence["run_url"],
                  "execution_timestamp_utc": run_evidence["execution_timestamp_utc"],
                  "row_counts": run_evidence["row_counts"],
                  "table_fingerprint": fingerprint},
        "identical_return_ids_and_values": bool(ids_values_match),
        "no_appended_copies": previous["row_counts"] == run_evidence["row_counts"],
        "note": "Fingerprints cover every return_id, tax_year and deduction_to_income value in "
                 "return_features and all three year tables; identical fingerprints plus equal "
                 "row counts show the two complete runs left the same IDs and values.",
    }
    (OUTPUTS_DIR / "two_run_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    assert ids_values_match, "Two complete runs produced different return IDs or values"
    assert comparison["no_appended_copies"], "Row counts changed between runs (appended copies?)"
    print("Two complete runs match: identical return IDs and values, unchanged row counts (no appended copies).")


EXPORTS = [
    (FEATURES_TABLE, "return_features.csv"),
    (TRAIN_TABLE, "features_train.csv"),
    (VALIDATION_TABLE, "features_validation.csv"),
    (TEST_TABLE, "features_test.csv"),
    (REJECTED_TABLE, "initial_rejected_records.csv"),
]
for table, filename in EXPORTS:
    spark.table(table).toPandas().to_csv(OUTPUTS_DIR / filename, index=False)
    print("exported", filename)

print(f"""
Handoff summary
- Inputs: {INITIAL_TABLE} (rejected), {SOURCE_TABLE} (accepted), {LINEAGE_VIEW} (lineage)
- Outputs: {FEATURES_TABLE}, {TRAIN_TABLE}, {VALIDATION_TABLE}, {TEST_TABLE}, {REJECTED_TABLE}
- Ratio: deduction_to_income = deduction / income; zero income keeps the row with a null ratio
- Checks: completeness, uniqueness, amount sanity; feature schema and need-to-know; readback with
  independently derived nonzero expected value and zero-income case; split coverage; two-run
  stability. Exports are in the Git folder's outputs/ directory.
""")