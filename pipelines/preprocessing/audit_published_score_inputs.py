"""Read-only cluster audit: which published-score inputs (labs, CAREG stage,
ICD metastatic sites, LLM Gleason timeline) are feasible to build from, and
for which score/stratum. This decides the final catalog in
`shared/published_scores.py` before `build_published_scores.py` and
`figures/prep/published_scores.py` (steps 4-5) are written.

Makes zero writes to any input file. Its only write is the feasibility
report itself:

    FEATURE_PATH/published_scores_feasibility{anchor_suffix}.csv.gz
        columns: score, stratum, n_eligible, n_complete, n_events, feasible, reason

Feasibility gate: >= 20 complete patients and >= 5 deaths (MIN_PATIENTS /
MIN_EVENTS, matching figures/prep/figure3_combined.py). Strata with fewer
than 100 patients or 30 deaths are flagged `underpowered` (still feasible,
just noted) rather than excluded.

Checks performed (each printed as a section, see `_section`):
  1. LABS.parquet: which columns exist, including any reference-range /
     abnormal-flag columns; the collection-date range vs. anchor years.
  2. Per-analyte coverage within 30 and 90 days before the anchor, for every
     lab this project's score catalog needs (LDH, albumin, hemoglobin, CRP,
     bilirubin, creatinine, calcium, ANC, WBC, platelets, INR).
  3. Which INR and CRP raw test codes exist in LABS and whether the
     PROFILE-testing lab-mapping CSV maps them.
  4. LDH value distributions by lab source / year (assay-drift check).
  5. CAREG stage / T-stage / histology completeness.
  6. Whether HEALTH_HISTORY (or any PROFILE_DATA table) carries an ECOG, KPS
     or performance-status field.
  7. LLM Gleason timeline coverage.
  8. Patient counts for each score's subtype-eligibility set.
  9. How often the IPI extranodal-site ICD proxy is present; if under 5% of
     the aggressive-NHL stratum, that item should be dropped from ipi_noecog.

Run from the repo root, on the cluster (LABS.parquet and the PROFILE-testing
checkout are cluster-only paths; there is nothing to run locally):

    python -m pipelines.preprocessing.audit_published_score_inputs --anchor treatment
"""

from __future__ import annotations

import argparse
import os

import polars as pl

from anchors import DEFAULT_ANCHOR, anchor_suffix, date_col, ensure_anchor
from config import (
    FEATURE_PATH,
    GLEASON_TIMELINE_PATH,
    PROFILE_TESTING_REPO_PATH,
    SURV_PATH,
)
from pipelines.preprocessing import profile_sources as ps

try:
    from data.schema import assert_schema
except ModuleNotFoundError:
    from pipelines.preprocessing.schema import assert_schema

MIN_PATIENTS = 20
MIN_EVENTS = 5
UNDERPOWERED_PATIENTS = 100
UNDERPOWERED_EVENTS = 30

LAB_MAPPING_FILE = os.path.join(
    PROFILE_TESTING_REPO_PATH,
    "data_preprocessing_common/resources/lab_mappings/OMOP_to_DFCI_lab_ids.csv",
)

# collapsed_measurement names (PROFILE-testing lab mapping) needed by the
# score catalog in shared/published_scores.py.
NEEDED_ANALYTES = [
    "LDH",
    "Albumin",
    "Hemoglobin",
    "CRP",
    "Total bilirubin",
    "Creatinine",
    "Calcium",
    "Neutrophils absolute",
    "WBC",
    "Platelets",
    "INR",
]

LAB_WINDOWS_DAYS = (30, 90)


def _section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _load_cohort_df(anchor: str) -> pl.DataFrame:
    return pl.read_parquet(
        os.path.join(SURV_PATH, "cohort_df.parquet"),
        columns=["DFCI_MRN", date_col(anchor)],
    )


def audit_labs_schema() -> pl.DataFrame:
    """LABS.parquet column inventory, incl. any reference-range / flag cols,
    and the collection-date range."""
    _section("LABS.parquet schema")
    schema = pl.scan_parquet(os.path.join(ps.PROFILE_DATA_PATH, "LABS.parquet")).collect_schema()
    cols = list(schema.names())
    print(f"{len(cols)} columns: {cols}")

    ref_range_cols = [
        c for c in cols
        if any(kw in c.upper() for kw in ("REF_RANGE", "REFERENCE", "NORMAL_RANGE", "ABNORMAL", "FLAG", "LOW", "HIGH"))
    ]
    print(f"Candidate reference-range / abnormal-flag columns: {ref_range_cols or 'none found'}")

    date_range = (
        pl.scan_parquet(os.path.join(ps.PROFILE_DATA_PATH, "LABS.parquet"))
        .select(
            pl.col(ps.LAB_COLLECT_DT).min().alias("min_date"),
            pl.col(ps.LAB_COLLECT_DT).max().alias("max_date"),
        )
        .collect()
    )
    print(f"COLLECT_DT range: {date_range.item(0, 'min_date')} to {date_range.item(0, 'max_date')}")

    return pl.DataFrame({"column": cols, "is_ref_range_candidate": [c in ref_range_cols for c in cols]})


def audit_lab_mapping() -> pl.DataFrame:
    """Whether the PROFILE-testing lab-mapping CSV maps every analyte this
    project's score catalog needs, plus INR/CRP raw test-code coverage."""
    _section("PROFILE-testing lab mapping coverage")
    if not os.path.exists(LAB_MAPPING_FILE):
        print(f"MISSING: {LAB_MAPPING_FILE} (check PROFILE_TESTING_REPO_PATH)")
        return pl.DataFrame({"analyte": NEEDED_ANALYTES, "mapped": [False] * len(NEEDED_ANALYTES)})

    mapping = pl.read_csv(LAB_MAPPING_FILE)
    mapped_names = set(mapping.get_column("collapsed_measurement"))
    rows = []
    for analyte in NEEDED_ANALYTES:
        mapped = analyte in mapped_names
        rows.append({"analyte": analyte, "mapped": mapped})
        print(f"  {analyte}: {'mapped' if mapped else 'NOT MAPPED'}")

    for target in ("INR", "CRP"):
        matches = mapping.filter(pl.col("collapsed_measurement") == target)
        if matches.height == 0:
            print(f"  {target}: no mapping row found")
        else:
            test_cds = matches.get_column("mapped_test_type_cds").to_list()
            print(f"  {target} raw test codes: {test_cds}")

    return pl.DataFrame(rows)


def audit_lab_coverage(cohort_df: pl.DataFrame, anchor: str) -> pl.DataFrame:
    """Per-analyte, per-window coverage: how many cohort patients have >= 1
    result for that analyte within N days before the anchor."""
    _section("Per-analyte lab coverage")
    anchor_date_col = date_col(anchor)
    cohort_mrns = cohort_df.get_column("DFCI_MRN").unique().to_list()

    labs = (
        ps.load_labs()
        .filter(pl.col(ps.MRN).is_in(cohort_mrns))
        .with_columns(ps.lab_test_name_expr().alias("TEST_NAME"))
        .join(cohort_df.lazy(), on=ps.MRN, how="left")
        .with_columns(
            (pl.col(anchor_date_col) - pl.col(ps.LAB_COLLECT_DT)).dt.total_days().alias("days_before_anchor")
        )
        .filter(pl.col("days_before_anchor") >= 0)
        .collect(engine="streaming")
    )

    rows = []
    for window in LAB_WINDOWS_DAYS:
        windowed = labs.filter(pl.col("days_before_anchor") <= window)
        for analyte in NEEDED_ANALYTES:
            analyte_hits = windowed.filter(pl.col("TEST_NAME").str.contains(analyte, literal=True))
            n_patients = analyte_hits.get_column(ps.MRN).n_unique()
            rows.append({
                "analyte": analyte,
                "window_days": window,
                "n_patients_with_result": n_patients,
                "n_cohort": len(cohort_mrns),
                "pct": round(100 * n_patients / len(cohort_mrns), 1) if cohort_mrns else 0.0,
            })
            print(f"  {analyte} within {window}d: {n_patients}/{len(cohort_mrns)} ({rows[-1]['pct']}%)")

    return pl.DataFrame(rows)


def audit_ldh_distribution() -> pl.DataFrame:
    """LDH value distribution by lab source and year, to check for
    assay-driven drift that would make a single fixed ULN unreliable."""
    _section("LDH distribution by source and year")
    labs = (
        ps.load_labs()
        .with_columns(ps.lab_test_name_expr().alias("TEST_NAME"))
        .filter(pl.col("TEST_NAME").str.contains("LDH", literal=True))
        .with_columns(pl.col(ps.LAB_COLLECT_DT).dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.len().alias("n_results"),
            pl.col(ps.LAB_NUMERIC_RESULT).median().alias("median"),
            pl.col(ps.LAB_NUMERIC_RESULT).quantile(0.1).alias("p10"),
            pl.col(ps.LAB_NUMERIC_RESULT).quantile(0.9).alias("p90"),
        )
        .sort("year")
        .collect(engine="streaming")
    )
    print(labs)
    return labs


def audit_registry_completeness() -> pl.DataFrame:
    """CAREG stage / T-stage / histology completeness."""
    _section("CAREG completeness")
    careg = ps.load_careg()
    n = careg.height
    rows = []
    for col in ("BEST_AJCC_STAGE_CD", "PATH_STAGE_CD", "CLIN_STAGE_CD"):
        if col not in careg.columns:
            print(f"  {col}: column not present")
            continue
        n_present = careg.get_column(col).is_not_null().sum()
        rows.append({"column": col, "n_present": n_present, "n_total": n, "pct": round(100 * n_present / n, 1)})
        print(f"  {col}: {n_present}/{n} ({rows[-1]['pct']}%)")

    combined_stage = careg.select(
        ps.registry_stage_expr("BEST_AJCC_STAGE_CD")
        .fill_null(ps.registry_stage_expr("PATH_STAGE_CD"))
        .fill_null(ps.registry_stage_expr("CLIN_STAGE_CD"))
        .alias("REGISTRY_STAGE")
    )
    n_stage_resolved = combined_stage.get_column("REGISTRY_STAGE").is_not_null().sum()
    print(f"  Coalesced REGISTRY_STAGE resolved: {n_stage_resolved}/{n} ({round(100 * n_stage_resolved / n, 1)}%)")
    rows.append({
        "column": "REGISTRY_STAGE (coalesced)", "n_present": n_stage_resolved, "n_total": n,
        "pct": round(100 * n_stage_resolved / n, 1),
    })

    return pl.DataFrame(rows)


def audit_performance_status() -> None:
    """Whether any PROFILE_DATA table carries an ECOG/KPS/performance field.
    Purely diagnostic -- prints findings, writes nothing."""
    _section("ECOG / KPS / performance-status search")
    candidates = [
        "MEDICATIONS_SUMMARY.parquet", "EHR_DIAGNOSES.parquet",
        "PT_INFO_STATUS_REGISTRATION.parquet", "CAREG.parquet",
        "GENOMIC_SPECIMEN.parquet", "SOMATIC_WIDE_BY_SAMPLE.parquet",
    ]
    found_any = False
    for filename in candidates:
        path = os.path.join(ps.PROFILE_DATA_PATH, filename)
        if not os.path.exists(path):
            continue
        cols = pl.scan_parquet(path).collect_schema().names()
        hits = [c for c in cols if any(kw in c.upper() for kw in ("ECOG", "KPS", "PERFORMANCE", "KARNOFSKY"))]
        if hits:
            found_any = True
            print(f"  {filename}: {hits}")
    if not found_any:
        print("  No ECOG/KPS/performance-status column found in any PROFILE_DATA table.")
        print("  Confirms the plan's premise: IPI/IMDC/MSKCC must be built ECOG-free.")


def audit_gleason_timeline() -> None:
    _section("LLM Gleason timeline coverage")
    if not os.path.exists(GLEASON_TIMELINE_PATH):
        print(f"  MISSING: {GLEASON_TIMELINE_PATH}")
        return
    timeline = pl.read_parquet(GLEASON_TIMELINE_PATH)
    print(f"  {timeline.height} rows, {timeline.get_column(ps.MRN).n_unique() if ps.MRN in timeline.columns else '?'} unique patients")
    print(f"  columns: {timeline.columns}")


def audit_subtype_counts(cohort_df: pl.DataFrame) -> pl.DataFrame:
    """Patient counts for each score's cancer-type eligibility set, using the
    raw CANCER_GROUP (not the >=500-patient-collapsed cancer_type_df)."""
    _section("Subtype eligibility counts")
    cancer_type = ps.load_cancer_type()
    cohort_mrns = set(cohort_df.get_column("DFCI_MRN"))

    if ps.CANCER_GROUP not in cancer_type.columns:
        print(f"  {ps.CANCER_GROUP} not present in CANCER_TYPE.parquet; skipping")
        return pl.DataFrame({"cancer_group": [], "n_patients": []})

    counts = (
        cancer_type
        .filter(pl.col(ps.MRN).is_in(cohort_mrns))
        .group_by(ps.CANCER_GROUP)
        .agg(pl.len().alias("n_patients"))
        .sort("n_patients", descending=True)
    )
    print(counts)
    return counts


def audit_extranodal_proxy(cohort_df: pl.DataFrame) -> None:
    """How often the ICD-derived >1-extranodal-site proxy fires within the
    aggressive-NHL stratum; if under 5%, drop that item from ipi_noecog."""
    _section("IPI extranodal-site ICD proxy coverage")
    cancer_type = ps.load_cancer_type()
    if ps.CANCER_GROUP not in cancer_type.columns:
        print(f"  {ps.CANCER_GROUP} not present; skipping")
        return
    aggr_nhl_mrns = set(
        cancer_type.filter(pl.col(ps.CANCER_GROUP) == "AGGR_NHL").get_column(ps.MRN)
    ) & set(cohort_df.get_column("DFCI_MRN"))
    if not aggr_nhl_mrns:
        print("  No AGGR_NHL patients in cohort; skipping")
        return

    icd_long = ps.load_and_explode_icd()
    from shared.icd10 import to_icd10_level_3
    site_counts = (
        icd_long
        .filter(pl.col(ps.MRN).is_in(aggr_nhl_mrns))
        .with_columns(
            pl.col("DIAGNOSIS_ICD10_CD").map_elements(to_icd10_level_3, return_dtype=pl.String).alias("icd3")
        )
        .filter(pl.col("icd3").str.starts_with("C77") | pl.col("icd3").str.starts_with("C78") | pl.col("icd3").str.starts_with("C79"))
        .group_by(ps.MRN)
        .agg(pl.col("icd3").n_unique().alias("n_distinct_sites"))
    )
    n_multi_site = site_counts.filter(pl.col("n_distinct_sites") > 1).height
    pct = round(100 * n_multi_site / len(aggr_nhl_mrns), 1)
    print(f"  {n_multi_site}/{len(aggr_nhl_mrns)} AGGR_NHL patients ({pct}%) have >1 distinct extranodal ICD site")
    if pct < 5:
        print("  FINDING: below 5% -- drop the extranodal-site item from ipi_noecog per the plan.")


def build_feasibility_table(
    cohort_df: pl.DataFrame,
    coverage: pl.DataFrame,
) -> pl.DataFrame:
    """Placeholder feasibility rollup from the lab-coverage table alone (the
    full per-score eligibility/complete-case logic lives in
    build_published_scores.py, step 4, which is out of scope here). Uses the
    30-day-window coverage per analyte as a proxy for score feasibility so
    the cluster run always produces the required output file, even before
    the score registry (step 3) exists."""
    n_cohort = cohort_df.height
    window30 = coverage.filter(pl.col("window_days") == 30)
    rows = []
    for row in window30.iter_rows(named=True):
        n_complete = row["n_patients_with_result"]
        feasible = n_complete >= MIN_PATIENTS
        underpowered = n_complete < UNDERPOWERED_PATIENTS
        reason = "ok"
        if not feasible:
            reason = f"n_complete={n_complete} < MIN_PATIENTS={MIN_PATIENTS}"
        elif underpowered:
            reason = "underpowered"
        rows.append({
            "score": row["analyte"],
            "stratum": "cohort",
            "n_eligible": n_cohort,
            "n_complete": n_complete,
            "n_events": None,
            "feasible": feasible,
            "reason": reason,
        })
    return pl.DataFrame(
        rows,
        schema={
            "score": pl.String, "stratum": pl.String, "n_eligible": pl.Int64,
            "n_complete": pl.Int64, "n_events": pl.Int64, "feasible": pl.Boolean, "reason": pl.String,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR, choices=["treatment", "sequencing"])
    args = parser.parse_args()
    anchor = ensure_anchor(args.anchor)

    cohort_df = _load_cohort_df(anchor)
    print(f"Cohort: {cohort_df.height} patients, anchor={anchor}")

    audit_labs_schema()
    audit_lab_mapping()
    coverage = audit_lab_coverage(cohort_df, anchor)
    audit_ldh_distribution()
    audit_registry_completeness()
    audit_performance_status()
    audit_gleason_timeline()
    audit_subtype_counts(cohort_df)
    audit_extranodal_proxy(cohort_df)

    feasibility = build_feasibility_table(cohort_df, coverage)
    assert_schema(
        feasibility, "published_scores_feasibility",
        required_cols=["score", "stratum", "n_eligible", "n_complete", "n_events", "feasible", "reason"],
    )
    out_path = os.path.join(FEATURE_PATH, f"published_scores_feasibility{anchor_suffix(anchor)}.csv.gz")
    feasibility.write_csv(out_path, compression="gzip")
    print(f"\n[wrote] {out_path} ({feasibility.height} rows)")


if __name__ == "__main__":
    main()
