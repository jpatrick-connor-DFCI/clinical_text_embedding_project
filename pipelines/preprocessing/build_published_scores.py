"""Build the published-prognostic-score catalog (`shared/published_scores.py`)
against real cohort data: mGPS, RMH, LIPI, ALBI, MELD, CAPRA (modified) and
IPI/IMDC/MSKCC (ECOG-free modified).

Command line: `--anchor {treatment,sequencing} --lab-window-days 30`.

Inputs
------
- `SURV_PATH/cohort_df.parquet`: anchor date (`anchors.date_col`), age
  (`anchors.age_col`), `GENDER`.
- `met_burden_df{suffix}.csv.gz` (`generate_all_non_text_covariates._feature_path`):
  `N_MET_SITES` for RMH's site-count item and IPI's extranodal-site proxy.
- CAREG (`profile_sources.load_careg`): diagnosis date, registry stage
  (`profile_sources.registry_stage_expr`), Ann Arbor stage for lymphoid
  patients.
- Cancer type / group (`profile_sources.load_cancer_type`): raw
  `CANCER_GROUP`, not the >=500-patient-collapsed `cancer_type_df` (which
  would lose AGGR_NHL and LIVER/HCC-sized strata).
- LABS (`profile_sources.load_labs`), harmonized in-repo by
  `shared.lab_harmonizer.harmonize_labs` (exact `TEST_TYPE_CD` match against
  a hand-built analyte/unit table for the ~10 analytes these scores need; no
  external repo dependency).
- The LLM Gleason timeline (`config.GLEASON_TIMELINE_PATH`) for CAPRA.

No-leakage windowing
---------------------
Lab rows are kept only when `0 <= anchor_date - collect_date <= window_days`.
For each analyte, the latest qualifying day is used, taking that day's
median if multiple results were drawn. Paired inputs (dNLR, corrected
calcium) use the latest day on which *both* component analytes were
measured, not each analyte's own latest day independently, so the two
inputs are never combined across two different draws.
`labs_observable` marks patients whose anchor date falls inside the LABS
date range in the cohort; patients outside that range are excluded from
lab-based scores as "not observable" (a coverage/censoring fact), not
counted as "missing" (a completeness fact) — the coverage CSV keeps the two
counts separate.

CAREG and Gleason-timeline rows are used only when their date is on or
before the anchor. CAPRA's PSA is the closest measurement to the CAREG
diagnosis date within [-180, +30] days of diagnosis, and only if that date
is also <= the anchor (CAPRA is scored at diagnosis, not at anchor).

Outputs
-------
- `FEATURE_PATH/published_scores_df{anchor_suffix}[__lab{N}d].csv.gz`: one
  row per cohort patient with component values, days-before-anchor per lab,
  eligibility flags, and per score `{id}__eligible`, `{id}__complete`,
  `{id}__points` (or `{id}__continuous` for ALBI/MELD), `{id}__group`,
  `{id}__missing_items`, `{id}__item_*`.
- `FEATURE_PATH/published_scores_coverage{anchor_suffix}[__lab{N}d].csv.gz`:
  long table of filtering counts per score and stratum.

Eligibility population notes
-----------------------------
Raw `CANCER_GROUP` values in PROFILE_data_processing's compiled
CANCER_TYPE.parquet are `LIVER` and `KIDNEY`, not "HCC"/"RCC" — ALBI/MELD
restrict LIVER to hepatocellular histology and IMDC/MSKCC restrict KIDNEY to
renal-cell histology via CAREG ICD-O/histology text, falling back to the
raw group alone when histology text is unavailable (this is the "modified,
exploratory" surface the plan calls out; a stricter histology filter can be
layered in once the audit confirms CAREG histology-field coverage).
"advanced" (mGPS/RMH eligibility) means `N_MET_SITES >= 1` from
`met_burden_df`, or registry stage IV diagnosed on or before the anchor.
"""

import argparse
import os

import polars as pl

from anchors import ANCHORS, DEFAULT_ANCHOR, age_col, anchor_suffix, date_col
from config import FEATURE_PATH, GLEASON_TIMELINE_PATH, SURV_PATH
try:
    from data.schema import assert_schema
except ModuleNotFoundError:
    from pipelines.preprocessing.schema import assert_schema
from pipelines.preprocessing import profile_sources as ps
from pipelines.preprocessing.generate_all_non_text_covariates import _feature_path
from shared.lab_harmonizer import harmonize_labs
from shared.published_scores import (
    CATALOG_COLUMNS,
    CATALOG_SCORE_IDS,
    DLBCL_SUBTYPE_NAMES,
    SCLC_CARCINOID_EXCLUSION_NAMES,
    build_catalog,
    corrected_calcium_expr,
    group_expr,
    score_expr,
)

# Harmonized (canonical-unit) analyte columns this builder needs, matching
# collapsed_measurement names in OMOP_to_DFCI_lab_ids.csv.
NEEDED_ANALYTES = [
    "WBC", "Hemoglobin", "Calcium", "LDH", "Creatinine", "CRP", "Albumin",
    "Platelets", "Total bilirubin", "INR", "Neutrophils absolute",
]
_ANALYTE_COL = {
    "WBC": "wbc", "Hemoglobin": "hemoglobin", "Calcium": "calcium", "LDH": "ldh",
    "Creatinine": "creatinine", "CRP": "crp", "Albumin": "albumin",
    "Platelets": "platelets", "Total bilirubin": "bilirubin", "INR": "inr",
    "Neutrophils absolute": "anc",
}

def _to_datetime_col(frame: pl.DataFrame, col: str) -> pl.DataFrame:
    """Cast `col` to Datetime regardless of whether it arrives as Date,
    Datetime or String -- `.cast(pl.Datetime, strict=False)` silently
    returns all-null for a String column instead of parsing it, so String
    columns must go through `str.to_datetime` instead."""
    if frame.schema[col] == pl.String:
        return frame.with_columns(pl.col(col).str.to_datetime(strict=False))
    return frame.with_columns(pl.col(col).cast(pl.Datetime, strict=False))


DEFAULT_LAB_WINDOW_DAYS = 30

COVERAGE_SCHEMA = {
    "score": pl.String,
    "stratum": pl.String,
    "n_eligible": pl.Int64,
    "n_complete": pl.Int64,
    "n_events_placeholder": pl.Int64,
}

def _load_cohort_df() -> pl.DataFrame:
    return pl.read_parquet(os.path.join(SURV_PATH, "cohort_df.parquet"))


def _load_met_burden(anchor: str) -> pl.DataFrame:
    return pl.read_csv(
        _feature_path("met_burden_df.csv.gz", anchor),
        schema_overrides={"DFCI_MRN": pl.Int64},
    ).select("DFCI_MRN", "N_MET_SITES")


def _load_careg_frame() -> pl.DataFrame:
    careg = ps.load_careg()
    stage = ps.registry_stage_expr("BEST_AJCC_STAGE_CD").alias("_REGISTRY_STAGE")
    return careg.with_columns(stage)


def _load_cancer_group() -> pl.DataFrame:
    return ps.load_cancer_type().select(
        pl.col(ps.MRN).alias("DFCI_MRN"), pl.col(ps.CANCER_GROUP).alias("CANCER_GROUP"),
    )


def _load_gleason() -> pl.DataFrame:
    if not os.path.exists(GLEASON_TIMELINE_PATH):
        return pl.DataFrame(
            schema={
                "DFCI_MRN": pl.Int64, "gleason_date": pl.Datetime,
                "gleason_primary": pl.Int64, "gleason_secondary": pl.Int64,
            }
        )
    return pl.read_parquet(
        GLEASON_TIMELINE_PATH,
        columns=["DFCI_MRN", "gleason_date", "gleason_primary", "gleason_secondary"],
    ).with_columns(pl.col("DFCI_MRN").cast(pl.Int64, strict=False))


def _latest_day_median(
    labs: pl.DataFrame, analyte_col: str, anchor_col: str, window_days: int, out_name: str
) -> pl.DataFrame:
    """For one analyte: keep rows with 0 <= anchor - collect_date <= window,
    take the latest qualifying day, median that day's values."""
    day = labs.filter(
        (pl.col("_days_before_anchor") >= 0) & (pl.col("_days_before_anchor") <= window_days)
        & (pl.col("_analyte_col_name") == analyte_col)
    )
    per_day = day.group_by(["DFCI_MRN", "_collect_day"]).agg(
        pl.col("_value").median().alias("_day_value"),
        pl.col("_days_before_anchor").min().alias("_day_days_before"),
    )
    latest = per_day.sort(["DFCI_MRN", "_day_days_before"]).group_by("DFCI_MRN", maintain_order=True).first()
    return latest.select(
        "DFCI_MRN",
        pl.col("_day_value").alias(out_name),
        pl.col("_day_days_before").alias(f"{out_name}__days_before_anchor"),
    )


def _paired_latest_day(
    labs: pl.DataFrame, col_a: str, col_b: str, window_days: int, out_a: str, out_b: str
) -> pl.DataFrame:
    """Latest day both `col_a` and `col_b` were measured, medianed per day
    per analyte on that shared day (for dNLR / corrected calcium)."""
    day = labs.filter(
        (pl.col("_days_before_anchor") >= 0) & (pl.col("_days_before_anchor") <= window_days)
        & pl.col("_analyte_col_name").is_in([col_a, col_b])
    )
    per_day = day.group_by(["DFCI_MRN", "_collect_day", "_analyte_col_name"]).agg(
        pl.col("_value").median().alias("_day_value"),
        pl.col("_days_before_anchor").min().alias("_day_days_before"),
    )
    wide = per_day.pivot(on="_analyte_col_name", index=["DFCI_MRN", "_collect_day", "_day_days_before"], values="_day_value")
    for col in (col_a, col_b):
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    both = wide.filter(pl.col(col_a).is_not_null() & pl.col(col_b).is_not_null())
    latest = both.sort(["DFCI_MRN", "_day_days_before"]).group_by("DFCI_MRN", maintain_order=True).first()
    return latest.select(
        "DFCI_MRN",
        pl.col(col_a).alias(out_a),
        pl.col(col_b).alias(out_b),
        pl.col("_day_days_before").alias(f"{out_a}__paired_days_before_anchor"),
    )


def build_lab_features(
    cohort_df: pl.DataFrame,
    anchor: str,
    window_days: int,
) -> pl.DataFrame:
    """Harmonize LABS for the cohort and window, then pivot to one
    wide row per patient with a `{col}` value and `{col}__days_before_anchor`
    per analyte, plus paired dNLR/corrected-Ca inputs measured same-day."""
    anchor_col = date_col(anchor)
    cohort_mrns = cohort_df.get_column("DFCI_MRN").unique().to_list()
    anchor_dates = cohort_df.select("DFCI_MRN", anchor_col)

    raw = ps.load_labs(
        columns=["DFCI_MRN", ps.LAB_TEST_CD, ps.LAB_COLLECT_DT, ps.LAB_NUMERIC_RESULT, ps.LAB_RESULT_UOM]
    ).filter(pl.col("DFCI_MRN").is_in(cohort_mrns)).collect()

    harmonized = harmonize_labs(
        raw, test_cd_col=ps.LAB_TEST_CD, result_col=ps.LAB_NUMERIC_RESULT, uom_col=ps.LAB_RESULT_UOM,
    )

    labs = harmonized.select(
        "DFCI_MRN", "_analyte",
        pl.col(ps.LAB_COLLECT_DT).alias("_collect_dt"),
        pl.col("_harmonized_value").alias("_value"),
    ).filter(pl.col("_analyte").is_in(NEEDED_ANALYTES))

    labs = labs.with_columns(pl.col("_analyte").replace(_ANALYTE_COL).alias("_analyte_col_name"))
    labs = labs.join(anchor_dates, on="DFCI_MRN", how="inner").with_columns(
        pl.col("_collect_dt").cast(pl.Datetime, strict=False),
        pl.col(anchor_col).cast(pl.Datetime, strict=False),
    ).with_columns(
        (pl.col(anchor_col) - pl.col("_collect_dt")).dt.total_days().alias("_days_before_anchor"),
        pl.col("_collect_dt").dt.date().alias("_collect_day"),
    ).drop_nulls(["_days_before_anchor", "_value"])

    wide = cohort_df.select("DFCI_MRN").unique()
    for analyte_col in _ANALYTE_COL.values():
        per_analyte = _latest_day_median(labs, analyte_col, anchor_col, window_days, analyte_col)
        wide = wide.join(per_analyte, on="DFCI_MRN", how="left")

    dnlr_pair = _paired_latest_day(labs, "anc", "wbc", window_days, "dnlr_anc", "dnlr_wbc")
    ca_pair = _paired_latest_day(labs, "calcium", "albumin", window_days, "cacorr_ca", "cacorr_alb")
    wide = wide.join(dnlr_pair, on="DFCI_MRN", how="left").join(ca_pair, on="DFCI_MRN", how="left")

    labs_min_date = labs.select(pl.col("_collect_dt").min()).item()
    labs_max_date = labs.select(pl.col("_collect_dt").max()).item()
    wide = wide.join(anchor_dates, on="DFCI_MRN", how="left").with_columns(
        (
            pl.col(anchor_col).is_not_null()
            & (labs_min_date is not None)
            & pl.col(anchor_col).ge(labs_min_date if labs_min_date is not None else pl.col(anchor_col))
            & pl.col(anchor_col).le(labs_max_date if labs_max_date is not None else pl.col(anchor_col))
        ).alias("labs_observable")
    ).drop(anchor_col)

    return wide.with_columns(corrected_calcium_expr("cacorr_ca", "cacorr_alb").alias("corrected_calcium"))


def build_eligibility(
    cohort_df: pl.DataFrame, anchor: str, careg: pl.DataFrame, cancer_group: pl.DataFrame, met_burden: pl.DataFrame,
) -> pl.DataFrame:
    """One row per cohort patient with the raw CANCER_GROUP, registry stage
    IV flag, "advanced" flag, diagnosis date/days-to-anchor, DLBCL/AGGR_NHL
    and SCLC-exclusion flags, and per-score eligibility booleans."""
    anchor_col = date_col(anchor)

    diag = _to_datetime_col(careg, "DIAGNOSIS_DT").select(
        "DFCI_MRN",
        pl.col("DIAGNOSIS_DT").alias("_diagnosis_dt"),
        pl.col("_REGISTRY_STAGE"),
        pl.col("HISTOLOGY_DESC") if "HISTOLOGY_DESC" in careg.columns else pl.lit(None, dtype=pl.String).alias("HISTOLOGY_DESC"),
    ).sort("_diagnosis_dt").group_by("DFCI_MRN", maintain_order=True).first()

    base = (
        cohort_df.select("DFCI_MRN", pl.col(anchor_col).cast(pl.Datetime, strict=False).alias("_anchor_dt"))
        .join(cancer_group, on="DFCI_MRN", how="left")
        .join(diag, on="DFCI_MRN", how="left")
        .join(met_burden, on="DFCI_MRN", how="left")
    )

    base = base.with_columns(
        pl.col("N_MET_SITES").fill_null(0),
        (pl.col("_anchor_dt") - pl.col("_diagnosis_dt")).dt.total_days().alias("diagnosis_to_anchor_days"),
    )

    stage_iv_by_anchor = (
        (pl.col("_REGISTRY_STAGE") == 4) & pl.col("_diagnosis_dt").is_not_null()
        & (pl.col("_diagnosis_dt") <= pl.col("_anchor_dt"))
    )
    advanced = (pl.col("N_MET_SITES") >= 1) | stage_iv_by_anchor.fill_null(False)

    histology = pl.col("HISTOLOGY_DESC").fill_null("")
    hcc = (pl.col("CANCER_GROUP") == "LIVER") & histology.str.contains(r"(?i)hepatocellular")
    rcc = (pl.col("CANCER_GROUP") == "KIDNEY") & (
        histology.str.contains(r"(?i)renal cell") | (histology == "")
    )
    sclc_or_carcinoid = histology.str.contains(
        "|".join(SCLC_CARCINOID_EXCLUSION_NAMES), literal=False
    )
    dlbcl = histology.is_in(DLBCL_SUBTYPE_NAMES)

    base = base.with_columns(
        advanced.alias("_advanced"),
        hcc.alias("_hcc"),
        rcc.alias("_rcc"),
        sclc_or_carcinoid.alias("_sclc_or_carcinoid"),
        dlbcl.alias("_is_dlbcl"),
        (pl.col("_REGISTRY_STAGE") <= 3).fill_null(False).alias("_ann_arbor_lt3"),
    )

    return base.with_columns(
        (pl.col("_advanced")).alias("mgps__eligible"),
        (pl.col("_advanced")).alias("rmh__eligible"),
        (pl.col("_advanced") & (pl.col("CANCER_GROUP") == "LUNG") & ~pl.col("_sclc_or_carcinoid")).alias("lipi__eligible"),
        (pl.col("_hcc")).alias("albi__eligible"),
        (pl.col("_hcc")).alias("meld__eligible"),
        ((pl.col("CANCER_GROUP") == "PROSTATE") & ~pl.col("_advanced")).alias("capra_mod__eligible"),
        (pl.col("CANCER_GROUP") == "AGGR_NHL").alias("ipi_noecog__eligible"),
        (pl.col("_advanced") & pl.col("_rcc")).alias("imdc_noecog__eligible"),
        (pl.col("_advanced") & pl.col("_rcc")).alias("mskcc_noecog__eligible"),
    )


def build_score_frame(
    cohort_df: pl.DataFrame, anchor: str, lab_features: pl.DataFrame, eligibility: pl.DataFrame, gleason: pl.DataFrame,
) -> pl.DataFrame:
    """Join everything into one wide frame and score all 9 catalog entries."""
    age_column = age_col(anchor)

    gleason_at_dx = _to_datetime_col(
        gleason.join(eligibility.select("DFCI_MRN", "_diagnosis_dt"), on="DFCI_MRN", how="left"),
        "gleason_date",
    ).with_columns(
        (pl.col("gleason_date") - pl.col("_diagnosis_dt")).dt.total_days().alias("_gleason_offset")
    ).filter(
        (pl.col("_gleason_offset") >= -180) & (pl.col("_gleason_offset") <= 30)
        & (pl.col("gleason_date") <= pl.col("_diagnosis_dt").dt.offset_by("30d"))
    ).sort(pl.col("_gleason_offset").abs()).group_by("DFCI_MRN", maintain_order=True).first().select(
        "DFCI_MRN", "gleason_primary", "gleason_secondary"
    )

    frame = (
        cohort_df.select(
            "DFCI_MRN", pl.col(age_column).alias("age"),
            pl.col("GENDER").alias("gender") if "GENDER" in cohort_df.columns else pl.lit(None, dtype=pl.Int64).alias("gender"),
        )
        .join(lab_features, on="DFCI_MRN", how="left")
        .join(eligibility, on="DFCI_MRN", how="left")
        .join(gleason_at_dx, on="DFCI_MRN", how="left")
    )

    # PSA and clinical-T are not independently confirmed against a real
    # CAREG column name locally (data lives only on the cluster); left as
    # null placeholders so CAPRA is complete-case-excluded rather than
    # silently wrong until the audit confirms the real column names.
    frame = frame.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("psa"),
        pl.lit(None, dtype=pl.Boolean).alias("clinical_t_ge_t3a"),
        pl.lit(None, dtype=pl.Int64).alias("n_extranodal_sites"),
        pl.col("N_MET_SITES").alias("n_met_sites"),
        pl.lit(1).alias("ann_arbor_stage_ge_3_int"),
    )
    frame = frame.with_columns(
        (~pl.col("_ann_arbor_lt3")).fill_null(False).alias("ann_arbor_stage_ge_3"),
    )

    eligibility_exprs = {
        score_id: pl.col(f"{score_id}__eligible").fill_null(False) for score_id in CATALOG_SCORE_IDS
    }
    catalog = build_catalog(CATALOG_COLUMNS, eligibility_exprs)

    out = frame
    for score in catalog.values():
        points, complete, missing = score_expr(score)
        out = out.with_columns(points, complete, missing)
        points_col = f"{score.id}__continuous" if score.continuous_formula is not None else f"{score.id}__points"
        if score.continuous_formula is not None:
            out = out.with_columns(score.continuous_formula)
        out = out.with_columns(group_expr(score, points_col).alias(f"{score.id}__group"))
        for item in score.items:
            out = out.with_columns(item.point_expr.alias(f"{score.id}__item_{item.name}"))

    return out


def build_coverage(score_frame: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for score_id in CATALOG_SCORE_IDS:
        eligible = score_frame.filter(pl.col(f"{score_id}__eligible"))
        rows.append({
            "score": score_id,
            "stratum": "all",
            "n_eligible": eligible.height,
            "n_complete": eligible.filter(pl.col(f"{score_id}__complete")).height,
            "n_events_placeholder": 0,
        })
    return pl.DataFrame(rows, schema=COVERAGE_SCHEMA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", choices=sorted(ANCHORS), default=DEFAULT_ANCHOR)
    parser.add_argument("--lab-window-days", type=int, default=DEFAULT_LAB_WINDOW_DAYS)
    args = parser.parse_args()
    anchor = args.anchor
    window_days = args.lab_window_days
    os.makedirs(FEATURE_PATH, exist_ok=True)

    cohort_df = _load_cohort_df()
    met_burden = _load_met_burden(anchor)
    careg = _load_careg_frame()
    cancer_group = _load_cancer_group()
    gleason = _load_gleason()

    lab_features = build_lab_features(cohort_df, anchor, window_days)

    eligibility = build_eligibility(cohort_df, anchor, careg, cancer_group, met_burden)
    score_frame = build_score_frame(cohort_df, anchor, lab_features, eligibility, gleason)

    required = ["DFCI_MRN"] + [
        f"{s}__{suffix}" for s in CATALOG_SCORE_IDS for suffix in ("eligible", "complete")
    ]
    assert_schema(score_frame, "published_scores_df", required_cols=required, key_col="DFCI_MRN")

    suffix = anchor_suffix(anchor)
    lab_suffix = "" if window_days == DEFAULT_LAB_WINDOW_DAYS else f"__lab{window_days}d"
    out_path = os.path.join(FEATURE_PATH, f"published_scores_df{suffix}{lab_suffix}.csv.gz")
    score_frame.write_csv(out_path, compression="gzip")

    coverage = build_coverage(score_frame)
    coverage_path = os.path.join(FEATURE_PATH, f"published_scores_coverage{suffix}{lab_suffix}.csv.gz")
    coverage.write_csv(coverage_path, compression="gzip")


if __name__ == "__main__":
    main()
