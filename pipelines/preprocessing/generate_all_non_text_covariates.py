"""Build non-text clinical, genomic, and treatment covariates for the cohort.

Split into one function per feature file. Six files total; five keep their
original filenames and column contracts (CANCER_TYPE_*, CANCER_STAGE_*,
*_AMP/_DEL/_SNV/_SV/_FUSION, PGS*, PX_on_*). The sixth, met_burden_df.csv.gz
(build_met_burden_df), is new and requires matching registration in
`pipelines/training/slurm_array_utils.py` and friends - see that module's
`metburden` entries.

Cancer type and stage come from PROFILE_data_processing's compiled
CANCER_ANNOTATIONS parquets (CANCER_TYPE.parquet, CANCER_STAGE.parquet,
note-regex derived), not the GENOMIC_SPECIMEN/CAREG placeholders used
previously.
"""

import argparse
import os

import polars as pl

from anchors import ANCHORS, DEFAULT_ANCHOR, anchor_suffix, date_col
from config import FEATURE_PATH, MED_CLASSES_FILE, PRS_MATRIX_FILE, PROFILE_PATH, SURV_PATH
try:
    from data.schema import assert_schema
except ModuleNotFoundError:
    from pipelines.preprocessing.schema import assert_schema
from pipelines.preprocessing import profile_sources as ps
from shared.icd10 import MET_SITE_GROUPS, met_site_group
from shared.stages import normalize_stage


def _load_cohort_df() -> pl.DataFrame:
    return pl.read_parquet(os.path.join(SURV_PATH, "cohort_df.parquet"))


def build_cancer_type_df(cohort_df: pl.DataFrame) -> pl.DataFrame:
    """CANCER_TYPE from the compiled CANCER_TYPE.parquet's CANCER_GROUP column
    (genomics-first, ICD-fallback; see PROFILE_data_processing/
    CANCER_ANNOTATIONS_PLAN.md), restricted to the cohort and renamed to the
    existing CANCER_TYPE column contract."""
    cohort_mrns = cohort_df.get_column("DFCI_MRN").unique().to_list()

    cancer_type = (
        ps.load_cancer_type()
        .select([ps.MRN, pl.col(ps.CANCER_GROUP).alias("CANCER_TYPE")])
        .filter(pl.col(ps.MRN).is_in(cohort_mrns))
        .drop_nulls()
    )

    frequent_types = (
        cancer_type.group_by("CANCER_TYPE").len()
        .filter(pl.col("len") >= 500)
        .get_column("CANCER_TYPE")
        .to_list()
    )
    cancer_type_df = cancer_type.with_columns(
        pl.when(pl.col("CANCER_TYPE").is_in(frequent_types))
        .then(pl.col("CANCER_TYPE"))
        .otherwise(pl.lit("OTHER"))
        .alias("CANCER_TYPE")
    )
    # Preserve the raw (collapsed) label alongside the dummies. `drop_first=True`
    # removes the reference category's column, so patients in that category are
    # all-zero across CANCER_TYPE_* and would be mislabeled by a downstream idxmax.
    # Keeping the string label lets consumers recover the true type directly. The
    # column is named 'CANCER_TYPE' (no trailing underscore) so it is never picked
    # up by `startswith('CANCER_TYPE_')` feature-column filters.
    cancer_type_dummies = cancer_type_df.select("CANCER_TYPE").to_dummies(
        columns="CANCER_TYPE", drop_first=True
    )
    return cancer_type_df.hstack(cancer_type_dummies)


def build_cancer_stage_df() -> pl.DataFrame:
    """CANCER_STAGE from the compiled CANCER_STAGE.parquet's STAGE column
    (note-regex derived, earliest note observation per patient; see
    PROFILE_data_processing/CANCER_ANNOTATIONS_PLAN.md), normalized via
    shared.stages.normalize_stage() to collapse to I/II/III/IV (in-situ
    stage 0 normalizes to None and is dropped, matching the prior sentinel
    handling). Also emits the raw CANCER_STAGE string column so figure prep no
    longer needs a drop_first reconstruction fallback, and load_stage_map()
    can read this file directly. Note-regex derived rather than registry
    derived (CANCER_STAGE_REGISTRY.parquet) to match the two independent
    sources' intended default; registry stage remains available via
    ps.load_cancer_stage(registry=True) if needed."""
    note_stage = (
        ps.load_cancer_stage(registry=False)
        .select([ps.MRN, ps.STAGE])
        .rename({ps.MRN: "DFCI_MRN"})
    )
    note_stage = note_stage.with_columns(
        pl.col(ps.STAGE).map_elements(normalize_stage, return_dtype=pl.String).alias("CANCER_STAGE")
    ).drop_nulls("CANCER_STAGE").select(["DFCI_MRN", "CANCER_STAGE"])
    stage_dummies = note_stage.select("CANCER_STAGE").to_dummies(
        columns="CANCER_STAGE", drop_first=True
    )
    return note_stage.hstack(stage_dummies)


def build_somatic_data_df(
    cohort_df: pl.DataFrame, anchor: str = DEFAULT_ANCHOR, anchor_col: str | None = None
) -> pl.DataFrame:
    """SOMATIC_WIDE_BY_SAMPLE.parquet joined to GENOMIC_SPECIMEN on the 3-column
    key (never UNIQUE_SAMPLE_ID). A result is eligible only when REPORT_DT is
    on or before the active anchor; the most recently available report is used.
    TEST_TYPE != RAPIDHEME_CLINICAL.

    `anchor_col` overrides the registry's date column for `anchor`, for callers
    whose landmark is not one of the registered anchors — the biomarker pipeline
    passes its line-specific `treatment_start_date` so a report issued after the
    ICI landmark cannot enter the marker set."""
    cohort_mrns = cohort_df.get_column("DFCI_MRN").unique().to_list()
    anchor_col = anchor_col if anchor_col is not None else date_col(anchor)
    anchor_df = cohort_df.select(["DFCI_MRN", anchor_col])

    genomic_spec = ps.load_genomic_specimen(exclude_rapidheme=True)
    genomic_spec = genomic_spec.filter(pl.col(ps.MRN).is_in(cohort_mrns))

    genomic_spec = genomic_spec.drop_nulls(subset=[ps.REPORT_DT])
    genomic_spec = genomic_spec.join(anchor_df, on="DFCI_MRN", how="inner").filter(
        pl.col(ps.REPORT_DT) <= pl.col(anchor_col)
    )

    selected_sample = (
        genomic_spec.sort(["DFCI_MRN", ps.REPORT_DT], descending=[False, True])
        .group_by("DFCI_MRN", maintain_order=True)
        .agg(pl.all().first())
    )

    somatic_wide = ps.load_somatic_wide()
    complete_somatic = selected_sample.join(somatic_wide, on=ps.SOMATIC_GROUP_KEY, how="inner")

    metadata_cols = selected_sample.columns
    # SOMATIC_WIDE_BY_SAMPLE also carries specimen metadata such as
    # SAMPLE_COLLECTION_DT. Keep the selected specimen's metadata once and
    # append only genuinely new somatic feature columns.
    metadata_col_set = set(metadata_cols)
    feature_cols = [column for column in somatic_wide.columns if column not in metadata_col_set]

    complete_somatic = complete_somatic.select(metadata_cols + feature_cols)
    return complete_somatic


def build_germline_data_df(cohort_df: pl.DataFrame) -> pl.DataFrame:
    """PRS scores unchanged (mjsaleh TSV), bridged to DFCI_MRN via
    PROFILE_2024_idmap.csv joined against the new cohort rather than
    px_metadata_min. PRS coverage is limited to MRNs in the 2024 idmap, so
    this modality covers fewer patients than the others.

    Patients with multiple mapped sample IDs get per-PGS-column averages
    across their samples (see the conflict handling below), not a single
    sample's raw values."""
    idmap = pl.read_csv(
        os.path.join(PROFILE_PATH, "PROFILE_2024_idmap.csv"),
        columns=["DFCI_MRN", "cbio_sample_id"],
    ).with_columns(
        pl.col("DFCI_MRN").cast(pl.Int64, strict=False),
        pl.col("cbio_sample_id").cast(pl.String),
    )
    cohort_idmap = idmap.filter(pl.col("DFCI_MRN").is_in(cohort_df.get_column("DFCI_MRN")))
    joined = (
        pl.read_csv(PRS_MATRIX_FILE, separator="\t")
        .rename({"IID": "cbio_sample_id"})
        .with_columns(pl.col("cbio_sample_id").cast(pl.String))
        .join(cohort_idmap, on="cbio_sample_id", how="inner")
    )
    pgs_cols = [column for column in joined.columns if "PGS" in column]
    if not pgs_cols:
        raise ValueError(f"No PGS columns found in {PRS_MATRIX_FILE}")

    # Multiple PROFILE sample IDs may map to one patient (e.g. resequenced or
    # relabeled samples), and their PGS values do not always agree. Rather
    # than picking one sample's values arbitrarily, average each PGS column
    # across a patient's samples (nulls excluded, matching the pre-average
    # conflict check this replaced). Non-PGS columns still take first() per
    # patient, since averaging a sample ID or other identifier is meaningless.
    other_cols = [column for column in joined.columns if column not in pgs_cols and column != "DFCI_MRN"]
    return (
        joined.sort(["DFCI_MRN", "cbio_sample_id"])
        .group_by("DFCI_MRN", maintain_order=True)
        .agg(
            [pl.col(column).first() for column in other_cols]
            + [pl.col(column).mean().alias(column) for column in pgs_cols]
        )
    )


def build_treatment_by_line_df(
    cohort_df: pl.DataFrame, anchor: str = DEFAULT_ANCHOR
) -> pl.DataFrame:
    """Patient-level treatment classes started on or before the anchor."""
    med_classes = pl.read_csv(MED_CLASSES_FILE).unique("MED_NAME", keep="last")
    long = ps.unpivot_medications_summary().rename({"SLOT": "treatment_line", "DRUG": "MED_NAME"})
    anchor_col = date_col(anchor)
    long = (
        long.sort(["DFCI_MRN", "START_DT"])
        .join(med_classes, on="MED_NAME", how="left")
        .join(cohort_df.select(["DFCI_MRN", anchor_col]), on="DFCI_MRN", how="inner")
        .filter(pl.col("START_DT") <= pl.col(anchor_col))
        .with_columns(pl.col("MOA_Category").fill_null("OTHER").alias("PX_on"))
        .to_dummies(columns="PX_on")
        .rename({"START_DT": "treatment_start_date"})
    )
    px_cols = [c for c in long.columns if c.startswith("PX_on_")]
    aggregated = long.group_by("DFCI_MRN").agg(
        # Preserve the legacy one-row-per-patient, treatment_line==1 contract
        # consumed by figures/trajectory code.  The row now represents the
        # complete anchor-observable treatment history.
        pl.lit(1).alias("treatment_line"),
        pl.col("treatment_start_date").max(),
        *[pl.col(c).max().alias(c) for c in px_cols],
    )
    # Absence of pre-anchor treatment is a real zero, not missingness.  This is
    # especially important at sequencing, where requiring a treatment row
    # would condition the comparison cohort on subsequent treatment uptake.
    return (
        cohort_df.filter(pl.col(anchor_col).is_not_null())
        .select("DFCI_MRN").unique()
        .join(aggregated, on="DFCI_MRN", how="left")
        .with_columns(
            pl.col("treatment_line").fill_null(1).cast(pl.Int64),
            *[pl.col(c).fill_null(0).cast(pl.Int64) for c in px_cols],
        )
    )


def _feature_path(filename: str, anchor: str) -> str:
    stem, ext = filename.split(".csv", 1)
    return os.path.join(FEATURE_PATH, f"{stem}{anchor_suffix(anchor)}.csv{ext}")


def build_met_burden_df(cohort_df: pl.DataFrame, anchor: str = DEFAULT_ANCHOR) -> pl.DataFrame:
    """Count of distinct pre-index metastatic organ groups (C77-C79 ICD-10
    codes, see shared.icd10.met_site_group), plus a per-group 0/1 indicator.

    Pre-index = START_DT <= the active anchor's date, recomputed here rather
    than trusting the on-disk TIME_TO_ICD (which is locked to the treatment
    anchor regardless of `anchor`; see extract_ICD_times.py and the same
    recompute in generate_embedding_prediction_datasets.py). Zero-filled to
    every cohort patient, so this file has 100% cohort coverage by
    construction - see the "do not add to _get_common_feature_mrns" note in
    slurm_array_utils.py.

    LEAKAGE WARNING: MET_SITE_{site} describes the same anatomy as the
    death_met scheme's `{site}M` outcome events, observed through a
    different instrument (ICD coding vs. the external clinical_to_{site}_met
    extraction). Training code removes both the concordant indicator and its
    contribution to the aggregate count for that event.

    The output is routed per anchor, so sequencing recomputes the ICD window
    relative to sequencing and writes an anchor-suffixed file.
    """
    cohort_mrns = cohort_df.get_column("DFCI_MRN").unique().to_list()
    anchor_date_col = date_col(anchor)

    icd_long = pl.read_parquet(
        os.path.join(SURV_PATH, "timestamped_icd_info.parquet"),
        columns=["DFCI_MRN", "START_DT", "DIAGNOSIS_ICD10_CD"],
    ).filter(pl.col("DFCI_MRN").is_in(cohort_mrns))

    icd_long = icd_long.join(
        cohort_df.select(["DFCI_MRN", anchor_date_col]),
        on="DFCI_MRN", how="left",
    ).with_columns(
        pl.col("START_DT").cast(pl.Datetime, strict=False),
        pl.col(anchor_date_col).cast(pl.Datetime, strict=False),
    ).with_columns(
        (pl.col("START_DT") - pl.col(anchor_date_col)).dt.total_days().alias("days_to_icd")
    ).filter(pl.col("days_to_icd").is_not_null() & (pl.col("days_to_icd") <= 0))

    site_df = icd_long.with_columns(
        pl.col("DIAGNOSIS_ICD10_CD").map_elements(met_site_group, return_dtype=pl.String).alias("site")
    ).drop_nulls("site").select(["DFCI_MRN", "site"]).unique()

    site_indicators = site_df.with_columns(pl.lit(1).cast(pl.Int32).alias("present")).pivot(
        on="site", index="DFCI_MRN", values="present"
    )
    # pivot only emits columns for groups actually observed in the cohort;
    # add any missing group explicitly so the column set is never data-dependent.
    for group in MET_SITE_GROUPS:
        if group not in site_indicators.columns:
            site_indicators = site_indicators.with_columns(pl.lit(0).cast(pl.Int32).alias(group))
    site_indicators = site_indicators.select(
        ["DFCI_MRN"] + [pl.col(group).fill_null(0).cast(pl.Int32) for group in MET_SITE_GROUPS]
    ).rename({group: f"MET_SITE_{group}" for group in MET_SITE_GROUPS})

    site_indicators = site_indicators.with_columns(
        pl.sum_horizontal([f"MET_SITE_{group}" for group in MET_SITE_GROUPS]).cast(pl.Int32).alias("N_MET_SITES")
    )

    met_burden_df = (
        cohort_df.select(pl.col("DFCI_MRN").unique())
        .join(site_indicators, on="DFCI_MRN", how="left")
    )
    fill_cols = ["N_MET_SITES"] + [f"MET_SITE_{group}" for group in MET_SITE_GROUPS]
    met_burden_df = met_burden_df.with_columns(
        [pl.col(c).fill_null(0).cast(pl.Int32) for c in fill_cols]
    )
    return met_burden_df.select(["DFCI_MRN"] + fill_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", choices=sorted(ANCHORS), default=DEFAULT_ANCHOR)
    args = parser.parse_args()
    anchor = args.anchor
    os.makedirs(FEATURE_PATH, exist_ok=True)

    cohort_df = _load_cohort_df()

    cancer_type_df = build_cancer_type_df(cohort_df)
    assert_schema(cancer_type_df, "cancer_type_df", required_cols=["DFCI_MRN", "CANCER_TYPE"], key_col="DFCI_MRN")
    cancer_type_df.write_csv(_feature_path('cancer_type_df.csv.gz', anchor), compression="gzip")

    cancer_stage_df = build_cancer_stage_df()
    assert_schema(cancer_stage_df, "cancer_stage_df", required_cols=["DFCI_MRN", "CANCER_STAGE"], key_col="DFCI_MRN")
    cancer_stage_df.write_csv(_feature_path('cancer_stage_df.csv.gz', anchor), compression="gzip")

    complete_somatic_data_df = build_somatic_data_df(cohort_df, anchor)
    assert_schema(complete_somatic_data_df, "complete_somatic_data_df", required_cols=["DFCI_MRN"], key_col="DFCI_MRN")
    complete_somatic_data_df.write_csv(
        _feature_path('complete_somatic_data_df.csv.gz', anchor), compression="gzip"
    )

    complete_germline_data_df = build_germline_data_df(cohort_df)
    assert_schema(
        complete_germline_data_df,
        "complete_germline_data_df",
        required_cols=["DFCI_MRN"],
        key_col="DFCI_MRN",
        unique_key=True,
    )
    complete_germline_data_df.write_csv(
        os.path.join(FEATURE_PATH, 'complete_germline_data_df.csv.gz'), compression="gzip"
    )

    categorical_treatment_data_by_line = build_treatment_by_line_df(cohort_df, anchor)
    assert_schema(
        categorical_treatment_data_by_line,
        "categorical_treatment_data_by_line",
        required_cols=["DFCI_MRN", "treatment_line"],
        key_col="DFCI_MRN",
    )
    categorical_treatment_data_by_line.write_csv(
        _feature_path('categorical_treatment_data_by_line.csv.gz', anchor), compression="gzip"
    )

    met_burden_df = build_met_burden_df(cohort_df, anchor)
    assert_schema(
        met_burden_df,
        "met_burden_df",
        required_cols=["DFCI_MRN", "N_MET_SITES"] + [f"MET_SITE_{s}" for s in MET_SITE_GROUPS],
        key_col="DFCI_MRN",
    )
    met_burden_df.write_csv(_feature_path('met_burden_df.csv.gz', anchor), compression="gzip")


if __name__ == "__main__":
    main()
