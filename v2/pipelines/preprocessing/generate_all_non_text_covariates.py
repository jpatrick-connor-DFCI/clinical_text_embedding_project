"""Build non-text clinical, genomic, and treatment covariates for the v2 cohort.

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

import os

import polars as pl

from anchors import DEFAULT_ANCHOR, date_col
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


def build_somatic_data_df(cohort_df: pl.DataFrame) -> pl.DataFrame:
    """SOMATIC_WIDE_BY_SAMPLE.parquet joined to GENOMIC_SPECIMEN on the 3-column
    key (never UNIQUE_SAMPLE_ID). Sequencing date = min(SAMPLE_COLLECTION_DT,
    TEST_ORDER_DT, REPORT_DT); keep days_from_sequencing_to_first_treatment
    >= 0; pick the argmin per patient. TEST_TYPE != RAPIDHEME_CLINICAL."""
    cohort_mrns = cohort_df.get_column("DFCI_MRN").unique().to_list()
    tstart_df = cohort_df.select(["DFCI_MRN", "first_treatment_date"])

    genomic_spec = ps.load_genomic_specimen(exclude_rapidheme=True)
    genomic_spec = genomic_spec.filter(pl.col(ps.MRN).is_in(cohort_mrns))

    genomic_spec = genomic_spec.with_columns(
        pl.min_horizontal([ps.SAMPLE_COLLECTION_DT, ps.TEST_ORDER_DT, ps.REPORT_DT]).alias("SEQUENCING_DT")
    ).drop_nulls(subset=["SEQUENCING_DT"])

    genomic_spec = genomic_spec.join(tstart_df, on="DFCI_MRN", how="inner")
    genomic_spec = genomic_spec.with_columns(
        (pl.col("first_treatment_date") - pl.col("SEQUENCING_DT")).dt.total_days().alias(
            "days_from_sequencing_to_first_treatment"
        )
    ).filter(pl.col("days_from_sequencing_to_first_treatment") >= 0)

    selected_sample = (
        genomic_spec.sort(["DFCI_MRN", "days_from_sequencing_to_first_treatment"])
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
    this modality covers fewer patients than the others."""
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

    # Multiple PROFILE sample IDs may map to one patient. Patient-level PRS
    # values must agree before collapsing those rows; otherwise a random row
    # choice could hide an upstream identity error and duplicate a patient
    # across model folds.
    conflicts = joined.group_by("DFCI_MRN").agg(
        [pl.col(column).drop_nulls().n_unique().alias(column) for column in pgs_cols]
    ).filter(pl.max_horizontal(pgs_cols) > 1)
    if conflicts.height:
        raise ValueError(
            f"PRS values disagree across sample IDs for {conflicts.height} patient(s)."
        )

    return (
        joined.sort(["DFCI_MRN", "cbio_sample_id"])
        .group_by("DFCI_MRN", maintain_order=True)
        .agg(pl.all().first())
    )


def build_treatment_by_line_df() -> pl.DataFrame:
    """Build from the unpivoted MEDICATIONS_SUMMARY long frame; slot order
    gives treatment_line (1-7) directly. Maps MED_NCI_PREFERRED_NM through
    MED_CLASSES_FILE to PX_on_* dummies. Drops the MED_LINES_FILE dependency
    and the cumcount() that silently overrode the source's own LINE column."""
    med_classes = pl.read_csv(MED_CLASSES_FILE).unique("MED_NAME", keep="last")
    long = ps.unpivot_medications_summary().rename({"SLOT": "treatment_line", "DRUG": "MED_NAME"})
    return (
        long.sort(["DFCI_MRN", "START_DT"])
        .join(med_classes, on="MED_NAME", how="left")
        .with_columns(pl.col("MOA_Category").fill_null("OTHER").alias("PX_on"))
        .to_dummies(columns="PX_on")
        .rename({"START_DT": "treatment_start_date"})
    )


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
    extraction). Training code must drop the concordant site indicator per
    event - see the per-event exclusion in run_feature_comp_task.py.

    ANCHOR LIMITATION: only the treatment anchor is validated. Under the
    sequencing anchor, treatment-anchored met burden would include the
    sequencing-to-treatment interval (sequencing generally precedes
    treatment; see build_somatic_data_df's
    days_from_sequencing_to_first_treatment >= 0 filter), which leaks
    post-anchor information. `main()` only calls this with the default
    anchor; a sequencing arm needs a routed, anchor-suffixed output file
    (see anchors.anchor_suffix / _suffixed in
    generate_embedding_prediction_datasets.py), not just passing anchor
    through as-is.
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
    os.makedirs(FEATURE_PATH, exist_ok=True)

    cohort_df = _load_cohort_df()

    cancer_type_df = build_cancer_type_df(cohort_df)
    assert_schema(cancer_type_df, "cancer_type_df", required_cols=["DFCI_MRN", "CANCER_TYPE"], key_col="DFCI_MRN")
    cancer_type_df.write_csv(os.path.join(FEATURE_PATH, 'cancer_type_df.csv.gz'), compression="gzip")

    cancer_stage_df = build_cancer_stage_df()
    assert_schema(cancer_stage_df, "cancer_stage_df", required_cols=["DFCI_MRN", "CANCER_STAGE"], key_col="DFCI_MRN")
    cancer_stage_df.write_csv(os.path.join(FEATURE_PATH, 'cancer_stage_df.csv.gz'), compression="gzip")

    complete_somatic_data_df = build_somatic_data_df(cohort_df)
    assert_schema(complete_somatic_data_df, "complete_somatic_data_df", required_cols=["DFCI_MRN"], key_col="DFCI_MRN")
    complete_somatic_data_df.write_csv(
        os.path.join(FEATURE_PATH, 'complete_somatic_data_df.csv.gz'), compression="gzip"
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

    categorical_treatment_data_by_line = build_treatment_by_line_df()
    assert_schema(
        categorical_treatment_data_by_line,
        "categorical_treatment_data_by_line",
        required_cols=["DFCI_MRN", "treatment_line"],
        key_col=None,
    )
    categorical_treatment_data_by_line.write_csv(
        os.path.join(FEATURE_PATH, 'categorical_treatment_data_by_line.csv.gz'), compression="gzip"
    )

    met_burden_df = build_met_burden_df(cohort_df)
    assert_schema(
        met_burden_df,
        "met_burden_df",
        required_cols=["DFCI_MRN", "N_MET_SITES"] + [f"MET_SITE_{s}" for s in MET_SITE_GROUPS],
        key_col="DFCI_MRN",
    )
    met_burden_df.write_csv(os.path.join(FEATURE_PATH, 'met_burden_df.csv.gz'), compression="gzip")


if __name__ == "__main__":
    main()
