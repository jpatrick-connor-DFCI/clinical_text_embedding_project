"""Build non-text clinical, genomic, and treatment covariates for the v2 cohort.

Split into one function per feature file. All five keep their existing
filenames and column contracts (CANCER_TYPE_*, CANCER_STAGE_*, *_AMP/_DEL/
_SNV/_SV/_FUSION, PGS*, PX_on_*) so `pipelines/training/slurm_array_utils.py`
needs zero changes to keep selecting features by name pattern.

Cancer type and stage come from PROFILE_data_processing's compiled
CANCER_ANNOTATIONS parquets (CANCER_TYPE.parquet, CANCER_STAGE.parquet,
note-regex derived), not the GENOMIC_SPECIMEN/CAREG placeholders used
previously.
"""

import os

import pandas as pd
import polars as pl

from config import FEATURE_PATH, MED_CLASSES_FILE, PRS_MATRIX_FILE, PROFILE_PATH, SURV_PATH
from data.schema import assert_schema
from pipelines.preprocessing import profile_sources as ps
from shared.stages import normalize_stage


def _load_cohort_df() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(SURV_PATH, "cohort_df.parquet"))


def build_cancer_type_df(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """CANCER_TYPE from the compiled CANCER_TYPE.parquet's CANCER_GROUP column
    (genomics-first, ICD-fallback; see PROFILE_data_processing/
    CANCER_ANNOTATIONS_PLAN.md), restricted to the cohort and renamed to the
    existing CANCER_TYPE column contract."""
    cohort_mrns = cohort_df["DFCI_MRN"].unique().tolist()

    cancer_type = (
        ps.load_cancer_type()
        .select([ps.MRN, pl.col(ps.CANCER_GROUP).alias("CANCER_TYPE")])
        .filter(pl.col(ps.MRN).is_in(cohort_mrns))
        .drop_nulls()
    )

    cancer_type_df = cancer_type.to_pandas()

    cancer_type_counts = cancer_type_df['CANCER_TYPE'].value_counts()
    types_to_keep = cancer_type_counts[cancer_type_counts >= 500].index.tolist()
    cancer_type_df['CANCER_TYPE'] = cancer_type_df['CANCER_TYPE'].where(
        cancer_type_df['CANCER_TYPE'].isin(types_to_keep), 'OTHER'
    )
    # Preserve the raw (collapsed) label alongside the dummies. `drop_first=True`
    # removes the reference category's column, so patients in that category are
    # all-zero across CANCER_TYPE_* and would be mislabeled by a downstream idxmax.
    # Keeping the string label lets consumers recover the true type directly. The
    # column is named 'CANCER_TYPE' (no trailing underscore) so it is never picked
    # up by `startswith('CANCER_TYPE_')` feature-column filters.
    _raw_cancer_type = cancer_type_df['CANCER_TYPE'].copy()
    cancer_type_df = pd.get_dummies(cancer_type_df, columns=['CANCER_TYPE'], drop_first=True)
    cancer_type_df.insert(1, 'CANCER_TYPE', _raw_cancer_type.values)
    return cancer_type_df


def build_cancer_stage_df() -> pd.DataFrame:
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
        .to_pandas()
    )

    note_stage["CANCER_STAGE"] = note_stage[ps.STAGE].map(normalize_stage)
    earliest_stage = note_stage.dropna(subset=["CANCER_STAGE"])[["DFCI_MRN", "CANCER_STAGE"]]

    _raw_cancer_stage = earliest_stage["CANCER_STAGE"].copy()
    cancer_stage_df = pd.get_dummies(earliest_stage, columns=["CANCER_STAGE"], drop_first=True)
    cancer_stage_df.insert(1, "CANCER_STAGE", _raw_cancer_stage.values)
    return cancer_stage_df


def build_somatic_data_df(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """SOMATIC_WIDE_BY_SAMPLE.parquet joined to GENOMIC_SPECIMEN on the 3-column
    key (never UNIQUE_SAMPLE_ID). Sequencing date = min(SAMPLE_COLLECTION_DT,
    TEST_ORDER_DT, REPORT_DT); keep days_from_sequencing_to_first_treatment
    >= 0; pick the argmin per patient. TEST_TYPE != RAPIDHEME_CLINICAL."""
    cohort_mrns = cohort_df["DFCI_MRN"].unique().tolist()
    tstart_df = pl.from_pandas(cohort_df[["DFCI_MRN", "first_treatment_date"]])

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

    metadata_cols = [c for c in selected_sample.columns]
    feature_cols = [c for c in somatic_wide.columns if c not in ps.SOMATIC_GROUP_KEY]

    complete_somatic = complete_somatic.select(metadata_cols + feature_cols)
    return complete_somatic.to_pandas()


def build_germline_data_df(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """PRS scores unchanged (mjsaleh TSV), bridged to DFCI_MRN via
    PROFILE_2024_idmap.csv joined against the new cohort rather than
    px_metadata_min. PRS coverage is limited to MRNs in the 2024 idmap, so
    this modality covers fewer patients than the others."""
    idmap = pd.read_csv(
        os.path.join(PROFILE_PATH, "PROFILE_2024_idmap.csv"),
        usecols=["DFCI_MRN", "cbio_sample_id"],
    )
    cohort_idmap = idmap.loc[idmap["DFCI_MRN"].isin(cohort_df["DFCI_MRN"])]

    prs_df = (
        pd.read_csv(PRS_MATRIX_FILE, sep="\t")
        .rename(columns={"IID": "cbio_sample_id"})
        .merge(cohort_idmap[["cbio_sample_id", "DFCI_MRN"]], on="cbio_sample_id")
    )
    return prs_df


def build_treatment_by_line_df() -> pd.DataFrame:
    """Build from the unpivoted MEDICATIONS_SUMMARY long frame; slot order
    gives treatment_line (1-7) directly. Maps MED_NCI_PREFERRED_NM through
    MED_CLASSES_FILE to PX_on_* dummies. Drops the MED_LINES_FILE dependency
    and the cumcount() that silently overrode the source's own LINE column."""
    med_classes = pd.read_csv(MED_CLASSES_FILE)
    med_class_dict = dict(zip(med_classes['MED_NAME'], med_classes['MOA_Category']))

    long = ps.unpivot_medications_summary().to_pandas()
    long = long.rename(columns={"SLOT": "treatment_line", "DRUG": "MED_NAME"})
    long = long.sort_values(["DFCI_MRN", "START_DT"])

    classes = long["MED_NAME"].map(med_class_dict).fillna("OTHER")
    dummies = pd.get_dummies(classes, prefix="PX_on", dtype=int)

    treatment_df = pd.concat([long.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    treatment_df = treatment_df.rename(columns={"START_DT": "treatment_start_date"})
    return treatment_df


def main() -> None:
    os.makedirs(FEATURE_PATH, exist_ok=True)

    cohort_df = _load_cohort_df()

    cancer_type_df = build_cancer_type_df(cohort_df)
    assert_schema(cancer_type_df, "cancer_type_df", required_cols=["DFCI_MRN", "CANCER_TYPE"], key_col="DFCI_MRN")
    cancer_type_df.to_csv(os.path.join(FEATURE_PATH, 'cancer_type_df.csv.gz'), index=False)

    cancer_stage_df = build_cancer_stage_df()
    assert_schema(cancer_stage_df, "cancer_stage_df", required_cols=["DFCI_MRN", "CANCER_STAGE"], key_col="DFCI_MRN")
    cancer_stage_df.to_csv(os.path.join(FEATURE_PATH, 'cancer_stage_df.csv.gz'), index=False)

    complete_somatic_data_df = build_somatic_data_df(cohort_df)
    assert_schema(complete_somatic_data_df, "complete_somatic_data_df", required_cols=["DFCI_MRN"], key_col="DFCI_MRN")
    complete_somatic_data_df.to_csv(os.path.join(FEATURE_PATH, 'complete_somatic_data_df.csv.gz'), index=False)

    complete_germline_data_df = build_germline_data_df(cohort_df)
    assert_schema(complete_germline_data_df, "complete_germline_data_df", required_cols=["DFCI_MRN"], key_col=None)
    complete_germline_data_df.to_csv(os.path.join(FEATURE_PATH, 'complete_germline_data_df.csv.gz'), index=False)

    categorical_treatment_data_by_line = build_treatment_by_line_df()
    assert_schema(
        categorical_treatment_data_by_line,
        "categorical_treatment_data_by_line",
        required_cols=["DFCI_MRN", "treatment_line"],
        key_col=None,
    )
    categorical_treatment_data_by_line.to_csv(
        os.path.join(FEATURE_PATH, 'categorical_treatment_data_by_line.csv.gz'), index=False
    )


if __name__ == "__main__":
    main()
