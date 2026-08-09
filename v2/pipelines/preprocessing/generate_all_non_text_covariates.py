"""Build non-text clinical, genomic, and treatment covariates for the v2 cohort.

Split into one function per feature file. All five keep their existing
filenames and column contracts (CANCER_TYPE_*, CANCER_STAGE_*, *_AMP/_DEL/
_SNV/_SV/_FUSION, PGS*, PX_on_*) so `pipelines/training/slurm_array_utils.py`
needs zero changes to keep selecting features by name pattern.

Cancer type and stage are TODO placeholders derived from GENOMIC_SPECIMEN/
CAREG until a compiled version is available; the column contracts are fixed
now so that swap will be a loader change only.
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
    """CANCER_TYPE from GENOMIC_SPECIMEN.CANCER_TYPE (patient-level, from the
    selected sample), falling back to a CAREG.SITE_CD site grouping for
    non-sequenced patients. # TODO: replace with compiled cancer type."""
    cohort_mrns = cohort_df["DFCI_MRN"].unique().tolist()

    genomic_spec = ps.load_genomic_specimen()
    genomic_spec = genomic_spec.filter(pl.col(ps.MRN).is_in(cohort_mrns))
    # One CANCER_TYPE per patient: take the first non-null value per MRN.
    genomic_type = (
        genomic_spec.select([ps.MRN, ps.CANCER_TYPE])
        .drop_nulls()
        .group_by(ps.MRN)
        .agg(pl.col(ps.CANCER_TYPE).first())
    )

    careg = ps.load_careg(columns=[ps.MRN, ps.SITE_CD])
    careg_type = (
        careg.filter(pl.col(ps.MRN).is_in(cohort_mrns))
        .select([ps.MRN, pl.col(ps.SITE_CD).alias(ps.CANCER_TYPE)])
        .drop_nulls()
        .group_by(ps.MRN)
        .agg(pl.col(ps.CANCER_TYPE).first())
    )

    covered_mrns = set(genomic_type[ps.MRN].to_list())
    fallback = careg_type.filter(~pl.col(ps.MRN).is_in(list(covered_mrns)))

    cancer_type_df = pl.concat([genomic_type, fallback], how="vertical").to_pandas()

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
    """CANCER_STAGE from CAREG.BEST_AJCC_STAGE_CD, dropping the '88'/'99'
    sentinels, one row per patient by earliest DIAGNOSIS_DT, normalized via
    shared.stages.normalize_stage(). Also emits the raw CANCER_STAGE string
    column so figure prep no longer needs a drop_first reconstruction
    fallback, and load_stage_map() can read this file directly.
    # TODO: replace with compiled cancer stage."""
    careg = ps.load_careg(columns=[ps.MRN, ps.DIAGNOSIS_DT, ps.BEST_AJCC_STAGE_CD])
    careg = careg.filter(
        (~pl.col(ps.BEST_AJCC_STAGE_CD).is_in(list(ps.STAGE_SENTINELS)))
        & pl.col(ps.BEST_AJCC_STAGE_CD).is_not_null()
    )
    earliest_stage = (
        careg.sort([ps.MRN, ps.DIAGNOSIS_DT])
        .group_by(ps.MRN, maintain_order=True)
        .agg(pl.col(ps.BEST_AJCC_STAGE_CD).first())
        .to_pandas()
    )

    earliest_stage["CANCER_STAGE"] = earliest_stage[ps.BEST_AJCC_STAGE_CD].map(normalize_stage)
    earliest_stage = earliest_stage.dropna(subset=["CANCER_STAGE"])
    earliest_stage = earliest_stage.rename(columns={ps.MRN: "DFCI_MRN"})[["DFCI_MRN", "CANCER_STAGE"]]

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


def build_mean_lab_vals_df(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """LABS.parquet: top-40 TEST_TYPE_DESCR, the 9999999.00 sentinel drop,
    mean/std, and the _missing indicator block. The _missing suffix is
    load-bearing: run_feature_comp_task.py treats a column as imputable only
    if a sibling {col}_missing exists. LABS is absent from the 2026_03
    release, so lab coverage ends with 2025_03 data."""
    cohort_mrns = cohort_df["DFCI_MRN"].unique().tolist()
    labs_df = ps.load_labs(
        columns=[ps.MRN, "SPECIMEN_COLLECT_DT", "TEST_TYPE_DESCR", "NUMERIC_RESULT"]
    ).to_pandas()
    labs_df = labs_df.loc[labs_df["DFCI_MRN"].isin(cohort_mrns)]

    test_count_df = labs_df['TEST_TYPE_DESCR'].value_counts().reset_index()
    test_count_df['rank_index'] = range(len(test_count_df))
    tests_to_include = test_count_df.loc[test_count_df['rank_index'] < 40]

    lab_subset_df = (
        labs_df.merge(tests_to_include['TEST_TYPE_DESCR'], on=['TEST_TYPE_DESCR'], how='inner')
        .merge(cohort_df[['DFCI_MRN', 'first_treatment_date']], on='DFCI_MRN')
    )
    lab_subset_df = lab_subset_df.loc[lab_subset_df['NUMERIC_RESULT'] != 9999999.00].dropna()

    lab_subset_df['LAB_TIME_REL_FIRST_TREATMENT_START'] = (
        pd.to_datetime(lab_subset_df['SPECIMEN_COLLECT_DT']) - pd.to_datetime(lab_subset_df['first_treatment_date'])
    ).dt.days

    mean_lab_df = (
        lab_subset_df.loc[lab_subset_df['LAB_TIME_REL_FIRST_TREATMENT_START'] < 0]
        .groupby(['DFCI_MRN', 'TEST_TYPE_DESCR'])['NUMERIC_RESULT']
        .agg(['mean', 'std'])
        .reset_index()
        .pivot(index='DFCI_MRN', columns='TEST_TYPE_DESCR')
    )

    mean_lab_df.columns = [f'{lab}_{stat}' for lab, stat in mean_lab_df.columns]
    mean_lab_df = mean_lab_df.reset_index()

    X = mean_lab_df.drop(columns=['DFCI_MRN']).astype('float32')

    # Save raw (unimputed) lab values alongside missingness indicators.
    # Mean imputation is deferred to per-fold preprocessing in cox_models.py
    # to avoid leaking test-set statistics into the imputed training features.
    final_mean_lab_df = pd.concat(
        [X, X.isna().astype('int8').add_suffix('_missing')],
        axis=1
    )
    final_mean_lab_df.insert(0, 'DFCI_MRN', mean_lab_df['DFCI_MRN'])
    return final_mean_lab_df


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

    mean_lab_vals_df = build_mean_lab_vals_df(cohort_df)
    assert_schema(mean_lab_vals_df, "mean_lab_vals_pre_first_treatment", required_cols=["DFCI_MRN"], key_col="DFCI_MRN")
    mean_lab_vals_df.to_csv(os.path.join(FEATURE_PATH, 'mean_lab_vals_pre_first_treatment.csv.gz'), index=False)


if __name__ == "__main__":
    main()
