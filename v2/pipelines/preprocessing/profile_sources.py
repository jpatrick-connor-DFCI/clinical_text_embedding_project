"""Typed polars loaders for the PROFILE_DATA parquet ecosystem.

Every PROFILE_DATA column name used anywhere in v2 should be spelled once,
here, as a module-level constant, and every load of a PROFILE_DATA parquet
should go through one of the loaders below. Before this module, column names
were bare string literals repeated at each use site across ~20 files.

PROFILE_DATA parquets are multi-release-deduped (2021_11 -> 2026_03), typed
`DFCI_MRN: Int64` and `pl.Date` throughout (except `EVENT_DATE` in the notes
metadata, which is tz-aware UTC). Loaders return polars DataFrames; callers
convert to pandas at the boundary where downstream v2 code expects it.
"""

import os

import polars as pl

from config import CANCER_ANNOTATIONS_PATH, CLINICAL_NOTES_PATH, PROFILE_DATA_PATH

# --- Shared column-name constants ---
MRN = "DFCI_MRN"

# MEDICATIONS_SUMMARY (wide, one row per patient, 7 fixed regimen slots)
MED_SLOTS = range(1, 8)
MED_NM_COL = "MED_NCI_PREFERRED_NM"
MED_START_COL = "MED_START_DT"
MED_END_COL = "MED_END_DT"
MED_LAST_ADMIN_COL = "MED_LAST_ADMIN_DT"
MED_CATEG_COL = "MED_ANTINEO_DRUG_CATEG"

# EHR_DIAGNOSES (ICD-10 only, 3 code/name slots per row)
ICD_CD = "DIAGNOSIS_ICD10_CD"
ICD_NM = "DIAGNOSIS_ICD10_NM"
ICD_START_DT = "START_DT"
ICD_END_DT = "END_DT"

# PT_INFO_STATUS_REGISTRATION
BIRTH_DT = "BIRTH_DT"
HYBRID_DEATH_DT = "HYBRID_DEATH_DT"
DERIVED_LAST_ALIVE_DATE = "DERIVED_LAST_ALIVE_DATE"
GENDER_NM = "GENDER_NM"

# CAREG
DIAGNOSIS_DT = "DIAGNOSIS_DT"
SITE_CD = "SITE_CD"
BEST_AJCC_STAGE_CD = "BEST_AJCC_STAGE_CD"
STAGE_SENTINELS = ("88", "99")

# GENOMIC_SPECIMEN / SOMATIC_WIDE_BY_SAMPLE
SAMPLE_ACCESSION_NBR = "SAMPLE_ACCESSION_NBR"
TEST_TYPE = "TEST_TYPE"
CANCER_TYPE = "CANCER_TYPE"
SAMPLE_COLLECTION_DT = "SAMPLE_COLLECTION_DT"
TEST_ORDER_DT = "TEST_ORDER_DT"
REPORT_DT = "REPORT_DT"
SOMATIC_GROUP_KEY = [MRN, SAMPLE_ACCESSION_NBR, TEST_TYPE]
RAPIDHEME_TEST_TYPE = "RAPIDHEME_CLINICAL"

# CANCER_ANNOTATIONS (compiled by PROFILE_data_processing/derive_cancer_annotations.ipynb)
CANCER_GROUP = "CANCER_GROUP"
STAGE = "STAGE"

# CLINICAL_NOTES metadata (shared across PROGRESS/PATHOLOGY/IMAGING)
RPT_ID = "RPT_ID"
RPT_TEXT = "RPT_TEXT"
EVENT_DATE = "EVENT_DATE"
PROC_DESC = "PROC_DESC"
RPT_TYPE = "RPT_TYPE"
INP_RPT_TYPE = "INP_RPT_TYPE"
PROVIDER_TYPE = "PROVIDER_TYPE"
ENCOUNTER_TYPE_DESC = "ENCOUNTER_TYPE_DESC"


def _path(filename: str) -> str:
    return os.path.join(PROFILE_DATA_PATH, filename)


def load_medications_summary(columns: list[str] | None = None) -> pl.DataFrame:
    """Wide MEDICATIONS_SUMMARY.parquet, one row per DFCI_MRN."""
    if columns is None:
        return pl.read_parquet(_path("MEDICATIONS_SUMMARY.parquet"))
    return pl.read_parquet(_path("MEDICATIONS_SUMMARY.parquet"), columns=columns)


def load_ehr_diagnoses(columns: list[str] | None = None) -> pl.DataFrame:
    """EHR_DIAGNOSES.parquet, 3 ICD-10 code/name slots per row."""
    if columns is None:
        return pl.read_parquet(_path("EHR_DIAGNOSES.parquet"))
    return pl.read_parquet(_path("EHR_DIAGNOSES.parquet"), columns=columns)


def load_patient_status(columns: list[str] | None = None) -> pl.DataFrame:
    """PT_INFO_STATUS_REGISTRATION.parquet: birth, death, last-alive, gender."""
    if columns is None:
        return pl.read_parquet(_path("PT_INFO_STATUS_REGISTRATION.parquet"))
    return pl.read_parquet(_path("PT_INFO_STATUS_REGISTRATION.parquet"), columns=columns)


def load_careg(columns: list[str] | None = None) -> pl.DataFrame:
    """CAREG.parquet: cancer registry (stage, site, diagnosis date)."""
    if columns is None:
        return pl.read_parquet(_path("CAREG.parquet"))
    return pl.read_parquet(_path("CAREG.parquet"), columns=columns)


def load_genomic_specimen(exclude_rapidheme: bool = True) -> pl.DataFrame:
    """GENOMIC_SPECIMEN.parquet, optionally excluding the heme clinical panel."""
    lf = pl.scan_parquet(_path("GENOMIC_SPECIMEN.parquet"))
    if exclude_rapidheme:
        lf = lf.filter(pl.col(TEST_TYPE) != RAPIDHEME_TEST_TYPE)
    return lf.collect(engine="streaming")


def _cancer_annotations_path(filename: str) -> str:
    return os.path.join(CANCER_ANNOTATIONS_PATH, filename)


def load_cancer_type() -> pl.DataFrame:
    """CANCER_TYPE.parquet: one row per DFCI_MRN, CANCER_GROUP genomics-first
    with ICD fallback (see PROFILE_data_processing/CANCER_ANNOTATIONS_PLAN.md)."""
    return pl.read_parquet(_cancer_annotations_path("CANCER_TYPE.parquet"))


def load_cancer_stage(registry: bool = False) -> pl.DataFrame:
    """CANCER_STAGE.parquet (note-regex derived) by default, or
    CANCER_STAGE_REGISTRY.parquet (CAREG BEST_AJCC_STAGE_CD derived) if
    registry=True. Both are one row per DFCI_MRN with a STAGE column in
    {0, 1, 2, 3, 4}; the two sources are never coalesced upstream."""
    filename = "CANCER_STAGE_REGISTRY.parquet" if registry else "CANCER_STAGE.parquet"
    return pl.read_parquet(_cancer_annotations_path(filename))


def load_somatic_wide() -> pl.DataFrame:
    """SOMATIC_WIDE_BY_SAMPLE.parquet: one row per (MRN, SAMPLE_ACCESSION_NBR, TEST_TYPE),
    already emitting *_SNV/_AMP/_DEL/_SV/_FUSION indicator columns."""
    return pl.read_parquet(_path("SOMATIC_WIDE_BY_SAMPLE.parquet"))


def load_medication_after_death_qc() -> pl.DataFrame:
    """MEDICATION_AFTER_DEATH_QC.parquet: audits (does not remove) medication
    dates recorded after a patient's death date."""
    return pl.read_parquet(_path("MEDICATION_AFTER_DEATH_QC.parquet"))


def _note_metadata_path(note_type: str) -> str:
    return os.path.join(CLINICAL_NOTES_PATH, f"{note_type}.parquet")


def load_note_metadata(note_type: str, columns: list[str] | None = None) -> pl.DataFrame:
    """One of PROGRESS_NOTES.parquet, PATHOLOGY_NOTES.parquet, IMAGING_NOTES.parquet.

    `note_type` is the parquet basename stem, e.g. 'PROGRESS_NOTES'.
    PROGRESS_NOTES.parquet already includes discharge summaries merged in.
    `EVENT_DATE` is tz-aware UTC; localize to naive before comparing to other
    (naive) date columns.
    """
    path = _note_metadata_path(note_type)
    if columns is None:
        return pl.read_parquet(path)
    available_columns = set(pl.scan_parquet(path).collect_schema().names())
    selected_columns = [column for column in columns if column in available_columns]
    return pl.scan_parquet(path).select(selected_columns).collect(engine="streaming")


def unpivot_medications_summary() -> pl.DataFrame:
    """Unpivot the 7 fixed MEDICATIONS_SUMMARY regimen slots to long format:
    (DFCI_MRN, SLOT, DRUG, DRUG_CATEG, START_DT, END_DT, LAST_ADMIN_DT), one row
    per non-null-START_DT slot per patient."""
    needed_cols = [MRN] + [
        f"{prefix}_{i}"
        for i in MED_SLOTS
        for prefix in (MED_NM_COL, MED_START_COL, MED_END_COL, MED_LAST_ADMIN_COL, MED_CATEG_COL)
    ]
    wide = load_medications_summary(columns=needed_cols)

    slot_frames = []
    for i in MED_SLOTS:
        slot_frames.append(
            wide.select(
                pl.col(MRN),
                pl.lit(i).alias("SLOT"),
                pl.col(f"{MED_NM_COL}_{i}").alias("DRUG"),
                pl.col(f"{MED_CATEG_COL}_{i}").alias("DRUG_CATEG"),
                pl.col(f"{MED_START_COL}_{i}").alias("START_DT"),
                pl.col(f"{MED_END_COL}_{i}").alias("END_DT"),
                pl.col(f"{MED_LAST_ADMIN_COL}_{i}").alias("LAST_ADMIN_DT"),
            )
        )

    long = pl.concat(slot_frames, how="vertical").filter(pl.col("START_DT").is_not_null())
    return long


def load_and_explode_icd() -> pl.DataFrame:
    """Melt the 3 ICD-10 code/name slots per EHR_DIAGNOSES row into one code per
    row via concat_list + explode + unnest, dropping rows with an empty code.

    This mirrors the pattern in PROFILE_data_processing/analyze_parquet_files.ipynb
    (cell 1), independently confirmed against the COMPASS repo's
    `load_and_explode_icd()`.
    """
    diagnosis_cols = [
        "DIAGNOSIS_ICD10_CD",
        "DIAGNOSIS_ICD10_NM",
        "DIAGNOSIS_ICD10_CD2",
        "DIAGNOSIS_ICD10_NM2",
        "DIAGNOSIS_ICD10_CD3",
        "DIAGNOSIS_ICD10_NM3",
    ]
    ehr_df = load_ehr_diagnoses()

    df_long = (
        ehr_df
        .with_columns(
            diagnosis=pl.concat_list(
                [
                    pl.struct(
                        pl.col("DIAGNOSIS_ICD10_CD").alias("CD"),
                        pl.col("DIAGNOSIS_ICD10_NM").alias("NM"),
                    ),
                    pl.struct(
                        pl.col("DIAGNOSIS_ICD10_CD2").alias("CD"),
                        pl.col("DIAGNOSIS_ICD10_NM2").alias("NM"),
                    ),
                    pl.struct(
                        pl.col("DIAGNOSIS_ICD10_CD3").alias("CD"),
                        pl.col("DIAGNOSIS_ICD10_NM3").alias("NM"),
                    ),
                ]
            )
        )
        .drop(diagnosis_cols)
        .explode("diagnosis", empty_as_null=True)
        .unnest("diagnosis")
        .rename({"CD": "DIAGNOSIS_ICD10_CD", "NM": "DIAGNOSIS_ICD10_NM"})
        .filter(
            pl.col("DIAGNOSIS_ICD10_CD")
            .fill_null("")
            .str.strip_chars()
            .ne("")
        )
    )
    return df_long
