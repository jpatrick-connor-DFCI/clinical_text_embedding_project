"""Unpack multi-slot ICD-10 diagnoses into long format and time them relative to first treatment."""

import os

import pandas as pd
import polars as pl

from config import SURV_PATH
from data.schema import assert_schema
from pipelines.preprocessing import profile_sources as ps


def main() -> None:
    cohort_df = pd.read_parquet(os.path.join(SURV_PATH, "cohort_df.parquet"))
    cohort_mrns = cohort_df["DFCI_MRN"].unique().tolist()
    mrn_tstart_dict = dict(zip(cohort_df["DFCI_MRN"], cohort_df["first_treatment_date"]))

    df_long = ps.load_and_explode_icd()
    df_long = df_long.filter(pl.col("DFCI_MRN").is_in(cohort_mrns))

    split_ehr_icd_subset = df_long.select(
        ["DFCI_MRN", "START_DT", "DIAGNOSIS_ICD10_CD", "DIAGNOSIS_ICD10_NM"]
    ).to_pandas()

    split_ehr_icd_subset["FIRST_TREATMENT_START_DT"] = split_ehr_icd_subset["DFCI_MRN"].map(mrn_tstart_dict)
    split_ehr_icd_subset["START_DT"] = pd.to_datetime(split_ehr_icd_subset["START_DT"])
    split_ehr_icd_subset["FIRST_TREATMENT_START_DT"] = pd.to_datetime(split_ehr_icd_subset["FIRST_TREATMENT_START_DT"])
    split_ehr_icd_subset["TIME_TO_ICD"] = (
        split_ehr_icd_subset["START_DT"] - split_ehr_icd_subset["FIRST_TREATMENT_START_DT"]
    ).dt.days

    assert_schema(
        split_ehr_icd_subset,
        "timestamped_icd_info",
        required_cols=[
            "DFCI_MRN",
            "START_DT",
            "DIAGNOSIS_ICD10_CD",
            "DIAGNOSIS_ICD10_NM",
            "FIRST_TREATMENT_START_DT",
            "TIME_TO_ICD",
        ],
        key_col=None,
    )

    split_ehr_icd_subset.to_parquet(os.path.join(SURV_PATH, 'timestamped_icd_info.parquet'), index=False)


if __name__ == "__main__":
    main()
