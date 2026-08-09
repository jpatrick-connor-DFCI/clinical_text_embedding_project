"""Unpack multi-slot ICD-10 diagnoses into long format and time them relative to first treatment."""

import os

import polars as pl

try:
    from config import SURV_PATH
    from data.schema import assert_schema
    from pipelines.preprocessing import profile_sources as ps
except ModuleNotFoundError:
    from v2.config import SURV_PATH
    from v2.data.schema import assert_schema
    from v2.pipelines.preprocessing import profile_sources as ps


def main() -> None:
    cohort_df = pl.read_parquet(os.path.join(SURV_PATH, "cohort_df.parquet"))
    cohort_mrns = cohort_df.get_column("DFCI_MRN").unique().to_list()

    df_long = ps.load_and_explode_icd()
    df_long = df_long.filter(pl.col("DFCI_MRN").is_in(cohort_mrns))

    split_ehr_icd_subset = df_long.select(
        ["DFCI_MRN", "START_DT", "DIAGNOSIS_ICD10_CD", "DIAGNOSIS_ICD10_NM"]
    ).join(
        cohort_df.select(["DFCI_MRN", pl.col("first_treatment_date").alias("FIRST_TREATMENT_START_DT")]),
        on="DFCI_MRN", how="left",
    ).with_columns(
        pl.col("START_DT").cast(pl.Datetime, strict=False),
        pl.col("FIRST_TREATMENT_START_DT").cast(pl.Datetime, strict=False),
    ).with_columns(
        (pl.col("START_DT") - pl.col("FIRST_TREATMENT_START_DT")).dt.total_days().alias("TIME_TO_ICD")
    )

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

    split_ehr_icd_subset.write_parquet(os.path.join(SURV_PATH, 'timestamped_icd_info.parquet'))


if __name__ == "__main__":
    main()
