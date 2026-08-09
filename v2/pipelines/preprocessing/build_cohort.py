"""Build the v2 cohort: index date, demographics, and survival outcomes.

Replaces `follow_up_vte_df_cohort.csv` (Intae's VTE project) as the source of
cohort membership, index date, age, sex, and survival outcomes. The cohort is
now defined from PROFILE_DATA directly rather than inherited from an
externally-owned project, so it is not expected to match the old VTE cohort
(it is larger and drops the vte/tt_vte endpoint, which has no PROFILE_DATA
source).

Writes:
  - SURV_PATH/cohort_df.parquet
  - SURV_PATH/cohort_attrition.json
"""

import json
import os

import polars as pl

from config import SURV_PATH
from data.schema import assert_schema
from pipelines.preprocessing import profile_sources as ps


def _first_treatment_dates() -> pl.DataFrame:
    """Per-patient first treatment date + anchor drug/category from the argmin slot."""
    long = ps.unpivot_medications_summary()

    first_per_patient = (
        long.sort(["DFCI_MRN", "START_DT", "SLOT"])
        .group_by("DFCI_MRN", maintain_order=True)
        .agg(pl.all().first())
        .select(
            pl.col("DFCI_MRN"),
            pl.col("START_DT").alias("first_treatment_date"),
            pl.col("DRUG").alias("ANCHOR_DRUG"),
            pl.col("DRUG_CATEG").alias("ANCHOR_DRUG_CATEG"),
        )
    )
    return first_per_patient


def _pre_index_note_mrns(cohort_mrns: pl.Series, first_treatment_df: pl.DataFrame) -> set:
    """MRNs with >= 1 clinical note strictly before their first_treatment_date,
    across the three note-metadata parquets. EVENT_DATE is tz-aware UTC (the
    only tz-aware column in PROFILE_DATA) and must be localized to naive
    before comparing against the (naive) first_treatment_date."""
    mrns_with_note: set = set()
    for note_type in ("PROGRESS_NOTES", "PATHOLOGY_NOTES", "IMAGING_NOTES"):
        meta = ps.load_note_metadata(note_type, columns=[ps.MRN, ps.EVENT_DATE])
        event_date_expr = pl.col(ps.EVENT_DATE)
        if meta.schema[ps.EVENT_DATE] == pl.String:
            event_date_expr = event_date_expr.str.to_datetime(time_zone="UTC", strict=False)
        meta = (
            meta.filter(pl.col(ps.MRN).is_in(cohort_mrns))
            .with_columns(
                event_date_expr
                .dt.replace_time_zone(None)
                .dt.date()
                .alias("_EVENT_DATE_NAIVE")
            )
            .join(first_treatment_df.select(["DFCI_MRN", "first_treatment_date"]), on="DFCI_MRN", how="inner")
            .filter(pl.col("_EVENT_DATE_NAIVE") < pl.col("first_treatment_date"))
        )
        mrns_with_note.update(meta["DFCI_MRN"].unique().to_list())
    return mrns_with_note


def _patient_status() -> pl.DataFrame:
    """Coalesced per-patient demographics/death from PT_INFO_STATUS_REGISTRATION,
    which contains one row per OncDRS release and must be coalesced: stable
    attributes (birth, gender) via first(), volatile/updated attributes
    (death, last-alive) via max(), matching the COMPASS reference pattern."""
    status = ps.load_patient_status(
        columns=[ps.MRN, ps.BIRTH_DT, ps.HYBRID_DEATH_DT, ps.DERIVED_LAST_ALIVE_DATE, ps.GENDER_NM]
    )
    coalesced = status.group_by(ps.MRN).agg(
        pl.col(ps.BIRTH_DT).drop_nulls().first().alias(ps.BIRTH_DT),
        pl.col(ps.GENDER_NM).drop_nulls().first().alias(ps.GENDER_NM),
        pl.col(ps.HYBRID_DEATH_DT).drop_nulls().max().alias(ps.HYBRID_DEATH_DT),
        pl.col(ps.DERIVED_LAST_ALIVE_DATE).drop_nulls().max().alias(ps.DERIVED_LAST_ALIVE_DATE),
    )
    return coalesced


def main() -> None:
    os.makedirs(SURV_PATH, exist_ok=True)
    attrition: dict[str, int] = {}

    first_treatment_df = _first_treatment_dates()
    attrition["patients_with_first_treatment_date"] = first_treatment_df.height

    treatment_mrns = first_treatment_df["DFCI_MRN"]

    mrns_with_pre_index_note = _pre_index_note_mrns(treatment_mrns, first_treatment_df)
    attrition["patients_with_pre_index_note"] = len(mrns_with_pre_index_note)

    cohort_df = first_treatment_df.filter(pl.col("DFCI_MRN").is_in(list(mrns_with_pre_index_note)))
    attrition["cohort_after_treatment_and_note_filter"] = cohort_df.height

    status_df = _patient_status()
    cohort_df = cohort_df.join(status_df, on="DFCI_MRN", how="left")

    cohort_df = cohort_df.with_columns(
        (
            (pl.col("first_treatment_date") - pl.col(ps.BIRTH_DT)).dt.total_days() / 365.2425
        ).alias("AGE_AT_TREATMENTSTART"),
        pl.col(ps.GENDER_NM)
        .map_elements(lambda g: {"MALE": 0, "FEMALE": 1}.get(g), return_dtype=pl.Int64)
        .alias("GENDER"),
        pl.col(ps.HYBRID_DEATH_DT).alias("death_date"),
        pl.col(ps.DERIVED_LAST_ALIVE_DATE).alias("last_contact_date"),
    )

    cohort_df = cohort_df.with_columns(
        pl.col("death_date").is_not_null().cast(pl.Int8).alias("death"),
    )
    cohort_df = cohort_df.with_columns(
        pl.coalesce(["death_date", "last_contact_date"]).alias("_follow_up_end_date"),
    )
    cohort_df = cohort_df.with_columns(
        (pl.col("_follow_up_end_date") - pl.col("first_treatment_date")).dt.total_days().alias("tt_death"),
    )
    cohort_df = cohort_df.drop("_follow_up_end_date")

    # Guard 1: drop patients whose tt_death <= 0 (nonpositive/undefined follow-up).
    n_before_tt_guard = cohort_df.height
    cohort_df = cohort_df.filter(pl.col("tt_death") > 0)
    attrition["dropped_nonpositive_tt_death"] = n_before_tt_guard - cohort_df.height

    # Guard 2: cross-check against MEDICATION_AFTER_DEATH_QC.parquet and drop any
    # patient whose first_treatment_date is after their death_date.
    n_before_death_guard = cohort_df.height
    try:
        qc_df = ps.load_medication_after_death_qc()
        qc_mrns = set(qc_df[ps.MRN].to_list())
        attrition["mrns_flagged_in_medication_after_death_qc"] = len(qc_mrns)
    except FileNotFoundError:
        attrition["mrns_flagged_in_medication_after_death_qc"] = 0

    cohort_df = cohort_df.filter(
        pl.col("death_date").is_null() | (pl.col("first_treatment_date") <= pl.col("death_date"))
    )
    attrition["dropped_post_mortem_index_date"] = n_before_death_guard - cohort_df.height

    attrition["final_cohort_size"] = cohort_df.height

    cohort_pd = cohort_df.select(
        [
            "DFCI_MRN",
            "first_treatment_date",
            "ANCHOR_DRUG",
            "ANCHOR_DRUG_CATEG",
            "AGE_AT_TREATMENTSTART",
            "GENDER",
            "death_date",
            "last_contact_date",
            "death",
            "tt_death",
        ]
    ).to_pandas()

    assert_schema(
        cohort_pd,
        "cohort_df",
        required_cols=[
            "DFCI_MRN",
            "first_treatment_date",
            "AGE_AT_TREATMENTSTART",
            "GENDER",
            "death_date",
            "last_contact_date",
            "death",
            "tt_death",
        ],
        key_col="DFCI_MRN",
        unique_key=True,
    )

    out_path = os.path.join(SURV_PATH, "cohort_df.parquet")
    cohort_pd.to_parquet(out_path, index=False)

    attrition_path = os.path.join(SURV_PATH, "cohort_attrition.json")
    with open(attrition_path, "w") as f:
        f.write(json.dumps(attrition, indent=2))

    print(f"Saved cohort ({cohort_pd.shape[0]} patients) to {out_path}")
    print(f"Saved attrition counts to {attrition_path}")
    print(json.dumps(attrition, indent=2))


if __name__ == "__main__":
    main()
