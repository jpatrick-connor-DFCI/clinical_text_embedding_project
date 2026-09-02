"""Build the cohort: index date, demographics, and survival outcomes.

Replaces `follow_up_vte_df_cohort.csv` (Intae's VTE project) as the source of
cohort membership, index date, age, sex, and survival outcomes. The cohort is
now defined from PROFILE_DATA directly rather than inherited from an
externally-owned project, so it is not expected to match the old VTE cohort
(it is larger and drops the vte/tt_vte endpoint, which has no PROFILE_DATA
source).

Carries both time-zero anchors (see anchors.py): `treatment`
(first_treatment_date, the original/default) and `sequencing` (earliest
genomic-specimen date, the OS-anchor-sensitivity arm). Every
treatment-anchor column name and value is
unchanged from before this anchor split — only new sequencing-anchor columns
are additive.

Writes:
  - SURV_PATH/cohort_df.parquet
  - SURV_PATH/cohort_attrition.json
"""

import json
import os

import polars as pl

from config import SURV_PATH
try:
    from data.schema import assert_schema
except ModuleNotFoundError:
    from pipelines.preprocessing.schema import assert_schema
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


def _sequencing_dates() -> pl.DataFrame:
    """Per-MRN earliest genomic report date.

    Somatic results are not available at specimen collection or test order.
    Using the earliest of those timestamps made the sequencing arm start
    before its defining features were observable.  REPORT_DT is therefore
    the availability timestamp and the only valid sequencing anchor.
    """
    genomic_spec = ps.load_genomic_specimen(exclude_rapidheme=True)
    genomic_spec = genomic_spec.with_columns(
        pl.col(ps.REPORT_DT).alias("sequencing_date")
    ).drop_nulls(subset=["sequencing_date"])

    earliest_per_patient = (
        genomic_spec.sort(["DFCI_MRN", "sequencing_date"])
        .group_by("DFCI_MRN", maintain_order=True)
        .agg(pl.col("sequencing_date").first())
    )
    return earliest_per_patient


def _pre_index_note_mrns(cohort_mrns: pl.Series, anchor_dates_df: pl.DataFrame, date_col: str) -> set:
    """MRNs with >= 1 clinical note strictly before their anchor date,
    across the three note-metadata parquets. EVENT_DATE is tz-aware UTC (the
    only tz-aware column in PROFILE_DATA) and must be localized to naive
    before comparing against the (naive) anchor date."""
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
            .join(anchor_dates_df.select(["DFCI_MRN", date_col]), on="DFCI_MRN", how="inner")
            .filter(pl.col("_EVENT_DATE_NAIVE") < pl.col(date_col))
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
    sequencing_df = _sequencing_dates()
    attrition["patients_with_sequencing_date"] = sequencing_df.height

    treatment_mrns = first_treatment_df["DFCI_MRN"]
    mrns_with_pre_index_note = _pre_index_note_mrns(
        treatment_mrns, first_treatment_df, "first_treatment_date"
    )
    attrition["patients_with_pre_index_note"] = len(mrns_with_pre_index_note)

    sequencing_mrns = sequencing_df["DFCI_MRN"]
    mrns_with_pre_seq_note = _pre_index_note_mrns(
        sequencing_mrns, sequencing_df, "sequencing_date"
    )
    attrition["patients_with_pre_sequencing_note"] = len(mrns_with_pre_seq_note)

    # Build the union before applying anchor-specific eligibility.  Starting
    # from treated patients made eventual treatment a future inclusion
    # criterion for the sequencing arm.
    candidate_mrns = sorted(set(treatment_mrns.to_list()) | set(sequencing_mrns.to_list()))
    cohort_df = (
        pl.DataFrame({"DFCI_MRN": candidate_mrns}, schema={"DFCI_MRN": pl.Int64})
        .join(first_treatment_df, on="DFCI_MRN", how="left")
        .join(sequencing_df, on="DFCI_MRN", how="left")
    )

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
        (pl.col("_follow_up_end_date") - pl.col("sequencing_date")).dt.total_days().alias("tt_death_sequencing"),
    )
    cohort_df = cohort_df.drop("_follow_up_end_date")

    # Audit medication records after death; anchor-specific eligibility below
    # excludes invalid treatment anchors without removing sequencing-only rows.
    try:
        qc_df = ps.load_medication_after_death_qc()
        qc_mrns = set(qc_df[ps.MRN].to_list())
        attrition["mrns_flagged_in_medication_after_death_qc"] = len(qc_mrns)
    except FileNotFoundError:
        attrition["mrns_flagged_in_medication_after_death_qc"] = 0

    # tt_death / AGE_AT_TREATMENTSTART remain treatment-anchor aliases for
    # backwards compatibility; they may be null on sequencing-only rows.
    cohort_df = cohort_df.with_columns(
        pl.col("tt_death").alias("tt_death_treatment"),
        (
            pl.col("first_treatment_date").is_not_null()
            & pl.col("DFCI_MRN").is_in(list(mrns_with_pre_index_note))
            & (pl.col("tt_death") > 0)
            & (pl.col("death_date").is_null() | (pl.col("first_treatment_date") <= pl.col("death_date")))
        ).fill_null(False).alias("eligible_treatment"),
    )

    # === Sequencing anchor ===
    seq_attrition: dict[str, int] = {
        "patients_with_sequencing_date": sequencing_df.height,
        "patients_with_pre_sequencing_note": len(mrns_with_pre_seq_note),
    }
    cohort_df = cohort_df.with_columns(
        (
            (pl.col("sequencing_date") - pl.col(ps.BIRTH_DT)).dt.total_days() / 365.2425
        ).alias("AGE_AT_SEQUENCING"),
    )

    cohort_df = cohort_df.with_columns(
        (
            pl.col("sequencing_date").is_not_null()
            & pl.col("DFCI_MRN").is_in(list(mrns_with_pre_seq_note))
            & (pl.col("tt_death_sequencing") > 0)
            & (pl.col("death_date").is_null() | (pl.col("sequencing_date") <= pl.col("death_date")))
        ).fill_null(False).alias("eligible_sequencing"),
    )

    seq_attrition["eligible_sequencing_count"] = int(cohort_df["eligible_sequencing"].sum())
    seq_attrition["eligible_treatment_count"] = int(cohort_df["eligible_treatment"].sum())
    seq_attrition["eligible_both_anchors_count"] = int(
        (cohort_df["eligible_treatment"] & cohort_df["eligible_sequencing"]).sum()
    )
    attrition["sequencing_anchor"] = seq_attrition

    # Rows eligible for neither analysis are not part of the modeled cohort.
    cohort_df = cohort_df.filter(pl.col("eligible_treatment") | pl.col("eligible_sequencing"))
    attrition["final_union_cohort_size"] = cohort_df.height

    cohort_out = cohort_df.select(
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
            "tt_death_treatment",
            "eligible_treatment",
            "sequencing_date",
            "AGE_AT_SEQUENCING",
            "tt_death_sequencing",
            "eligible_sequencing",
        ]
    )

    assert_schema(
        cohort_out,
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
    cohort_out.write_parquet(out_path)

    attrition_path = os.path.join(SURV_PATH, "cohort_attrition.json")
    with open(attrition_path, "w") as f:
        f.write(json.dumps(attrition, indent=2))

    print(f"Saved cohort ({cohort_out.height} patients) to {out_path}")
    print(f"Saved attrition counts to {attrition_path}")
    print(json.dumps(attrition, indent=2))


if __name__ == "__main__":
    main()
