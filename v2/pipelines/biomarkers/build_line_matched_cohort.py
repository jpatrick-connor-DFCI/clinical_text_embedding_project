"""Build two first-line ICI initiation cohorts for biomarker analysis.

Cohort 1 (first_line_unmatched):
  - All patients observed at line 1
  - Exposure is whether line 1 contains ICI
  - No 1:1 matching — all eligible patients included

Cohort 2 (first_line_matched):
  - The same first-line new-user population
  - 1:1 exact matching on cancer type without replacement

Both cohorts are restricted to patients with a usable somatic profile at their
line-1 landmark (a non-RAPIDHEME genomic specimen reported on or before that
date).  Unsequenced patients contribute no marker to a somatic-biomarker
analysis, and restricting here rather than at the downstream somatic join is
what keeps cohort 2's 1:1 pairs intact.

Neither cohort conditions control eligibility on eventual ICI receipt or on
the maximum line a patient later reaches.  Those post-landmark definitions
were a source of immortal-time/selection leakage in the former design.

Lines of therapy, line start dates, and ICI exposure are derived from
PROFILE_DATA's MEDICATIONS_SUMMARY.parquet via
`pipelines.biomarkers.profile_lines`, not from the lab-owned
ALL_MEDICATION_LINES.csv this module used to read.  Cancer type comes from the
compiled CANCER_TYPE.parquet through the same `build_cancer_type_df` the
preprocessing pipeline uses, rather than from its written-out CSV.

Usage:
  python -m pipelines.biomarkers.build_line_matched_cohort
"""

import os
import random

import numpy as np
import polars as pl

from config import MATCHED_COHORT_PATH, SURV_PATH
from data.schema import assert_schema
from pipelines.biomarkers.profile_lines import derive_lines_of_therapy
from pipelines.preprocessing import profile_sources as ps
from pipelines.preprocessing.generate_all_non_text_covariates import build_cancer_type_df


def match_cohort_1to1(cases_df, controls_df):
    """Exact 1:1 match on (cancer_type, line_category) without replacement.

    Each control patient (DFCI_MRN) can only be used once globally, even if
    they appear in multiple line_category slots.
    """
    matched_cases = []
    matched_controls = []
    used_control_mrns = set()

    for (ctype, lcat), stratum_cases in cases_df.group_by(['cancer_type', 'line_category']):
        stratum_controls = controls_df.filter(
            (pl.col('cancer_type') == ctype) &
            (pl.col('line_category') == lcat) &
            (~pl.col('DFCI_MRN').is_in(used_control_mrns))
        )

        if stratum_controls.is_empty():
            print(f"  WARNING: No controls for ({ctype}, line={lcat}), "
                  f"dropping {len(stratum_cases)} ICI cases")
            continue

        # Shuffle controls for random selection
        stratum_controls = stratum_controls.sample(fraction=1.0, seed=42)

        n_cases = len(stratum_cases)
        n_available = len(stratum_controls)
        n_matchable = min(n_cases, n_available)

        if n_matchable < n_cases:
            sampled_cases = stratum_cases.sample(n=n_matchable, seed=42)
            matched_cases.append(sampled_cases)
            selected_controls = stratum_controls.head(n_matchable)
            matched_controls.append(selected_controls)
            used_control_mrns.update(selected_controls['DFCI_MRN'].to_list())
            print(f"  ({ctype}, line={lcat}): {n_matchable}/{n_cases} cases matched 1:1 "
                  f"(only {n_available} controls available)")
        else:
            matched_cases.append(stratum_cases)
            selected_controls = stratum_controls.head(n_cases)
            matched_controls.append(selected_controls)
            used_control_mrns.update(selected_controls['DFCI_MRN'].to_list())
            print(f"  ({ctype}, line={lcat}): {n_cases} cases x 1 control "
                  f"({n_available} available)")

    if not matched_cases:
        return pl.DataFrame()

    all_cases = pl.concat(matched_cases)
    all_controls = pl.concat(matched_controls)
    return pl.concat([all_cases, all_controls], how='diagonal_relaxed')


def _assert_one_to_one(left, right, on, name="join"):
    key_counts = right.group_by(on).agg(pl.len().alias("_n"))
    if (key_counts["_n"] > 1).any():
        raise ValueError(f"{name}: right side is not unique on {on}")


def build_first_line_new_user_df(
    lines_df: pl.DataFrame,
    cohort_mrns: set,
    cancer_type_map: dict,
) -> pl.DataFrame:
    """Define exposure at line 1 without consulting later treatment history.

    `lines_df` is `profile_lines.derive_lines_of_therapy()` output, already one
    row per (DFCI_MRN, LINE) with HAS_ICI and treatment_start_date resolved, so
    no per-patient collapse is needed here — only the restriction to line 1.
    """
    first_line = (
        lines_df.filter(
            pl.col('DFCI_MRN').is_in(cohort_mrns) & (pl.col('LINE') == 1)
        )
        .select(
            'DFCI_MRN',
            pl.col('HAS_ICI').cast(pl.Int64).alias('PX_on_ICI'),
            pl.lit(1).cast(pl.Int64).alias('line_category'),
            pl.col('DFCI_MRN').replace_strict(cancer_type_map, default=None).alias('cancer_type'),
            pl.col('treatment_start_date').cast(pl.Datetime),
        )
        .drop_nulls(subset=['cancer_type'])
    )
    # derive_lines_of_therapy() is unique on (DFCI_MRN, LINE) by construction;
    # this asserts that rather than assuming it.
    _assert_one_to_one(first_line, first_line, ['DFCI_MRN', 'line_category'])
    return first_line


def sequenced_before_landmark(landmark_df: pl.DataFrame) -> pl.DataFrame:
    """MRNs with a usable somatic profile at their landmark, one row per patient.

    Eligibility is a non-RAPIDHEME genomic specimen whose REPORT_DT is on or
    before the patient's line-1 start date — exactly the criterion
    `build_somatic_data_df` applies when it selects each patient's marker set.
    Applying it here, before matching, is what keeps the 1:1 pairs intact: a
    patient without a pre-landmark specimen has no markers and would otherwise
    be dropped by generate_IPTW_df's somatic inner join *after* being matched,
    orphaning their partner.

    Args:
        landmark_df: DFCI_MRN plus `treatment_start_date` (the line-1 landmark).

    Returns:
        DFCI_MRN and `sequencing_date` (the latest eligible REPORT_DT).
    """
    landmark = landmark_df.select(
        'DFCI_MRN', pl.col('treatment_start_date').cast(pl.Date).alias('_landmark'))
    specimens = (
        ps.load_genomic_specimen(exclude_rapidheme=True)
        .select([ps.MRN, ps.REPORT_DT])
        .drop_nulls(subset=[ps.REPORT_DT])
        .join(landmark, on='DFCI_MRN', how='inner')
        .filter(pl.col(ps.REPORT_DT) <= pl.col('_landmark'))
    )
    return (
        specimens.group_by('DFCI_MRN')
        .agg(pl.col(ps.REPORT_DT).max().alias('sequencing_date'))
    )


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    os.makedirs(MATCHED_COHORT_PATH, exist_ok=True)

    # === Load data ===
    surv_df = pl.read_parquet(os.path.join(SURV_PATH, 'death_met_surv_df.parquet'))
    surv_df = surv_df.with_columns(
        pl.col('first_treatment_date').str.to_datetime(strict=False)
        if surv_df.schema['first_treatment_date'] == pl.String
        else pl.col('first_treatment_date')
    )
    surv_mrns = set(surv_df['DFCI_MRN'].unique().to_list())

    # === Lines of therapy from PROFILE_DATA ===
    med_long = ps.unpivot_medications_summary()
    lines_df = derive_lines_of_therapy(med_long)

    # === Cancer type from PROFILE_DATA ===
    # Built on the full cohort, not the biomarker subset, so the >=500-patient
    # OTHER collapse in build_cancer_type_df produces the same labels the rest
    # of the project uses.
    full_cohort_df = pl.read_parquet(os.path.join(SURV_PATH, 'cohort_df.parquet'))
    cancer_type_df = build_cancer_type_df(full_cohort_df).select(['DFCI_MRN', 'CANCER_TYPE'])
    cancer_type_map = dict(zip(cancer_type_df['DFCI_MRN'].to_list(),
                               cancer_type_df['CANCER_TYPE'].to_list()))

    # === Restrict to sequenced patients ===
    # This is a somatic-biomarker analysis: a patient with no usable genomic
    # profile at their landmark contributes no marker and cannot inform any
    # hypothesis being tested.  The restriction is applied here, before
    # matching, rather than being left to generate_IPTW_df's somatic join.
    line1 = lines_df.filter(pl.col('LINE') == 1).select(['DFCI_MRN', 'treatment_start_date'])
    sequenced = sequenced_before_landmark(line1)
    sequenced_mrns = set(sequenced['DFCI_MRN'].to_list())
    cohort_mrns = surv_mrns & sequenced_mrns

    # First-line new-user design.  Exposure is defined using only the regimen
    # at the landmark itself; later treatment choices never affect membership.
    first_line = build_first_line_new_user_df(lines_df, cohort_mrns, cancer_type_map)

    # === Coverage funnel ===
    # A silent drop at any of these steps is the most likely failure mode of the
    # PROFILE_DATA migration, so report each one explicitly.
    med_mrns = set(med_long['DFCI_MRN'].unique().to_list())
    line1_mrns = set(lines_df.filter(pl.col('LINE') == 1)['DFCI_MRN'].unique().to_list())
    print("\nCoverage funnel:")
    print(f"  death_met_surv_df patients:          {len(surv_mrns)}")
    print(f"  ...with any MEDICATIONS_SUMMARY row: {len(surv_mrns & med_mrns)}")
    print(f"  ...with a derived line 1:            {len(surv_mrns & line1_mrns)}")
    print(f"  ...sequenced on or before line 1:    {len(cohort_mrns)}")
    print(f"  ...with a cancer type (final):       {first_line.height}")
    print(f"  Derived lines per patient: median="
          f"{lines_df.group_by('DFCI_MRN').agg(pl.len()).get_column('len').median()}, "
          f"max={lines_df.group_by('DFCI_MRN').agg(pl.len()).get_column('len').max()}")

    # === Add first_treatment_date for downstream use ===
    surv_dates = (surv_df.select(['DFCI_MRN', 'first_treatment_date'])
                  .unique(subset='DFCI_MRN', keep='first'))

    output_cols = [
        'DFCI_MRN', 'PX_on_ICI', 'line_category', 'cancer_type',
        'treatment_start_date',
    ]

    # ================================================================
    # Cohort 1: all first-line initiators, without future-history restrictions.
    # ================================================================
    print("\n" + "=" * 60)
    print("Cohort 1: First-line ICI vs non-ICI initiators, unmatched")
    print("=" * 60)

    cohort1 = first_line.select(output_cols)
    _assert_one_to_one(cohort1, surv_dates, ['DFCI_MRN'])
    cohort1 = cohort1.join(surv_dates, on='DFCI_MRN', how='inner')
    assert_schema(cohort1, 'matched_cohort_cohort1',
                  output_cols + ['first_treatment_date'], key_col='DFCI_MRN')

    n_ici = int(cohort1['PX_on_ICI'].sum())
    n_ctrl = len(cohort1) - n_ici
    print(f"Cohort 1: {n_ici} ICI + {n_ctrl} controls = {len(cohort1)} total")
    print(f"  Cancer type distribution:")
    print(cohort1.group_by(['PX_on_ICI', 'cancer_type']).agg(pl.len()).pivot(
        on='cancer_type', index='PX_on_ICI', values='len').fill_null(0))

    out_path = os.path.join(MATCHED_COHORT_PATH, 'matched_cohort_cohort1.parquet')
    cohort1.write_parquet(out_path)
    print(f"  Saved to {out_path}")

    # ================================================================
    # Cohort 2: 1:1 matched first-line new users.
    # ================================================================
    print("\n" + "=" * 60)
    print("Cohort 2: First-line ICI vs non-ICI, 1:1 matched")
    print("=" * 60)

    cohort2 = match_cohort_1to1(
        first_line.filter(pl.col('PX_on_ICI') == 1),
        first_line.filter(pl.col('PX_on_ICI') == 0),
    )

    if cohort2.is_empty():
        print("\nCohort 2: No matched patients.")
    else:
        cohort2 = cohort2.select(output_cols)
        _assert_one_to_one(cohort2, surv_dates, ['DFCI_MRN'])
        cohort2 = cohort2.join(surv_dates, on='DFCI_MRN', how='inner')
        assert_schema(cohort2, 'matched_cohort_cohort2',
                      output_cols + ['first_treatment_date'], key_col='DFCI_MRN')
        shift = (cohort2['treatment_start_date'] - cohort2['first_treatment_date']).dt.total_days()
        if (shift < 0).any():
            raise ValueError("Found a first-line treatment date before the cohort anchor.")

        n_ici = int(cohort2['PX_on_ICI'].sum())
        n_ctrl = len(cohort2) - n_ici
        print(f"\nCohort 2: {n_ici} ICI + {n_ctrl} controls = {len(cohort2)} total")
        print(f"  Line distribution:")
        print(cohort2.group_by(['PX_on_ICI', 'line_category']).agg(pl.len()).pivot(
            on='line_category', index='PX_on_ICI', values='len').fill_null(0))

        out_path = os.path.join(MATCHED_COHORT_PATH, 'matched_cohort_cohort2.parquet')
        cohort2.write_parquet(out_path)
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
