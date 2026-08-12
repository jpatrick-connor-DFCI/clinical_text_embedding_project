"""Build two ICI vs never-ICI cohorts for biomarker analysis.

Cohort 1 (first_line_unmatched):
  - ICI patients whose first ICI was at line 1
  - Never-ICI controls whose max line of therapy was 1
  - No 1:1 matching — all eligible patients included

Cohort 2 (line_matched_1to3):
  - ICI patients whose first ICI was at line 1, 2, or 3
  - Never-ICI controls whose max line was 1, 2, or 3
  - 1:1 exact matching on (cancer_type, line_category) without replacement

Usage:
  python -m pipelines.biomarkers.build_line_matched_cohort
"""

import gzip
import os
import random

import numpy as np
import polars as pl

from config import FEATURE_PATH, IO_START_FILE, MATCHED_COHORT_PATH, MED_LINES_FILE, SURV_PATH


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


def build_line_start_dates(med_lines_df: pl.DataFrame) -> pl.DataFrame:
    """Return one observed treatment-start date per patient and therapy line."""
    required = {"DFCI_MRN", "LINE", "MED_START_DT"}
    missing = required - set(med_lines_df.columns)
    if missing:
        raise ValueError(
            f"{MED_LINES_FILE} is missing line-timing columns: {sorted(missing)}"
        )

    line_dates = med_lines_df.select(sorted(required))
    line_dates = line_dates.with_columns(
        pl.col("LINE").cast(pl.Float64, strict=False),
        pl.col("MED_START_DT").str.to_datetime(strict=False).alias("treatment_start_date"),
    )
    line_dates = line_dates.drop_nulls(
        subset=["DFCI_MRN", "LINE", "treatment_start_date"]
    )
    line_dates = line_dates.with_columns(pl.col("LINE").cast(pl.Int64))
    return (
        line_dates.group_by(["DFCI_MRN", "LINE"])
        .agg(pl.col("treatment_start_date").min())
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
    cohort_mrns = set(surv_df['DFCI_MRN'].unique().to_list())

    io_start_df = pl.read_csv(IO_START_FILE).rename({'MRN': 'DFCI_MRN'})
    ici_mrns = set(io_start_df['DFCI_MRN'].unique().to_list())

    med_lines_df = pl.read_csv(MED_LINES_FILE).rename({'MRN': 'DFCI_MRN'})
    line_start_dates = build_line_start_dates(med_lines_df)
    treated_mrns = set(line_start_dates['DFCI_MRN'].unique().to_list())

    cancer_type_df = pl.read_csv(os.path.join(FEATURE_PATH, 'cancer_type_df.csv.gz'))
    if 'CANCER_TYPE' not in cancer_type_df.columns:
        raise ValueError(
            "cancer_type_df.csv.gz must contain the raw CANCER_TYPE column; "
            "reconstructing it from drop-first dummy columns is not valid."
        )
    cancer_type_df = cancer_type_df.with_columns(pl.col('CANCER_TYPE').alias('cancer_type'))
    cancer_type_map = dict(zip(cancer_type_df['DFCI_MRN'].to_list(), cancer_type_df['cancer_type'].to_list()))

    # === Determine ICI line for ICI patients ===
    # ICI line = the earliest LINE where HAS_ICI == 1 for patients in IO_START
    ici_lines = (
        med_lines_df.filter(
            (pl.col('DFCI_MRN').is_in(ici_mrns & cohort_mrns)) &
            (pl.col('HAS_ICI') == 1)
        )
        .group_by('DFCI_MRN')
        .agg(pl.col('LINE').min().alias('ici_line'))
    )
    ici_lines = ici_lines.with_columns(
        pl.col('ici_line').cast(pl.Int64),
        pl.lit(1).alias('PX_on_ICI'),
    ).with_columns(pl.col('ici_line').alias('line_category'))
    _assert_one_to_one(ici_lines, line_start_dates.rename({'LINE': 'ici_line'}), ['DFCI_MRN', 'ici_line'])
    ici_lines = ici_lines.join(
        line_start_dates.rename({'LINE': 'ici_line'}),
        on=['DFCI_MRN', 'ici_line'], how='inner',
    )

    # === Determine lines of therapy for never-ICI patients ===
    never_ici_mrns = (cohort_mrns & treated_mrns) - ici_mrns

    # Max line per control (used for cohort 1 as a covariate, and to determine
    # which lines a control is eligible for in cohort 2 matching)
    control_max_lines = (
        med_lines_df.filter(pl.col('DFCI_MRN').is_in(never_ici_mrns))
        .group_by('DFCI_MRN')
        .agg(pl.col('LINE').max().alias('max_line'))
    )
    # For cohort 1, only controls whose observed maximum line is one are eligible.
    control_lines = control_max_lines.with_columns(
        pl.col('max_line').cast(pl.Int64),
        pl.lit(0).alias('PX_on_ICI'),
    ).with_columns(pl.col('max_line').alias('line_category'))
    _assert_one_to_one(control_lines, line_start_dates.rename({'LINE': 'line_category'}), ['DFCI_MRN', 'line_category'])
    control_lines = control_lines.join(
        line_start_dates.rename({'LINE': 'line_category'}),
        on=['DFCI_MRN', 'line_category'], how='inner',
    )

    # === Add cancer type ===
    ici_lines = ici_lines.with_columns(
        pl.col('DFCI_MRN').replace_strict(cancer_type_map, default=None).alias('cancer_type')
    )
    control_lines = control_lines.with_columns(
        pl.col('DFCI_MRN').replace_strict(cancer_type_map, default=None).alias('cancer_type')
    )

    # Drop patients without cancer type mapping
    ici_lines = ici_lines.drop_nulls(subset=['cancer_type'])
    control_lines = control_lines.drop_nulls(subset=['cancer_type'])

    print(f"ICI patients with line + cancer type: {len(ici_lines)}")
    print(f"Never-ICI controls with line + cancer type: {len(control_lines)}")
    print(f"\nICI line_category distribution:")
    print(ici_lines['line_category'].value_counts().sort('line_category'))
    print(f"\nControl line_category distribution:")
    print(control_lines['line_category'].value_counts().sort('line_category'))

    # === Add first_treatment_date for downstream use ===
    surv_dates = (surv_df.select(['DFCI_MRN', 'first_treatment_date'])
                  .unique(subset='DFCI_MRN', keep='first'))

    output_cols = [
        'DFCI_MRN', 'PX_on_ICI', 'line_category', 'cancer_type',
        'treatment_start_date',
    ]

    # ================================================================
    # Cohort 1: First-line ICI vs never-ICI patients who only reached line 1.
    # ================================================================
    print("\n" + "=" * 60)
    print("Cohort 1: First-line ICI vs line-1-only never-ICI, unmatched")
    print("=" * 60)

    ici_line1 = ici_lines.filter(pl.col('line_category') == 1)
    control_line1_only = control_lines.filter(pl.col('max_line') == 1)

    cohort1 = pl.concat(
        [ici_line1.select(output_cols), control_line1_only.select(output_cols)]
    )
    _assert_one_to_one(cohort1, surv_dates, ['DFCI_MRN'])
    cohort1 = cohort1.join(surv_dates, on='DFCI_MRN', how='inner')

    n_ici = int(cohort1['PX_on_ICI'].sum())
    n_ctrl = len(cohort1) - n_ici
    print(f"Cohort 1: {n_ici} ICI + {n_ctrl} controls = {len(cohort1)} total")
    print(f"  Cancer type distribution:")
    print(cohort1.group_by(['PX_on_ICI', 'cancer_type']).agg(pl.len()).pivot(
        on='cancer_type', index='PX_on_ICI', values='len').fill_null(0))

    with gzip.open(os.path.join(MATCHED_COHORT_PATH, 'matched_cohort_cohort1.csv.gz'), 'wb') as f:
        cohort1.write_csv(f)
    print(f"  Saved to {os.path.join(MATCHED_COHORT_PATH, 'matched_cohort_cohort1.csv.gz')}")

    # ================================================================
    # Cohort 2: Lines 1-3, 1:1 matched on (cancer_type, line_category)
    # A control with max_line=3 is eligible at lines 1, 2, and 3.
    # Each control patient can only be used once across all strata.
    # ================================================================
    print("\n" + "=" * 60)
    print("Cohort 2: Lines 1-3, 1:1 matched")
    print("=" * 60)

    ici_1to3 = ici_lines.filter(pl.col('line_category').is_in([1, 2, 3]))

    # Expand controls: each control is eligible at every line up to their max_line (capped at 3)
    ctrl_expanded_rows = []
    for row in control_max_lines.iter_rows(named=True):
        max_l = min(int(row['max_line']), 3)
        for line in range(1, max_l + 1):
            ctrl_expanded_rows.append({
                'DFCI_MRN': row['DFCI_MRN'],
                'line_category': line,
                'PX_on_ICI': 0,
            })
    ctrl_expanded = pl.DataFrame(ctrl_expanded_rows)
    ctrl_expanded = ctrl_expanded.join(
        line_start_dates.rename({'LINE': 'line_category'}),
        on=['DFCI_MRN', 'line_category'], how='inner',
    )
    ctrl_expanded = ctrl_expanded.with_columns(
        pl.col('DFCI_MRN').replace_strict(cancer_type_map, default=None).alias('cancer_type')
    )
    ctrl_expanded = ctrl_expanded.drop_nulls(subset=['cancer_type'])

    print(f"Eligible ICI (lines 1-3): {len(ici_1to3)}")
    print(f"Eligible control-line slots: {len(ctrl_expanded)} "
          f"({ctrl_expanded['DFCI_MRN'].n_unique()} unique controls)")

    cohort2 = match_cohort_1to1(ici_1to3, ctrl_expanded)

    if cohort2.is_empty():
        print("\nCohort 2: No matched patients.")
    else:
        cohort2 = cohort2.select(output_cols)
        _assert_one_to_one(cohort2, surv_dates, ['DFCI_MRN'])
        cohort2 = cohort2.join(surv_dates, on='DFCI_MRN', how='inner')
        shift = (cohort2['treatment_start_date'] - cohort2['first_treatment_date']).dt.total_days()
        if (shift < 0).any():
            raise ValueError("Found a line-specific treatment date before first treatment.")
        later_line = cohort2['line_category'] > 1
        if later_line.any() and not (shift.filter(later_line) > 0).any():
            raise ValueError(
                "All line-2/3 landmarks equal first treatment; MED_START_DT does not "
                "provide valid line-specific timing."
            )

        n_ici = int(cohort2['PX_on_ICI'].sum())
        n_ctrl = len(cohort2) - n_ici
        print(f"\nCohort 2: {n_ici} ICI + {n_ctrl} controls = {len(cohort2)} total")
        print(f"  Line distribution:")
        print(cohort2.group_by(['PX_on_ICI', 'line_category']).agg(pl.len()).pivot(
            on='line_category', index='PX_on_ICI', values='len').fill_null(0))

        with gzip.open(os.path.join(MATCHED_COHORT_PATH, 'matched_cohort_cohort2.csv.gz'), 'wb') as f:
            cohort2.write_csv(f)
        print(f"  Saved to {os.path.join(MATCHED_COHORT_PATH, 'matched_cohort_cohort2.csv.gz')}")


if __name__ == "__main__":
    main()
