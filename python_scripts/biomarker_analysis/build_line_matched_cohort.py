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
  python build_line_matched_cohort.py
"""

import os
import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# === Paths ===
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
CLINICAL_FEATURE_PATH = os.path.join(DATA_PATH, 'clinical_and_genomic_features/')
COHORT_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/matched_cohorts/')
os.makedirs(COHORT_PATH, exist_ok=True)

MED_LINES_FILE = '/data/gusev/USERS/mjsaleh/profile_lines_of_rx/ALL_MEDICATION_LINES.csv'
IO_START_FILE = '/data/gusev/USERS/mjsaleh/IO_START.csv'
TREATMENT_FILE = '/data/gusev/USERS/mjsaleh/profile_lines_of_rx/profile_rxlines.csv'


# === Load data ===
surv_df = pd.read_csv(os.path.join(SURV_PATH, 'death_met_surv_df.csv.gz'))
surv_df['first_treatment_date'] = pd.to_datetime(surv_df['first_treatment_date'])
cohort_mrns = set(surv_df['DFCI_MRN'].unique())

io_start_df = pd.read_csv(IO_START_FILE, index_col=0).rename(columns={'MRN': 'DFCI_MRN'})
ici_mrns = set(io_start_df['DFCI_MRN'].unique())

treatment_df = pd.read_csv(TREATMENT_FILE)
treated_mrns = set(treatment_df['MRN'].unique())

med_lines_df = pd.read_csv(MED_LINES_FILE).rename(columns={'MRN': 'DFCI_MRN'})

cancer_type_df = pd.read_csv(os.path.join(CLINICAL_FEATURE_PATH, 'cancer_type_df.csv.gz'))
cancer_type_cols = [c for c in cancer_type_df.columns if c.startswith('CANCER_TYPE_')]
cancer_type_df['cancer_type'] = cancer_type_df[cancer_type_cols].idxmax(axis=1)
cancer_type_map = dict(zip(cancer_type_df['DFCI_MRN'], cancer_type_df['cancer_type']))


# === Determine ICI line for ICI patients ===
# ICI line = the earliest LINE where HAS_ICI == 1 for patients in IO_START
ici_lines = (
    med_lines_df.loc[
        (med_lines_df['DFCI_MRN'].isin(ici_mrns & cohort_mrns)) &
        (med_lines_df['HAS_ICI'] == 1)
    ]
    .groupby('DFCI_MRN')['LINE']
    .min()
    .reset_index()
    .rename(columns={'LINE': 'ici_line'})
)
ici_lines['line_category'] = ici_lines['ici_line'].astype(int)
ici_lines['PX_on_ICI'] = 1

# === Determine lines of therapy for never-ICI patients ===
never_ici_mrns = (cohort_mrns & treated_mrns) - ici_mrns

# Max line per control (used for cohort 1 as a covariate, and to determine
# which lines a control is eligible for in cohort 2 matching)
control_max_lines = (
    med_lines_df.loc[med_lines_df['DFCI_MRN'].isin(never_ici_mrns)]
    .groupby('DFCI_MRN')['LINE']
    .max()
    .reset_index()
    .rename(columns={'LINE': 'max_line'})
)
# For cohort 1 (unmatched), line_category = max line reached
control_lines = control_max_lines.copy()
control_lines['line_category'] = control_lines['max_line'].astype(int)
control_lines['PX_on_ICI'] = 0

# === Add cancer type ===
ici_lines['cancer_type'] = ici_lines['DFCI_MRN'].map(cancer_type_map)
control_lines['cancer_type'] = control_lines['DFCI_MRN'].map(cancer_type_map)

# Drop patients without cancer type mapping
ici_lines = ici_lines.dropna(subset=['cancer_type']).copy()
control_lines = control_lines.dropna(subset=['cancer_type']).copy()

print(f"ICI patients with line + cancer type: {len(ici_lines)}")
print(f"Never-ICI controls with line + cancer type: {len(control_lines)}")
print(f"\nICI line_category distribution:")
print(ici_lines['line_category'].value_counts().sort_index())
print(f"\nControl line_category distribution:")
print(control_lines['line_category'].value_counts().sort_index())

# === Add first_treatment_date for downstream use ===
surv_dates = (surv_df[['DFCI_MRN', 'first_treatment_date']]
              .drop_duplicates(subset='DFCI_MRN', keep='first'))

output_cols = ['DFCI_MRN', 'PX_on_ICI', 'line_category', 'cancer_type']


# ================================================================
# Cohort 1: First-line ICI vs all never-ICI, no matching
# ================================================================
print("\n" + "=" * 60)
print("Cohort 1: First-line ICI vs all never-ICI, unmatched")
print("=" * 60)

ici_line1 = ici_lines[ici_lines['line_category'] == 1].copy()

cohort1 = pd.concat([ici_line1[output_cols], control_lines[output_cols]], ignore_index=True)
cohort1 = cohort1.merge(surv_dates, on='DFCI_MRN')
cohort1 = cohort1.rename(columns={'first_treatment_date': 'treatment_start_date'})

n_ici = int(cohort1['PX_on_ICI'].sum())
n_ctrl = len(cohort1) - n_ici
print(f"Cohort 1: {n_ici} ICI + {n_ctrl} controls = {len(cohort1)} total")
print(f"  Cancer type distribution:")
print(cohort1.groupby(['PX_on_ICI', 'cancer_type']).size().unstack(fill_value=0))

cohort1.to_csv(os.path.join(COHORT_PATH, 'matched_cohort_cohort1.csv.gz'), index=False)
print(f"  Saved to {os.path.join(COHORT_PATH, 'matched_cohort_cohort1.csv.gz')}")


# ================================================================
# Cohort 2: Lines 1-3, 1:1 matched on (cancer_type, line_category)
# A control with max_line=3 is eligible at lines 1, 2, and 3.
# Each control patient can only be used once across all strata.
# ================================================================
print("\n" + "=" * 60)
print("Cohort 2: Lines 1-3, 1:1 matched")
print("=" * 60)

ici_1to3 = ici_lines[ici_lines['line_category'].isin([1, 2, 3])].copy()

# Expand controls: each control is eligible at every line up to their max_line (capped at 3)
ctrl_expanded_rows = []
for _, row in control_max_lines.iterrows():
    max_l = min(int(row['max_line']), 3)
    for line in range(1, max_l + 1):
        ctrl_expanded_rows.append({
            'DFCI_MRN': row['DFCI_MRN'],
            'line_category': line,
            'PX_on_ICI': 0,
        })
ctrl_expanded = pd.DataFrame(ctrl_expanded_rows)
ctrl_expanded['cancer_type'] = ctrl_expanded['DFCI_MRN'].map(cancer_type_map)
ctrl_expanded = ctrl_expanded.dropna(subset=['cancer_type']).copy()

print(f"Eligible ICI (lines 1-3): {len(ici_1to3)}")
print(f"Eligible control-line slots: {len(ctrl_expanded)} "
      f"({ctrl_expanded['DFCI_MRN'].nunique()} unique controls)")


def match_cohort_1to1(cases_df, controls_df):
    """Exact 1:1 match on (cancer_type, line_category) without replacement.

    Each control patient (DFCI_MRN) can only be used once globally, even if
    they appear in multiple line_category slots.
    """
    matched_cases = []
    matched_controls = []
    used_control_mrns = set()

    for (ctype, lcat), stratum_cases in cases_df.groupby(['cancer_type', 'line_category']):
        stratum_controls = controls_df.loc[
            (controls_df['cancer_type'] == ctype) &
            (controls_df['line_category'] == lcat) &
            (~controls_df['DFCI_MRN'].isin(used_control_mrns))
        ].copy()

        if stratum_controls.empty:
            print(f"  WARNING: No controls for ({ctype}, line={lcat}), "
                  f"dropping {len(stratum_cases)} ICI cases")
            continue

        # Shuffle controls for random selection
        stratum_controls = stratum_controls.sample(frac=1, random_state=42).reset_index(drop=True)

        n_cases = len(stratum_cases)
        n_available = len(stratum_controls)
        n_matchable = min(n_cases, n_available)

        if n_matchable < n_cases:
            sampled_cases = stratum_cases.sample(n=n_matchable, random_state=42)
            matched_cases.append(sampled_cases)
            selected_controls = stratum_controls.head(n_matchable)
            matched_controls.append(selected_controls)
            used_control_mrns.update(selected_controls['DFCI_MRN'])
            print(f"  ({ctype}, line={lcat}): {n_matchable}/{n_cases} cases matched 1:1 "
                  f"(only {n_available} controls available)")
        else:
            matched_cases.append(stratum_cases)
            selected_controls = stratum_controls.head(n_cases)
            matched_controls.append(selected_controls)
            used_control_mrns.update(selected_controls['DFCI_MRN'])
            print(f"  ({ctype}, line={lcat}): {n_cases} cases x 1 control "
                  f"({n_available} available)")

    if not matched_cases:
        return pd.DataFrame()

    all_cases = pd.concat(matched_cases, ignore_index=True)
    all_controls = pd.concat(matched_controls, ignore_index=True)
    return pd.concat([all_cases, all_controls], ignore_index=True)


cohort2 = match_cohort_1to1(ici_1to3, ctrl_expanded)

if cohort2.empty:
    print("\nCohort 2: No matched patients.")
else:
    cohort2 = cohort2[output_cols].merge(surv_dates, on='DFCI_MRN')
    cohort2 = cohort2.rename(columns={'first_treatment_date': 'treatment_start_date'})

    n_ici = int(cohort2['PX_on_ICI'].sum())
    n_ctrl = len(cohort2) - n_ici
    print(f"\nCohort 2: {n_ici} ICI + {n_ctrl} controls = {len(cohort2)} total")
    print(f"  Line distribution:")
    print(cohort2.groupby(['PX_on_ICI', 'line_category']).size().unstack(fill_value=0))

    cohort2.to_csv(os.path.join(COHORT_PATH, 'matched_cohort_cohort2.csv.gz'), index=False)
    print(f"  Saved to {os.path.join(COHORT_PATH, 'matched_cohort_cohort2.csv.gz')}")
