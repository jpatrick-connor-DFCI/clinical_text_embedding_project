"""Build line-matched ICI vs never-ICI cohorts.

For each ICI patient, determines the line at which they received ICI (1, 2, 3, or 4+).
For each never-ICI patient, determines their max line of therapy (1, 2, 3, or 4+).
Performs exact matching on (cancer_type, line_category) with two schemes:
  - 1:1 matching (one control per case, without replacement)
  - 1:k matching (up to k=3 controls per case, without replacement)

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

MAX_K = 3  # max controls per case for 1:k matching

# === Load data ===
surv_df = pd.read_csv(os.path.join(SURV_PATH, 'death_met_surv_df.csv'))
surv_df['first_treatment_date'] = pd.to_datetime(surv_df['first_treatment_date'])
cohort_mrns = set(surv_df['DFCI_MRN'].unique())

io_start_df = pd.read_csv(IO_START_FILE, index_col=0).rename(columns={'MRN': 'DFCI_MRN'})
ici_mrns = set(io_start_df['DFCI_MRN'].unique())

treatment_df = pd.read_csv(TREATMENT_FILE)
treated_mrns = set(treatment_df['MRN'].unique())

med_lines_df = pd.read_csv(MED_LINES_FILE).rename(columns={'MRN': 'DFCI_MRN'})

cancer_type_df = pd.read_csv(os.path.join(CLINICAL_FEATURE_PATH, 'cancer_type_df.csv'))
cancer_type_cols = [c for c in cancer_type_df.columns if c.startswith('CANCER_TYPE_')]
cancer_type_df['cancer_type'] = cancer_type_df[cancer_type_cols].idxmax(axis=1)
cancer_type_map = dict(zip(cancer_type_df['DFCI_MRN'], cancer_type_df['cancer_type']))


def bin_line(line_num):
    """Bin line of therapy into categories: 1, 2, 3, or 4+."""
    if line_num <= 3:
        return int(line_num)
    return 4


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
ici_lines['line_category'] = ici_lines['ici_line'].apply(bin_line)
ici_lines['PX_on_ICI'] = 1

# === Determine max line for never-ICI patients ===
never_ici_mrns = (cohort_mrns & treated_mrns) - ici_mrns

control_lines = (
    med_lines_df.loc[med_lines_df['DFCI_MRN'].isin(never_ici_mrns)]
    .groupby('DFCI_MRN')['LINE']
    .max()
    .reset_index()
    .rename(columns={'LINE': 'max_line'})
)
control_lines['line_category'] = control_lines['max_line'].apply(bin_line)
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

# === Matching ===
def match_cohort(cases_df, controls_df, max_controls_per_case):
    """Exact match on (cancer_type, line_category) without replacement.

    Returns matched case-control dataframe.
    """
    matched_cases = []
    matched_controls = []

    for (ctype, lcat), stratum_cases in cases_df.groupby(['cancer_type', 'line_category']):
        stratum_controls = controls_df.loc[
            (controls_df['cancer_type'] == ctype) &
            (controls_df['line_category'] == lcat)
        ].copy()

        if stratum_controls.empty:
            print(f"  WARNING: No controls for ({ctype}, line={lcat}), "
                  f"dropping {len(stratum_cases)} ICI cases")
            continue

        # Shuffle controls for random selection
        stratum_controls = stratum_controls.sample(frac=1, random_state=42).reset_index(drop=True)

        n_cases = len(stratum_cases)
        n_available = len(stratum_controls)
        n_per_case = min(max_controls_per_case, n_available // n_cases) if n_cases > 0 else 0

        if n_per_case == 0 and n_available > 0:
            # Fewer controls than cases: match as many cases as we have controls (1:1)
            n_matchable = min(n_cases, n_available)
            sampled_cases = stratum_cases.sample(n=n_matchable, random_state=42)
            matched_cases.append(sampled_cases)
            matched_controls.append(stratum_controls.head(n_matchable))
            print(f"  ({ctype}, line={lcat}): {n_matchable}/{n_cases} cases matched 1:1 "
                  f"(only {n_available} controls available)")
        else:
            matched_cases.append(stratum_cases)
            # Assign controls evenly across cases without replacement
            n_total_controls = n_per_case * n_cases
            matched_controls.append(stratum_controls.head(n_total_controls))
            print(f"  ({ctype}, line={lcat}): {n_cases} cases x {n_per_case} controls "
                  f"({n_available} available)")

    if not matched_cases:
        return pd.DataFrame()

    all_cases = pd.concat(matched_cases, ignore_index=True)
    all_controls = pd.concat(matched_controls, ignore_index=True)
    return pd.concat([all_cases, all_controls], ignore_index=True)


# --- 1:1 matching ---
print("\n=== 1:1 Matching ===")
matched_1to1 = match_cohort(ici_lines, control_lines, max_controls_per_case=1)

# --- 1:k matching ---
print(f"\n=== 1:{MAX_K} Matching ===")
matched_1tok = match_cohort(ici_lines, control_lines, max_controls_per_case=MAX_K)

# === Add first_treatment_date back for downstream use ===
surv_dates = (surv_df[['DFCI_MRN', 'first_treatment_date']]
              .drop_duplicates(subset='DFCI_MRN', keep='first'))

for label, matched_df in [('1to1', matched_1to1), ('1tok', matched_1tok)]:
    if matched_df.empty:
        print(f"\n{label}: No matched patients, skipping.")
        continue

    out = matched_df[['DFCI_MRN', 'PX_on_ICI', 'line_category', 'cancer_type']].merge(
        surv_dates, on='DFCI_MRN')
    out = out.rename(columns={'first_treatment_date': 'treatment_start_date'})

    n_ici = out['PX_on_ICI'].sum()
    n_ctrl = len(out) - n_ici
    print(f"\n{label}: {int(n_ici)} ICI + {int(n_ctrl)} controls = {len(out)} total")
    print(f"  Line distribution:")
    print(out.groupby(['PX_on_ICI', 'line_category']).size().unstack(fill_value=0))

    out.to_csv(os.path.join(COHORT_PATH, f'matched_cohort_{label}.csv'), index=False)
    print(f"  Saved to {os.path.join(COHORT_PATH, f'matched_cohort_{label}.csv')}")
