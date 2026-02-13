"""Run Risk Based Analysis script for biomarker analysis workflows."""

import os
import time
import random
from tqdm import tqdm
import pandas as pd
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests

random.seed(42)  # set seed for reproducibility


def one_hot_panel_version(df: pd.DataFrame) -> pd.DataFrame:
    panel_raw_cols = [col for col in df.columns if col.strip().upper() == 'PANEL_VERSION']
    if not panel_raw_cols:
        return df

    out = df.copy()
    panel_raw_col = panel_raw_cols[0]
    existing_panel_one_hot = [col for col in out.columns if col.upper().startswith('PANEL_VERSION_')]
    if existing_panel_one_hot:
        out = out.drop(columns=existing_panel_one_hot)

    panel_values = out[panel_raw_col].fillna('MISSING').astype(str)
    panel_dummies = pd.get_dummies(panel_values, prefix='PANEL_VERSION', dtype=int)
    out = pd.concat([out.drop(columns=[panel_raw_col]), panel_dummies], axis=1)
    return out


# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')
RISK_RUN_PATH = os.path.join(MARKER_PATH, 'text_risk_runs/')
os.makedirs(RISK_RUN_PATH, exist_ok=True)

biomarker_df = pd.read_csv(os.path.join(MARKER_PATH, 'ICI_biomarker_discovery.csv')).drop_duplicates(subset=['DFCI_MRN'], keep='first')
biomarker_df = one_hot_panel_version(biomarker_df)

death_df = biomarker_df.copy()

cancer_type_counts = death_df[[col for col in death_df.columns if col.startswith('CANCER_TYPE_') and ('OTHER' not in col)]].sum(axis=0).sort_values(ascending=False)
cancer_types_to_test = cancer_type_counts[cancer_type_counts >= 100].index.tolist()

panel_cols = [
    col for col in death_df
    if col.upper().startswith('PANEL_VERSION_')
    and pd.to_numeric(death_df[col], errors='coerce').fillna(0).sum() > 0
]
base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART'] + panel_cols
excluded_cols = ['DFCI_MRN', 'tt_death', 'death'] + base_vars + [col for col in biomarker_df.columns if col.startswith('CANCER_TYPE')]
mutation_tags = ('_SNV', '_SV', '_FUSION', '_DEL', '_AMP', '_CNV')
genomics_cols = [
    col for col in biomarker_df.columns
    if (col not in excluded_cols) and any(tag in col.upper() for tag in mutation_tags)
]

for cancer_type in cancer_types_to_test:
    
    print(f'Starting {cancer_type}.')
    start_time = time.time()
    
    type_df = death_df.loc[death_df[cancer_type]].copy()

    if type_df.empty:
        print(f'No patients available for {cancer_type}; skipping.')
        continue

    marker_dfs = []
    markers_to_test = []
    for marker in genomics_cols:
        prevalence = pd.to_numeric(type_df[marker], errors='coerce').sum(skipna=True) / len(type_df)
        if prevalence >= 0.01:
            markers_to_test.append(marker)

    failed_markers = []
    for test_col in tqdm(markers_to_test):
        try:
            cols_base = ['tt_death', 'death'] + base_vars + [test_col]
            base_cph = CoxPHFitter()
            base_cph.fit(type_df[cols_base], duration_col='tt_death', event_col='death', robust=True)
            base_sum = base_cph.summary.reset_index()
            base_entry = base_sum.loc[base_sum['covariate'] == test_col]
            base_entry.columns = [col + '_without_text_risk' for col in base_entry.columns]

            cols_risk = ['tt_death', 'death'] + base_vars + ['ICI_risk_score', test_col]
            risk_cph = CoxPHFitter()
            risk_cph.fit(type_df[cols_risk], duration_col='tt_death', event_col='death', robust=True)
            risk_sum = risk_cph.summary.reset_index()
            risk_entry = risk_sum.loc[risk_sum['covariate'] == test_col]
            risk_entry.columns = [col + '_with_text_risk' for col in risk_entry.columns]

            entry = pd.concat([base_entry.reset_index(drop=True), risk_entry.reset_index(drop=True)], axis=1)
            entry.insert(0, 'covariate', test_col)
            entry.insert(1, 'c_index_without_text_risk', base_cph.concordance_index_)
            entry.insert(1, 'c_index_with_text_risk', risk_cph.concordance_index_)

            marker_dfs.append(entry)

        except Exception as e:
            failed_markers.append((test_col, str(e)))
            continue

    if failed_markers:
        first_marker, first_error = failed_markers[0]
        print(f'{cancer_type}: {len(failed_markers)} marker fits failed. First: {first_marker} -> {first_error}')

    if not marker_dfs:
        print(f'{cancer_type}: no successful marker fits; skipping output.')
        print(f'{cancer_type} finished. Time elapsed = {(time.time() - start_time) / 60 : 0.2f}')
        continue

    type_IO_marker_df = pd.concat(marker_dfs, ignore_index=True)
    if type_IO_marker_df.empty:
        print(f'{cancer_type}: empty result table after fitting; skipping output.')
        print(f'{cancer_type} finished. Time elapsed = {(time.time() - start_time) / 60 : 0.2f}')
        continue

    reject, pvals_corrected, _, _ = multipletests(type_IO_marker_df["p_without_text_risk"], alpha=0.05, method="fdr_bh")
    risk_reject, risk_pvals_corrected, _, _ = multipletests(type_IO_marker_df["p_with_text_risk"], alpha=0.05, method="fdr_bh")
    
    type_IO_marker_df['corrected_p_without_text_risk'] = pvals_corrected
    type_IO_marker_df['corrected_p_with_text_risk'] = risk_pvals_corrected
    
    type_IO_marker_df['significant_without_text_risk'] = reject
    type_IO_marker_df['significant_with_text_risk'] = risk_reject
    
    all_sig_type_hits = type_IO_marker_df.loc[(type_IO_marker_df['significant_without_text_risk']) |
                                         (type_IO_marker_df['significant_with_text_risk'])]

    all_sig_type_hits.to_csv(
        os.path.join(RISK_RUN_PATH, f"all_sig_{cancer_type.replace('CANCER_TYPE_', '').lower()}_hits.csv"),
        index=False,
    )

    print(f'{cancer_type} finished. Time elapsed = {(time.time() - start_time) / 60 : 0.2f}')
