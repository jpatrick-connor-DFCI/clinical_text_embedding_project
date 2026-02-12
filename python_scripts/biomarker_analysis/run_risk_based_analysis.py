"""Run Risk Based Analysis script for biomarker analysis workflows."""

import os
import time
import random
from tqdm import tqdm
import pandas as pd
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests

random.seed(42)  # set seed for reproducibility

# Paths
IO_PATH = '/data/gusev/USERS/mjsaleh/'
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')
RISK_RUN_PATH = os.path.join(MARKER_PATH, 'text_risk_runs/')

biomarker_df = pd.read_csv(os.path.join(MARKER_PATH, 'IO_biomarker_discovery.csv')).drop_duplicates(subset=['DFCI_MRN'], keep='first')
irAE_df = pd.read_csv(os.path.join(IO_PATH, 'IO_START.csv'), index_col=0).rename(columns={'MRN' : 'DFCI_MRN'})

death_df = biomarker_df.loc[biomarker_df['DFCI_MRN'].isin(irAE_df['DFCI_MRN'].unique())].copy()

base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART'] + \
            [col for col in death_df if col.startswith('CANCER_TYPE') and death_df[col].sum() > 0] + \
            [col for col in death_df if col.startswith('PANEL_VERSION') and death_df[col].sum() > 0]
genomics_cols = [col for col in biomarker_df if not col in ['DFCI_MRN', 'tt_death', 'death'] + base_vars]

cancer_type_counts = death_df[[col for col in death_df.columns if col.startswith('CANCER_TYPE_') and ('OTHER' not in col)]].sum(axis=0).sort_values(ascending=False)
cancer_types_to_test = cancer_type_counts[cancer_type_counts >= 100].index.tolist()

base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART'] + \
            [col for col in death_df if col.startswith('PANEL_VERSION') and death_df[col].sum() > 0]
genomics_cols = [col for col in biomarker_df if not col in ['DFCI_MRN', 'tt_death', 'death'] + base_vars + [col for col in biomarker_df.columns if col.startswith('CANCER_TYPE')]]

for cancer_type in cancer_types_to_test:
    
    print(f'Starting {cancer_type}.')
    start_time = time.time()
    
    type_df = death_df.loc[death_df[cancer_type]].copy()

    marker_dfs = []
    markers_to_test = [m for m in genomics_cols if (type_df[m].sum()/len(type_df)) >= 0.01]
    for test_col in tqdm(markers_to_test):
        try:
            cols_base = ['tt_death', 'death'] + base_vars + [test_col]
            base_cph = CoxPHFitter()
            base_cph.fit(type_df[cols_base], duration_col='tt_death', event_col='death', robust=True)
            base_sum = base_cph.summary.reset_index()
            base_entry = base_sum.loc[base_sum['covariate'] == test_col]
            base_entry.columns = [col + '_without_text_risk' for col in base_entry.columns]

            cols_risk = ['tt_death', 'death'] + base_vars + ['IO_risk_score', test_col]
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
            continue
    try: 
        type_IO_marker_df = pd.concat(marker_dfs)
    
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
    
    except:
        print(f'{cancer_type} finished. Time elapsed = {(time.time() - start_time) / 60 : 0.2f}')
        
        
