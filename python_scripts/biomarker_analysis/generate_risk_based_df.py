"""Generate risk-based biomarker discovery dataset with text-embedding risk scores.

Time origin for all patients is LOT_start_date for line 1, matching the
prediction time used to generate propensity scores in ICI_LRs.py.
"""

import os
import random
import numpy as np
import pandas as pd
from embed_surv_utils import run_grid_CoxPH_parallel, get_heldout_risk_scores_CoxPH, generate_survival_embedding_df

random.seed(42)  # set seed for reproducibility

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
ICI_PATH = os.path.join(DATA_PATH, 'treatment_prediction/ICI_propensity/w_30_day_buffer/')
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/processed_datasets/')
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')

# --- Load base survival data (tt_death measured from first_treatment_date) ---
tt_death_df = pd.read_csv(os.path.join(SURV_PATH, 'death_met_surv_df.csv'))
tt_death_df['first_treatment_date'] = pd.to_datetime(tt_death_df['first_treatment_date'])

# --- Load line 1 treatment start dates (propensity score time origin) ---
treatment_df = pd.read_csv('/data/gusev/USERS/mjsaleh/profile_lines_of_rx/profile_rxlines.csv')
treatment_df['LOT_start_date'] = pd.to_datetime(treatment_df['LOT_start_date'])
treatment_df = treatment_df.sort_values(['MRN', 'LOT_start_date'])
treatment_df['treatment_line'] = treatment_df.groupby('MRN').cumcount() + 1
line1_starts = (treatment_df
                .loc[treatment_df['treatment_line'] == 1, ['MRN', 'LOT_start_date']]
                .rename(columns={'MRN': 'DFCI_MRN', 'LOT_start_date': 'line1_start_date'}))

# --- Load note embeddings ---
notes_meta = pd.read_csv(os.path.join(NOTES_PATH, 'full_VTE_embeddings_metadata.csv'))
embeddings = np.load(os.path.join(NOTES_PATH, 'full_VTE_embeddings_as_array.npy'))

# --- Load genomic / clinical features ---
cancer_type_df = pd.read_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/cancer_type_df.csv'))
somatic_df = pd.read_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/complete_somatic_data_df.csv'))
mutation_tags = ('_SNV', '_SV', '_FUSION', '_DEL', '_AMP', '_CNV')
panel_cols = [col for col in somatic_df.columns if col.upper().startswith('PANEL_VERSION')]
mutation_cols = [col for col in somatic_df.columns if any(tag in col.upper() for tag in mutation_tags)]
somatic_keep_cols = ['DFCI_MRN'] + panel_cols + mutation_cols
somatic_keep_cols = list(dict.fromkeys(somatic_keep_cols))
somatic_df = somatic_df[somatic_keep_cols].copy()

# --- Load propensity predictions to identify ICI patients ---
line1_preds = pd.read_csv(os.path.join(ICI_PATH, 'line_1_predictions.csv'))
line1_preds = line1_preds[['DFCI_MRN', 'ground_truth']].dropna().copy()
line1_preds['ground_truth'] = line1_preds['ground_truth'].astype(int)
ICI_mrns = line1_preds.loc[line1_preds['ground_truth'] == 1, 'DFCI_MRN'].unique()

# --- Merge line 1 start dates and recompute tt_death ---
surv_df = tt_death_df.merge(line1_starts, on='DFCI_MRN')
days_offset = (surv_df['line1_start_date'] - surv_df['first_treatment_date']).dt.days
surv_df['tt_death'] = surv_df['tt_death'] - days_offset
surv_df = surv_df.loc[surv_df['tt_death'] > 0].copy()

# --- Build note timing relative to line 1 start date for ALL patients ---
line1_start_map = dict(zip(line1_starts['DFCI_MRN'], line1_starts['line1_start_date']))
notes_meta['ANALYSIS_START_DT'] = notes_meta['DFCI_MRN'].map(line1_start_map)
notes_meta['NOTE_TIME_REL_ANALYSIS_START_DT'] = (
    pd.to_datetime(notes_meta['NOTE_DATETIME']) - pd.to_datetime(notes_meta['ANALYSIS_START_DT'])
).dt.days

# --- Generate embedding features with time-decay-mean pooling ---
note_types = ['Clinician', 'Imaging', 'Pathology']
ICI_prediction_df = (generate_survival_embedding_df(
                        notes_meta, surv_df[['DFCI_MRN', 'death', 'tt_death']], embeddings,
                        note_types=note_types,
                        pool_fx={key: 'time_decay_mean' for key in note_types},
                        decay_param=0.01,
                        note_timing_col='NOTE_TIME_REL_ANALYSIS_START_DT')
                    .merge(surv_df[['DFCI_MRN', 'GENDER', 'AGE_AT_TREATMENTSTART']], on='DFCI_MRN')
                    .merge(cancer_type_df, on='DFCI_MRN')).dropna()

base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART'] + [col for col in ICI_prediction_df if col.startswith('CANCER_TYPE')]
embed_cols = [c for c in ICI_prediction_df.columns if 'EMBEDDING' in c or '2015' in c]
continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols

# Grid search for best penalized CoxPH hyperparameters (trained on ALL patients)
event = 'death'
alphas_to_test = np.logspace(-5, 0, 25)
l1_ratios = [0.5, 1.0]

_, ICI_val_results, _ = run_grid_CoxPH_parallel(
    ICI_prediction_df, base_vars, continuous_vars, embed_cols,
    l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=5000, verbose=5)

ICI_l1_ratio, ICI_alpha = ICI_val_results.sort_values(by='mean_auc(t)', ascending=False).iloc[0][['l1_ratio', 'alpha']]

# Get held-out risk scores using best hyperparameters
trained_ICI = (get_heldout_risk_scores_CoxPH(
                  ICI_prediction_df, base_vars, continuous_vars, embed_cols,
                  event_col=event, tstop_col=f'tt_{event}', penalized=True,
                  l1_ratio=ICI_l1_ratio, alpha=ICI_alpha, max_iter=5000)
              .rename(columns={'risk_score': 'ICI_risk_score'}))

# --- Build final biomarker df (ICI patients only) ---
biomarker_df = (surv_df[['DFCI_MRN', 'tt_death', 'death', 'GENDER', 'AGE_AT_TREATMENTSTART']]
                .loc[surv_df['DFCI_MRN'].isin(ICI_mrns)]
                .merge(somatic_df, on='DFCI_MRN')
                .merge(cancer_type_df, on='DFCI_MRN')
                .merge(trained_ICI, on='DFCI_MRN')
                .drop_duplicates(subset=['DFCI_MRN'], keep='first'))

biomarker_df.to_csv(os.path.join(MARKER_PATH, 'ICI_biomarker_discovery.csv'), index=False)
