"""Generate IPTW dataset for biomarker analysis (ICI vs never-ICI).

Treatment groups:
  - ICI:     patients in IO_START.csv
  - non-ICI: patients in survival cohort NOT in IO_START.csv

Time origin for all patients is first_treatment_date (= treatment_start_date
saved by ICI_LRs.py), so no time-origin shift is needed.

Usage:
  python generate_IPTW_df.py --cohort all_ICI
  python generate_IPTW_df.py --cohort first_line
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
from embed_surv_utils import run_grid_CoxPH_parallel, get_heldout_risk_scores_CoxPH, generate_survival_embedding_df

random.seed(42)

parser = argparse.ArgumentParser(description='Generate IPTW dataset for biomarker analysis.')
parser.add_argument('--cohort', choices=['all_ICI', 'first_line'], required=True,
                    help='ICI cohort definition: all_ICI (any line) or first_line (first-line only)')
args = parser.parse_args()

COHORT = args.cohort

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
COHORT_PATHS = {
    'all_ICI': {
        'propensity': 'treatment_prediction/all_ICI_propensity/w_30_day_buffer/',
        'prediction_data': 'treatment_prediction/all_ICI_prediction_data/',
    },
    'first_line': {
        'propensity': 'treatment_prediction/first_line_ICI_propensity/w_30_day_buffer/',
        'prediction_data': 'treatment_prediction/first_line_ICI_prediction_data/',
    },
}
ICI_PATH = os.path.join(DATA_PATH, COHORT_PATHS[COHORT]['propensity'])
ICI_DATA_PATH = os.path.join(DATA_PATH, COHORT_PATHS[COHORT]['prediction_data'])
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/processed_datasets/')
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')
os.makedirs(MARKER_PATH, exist_ok=True)

print(f"[generate_IPTW_df] Cohort: {COHORT}")
print(f"[generate_IPTW_df] Propensity path: {ICI_PATH}")
print(f"[generate_IPTW_df] Prediction data path: {ICI_DATA_PATH}")

# --- Load base survival data (tt_death measured from first_treatment_date) ---
tt_death_df = pd.read_csv(os.path.join(SURV_PATH, 'death_met_surv_df.csv'))
tt_death_df['first_treatment_date'] = pd.to_datetime(tt_death_df['first_treatment_date'])
tt_death_df = tt_death_df[['DFCI_MRN', 'first_treatment_date', 'tt_death', 'death',
                            'GENDER', 'AGE_AT_TREATMENTSTART']].copy()

# --- Load per-patient prediction times (saved by ICI_LRs.py) ---
# treatment_start_date = first_treatment_date for all patients
prediction_times = pd.read_csv(os.path.join(ICI_DATA_PATH, 'prediction_times.csv'))
prediction_times['treatment_start_date'] = pd.to_datetime(prediction_times['treatment_start_date'])
prediction_times = prediction_times[['DFCI_MRN', 'treatment_start_date']].drop_duplicates(subset='DFCI_MRN', keep='first')

# --- Load genomic / clinical features ---
cancer_type_df = pd.read_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/cancer_type_df.csv'))
somatic_df = pd.read_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/complete_somatic_data_df.csv'))
mutation_tags = ('_SNV', '_SV', '_FUSION', '_DEL', '_AMP', '_CNV')
panel_cols = [col for col in somatic_df.columns if col.upper().startswith('PANEL_VERSION')]
mutation_cols = [col for col in somatic_df.columns if any(tag in col.upper() for tag in mutation_tags)]
somatic_keep_cols = ['DFCI_MRN'] + panel_cols + mutation_cols
somatic_keep_cols = list(dict.fromkeys(somatic_keep_cols))
somatic_df = somatic_df[somatic_keep_cols].copy()

# --- Load propensity predictions (30-day buffer) ---
line1_preds = pd.read_csv(os.path.join(ICI_PATH, 'line_1_predictions.csv'))
required_pred_cols = {'DFCI_MRN', 'ground_truth', 'model_probs'}
if not required_pred_cols.issubset(set(line1_preds.columns)):
    raise ValueError(f"line_1_predictions.csv must contain columns: {sorted(required_pred_cols)}")
line1_preds = line1_preds[['DFCI_MRN', 'ground_truth', 'model_probs']].dropna().copy()
line1_preds['ground_truth'] = line1_preds['ground_truth'].astype(int)

# --- Build unified patient dataframe ---
# Merge all data sources; restrict to patients with propensity predictions
patient_df = pd.get_dummies(tt_death_df
              .merge(prediction_times, on='DFCI_MRN')
              .merge(somatic_df, on='DFCI_MRN')
              .merge(cancer_type_df, on='DFCI_MRN')
              .merge(line1_preds, on='DFCI_MRN')
              .drop_duplicates(subset=['DFCI_MRN'], keep='first'), columns=['PANEL_VERSION'], drop_first=True)

# Drop patients with non-positive survival
patient_df = patient_df.loc[patient_df['tt_death'] > 0].copy()

# --- Assign treatment group and propensity scores ---
patient_df['PX_on_ICI'] = patient_df['ground_truth'].astype(int)
patient_df['ICI_prediction'] = patient_df['model_probs']

# --- Compute text_risk_score from note embeddings (all patients) ---
notes_meta = pd.read_csv(os.path.join(NOTES_PATH, 'full_VTE_embeddings_metadata.csv'))
embeddings = np.load(os.path.join(NOTES_PATH, 'full_VTE_embeddings_as_array.npy'))

start_date_map = dict(zip(prediction_times['DFCI_MRN'], prediction_times['treatment_start_date']))
notes_meta['ANALYSIS_START_DT'] = notes_meta['DFCI_MRN'].map(start_date_map)
notes_meta['NOTE_TIME_REL_ANALYSIS_START_DT'] = (
    pd.to_datetime(notes_meta['NOTE_DATETIME']) - pd.to_datetime(notes_meta['ANALYSIS_START_DT'])
).dt.days

note_types = ['Clinician', 'Imaging', 'Pathology']
embedding_df = (generate_survival_embedding_df(
                    notes_meta, patient_df[['DFCI_MRN', 'death', 'tt_death']], embeddings,
                    note_types=note_types,
                    pool_fx={key: 'time_decay_mean' for key in note_types},
                    decay_param=0.01,
                    note_timing_col='NOTE_TIME_REL_ANALYSIS_START_DT')
                .merge(patient_df[['DFCI_MRN', 'GENDER', 'AGE_AT_TREATMENTSTART']], on='DFCI_MRN')
                .merge(cancer_type_df, on='DFCI_MRN')).dropna()

risk_base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART'] + [c for c in embedding_df if c.startswith('CANCER_TYPE')]
embed_cols = [c for c in embedding_df.columns if 'EMBEDDING' in c or '2015' in c]
continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols

alphas_to_test = np.logspace(-5, 0, 25)
l1_ratios = [0.5, 1.0]

_, val_results, _ = run_grid_CoxPH_parallel(
    embedding_df, risk_base_vars, continuous_vars, embed_cols,
    l1_ratios, alphas_to_test, event_col='death', tstop_col='tt_death', max_iter=5000, verbose=5)

best_l1, best_alpha = val_results.sort_values(by='mean_auc(t)', ascending=False).iloc[0][['l1_ratio', 'alpha']]

risk_scores = (get_heldout_risk_scores_CoxPH(
                  embedding_df, risk_base_vars, continuous_vars, embed_cols,
                  event_col='death', tstop_col='tt_death', penalized=True,
                  l1_ratio=best_l1, alpha=best_alpha, max_iter=5000)
              .rename(columns={'risk_score': 'text_risk_score'}))

patient_df = patient_df.merge(risk_scores[['DFCI_MRN', 'text_risk_score']], on='DFCI_MRN', how='left')

# --- Select final columns ---
required_cols = ['DFCI_MRN', 'tt_death', 'death']
base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
drop_cols = set(required_cols + base_vars + ['PX_on_ICI', 'ICI_prediction', 'text_risk_score',
                'first_treatment_date', 'treatment_start_date', 'ground_truth', 'model_probs'])
biomarker_cols = [col for col in patient_df.columns if col not in drop_cols]

output_cols = required_cols + base_vars + biomarker_cols + ['PX_on_ICI', 'ICI_prediction']
if 'text_risk_score' in patient_df.columns:
    output_cols.append('text_risk_score')
interaction_ICI_df = patient_df[output_cols].copy()

interaction_ICI_df = interaction_ICI_df.dropna(subset=['ICI_prediction', 'tt_death', 'death']).copy()
interaction_ICI_df['PX_on_ICI'] = interaction_ICI_df['PX_on_ICI'].astype(int)
interaction_ICI_df['death'] = interaction_ICI_df['death'].astype(int)

output_file = os.path.join(MARKER_PATH, f'IPTW_ICI_interaction_runs_df_{COHORT}.csv')
interaction_ICI_df.to_csv(output_file, index=False)
print(f"[generate_IPTW_df] Saved {len(interaction_ICI_df)} patients to {output_file}")
