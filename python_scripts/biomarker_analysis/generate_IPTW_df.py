"""Generate IPTW dataset for biomarker analysis (first-line ICI vs never-ICI).

Treatment groups:
  - ICI:     patients who receive ICI at first line of treatment
  - non-ICI: patients who NEVER receive ICI at any line of treatment

Time origin for all patients is line 1 LOT_start_date.

Includes text-embedding risk scores (purely prognostic, no treatment indicator)
trained on ALL patients via cross-validated penalized CoxPH, for use as a
covariate in the doubly-robust-like IPTW analysis.
"""

import os
import random
import numpy as np
import pandas as pd
from embed_surv_utils import (
    run_grid_CoxPH_parallel,
    get_heldout_risk_scores_CoxPH,
    generate_survival_embedding_df,
)

random.seed(42)  # set seed for reproducibility

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
ICI_PATH = os.path.join(DATA_PATH, 'treatment_prediction/ICI_propensity/w_30_day_buffer/')
ICI_DATA_PATH = os.path.join(DATA_PATH, 'treatment_prediction/line_ICI_prediction_data/')
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/processed_datasets/')
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')

# --- Load base survival data (tt_death measured from first_treatment_date) ---
tt_death_df = pd.read_csv(os.path.join(SURV_PATH, 'death_met_surv_df.csv'))
tt_death_df['first_treatment_date'] = pd.to_datetime(tt_death_df['first_treatment_date'])
tt_death_df = tt_death_df[['DFCI_MRN', 'first_treatment_date', 'tt_death', 'death',
                            'GENDER', 'AGE_AT_TREATMENTSTART']].copy()

# --- Load per-patient prediction times (saved by ICI_LRs.py) ---
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

# --- Load propensity predictions (line 1, 30-day buffer) ---
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

# --- Recompute tt_death from line 1 start ---
# Original tt_death is measured from first_treatment_date; shift to treatment_start_date
days_offset = (patient_df['treatment_start_date'] - patient_df['first_treatment_date']).dt.days
patient_df['tt_death'] = patient_df['tt_death'] - days_offset

# Drop patients with non-positive survival from their prediction time
patient_df = patient_df.loc[patient_df['tt_death'] > 0].copy()

# --- Assign treatment group and propensity scores ---
patient_df['PX_on_ICI'] = patient_df['ground_truth'].astype(int)
patient_df['ICI_prediction'] = patient_df['model_probs']

# =====================================================================
# Compute text-embedding risk scores (purely prognostic, no treatment)
# =====================================================================
# Train penalized CoxPH: survival ~ embeddings + demographics on ALL patients
# Cross-validated held-out predictions to avoid overfitting
print("Loading note embeddings for text risk score computation...")
notes_meta = pd.read_csv(os.path.join(NOTES_PATH, 'full_VTE_embeddings_metadata.csv'))
embeddings = np.load(os.path.join(NOTES_PATH, 'full_VTE_embeddings_as_array.npy'))

# Build note timing relative to treatment_start_date (line 1 for all patients)
start_date_map = dict(zip(prediction_times['DFCI_MRN'], prediction_times['treatment_start_date']))
notes_meta['ANALYSIS_START_DT'] = notes_meta['DFCI_MRN'].map(start_date_map)
notes_meta['NOTE_TIME_REL_ANALYSIS_START_DT'] = (
    pd.to_datetime(notes_meta['NOTE_DATETIME']) - pd.to_datetime(notes_meta['ANALYSIS_START_DT'])
).dt.days

# Generate embedding features with time-decay-mean pooling
note_types = ['Clinician', 'Imaging', 'Pathology']
surv_for_risk = patient_df[['DFCI_MRN', 'death', 'tt_death']].copy()
risk_embedding_df = (generate_survival_embedding_df(
                        notes_meta, surv_for_risk, embeddings,
                        note_types=note_types,
                        pool_fx={key: 'time_decay_mean' for key in note_types},
                        decay_param=0.01,
                        note_timing_col='NOTE_TIME_REL_ANALYSIS_START_DT')
                    .merge(patient_df[['DFCI_MRN', 'GENDER', 'AGE_AT_TREATMENTSTART']], on='DFCI_MRN')
                    .merge(cancer_type_df, on='DFCI_MRN')).dropna()

# Purely prognostic model: embeddings + demographics + cancer type (NO treatment indicator)
risk_base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART'] + [
    col for col in risk_embedding_df if col.startswith('CANCER_TYPE')]
embed_cols = [c for c in risk_embedding_df.columns if 'EMBEDDING' in c or '2015' in c]
continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols

print("Running grid search for text risk model hyperparameters...")
event = 'death'
alphas_to_test = np.logspace(-5, 0, 25)
l1_ratios = [0.5, 1.0]

_, val_results, _ = run_grid_CoxPH_parallel(
    risk_embedding_df, risk_base_vars, continuous_vars, embed_cols,
    l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=5000, verbose=5)

best_l1_ratio, best_alpha = val_results.sort_values(
    by='mean_auc(t)', ascending=False).iloc[0][['l1_ratio', 'alpha']]
print(f"Best risk model: l1_ratio={best_l1_ratio}, alpha={best_alpha}")

print("Generating held-out text risk scores...")
risk_scores = (get_heldout_risk_scores_CoxPH(
                  risk_embedding_df, risk_base_vars, continuous_vars, embed_cols,
                  event_col=event, tstop_col=f'tt_{event}', penalized=True,
                  l1_ratio=best_l1_ratio, alpha=best_alpha, max_iter=5000)
              .rename(columns={'risk_score': 'text_risk_score'}))

# Merge risk scores into patient dataframe
patient_df = patient_df.merge(risk_scores[['DFCI_MRN', 'text_risk_score']], on='DFCI_MRN', how='left')
print(f"Text risk scores: {patient_df['text_risk_score'].notna().sum()} / {len(patient_df)} patients")

# --- Select final columns ---
required_cols = ['DFCI_MRN', 'tt_death', 'death']
base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
drop_cols = set(required_cols + base_vars + ['PX_on_ICI', 'ICI_prediction', 'text_risk_score',
                'first_treatment_date', 'treatment_start_date', 'ground_truth', 'model_probs'])
biomarker_cols = [col for col in patient_df.columns if col not in drop_cols]

interaction_ICI_df = patient_df[required_cols + base_vars + biomarker_cols
                               + ['PX_on_ICI', 'ICI_prediction', 'text_risk_score']].copy()

interaction_ICI_df = interaction_ICI_df.dropna(subset=['ICI_prediction', 'tt_death', 'death']).copy()
interaction_ICI_df['PX_on_ICI'] = interaction_ICI_df['PX_on_ICI'].astype(int)
interaction_ICI_df['death'] = interaction_ICI_df['death'].astype(int)

interaction_ICI_df.to_csv(os.path.join(MARKER_PATH, 'IPTW_ICI_interaction_runs_df.csv'), index=False)
