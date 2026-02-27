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
import pandas as pd

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

# --- Select final columns ---
required_cols = ['DFCI_MRN', 'tt_death', 'death']
base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
drop_cols = set(required_cols + base_vars + ['PX_on_ICI', 'ICI_prediction',
                'first_treatment_date', 'treatment_start_date', 'ground_truth', 'model_probs'])
biomarker_cols = [col for col in patient_df.columns if col not in drop_cols]

output_cols = required_cols + base_vars + biomarker_cols + ['PX_on_ICI', 'ICI_prediction']
interaction_ICI_df = patient_df[output_cols].copy()

interaction_ICI_df = interaction_ICI_df.dropna(subset=['ICI_prediction', 'tt_death', 'death']).copy()
interaction_ICI_df['PX_on_ICI'] = interaction_ICI_df['PX_on_ICI'].astype(int)
interaction_ICI_df['death'] = interaction_ICI_df['death'].astype(int)

output_file = os.path.join(MARKER_PATH, f'IPTW_ICI_interaction_runs_df_{COHORT}.csv')
interaction_ICI_df.to_csv(output_file, index=False)
print(f"[generate_IPTW_df] Saved {len(interaction_ICI_df)} patients to {output_file}")
