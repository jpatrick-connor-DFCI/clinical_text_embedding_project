"""Generate IPTW dataset for biomarker analysis (ICI vs non-ICI with propensity scores).

Time origin for all patients is LOT_start_date for line 1, matching the
prediction time used to generate propensity scores in ICI_LRs.py.
"""

import os
import random
import pandas as pd

random.seed(42)  # set seed for reproducibility

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
ICI_PATH = os.path.join(DATA_PATH, 'treatment_prediction/ICI_propensity/w_30_day_buffer/')
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')
INTAE_DATA_PATH = '/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/'

# --- Load base survival data (tt_death measured from first_treatment_date) ---
tt_death_df = pd.read_csv(os.path.join(SURV_PATH, 'death_met_surv_df.csv'))
tt_death_df['first_treatment_date'] = pd.to_datetime(tt_death_df['first_treatment_date'])
tt_death_df = tt_death_df[['DFCI_MRN', 'first_treatment_date', 'tt_death', 'death',
                            'GENDER', 'AGE_AT_TREATMENTSTART']].copy()

# --- Load line 1 treatment start dates (propensity score time origin) ---
treatment_df = pd.read_csv('/data/gusev/USERS/mjsaleh/profile_lines_of_rx/profile_rxlines.csv')
treatment_df['LOT_start_date'] = pd.to_datetime(treatment_df['LOT_start_date'])
treatment_df = treatment_df.sort_values(['MRN', 'LOT_start_date'])
treatment_df['treatment_line'] = treatment_df.groupby('MRN').cumcount() + 1
line1_starts = (treatment_df
                .loc[treatment_df['treatment_line'] == 1, ['MRN', 'LOT_start_date']]
                .rename(columns={'MRN': 'DFCI_MRN', 'LOT_start_date': 'line1_start_date'}))

# --- Load genomic / clinical features ---
cancer_type_df = pd.read_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/cancer_type_df.csv'))
somatic_df = pd.read_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/complete_somatic_data_df.csv'))

# --- Load propensity predictions (line 1, 30-day buffer) ---
line1_preds = pd.read_csv(os.path.join(ICI_PATH, 'line_1_predictions.csv'))
required_pred_cols = {'DFCI_MRN', 'ground_truth', 'model_probs'}
if not required_pred_cols.issubset(set(line1_preds.columns)):
    raise ValueError(f"line_1_predictions.csv must contain columns: {sorted(required_pred_cols)}")
line1_preds = line1_preds[['DFCI_MRN', 'ground_truth', 'model_probs']].dropna().copy()
line1_preds['ground_truth'] = line1_preds['ground_truth'].astype(int)

# --- Build unified patient dataframe ---
# Merge all data sources; restrict to patients with propensity predictions
patient_df = (tt_death_df
              .merge(line1_starts, on='DFCI_MRN')
              .merge(somatic_df, on='DFCI_MRN')
              .merge(cancer_type_df, on='DFCI_MRN')
              .merge(line1_preds, on='DFCI_MRN')
              .drop_duplicates(subset=['DFCI_MRN'], keep='first'))

# --- Recompute tt_death from line 1 start date ---
# Original tt_death is measured from first_treatment_date; shift to line1_start_date
days_offset = (patient_df['line1_start_date'] - patient_df['first_treatment_date']).dt.days
patient_df['tt_death'] = patient_df['tt_death'] - days_offset

# Drop patients with non-positive survival from line 1 start
patient_df = patient_df.loc[patient_df['tt_death'] > 0].copy()

# --- Assign treatment group and propensity scores ---
patient_df['PX_on_ICI'] = patient_df['ground_truth'].astype(int)
patient_df['IO_prediction'] = patient_df['model_probs']

# --- Select final columns ---
required_cols = ['DFCI_MRN', 'tt_death', 'death']
base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
drop_cols = set(required_cols + base_vars + ['PX_on_ICI', 'IO_prediction',
                'first_treatment_date', 'line1_start_date', 'ground_truth', 'model_probs'])
biomarker_cols = [col for col in patient_df.columns if col not in drop_cols]

interaction_IO_df = patient_df[required_cols + base_vars + biomarker_cols
                               + ['PX_on_ICI', 'IO_prediction']].copy()

interaction_IO_df = interaction_IO_df.dropna(subset=['IO_prediction', 'tt_death', 'death']).copy()
interaction_IO_df['PX_on_ICI'] = interaction_IO_df['PX_on_ICI'].astype(int)
interaction_IO_df['death'] = interaction_IO_df['death'].astype(int)

interaction_IO_df.to_csv(os.path.join(MARKER_PATH, 'IPTW_IO_interaction_runs_df.csv'), index=False)
