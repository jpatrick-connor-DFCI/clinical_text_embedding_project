"""Generate IPTW dataset for biomarker analysis (ICI vs never-ICI).

Builds on cohort-specific propensity scores from ICI_LRs.py.
Includes line_category as a covariate (dummy-coded, line 1 as reference).
Also includes clinical text embeddings for prognostic score adjustment.

Usage:
  python generate_IPTW_df.py --cohort cohort1 --ps_model embeddings_only
  python generate_IPTW_df.py --cohort cohort2 --ps_model all_covariates
"""

import os
import argparse
import random
import pandas as pd

random.seed(42)

parser = argparse.ArgumentParser(description='Generate IPTW dataset for biomarker analysis.')
parser.add_argument('--cohort', choices=['cohort1', 'cohort2'], required=True,
                    help='cohort1=first-line unmatched, cohort2=line-matched lines 1-3')
parser.add_argument('--ps_model', choices=['covariates_only', 'covariates_plus_embeddings'], required=True,
                    help='Propensity score model: covariates_only or covariates_plus_embeddings')
args = parser.parse_args()

COHORT = args.cohort
PS_MODEL = args.ps_model

# === Paths ===
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
PRED_BASE = os.path.join(DATA_PATH, f'treatment_prediction/{COHORT}/')
PRED_DATA_PATH = os.path.join(PRED_BASE, 'prediction_data/')
PS_PATH = os.path.join(PRED_BASE, f'{PS_MODEL}_propensity/w_30_day_buffer/')
MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')
os.makedirs(MARKER_PATH, exist_ok=True)

print(f"[generate_IPTW_df] Cohort: {COHORT}, PS model: {PS_MODEL}")

# === Load base survival data ===
tt_death_df = pd.read_csv(os.path.join(SURV_PATH, 'death_met_surv_df.csv'))
tt_death_df['first_treatment_date'] = pd.to_datetime(tt_death_df['first_treatment_date'])
tt_death_df = tt_death_df[['DFCI_MRN', 'first_treatment_date', 'tt_death', 'death',
                            'GENDER', 'AGE_AT_TREATMENTSTART']].copy()

# === Load prediction times (includes line_category) ===
prediction_times = pd.read_csv(os.path.join(PRED_DATA_PATH, 'prediction_times.csv'))
prediction_times['treatment_start_date'] = pd.to_datetime(prediction_times['treatment_start_date'])
prediction_times = prediction_times.drop_duplicates(subset='DFCI_MRN', keep='first')

# === Load genomic / clinical features ===
cancer_type_df = pd.read_csv(
    os.path.join(DATA_PATH, 'clinical_and_genomic_features/cancer_type_df.csv'))
somatic_df = pd.read_csv(
    os.path.join(DATA_PATH, 'clinical_and_genomic_features/complete_somatic_data_df.csv'))
mutation_tags = ('_SNV', '_SV', '_FUSION', '_DEL', '_AMP')
panel_cols = [col for col in somatic_df.columns if col.upper().startswith('PANEL_VERSION')]
mutation_cols = [col for col in somatic_df.columns if any(tag in col.upper() for tag in mutation_tags)]
somatic_keep_cols = ['DFCI_MRN'] + panel_cols + mutation_cols
somatic_keep_cols = list(dict.fromkeys(somatic_keep_cols))
somatic_df = somatic_df[somatic_keep_cols].copy()

# === Load propensity predictions (30-day buffer) ===
preds = pd.read_csv(os.path.join(PS_PATH, 'predictions.csv'))
required_pred_cols = {'DFCI_MRN', 'ground_truth', 'model_probs'}
if not required_pred_cols.issubset(set(preds.columns)):
    raise ValueError(f"predictions.csv must contain columns: {sorted(required_pred_cols)}")
preds = preds[['DFCI_MRN', 'ground_truth', 'model_probs']].dropna().copy()
preds['ground_truth'] = preds['ground_truth'].astype(int)

# === Load clinical text embeddings (30-day buffer) for prognostic score ===
EMBED_BUFFER = 30
embed_file = os.path.join(PRED_DATA_PATH, f'w_{EMBED_BUFFER}_day_buffer/',
                          f'ICI_prediction_df_w_{EMBED_BUFFER}_day_buffer.csv')
embed_df = pd.read_csv(embed_file)
embedding_cols = [c for c in embed_df.columns
                  if ('IMAGING' in c) or ('PATHOLOGY' in c) or ('CLINICIAN' in c)]
embed_df = embed_df[['DFCI_MRN'] + embedding_cols].drop_duplicates(subset='DFCI_MRN', keep='first')
print(f"[generate_IPTW_df] Loaded {len(embedding_cols)} embedding columns from {embed_file}")

# === Build unified patient dataframe ===
patient_df = (tt_death_df
              .merge(prediction_times, on='DFCI_MRN')
              .merge(somatic_df, on='DFCI_MRN')
              .merge(cancer_type_df, on='DFCI_MRN')
              .merge(preds, on='DFCI_MRN')
              .merge(embed_df, on='DFCI_MRN', how='left')
              .drop_duplicates(subset=['DFCI_MRN'], keep='first'))

# One-hot encode panel version
if 'PANEL_VERSION' in patient_df.columns:
    patient_df = pd.get_dummies(patient_df, columns=['PANEL_VERSION'], drop_first=True)

# Dummy-code line_category (drop line 1 as reference)
patient_df = pd.get_dummies(patient_df, columns=['line_category'], prefix='LINE', drop_first=True, dtype=int)

# Drop patients with non-positive survival
patient_df = patient_df.loc[patient_df['tt_death'] > 0].copy()

# === Assign treatment group and propensity scores ===
patient_df['PX_on_ICI'] = patient_df['ground_truth'].astype(int)
patient_df['ICI_prediction'] = patient_df['model_probs']

# === Select final columns ===
required_cols = ['DFCI_MRN', 'tt_death', 'death']
base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
line_cols = [col for col in patient_df.columns if col.startswith('LINE_')]
meta_cols = ['PX_on_ICI', 'ICI_prediction', 'first_treatment_date', 'treatment_start_date',
             'ground_truth', 'model_probs']
drop_cols = set(required_cols + base_vars + line_cols + meta_cols + embedding_cols)
biomarker_cols = [col for col in patient_df.columns if col not in drop_cols]

output_cols = (required_cols + base_vars + line_cols + biomarker_cols +
               embedding_cols + ['PX_on_ICI', 'ICI_prediction'])
print(f"[generate_IPTW_df] {len(embedding_cols)} embedding cols included for prognostic score")
interaction_ICI_df = patient_df[output_cols].copy()

interaction_ICI_df = interaction_ICI_df.dropna(subset=['ICI_prediction', 'tt_death', 'death']).copy()
interaction_ICI_df['PX_on_ICI'] = interaction_ICI_df['PX_on_ICI'].astype(int)
interaction_ICI_df['death'] = interaction_ICI_df['death'].astype(int)

output_file = os.path.join(MARKER_PATH, f'IPTW_df_{COHORT}_{PS_MODEL}.csv')
interaction_ICI_df.to_csv(output_file, index=False)
print(f"[generate_IPTW_df] Saved {len(interaction_ICI_df)} patients to {output_file}")
print(f"  ICI: {interaction_ICI_df['PX_on_ICI'].sum()}, "
      f"Controls: {(~interaction_ICI_df['PX_on_ICI'].astype(bool)).sum()}")
print(f"  Line dummies: {line_cols}")
