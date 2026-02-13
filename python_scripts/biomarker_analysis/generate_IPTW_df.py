"""Generate IPTW dataset for biomarker analysis (IO vs non-IO with propensity scores)."""

import os
import random
import pandas as pd

random.seed(42)  # set seed for reproducibility

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
ICI_PATH = os.path.join(DATA_PATH, 'treatment_prediction/ICI_propensity/w_30_day_buffer/')
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')

# Load IO patient base df (from generate_IO_patient_df.py)
IO_patient_base_df = pd.read_csv(os.path.join(MARKER_PATH, 'IO_patient_base_df.csv')).drop_duplicates(subset=['DFCI_MRN'], keep='first')

# Load datasets for non-IO patients
cancer_type_df = pd.read_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/cancer_type_df.csv'))
tt_death_df = pd.read_csv(os.path.join(SURV_PATH, 'death_met_surv_df.csv'))
somatic_df = pd.read_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/complete_somatic_data_df.csv'))

# Load propensity predictions
line1_preds = pd.read_csv(os.path.join(ICI_PATH, 'line_1_predictions.csv'))
required_pred_cols = {'DFCI_MRN', 'ground_truth', 'model_probs'}
if not required_pred_cols.issubset(set(line1_preds.columns)):
    raise ValueError(f"line_1_predictions.csv must contain columns: {sorted(required_pred_cols)}")

line1_preds = line1_preds[['DFCI_MRN', 'ground_truth', 'model_probs']].dropna().copy()
line1_preds['ground_truth'] = line1_preds['ground_truth'].astype(int)

line1_mrns = line1_preds['DFCI_MRN'].unique()
line1_io_mrns = line1_preds.loc[line1_preds['ground_truth'] == 1, 'DFCI_MRN'].unique()
line1_non_io_mrns = line1_preds.loc[line1_preds['ground_truth'] == 0, 'DFCI_MRN'].unique()

# IO patients from base df
line1_IO_data = IO_patient_base_df.loc[IO_patient_base_df['DFCI_MRN'].isin(line1_io_mrns)]

# Non-IO patients assembled from survival + somatic + cancer type
line1_non_IO_data = (tt_death_df.loc[(tt_death_df['DFCI_MRN'].isin(line1_mrns)) &
                                     (tt_death_df['DFCI_MRN'].isin(line1_non_io_mrns))]
                     .merge(somatic_df, on='DFCI_MRN')
                     .merge(cancer_type_df, on='DFCI_MRN')).drop_duplicates(subset=['DFCI_MRN'], keep='first')

cols_to_include = list(set(line1_IO_data.columns) & set(line1_non_IO_data.columns))
required_cols = ['DFCI_MRN', 'tt_death', 'death']

base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
biomarker_cols = [col for col in cols_to_include if col not in (required_cols + base_vars)]

line1_IO_data['PX_on_IO'] = 1
line1_non_IO_data['PX_on_IO'] = 0

# Build analysis dataframe
interaction_IO_df = pd.concat([
    line1_IO_data[required_cols + base_vars + biomarker_cols + ['PX_on_IO']],
    line1_non_IO_data[required_cols + base_vars + biomarker_cols + ['PX_on_IO']]
], ignore_index=True)

# Map propensity scores from predictions
ps_map = dict(zip(line1_preds['DFCI_MRN'], line1_preds['model_probs']))
interaction_IO_df['IO_prediction'] = interaction_IO_df['DFCI_MRN'].map(ps_map)
interaction_IO_df['PX_on_IO'] = interaction_IO_df['DFCI_MRN'].map(dict(zip(line1_preds['DFCI_MRN'], line1_preds['ground_truth'])))

interaction_IO_df = interaction_IO_df.dropna(subset=['IO_prediction','tt_death','death']).copy()
interaction_IO_df['PX_on_IO'] = interaction_IO_df['PX_on_IO'].astype(int)
interaction_IO_df['death'] = interaction_IO_df['death'].astype(int)

interaction_IO_df.to_csv(os.path.join(MARKER_PATH, 'IPTW_IO_interaction_runs_df.csv'), index=False)
