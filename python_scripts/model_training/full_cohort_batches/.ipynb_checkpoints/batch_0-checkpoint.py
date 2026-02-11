import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from embed_surv_utils import run_base_CoxPH, run_grid_CoxPH_parallel

os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/clinical_text_project/data/'
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/')
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/VTE_data/processed_datasets/')
STAGE_PATH = '/data/gusev/PROFILE/CLINICAL/OncDRS/DERIVED_FROM_CLINICAL_TEXTS_2024_03/derived_files/cancer_stage/'
OUTPUT_PATH = os.path.join(RESULTS_PATH, 'phecode_model_comps_full_cohort_all_notes/')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load datasets
cancer_type_df = pd.read_csv('/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/first_treatments_dfci_w_inferred_cancers.csv', 
                             usecols=['DFCI_MRN', 'med_genomics_merged_cancer_group']).dropna()
cancer_type_df.rename(columns={'med_genomics_merged_cancer_group': 'CANCER_TYPE'}, inplace=True)

time_decayed_events_df = pd.read_csv(os.path.join(SURV_PATH, 'time-to-phecode/time_decayed_events_df.csv')).dropna()

# Define model columns
base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
events = [col.split('_', 1)[1] for col in time_decayed_events_df.columns if col.startswith('tt')]
tt_events = [f"tt_{event}" for event in events]
required_cols = ['DFCI_MRN'] + base_vars + events + tt_events

# Assemble prediction datasets
base_pred_df = pd.get_dummies(time_decayed_events_df[required_cols].merge(cancer_type_df, on='DFCI_MRN'),
                              columns=['CANCER_TYPE'], drop_first=True)

# Column groups
embed_cols = [c for c in time_decayed_events_df.columns if 'EMBEDDING' in c or '2015' in c]
continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols
cancer_type_cols = [col for col in base_pred_df if 'CANCER_TYPE' in col]

embeddings_pred_df = time_decayed_events_df[required_cols + embed_cols].merge(base_pred_df[['DFCI_MRN'] + cancer_type_cols], on='DFCI_MRN')

# CoxPH hyperparameters
alphas_to_test = np.logspace(-5, 0, 30)
l1_ratios = [0.5, 1.0]

events_data = time_decayed_events_df[[event for event in events]]
event_freq = events_data.sum(axis=0) / len(events_data)
events_to_include = event_freq[event_freq >= 0.05].index
events_sub = list(set(events_to_include) - set(os.listdir(OUTPUT_PATH)))

for event in tqdm(events_sub[0:10]):
    
    base_results = run_base_CoxPH(
        base_pred_df, base_vars + cancer_type_cols, ['AGE_AT_TREATMENTSTART'],
        event_col=event, tstop_col=f'tt_{event}')
    
    embed_plus_type_test_results, embed_plus_type_val_results, _ = run_grid_CoxPH_parallel(
        embeddings_pred_df, base_vars + cancer_type_cols, continuous_vars, embed_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=5000)
    
    event_path = os.path.join(OUTPUT_PATH, event)
    os.makedirs(event_path, exist_ok=True)
    
    base_results.to_csv(os.path.join(event_path, 'coxPH_base_model_metrics.csv'))
    
    embed_plus_type_val_results.to_csv(os.path.join(event_path, 'coxPH_decayed_embeddings_plus_type_val_metrics.csv'), index=False)
    embed_plus_type_test_results.to_csv(os.path.join(event_path, 'coxPH_decayed_embeddings_plus_type_test_metrics.csv'), index=False)