# === Imports ===
import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from embed_surv_utils import run_grid_CoxPH_parallel, run_base_CoxPH

# === Paths ===
PROJ_PATH = '/data/gusev/USERS/jpconnor/clinical_text_project/'
FIGURE_PATH = os.path.join(PROJ_PATH, 'figures/model_metrics/')
DATA_PATH = os.path.join(PROJ_PATH, 'data/')
FEATURE_PATH = os.path.join(DATA_PATH, 'clinical_and_genomic_features/')
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/')
OUTPUT_PATH = os.path.join(RESULTS_PATH, 'icd10_level_3_full_cohort/')
os.makedirs(OUTPUT_PATH, exist_ok=True)

os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

# Load text data
time_decayed_events_df = pd.read_csv(os.path.join(SURV_PATH, 'time-to-icd/time_decayed_events_df.csv'))

# load clinical and genomic features
cancer_type_df = pd.read_csv(os.path.join(FEATURE_PATH, 'cancer_type_df.csv'))

# feature classes
type_cols = [col for col in cancer_type_df.columns if 'CANCER_TYPE_' in col]
embed_cols = [col for col in time_decayed_events_df.columns if ('EMBEDDING' in col or '2015' in col)]

# other vars
base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols

full_prediction_df = time_decayed_events_df.merge(cancer_type_df[['DFCI_MRN'] + type_cols], on='DFCI_MRN')

met_events = ['brainM', 'boneM', 'adrenalM', 'liverM', 'lungM', 'nodeM', 'peritonealM']
event_pairs = [(col, f'tt_{col}') for col in met_events]

baseline_met_cols = []
for ind_col, time_col in event_pairs:
    col = f"{ind_col}_baseline_met"
    full_prediction_df[col] = (full_prediction_df[ind_col] == 1) & (full_prediction_df[time_col] <= 0)
    baseline_met_cols.append(col)

full_prediction_df["has_baseline_met"] = full_prediction_df[baseline_met_cols].any(axis=1)
met_free_prediction_df = full_prediction_df.loc[~full_prediction_df["has_baseline_met"]].copy()

# CoxPH hyperparameters
alphas_to_test = np.logspace(-5, 0, 30)
l1_ratios = [0.5, 1.0]

max_iter=1_000
for met_event in tqdm(met_events):
    event_path = os.path.join(OUTPUT_PATH, met_event)
    os.makedirs(event_path, exist_ok=True)
    
    event_pred_df = met_free_prediction_df.loc[met_free_prediction_df[f'tt_{met_event}'] > 0].copy()
    if met_event == 'brainM':
        event_pred_df = event_pred_df.loc[~event_pred_df['CANCER_TYPE_BRAIN']]
        
    print(met_event)
    print(len(event_pred_df))
    
    # text
    text_start = time.time()
    text_test, text_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'] + embed_cols, embed_cols,
        l1_ratios, alphas_to_test, event_col=met_event, tstop_col=f'tt_{met_event}', max_iter=max_iter)
    print(f'text complete in {(time.time() - text_start) / 60 : 0.2f} \n')
    
    text_test.to_csv(os.path.join(event_path, 'text_test.csv'), index=False)
    text_val.to_csv(os.path.join(event_path, 'text_val.csv'), index=False)
    
    base_results = run_base_CoxPH(
        event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'],
        event_col=met_event, tstop_col=f'tt_{met_event}')

    base_results.to_csv(os.path.join(event_path, 'type_model_metrics.csv'))