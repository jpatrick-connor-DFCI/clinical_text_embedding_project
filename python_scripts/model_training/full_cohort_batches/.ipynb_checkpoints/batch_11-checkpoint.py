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

# Find all time-to-event columns
met_events = ['brainM', 'boneM', 'adrenalM', 'liverM', 'lungM', 'nodeM', 'peritonealM']
events = [col.split('_', 1)[1] for col in time_decayed_events_df.columns if (col.startswith('tt') and (col not in met_events))]
tt_events = [f"tt_{e}" for e in events]

# CoxPH hyperparameters
alphas_to_test = np.logspace(-5, 0, 30)
l1_ratios = [0.5, 1.0]

max_iter=1_000
for event in tqdm(events[275:300]):
    event_path = os.path.join(OUTPUT_PATH, event)
    os.makedirs(event_path, exist_ok=True)
    
    event_pred_df = full_prediction_df.loc[full_prediction_df[f'tt_{event}'] > 0].copy()

    print(event)
    
    # text
    text_start = time.time()
    text_test, text_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'] + embed_cols, embed_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=max_iter)
    print(f'text complete in {(time.time() - text_start) / 60 : 0.2f} \n')
    
    text_test.to_csv(os.path.join(event_path, 'text_test.csv'), index=False)
    text_val.to_csv(os.path.join(event_path, 'text_val.csv'), index=False)
    
    base_results = run_base_CoxPH(
        event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'],
        event_col=event, tstop_col=f'tt_{event}')

    base_results.to_csv(os.path.join(event_path, 'type_model_metrics.csv'))