# === Imports ===
import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from embed_surv_utils import run_grid_CoxPH_parallel, run_base_CoxPH

month_offset = 3

# === Paths ===
FIGURE_PATH = '/data/gusev/USERS/jpconnor/figures/clinical_text_embedding_project/model_metrics/'
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
FEATURE_PATH = os.path.join(DATA_PATH, 'clinical_and_genomic_features/')
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
LANDMARK_PATH = os.path.join(SURV_PATH, 'time-to-icd/landmark_datasets/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/icd_results/')
OUTPUT_PATH = os.path.join(RESULTS_PATH, f'landmark_results/plus_{month_offset}_months/')
os.makedirs(OUTPUT_PATH, exist_ok=True)

os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

# Load text data
time_decayed_events_df = pd.read_csv(os.path.join(LANDMARK_PATH, f'time_decayed_events_df_plus_{month_offset}_months.csv'))

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
events = [col.split('_', 1)[1] for col in time_decayed_events_df.columns if (col.startswith('tt') and (col.split('_', 1)[1] not in met_events))]
tt_events = [f"tt_{e}" for e in events]

# read in twelve month dict
twelve_month_events_df = pd.read_csv(os.path.join(LANDMARK_PATH, 'time_decayed_events_df_plus_12_months.csv'),
                                    usecols=['DFCI_MRN'] + events + tt_events)
event_mrn_dict = {event : twelve_month_events_df.loc[twelve_month_events_df[f'tt_{event}'] > 0, 'DFCI_MRN'].unique().tolist() for event in events} 

# CoxPH hyperparameters
alphas_to_test = np.logspace(-5, 0, 30)
l1_ratios = [0.5, 1.0]

max_iter=1_000
for event in tqdm(events[100:200]):
    event_path = os.path.join(OUTPUT_PATH, event)
    os.makedirs(event_path, exist_ok=True)
    
    event_pred_df = full_prediction_df.loc[full_prediction_df['DFCI_MRN'].isin(event_mrn_dict[event])].copy()

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