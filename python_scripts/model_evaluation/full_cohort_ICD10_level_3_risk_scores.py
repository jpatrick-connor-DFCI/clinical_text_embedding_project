# === Imports ===
import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from functools import reduce
from embed_surv_utils import run_grid_CoxPH_parallel, get_heldout_risk_scores_CoxPH

# === Paths ===
FIGURE_PATH = '/data/gusev/USERS/jpconnor/figures/clinical_text_embedding_project/model_metrics/'
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/icd_results/')
FEATURE_PATH = os.path.join(DATA_PATH, 'clinical_and_genomic_features/')
FULL_COHORT_PATH = os.path.join(RESULTS_PATH, 'full_cohort/')
FEATURE_COMPS_PATH = os.path.join(RESULTS_PATH, 'feature_comps/')
HELD_OUT_PATH = os.path.join(RESULTS_PATH, 'full_cohort_predicted_risk/')
os.makedirs(HELD_OUT_PATH, exist_ok=True)

events = list(set(os.listdir(FULL_COHORT_PATH)) & set(os.listdir(FEATURE_COMPS_PATH)))

os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

# Load text data
time_decayed_events_df = pd.read_csv(os.path.join(SURV_PATH, 'time-to-icd/time_decayed_events_df.csv'))

# load clinical and genomic features
mrn_stage_df = pd.read_csv(os.path.join(FEATURE_PATH, 'cancer_stage_df.csv'))
cancer_type_df = pd.read_csv(os.path.join(FEATURE_PATH, 'cancer_type_df.csv'))
somatic_df = pd.read_csv(os.path.join(FEATURE_PATH, 'PROFILE_2024_MUTATION_CARRIERS.csv'))
prs_df = pd.read_csv(os.path.join(FEATURE_PATH, 'PGS_DATA_VTE_COHORT.csv'))
treatment_df = pd.read_csv(os.path.join(FEATURE_PATH, 'categorical_treatment_data_by_line.csv'))
labs_df = pd.read_csv(os.path.join(FEATURE_PATH, 'mean_lab_vals_pre_first_treatment.csv'))

# feature classes
stage_cols = [col for col in mrn_stage_df.columns if 'CANCER_STAGE_' in col]
type_cols = [col for col in cancer_type_df.columns if 'CANCER_TYPE_' in col]
somatic_cols = [col for col in somatic_df.columns if col != 'DFCI_MRN']
prs_cols = [col for col in prs_df.columns if 'PGS' in col]
treatment_cols = [col for col in treatment_df.columns if 'PX_on_' in col]
embed_cols = [col for col in time_decayed_events_df.columns if ('EMBEDDING' in col or '2015' in col)]
labs_cols = [col for col in labs_df.columns if col != 'DFCI_MRN']

# other vars
base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols + prs_cols

full_prediction_df = time_decayed_events_df.merge(cancer_type_df[['DFCI_MRN'] + type_cols], on='DFCI_MRN')

# Find all time-to-event columns
events = [col.split('_', 1)[1] for col in time_decayed_events_df.columns if col.startswith('tt')]
tt_events = [f"tt_{e}" for e in events]

# CoxPH hyperparameters
alphas_to_test = np.logspace(-5, 0, 30)
l1_ratios = [0.5, 1.0]

max_iter=1_000
event = 'death'

event_path = os.path.join(FULL_COHORT_PATH, event)
os.makedirs(event_path, exist_ok=True)

event_pred_df = full_prediction_df.loc[full_prediction_df[f'tt_{event}'] > 0].copy()

event_l1, event_alpha = pd.read_csv(os.path.join(event_path, 'text_val.csv')).sort_values(
                                    by='mean_c_index', ascending=False)[['l1_ratio', 'alpha']].values[0]

event_risk_scores = get_heldout_risk_scores_CoxPH(event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'], embed_cols,
                                             event_col=event, tstop_col=f'tt_{event}', max_iter=max_iter, penalized=True, verbose=10,
                                             l1_ratio=event_l1, alpha=event_alpha).rename(columns={'risk_score' : 'text_risk_score'})
event_risk_scores.to_csv(os.path.join(HELD_OUT_PATH, 'death_held_out_preds.csv'), index=False)