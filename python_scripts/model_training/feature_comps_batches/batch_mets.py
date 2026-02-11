# === Imports ===
import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from embed_surv_utils import run_grid_CoxPH_parallel

# === Paths ===
FIGURE_PATH = '/data/gusev/USERS/jpconnor/figures/clinical_text_embedding_project/model_metrics/'
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
FEATURE_PATH = os.path.join(DATA_PATH, 'clinical_and_genomic_features/')
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/icd_results/')
OUTPUT_PATH = os.path.join(RESULTS_PATH, 'feature_comps/')
STAGE_PATH = '/data/gusev/PROFILE/CLINICAL/OncDRS/DERIVED_FROM_CLINICAL_TEXTS_2024_03/derived_files/cancer_stage/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

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

# PCA dictionary
prs_pca_dict = {'PGS' : (prs_cols, 1500)}

full_prediction_df = (time_decayed_events_df
                      .merge(somatic_df[['DFCI_MRN'] + somatic_cols], on='DFCI_MRN')
                      .merge(prs_df[['DFCI_MRN'] + prs_cols], on='DFCI_MRN')
                      .merge(treatment_df.loc[treatment_df['treatment_line'] == 1, ['DFCI_MRN'] + treatment_cols], on='DFCI_MRN')
                      .merge(cancer_type_df[['DFCI_MRN'] + type_cols], on='DFCI_MRN')
                      .merge(mrn_stage_df[['DFCI_MRN'] + stage_cols], on='DFCI_MRN')
                      .merge(labs_df[['DFCI_MRN'] + labs_cols], on='DFCI_MRN'))

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
    
    ## Single modality testing
    
    # stage
    stage_start = time.time()
    stage_test, stage_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'], stage_cols,
        l1_ratios, alphas_to_test, event_col=met_event, tstop_col=f'tt_{met_event}', max_iter=max_iter)

    print(f'stage complete in {(time.time() - stage_start) / 60 : 0.2f}')
    stage_test.to_csv(os.path.join(event_path, 'stage_test.csv'), index=False)
    stage_val.to_csv(os.path.join(event_path, 'stage_val.csv'), index=False)
    
    
    # treatment
    treatment_start = time.time()
    treatment_test, treatment_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'], treatment_cols,
        l1_ratios, alphas_to_test, event_col=met_event, tstop_col=f'tt_{met_event}', max_iter=max_iter)

    print(f'treatment complete in {(time.time() - treatment_start) / 60 : 0.2f}')
    treatment_test.to_csv(os.path.join(event_path, 'treatment_test.csv'), index=False)
    treatment_val.to_csv(os.path.join(event_path, 'treatment_val.csv'), index=False)

    # labs
    labs_start = time.time()
    lab_test, lab_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'] + labs_cols, labs_cols,
        l1_ratios, alphas_to_test, event_col=met_event, tstop_col=f'tt_{met_event}', max_iter=max_iter)

    print(f'labs complete in {(time.time() - labs_start) / 60 : 0.2f}')
    lab_test.to_csv(os.path.join(event_path, 'labs_test.csv'), index=False)
    lab_val.to_csv(os.path.join(event_path, 'labs_val.csv'), index=False)
    
    # somatic
    somatic_start = time.time()
    somatic_test, somatic_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'], somatic_cols,
        l1_ratios, alphas_to_test, event_col=met_event, tstop_col=f'tt_{met_event}', max_iter=max_iter)
    print(f'somatic complete in {(time.time() - somatic_start) / 60 : 0.2f}')
    
    somatic_test.to_csv(os.path.join(event_path, 'somatic_test.csv'), index=False)
    somatic_val.to_csv(os.path.join(event_path, 'somatic_val.csv'), index=False)
    
    # PRS
    prs_start = time.time()
    prs_test, prs_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'], prs_cols,
        l1_ratios, alphas_to_test, pca_config=prs_pca_dict, event_col=met_event, tstop_col=f'tt_{met_event}', max_iter=max_iter)
    print(f'prs complete in {(time.time() - prs_start) / 60 : 0.2f}')
    
    prs_test.to_csv(os.path.join(event_path, 'prs_test.csv'), index=False)
    prs_val.to_csv(os.path.join(event_path, 'prs_val.csv'), index=False)
    
    # text
    text_start = time.time()
    text_test, text_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'] + embed_cols, embed_cols,
        l1_ratios, alphas_to_test, event_col=met_event, tstop_col=f'tt_{met_event}', max_iter=max_iter)
    print(f'text complete in {(time.time() - text_start) / 60 : 0.2f} \n')
    
    text_test.to_csv(os.path.join(event_path, 'text_test.csv'), index=False)
    text_val.to_csv(os.path.join(event_path, 'text_val.csv'), index=False)