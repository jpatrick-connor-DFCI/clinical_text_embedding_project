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
HELD_OUT_PATH = os.path.join(RESULTS_PATH, 'feature_specific_risk/')
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

# PCA dictionary
prs_pca_dict = {'PGS' : (prs_cols, 1500)}

full_prediction_df = (time_decayed_events_df
                      .merge(somatic_df[['DFCI_MRN'] + somatic_cols], on='DFCI_MRN')
                      .merge(prs_df[['DFCI_MRN'] + prs_cols], on='DFCI_MRN')
                      .merge(treatment_df.loc[treatment_df['treatment_line'] == 1, ['DFCI_MRN'] + treatment_cols], on='DFCI_MRN')
                      .merge(cancer_type_df[['DFCI_MRN'] + type_cols], on='DFCI_MRN')
                      .merge(mrn_stage_df[['DFCI_MRN'] + stage_cols], on='DFCI_MRN')
                      .merge(labs_df[['DFCI_MRN'] + labs_cols], on='DFCI_MRN'))

# Find all time-to-event columns
met_events = ['brainM', 'boneM', 'adrenalM', 'liverM', 'lungM', 'nodeM', 'peritonealM']
event_pairs = [(col, f'tt_{col}') for col in met_events]
events = [col.split('_', 1)[1] for col in time_decayed_events_df.columns if (col.startswith('tt') and col not in met_events)]

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

for event in tqdm(met_events):
    event_path = os.path.join(FEATURE_COMPS_PATH, event)
    os.makedirs(event_path, exist_ok=True)
    
    event_pred_df = met_free_prediction_df.loc[met_free_prediction_df[f'tt_{event}'] > 0].copy()
    if event == 'brainM':
        event_pred_df = event_pred_df.loc[~event_pred_df['CANCER_TYPE_BRAIN']]
        
    stage_l1, stage_alpha = pd.read_csv(os.path.join(event_path, 'stage_val.csv')).sort_values(
                                        by='mean_c_index', ascending=False)[['l1_ratio', 'alpha']].values[0]
    stage_scores = get_heldout_risk_scores_CoxPH(event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'], stage_cols,
                                                 event_col=event, tstop_col=f'tt_{event}', max_iter=max_iter, penalized=True,
                                                 l1_ratio=stage_l1, alpha=stage_alpha).rename(columns={'risk_score' : 'stage_risk_score'})
    
    treatment_l1, treatment_alpha = pd.read_csv(os.path.join(event_path, 'treatment_val.csv')).sort_values(
                                                by='mean_c_index', ascending=False)[['l1_ratio', 'alpha']].values[0]
    treatment_scores = get_heldout_risk_scores_CoxPH(event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'], treatment_cols,
                                                     event_col=event, tstop_col=f'tt_{event}', max_iter=max_iter, penalized=True, 
                                                     l1_ratio=treatment_l1, alpha=treatment_alpha).rename(columns={'risk_score' : 'treatment_risk_score'})
    
    lab_l1, lab_alpha = pd.read_csv(os.path.join(event_path, 'labs_val.csv')).sort_values(
                                    by='mean_c_index', ascending=False)[['l1_ratio', 'alpha']].values[0]
    lab_scores = get_heldout_risk_scores_CoxPH(event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'], labs_cols,
                                               event_col=event, tstop_col=f'tt_{event}', max_iter=max_iter, penalized=True,
                                               l1_ratio=lab_l1, alpha=lab_alpha).rename(columns={'risk_score' : 'labs_risk_score'})
    
    somatic_l1, somatic_alpha = pd.read_csv(os.path.join(event_path, 'somatic_val.csv')).sort_values(
                                            by='mean_c_index', ascending=False)[['l1_ratio', 'alpha']].values[0]
    somatic_scores = get_heldout_risk_scores_CoxPH(event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'], somatic_cols,
                                                   event_col=event, tstop_col=f'tt_{event}', max_iter=max_iter, penalized=True,
                                                   l1_ratio=somatic_l1, alpha=somatic_alpha).rename(columns={'risk_score' : 'somatic_risk_score'})
    
    prs_l1, prs_alpha = pd.read_csv(os.path.join(event_path, 'prs_val.csv')).sort_values(
                                    by='mean_c_index', ascending=False)[['l1_ratio', 'alpha']].values[0]
    prs_scores = get_heldout_risk_scores_CoxPH(event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'], prs_cols,
                                               event_col=event, tstop_col=f'tt_{event}', max_iter=max_iter, penalized=True,
                                               l1_ratio=prs_l1, alpha=prs_alpha).rename(columns={'risk_score' : 'prs_risk_score'})
    
    text_l1, text_alpha = pd.read_csv(os.path.join(event_path, 'text_val.csv')).sort_values(
                                      by='mean_c_index', ascending=False)[['l1_ratio', 'alpha']].values[0]
    text_scores = get_heldout_risk_scores_CoxPH(event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'], embed_cols,
                                                event_col=event, tstop_col=f'tt_{event}', max_iter=max_iter, penalized=True,
                                                l1_ratio=text_l1, alpha=text_alpha).rename(columns={'risk_score' : 'text_risk_score'})
    
    complete_risk_df = reduce(lambda left, right: left.merge(right, on="DFCI_MRN", how="inner"), 
                             [stage_scores, treatment_scores, somatic_scores, prs_scores, text_scores])
    
    complete_risk_df.to_csv(os.path.join(HELD_OUT_PATH, f'{event}_held_out_preds.csv'), index=False)

# for event in events:
for event in tqdm(events):
    event_path = os.path.join(FEATURE_COMPS_PATH, event)
    os.makedirs(event_path, exist_ok=True)
    
    event_pred_df = full_prediction_df.loc[full_prediction_df[f'tt_{event}'] > 0].copy()
    
    stage_l1, stage_alpha = pd.read_csv(os.path.join(event_path, 'stage_val.csv')).sort_values(
                                        by='mean_c_index', ascending=False)[['l1_ratio', 'alpha']].values[0]
    stage_scores = get_heldout_risk_scores_CoxPH(event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'], stage_cols,
                                                 event_col=event, tstop_col=f'tt_{event}', max_iter=max_iter, penalized=True,
                                                 l1_ratio=stage_l1, alpha=stage_alpha).rename(columns={'risk_score' : 'stage_risk_score'})
    
    treatment_l1, treatment_alpha = pd.read_csv(os.path.join(event_path, 'treatment_val.csv')).sort_values(
                                                by='mean_c_index', ascending=False)[['l1_ratio', 'alpha']].values[0]
    treatment_scores = get_heldout_risk_scores_CoxPH(event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'], treatment_cols,
                                                     event_col=event, tstop_col=f'tt_{event}', max_iter=max_iter, penalized=True, 
                                                     l1_ratio=treatment_l1, alpha=treatment_alpha).rename(columns={'risk_score' : 'treatment_risk_score'})
    
    lab_l1, lab_alpha = pd.read_csv(os.path.join(event_path, 'labs_val.csv')).sort_values(
                                    by='mean_c_index', ascending=False)[['l1_ratio', 'alpha']].values[0]
    lab_scores = get_heldout_risk_scores_CoxPH(event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'], labs_cols,
                                               event_col=event, tstop_col=f'tt_{event}', max_iter=max_iter, penalized=True,
                                               l1_ratio=lab_l1, alpha=lab_alpha).rename(columns={'risk_score' : 'labs_risk_score'})
    
    somatic_l1, somatic_alpha = pd.read_csv(os.path.join(event_path, 'somatic_val.csv')).sort_values(
                                            by='mean_c_index', ascending=False)[['l1_ratio', 'alpha']].values[0]
    somatic_scores = get_heldout_risk_scores_CoxPH(event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'], somatic_cols,
                                                   event_col=event, tstop_col=f'tt_{event}', max_iter=max_iter, penalized=True,
                                                   l1_ratio=somatic_l1, alpha=somatic_alpha).rename(columns={'risk_score' : 'somatic_risk_score'})
    
    prs_l1, prs_alpha = pd.read_csv(os.path.join(event_path, 'prs_val.csv')).sort_values(
                                    by='mean_c_index', ascending=False)[['l1_ratio', 'alpha']].values[0]
    prs_scores = get_heldout_risk_scores_CoxPH(event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'], prs_cols,
                                               event_col=event, tstop_col=f'tt_{event}', max_iter=max_iter, penalized=True,
                                               l1_ratio=prs_l1, alpha=prs_alpha).rename(columns={'risk_score' : 'prs_risk_score'})
    
    text_l1, text_alpha = pd.read_csv(os.path.join(event_path, 'text_val.csv')).sort_values(
                                      by='mean_c_index', ascending=False)[['l1_ratio', 'alpha']].values[0]
    text_scores = get_heldout_risk_scores_CoxPH(event_pred_df, base_vars + type_cols, ['AGE_AT_TREATMENTSTART'], embed_cols,
                                                event_col=event, tstop_col=f'tt_{event}', max_iter=max_iter, penalized=True,
                                                l1_ratio=text_l1, alpha=text_alpha).rename(columns={'risk_score' : 'text_risk_score'})
    
    complete_risk_df = reduce(lambda left, right: left.merge(right, on="DFCI_MRN", how="inner"), 
                             [stage_scores, treatment_scores, somatic_scores, prs_scores, text_scores])
    
    complete_risk_df.to_csv(os.path.join(HELD_OUT_PATH, f'{event}_held_out_preds.csv'), index=False)