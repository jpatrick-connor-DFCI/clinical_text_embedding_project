# === Imports ===
import os
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from embed_surv_utils import run_grid_CoxPH_parallel, get_heldout_risk_scores_CoxPH

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# === Paths ===
PROJ_PATH = '/data/gusev/USERS/jpconnor/clinical_text_project/'
FIGURE_PATH = os.path.join(PROJ_PATH, 'figures/model_metrics/')
DATA_PATH = os.path.join(PROJ_PATH, 'data/')
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/')
OUTPUT_PATH = os.path.join(RESULTS_PATH, 'phecode_complete_model/')
STAGE_PATH = '/data/gusev/PROFILE/CLINICAL/OncDRS/DERIVED_FROM_CLINICAL_TEXTS_2024_03/derived_files/cancer_stage/'

os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

# Load text data
time_decayed_events_df = pd.read_csv(os.path.join(SURV_PATH, 'time-to-phecode/time_decayed_events_df.csv'))

# Load datasets
mrn_stage_dict = pickle.load(open(os.path.join(STAGE_PATH, 'dfci_cancer_mrn_to_derived_cancer_stage.pkl'), 'rb'))
mrn_stage_df = pd.get_dummies(pd.DataFrame({'DFCI_MRN' : mrn_stage_dict.keys(), 
                                            'CANCER_STAGE' : mrn_stage_dict.values()}),
                              columns=['CANCER_STAGE'], drop_first=True)

# Load cancer types
cancer_type_df = pd.read_csv(
    '/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/first_treatments_dfci_w_inferred_cancers.csv',
    usecols=['DFCI_MRN', 'med_genomics_merged_cancer_group']).rename(columns={'med_genomics_merged_cancer_group': 'CANCER_TYPE'})
cancer_type_sub = cancer_type_df.loc[cancer_type_df['DFCI_MRN'].isin(time_decayed_events_df['DFCI_MRN'].unique())]

cancer_type_counts = cancer_type_sub['CANCER_TYPE'].value_counts()
types_to_keep = cancer_type_counts[cancer_type_counts >= 500].index.tolist()
cancer_type_sub['CANCER_TYPE'] = cancer_type_sub['CANCER_TYPE'].where(cancer_type_sub['CANCER_TYPE'].isin(types_to_keep), 'OTHER')
cancer_type_sub = pd.get_dummies(cancer_type_sub, columns=['CANCER_TYPE'], drop_first=True)

# Load genomics
somatic_df = pd.read_csv(os.path.join(DATA_PATH, 'PROFILE_2024_MUTATION_CARRIERS.csv')).drop_duplicates(subset='DFCI_MRN', keep='first')
somatic_df = pd.get_dummies(somatic_df.drop(columns=['sample_id', 'cbio_sample_id', 'cbio_patient_id', 
                                                     'onco_tree_code', 'briefcase', 'riker_pipeline_version', 
                                                     'riker_run_version', 'CANCER_TYPE']), columns=['PANEL_VERSION'])

# Load PRS
idmap = pd.read_csv("/data/gusev/PROFILE/CLINICAL/PROFILE_2024_idmap.csv").rename(columns={'MRN' : 'DFCI_MRN'})
prs_df = (pd.read_csv('/data/gusev/USERS/mjsaleh/PRS_PGScatalog/pgs_matrix_with_avg.tsv', sep='\t')
          .merge(idmap[['cbio_sample_id', 'DFCI_MRN']].rename(columns={'cbio_sample_id' : 'IID'}))
          .drop_duplicates(subset='DFCI_MRN', keep='first'))

# Example: Suppose you have a DataFrame X with numeric features
# X = pd.read_csv('data.csv')
X = prs_df[[col for col in prs_df.columns if 'PGS' in col]].values

# 1. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Apply PCA (e.g., reduce to 2 dimensions)
pca = PCA(n_components=1500)
X_pca = pca.fit_transform(X_scaled)

# 3. Convert back to DataFrame for convenience
prs_pca_df = pd.DataFrame(X_pca, columns=[f'PGS_PC_{idx+1}' for idx in range(1500)])
prs_pca_df['DFCI_MRN'] = prs_df['DFCI_MRN'].copy()

# Load treatment data
treatment_df = (pd.read_csv("/data/gusev/USERS/mjsaleh/profile_lines_of_rx/profile_rxlines.csv")
                .rename(columns={"MRN": "DFCI_MRN", "LOT_start_date": "treatment_start_date"}))
treatment_df["treatment_start_date"] = pd.to_datetime(treatment_df["treatment_start_date"])
treatment_df = treatment_df.sort_values(["DFCI_MRN", "treatment_start_date"])
treatment_df["treatment_line"] = treatment_df.groupby("DFCI_MRN").cumcount() + 1
treatments = (treatment_df["Treatment_type"]
              .str.replace(";", "", regex=False)
              .str.split().explode())
dummies = pd.get_dummies(treatments, prefix="PX_on", drop_first=True).groupby(level=0).max()
treatment_df = pd.concat([treatment_df, dummies], axis=1)

# define prediction column types
somatic_mutation_cols = [col for col in somatic_df.columns if col != 'DFCI_MRN']
prs_cols = [col for col in prs_pca_df.columns if 'PGS' in col]
treatment_cols = [c for c in treatment_df if c.startswith("PX_on_")]
cancer_type_cols = [col for col in cancer_type_sub.columns if col.startswith('CANCER_TYPE')]
stage_cols = [col for col in mrn_stage_df.columns if col.startswith('CANCER_STAGE')]
embed_cols = [c for c in time_decayed_events_df.columns if ('EMBEDDING' in c or '2015' in c)]
base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols + prs_cols

full_prediction_df = (time_decayed_events_df
                      .merge(somatic_df[['DFCI_MRN'] + somatic_mutation_cols], on='DFCI_MRN')
                      .merge(prs_pca_df[['DFCI_MRN'] + prs_cols], on='DFCI_MRN')
                      .merge(treatment_df.loc[treatment_df['treatment_line'] == 1, ['DFCI_MRN'] + treatment_cols], on='DFCI_MRN')
                      .merge(cancer_type_sub[['DFCI_MRN'] + cancer_type_cols], on='DFCI_MRN')
                      .merge(mrn_stage_df[['DFCI_MRN'] + stage_cols], on='DFCI_MRN'))

# Find all time-to-event columns
events = [col.split('_', 1)[1] for col in time_decayed_events_df.columns if col.startswith('tt')]
tt_events = [f"tt_{e}" for e in events]

# CoxPH hyperparameters
alphas_to_test = np.logspace(-5, 0, 30)
l1_ratios = [0.5, 1.0]

events_data = time_decayed_events_df[[event for event in events]]
event_freq = events_data.sum(axis=0) / len(events_data)
events_to_include = event_freq[event_freq >= 0.05].index

for event in tqdm(final_events):
    
    event_path = os.path.join(OUTPUT_PATH, event)
    os.makedirs(event_path, exist_ok=True)
    
    event_pred_df = full_prediction_df.loc[full_prediction_df[f'tt_{event}'] > 0].copy()

    # type + stage + somatic
    print('type + stage + somatic')
    type_stage_somatic_test, type_stage_somatic_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + cancer_type_cols + stage_cols, ['AGE_AT_TREATMENTSTART'], somatic_mutation_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=5000)

    type_stage_somatic_test.to_csv(os.path.join(event_path, 'type_stage_somatic_test.csv'), index=False)
    type_stage_somatic_val.to_csv(os.path.join(event_path, 'type_stage_somatic_val.csv'), index=False)
    
    # type + stage + prs
    print('type + stage + prs')
    type_stage_prs_test, type_stage_prs_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + cancer_type_cols + stage_cols, ['AGE_AT_TREATMENTSTART'] + prs_cols, prs_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=5000)
    
    type_stage_prs_test.to_csv(os.path.join(event_path, 'type_stage_prs_test.csv'), index=False)
    type_stage_prs_val.to_csv(os.path.join(event_path, 'type_stage_prs_val.csv'), index=False)

    # type + stage + somatic + prs    
    print('type + stage + somatic + prs')
    type_stage_somatic_prs_test, type_stage_somatic_prs_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + cancer_type_cols + stage_cols, ['AGE_AT_TREATMENTSTART'] + prs_cols, somatic_mutation_cols + prs_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=5000)

    type_stage_somatic_prs_test.to_csv(os.path.join(event_path, 'type_stage_somatic_prs_test.csv'), index=False)
    type_stage_somatic_prs_val.to_csv(os.path.join(event_path, 'type_stage_somatic_prs_val.csv'), index=False)

    # type + stage + treatment
    print('type + stage + treatment')
    type_stage_treatment_test, type_stage_treatment_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + cancer_type_cols + stage_cols, ['AGE_AT_TREATMENTSTART'], treatment_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=5000)

    type_stage_treatment_test.to_csv(os.path.join(event_path, 'type_stage_treatment_test.csv'), index=False)
    type_stage_treatment_val.to_csv(os.path.join(event_path, 'type_stage_treatment_val.csv'), index=False)

    # type + stage + treatment + somatic
    print('type + stage + treatment + somatic')
    type_stage_treatment_somatic_test, type_stage_treatment_somatic_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + cancer_type_cols + stage_cols, ['AGE_AT_TREATMENTSTART'], somatic_mutation_cols + treatment_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=5000)

    type_stage_treatment_somatic_test.to_csv(os.path.join(event_path, 'type_stage_treatment_somatic_test.csv'), index=False)
    type_stage_treatment_somatic_val.to_csv(os.path.join(event_path, 'type_stage_treatment_somatic_val.csv'), index=False)
    
    # type + stage + treatment + prs
    print('type + stage + treatment + prs')
    type_stage_treatment_prs_test, type_stage_treatment_prs_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + cancer_type_cols + stage_cols, ['AGE_AT_TREATMENTSTART'] + prs_cols, treatment_cols + prs_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=5000)

    type_stage_treatment_prs_test.to_csv(os.path.join(event_path, 'type_stage_treatment_prs_test.csv'), index=False)
    type_stage_treatment_prs_val.to_csv(os.path.join(event_path, 'type_stage_treatment_prs_val.csv'), index=False)
    
    # type + stage + treatment + somatic + prs
    print('type + stage + treatment + somatic + prs')
    type_stage_treatment_somatic_prs_test, type_stage_treatment_somatic_prs_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + cancer_type_cols + stage_cols, ['AGE_AT_TREATMENTSTART'] + prs_cols, treatment_cols + somatic_mutation_cols + prs_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=5000)
    
    type_stage_treatment_somatic_prs_test.to_csv(os.path.join(event_path, 'type_stage_treatment_somatic_prs_test.csv'), index=False)
    type_stage_treatment_somatic_prs_val.to_csv(os.path.join(event_path, 'type_stage_treatment_somatic_prs_val.csv'), index=False)
    
    # type + stage + text
    print('type + stage + text')
    type_stage_text_test, type_stage_text_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + cancer_type_cols + stage_cols, ['AGE_AT_TREATMENTSTART'] + embed_cols, embed_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=5000)

    type_stage_text_test.to_csv(os.path.join(event_path, 'type_stage_text_test.csv'), index=False)
    type_stage_text_val.to_csv(os.path.join(event_path, 'type_stage_text_val.csv'), index=False)    
    
    # type + stage + somatic + prs + text
    print('type + stage + somatic + prs + text')
    type_stage_somatic_prs_text_test, type_stage_somatic_prs_text_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + cancer_type_cols + stage_cols, ['AGE_AT_TREATMENTSTART'] + prs_cols + embed_cols,
        somatic_mutation_cols + prs_cols + embed_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=5000)

    type_stage_somatic_prs_text_test.to_csv(os.path.join(event_path, 'type_stage_somatic_prs_text_test.csv'), index=False)
    type_stage_somatic_prs_text_val.to_csv(os.path.join(event_path, 'type_stage_somatic_prs_text_val.csv'), index=False)

    # type + stage + somatic + prs + treatment + text
    print('type + stage + somatic + prs + treatment + text')
    type_stage_somatic_prs_treatment_text_test, type_stage_somatic_prs_treatment_text_val, _ = run_grid_CoxPH_parallel(
        event_pred_df, base_vars + cancer_type_cols + stage_cols, ['AGE_AT_TREATMENTSTART'] + prs_cols + embed_cols,
        somatic_mutation_cols + prs_cols + embed_cols + treatment_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=5000)

    type_stage_somatic_prs_treatment_text_test.to_csv(os.path.join(event_path, 'type_stage_somatic_prs_treatment_text_test.csv'), index=False)
    type_stage_somatic_prs_treatment_text_val.to_csv(os.path.join(event_path, 'type_stage_somatic_prs_treatment_text_val.csv'), index=False)