import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from embed_surv_utils import run_base_CoxPH, run_grid_CoxPH_parallel

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
DATA_PATH = "/data/gusev/USERS/jpconnor/clinical_text_project/data/"
SURV_PATH = os.path.join(DATA_PATH, "survival_data/")
NOTES_PATH = os.path.join(DATA_PATH, "batched_datasets/VTE_data/processed_datasets/")
ICI_PRED_PATH = os.path.join(DATA_PATH, "treatment_prediction/line_ICI_prediction_data/")
LINE_PRED_PATH = os.path.join(DATA_PATH,'treatment_prediction/time-to-next-treatment/')
OUTPUT_PATH = os.path.join(LINE_PRED_PATH, 'results/')

full_ttnt_df = pd.read_csv(os.path.join(LINE_PRED_PATH, 'full_tt_next_treatment_pred_df.csv'))

cancer_type_df = pd.read_csv('/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/first_treatments_dfci_w_inferred_cancers.csv',
                             usecols=['DFCI_MRN', 'med_genomics_merged_cancer_group']).rename(columns={'med_genomics_merged_cancer_group' : 'CANCER_TYPE'})

tt_phecodes_df = pd.read_csv(os.path.join(SURV_PATH, 'time-to-phecode/tt_vte_plus_phecodes.csv'))

full_ttnt_df = (full_ttnt_df
                .merge(tt_phecodes_df[['DFCI_MRN', 'AGE_AT_TREATMENTSTART', 'GENDER']], on='DFCI_MRN')
                .merge(cancer_type_df, on='DFCI_MRN'))

# Define model columns
target_cols = ['time_on_treatment', 'event']
base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
embed_cols = [c for c in full_ttnt_df.columns if 'EMBEDDING' in c or '2015' in c]
continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols

# CoxPH hyperparameters
alphas_to_test = np.logspace(-5, 0, 30)
l1_ratios = [0.5, 1.0]

for line in tqdm(list(range(1, 7))):
    line_pred_df = full_ttnt_df.loc[full_ttnt_df['treatment_line'] == line, base_vars + ['CANCER_TYPE'] + embed_cols + target_cols].dropna()
    
    cancer_type_counts = line_pred_df['CANCER_TYPE'].value_counts()
    rare_cancers = cancer_type_counts[cancer_type_counts < (5/100) * len(line_pred_df)].index
    line_pred_df['CANCER_TYPE'] = line_pred_df['CANCER_TYPE'].where(~line_pred_df['CANCER_TYPE'].isin(rare_cancers), 'OTHER')
    
    line_pred_df = pd.get_dummies(line_pred_df, columns=['CANCER_TYPE'], drop_first=True)
    cancer_type_cols = [col for col in line_pred_df.columns if 'CANCER_TYPE' in col]
    
    line_pred_df = line_pred_df.loc[line_pred_df['time_on_treatment'] > 0]
    
    base_results = run_base_CoxPH(
        line_pred_df, base_vars + cancer_type_cols, ['AGE_AT_TREATMENTSTART'],
        event_col='event', tstop_col='time_on_treatment')
    
    embed_plus_type_test_results, embed_plus_type_val_results, _ = run_grid_CoxPH_parallel(
        line_pred_df, base_vars + cancer_type_cols, continuous_vars, embed_cols,
        l1_ratios, alphas_to_test, event_col='event', tstop_col='time_on_treatment', max_iter=5000)

    line_path = os.path.join(OUTPUT_PATH, f'line_{line}')
    os.makedirs(line_path, exist_ok=True)
    
    base_results.to_csv(os.path.join(line_path, 'coxPH_base_model_metrics.csv'), index=False)
    
    embed_plus_type_val_results.to_csv(os.path.join(line_path, 'coxPH_decayed_embeddings_plus_type_val_metrics.csv'), index=False)
    embed_plus_type_test_results.to_csv(os.path.join(line_path, 'coxPH_decayed_embeddings_plus_type_test_metrics.csv'), index=False)

metric_data = []
for line in range(1,7):
    line_pred_df = full_ttnt_df.loc[(full_ttnt_df['treatment_line'] == line) & 
                                    (full_ttnt_df['time_on_treatment'] > 0), base_vars + ['CANCER_TYPE'] + embed_cols + target_cols].dropna()
    num_on_line = len(line_pred_df)
    num_on_last_line = num_on_line - line_pred_df['event'].sum()
    
    line_path = os.path.join(OUTPUT_PATH, f'line_{line}/')
    
    base_results = pd.read_csv(os.path.join(line_path, 'coxPH_base_model_metrics.csv'))
    embed_type_results = pd.read_csv(os.path.join(line_path, 'coxPH_decayed_embeddings_plus_type_test_metrics.csv'))
    
    base_entry = base_results.loc[base_results['eval_data'] == 'test_data'].iloc[0]['mean_auc(t)']
    embed_type_entry = embed_type_results.sort_values(by='mean_auc(t)', ascending=False).iloc[0]['mean_auc(t)']
    
    metric_data.append([line, num_on_line, num_on_last_line, base_entry, embed_type_entry])
    
metric_df = pd.DataFrame(metric_data, columns=['treatment_line', 'num_px', 'num_px_on_last_line', 'base_model_mean_auc(t)', 'embed_model_mean_auc(t)'])

metric_df['text_model_auc_improvement'] = metric_df['embed_model_mean_auc(t)'] - metric_df['base_model_mean_auc(t)']