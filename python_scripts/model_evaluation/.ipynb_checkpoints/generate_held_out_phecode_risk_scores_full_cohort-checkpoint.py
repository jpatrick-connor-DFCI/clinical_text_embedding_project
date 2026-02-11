import os
import pandas as pd
from tqdm import tqdm
from embed_surv_utils import get_heldout_risk_scores_CoxPH

os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/clinical_text_project/data/'
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/')
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/VTE_data/processed_datasets/')
STAGE_PATH = '/data/gusev/PROFILE/CLINICAL/OncDRS/DERIVED_FROM_CLINICAL_TEXTS_2024_03/derived_files/cancer_stage/'
METRICS_PATH = os.path.join(RESULTS_PATH, 'phecode_model_comps_full_cohort/')
OUTPUT_PATH = os.path.join(RESULTS_PATH, 'phecode_held_out_risk_scores/full_cohort/')
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

events = os.listdir(METRICS_PATH)
for event in tqdm(events):
    EVENT_PATH = os.path.join(OUTPUT_PATH, event)
    os.makedirs(EVENT_PATH, exist_ok=True)
    
    event_val_metrics = pd.read_csv(os.path.join(METRICS_PATH, event, 'coxPH_decayed_embeddings_plus_type_val_metrics.csv'))
    l1_ratio, alpha = event_val_metrics.sort_values(by='mean_auc(t)', ascending=False).iloc[0][['l1_ratio', 'alpha']]
    
    base_risk_predictions = get_heldout_risk_scores_CoxPH(embeddings_pred_df, base_vars + cancer_type_cols, ['AGE_AT_TREATMENTSTART'],
                                                          [], event_col=event, tstop_col=f'tt_{event}', penalized=False).rename(columns={'risk_score' : 'base_risk_score'})
    text_risk_predictions = get_heldout_risk_scores_CoxPH(embeddings_pred_df, base_vars + cancer_type_cols, continuous_vars,
                                                          embed_cols, event_col=event, tstop_col=f'tt_{event}', penalized=True,
                                                          l1_ratio=l1_ratio, alpha=alpha).rename(columns={'risk_score' : 'text_risk_score'})

    complete_risk_predictions = base_risk_predictions.merge(text_risk_predictions, on='DFCI_MRN')
    complete_risk_predictions.to_csv(os.path.join(EVENT_PATH, 'held_out_risk_predictions.csv'), index=False)