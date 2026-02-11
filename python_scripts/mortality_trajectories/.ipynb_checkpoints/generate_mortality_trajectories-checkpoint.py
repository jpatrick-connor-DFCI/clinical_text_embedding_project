import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from embed_surv_utils import run_grid_CoxPH_parallel, get_heldout_risk_scores_CoxPH, generate_survival_embedding_df

# Paths
PROJ_PATH = '/data/gusev/USERS/jpconnor/clinical_text_project/'
FIGURE_PATH = os.path.join(PROJ_PATH, 'figures/model_metrics/')
DATA_PATH = os.path.join(PROJ_PATH, 'data/')
FEATURE_PATH = os.path.join(DATA_PATH, 'clinical_and_genomic_features/')
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/icd_results/')
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/processed_datasets/')
TRAJECTORY_PATH = os.path.join(RESULTS_PATH, 'mortality_trajectories/')
os.makedirs(TRAJECTORY_PATH, exist_ok=True)

os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

# Load datasets
cancer_type_df = pd.read_csv(os.path.join(FEATURE_PATH, 'cancer_type_df.csv'))
time_decayed_events_df = pd.read_csv(os.path.join(SURV_PATH, 'time-to-icd/time_decayed_events_df.csv'))
full_prediction_df = time_decayed_events_df.merge(cancer_type_df, on='DFCI_MRN').dropna()

# Define model columns
base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
events = [col.split('_', 1)[1] for col in time_decayed_events_df.columns if col.startswith('tt')]
tt_events = [f"tt_{event}" for event in events]

# Column groups
embed_cols = [c for c in full_prediction_df.columns if 'EMBEDDING' in c or '2015' in c]
continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols
type_cols = [c for c in full_prediction_df.columns if 'CANCER_TYPE' in c]

## Train the baseline cancer model
event='death'
alphas_to_test = np.logspace(-5, 0, 30)
l1_ratios = [0.5, 1.0]

_, embed_val_results, pan_cancer_model = run_grid_CoxPH_parallel(
    full_prediction_df, base_vars + type_cols, continuous_vars, embed_cols,
    l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}',
    max_iter=1000, verbose=5)

opt_l1_ratio, opt_alpha = embed_val_results.sort_values(by='mean_auc(t)', ascending=False).iloc[0][['l1_ratio', 'alpha']]

## Generate monthly data frames
notes_meta = pd.read_csv(NOTES_PATH + 'full_VTE_embeddings_metadata.csv')
embeddings_data = np.load(open(NOTES_PATH + 'full_VTE_embeddings_as_array.npy', 'rb'))
events_data = pd.read_csv(SURV_PATH + 'time-to-icd/tt_vte_plus_icd_level_3s.csv')

note_types = ['Clinician', 'Imaging', 'Pathology']
months_to_test = [i*3 for i in range(1,21)]

cohort_mrns = full_prediction_df['DFCI_MRN'].unique().tolist()
trajectory_predictions_df = pd.DataFrame({'DFCI_MRN' : cohort_mrns} | {f'plus_{month_adj}_months_data' : [np.nan for _ in range(len(cohort_mrns))] for month_adj in months_to_test})

risk_scores = get_heldout_risk_scores_CoxPH(full_prediction_df, base_vars + type_cols, continuous_vars, embed_cols,
                                            event_col=event, tstop_col=f'tt_{event}', id_col='DFCI_MRN', penalized=True, 
                                            l1_ratio=opt_l1_ratio, alpha=opt_alpha, max_iter=1000, verbose=5)

trajectory_predictions_df['plus_0_months_data'] = (trajectory_predictions_df['DFCI_MRN']
                                                   .map(dict(zip(risk_scores['DFCI_MRN'], risk_scores['risk_score'])) |
                                                                 {mrn : np.nan for mrn in cohort_mrns 
                                                                  if mrn not in risk_scores['DFCI_MRN'].unique()}))

prev_mrns = cohort_mrns
for month_adj in tqdm(months_to_test):
    notes_meta_copy = notes_meta.loc[notes_meta['DFCI_MRN'].isin(prev_mrns)].copy()
    events_data_copy = events_data.loc[events_data['DFCI_MRN'].isin(prev_mrns)].copy()
    monthly_data = (generate_survival_embedding_df(notes_meta_copy, events_data_copy, embeddings_data, note_types=note_types,
                                                  pool_fx={key : 'time_decay_mean' for key in note_types}, decay_param=0.01,
                                                  max_note_window=month_adj*30)[['DFCI_MRN', event, f'tt_{event}'] + base_vars + embed_cols]
                    .merge(cancer_type_df, on='DFCI_MRN').dropna())
    monthly_data = monthly_data.loc[monthly_data[f'tt_{event}'] > 0]
    
    try:
        risk_scores = get_heldout_risk_scores_CoxPH(monthly_data, base_vars + type_cols, continuous_vars, embed_cols,
                                                    event_col=event, tstop_col=f'tt_{event}', id_col='DFCI_MRN', penalized=True, 
                                                    l1_ratio=opt_l1_ratio, alpha=opt_alpha, max_iter=1000, verbose=0)
        trajectory_predictions_df[f'plus_{month_adj}_months_data'] = (trajectory_predictions_df['DFCI_MRN']
                                                                      .map(dict(zip(risk_scores['DFCI_MRN'], risk_scores['risk_score'])) | 
                                                                           {mrn : np.nan for mrn in cohort_mrns if mrn not in risk_scores['DFCI_MRN'].unique()}))
        prev_mrns = risk_scores['DFCI_MRN'].unique()
    
    except:
        continue
    
trajectory_predictions_df.to_csv(os.path.join(TRAJECTORY_PATH, 'survival_trajectories.csv'), index=False)