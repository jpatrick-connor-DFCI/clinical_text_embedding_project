import os
import pickle
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from statsmodels.stats.multitest import multipletests
from embed_surv_utils import run_grid_CoxPH_parallel, get_heldout_risk_scores_CoxPH, generate_survival_embedding_df

random.seed(42)  # set seed for reproducibility

# Paths
IO_PATH = '/data/gusev/USERS/mjsaleh/'
FIGURE_PATH = '/data/gusev/USERS/jpconnor/figures/clinical_text_embedding_project/model_metrics/'
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/')
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/VTE_data/processed_datasets/')
STAGE_PATH = '/data/gusev/PROFILE/CLINICAL/OncDRS/DERIVED_FROM_CLINICAL_TEXTS_2024_03/derived_files/cancer_stage/'
OUTPUT_PATH = os.path.join(RESULTS_PATH, 'phecode_model_comps_final')
os.makedirs(OUTPUT_PATH, exist_ok=True)

os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

# Load datasets
cancer_type_df = pd.read_csv('/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/first_treatments_dfci_w_inferred_cancers.csv',
                             usecols=['DFCI_MRN', 'med_genomics_merged_cancer_group']).rename(columns={'med_genomics_merged_cancer_group' : 'CANCER_TYPE'})

notes_meta = pd.read_csv(os.path.join(NOTES_PATH, 'full_VTE_embeddings_metadata.csv'))
embeddings = np.load(os.path.join(NOTES_PATH, 'full_VTE_embeddings_as_array.npy'))

tt_phecodes_df = pd.read_csv(os.path.join(SURV_PATH, 'phecode_surv_df.csv'))
irAE_df = pd.read_csv(os.path.join(IO_PATH, 'IO_START.csv'), index_col=0).rename(columns={'MRN' : 'DFCI_MRN'})

vte_data = pd.read_csv("/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/follow_up_vte_df_cohort.csv")
vte_data_sub = vte_data[
    ["DFCI_MRN", "AGE_AT_FIRST_TREAT", "BIOLOGICAL_SEX", "first_treatment_date", 
     "death_date", "last_contact_date", "tt_death", "death", "tt_vte", "vte"]
].copy()
vte_data_sub["last_contact_date"] = pd.to_datetime(vte_data_sub["last_contact_date"])
tt_phecodes_df['last_contact_date'] = tt_phecodes_df['DFCI_MRN'].map(dict(zip(vte_data_sub['DFCI_MRN'], vte_data_sub['last_contact_date'])))

irAE_df = irAE_df.merge(tt_phecodes_df[['DFCI_MRN', 'death', 'GENDER', 'AGE_AT_TREATMENTSTART', 'last_contact_date']], on='DFCI_MRN')
IO_mrns = irAE_df['DFCI_MRN'].unique()

irAE_df['tt_death'] = (irAE_df['last_contact_date'] - pd.to_datetime(irAE_df['IO_START'])).dt.days

IO_mrns = irAE_df['DFCI_MRN'].unique()
tt_phecodes_df_wo_IO_mrns = tt_phecodes_df.loc[~tt_phecodes_df['DFCI_MRN'].isin(IO_mrns)]

tstart_dict_w_IOs = dict(zip(irAE_df['DFCI_MRN'], irAE_df['IO_START'])) | \
                    dict(zip(tt_phecodes_df_wo_IO_mrns['DFCI_MRN'], tt_phecodes_df_wo_IO_mrns['first_treatment_date']))

notes_meta['IO_ANALYSIS_START_DT'] = notes_meta['DFCI_MRN'].map(tstart_dict_w_IOs)
notes_meta['NOTE_TIME_REL_IO_ANALYSIS_START_DT'] = (pd.to_datetime(notes_meta['NOTE_DATETIME']) - pd.to_datetime(notes_meta['IO_ANALYSIS_START_DT'])).dt.days

tt_phecodes_df_w_IO_mrns = pd.concat([tt_phecodes_df_wo_IO_mrns[['DFCI_MRN', 'death', 'tt_death']], 
                                      irAE_df[['DFCI_MRN', 'death', 'tt_death']]])

note_types = ['Clinician', 'Imaging', 'Pathology']
IO_prediction_df = pd.get_dummies(generate_survival_embedding_df(notes_meta, tt_phecodes_df_w_IO_mrns, embeddings, note_types=note_types,
                                                                 pool_fx={key : 'time_decay_mean' for key in note_types}, decay_param=0.01,
                                                                 note_timing_col='NOTE_TIME_REL_IO_ANALYSIS_START_DT')
                                  .merge(tt_phecodes_df[['DFCI_MRN', 'GENDER', 'AGE_AT_TREATMENTSTART']], on='DFCI_MRN')
                                  .merge(cancer_type_df, on='DFCI_MRN'), columns=['CANCER_TYPE'], drop_first=True).dropna()

IO_prediction_df.dropna(inplace=True)

base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART'] + [col for col in IO_prediction_df if col.startswith('CANCER_TYPE')]
embed_cols = [c for c in IO_prediction_df.columns if 'EMBEDDING' in c or '2015' in c]
continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols

IO_mrns = list(set(irAE_df['DFCI_MRN'].unique()).intersection(set(IO_prediction_df['DFCI_MRN'].unique())))
eval_mrns = random.sample(IO_mrns, int(len(IO_mrns) // 2))

IO_train_mrns = list(set(IO_mrns) - set(eval_mrns))

held_out_pred_df = IO_prediction_df.loc[IO_prediction_df['DFCI_MRN'].isin(eval_mrns)]

pan_treatment_pred_df = IO_prediction_df.loc[~IO_prediction_df['DFCI_MRN'].isin(eval_mrns)]
just_IO_pred_df = IO_prediction_df.loc[IO_prediction_df['DFCI_MRN'].isin(IO_train_mrns)]

## Train the pan cancer model
event='death'
alphas_to_test = np.logspace(-5, 0, 30)
l1_ratios = [0.5, 1.0]

_, pan_treatment_val_results, pan_treatment_model = run_grid_CoxPH_parallel(
    pan_treatment_pred_df, base_vars, continuous_vars, embed_cols, l1_ratios, 
    alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=5000, verbose=5)

pan_treatment_l1_ratio, pan_treatment_alpha = pan_treatment_val_results.sort_values(by='mean_auc(t)', ascending=False).iloc[0][['l1_ratio', 'alpha']]

trained_pan_treatment = (get_heldout_risk_scores_CoxPH(pan_treatment_pred_df, base_vars, continuous_vars, embed_cols,
                                                       event_col=event, tstop_col=f'tt_{event}', penalized=True, max_iter=5000,
                                                       l1_ratio=pan_treatment_l1_ratio, alpha=pan_treatment_alpha)
                         .rename(columns={'risk_score' : 'pan_treatment_risk_score'}))

_, IO_val_results, IO_model = run_grid_CoxPH_parallel(
    just_IO_pred_df, base_vars, continuous_vars, embed_cols, 
    l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=5000, verbose=5)

IO_l1_ratio, IO_alpha = IO_val_results.sort_values(by='mean_auc(t)', ascending=False).iloc[0][['l1_ratio', 'alpha']]

trained_IO = (get_heldout_risk_scores_CoxPH(just_IO_pred_df, base_vars, continuous_vars, embed_cols,
                                            event_col=event, tstop_col=f'tt_{event}', penalized=True,
                                            max_iter=5000, l1_ratio=IO_l1_ratio, alpha=IO_alpha)
              .rename(columns={'risk_score' : 'IO_risk_score'}))

complete_risk_scores = (trained_IO
                        .merge(trained_pan_treatment, on='DFCI_MRN')
                        .merge(just_IO_pred_df[['DFCI_MRN', f'tt_{event}', event]], on='DFCI_MRN'))
train_times = complete_risk_scores[f'tt_{event}']
train_events = complete_risk_scores[event].astype(bool)

c_index_pan_train = concordance_index_censored(train_events, train_times, complete_risk_scores['pan_treatment_risk_score'])[0]
c_index_within_train = concordance_index_censored(train_events, train_times, complete_risk_scores['IO_risk_score'])[0]

print(f'C-index using pan treatment model on training set = {c_index_pan_train : 0.3f}')
print(f'C-index using within IO model on training set = {c_index_within_train : 0.3f}')

held_out_pred_df[continuous_vars] = StandardScaler().fit_transform(held_out_pred_df[continuous_vars])

IO_scores = IO_model.predict(held_out_pred_df[base_vars + embed_cols])
pan_scores = pan_treatment_model.predict(held_out_pred_df[base_vars + embed_cols])
dfci_mrns = held_out_pred_df['DFCI_MRN'].tolist()

held_out_risk_scores = (pd.DataFrame({'DFCI_MRN' : dfci_mrns, 'IO_risk_score' : IO_scores, 'pan_risk_score' : pan_scores})
                        .merge(held_out_pred_df[['DFCI_MRN', f'tt_{event}', event]], on='DFCI_MRN'))

held_out_times = held_out_risk_scores[f'tt_{event}']
held_out_events = held_out_risk_scores[event].astype(bool)

c_index_pan_held_out = concordance_index_censored(held_out_events, held_out_times, held_out_risk_scores['pan_risk_score'])[0]
c_index_within_held_out = concordance_index_censored(held_out_events, held_out_times, held_out_risk_scores['IO_risk_score'])[0]

print(f'C-index using pan treatment model on held out data = {c_index_pan_held_out : 0.3f}')
print(f'C-index using IO model on held out data = {c_index_within_held_out : 0.3f}')

ICI_data = pd.DataFrame({'data' : ['training', 'held_out'], 'pan_c_index' : [c_index_pan_train, c_index_pan_held_out], 
                         'within_c_index' : [c_index_within_train, c_index_within_held_out]})

ICI_data.to_csv(os.path.join(RESULTS_PATH, 'pan_vs_within_IO_results.csv'), index=False)