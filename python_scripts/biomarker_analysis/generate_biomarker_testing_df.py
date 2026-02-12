import os
import random
import numpy as np
import pandas as pd
from embed_surv_utils import run_grid_CoxPH_parallel, get_heldout_risk_scores_CoxPH, generate_survival_embedding_df

random.seed(42)  # set seed for reproducibility

# Paths
IO_PATH = '/data/gusev/USERS/mjsaleh/'
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/processed_datasets/')
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')

# Load datasets
cancer_type_df = pd.read_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/cancer_type_df.csv'))

notes_meta = pd.read_csv(os.path.join(NOTES_PATH, 'full_VTE_embeddings_metadata.csv'))
embeddings = np.load(os.path.join(NOTES_PATH, 'full_VTE_embeddings_as_array.npy'))

tt_phecodes_df = pd.read_csv(os.path.join(SURV_PATH, 'phecode_surv_df.csv'))
irAE_df = pd.read_csv(os.path.join(IO_PATH, 'IO_START.csv'), index_col=0).rename(columns={'MRN' : 'DFCI_MRN'})

vte_data = pd.read_csv("/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/follow_up_vte_df_cohort.csv")
vte_data_sub = vte_data[["DFCI_MRN", "AGE_AT_FIRST_TREAT", "BIOLOGICAL_SEX", "first_treatment_date", "death_date", "last_contact_date", "tt_death", "death", "tt_vte", "vte"]].copy()

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
IO_prediction_df = (generate_survival_embedding_df(notes_meta, tt_phecodes_df_w_IO_mrns, embeddings, note_types=note_types,
                                                   pool_fx={key : 'time_decay_mean' for key in note_types}, decay_param=0.01,
                                                   note_timing_col='NOTE_TIME_REL_IO_ANALYSIS_START_DT')
                    .merge(tt_phecodes_df[['DFCI_MRN', 'GENDER', 'AGE_AT_TREATMENTSTART']], on='DFCI_MRN')
                    .merge(cancer_type_df, on='DFCI_MRN')).dropna()

base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART'] + [col for col in IO_prediction_df if col.startswith('CANCER_TYPE')]
embed_cols = [c for c in IO_prediction_df.columns if 'EMBEDDING' in c or '2015' in c]
continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols

IO_mrns = list(set(irAE_df['DFCI_MRN'].unique()).intersection(set(IO_prediction_df['DFCI_MRN'].unique())))

event='death'
alphas_to_test = np.logspace(-5, 0, 25)
l1_ratios = [0.5, 1.0]


_, IO_val_results, _ = run_grid_CoxPH_parallel(
    IO_prediction_df, base_vars, continuous_vars, embed_cols,
    l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=3000, verbose=5)

IO_l1_ratio, IO_alpha = IO_val_results.sort_values(by='mean_auc(t)', ascending=False).iloc[0][['l1_ratio', 'alpha']]

trained_IO = (get_heldout_risk_scores_CoxPH(IO_prediction_df, base_vars, continuous_vars, embed_cols,
                                            event_col=event, tstop_col=f'tt_{event}', penalized=True,
                                            l1_ratio=IO_l1_ratio, alpha=IO_alpha, max_iter=3000)
              .rename(columns={'risk_score' : 'IO_risk_score'}))

somatic_df = pd.read_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/complete_somatic_data_df.csv'))

biomarker_df = (irAE_df[['DFCI_MRN', 'tt_death', 'death', 'GENDER', 'AGE_AT_TREATMENTSTART']]
                .merge(cancer_type_df, on='DFCI_MRN')
                .merge(somatic_df, on='DFCI_MRN')
                .merge(trained_IO, on='DFCI_MRN'))

biomarker_df.to_csv(os.path.join(MARKER_PATH, 'IO_biomarker_discovery.csv'), index=False)
