import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from embed_surv_utils import map_time_to_event, generate_survival_embedding_df

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/processed_datasets/')
INTAE_DATA_PATH = '/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/'
METS_PROJECT = '/data/gusev/Recurrent_Mets_Project/'
PROCESSED_DATA_PATH = os.path.join(METS_PROJECT, 'clinical_to_ag/')

# Load data
vte_data = pd.read_csv(os.path.join(INTAE_DATA_PATH, 'follow_up_vte_df_cohort.csv'))
vte_data_sub = vte_data[['DFCI_MRN', 'AGE_AT_FIRST_TREAT', 'BIOLOGICAL_SEX', 'first_treatment_date', 
                         'death_date', 'last_contact_date', 'tt_death', 'death', 'tt_vte', 'vte']].copy()

# First treatment date dict
mrn_tstart_dict = dict(zip(vte_data_sub['DFCI_MRN'], vte_data_sub['first_treatment_date'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d'))))

# ICD → phecode mapping
icd_to_phecode_map = pd.read_csv(os.path.join(DATA_PATH, 'code_data/icd_to_phecode_map.csv'))

# EHR ICD info
split_ehr_icd_subset = pd.read_csv(os.path.join(SURV_PATH, 'time-to-icd/timestamped_icd_info.csv'))
split_ehr_icd_subset = split_ehr_icd_subset.loc[
    split_ehr_icd_subset['DIAGNOSIS_ICD10_CD'].isin(icd_to_phecode_map['ICD10_CD'].unique())
].copy()
split_ehr_icd_subset['PHECODE'] = split_ehr_icd_subset['DIAGNOSIS_ICD10_CD'].map(dict(zip(icd_to_phecode_map['ICD10_CD'], icd_to_phecode_map['PHECODE'])))
split_ehr_icd_subset['START_DT'] = pd.to_datetime(split_ehr_icd_subset['START_DT'], errors='coerce')
split_ehr_icd_subset = (split_ehr_icd_subset
                        .sort_values(['DFCI_MRN', 'PHECODE', 'START_DT'])
                        .drop_duplicates(subset=['DFCI_MRN', 'PHECODE'], keep='first'))

# Generate time-to-event for phecodes
phecodes_to_analyze = split_ehr_icd_subset['PHECODE'].unique()

for phecode in tqdm(phecodes_to_analyze, desc="Generating phecode events"):
    phecode_data_sub = split_ehr_icd_subset[split_ehr_icd_subset['PHECODE'] == phecode]
    vte_data_sub['tt_' + str(phecode)], vte_data_sub[str(phecode)] = map_time_to_event(phecode_data_sub, vte_data_sub, 'DFCI_MRN', str(phecode), 'TIME_TO_ICD')

# Filter phecodes with ≥5% prevalence
events = [re.split('_', col)[1] for col in vte_data_sub.columns if col.startswith('tt_')]
events_ct_df = pd.DataFrame({'event': events, 
                             'num_px': [vte_data_sub[event].dropna().shape[0] for event in events], 
                             'num_events': [vte_data_sub[event].sum() for event in events]})
events_ct_df['event_prevalence'] = events_ct_df['num_events'] / events_ct_df['num_px']
# events_ct_df = events_ct_df.loc[events_ct_df['event_prevalence'] >= 0.05]

base_cols = ['DFCI_MRN', 'AGE_AT_FIRST_TREAT', 'BIOLOGICAL_SEX', 'first_treatment_date']
event_cols = events_ct_df['event'].tolist()
tt_event_cols = ['tt_' + event for event in event_cols]

# Add metastatic disease events
met_sites = ['brain', 'bone', 'adrenal', 'liver', 'lung', 'node', 'peritoneal']

dfs_to_concat = [pd.read_csv(os.path.join(PROCESSED_DATA_PATH, f'clinical_to_{site}_met.csv')).loc[lambda df: df['event'] == 1, ['dfci_mrn', 'date', 'type']] for site in met_sites]
met_date_df = pd.concat(dfs_to_concat)
met_date_df.rename(columns={'dfci_mrn': 'DFCI_MRN', 'date': 'MET_DATE', 'type': 'MET_LOCATION'}, inplace=True)

# Keep only MRNs in dataset
met_date_df = met_date_df.loc[met_date_df['DFCI_MRN'].isin(vte_data_sub['DFCI_MRN'])].copy()

# Map first treatment date
met_date_df['first_treatment_date'] = met_date_df['DFCI_MRN'].map(dict(zip(vte_data_sub['DFCI_MRN'], vte_data_sub['first_treatment_date']))).apply(lambda x : datetime.strptime(x, '%Y-%m-%d'))

# Compute TIME_TO_MET
met_date_df['MET_DATE'] = met_date_df['MET_DATE'].apply(lambda x: x.split(' ')[0])
met_date_df['TIME_TO_MET'] = (met_date_df['MET_DATE'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d')) 
                              - met_date_df['first_treatment_date']).dt.days

# Add metastasis events
for met_loc in met_date_df['MET_LOCATION'].unique():
    cur_met_data_sub = met_date_df[met_date_df['MET_LOCATION'] == met_loc]
    vte_data_sub['tt_' + met_loc], vte_data_sub[met_loc] = map_time_to_event(cur_met_data_sub, vte_data_sub, 'DFCI_MRN', met_loc, 'TIME_TO_MET')

vte_data_sub['AGE_AT_TREATMENTSTART'] = vte_data_sub['AGE_AT_FIRST_TREAT']
vte_data_sub['GENDER'] = vte_data_sub['BIOLOGICAL_SEX'].map({'MALE' : 0, 'FEMALE' : 1})
vte_data_sub.drop(columns=['AGE_AT_FIRST_TREAT', 'BIOLOGICAL_SEX', 'death_date', 'last_contact_date'], inplace=True)
    
# Save final dataset
events_data_sub = vte_data_sub
events_data_sub.to_csv(os.path.join(SURV_PATH, 'phecode_surv_df.csv'), index=False)

# Create embedding prediction dataset
embeddings_data = np.load(os.path.join(NOTES_PATH, 'full_VTE_embeddings_as_array.npy'))
notes_meta = pd.read_csv(os.path.join(NOTES_PATH, 'full_VTE_embeddings_metadata.csv'))

note_types = ['Clinician', 'Imaging', 'Pathology']
monthly_data = generate_survival_embedding_df(notes_meta, events_data_sub, embeddings_data, note_types=note_types,
                                              note_timing_col='NOTE_TIME_REL_FIRST_TREATMENT_START', continuous_window=False,
                                              pool_fx={key : 'time_decay_mean' for key in note_types}, decay_param=0.01).dropna()
monthly_data.to_csv(os.path.join(SURV_PATH, 'phecode_embedding_prediction_df.csv'), index=False)
