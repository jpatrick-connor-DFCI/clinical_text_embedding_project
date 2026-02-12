import os
import re
import numpy as np
import pandas as pd
from typing import Optional
from tqdm import tqdm
from embed_surv_utils import map_time_to_event, generate_survival_embedding_df

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
CODE_PATH = os.path.join(DATA_PATH, 'code_data/')
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/processed_datasets/')
INTAE_DATA_PATH = '/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/'
METS_PROJECT = '/data/gusev/Recurrent_Mets_Project/'
PROCESSED_DATA_PATH = os.path.join(METS_PROJECT, 'clinical_to_ag/')


def _normalize_icd10_undotted(code: str) -> Optional[str]:
    if pd.isna(code):
        return None
    code = str(code).strip().upper()
    code = re.sub(r"[^A-Z0-9.]", "", code)
    code = code.replace(".", "")
    return code if code else None


def _to_icd10_level_3(code: str) -> Optional[str]:
    code = _normalize_icd10_undotted(code)
    if code is None or len(code) < 3:
        return None
    return code[:3]


# Load endpoints to model
endpoint_file = os.path.join(CODE_PATH, 'cancer_endpoints_icd10_level3.csv')
endpoint_df = pd.read_csv(endpoint_file)
if 'icd_code' not in endpoint_df.columns:
    raise ValueError(f"Expected column 'icd_code' in {endpoint_file}. Found: {list(endpoint_df.columns)}")

icds_to_analyze = (
    endpoint_df['icd_code']
    .map(_to_icd10_level_3)
    .dropna()
    .drop_duplicates()
    .tolist()
)
if len(icds_to_analyze) == 0:
    raise ValueError(f"No usable level-3 ICD codes found in {endpoint_file}.")

# Load cohort data
vte_data = pd.read_csv(os.path.join(INTAE_DATA_PATH, 'follow_up_vte_df_cohort.csv'))
vte_data_sub = vte_data[
    ['DFCI_MRN', 'AGE_AT_FIRST_TREAT', 'BIOLOGICAL_SEX', 'first_treatment_date',
     'death_date', 'last_contact_date', 'tt_death', 'death', 'tt_vte', 'vte']
].copy()

# Load and filter EHR ICD records
split_ehr_icd_subset = pd.read_csv(os.path.join(SURV_PATH, 'timestamped_icd_info.csv'))
split_ehr_icd_subset['ICD10_LEVEL_3_CD'] = split_ehr_icd_subset['DIAGNOSIS_ICD10_CD'].map(_to_icd10_level_3)
split_ehr_icd_subset = split_ehr_icd_subset.loc[
    split_ehr_icd_subset['ICD10_LEVEL_3_CD'].isin(set(icds_to_analyze))
].copy()
split_ehr_icd_subset['START_DT'] = pd.to_datetime(split_ehr_icd_subset['START_DT'], errors='coerce')
split_ehr_icd_subset = (
    split_ehr_icd_subset
    .sort_values(['DFCI_MRN', 'ICD10_LEVEL_3_CD', 'START_DT'])
    .drop_duplicates(subset=['DFCI_MRN', 'ICD10_LEVEL_3_CD'], keep='first')
)

# Generate time-to-event features for only requested ICD endpoints
for icd in tqdm(icds_to_analyze, desc="Generating level-3 ICD events"):
    icd_data_sub = split_ehr_icd_subset.loc[split_ehr_icd_subset['ICD10_LEVEL_3_CD'] == icd]
    vte_data_sub[f'tt_{icd}'], vte_data_sub[icd] = map_time_to_event(
        icd_data_sub, vte_data_sub, 'DFCI_MRN', icd, 'TIME_TO_ICD'
    )

# Add metastatic disease events
met_sites = ['brain', 'bone', 'adrenal', 'liver', 'lung', 'node', 'peritoneal']
dfs_to_concat = [
    pd.read_csv(os.path.join(PROCESSED_DATA_PATH, f'clinical_to_{site}_met.csv'))
    .loc[lambda df: df['event'] == 1, ['dfci_mrn', 'date', 'type']]
    for site in met_sites
]
met_date_df = pd.concat(dfs_to_concat, ignore_index=True)
met_date_df.rename(columns={'dfci_mrn': 'DFCI_MRN', 'date': 'MET_DATE', 'type': 'MET_LOCATION'}, inplace=True)

met_date_df = met_date_df.loc[met_date_df['DFCI_MRN'].isin(vte_data_sub['DFCI_MRN'])].copy()
mrn_tstart_dict = dict(zip(vte_data_sub['DFCI_MRN'], pd.to_datetime(vte_data_sub['first_treatment_date'], errors='coerce')))
met_date_df['first_treatment_date'] = met_date_df['DFCI_MRN'].map(mrn_tstart_dict)
met_date_df['MET_DATE'] = pd.to_datetime(met_date_df['MET_DATE'].astype(str).str.split(' ').str[0], errors='coerce')
met_date_df['TIME_TO_MET'] = (met_date_df['MET_DATE'] - met_date_df['first_treatment_date']).dt.days
met_date_df = met_date_df.dropna(subset=['TIME_TO_MET'])

met_events_added = []
for met_loc in sorted(met_date_df['MET_LOCATION'].dropna().unique()):
    cur_met_data_sub = met_date_df.loc[met_date_df['MET_LOCATION'] == met_loc]
    vte_data_sub[f'tt_{met_loc}'], vte_data_sub[met_loc] = map_time_to_event(
        cur_met_data_sub, vte_data_sub, 'DFCI_MRN', met_loc, 'TIME_TO_MET'
    )
    met_events_added.append(str(met_loc))

vte_data_sub['AGE_AT_TREATMENTSTART'] = vte_data_sub['AGE_AT_FIRST_TREAT']
vte_data_sub['GENDER'] = vte_data_sub['BIOLOGICAL_SEX'].map({'MALE': 0, 'FEMALE': 1})
vte_data_sub.drop(columns=['AGE_AT_FIRST_TREAT', 'BIOLOGICAL_SEX', 'death_date', 'last_contact_date'], inplace=True)

# Save final survival dataset
base_cols = ['DFCI_MRN', 'first_treatment_date', 'AGE_AT_TREATMENTSTART', 'GENDER']
event_cols = [event for event in icds_to_analyze if event in vte_data_sub.columns]
event_cols += [event for event in met_events_added if event in vte_data_sub.columns and event not in event_cols]
tt_event_cols = [f'tt_{event}' for event in event_cols]
events_data_sub = vte_data_sub[base_cols + event_cols + tt_event_cols]
events_data_sub.to_csv(os.path.join(SURV_PATH, 'level_3_ICD_surv_df.csv'), index=False)

# Create embedding prediction dataset
embeddings_data = np.load(os.path.join(NOTES_PATH, 'full_VTE_embeddings_as_array.npy'))
notes_meta = pd.read_csv(os.path.join(NOTES_PATH, 'full_VTE_embeddings_metadata.csv'))

note_types = ['Clinician', 'Imaging', 'Pathology']
monthly_data = generate_survival_embedding_df(
    notes_meta,
    events_data_sub,
    embeddings_data,
    note_types=note_types,
    note_timing_col='NOTE_TIME_REL_FIRST_TREATMENT_START',
    continuous_window=False,
    pool_fx={key: 'time_decay_mean' for key in note_types},
    decay_param=0.01,
).dropna()
monthly_data.to_csv(os.path.join(SURV_PATH, 'level_3_ICD_embedding_prediction_df.csv'), index=False)
