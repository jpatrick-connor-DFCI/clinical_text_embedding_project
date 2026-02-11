import os
import re
import icd10
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from embed_surv_utils import map_time_to_event

import pandas as pd

# your table as a DataFrame
chapters = pd.DataFrame(
    [
        ("I",   "A00-B99", "Certain infectious and parasitic diseases"),
        ("II",  "C00-D48", "Neoplasms"),
        ("III", "D50-D89", "Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism"),
        ("IV",  "E00-E90", "Endocrine, nutritional and metabolic diseases"),
        ("V",   "F00-F99", "Mental and behavioural disorders"),
        ("VI",  "G00-G99", "Diseases of the nervous system"),
        ("VII", "H00-H59", "Diseases of the eye and adnexa"),
        ("VIII","H60-H95", "Diseases of the ear and mastoid process"),
        ("IX",  "I00-I99", "Diseases of the circulatory system"),
        ("X",   "J00-J99", "Diseases of the respiratory system"),
        ("XI",  "K00-K93", "Diseases of the digestive system"),
        ("XII", "L00-L99", "Diseases of the skin and subcutaneous tissue"),
        ("XIII","M00-M99", "Diseases of the musculoskeletal system and connective tissue"),
        ("XIV", "N00-N99", "Diseases of the genitourinary system"),
        ("XV",  "O00-O99", "Pregnancy, childbirth and the puerperium"),
        ("XVI", "P00-P96", "Certain conditions originating in the perinatal period"),
        ("XVII","Q00-Q99", "Congenital malformations, deformations and chromosomal abnormalities"),
        ("XVIII","R00-R99","Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified"),
        ("XIX", "S00-T98", "Injury, poisoning and certain other consequences of external causes"),
        ("XX",  "V01-Y98", "External causes of morbidity and mortality"),
        ("XXI", "Z00-Z99", "Factors influencing health status and contact with health services"),
        ("XXII","U00-U99", "Codes for special purposes"),
    ],
    columns=["Chapter", "Block", "Title"],
)

def _norm_icd10(code: str) -> str:
    """Normalize to undotted, uppercase; keep only first 3 chars for chapter/block matching."""
    if code is None:
        return ""
    code = str(code).strip().upper().replace(".", "")
    return code[:3]  # e.g., "C50.9" -> "C50", "A041" -> "A04"

def add_icd10_chapter_block(df: pd.DataFrame, code_col: str, chapters: pd.DataFrame = chapters) -> pd.DataFrame:
    """
    Adds Chapter, Block (range), and Title using the provided ICD-10 chapter/block range table.
    Works for codes like 'C50.9', 'C509', 'A04.1', etc. using the first 3 characters.
    """
    out = df.copy()
    key = out[code_col].map(_norm_icd10)

    # parse block ranges into start/end like "A00", "B99"
    ranges = chapters.copy()
    ranges[["start", "end"]] = ranges["Block"].str.split("-", expand=True)
    ranges["start"] = ranges["start"].str.upper()
    ranges["end"] = ranges["end"].str.upper()

    # vectorized-ish match (22 ranges, so per-row scan is plenty fast)
    chap, block, title = [], [], []
    for k in key:
        hit = ranges[(ranges["start"] <= k) & (ranges["end"] >= k)]
        if hit.empty:
            chap.append(None); block.append(None); title.append(None)
        else:
            r = hit.iloc[0]
            chap.append(r["Chapter"]); block.append(r["Block"]); title.append(r["Title"])

    out["Chapter"] = chap
    out["Block"] = block
    out["Title"] = title
    return out

# ---- example ----
# df = pd.DataFrame({"icd10": ["A04.1", "C50.9", "D64", "H90.3", "U07.1", "Y93.9"]})
# df2 = add_icd10_chapter_block(df, "icd10")
# print(df2)

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/clinical_text_project/data/'
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
INTAE_DATA_PATH = '/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/'
METS_PROJECT = '/data/gusev/Recurrent_Mets_Project/'
PROCESSED_DATA_PATH = os.path.join(METS_PROJECT, 'clinical_to_ag/')

# Load data
vte_data = pd.read_csv(os.path.join(INTAE_DATA_PATH, 'follow_up_vte_df_cohort.csv'))
vte_data_sub = vte_data[['DFCI_MRN', 'AGE_AT_FIRST_TREAT', 'BIOLOGICAL_SEX', 'first_treatment_date', 
                         'death_date', 'last_contact_date', 'tt_death', 'death', 'tt_vte', 'vte']].copy()

# First treatment date dict
mrn_tstart_dict = dict(zip(vte_data_sub['DFCI_MRN'], vte_data_sub['first_treatment_date'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d'))))

# EHR ICD info
split_ehr_icd_subset = pd.read_csv(os.path.join(SURV_PATH, 'time-to-icd/IO_post_treatment_icd_info.csv'))
split_ehr_icd_subset['ICD10_LEVEL_3_CD'] = split_ehr_icd_subset['DIAGNOSIS_ICD10_CD'].apply(lambda x : x.split('.')[0])
icd_descr_lookup = {key : icd10.find(key).description for key in split_ehr_icd_subset['ICD10_LEVEL_3_CD'].unique() if icd10.find(key) is not None}
split_ehr_icd_subset['ICD10_LEVEL_3_NM'] = split_ehr_icd_subset['ICD10_LEVEL_3_CD'].map(icd_descr_lookup)

common_icds = split_ehr_icd_subset[['ICD10_LEVEL_3_CD', 'ICD10_LEVEL_3_NM']].value_counts().reset_index()

common_icds_w_descr = add_icd10_chapter_block(common_icds, 'ICD10_LEVEL_3_CD')
common_icds_to_select = common_icds_w_descr.loc[~common_icds_w_descr['Chapter'].isin(['II',   # Neoplasms
                                                                                      'XV',   # Pregnancy
                                                                                      'XVI',  # Conditions in the perinatal period
                                                                                      'XVII', # Congenital malformations
                                                                                      'XIX',  # Inury/poisoning
                                                                                      'XX',   # External causes of morbidity and mortality
                                                                                      'XXI',  # Factors influencing health status
                                                                                      'XXII', # Codes for special purposes
                                                                                     ])]

split_ehr_icd_subset = split_ehr_icd_subset.loc[(split_ehr_icd_subset['TIME_TO_ICD'] > 0) & 
                                                (split_ehr_icd_subset['ICD10_LEVEL_3_CD'].isin(common_icds_to_select['ICD10_LEVEL_3_CD'].unique()))].copy()

# Generate time-to-event for icds
icds_to_analyze = split_ehr_icd_subset['ICD10_LEVEL_3_CD'].unique()

for icd in tqdm(icds_to_analyze, desc="Generating icd events"):
    icd_data_sub = split_ehr_icd_subset[split_ehr_icd_subset['ICD10_LEVEL_3_CD'] == icd]
    vte_data_sub['tt_' + str(icd)], vte_data_sub[str(icd)] = map_time_to_event(icd_data_sub, vte_data_sub, 'DFCI_MRN', str(icd), 'TIME_TO_ICD')

# Filter icds with ≥5% prevalence
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
events_to_analyze = [col[3:] for col in vte_data_sub.columns if col.startswith('tt_')]

events = []; num_prior_events = []; num_post_events = []; mean_tte = []; num_at_risk_pxs = [];
for event in events_to_analyze:
    event_data = vte_data_sub[[f'tt_{event}', event]]
    prior_event_data = event_data.loc[event_data[f'tt_{event}'] <= 0]
    post_event_data = event_data.loc[(event_data[event]) & (event_data[f'tt_{event}'] > 0)]
    
    num_prior_events.append(len(prior_event_data))
    num_post_events.append(len(post_event_data))
    mean_tte.append(post_event_data[f'tt_{event}'].mean())
    num_at_risk_pxs.append(len(event_data) - len(prior_event_data))

event_freq_df = pd.DataFrame({'event' : events_to_analyze, 'num_prior_events' : num_prior_events,
                              'num_post_events' : num_post_events, 'mean_tte' : mean_tte, 'num_at_risk_pxs' : num_at_risk_pxs})
event_freq_df['event_freq'] = (event_freq_df['num_post_events'] / event_freq_df['num_at_risk_pxs'])
events_to_include = event_freq_df.loc[event_freq_df['event_freq'] >= 0.01, 'event'].tolist()

base_cols = ['DFCI_MRN', 'first_treatment_date', 'AGE_AT_TREATMENTSTART', 'GENDER']
(vte_data_sub[base_cols + [event for event in events_to_include] + [f'tt_{event}' for event in events_to_include]]
    .to_csv(os.path.join(SURV_PATH, 'time-to-icd/tt_vte_plus_icd_level_3s.csv'), index=False))