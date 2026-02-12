import os
import pandas as pd
from datetime import datetime

DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
DIAGNOSTICS_PATH = '/data/gusev/PROFILE/CLINICAL/OncDRS/ALL_2025_03/'
INTAE_DATA_PATH = '/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/'
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/')

ehr_icds = pd.read_csv(os.path.join(DIAGNOSTICS_PATH, 'EHR_DIAGNOSIS.csv'))
vte_data = pd.read_csv(os.path.join(INTAE_DATA_PATH, 'follow_up_vte_df_cohort.csv'))

vte_data_sub = vte_data[['DFCI_MRN', 'AGE_AT_FIRST_TREAT', 'BIOLOGICAL_SEX', 'first_treatment_date', 'death_date', 
                         'last_contact_date', 'tt_death', 'death', 'tt_vte', 'vte']]

mrn_tstart_df = vte_data_sub[['DFCI_MRN', 'first_treatment_date']].drop_duplicates()
mrn_tstart_dict = dict(zip(mrn_tstart_df['DFCI_MRN'], mrn_tstart_df['first_treatment_date'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d'))))

vte_mrns = vte_data_sub['DFCI_MRN'].unique()

# extract icds for the mrns in question
ehr_icd_subset = ehr_icds.loc[ehr_icds['DFCI_MRN'].isin(vte_mrns)][['DFCI_MRN', 'START_DT', 'DIAGNOSIS_ICD10_LIST', 'DIAGNOSIS_ICD10_CD', 
                                                                    'DIAGNOSIS_ICD10_NM', 'DIAGNOSIS_ICD10_CD2', 'DIAGNOSIS_ICD10_NM2', 
                                                                    'DIAGNOSIS_ICD10_CD3', 'DIAGNOSIS_ICD10_NM3', 'DIAGNOSIS_CONVERTED_ICD10_IND']]

# remove entries with no defined start date and with no defined codes
ehr_icd_subset = ehr_icd_subset.loc[(~ehr_icd_subset['START_DT'].isna()) &
                                    (~ehr_icd_subset['DIAGNOSIS_ICD10_LIST'].isna())]

# unpack multiple codes instances
single_diagnoses = ehr_icd_subset.loc[(ehr_icd_subset['DIAGNOSIS_ICD10_CD2'].isna()) &
                                      (ehr_icd_subset['DIAGNOSIS_ICD10_CD3'].isna())]

double_diagnoses = ehr_icd_subset.loc[(~ehr_icd_subset['DIAGNOSIS_ICD10_CD2'].isna()) &
                                      (ehr_icd_subset['DIAGNOSIS_ICD10_CD3'].isna())]

triple_diagnoses = ehr_icd_subset.loc[(~ehr_icd_subset['DIAGNOSIS_ICD10_CD2'].isna()) &
                                      (~ehr_icd_subset['DIAGNOSIS_ICD10_CD3'].isna())]

single_diag_set1 = single_diagnoses[['DFCI_MRN', 'START_DT', 'DIAGNOSIS_ICD10_CD', 'DIAGNOSIS_ICD10_NM']]

double_diag_set1 = double_diagnoses[['DFCI_MRN', 'START_DT', 'DIAGNOSIS_ICD10_CD', 'DIAGNOSIS_ICD10_NM']]
double_diag_set2 = double_diagnoses[['DFCI_MRN', 'START_DT', 'DIAGNOSIS_ICD10_CD2', 'DIAGNOSIS_ICD10_NM2']].rename(columns={'DIAGNOSIS_ICD10_CD2' : 'DIAGNOSIS_ICD10_CD', 'DIAGNOSIS_ICD10_NM2' : 'DIAGNOSIS_ICD10_NM'})

triple_diag_set1 = triple_diagnoses[['DFCI_MRN', 'START_DT', 'DIAGNOSIS_ICD10_CD', 'DIAGNOSIS_ICD10_NM']]
triple_diag_set2 = triple_diagnoses[['DFCI_MRN', 'START_DT', 'DIAGNOSIS_ICD10_CD2', 'DIAGNOSIS_ICD10_NM2']].rename(columns={'DIAGNOSIS_ICD10_CD2' : 'DIAGNOSIS_ICD10_CD', 'DIAGNOSIS_ICD10_NM2' : 'DIAGNOSIS_ICD10_NM'})
triple_diag_set3 = triple_diagnoses[['DFCI_MRN', 'START_DT', 'DIAGNOSIS_ICD10_CD3', 'DIAGNOSIS_ICD10_NM3']].rename(columns={'DIAGNOSIS_ICD10_CD3' : 'DIAGNOSIS_ICD10_CD', 'DIAGNOSIS_ICD10_NM3' : 'DIAGNOSIS_ICD10_NM'})

split_ehr_icd_subset = pd.concat([single_diag_set1, double_diag_set1, double_diag_set2, triple_diag_set1, triple_diag_set2, triple_diag_set3])

# add in treatment start date
split_ehr_icd_subset['FIRST_TREATMENT_START_DT'] = split_ehr_icd_subset['DFCI_MRN'].map(mrn_tstart_dict)
split_ehr_icd_subset['TIME_TO_ICD'] = (split_ehr_icd_subset['START_DT'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S')) - split_ehr_icd_subset['FIRST_TREATMENT_START_DT']).apply(lambda x : x.days)

split_ehr_icd_subset.to_csv(os.path.join(SURV_PATH, 'time-to-icd/IO_post_treatment_icd_info.csv'), index=False)