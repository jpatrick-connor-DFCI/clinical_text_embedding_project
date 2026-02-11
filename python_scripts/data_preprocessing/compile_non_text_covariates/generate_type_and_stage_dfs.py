# === Imports ===
import os
import pickle
import pandas as pd

# === Paths ===
PROJ_PATH = '/data/gusev/USERS/jpconnor/clinical_text_project/'
FIGURE_PATH = os.path.join(PROJ_PATH, 'figures/model_metrics/')
DATA_PATH = os.path.join(PROJ_PATH, 'data/')
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
STAGE_PATH = '/data/gusev/PROFILE/CLINICAL/OncDRS/DERIVED_FROM_CLINICAL_TEXTS_2024_03/derived_files/cancer_stage/'

# Load text data
time_decayed_events_df = pd.read_csv(os.path.join(SURV_PATH, 'time-to-icd/time_decayed_events_df.csv'), usecols=['DFCI_MRN'])

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

cancer_type_sub.to_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/cancer_type_df.csv'), index=False)
mrn_stage_df.to_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/cancer_stage_df.csv'), index=False)