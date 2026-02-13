"""Generate IO patient base dataframe with IO_START-adjusted survival times."""

import os
import random
import pandas as pd

random.seed(42)  # set seed for reproducibility

# Paths
IO_PATH = '/data/gusev/USERS/mjsaleh/'
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')

# Load datasets
cancer_type_df = pd.read_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/cancer_type_df.csv'))
tt_death_df = pd.read_csv(os.path.join(SURV_PATH, 'death_met_surv_df.csv'))
irAE_df = pd.read_csv(os.path.join(IO_PATH, 'IO_START.csv'), index_col=0).rename(columns={'MRN' : 'DFCI_MRN'})
somatic_df = pd.read_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/complete_somatic_data_df.csv'))

vte_data = pd.read_csv("/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/follow_up_vte_df_cohort.csv")
vte_data_sub = vte_data[["DFCI_MRN", "last_contact_date"]].copy()
vte_data_sub["last_contact_date"] = pd.to_datetime(vte_data_sub["last_contact_date"])

# Add last_contact_date to survival df
tt_death_df['last_contact_date'] = tt_death_df['DFCI_MRN'].map(dict(zip(vte_data_sub['DFCI_MRN'], vte_data_sub['last_contact_date'])))

# Merge IO patients with survival demographics
irAE_df = irAE_df.merge(tt_death_df[['DFCI_MRN', 'death', 'GENDER', 'AGE_AT_TREATMENTSTART', 'last_contact_date']], on='DFCI_MRN')

# Adjust tt_death for IO patients (time from IO_START to last contact)
irAE_df['tt_death'] = (irAE_df['last_contact_date'] - pd.to_datetime(irAE_df['IO_START'])).dt.days

# Build IO patient base dataframe
IO_patient_base_df = (irAE_df[['DFCI_MRN', 'tt_death', 'death', 'GENDER', 'AGE_AT_TREATMENTSTART']]
                      .merge(cancer_type_df, on='DFCI_MRN')
                      .merge(somatic_df, on='DFCI_MRN'))

IO_patient_base_df.to_csv(os.path.join(MARKER_PATH, 'IO_patient_base_df.csv'), index=False)
