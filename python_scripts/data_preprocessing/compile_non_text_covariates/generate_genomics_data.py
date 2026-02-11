import os 
import numpy as np
import pandas as pd

prof_path = '/data/gusev/PROFILE/CLINICAL/'
data_path = '/data/gusev/USERS/jpconnor/clinical_text_project/data/'

idmap = pd.read_csv(prof_path + 'PROFILE_2024_idmap.csv')
mut_2022_data = pd.read_csv(prof_path + 'PROFILE_2022.MUTATION_CARRIERS.csv', index_col=0).fillna(0)
mut_2024_data = pd.read_csv(prof_path + 'PROFILE_2024.MUTATION_CARRIERS.csv', index_col=0).fillna(0)

mut_2022_data['DFCI_MRN'] = mut_2022_data.index
mut_2024_data['DFCI_MRN'] = mut_2024_data.index.map(dict(zip(idmap['sample_id'], idmap['DFCI_MRN'])))

joint_cols = list(set(mut_2022_data.columns) & set(mut_2024_data.columns))
mut_merged = pd.concat([mut_2022_data[joint_cols], mut_2024_data[joint_cols]])

final_somatic_df = pd.get_dummies(idmap.merge(mut_merged, on='DFCI_MRN')
    .drop_duplicates(subset='DFCI_MRN', keep='first')
    .drop(columns=['sample_id', 'cbio_sample_id', 'cbio_patient_id', 'onco_tree_code', 
                   'briefcase', 'riker_pipeline_version', 'riker_run_version', 'CANCER_TYPE']), columns=['PANEL_VERSION'])

final_somatic_df.to_csv(data_path + 'PROFILE_2024_MUTATION_CARRIERS.csv', index=False)

prs_df = (pd.read_csv('/data/gusev/USERS/mjsaleh/PRS_PGScatalog/pgs_matrix_with_avg.tsv', sep='\t')
          .merge(idmap[['cbio_sample_id', 'DFCI_MRN']].rename(columns={'cbio_sample_id' : 'IID'}))
          .drop_duplicates(subset='DFCI_MRN', keep='first'))

prs_df.to_csv(data_path + 'PGS_DATA_VTE_COHORT.csv', index=False)