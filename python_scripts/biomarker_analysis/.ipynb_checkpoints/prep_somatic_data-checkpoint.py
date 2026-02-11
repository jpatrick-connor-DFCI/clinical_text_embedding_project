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

final_somatic_df = idmap.merge(mut_merged, on='DFCI_MRN')
final_somatic_df.to_csv(data_path + 'PROFILE_2024_MUTATION_CARRIERS.csv', index=False)