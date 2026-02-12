import os
import pandas as pd
from functools import reduce

# Paths
PROFILE_PATH = '/data/gusev/PROFILE/CLINICAL/'
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')
INTAE_DATA_PATH = os.path.join(PROFILE_PATH, 'robust_VTE_pred_project_2025_03_cohort/data/')

vte_data = pd.read_csv(os.path.join(INTAE_DATA_PATH, 'follow_up_vte_df_cohort.csv'), usecols=['DFCI_MRN', 'first_treatment_date'])

idmap = pd.read_csv(os.path.join(PROFILE_PATH, 'PROFILE_2024_idmap.csv'), usecols=['DFCI_MRN', 'sample_id', 'CANCER_TYPE', 'cbio_sample_id', 'PANEL_VERSION'])
px_metadata = (pd.read_csv(os.path.join(PROFILE_PATH, 'OncDRS/ALL_2025_03/SOMATIC_SPECIMEN.csv'), 
                           usecols=['DFCI_MRN', 'SAMPLE_BL_NBR', 'CANCER_TYPE', 'SAMPLE_COLLECTION_DT'])
               .rename(columns={'SAMPLE_BL_NBR' : 'sample_id'})
               .merge(idmap, on=['DFCI_MRN', 'sample_id', 'CANCER_TYPE'])
               .merge(vte_data, on='DFCI_MRN'))
px_metadata['SAMPLE_COLLECTION_DT'] = pd.to_datetime(px_metadata['SAMPLE_COLLECTION_DT'])
px_metadata['first_treatment_date'] = pd.to_datetime(px_metadata['first_treatment_date'])
px_metadata['days_from_sequencing_to_first_treatment'] = (px_metadata['first_treatment_date'] - px_metadata['SAMPLE_COLLECTION_DT']).apply(lambda x : x.days)
px_metadata_min = (px_metadata
                   .loc[px_metadata
                        .groupby("DFCI_MRN")["days_from_sequencing_to_first_treatment"]
                        .idxmin()]
                   .reset_index(drop=True))

dfs_to_merge = []
for mut_type in ['AMP', 'CNV', 'DEL', 'SNV']:
    if mut_type == 'SNV':
        cur_df = pd.read_csv(os.path.join(PROFILE_PATH, 'PROFILE_2024.MUTATION_CARRIERS.csv'), index_col=0)
    else:
        cur_df = pd.read_csv(os.path.join(PROFILE_PATH, f'PROFILE_2024.ALL_{mut_type}_CARRIERS.csv'), index_col=0)

    cur_df.columns = [col + f'_{mut_type}' for col in cur_df.columns]
    cur_df['sample_id'] = cur_df.index
    cur_df = cur_df.loc[cur_df['sample_id'].isin(px_metadata_min['sample_id'].unique())].reset_index(drop=True)
    dfs_to_merge.append(cur_df)

complete_mutation_data = (reduce(lambda left, right : pd.merge(left, right, on='sample_id', how='inner'), dfs_to_merge)
                          .dropna(axis=1, how='all')
                          .fillna(0)
                          .merge(px_metadata_min, on='sample_id'))

mutation_columns = [col for col in complete_mutation_data.columns if ('_AMP' in col) or ('_CNV' in col) or ('_DEL' in col) or ('_SNV' in col)]
metadata_columns = px_metadata_min.columns.tolist()

complete_mutation_data = complete_mutation_data[metadata_columns + mutation_columns]
complete_mutation_data[mutation_columns] = complete_mutation_data[mutation_columns].astype(int)

complete_mutation_data.to_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/complete_somatic_data_df.csv'), index=False)

prs_df = (pd.read_csv('/data/gusev/USERS/mjsaleh/PRS_PGScatalog/pgs_matrix_with_avg.tsv', sep='\t')
          .rename(columns={'IID' : 'cbio_sample_id'})
          .merge(px_metadata_min[['cbio_sample_id', 'DFCI_MRN']], on='cbio_sample_id'))
prs_df.to_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/complete_germline_data_df.csv'), index=False)