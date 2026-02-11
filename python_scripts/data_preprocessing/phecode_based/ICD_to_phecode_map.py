import os
import pandas as pd
from tqdm import tqdm
from embed_surv_utils import find_icd_code

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
DIAGNOSTICS_PATH = '/data/gusev/PROFILE/CLINICAL/OncDRS/ALL_2025_03/'
INTAE_DATA_PATH = '/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/'
CODE_PATH = os.path.join(DATA_PATH, 'code_data/')

# Load data
dfci_codes = pd.read_csv(os.path.join(CODE_PATH, 'DFCI_ICD10_codes.csv'), index_col=0)
dfci_phecodes = pd.read_csv(os.path.join(CODE_PATH, 'DFCI_Phecodes.csv'), index_col=0)

# Map ICD codes
icd_dict = dict(zip(dfci_codes['id'], dfci_codes['code']))
dfci_phecodes['ICD10'] = dfci_phecodes['id'].map(icd_dict)

# Deduplicate ICD–phecode mapping (keep highest phecode per ICD10)
deduped_data = []
map_cols = dfci_phecodes.columns

for icd in tqdm(dfci_phecodes['ICD10'].unique()):
    icd_subset = dfci_phecodes.loc[dfci_phecodes['ICD10'] == icd]
    entry_to_keep = icd_subset.sort_values(by='phecode', ascending=False).iloc[0].tolist()
    deduped_data.append(entry_to_keep)

assert len(deduped_data) == dfci_phecodes['ICD10'].nunique()

deduped_icd_df = pd.DataFrame(deduped_data, columns=map_cols)

# Load target ICDs/phecodes
icds_to_analyze = pd.read_csv(os.path.join(CODE_PATH, 'ICD10_Relevant_Cancer_Codes.csv'))
phecodes_to_predict = pd.read_csv(os.path.join(CODE_PATH, 'filtered_oncology_relevant_phecodes.csv'))
phecodes_to_predict['phecode_cat'] = phecodes_to_predict['phecode'].apply(lambda x: int(str(x).split('.')[0]))

phecode_descr = pd.read_csv(os.path.join(CODE_PATH, 'phecode_description.csv'))
phecode_descr.columns = ['phecode_cat', 'description']

phecodes_to_predict['phecode_cat_description'] = phecodes_to_predict['phecode_cat'].map(dict(zip(phecode_descr['phecode_cat'], phecode_descr['description'])))

# Old ICD codes joined with phecodes
old_icd_df = (icds_to_analyze[['ICD-10 Code']]
              .rename(columns={'ICD-10 Code': 'ICD10'})
              .merge(deduped_icd_df[['phecode', 'ICD10']], on='ICD10')
              [['phecode', 'ICD10']])

# Subset of phecodes to predict
subset_icd_df = deduped_icd_df.loc[deduped_icd_df['phecode'].isin(phecodes_to_predict['phecode'].unique()), ['phecode', 'ICD10'],]

# Combine and annotate
full_icd_phecode_df = pd.concat([old_icd_df, subset_icd_df]).drop_duplicates()
full_icd_phecode_df['description'] = full_icd_phecode_df['phecode'].map(dict(zip(deduped_icd_df['phecode'], deduped_icd_df['description'])))
full_icd_phecode_df['phecode_cat'] = full_icd_phecode_df['phecode'].apply(lambda x: int(str(x).split('.')[0]))
full_icd_phecode_df['phecode_cat_description'] = full_icd_phecode_df['phecode_cat'].map(dict(zip(phecode_descr['phecode_cat'], phecode_descr['description'])))
full_icd_phecode_df['ICD10_DESCR'] = full_icd_phecode_df['ICD10'].apply(find_icd_code)

# Reorder/rename columns
full_icd_phecode_df = full_icd_phecode_df[['ICD10', 'ICD10_DESCR', 'phecode', 'description', 'phecode_cat', 'phecode_cat_description']]
full_icd_phecode_df.columns = ['ICD10_CD', 'ICD10_CD_DESCR', 'PHECODE', 
                               'PHECODE_DESCR', 'PHECODE_CAT', 'PHECODE_CAT_DESCR']

# Save output
output_file = os.path.join(CODE_PATH, 'icd_to_phecode_map.csv')
full_icd_phecode_df.to_csv(output_file, index=False)
