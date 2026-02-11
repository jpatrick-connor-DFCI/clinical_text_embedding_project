import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Paths
BATCHED_DATA_PATH = '/data/gusev/USERS/jpconnor/clinical_text_project/data/batched_datasets/'
TOKEN_PATH = os.path.join(BATCHED_DATA_PATH, 'batched_tokens/tokens/')
META_PATH = os.path.join(BATCHED_DATA_PATH, 'batched_tokens/metadata/')
EMBEDS_PATH = os.path.join(BATCHED_DATA_PATH, 'embeddings/')
PROC_PATH = os.path.join(BATCHED_DATA_PATH, 'processed_datasets/')
SURVIVAL_FILE = '/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/follow_up_vte_df_cohort.csv'

os.makedirs(PROC_PATH, exist_ok=True)

# Load survival data
survival_df = pd.read_csv(SURVIVAL_FILE)
mrn_tstart_dict = dict(zip(survival_df['DFCI_MRN'].tolist(),
                           pd.to_datetime(survival_df['first_treatment_date'], format='%Y-%m-%d').tolist()))

# Detect all batch indices directly
batch_idxs = [int(f.split('_')[4]) for f in os.listdir(TOKEN_PATH) if f.endswith('.json')]
print(f'Found {len(batch_idxs)} batch indices.')

# Load metadata
metadata_list = []
embedding_tensor_list = []
for batch_idx in tqdm(batch_idxs, desc='Loading metadata and embeddings'):
    # load metadata
    cur_metadata = pd.read_json(os.path.join(META_PATH, f'VTE_notes_tokenized_batch_{batch_idx}_metadata.json'))
    cur_metadata['SUB_BATCH_FILE_ID'] = f'batch_{batch_idx}'
    cur_metadata['WITHIN_SUB_BATCH_INDEX'] = list(range(len(cur_metadata)))
    metadata_list.append(cur_metadata)
    
    # load embeddings
    embedding_tensor_list.append(torch.load(os.path.join(EMBEDS_PATH, f'VTE_notes_embeddings_batch_{batch_idx}.pt')))

metadata_df = pd.concat(metadata_list, ignore_index=True)
embeddings = torch.cat(embedding_tensor_list, dim=0).numpy()

# Add derived columns
metadata_df['EMBEDDING_INDEX'] = range(len(metadata_df))
metadata_df['NOTE_DATETIME'] = pd.to_datetime(metadata_df['EVENT_DATE'], format='%Y-%m-%dT%H:%M:%SZ')
metadata_df['FIRST_TREATMENT_START_DT'] = metadata_df['DFCI_MRN'].map(mrn_tstart_dict)
metadata_df['NOTE_TIME_REL_FIRST_TREATMENT_START'] = ((metadata_df['NOTE_DATETIME'] - metadata_df['FIRST_TREATMENT_START_DT']).dt.days)

# Save outputs
metadata_file = os.path.join(PROC_PATH, 'full_VTE_embeddings_metadata.csv')
embeds_file = os.path.join(PROC_PATH, 'full_VTE_embeddings_as_array.npy')

metadata_df.to_csv(metadata_file, index=False)
np.save(embeds_file, embeddings)

print(f'Saved merged metadata to {metadata_file}')
print(f'Saved merged embeddings to {embeds_file}')