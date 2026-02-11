import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from embed_surv_utils import generate_survival_embedding_df

# Paths
PROJ_PATH = '/data/gusev/USERS/jpconnor/clinical_text_project/'
DATA_PATH = os.path.join(PROJ_PATH, 'data/')
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/VTE_data/processed_datasets/')
OUTPUT_PATH = os.path.join(SURV_PATH, 'time-to-icd/landmark_datasets/')
os.makedirs(OUTPUT_PATH, exist_ok=True)

events_data_sub = pd.read_csv(os.path.join(SURV_PATH, 'time-to-icd/tt_vte_plus_icd_level_3s.csv'))
embeddings_data = np.load(os.path.join(NOTES_PATH, 'full_VTE_embeddings_as_array.npy'))
notes_meta = pd.read_csv(os.path.join(NOTES_PATH, 'full_VTE_embeddings_metadata.csv'))

note_types = ['Clinician', 'Imaging', 'Pathology']
offsets = [i*3 for i in range(13)]
tte_cols = [col for col in events_data_sub if col.startswith('tt_')]
for offset in tqdm(offsets):

    offset_events_data = events_data_sub.copy()
    offset_notes_meta = notes_meta.copy()
    
    offset_events_data[tte_cols] -= offset * 30
    offset_notes_meta['NOTE_TIME_REL_FIRST_TREATMENT_START'] += offset * 30

    monthly_data = generate_survival_embedding_df(offset_notes_meta, offset_events_data, embeddings_data, note_types=note_types,
                                              note_timing_col='NOTE_TIME_REL_FIRST_TREATMENT_START', continuous_window=False,
                                              pool_fx={key : 'time_decay_mean' for key in note_types}, decay_param=0.01).dropna()

    monthly_data.to_csv(os.path.join(OUTPUT_PATH, f'time_decayed_events_df_plus_{offset}_months.csv'), index=False)