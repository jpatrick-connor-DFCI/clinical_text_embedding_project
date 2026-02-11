import os
import pandas as pd
import numpy as np
from embed_surv_utils import generate_survival_embedding_df

# Paths
PROJ_PATH = '/data/gusev/USERS/jpconnor/clinical_text_project/'
DATA_PATH = os.path.join(PROJ_PATH, 'data/')
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/processed_datasets/')

events_data_sub = pd.read_csv(os.path.join(SURV_PATH, 'time-to-icd/tt_vte_plus_icd_level_3s.csv'))
embeddings_data = np.load(os.path.join(NOTES_PATH, 'full_VTE_embeddings_as_array.npy'))
notes_meta = pd.read_csv(os.path.join(NOTES_PATH, 'full_VTE_embeddings_metadata.csv'))

note_types = ['Clinician', 'Imaging', 'Pathology']
monthly_data = generate_survival_embedding_df(notes_meta, events_data_sub, embeddings_data, note_types=note_types,
                                              note_timing_col='NOTE_TIME_REL_FIRST_TREATMENT_START', continuous_window=False,
                                              pool_fx={key : 'time_decay_mean' for key in note_types}, decay_param=0.01).dropna()

monthly_data.to_csv(os.path.join(SURV_PATH, 'time-to-icd/time_decayed_events_df.csv'), index=False)