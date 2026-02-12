import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from embed_surv_utils import clean_text, deduplicate_texts

# Paths and constants
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
RAW_TEXT_PATH = "/data/gusev/PROFILE/CLINICAL/OncDRS/CLINICAL_TEXTS_2024_03/"
PREFIX = "RequestID-19-033-March24-"

COL_METADATA = [
    "PROC_DESC_STR", "PROVIDER_CRED_STR", "EVENT_DATE", "id", "DFCI_MRN",
    "ENCOUNTER_TYPE_DESC_STR", "RPT_DATE", "RPT_TYPE", "SOURCE_STR", "PROVIDER_CRED"
]
COLUMNS_TO_SAVE = COL_METADATA + ["CLINICAL_TEXT", "NOTE_TYPE"]

IMAGE_FILES = [f"{PREFIX}Imaging-{i}.json" for i in range(1, 8)]
PATHOLOGY_FILES = [f"{PREFIX}Path-{i}.json" for i in range(1, 11)]
CLINICIAN_FILES = [f"{PREFIX}ProgNote-{i}.json" for i in range(1, 12)]

FULL_FILENAMES = IMAGE_FILES + PATHOLOGY_FILES + CLINICIAN_FILES
FILE_SOURCES = (
    ["Imaging"] * len(IMAGE_FILES)
    + ["Pathology"] * len(PATHOLOGY_FILES)
    + ["Clinician"] * len(CLINICIAN_FILES)
)

VTE_DATA_PATH = "/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/"
BATCHED_DATA_PATH = os.path.join(DATA_PATH, "batched_datasets/")
BATCHED_TEXT_PATH = os.path.join(BATCHED_DATA_PATH, "batched_text/")

VTE_MRNS = pd.read_csv(VTE_DATA_PATH + "follow_up_vte_df_cohort.csv")["DFCI_MRN"].unique()

BATCH_SIZE = 50_000

# Extract and batch notes
note_count = 0
batch_count = 0
text_data_dict = {col: [] for col in COLUMNS_TO_SAVE}

for filename, source in tqdm(list(zip(FULL_FILENAMES, FILE_SOURCES))):
    docs = json.load(open((os.path.join(RAW_TEXT_PATH, filename)), 'r'))["response"]["docs"]

    for note in docs:
        if int(note["DFCI_MRN"]) not in VTE_MRNS:
            continue

        # Collect and clean text
        text_entries = [note[k] for k in note.keys() if "TEXT" in k]
        text_to_save = clean_text(" ".join(deduplicate_texts(text_entries)))

        # Fill metadata
        for col in COL_METADATA:
            text_data_dict[col].append(note.get(col, np.nan))

        text_data_dict["NOTE_TYPE"].append(source)
        text_data_dict["CLINICAL_TEXT"].append(text_to_save)

        note_count += 1

        # Dump batch if size exceeded
        if note_count >= BATCH_SIZE:
            with open(os.path.join(BATCHED_TEXT_PATH, f"VTE_notes_with_full_metadata_batch_{batch_count}.json"), 'w') as f:
                json.dump(text_data_dict, f)
                
            batch_count += 1
            note_count = 0
            text_data_dict = {col: [] for col in COLUMNS_TO_SAVE}

# Tokenization
BATCHED_TOKEN_PATH = os.path.join(BATCHED_DATA_PATH, "batched_tokens/")
TOKENIZER = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")

batch_files = sorted(os.listdir(BATCHED_TEXT_PATH))

for idx in tqdm(range(len(batch_files))):
    batch_file = f"VTE_notes_with_full_metadata_batch_{idx}.json"
    batch_path = os.path.join(BATCHED_DATA_PATH, "batched_text", batch_file)

    batched_data = json.load(open(batch_path), 'r')

    clinical_texts = batched_data.pop("CLINICAL_TEXT")
    tokenized = TOKENIZER(clinical_texts, padding="max_length", truncation=True)

    tokenized_dict = {"input_ids": tokenized["input_ids"],
                      "attention_mask": tokenized["attention_mask"]}

    with open(os.path.join(BATCHED_TOKEN_PATH, "tokens", f"VTE_notes_tokenized_batch_{idx}_tokens.json"), 'w') as f:
        json.dump(tokenized_dict, f)
    
    with open(os.path.join(BATCHED_TOKEN_PATH, "metadata",f"VTE_notes_tokenized_batch_{idx}_metadata.json")) as f:
        json.dump(batched_data, f)
