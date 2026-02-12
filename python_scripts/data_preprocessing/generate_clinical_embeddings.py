"""Generate Clinical Embeddings script for data preprocessing workflows."""

## RAN ON GCP ##
import os
import re
import json
import time

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel


# Dataset class
class TokenBatchDataset(Dataset):
    """Dataset wrapping tokenized inputs with attention masks."""

    def __init__(self, input_ids, att_masks):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.att_masks = torch.tensor(att_masks, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return idx, self.input_ids[idx], self.att_masks[idx]

# Paths
ROOT_PATH = '/home/patrickconnor/generate_clinical_embeddings'
TOKEN_PATH = os.path.join(ROOT_PATH, 'tokens')
EMBED_PATH = os.path.join(ROOT_PATH, 'embeddings')

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = AutoModel.from_pretrained('yikuan8/Clinical-Longformer').to(device)

# Get batch indices
batch_indices = {int(re.split('_', f)[4]) for f in TOKEN_PATH.iterdir() if f.name.startswith('VTE_notes_tokenized_batch_')}

# Process batches
for batch_idx in sorted(batch_indices):
    start_time = time.time()

    with open(os.path.join(TOKEN_PATH, f'VTE_notes_tokenized_batch_{batch_idx}_tokens.json')) as f:
        cur_batch = json.load(f)

    dataset = TokenBatchDataset(cur_batch['input_ids'], cur_batch['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    predictions = torch.empty((len(dataset), 768))

    for indices, tokens, masks in tqdm(dataloader, desc=f'Batch {batch_idx}'):
        tokens, masks = tokens.to(device), masks.to(device)

        with torch.no_grad():
            preds = embedding_model(tokens, masks)[1].cpu()

        predictions[indices] = preds

    torch.save(predictions, os.path.join(EMBED_PATH, f'VTE_notes_embeddings_batch_{batch_idx}.pt'))

    elapsed = (time.time() - start_time) / 60
    print(f'Batch {batch_idx} completed in {elapsed:.2f} minutes.\n')
