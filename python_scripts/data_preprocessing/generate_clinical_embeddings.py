"""Generate Clinical-Longformer note embeddings from tokenized note batches."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class TokenBatchDataset(Dataset):
    """Dataset wrapping ragged token sequences stored in flattened form."""

    def __init__(self, input_ids_flat: np.ndarray, row_splits: np.ndarray) -> None:
        if row_splits.ndim != 1:
            raise ValueError("row_splits must be one-dimensional.")
        if len(row_splits) == 0:
            raise ValueError("row_splits cannot be empty.")
        if int(row_splits[0]) != 0:
            raise ValueError("row_splits must start at 0.")
        if int(row_splits[-1]) != len(input_ids_flat):
            raise ValueError("row_splits does not match the flattened token length.")

        self.input_ids_flat = input_ids_flat
        self.row_splits = row_splits

    def __len__(self) -> int:
        return len(self.row_splits) - 1

    def __getitem__(self, idx: int) -> tuple[int, np.ndarray]:
        start_idx = int(self.row_splits[idx])
        end_idx = int(self.row_splits[idx + 1])
        return idx, self.input_ids_flat[start_idx:end_idx]


# Paths and constants
# This step typically runs on GCP after token batches are copied over from the
# preprocessing cluster.
GCP_ROOT = Path(os.environ.get("CLINICAL_EMBED_GCP_ROOT", "/home/patrickconnor/generate_clinical_embeddings"))
TOKEN_PATH = Path(os.environ.get("TOKEN_PATH", str(GCP_ROOT / "tokens")))
EMBED_PATH = Path(os.environ.get("EMBED_PATH", str(GCP_ROOT / "embeddings")))

TOKEN_FILE_PATTERNS = (
    (0, re.compile(r"VTE_notes_tokenized_batch_(\d+)_tokens\.npz$")),
    (1, re.compile(r"VTE_notes_tokenized_batch_(\d+)_tokens\.json$")),
)
MODEL_NAME = "yikuan8/Clinical-Longformer"
DATALOADER_BATCH_SIZE = 64


def discover_token_files(token_path: Path) -> dict[int, Path]:
    """Discover token batch files, preferring the newer binary format."""
    token_files: dict[int, tuple[int, Path]] = {}

    for file_path in token_path.iterdir():
        if not file_path.is_file():
            continue

        for priority, pattern in TOKEN_FILE_PATTERNS:
            match = pattern.fullmatch(file_path.name)
            if not match:
                continue

            batch_idx = int(match.group(1))
            existing = token_files.get(batch_idx)
            if existing is None or priority < existing[0]:
                token_files[batch_idx] = (priority, file_path)
            break

    if not token_files:
        raise FileNotFoundError(f"No token batch files found in {token_path}")

    return {batch_idx: file_path for batch_idx, (_, file_path) in sorted(token_files.items())}


def pack_token_sequences(token_sequences: list[list[int]], token_dtype: np.dtype | type[np.integer]) -> dict[str, np.ndarray]:
    """Pack variable-length token sequences into flattened arrays."""
    seq_lengths = np.fromiter((len(seq) for seq in token_sequences), dtype=np.int32, count=len(token_sequences))
    row_splits = np.empty(len(token_sequences) + 1, dtype=np.int64)
    row_splits[0] = 0
    row_splits[1:] = np.cumsum(seq_lengths, dtype=np.int64)

    if row_splits[-1] == 0:
        input_ids_flat = np.empty(0, dtype=token_dtype)
    else:
        input_ids_flat = np.concatenate([np.asarray(seq, dtype=token_dtype) for seq in token_sequences])

    return {"input_ids_flat": input_ids_flat, "row_splits": row_splits}


def load_legacy_json_batch(batch_path: Path) -> dict[str, np.ndarray]:
    """Load an older padded JSON token batch and convert it to ragged storage."""
    with open(batch_path) as handle:
        token_batch = json.load(handle)

    token_sequences = []
    for input_ids, attention_mask in zip(token_batch["input_ids"], token_batch["attention_mask"], strict=True):
        seq_len = int(np.sum(attention_mask))
        token_sequences.append(input_ids[:seq_len])

    max_token_id = max((max(seq) for seq in token_sequences if seq), default=0)
    token_dtype = np.uint16 if max_token_id <= np.iinfo(np.uint16).max else np.int32
    return pack_token_sequences(token_sequences, token_dtype)


def load_token_batch(batch_path: Path) -> dict[str, np.ndarray]:
    """Load one token batch from disk in either the new or legacy format."""
    if batch_path.suffix == ".npz":
        with np.load(batch_path, allow_pickle=False) as token_batch:
            return {
                "input_ids_flat": token_batch["input_ids_flat"],
                "row_splits": token_batch["row_splits"],
            }

    if batch_path.suffix == ".json":
        return load_legacy_json_batch(batch_path)

    raise ValueError(f"Unsupported token batch format: {batch_path}")


def build_collate_fn(pad_token_id: int):
    """Build a collator that pads to the longest note in each minibatch."""

    def collate_token_batch(batch: list[tuple[int, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.tensor([idx for idx, _ in batch], dtype=torch.long)
        seq_lengths = [len(seq) for _, seq in batch]
        max_seq_len = max(seq_lengths, default=0)

        input_ids = torch.full((len(batch), max_seq_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_seq_len), dtype=torch.long)

        for row_idx, (_, token_ids) in enumerate(batch):
            seq_len = len(token_ids)
            if seq_len == 0:
                continue

            input_ids[row_idx, :seq_len] = torch.as_tensor(token_ids, dtype=torch.long)
            attention_mask[row_idx, :seq_len] = 1

        return indices, input_ids, attention_mask

    return collate_token_batch


def main() -> None:
    """Generate pooled CLS embeddings for all token batches."""
    from transformers import AutoModel

    EMBED_PATH.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = device == "cuda"

    embedding_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    embedding_model.eval()
    hidden_size = embedding_model.config.hidden_size
    pad_token_id = embedding_model.config.pad_token_id
    if pad_token_id is None:
        raise ValueError("Model config must define pad_token_id for dynamic padding.")

    token_files = discover_token_files(TOKEN_PATH)
    collate_fn = build_collate_fn(pad_token_id)
    print(f"Found {len(token_files)} token batches in {TOKEN_PATH}.")

    for batch_idx, batch_path in token_files.items():
        start_time = time.time()
        cur_batch = load_token_batch(batch_path)

        dataset = TokenBatchDataset(cur_batch["input_ids_flat"], cur_batch["row_splits"])
        dataloader = DataLoader(
            dataset,
            batch_size=DATALOADER_BATCH_SIZE,
            shuffle=False,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        predictions = torch.empty((len(dataset), hidden_size), dtype=torch.float32)

        for indices, tokens, masks in tqdm(dataloader, desc=f"Batch {batch_idx}"):
            tokens = tokens.to(device, non_blocking=pin_memory)
            masks = masks.to(device, non_blocking=pin_memory)

            with torch.inference_mode():
                preds = embedding_model(input_ids=tokens, attention_mask=masks).pooler_output.cpu()

            predictions[indices] = preds

        torch.save(predictions, EMBED_PATH / f"VTE_notes_embeddings_batch_{batch_idx}.pt")

        elapsed = (time.time() - start_time) / 60
        print(f"Batch {batch_idx} completed in {elapsed:.2f} minutes.")


if __name__ == "__main__":
    main()
