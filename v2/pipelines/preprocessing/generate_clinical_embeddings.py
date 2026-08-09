"""Generate mean-pooled Clinical ModernBERT note embeddings from tokenized note batches."""

from __future__ import annotations

import io
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd
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

TOKEN_FILE_PATTERN = re.compile(r"clinical_notes_tokenized_batch_(\d+)_tokens\.npy\.zst$")
MODEL_NAME = "Simonlee711/Clinical_ModernBERT"
DATALOADER_BATCH_SIZE = 64


def discover_token_files(token_path: Path) -> dict[int, Path]:
    """Discover token batch files."""
    token_files: dict[int, Path] = {}

    for file_path in token_path.iterdir():
        if not file_path.is_file():
            continue

        match = TOKEN_FILE_PATTERN.fullmatch(file_path.name)
        if match:
            token_files[int(match.group(1))] = file_path

    if not token_files:
        raise FileNotFoundError(f"No token batch files found in {token_path}")

    return dict(sorted(token_files.items()))


def load_token_batch(batch_path: Path) -> dict[str, np.ndarray]:
    """Load one zstd-compressed npz token batch from disk."""
    with open(batch_path, "rb") as handle:
        raw = zstd.decompress(handle.read())
    with np.load(io.BytesIO(raw), allow_pickle=False) as token_batch:
        return {
            "input_ids_flat": token_batch["input_ids_flat"],
            "row_splits": token_batch["row_splits"],
        }


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
    """Generate attention-masked mean-pooled embeddings for all token batches."""
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
                # ModernBERT has no pooler head, so .pooler_output is unavailable.
                # Use attention-masked mean pooling over last_hidden_state; the
                # clamp guards against all-padding rows.
                out = embedding_model(input_ids=tokens, attention_mask=masks).last_hidden_state
                mask = masks.unsqueeze(-1).to(out.dtype)
                preds = ((out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)).cpu()

            predictions[indices] = preds

        embeddings_bytes = io.BytesIO()
        np.save(embeddings_bytes, predictions.numpy().astype(np.float16))
        with open(EMBED_PATH / f"clinical_notes_embeddings_batch_{batch_idx}.npy.zst", "wb") as handle:
            handle.write(zstd.compress(embeddings_bytes.getvalue(), level=15))

        elapsed = (time.time() - start_time) / 60
        print(f"Batch {batch_idx} completed in {elapsed:.2f} minutes.")


if __name__ == "__main__":
    main()
