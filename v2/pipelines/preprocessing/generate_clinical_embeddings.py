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


# Paths and constants. The batched_tokens directory is copied to the GPU machine
# separately and is expected to contain its original tokens/ and metadata/
# subdirectories. The generated embeddings/ directory is its sibling under data/.
DATA_ROOT = Path(os.environ.get("CLINICAL_EMBED_DATA_ROOT", "/home/patrickconnor/data"))
BATCHED_TOKENS_PATH = Path(
    os.environ.get("BATCHED_TOKENS_PATH", str(DATA_ROOT / "batched_tokens"))
)
TOKEN_PATH = Path(os.environ.get("TOKEN_PATH", str(BATCHED_TOKENS_PATH / "tokens")))
EMBED_PATH = Path(os.environ.get("EMBED_PATH", str(DATA_ROOT / "embeddings")))

TOKEN_FILE_PATTERN = re.compile(r"clinical_notes_tokenized_batch_(\d+)_tokens\.npy\.zst$")
MODEL_NAME = "Simonlee711/Clinical_ModernBERT"
MAX_BATCH_SIZE = int(os.environ.get("EMBED_MAX_BATCH_SIZE", "64"))
# Attention memory grows approximately with batch_size * sequence_length**2.
# This default permits one 8,192-token note, four 4,096-token notes, etc.
MAX_ATTENTION_ELEMENTS = int(os.environ.get("EMBED_MAX_ATTENTION_ELEMENTS", str(8192**2)))

if MAX_BATCH_SIZE < 1:
    raise ValueError("EMBED_MAX_BATCH_SIZE must be at least 1.")
if MAX_ATTENTION_ELEMENTS < 1:
    raise ValueError("EMBED_MAX_ATTENTION_ELEMENTS must be at least 1.")


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


def build_length_bucketed_batches(row_splits: np.ndarray) -> list[list[int]]:
    """Group similarly sized notes while respecting the attention-memory budget.

    Sorting by length minimizes dynamic-padding waste. Predictions are written
    back using their original row indices, so this does not change output order.
    """
    sequence_lengths = np.diff(row_splits)
    sorted_indices = np.argsort(sequence_lengths, kind="stable")
    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_max_length = 0

    for raw_idx in sorted_indices:
        idx = int(raw_idx)
        sequence_length = max(int(sequence_lengths[idx]), 1)
        candidate_max_length = max(current_max_length, sequence_length)
        candidate_size = len(current_batch) + 1
        exceeds_budget = candidate_size * candidate_max_length**2 > MAX_ATTENTION_ELEMENTS

        if current_batch and (candidate_size > MAX_BATCH_SIZE or exceeds_budget):
            batches.append(current_batch)
            current_batch = [idx]
            current_max_length = sequence_length
        else:
            current_batch.append(idx)
            current_max_length = candidate_max_length

    if current_batch:
        batches.append(current_batch)

    return batches


def embed_minibatch(
    embedding_model: torch.nn.Module,
    tokens: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool one minibatch, splitting and retrying after a CUDA OOM."""
    try:
        with torch.inference_mode():
            # The checkpoint has no trained pooler head. Use attention-masked
            # mean pooling over last_hidden_state instead.
            out = embedding_model(input_ids=tokens, attention_mask=masks).last_hidden_state
            mask = masks.unsqueeze(-1).to(out.dtype)
            return ((out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)).cpu()
    except torch.OutOfMemoryError as exc:
        if tokens.shape[0] == 1:
            raise RuntimeError(
                "CUDA ran out of memory for a single note with padded length "
                f"{tokens.shape[1]}. Reduce the preprocessing token-length limit or "
                "run on a GPU with more memory."
            ) from exc

        split_at = tokens.shape[0] // 2
        print(
            "CUDA OOM for minibatch "
            f"(size={tokens.shape[0]}, length={tokens.shape[1]}); retrying as smaller batches."
        )
        torch.cuda.empty_cache()
        left = embed_minibatch(embedding_model, tokens[:split_at], masks[:split_at])
        right = embed_minibatch(embedding_model, tokens[split_at:], masks[split_at:])
        return torch.cat((left, right), dim=0)


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
        minibatches = build_length_bucketed_batches(cur_batch["row_splits"])
        dataloader = DataLoader(
            dataset,
            batch_sampler=minibatches,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        predictions = torch.empty((len(dataset), hidden_size), dtype=torch.float32)

        for indices, tokens, masks in tqdm(dataloader, desc=f"Batch {batch_idx}"):
            tokens = tokens.to(device, non_blocking=pin_memory)
            masks = masks.to(device, non_blocking=pin_memory)
            preds = embed_minibatch(embedding_model, tokens, masks)
            predictions[indices] = preds

        embeddings_bytes = io.BytesIO()
        np.save(embeddings_bytes, predictions.numpy().astype(np.float16))
        with open(EMBED_PATH / f"clinical_notes_embeddings_batch_{batch_idx}.npy.zst", "wb") as handle:
            handle.write(zstd.compress(embeddings_bytes.getvalue(), level=15))

        elapsed = (time.time() - start_time) / 60
        print(f"Batch {batch_idx} completed in {elapsed:.2f} minutes.")


if __name__ == "__main__":
    main()
