"""Generate mean-pooled Clinical ModernBERT note embeddings from tokenized note batches."""

from __future__ import annotations

import argparse
import io
import os
import re
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zstandard as zstd
from torch.utils.data import Dataset
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
EMBED_FILE_PATTERN = re.compile(r"clinical_notes_embeddings_batch_(\d+)\.npy\.zst$")
MODEL_NAME = "Simonlee711/Clinical_ModernBERT"
MAX_BATCH_SIZE = int(os.environ.get("EMBED_MAX_BATCH_SIZE", "128"))
# Each padding-free minibatch is limited by the sum of sequence_length**2.
MAX_ATTENTION_ELEMENTS = int(os.environ.get("EMBED_MAX_ATTENTION_ELEMENTS", str(8192**2)))
COLLATE_WORKERS = int(
    os.environ.get("EMBED_COLLATE_WORKERS", os.environ.get("EMBED_DATALOADER_WORKERS", "2"))
)
COLLATE_PREFETCH_FACTOR = int(
    os.environ.get(
        "EMBED_COLLATE_PREFETCH_FACTOR",
        os.environ.get("EMBED_DATALOADER_PREFETCH_FACTOR", "2"),
    )
)
ZSTD_LEVEL = int(os.environ.get("EMBED_ZSTD_LEVEL", "3"))
COMPILE_MODEL = os.environ.get("EMBED_COMPILE", "1").lower() not in {"0", "false", "no"}
COMPILE_MODE = os.environ.get("EMBED_COMPILE_MODE", "default")

if MAX_BATCH_SIZE < 1:
    raise ValueError("EMBED_MAX_BATCH_SIZE must be at least 1.")
if MAX_ATTENTION_ELEMENTS < 1:
    raise ValueError("EMBED_MAX_ATTENTION_ELEMENTS must be at least 1.")
if COLLATE_WORKERS < 0:
    raise ValueError("EMBED_COLLATE_WORKERS cannot be negative.")
if COLLATE_PREFETCH_FACTOR < 1:
    raise ValueError("EMBED_COLLATE_PREFETCH_FACTOR must be at least 1.")
if not 1 <= ZSTD_LEVEL <= 22:
    raise ValueError("EMBED_ZSTD_LEVEL must be between 1 and 22.")


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


def discover_embedded_batches(embed_path: Path) -> set[int]:
    """Return batch indices with completed embedding output files."""
    if not embed_path.exists():
        return set()

    embedded_batches: set[int] = set()
    for file_path in embed_path.iterdir():
        if not file_path.is_file():
            continue

        match = EMBED_FILE_PATTERN.fullmatch(file_path.name)
        if match:
            embedded_batches.add(int(match.group(1)))

    return embedded_batches


def embedding_output_path(batch_idx: int) -> Path:
    """Return the final output path for one embedding batch."""
    return EMBED_PATH / f"clinical_notes_embeddings_batch_{batch_idx}.npy.zst"


def load_token_batch(batch_path: Path) -> dict[str, np.ndarray]:
    """Load one zstd-compressed npz token batch from disk."""
    with open(batch_path, "rb") as handle:
        raw = zstd.decompress(handle.read())
    with np.load(io.BytesIO(raw), allow_pickle=False) as token_batch:
        return {
            "input_ids_flat": token_batch["input_ids_flat"],
            "row_splits": token_batch["row_splits"],
        }


def collate_packed_batch(
    batch: list[tuple[int, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Flatten notes and construct the metadata required by FlashAttention-2.

    ModernBERT treats the flattened tokens as independent sequences using
    restarted position IDs and cumulative sequence boundaries. No padding or
    attention-mask tensor is created.
    """
    indices = torch.tensor([idx for idx, _ in batch], dtype=torch.long)
    sequence_lengths_np = np.asarray([len(seq) for _, seq in batch], dtype=np.int32)
    if np.any(sequence_lengths_np == 0):
        raise ValueError("Padding-free embedding requires every note to contain at least one token.")

    input_ids_np = np.concatenate([seq for _, seq in batch]).astype(np.int64, copy=False)
    position_ids_np = np.concatenate(
        [np.arange(sequence_length, dtype=np.int64) for sequence_length in sequence_lengths_np]
    )
    cumulative_lengths_np = np.empty(len(sequence_lengths_np) + 1, dtype=np.int32)
    cumulative_lengths_np[0] = 0
    np.cumsum(sequence_lengths_np, out=cumulative_lengths_np[1:])

    return (
        indices,
        torch.from_numpy(input_ids_np).unsqueeze(0),
        torch.from_numpy(position_ids_np).unsqueeze(0),
        torch.from_numpy(sequence_lengths_np),
        torch.from_numpy(cumulative_lengths_np),
        int(sequence_lengths_np.max()),
    )


def collate_dataset_minibatch(
    dataset: TokenBatchDataset,
    minibatch_indices: list[int],
    pin_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Collate one sampler minibatch, optionally pinning GPU-bound tensors."""
    packed_batch = collate_packed_batch([dataset[idx] for idx in minibatch_indices])
    if not pin_memory:
        return packed_batch

    indices, tokens, position_ids, sequence_lengths, cumulative_lengths, max_length = packed_batch
    return (
        indices,
        tokens.pin_memory(),
        position_ids.pin_memory(),
        sequence_lengths.pin_memory(),
        cumulative_lengths.pin_memory(),
        max_length,
    )


def prefetch_collated_minibatches(
    dataset: TokenBatchDataset,
    minibatches: list[list[int]],
    pin_memory: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """Vectorize and pin upcoming minibatches in bounded background threads."""
    if COLLATE_WORKERS == 0:
        for minibatch_indices in minibatches:
            yield collate_dataset_minibatch(dataset, minibatch_indices, pin_memory)
        return

    minibatch_iter = iter(minibatches)
    max_pending = COLLATE_WORKERS * COLLATE_PREFETCH_FACTOR
    with ThreadPoolExecutor(max_workers=COLLATE_WORKERS, thread_name_prefix="collator") as executor:
        pending: deque[Future[Any]] = deque()
        for _ in range(max_pending):
            try:
                minibatch_indices = next(minibatch_iter)
            except StopIteration:
                break
            pending.append(
                executor.submit(collate_dataset_minibatch, dataset, minibatch_indices, pin_memory)
            )

        while pending:
            packed_batch = pending.popleft().result()
            try:
                minibatch_indices = next(minibatch_iter)
            except StopIteration:
                pass
            else:
                pending.append(
                    executor.submit(collate_dataset_minibatch, dataset, minibatch_indices, pin_memory)
                )
            yield packed_batch


def build_length_bucketed_batches(row_splits: np.ndarray) -> list[list[int]]:
    """Group similarly sized notes while respecting the attention-work budget.

    Padding-free attention work is proportional to the sum of each sequence's
    squared length. Predictions are written back using their original row
    indices, so length sorting does not change output order.
    """
    sequence_lengths = np.diff(row_splits)
    sorted_indices = np.argsort(sequence_lengths, kind="stable")
    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_attention_elements = 0

    for raw_idx in sorted_indices:
        idx = int(raw_idx)
        sequence_length = max(int(sequence_lengths[idx]), 1)
        candidate_size = len(current_batch) + 1
        candidate_attention_elements = current_attention_elements + sequence_length**2
        exceeds_budget = candidate_attention_elements > MAX_ATTENTION_ELEMENTS

        if current_batch and (candidate_size > MAX_BATCH_SIZE or exceeds_budget):
            batches.append(current_batch)
            current_batch = [idx]
            current_attention_elements = sequence_length**2
        else:
            current_batch.append(idx)
            current_attention_elements = candidate_attention_elements

    if current_batch:
        batches.append(current_batch)

    return batches


class ModelRunner:
    """Call a model through torch.compile, falling back if a graph fails."""

    def __init__(self, embedding_model: torch.nn.Module, compile_model: bool) -> None:
        self.eager_forward = embedding_model.forward
        self.forward: Callable[..., Any] = self.eager_forward
        self.compiled_active = False

        if compile_model:
            self.forward = torch.compile(
                self.eager_forward,
                fullgraph=True,
                dynamic=True,
                mode=COMPILE_MODE,
            )
            self.compiled_active = True

    def __call__(self, **model_inputs: Any) -> Any:
        if not self.compiled_active:
            return self.forward(**model_inputs)

        try:
            return self.forward(**model_inputs)
        except torch.OutOfMemoryError:
            raise
        except Exception as exc:
            print(
                "torch.compile failed; continuing with eager execution "
                f"({type(exc).__name__}: {exc}).",
                flush=True,
            )
            self.forward = self.eager_forward
            self.compiled_active = False
            return self.forward(**model_inputs)


def embed_minibatch(
    model_runner: ModelRunner,
    tokens: torch.Tensor,
    position_ids: torch.Tensor,
    sequence_lengths: torch.Tensor,
    cumulative_lengths: torch.Tensor,
    max_sequence_length: int,
) -> torch.Tensor:
    """Embed and mean-pool a packed minibatch, splitting after a CUDA OOM."""
    try:
        with torch.inference_mode():
            output = model_runner(
                input_ids=tokens,
                position_ids=position_ids,
                cu_seq_lens_q=cumulative_lengths,
                cu_seq_lens_k=cumulative_lengths,
                max_length_q=max_sequence_length,
                max_length_k=max_sequence_length,
            ).last_hidden_state.squeeze(0)
            # Accumulate means in FP32 even though the model runs in BF16.
            return torch.segment_reduce(
                output.float(),
                reduce="mean",
                lengths=sequence_lengths,
            ).cpu()
    except torch.OutOfMemoryError as exc:
        note_count = len(sequence_lengths)
        if note_count == 1:
            raise RuntimeError(
                "CUDA ran out of memory for a single note with length "
                f"{tokens.shape[1]}. Reduce the preprocessing token-length limit or "
                "run on a GPU with more memory."
            ) from exc

        split_at = note_count // 2
        token_split_at = int(sequence_lengths[:split_at].sum().item())
        print(
            "CUDA OOM for minibatch "
            f"(notes={note_count}, tokens={tokens.shape[1]}); retrying as smaller batches."
        )
        torch.cuda.empty_cache()

        left_lengths = sequence_lengths[:split_at]
        right_lengths = sequence_lengths[split_at:]
        left_cumulative_lengths = torch.cat(
            (cumulative_lengths.new_zeros(1), left_lengths.cumsum(dim=0))
        )
        right_cumulative_lengths = torch.cat(
            (cumulative_lengths.new_zeros(1), right_lengths.cumsum(dim=0))
        )
        left = embed_minibatch(
            model_runner,
            tokens[:, :token_split_at],
            position_ids[:, :token_split_at],
            left_lengths,
            left_cumulative_lengths,
            int(left_lengths.max().item()),
        )
        right = embed_minibatch(
            model_runner,
            tokens[:, token_split_at:],
            position_ids[:, token_split_at:],
            right_lengths,
            right_cumulative_lengths,
            int(right_lengths.max().item()),
        )
        return torch.cat((left, right), dim=0)


def save_embedding_batch(output_path: Path, predictions: np.ndarray) -> None:
    """Compress and atomically save one completed embedding batch."""
    embeddings_bytes = io.BytesIO()
    np.save(embeddings_bytes, predictions.astype(np.float16, copy=False))
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    with open(partial_path, "wb") as handle:
        handle.write(zstd.compress(embeddings_bytes.getvalue(), level=ZSTD_LEVEL))
    partial_path.replace(output_path)


def prefetch_token_batches(
    token_files: dict[int, Path],
) -> Iterator[tuple[int, Path, dict[str, np.ndarray]]]:
    """Load and decompress the next outer batch while the GPU processes this one."""
    items = iter(token_files.items())
    try:
        first_batch_idx, first_batch_path = next(items)
    except StopIteration:
        return

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="token-loader") as executor:
        batch_idx = first_batch_idx
        batch_path = first_batch_path
        load_future = executor.submit(load_token_batch, batch_path)
        for next_item in items:
            token_batch = load_future.result()
            next_batch_idx, next_batch_path = next_item
            load_future = executor.submit(load_token_batch, next_batch_path)
            yield batch_idx, batch_path, token_batch
            batch_idx = next_batch_idx
            batch_path = next_batch_path
        yield batch_idx, batch_path, load_future.result()


def embed_token_batch(
    model_runner: ModelRunner,
    token_batch: dict[str, np.ndarray],
    hidden_size: int,
    device: str,
    pin_memory: bool,
    batch_idx: int,
) -> torch.Tensor:
    """Embed all notes in one token batch."""
    dataset = TokenBatchDataset(token_batch["input_ids_flat"], token_batch["row_splits"])
    minibatches = build_length_bucketed_batches(token_batch["row_splits"])

    predictions = torch.empty((len(dataset), hidden_size), dtype=torch.float32)
    for indices, tokens, position_ids, sequence_lengths, cumulative_lengths, max_length in tqdm(
        prefetch_collated_minibatches(dataset, minibatches, pin_memory),
        total=len(minibatches),
        desc=f"Batch {batch_idx}",
    ):
        tokens = tokens.to(device, non_blocking=pin_memory)
        position_ids = position_ids.to(device, non_blocking=pin_memory)
        sequence_lengths = sequence_lengths.to(device, non_blocking=pin_memory)
        cumulative_lengths = cumulative_lengths.to(device, non_blocking=pin_memory)
        preds = embed_minibatch(
            model_runner,
            tokens,
            position_ids,
            sequence_lengths,
            cumulative_lengths,
            max_length,
        )
        predictions[indices] = preds

    return predictions


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test-notes",
        type=int,
        default=0,
        metavar="N",
        help="Embed the first N notes from one token batch without writing an output file.",
    )
    parser.add_argument(
        "--run-after-smoke-test",
        action="store_true",
        help="Continue the resumable full run with the already loaded model after the smoke test.",
    )
    args = parser.parse_args()
    if args.smoke_test_notes < 0:
        parser.error("--smoke-test-notes cannot be negative")
    return args


def slice_token_batch(token_batch: dict[str, np.ndarray], note_count: int) -> dict[str, np.ndarray]:
    """Return the first ``note_count`` notes from a flattened token batch."""
    row_splits = token_batch["row_splits"]
    kept_note_count = min(note_count, len(row_splits) - 1)
    kept_row_splits = row_splits[: kept_note_count + 1].copy()
    kept_token_count = int(kept_row_splits[-1])
    return {
        "input_ids_flat": token_batch["input_ids_flat"][:kept_token_count].copy(),
        "row_splits": kept_row_splits,
    }


def load_embedding_model(device: str) -> tuple[torch.nn.Module, ModelRunner]:
    """Load Clinical ModernBERT in BF16 with FlashAttention-2."""
    if device != "cuda":
        raise RuntimeError("Embedding generation requires a CUDA GPU for FlashAttention-2.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The visible CUDA GPU does not support BF16; use an Ampere-or-newer GPU.")

    from transformers import AutoModel

    try:
        embedding_model = AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
    except ImportError as exc:
        raise RuntimeError(
            "FlashAttention-2 is unavailable. Install flash-attn in the GPU environment "
            "before running embedding generation."
        ) from exc

    embedding_model.eval()
    return embedding_model, ModelRunner(embedding_model, compile_model=COMPILE_MODEL)


def main() -> None:
    """Generate padding-free mean-pooled embeddings for all token batches."""
    args = parse_args()
    EMBED_PATH.mkdir(parents=True, exist_ok=True)

    token_files = discover_token_files(TOKEN_PATH)
    print(f"Found {len(token_files)} token batches in {TOKEN_PATH}.")

    smoke_test = args.smoke_test_notes > 0
    embedded_batches = discover_embedded_batches(EMBED_PATH)
    pending_token_files = {
        batch_idx: batch_path
        for batch_idx, batch_path in token_files.items()
        if batch_idx not in embedded_batches
    }
    completed_count = len(token_files) - len(pending_token_files)
    if completed_count:
        print(f"Skipping {completed_count} previously embedded batches in {EMBED_PATH}.")
    if not pending_token_files and not smoke_test:
        print("All token batches already have embeddings; nothing to do.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = device == "cuda"
    embedding_model, model_runner = load_embedding_model(device)
    hidden_size = embedding_model.config.hidden_size
    print(
        "Inference settings: BF16, FlashAttention-2, padding-free batches, "
        f"torch.compile={'on' if COMPILE_MODEL else 'off'}, max notes={MAX_BATCH_SIZE}, "
        f"collate workers={COLLATE_WORKERS}, zstd level={ZSTD_LEVEL}."
    )

    if smoke_test:
        smoke_batch_idx = next(iter(token_files))
        print(
            f"Smoke test: embedding at most {args.smoke_test_notes} notes from batch "
            f"{smoke_batch_idx}; no output file will be written."
        )
        smoke_start_time = time.time()
        smoke_token_batch = slice_token_batch(
            load_token_batch(token_files[smoke_batch_idx]),
            args.smoke_test_notes,
        )
        smoke_predictions = embed_token_batch(
            model_runner,
            smoke_token_batch,
            hidden_size,
            device,
            pin_memory,
            smoke_batch_idx,
        )
        if not torch.isfinite(smoke_predictions).all():
            raise RuntimeError("Smoke test produced non-finite embeddings.")
        print(
            f"Smoke test passed: shape={tuple(smoke_predictions.shape)}, "
            f"dtype={smoke_predictions.dtype}, elapsed={time.time() - smoke_start_time:.2f} seconds."
        )
        if not args.run_after_smoke_test:
            return
        if not pending_token_files:
            print("All token batches already have embeddings; nothing else to do.")
            return

    write_future: Future[None] | None = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedding-writer") as writer:
        for batch_idx, _, token_batch in prefetch_token_batches(pending_token_files):
            start_time = time.time()
            predictions = embed_token_batch(
                model_runner,
                token_batch,
                hidden_size,
                device,
                pin_memory,
                batch_idx,
            )

            # Bound queued output memory to one batch while overlapping the
            # previous batch's compression/write with current GPU inference.
            if write_future is not None:
                write_future.result()
            output_path = embedding_output_path(batch_idx)
            write_future = writer.submit(save_embedding_batch, output_path, predictions.numpy())

            elapsed = (time.time() - start_time) / 60
            print(f"Batch {batch_idx} embedded in {elapsed:.2f} minutes; saving asynchronously.")

        if write_future is not None:
            write_future.result()


if __name__ == "__main__":
    main()
