"""Text preprocessing and tokenization for clinical note embedding workflows."""

from __future__ import annotations

import io
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import zstandard as zstd
from tqdm.auto import tqdm

from config import DATA_PATH as _DATA_PATH_STR, SURV_PATH
from pipelines.preprocessing import profile_sources as ps

# Paths and constants
DATA_PATH = Path(_DATA_PATH_STR)

# PROGRESS_NOTES.parquet already includes discharge summaries merged in.
NOTE_TYPE_SOURCES = (
    ("PROGRESS_NOTES", "Clinician"),
    ("PATHOLOGY_NOTES", "Pathology"),
    ("IMAGING_NOTES", "Imaging"),
)

COL_METADATA = [
    ps.RPT_ID,
    ps.EVENT_DATE,
    ps.MRN,
    ps.PROC_DESC,
    ps.RPT_TYPE,
    ps.INP_RPT_TYPE,
    ps.PROVIDER_TYPE,
    ps.ENCOUNTER_TYPE_DESC,
]
COLUMNS_TO_SAVE = COL_METADATA + ["CLINICAL_TEXT", "NOTE_TYPE"]

BATCHED_DATA_PATH = DATA_PATH / "batched_datasets"
BATCHED_TOKEN_PATH = BATCHED_DATA_PATH / "batched_tokens"
BATCHED_TOKEN_FILES_PATH = BATCHED_TOKEN_PATH / "tokens"
BATCHED_METADATA_PATH = BATCHED_TOKEN_PATH / "metadata"

BATCH_SIZE = 100_000
TOKENIZER_BATCH_SIZE = 16_384
TOKENIZER_MODEL_NAME = "Simonlee711/Clinical_ModernBERT"
TOKEN_BATCH_SUFFIX = ".npy.zst"

# Rayon threads inside the Rust tokenizer. This is the pipeline's only source of
# parallelism: the tokenizer parallelizes across the independent sequences in one
# encode call, which saturates the available cores without the memory multiplier
# that process-level parallelism over shards would incur (one shard's token
# sequences are the single largest allocation in the stage). If this is ever
# combined with multiprocessing, pin it to 1 in the workers to avoid
# oversubscription.
TOKENIZER_THREADS = int(os.environ.get("TOKENIZER_THREADS", os.cpu_count() or 4))

# Both variables are read when the tokenizers native library initializes, so they
# must be set before `transformers` is imported in main() -- setting them inside
# main() would silently do nothing.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("RAYON_NUM_THREADS", str(TOKENIZER_THREADS))

# stderr is not a tty when this module runs under subprocess.run from
# notebooks/1_data/01_preprocessing.ipynb, which disables every bar; FORCE_PROGRESS re-enables them.
SHOW_LIVE_PROGRESS = os.environ.get("FORCE_PROGRESS", "") == "1" or sys.stderr.isatty()
TOKEN_FILE_PATTERN = re.compile(r"clinical_notes_tokenized_batch_(\d+)_tokens\.npy\.zst$")
METADATA_FILE_PATTERN = re.compile(r"clinical_notes_tokenized_batch_(\d+)_metadata\.parquet$")


def token_batch_path(batch_idx: int) -> Path:
    return BATCHED_TOKEN_FILES_PATH / f"clinical_notes_tokenized_batch_{batch_idx}_tokens{TOKEN_BATCH_SUFFIX}"


def metadata_batch_path(batch_idx: int) -> Path:
    return BATCHED_METADATA_PATH / f"clinical_notes_tokenized_batch_{batch_idx}_metadata.parquet"


def archive_stale_batch_files(valid_batch_count: int) -> int:
    """Move obsolete generated batches outside discovery directories."""
    stale_files: list[tuple[Path, str]] = []
    for directory, pattern, kind in (
        (BATCHED_TOKEN_FILES_PATH, TOKEN_FILE_PATTERN, "tokens"),
        (BATCHED_METADATA_PATH, METADATA_FILE_PATTERN, "metadata"),
    ):
        for path in directory.iterdir():
            match = pattern.fullmatch(path.name)
            if match and int(match.group(1)) >= valid_batch_count:
                stale_files.append((path, kind))

    if not stale_files:
        return 0

    archive_root = BATCHED_TOKEN_PATH / "stale_batches" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for path, kind in stale_files:
        destination_dir = archive_root / kind
        destination_dir.mkdir(parents=True, exist_ok=True)
        path.replace(destination_dir / path.name)
    print(f"Archived {len(stale_files)} stale batch files under {archive_root}", flush=True)
    return len(stale_files)


def clean_text(text: str) -> str:
    """Normalize whitespace and strip uncommon non-alphanumeric characters."""
    text = re.sub(r"\s\s+", " ", text)
    return re.sub(r"[^A-Za-z0-9 .,?!()/]+", " ", text)


def init_batch_storage() -> dict[str, list[Any]]:
    """Create empty column-oriented storage for a single text batch."""
    return {col: [] for col in COLUMNS_TO_SAVE}


def ensure_output_dirs() -> None:
    """Create output directories for token batches and metadata."""
    for path in (
        BATCHED_TOKEN_PATH,
        BATCHED_TOKEN_FILES_PATH,
        BATCHED_METADATA_PATH,
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_cohort_mrns() -> set[int]:
    """Load cohort MRNs into a set for fast membership checks."""
    cohort_df = pl.read_parquet(os.path.join(SURV_PATH, "cohort_df.parquet"), columns=["DFCI_MRN"])
    return set(
        cohort_df.select(pl.col("DFCI_MRN").cast(pl.Int64, strict=False))
        .drop_nulls()
        .get_column("DFCI_MRN")
        .to_list()
    )


def load_note_source_df(note_type: str, note_type_label: str, cohort_mrns: set[int]) -> pl.DataFrame:
    """Load one note-metadata parquet, filter to cohort MRNs, and normalize to
    COLUMNS_TO_SAVE. The compiled parquets are already RPT_ID-deduped across
    all four raw pulls, newest-wins, so no cross-snapshot dedup is needed here."""
    available_cols = [c for c in COL_METADATA if c != ps.MRN] + [ps.MRN, ps.RPT_TEXT]
    note_df = ps.load_note_metadata(note_type, columns=available_cols)
    present_cols = [c for c in available_cols if c in note_df.columns]
    note_df = note_df.select(present_cols).with_columns(
        pl.col(ps.MRN).cast(pl.Int64, strict=False)
    ).filter(pl.col(ps.MRN).is_in(cohort_mrns))

    for col in COL_METADATA:
        if col not in note_df.columns:
            note_df = note_df.with_columns(pl.lit(None).alias(col))

    return note_df.with_columns(
        pl.lit(note_type_label).alias("NOTE_TYPE"),
        pl.col(ps.RPT_TEXT)
        .fill_null("")
        .cast(pl.String)
        .str.replace_all(r"\s\s+", " ")
        .str.replace_all(r"[^A-Za-z0-9 .,?!()/]+", " ")
        .str.strip_chars()
        .alias("CLINICAL_TEXT"),
    ).select(COLUMNS_TO_SAVE)


def resolve_token_dtype(tokenizer: Any) -> np.dtype:
    """Choose a compact dtype that can hold the tokenizer vocabulary."""
    # len(tokenizer) includes added/special tokens; tokenizer.vocab_size does not,
    # so relying on vocab_size alone can silently pick a dtype too small to hold
    # every id the tokenizer can actually emit.
    vocab_size = len(tokenizer)
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.int32


def pack_token_sequences(token_sequences: list[list[int]], token_dtype: np.dtype) -> dict[str, np.ndarray]:
    """Pack variable-length token sequences into a compact flattened representation."""
    seq_lengths = np.fromiter((len(seq) for seq in token_sequences), dtype=np.int32, count=len(token_sequences))
    row_splits = np.empty(len(token_sequences) + 1, dtype=np.int64)
    row_splits[0] = 0
    np.cumsum(seq_lengths, dtype=np.int64, out=row_splits[1:])

    # Fill a single preallocated buffer in place rather than building one ndarray
    # per sequence and np.concatenate-ing them, which allocates the full output a
    # second time and doubles peak memory. The empty-corpus case falls out for
    # free: np.empty(0, ...) with a loop that never executes.
    input_ids_flat = np.empty(int(row_splits[-1]), dtype=token_dtype)
    for seq, start, end in zip(token_sequences, row_splits[:-1], row_splits[1:]):
        if end > start:
            input_ids_flat[start:end] = seq

    return {
        "input_ids_flat": input_ids_flat,
        "row_splits": row_splits,
        "storage_version": np.array([2], dtype=np.uint8),
    }


def tokenize_texts(
    tokenizer: Any,
    clinical_texts: pl.Series,
    *,
    batch_idx: int | None = None,
) -> dict[str, np.ndarray]:
    """Tokenize notes into ragged sequences to minimize disk and transfer overhead."""
    token_sequences: list[list[int]] = []
    token_dtype = resolve_token_dtype(tokenizer)
    n_texts = len(clinical_texts)

    description = "Tokenizer chunks" if batch_idx is None else f"Tokenizing batch {batch_idx}"
    with tqdm(
        total=n_texts,
        desc=description,
        unit="notes",
        disable=not SHOW_LIVE_PROGRESS,
        leave=True,
        mininterval=1.0,
    ) as progress:
        for start_idx in range(0, n_texts, TOKENIZER_BATCH_SIZE):
            # Slicing the Series is zero-copy; only this chunk is materialized into
            # Python strings, and it is freed at the next iteration instead of
            # holding the whole batch's strings alive for the duration of the loop.
            text_chunk = clinical_texts.slice(start_idx, TOKENIZER_BATCH_SIZE).to_list()
            tokenized_chunk = tokenizer(
                text_chunk,
                padding=False,
                truncation=True,
                return_attention_mask=False,
            )
            token_sequences.extend(tokenized_chunk["input_ids"])
            progress.update(len(text_chunk))

    return pack_token_sequences(token_sequences, token_dtype)


def write_npz_zst(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write named arrays as an uncompressed .npz payload, then zstd-compress the
    whole thing. Keeps np.load-compatible internal structure (unzip with
    zstandard, then np.load(io.BytesIO(...))) while getting zstd's better ratio
    than npz's per-member deflate."""
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    with open(path, "wb") as handle:
        handle.write(zstd.compress(buf.getvalue(), level=3))


def batch_is_complete(batch_idx: int, batch_df: pl.DataFrame) -> bool:
    """Return whether both saved outputs fully match the current input batch."""
    token_path = token_batch_path(batch_idx)
    metadata_path = metadata_batch_path(batch_idx)
    if not token_path.is_file() or not metadata_path.is_file():
        return False

    try:
        saved_metadata = pl.read_parquet(metadata_path)
        identity_columns = [ps.RPT_ID, ps.MRN, "NOTE_TYPE"]
        expected_identity = batch_df.select(identity_columns)
        if saved_metadata.height != batch_df.height:
            return False
        if not saved_metadata.select(identity_columns).equals(expected_identity):
            return False

        with open(token_path, "rb") as handle:
            raw = zstd.decompress(handle.read())
        with np.load(io.BytesIO(raw), allow_pickle=False) as token_batch:
            input_ids_flat = token_batch["input_ids_flat"]
            row_splits = token_batch["row_splits"]
            if row_splits.ndim != 1 or len(row_splits) != batch_df.height + 1:
                return False
            if int(row_splits[0]) != 0 or int(row_splits[-1]) != len(input_ids_flat):
                return False
    except (OSError, ValueError, KeyError, zstd.ZstdError, pl.exceptions.PolarsError):
        return False

    return True


def save_tokenized_batch(batch_df: pl.DataFrame, tokenizer: Any, batch_idx: int) -> None:
    """Tokenize a text batch and save token arrays plus metadata."""
    clinical_texts = batch_df.get_column("CLINICAL_TEXT").fill_null("").cast(pl.String)
    tokenized_dict = tokenize_texts(tokenizer, clinical_texts, batch_idx=batch_idx)
    write_npz_zst(
        token_batch_path(batch_idx),
        tokenized_dict,
    )

    metadata_df = batch_df.drop("CLINICAL_TEXT")
    metadata_df.write_parquet(metadata_batch_path(batch_idx))


def flush_batch(
    batch_df: pl.DataFrame,
    tokenizer: Any,
    batch_idx: int,
) -> None:
    """Save one full batch of cleaned notes as token and metadata outputs."""
    save_tokenized_batch(batch_df, tokenizer, batch_idx)


def main() -> None:
    """Extract clinical text, batch it, and tokenize it for embedding generation."""
    from transformers import AutoTokenizer

    ensure_output_dirs()

    cohort_mrns = load_cohort_mrns()
    # use_fast=True makes a fallback to the ~20-100x slower Python tokenizer loud
    # instead of silent (from_pretrained falls back quietly on a tokenizers
    # version mismatch or a missing tokenizer.json), and the fallback would also
    # make TOKENIZERS_PARALLELISM/RAYON_NUM_THREADS above inert.
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("Fast (Rust) tokenizer required; got the slow Python fallback.")

    batch_count = 0
    kept_note_count = 0
    resumed_batch_count = 0

    for source_idx, (note_type, note_type_label) in enumerate(NOTE_TYPE_SOURCES, start=1):
        print(
            f"[{source_idx}/{len(NOTE_TYPE_SOURCES)}] "
            f"Loading and cleaning {note_type_label} notes...",
            flush=True,
        )
        source_notes = load_note_source_df(note_type, note_type_label, cohort_mrns)
        print(f"{note_type_label}: {source_notes.height:,} cohort notes", flush=True)

        batch_starts = range(0, source_notes.height, BATCH_SIZE)
        source_batch_count = max(1, (source_notes.height + BATCH_SIZE - 1) // BATCH_SIZE)
        for source_batch_idx, start_idx in enumerate(batch_starts, start=1):
            batch_df = source_notes.slice(start_idx, BATCH_SIZE)
            if batch_is_complete(batch_count, batch_df):
                resumed_batch_count += 1
                print(
                    f"  [{source_batch_idx}/{source_batch_count}] Resumed batch {batch_count} "
                    f"({batch_df.height:,} notes)",
                    flush=True,
                )
            else:
                print(
                    f"  [{source_batch_idx}/{source_batch_count}] Tokenizing batch {batch_count} "
                    f"({batch_df.height:,} notes)",
                    flush=True,
                )
                flush_batch(batch_df, tokenizer, batch_count)
            kept_note_count += batch_df.height
            batch_count += 1

    print(f"Saved {kept_note_count} cohort notes across {batch_count} batches.")
    print(f"Resumed from {resumed_batch_count} previously completed batches.")
    archive_stale_batch_files(batch_count)


if __name__ == "__main__":
    main()
