"""Text preprocessing and tokenization for clinical note embedding workflows."""

from __future__ import annotations

import io
import os
import re
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

BATCH_SIZE = 50_000
TOKENIZER_BATCH_SIZE = 2_048
TOKENIZER_MODEL_NAME = "Simonlee711/Clinical_ModernBERT"
TOKEN_BATCH_SUFFIX = ".npy.zst"


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
    note_df = ps.load_note_metadata(note_type)
    present_cols = [c for c in available_cols if c in note_df.columns]
    note_df = note_df.select(present_cols).with_columns(
        pl.col(ps.MRN).cast(pl.Int64, strict=False)
    ).filter(pl.col(ps.MRN).is_in(cohort_mrns))

    for col in COL_METADATA:
        if col not in note_df.columns:
            note_df = note_df.with_columns(pl.lit(None).alias(col))

    return note_df.with_columns(
        pl.lit(note_type_label).alias("NOTE_TYPE"),
        pl.col(ps.RPT_TEXT).fill_null("").cast(pl.String).map_elements(clean_text, return_dtype=pl.String).str.strip_chars().alias("CLINICAL_TEXT"),
    ).select(COLUMNS_TO_SAVE)


def resolve_token_dtype(tokenizer: Any) -> np.dtype:
    """Choose a compact dtype that can hold the tokenizer vocabulary."""
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None and int(vocab_size) <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.int32


def pack_token_sequences(token_sequences: list[list[int]], token_dtype: np.dtype) -> dict[str, np.ndarray]:
    """Pack variable-length token sequences into a compact flattened representation."""
    seq_lengths = np.fromiter((len(seq) for seq in token_sequences), dtype=np.int32, count=len(token_sequences))
    row_splits = np.empty(len(token_sequences) + 1, dtype=np.int64)
    row_splits[0] = 0
    row_splits[1:] = np.cumsum(seq_lengths, dtype=np.int64)

    if row_splits[-1] == 0:
        input_ids_flat = np.empty(0, dtype=token_dtype)
    else:
        input_ids_flat = np.concatenate([np.asarray(seq, dtype=token_dtype) for seq in token_sequences])

    return {
        "input_ids_flat": input_ids_flat,
        "row_splits": row_splits,
        "storage_version": np.array([2], dtype=np.uint8),
    }


def tokenize_texts(tokenizer: Any, clinical_texts: list[str]) -> dict[str, np.ndarray]:
    """Tokenize notes into ragged sequences to minimize disk and transfer overhead."""
    token_sequences: list[list[int]] = []
    token_dtype = resolve_token_dtype(tokenizer)

    for start_idx in range(0, len(clinical_texts), TOKENIZER_BATCH_SIZE):
        text_chunk = clinical_texts[start_idx:start_idx + TOKENIZER_BATCH_SIZE]
        tokenized_chunk = tokenizer(
            text_chunk,
            padding=False,
            truncation=True,
            return_attention_mask=False,
        )
        token_sequences.extend(tokenized_chunk["input_ids"])

    return pack_token_sequences(token_sequences, token_dtype)


def write_npz_zst(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write named arrays as an uncompressed .npz payload, then zstd-compress the
    whole thing. Keeps np.load-compatible internal structure (unzip with
    zstandard, then np.load(io.BytesIO(...))) while getting zstd's better ratio
    than npz's per-member deflate."""
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    with open(path, "wb") as handle:
        handle.write(zstd.compress(buf.getvalue(), level=15))


def save_tokenized_batch(batch_df: pl.DataFrame, tokenizer: Any, batch_idx: int) -> None:
    """Tokenize a text batch and save token arrays plus metadata."""
    clinical_texts = batch_df.get_column("CLINICAL_TEXT").fill_null("").cast(pl.String).to_list()
    tokenized_dict = tokenize_texts(tokenizer, clinical_texts)
    write_npz_zst(
        BATCHED_TOKEN_FILES_PATH / f"clinical_notes_tokenized_batch_{batch_idx}_tokens{TOKEN_BATCH_SUFFIX}",
        tokenized_dict,
    )

    metadata_df = batch_df.drop("CLINICAL_TEXT")
    metadata_df.write_parquet(BATCHED_METADATA_PATH / f"clinical_notes_tokenized_batch_{batch_idx}_metadata.parquet")


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
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)

    all_notes = pl.concat(
        [
            load_note_source_df(note_type, note_type_label, cohort_mrns)
            for note_type, note_type_label in NOTE_TYPE_SOURCES
        ],
        how="vertical_relaxed",
    )

    batch_count = 0
    kept_note_count = 0

    for start_idx in tqdm(range(0, all_notes.height, BATCH_SIZE), desc="Tokenizing note batches"):
        batch_df = all_notes.slice(start_idx, BATCH_SIZE)
        flush_batch(batch_df, tokenizer, batch_count)
        kept_note_count += batch_df.height
        batch_count += 1

    print(f"Saved {kept_note_count} cohort notes across {batch_count} batches.")


if __name__ == "__main__":
    main()
