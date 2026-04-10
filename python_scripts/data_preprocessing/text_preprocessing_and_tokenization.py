"""Text preprocessing and tokenization for clinical note embedding workflows."""

from __future__ import annotations

import json
import os
import re
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# Paths and constants
DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/")
RAW_TEXT_ROOT = Path("/data/gusev/PROFILE/CLINICAL/OncDRS")
RAW_TEXT_DIRECTORIES = (
    "CLINICAL_TEXTS_2025_11",
    "CLINICAL_TEXTS_2025_03",
    "CLINICAL_TEXTS_2024_03",
)
VTE_DATA_PATH = Path("/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/")

COL_METADATA = [
    "PROC_DESC_STR",
    "PROVIDER_CRED_STR",
    "EVENT_DATE",
    "id",
    "DFCI_MRN",
    "ENCOUNTER_TYPE_DESC_STR",
    "RPT_DATE",
    "RPT_TYPE",
    "SOURCE_STR",
    "PROVIDER_CRED",
]
COLUMNS_TO_SAVE = COL_METADATA + ["CLINICAL_TEXT", "NOTE_TYPE"]

FILE_PATTERNS = (
    ("*Imaging-*.json", "Imaging"),
    ("*Path-*.json", "Pathology"),
    ("*ProgNote-*.json", "Clinician"),
)

BATCHED_DATA_PATH = DATA_PATH / "batched_datasets"
BATCHED_TEXT_PATH = BATCHED_DATA_PATH / "batched_text"
BATCHED_TOKEN_PATH = BATCHED_DATA_PATH / "batched_tokens"
BATCHED_TOKEN_FILES_PATH = BATCHED_TOKEN_PATH / "tokens"
BATCHED_METADATA_PATH = BATCHED_TOKEN_PATH / "metadata"

BATCH_SIZE = 50_000
TOKENIZER_BATCH_SIZE = 2_048
TOKENIZER_MODEL_NAME = "yikuan8/Clinical-Longformer"
TOKEN_BATCH_SUFFIX = ".npz"


def resolve_text_batch_compression() -> tuple[str | dict[str, Any], str]:
    """Choose the best available on-disk compression for intermediate text batches."""
    if importlib.util.find_spec("zstandard") is not None:
        return {"method": "zstd", "level": 3}, ".jsonl.zst"

    return "gzip", ".jsonl.gz"


TEXT_BATCH_COMPRESSION, TEXT_BATCH_SUFFIX = resolve_text_batch_compression()


def clean_text(text: str) -> str:
    """Normalize whitespace and strip uncommon non-alphanumeric characters."""
    text = re.sub(r"\s\s+", " ", text)
    return re.sub(r"[^A-Za-z0-9 .,?!()/]+", " ", text)


def deduplicate_texts(text_list: list[str]) -> list[str]:
    """Remove duplicate text fragments while preserving original order."""
    seen = set()
    deduped = []

    for text in text_list:
        if text not in seen:
            seen.add(text)
            deduped.append(text)

    return deduped


def init_batch_storage() -> dict[str, list[Any]]:
    """Create empty column-oriented storage for a single text batch."""
    return {col: [] for col in COLUMNS_TO_SAVE}


def ensure_output_dirs() -> None:
    """Create output directories for text batches, token batches, and metadata."""
    for path in (
        BATCHED_TEXT_PATH,
        BATCHED_TOKEN_PATH,
        BATCHED_TOKEN_FILES_PATH,
        BATCHED_METADATA_PATH,
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_raw_text_dirs() -> list[Path]:
    """
    Resolve the raw note directories to scan.

    The default behavior uses the full union of the 2024_03, 2025_03, and
    2025_11 snapshots. A comma-separated `CLINICAL_TEXT_DIRS` environment
    variable can override this list if needed.
    """
    env_override = os.environ.get("CLINICAL_TEXT_DIRS")
    if env_override:
        return [Path(path.strip()) for path in env_override.split(",") if path.strip()]

    return [RAW_TEXT_ROOT / directory_name for directory_name in RAW_TEXT_DIRECTORIES]


def discover_raw_text_files(raw_text_dirs: list[Path]) -> list[tuple[Path, str]]:
    """Collect note files across all configured raw-text directories."""
    files_to_process: list[tuple[Path, str]] = []
    missing_dirs = [str(path) for path in raw_text_dirs if not path.exists()]

    for raw_text_dir in raw_text_dirs:
        if not raw_text_dir.exists():
            continue

        for pattern, source in FILE_PATTERNS:
            for file_path in sorted(raw_text_dir.glob(pattern)):
                files_to_process.append((file_path, source))

    if not files_to_process:
        raise FileNotFoundError(
            "No clinical note files were found in the configured raw text directories: "
            + ", ".join(str(path) for path in raw_text_dirs)
        )

    if missing_dirs:
        print("Warning: missing raw text directories:", ", ".join(missing_dirs))

    return files_to_process


def load_vte_mrns() -> set[int]:
    """Load cohort MRNs into a set for fast membership checks."""
    vte_df = pd.read_csv(VTE_DATA_PATH / "follow_up_vte_df_cohort.csv", usecols=["DFCI_MRN"])
    mrns = pd.to_numeric(vte_df["DFCI_MRN"], errors="coerce").dropna().astype(np.int64)
    return set(mrns.tolist())


def parse_mrn(mrn_value: Any) -> int | None:
    """Convert note MRN values to ints when possible."""
    if mrn_value is None:
        return None

    try:
        if pd.isna(mrn_value):
            return None
    except TypeError:
        pass

    try:
        return int(float(mrn_value))
    except (TypeError, ValueError):
        return None


def is_text_value(value: Any) -> bool:
    """Return True when a raw note field should contribute to the concatenated text."""
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    try:
        return not pd.isna(value)
    except TypeError:
        return True


def build_note_text(note: dict[str, Any]) -> str:
    """Extract, deduplicate, and clean all text fields from a single note."""
    text_entries = [
        str(value)
        for key, value in note.items()
        if "TEXT" in key and is_text_value(value)
    ]

    return clean_text(" ".join(deduplicate_texts(text_entries))).strip()


def build_note_dedupe_key(note: dict[str, Any], source: str) -> str:
    """
    Build a stable deduplication key for notes across overlapping raw snapshots.

    Newer directories are processed first, so duplicate note IDs from older
    snapshots are skipped automatically.
    """
    note_id = note.get("id")
    if note_id is not None and str(note_id) != "" and not pd.isna(note_id):
        return f"id::{note_id}"

    fallback_fields = (
        source,
        note.get("DFCI_MRN", ""),
        note.get("EVENT_DATE", ""),
        note.get("RPT_DATE", ""),
        note.get("RPT_TYPE", ""),
        note.get("PROC_DESC_STR", ""),
    )
    return "fallback::" + "||".join(str(value) for value in fallback_fields)


def append_note_to_batch(
    batch_storage: dict[str, list[Any]],
    note: dict[str, Any],
    source: str,
) -> None:
    """Append note metadata and cleaned text to the in-memory batch store."""
    for col in COL_METADATA:
        batch_storage[col].append(note.get(col, np.nan))

    batch_storage["NOTE_TYPE"].append(source)
    batch_storage["CLINICAL_TEXT"].append(build_note_text(note))


def save_text_batch(batch_df: pd.DataFrame, batch_idx: int) -> Path:
    """Persist a compressed text batch as newline-delimited JSON."""
    batch_path = BATCHED_TEXT_PATH / f"VTE_notes_with_full_metadata_batch_{batch_idx}{TEXT_BATCH_SUFFIX}"
    batch_df.to_json(
        batch_path,
        orient="records",
        lines=True,
        compression=TEXT_BATCH_COMPRESSION,
    )
    return batch_path


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


def save_tokenized_batch(batch_df: pd.DataFrame, tokenizer: Any, batch_idx: int) -> None:
    """Tokenize a text batch and save token arrays plus metadata."""
    clinical_texts = batch_df["CLINICAL_TEXT"].fillna("").astype(str).tolist()
    tokenized_dict = tokenize_texts(tokenizer, clinical_texts)
    np.savez_compressed(
        BATCHED_TOKEN_FILES_PATH / f"VTE_notes_tokenized_batch_{batch_idx}_tokens{TOKEN_BATCH_SUFFIX}",
        **tokenized_dict,
    )

    metadata_dict = batch_df.drop(columns=["CLINICAL_TEXT"]).to_dict(orient="list")
    with open(BATCHED_METADATA_PATH / f"VTE_notes_tokenized_batch_{batch_idx}_metadata.json", "w") as handle:
        json.dump(metadata_dict, handle)


def flush_batch(
    batch_storage: dict[str, list[Any]],
    tokenizer: Any,
    batch_idx: int,
) -> dict[str, list[Any]]:
    """Save one full batch of cleaned notes and its tokenized outputs."""
    batch_df = pd.DataFrame(batch_storage, columns=COLUMNS_TO_SAVE)
    save_text_batch(batch_df, batch_idx)
    save_tokenized_batch(batch_df, tokenizer, batch_idx)
    return init_batch_storage()


def iter_docs(file_path: Path) -> list[dict[str, Any]]:
    """Load note docs from a raw OncDRS JSON export."""
    with open(file_path, "r") as handle:
        payload = json.load(handle)
    return payload.get("response", {}).get("docs", [])


def main() -> None:
    """Extract clinical text, batch it, and tokenize it for embedding generation."""
    from transformers import AutoTokenizer

    ensure_output_dirs()

    raw_text_dirs = resolve_raw_text_dirs()
    files_to_process = discover_raw_text_files(raw_text_dirs)
    vte_mrns = load_vte_mrns()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)

    batch_storage = init_batch_storage()
    seen_note_keys: set[str] = set()
    batch_count = 0
    kept_note_count = 0
    skipped_duplicate_count = 0

    for file_path, source in tqdm(files_to_process, desc="Reading raw note files"):
        for note in iter_docs(file_path):
            mrn = parse_mrn(note.get("DFCI_MRN"))
            if mrn is None or mrn not in vte_mrns:
                continue

            note_key = build_note_dedupe_key(note, source)
            if note_key in seen_note_keys:
                skipped_duplicate_count += 1
                continue
            seen_note_keys.add(note_key)

            append_note_to_batch(batch_storage, note, source)
            kept_note_count += 1

            if len(batch_storage["CLINICAL_TEXT"]) >= BATCH_SIZE:
                batch_storage = flush_batch(batch_storage, tokenizer, batch_count)
                batch_count += 1

    if batch_storage["CLINICAL_TEXT"]:
        flush_batch(batch_storage, tokenizer, batch_count)
        batch_count += 1

    print(f"Processed {len(files_to_process)} raw note files across {len(raw_text_dirs)} directories.")
    print(f"Saved {kept_note_count} cohort notes across {batch_count} batches.")
    print(f"Skipped {skipped_duplicate_count} duplicate notes across overlapping snapshots.")


if __name__ == "__main__":
    main()
