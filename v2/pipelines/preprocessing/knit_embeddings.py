"""Knit batchwise note embeddings and metadata into full analysis-ready files."""

from __future__ import annotations

import io
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import zstandard as zstd
from tqdm.auto import tqdm

from config import DATA_PATH as _DATA_PATH_STR, SURV_PATH

# Paths and constants
# This step typically runs back on the main cluster after batch embeddings have
# been copied from the GCP embedding workspace into `EMBEDS_PATH`.
DATA_PATH = Path(_DATA_PATH_STR)
BATCHED_DATA_PATH = Path(os.environ.get("BATCHED_DATA_PATH", str(DATA_PATH / "batched_datasets")))
META_PATH = Path(os.environ.get("META_PATH", str(BATCHED_DATA_PATH / "batched_tokens" / "metadata")))
EMBEDS_PATH = Path(os.environ.get("EMBEDS_PATH", str(BATCHED_DATA_PATH / "embeddings")))
PROC_PATH = Path(os.environ.get("PROC_PATH", str(BATCHED_DATA_PATH / "processed_datasets")))
COHORT_FILE = Path(
    os.environ.get(
        "COHORT_FILE",
        str(Path(SURV_PATH) / "cohort_df.parquet"),
    )
)

METADATA_FILE_RE = re.compile(r"clinical_notes_tokenized_batch_(\d+)_metadata\.parquet$")
EMBED_FILE_RE = re.compile(r"clinical_notes_embeddings_batch_(\d+)\.npy\.zst$")


def discover_batch_indices(path: Path, pattern: re.Pattern[str]) -> set[int]:
    """Return all batch indices matching a filename pattern in a directory."""
    batch_indices = set()

    for file_path in path.iterdir():
        if not file_path.is_file():
            continue

        match = pattern.fullmatch(file_path.name)
        if match:
            batch_indices.add(int(match.group(1)))

    return batch_indices


def load_batch_metadata(batch_idx: int) -> pd.DataFrame:
    """Load one metadata batch from disk."""
    metadata_df = pd.read_parquet(META_PATH / f"clinical_notes_tokenized_batch_{batch_idx}_metadata.parquet")
    metadata_df["SUB_BATCH_FILE_ID"] = f"batch_{batch_idx}"
    metadata_df["WITHIN_SUB_BATCH_INDEX"] = np.arange(len(metadata_df))
    return metadata_df


def load_batch_embeddings(batch_idx: int) -> np.ndarray:
    """Load one zstd-compressed embeddings batch from disk."""
    with open(EMBEDS_PATH / f"clinical_notes_embeddings_batch_{batch_idx}.npy.zst", "rb") as handle:
        raw = zstd.decompress(handle.read())
    return np.load(io.BytesIO(raw))


def parse_note_datetimes(metadata_df: pd.DataFrame) -> pd.Series:
    """Parse note timestamps from EVENT_DATE, which is tz-aware UTC in the
    PROFILE_DATA notes parquets (the only tz-aware column in the ecosystem);
    convert to naive to match every other date column."""
    return pd.to_datetime(metadata_df["EVENT_DATE"], errors="coerce", utc=True).dt.tz_localize(None)


def main() -> None:
    """Merge per-batch metadata and ModernBERT embeddings into full cohort files."""
    PROC_PATH.mkdir(parents=True, exist_ok=True)

    cohort_df = pd.read_parquet(COHORT_FILE)
    cohort_df["DFCI_MRN"] = pd.to_numeric(cohort_df["DFCI_MRN"], errors="coerce")
    cohort_df["FIRST_TREATMENT_START_DT"] = pd.to_datetime(
        cohort_df["first_treatment_date"],
        errors="coerce",
    )
    mrn_tstart_dict = dict(
        zip(
            cohort_df["DFCI_MRN"].dropna().astype(np.int64),
            cohort_df["FIRST_TREATMENT_START_DT"],
        )
    )

    mrn_seqdate_dict: dict[int, pd.Timestamp] = {}
    if "sequencing_date" in cohort_df.columns:
        cohort_df["SEQUENCING_DT"] = pd.to_datetime(cohort_df["sequencing_date"], errors="coerce")
        mrn_seqdate_dict = dict(
            zip(
                cohort_df["DFCI_MRN"].dropna().astype(np.int64),
                cohort_df["SEQUENCING_DT"],
            )
        )

    metadata_batch_indices = discover_batch_indices(META_PATH, METADATA_FILE_RE)
    embedding_batch_indices = discover_batch_indices(EMBEDS_PATH, EMBED_FILE_RE)

    if not metadata_batch_indices:
        raise FileNotFoundError(f"No metadata batch files found in {META_PATH}")
    if not embedding_batch_indices:
        raise FileNotFoundError(f"No embedding batch files found in {EMBEDS_PATH}")

    missing_embeddings = sorted(metadata_batch_indices - embedding_batch_indices)
    if missing_embeddings:
        raise FileNotFoundError(
            "Missing embedding batches for metadata batch indices: "
            + ", ".join(str(batch_idx) for batch_idx in missing_embeddings)
        )

    batch_indices = sorted(metadata_batch_indices & embedding_batch_indices)
    print(f"Found {len(batch_indices)} completed batch indices.")

    metadata_list = []
    embedding_array_list = []
    for batch_idx in tqdm(batch_indices, desc="Loading metadata and embeddings"):
        cur_metadata = load_batch_metadata(batch_idx)
        cur_embeddings = load_batch_embeddings(batch_idx)

        if len(cur_metadata) != cur_embeddings.shape[0]:
            raise ValueError(
                f"Batch {batch_idx} has {len(cur_metadata)} metadata rows but "
                f"{cur_embeddings.shape[0]} embeddings."
            )

        metadata_list.append(cur_metadata)
        embedding_array_list.append(cur_embeddings)

    metadata_df = pd.concat(metadata_list, ignore_index=True)
    embeddings = np.concatenate(embedding_array_list, axis=0)

    metadata_df["DFCI_MRN"] = pd.to_numeric(metadata_df["DFCI_MRN"], errors="coerce")
    metadata_df["NOTE_DATETIME"] = parse_note_datetimes(metadata_df)
    valid_note_mask = metadata_df["NOTE_DATETIME"].notna().to_numpy()
    dropped_missing_note_dt = int((~valid_note_mask).sum())

    if dropped_missing_note_dt:
        metadata_df = metadata_df.loc[valid_note_mask].reset_index(drop=True)
        embeddings = embeddings[valid_note_mask]

    metadata_df["EMBEDDING_INDEX"] = np.arange(len(metadata_df))
    metadata_df["FIRST_TREATMENT_START_DT"] = metadata_df["DFCI_MRN"].map(mrn_tstart_dict)
    metadata_df["NOTE_TIME_REL_TREATMENT"] = (
        metadata_df["NOTE_DATETIME"] - metadata_df["FIRST_TREATMENT_START_DT"]
    ).dt.days
    # Alias of the treatment-anchor column so pre-anchor-split callers keep working.
    metadata_df["NOTE_TIME_REL_FIRST_TREATMENT_START"] = metadata_df["NOTE_TIME_REL_TREATMENT"]

    metadata_df["SEQUENCING_DT"] = metadata_df["DFCI_MRN"].map(mrn_seqdate_dict)
    metadata_df["NOTE_TIME_REL_SEQUENCING"] = (
        metadata_df["NOTE_DATETIME"] - metadata_df["SEQUENCING_DT"]
    ).dt.days

    metadata_file = PROC_PATH / "full_clinical_notes_embeddings_metadata.parquet"
    embeds_file = PROC_PATH / "full_clinical_notes_embeddings_as_array.npy.zst"

    metadata_df.to_parquet(metadata_file, index=False)
    embeddings_bytes = io.BytesIO()
    np.save(embeddings_bytes, embeddings.astype(np.float16))
    with open(embeds_file, 'wb') as f:
        f.write(zstd.compress(embeddings_bytes.getvalue(), level=15))

    missing_tstart = int(metadata_df["FIRST_TREATMENT_START_DT"].isna().sum())
    missing_seqdate = int(metadata_df["SEQUENCING_DT"].isna().sum())

    print(f"Saved merged metadata to {metadata_file}")
    print(f"Saved merged embeddings to {embeds_file}")
    print(f"Rows dropped for missing NOTE_DATETIME: {dropped_missing_note_dt}")
    print(f"Rows with missing FIRST_TREATMENT_START_DT: {missing_tstart}")
    print(f"Rows with missing SEQUENCING_DT: {missing_seqdate}")


if __name__ == "__main__":
    main()
