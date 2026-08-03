"""Knit batchwise note embeddings and metadata into full analysis-ready files."""

from __future__ import annotations

import gzip
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from config import DATA_PATH as _DATA_PATH_STR, INTAE_DATA_PATH

# Paths and constants
# This step typically runs back on the main cluster after batch embeddings have
# been copied from the GCP embedding workspace into `EMBEDS_PATH`.
DATA_PATH = Path(_DATA_PATH_STR)
BATCHED_DATA_PATH = Path(os.environ.get("BATCHED_DATA_PATH", str(DATA_PATH / "batched_datasets")))
META_PATH = Path(os.environ.get("META_PATH", str(BATCHED_DATA_PATH / "batched_tokens" / "metadata")))
EMBEDS_PATH = Path(os.environ.get("EMBEDS_PATH", str(BATCHED_DATA_PATH / "embeddings")))
PROC_PATH = Path(os.environ.get("PROC_PATH", str(BATCHED_DATA_PATH / "processed_datasets")))
SURVIVAL_FILE = Path(
    os.environ.get(
        "SURVIVAL_FILE",
        str(Path(INTAE_DATA_PATH) / "follow_up_vte_df_cohort.csv"),
    )
)

METADATA_FILE_RE = re.compile(r"VTE_notes_tokenized_batch_(\d+)_metadata\.json$")
EMBED_FILE_RE = re.compile(r"VTE_notes_embeddings_batch_(\d+)\.pt$")


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
    with open(META_PATH / f"VTE_notes_tokenized_batch_{batch_idx}_metadata.json") as handle:
        metadata_dict = json.load(handle)

    metadata_df = pd.DataFrame(metadata_dict)
    metadata_df["SUB_BATCH_FILE_ID"] = f"batch_{batch_idx}"
    metadata_df["WITHIN_SUB_BATCH_INDEX"] = np.arange(len(metadata_df))
    return metadata_df


def parse_note_datetimes(metadata_df: pd.DataFrame) -> pd.Series:
    """Parse note timestamps robustly across historical raw-note snapshots."""
    event_dt = pd.to_datetime(metadata_df["EVENT_DATE"], errors="coerce", utc=True).dt.tz_localize(None)
    rpt_dt = pd.to_datetime(metadata_df["RPT_DATE"], errors="coerce", utc=True).dt.tz_localize(None)
    return event_dt.fillna(rpt_dt)


def main() -> None:
    """Merge per-batch metadata and Longformer embeddings into full cohort files."""
    PROC_PATH.mkdir(parents=True, exist_ok=True)

    survival_df = pd.read_csv(SURVIVAL_FILE)
    survival_df["DFCI_MRN"] = pd.to_numeric(survival_df["DFCI_MRN"], errors="coerce")
    survival_df["FIRST_TREATMENT_START_DT"] = pd.to_datetime(
        survival_df["first_treatment_date"],
        errors="coerce",
    )
    mrn_tstart_dict = dict(
        zip(
            survival_df["DFCI_MRN"].dropna().astype(np.int64),
            survival_df["FIRST_TREATMENT_START_DT"],
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
    embedding_tensor_list = []
    for batch_idx in tqdm(batch_indices, desc="Loading metadata and embeddings"):
        cur_metadata = load_batch_metadata(batch_idx)
        cur_embeddings = torch.load(
            EMBEDS_PATH / f"VTE_notes_embeddings_batch_{batch_idx}.pt",
            map_location="cpu",
        )

        if len(cur_metadata) != cur_embeddings.shape[0]:
            raise ValueError(
                f"Batch {batch_idx} has {len(cur_metadata)} metadata rows but "
                f"{cur_embeddings.shape[0]} embeddings."
            )

        metadata_list.append(cur_metadata)
        embedding_tensor_list.append(cur_embeddings)

    metadata_df = pd.concat(metadata_list, ignore_index=True)
    embeddings = torch.cat(embedding_tensor_list, dim=0).numpy()

    metadata_df["DFCI_MRN"] = pd.to_numeric(metadata_df["DFCI_MRN"], errors="coerce")
    metadata_df["NOTE_DATETIME"] = parse_note_datetimes(metadata_df)
    valid_note_mask = metadata_df["NOTE_DATETIME"].notna().to_numpy()
    dropped_missing_note_dt = int((~valid_note_mask).sum())

    if dropped_missing_note_dt:
        metadata_df = metadata_df.loc[valid_note_mask].reset_index(drop=True)
        embeddings = embeddings[valid_note_mask]

    metadata_df["EMBEDDING_INDEX"] = np.arange(len(metadata_df))
    metadata_df["FIRST_TREATMENT_START_DT"] = metadata_df["DFCI_MRN"].map(mrn_tstart_dict)
    metadata_df["NOTE_TIME_REL_FIRST_TREATMENT_START"] = (
        metadata_df["NOTE_DATETIME"] - metadata_df["FIRST_TREATMENT_START_DT"]
    ).dt.days

    metadata_file = PROC_PATH / "full_VTE_embeddings_metadata.csv.gz"
    embeds_file = PROC_PATH / "full_VTE_embeddings_as_array.npy.gz"

    metadata_df.to_csv(metadata_file, index=False)
    with gzip.open(embeds_file, 'wb') as f:
        np.save(f, embeddings)

    missing_tstart = int(metadata_df["FIRST_TREATMENT_START_DT"].isna().sum())

    print(f"Saved merged metadata to {metadata_file}")
    print(f"Saved merged embeddings to {embeds_file}")
    print(f"Rows dropped for missing NOTE_DATETIME: {dropped_missing_note_dt}")
    print(f"Rows with missing FIRST_TREATMENT_START_DT: {missing_tstart}")


if __name__ == "__main__":
    main()
