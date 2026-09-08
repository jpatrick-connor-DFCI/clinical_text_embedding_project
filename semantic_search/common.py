"""Paths, constants and small loaders shared by the three semantic_search stages.

Layout under SEMANTIC_SEARCH_PATH:

    features/{space}_{window}.parquet          stage 1  DFCI_MRN + EMBEDDING_*
    clusters/{space}_{window}_labels.parquet   stage 2  DFCI_MRN, cluster
    clusters/{space}_{window}_coords.parquet   stage 2  DFCI_MRN, dim1, dim2
    clusters/{space}_{window}_meta.json        stage 2  k, seed, silhouette, ...
    results/*.csv                              stages 1-3
    figures/*.png                              stage 3
"""

from __future__ import annotations

import json
import os

import polars as pl

from config import SEMANTIC_SEARCH_PATH

# NOTE_TYPE values as they appear in the knitted embedding metadata.
NOTE_TYPES = ["Clinician", "Imaging", "Pathology"]

# Note-selection windows.
#   alltime       every note a patient has, no anchor -- a cross-sectional
#                 descriptor of the whole documented history.
#   pretreatment  notes strictly before first_treatment_date, matching the
#                 production survival arm (continuous_window=False,
#                 max_note_window=0).  Leak-free against tt_death.
WINDOWS = ["alltime", "pretreatment"]

# Feature spaces.  The single-type spaces and `merged` are one embedding
# dimension wide; `concat` is three.
SINGLE_TYPE_SPACES = ["clinician", "imaging", "pathology"]
SPACES = SINGLE_TYPE_SPACES + ["concat", "merged"]

# NOTE_TYPE -> space name for the single-type spaces.
SPACE_FOR_NOTE_TYPE = {nt: nt.lower() for nt in NOTE_TYPES}

# The literal NOTE_TYPE stamped over every row for the merged pooling pass.
MERGED_NOTE_TYPE = "All"

FEATURES_DIR = os.path.join(SEMANTIC_SEARCH_PATH, "features")
CLUSTERS_DIR = os.path.join(SEMANTIC_SEARCH_PATH, "clusters")
RESULTS_DIR = os.path.join(SEMANTIC_SEARCH_PATH, "results")
FIGURES_DIR = os.path.join(SEMANTIC_SEARCH_PATH, "figures")

NOTE_TIMING_COL = "NOTE_TIME_REL_FIRST_TREATMENT_START"
PATIENT_KEY = "DFCI_MRN"


def ensure_dirs() -> None:
    """Create the output subdirectories.  Idempotent; safe to call at import
    time of a stage module."""
    for path in (FEATURES_DIR, CLUSTERS_DIR, RESULTS_DIR, FIGURES_DIR):
        os.makedirs(path, exist_ok=True)


def _validate(space: str | None = None, window: str | None = None) -> None:
    if space is not None and space not in SPACES:
        raise ValueError(f"Unknown space {space!r}. Must be one of {SPACES}.")
    if window is not None and window not in WINDOWS:
        raise ValueError(f"Unknown window {window!r}. Must be one of {WINDOWS}.")


def feature_path(space: str, window: str) -> str:
    _validate(space, window)
    return os.path.join(FEATURES_DIR, f"{space}_{window}.parquet")


def labels_path(space: str, window: str) -> str:
    _validate(space, window)
    return os.path.join(CLUSTERS_DIR, f"{space}_{window}_labels.parquet")


def coords_path(space: str, window: str) -> str:
    _validate(space, window)
    return os.path.join(CLUSTERS_DIR, f"{space}_{window}_coords.parquet")


def cluster_meta_path(space: str, window: str) -> str:
    _validate(space, window)
    return os.path.join(CLUSTERS_DIR, f"{space}_{window}_meta.json")


def result_path(name: str) -> str:
    return os.path.join(RESULTS_DIR, f"{name}.csv")


def figure_path(name: str) -> str:
    return os.path.join(FIGURES_DIR, f"{name}.png")


def embedding_cols(df: pl.DataFrame) -> list[str]:
    """The feature columns of a stage-1 features frame, in file order.

    Matches both the canonical `EMBEDDING_{i}` names this arm writes and the
    `{TYPE}_EMBEDDING_{i}` names `pool_embedding_series_vectorized` emits, so it
    is usable on either an intermediate pooled frame or a written feature file.
    """
    return [c for c in df.columns if "EMBEDDING_" in c]


def load_features(space: str, window: str) -> pl.DataFrame:
    return pl.read_parquet(feature_path(space, window))


def load_labels(space: str, window: str) -> pl.DataFrame:
    return pl.read_parquet(labels_path(space, window))


def load_coords(space: str, window: str) -> pl.DataFrame:
    return pl.read_parquet(coords_path(space, window))


def load_cluster_meta(space: str, window: str) -> dict:
    with open(cluster_meta_path(space, window)) as fh:
        return json.load(fh)


def write_cluster_meta(space: str, window: str, meta: dict) -> str:
    ensure_dirs()
    path = cluster_meta_path(space, window)
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return path


def write_result(df: pl.DataFrame, name: str) -> str:
    """Write a result table, warning rather than failing on an empty frame.

    Mirrors `figures.io.save_figure_data`: a zero-row result is a real outcome
    worth seeing on disk (it says the test found nothing), not a crash.
    """
    ensure_dirs()
    path = result_path(name)
    if df.height == 0:
        print(f"  WARNING: {name} is empty (0 rows)", flush=True)
    df.write_csv(path)
    print(f"  wrote {path} ({df.height} rows)", flush=True)
    return path


def available_pairs(paths_fn=feature_path) -> list[tuple[str, str]]:
    """(space, window) pairs whose artifact already exists on disk."""
    return [(s, w) for s in SPACES for w in WINDOWS if os.path.exists(paths_fn(s, w))]
