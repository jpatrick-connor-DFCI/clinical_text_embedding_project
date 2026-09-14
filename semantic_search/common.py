"""Paths, constants and small loaders shared by the semantic_search stages.

Layout under SEMANTIC_SEARCH_PATH:

    features/{space}_{window}.parquet          stage 1  DFCI_MRN + EMBEDDING_*
    pcs/{space}_{window}_scores.parquet        stage 2  DFCI_MRN, PC1, ...
    pcs/{space}_{window}_loadings.parquet      stage 2  long-form PCA loadings
    pcs/{space}_{window}_transformer.joblib    stage 2  fitted preprocessing/PCA
    pcs/{space}_{window}_meta.json             stage 2  explained variance, ...
    results/*.csv                              stages 1-4
    predictions/*.parquet                     stage 4  out-of-fold predictions
    models/*.joblib                            stage 4  final fitted classifiers
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
DEFAULT_WINDOWS = ["alltime"]

# ``concat`` is retained exclusively for the supervised prediction workflow.
# Exploratory PCA and clinical associations must use a single note type at a
# time, so their three spaces each contain one 768-dimensional patient mean.
SPACES = ["concat"]
PC_SPACES = [note_type.lower() for note_type in NOTE_TYPES]
FEATURE_SPACES = SPACES + PC_SPACES

# CLUSTERS_DIR, FIGURES_DIR, and their helpers are retained only so earlier
# exploratory artifacts and notebooks remain readable. The active exploratory
# workflow writes PCS_DIR and PC association tables instead.
FEATURES_DIR = os.path.join(SEMANTIC_SEARCH_PATH, "features")
CLUSTERS_DIR = os.path.join(SEMANTIC_SEARCH_PATH, "clusters")
PCS_DIR = os.path.join(SEMANTIC_SEARCH_PATH, "pcs")
RESULTS_DIR = os.path.join(SEMANTIC_SEARCH_PATH, "results")
FIGURES_DIR = os.path.join(SEMANTIC_SEARCH_PATH, "figures")
PREDICTIONS_DIR = os.path.join(SEMANTIC_SEARCH_PATH, "predictions")
MODELS_DIR = os.path.join(SEMANTIC_SEARCH_PATH, "models")
PREDICTION_META_DIR = os.path.join(SEMANTIC_SEARCH_PATH, "prediction_meta")

NOTE_TIMING_COL = "NOTE_TIME_REL_FIRST_TREATMENT_START"
PATIENT_KEY = "DFCI_MRN"


def ensure_dirs() -> None:
    """Create the output subdirectories.  Idempotent; safe to call at import
    time of a stage module."""
    for path in (
        FEATURES_DIR,
        CLUSTERS_DIR,
        PCS_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        PREDICTIONS_DIR,
        MODELS_DIR,
        PREDICTION_META_DIR,
    ):
        os.makedirs(path, exist_ok=True)


def _validate(space: str | None = None, window: str | None = None) -> None:
    if space is not None and space not in FEATURE_SPACES:
        raise ValueError(f"Unknown space {space!r}. Must be one of {FEATURE_SPACES}.")
    if window is not None and window not in WINDOWS:
        raise ValueError(f"Unknown window {window!r}. Must be one of {WINDOWS}.")


def feature_path(space: str, window: str) -> str:
    _validate(space, window)
    return os.path.join(FEATURES_DIR, f"{space}_{window}.parquet")


def pc_scores_path(space: str, window: str) -> str:
    _validate(space, window)
    return os.path.join(PCS_DIR, f"{space}_{window}_scores.parquet")


def pc_loadings_path(space: str, window: str) -> str:
    _validate(space, window)
    return os.path.join(PCS_DIR, f"{space}_{window}_loadings.parquet")


def pc_transformer_path(space: str, window: str) -> str:
    _validate(space, window)
    return os.path.join(PCS_DIR, f"{space}_{window}_transformer.joblib")


def pc_meta_path(space: str, window: str) -> str:
    _validate(space, window)
    return os.path.join(PCS_DIR, f"{space}_{window}_meta.json")


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


def load_pc_scores(space: str, window: str) -> pl.DataFrame:
    return pl.read_parquet(pc_scores_path(space, window))


def load_pc_meta(space: str, window: str) -> dict:
    with open(pc_meta_path(space, window)) as fh:
        return json.load(fh)


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
