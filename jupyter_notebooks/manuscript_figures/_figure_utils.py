"""Shared utilities for manuscript figure notebooks.

Conventions
-----------
- Every panel is saved as a standalone PNG via `save_panel(fig, "fig2c")`.
- Output goes to `jupyter_notebooks/manuscript_figures/figures/`.
- Color palettes are kept consistent across figures (modality palette spans Fig 2-3;
  cluster palette is used in Fig 4).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
FIGURE_OUT_DIR = THIS_DIR / "figures"

# Project paths (cluster) — mirrors slurm_array_utils.py
DATA_PATH = "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/"
SURV_PATH = os.path.join(DATA_PATH, "time-to-event_analysis/")
FEATURE_PATH = os.path.join(DATA_PATH, "clinical_and_genomic_features/")
NOTES_PATH = os.path.join(DATA_PATH, "batched_datasets/processed_datasets/")
RESULTS_PATH = os.path.join(SURV_PATH, "results/")
BIOMARKER_PATH = os.path.join(DATA_PATH, "biomarker_analysis/")

# Pre-computed figure inputs (produced by data_generation/prep_figure_*.py)
FIGURE_DATA_DIR = os.path.join(RESULTS_PATH, "figure_data/")

# Result subdir per scheme — matches SCHEME_CONFIG.results_dir
SCHEME_RESULT_DIRS = {
    "death_met": "death_met_results",
    "icd3_post": "level_3_ICD_post_results",
    "icd4_post": "level_4_ICD_post_results",
    "phecode_post": "phecode_post_results",
}

EMBEDDING_FILES = {
    "death_met": "death_met_embedding_prediction_df.csv.gz",
    "icd3_post": "level_3_ICD_post_embedding_prediction_df.csv.gz",
    "icd4_post": "level_4_ICD_post_embedding_prediction_df.csv.gz",
    "phecode_post": "phecode_post_embedding_prediction_df.csv.gz",
}

# Make slurm_array_utils importable so notebooks can reuse loaders / helpers
_TRAINING_DIR = THIS_DIR.parents[1] / "python_scripts" / "model_training"
if str(_TRAINING_DIR) not in sys.path:
    sys.path.append(str(_TRAINING_DIR))


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------
# NOTE: "labs" temporarily disabled for Fig 3 regeneration — uncomment to
# restore. MODALITY_ORDER is only consumed by prep_figure_3 / plot_figure_3,
# so this scoped change does not affect Figs 1/2/4/5.
MODALITY_ORDER = ["stage", "treatment", "somatic", "prs", "text"]  # "labs",
MODALITY_COLORS = {
    "stage":     "#8C8C8C",
    "treatment": "#E8B72E",
    "labs":      "#5DA5DA",  # kept so re-enabling MODALITY_ORDER needs no other change
    "somatic":   "#60BD68",
    "prs":       "#B276B2",
    "text":      "#D62728",
}

MODEL_COLORS = {"text": "#D62728", "base": "#4A4A4A"}

# Used in Fig 4; up to 8 clusters
CLUSTER_COLORS = [
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728",
    "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
]


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_panel(fig: plt.Figure, name: str) -> Path:
    """Save `fig` as `figures/<name>.png` and return the path.

    `name` should be like 'fig2c' (no extension). PDFs are explicitly disabled.
    """
    FIGURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_OUT_DIR / f"{name}.png"
    fig.savefig(out, format="png")
    print(f"[saved] {out}")
    return out


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def scheme_results_dir(scheme: str) -> str:
    return os.path.join(RESULTS_PATH, SCHEME_RESULT_DIRS[scheme])


def full_cohort_event_dir(scheme: str, event: str) -> str:
    return os.path.join(scheme_results_dir(scheme), "full_cohort", event)


def full_cohort_risk_dir(scheme: str, event: str) -> str:
    return os.path.join(scheme_results_dir(scheme), "full_cohort_risk_scores", event)


def feature_held_out_dir(scheme: str, event: str) -> str:
    return os.path.join(scheme_results_dir(scheme), "held_out_risk_scores", event)


# ---------------------------------------------------------------------------
# Figure-data IO (prep scripts write, notebooks read)
# ---------------------------------------------------------------------------
def figure_data_path(name: str) -> str:
    """Resolve a filename like 'fig2_metrics.csv' to its absolute path."""
    return os.path.join(FIGURE_DATA_DIR, name)


def save_figure_data(df: pd.DataFrame, name: str) -> str:
    """Write a prepared CSV to FIGURE_DATA_DIR and return the path."""
    if len(df.columns) == 0:
        raise ValueError(f"Refusing to write schema-less figure data for {name}")
    os.makedirs(FIGURE_DATA_DIR, exist_ok=True)
    out = figure_data_path(name)
    df.to_csv(out, index=False)
    print(f"[wrote] {out}  ({len(df):,} rows)")
    return out


def load_figure_data(name: str) -> pd.DataFrame:
    """Read a prepared CSV from FIGURE_DATA_DIR."""
    return pd.read_csv(figure_data_path(name))


def list_trained_events(scheme: str) -> list[str]:
    """Events under <scheme>/full_cohort/ that have both text and base test files."""
    train_root = os.path.join(scheme_results_dir(scheme), "full_cohort")
    if not os.path.isdir(train_root):
        return []
    out = []
    for ev in sorted(os.listdir(train_root)):
        tp = os.path.join(train_root, ev, "text_test.csv")
        bp = os.path.join(train_root, ev, "base_test.csv")
        if os.path.exists(tp) and os.path.exists(bp):
            out.append(ev)
    return out
