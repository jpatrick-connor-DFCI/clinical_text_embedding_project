"""Shared helpers for manuscript figure-generation scripts."""

from __future__ import annotations

import os
import warnings

import matplotlib as mpl


DATA_PATH = "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/"
SURV_PATH = os.path.join(DATA_PATH, "time-to-event_analysis/")
FEATURE_PATH = os.path.join(DATA_PATH, "clinical_and_genomic_features/")
RESULTS_PATH = os.path.join(SURV_PATH, "results/")
COMPILED_DIR = os.path.join(RESULTS_PATH, "compiled_all_schemes/")
DEFAULT_OUTPUT_DIR = os.path.join(DATA_PATH, "figures/manuscript/")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_OUTPUT_DIR = os.path.join(REPO_ROOT, "generated_figures", "manuscript")


def resolve_output_dir() -> str:
    for path in (DEFAULT_OUTPUT_DIR, LOCAL_OUTPUT_DIR):
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except PermissionError:
            continue

    warnings.warn(
        f"Falling back to writable local figure output directory: {LOCAL_OUTPUT_DIR}",
        stacklevel=2,
    )
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    return LOCAL_OUTPUT_DIR


OUTPUT_DIR = resolve_output_dir()

# compile_all_scheme_results.ipynb defines thresholds [0.01, 0.025, 0.05]
# and writes labels via _threshold_to_label(...) -> 1pct, 2_5pct, 5pct.
PREVALENCE_FILTERS_TO_PLOT = [None, "1pct", "2_5pct", "5pct"]


def prevalence_filter_to_suffix(prevalence_filter: str | None) -> str:
    return "unfiltered" if prevalence_filter in (None, "") else prevalence_filter


def prevalence_filter_to_label(prevalence_filter: str | None) -> str:
    return "unfiltered event set" if prevalence_filter in (None, "") else f"{prevalence_filter} event set"


def get_compiled_metrics_path(prevalence_filter: str | None) -> str:
    if prevalence_filter in (None, ""):
        path = os.path.join(COMPILED_DIR, "all_schemes_compiled_metrics.csv")
    else:
        path = os.path.join(COMPILED_DIR, f"all_schemes_compiled_metrics_{prevalence_filter}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Compiled metrics not found for filter {prevalence_filter!r}: {path}")
    return path


def set_manuscript_style(legend_fontsize: float = 8) -> None:
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid")
    mpl.rcParams.update({
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": legend_fontsize,
    })
