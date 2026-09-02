"""The endpoint-scheme registry.

Consolidates three previously-independent copies of the same (scheme ->
embedding file, results dir) mapping: `SCHEME_CONFIG` (slurm_array_utils.py),
and `SCHEME_RESULT_DIRS` + `EMBEDDING_FILES` (_figure_utils.py). All three
carried the same four entries; confirmed byte-identical before merging.
"""
import os

import polars as pl

from anchors import DEFAULT_ANCHOR, anchor_suffix, ensure_anchor
from config import SURV_PATH

SCHEMES: dict[str, dict[str, str]] = {
    "icd3_post": {
        "embedding_file": "level_3_ICD_post_embedding_prediction_df.parquet",
        "results_dir": "level_3_ICD_post_results",
    },
    "icd4_post": {
        "embedding_file": "level_4_ICD_post_embedding_prediction_df.parquet",
        "results_dir": "level_4_ICD_post_results",
    },
    "phecode_post": {
        "embedding_file": "phecode_post_embedding_prediction_df.parquet",
        "results_dir": "phecode_post_results",
    },
    "death_met": {
        "embedding_file": "death_met_embedding_prediction_df.parquet",
        "results_dir": "death_met_results",
    },
}


def ensure_scheme(scheme: str) -> str:
    if scheme not in SCHEMES:
        valid = ", ".join(sorted(SCHEMES))
        raise ValueError(f"Unsupported scheme '{scheme}'. Valid options: {valid}")
    return scheme


def embedding_file(scheme: str, anchor: str = DEFAULT_ANCHOR) -> str:
    """Embedding-prediction filename for `scheme`. Non-default anchors insert
    `anchor_suffix()` before the extension, e.g.
    `death_met_embedding_prediction_df__sequencing.parquet`; `treatment`
    reproduces the current filename exactly."""
    base = SCHEMES[ensure_scheme(scheme)]["embedding_file"]
    suffix = anchor_suffix(ensure_anchor(anchor))
    if not suffix:
        return base
    stem, ext = os.path.splitext(base)
    return f"{stem}{suffix}{ext}"


def scheme_results_dir(scheme: str, anchor: str = DEFAULT_ANCHOR) -> str:
    """Results directory for `scheme`. Non-default anchors nest under
    `<scheme_results_dir>/anchor_<anchor>/`, leaving the `treatment` path
    byte-identical to before anchors existed."""
    base = os.path.join(SURV_PATH, "results", SCHEMES[ensure_scheme(scheme)]["results_dir"])
    ensure_anchor(anchor)
    if anchor == DEFAULT_ANCHOR:
        return base
    return os.path.join(base, f"anchor_{anchor}")


def get_output_dir(scheme: str, run_type: str, anchor: str = DEFAULT_ANCHOR) -> str:
    """Result subdirectory for a training run, created if missing."""
    valid_run_types = {"full_cohort", "feature_comps", "full_cohort_risk_scores"}
    if run_type not in valid_run_types:
        raise ValueError(f"run_type must be one of {sorted(valid_run_types)}")
    out = os.path.join(scheme_results_dir(scheme, anchor), run_type)
    os.makedirs(out, exist_ok=True)
    return out


def full_cohort_event_dir(scheme: str, event: str, anchor: str = DEFAULT_ANCHOR) -> str:
    return os.path.join(scheme_results_dir(scheme, anchor), "full_cohort", event)


def full_cohort_risk_dir(scheme: str, event: str, anchor: str = DEFAULT_ANCHOR) -> str:
    return os.path.join(scheme_results_dir(scheme, anchor), "full_cohort_risk_scores", event)


def feature_held_out_dir(scheme: str, event: str, anchor: str = DEFAULT_ANCHOR) -> str:
    return os.path.join(scheme_results_dir(scheme, anchor), "held_out_risk_scores", event)


def load_embedding_prediction_df(scheme: str, anchor: str = DEFAULT_ANCHOR) -> pl.DataFrame:
    return pl.read_parquet(os.path.join(SURV_PATH, embedding_file(scheme, anchor)))


def list_trained_events(scheme: str, anchor: str = DEFAULT_ANCHOR) -> list[str]:
    """Events under <scheme>/full_cohort/ that have both text and base test files."""
    train_root = os.path.join(scheme_results_dir(scheme, anchor), "full_cohort")
    if not os.path.isdir(train_root):
        return []
    out = []
    for event in sorted(os.listdir(train_root)):
        text_fp = os.path.join(train_root, event, "text_test.csv")
        base_fp = os.path.join(train_root, event, "base_test.csv")
        if os.path.exists(text_fp) and os.path.exists(base_fp):
            out.append(event)
    return out
