"""Slurm Array Utils script for model training workflows."""

import functools
import os
from typing import Any

import numpy as np
import pandas as pd

from config import FEATURE_PATH
from schemes import load_embedding_prediction_df

DEFAULT_ALPHAS = np.logspace(-5, 0, 25).tolist()
DEFAULT_L1_RATIOS = [0.5, 1.0]


@functools.cache
def _get_common_feature_mrns() -> set:
    """Return the intersection of patient MRNs across all feature files.

    Used by load_feature_modalities_df to enforce a cohort-agnostic comparison:
    every modality model trains on the same set of patients regardless of which
    features are actually used.

    Cached in-process (Phase-A A3): this reads 5 gzipped CSVs and the result is a constant
    for the lifetime of the process, so repeated calls (e.g. one per modality when A2's
    ``--modality all`` loop or a single-process multi-modality run calls it more than once)
    reuse the same computed set instead of re-reading from disk. Callers must not mutate the
    returned set in place — it is the cached object itself, not a copy.
    """
    somatic_mrns = set(pd.read_csv(os.path.join(FEATURE_PATH, "complete_somatic_data_df.csv.gz"), usecols=["DFCI_MRN"])["DFCI_MRN"])
    prs_mrns = set(pd.read_csv(os.path.join(FEATURE_PATH, "complete_germline_data_df.csv.gz"), usecols=["DFCI_MRN"])["DFCI_MRN"])
    stage_mrns = set(pd.read_csv(os.path.join(FEATURE_PATH, "cancer_stage_df.csv.gz"), usecols=["DFCI_MRN"])["DFCI_MRN"])
    labs_mrns = set(pd.read_csv(os.path.join(FEATURE_PATH, "mean_lab_vals_pre_first_treatment.csv.gz"), usecols=["DFCI_MRN"])["DFCI_MRN"])
    treatment_mrns = set(
        pd.read_csv(os.path.join(FEATURE_PATH, "categorical_treatment_data_by_line.csv.gz"), usecols=["DFCI_MRN", "treatment_line"])
        .query("treatment_line == 1")["DFCI_MRN"]
    )
    common = somatic_mrns & prs_mrns & stage_mrns & labs_mrns & treatment_mrns
    print(f"  Common feature cohort: {len(common)} patients (intersection of all modality files)")
    return common


def _get_n_jobs(default: int | None = None) -> int:
    if default is not None:
        return int(default)
    return int(os.getenv("SLURM_CPUS_PER_TASK", "1"))


def load_cancer_type_df() -> tuple[pd.DataFrame, list[str]]:
    cancer_type_df = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_type_df.csv.gz"))
    type_cols = [col for col in cancer_type_df.columns if col.startswith("CANCER_TYPE_")]
    return cancer_type_df, type_cols


def get_events_from_df(df: pd.DataFrame) -> list[str]:
    return [col[3:] for col in df.columns if col.startswith("tt_")]


def build_full_prediction_df(scheme: str) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    emb_df = load_embedding_prediction_df(scheme)
    cancer_type_df, type_cols = load_cancer_type_df()
    full_prediction_df = emb_df.merge(cancer_type_df[["DFCI_MRN"] + type_cols], on="DFCI_MRN")
    embed_cols = [col for col in full_prediction_df.columns if ("EMBEDDING" in col or "2015" in col)]
    events = get_events_from_df(full_prediction_df)
    return full_prediction_df, type_cols, embed_cols, events


def filter_event_rows(full_prediction_df: pd.DataFrame, event: str) -> pd.DataFrame:
    tt_col = f"tt_{event}"
    if tt_col not in full_prediction_df.columns:
        raise ValueError(f"Missing column '{tt_col}' in prediction dataframe.")
    # Drop rows with NaN or non-positive time-to-event values
    mask = full_prediction_df[tt_col].notna() & (full_prediction_df[tt_col] > 0)
    # Also drop rows where the event indicator itself is NaN
    if event in full_prediction_df.columns:
        mask = mask & full_prediction_df[event].notna()
    event_pred_df = full_prediction_df.loc[mask].copy()
    if event == "brainM" and "CANCER_TYPE_BRAIN" in event_pred_df.columns:
        event_pred_df = event_pred_df.loc[event_pred_df["CANCER_TYPE_BRAIN"].fillna(0).astype(bool) == False].copy()
    return event_pred_df


def load_feature_modalities_df(
    scheme: str,
    modality: str | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any], dict[str, list[str]]]:
    """Load feature data for the given scheme.

    When *modality* is one of the six modality names, only that modality's CSV is read and
    merged, avoiding unnecessary I/O for large files (e.g. PRS) in single-modality jobs.
    When *modality* is None all modalities are loaded but the common-cohort intersection
    (below) is skipped (original behaviour).
    When *modality* is the literal string "all", all six modalities are loaded AND the
    common-cohort intersection is applied — i.e. the union of every single-modality run's
    I/O and row-filtering, so the returned frame is exactly equivalent to loading each
    modality separately (same rows, same per-modality column values) and can be looped
    in-process over all six `modality_cfg` entries.
    """
    emb_df = load_embedding_prediction_df(scheme)
    cancer_type_df = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_type_df.csv.gz"))
    type_cols = [col for col in cancer_type_df.columns if col.startswith("CANCER_TYPE_")]
    embed_cols = [col for col in emb_df.columns if ("EMBEDDING" in col or "2015" in col)]

    _to_load = (
        {modality}
        if modality and modality != "all"
        else {"stage", "treatment", "labs", "somatic", "prs", "text"}
    )

    # Load only what is needed
    mrn_stage_df = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_stage_df.csv.gz")) if "stage" in _to_load else None
    somatic_df = pd.read_csv(os.path.join(FEATURE_PATH, "complete_somatic_data_df.csv.gz")) if "somatic" in _to_load else None
    prs_df = pd.read_csv(os.path.join(FEATURE_PATH, "complete_germline_data_df.csv.gz")) if "prs" in _to_load else None
    treatment_df = pd.read_csv(os.path.join(FEATURE_PATH, "categorical_treatment_data_by_line.csv.gz")) if "treatment" in _to_load else None
    labs_df = pd.read_csv(os.path.join(FEATURE_PATH, "mean_lab_vals_pre_first_treatment.csv.gz")) if "labs" in _to_load else None

    stage_cols = [col for col in mrn_stage_df.columns if col.startswith("CANCER_STAGE_")] if mrn_stage_df is not None else []
    somatic_cols = [col for col in somatic_df.columns if col.endswith(('_AMP', '_DEL', '_SNV', '_SV', '_FUSION'))] if somatic_df is not None else []
    prs_cols = [col for col in prs_df.columns if "PGS" in col] if prs_df is not None else []
    treatment_cols = [col for col in treatment_df.columns if col.startswith("PX_on_")] if treatment_df is not None else []
    labs_cols = [col for col in labs_df.columns if col != "DFCI_MRN"] if labs_df is not None else []

    # For single-modality runs (and "all", which must match them exactly), restrict to the
    # intersection of all feature files so all modality models train on the same cohort
    # (cohort-agnostic comparison). Only bare modality=None (used elsewhere for the original,
    # cohort-unrestricted behaviour) skips this.
    if modality is not None:
        common_mrns = _get_common_feature_mrns()
        emb_df = emb_df[emb_df["DFCI_MRN"].isin(common_mrns)].copy()

    # Base merge: embedding + cancer type (always needed)
    full_prediction_df = emb_df.merge(cancer_type_df[["DFCI_MRN"] + type_cols], on="DFCI_MRN")

    if somatic_df is not None:
        full_prediction_df = full_prediction_df.merge(somatic_df[["DFCI_MRN"] + somatic_cols], on="DFCI_MRN")
    if prs_df is not None:
        full_prediction_df = full_prediction_df.merge(prs_df[["DFCI_MRN"] + prs_cols], on="DFCI_MRN")
    if treatment_df is not None:
        full_prediction_df = full_prediction_df.merge(
            treatment_df.loc[treatment_df["treatment_line"] == 1, ["DFCI_MRN"] + treatment_cols],
            on="DFCI_MRN",
        )
    if mrn_stage_df is not None:
        full_prediction_df = full_prediction_df.merge(mrn_stage_df[["DFCI_MRN"] + stage_cols], on="DFCI_MRN")
    if labs_df is not None:
        full_prediction_df = full_prediction_df.merge(labs_df[["DFCI_MRN"] + labs_cols], on="DFCI_MRN")

    modality_cfg: dict[str, dict[str, Any]] = {
        "stage": {
            "continuous_vars": ["AGE_AT_TREATMENTSTART"],
            "penalized_cols": stage_cols,
            "pca_config": None,
        },
        "treatment": {
            "continuous_vars": ["AGE_AT_TREATMENTSTART"],
            "penalized_cols": treatment_cols,
            "pca_config": None,
        },
        "labs": {
            "continuous_vars": ["AGE_AT_TREATMENTSTART"] + labs_cols,
            "penalized_cols": labs_cols,
            "pca_config": None,
        },
        "somatic": {
            "continuous_vars": ["AGE_AT_TREATMENTSTART"],
            "penalized_cols": somatic_cols,
            "pca_config": None,
        },
        "prs": {
            "continuous_vars": ["AGE_AT_TREATMENTSTART"],
            "penalized_cols": prs_cols,
            "pca_config": {"PGS": (prs_cols, 1500)},
        },
        "text": {
            "continuous_vars": ["AGE_AT_TREATMENTSTART"] + embed_cols,
            "penalized_cols": embed_cols,
            "pca_config": None,
        },
    }
    feature_cols = {
        "type_cols": type_cols,
        "embed_cols": embed_cols,
    }
    return full_prediction_df, type_cols, embed_cols, modality_cfg, feature_cols


MIN_EVENTS_FOR_CV = 50
MIN_NON_EVENTS_FOR_CV = 50


def validate_cox_inputs(
    df: pd.DataFrame,
    event_col: str,
    tstop_col: str,
    feature_cols: list[str],
    label: str = "",
) -> tuple[pd.DataFrame, set[str]]:
    """Validate data suitability for Cox PH models before fitting.

    Returns (possibly filtered DataFrame, set of dropped constant columns).
    Raises ValueError for conditions that make fitting impossible.
    """
    tag = f"[{label}] " if label else ""

    # --- survival columns ---
    assert tstop_col in df.columns, f"{tag}Missing time column '{tstop_col}'"
    assert event_col in df.columns, f"{tag}Missing event column '{event_col}'"

    n_nan_time = df[tstop_col].isna().sum()
    n_nan_event = df[event_col].isna().sum()
    if n_nan_time > 0 or n_nan_event > 0:
        raise ValueError(
            f"{tag}{n_nan_time} NaN times, {n_nan_event} NaN events — "
            "filter_event_rows should have removed these"
        )

    if (df[tstop_col] <= 0).any():
        n_bad = (df[tstop_col] <= 0).sum()
        raise ValueError(f"{tag}{n_bad} rows with non-positive time-to-event")

    # event must be binary 0/1
    unique_events = set(df[event_col].unique())
    if not unique_events.issubset({0, 1, 0.0, 1.0, True, False}):
        raise ValueError(
            f"{tag}Event column '{event_col}' has non-binary values: "
            f"{sorted(unique_events - {0, 1, 0.0, 1.0, True, False})}"
        )

    n_events = int(df[event_col].sum())
    n_non_events = len(df) - n_events
    if n_events < MIN_EVENTS_FOR_CV:
        raise ValueError(
            f"{tag}Only {n_events} positive events (need >= {MIN_EVENTS_FOR_CV} "
            "for stratified 5-fold CV)"
        )
    if n_non_events < MIN_NON_EVENTS_FOR_CV:
        raise ValueError(
            f"{tag}Only {n_non_events} censored observations (need >= {MIN_NON_EVENTS_FOR_CV} "
            "for stratified 5-fold CV)"
        )

    # --- feature columns ---
    present = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(present)
    if missing:
        raise ValueError(f"{tag}Missing feature columns: {sorted(missing)}")

    constant_cols = set(c for c in present if df[c].nunique(dropna=False) <= 1)
    if constant_cols:
        print(f"{tag}Dropping {len(constant_cols)} constant feature columns: {sorted(constant_cols)}")
        df = df.drop(columns=constant_cols)

    # NaN in features: warn (run_grid_CoxPH_parallel drops these internally)
    non_constant = [c for c in present if c not in constant_cols]
    n_feat_nan = df[non_constant].isna().any(axis=1).sum()
    if n_feat_nan > 0:
        print(f"{tag}Warning: {n_feat_nan}/{len(df)} rows have NaN in feature columns")

    return df, constant_cols


def get_best_hparams(val_fp: str) -> tuple[float, float]:
    """Return (l1_ratio, alpha) from the CV row with highest mean_auc(t)."""
    val_df = pd.read_csv(val_fp)
    best = val_df.sort_values(by="mean_auc(t)", ascending=False).iloc[0]
    return float(best["l1_ratio"]), float(best["alpha"])


def parse_manifest_line(line: str, expected_fields: int) -> list[str]:
    fields = line.rstrip("\n").split("\t")
    if len(fields) != expected_fields:
        raise ValueError(f"Expected {expected_fields} tab-separated fields, got {len(fields)}: {line!r}")
    return fields


__all__ = [
    "FEATURE_PATH",
    "DEFAULT_ALPHAS",
    "DEFAULT_L1_RATIOS",
    "_get_n_jobs",
    "_get_common_feature_mrns",
    "build_full_prediction_df",
    "filter_event_rows",
    "get_best_hparams",
    "get_events_from_df",
    "load_feature_modalities_df",
    "parse_manifest_line",
    "validate_cox_inputs",
]
