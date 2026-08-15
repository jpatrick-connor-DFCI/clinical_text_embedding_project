"""Slurm Array Utils script for model training workflows."""

import functools
import gzip
import io
import os
from typing import Any

import numpy as np
import polars as pl

from anchors import DEFAULT_ANCHOR, age_col, anchor_suffix
from config import FEATURE_PATH
from schemes import load_embedding_prediction_df
from shared.polars_utils import finite_or_zero

DEFAULT_ALPHAS = np.logspace(-5, 0, 25).tolist()
DEFAULT_L1_RATIOS = [0.5, 1.0]


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated CLI list, used by lightweight smoke runs."""
    try:
        values = [float(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Expected comma-separated numbers, got {value!r}") from exc
    if not values:
        raise ValueError("Expected at least one number")
    return values


def write_ipcw_reference_csv(df: pl.DataFrame, path: str) -> None:
    """Write compact IPCW audit rows as a gzip-compressed CSV."""
    buffer = io.StringIO()
    df.write_csv(buffer)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def _feature_file(filename: str, anchor: str) -> str:
    """Return an anchor-routed feature path; treatment keeps legacy names."""
    stem, ext = filename.split(".csv", 1)
    return os.path.join(FEATURE_PATH, f"{stem}{anchor_suffix(anchor)}.csv{ext}")


@functools.cache
def _get_common_feature_mrns(anchor: str = DEFAULT_ANCHOR) -> set:
    """Return the intersection of patient MRNs across all feature files.

    Used by load_feature_modalities_df to enforce a cohort-agnostic comparison:
    every modality model trains on the same set of patients regardless of which
    features are actually used.

    met_burden_df.csv.gz (the `metburden` modality) is intentionally excluded
    from this intersection. It is zero-filled to every cohort patient (see
    build_met_burden_df in generate_all_non_text_covariates.py), so its MRN
    set is always a superset of every other file's and adding it here would
    be a guaranteed no-op read on the happy path. The one way it could change
    the result is if the file were stale relative to cohort_df.parquet, in
    which case including it would silently shrink the cohort for every other
    modality too — the worst failure mode. Leaving it out means a stale
    met_burden_df degrades only the metburden modality (see the left-join +
    fillna(0) in load_feature_modalities_df below).

    Cached in-process (Phase-A A3): this reads 5 gzipped CSVs and the result is a constant
    for the lifetime of the process, so repeated calls (e.g. one per modality when A2's
    ``--modality all`` loop or a single-process multi-modality run calls it more than once)
    reuse the same computed set instead of re-reading from disk. Callers must not mutate the
    returned set in place — it is the cached object itself, not a copy.
    """
    somatic_mrns = set(pl.read_csv(_feature_file("complete_somatic_data_df.csv.gz", anchor), columns=["DFCI_MRN"])["DFCI_MRN"])
    prs_mrns = set(pl.read_csv(os.path.join(FEATURE_PATH, "complete_germline_data_df.csv.gz"), columns=["DFCI_MRN"])["DFCI_MRN"])
    stage_mrns = set(pl.read_csv(_feature_file("cancer_stage_df.csv.gz", anchor), columns=["DFCI_MRN"])["DFCI_MRN"])
    treatment_mrns = set(
        pl.read_csv(_feature_file("categorical_treatment_data_by_line.csv.gz", anchor), columns=["DFCI_MRN"])["DFCI_MRN"]
    )
    common = somatic_mrns & prs_mrns & stage_mrns & treatment_mrns
    print(f"  Common feature cohort: {len(common)} patients (intersection of all modality files)")
    return common


def _get_n_jobs(default: int | None = None) -> int:
    if default is not None:
        return int(default)
    return int(os.getenv("SLURM_CPUS_PER_TASK", "1"))


def load_cancer_type_df(anchor: str = DEFAULT_ANCHOR) -> tuple[pl.DataFrame, list[str]]:
    cancer_type_df = pl.read_csv(_feature_file("cancer_type_df.csv.gz", anchor))
    type_cols = [col for col in cancer_type_df.columns if col.startswith("CANCER_TYPE_")]
    return cancer_type_df, type_cols


def get_events_from_df(df: pl.DataFrame) -> list[str]:
    return [col[3:] for col in df.columns if col.startswith("tt_")]


def build_full_prediction_df(
    scheme: str, anchor: str = DEFAULT_ANCHOR
) -> tuple[pl.DataFrame, list[str], list[str], list[str]]:
    emb_df = load_embedding_prediction_df(scheme, anchor)
    cancer_type_df, type_cols = load_cancer_type_df(anchor)
    full_prediction_df = emb_df.join(cancer_type_df.select(["DFCI_MRN"] + type_cols), on="DFCI_MRN")
    embed_cols = [col for col in full_prediction_df.columns if ("EMBEDDING" in col or "2015" in col)]
    events = get_events_from_df(full_prediction_df)
    return full_prediction_df, type_cols, embed_cols, events


def filter_event_rows(full_prediction_df: pl.DataFrame, event: str) -> pl.DataFrame:
    tt_col = f"tt_{event}"
    if tt_col not in full_prediction_df.columns:
        raise ValueError(f"Missing column '{tt_col}' in prediction dataframe.")
    # Drop null, NaN, infinite, and non-positive time-to-event values.
    mask = pl.col(tt_col).cast(pl.Float64, strict=False).is_finite() & (pl.col(tt_col) > 0)
    # Also drop rows where the event indicator is null, NaN, or infinite.
    if event in full_prediction_df.columns:
        mask = mask & pl.col(event).cast(pl.Float64, strict=False).is_finite()
    event_pred_df = full_prediction_df.filter(mask)
    if event == "brainM" and "CANCER_TYPE_BRAIN" in event_pred_df.columns:
        event_pred_df = event_pred_df.filter(
            finite_or_zero("CANCER_TYPE_BRAIN").cast(pl.Boolean) == False
        )
    return event_pred_df


def load_feature_modalities_df(
    scheme: str,
    modality: str | None = None,
    anchor: str = DEFAULT_ANCHOR,
) -> tuple[pl.DataFrame, list[str], list[str], dict[str, Any], dict[str, list[str]]]:
    """Load feature data for the given scheme.

    When *modality* is one of the seven modality names, only that modality's CSV is read and
    merged, avoiding unnecessary I/O for large files (e.g. PRS) in single-modality jobs.
    When *modality* is None all modalities are loaded but the common-cohort intersection
    (below) is skipped (original behaviour).
    When *modality* is the literal string "all", all seven modalities are loaded AND the
    common-cohort intersection is applied — i.e. the union of every single-modality run's
    I/O and row-filtering, so the returned frame is exactly equivalent to loading each
    modality separately (same rows, same per-modality column values) and can be looped
    in-process over all seven `modality_cfg` entries.

    Every temporal feature file is anchor-routed; treatment keeps the legacy
    filename while sequencing uses the anchors.anchor_suffix convention.
    """
    emb_df = load_embedding_prediction_df(scheme, anchor)
    cancer_type_df = pl.read_csv(_feature_file("cancer_type_df.csv.gz", anchor))
    type_cols = [col for col in cancer_type_df.columns if col.startswith("CANCER_TYPE_")]
    embed_cols = [col for col in emb_df.columns if ("EMBEDDING" in col or "2015" in col)]

    _to_load = (
        {modality}
        if modality and modality != "all"
        else {"stage", "treatment", "somatic", "prs", "text", "metburden"}
    )

    # Load only what is needed
    mrn_stage_df = pl.read_csv(_feature_file("cancer_stage_df.csv.gz", anchor)) if "stage" in _to_load else None
    somatic_df = pl.read_csv(_feature_file("complete_somatic_data_df.csv.gz", anchor)) if "somatic" in _to_load else None
    prs_df = pl.read_csv(os.path.join(FEATURE_PATH, "complete_germline_data_df.csv.gz")) if "prs" in _to_load else None
    treatment_df = pl.read_csv(_feature_file("categorical_treatment_data_by_line.csv.gz", anchor)) if "treatment" in _to_load else None
    met_burden_df = pl.read_csv(_feature_file("met_burden_df.csv.gz", anchor)) if "metburden" in _to_load else None

    stage_cols = [col for col in mrn_stage_df.columns if col.startswith("CANCER_STAGE_")] if mrn_stage_df is not None else []
    somatic_cols = [col for col in somatic_df.columns if col.endswith(('_AMP', '_DEL', '_SNV', '_SV', '_FUSION'))] if somatic_df is not None else []
    prs_cols = [col for col in prs_df.columns if "PGS" in col] if prs_df is not None else []
    treatment_cols = [col for col in treatment_df.columns if col.startswith("PX_on_")] if treatment_df is not None else []
    met_site_cols = [col for col in met_burden_df.columns if col.startswith("MET_SITE_")] if met_burden_df is not None else []

    # For single-modality runs (and "all", which must match them exactly), restrict to the
    # intersection of all feature files so all modality models train on the same cohort
    # (cohort-agnostic comparison). Only bare modality=None (used elsewhere for the original,
    # cohort-unrestricted behaviour) skips this.
    if modality is not None:
        common_mrns = _get_common_feature_mrns(anchor)
        emb_df = emb_df.filter(pl.col("DFCI_MRN").is_in(common_mrns))

    # Base merge: embedding + cancer type (always needed)
    full_prediction_df = emb_df.join(cancer_type_df.select(["DFCI_MRN"] + type_cols), on="DFCI_MRN")

    if somatic_df is not None:
        full_prediction_df = full_prediction_df.join(somatic_df.select(["DFCI_MRN"] + somatic_cols), on="DFCI_MRN")
    if prs_df is not None:
        full_prediction_df = full_prediction_df.join(prs_df.select(["DFCI_MRN"] + prs_cols), on="DFCI_MRN")
    if treatment_df is not None:
        full_prediction_df = full_prediction_df.join(
            treatment_df.select(["DFCI_MRN"] + treatment_cols),
            on="DFCI_MRN",
        )
    if mrn_stage_df is not None:
        full_prediction_df = full_prediction_df.join(mrn_stage_df.select(["DFCI_MRN"] + stage_cols), on="DFCI_MRN")
    if met_burden_df is not None:
        # left join + fillna(0), unlike the inner joins above: met_burden_df is
        # zero-filled to the full cohort by construction, so a missing MRN here
        # means a stale feature file, not a patient who lacks metastatic burden
        # data. An inner join would silently shrink the cohort for this modality
        # alone on a stale file; left join + fillna(0) degrades gracefully and
        # matches the file's intended zero-fill semantics.
        met_cols = ["N_MET_SITES"] + met_site_cols
        full_prediction_df = full_prediction_df.join(
            met_burden_df.select(["DFCI_MRN"] + met_cols), on="DFCI_MRN", how="left"
        )
        full_prediction_df = full_prediction_df.with_columns(
            [finite_or_zero(c).alias(c) for c in met_cols]
        )

    anchor_age_col = age_col(anchor)
    modality_cfg: dict[str, dict[str, Any]] = {
        "stage": {
            "continuous_vars": [anchor_age_col],
            "penalized_cols": stage_cols,
            "pca_config": None,
        },
        "treatment": {
            "continuous_vars": [anchor_age_col],
            "penalized_cols": treatment_cols,
            "pca_config": None,
        },
        "somatic": {
            "continuous_vars": [anchor_age_col],
            "penalized_cols": somatic_cols,
            "pca_config": None,
        },
        "prs": {
            "continuous_vars": [anchor_age_col],
            "penalized_cols": prs_cols,
            "pca_config": {"PGS": (prs_cols, 1500)},
        },
        "text": {
            "continuous_vars": [anchor_age_col] + embed_cols,
            "penalized_cols": embed_cols,
            "pca_config": None,
        },
        "metburden": {
            # N_MET_SITES unpenalized (robust aggregate signal); the eight
            # MET_SITE_* site indicators penalized (sparse, site-pattern signal).
            # N_MET_SITES is an exact sum of the indicators, so under the
            # elastic net (l1_ratio in {0.5, 1.0}) individual indicator
            # coefficients read as "deviation from the additive-count effect",
            # not standalone site effects. See build_met_burden_df's docstring
            # for the LEAKAGE WARNING re: the *M outcome events — the
            # concordant indicator and its count contribution are removed per-event in
            # run_feature_comp_task.py, not here (this cfg is shared across events).
            "continuous_vars": [anchor_age_col, "N_MET_SITES"],
            "penalized_cols": met_site_cols,
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
    df: pl.DataFrame,
    event_col: str,
    tstop_col: str,
    feature_cols: list[str],
    label: str = "",
) -> tuple[pl.DataFrame, set[str]]:
    """Validate data suitability for Cox PH models before fitting.

    Returns (possibly filtered DataFrame, set of dropped constant columns).
    Raises ValueError for conditions that make fitting impossible.
    """
    tag = f"[{label}] " if label else ""

    # --- survival columns ---
    assert tstop_col in df.columns, f"{tag}Missing time column '{tstop_col}'"
    assert event_col in df.columns, f"{tag}Missing event column '{event_col}'"

    n_bad_time = df.select(
        (~pl.col(tstop_col).cast(pl.Float64, strict=False).is_finite().fill_null(False)).sum()
    ).item()
    n_bad_event = df.select(
        (~pl.col(event_col).cast(pl.Float64, strict=False).is_finite().fill_null(False)).sum()
    ).item()
    if n_bad_time > 0 or n_bad_event > 0:
        raise ValueError(
            f"{tag}{n_bad_time} invalid times, {n_bad_event} invalid events — "
            "filter_event_rows should have removed these"
        )

    if (df[tstop_col] <= 0).any():
        n_bad = (df[tstop_col] <= 0).sum()
        raise ValueError(f"{tag}{n_bad} rows with non-positive time-to-event")

    # event must be binary 0/1
    unique_events = set(df[event_col].unique().to_list())
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

    constant_cols = set(c for c in present if df[c].n_unique() <= 1)
    if constant_cols:
        print(f"{tag}Dropping {len(constant_cols)} constant feature columns: {sorted(constant_cols)}")
        df = df.drop(constant_cols)

    # Null/NaN/infinite feature values are imputed or removed by the caller.
    non_constant = [c for c in present if c not in constant_cols]
    n_feat_invalid = df.select(
        pl.any_horizontal([
            ~pl.col(c).cast(pl.Float64, strict=False).is_finite().fill_null(False)
            for c in non_constant
        ]).sum()
    ).item() if non_constant else 0
    if n_feat_invalid > 0:
        print(
            f"{tag}Warning: {n_feat_invalid}/{len(df)} rows have invalid "
            "feature values"
        )

    return df, constant_cols


def get_best_hparams(val_fp: str) -> tuple[float, float]:
    """Return (l1_ratio, alpha) from the CV row with highest mean_auc(t)."""
    val_df = pl.read_csv(val_fp)
    valid = val_df.filter(pl.col("mean_auc(t)").is_finite())
    if valid.is_empty():
        raise ValueError(f"No finite mean_auc(t) values in {val_fp}")
    best = valid.sort(by="mean_auc(t)", descending=True).row(0, named=True)
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
    "write_ipcw_reference_csv",
    "validate_cox_inputs",
]
