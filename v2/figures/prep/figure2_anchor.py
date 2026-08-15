"""Pre-compute inputs for the anchor-sensitivity supplement (treatment vs sequencing time-zero).

Compares full-cohort text/base model metrics under the two `anchors.py` time-zero definitions.
Each arm is reported both on its own natural (eligible) cohort and on the intersection of
patients eligible under both anchors, so a metric shift between anchors can be attributed to the
timescale itself rather than to a change in cohort composition.

Writes to FIGURE_DATA_DIR:
- fig2_anchor_sensitivity.csv     anchor, scheme, event, model, n, n_events, cindex, mean_auc, ibs
                                   (model in {text, base}; one row per anchor x scheme x event x
                                   model, restricted to events trained under both anchors, each
                                   evaluated on both its natural cohort and the both-anchors-
                                   eligible intersection — see `cohort` column: "natural" or
                                   "intersection")
- fig2_anchor_cohort_overlap.csv  scheme, n_treatment, n_sequencing, n_intersection
                                   (patient-level eligible-cohort sizes per scheme, from each
                                   anchor's embedding_prediction_df)
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

from anchors import ANCHORS
from figures.io import save_figure_data
from pipelines.training.slurm_array_utils import filter_event_rows
from schemes import full_cohort_event_dir, full_cohort_risk_dir, list_trained_events, load_embedding_prediction_df
from shared.polars_utils import filter_finite_rows

SCHEMES = ["death_met", "icd3_post", "icd4_post", "phecode_post"]
ANCHOR_LIST = sorted(ANCHORS.keys())

ANCHOR_SENSITIVITY_COLUMNS = [
    "anchor", "scheme", "event", "model", "cohort", "n", "n_events", "cindex", "mean_auc", "ibs",
]
COHORT_OVERLAP_COLUMNS = ["scheme", "n_treatment", "n_sequencing", "n_intersection"]

# Time-dependent-AUC/IBS evaluation grid, matching figure2.py's convention (5th-95th
# percentile of observed follow-up, 50 points).
AUC_TIME_GRID_POINTS = 50


def _eval_times(tt: pl.Series) -> pl.Series:
    lo, hi = tt.quantile(0.05), tt.quantile(0.95)
    return pl.Series(
        [lo + (hi - lo) * i / (AUC_TIME_GRID_POINTS - 1) for i in range(AUC_TIME_GRID_POINTS)]
    )


def _score_predictor(
    evaluation_df: pl.DataFrame,
    reference_df: pl.DataFrame,
    event_col: str,
    time_col: str,
    score_col: str,
) -> tuple[float, float]:
    """(cindex, mean_auc) for a risk score on (event, time).

    IBS is not computed here: it requires predicted survival-function curves, which
    the held-out risk-score CSVs (a scalar linear predictor per patient) don't carry
    — only the CV-time text_test.csv/base_test.csv files (used by
    _natural_cohort_metrics) have it. The intersection-cohort rows leave ibs NaN.
    """
    event_arr = evaluation_df[event_col].cast(pl.Boolean).to_numpy()
    time_arr = evaluation_df[time_col].cast(pl.Float64).to_numpy()
    score_arr = evaluation_df[score_col].cast(pl.Float64).to_numpy()
    try:
        cindex = concordance_index_censored(event_arr, time_arr, score_arr)[0]
    except (ValueError, ZeroDivisionError):
        cindex = float("nan")

    if "outer_fold" not in evaluation_df.columns or "outer_fold" not in reference_df.columns:
        raise ValueError("Risk scores predate nested CV; regenerate files with outer_fold metadata")
    fold_aucs, fold_weights = [], []
    for fold in evaluation_df["outer_fold"].unique().sort().to_list():
        fold_eval = evaluation_df.filter(pl.col("outer_fold") == fold)
        fold_ref = reference_df.filter(pl.col("outer_fold") != fold)
        if fold_eval.is_empty() or fold_ref.is_empty():
            continue
        eval_time = fold_eval[time_col].cast(pl.Float64).to_numpy()
        ref_time = fold_ref[time_col].cast(pl.Float64).to_numpy()
        lo, hi = np.percentile(ref_time, [5, 95])
        et = np.linspace(lo, hi, AUC_TIME_GRID_POINTS)
        et = et[(et > eval_time.min()) & (et < eval_time.max())]
        if len(et) == 0:
            continue
        try:
            y_ref = Surv.from_arrays(
                fold_ref[event_col].cast(pl.Boolean).to_numpy(), ref_time
            )
            y_eval = Surv.from_arrays(
                fold_eval[event_col].cast(pl.Boolean).to_numpy(), eval_time
            )
            fold_aucs.append(float(cumulative_dynamic_auc(
                y_ref, y_eval, fold_eval[score_col].to_numpy(), et
            )[1]))
            fold_weights.append(fold_eval.height)
        except (ValueError, ZeroDivisionError):
            pass
    mean_auc = float(np.average(fold_aucs, weights=fold_weights)) if fold_aucs else float("nan")
    return cindex, mean_auc


def _natural_cohort_metrics(anchor: str) -> pl.DataFrame:
    """One row per (scheme, event, model) using each event's own test_data metrics file,
    already computed on that anchor's natural (eligible) held-out test split."""
    rows = []
    for scheme in SCHEMES:
        for event in list_trained_events(scheme, anchor):
            d = full_cohort_event_dir(scheme, event, anchor)
            try:
                text = pl.read_csv(os.path.join(d, "text_test.csv")).row(0, named=True)
                base = pl.read_csv(os.path.join(d, "base_test.csv")).row(0, named=True)
            except (FileNotFoundError, KeyError, IndexError) as e:
                print(f"  [{anchor}:{scheme}:{event}] skipped — {type(e).__name__}: {e}")
                continue
            pred_df = load_embedding_prediction_df(scheme, anchor)
            event_df = filter_event_rows(pred_df, event)
            n = len(event_df)
            n_events = int(event_df[event].sum()) if event in event_df.columns else None
            for model, row in (("text", text), ("base", base)):
                rows.append({
                    "anchor": anchor, "scheme": scheme, "event": event, "model": model,
                    "cohort": "natural", "n": n, "n_events": n_events,
                    "cindex": row["mean_c_index"], "mean_auc": row["mean_auc(t)"],
                    "ibs": row["mean_ibs"],
                })
    return pl.DataFrame(rows, schema=ANCHOR_SENSITIVITY_COLUMNS) if rows else pl.DataFrame(schema=ANCHOR_SENSITIVITY_COLUMNS)


def _intersection_mrns() -> dict[str, frozenset]:
    """Per-scheme MRN sets eligible under both anchors (natural embedding_prediction_df)."""
    out = {}
    for scheme in SCHEMES:
        mrn_sets = []
        for anchor in ANCHOR_LIST:
            try:
                df = load_embedding_prediction_df(scheme, anchor)
            except FileNotFoundError:
                mrn_sets.append(frozenset())
                continue
            mrn_sets.append(frozenset(df["DFCI_MRN"]))
        out[scheme] = mrn_sets[0].intersection(*mrn_sets[1:]) if mrn_sets else frozenset()
    return out


def _intersection_cohort_metrics(anchor: str, intersection_mrns: dict[str, frozenset]) -> pl.DataFrame:
    """Re-score each trained event's held-out risk scores restricted to the
    both-anchors-eligible intersection, so metric shifts can be attributed to the
    timescale rather than to a change in cohort composition."""
    rows = []
    for scheme in SCHEMES:
        mrns = intersection_mrns.get(scheme, frozenset())
        if not mrns:
            continue
        pred_df = load_embedding_prediction_df(scheme, anchor)
        for event in list_trained_events(scheme, anchor):
            risk_dir = full_cohort_risk_dir(scheme, event, anchor)
            text_fp = os.path.join(risk_dir, "text_risk_scores.csv")
            base_fp = os.path.join(risk_dir, "base_risk_scores.csv")
            if not (os.path.exists(text_fp) and os.path.exists(base_fp)):
                continue
            event_df = filter_event_rows(pred_df, event)
            event_df = event_df.filter(pl.col("DFCI_MRN").is_in(mrns))
            if event_df.is_empty():
                continue
            surv_cols = ["DFCI_MRN", event, f"tt_{event}"]
            for model, fp in (("text", text_fp), ("base", base_fp)):
                risk_df = pl.read_csv(fp)
                reference = risk_df.join(
                    filter_event_rows(pred_df, event).select(surv_cols), on="DFCI_MRN"
                )
                merged = reference.filter(pl.col("DFCI_MRN").is_in(mrns))
                score_col = "text_risk_score" if "text_risk_score" in merged.columns else "base_risk_score"
                merged = filter_finite_rows(merged, [score_col, event, f"tt_{event}"])
                if merged.is_empty():
                    continue
                cindex, mean_auc = _score_predictor(
                    merged, reference, event, f"tt_{event}", score_col,
                )
                rows.append({
                    "anchor": anchor, "scheme": scheme, "event": event, "model": model,
                    "cohort": "intersection", "n": len(merged), "n_events": int(merged[event].sum()),
                    "cindex": cindex, "mean_auc": mean_auc, "ibs": float("nan"),
                })
    return pl.DataFrame(rows, schema=ANCHOR_SENSITIVITY_COLUMNS) if rows else pl.DataFrame(schema=ANCHOR_SENSITIVITY_COLUMNS)


def _cohort_overlap(intersection_mrns: dict[str, frozenset]) -> pl.DataFrame:
    rows = []
    for scheme in SCHEMES:
        sizes = {}
        for anchor in ANCHOR_LIST:
            try:
                df = load_embedding_prediction_df(scheme, anchor)
                sizes[anchor] = df["DFCI_MRN"].n_unique()
            except FileNotFoundError:
                sizes[anchor] = 0
        rows.append({
            "scheme": scheme,
            "n_treatment": sizes.get("treatment", 0),
            "n_sequencing": sizes.get("sequencing", 0),
            "n_intersection": len(intersection_mrns.get(scheme, frozenset())),
        })
    return pl.DataFrame(rows, schema=COHORT_OVERLAP_COLUMNS)


def main() -> None:
    intersection_mrns = _intersection_mrns()

    frames = []
    for anchor in ANCHOR_LIST:
        frames.append(_natural_cohort_metrics(anchor))
        frames.append(_intersection_cohort_metrics(anchor, intersection_mrns))
    sensitivity_df = pl.concat(frames, how="vertical") if frames else pl.DataFrame(
        schema=ANCHOR_SENSITIVITY_COLUMNS
    )
    save_figure_data(sensitivity_df, "fig2_anchor_sensitivity.csv")
    save_figure_data(_cohort_overlap(intersection_mrns), "fig2_anchor_cohort_overlap.csv")


if __name__ == "__main__":
    main()
