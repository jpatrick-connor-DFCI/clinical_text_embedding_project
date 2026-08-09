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

import pandas as pd
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

from anchors import ANCHORS
from figures.io import save_figure_data
from pipelines.training.slurm_array_utils import filter_event_rows
from schemes import full_cohort_event_dir, full_cohort_risk_dir, list_trained_events, load_embedding_prediction_df

SCHEMES = ["death_met", "icd3_post", "icd4_post", "phecode_post"]
ANCHOR_LIST = sorted(ANCHORS.keys())

ANCHOR_SENSITIVITY_COLUMNS = [
    "anchor", "scheme", "event", "model", "cohort", "n", "n_events", "cindex", "mean_auc", "ibs",
]
COHORT_OVERLAP_COLUMNS = ["scheme", "n_treatment", "n_sequencing", "n_intersection"]

# Time-dependent-AUC/IBS evaluation grid, matching figure2.py's convention (5th-95th
# percentile of observed follow-up, 50 points).
AUC_TIME_GRID_POINTS = 50


def _eval_times(tt: pd.Series) -> pd.Series:
    lo, hi = tt.quantile(0.05), tt.quantile(0.95)
    return pd.Series(pd.array(
        [lo + (hi - lo) * i / (AUC_TIME_GRID_POINTS - 1) for i in range(AUC_TIME_GRID_POINTS)]
    ))


def _score_predictor(
    event_flag: pd.Series, tt: pd.Series, risk_score: pd.Series,
) -> tuple[float, float]:
    """(cindex, mean_auc) for a risk score on (event, time).

    IBS is not computed here: it requires predicted survival-function curves, which
    the held-out risk-score CSVs (a scalar linear predictor per patient) don't carry
    — only the CV-time text_test.csv/base_test.csv files (used by
    _natural_cohort_metrics) have it. The intersection-cohort rows leave ibs NaN.
    """
    event_arr = event_flag.astype(bool).to_numpy()
    time_arr = tt.astype(float).to_numpy()
    score_arr = risk_score.astype(float).to_numpy()
    try:
        cindex = concordance_index_censored(event_arr, time_arr, score_arr)[0]
    except (ValueError, ZeroDivisionError):
        cindex = float("nan")

    y = Surv.from_arrays(event_arr, time_arr)
    et = _eval_times(tt)
    et = et[(et > time_arr.min()) & (et < time_arr.max())].to_numpy()
    mean_auc = float("nan")
    if len(et) > 0:
        try:
            mean_auc = cumulative_dynamic_auc(y, y, score_arr, et)[1]
        except (ValueError, ZeroDivisionError):
            pass
    return cindex, mean_auc


def _natural_cohort_metrics(anchor: str) -> pd.DataFrame:
    """One row per (scheme, event, model) using each event's own test_data metrics file,
    already computed on that anchor's natural (eligible) held-out test split."""
    rows = []
    for scheme in SCHEMES:
        for event in list_trained_events(scheme, anchor):
            d = full_cohort_event_dir(scheme, event, anchor)
            try:
                text = pd.read_csv(os.path.join(d, "text_test.csv")).iloc[0]
                base = pd.read_csv(os.path.join(d, "base_test.csv")).iloc[0]
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
    return pd.DataFrame(rows, columns=ANCHOR_SENSITIVITY_COLUMNS)


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


def _intersection_cohort_metrics(anchor: str, intersection_mrns: dict[str, frozenset]) -> pd.DataFrame:
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
            event_df = event_df[event_df["DFCI_MRN"].isin(mrns)]
            if event_df.empty:
                continue
            surv_cols = ["DFCI_MRN", event, f"tt_{event}"]
            for model, fp in (("text", text_fp), ("base", base_fp)):
                risk_df = pd.read_csv(fp)
                merged = risk_df.merge(event_df[surv_cols], on="DFCI_MRN").dropna()
                if merged.empty:
                    continue
                score_col = "text_risk_score" if "text_risk_score" in merged.columns else "base_risk_score"
                cindex, mean_auc = _score_predictor(
                    merged[event], merged[f"tt_{event}"], merged[score_col],
                )
                rows.append({
                    "anchor": anchor, "scheme": scheme, "event": event, "model": model,
                    "cohort": "intersection", "n": len(merged), "n_events": int(merged[event].sum()),
                    "cindex": cindex, "mean_auc": mean_auc, "ibs": float("nan"),
                })
    return pd.DataFrame(rows, columns=ANCHOR_SENSITIVITY_COLUMNS)


def _cohort_overlap(intersection_mrns: dict[str, frozenset]) -> pd.DataFrame:
    rows = []
    for scheme in SCHEMES:
        sizes = {}
        for anchor in ANCHOR_LIST:
            try:
                df = load_embedding_prediction_df(scheme, anchor)
                sizes[anchor] = df["DFCI_MRN"].nunique()
            except FileNotFoundError:
                sizes[anchor] = 0
        rows.append({
            "scheme": scheme,
            "n_treatment": sizes.get("treatment", 0),
            "n_sequencing": sizes.get("sequencing", 0),
            "n_intersection": len(intersection_mrns.get(scheme, frozenset())),
        })
    return pd.DataFrame(rows, columns=COHORT_OVERLAP_COLUMNS)


def main() -> None:
    intersection_mrns = _intersection_mrns()

    frames = []
    for anchor in ANCHOR_LIST:
        frames.append(_natural_cohort_metrics(anchor))
        frames.append(_intersection_cohort_metrics(anchor, intersection_mrns))
    sensitivity_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=ANCHOR_SENSITIVITY_COLUMNS
    )
    save_figure_data(sensitivity_df, "fig2_anchor_sensitivity.csv")
    save_figure_data(_cohort_overlap(intersection_mrns), "fig2_anchor_cohort_overlap.csv")


if __name__ == "__main__":
    main()
