"""Pre-compute inputs for Figure 2 (text vs base full-cohort prediction).

Writes to FIGURE_DATA_DIR:
- fig2_full_cohort_metrics.csv      scheme, event, text_cindex, base_cindex, text_auc, base_auc
- fig2_cancer_endpoint_heatmap.csv  event, cancer_type, text_cindex, n, n_events
- fig2_km_examples.csv              DFCI_MRN, event, text_risk_score, event_indicator, time,
                                    text_tertile
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _figure_utils import (
    FEATURE_PATH, SURV_PATH,
    full_cohort_event_dir, full_cohort_risk_dir, list_trained_events,
    save_figure_data,
)


SCHEMES = ["death_met", "icd3_post", "icd4_post", "phecode_post"]
DEATH_EVENTS = ["death", "brainM", "boneM", "adrenalM", "liverM", "lungM", "nodeM", "peritonealM"]

FULL_COHORT_METRIC_COLUMNS = [
    "scheme", "event", "text_cindex", "base_cindex", "text_auc", "base_auc",
]
CANCER_ENDPOINT_COLUMNS = ["event", "cancer_type", "text_cindex", "n", "n_events"]
KM_EXAMPLE_COLUMNS = ["DFCI_MRN", "event", "text_risk_score", "event_indicator", "time", "text_tertile"]


def _full_cohort_metrics() -> pd.DataFrame:
    rows = []
    n_skipped = 0
    for scheme in SCHEMES:
        for ev in list_trained_events(scheme):
            d = full_cohort_event_dir(scheme, ev)
            try:
                text = pd.read_csv(os.path.join(d, "text_test.csv")).iloc[0]
                base = pd.read_csv(os.path.join(d, "base_test.csv")).iloc[0]
            except (FileNotFoundError, KeyError, IndexError) as e:
                print(f"  [{scheme}:{ev}] skipped — {type(e).__name__}: {e}")
                n_skipped += 1
                continue
            rows.append({
                "scheme": scheme, "event": ev,
                "text_cindex": text["mean_c_index"], "base_cindex": base["mean_c_index"],
                "text_auc": text["mean_auc(t)"], "base_auc": base["mean_auc(t)"],
            })
    if n_skipped:
        print(f"  total skipped: {n_skipped}")
    return pd.DataFrame(rows, columns=FULL_COHORT_METRIC_COLUMNS)


def _merge_risk_with_surv(event: str, surv_df: pd.DataFrame) -> pd.DataFrame | None:
    rd = full_cohort_risk_dir("death_met", event)
    tp = os.path.join(rd, "text_risk_scores.csv")
    bp = os.path.join(rd, "base_risk_scores.csv")
    if not (os.path.exists(tp) and os.path.exists(bp)):
        return None
    text_rs = pd.read_csv(tp)
    base_rs = pd.read_csv(bp)
    merged = (text_rs.merge(base_rs, on="DFCI_MRN")
                     .merge(surv_df[["DFCI_MRN", event, f"tt_{event}"]], on="DFCI_MRN")
                     .dropna())
    merged = merged[merged[f"tt_{event}"] > 0]
    return merged


def _safe_tertiles(scores: pd.Series, label: str) -> pd.Series:
    """qcut into low/mid/high, falling back to rank-based bins when ties at
    boundaries would otherwise raise. Base-model risk often has many ties
    (e.g. age + sex + cancer-type alone collapses many patients onto identical
    linear predictors)."""
    labels = ["low", "mid", "high"]
    try:
        return pd.qcut(scores, 3, labels=labels).astype(str)
    except ValueError:
        ranks = scores.rank(method="first")
        out = pd.qcut(ranks, 3, labels=labels).astype(str)
        print(f"  [{label}] qcut hit duplicate edges; used rank-based tertiles")
        return out


def _cancer_type_labels(cancer_df: pd.DataFrame) -> pd.DataFrame:
    type_cols = [c for c in cancer_df.columns if c.startswith("CANCER_TYPE_")]
    if "CANCER_TYPE" in cancer_df.columns:
        labels = cancer_df["CANCER_TYPE"].astype(str)
    elif type_cols:
        labels = cancer_df[type_cols].apply(pd.to_numeric, errors="coerce").fillna(0).idxmax(axis=1)
        labels = labels.str.replace("CANCER_TYPE_", "", regex=False)
    else:
        return pd.DataFrame(columns=["DFCI_MRN", "cancer_type"])
    return pd.DataFrame({
        "DFCI_MRN": cancer_df["DFCI_MRN"],
        "cancer_type": labels.astype(str).str.replace("_", " ", regex=False),
    })


def _cancer_endpoint_heatmap(surv_df: pd.DataFrame, top_n_cancers: int = 8,
                             min_n: int = 50, min_events: int = 5) -> pd.DataFrame:
    cancer_df = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_type_df.csv.gz"))
    labels = _cancer_type_labels(cancer_df)
    if labels.empty:
        return pd.DataFrame(columns=CANCER_ENDPOINT_COLUMNS)
    top_cancers = labels["cancer_type"].value_counts().head(top_n_cancers).index.tolist()
    rows = []
    for ev in DEATH_EVENTS:
        m = _merge_risk_with_surv(ev, surv_df)
        if m is None:
            continue
        merged = m.merge(labels, on="DFCI_MRN", how="inner")
        for cancer_type in top_cancers:
            sub = merged[merged["cancer_type"] == cancer_type].dropna(subset=["text_risk_score", ev, f"tt_{ev}"])
            n = len(sub)
            n_events = int(sub[ev].sum()) if n else 0
            if n < min_n or n_events < min_events:
                continue
            try:
                cidx = concordance_index_censored(
                    sub[ev].astype(bool).values,
                    sub[f"tt_{ev}"].astype(float).values,
                    sub["text_risk_score"].astype(float).values,
                )[0]
            except (ValueError, ZeroDivisionError):
                continue
            rows.append({
                "event": ev,
                "cancer_type": cancer_type,
                "text_cindex": cidx,
                "n": n,
                "n_events": n_events,
            })
    return pd.DataFrame(rows, columns=CANCER_ENDPOINT_COLUMNS)


def _km_examples(surv_df: pd.DataFrame, events: list[str] | None = None) -> pd.DataFrame:
    events = events or ["death", "liverM", "brainM"]
    rows = []
    for ev in events:
        m = _merge_risk_with_surv(ev, surv_df)
        if m is None or len(m) < 100:
            continue
        m = m.copy()
        m["text_tertile"] = _safe_tertiles(m["text_risk_score"], f"text/{ev}")
        cur = m[["DFCI_MRN", "text_risk_score", ev, f"tt_{ev}", "text_tertile"]].rename(
            columns={ev: "event_indicator", f"tt_{ev}": "time"}
        )
        cur["event"] = ev
        rows.append(cur[KM_EXAMPLE_COLUMNS])
    if not rows:
        return pd.DataFrame(columns=KM_EXAMPLE_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    surv_df = pd.read_csv(os.path.join(SURV_PATH, "death_met_surv_df.csv.gz"))

    save_figure_data(_full_cohort_metrics(), "fig2_full_cohort_metrics.csv")
    save_figure_data(_cancer_endpoint_heatmap(surv_df), "fig2_cancer_endpoint_heatmap.csv")
    save_figure_data(_km_examples(surv_df), "fig2_km_examples.csv")


if __name__ == "__main__":
    main()
