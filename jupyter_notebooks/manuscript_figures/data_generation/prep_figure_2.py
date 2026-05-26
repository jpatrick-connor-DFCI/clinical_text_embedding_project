"""Pre-compute inputs for Figure 2 (text vs base full-cohort prediction).

Writes to FIGURE_DATA_DIR:
- fig2_full_cohort_metrics.csv    scheme, event, text_cindex, base_cindex, text_auc, base_auc
- fig2_bootstrap_deltas.csv       event, delta, lo, hi, n
- fig2_km_death_data.csv          DFCI_MRN, text_risk_score, base_risk_score, death, tt_death,
                                  text_tertile, base_tertile
- fig2_td_auc.csv                 event, time_months, auc_text, auc_base
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _figure_utils import (
    SURV_PATH,
    full_cohort_event_dir, full_cohort_risk_dir, list_trained_events,
    save_figure_data,
)


SCHEMES = ["death_met", "icd3_post", "icd4_post", "phecode_post"]
DEATH_EVENTS = ["death", "brainM", "boneM", "adrenalM", "liverM", "lungM", "nodeM", "peritonealM"]
TD_AUC_EVENTS = ["death", "liverM", "brainM"]
EVAL_MONTHS = np.array([3, 6, 12, 18, 24, 36])
BOOTSTRAP_SEEDS = {event: i for i, event in enumerate(DEATH_EVENTS)}

FULL_COHORT_METRIC_COLUMNS = [
    "scheme", "event", "text_cindex", "base_cindex", "text_auc", "base_auc",
]
BOOTSTRAP_DELTA_COLUMNS = ["event", "delta", "lo", "hi", "n"]
KM_DEATH_COLUMNS = [
    "DFCI_MRN", "text_risk_score", "base_risk_score", "death", "tt_death",
    "text_tertile", "base_tertile",
]
TD_AUC_COLUMNS = ["event", "time_months", "auc_text", "auc_base"]


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


def _bootstrap_deltas(surv_df: pd.DataFrame, n_boot: int = 200) -> pd.DataFrame:
    rows = []
    for ev in DEATH_EVENTS:
        m = _merge_risk_with_surv(ev, surv_df)
        if m is None or len(m) < 100:
            print(f"  [{ev}] skipped — too few patients with risk + survival")
            continue
        # Per-event RNG so reruns of a single event are reproducible without
        # depending on the order of preceding events.
        rng = np.random.default_rng(BOOTSTRAP_SEEDS[ev])
        events = m[ev].values.astype(bool)
        times = m[f"tt_{ev}"].values.astype(float)
        text_rs = m["text_risk_score"].values
        base_rs = m["base_risk_score"].values
        text_c0 = concordance_index_censored(events, times, text_rs)[0]
        base_c0 = concordance_index_censored(events, times, base_rs)[0]
        deltas = []
        n = len(events)
        n_boot_failed = 0
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                tc = concordance_index_censored(events[idx], times[idx], text_rs[idx])[0]
                bc = concordance_index_censored(events[idx], times[idx], base_rs[idx])[0]
                deltas.append(tc - bc)
            except (ValueError, ZeroDivisionError):
                n_boot_failed += 1
                continue
        if n_boot_failed:
            print(f"  [{ev}] {n_boot_failed}/{n_boot} bootstrap iterations failed")
        deltas = np.array(deltas)
        if len(deltas) == 0:
            print(f"  [{ev}] skipped — all bootstrap iterations failed")
            continue
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        rows.append({"event": ev, "delta": text_c0 - base_c0, "lo": lo, "hi": hi, "n": n})
    return pd.DataFrame(rows, columns=BOOTSTRAP_DELTA_COLUMNS).sort_values("delta").reset_index(drop=True)


def _safe_tertiles(scores: pd.Series, label: str) -> pd.Series:
    """qcut into low/mid/high, falling back to rank-based bins when ties at
    boundaries would otherwise raise. Base-model risk often has many ties
    (e.g. age + sex + cancer-type alone collapses many patients onto identical
    linear predictors)."""
    labels = ["low", "mid", "high"]
    try:
        return pd.qcut(scores, 3, labels=labels).astype(str)
    except ValueError:
        # Rank-based fallback breaks ties by row order so every row gets a bin.
        ranks = scores.rank(method="first")
        out = pd.qcut(ranks, 3, labels=labels).astype(str)
        print(f"  [_km_death] {label}: qcut hit duplicate edges; used rank-based tertiles")
        return out


def _km_death(surv_df: pd.DataFrame) -> pd.DataFrame:
    m = _merge_risk_with_surv("death", surv_df)
    if m is None:
        return pd.DataFrame(columns=KM_DEATH_COLUMNS)
    m = m.copy()
    m["text_tertile"] = _safe_tertiles(m["text_risk_score"], "text")
    m["base_tertile"] = _safe_tertiles(m["base_risk_score"], "base")
    return m[["DFCI_MRN", "text_risk_score", "base_risk_score",
              "death", "tt_death", "text_tertile", "base_tertile"]]


def _td_auc(surv_df: pd.DataFrame) -> pd.DataFrame:
    eval_days = EVAL_MONTHS * 30.44
    rows = []
    for ev in TD_AUC_EVENTS:
        m = _merge_risk_with_surv(ev, surv_df)
        if m is None:
            print(f"  [{ev}] skipped — risk-score files not found")
            continue
        y = Surv.from_arrays(event=m[ev].astype(bool).values, time=m[f"tt_{ev}"].values)
        max_t = m[f"tt_{ev}"].max() * 0.9
        times = eval_days[eval_days < max_t]
        if len(times) == 0:
            print(f"  [{ev}] skipped — max follow-up {m[f'tt_{ev}'].max():.0f} days "
                  "too short for any evaluation timepoint")
            continue
        if len(times) < 3:
            print(f"  [{ev}] warning: only {len(times)} timepoints fit within follow-up "
                  f"(max {m[f'tt_{ev}'].max():.0f} days); AUC curve will be sparse")
        try:
            auc_text, _ = cumulative_dynamic_auc(y, y, m["text_risk_score"].values, times)
            auc_base, _ = cumulative_dynamic_auc(y, y, m["base_risk_score"].values, times)
        except ValueError as e:
            print(f"  [{ev}] cumulative_dynamic_auc failed: {e}")
            continue
        for t, at, ab in zip(times / 30.44, auc_text, auc_base):
            rows.append({"event": ev, "time_months": t, "auc_text": at, "auc_base": ab})
    return pd.DataFrame(rows, columns=TD_AUC_COLUMNS)


def main() -> None:
    surv_df = pd.read_csv(os.path.join(SURV_PATH, "death_met_surv_df.csv.gz"))

    save_figure_data(_full_cohort_metrics(), "fig2_full_cohort_metrics.csv")
    save_figure_data(_bootstrap_deltas(surv_df), "fig2_bootstrap_deltas.csv")
    save_figure_data(_km_death(surv_df), "fig2_km_death_data.csv")
    save_figure_data(_td_auc(surv_df), "fig2_td_auc.csv")


if __name__ == "__main__":
    main()
