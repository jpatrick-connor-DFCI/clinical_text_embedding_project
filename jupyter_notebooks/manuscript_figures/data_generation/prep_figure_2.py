"""Pre-compute inputs for Figure 2 (text vs base full-cohort prediction).

Writes to FIGURE_DATA_DIR:
- fig2_full_cohort_metrics.csv      scheme, event, text_cindex, base_cindex, text_auc, base_auc
- fig2_within_vs_pan_cancer.csv     stratum, auc_pan, auc_within, delta, cindex_pan, cindex_within,
                                    cindex_delta, n_heldout, is_overall
- fig2_within_vs_pan_treatment.csv  stratum, auc_pan, auc_within, delta, cindex_pan, cindex_within,
                                    cindex_delta, n_heldout, is_overall
- fig2_km_tertiles.csv              DFCI_MRN, text_risk_score, base_risk_score, death, tt_death,
                                    text_tertile, base_tertile
- fig2_km_stage_vs_risk.csv         DFCI_MRN, tt_death, death, text_risk_score, stage_group,
                                    stage_ordinal, risk_quartile   (known-stage cohort)
- fig2_stage_vs_risk_cindex.csv     predictor, cindex, n   (stage ordinal vs text risk score, OS)
- fig2_stage_vs_risk_cindex_by_stage.csv  stage_group, cindex, n   (within-stage C-index of text
                                    risk score for OS; FigS2 stage-panel annotation, cindex mode)
- fig2_stage_vs_risk_auc.csv        stage_group, mean_auc, n   (within-stage AUC(t) of text
                                    risk score for OS; FigS2 stage-panel annotation, auc mode)
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _figure_utils import (
    FEATURE_PATH, RESULTS_PATH, SURV_PATH,
    full_cohort_event_dir, full_cohort_risk_dir, list_trained_events,
    save_figure_data,
)

# Raw, complete stage labels (avoids the drop_first ambiguity in cancer_stage_df.csv.gz).
# Mirrors generate_all_non_text_covariates.py:17.
STAGE_PATH = (
    "/data/gusev/PROFILE/CLINICAL/OncDRS/DERIVED_FROM_CLINICAL_TEXTS_2024_03/"
    "derived_files/cancer_stage/dfci_cancer_mrn_to_derived_cancer_stage.pkl"
)


SCHEMES = ["death_met", "icd3_post", "icd4_post", "phecode_post"]

# Major-stage ordering / ordinal encoding for the stage-vs-risk comparison
STAGE_ORDER = ["I", "II", "III", "IV"]
STAGE_ORDINAL = {"I": 1, "II": 2, "III": 3, "IV": 4}
RISK_QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]

FULL_COHORT_METRIC_COLUMNS = [
    "scheme", "event", "text_cindex", "base_cindex", "text_auc", "base_auc",
]
WITHIN_VS_PAN_COLUMNS = [
    "stratum", "auc_pan", "auc_within", "delta",
    "cindex_pan", "cindex_within", "cindex_delta",
    "n_heldout", "is_overall",
]
# (subdir, filename, stratum-column) for each within-vs-pan comparison written by Pipeline 3.
_WITHIN_VS_PAN_SPEC = {
    "cancer":    ("pan_vs_within_cancer",    "metrics_by_cancer_type.csv", "CANCER_TYPE"),
    "treatment": ("pan_vs_within_treatment", "metrics_by_treatment.csv",   "TREATMENT"),
}
KM_TERTILE_COLUMNS = [
    "DFCI_MRN", "text_risk_score", "base_risk_score", "death", "tt_death",
    "text_tertile", "base_tertile",
]
STAGE_VS_RISK_COLUMNS = [
    "DFCI_MRN", "tt_death", "death", "text_risk_score",
    "stage_group", "stage_ordinal", "risk_quartile",
]
STAGE_VS_RISK_CINDEX_COLUMNS = ["predictor", "cindex", "n"]
STAGE_VS_RISK_AUC_COLUMNS = ["stage_group", "mean_auc", "n"]
# Number of evaluation points on the 5th-95th percentile time grid, matching
# within_vs_pan_cancer_models.py's mean-AUC(t) convention.
AUC_TIME_GRID_POINTS = 50


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


def _safe_quantiles(scores: pd.Series, n: int, labels: list[str], label: str) -> pd.Series:
    """qcut into n equal-frequency bins, falling back to rank-based bins when ties at
    boundaries would otherwise raise. Base-model risk often has many ties
    (e.g. age + sex + cancer-type alone collapses many patients onto identical
    linear predictors)."""
    try:
        return pd.qcut(scores, n, labels=labels).astype(str)
    except ValueError:
        ranks = scores.rank(method="first")
        out = pd.qcut(ranks, n, labels=labels).astype(str)
        print(f"  [{label}] qcut hit duplicate edges; used rank-based bins (n={n})")
        return out


def _safe_tertiles(scores: pd.Series, label: str) -> pd.Series:
    """Backwards-compatible low/mid/high tertiles (used by the Fig 2D panel)."""
    return _safe_quantiles(scores, 3, ["low", "mid", "high"], label)


def _km_tertiles(surv_df: pd.DataFrame) -> pd.DataFrame:
    """Patient-level table for the Fig 2D text-vs-base tertile KM panel."""
    m = _merge_risk_with_surv("death", surv_df)
    if m is None:
        return pd.DataFrame(columns=KM_TERTILE_COLUMNS)
    m = m.copy()
    m["text_tertile"] = _safe_tertiles(m["text_risk_score"], "text")
    m["base_tertile"] = _safe_tertiles(m["base_risk_score"], "base")
    return m[KM_TERTILE_COLUMNS]


_STAGE_TOKEN = re.compile(r"^(IV|III|II|I|4|3|2|1)[A-D]?$")


def _normalize_stage(raw) -> str | None:
    """Map a raw stage string to a major stage in {I, II, III, IV}, collapsing
    substages (e.g. IVA -> IV), arabic numerals (4 -> IV), and float repr (2.0 -> II).
    Returns None for unknown / in-situ (0) / unstageable values so they can be dropped."""
    if pd.isna(raw):
        return None
    s = str(raw).upper().strip().replace("STAGE", "").strip()
    s = re.sub(r"\.0+$", "", s)  # the source pickle stores stages as floats (e.g. 2.0)
    m = _STAGE_TOKEN.match(s)
    if not m:
        return None
    arabic_to_roman = {"1": "I", "2": "II", "3": "III", "4": "IV"}
    token = m.group(1)
    return arabic_to_roman.get(token, token)


def _major_stage_labels() -> pd.DataFrame:
    """DFCI_MRN -> stage_group (I/II/III/IV) from the raw derived-stage pickle.

    Falls back to reconstructing from the one-hot cancer_stage_df.csv.gz when the
    pickle is unreadable (all-zero rows = the drop_first reference stage)."""
    try:
        with open(STAGE_PATH, "rb") as f:
            mrn_to_stage = pickle.load(f)
        df = pd.DataFrame({"DFCI_MRN": list(mrn_to_stage.keys()),
                           "stage_group": [_normalize_stage(v) for v in mrn_to_stage.values()]})
    except (FileNotFoundError, OSError, pickle.UnpicklingError) as e:
        print(f"  stage pickle unavailable ({type(e).__name__}); reconstructing from one-hot CSV")
        oh = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_stage_df.csv.gz"))
        stage_cols = [c for c in oh.columns if c.startswith("CANCER_STAGE_")]
        present = [_normalize_stage(c.replace("CANCER_STAGE_", "")) for c in stage_cols]
        # drop_first reference = the single major stage absent from the columns
        reference = next((s for s in STAGE_ORDER if s not in present), None)
        active = oh[stage_cols].to_numpy()
        labels = []
        for i in range(len(oh)):
            row = active[i]
            if row.max() <= 0:
                labels.append(reference)
            else:
                labels.append(present[int(row.argmax())])
        df = pd.DataFrame({"DFCI_MRN": oh["DFCI_MRN"], "stage_group": labels})

    df = df.dropna(subset=["stage_group"])
    df = df[df["stage_group"].isin(STAGE_ORDER)].drop_duplicates(subset=["DFCI_MRN"])
    print("  normalized stage value_counts:\n"
          + df["stage_group"].value_counts().reindex(STAGE_ORDER).to_string())
    return df


def _stage_vs_risk(surv_df: pd.DataFrame) -> pd.DataFrame:
    """Patient-level table for the Fig 2E stage-vs-text-risk KM panel.

    Cohort = patients with a known major stage AND a death text risk score; this same
    set drives both KM subpanels and both c-indices for an apples-to-apples comparison."""
    m = _merge_risk_with_surv("death", surv_df)
    if m is None:
        return pd.DataFrame(columns=STAGE_VS_RISK_COLUMNS)
    stage_df = _major_stage_labels()
    m = m.merge(stage_df, on="DFCI_MRN", how="inner")
    if m.empty:
        return pd.DataFrame(columns=STAGE_VS_RISK_COLUMNS)
    m = m.copy()
    m["stage_ordinal"] = m["stage_group"].map(STAGE_ORDINAL)
    m["risk_quartile"] = _safe_quantiles(m["text_risk_score"], 4, RISK_QUARTILE_LABELS, "text_risk")
    return m[STAGE_VS_RISK_COLUMNS]


def _stage_vs_risk_cindex(df: pd.DataFrame) -> pd.DataFrame:
    """Concordance of clinical stage (ordinal) vs text risk score for predicting OS,
    on the shared cohort from _stage_vs_risk. Higher estimate = higher risk for both."""
    if df.empty:
        return pd.DataFrame(columns=STAGE_VS_RISK_CINDEX_COLUMNS)
    event = df["death"].astype(bool).to_numpy()
    time = df["tt_death"].astype(float).to_numpy()
    rows = []
    for predictor, estimate in [
        ("stage", df["stage_ordinal"].astype(float).to_numpy()),
        ("text_risk", df["text_risk_score"].astype(float).to_numpy()),
    ]:
        try:
            cidx = concordance_index_censored(event, time, estimate)[0]
        except (ValueError, ZeroDivisionError) as e:
            print(f"  c-index failed for {predictor}: {e}")
            cidx = float("nan")
        rows.append({"predictor": predictor, "cindex": cidx, "n": len(df)})
    return pd.DataFrame(rows, columns=STAGE_VS_RISK_CINDEX_COLUMNS)


def _stage_vs_risk_cindex_by_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Concordance of the text risk score for OS, within each stage.

    Per-stage analogue of _stage_vs_risk_cindex (which reports one pooled
    cindex for the whole known-stage cohort), matching the per-stage grouping
    of _stage_vs_risk_auc so FigS2 can annotate either metric per stage panel.
    """
    if df.empty:
        return pd.DataFrame(columns=STAGE_VS_RISK_AUC_COLUMNS[:1] + ["cindex", "n"])
    rows = []
    for stage_lbl, sub in df.groupby("stage_group"):
        event = sub["death"].astype(bool).to_numpy()
        time = sub["tt_death"].astype(float).to_numpy()
        try:
            cidx = concordance_index_censored(
                event, time, sub["text_risk_score"].astype(float).to_numpy())[0]
        except (ValueError, ZeroDivisionError) as e:
            print(f"  c-index failed for stage {stage_lbl}: {e}")
            cidx = float("nan")
        rows.append({"stage_group": stage_lbl, "cindex": cidx, "n": len(sub)})
    return pd.DataFrame(rows, columns=["stage_group", "cindex", "n"])


def _stage_vs_risk_auc(df: pd.DataFrame) -> pd.DataFrame:
    """Mean time-dependent AUC of the text risk score for OS, within each stage.

    IPCW reference + eval-time grid (5th-95th percentile, 50 points) are fit on the
    pooled known-stage cohort (this table has no train/test split, unlike
    within_vs_pan_cancer_models.py's held-out data), then cumulative_dynamic_auc is
    evaluated per stage subgroup against that shared reference — mirroring the
    project-standard mean-AUC(t) definition used everywhere else in the pipeline.
    """
    if df.empty:
        return pd.DataFrame(columns=STAGE_VS_RISK_AUC_COLUMNS)
    y_ref = Surv.from_arrays(df["death"].astype(bool), df["tt_death"].astype(float))
    lo, hi = np.percentile(df["tt_death"].astype(float), [5, 95])
    base_eval_times = np.linspace(lo, hi, AUC_TIME_GRID_POINTS)

    rows = []
    for stage_lbl, sub in df.groupby("stage_group"):
        et = base_eval_times[(base_eval_times > sub["tt_death"].min())
                             & (base_eval_times < sub["tt_death"].max())]
        mean_auc = float("nan")
        if len(et) > 0:
            try:
                y_test = Surv.from_arrays(sub["death"].astype(bool), sub["tt_death"].astype(float))
                mean_auc = cumulative_dynamic_auc(
                    y_ref, y_test, sub["text_risk_score"].astype(float).to_numpy(), et,
                )[1]
            except (ValueError, ZeroDivisionError) as e:
                print(f"  AUC(t) failed for stage {stage_lbl}: {e}")
        rows.append({"stage_group": stage_lbl, "mean_auc": mean_auc, "n": len(sub)})
    return pd.DataFrame(rows, columns=STAGE_VS_RISK_AUC_COLUMNS)


def _within_vs_pan(kind: str) -> pd.DataFrame:
    """Read a Pipeline-3 pan-vs-within metrics CSV (mean time-dependent AUC + C-index).

    Maps the upstream schema to the unified figure schema, keeps the `Overall`
    row, drops NaN-AUC strata, and applies an n>=30 floor to per-stratum rows
    (the treatment script writes all strata; the cancer script already filters).
    Returns a header-only frame if the upstream file is missing or pre-dates the
    AUC columns (re-run the Pipeline-3 script), so the R panel degrades gracefully.
    C-index columns are passed through when present so the R side can render a
    parallel C-index version of this panel (falls back to NaN otherwise).
    """
    subdir, fname, stratum_col = _WITHIN_VS_PAN_SPEC[kind]
    fp = os.path.join(RESULTS_PATH, subdir, fname)
    if not os.path.exists(fp):
        print(f"  missing {fp}; skipping within-vs-pan {kind}")
        return pd.DataFrame(columns=WITHIN_VS_PAN_COLUMNS)
    df = pd.read_csv(fp)
    if not {stratum_col, "AUC_PAN", "AUC_WITHIN", "N_HELDOUT"}.issubset(df.columns):
        print(f"  {fp} has no AUC columns — re-run the Pipeline-3 script; skipping {kind}")
        return pd.DataFrame(columns=WITHIN_VS_PAN_COLUMNS)
    has_cindex = {"CINDEX_PAN", "CINDEX_WITHIN"}.issubset(df.columns)
    if not has_cindex:
        print(f"  {fp} has no CINDEX columns — re-run the Pipeline-3 script for cindex panel")
    out = pd.DataFrame({
        "stratum": df[stratum_col].astype(str),
        "auc_pan": df["AUC_PAN"],
        "auc_within": df["AUC_WITHIN"],
        "delta": df["DELTA_AUC_WITHIN_MINUS_PAN"] if "DELTA_AUC_WITHIN_MINUS_PAN" in df
                 else df["AUC_WITHIN"] - df["AUC_PAN"],
        "cindex_pan": df["CINDEX_PAN"] if has_cindex else np.nan,
        "cindex_within": df["CINDEX_WITHIN"] if has_cindex else np.nan,
        "n_heldout": df["N_HELDOUT"],
    })
    out["cindex_delta"] = out["cindex_within"] - out["cindex_pan"]
    out["is_overall"] = out["stratum"] == "Overall"
    out = out.dropna(subset=["auc_pan", "auc_within"])
    out = out[out["is_overall"] | (out["n_heldout"] >= 30)]
    return out[WITHIN_VS_PAN_COLUMNS]


def main() -> None:
    surv_df = pd.read_csv(os.path.join(SURV_PATH, "death_met_surv_df.csv.gz"))

    save_figure_data(_full_cohort_metrics(), "fig2_full_cohort_metrics.csv")
    save_figure_data(_within_vs_pan("cancer"), "fig2_within_vs_pan_cancer.csv")
    save_figure_data(_within_vs_pan("treatment"), "fig2_within_vs_pan_treatment.csv")
    save_figure_data(_km_tertiles(surv_df), "fig2_km_tertiles.csv")

    stage_vs_risk_df = _stage_vs_risk(surv_df)
    save_figure_data(stage_vs_risk_df, "fig2_km_stage_vs_risk.csv")
    save_figure_data(_stage_vs_risk_cindex(stage_vs_risk_df), "fig2_stage_vs_risk_cindex.csv")
    save_figure_data(_stage_vs_risk_cindex_by_stage(stage_vs_risk_df),
                      "fig2_stage_vs_risk_cindex_by_stage.csv")
    save_figure_data(_stage_vs_risk_auc(stage_vs_risk_df), "fig2_stage_vs_risk_auc.csv")


if __name__ == "__main__":
    main()
