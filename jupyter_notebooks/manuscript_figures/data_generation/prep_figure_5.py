"""Pre-compute inputs for Figure 5 (ICI biomarker discovery).

Writes to FIGURE_DATA_DIR:
- fig5_ps_predictions.csv         DFCI_MRN, ps_model, model_probs, ground_truth,
                                  cancer_type
- fig5_volcano_track2.csv         marker, cancer, HR_markerxICI, p_markerxICI,
                                  log_hr, neglog10_p, significant
- fig5_robust_hits.csv            marker, cancer_type, spec, HR_markerxICI,
                                  CI95_marker_ICI_low, CI95_marker_ICI_high,
                                  p_markerxICI, n_specs, mean_HR,
                                  direction_consistent
- fig5_km_top_hit.csv             DFCI_MRN, marker_value, PX_on_ICI, death, tt_death
- fig5_km_examples.csv            example_id, title, marker, cancer, marker_value, PX_on_ICI,
                                  death, tt_death
- fig5_top_hit_meta.csv           marker, cancer, ps_model
- fig5_love_smd.csv               covariate, smd_unweighted, smd_weighted  (primary-spec balance)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _figure_utils import (
    BIOMARKER_PATH, DATA_PATH, FEATURE_PATH,
    save_figure_data,
)


COHORT = "cohort2"
PS_BUFFER = 30
TRACK2_WEIGHT = "ATE"
PRIMARY_PS_MODEL = "covariates_plus_embeddings"
COMPILED_DIR = os.path.join(BIOMARKER_PATH, "compiled_results/")
PS_BASE = os.path.join(DATA_PATH, f"treatment_prediction/{COHORT}/")
PRIMARY_SPEC_DIR = os.path.join(BIOMARKER_PATH, f"IPTW_runs_{COHORT}_covariates_plus_embeddings/")

PS_PREDICTION_COLUMNS = ["DFCI_MRN", "ps_model", "model_probs", "ground_truth", "cancer_type", "cohort"]
VOLCANO_COLUMNS = [
    "marker", "cancer", "HR_markerxICI", "p_markerxICI",
    "log_hr", "neglog10_p", "significant",
]
ROBUST_COLUMNS = [
    "marker", "cancer_type", "spec", "HR_markerxICI",
    "CI95_marker_ICI_low", "CI95_marker_ICI_high", "p_markerxICI",
    "cohort", "ps_model", "weight_type", "n_specs", "mean_HR",
    "direction_consistent", "n_significant_markers",
]
KM_COLUMNS = ["DFCI_MRN", "marker_value", "PX_on_ICI", "death", "tt_death"]
KM_EXAMPLE_COLUMNS = [
    "DFCI_MRN", "example_id", "title", "marker", "cancer", "marker_value",
    "PX_on_ICI", "death", "tt_death", "hr",
]
TOP_HIT_META_COLUMNS = ["marker", "cancer", "cohort", "ps_model", "weight_type"]
LOVE_SMD_COLUMNS = ["covariate", "smd_unweighted", "smd_weighted"]
LOVE_TOP_N = 15  # covariates shown beyond the always-kept demographics


def _patient_cancer_type() -> pd.DataFrame:
    fp = os.path.join(FEATURE_PATH, "cancer_type_df.csv.gz")
    if not os.path.exists(fp):
        return pd.DataFrame(columns=["DFCI_MRN", "cancer_type"])
    cancer = pd.read_csv(fp)
    type_cols = [c for c in cancer.columns if c.startswith("CANCER_TYPE_")]
    if "DFCI_MRN" not in cancer.columns or not type_cols:
        return pd.DataFrame(columns=["DFCI_MRN", "cancer_type"])
    out = cancer[["DFCI_MRN"] + type_cols].copy()
    out["cancer_type"] = out[type_cols].idxmax(axis=1).str.replace("CANCER_TYPE_", "", regex=False)
    return out[["DFCI_MRN", "cancer_type"]]


def _ps_predictions() -> pd.DataFrame:
    cancer = _patient_cancer_type()
    rows = []
    for ps_model in ("covariates_only", "covariates_plus_embeddings"):
        fp = os.path.join(PS_BASE, f"{ps_model}_propensity/w_{PS_BUFFER}_day_buffer/predictions.csv.gz")
        if not os.path.exists(fp):
            print(f"  missing {fp}")
            continue
        d = pd.read_csv(fp)
        required = {"DFCI_MRN", "model_probs", "ground_truth"}
        if not required.issubset(d.columns):
            print(f"  skipping {fp}: missing {sorted(required - set(d.columns))}")
            continue
        d = d[["DFCI_MRN", "model_probs", "ground_truth"]].copy()
        d["ps_model"] = ps_model
        d["cohort"] = COHORT
        if not cancer.empty:
            d = d.merge(cancer, on="DFCI_MRN", how="left")
        else:
            d["cancer_type"] = pd.NA
        rows.append(d.reindex(columns=PS_PREDICTION_COLUMNS))
    if not rows:
        return pd.DataFrame(columns=PS_PREDICTION_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def _volcano_track2() -> pd.DataFrame:
    if not os.path.isdir(PRIMARY_SPEC_DIR):
        print(f"  no primary spec dir at {PRIMARY_SPEC_DIR}")
        return pd.DataFrame(columns=VOLCANO_COLUMNS)
    files = [os.path.join(PRIMARY_SPEC_DIR, f) for f in os.listdir(PRIMARY_SPEC_DIR)
             if f.endswith(f"_track2_{TRACK2_WEIGHT}_interaction.csv.gz")]
    if not files:
        return pd.DataFrame(columns=VOLCANO_COLUMNS)
    frames = [pd.read_csv(fp).assign(cancer=os.path.basename(fp).split("_track2_")[0])
              for fp in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["HR_markerxICI", "p_markerxICI"]).copy()
    df["log_hr"] = np.log(df["HR_markerxICI"])
    df["neglog10_p"] = -np.log10(df["p_markerxICI"].clip(lower=1e-300))
    df["significant"] = df.get("significant_predictive",
                                pd.Series(False, index=df.index)).astype(bool)
    return df[["marker", "cancer", "HR_markerxICI", "p_markerxICI",
               "log_hr", "neglog10_p", "significant"]]


def _robust_hits() -> pd.DataFrame:
    fp = os.path.join(COMPILED_DIR, "track2_all_significant_hits.csv.gz")
    if not os.path.exists(fp):
        print(f"  missing {fp}")
        return pd.DataFrame(columns=ROBUST_COLUMNS)
    try:
        hits = pd.read_csv(fp)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=ROBUST_COLUMNS)
    if hits.empty:
        return pd.DataFrame(columns=ROBUST_COLUMNS)
    hits["spec"] = (hits["cohort"].astype(str) + "|" +
                    hits["ps_model"].astype(str) + "|" +
                    hits["weight_type"].astype(str))

    # Geometric mean for HR aggregation: HRs are multiplicative, so the
    # arithmetic mean over-weights large HRs. Drop pathological HRs (<=0, inf,
    # NaN, or already-flagged extremes) before averaging so a single model-
    # separation row can't dominate the geometric mean.
    EXTREME = 50.0

    def _is_clean(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        return s.gt(0) & np.isfinite(s) & s.between(1 / EXTREME, EXTREME)

    def _clean_geomean(s: pd.Series) -> float:
        s = pd.to_numeric(s, errors="coerce")
        s = s[_is_clean(s)]
        if s.empty:
            return float("nan")
        return float(np.exp(np.log(s).mean()))

    def _clean_direction(s: pd.Series) -> bool:
        s = pd.to_numeric(s, errors="coerce")
        s = s[_is_clean(s)]
        if s.empty:
            return False
        return bool((s > 1).all() or (s < 1).all())

    def _n_clean_specs(s: pd.Series) -> int:
        # Count only specs whose HR survives the cleaning filter, so n_specs
        # matches what actually contributed to mean_HR / direction_consistent.
        return int(_is_clean(s).sum())

    grp = (hits.groupby(["marker", "cancer_type"])
                .agg(n_specs=("HR_markerxICI", _n_clean_specs),
                     mean_HR=("HR_markerxICI", _clean_geomean),
                     direction_consistent=("HR_markerxICI", _clean_direction))
                .reset_index())
    robust = grp[(grp["n_specs"] >= 2) & grp["direction_consistent"]].copy()
    cols = ["marker", "cancer_type", "spec", "HR_markerxICI",
            "CI95_marker_ICI_low", "CI95_marker_ICI_high", "p_markerxICI"]
    cols = [c for c in cols if c in hits.columns]
    out = (hits.merge(robust[["marker", "cancer_type"]],
                       on=["marker", "cancer_type"], how="inner")
                [cols + ["cohort", "ps_model", "weight_type"]]
                .merge(robust, on=["marker", "cancer_type"], how="left"))
    # Denominator for the survivorship-aware panel B annotation: markers significant
    # in >=1 spec (the universe these robust hits were filtered from).
    out["n_significant_markers"] = hits[["marker", "cancer_type"]].drop_duplicates().shape[0]
    return out.reindex(columns=ROBUST_COLUMNS)


def _km_top_hit() -> tuple[pd.DataFrame, pd.DataFrame]:
    t2_fp = os.path.join(COMPILED_DIR, "track2_all_significant_hits.csv.gz")
    if not os.path.exists(t2_fp):
        return pd.DataFrame(columns=KM_COLUMNS), pd.DataFrame(columns=TOP_HIT_META_COLUMNS)
    try:
        hits = pd.read_csv(t2_fp)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=KM_COLUMNS), pd.DataFrame(columns=TOP_HIT_META_COLUMNS)
    if hits.empty:
        return pd.DataFrame(columns=KM_COLUMNS), pd.DataFrame(columns=TOP_HIT_META_COLUMNS)
    # Restrict "top hit" to the primary spec so panel D shows what the methods
    # section claims is the primary analysis, not the most-extreme p-value
    # across all sensitivity specifications.
    primary = hits[
        (hits["ps_model"] == PRIMARY_PS_MODEL)
        & (hits["weight_type"].str.upper() == TRACK2_WEIGHT.upper())
        & (hits["cohort"] == COHORT)
    ]
    if primary.empty:
        print(f"  no hits in primary spec ({COHORT}, {PRIMARY_PS_MODEL}, {TRACK2_WEIGHT}); "
              "falling back to overall best p")
        primary = hits
    top = primary.sort_values("p_markerxICI").iloc[0]
    marker = top["marker"]
    cancer = top["cancer_type"]
    cohort = top["cohort"]
    ps_model = top["ps_model"]
    weight_type = top.get("weight_type", TRACK2_WEIGHT)
    print(f"  top hit: {marker} in {cancer} ({cohort}, {ps_model}, {weight_type})")
    iptw_fp = os.path.join(BIOMARKER_PATH, f"IPTW_df_{cohort}_{ps_model}.csv.gz")
    if not os.path.exists(iptw_fp):
        print(f"  missing {iptw_fp}")
        return pd.DataFrame(columns=KM_COLUMNS), pd.DataFrame(columns=TOP_HIT_META_COLUMNS)
    iptw = pd.read_csv(iptw_fp)
    cancer_col = f"CANCER_TYPE_{cancer}"
    if cancer != "pan_cancer" and cancer_col in iptw.columns:
        iptw = iptw[iptw[cancer_col] == 1]
    iptw = iptw.dropna(subset=[marker, "PX_on_ICI", "tt_death", "death"])
    iptw = iptw[iptw["tt_death"] > 0]
    km = iptw[["DFCI_MRN", marker, "PX_on_ICI", "death", "tt_death"]].rename(
        columns={marker: "marker_value"}
    )
    meta = pd.DataFrame([{
        "marker": marker, "cancer": cancer, "cohort": cohort, "ps_model": ps_model,
        "weight_type": weight_type,
    }])
    return km, meta


def _select_km_example_hits(hits: pd.DataFrame, max_examples: int = 3) -> pd.DataFrame:
    primary = hits[
        (hits["ps_model"] == PRIMARY_PS_MODEL)
        & (hits["weight_type"].str.upper() == TRACK2_WEIGHT.upper())
        & (hits["cohort"] == COHORT)
    ].copy()
    if primary.empty:
        primary = hits.copy()
    primary = primary.dropna(subset=["marker", "cancer_type", "p_markerxICI", "HR_markerxICI"])
    if primary.empty:
        return primary
    primary = primary.sort_values("p_markerxICI")

    chosen = []
    benefit = primary[primary["HR_markerxICI"] < 1]
    harm = primary[primary["HR_markerxICI"] > 1]
    if not benefit.empty:
        chosen.append(benefit.iloc[0])
    if not harm.empty:
        chosen.append(harm.iloc[0])
    for _, row in primary.iterrows():
        key = (row["marker"], row["cancer_type"])
        if any((r["marker"], r["cancer_type"]) == key for r in chosen):
            continue
        chosen.append(row)
        if len(chosen) >= max_examples:
            break
    return pd.DataFrame(chosen[:max_examples])


def _km_examples() -> pd.DataFrame:
    t2_fp = os.path.join(COMPILED_DIR, "track2_all_significant_hits.csv.gz")
    if not os.path.exists(t2_fp):
        return pd.DataFrame(columns=KM_EXAMPLE_COLUMNS)
    try:
        hits = pd.read_csv(t2_fp)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=KM_EXAMPLE_COLUMNS)
    if hits.empty:
        return pd.DataFrame(columns=KM_EXAMPLE_COLUMNS)

    examples = _select_km_example_hits(hits)
    if examples.empty:
        return pd.DataFrame(columns=KM_EXAMPLE_COLUMNS)

    frames = []
    for i, row in enumerate(examples.itertuples(index=False), start=1):
        marker = getattr(row, "marker")
        cancer = getattr(row, "cancer_type")
        cohort = getattr(row, "cohort")
        ps_model = getattr(row, "ps_model")
        # NOTE: HR_markerxICI is the interaction-term HR; CI95_marker_ICI_* upstream
        # are the *marker main-effect* CI (different term) and would mismatch, so we
        # do not carry them through. run_IPTW_analysis emits no SE for the interaction.
        hr = getattr(row, "HR_markerxICI")
        iptw_fp = os.path.join(BIOMARKER_PATH, f"IPTW_df_{cohort}_{ps_model}.csv.gz")
        if not os.path.exists(iptw_fp):
            print(f"  missing {iptw_fp}")
            continue
        iptw = pd.read_csv(iptw_fp)
        cancer_col = f"CANCER_TYPE_{cancer}"
        if cancer != "pan_cancer" and cancer_col in iptw.columns:
            iptw = iptw[iptw[cancer_col] == 1]
        required = [marker, "PX_on_ICI", "tt_death", "death"]
        if not set(required).issubset(iptw.columns):
            print(f"  skipping KM example {marker}/{cancer}: missing required columns")
            continue
        cur = iptw.dropna(subset=required).copy()
        cur = cur[cur["tt_death"] > 0]
        if cur.empty:
            continue
        direction = "ICI Benefit" if hr < 1 else "ICI Harm"
        title = f"{marker} - {direction}"
        cur = cur[["DFCI_MRN", marker, "PX_on_ICI", "death", "tt_death"]].rename(
            columns={marker: "marker_value"}
        )
        cur["example_id"] = i
        cur["title"] = title
        cur["marker"] = marker
        cur["cancer"] = cancer
        cur["hr"] = hr
        frames.append(cur.reindex(columns=KM_EXAMPLE_COLUMNS))
    if not frames:
        return pd.DataFrame(columns=KM_EXAMPLE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _smd(x: np.ndarray, treat: np.ndarray, w: np.ndarray | None = None) -> float:
    """Standardized mean difference between treated/control (optionally IPTW-weighted)."""
    t, c = treat == 1, treat == 0
    if x[t].size == 0 or x[c].size == 0:
        return np.nan
    if w is None:
        mt, mc = x[t].mean(), x[c].mean()
        vt, vc = x[t].var(ddof=1), x[c].var(ddof=1)
    else:
        wt, wc = w[t], w[c]
        mt, mc = np.average(x[t], weights=wt), np.average(x[c], weights=wc)
        vt = np.average((x[t] - mt) ** 2, weights=wt)
        vc = np.average((x[c] - mc) ** 2, weights=wc)
    denom = np.sqrt((vt + vc) / 2)
    return float((mt - mc) / denom) if denom > 0 else 0.0


def _love_smd() -> pd.DataFrame:
    """Covariate balance (SMD) before vs after IPTW for the primary spec, for the love plot.

    Recomputes stabilized ATE weights from the held-out propensity (ICI_prediction) so the panel
    is self-contained, mirroring run_IPTW_analysis.compute_smd + the ATE-weight formula.
    """
    fp = os.path.join(BIOMARKER_PATH, f"IPTW_df_{COHORT}_{PRIMARY_PS_MODEL}.csv.gz")
    if not os.path.exists(fp):
        print(f"  no IPTW df at {fp}; emitting empty love-plot data")
        return pd.DataFrame(columns=LOVE_SMD_COLUMNS)
    df = pd.read_csv(fp)
    if not {"PX_on_ICI", "ICI_prediction"}.issubset(df.columns):
        print("  IPTW df missing PX_on_ICI/ICI_prediction; emitting empty love-plot data")
        return pd.DataFrame(columns=LOVE_SMD_COLUMNS)

    # Rows without a propensity/treatment value can't be weighted — drop them so the
    # weighted SMDs don't collapse to NaN.
    df = df.dropna(subset=["PX_on_ICI", "ICI_prediction"])
    if df.empty:
        return pd.DataFrame(columns=LOVE_SMD_COLUMNS)

    base = [c for c in ("GENDER", "AGE_AT_TREATMENTSTART") if c in df.columns]
    prefixed = [c for c in df.columns
                if c.startswith(("LINE_", "CANCER_TYPE_", "PANEL_VERSION_"))]
    covars = base + prefixed
    if not covars:
        return pd.DataFrame(columns=LOVE_SMD_COLUMNS)

    treat = df["PX_on_ICI"].to_numpy()
    ps = df["ICI_prediction"].clip(1e-6, 1 - 1e-6).to_numpy()
    p_treat = treat.mean()
    w = np.where(treat == 1, p_treat / ps, (1 - p_treat) / (1 - ps))
    lo, hi = np.percentile(w, [1, 99])
    w = np.clip(w, lo, hi)

    rows = []
    for cov in covars:
        x = pd.to_numeric(df[cov], errors="coerce").to_numpy(dtype=float)
        if np.isnan(x).any():
            continue
        rows.append({
            "covariate": cov,
            "smd_unweighted": _smd(x, treat),
            "smd_weighted": _smd(x, treat, w),
        })
    out = pd.DataFrame(rows, columns=LOVE_SMD_COLUMNS)
    if out.empty:
        return out
    # Keep demographics + the most-imbalanced covariates for a readable love plot
    out["_abs"] = out["smd_unweighted"].abs()
    keep_base = out[out["covariate"].isin(base)]
    keep_top = out[~out["covariate"].isin(base)].nlargest(LOVE_TOP_N, "_abs")
    return pd.concat([keep_base, keep_top]).drop(columns="_abs").reset_index(drop=True)


def main() -> None:
    save_figure_data(_ps_predictions(), "fig5_ps_predictions.csv")
    save_figure_data(_volcano_track2(), "fig5_volcano_track2.csv")
    save_figure_data(_robust_hits(), "fig5_robust_hits.csv")
    km, meta = _km_top_hit()
    save_figure_data(km, "fig5_km_top_hit.csv")
    save_figure_data(_km_examples(), "fig5_km_examples.csv")
    save_figure_data(meta, "fig5_top_hit_meta.csv")
    save_figure_data(_love_smd(), "fig5_love_smd.csv")


if __name__ == "__main__":
    main()
