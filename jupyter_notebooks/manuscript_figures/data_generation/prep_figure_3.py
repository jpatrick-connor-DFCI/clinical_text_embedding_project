"""Pre-compute inputs for Figure 3 (modality comparison).

Writes to FIGURE_DATA_DIR:
- fig3_modality_cindex.csv        scheme, event, modality, cindex, auc  (both metrics side by side;
                                  auc is the sibling mean_auc(t) column from the same {mod}_test.csv)
- fig3_modality_avg_rank_cindex.csv   modality, mean_rank, sem_rank, n_events  (ranked by cindex)
- fig3_modality_avg_rank_auc.csv      modality, mean_rank, sem_rank, n_events  (ranked by mean AUC(t))
- fig3_modality_ranks_long_cindex.csv scheme, event, modality, rank  (per-endpoint, cindex-ranked)
- fig3_modality_ranks_long_auc.csv    scheme, event, modality, rank  (per-endpoint, AUC(t)-ranked)
                                  (the two ranks_long files let the R tier run a Friedman test across
                                  modalities for whichever metric is active)
- fig3_joint_betas.csv            scheme, event, modality, beta, se, hr, p_value, n, n_events
- fig3_risk_score_corr.csv        modality x modality correlation, plus n_patients in metadata row
"""

from __future__ import annotations

import os
import sys
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _figure_utils import (
    MODALITY_ORDER, SCHEME_RESULT_DIRS, feature_held_out_dir, scheme_results_dir,
    save_figure_data,
)
from slurm_array_utils import get_events_from_df, load_embedding_prediction_df


SCHEMES = list(SCHEME_RESULT_DIRS)
DEATH_SCHEME = "death_met"    # scheme for the correlation heatmap

MODALITY_CINDEX_COLUMNS = ["scheme", "event", "modality", "cindex", "auc"]
MODALITY_AVG_RANK_COLUMNS = ["modality", "mean_rank", "sem_rank", "n_events"]
MODALITY_RANKS_LONG_COLUMNS = ["scheme", "event", "modality", "rank"]
JOINT_BETA_COLUMNS = [
    "scheme", "event", "modality",
    "beta", "se", "hr", "p_value", "n", "n_events",
]
RISK_SCORE_CORR_COLUMNS = ["modality", *MODALITY_ORDER, "n_patients"]


def _modality_cindex(scheme: str) -> pd.DataFrame:
    root = os.path.join(scheme_results_dir(scheme), "feature_comps")
    events = sorted(os.listdir(root)) if os.path.isdir(root) else []
    rows = []
    n_failed = 0
    for ev in events:
        d = os.path.join(root, ev)
        for mod in MODALITY_ORDER:
            fp = os.path.join(d, f"{mod}_test.csv")
            if not os.path.exists(fp):
                continue
            try:
                row = pd.read_csv(fp).iloc[0]
                rows.append({
                    "scheme": scheme, "event": ev, "modality": mod,
                    "cindex": row["mean_c_index"], "auc": row["mean_auc(t)"],
                })
            except (KeyError, IndexError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                print(f"  [{ev}:{mod}] failed to load test metrics — {type(e).__name__}: {e}")
                n_failed += 1
                continue
    if n_failed:
        print(f"  total skipped: {n_failed}")
    return pd.DataFrame(rows, columns=MODALITY_CINDEX_COLUMNS)


def _joint_betas(scheme: str) -> pd.DataFrame:
    """Refit the joint Cox model per (scheme, event) so we have p-values.

    The held-out modality risk scores are produced at training time by
    `run_feature_comp_task.py` (written under `<scheme>/held_out_risk_scores/`).
    sksurv's `CoxPHSurvivalAnalysis` does not expose SEs, so for Fig 3A we redo
    the joint fit here with `lifelines.CoxPHFitter` on the same merged inputs
    (held-out modality risk scores × event surv data), standardizing features.
    """
    risk_root = os.path.normpath(
        os.path.join(scheme_results_dir(scheme), "held_out_risk_scores")
    )
    if not os.path.isdir(risk_root):
        print(f"  missing {risk_root}; skipping joint refit")
        return pd.DataFrame(columns=JOINT_BETA_COLUMNS)
    try:
        tte_df = load_embedding_prediction_df(scheme)
    except FileNotFoundError as exc:
        print(f"  {scheme}: cannot load embedding prediction df ({exc})")
        return pd.DataFrame(columns=JOINT_BETA_COLUMNS)
    available_events = sorted(set(get_events_from_df(tte_df)))
    rows: list[dict[str, object]] = []
    for event in available_events:
        event_dir = os.path.join(risk_root, event)
        if not os.path.isdir(event_dir):
            continue
        risk_dfs = []
        for mod in MODALITY_ORDER:
            fp = os.path.join(event_dir, f"{mod}_risk_scores.csv")
            if not os.path.exists(fp):
                continue
            d = pd.read_csv(fp)
            risk_col = f"{mod}_risk_score"
            if risk_col not in d.columns and "risk_score" in d.columns:
                d = d.rename(columns={"risk_score": risk_col})
            if risk_col not in d.columns:
                continue
            risk_dfs.append(d[["DFCI_MRN", risk_col]].drop_duplicates(subset=["DFCI_MRN"]))
        if len(risk_dfs) < 2:
            continue
        merged = reduce(lambda l, r: l.merge(r, on="DFCI_MRN", how="inner"), risk_dfs)
        keep = ["DFCI_MRN", event, f"tt_{event}"]
        if not set(keep).issubset(tte_df.columns):
            continue
        merged = merged.merge(tte_df[keep], on="DFCI_MRN", how="inner")
        risk_cols = [c for c in merged.columns if c.endswith("_risk_score")]
        # Drop rows with NaN in *any* feature or in event/time. The per-modality
        # held-out risk-score files don't all cover the same patients (their
        # outer join can leave NaNs for partially-overlapping cohorts), and
        # lifelines refuses NaNs outright.
        merged = merged.dropna(subset=risk_cols + [event, f"tt_{event}"])
        merged = merged.loc[merged[f"tt_{event}"] > 0]
        # Drop constant risk-score columns within this event slice; a feature
        # with zero variance breaks StandardScaler and makes the Cox design
        # matrix singular.
        constant_cols = [c for c in risk_cols if merged[c].nunique(dropna=False) <= 1]
        if constant_cols:
            print(f"  [{scheme}/{event}] dropping constant risk columns: {constant_cols}")
            merged = merged.drop(columns=constant_cols)
            risk_cols = [c for c in risk_cols if c not in constant_cols]
        if len(risk_cols) < 2:
            continue
        n = len(merged)
        n_events = int(merged[event].sum()) if n else 0
        if n < 20 or n_events < 5 or n - n_events < 5:
            continue
        try:
            X = StandardScaler().fit_transform(merged[risk_cols].astype(float))
        except ValueError as exc:
            print(f"  [{scheme}/{event}] standardize failed: {exc}")
            continue
        if not np.isfinite(X).all():
            print(f"  [{scheme}/{event}] non-finite values after standardize; skipping")
            continue
        fit_df = pd.DataFrame(X, columns=risk_cols, index=merged.index)
        fit_df[event] = merged[event].astype(int).values
        fit_df[f"tt_{event}"] = merged[f"tt_{event}"].astype(float).values
        # Small L2 ridge stabilizes the fit on highly-collinear modality risk
        # scores. Without it, near-rank-deficient designs produce huge
        # cancelling betas (|beta| > 1e4) and meaningless p-values. With a
        # penalty of 1e-2 on standardized features, well-identified coefficients
        # are essentially unchanged while pathological events stay finite.
        cph = CoxPHFitter(penalizer=1e-2, l1_ratio=0.0)
        # lifelines wraps several upstream failure modes (collinearity,
        # remaining NaNs slipping through, Newton-step singularities) in a mix
        # of ConvergenceError / TypeError / ValueError / LinAlgError. Catch all
        # of them so one pathological event can't kill the entire scheme.
        try:
            cph.fit(fit_df, duration_col=f"tt_{event}", event_col=event)
        except (ConvergenceError, TypeError, ValueError,
                np.linalg.LinAlgError) as exc:
            print(f"  [{scheme}/{event}] CoxPH refit failed: {type(exc).__name__}: {exc}")
            continue
        summary = cph.summary
        # Backstop: even with the ridge, drop the (scheme, event) entirely if
        # any coefficient is clearly numerical breakdown rather than biology.
        # On standardized features, |beta| > 5 (HR > ~150 per SD) is implausible
        # and almost always reflects residual separation / collinearity.
        if (summary["coef"].abs() > 5).any():
            offenders = summary.loc[summary["coef"].abs() > 5, "coef"].to_dict()
            print(f"  [{scheme}/{event}] dropping fit with pathological betas: {offenders}")
            continue
        for risk_col, srow in summary.iterrows():
            rows.append({
                "scheme": scheme,
                "event": event,
                "modality": str(risk_col).replace("_risk_score", ""),
                "beta": float(srow["coef"]),
                "se": float(srow["se(coef)"]),
                "hr": float(srow["exp(coef)"]),
                "p_value": float(srow["p"]),
                "n": n,
                "n_events": n_events,
            })
    return pd.DataFrame(rows, columns=JOINT_BETA_COLUMNS)


def _modality_cindex_all() -> pd.DataFrame:
    frames = [_modality_cindex(scheme) for scheme in SCHEMES]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=MODALITY_CINDEX_COLUMNS)


def _joint_betas_all() -> pd.DataFrame:
    frames = [_joint_betas(scheme) for scheme in SCHEMES]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=JOINT_BETA_COLUMNS)


def _modality_avg_rank(metrics_df: pd.DataFrame, value_col: str = "cindex") -> pd.DataFrame:
    """Average performance rank per modality across endpoints.

    For each (scheme, event) endpoint, rank the modalities by `value_col` (cindex or
    auc; 1 = best); average those ranks per modality. Restricted to *complete-case*
    endpoints where every modality in MODALITY_ORDER has a value, so all averages
    rank the same set.
    """
    if metrics_df.empty:
        return pd.DataFrame(columns=MODALITY_AVG_RANK_COLUMNS)

    mat = metrics_df.pivot_table(
        index=["scheme", "event"], columns="modality", values=value_col, aggfunc="mean",
    )
    modalities = [m for m in MODALITY_ORDER if m in mat.columns]
    mat = mat.reindex(columns=modalities)
    mat = mat.dropna(how="any")  # complete-case endpoints only
    if mat.empty:
        return pd.DataFrame(columns=MODALITY_AVG_RANK_COLUMNS)

    # Higher value -> better -> rank 1; ties share the average rank.
    ranks = mat.rank(axis=1, ascending=False, method="average")
    n_events = len(ranks)
    out = pd.DataFrame({
        "modality": modalities,
        "mean_rank": [ranks[m].mean() for m in modalities],
        "sem_rank": [ranks[m].std(ddof=1) / np.sqrt(n_events) if n_events > 1 else 0.0
                     for m in modalities],
        "n_events": n_events,
    })
    return out.reindex(columns=MODALITY_AVG_RANK_COLUMNS)


def _modality_ranks_long(metrics_df: pd.DataFrame, value_col: str = "cindex") -> pd.DataFrame:
    """Per-endpoint modality ranks (1 = best) by `value_col`, complete-case only.

    Long-format companion to _modality_avg_rank's aggregated mean/SEM, so the
    R tier can run a Friedman test (repeated measures across modalities, one
    block per endpoint) for Fig 3C.
    """
    if metrics_df.empty:
        return pd.DataFrame(columns=MODALITY_RANKS_LONG_COLUMNS)

    mat = metrics_df.pivot_table(
        index=["scheme", "event"], columns="modality", values=value_col, aggfunc="mean",
    )
    modalities = [m for m in MODALITY_ORDER if m in mat.columns]
    mat = mat.reindex(columns=modalities)
    mat = mat.dropna(how="any")
    if mat.empty:
        return pd.DataFrame(columns=MODALITY_RANKS_LONG_COLUMNS)

    ranks = mat.rank(axis=1, ascending=False, method="average")
    out = ranks.reset_index().melt(
        id_vars=["scheme", "event"], var_name="modality", value_name="rank",
    )
    return out.reindex(columns=MODALITY_RANKS_LONG_COLUMNS)


def _risk_score_corr(scheme: str, event: str = "death") -> pd.DataFrame:
    risk_dir = feature_held_out_dir(scheme, event)
    dfs = []
    for mod in MODALITY_ORDER:
        fp = os.path.join(risk_dir, f"{mod}_risk_scores.csv")
        if not os.path.exists(fp):
            print(f"  missing {fp}")
            continue
        d = pd.read_csv(fp)
        risk_col = f"{mod}_risk_score"
        if risk_col not in d.columns and "risk_score" in d.columns:
            d = d.rename(columns={"risk_score": risk_col})
        dfs.append(d[["DFCI_MRN", risk_col]])
    if not dfs:
        return pd.DataFrame(columns=RISK_SCORE_CORR_COLUMNS)
    merged = reduce(lambda l, r: l.merge(r, on="DFCI_MRN", how="inner"), dfs)
    cols = [c for c in merged.columns if c.endswith("_risk_score")]
    corr = merged[cols].corr()
    corr.index = [c.replace("_risk_score", "") for c in corr.index]
    corr.columns = [c.replace("_risk_score", "") for c in corr.columns]
    out = corr.reset_index().rename(columns={"index": "modality"})
    out["n_patients"] = len(merged)
    # Enforce stable schema: always emit all MODALITY_ORDER columns, NaN where
    # the modality risk-score file was missing for this scheme/event.
    return out.reindex(columns=RISK_SCORE_CORR_COLUMNS)


def main() -> None:
    metrics_all = _modality_cindex_all()
    save_figure_data(metrics_all, "fig3_modality_cindex.csv")
    for value_col, tag in [("cindex", "cindex"), ("auc", "auc")]:
        save_figure_data(_modality_avg_rank(metrics_all, value_col),
                          f"fig3_modality_avg_rank_{tag}.csv")
        save_figure_data(_modality_ranks_long(metrics_all, value_col),
                          f"fig3_modality_ranks_long_{tag}.csv")
    save_figure_data(_joint_betas_all(), "fig3_joint_betas.csv")
    save_figure_data(_risk_score_corr(DEATH_SCHEME, "death"), "fig3_risk_score_corr.csv")


if __name__ == "__main__":
    main()
