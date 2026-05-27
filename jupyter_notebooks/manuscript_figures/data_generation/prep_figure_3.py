"""Pre-compute inputs for Figure 3 (modality comparison).

Writes to FIGURE_DATA_DIR:
- fig3_modality_cindex.csv        scheme, event, modality, cindex
- fig3_joint_betas.csv            scheme, event, modality, beta, se, hr, p_value, n, n_events
- fig3_risk_score_corr.csv        modality x modality correlation, plus n_patients in metadata row
- fig3_univariate_vs_joint.csv    modality, univariate_auc, joint_auc
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


SCHEME = "icd3_post"          # representative scheme for legacy correlation / joint-AUC helpers
SCHEMES = list(SCHEME_RESULT_DIRS)
DEATH_SCHEME = "death_met"    # scheme for the correlation heatmap

MODALITY_CINDEX_COLUMNS = ["scheme", "event", "modality", "cindex"]
JOINT_BETA_COLUMNS = [
    "scheme", "event", "modality",
    "beta", "se", "hr", "p_value", "n", "n_events",
]
RISK_SCORE_CORR_COLUMNS = ["modality", *MODALITY_ORDER, "n_patients"]
UNIVARIATE_VS_JOINT_COLUMNS = ["modality", "univariate_auc", "joint_auc"]


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
                rows.append({"scheme": scheme, "event": ev, "modality": mod, "cindex": row["mean_c_index"]})
            except (KeyError, IndexError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                print(f"  [{ev}:{mod}] failed to load test metrics — {type(e).__name__}: {e}")
                n_failed += 1
                continue
    if n_failed:
        print(f"  total skipped: {n_failed}")
    return pd.DataFrame(rows, columns=MODALITY_CINDEX_COLUMNS)


def _joint_betas(scheme: str) -> pd.DataFrame:
    """Refit the joint Cox model per (scheme, event) so we have p-values.

    Upstream `feature_risk_score_coxph.py` saves point estimates only (sksurv's
    `CoxPHSurvivalAnalysis` does not expose SEs). For Fig 3A we need
    significance, so we redo the joint fit with `lifelines.CoxPHFitter` on the
    same merged inputs (held-out modality risk scores × event surv data) and
    standardize features to mirror the upstream pipeline.
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
        cph = CoxPHFitter()
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


def _univariate_vs_joint(scheme: str) -> pd.DataFrame:
    base = os.path.join(scheme_results_dir(scheme), "risk_score_coxph")
    univ_fp = os.path.join(base, "univariate_modality_metrics.csv")
    joint_fp = os.path.join(base, "joint_model_metrics.csv")
    if not (os.path.exists(univ_fp) and os.path.exists(joint_fp)):
        print(f"  missing univariate/joint metrics; skipping")
        return pd.DataFrame(columns=UNIVARIATE_VS_JOINT_COLUMNS)
    univ = pd.read_csv(univ_fp)
    joint = pd.read_csv(joint_fp)
    univ_mean = univ.groupby("modality")["mean_auc(t)"].mean()
    joint_mean = joint["mean_auc(t)"].mean()
    rows = [{"modality": mod, "univariate_auc": univ_mean.get(mod), "joint_auc": joint_mean}
            for mod in MODALITY_ORDER if mod in univ_mean.index]
    return pd.DataFrame(rows, columns=UNIVARIATE_VS_JOINT_COLUMNS)


def main() -> None:
    save_figure_data(_modality_cindex_all(), "fig3_modality_cindex.csv")
    save_figure_data(_joint_betas_all(), "fig3_joint_betas.csv")
    save_figure_data(_risk_score_corr(DEATH_SCHEME, "death"), "fig3_risk_score_corr.csv")
    save_figure_data(_univariate_vs_joint(SCHEME), "fig3_univariate_vs_joint.csv")


if __name__ == "__main__":
    main()
