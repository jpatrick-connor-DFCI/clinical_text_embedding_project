"""Pre-compute inputs for Figure 3 (modality comparison).

Writes to FIGURE_DATA_DIR:
- fig3_modality_cindex.csv        event, modality, cindex
- fig3_joint_betas.csv            event, modality, beta
- fig3_risk_score_corr.csv        modality x modality correlation, plus n_patients in metadata row
- fig3_univariate_vs_joint.csv    modality, univariate_auc, joint_auc
"""

from __future__ import annotations

import os
import sys
from functools import reduce
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _figure_utils import (
    MODALITY_ORDER, feature_held_out_dir, scheme_results_dir,
    save_figure_data,
)


SCHEME = "icd3_post"          # representative scheme for panels A/B/D
DEATH_SCHEME = "death_met"    # scheme for the correlation heatmap

MODALITY_CINDEX_COLUMNS = ["event", "modality", "cindex"]
JOINT_BETA_COLUMNS = ["event", "modality", "beta"]
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
                rows.append({"event": ev, "modality": mod, "cindex": row["mean_c_index"]})
            except (KeyError, IndexError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                print(f"  [{ev}:{mod}] failed to load test metrics — {type(e).__name__}: {e}")
                n_failed += 1
                continue
    if n_failed:
        print(f"  total skipped: {n_failed}")
    return pd.DataFrame(rows, columns=MODALITY_CINDEX_COLUMNS)


def _joint_betas(scheme: str) -> pd.DataFrame:
    fp = os.path.join(scheme_results_dir(scheme), "risk_score_coxph", "joint_model_betas.csv")
    if not os.path.exists(fp):
        print(f"  missing {fp}; skipping")
        return pd.DataFrame(columns=JOINT_BETA_COLUMNS)
    return pd.read_csv(fp)[JOINT_BETA_COLUMNS]


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
    save_figure_data(_modality_cindex(SCHEME), "fig3_modality_cindex.csv")
    save_figure_data(_joint_betas(SCHEME), "fig3_joint_betas.csv")
    save_figure_data(_risk_score_corr(DEATH_SCHEME, "death"), "fig3_risk_score_corr.csv")
    save_figure_data(_univariate_vs_joint(SCHEME), "fig3_univariate_vs_joint.csv")


if __name__ == "__main__":
    main()
