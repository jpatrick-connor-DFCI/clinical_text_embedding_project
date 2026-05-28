"""Pre-compute inputs for Figure 4 (mortality trajectories).

Writes to FIGURE_DATA_DIR:
- fig4_trajectories_with_clusters.csv  DFCI_MRN, cluster, <month columns kept>  (full cohort)
- fig4_trajectories_heatmap.csv        same schema, downsampled to <= HEATMAP_ROWS_PER_CLUSTER
                                       per cluster and ordered by within-cluster mean risk
                                       (panel A reads this so the heatmap renders cleanly)
- fig4_cluster_means.csv               cluster, month, mean, sem  (computed on full cohort)
- fig4_cluster_composition_cancer.csv  cluster, top categories, OTHER
- fig4_cluster_composition_stage.csv   same
- fig4_cluster_composition_treatment.csv same
- fig4_km_data.csv                     DFCI_MRN, cluster, death, tt_death
- fig4_cluster_severity.csv            cluster, mean_met_sites, rmst_months, n_patients
- fig4_silhouette.csv                  k, silhouette  (cluster-count justification, appendix)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _figure_utils import (
    FEATURE_PATH, RESULTS_PATH, SURV_PATH,
    save_figure_data,
)


N_CLUSTERS = 4
DEFAULT_DECAY = 0.1
HEATMAP_ROWS_PER_CLUSTER = 500

CLUSTER_MEAN_COLUMNS = ["cluster", "month", "mean", "sem", "n_patients"]
COMPOSITION_COLUMNS = ["cluster", "OTHER"]
KM_COLUMNS = ["DFCI_MRN", "cluster", "death", "tt_death"]
SEVERITY_COLUMNS = ["cluster", "mean_met_sites", "rmst_months", "n_patients"]
SILHOUETTE_COLUMNS = ["k", "silhouette"]
MET_EVENTS = ["brainM", "boneM", "adrenalM", "liverM", "lungM", "nodeM", "peritonealM"]
RMST_TAU_MONTHS = 120  # 5y past the 60-mo landmark entry requirement; avoids saturation at the entry cap
SILHOUETTE_K_RANGE = range(2, 9)


def _default_trajectory_input(decay: float) -> str:
    """Canonical output path of generate_mortality_trajectories.py."""
    return os.path.join(
        RESULTS_PATH, "death_met_results", "mortality_trajectories",
        f"survival_trajectories_w_decay_param_{decay}.csv",
    )


def _scaled_trajectory_matrix(input_path: str) -> tuple[pd.DataFrame, np.ndarray | None, list[str]]:
    """Load, filter, impute (ffill/bfill) and z-scale the trajectory matrix.

    Returns (traj_sub with imputed month columns, X_z or None if too few patients,
    months_to_keep). Shared by clustering and the silhouette scan so both operate on
    exactly the same feature matrix.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Trajectory input not found: {input_path}\n"
            "Run python_scripts/mortality_trajectories/generate_mortality_trajectories.py first, "
            "or pass --input to override."
        )
    traj_df = pd.read_csv(input_path)
    monthly_cols = [c for c in traj_df.columns if c != "DFCI_MRN"]
    missing_colwise = traj_df[monthly_cols].isna().sum(axis=0)
    months_to_keep = missing_colwise[missing_colwise < 0.9 * len(traj_df)].index.tolist()
    missing_rowwise = (~traj_df[monthly_cols].isna()).sum(axis=1)
    pxs_to_keep = missing_rowwise[missing_rowwise > 18].index
    traj_sub = traj_df.loc[pxs_to_keep, ["DFCI_MRN"] + months_to_keep].copy()

    if len(traj_sub) < N_CLUSTERS:
        return traj_sub, None, months_to_keep

    X = traj_sub[months_to_keep].values
    X_filled = pd.DataFrame(X).ffill(axis=1).bfill(axis=1).values
    X_z = StandardScaler().fit_transform(X_filled)
    # Replace original X with filled X so downstream uses the same values
    for i, col in enumerate(months_to_keep):
        traj_sub[col] = X_filled[:, i]
    return traj_sub, X_z, months_to_keep


def _cluster_trajectories(input_path: str) -> tuple[pd.DataFrame, list[str]]:
    traj_sub, X_z, months_to_keep = _scaled_trajectory_matrix(input_path)
    if X_z is None:
        print(f"  only {len(traj_sub)} patients survived filtering; need >= {N_CLUSTERS} "
              "for clustering. Emitting empty figure-data.")
        empty = pd.DataFrame(columns=["DFCI_MRN", "cluster", *months_to_keep])
        return empty, months_to_keep

    model = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")
    raw_labels = model.fit_predict(X_z)

    # Relabel clusters by ascending mean risk so cluster 0 is always lowest-risk.
    # This stabilizes labels across reruns / input filtering changes.
    risk = traj_sub[months_to_keep].to_numpy()
    cluster_mean_risk = {k: risk[raw_labels == k].mean() for k in np.unique(raw_labels)}
    label_order = sorted(cluster_mean_risk, key=cluster_mean_risk.get)
    relabel = {old: new for new, old in enumerate(label_order)}
    traj_sub["cluster"] = pd.Series(raw_labels, index=traj_sub.index).map(relabel)
    return traj_sub, months_to_keep


def _silhouette_scan(input_path: str) -> pd.DataFrame:
    """Silhouette score vs k on the same scaled trajectory matrix — justifies N_CLUSTERS."""
    _, X_z, _ = _scaled_trajectory_matrix(input_path)
    if X_z is None or len(X_z) <= max(SILHOUETTE_K_RANGE):
        return pd.DataFrame(columns=SILHOUETTE_COLUMNS)
    rows = []
    for k in SILHOUETTE_K_RANGE:
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_z)
        rows.append({"k": int(k), "silhouette": float(silhouette_score(X_z, labels))})
    return pd.DataFrame(rows, columns=SILHOUETTE_COLUMNS)


def _cluster_means(traj_sub: pd.DataFrame, months_to_keep: list[str]) -> pd.DataFrame:
    rows = []
    for k in sorted(traj_sub["cluster"].unique()):
        sub = traj_sub.loc[traj_sub["cluster"] == k, months_to_keep].values
        mean = sub.mean(axis=0)
        sem = sub.std(axis=0) / np.sqrt(sub.shape[0])
        for col, m, s in zip(months_to_keep, mean, sem):
            rows.append({"cluster": int(k), "month": col, "mean": m, "sem": s,
                         "n_patients": sub.shape[0]})
    return pd.DataFrame(rows, columns=CLUSTER_MEAN_COLUMNS)


def _composition(traj_sub: pd.DataFrame, lookup_df: pd.DataFrame,
                 cols: list[str], prefix: str, top_n: int = 8) -> pd.DataFrame:
    if traj_sub.empty:
        return pd.DataFrame(columns=COMPOSITION_COLUMNS)
    merged = traj_sub[["DFCI_MRN", "cluster"]].merge(
        lookup_df[["DFCI_MRN"] + cols], on="DFCI_MRN", how="left",
    )
    agg = merged.groupby("cluster")[cols].sum()
    agg = agg.div(agg.sum(axis=1), axis=0)
    top = agg.sum(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
    keep = agg[top].copy()
    keep["OTHER"] = 1 - keep.sum(axis=1)
    keep.columns = [c.replace(prefix, "") for c in keep.columns]
    return keep.reset_index()


def _heatmap_downsample(traj_sub: pd.DataFrame, months_to_keep: list[str],
                         rows_per_cluster: int = HEATMAP_ROWS_PER_CLUSTER) -> pd.DataFrame:
    """Within each cluster, sample at most `rows_per_cluster` patients and order
    them by within-cluster mean trajectory risk so the heatmap shows clean cluster
    blocks rather than an unreadable smear of 30k+ rows.
    """
    if traj_sub.empty:
        return traj_sub.iloc[0:0].copy()
    rng = np.random.default_rng(0)
    chunks = []
    for k, g in traj_sub.groupby("cluster"):
        if len(g) > rows_per_cluster:
            idx = rng.choice(len(g), size=rows_per_cluster, replace=False)
            g = g.iloc[idx]
        # Order by within-patient mean risk so each cluster block reads top→bottom
        g = g.assign(_mean=g[months_to_keep].mean(axis=1)).sort_values("_mean").drop(columns="_mean")
        chunks.append(g)
    return pd.concat(chunks, ignore_index=True)


def _km_data(traj_sub: pd.DataFrame) -> pd.DataFrame:
    if traj_sub.empty:
        return pd.DataFrame(columns=KM_COLUMNS)
    surv_df = pd.read_csv(os.path.join(SURV_PATH, "death_met_surv_df.csv.gz"))
    out = (traj_sub[["DFCI_MRN", "cluster"]]
           .merge(surv_df[["DFCI_MRN", "death", "tt_death"]], on="DFCI_MRN", how="inner")
           .dropna())
    out = out[out["tt_death"] > 0]
    return out


def _cluster_severity(traj_sub: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster disease-severity metrics for the Fig 4C characteristics panel.

    - mean_met_sites: mean number of distinct metastasis sites per patient (0-7).
    - rmst_months: cluster restricted mean survival time over [0, RMST_TAU_MONTHS] months.
    """
    if traj_sub.empty:
        return pd.DataFrame(columns=SEVERITY_COLUMNS)
    surv_df = pd.read_csv(os.path.join(SURV_PATH, "death_met_surv_df.csv.gz"))
    met_cols = [c for c in MET_EVENTS if c in surv_df.columns]
    keep_cols = ["DFCI_MRN", "death", "tt_death"] + met_cols
    merged = traj_sub[["DFCI_MRN", "cluster"]].merge(
        surv_df[keep_cols], on="DFCI_MRN", how="inner",
    )
    if met_cols:
        merged["n_met_sites"] = merged[met_cols].fillna(0).clip(upper=1).sum(axis=1)
    else:
        merged["n_met_sites"] = np.nan

    rows = []
    for k, sub in merged.groupby("cluster"):
        valid = sub.dropna(subset=["death", "tt_death"])
        valid = valid[valid["tt_death"] > 0]
        rmst = np.nan
        if not valid.empty:
            kmf = KaplanMeierFitter().fit(valid["tt_death"] / 30.44, valid["death"])
            try:
                rmst = float(restricted_mean_survival_time(kmf, t=RMST_TAU_MONTHS))
            except Exception as e:  # pragma: no cover - defensive
                print(f"  RMST failed for cluster {k}: {e}")
                rmst = np.nan
        rows.append({
            "cluster": int(k),
            "mean_met_sites": sub["n_met_sites"].mean(),
            "rmst_months": rmst,
            "n_patients": int(len(sub)),
        })
    return pd.DataFrame(rows, columns=SEVERITY_COLUMNS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decay", type=float, default=DEFAULT_DECAY,
                        help="Trajectory decay parameter — selects the input filename")
    parser.add_argument("--input", type=str, default=None,
                        help="Override trajectory input path entirely")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = args.input or _default_trajectory_input(args.decay)
    print(f"  trajectory input: {input_path}")

    cancer_type_df = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_type_df.csv.gz"))
    stage_df = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_stage_df.csv.gz"))
    treatment_df = pd.read_csv(os.path.join(FEATURE_PATH, "categorical_treatment_data_by_line.csv.gz"))
    type_cols = [c for c in cancer_type_df.columns if c.startswith("CANCER_TYPE_")]
    stage_cols = [c for c in stage_df.columns if c.startswith("CANCER_STAGE_")]
    tx_cols = [c for c in treatment_df.columns if c.startswith("PX_on_")]
    tx1 = treatment_df.loc[treatment_df["treatment_line"] == 1]

    traj_sub, months_to_keep = _cluster_trajectories(input_path)
    save_figure_data(traj_sub, "fig4_trajectories_with_clusters.csv")
    save_figure_data(_heatmap_downsample(traj_sub, months_to_keep),
                     "fig4_trajectories_heatmap.csv")
    save_figure_data(_cluster_means(traj_sub, months_to_keep), "fig4_cluster_means.csv")
    save_figure_data(_composition(traj_sub, cancer_type_df, type_cols, "CANCER_TYPE_"),
                     "fig4_cluster_composition_cancer.csv")
    save_figure_data(_composition(traj_sub, stage_df, stage_cols, "CANCER_STAGE_"),
                     "fig4_cluster_composition_stage.csv")
    save_figure_data(_composition(traj_sub, tx1, tx_cols, "PX_on_"),
                     "fig4_cluster_composition_treatment.csv")
    save_figure_data(_km_data(traj_sub), "fig4_km_data.csv")
    save_figure_data(_cluster_severity(traj_sub), "fig4_cluster_severity.csv")
    save_figure_data(_silhouette_scan(input_path), "fig4_silhouette.csv")


if __name__ == "__main__":
    main()
