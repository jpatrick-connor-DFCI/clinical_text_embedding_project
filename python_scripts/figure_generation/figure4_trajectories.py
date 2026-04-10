"""Figure 4 — Longitudinal mortality risk trajectory analysis (3-panel).

Panels:
  A: Mean mortality risk trajectories for k clusters (with 95% CI bands)
  B: Kaplan-Meier overall survival stratified by trajectory cluster
  C: Clinical characteristics by cluster (grouped bar chart)

Data sources:
  - Trajectory CSV:
      {RESULTS_PATH}/death_met_results/mortality_trajectories/
      survival_trajectories_w_decay_param_{DECAY_PARAM}.csv
      Columns: DFCI_MRN, plus_0_months_data, plus_3_months_data, ..., plus_60_months_data

  - Survival data:
      {SURV_PATH}/death_met_embedding_prediction_df.csv
      Columns include: DFCI_MRN, death, tt_death

  - Cancer stage (for clinical char panel):
      {FEATURE_PATH}/cancer_stage_df.csv
      Columns: DFCI_MRN, CANCER_STAGE_2.0, CANCER_STAGE_3.0, CANCER_STAGE_4.0, ...

  - Treatment (ICI flag):
      {FEATURE_PATH}/categorical_treatment_data_by_line.csv
      Columns: DFCI_MRN, treatment_line, PX_on_ICI_1 (or similar)

  - Sex from survival/demographics:
      death_met_embedding_prediction_df.csv includes GENDER column
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_generation_utils import (
    FEATURE_PATH,
    OUTPUT_DIR,
    RESULTS_PATH,
    SURV_PATH,
    set_manuscript_style,
)
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Trajectory file — adjust scheme/decay param to match your computed results
TRAJECTORY_SCHEME = "death_met"
TRAJECTORY_RESULTS_DIR = "death_met_results"
DECAY_PARAM = 0.1   # adjust to match your file name
TRAJECTORY_FILE = os.path.join(
    RESULTS_PATH, TRAJECTORY_RESULTS_DIR, "mortality_trajectories",
    f"survival_trajectories_w_decay_param_{DECAY_PARAM}.csv"
)

# Clustering
N_CLUSTERS = 4           # adjust based on elbow/silhouette analysis
RANDOM_SEED = 42

# Cluster display names — order clusters by mean risk (assigned in code)
# These will be filled in dynamically; override here if you want fixed labels.
CLUSTER_LABELS_OVERRIDE: dict[int, str] | None = None
# e.g.: {1: "Stable Low", 2: "Decreasing → Rebounding", 3: "Stable High", 4: "Rapidly Increasing"}

CLUSTER_COLORS = ["#2166ac", "#92c5de", "#f4a582", "#d73027"]


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def set_style():
    set_manuscript_style()


# ---------------------------------------------------------------------------
# Load and prepare trajectory data
# ---------------------------------------------------------------------------

def load_trajectories() -> pd.DataFrame:
    """Load wide-format trajectory CSV and return it."""
    if not os.path.isfile(TRAJECTORY_FILE):
        raise FileNotFoundError(f"Trajectory file not found: {TRAJECTORY_FILE}")
    return pd.read_csv(TRAJECTORY_FILE)


def wide_to_long(surv_traj: pd.DataFrame) -> pd.DataFrame:
    """Melt wide trajectory DF to long format with columns: DFCI_MRN, time, risk_score."""
    risk_cols = [c for c in surv_traj.columns if c.startswith("plus_")]
    long = (surv_traj
            .loc[~surv_traj["plus_0_months_data"].isna()]
            .melt(id_vars="DFCI_MRN", value_vars=risk_cols,
                  var_name="time_str", value_name="risk_score")
            .dropna(subset=["risk_score"]))
    long["time"] = long["time_str"].str.extract(r"plus_(\d+)_months").astype(int)
    return long.sort_values(["DFCI_MRN", "time"]).drop(columns="time_str")


def extract_trajectory_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """Extract summary features per patient for clustering."""
    def features(grp):
        grp = grp.sort_values("time")
        rs = grp["risk_score"].values
        t  = grp["time"].values
        slope = np.polyfit(t, rs, 1)[0] if len(t) > 1 else 0.0
        auc   = np.trapezoid(rs, t) if len(t) > 1 else 0.0
        return pd.Series({
            "baseline":   rs[0],
            "final":      rs[-1],
            "slope":      slope,
            "auc":        auc,
            "min_risk":   rs.min(),
            "max_risk":   rs.max(),
            "rebound":    rs[-1] - rs.min(),
        })
    return (long_df.groupby("DFCI_MRN")[["time", "risk_score"]]
            .apply(features)
            .reset_index())


def cluster_patients(features_df: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> pd.DataFrame:
    """K-Means cluster patients on standardized trajectory features."""
    feat_cols = ["baseline", "final", "slope", "auc", "min_risk", "max_risk", "rebound"]
    X = features_df[feat_cols].fillna(features_df[feat_cols].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=20)
    features_df = features_df.copy()
    features_df["cluster_raw"] = km.fit_predict(X_scaled)

    # Order clusters by mean baseline risk (ascending)
    mean_baseline = features_df.groupby("cluster_raw")["baseline"].mean().sort_values()
    rank_map = {c: i + 1 for i, c in enumerate(mean_baseline.index)}
    features_df["cluster"] = features_df["cluster_raw"].map(rank_map)
    return features_df


def apply_cluster_labels(cluster_df: pd.DataFrame) -> dict[int, str]:
    """Return display name for each cluster. Override if CLUSTER_LABELS_OVERRIDE is set."""
    if CLUSTER_LABELS_OVERRIDE:
        return CLUSTER_LABELS_OVERRIDE
    # Auto-generate labels from cluster statistics
    stats = cluster_df.groupby("cluster")[["baseline", "final", "slope"]].mean()
    labels = {}
    name_counts: dict[str, int] = {}
    for c in sorted(stats.index):
        base, final, slope = stats.loc[c, ["baseline", "final", "slope"]]
        n = (cluster_df["cluster"] == c).sum()
        if slope > 0.005 and final > 0.6:
            base_label = "Rapidly Increasing"
        elif slope > 0.002 and base > 0.5:
            base_label = "Stable High"
        elif abs(slope) <= 0.002 and base < 0.35:
            base_label = "Stable Low"
        else:
            base_label = "Decreasing → Rebounding" if final > base * 0.8 else "Decreasing"

        name_counts[base_label] = name_counts.get(base_label, 0) + 1
        if name_counts[base_label] > 1:
            base_label = f"{base_label} {name_counts[base_label]}"
        labels[c] = f"{base_label} (n={n})"
    return labels


# ---------------------------------------------------------------------------
# Panel A — mean trajectory lines
# ---------------------------------------------------------------------------

def draw_panel_a(ax: plt.Axes, long_df: pd.DataFrame,
                 cluster_df: pd.DataFrame, cluster_labels: dict[int, str]) -> None:
    merged = long_df.merge(cluster_df[["DFCI_MRN", "cluster"]], on="DFCI_MRN")

    times = sorted(merged["time"].unique())
    for i, (cluster_id, color) in enumerate(zip(sorted(cluster_labels.keys()), CLUSTER_COLORS)):
        sub = merged[merged["cluster"] == cluster_id]
        mean_traj = sub.groupby("time")["risk_score"].mean()
        sem_traj  = sub.groupby("time")["risk_score"].sem()

        ax.plot(mean_traj.index, mean_traj.values,
                color=color, lw=2.2, label=cluster_labels[cluster_id])
        ax.fill_between(mean_traj.index,
                        mean_traj.values - 1.96 * sem_traj.values,
                        mean_traj.values + 1.96 * sem_traj.values,
                        alpha=0.18, color=color)

    ax.set_xlabel("Months since treatment start")
    ax.set_ylabel("Predicted mortality risk score")
    ax.set_title(f"Mortality Risk Trajectories ({N_CLUSTERS} Clusters)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.4)


# ---------------------------------------------------------------------------
# Panel B — KM survival curves by cluster
# ---------------------------------------------------------------------------

def draw_panel_b(ax: plt.Axes, cluster_df: pd.DataFrame,
                 cluster_labels: dict[int, str]) -> None:
    """KM curves using actual death/tt_death from the cohort."""
    surv_df = pd.read_csv(
        os.path.join(SURV_PATH, "death_met_embedding_prediction_df.csv"),
        usecols=["DFCI_MRN", "death", "tt_death"]
    ).rename(columns={"death": "event", "tt_death": "time"})
    surv_df["event"] = surv_df["event"].astype(bool)

    merged = cluster_df[["DFCI_MRN", "cluster"]].merge(surv_df, on="DFCI_MRN")
    merged = merged.dropna(subset=["event", "time"])

    at_risk_times = [0, 24, 48, 72, 96, 120]

    for cluster_id, color in zip(sorted(cluster_labels.keys()), CLUSTER_COLORS):
        sub = merged[merged["cluster"] == cluster_id]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["time"] / 30.44, sub["event"],   # convert days → months
                label=cluster_labels[cluster_id])
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=1.8)

    # Log-rank p-value
    try:
        result = multivariate_logrank_test(
            merged["time"], merged["cluster"], merged["event"]
        )
        p = result.p_value
        p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    except Exception:
        p_str = ""

    ax.set_xlabel("Months since treatment")
    ax.set_ylabel("Survival probability")
    ax.set_title("Overall Survival by Trajectory Cluster")
    ax.set_ylim(-0.02, 1.05)
    if p_str:
        ax.text(0.97, 0.97, p_str, transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc"))
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    # At-risk table
    for qi, (cluster_id, color) in enumerate(zip(sorted(cluster_labels.keys()), CLUSTER_COLORS)):
        sub = merged[merged["cluster"] == cluster_id]
        y_pos = -0.14 - 0.06 * qi
        ax.text(-0.01, y_pos, cluster_labels[cluster_id][:20],
                transform=ax.transAxes, fontsize=6, ha="right", va="top", color=color)
        for ti, t in enumerate(at_risk_times):
            n = (sub["time"] / 30.44 >= t).sum()
            x_frac = t / (ax.get_xlim()[1] or 120)
            ax.text(x_frac, y_pos, str(n), transform=ax.transAxes,
                    fontsize=5.5, ha="center", va="top", color=color)


# ---------------------------------------------------------------------------
# Panel C — clinical characteristics by cluster
# ---------------------------------------------------------------------------

def load_clinical_features(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """Merge cluster assignments with sex, stage IV, ICI treatment."""
    # Demographics + death indicator
    surv_df = pd.read_csv(
        os.path.join(SURV_PATH, "death_met_embedding_prediction_df.csv"),
        usecols=["DFCI_MRN", "GENDER", "death", "tt_death"]
    )

    # Stage
    stage_df = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_stage_df.csv"))
    stage_cols_4 = [c for c in stage_df.columns if "STAGE_4" in c or "STAGE_IV" in c.upper()]
    if stage_cols_4:
        stage_df["stage_iv"] = stage_df[stage_cols_4].any(axis=1).astype(int)
    else:
        # Fall back to numeric STAGE column if present
        if "STAGE" in stage_df.columns:
            stage_df["stage_iv"] = (stage_df["STAGE"] == 4).astype(int)
        else:
            stage_df["stage_iv"] = 0
    stage_df = stage_df[["DFCI_MRN", "stage_iv"]]

    # ICI treatment flag
    treat_df = pd.read_csv(os.path.join(FEATURE_PATH, "categorical_treatment_data_by_line.csv"))
    ici_col = next((c for c in treat_df.columns if "ICI" in c.upper() and "_1" in c), None)
    if ici_col:
        line1_mask = pd.to_numeric(treat_df["treatment_line"], errors="coerce") == 1
        ici_df = treat_df.loc[line1_mask, ["DFCI_MRN", ici_col]].copy()
        ici_df = ici_df.rename(columns={ici_col: "ici_treated"})
    else:
        # If treatment_line column exists, mark anyone with any ICI
        ici_cols = [c for c in treat_df.columns if "ICI" in c.upper() and c != "DFCI_MRN"]
        if ici_cols:
            ici_df = treat_df[["DFCI_MRN"]].copy()
            ici_df["ici_treated"] = treat_df[ici_cols].any(axis=1).astype(int)
            ici_df = ici_df.drop_duplicates("DFCI_MRN")
        else:
            ici_df = cluster_df[["DFCI_MRN"]].copy()
            ici_df["ici_treated"] = 0

    merged = (cluster_df[["DFCI_MRN", "cluster"]]
              .merge(surv_df, on="DFCI_MRN", how="left")
              .merge(stage_df, on="DFCI_MRN", how="left")
              .merge(ici_df, on="DFCI_MRN", how="left"))

    gender = merged["GENDER"].astype("string").str.strip().str.upper()
    merged["male"] = gender.isin(["M", "MALE"]).astype(float)
    merged["stage_iv"] = pd.to_numeric(merged["stage_iv"], errors="coerce")
    merged["ici_treated"] = pd.to_numeric(merged["ici_treated"], errors="coerce")
    return merged


def draw_panel_c(ax: plt.Axes, clinical_df: pd.DataFrame,
                 cluster_labels: dict[int, str]) -> None:
    features = {
        "% Male": "male",
        "% Stage IV": "stage_iv",
        "% ICI Treated": "ici_treated",
    }
    cluster_ids = sorted(cluster_labels.keys())
    n_features = len(features)
    n_clusters = len(cluster_ids)
    x = np.arange(n_features)
    width = 0.8 / n_clusters

    for ci, (cluster_id, color) in enumerate(zip(cluster_ids, CLUSTER_COLORS)):
        sub = clinical_df[clinical_df["cluster"] == cluster_id]
        vals = [100 * sub[col].mean() for col in features.values()]
        offset = (ci - n_clusters / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      color=color, edgecolor="white", linewidth=0.4,
                      label=cluster_labels[cluster_id])

    # Chi-square significance stars above each feature group
    for fi, (feat_name, col) in enumerate(features.items()):
        ct = clinical_df.dropna(subset=[col])
        contingency = pd.crosstab(ct["cluster"], ct[col])
        if contingency.shape == (n_clusters, 2):
            try:
                _, p, _, _ = chi2_contingency(contingency)
                stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                if stars:
                    ax.text(fi, ax.get_ylim()[1] + 1, stars,
                            ha="center", va="bottom", fontsize=10)
            except Exception:
                pass

    ax.set_xticks(x)
    ax.set_xticklabels(list(features.keys()))
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Clinical Characteristics by Trajectory Cluster")
    ax.legend(frameon=False, fontsize=7, loc="upper right",
              bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="y", alpha=0.4)
    ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_style()

    print(f"Loading trajectory data from: {TRAJECTORY_FILE}")
    surv_traj = load_trajectories()

    print("Converting to long format and extracting features...")
    long_df = wide_to_long(surv_traj)
    features_df = extract_trajectory_features(long_df)

    print(f"Clustering {len(features_df)} patients into {N_CLUSTERS} clusters...")
    cluster_df = cluster_patients(features_df, n_clusters=N_CLUSTERS)
    cluster_labels = apply_cluster_labels(cluster_df)

    print("Cluster sizes:")
    for c, label in cluster_labels.items():
        print(f"  {label}: {(cluster_df['cluster'] == c).sum()} patients")

    print("Loading clinical features for Panel C...")
    clinical_df = load_clinical_features(cluster_df)

    # Build figure — 2-row layout: A and B side by side (top), C spanning bottom
    fig = plt.figure(figsize=(14, 12))
    ax_a = fig.add_axes([0.07, 0.54, 0.40, 0.38])
    ax_b = fig.add_axes([0.57, 0.54, 0.38, 0.38])
    ax_c = fig.add_axes([0.10, 0.07, 0.80, 0.34])

    draw_panel_a(ax_a, long_df, cluster_df, cluster_labels)
    draw_panel_b(ax_b, cluster_df, cluster_labels)
    draw_panel_c(ax_c, clinical_df, cluster_labels)

    for label, ax in {"A": ax_a, "B": ax_b, "C": ax_c}.items():
        ax.text(-0.10, 1.04, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="right")

    out_stem = os.path.join(OUTPUT_DIR, "figure4_trajectories")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", facecolor="white", bbox_inches="tight")
    print(f"Saved figure 4 → {out_stem}.png/.pdf")


if __name__ == "__main__":
    main()
