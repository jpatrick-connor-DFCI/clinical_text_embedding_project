"""Render Figure 4 panels using the old trajectory-cluster target."""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from _figure_utils import apply_style, load_figure_data, save_panel, CLUSTER_COLORS


apply_style()

N_CLUSTERS = 4  # mirrors prep_figure_4.N_CLUSTERS; used to mark the chosen k in figS1a


def _missing(ax: plt.Axes, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#777")
    ax.set_axis_off()


def _month_to_float(values: pd.Series) -> pd.Series:
    return values.astype(str).str.extract(r"(\d+)")[0].astype(float)


def _cluster_label(cluster: int, n: int) -> str:
    names = ["Stable Low", "Intermediate", "Stable High", "Rapidly Increasing", "Rebounding"]
    name = names[min(cluster, len(names) - 1)]
    return f"{name} (n={n:,})"


heatmap = load_figure_data("fig4_trajectories_heatmap.csv")
km = load_figure_data("fig4_km_data.csv")


# %% fig4a: per-patient mortality-risk trajectory heatmap, grouped by cluster
fig, ax = plt.subplots(figsize=(6.4, 4.8))
if heatmap.empty:
    _missing(ax, "fig4_trajectories_heatmap.csv empty")
else:
    hm = heatmap.sort_values("cluster", kind="stable").reset_index(drop=True)
    month_cols = [c for c in hm.columns if c not in ("DFCI_MRN", "cluster")]
    nums = _month_to_float(pd.Series(month_cols)).to_numpy()
    order = np.argsort(nums)
    month_cols = [month_cols[i] for i in order]
    x_months = nums[order]
    M = hm[month_cols].to_numpy(dtype=float)

    im = ax.imshow(M, aspect="auto", cmap="magma", interpolation="nearest")
    n_months = len(month_cols)
    xticks = np.unique(np.linspace(0, n_months - 1, min(n_months, 7)).astype(int))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x_months[t]:.0f}" for t in xticks])

    # White separators between clusters; cluster names as y-tick labels at block centers
    boundaries = hm.groupby("cluster", sort=True).size()
    y0, mids, ylabels = 0, [], []
    for k, n in boundaries.items():
        if y0 > 0:
            ax.axhline(y0 - 0.5, color="white", lw=1.2)
        mids.append(y0 + n / 2 - 0.5)
        ylabels.append(_cluster_label(int(k), int(n)))
        y0 += n
    ax.set_yticks(mids)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("Months post-treatment")
    ax.set_title("Mortality-Risk Trajectories by Cluster")
    fig.colorbar(im, ax=ax, label="Std. mortality risk", fraction=0.046, pad=0.04)
save_panel(fig, "fig4a")
plt.close(fig)


# %% fig4b: conditional KM survival from month 60 onwards
# Cluster assignment requires a 60-month risk-score trajectory, so every
# clustered patient is by construction alive at month 60. Plotting from time 0
# pretends those 60 months were observed for cluster assignment, which is
# immortal-time bias. Condition on surviving to month 60 (left-truncation at
# the cluster-entry boundary) so the curves start at S(60) = 1.0.
ENTRY_MONTHS = 60.0
fig, ax = plt.subplots(figsize=(6.4, 4.8))
if km.empty:
    _missing(ax, "fig4_km_data.csv empty")
else:
    km_months = km.assign(_months=km["tt_death"] / 30.44)
    km_at_risk = km_months[km_months["_months"] >= ENTRY_MONTHS]
    if km_at_risk.empty:
        _missing(ax, f"no patients followed past {int(ENTRY_MONTHS)} months")
    else:
        kmf = KaplanMeierFitter()
        for k in sorted(km_at_risk["cluster"].dropna().unique()):
            sub = km_at_risk[km_at_risk["cluster"] == k]
            if sub.empty:
                continue
            color = CLUSTER_COLORS[int(k) % len(CLUSTER_COLORS)]
            label = _cluster_label(int(k), len(sub))
            kmf.fit(
                durations=sub["_months"],
                event_observed=sub["death"],
                entry=np.full(len(sub), ENTRY_MONTHS),
                label=label,
            )
            kmf.plot_survival_function(ax=ax, ci_show=False, color=color, lw=2)
        try:
            lr = multivariate_logrank_test(
                km_at_risk["_months"],
                km_at_risk["cluster"],
                km_at_risk["death"],
            )
            ax.text(0.03, 0.08, f"Log-rank p={lr.p_value:.1e}",
                    transform=ax.transAxes, fontsize=8, style="italic")
        except Exception as exc:
            print(f"logrank test failed: {exc}")
        ax.set_xlim(ENTRY_MONTHS, 120)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("Months from first treatment")
        ax.set_ylabel("Overall Survival Probability (conditional)")
        ax.set_title(f"KM Overall Survival by Cluster\n"
                     f"(conditional on survival to month {int(ENTRY_MONTHS)})")
        ax.legend(fontsize=7, loc="upper right")
save_panel(fig, "fig4b")
plt.close(fig)


# %% fig4c: disease-severity characteristics by trajectory cluster
stage = load_figure_data("fig4_cluster_composition_stage.csv")
treatment = load_figure_data("fig4_cluster_composition_treatment.csv")
severity = load_figure_data("fig4_cluster_severity.csv")

# Composition CSVs carry one column per top category with the prefix stripped
# (e.g. "CANCER_STAGE_IV" -> "IV"), plus a trailing "OTHER" column. Match exact
# tokens so "IV" doesn't catch "IIV" and "ICI" doesn't catch "ICI_PLUS_CHEMO".
STAGE_IV_TOKENS = re.compile(r"^(IV|4(\.0+)?)[A-D]?$", re.IGNORECASE)
ICI_TOKENS = re.compile(
    r"^(ICI|IMMUNOTHERAPY|PD1|PDL1|PD_?L1|"
    r"IMMUNE[ _]CHECKPOINT[ _]INHIBITORS?|"
    r"CHECKPOINT(?:_INHIBITOR)?)$",
    re.IGNORECASE,
)


def _share_by_token(comp_df: pd.DataFrame, token_re: re.Pattern) -> dict[int, float]:
    """{cluster: percentage} summing composition columns whose name matches token_re."""
    if comp_df.empty:
        return {}
    idx = comp_df.set_index("cluster")
    cols = [c for c in idx.columns if c != "OTHER" and token_re.match(str(c))]
    if not cols:
        return {}
    return {int(k): v for k, v in (100 * idx[cols].sum(axis=1)).items()}


stage_iv = _share_by_token(stage, STAGE_IV_TOKENS)
ici = _share_by_token(treatment, ICI_TOKENS)
met_sites: dict[int, float] = {}
rmst: dict[int, float] = {}
if not severity.empty:
    sev_idx = severity.set_index("cluster")
    met_sites = {int(k): v for k, v in sev_idx["mean_met_sites"].dropna().items()}
    rmst = {int(k): v for k, v in sev_idx["rmst_months"].dropna().items()}

# (title, y-axis units, {cluster: value})
characteristics = [
    ("% Stage IV", "Percentage (%)", stage_iv),
    ("% ICI Treated", "Percentage (%)", ici),
    ("Mean # met sites", "Sites (0-7)", met_sites),
    ("10-yr RMST", "Months", rmst),
]

cluster_ids = sorted({int(k) for _, _, d in characteristics for k in d})

fig, axes = plt.subplots(1, len(characteristics), figsize=(12.0, 3.8))
axes = np.atleast_1d(axes)
if not cluster_ids:
    for ax in axes:
        _missing(ax, "cluster characteristic data empty")
else:
    colors = [CLUSTER_COLORS[k % len(CLUSTER_COLORS)] for k in cluster_ids]
    for ax, (title, units, values) in zip(axes, characteristics):
        if not values:
            _missing(ax, "no data")
            continue
        heights = [values.get(k, np.nan) for k in cluster_ids]
        ax.bar(range(len(cluster_ids)), heights, color=colors, edgecolor="white")
        # Flag clusters with no survival estimate (RMST should always be defined)
        if units == "Months":
            for xi, k in enumerate(cluster_ids):
                if k not in values:
                    ax.text(xi, 0, "n/a", ha="center", va="bottom", fontsize=7, color="#777")
        ax.set_xticks(range(len(cluster_ids)))
        ax.set_xticklabels([str(k) for k in cluster_ids])
        ax.set_xlabel("Cluster")
        ax.set_ylabel(units)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.35)
        ax.grid(axis="x", visible=False)
        if title.startswith("%"):
            ax.set_ylim(0, 100)
    fig.suptitle("Disease-Severity Characteristics by Trajectory Cluster",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
save_panel(fig, "fig4c")
plt.close(fig)


# %% figS1a (appendix): silhouette score vs k, justifying the chosen cluster count
sil = load_figure_data("fig4_silhouette.csv")
fig, ax = plt.subplots(figsize=(6.0, 4.4))
if sil.empty:
    _missing(ax, "fig4_silhouette.csv empty")
else:
    sil = sil.sort_values("k")
    ax.plot(sil["k"], sil["silhouette"], marker="o", color="#2E86C1", lw=2)
    best_k = int(sil.loc[sil["silhouette"].idxmax(), "k"])
    ax.axvline(N_CLUSTERS, color="#E74C3C", ls="--", lw=1.5,
               label=f"chosen k={N_CLUSTERS}")
    ax.scatter([best_k], [sil["silhouette"].max()], color="#E74C3C", zorder=5,
               label=f"best silhouette (k={best_k})")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Mean silhouette score")
    ax.set_title("Trajectory Cluster-Count Selection")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.35)
save_panel(fig, "figS1a")
plt.close(fig)
