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


def _missing(ax: plt.Axes, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#777")
    ax.set_axis_off()


def _month_to_float(values: pd.Series) -> pd.Series:
    return values.astype(str).str.extract(r"(\d+)")[0].astype(float)


def _cluster_label(cluster: int, n: int) -> str:
    names = ["Stable Low", "Intermediate", "Stable High", "Rapidly Increasing", "Rebounding"]
    name = names[min(cluster, len(names) - 1)]
    return f"{name} (n={n:,})"


means = load_figure_data("fig4_cluster_means.csv")
km = load_figure_data("fig4_km_data.csv")


# %% fig4a: risk trajectories by cluster
fig, ax = plt.subplots(figsize=(6.4, 4.8))
if means.empty:
    _missing(ax, "fig4_cluster_means.csv empty")
else:
    means = means.copy()
    means["month_num"] = _month_to_float(means["month"])
    for k, g in means.groupby("cluster"):
        g = g.sort_values("month_num")
        n = int(g["n_patients"].iloc[0])
        color = CLUSTER_COLORS[int(k) % len(CLUSTER_COLORS)]
        ax.plot(g["month_num"], g["mean"], lw=2.2, color=color, label=_cluster_label(int(k), n))
        ax.fill_between(g["month_num"], g["mean"] - 1.96 * g["sem"], g["mean"] + 1.96 * g["sem"],
                        color=color, alpha=0.18)
    ax.set_xlabel("Months post-treatment")
    ax.set_ylabel("Standardized Mortality Risk Score")
    ax.set_title("Risk Score Trajectories by Cluster")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.35)
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


# %% fig4c: clinical characteristics by trajectory cluster
stage = load_figure_data("fig4_cluster_composition_stage.csv")
treatment = load_figure_data("fig4_cluster_composition_treatment.csv")
cancer = load_figure_data("fig4_cluster_composition_cancer.csv")
fig, ax = plt.subplots(figsize=(10.6, 4.4))
if stage.empty and treatment.empty and cancer.empty and km.empty:
    _missing(ax, "cluster characteristic data empty")
else:
    cluster_ids = sorted(set(
        pd.concat([
            stage.get("cluster", pd.Series(dtype=float)),
            treatment.get("cluster", pd.Series(dtype=float)),
            cancer.get("cluster", pd.Series(dtype=float)),
            km.get("cluster", pd.Series(dtype=float)),
        ]).dropna().astype(int)
    ))
    features: list[tuple[str, dict[int, float]]] = []

    # Composition CSVs (per prep_figure_4._composition) carry one column per
    # top category with the prefix already stripped (e.g. "CANCER_STAGE_IV" -> "IV"),
    # plus a trailing "OTHER" column. Use exact-token matching so "IV" doesn't
    # accidentally match e.g. "IIV", and "ICI" doesn't match a longer combo
    # like "ICI_PLUS_CHEMO" — list those alternates explicitly.
    STAGE_IV_TOKENS = re.compile(r"^(IV|4)[A-D]?$", re.IGNORECASE)
    ICI_TOKENS = re.compile(
        r"^(ICI|IMMUNOTHERAPY|PD1|PDL1|PD_?L1|CHECKPOINT(?:_INHIBITOR)?)$",
        re.IGNORECASE,
    )

    def _category_cols(idx: pd.DataFrame) -> list[str]:
        return [c for c in idx.columns if c != "OTHER"]

    def _top_cat_share(idx: pd.DataFrame, label_suffix: str) -> tuple[str, dict[int, float]] | None:
        cats = _category_cols(idx)
        if not cats:
            return None
        col = idx[cats].sum().sort_values(ascending=False).index[0]
        label = f"{str(col).replace('_', ' ')} ({label_suffix})"
        return label, (100 * idx[col]).to_dict()

    if not stage.empty:
        stage_idx = stage.set_index("cluster")
        stage4_cols = [c for c in stage_idx.columns
                       if c != "OTHER" and STAGE_IV_TOKENS.match(str(c))]
        if stage4_cols:
            features.append(("% Stage IV", (100 * stage_idx[stage4_cols].sum(axis=1)).to_dict()))
        else:
            top = _top_cat_share(stage_idx, "stage")
            if top is not None:
                print(f"  no Stage IV columns detected; falling back to top stage: {top[0]}")
                features.append(top)

    if not treatment.empty:
        tx_idx = treatment.set_index("cluster")
        ici_cols = [c for c in tx_idx.columns
                    if c != "OTHER" and ICI_TOKENS.match(str(c))]
        if ici_cols:
            features.append(("% ICI Treated", (100 * tx_idx[ici_cols].sum(axis=1)).to_dict()))
        else:
            top = _top_cat_share(tx_idx, "tx")
            if top is not None:
                print(f"  no ICI columns detected; falling back to top treatment: {top[0]}")
                features.append(top)

    if not cancer.empty:
        cancer_idx = cancer.set_index("cluster")
        top = _top_cat_share(cancer_idx, "cancer")
        if top is not None:
            features.append(top)

    if not km.empty:
        alive5 = {}
        for k, sub in km.groupby("cluster"):
            alive5[int(k)] = 100 * ((sub["tt_death"] / 30.44 >= 60) | (sub["death"] == 0)).mean()
        features.append(("Alive at 5 years", alive5))

    if not features:
        _missing(ax, "no supported clinical characteristics")
    else:
        x = np.arange(len(features))
        width = 0.78 / max(len(cluster_ids), 1)
        for i, k in enumerate(cluster_ids):
            vals = [feat_map.get(k, np.nan) for _, feat_map in features]
            offset = (i - len(cluster_ids) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width=width * 0.92,
                   color=CLUSTER_COLORS[int(k) % len(CLUSTER_COLORS)],
                   edgecolor="white", label=f"Cluster {k}")
        ax.set_xticks(x)
        ax.set_xticklabels([name for name, _ in features])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Clinical Characteristics by Trajectory Cluster")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.35)
        ax.grid(axis="x", visible=False)
save_panel(fig, "fig4c")
plt.close(fig)
