"""Render Figure 2 panels using the old text-vs-base target."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from _figure_utils import apply_style, load_figure_data, save_panel


apply_style()

SCHEME_COLORS = {
    "death_met": "#E74C3C",
    "icd3_post": "#3498DB",
    "icd4_post": "#2ECC71",
    "phecode_post": "#9B59B6",
}
SCHEME_MARKERS = {"death_met": "D", "icd3_post": "o", "icd4_post": "^", "phecode_post": "s"}
SCHEME_LABELS = {"death_met": "death_met", "icd3_post": "ICD3", "icd4_post": "ICD4", "phecode_post": "PhecodeX"}
RISK_COLORS = {"low": "#2E86C1", "mid": "#F28E2B", "high": "#E74C3C"}
# Shared low->high palette for the stage-vs-risk panel (stage I->IV, quartile Q1->Q4)
ORDINAL4 = ["#2E86C1", "#58A55C", "#F28E2B", "#E74C3C"]


def _missing(ax: plt.Axes, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#777")
    ax.set_axis_off()


metrics = load_figure_data("fig2_full_cohort_metrics.csv")


# %% fig2a: text vs base scatter
fig, ax = plt.subplots(figsize=(6.4, 5.0))
if metrics.empty:
    _missing(ax, "fig2_full_cohort_metrics.csv empty")
else:
    plot_df = metrics.dropna(subset=["base_cindex", "text_cindex"]).copy()
    plot_df["delta"] = plot_df["text_cindex"] - plot_df["base_cindex"]
    for scheme, g in plot_df.groupby("scheme"):
        ax.scatter(g["base_cindex"], g["text_cindex"],
                   marker=SCHEME_MARKERS.get(scheme, "o"),
                   color=SCHEME_COLORS.get(scheme, "#777"),
                   s=26, alpha=0.55, edgecolors="none",
                   label=f"{SCHEME_LABELS.get(scheme, scheme)} (n={len(g)})")
    lo = max(0.45, plot_df[["base_cindex", "text_cindex"]].min().min() - 0.02)
    hi = min(1.0, plot_df[["base_cindex", "text_cindex"]].max().max() + 0.02)
    ax.plot([lo, hi], [lo, hi], ls="--", color="#666", lw=1)
    top = plot_df.nlargest(min(5, len(plot_df)), "delta")
    for _, row in top.iterrows():
        label = str(row["event"]).replace("_", " ")[:28]
        ax.annotate(label, (row["base_cindex"], row["text_cindex"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=6.5,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#999"))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Base Model C-index")
    ax.set_ylabel("Text Model C-index")
    ax.set_title("Text vs. Base Model Performance")
    ax.legend(loc="lower right", fontsize=7)
save_panel(fig, "fig2a")
plt.close(fig)


# %% fig2b: delta violin by scheme
fig, ax = plt.subplots(figsize=(6.0, 4.8))
if metrics.empty:
    _missing(ax, "fig2_full_cohort_metrics.csv empty")
else:
    plot_df = metrics.dropna(subset=["base_cindex", "text_cindex"]).copy()
    plot_df["delta"] = plot_df["text_cindex"] - plot_df["base_cindex"]
    order = [s for s in SCHEME_LABELS if s in set(plot_df["scheme"])]
    values = [plot_df.loc[plot_df["scheme"] == s, "delta"].values for s in order]
    parts = ax.violinplot(values, positions=np.arange(len(order)), widths=0.65,
                          showmeans=False, showmedians=True, showextrema=True)
    for body, scheme in zip(parts["bodies"], order):
        body.set_facecolor(SCHEME_COLORS[scheme])
        body.set_alpha(0.45)
        body.set_edgecolor("none")
    parts["cmedians"].set_color("#111")
    rng = np.random.default_rng(0)
    for i, (scheme, vals) in enumerate(zip(order, values)):
        if len(vals) == 0:
            continue
        x = i + rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(x, vals, s=10, color=SCHEME_COLORS[scheme], alpha=0.35, edgecolors="none")
        ax.text(i, ax.get_ylim()[1] * 0.94, f"n={len(vals)}\nmed={np.median(vals):+.3f}",
                ha="center", va="top", fontsize=7)
    ax.axhline(0, color="#333", ls="--", lw=1)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([SCHEME_LABELS[s] for s in order])
    ax.set_ylabel("Delta C-index (Text - Base)")
    ax.set_title("C-index Improvement by Scheme")
save_panel(fig, "fig2b")
plt.close(fig)


# %% fig2c: heatmap of text model C-index by endpoint and scheme
cancer_heat = load_figure_data("fig2_cancer_endpoint_heatmap.csv")
fig, ax = plt.subplots(figsize=(10.8, 4.2))
if cancer_heat.empty:
    _missing(ax, "fig2_cancer_endpoint_heatmap.csv empty")
else:
    plot_df = cancer_heat.dropna(subset=["text_cindex"]).copy()
    event_order = (
        plot_df.groupby("event")["text_cindex"].mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    cancer_order = (
        plot_df.groupby("cancer_type")["n"].sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    mat = plot_df.pivot_table(index="event", columns="cancer_type", values="text_cindex", aggfunc="mean")
    mat = mat.reindex(index=event_order, columns=cancer_order)
    im = ax.imshow(mat.values, aspect="auto", cmap="Blues", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([str(e).replace("_", " ") for e in mat.index], fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iloc[i, j]
            if pd.isna(v):
                continue
            ax.text(j, i, "*" if v >= 0.60 else "", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
    ax.set_title("Text Model C-index by Cancer Type and Endpoint")
    fig.colorbar(im, ax=ax, label="C-index", fraction=0.03, pad=0.02)
save_panel(fig, "fig2c")
plt.close(fig)


# %% fig2d: mortality by risk-score tertile (text solid, base dashed)
km_tert = load_figure_data("fig2_km_tertiles.csv")
fig, ax = plt.subplots(figsize=(6.4, 5.0))
if km_tert.empty:
    _missing(ax, "fig2_km_tertiles.csv empty")
else:
    months = km_tert["tt_death"] / 30.44
    death = km_tert["death"].astype(int)
    kmf = KaplanMeierFitter()
    for t in ("low", "mid", "high"):
        sub = km_tert[km_tert["text_tertile"] == t]
        if sub.empty:
            continue
        kmf.fit(sub["tt_death"] / 30.44, sub["death"].astype(int),
                label=f"text {t} (n={len(sub):,})")
        kmf.plot_survival_function(ax=ax, ci_show=False,
                                   color=RISK_COLORS[t], lw=2)
    for t in ("low", "mid", "high"):
        sub = km_tert[km_tert["base_tertile"] == t]
        if sub.empty:
            continue
        kmf.fit(sub["tt_death"] / 30.44, sub["death"].astype(int),
                label=f"base {t} (n={len(sub):,})")
        kmf.plot_survival_function(ax=ax, ci_show=False,
                                   color=RISK_COLORS[t], lw=1.2,
                                   linestyle="--")
    try:
        lr_text = multivariate_logrank_test(months, km_tert["text_tertile"], death)
        lr_base = multivariate_logrank_test(months, km_tert["base_tertile"], death)
        ax.text(0.03, 0.06,
                f"text logrank p={lr_text.p_value:.1e}\n"
                f"base logrank p={lr_base.p_value:.1e}",
                transform=ax.transAxes, fontsize=7.5, style="italic")
    except Exception as exc:
        print(f"  logrank test failed: {exc}")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Months from first treatment")
    ax.set_ylabel("Overall survival")
    ax.set_title("Mortality by Risk-Score Tertile\n(text solid, base dashed)")
    ax.legend(loc="upper right", fontsize=6.5, ncol=2)
save_panel(fig, "fig2d")
plt.close(fig)


# %% fig2e: survival by cancer stage vs text risk-score quartile (side-by-side)
sr = load_figure_data("fig2_km_stage_vs_risk.csv")
cidx_tbl = load_figure_data("fig2_stage_vs_risk_cindex.csv")
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
if sr.empty:
    _missing(axL, "fig2_km_stage_vs_risk.csv empty")
    _missing(axR, "fig2_km_stage_vs_risk.csv empty")
else:
    months = sr["tt_death"] / 30.44
    death = sr["death"].astype(int)

    def _cindex_for(predictor: str) -> float:
        if cidx_tbl.empty or "predictor" not in cidx_tbl.columns:
            return float("nan")
        hit = cidx_tbl.loc[cidx_tbl["predictor"] == predictor, "cindex"]
        return float(hit.iloc[0]) if len(hit) else float("nan")

    def _km_by_group(ax, group_col, groups, title):
        kmf = KaplanMeierFitter()
        for grp, color in zip(groups, ORDINAL4):
            sub = sr[sr[group_col] == grp]
            if sub.empty:
                continue
            kmf.fit(sub["tt_death"] / 30.44, sub["death"].astype(int),
                    label=f"{grp} (n={len(sub):,})")
            kmf.plot_survival_function(ax=ax, ci_show=False, color=color, lw=2)
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("Months from first treatment")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=6.5)

    _km_by_group(axL, "stage_group", ["I", "II", "III", "IV"], "Survival by Cancer Stage")
    _km_by_group(axR, "risk_quartile", ["Q1", "Q2", "Q3", "Q4"],
                 "Survival by Text Risk-Score Quartile")
    axL.set_ylabel("Overall survival")

    try:
        lr_stage = multivariate_logrank_test(months, sr["stage_group"], death)
        axL.text(0.03, 0.06,
                 f"c-index={_cindex_for('stage'):.3f}\nlogrank p={lr_stage.p_value:.1e}",
                 transform=axL.transAxes, fontsize=7.5, style="italic")
        lr_risk = multivariate_logrank_test(months, sr["risk_quartile"], death)
        axR.text(0.03, 0.06,
                 f"c-index={_cindex_for('text_risk'):.3f}\nlogrank p={lr_risk.p_value:.1e}",
                 transform=axR.transAxes, fontsize=7.5, style="italic")
    except Exception as exc:
        print(f"  logrank test failed: {exc}")
save_panel(fig, "fig2e")
plt.close(fig)
