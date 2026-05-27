"""Render Figure 5 panels using the old biomarker-discovery target.

Target content from the old rendered mockup:
- A: propensity-model ROC curves with an inset cancer-type AUC bar plot
- B: biomarker association robustness across analysis specifications
- C: three representative marker-by-ICI Kaplan-Meier interaction panels
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter

from _figure_utils import apply_style, load_figure_data, save_panel


apply_style()

BENEFIT_COLOR = "#2E6F9E"  # ICI benefit (HR<1)
HARM_COLOR = "#E76F51"     # ICI harm (HR>1)
TEAL = "#2A9D8F"
NS_GRAY = "#999999"
LIGHT_GRAY = "#EAEAEA"


def _missing(ax: plt.Axes, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#777")
    ax.set_axis_off()


def _pretty_model(model: str) -> str:
    labels = {
        "covariates_only": "Covariates only",
        "covariates_plus_embeddings": "Text + covariates",
    }
    return labels.get(str(model), str(model).replace("_", " "))


def _pretty_spec(spec: str) -> str:
    parts = str(spec).split("|")
    if len(parts) == 3:
        cohort, ps_model, weight = parts
        ps = "embed PS" if "embedding" in ps_model else "covar PS"
        return f"{weight}\n({ps})"
    return str(spec).replace("_", " ")


def _roc_curve_auc(y_true: pd.Series, score: pd.Series) -> tuple[np.ndarray, np.ndarray, float]:
    y = pd.to_numeric(y_true, errors="coerce").to_numpy()
    s = pd.to_numeric(score, errors="coerce").to_numpy()
    keep = np.isfinite(y) & np.isfinite(s)
    y = y[keep].astype(int)
    s = s[keep]
    if len(np.unique(y)) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    order = np.argsort(-s)
    y = y[order]
    threshold_idx = np.r_[np.where(np.diff(s[order]) != 0)[0], len(y) - 1]
    tp = np.cumsum(y == 1)[threshold_idx]
    fp = np.cumsum(y == 0)[threshold_idx]
    tpr = np.r_[0.0, tp / max(tp[-1], 1), 1.0]
    fpr = np.r_[0.0, fp / max(fp[-1], 1), 1.0]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def _km_title(marker: str, cancer: str, hr: float | None = None) -> str:
    marker = str(marker).replace("_", " ")
    cancer = str(cancer).replace("_", " ")
    if hr is None or not np.isfinite(hr):
        return f"{marker}\n{cancer}"
    direction = "ICI benefit" if hr < 1 else "ICI harm"
    return f"{marker}\n{direction}"


ps_predictions = load_figure_data("fig5_ps_predictions.csv")
robust = load_figure_data("fig5_robust_hits.csv")
km_examples = load_figure_data("fig5_km_examples.csv")
km_top = load_figure_data("fig5_km_top_hit.csv")
meta = load_figure_data("fig5_top_hit_meta.csv")


# %% fig5a: propensity ROC + cancer-type AUC inset
fig, ax = plt.subplots(figsize=(8.0, 6.0))
if ps_predictions.empty:
    _missing(ax, "fig5_ps_predictions.csv empty")
else:
    ps = ps_predictions.dropna(subset=["model_probs", "ground_truth"]).copy()
    colors = {
        "covariates_only": "#F28E2B",
        "covariates_plus_embeddings": TEAL,
    }
    model_order = ["covariates_only", "covariates_plus_embeddings"]
    model_order += [m for m in ps["ps_model"].dropna().unique() if m not in model_order]
    for model in model_order:
        sub = ps[ps["ps_model"] == model]
        if sub.empty:
            continue
        fpr, tpr, auc = _roc_curve_auc(sub["ground_truth"], sub["model_probs"])
        label = f"{_pretty_model(model)} (AUC={auc:.2f})" if np.isfinite(auc) else _pretty_model(model)
        ax.plot(fpr, tpr, lw=2.0, color=colors.get(model, "#457B9D"), label=label)

    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1.0, label="Random classifier")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Propensity Score Model: Predicting ICI Receipt", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", frameon=True)

    text_model = ps[ps["ps_model"] == "covariates_plus_embeddings"].dropna(subset=["cancer_type"])
    rows = []
    for cancer, sub in text_model.groupby("cancer_type"):
        y = pd.to_numeric(sub["ground_truth"], errors="coerce")
        if len(sub) < 30 or y.nunique(dropna=True) < 2:
            continue
        _, _, auc = _roc_curve_auc(sub["ground_truth"], sub["model_probs"])
        rows.append({"cancer_type": cancer, "auc": auc, "n": len(sub)})
    auc_by_cancer = pd.DataFrame(rows).dropna(subset=["auc"]).sort_values("auc").tail(6)
    if not auc_by_cancer.empty:
        inset = ax.inset_axes([0.36, 0.08, 0.46, 0.36])
        y_pos = np.arange(len(auc_by_cancer))
        inset.barh(y_pos, auc_by_cancer["auc"], color=plt.cm.tab10(np.linspace(0, 0.6, len(y_pos))))
        inset.set_yticks(y_pos)
        inset.set_yticklabels(auc_by_cancer["cancer_type"], fontsize=7)
        inset.set_xlim(max(0.5, auc_by_cancer["auc"].min() - 0.08), min(1.0, auc_by_cancer["auc"].max() + 0.08))
        inset.set_xlabel("AUC", fontsize=7)
        inset.set_title("AUC by Cancer Type\n(Text model)", fontsize=8, fontweight="bold")
        inset.tick_params(axis="x", labelsize=7)
        for yi, row in enumerate(auc_by_cancer.itertuples(index=False)):
            inset.text(row.auc + 0.005, yi, f"{row.auc:.2f}", va="center", fontsize=7)
save_panel(fig, "fig5a")
plt.close(fig)


# %% fig5b: biomarker association robustness
fig, ax = plt.subplots(figsize=(8.0, 6.0))
if robust.empty:
    _missing(ax, "fig5_robust_hits.csv empty")
else:
    r = robust.dropna(subset=["marker", "spec", "HR_markerxICI"]).copy()
    r["support"] = -np.log10(pd.to_numeric(r["p_markerxICI"], errors="coerce").clip(lower=1e-300))
    r["marker_key"] = r["marker"].astype(str) + " (" + r["cancer_type"].astype(str) + ")"
    marker_rank = (r.groupby("marker_key")
                    .agg(n_specs=("spec", "nunique"),
                         support=("support", "max"),
                         mean_HR=("mean_HR", "median"))
                    .sort_values(["n_specs", "support"], ascending=[False, False])
                    .head(15))
    markers = marker_rank.index.tolist()
    specs = r["spec"].dropna().drop_duplicates().tolist()[:6]
    sub = r[r["marker_key"].isin(markers) & r["spec"].isin(specs)]

    for x in range(len(specs)):
        ax.axvline(x, color=LIGHT_GRAY, lw=0.8, zorder=0)
    for y in range(len(markers)):
        ax.axhline(y, color=LIGHT_GRAY, lw=0.8, zorder=0)
    for y in range(len(markers)):
        for x in range(len(specs)):
            ax.scatter(x, y, s=22, facecolors="white", edgecolors=NS_GRAY, lw=0.8, zorder=1)

    for _, row in sub.iterrows():
        x = specs.index(row["spec"])
        y = markers.index(row["marker_key"])
        hr = float(row["HR_markerxICI"])
        support = float(row["support"]) if np.isfinite(row["support"]) else 0.0
        color = BENEFIT_COLOR if hr < 1 else HARM_COLOR
        ax.scatter(x, y, s=28 + min(support, 4) * 26, color=color,
                   edgecolors="white", lw=0.5, zorder=3)

    for y, marker in enumerate(markers):
        hr = marker_rank.loc[marker, "mean_HR"]
        if pd.notna(hr):
            ax.text(len(specs) + 0.3, y, f"HR={hr:.2f}", va="center", fontsize=7, color="#666")

    ax.set_xticks(range(len(specs)))
    ax.set_xticklabels([_pretty_spec(s) for s in specs], fontsize=8)
    ax.set_yticks(range(len(markers)))
    ax.set_yticklabels(markers, fontsize=8, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(-0.55, len(specs) + 1.2)
    ax.set_title("Biomarker Association Robustness", fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BENEFIT_COLOR,
               markeredgecolor="white", markersize=8, label="Sig., ICI benefit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=HARM_COLOR,
               markeredgecolor="white", markersize=8, label="Sig., ICI harm"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=NS_GRAY, markersize=8, label="Not significant"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="lower left", frameon=True)
save_panel(fig, "fig5b")
plt.close(fig)


# %% fig5c: three representative KM interaction panels
fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), sharey=True)


def _plot_km(ax: plt.Axes, data: pd.DataFrame, title: str) -> None:
    if data.empty:
        _missing(ax, "no KM data")
        return
    kmf = KaplanMeierFitter()
    strata = [
        ("Marker+ / ICI+", (data["marker_value"] == 1) & (data["PX_on_ICI"] == 1), "#1B4F72"),
        ("Marker- / ICI+", (data["marker_value"] == 0) & (data["PX_on_ICI"] == 1), "#5DADE2"),
        ("Marker- / ICI-", (data["marker_value"] == 0) & (data["PX_on_ICI"] == 0), "#F28E2B"),
        ("Marker+ / ICI-", (data["marker_value"] == 1) & (data["PX_on_ICI"] == 0), "#C0392B"),
    ]
    plotted = 0
    for label, mask, color in strata:
        sub = data[mask].dropna(subset=["tt_death", "death"])
        sub = sub[sub["tt_death"] > 0]
        if len(sub) < 5:
            continue
        kmf.fit(sub["tt_death"] / 30.44, sub["death"], label=f"{label} (n={len(sub):,})")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color, lw=2.0)
        plotted += 1
    if plotted == 0:
        _missing(ax, "insufficient KM strata")
        return
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Months")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")


if not km_examples.empty:
    example_ids = list(km_examples["example_id"].dropna().unique())[:3]
    for ax, ex_id in zip(axes, example_ids):
        data = km_examples[km_examples["example_id"] == ex_id]
        title = str(data["title"].iloc[0]) + "\n" + str(data["cancer"].iloc[0])
        _plot_km(ax, data, title)
    for ax in axes[len(example_ids):]:
        _missing(ax, "no additional example")
elif not km_top.empty:
    title = "Top hit"
    if not meta.empty:
        m = meta.iloc[0]
        title = _km_title(str(m["marker"]), str(m["cancer"]))
    _plot_km(axes[0], km_top, title)
    for ax in axes[1:]:
        _missing(ax, "no additional example")
else:
    for ax in axes:
        _missing(ax, "fig5_km_examples.csv empty")
axes[0].set_ylabel("Survival Probability")
fig.tight_layout()
save_panel(fig, "fig5c")
plt.close(fig)
