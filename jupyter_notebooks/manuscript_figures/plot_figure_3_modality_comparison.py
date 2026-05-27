"""Render Figure 3 panels using the old feature-comparison target."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _figure_utils import apply_style, load_figure_data, save_panel, MODALITY_ORDER, MODALITY_COLORS


apply_style()

DISPLAY = {m: m.title() if m != "prs" else "PRS" for m in MODALITY_ORDER}
FDR_ALPHA = 0.05


def _missing(ax: plt.Axes, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#777")
    ax.set_axis_off()


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / ranks)[::-1])[::-1]
    return np.clip(q, 0, 1)


cindex_df = load_figure_data("fig3_modality_cindex.csv")
betas = load_figure_data("fig3_joint_betas.csv")

# Compute joint-model significance once per (scheme, event, modality) row.
# We BH-correct within each (scheme, event) cell because that is the family of
# tests jointly produced by one Cox fit; the across-modality comparison in
# panels A/B then counts the per-fit survivors.
if not betas.empty and "p_value" in betas.columns:
    betas = betas.copy()
    betas["q_value"] = np.nan
    for _, grp in betas.groupby(["scheme", "event"], dropna=False):
        betas.loc[grp.index, "q_value"] = _bh_fdr(
            grp["p_value"].fillna(1.0).to_numpy()
        )
    betas["sig"] = (betas["q_value"] < FDR_ALPHA) & betas["beta"].notna()
elif not betas.empty:
    betas = betas.copy()
    betas["sig"] = False


# %% fig3a: significant endpoints per modality (from joint Cox, BH-FDR < 0.05)
fig, ax = plt.subplots(figsize=(6.0, 4.8))
if betas.empty or "sig" not in betas.columns:
    _missing(ax, "fig3_joint_betas.csv missing p-values")
else:
    counts = (betas.groupby("modality")["sig"].sum()
              .reindex(MODALITY_ORDER).fillna(0).astype(int))
    x = np.arange(len(counts))
    colors = [MODALITY_COLORS[m] for m in counts.index]
    ax.bar(x, counts.values, color=colors, edgecolor="white", width=0.6)
    top = max(int(counts.max()), 1)
    for xi, value in zip(x, counts.values):
        ax.text(xi, value + top * 0.03, f"{value}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in counts.index], rotation=20, ha="right")
    ax.set_ylabel(f"# endpoints (joint Cox BH-FDR < {FDR_ALPHA:.2f})")
    ax.set_title("Significant Endpoints per Modality\n(joint Cox model)")
    ax.grid(axis="y", alpha=0.35)
    ax.grid(axis="x", visible=False)
save_panel(fig, "fig3a")
plt.close(fig)


# %% fig3b: pairwise endpoint overlap (significant in joint Cox)
fig, ax = plt.subplots(figsize=(6.0, 5.0))
if betas.empty or "sig" not in betas.columns:
    _missing(ax, "fig3_joint_betas.csv missing p-values")
else:
    sig_betas = betas[betas["sig"]]
    sig_sets = {
        m: set(
            sig_betas.loc[sig_betas["modality"] == m, ["scheme", "event"]]
            .apply(tuple, axis=1)
        )
        for m in MODALITY_ORDER
    }
    matrix = np.zeros((len(MODALITY_ORDER), len(MODALITY_ORDER)))
    for i, m1 in enumerate(MODALITY_ORDER):
        for j, m2 in enumerate(MODALITY_ORDER):
            matrix[i, j] = len(sig_sets[m1] & sig_sets[m2]) if i != j else len(sig_sets[m1])
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    im = ax.imshow(np.where(mask, np.nan, matrix), cmap="Blues", aspect="auto")
    max_val = np.nanmax(matrix) if matrix.size else 0
    for i in range(len(MODALITY_ORDER)):
        for j in range(len(MODALITY_ORDER)):
            if mask[i, j]:
                continue
            v = int(matrix[i, j])
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > max_val * 0.55 else "#222",
                    fontsize=8, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(MODALITY_ORDER)))
    ax.set_xticklabels([DISPLAY[m] for m in MODALITY_ORDER], rotation=35, ha="right")
    ax.set_yticks(range(len(MODALITY_ORDER)))
    ax.set_yticklabels([DISPLAY[m] for m in MODALITY_ORDER])
    ax.set_title(f"Pairwise Endpoint Overlap\n(joint Cox BH-FDR < {FDR_ALPHA:.2f})")
    fig.colorbar(im, ax=ax, label="Overlap count", fraction=0.04, pad=0.02)
save_panel(fig, "fig3b")
plt.close(fig)


# %% fig3c: text vs best competitor
fig, ax = plt.subplots(figsize=(6.0, 5.0))
if cindex_df.empty or "text" not in set(cindex_df["modality"]):
    _missing(ax, "text modality unavailable")
else:
    index_cols = ["scheme", "event"] if "scheme" in cindex_df.columns else ["event"]
    mat = cindex_df.pivot_table(index=index_cols, columns="modality", values="cindex", aggfunc="mean")
    competitors = [m for m in MODALITY_ORDER if m != "text" and m in mat.columns]
    plot_df = mat.dropna(subset=["text"]).copy()
    plot_df = plot_df.dropna(subset=competitors, how="all")
    plot_df["best_comp_cindex"] = plot_df[competitors].max(axis=1)
    plot_df["best_comp_modality"] = plot_df[competitors].idxmax(axis=1)
    plot_df = plot_df.dropna(subset=["best_comp_cindex"])
    for mod in competitors:
        sub = plot_df[plot_df["best_comp_modality"] == mod]
        if sub.empty:
            continue
        ax.scatter(sub["best_comp_cindex"], sub["text"], s=26, alpha=0.65,
                   color=MODALITY_COLORS[mod], label=DISPLAY[mod], edgecolors="none")
    lo = max(0.45, plot_df[["best_comp_cindex", "text"]].min().min() - 0.02)
    hi = min(1.0, plot_df[["best_comp_cindex", "text"]].max().max() + 0.02)
    ax.plot([lo, hi], [lo, hi], ls="--", color="#666", lw=1)
    n_text = int((plot_df["text"] > plot_df["best_comp_cindex"]).sum())
    n_comp = int((plot_df["text"] <= plot_df["best_comp_cindex"]).sum())
    ax.text(0.04, 0.93, f"Text better: N={n_text}", transform=ax.transAxes,
            fontsize=8, color="#34495E", style="italic")
    ax.text(0.54, 0.12, f"Competitor better: N={n_comp}", transform=ax.transAxes,
            fontsize=8, color="#E74C3C", style="italic")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Best Competitor C-index")
    ax.set_ylabel("Text Model C-index")
    ax.set_title("Text vs. Best-Competing Modality")
    ax.legend(title="Best competitor", fontsize=7, title_fontsize=7.5, loc="lower right")
save_panel(fig, "fig3c")
plt.close(fig)


# %% fig3d: joint Cox log HR by modality
fig, ax = plt.subplots(figsize=(6.0, 5.0))
if betas.empty:
    _missing(ax, "fig3_joint_betas.csv empty")
else:
    rng = np.random.default_rng(0)
    positions = []
    labels = []
    for i, mod in enumerate(MODALITY_ORDER):
        vals = betas.loc[betas["modality"] == mod, "beta"].dropna().values
        if len(vals) == 0:
            continue
        positions.append(i)
        labels.append(DISPLAY[mod])
        parts = ax.violinplot([vals], positions=[i], widths=0.6,
                              showmeans=False, showmedians=True, showextrema=True)
        for body in parts["bodies"]:
            body.set_facecolor(MODALITY_COLORS[mod])
            body.set_alpha(0.45)
            body.set_edgecolor("none")
        parts["cmedians"].set_color("#111")
        x = i + rng.uniform(-0.16, 0.16, len(vals))
        ax.scatter(x, vals, s=9, color=MODALITY_COLORS[mod], alpha=0.30, edgecolors="none")
    ax.axhline(0, color="#333", ls="--", lw=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("log(HR) from Joint Model")
    ax.set_title("Joint Cox Model: Log HR by Modality")
    ax.grid(axis="y", alpha=0.35)
    ax.grid(axis="x", visible=False)
save_panel(fig, "fig3d")
plt.close(fig)
