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
IQR_WHISKER = 1.5  # Tukey fence multiplier for trimming extreme log-HRs in fig3d (display-only)


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


# %% fig3a: significant endpoints per modality (complete-case endpoints, joint Cox BH-FDR < 0.05)
fig, ax = plt.subplots(figsize=(6.0, 4.8))
if betas.empty or "sig" not in betas.columns:
    _missing(ax, "fig3_joint_betas.csv missing p-values")
else:
    # Restrict to complete-case endpoints (every modality fit) so denominators are equal.
    present_sets = (betas.dropna(subset=["beta"])
                    .groupby(["scheme", "event"])["modality"].agg(set))
    complete_eps = {ep for ep, mods in present_sets.items()
                    if set(MODALITY_ORDER).issubset(mods)}
    cc = betas[betas.set_index(["scheme", "event"]).index.isin(complete_eps)]
    counts = (cc.groupby("modality")["sig"].sum()
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
    ax.set_title(f"Significant Endpoints per Modality\n(joint Cox, {len(complete_eps)} complete-case endpoints)")
    ax.grid(axis="y", alpha=0.35)
    ax.grid(axis="x", visible=False)
save_panel(fig, "fig3a")
plt.close(fig)


# %% fig3b: modality risk-score correlation (death_met death endpoint)
corr_df = load_figure_data("fig3_risk_score_corr.csv")
fig, ax = plt.subplots(figsize=(6.0, 5.0))
if corr_df.empty or "modality" not in corr_df.columns:
    _missing(ax, "fig3_risk_score_corr.csv empty")
else:
    cm = corr_df.set_index("modality")
    mods = [m for m in MODALITY_ORDER if m in cm.index and m in cm.columns]
    mat = cm.loc[mods, mods].astype(float)
    im = ax.imshow(mat.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(mods)):
        for j in range(len(mods)):
            v = mat.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 else "#222", fontsize=8)
    ax.set_xticks(range(len(mods)))
    ax.set_xticklabels([DISPLAY[m] for m in mods], rotation=35, ha="right")
    ax.set_yticks(range(len(mods)))
    ax.set_yticklabels([DISPLAY[m] for m in mods])
    title = "Modality Risk-Score Correlation (death)"
    if "n_patients" in corr_df.columns and corr_df["n_patients"].notna().any():
        title += f"\n(n={int(corr_df['n_patients'].dropna().iloc[0]):,})"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Pearson r", fraction=0.046, pad=0.04)
save_panel(fig, "fig3b")
plt.close(fig)


# %% fig3c: average modality rank across endpoints (1 = best)
avg_rank = load_figure_data("fig3_modality_avg_rank.csv")
fig, ax = plt.subplots(figsize=(6.0, 5.0))
if avg_rank.empty:
    _missing(ax, "fig3_modality_avg_rank.csv empty")
else:
    rank_df = avg_rank.dropna(subset=["mean_rank"]).sort_values("mean_rank", ascending=False)
    y = np.arange(len(rank_df))
    ax.barh(y, rank_df["mean_rank"], xerr=rank_df["sem_rank"],
            color=[MODALITY_COLORS.get(m, "#777") for m in rank_df["modality"]],
            edgecolor="white", error_kw=dict(ecolor="#333", lw=1, capsize=3))
    for yi, row in enumerate(rank_df.itertuples(index=False)):
        ax.text(row.mean_rank + (row.sem_rank or 0) + 0.05, yi, f"{row.mean_rank:.2f}",
                va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY.get(m, m) for m in rank_df["modality"]])
    n_events = int(rank_df["n_events"].iloc[0]) if "n_events" in rank_df.columns else 0
    ax.set_xlim(0.5, len(rank_df) + 0.5)
    ax.set_xlabel("Average rank across endpoints (1 = best)")
    ax.set_title(f"Average Modality Rank\n(complete-case endpoints, n={n_events})")
    ax.grid(axis="x", alpha=0.35)
    ax.grid(axis="y", visible=False)
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
    n_trimmed = 0
    for i, mod in enumerate(MODALITY_ORDER):
        sub = betas.loc[betas["modality"] == mod]
        # Standardized coefficient (Wald z = beta/SE): scale-free and stable, unlike raw log-HR
        vals = (sub["beta"] / sub["se"]).replace([np.inf, -np.inf], np.nan).dropna().values
        if len(vals) == 0:
            continue
        # Tukey-fence trim of extreme z-scores (unstable joint fits) for this modality
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - IQR_WHISKER * iqr, q3 + IQR_WHISKER * iqr
        kept = vals[(vals >= lo) & (vals <= hi)]
        n_trimmed += len(vals) - len(kept)
        if len(kept) == 0:
            continue
        vals = kept
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
    for zc in (-1.96, 1.96):
        ax.axhline(zc, color="#999", ls=":", lw=1)
    ax.text(0.015, 0.015,
            "z from L2-penalized fit; ±1.96 lines are descriptive, not an exact test",
            transform=ax.transAxes, fontsize=6.5, style="italic", color="#777")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Joint Cox standardized coefficient (z = β/SE)")
    title = "Joint Cox Model: Standardized Coefficient by Modality"
    if n_trimmed:
        title += f"\n({n_trimmed} extreme outliers trimmed, Tukey {IQR_WHISKER}×IQR)"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.35)
    ax.grid(axis="x", visible=False)
save_panel(fig, "fig3d")
plt.close(fig)
