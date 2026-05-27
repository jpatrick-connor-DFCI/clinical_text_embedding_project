"""Render Figure 1 panels using the old manuscript overview target.

The prepared CSV inputs are unchanged. This script only changes the visual
surface so the generated panels match the historical mockup content:
pipeline schematic, endpoint counts, patient timeline, and cohort composition.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle

from _figure_utils import apply_style, load_figure_data, save_panel


apply_style()

SCHEME_LABELS = {
    "death_met": "death_met",
    "icd3_post": "ICD-10\nlevel-3",
    "icd4_post": "ICD-10\nlevel-4",
    "phecode_post": "PhecodeX",
}
SCHEME_COLORS = {
    "death_met": "#E74C3C",
    "icd3_post": "#3498DB",
    "icd4_post": "#2ECC71",
    "phecode_post": "#9B59B6",
}


def _missing(ax: plt.Axes, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#777")
    ax.set_axis_off()


# %% fig1a: pipeline schematic
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis("off")

note_boxes = [
    ("Clinician\nNotes", "#B9DCF2", 4.6),
    ("Imaging\nReports", "#B8E3C6", 3.0),
    ("Pathology\nReports", "#F8D9A3", 1.4),
]
for label, color, y in note_boxes:
    box = FancyBboxPatch((0.15, y - 0.45), 1.8, 0.9, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor="#555", linewidth=1.1)
    ax.add_patch(box)
    for j in range(3):
        ax.plot([0.35, 1.75], [y + 0.22 - j * 0.18, y + 0.22 - j * 0.18],
                color="#777", lw=0.8, alpha=0.65)
    ax.text(1.05, y - 0.05, label, ha="center", va="center", fontsize=9, fontweight="bold")

ax.annotate("", xy=(2.65, 3.0), xytext=(1.95, 3.0),
            arrowprops=dict(arrowstyle="->", lw=1.7, color="#2C3E50"))

rng = np.random.default_rng(1)
cmap = plt.get_cmap("Blues")
for r in range(4):
    for c in range(6):
        ax.add_patch(Rectangle((3.05 + c * 0.36, 2.25 + r * 0.36), 0.29, 0.29,
                               facecolor=cmap(0.35 + 0.6 * rng.random()),
                               edgecolor="white", lw=0.5))
ax.text(4.12, 4.35, "Clinical-Longformer", ha="center", va="bottom",
        fontsize=10, fontweight="bold", color="#1F5E83")
ax.text(4.12, 2.05, "Time-decay weighted pooling | Pre-treatment only",
        ha="center", va="top", fontsize=8, color="#555", style="italic")

ax.annotate("", xy=(5.65, 3.0), xytext=(5.25, 3.0),
            arrowprops=dict(arrowstyle="->", lw=1.7, color="#2C3E50"))

cloud_center = (6.75, 3.0)
for dx, dy, color in [(-0.15, 0.45, "#F28E2B"), (-0.35, -0.25, "#3498DB"), (0.35, -0.65, "#2ECC71")]:
    ax.scatter(rng.normal(cloud_center[0] + dx, 0.28, 55),
               rng.normal(cloud_center[1] + dy, 0.25, 55),
               s=12, color=color, alpha=0.65)
ax.add_patch(Ellipse(cloud_center, width=2.2, height=3.0, angle=15,
                     fill=False, edgecolor="#95A5A6", lw=1.2))
ax.text(cloud_center[0], 4.65, "768-dim Patient\nEmbeddings", ha="center",
        fontsize=9, fontweight="bold", color="#1F5E83")

ax.annotate("", xy=(8.35, 4.05), xytext=(7.65, 3.65),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#2C3E50"))
ax.annotate("", xy=(8.35, 2.05), xytext=(7.65, 2.45),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#2C3E50"))

surv_box = FancyBboxPatch((8.35, 3.35), 2.75, 1.1, boxstyle="round,pad=0.08",
                          facecolor="#DDEFFB", edgecolor="#2E86C1", lw=1.4)
bio_box = FancyBboxPatch((8.35, 1.55), 2.75, 1.1, boxstyle="round,pad=0.08",
                         facecolor="#DDF5E8", edgecolor="#27AE60", lw=1.4)
ax.add_patch(surv_box)
ax.add_patch(bio_box)
ax.text(9.72, 4.15, "Survival Prediction", ha="center", va="center",
        fontsize=9, fontweight="bold", color="#1F5E83")
ax.plot([8.7, 10.75], [3.86, 3.78], color="#2E86C1", lw=1.6)
ax.text(9.72, 2.22, "ICI Biomarker\nDiscovery", ha="center", va="center",
        fontsize=9, fontweight="bold", color="#1F5E83")
ax.scatter(rng.normal(9.72, 0.48, 70), rng.normal(1.92, 0.22, 70),
           s=9, color="#95A5A6", alpha=0.55)
ax.scatter(rng.normal(9.2, 0.16, 6), rng.normal(2.05, 0.18, 6),
           s=14, color="#E74C3C", alpha=0.75)

save_panel(fig, "fig1a")
plt.close(fig)


# %% fig1b: endpoint counts
try:
    endpoint_counts = load_figure_data("fig1_endpoint_counts.csv")
except FileNotFoundError:
    endpoint_counts = pd.DataFrame()

fig, ax = plt.subplots(figsize=(5.4, 4.4))
if endpoint_counts.empty:
    _missing(ax, "fig1_endpoint_counts.csv unavailable")
else:
    counts = endpoint_counts.set_index("scheme")["n_endpoints"].reindex(SCHEME_LABELS).fillna(0).astype(int)
    y = np.arange(len(counts))
    colors = [SCHEME_COLORS[s] for s in counts.index]
    bars = ax.barh(y, counts.values, color=colors, edgecolor="white", height=0.62)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_width() + max(counts.max() * 0.015, 1), bar.get_y() + bar.get_height() / 2,
                f"{value:,}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([SCHEME_LABELS[s] for s in counts.index])
    ax.set_xlabel("Number of endpoints")
    ax.set_title("Outcome Endpoints")
    ax.grid(axis="x", alpha=0.35)
    ax.grid(axis="y", visible=False)
save_panel(fig, "fig1b")
plt.close(fig)


# %% fig1c: patient timeline schematic
vol = load_figure_data("fig1_note_volume.csv")
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.set_xlim(-24, 60)
ax.set_ylim(0, 1)
ax.axvspan(-24, 0, ymin=0.38, ymax=0.50, color="#A9D5F3", alpha=0.9)
ax.axvspan(0, 60, ymin=0.38, ymax=0.50, color="#F1948A", alpha=0.9)
ax.axvline(0, color="#2C3E50", ls="--", lw=2)
ax.text(0, 0.78, "Treatment\nStart", ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(-12, 0.78, "Note collection\nwindow", ha="center", va="center", fontsize=9, color="#2E86C1")
ax.annotate("", xy=(-12, 0.49), xytext=(-15, 0.72),
            arrowprops=dict(arrowstyle="->", color="#2E86C1", lw=1.4))
ax.text(34, 0.78, "Outcome\nwindow", ha="center", va="center", fontsize=9, color="#E74C3C")
ax.annotate("", xy=(30, 0.50), xytext=(34, 0.72),
            arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.4))

rng = np.random.default_rng(4)
note_types = list(vol["note_type"].dropna().unique())[:3] if not vol.empty else []
if not note_types:
    note_types = ["Clinician Notes", "Imaging Reports", "Pathology Reports"]
markers = ["o", "^", "s"]
colors = ["#2E86C1", "#27AE60", "#F28E2B"]
for i, note_type in enumerate(note_types):
    xs = rng.uniform(-22, -1, 12)
    ys = rng.normal(0.58, 0.012, 12)
    ax.scatter(xs, ys, s=36, marker=markers[i % len(markers)], color=colors[i % len(colors)],
               label=str(note_type).replace("_", " ").title(), alpha=0.8)
ax.text(-24, 0.34, "Diagnosis", ha="left", va="top", color="#666", fontsize=8)
ax.text(58, 0.34, "End of\nFollow-up", ha="right", va="top", color="#666", fontsize=8)
ax.set_xlabel("Time (months relative to treatment start)")
ax.set_yticks([])
ax.set_title("Patient Timeline")
ax.legend(loc="lower right", fontsize=7)
save_panel(fig, "fig1c")
plt.close(fig)


# %% fig1d: cohort composition (pie of cancer types)
ct = load_figure_data("fig1_cancer_type_counts.csv")
fig, ax = plt.subplots(figsize=(7.0, 5.2))
if ct.empty:
    _missing(ax, "fig1_cancer_type_counts.csv empty")
else:
    plot_df = ct.head(10).copy()
    plot_df["category"] = plot_df["category"].astype(str).str.replace("_", " ", regex=False)
    plot_df = plot_df.sort_values("n", ascending=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_df)))
    total = int(plot_df["n"].sum())
    wedges, _, autotexts = ax.pie(
        plot_df["n"].values,
        labels=None,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
        startangle=90,
        pctdistance=0.78,
        wedgeprops=dict(edgecolor="white", linewidth=1.2),
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax.legend(
        wedges,
        [f"{c} (n={int(n):,})" for c, n in zip(plot_df["category"], plot_df["n"])],
        loc="center left",
        bbox_to_anchor=(0.95, 0.5),
        fontsize=7.5,
        frameon=False,
    )
    ax.set_title(f"Cohort Composition  (N={total:,})")
save_panel(fig, "fig1d")
plt.close(fig)
