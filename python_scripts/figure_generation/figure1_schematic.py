"""Figure 1 — Model schematic, endpoint overview, patient timeline, cohort composition.

Panels:
  A: Creative pipeline schematic (pure matplotlib drawing, no data)
  B: Bar chart of endpoint counts across 4 outcome schemes
  C: Patient timeline schematic (pre/post treatment note collection)
  D: Cohort composition by cancer type

Data sources:
  - Panel B: compiled_all_schemes_metrics.csv (or counted from embedding prediction DFs)
  - Panel D: cancer_type_df.csv merged with the death_met cohort
"""

from __future__ import annotations

import os
import sys

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Paths — edit DATA_PATH if running on a different cluster mount point
# ---------------------------------------------------------------------------
DATA_PATH = "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/"
SURV_PATH = os.path.join(DATA_PATH, "time-to-event_analysis/")
FEATURE_PATH = os.path.join(DATA_PATH, "clinical_and_genomic_features/")
RESULTS_PATH = os.path.join(SURV_PATH, "results/")
COMPILED_DIR = os.path.join(RESULTS_PATH, "compiled_all_schemes/")

SCHEME_CONFIG = {
    "icd3_post":    {"results_dir": "level_3_ICD_post_results",  "label": "ICD-10 Level 3"},
    "icd4_post":    {"results_dir": "level_4_ICD_post_results",  "label": "ICD-10 Level 4"},
    "phecode_post": {"results_dir": "phecode_post_results",       "label": "Phecodes"},
    "death_met":    {"results_dir": "death_met_results",          "label": "Death + Metastasis"},
}
SCHEME_COLORS = {
    "death_met":    "#1b6f8a",
    "icd3_post":    "#3d8c40",
    "icd4_post":    "#d17c29",
    "phecode_post": "#8f3b76",
}

OUTPUT_DIR = os.path.join(DATA_PATH, "figures/manuscript/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def set_style():
    import seaborn as sns
    sns.set_theme(context="paper", style="whitegrid")
    mpl.rcParams.update({
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })


# ---------------------------------------------------------------------------
# Panel A — pipeline schematic
# ---------------------------------------------------------------------------

def draw_panel_a(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # --- Note type boxes (left) ---
    note_boxes = [
        ("Clinician\nNotes",   "#dbeafe", 4.5),
        ("Imaging\nReports",   "#dcfce7", 3.0),
        ("Pathology\nReports", "#fef3c7", 1.5),
    ]
    for label, color, y in note_boxes:
        box = FancyBboxPatch((0.1, y - 0.55), 1.55, 1.0,
                             boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor="#888", linewidth=0.8)
        ax.add_patch(box)
        ax.text(0.88, y - 0.05, label, ha="center", va="center", fontsize=7.5, fontweight="bold")
        # Simulated text lines inside box
        for i in range(3):
            ax.plot([0.2, 1.5], [y - 0.3 + i * 0.18, y - 0.3 + i * 0.18],
                    color="#aaa", lw=0.8, alpha=0.6)

    # Arrow from note boxes to transformer
    ax.annotate("", xy=(2.1, 3.0), xytext=(1.7, 3.0),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))

    # --- Clinical-Longformer block ---
    tf_x, tf_y, tf_w, tf_h = 2.1, 1.4, 1.9, 3.2
    # Draw grid of colored squares representing attention
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("Blues")
    sq_size = 0.27
    for row in range(10):
        for col in range(6):
            val = rng.random()
            c = cmap(0.3 + 0.7 * val)
            rect = mpatches.Rectangle(
                (tf_x + 0.06 + col * (sq_size + 0.02),
                 tf_y + 0.06 + row * (sq_size + 0.02)),
                sq_size, sq_size,
                facecolor=c, edgecolor="white", linewidth=0.4
            )
            ax.add_patch(rect)
    border = FancyBboxPatch((tf_x, tf_y), tf_w, tf_h,
                            boxstyle="round,pad=0.06",
                            facecolor="none", edgecolor="#1b6f8a", linewidth=1.5)
    ax.add_patch(border)
    ax.text(tf_x + tf_w / 2, tf_y + tf_h + 0.18, "Clinical-Longformer",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1b6f8a")
    ax.text(tf_x + tf_w / 2, tf_y - 0.22,
            "Time-decay pooling | Pre-treatment only",
            ha="center", va="top", fontsize=6.5, color="#555", style="italic")

    # Arrow to embedding scatter
    ax.annotate("", xy=(4.4, 3.0), xytext=(4.05, 3.0),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))

    # --- Embedding scatter (center-right) ---
    emb_cx, emb_cy = 5.3, 3.0
    rng2 = np.random.default_rng(1)
    note_colors_emb = ["#3b82f6", "#22c55e", "#f59e0b"]
    note_labels_emb = ["Clinician", "Imaging", "Pathology"]
    for i, (nc, nl) in enumerate(zip(note_colors_emb, note_labels_emb)):
        xs = rng2.normal(emb_cx + (i - 1) * 0.25, 0.28, 60)
        ys = rng2.normal(emb_cy + (i - 1) * 0.18, 0.28, 60)
        ax.scatter(xs, ys, c=nc, s=7, alpha=0.55, label=nl)

    # Ellipse border
    from matplotlib.patches import Ellipse
    ell = Ellipse((emb_cx, emb_cy), width=1.85, height=1.45,
                  facecolor="none", edgecolor="#777", linewidth=1.0, linestyle="--")
    ax.add_patch(ell)
    ax.text(emb_cx, emb_cy + 0.85, "768-dim\nEmbeddings", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#444")

    legend_handles = [mpatches.Patch(color=c, label=l)
                      for c, l in zip(note_colors_emb, note_labels_emb)]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(emb_cx / 10, -0.01), ncol=3,
              frameon=False, fontsize=6.5)

    # Arrows to downstream boxes
    ax.annotate("", xy=(6.6, 4.3), xytext=(6.1, 3.3),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))
    ax.annotate("", xy=(6.6, 1.8), xytext=(6.1, 2.7),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))

    # --- Downstream box 1: Survival Prediction ---
    bp1 = FancyBboxPatch((6.6, 3.7), 3.2, 1.3,
                         boxstyle="round,pad=0.1",
                         facecolor="#eff6ff", edgecolor="#1b6f8a", linewidth=1.3)
    ax.add_patch(bp1)
    ax.text(8.2, 4.85, "Survival Prediction", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#1b6f8a")
    # Mini KM curve
    t = np.linspace(0, 1, 30)
    km_y = np.exp(-1.5 * t)
    ax.plot(6.85 + t * 1.6, 3.85 + km_y * 0.7, color="#1b6f8a", lw=1.5)
    ax.text(8.65, 3.85, "Penalized Cox PH\n+ elastic net",
            ha="center", va="center", fontsize=6.5, color="#555")

    # --- Downstream box 2: Biomarker Discovery ---
    bp2 = FancyBboxPatch((6.6, 1.1), 3.2, 1.3,
                         boxstyle="round,pad=0.1",
                         facecolor="#fdf4ff", edgecolor="#8f3b76", linewidth=1.3)
    ax.add_patch(bp2)
    ax.text(8.2, 2.25, "ICI Biomarker Discovery", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#8f3b76")
    # Mini volcano
    vx = rng2.normal(0, 0.35, 80)
    vy = rng2.uniform(0, 2.0, 80)
    ax.scatter(7.7 + vx * 0.8, 1.3 + vy * 0.45,
               s=4, c=["#8f3b76" if abs(x) > 0.5 and y > 1.0 else "#ccc"
                        for x, y in zip(vx, vy)],
               alpha=0.7, zorder=5)
    ax.text(8.65, 1.3, "IPTW-weighted Cox\ninteraction model",
            ha="center", va="center", fontsize=6.5, color="#555")


# ---------------------------------------------------------------------------
# Panel B — endpoint counts
# ---------------------------------------------------------------------------

def count_endpoints_per_scheme(compiled_csv: str) -> pd.Series:
    """Count unique events per scheme from compiled metrics CSV."""
    df = pd.read_csv(compiled_csv, usecols=["scheme", "event"])
    return df.groupby("scheme")["event"].nunique()


def draw_panel_b(ax: plt.Axes, compiled_csv: str) -> None:
    counts = count_endpoints_per_scheme(compiled_csv)

    scheme_order = ["death_met", "icd3_post", "icd4_post", "phecode_post"]
    labels = [SCHEME_CONFIG[s]["label"] for s in scheme_order]
    values = [counts.get(s, 0) for s in scheme_order]
    colors = [SCHEME_COLORS[s] for s in scheme_order]

    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8.5, fontweight="bold")

    ax.set_xlabel("Number of endpoints analyzed")
    ax.set_title("Endpoints Analyzed per Scheme")
    ax.invert_yaxis()
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False)
    ax.grid(axis="x", alpha=0.5)
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------------------
# Panel C — patient timeline schematic
# ---------------------------------------------------------------------------

def draw_panel_c(ax: plt.Axes) -> None:
    ax.set_xlim(-26, 62)
    ax.set_ylim(-0.5, 3.5)
    ax.axis("off")

    # Pre-treatment bar (light blue)
    ax.barh(0, width=24, left=-24, height=0.35, color="#bfdbfe", edgecolor="#888", linewidth=0.8)
    # Post-treatment bar (light salmon)
    ax.barh(0, width=60, left=0, height=0.35, color="#fecaca", edgecolor="#888", linewidth=0.8)

    # Treatment start line
    ax.axvline(0, ymin=0.05, ymax=0.72, color="#444", lw=2, linestyle="--", zorder=5)
    ax.text(0, 0.85, "Treatment\nStart", ha="center", va="bottom", fontsize=8,
            fontweight="bold", color="#444")

    # Note markers (simulated pre-treatment notes)
    rng = np.random.default_rng(7)
    for note_type, color, marker, y_base in [
        ("Clinician Notes", "#3b82f6", "o", 1.5),
        ("Imaging Reports", "#22c55e", "^", 2.1),
        ("Pathology Reports", "#f59e0b", "s", 2.7),
    ]:
        # Denser near treatment start
        xs = -rng.exponential(5, 18)
        xs = np.clip(xs, -23, -0.5)
        ys = np.full_like(xs, y_base) + rng.normal(0, 0.07, len(xs))
        ax.scatter(xs, ys, c=color, marker=marker, s=28, alpha=0.75, zorder=4,
                   label=note_type)

    # Annotation arrows
    ax.annotate("", xy=(-12, 0.9), xytext=(-12, 0.55),
                arrowprops=dict(arrowstyle="-|>", color="#1b6f8a", lw=1.2))
    ax.text(-12, 0.92, "Note collection window", ha="center", va="bottom",
            fontsize=7.5, color="#1b6f8a", style="italic")

    ax.annotate("", xy=(30, 0.9), xytext=(30, 0.55),
                arrowprops=dict(arrowstyle="-|>", color="#b91c1c", lw=1.2))
    ax.text(30, 0.92, "Outcome window", ha="center", va="bottom",
            fontsize=7.5, color="#b91c1c", style="italic")

    # X axis label
    ax.text(18, -0.45, "Time (months relative to treatment start)", ha="center",
            fontsize=8.5, color="#333")
    # Tick marks
    for t in [-24, -12, 0, 12, 24, 36, 48, 60]:
        ax.plot([t, t], [-0.08, 0.0], color="#555", lw=0.8)
        ax.text(t, -0.22, str(t), ha="center", fontsize=7.5, color="#555")

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0),
              frameon=False, fontsize=7.5, ncol=1)


# ---------------------------------------------------------------------------
# Panel D — cohort composition
# ---------------------------------------------------------------------------

def load_cohort_cancer_types(cancer_type_csv: str, cohort_csv: str) -> pd.Series:
    """Return patient counts by dominant cancer type for patients in the cohort."""
    cohort = pd.read_csv(cohort_csv, usecols=["DFCI_MRN"])
    ct = pd.read_csv(cancer_type_csv)
    ct = ct[ct["DFCI_MRN"].isin(cohort["DFCI_MRN"])].copy()

    type_cols = [c for c in ct.columns if c.startswith("CANCER_TYPE_")]
    if not type_cols:
        raise ValueError(f"No CANCER_TYPE_* columns found in {cancer_type_csv}")

    ct["dominant_type"] = ct[type_cols].idxmax(axis=1).str.replace("CANCER_TYPE_", "")
    return ct["dominant_type"].value_counts()


def draw_panel_d(ax: plt.Axes, cancer_counts: pd.Series, top_n: int = 10) -> None:
    top = cancer_counts.head(top_n).sort_values()
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(top)))

    bars = ax.barh(top.index, top.values, color=colors, edgecolor="white", height=0.7)
    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8)

    ax.set_xlabel("Number of patients")
    ax.set_title("Cohort Composition by Cancer Type")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False)
    ax.grid(axis="x", alpha=0.5)
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_style()

    compiled_csv = os.path.join(COMPILED_DIR, "all_schemes_compiled_metrics.csv")
    cancer_type_csv = os.path.join(FEATURE_PATH, "cancer_type_df.csv")
    cohort_csv = os.path.join(SURV_PATH, "death_met_embedding_prediction_df.csv")

    # Check required files exist
    missing = [f for f in [compiled_csv, cancer_type_csv, cohort_csv] if not os.path.isfile(f)]
    if missing:
        print("ERROR: Missing required data files:")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)

    cancer_counts = load_cohort_cancer_types(cancer_type_csv, cohort_csv)

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, :])   # spans full top row
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])   # will be replaced with timeline — no, let's do 2x2 bottom

    # Actually use 2x2 bottom row
    fig = plt.figure(figsize=(16, 11))
    ax_a = fig.add_axes([0.03, 0.50, 0.94, 0.45])
    ax_b = fig.add_axes([0.06, 0.06, 0.25, 0.36])
    ax_c = fig.add_axes([0.37, 0.06, 0.30, 0.36])
    ax_d = fig.add_axes([0.70, 0.06, 0.27, 0.36])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b, compiled_csv)
    draw_panel_c(ax_c)
    draw_panel_d(ax_d, cancer_counts)

    panel_labels = {"A": ax_a, "B": ax_b, "C": ax_c, "D": ax_d}
    for label, ax in panel_labels.items():
        ax.text(-0.03, 1.03, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="right")

    caption = (
        "Figure 1. Overview of the clinical text embedding framework. "
        "(A) Model pipeline: pre-treatment clinical notes are embedded via Clinical-Longformer, "
        "pooled into patient-level representations, and used for survival prediction and ICI biomarker discovery. "
        "(B) Number of endpoints analyzed across four outcome schemes. "
        "(C) Patient timeline schematic showing pre-treatment note collection and post-treatment outcome windows. "
        "(D) Study cohort composition by cancer type."
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7.5,
             style="italic", wrap=True,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#ddd", alpha=0.8))

    out_stem = os.path.join(OUTPUT_DIR, "figure1_schematic")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", facecolor="white", bbox_inches="tight")
    print(f"Saved figure 1 → {out_stem}.png/.pdf")


if __name__ == "__main__":
    main()
