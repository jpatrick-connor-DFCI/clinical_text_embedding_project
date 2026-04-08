"""Figure 1 — Model schematic, endpoint overview, delta C-index exemplars, and mortality KM.

Panels:
  A: Creative pipeline schematic (pure matplotlib drawing, no data)
  B: Bar chart of endpoint counts across 4 outcome schemes
  C: Top positive and negative event-level delta C-index values
  D: All-cause mortality Kaplan-Meier curves by text risk quartile

Data sources:
  - Panels B, C: compiled_all_schemes_metrics.csv
  - Panel D: death_met held-out risk scores + death_met_embedding_prediction_df.csv
"""

from __future__ import annotations

import os
import warnings

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
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

# Generate one separate figure per evidence-backed prevalence filter.
# compile_all_scheme_results.ipynb defines thresholds [0.01, 0.025, 0.05]
# and writes labels via _threshold_to_label(...) -> 1pct, 2_5pct, 5pct.
PREVALENCE_FILTERS_TO_PLOT = [None, "1pct", "2_5pct", "5pct"]


def prevalence_filter_to_suffix(prevalence_filter: str | None) -> str:
    return "unfiltered" if prevalence_filter in (None, "") else prevalence_filter


def prevalence_filter_to_label(prevalence_filter: str | None) -> str:
    return "unfiltered event set" if prevalence_filter in (None, "") else f"{prevalence_filter} event set"


def get_compiled_metrics_path(prevalence_filter: str | None) -> str:
    if prevalence_filter in (None, ""):
        path = os.path.join(COMPILED_DIR, "all_schemes_compiled_metrics.csv")
    else:
        path = os.path.join(COMPILED_DIR, f"all_schemes_compiled_metrics_{prevalence_filter}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Compiled metrics not found for filter {prevalence_filter!r}: {path}")
    return path


def load_compiled_metrics(compiled_csv: str) -> pd.DataFrame:
    df = pd.read_csv(compiled_csv)
    if "delta_c_index" not in df.columns:
        required = {"text_full_cohort_mean_c_index", "base_mean_c_index"}
        if not required.issubset(df.columns):
            missing = sorted(required - set(df.columns))
            raise ValueError(f"Compiled metrics missing required columns: {missing}")
        df["delta_c_index"] = (
            df["text_full_cohort_mean_c_index"] - df["base_mean_c_index"]
        )
    return df

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
# Panel C — top positive/negative delta C-index events
# ---------------------------------------------------------------------------

def _short_scheme_label(scheme: str) -> str:
    return {
        "death_met": "Death+Met",
        "icd3_post": "ICD-10 L3",
        "icd4_post": "ICD-10 L4",
        "phecode_post": "Phecodes",
    }.get(scheme, scheme)


def _event_label(row: pd.Series, max_len: int = 38) -> str:
    desc = str(row.get("event_description", row.get("event", "")))
    label = f"{desc} [{_short_scheme_label(row['scheme'])}]"
    return label if len(label) <= max_len else label[: max_len - 1] + "…"


def draw_panel_c(ax: plt.Axes, compiled_csv: str, top_n: int = 5) -> None:
    df = load_compiled_metrics(compiled_csv)
    plot_df = df.dropna(subset=["delta_c_index"]).copy()
    if plot_df.empty:
        ax.text(0.5, 0.5, "No delta C-index data available",
                ha="center", va="center", transform=ax.transAxes)
        return

    if "event_description" not in plot_df.columns:
        plot_df["event_description"] = plot_df["event"]

    pos = plot_df[plot_df["delta_c_index"] > 0].nlargest(top_n, "delta_c_index")
    neg = plot_df[plot_df["delta_c_index"] < 0].nsmallest(top_n, "delta_c_index")
    ranked = pd.concat([neg, pos], ignore_index=True)
    if ranked.empty:
        ax.text(0.5, 0.5, "No positive or negative delta C-index events found",
                ha="center", va="center", transform=ax.transAxes)
        return

    ranked["label"] = ranked.apply(_event_label, axis=1)
    ranked = ranked.sort_values("delta_c_index")
    colors = ["#b55852" if val < 0 else "#1665a2" for val in ranked["delta_c_index"]]

    bars = ax.barh(ranked["label"], ranked["delta_c_index"], color=colors, edgecolor="white", height=0.72)
    ax.axvline(0, color="#444", linestyle="--", lw=1.0)

    x_span = max(0.01, ranked["delta_c_index"].abs().max())
    ax.set_xlim(-x_span * 1.25, x_span * 1.25)
    for bar, val in zip(bars, ranked["delta_c_index"]):
        x = val + (0.005 if val >= 0 else -0.005)
        ha = "left" if val >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}",
                va="center", ha=ha, fontsize=7.5)

    ax.set_xlabel("Delta C-index (text - base)")
    ax.set_title("Top Positive and Negative Event-Level Gains")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelsize=7.2)
    ax.grid(axis="x", alpha=0.5)
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------------------
# Panel D — all-cause mortality
# ---------------------------------------------------------------------------

def load_text_risk_scores(event: str = "death", scheme: str = "death_met") -> pd.DataFrame | None:
    results_dir = SCHEME_CONFIG[scheme]["results_dir"]
    fpath = os.path.join(
        RESULTS_PATH, results_dir, "held_out_risk_scores", event, "text_risk_scores.csv"
    )
    if not os.path.isfile(fpath):
        return None
    df = pd.read_csv(fpath)
    if "risk_score" not in df.columns:
        score_cols = [c for c in df.columns if "risk_score" in c.lower()]
        if score_cols:
            df = df.rename(columns={score_cols[0]: "risk_score"})
    return df


def load_death_survival_data() -> pd.DataFrame:
    fpath = os.path.join(SURV_PATH, "death_met_embedding_prediction_df.csv")
    cols = ["DFCI_MRN", "death", "tt_death"]
    df = pd.read_csv(fpath, usecols=cols)
    df = df.rename(columns={"death": "event_indicator", "tt_death": "time"})
    df = df.dropna(subset=["event_indicator", "time"])
    df["event_indicator"] = df["event_indicator"].astype(bool)
    return df


def draw_panel_d(ax: plt.Axes) -> None:
    scores = load_text_risk_scores()
    if scores is None:
        ax.text(0.5, 0.5, "No text risk scores available for death_met/death",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        return

    surv = load_death_survival_data()
    merged = scores.merge(surv, on="DFCI_MRN").dropna(subset=["risk_score", "time", "event_indicator"])
    if merged.empty:
        ax.text(0.5, 0.5, "No analyzable mortality records available",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        return

    try:
        quartile_codes = pd.qcut(merged["risk_score"], q=4, labels=False, duplicates="drop")
    except ValueError:
        quartile_codes = pd.Series(np.nan, index=merged.index)
    merged = merged.assign(quartile_code=quartile_codes).dropna(subset=["quartile_code"]).copy()
    merged["quartile_code"] = merged["quartile_code"].astype(int)

    if merged["quartile_code"].nunique() < 2:
        ax.text(0.5, 0.5, "Risk scores do not support quartile stratification",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        return

    quartile_labels = {0: "Q1 (low)", 1: "Q2", 2: "Q3", 3: "Q4 (high)"}
    quartile_order = sorted(merged["quartile_code"].unique())
    colors = ["#2166ac", "#74add1", "#f4a582", "#d73027"]

    for idx, color in zip(quartile_order, colors):
        label = quartile_labels[idx]
        sub = merged[merged["quartile_code"] == idx]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["time"], sub["event_indicator"], label=f"{label} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color, linewidth=1.8)

    try:
        result = multivariate_logrank_test(
            merged["time"], merged["quartile_code"], merged["event_indicator"]
        )
        p_val = result.p_value
        p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    except Exception:
        p_str = ""

    ax.set_title("All-cause Mortality by Text Risk Quartile")
    ax.set_xlabel("Time to death")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(-0.02, 1.05)
    if p_str:
        ax.text(0.97, 0.97, p_str, transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc"))
    ax.legend(fontsize=6.8, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.5)
    ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_style()
    generated = 0

    for prevalence_filter in PREVALENCE_FILTERS_TO_PLOT:
        try:
            compiled_csv = get_compiled_metrics_path(prevalence_filter)
        except FileNotFoundError as exc:
            warnings.warn(str(exc), stacklevel=2)
            continue

        prevalence_label = prevalence_filter_to_label(prevalence_filter)
        suffix = prevalence_filter_to_suffix(prevalence_filter)

        fig = plt.figure(figsize=(16, 11))
        ax_a = fig.add_axes([0.03, 0.50, 0.94, 0.45])
        ax_b = fig.add_axes([0.06, 0.06, 0.25, 0.36])
        ax_c = fig.add_axes([0.37, 0.06, 0.30, 0.36])
        ax_d = fig.add_axes([0.70, 0.06, 0.27, 0.36])

        draw_panel_a(ax_a)
        draw_panel_b(ax_b, compiled_csv)
        draw_panel_c(ax_c, compiled_csv)
        draw_panel_d(ax_d)

        panel_labels = {"A": ax_a, "B": ax_b, "C": ax_c, "D": ax_d}
        for label, ax in panel_labels.items():
            ax.text(-0.03, 1.03, label, transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="top", ha="right")

        fig.text(0.98, 0.975, f"Panels B-C: {prevalence_label}", ha="right", va="top",
                 fontsize=8, color="#444")
        caption = (
            "Figure 1. Overview of the clinical text embedding framework. "
            "(A) Model pipeline: pre-treatment clinical notes are embedded via Clinical-Longformer, "
            "pooled into patient-level representations, and used for survival prediction and ICI biomarker discovery. "
            f"(B) Number of endpoints analyzed across four outcome schemes for the {prevalence_label}. "
            "(C) Top positive and negative event-level delta C-index values (text minus base). "
            "(D) Kaplan-Meier survival curves for all-cause mortality stratified by text-derived risk quartile."
        )
        fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7.5,
                 style="italic", wrap=True,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#ddd", alpha=0.8))

        out_stem = os.path.join(OUTPUT_DIR, f"figure1_schematic_{suffix}")
        for ext in ("png", "pdf"):
            fig.savefig(f"{out_stem}.{ext}", facecolor="white", bbox_inches="tight")
        plt.close(fig)
        generated += 1
        print(f"Saved figure 1 ({prevalence_label}) → {out_stem}.png/.pdf")

    if generated == 0:
        raise FileNotFoundError("No figure1 outputs were generated because no compiled metrics files were found.")


if __name__ == "__main__":
    main()
