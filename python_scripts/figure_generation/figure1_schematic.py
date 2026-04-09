"""Figure 1 — Study overview figure.

Panels:
  A: Overview of notes to patient-level prediction
  B: Example patient timeline from notes to follow-up
  C: Outcome endpoint families and representative endpoints
  D: Cancer type distribution in the mortality cohort

Data sources:
  - Panel C: compiled_all_schemes_metrics.csv
  - Panel D: cancer_type_df.csv + death_met_embedding_prediction_df.csv
"""

from __future__ import annotations

import os
import textwrap
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_generation_utils import (
    FEATURE_PATH,
    OUTPUT_DIR,
    PREVALENCE_FILTERS_TO_PLOT,
    SURV_PATH,
    get_compiled_metrics_path,
    prevalence_filter_to_label,
    prevalence_filter_to_suffix,
    set_manuscript_style,
)
from matplotlib.patches import FancyBboxPatch


SCHEME_CONFIG = {
    "death_met": {
        "label": "Death + Metastasis",
        "subtitle": "Composite survival endpoints",
    },
    "icd3_post": {
        "label": "ICD-10 Level 3",
        "subtitle": "Broad diagnosis families",
    },
    "icd4_post": {
        "label": "ICD-10 Level 4",
        "subtitle": "Granular diagnosis codes",
    },
    "phecode_post": {
        "label": "Phecodes",
        "subtitle": "Phenotype groupings",
    },
}
SCHEME_COLORS = {
    "death_met": "#1b6f8a",
    "icd3_post": "#3d8c40",
    "icd4_post": "#d17c29",
    "phecode_post": "#8f3b76",
}


def load_compiled_metrics(compiled_csv: str) -> pd.DataFrame:
    df = pd.read_csv(compiled_csv)
    if "event_description" not in df.columns:
        df["event_description"] = df["event"]
    if "delta_c_index" not in df.columns:
        required = {"text_full_cohort_mean_c_index", "base_mean_c_index"}
        if required.issubset(df.columns):
            df["delta_c_index"] = (
                df["text_full_cohort_mean_c_index"] - df["base_mean_c_index"]
            )
    return df


def set_style() -> None:
    set_manuscript_style()


def draw_panel_a(ax: plt.Axes) -> None:
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Notes to Prediction Overview", loc="left")

    note_boxes = [
        ("Clinician\nNotes", "#dbeafe", 4.5),
        ("Imaging\nReports", "#dcfce7", 3.0),
        ("Pathology\nReports", "#fef3c7", 1.5),
    ]
    for label, color, y in note_boxes:
        box = FancyBboxPatch(
            (0.10, y - 0.55),
            1.55,
            1.0,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor="#888",
            linewidth=0.8,
        )
        ax.add_patch(box)
        ax.text(0.88, y - 0.05, label, ha="center", va="center", fontsize=7.5, fontweight="bold")
        for i in range(3):
            ax.plot(
                [0.20, 1.50],
                [y - 0.30 + i * 0.18, y - 0.30 + i * 0.18],
                color="#aaa",
                lw=0.8,
                alpha=0.6,
            )

    ax.annotate("", xy=(2.1, 3.0), xytext=(1.7, 3.0),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))

    tf_x, tf_y, tf_w, tf_h = 2.1, 1.4, 1.9, 3.2
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("Blues")
    sq_size = 0.27
    for row in range(10):
        for col in range(6):
            rect = mpatches.Rectangle(
                (tf_x + 0.06 + col * (sq_size + 0.02), tf_y + 0.06 + row * (sq_size + 0.02)),
                sq_size,
                sq_size,
                facecolor=cmap(0.3 + 0.7 * rng.random()),
                edgecolor="white",
                linewidth=0.4,
            )
            ax.add_patch(rect)
    border = FancyBboxPatch(
        (tf_x, tf_y),
        tf_w,
        tf_h,
        boxstyle="round,pad=0.06",
        facecolor="none",
        edgecolor="#1b6f8a",
        linewidth=1.5,
    )
    ax.add_patch(border)
    ax.text(tf_x + tf_w / 2, tf_y + tf_h + 0.18, "Clinical-Longformer",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1b6f8a")
    ax.text(tf_x + tf_w / 2, tf_y - 0.22, "Time-decay pooling | Pre-treatment only",
            ha="center", va="top", fontsize=6.5, color="#555", style="italic")

    ax.annotate("", xy=(4.4, 3.0), xytext=(4.05, 3.0),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))

    from matplotlib.patches import Ellipse

    emb_cx, emb_cy = 5.2, 3.0
    rng2 = np.random.default_rng(1)
    for idx, color in enumerate(["#3b82f6", "#22c55e", "#f59e0b"]):
        xs = rng2.normal(emb_cx + (idx - 1) * 0.25, 0.28, 60)
        ys = rng2.normal(emb_cy + (idx - 1) * 0.18, 0.28, 60)
        ax.scatter(xs, ys, c=color, s=7, alpha=0.55)
    ell = Ellipse((emb_cx, emb_cy), width=1.85, height=1.45,
                  facecolor="none", edgecolor="#777", linewidth=1.0, linestyle="--")
    ax.add_patch(ell)
    ax.text(emb_cx, emb_cy + 0.85, "768-dim\nNote Embeddings",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="#444")

    ax.annotate("", xy=(6.5, 3.0), xytext=(6.1, 3.0),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))

    pooled_box = FancyBboxPatch(
        (6.6, 1.9),
        2.0,
        2.2,
        boxstyle="round,pad=0.1",
        facecolor="#eef4ff",
        edgecolor="#3b82f6",
        linewidth=1.2,
    )
    ax.add_patch(pooled_box)
    ax.text(7.6, 3.75, "Patient-Level\nEmbedding",
            ha="center", va="center", fontsize=8, fontweight="bold", color="#1d4ed8")
    for i, width in enumerate([1.3, 1.6, 1.1, 1.5, 0.9]):
        ax.plot([6.95, 6.95 + width], [3.15 - i * 0.25, 3.15 - i * 0.25],
                color="#60a5fa", lw=2.0, solid_capstyle="round")
    ax.text(7.6, 2.15, "Notes pooled before\nsurvival modeling",
            ha="center", va="center", fontsize=6.5, color="#555")

    ax.annotate("", xy=(9.0, 3.0), xytext=(8.6, 3.0),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))

    pred_box = FancyBboxPatch(
        (9.0, 1.55),
        2.5,
        2.9,
        boxstyle="round,pad=0.1",
        facecolor="#effaf5",
        edgecolor="#2f855a",
        linewidth=1.2,
    )
    ax.add_patch(pred_box)
    ax.text(10.25, 4.12, "Risk Prediction",
            ha="center", va="center", fontsize=8, fontweight="bold", color="#1f6f4a")
    ax.text(10.25, 3.58, "Penalized Cox PH",
            ha="center", va="center", fontsize=6.8, color="#555")

    t = np.linspace(0, 1, 30)
    for offset, color, rate in zip(
        [0.0, 0.12, 0.24],
        ["#1d4ed8", "#60a5fa", "#f59e0b"],
        [1.8, 1.1, 0.7],
    ):
        ax.plot(9.35 + t * 1.55, 1.95 + offset + np.exp(-rate * t) * 0.85, color=color, lw=1.4)
    ax.text(10.25, 1.88, "Low- to high-risk\nsurvival strata",
            ha="center", va="top", fontsize=6.5, color="#555")


def draw_panel_b(ax: plt.Axes) -> None:
    ax.set_xlim(-24, 36)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Example Patient Timeline")

    ax.axvspan(-24, 0, ymin=0.18, ymax=0.88, color="#eef4ff", alpha=0.8)
    ax.axvspan(0, 36, ymin=0.18, ymax=0.88, color="#f8fafc", alpha=1.0)
    ax.hlines(0.46, -24, 36, color="#6b7280", lw=2.0)

    for month in [-24, -12, -6, 0, 6, 12, 24, 36]:
        ax.vlines(month, 0.42, 0.50, color="#6b7280", lw=1.0)
        tick_label = f"{month:+d}" if month != 0 else "0"
        ax.text(month, 0.30, tick_label, ha="center", va="top", fontsize=7)

    ax.text(-12, 0.88, "Pre-treatment note window",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1d4ed8")
    ax.text(18, 0.88, "Follow-up window",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="#475569")

    note_points = [
        (-20, "Clinician", "#3b82f6"),
        (-15, "Imaging", "#22c55e"),
        (-10, "Pathology", "#f59e0b"),
        (-4, "Clinician", "#3b82f6"),
    ]
    for x, label, color in note_points:
        ax.scatter(x, 0.62, s=48, color=color, edgecolors="white", linewidths=0.7, zorder=3)
        ax.text(x, 0.70, label, ha="center", va="bottom", fontsize=6.2, color=color)

    ax.vlines(0, 0.18, 0.76, color="#1b6f8a", linestyle="--", lw=1.4)
    ax.scatter(0, 0.46, s=54, color="#1b6f8a", zorder=4)
    ax.text(0, 0.79, "Treatment start",
            ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#1b6f8a")

    milestone_points = [
        (8, "Risk score\nupdated", "#2563eb"),
        (18, "Incident\nendpoint", "#7c3aed"),
        (32, "Death or\ncensoring", "#dc2626"),
    ]
    for x, label, color in milestone_points:
        ax.scatter(x, 0.46, s=46, color=color, edgecolors="white", linewidths=0.7, zorder=3)
        ax.text(x, 0.65 if x < 20 else 0.63, label, ha="center", va="bottom", fontsize=6.4, color=color)

    ax.annotate("", xy=(0, 0.16), xytext=(-24, 0.16),
                arrowprops=dict(arrowstyle="<->", color="#3b82f6", lw=1.0))
    ax.text(-12, 0.10, "Notes pooled up to treatment start",
            ha="center", va="top", fontsize=6.4, color="#3b82f6")
    ax.annotate("", xy=(36, 0.16), xytext=(0, 0.16),
                arrowprops=dict(arrowstyle="<->", color="#64748b", lw=1.0))
    ax.text(18, 0.10, "Time-to-event follow-up after treatment",
            ha="center", va="top", fontsize=6.4, color="#475569")


def _clean_endpoint_label(label: str, max_len: int = 32) -> str:
    label = str(label).replace("_", " ").strip()
    if not label or label.lower() == "nan":
        return "Unavailable"
    return textwrap.shorten(label, width=max_len, placeholder="…")


def summarize_endpoint_cards(compiled_csv: str, example_n: int = 3) -> dict[str, dict[str, object]]:
    df = load_compiled_metrics(compiled_csv)
    summary: dict[str, dict[str, object]] = {}

    for scheme in ["death_met", "icd3_post", "icd4_post", "phecode_post"]:
        sub = df[df["scheme"] == scheme].copy()
        if sub.empty:
            summary[scheme] = {"count": 0, "examples": []}
            continue

        sub["event_display"] = sub["event_description"].fillna(sub["event"]).astype(str)
        sort_cols = ["event_display"]
        ascending = [True]
        if "text_full_cohort_mean_c_index" in sub.columns:
            sort_cols = ["text_full_cohort_mean_c_index", "event_display"]
            ascending = [False, True]
        sub = sub.sort_values(sort_cols, ascending=ascending)

        examples: list[str] = []
        for desc in sub["event_display"]:
            cleaned = _clean_endpoint_label(desc)
            if cleaned not in examples:
                examples.append(cleaned)
            if len(examples) == example_n:
                break

        summary[scheme] = {
            "count": int(sub["event"].nunique()),
            "examples": examples,
        }

    return summary


def draw_panel_c(ax: plt.Axes, compiled_csv: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Outcome Endpoints by Scheme")

    cards = summarize_endpoint_cards(compiled_csv)
    card_positions = {
        "death_met": (0.03, 0.54),
        "icd3_post": (0.52, 0.54),
        "icd4_post": (0.03, 0.08),
        "phecode_post": (0.52, 0.08),
    }
    card_w = 0.44
    card_h = 0.34

    for scheme, (x0, y0) in card_positions.items():
        color = SCHEME_COLORS[scheme]
        card = FancyBboxPatch(
            (x0, y0),
            card_w,
            card_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#fbfdff",
            edgecolor=color,
            linewidth=1.2,
        )
        ax.add_patch(card)

        info = cards[scheme]
        ax.text(x0 + 0.03, y0 + card_h - 0.06, SCHEME_CONFIG[scheme]["label"],
                ha="left", va="top", fontsize=8, fontweight="bold", color=color)
        ax.text(x0 + 0.03, y0 + card_h - 0.12, SCHEME_CONFIG[scheme]["subtitle"],
                ha="left", va="top", fontsize=6.4, color="#555")
        ax.text(x0 + 0.03, y0 + card_h - 0.18, f"Endpoints: {info['count']}",
                ha="left", va="top", fontsize=7, fontweight="bold", color="#111")
        ax.text(x0 + 0.03, y0 + card_h - 0.24, "Examples",
                ha="left", va="top", fontsize=6.7, color="#555")

        examples = info["examples"] or ["No endpoints available"]
        for idx, example in enumerate(examples[:3]):
            ax.text(
                x0 + 0.04,
                y0 + card_h - 0.31 - idx * 0.07,
                textwrap.fill(f"{idx + 1}. {example}", width=22),
                ha="left",
                va="top",
                fontsize=6.5,
                color="#222",
            )


def load_cancer_type_distribution(top_n: int = 8) -> pd.Series:
    cohort_mrns = pd.read_csv(
        os.path.join(SURV_PATH, "death_met_embedding_prediction_df.csv"),
        usecols=["DFCI_MRN"],
    )["DFCI_MRN"]

    cancer_df = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_type_df.csv"))
    cancer_df = cancer_df[cancer_df["DFCI_MRN"].isin(cohort_mrns)].copy()
    type_cols = [c for c in cancer_df.columns if c.startswith("CANCER_TYPE_")]

    if type_cols:
        cancer_df["cancer_type"] = (
            cancer_df[type_cols]
            .idxmax(axis=1)
            .str.replace("CANCER_TYPE_", "", regex=False)
        )
    elif "CANCER_TYPE" in cancer_df.columns:
        cancer_df["cancer_type"] = cancer_df["CANCER_TYPE"].astype(str)
    else:
        raise ValueError("Could not identify cancer type columns in cancer_type_df.csv")

    counts = cancer_df["cancer_type"].dropna().astype(str).value_counts()
    counts.index = counts.index.str.replace("_", " ", regex=False)
    if len(counts) > top_n:
        other_n = int(counts.iloc[top_n:].sum())
        counts = counts.head(top_n)
        if other_n > 0:
            counts.loc["OTHER"] = other_n
    return counts.sort_values()


def draw_panel_d(ax: plt.Axes) -> None:
    try:
        counts = load_cancer_type_distribution()
    except (FileNotFoundError, ValueError) as exc:
        ax.text(0.5, 0.5, str(exc), ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return

    if counts.empty:
        ax.text(0.5, 0.5, "No cancer type data available",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        return

    total = int(counts.sum())
    colors = plt.cm.YlGnBu(np.linspace(0.35, 0.85, len(counts)))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white", height=0.72)

    for bar, val in zip(bars, counts.values):
        pct = 100 * val / max(total, 1)
        ax.text(
            bar.get_width() + max(counts.max() * 0.02, 1.0),
            bar.get_y() + bar.get_height() / 2,
            f"{val} ({pct:.1f}%)",
            va="center",
            fontsize=7,
        )

    ax.set_xlabel("Patients (n)")
    ax.set_title("Cancer Type Distribution")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelsize=7)
    ax.grid(axis="x", alpha=0.5)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()


def main() -> None:
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

        fig = plt.figure(figsize=(16, 10.2))
        ax_a = fig.add_axes([0.04, 0.56, 0.92, 0.33])
        ax_b = fig.add_axes([0.05, 0.10, 0.28, 0.34])
        ax_c = fig.add_axes([0.36, 0.10, 0.30, 0.34])
        ax_d = fig.add_axes([0.70, 0.10, 0.24, 0.34])

        draw_panel_a(ax_a)
        draw_panel_b(ax_b)
        draw_panel_c(ax_c, compiled_csv)
        draw_panel_d(ax_d)

        for label, ax in {"A": ax_a, "B": ax_b, "C": ax_c, "D": ax_d}.items():
            ax.text(-0.03, 1.03, label, transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="top", ha="right")

        fig.text(0.98, 0.975, f"Panel C event set: {prevalence_label}",
                 ha="right", va="top", fontsize=8, color="#444")

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
