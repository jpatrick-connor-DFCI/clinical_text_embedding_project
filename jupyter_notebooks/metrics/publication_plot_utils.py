"""Publication-grade plotting utilities for cross-scheme model summaries."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCHEME_DISPLAY_NAMES = {
    "death_met": "Death + Metastasis",
    "icd3_post": "ICD-10 Level 3",
    "icd4_post": "ICD-10 Level 4",
    "phecode_post": "Phecodes",
}

SCHEME_PALETTE = {
    "death_met": "#1b6f8a",
    "icd3_post": "#3d8c40",
    "icd4_post": "#d17c29",
    "phecode_post": "#8f3b76",
}

MODEL_PALETTE = {
    "Base": "#4d4d4d",
    "Text (full cohort)": "#c53b31",
}

FEATURE_DISPLAY_NAMES = {
    "stage": "Stage",
    "treatment": "Treatment",
    "labs": "Labs",
    "somatic": "Somatic",
    "prs": "PRS",
    "text": "Text",
}

FEATURE_PALETTE = {
    "Stage": "#718096",
    "Treatment": "#c48f00",
    "Labs": "#188977",
    "Somatic": "#b55852",
    "PRS": "#7d62a8",
    "Text": "#1665a2",
}

CATEGORY_DISPLAY_NAMES = {
    "Death": "Death",
    "Metastasis": "Metastasis",
    "VTE": "VTE",
    "icd3_post": "ICD-10 L3",
    "icd4_post": "ICD-10 L4",
    "phecode_post": "Phecodes",
}


def set_publication_style() -> None:
    """Apply a clean manuscript-friendly plotting theme."""
    sns.set_theme(context="paper", style="whitegrid")
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.labelweight": "medium",
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#444444",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "grid.color": "#d7d7d7",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
        }
    )


def _ensure_output_dir(output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_figure(fig: plt.Figure, output_dir: str | Path, stem: str) -> None:
    out = _ensure_output_dir(output_dir)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{suffix}", facecolor="white")


def _scheme_label(scheme: str) -> str:
    return SCHEME_DISPLAY_NAMES.get(scheme, scheme)


def _scheme_palette_with_labels() -> dict[str, str]:
    return {_scheme_label(key): value for key, value in SCHEME_PALETTE.items()}


def _category_label(category: str) -> str:
    return CATEGORY_DISPLAY_NAMES.get(category, category)


def _wrap_event_label(event_description: str, scheme: str, width: int = 44) -> str:
    scheme_label = _scheme_label(scheme)
    return textwrap.fill(f"{event_description} [{scheme_label}]", width=width)


def _feature_long_df(
    results_df: pd.DataFrame,
    feature_names: list[str],
    metric: str = "mean_c_index",
) -> pd.DataFrame:
    metric_cols = {}
    for feature in feature_names:
        col = "text_feat_comp_mean_c_index" if feature == "text" else f"{feature}_{metric}"
        if col in results_df.columns:
            metric_cols[feature] = col

    if not metric_cols:
        return pd.DataFrame(columns=["event", "scheme", "feature", "value"])

    long_df = pd.concat(
        [
            results_df[["event", "scheme"]].assign(feature=FEATURE_DISPLAY_NAMES.get(feature, feature), value=results_df[col])
            for feature, col in metric_cols.items()
        ],
        ignore_index=True,
    ).dropna(subset=["value"])
    return long_df


def _draw_delta_c_index_by_scheme(
    ax: plt.Axes,
    results_df: pd.DataFrame,
    scheme_order: list[str],
) -> None:
    plot_df = results_df[["scheme", "delta_c_index"]].dropna().copy()
    plot_df["scheme_label"] = plot_df["scheme"].map(_scheme_label)
    order = [_scheme_label(s) for s in scheme_order if s in plot_df["scheme"].unique()]
    palette = _scheme_palette_with_labels()

    sns.boxplot(
        data=plot_df,
        y="scheme_label",
        x="delta_c_index",
        hue="scheme_label",
        order=order,
        dodge=False,
        palette=palette,
        width=0.56,
        linewidth=1.0,
        fliersize=0,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()

    sns.stripplot(
        data=plot_df,
        y="scheme_label",
        x="delta_c_index",
        order=order,
        color="#202020",
        alpha=0.35,
        jitter=0.18,
        size=3.0,
        ax=ax,
    )

    ax.axvline(0, color="#444444", linestyle=(0, (4, 2)), linewidth=1.0)
    ax.set_xlabel("Change in held-out C-index from text embeddings")
    ax.set_ylabel("")
    ax.set_title("Performance Gain by Endpoint Scheme")

    for row_idx, label in enumerate(order):
        subset = plot_df.loc[plot_df["scheme_label"] == label, "delta_c_index"]
        if subset.empty:
            continue
        ax.text(
            1.01,
            row_idx,
            f"n={len(subset)}  med={subset.median():+.3f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=8,
            color="#444444",
        )

    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)


def plot_delta_c_index_by_scheme(
    results_df: pd.DataFrame,
    output_dir: str | Path,
    scheme_order: list[str],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    _draw_delta_c_index_by_scheme(ax, results_df, scheme_order)
    fig.subplots_adjust(right=0.82)
    _save_figure(fig, output_dir, "delta_c_index_by_scheme_publication")
    return fig


def _draw_base_vs_text_scatter(
    ax: plt.Axes,
    results_df: pd.DataFrame,
    scheme_order: list[str],
) -> None:
    plot_df = results_df.dropna(subset=["base_mean_c_index", "text_full_cohort_mean_c_index"]).copy()
    if plot_df.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return

    min_val = plot_df[["base_mean_c_index", "text_full_cohort_mean_c_index"]].min().min()
    max_val = plot_df[["base_mean_c_index", "text_full_cohort_mean_c_index"]].max().max()
    padding = 0.02
    lims = [max(0.45, min_val - padding), min(1.0, max_val + padding)]

    for scheme in scheme_order:
        subset = plot_df.loc[plot_df["scheme"] == scheme]
        if subset.empty:
            continue
        ax.scatter(
            subset["base_mean_c_index"],
            subset["text_full_cohort_mean_c_index"],
            label=f"{_scheme_label(scheme)} (n={len(subset)})",
            color=SCHEME_PALETTE.get(scheme, "#4d4d4d"),
            alpha=0.68,
            s=34,
            edgecolors="white",
            linewidths=0.4,
        )

    improved_pct = 100 * (
        plot_df["text_full_cohort_mean_c_index"] > plot_df["base_mean_c_index"]
    ).mean()
    median_delta = (
        plot_df["text_full_cohort_mean_c_index"] - plot_df["base_mean_c_index"]
    ).median()

    ax.plot(lims, lims, color="#444444", linestyle=(0, (4, 2)), linewidth=1.0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Base model held-out C-index")
    ax.set_ylabel("Text model held-out C-index")
    ax.set_title("Base vs Text Model Discrimination")
    ax.text(
        0.03,
        0.97,
        f"{improved_pct:.1f}% improved\nMedian delta = {median_delta:+.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True)


def plot_base_vs_text_scatter(
    results_df: pd.DataFrame,
    output_dir: str | Path,
    scheme_order: list[str],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    _draw_base_vs_text_scatter(ax, results_df, scheme_order)
    _save_figure(fig, output_dir, "scatter_base_vs_text_all_schemes_publication")
    return fig


def _draw_model_performance_by_category(
    ax: plt.Axes,
    results_df: pd.DataFrame,
    category_order: list[str],
) -> None:
    long_df = (
        results_df.melt(
            id_vars=["event", "event_category"],
            value_vars=["base_mean_c_index", "text_full_cohort_mean_c_index"],
            var_name="model",
            value_name="c_index",
        )
        .dropna(subset=["c_index"])
        .copy()
    )
    long_df["model"] = long_df["model"].map(
        {
            "base_mean_c_index": "Base",
            "text_full_cohort_mean_c_index": "Text (full cohort)",
        }
    )
    categories_present = [cat for cat in category_order if cat in long_df["event_category"].unique()]
    if not categories_present:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return

    summary_df = (
        long_df.groupby(["event_category", "model"])
        .agg(mean_c_index=("c_index", "mean"), sem=("c_index", "sem"), n=("c_index", "size"))
        .reset_index()
    )
    summary_df["ci95"] = 1.96 * summary_df["sem"].fillna(0.0)

    x = np.arange(len(categories_present))
    offsets = {"Base": -0.12, "Text (full cohort)": 0.12}
    category_counts = long_df.groupby("event_category")["event"].nunique().to_dict()

    for model_name in ("Base", "Text (full cohort)"):
        model_df = (
            summary_df.loc[summary_df["model"] == model_name]
            .set_index("event_category")
            .reindex(categories_present)
        )
        xpos = x + offsets[model_name]
        ax.errorbar(
            xpos,
            model_df["mean_c_index"],
            yerr=model_df["ci95"],
            fmt="o-",
            markersize=6,
            linewidth=2.0,
            capsize=3,
            color=MODEL_PALETTE[model_name],
            label=model_name,
        )

    xlabels = [
        f"{_category_label(cat)}\n(n={category_counts.get(cat, 0)})" for cat in categories_present
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Mean held-out C-index")
    ax.set_xlabel("")
    ax.set_title("Performance by Endpoint Category")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


def plot_model_performance_by_category(
    results_df: pd.DataFrame,
    output_dir: str | Path,
    category_order: list[str],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    _draw_model_performance_by_category(ax, results_df, category_order)
    _save_figure(fig, output_dir, "model_performance_by_category_publication")
    return fig


def _draw_feature_comparison_distribution(
    ax: plt.Axes,
    results_df: pd.DataFrame,
    feature_names: list[str],
) -> None:
    long_df = _feature_long_df(results_df, feature_names)
    if long_df.empty:
        ax.text(0.5, 0.5, "No feature-comparison metrics available", ha="center", va="center")
        return

    order = (
        long_df.groupby("feature")["value"].median().sort_values(ascending=False).index.tolist()
    )
    palette = {feature: FEATURE_PALETTE.get(feature, "#4d4d4d") for feature in order}

    sns.boxplot(
        data=long_df,
        x="feature",
        y="value",
        hue="feature",
        order=order,
        dodge=False,
        palette=palette,
        width=0.58,
        linewidth=1.0,
        fliersize=0,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()

    sns.stripplot(
        data=long_df,
        x="feature",
        y="value",
        order=order,
        color="#202020",
        alpha=0.22,
        jitter=0.18,
        size=2.8,
        ax=ax,
    )

    counts = long_df.groupby("feature")["value"].size().to_dict()
    ax.axhline(0.5, color="#444444", linestyle=(0, (4, 2)), linewidth=1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Held-out C-index")
    ax.set_title("Single-Modality Performance Across Endpoints")
    ax.set_xticklabels([f"{feature}\n(n={counts.get(feature, 0)})" for feature in order])
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


def plot_feature_comparison_distribution(
    results_df: pd.DataFrame,
    output_dir: str | Path,
    feature_names: list[str],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    _draw_feature_comparison_distribution(ax, results_df, feature_names)
    _save_figure(fig, output_dir, "feature_comparison_distribution_publication")
    return fig


def plot_top_text_gain_events(
    results_df: pd.DataFrame,
    output_dir: str | Path,
    top_n: int = 15,
) -> plt.Figure:
    plot_df = (
        results_df.dropna(subset=["delta_c_index"])
        .nlargest(top_n, "delta_c_index")
        .copy()
        .sort_values("delta_c_index", ascending=True)
    )
    plot_df["label"] = [
        _wrap_event_label(event_description, scheme) for event_description, scheme in zip(plot_df["event_description"], plot_df["scheme"])
    ]

    fig, ax = plt.subplots(figsize=(9.6, max(4.6, 0.42 * len(plot_df) + 1.0)))
    ax.hlines(plot_df["label"], 0, plot_df["delta_c_index"], color="#cfcfcf", linewidth=1.5)
    ax.scatter(
        plot_df["delta_c_index"],
        plot_df["label"],
        s=58,
        color=[SCHEME_PALETTE.get(scheme, "#4d4d4d") for scheme in plot_df["scheme"]],
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    ax.axvline(0, color="#444444", linestyle=(0, (4, 2)), linewidth=1.0)
    ax.set_xlabel("Change in held-out C-index from text embeddings")
    ax.set_ylabel("")
    ax.set_title(f"Top {len(plot_df)} Endpoint Gains from Text Embeddings")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    _save_figure(fig, output_dir, "top_text_gain_events_publication")
    return fig


def plot_publication_summary(
    results_df: pd.DataFrame,
    output_dir: str | Path,
    scheme_order: list[str],
    category_order: list[str],
    feature_names: list[str],
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.2))

    _draw_delta_c_index_by_scheme(axes[0, 0], results_df, scheme_order)
    _draw_base_vs_text_scatter(axes[0, 1], results_df, scheme_order)
    _draw_model_performance_by_category(axes[1, 0], results_df, category_order)
    _draw_feature_comparison_distribution(axes[1, 1], results_df, feature_names)

    for idx, ax in enumerate(axes.flat):
        ax.text(
            -0.14,
            1.06,
            chr(ord("A") + idx),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.suptitle("Cross-Scheme Survival Modeling Summary", y=1.02, fontsize=14, fontweight="semibold")
    fig.subplots_adjust(right=0.86, hspace=0.32, wspace=0.28)
    _save_figure(fig, output_dir, "all_schemes_publication_summary")
    return fig
