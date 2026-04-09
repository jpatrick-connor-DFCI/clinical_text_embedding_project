"""Figure 2 — Text embedding model prediction results.

Panels:
  A: Scatter — text vs. base model C-index across all endpoint schemes, colored by event category
  B: Violin — delta C-index (text − base) distribution per scheme
  C: Heatmap — text model C-index for top endpoints × cancer type (requires per-event risk scores)
  D: KM curve — survival by risk quartile for all-cause mortality

Data sources:
  - Panels A, B: compiled_all_schemes_compiled_metrics.csv
  - Panel C: held_out_risk_scores/{event}/text_risk_scores.csv + cancer_type_df.csv
             + {scheme_embedding_df}.csv (for tt_{event} and event indicator)
  - Panel D: same as Panel C
"""

from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from figure_generation_utils import (
    COMPILED_DIR,
    FEATURE_PATH,
    OUTPUT_DIR,
    PREVALENCE_FILTERS_TO_PLOT,
    RESULTS_PATH,
    SURV_PATH,
    get_compiled_metrics_path,
    prevalence_filter_to_label,
    prevalence_filter_to_suffix,
    set_manuscript_style,
)
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.cluster.hierarchy import dendrogram, linkage
from sksurv.metrics import concordance_index_censored

SCHEME_CONFIG = {
    "icd3_post":    {"embedding_file": "level_3_ICD_post_embedding_prediction_df.csv",
                     "results_dir":    "level_3_ICD_post_results"},
    "icd4_post":    {"embedding_file": "level_4_ICD_post_embedding_prediction_df.csv",
                     "results_dir":    "level_4_ICD_post_results"},
    "phecode_post": {"embedding_file": "phecode_post_embedding_prediction_df.csv",
                     "results_dir":    "phecode_post_results"},
    "death_met":    {"embedding_file": "death_met_embedding_prediction_df.csv",
                     "results_dir":    "death_met_results"},
}
SCHEME_DISPLAY = {
    "death_met":    "Death + Metastasis",
    "icd3_post":    "ICD-10 Level 3",
    "icd4_post":    "ICD-10 Level 4",
    "phecode_post": "Phecodes",
}
SCHEME_COLORS = {
    "death_met":    "#1b6f8a",
    "icd3_post":    "#3d8c40",
    "icd4_post":    "#d17c29",
    "phecode_post": "#8f3b76",
}
SCHEME_MARKERS = {
    "death_met":    "D",
    "icd3_post":    "o",
    "icd4_post":    "^",
    "phecode_post": "s",
}

# Panel C — top N events to show in heatmap, top M cancer types
TOP_N_EVENTS = 20
TOP_M_CANCER_TYPES = 8

# Panel D — keep the manuscript KM example focused on all-cause mortality.
KM_EVENT = ("death_met", "death", "All-cause Mortality")

# Significance threshold for C-index in heatmap
HEATMAP_SIG_CINDEX = 0.65


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def set_style():
    set_manuscript_style()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_compiled_metrics(prevalence_filter: str | None = "1pct", allow_fallback: bool = True) -> pd.DataFrame:
    """Load compiled cross-scheme metrics. Falls back to unfiltered if filtered file missing."""
    if prevalence_filter in (None, ""):
        path = get_compiled_metrics_path(None)
    else:
        filtered = os.path.join(COMPILED_DIR, f"all_schemes_compiled_metrics_{prevalence_filter}.csv")
        base = os.path.join(COMPILED_DIR, "all_schemes_compiled_metrics.csv")
        path = filtered if os.path.isfile(filtered) else base
        if not allow_fallback and path != filtered:
            raise FileNotFoundError(f"Compiled metrics not found for filter {prevalence_filter!r}: {filtered}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Compiled metrics not found: {path}")
    print(f"Loading compiled metrics from: {path}")
    return pd.read_csv(path)


def load_embedding_prediction_df(scheme: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Load the embedding prediction DataFrame for a given scheme."""
    fname = SCHEME_CONFIG[scheme]["embedding_file"]
    fpath = os.path.join(SURV_PATH, fname)
    return pd.read_csv(fpath, usecols=usecols)


def load_risk_scores(scheme: str, event: str, modality: str = "text") -> pd.DataFrame:
    """Load held-out risk scores for a given scheme/event/modality."""
    results_dir = SCHEME_CONFIG[scheme]["results_dir"]
    fpath = os.path.join(
        RESULTS_PATH, results_dir, "held_out_risk_scores", event,
        f"{modality}_risk_scores.csv"
    )
    if not os.path.isfile(fpath):
        return None
    df = pd.read_csv(fpath)
    # Standardize column name
    if "risk_score" not in df.columns:
        score_cols = [c for c in df.columns if "risk_score" in c.lower()]
        if score_cols:
            df = df.rename(columns={score_cols[0]: "risk_score"})
    return df


def load_survival_data(scheme: str, event: str) -> pd.DataFrame:
    """Return (DFCI_MRN, tt_{event}, {event}) for a given scheme/event."""
    cols = ["DFCI_MRN", event, f"tt_{event}"]
    df = load_embedding_prediction_df(scheme, usecols=cols)
    df = df.rename(columns={event: "event_indicator", f"tt_{event}": "time"})
    df = df.dropna(subset=["event_indicator", "time"])
    df["event_indicator"] = df["event_indicator"].astype(bool)
    return df


def load_cancer_types(cohort_mrns: pd.Series) -> pd.DataFrame:
    """Return DataFrame with DFCI_MRN and dominant cancer type."""
    ct = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_type_df.csv"))
    ct = ct[ct["DFCI_MRN"].isin(cohort_mrns)].copy()
    type_cols = [c for c in ct.columns if c.startswith("CANCER_TYPE_")]
    ct["cancer_type"] = ct[type_cols].idxmax(axis=1).str.replace("CANCER_TYPE_", "", regex=False)
    return ct[["DFCI_MRN", "cancer_type"]]


# ---------------------------------------------------------------------------
# Panel A — multi-scheme scatter
# ---------------------------------------------------------------------------

def draw_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["base_mean_c_index", "text_full_cohort_mean_c_index"]).copy()
    if "delta_c_index" not in plot_df.columns:
        plot_df["delta_c_index"] = (
            plot_df["text_full_cohort_mean_c_index"] - plot_df["base_mean_c_index"]
        )
    plot_df["scheme_label"] = plot_df["scheme"].map(SCHEME_DISPLAY)

    lim_lo = max(0.48, plot_df[["base_mean_c_index", "text_full_cohort_mean_c_index"]].min().min() - 0.01)
    lim_hi = min(1.01, plot_df[["base_mean_c_index", "text_full_cohort_mean_c_index"]].max().max() + 0.01)

    for scheme in ["death_met", "icd3_post", "icd4_post", "phecode_post"]:
        sub = plot_df[plot_df["scheme"] == scheme]
        if sub.empty:
            continue
        ax.scatter(
            sub["base_mean_c_index"],
            sub["text_full_cohort_mean_c_index"],
            label=f"{SCHEME_DISPLAY[scheme]} (n={len(sub)})",
            color=SCHEME_COLORS[scheme],
            marker=SCHEME_MARKERS[scheme],
            s=20, alpha=0.55, edgecolors="none",
        )

    # Reference line
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color="#555", linestyle="--", lw=1.0, zorder=0)

    # Annotation: % improved and median delta
    delta = plot_df["delta_c_index"]
    pct_improved = 100 * (delta > 0).mean()
    ax.text(0.04, 0.96,
            f"{pct_improved:.0f}% of endpoints: text > base\n"
            f"Median ΔC-index = {delta.median():+.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc"))

    # Annotate top gainers if event_description column exists
    if "event_description" in plot_df.columns:
        top5 = plot_df.nlargest(5, "delta_c_index")
        for _, row in top5.iterrows():
            desc = str(row.get("event_description", row.get("event", "")))
            desc = desc[:30] + "…" if len(desc) > 30 else desc
            ax.annotate(desc,
                        xy=(row["base_mean_c_index"], row["text_full_cohort_mean_c_index"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=6.5,
                        arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6))

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Base model C-index")
    ax.set_ylabel("Text model C-index")
    ax.set_title("Base vs. Text Model Discrimination")
    ax.legend(frameon=False, loc="lower right", fontsize=7, markerscale=1.2)


# ---------------------------------------------------------------------------
# Panel B — delta C-index violin by scheme
# ---------------------------------------------------------------------------

def draw_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    plot_df = df.copy()
    if "delta_c_index" not in plot_df.columns:
        plot_df["delta_c_index"] = (
            plot_df["text_full_cohort_mean_c_index"] - plot_df["base_mean_c_index"]
        )
    plot_df = plot_df.dropna(subset=["delta_c_index"]).copy()
    plot_df["scheme_label"] = plot_df["scheme"].map(SCHEME_DISPLAY)

    order = [SCHEME_DISPLAY[s] for s in ["death_met", "icd3_post", "icd4_post", "phecode_post"]
             if SCHEME_DISPLAY[s] in plot_df["scheme_label"].unique()]
    palette = {SCHEME_DISPLAY[s]: SCHEME_COLORS[s] for s in SCHEME_COLORS}

    sns.violinplot(data=plot_df, x="scheme_label", y="delta_c_index",
                   order=order, palette=palette, inner="box",
                   cut=0, linewidth=0.8, ax=ax)
    sns.stripplot(data=plot_df, x="scheme_label", y="delta_c_index",
                  order=order, color="#333", alpha=0.18, jitter=True, size=2, ax=ax)

    ax.axhline(0, color="#444", linestyle="--", lw=1.0)
    ax.set_xlabel("")
    ax.set_ylabel("ΔC-index (text − base)")
    ax.set_title("Text Embedding Gain by Outcome Scheme")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

    for i, label in enumerate(order):
        sub = plot_df[plot_df["scheme_label"] == label]["delta_c_index"]
        ax.text(i, ax.get_ylim()[1] * 0.97,
                f"n={len(sub)}\nmed={sub.median():+.3f}",
                ha="center", va="top", fontsize=6.5)


# ---------------------------------------------------------------------------
# Panel C — cancer-type-stratified heatmap of text C-indices
# ---------------------------------------------------------------------------

def compute_per_cancer_type_cindex(
    scheme: str, event: str, cancer_type_df: pd.DataFrame,
    min_events: int = 10
) -> dict[str, float]:
    """Compute per-cancer-type C-index for a given event using held-out risk scores."""
    scores = load_risk_scores(scheme, event, modality="text")
    if scores is None:
        return {}

    surv = load_survival_data(scheme, event)
    merged = scores.merge(surv, on="DFCI_MRN").merge(cancer_type_df, on="DFCI_MRN")
    merged = merged.dropna(subset=["risk_score", "time", "event_indicator"])

    result = {}
    for ct, grp in merged.groupby("cancer_type"):
        n_events = grp["event_indicator"].sum()
        if len(grp) < 20 or n_events < min_events:
            continue
        try:
            c_stat, _, _, _, _ = concordance_index_censored(
                grp["event_indicator"].values.astype(bool),
                grp["time"].values,
                grp["risk_score"].values,
            )
            result[ct] = c_stat
        except Exception:
            pass
    return result


def build_heatmap_matrix(
    compiled_df: pd.DataFrame,
    cancer_type_df: pd.DataFrame,
    top_n: int = TOP_N_EVENTS,
    top_m: int = TOP_M_CANCER_TYPES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (events × cancer_types) C-index matrix.
    Returns (heatmap_df, meta_df) where meta_df has event descriptions.
    """
    # Select top N events by text C-index
    top_events = (compiled_df
                  .dropna(subset=["text_full_cohort_mean_c_index"])
                  .nlargest(top_n, "text_full_cohort_mean_c_index")
                  [["scheme", "event", "event_description", "text_full_cohort_mean_c_index"]]
                  .reset_index(drop=True))

    # Get top M cancer types by frequency
    top_cts = cancer_type_df["cancer_type"].value_counts().head(top_m).index.tolist()
    ct_sub = cancer_type_df[cancer_type_df["cancer_type"].isin(top_cts)].copy()

    rows = []
    for _, meta in top_events.iterrows():
        per_ct = compute_per_cancer_type_cindex(meta["scheme"], meta["event"], ct_sub)
        row = {ct: per_ct.get(ct, np.nan) for ct in top_cts}
        row["event_description"] = meta.get("event_description", meta["event"])
        rows.append(row)

    df = pd.DataFrame(rows)
    meta_df = df[["event_description"]].copy()
    heatmap_df = df.drop(columns=["event_description"])
    heatmap_df.index = meta_df["event_description"].values
    return heatmap_df, meta_df


def draw_panel_c(ax: plt.Axes, heatmap_df: pd.DataFrame) -> None:
    # Cluster rows
    valid_rows = heatmap_df.dropna(how="all")
    if valid_rows.empty:
        ax.text(0.5, 0.5, "No cancer-type C-index data available",
                ha="center", va="center", transform=ax.transAxes)
        return

    if len(valid_rows) == 1:
        ordered = valid_rows.copy()
    else:
        imputed = valid_rows.fillna(0.5)
        link = linkage(imputed.values, method="ward")
        order = dendrogram(link, no_plot=True)["leaves"]
        ordered = valid_rows.iloc[order]

    # Shorten labels
    def shorten(s, n=35):
        return s[:n] + "…" if len(str(s)) > n else str(s)

    ordered.index = [shorten(i) for i in ordered.index]

    mask = ordered.isna()
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#e0e0e0")

    im = ax.imshow(ordered.values, cmap=cmap, vmin=0.5, vmax=1.0,
                   aspect="auto", interpolation="nearest")

    # Asterisks for high C-index
    for r in range(ordered.shape[0]):
        for c in range(ordered.shape[1]):
            val = ordered.iloc[r, c]
            if not np.isnan(val) and val >= HEATMAP_SIG_CINDEX:
                ax.text(c, r, "*", ha="center", va="center", fontsize=7, color="white")

    ax.set_xticks(range(len(ordered.columns)))
    ax.set_xticklabels(ordered.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(ordered.index)))
    ax.set_yticklabels(ordered.index, fontsize=7)
    ax.set_title("Text C-index by Cancer Type (top endpoints)")

    plt.colorbar(im, ax=ax, label="C-index", fraction=0.04, pad=0.02)
    ax.text(1.15, -0.05, "* C-index ≥ 0.65", transform=ax.transAxes,
            fontsize=7, ha="right", color="#555")


# ---------------------------------------------------------------------------
# Panel D — KM curves by risk quartile
# ---------------------------------------------------------------------------

def draw_km_quartiles(ax: plt.Axes, scheme: str, event: str,
                      display_name: str, n_at_risk_times: list[int] | None = None) -> None:
    """Draw KM curves stratified by text risk score quartile."""
    if n_at_risk_times is None:
        n_at_risk_times = [0, 12, 24, 36, 48]

    scores = load_risk_scores(scheme, event, modality="text")
    if scores is None:
        ax.text(0.5, 0.5, f"No risk scores:\n{scheme}/{event}",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        return

    surv = load_survival_data(scheme, event)
    merged = scores.merge(surv, on="DFCI_MRN").dropna(subset=["risk_score", "time", "event_indicator"])

    merged["quartile"] = pd.qcut(merged["risk_score"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    quartile_order = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    colors = ["#2166ac", "#74add1", "#f4a582", "#d73027"]

    kmfs = {}
    for q, color in zip(quartile_order, colors):
        sub = merged[merged["quartile"] == q]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["time"], sub["event_indicator"], label=q)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=1.6)
        kmfs[q] = (kmf, sub)

    # Log-rank test
    try:
        result = multivariate_logrank_test(
            merged["time"], merged["quartile"], merged["event_indicator"]
        )
        p_val = result.p_value
        p_str = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    except Exception:
        p_str = ""

    ax.set_title(display_name, fontsize=9)
    ax.set_xlabel("Months since treatment")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(-0.02, 1.05)
    if p_str:
        ax.text(0.97, 0.97, p_str, transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc"))
    ax.legend(fontsize=7, frameon=False, loc="upper right")

    # At-risk table below the plot
    ax.set_xlim(left=0)
    max_time = merged["time"].max()
    at_risk_times = [t for t in n_at_risk_times if t <= max_time]
    if at_risk_times:
        table_y_start = -0.28
        dy = 0.065
        ax.text(-0.1, table_y_start, "At risk:", transform=ax.transAxes,
                fontsize=6.5, ha="right", va="top")
        for qi, (q, color) in enumerate(zip(quartile_order, colors)):
            y_pos = table_y_start - dy * (qi + 1)
            ax.text(-0.1, y_pos, q, transform=ax.transAxes,
                    fontsize=6.5, ha="right", va="top", color=color)
            kmf = kmfs[q][0]
            sub_df = kmfs[q][1]
            for t in at_risk_times:
                n_at_risk = (sub_df["time"] >= t).sum()
                # x position in data coordinates → axes transform
                ax_x = t / (ax.get_xlim()[1] or max_time)
                ax.text(ax_x, y_pos, str(n_at_risk), transform=ax.transAxes,
                        fontsize=6, ha="center", va="top", color=color)


def draw_panel_d(
    ax: plt.Axes,
) -> None:
    draw_km_quartiles(ax, *KM_EVENT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_style()

    # Load cancer type data for panel C
    # Use all MRNs from any scheme (use death_met cohort as proxy)
    print("Loading cancer type data for panel C...")
    cohort_mrns = load_embedding_prediction_df("death_met", usecols=["DFCI_MRN"])["DFCI_MRN"]
    cancer_type_df = load_cancer_types(cohort_mrns)
    generated = 0

    for prevalence_filter in PREVALENCE_FILTERS_TO_PLOT:
        try:
            df = load_compiled_metrics(prevalence_filter, allow_fallback=False)
        except FileNotFoundError as exc:
            warnings.warn(str(exc), stacklevel=2)
            continue

        prevalence_label = prevalence_filter_to_label(prevalence_filter)
        suffix = prevalence_filter_to_suffix(prevalence_filter)

        print(
            "Building per-cancer-type C-index heatmap "
            f"(top {TOP_N_EVENTS} events × top {TOP_M_CANCER_TYPES} types) for {prevalence_label}..."
        )
        heatmap_df, _ = build_heatmap_matrix(df, cancer_type_df, top_n=TOP_N_EVENTS, top_m=TOP_M_CANCER_TYPES)

        fig = plt.figure(figsize=(15, 14))

        ax_a = fig.add_axes([0.06, 0.56, 0.38, 0.38])
        ax_b = fig.add_axes([0.56, 0.56, 0.38, 0.38])
        ax_c = fig.add_axes([0.06, 0.08, 0.38, 0.40])
        ax_d = fig.add_axes([0.56, 0.14, 0.38, 0.34])

        draw_panel_a(ax_a, df)
        draw_panel_b(ax_b, df)
        draw_panel_c(ax_c, heatmap_df)
        draw_panel_d(ax_d)

        panel_labels = {"A": ax_a, "B": ax_b, "C": ax_c, "D": ax_d}
        for label, ax in panel_labels.items():
            ax.text(-0.12, 1.04, label, transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="top", ha="right")

        fig.text(0.98, 0.975, f"Panels A-C event set: {prevalence_label}", ha="right", va="top",
                 fontsize=8, color="#444")
        caption = (
            "Figure 2. Text embedding model performance across outcome types. "
            f"Panels A-C use the {prevalence_label}. "
            "(A) Text vs. base model C-index across all endpoints; marker shape = scheme, "
            "diagonal = equal performance. "
            "(B) Distribution of C-index improvement (text − base) by outcome scheme. "
            "(C) Cancer-type-stratified text C-index heatmap for top-performing endpoints; "
            "gray = insufficient events; * indicates C-index ≥ 0.65. "
            "(D) Kaplan-Meier survival curves by embedding-derived risk quartile for all-cause mortality."
        )
        fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=7.5,
                 style="italic",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#ddd", alpha=0.8))

        out_stem = os.path.join(OUTPUT_DIR, f"figure2_text_results_{suffix}")
        for ext in ("png", "pdf"):
            fig.savefig(f"{out_stem}.{ext}", facecolor="white", bbox_inches="tight")
        plt.close(fig)
        generated += 1
        print(f"Saved figure 2 ({prevalence_label}) → {out_stem}.png/.pdf")

    if generated == 0:
        raise FileNotFoundError("No figure2 outputs were generated because no compiled metrics files were found.")


if __name__ == "__main__":
    main()
