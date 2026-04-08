"""Figure 3 — Feature class contribution to survival prediction.

Panels:
  A: Stacked bar — number of significantly predicted endpoints per modality × scheme
  B: Pairwise overlap heatmap between modalities
  C: Scatter — text C-index (Y) vs. best-competing-modality C-index (X)
  D: Violin — log(HR) coefficients from a joint Cox model (all 6 risk scores as features)

Data sources:
  - Panels A, B, C: compiled_all_schemes_compiled_metrics.csv
  - Panel D: held_out_risk_scores/{event}/{modality}_risk_scores.csv
             + survival data from embedding prediction DFs
             → joint CoxPH fitted per event, coefficients cached to a CSV

Notes for Panel D:
  The joint Cox model is fitted per event over all 6 modality risk scores.
  This requires that all 6 modality risk score files exist for a given event.
  Results are cached to avoid re-running on every call.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/"
SURV_PATH = os.path.join(DATA_PATH, "time-to-event_analysis/")
RESULTS_PATH = os.path.join(SURV_PATH, "results/")
COMPILED_DIR = os.path.join(RESULTS_PATH, "compiled_all_schemes/")
OUTPUT_DIR = os.path.join(DATA_PATH, "figures/manuscript/")
JOINT_COX_CACHE = os.path.join(OUTPUT_DIR, "joint_cox_coefs.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
SCHEME_COLORS = {
    "death_met":    "#1b6f8a",
    "icd3_post":    "#3d8c40",
    "icd4_post":    "#d17c29",
    "phecode_post": "#8f3b76",
}
SCHEME_DISPLAY = {
    "death_met":    "Death + Met",
    "icd3_post":    "ICD-10 L3",
    "icd4_post":    "ICD-10 L4",
    "phecode_post": "Phecodes",
}

MODALITIES = ["text", "stage", "treatment", "labs", "somatic", "prs"]
MODALITY_DISPLAY = {
    "text": "Text", "stage": "Stage", "treatment": "Treatment",
    "labs": "Labs", "somatic": "Somatic", "prs": "PRS",
}
MODALITY_COLORS = {
    "Text":      "#1665a2",
    "Stage":     "#718096",
    "Treatment": "#c48f00",
    "Labs":      "#188977",
    "Somatic":   "#b55852",
    "PRS":       "#7d62a8",
}

# Significance threshold: C-index column must exceed 0.5 at FDR < 0.05
# In compiled metrics, we use a simple threshold (column present and > 0.5+delta)
CINDEX_SIG_THRESHOLD = 0.55   # adjust to match your FDR-based significance cutoff

# Panel D: max number of events to use for joint Cox fitting
MAX_JOINT_COX_EVENTS = 300


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def set_style():
    sns.set_theme(context="paper", style="whitegrid")
    mpl.rcParams.update({
        "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_compiled_metrics() -> pd.DataFrame:
    for fname in ["all_schemes_compiled_metrics_1pct.csv",
                  "all_schemes_compiled_metrics.csv"]:
        fpath = os.path.join(COMPILED_DIR, fname)
        if os.path.isfile(fpath):
            print(f"Loading compiled metrics: {fpath}")
            return pd.read_csv(fpath)
    raise FileNotFoundError("Compiled metrics CSV not found in " + COMPILED_DIR)


def get_cindex_col(modality: str) -> str:
    """Return the compiled-metrics column name for a modality's C-index."""
    if modality == "text":
        return "text_feat_comp_mean_c_index"
    return f"{modality}_mean_c_index"


def is_significant(df: pd.DataFrame, modality: str,
                   threshold: float = CINDEX_SIG_THRESHOLD) -> pd.Series:
    """Boolean mask: modality has significant C-index for each endpoint."""
    col = get_cindex_col(modality)
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].notna() & (df[col] > threshold)


def load_risk_scores(scheme: str, event: str, modality: str) -> pd.DataFrame | None:
    results_dir = SCHEME_CONFIG[scheme]["results_dir"]
    fpath = os.path.join(
        RESULTS_PATH, results_dir, "held_out_risk_scores", event,
        f"{modality}_risk_scores.csv"
    )
    if not os.path.isfile(fpath):
        return None
    df = pd.read_csv(fpath)
    score_cols = [c for c in df.columns if "risk_score" in c.lower() and c != "DFCI_MRN"]
    if score_cols:
        df = df.rename(columns={score_cols[0]: f"{modality}_risk_score"})
    return df


def load_survival_data(scheme: str, event: str) -> pd.DataFrame:
    fname = SCHEME_CONFIG[scheme]["embedding_file"]
    fpath = os.path.join(SURV_PATH, fname)
    cols = ["DFCI_MRN", event, f"tt_{event}"]
    df = pd.read_csv(fpath, usecols=cols)
    df = df.rename(columns={event: "event_indicator", f"tt_{event}": "time"})
    df = df.dropna(subset=["event_indicator", "time"])
    df["event_indicator"] = df["event_indicator"].astype(bool)
    return df


# ---------------------------------------------------------------------------
# Panel A — stacked bar: significant endpoints per modality × scheme
# ---------------------------------------------------------------------------

def draw_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    scheme_order = ["death_met", "icd3_post", "icd4_post", "phecode_post"]
    records = []
    for mod in MODALITIES:
        for scheme in scheme_order:
            sub = df[df["scheme"] == scheme]
            n_sig = is_significant(sub, mod).sum()
            records.append({
                "modality": MODALITY_DISPLAY[mod],
                "scheme": SCHEME_DISPLAY[scheme],
                "n_sig": n_sig,
            })
    count_df = pd.DataFrame(records)

    pivot = count_df.pivot(index="modality", columns="scheme", values="n_sig")
    scheme_labels = [SCHEME_DISPLAY[s] for s in scheme_order]
    pivot = pivot[[s for s in scheme_labels if s in pivot.columns]]

    mod_order = [MODALITY_DISPLAY[m] for m in MODALITIES]
    pivot = pivot.reindex(mod_order)

    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for j, (scheme_label, scheme_key) in enumerate(zip(scheme_labels, scheme_order)):
        if scheme_label not in pivot.columns:
            continue
        vals = pivot[scheme_label].fillna(0).values
        ax.bar(x, vals, bottom=bottom, label=scheme_label,
               color=SCHEME_COLORS[scheme_key], edgecolor="white", linewidth=0.5)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(mod_order, rotation=20, ha="right")
    ax.set_ylabel("Significant endpoints (n)")
    ax.set_title("Significant Endpoints per Modality")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.5)
    ax.grid(axis="x", visible=False)

    # Total count labels
    for xi, total in zip(x, bottom):
        ax.text(xi, total + 1, str(int(total)), ha="center", va="bottom", fontsize=7.5)


# ---------------------------------------------------------------------------
# Panel B — pairwise overlap heatmap
# ---------------------------------------------------------------------------

def draw_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    labels = [MODALITY_DISPLAY[m] for m in MODALITIES]
    n = len(labels)
    matrix = np.zeros((n, n))

    sig_masks = {MODALITY_DISPLAY[m]: is_significant(df, m) for m in MODALITIES}

    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if i == j:
                matrix[i, j] = sig_masks[li].sum()
            else:
                matrix[i, j] = (sig_masks[li] & sig_masks[lj]).sum()

    # Show only lower triangle + diagonal
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)

    im = ax.imshow(np.where(mask, np.nan, matrix),
                   cmap="Blues", vmin=0, aspect="auto", interpolation="nearest")

    for i in range(n):
        for j in range(n):
            if not mask[i, j]:
                val = int(matrix[i, j])
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8, color="white" if matrix[i, j] > matrix.max() * 0.6 else "#333")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_title("Pairwise Overlap in Significant Endpoints")
    plt.colorbar(im, ax=ax, label="# endpoints", fraction=0.04, pad=0.02)


# ---------------------------------------------------------------------------
# Panel C — text (Y) vs. best competitor (X) scatter
# ---------------------------------------------------------------------------

def draw_panel_c(ax: plt.Axes, df: pd.DataFrame) -> None:
    text_col = get_cindex_col("text")
    competitor_cols = {m: get_cindex_col(m) for m in MODALITIES if m != "text"
                       and get_cindex_col(m) in df.columns}

    sig_text = is_significant(df, "text")
    plot_df = df[sig_text].copy()
    if plot_df.empty:
        ax.text(0.5, 0.5, "No significant text endpoints", ha="center", va="center",
                transform=ax.transAxes)
        return

    # Best competitor per endpoint
    comp_matrix = plot_df[[v for v in competitor_cols.values()]].copy()
    comp_matrix.columns = list(competitor_cols.keys())
    plot_df["best_comp_cindex"] = comp_matrix.max(axis=1)
    plot_df["best_comp_modality"] = comp_matrix.idxmax(axis=1).map(MODALITY_DISPLAY)
    plot_df["text_cindex"] = plot_df[text_col]

    plot_df = plot_df.dropna(subset=["text_cindex", "best_comp_cindex"])

    for mod_key in [m for m in MODALITIES if m != "text"]:
        mod_display = MODALITY_DISPLAY[mod_key]
        sub = plot_df[plot_df["best_comp_modality"] == mod_display]
        if sub.empty:
            continue
        ax.scatter(sub["best_comp_cindex"], sub["text_cindex"],
                   label=mod_display, color=MODALITY_COLORS[mod_display],
                   s=18, alpha=0.6, edgecolors="none")

    lims = [
        max(0.49, plot_df[["text_cindex", "best_comp_cindex"]].min().min() - 0.01),
        min(1.01, plot_df[["text_cindex", "best_comp_cindex"]].max().max() + 0.01),
    ]
    ax.plot(lims, lims, color="#555", linestyle="--", lw=1.0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    n_text_wins = (plot_df["text_cindex"] > plot_df["best_comp_cindex"]).sum()
    n_comp_wins = (plot_df["text_cindex"] <= plot_df["best_comp_cindex"]).sum()
    ax.text(0.04, 0.96,
            f"Text wins: {n_text_wins}\nCompetitor wins: {n_comp_wins}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc"))

    ax.set_xlabel("Best competitor C-index")
    ax.set_ylabel("Text C-index")
    ax.set_title("Text vs. Best Competing Modality")
    ax.legend(frameon=False, fontsize=7, loc="lower right")


# ---------------------------------------------------------------------------
# Panel D — joint Cox model log(HR) violins
# ---------------------------------------------------------------------------

def fit_joint_cox_for_event(scheme: str, event: str,
                             min_events: int = 15) -> dict[str, float] | None:
    """Fit a joint CoxPH with all 6 modality risk scores as predictors.
    Returns dict of {modality: log_HR} or None if insufficient data.
    """
    # Load survival data
    try:
        surv = load_survival_data(scheme, event)
    except Exception:
        return None

    if surv["event_indicator"].sum() < min_events:
        return None

    # Load all 6 risk scores
    merged = surv.copy()
    for mod in MODALITIES:
        rs = load_risk_scores(scheme, event, mod)
        if rs is None:
            return None  # require all modalities
        rs = rs.rename(columns={c: f"{mod}_risk_score"
                                 for c in rs.columns if c != "DFCI_MRN"})
        merged = merged.merge(rs[["DFCI_MRN", f"{mod}_risk_score"]], on="DFCI_MRN", how="inner")

    merged = merged.dropna()
    if len(merged) < 50 or merged["event_indicator"].sum() < min_events:
        return None

    # Standardize risk scores
    score_cols = [f"{m}_risk_score" for m in MODALITIES]
    merged[score_cols] = (merged[score_cols] - merged[score_cols].mean()) / (merged[score_cols].std() + 1e-8)

    fit_df = merged[["time", "event_indicator"] + score_cols].copy()
    fit_df["event_indicator"] = fit_df["event_indicator"].astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(fit_df, duration_col="time", event_col="event_indicator")
            coefs = {}
            for mod in MODALITIES:
                col = f"{mod}_risk_score"
                if col in cph.params_.index:
                    coefs[MODALITY_DISPLAY[mod]] = float(cph.params_[col])
            return coefs if coefs else None
        except Exception:
            return None


def compute_or_load_joint_cox_coefs(df: pd.DataFrame,
                                    cache_path: str = JOINT_COX_CACHE,
                                    max_events: int = MAX_JOINT_COX_EVENTS) -> pd.DataFrame:
    """Compute joint Cox coefficients for all events, using cache if available."""
    if os.path.isfile(cache_path):
        print(f"Loading joint Cox coefficients from cache: {cache_path}")
        return pd.read_csv(cache_path)

    print("Computing joint Cox model coefficients (this may take a while)...")
    records = []
    events_to_run = df.head(max_events)

    for _, row in events_to_run.iterrows():
        scheme, event = row["scheme"], row["event"]
        coefs = fit_joint_cox_for_event(scheme, event)
        if coefs:
            coefs["scheme"] = scheme
            coefs["event"] = event
            records.append(coefs)

    if not records:
        return pd.DataFrame()

    coef_df = pd.DataFrame(records)
    coef_df.to_csv(cache_path, index=False)
    print(f"Saved joint Cox coefficients → {cache_path}")
    return coef_df


def draw_panel_d(ax: plt.Axes, coef_df: pd.DataFrame) -> None:
    if coef_df.empty:
        ax.text(0.5, 0.5, "Joint Cox coefficients not available",
                ha="center", va="center", transform=ax.transAxes)
        return

    mod_display_order = [MODALITY_DISPLAY[m] for m in MODALITIES]
    available_mods = [m for m in mod_display_order if m in coef_df.columns]

    long_df = coef_df[available_mods].melt(var_name="Modality", value_name="log_HR")
    long_df = long_df.dropna(subset=["log_HR"])

    palette = {m: MODALITY_COLORS[m] for m in available_mods}

    sns.violinplot(data=long_df, x="Modality", y="log_HR",
                   order=available_mods, palette=palette,
                   inner="box", cut=0, linewidth=0.8, ax=ax)
    sns.stripplot(data=long_df, x="Modality", y="log_HR",
                  order=available_mods, color="#333", alpha=0.12,
                  jitter=True, size=2.0, ax=ax)

    ax.axhline(0, color="#444", linestyle="--", lw=1.0)
    ax.set_xlabel("")
    ax.set_ylabel("log(HR) from joint Cox model")
    ax.set_title("Marginal Effect of Each Modality\n(Joint Cox with all risk scores)")
    ax.set_xticklabels(available_mods, rotation=20, ha="right")

    # Wilcoxon test vs 0 for each modality
    for i, mod in enumerate(available_mods):
        vals = coef_df[mod].dropna()
        if len(vals) < 5:
            continue
        try:
            _, p = wilcoxon(vals)
        except Exception:
            continue
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(i, ax.get_ylim()[1] * 0.97, stars,
                ha="center", va="top", fontsize=8, fontweight="bold")

    ax.grid(axis="y", alpha=0.5)
    ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_style()

    df = load_compiled_metrics()

    # Panel D: compute or load joint Cox coefficients
    coef_df = compute_or_load_joint_cox_coefs(df)

    fig = plt.figure(figsize=(15, 13))
    ax_a = fig.add_axes([0.07, 0.55, 0.38, 0.38])
    ax_b = fig.add_axes([0.56, 0.55, 0.38, 0.38])
    ax_c = fig.add_axes([0.07, 0.07, 0.38, 0.38])
    ax_d = fig.add_axes([0.56, 0.07, 0.38, 0.38])

    draw_panel_a(ax_a, df)
    draw_panel_b(ax_b, df)
    draw_panel_c(ax_c, df)
    draw_panel_d(ax_d, coef_df)

    for label, ax in {"A": ax_a, "B": ax_b, "C": ax_c, "D": ax_d}.items():
        ax.text(-0.12, 1.04, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="right")

    caption = (
        "Figure 3. Feature class contributions to survival prediction. "
        "(A) Number of significantly predicted endpoints per modality, stratified by outcome scheme. "
        "(B) Pairwise overlap in significantly predicted endpoints between modalities; "
        "diagonal = total significant per modality. "
        "(C) Text C-index (y-axis) vs. best-competing-modality C-index for endpoints where text "
        "is significant; points above diagonal indicate text outperforms all other modalities. "
        "(D) Distribution of log(HR) coefficients from a joint Cox model including all six "
        "modality-specific risk scores; asterisks indicate deviation from zero (Wilcoxon, "
        "***p<0.001, **p<0.01, *p<0.05)."
    )
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=7.5, style="italic",
             bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#ddd", alpha=0.8))

    out_stem = os.path.join(OUTPUT_DIR, "figure3_feature_comps")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", facecolor="white", bbox_inches="tight")
    print(f"Saved figure 3 → {out_stem}.png/.pdf")


if __name__ == "__main__":
    main()
