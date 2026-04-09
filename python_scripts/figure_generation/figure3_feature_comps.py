"""Figure 3 — Feature class contribution to survival prediction.

Panels:
  A: Stacked bar — number of significantly predicted endpoints per modality × scheme
  B: Pairwise overlap heatmap between modalities
  C: Scatter — text C-index (Y) vs. best-competing-modality C-index (X)
  D: Joint-model log HRs across endpoints for each modality

Data sources:
  - Panels A, B, C: risk_score_coxph/univariate_modality_metrics.csv
  - Panel D: risk_score_coxph/joint_model_betas.csv

These files are produced by python_scripts/model_evaluation/feature_risk_score_coxph.py.
"""

from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from figure_generation_utils import (
    OUTPUT_DIR,
    PREVALENCE_FILTERS_TO_PLOT,
    RESULTS_PATH,
    get_compiled_metrics_path,
    prevalence_filter_to_label,
    prevalence_filter_to_suffix,
    set_manuscript_style,
)

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

MODALITIES = ["text", "stage", "treatment", "somatic", "prs"]
MODALITY_DISPLAY = {
    "text": "Text", "stage": "Stage", "treatment": "Treatment",
    "somatic": "Somatic", "prs": "PRS",
}
MODALITY_COLORS = {
    "Text":      "#1665a2",
    "Stage":     "#718096",
    "Treatment": "#c48f00",
    "Somatic":   "#b55852",
    "PRS":       "#7d62a8",
}

RISK_SCORE_COXPH_DIRNAME = "risk_score_coxph"
UNIVAR_METRICS_FILE = "univariate_modality_metrics.csv"
JOINT_BETAS_FILE = "joint_model_betas.csv"

# Significance threshold for the second-stage held-out risk-score CoxPH runs.
CINDEX_SIG_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def set_style():
    set_manuscript_style()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_risk_score_coxph_dir(scheme: str) -> str:
    results_dir = SCHEME_CONFIG[scheme]["results_dir"]
    return os.path.join(RESULTS_PATH, results_dir, RISK_SCORE_COXPH_DIRNAME)


def load_feature_risk_score_run_outputs() -> pd.DataFrame:
    """Load tidy per-event outputs from feature_risk_score_coxph.py across schemes."""
    univariate_frames: list[pd.DataFrame] = []
    missing_univar: list[str] = []

    for scheme in sorted(SCHEME_CONFIG):
        out_dir = get_risk_score_coxph_dir(scheme)
        univar_fp = os.path.join(out_dir, UNIVAR_METRICS_FILE)

        if os.path.isfile(univar_fp):
            cur = pd.read_csv(univar_fp)
            cur["scheme"] = scheme
            univariate_frames.append(cur)
        else:
            missing_univar.append(univar_fp)

    if not univariate_frames:
        raise FileNotFoundError(
            "No risk-score CoxPH outputs found. Run "
            "python_scripts/model_evaluation/feature_risk_score_coxph.py first."
        )

    if missing_univar:
        warnings.warn(
            "Missing univariate modality metrics for some schemes:\n  "
            + "\n  ".join(missing_univar),
            stacklevel=2,
        )

    univar_df = pd.concat(univariate_frames, ignore_index=True)
    return prepare_univariate_metrics(univar_df)


def load_joint_model_betas() -> pd.DataFrame:
    """Load joint-model beta coefficients across schemes."""
    beta_frames: list[pd.DataFrame] = []
    missing_beta: list[str] = []

    for scheme in sorted(SCHEME_CONFIG):
        out_dir = get_risk_score_coxph_dir(scheme)
        beta_fp = os.path.join(out_dir, JOINT_BETAS_FILE)

        if os.path.isfile(beta_fp):
            cur = pd.read_csv(beta_fp)
            cur["scheme"] = scheme
            beta_frames.append(cur)
        else:
            missing_beta.append(beta_fp)

    if not beta_frames:
        raise FileNotFoundError(
            "No joint-model beta outputs found. Run "
            "python_scripts/model_evaluation/feature_risk_score_coxph.py first."
        )

    if missing_beta:
        warnings.warn(
            "Missing joint-model beta outputs for some schemes:\n  "
            + "\n  ".join(missing_beta),
            stacklevel=2,
        )

    beta_df = pd.concat(beta_frames, ignore_index=True)
    required_cols = {"scheme", "event", "modality", "beta"}
    missing_cols = sorted(required_cols - set(beta_df.columns))
    if missing_cols:
        raise ValueError(f"Joint-model betas missing required columns: {missing_cols}")

    beta_df = beta_df[beta_df["modality"].isin(MODALITIES)].copy()
    beta_df["beta"] = pd.to_numeric(beta_df["beta"], errors="coerce")
    beta_df["is_valid"] = beta_df["beta"].notna() & np.isfinite(beta_df["beta"])
    beta_df = beta_df.drop_duplicates(subset=["scheme", "event", "modality"], keep="last")
    return beta_df


def prepare_univariate_metrics(univar_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"scheme", "event", "modality", "mean_c_index"}
    missing_cols = sorted(required_cols - set(univar_df.columns))
    if missing_cols:
        raise ValueError(f"Univariate metrics missing required columns: {missing_cols}")

    plot_df = univar_df.copy()
    if "error" not in plot_df.columns:
        plot_df["error"] = ""

    plot_df["error"] = plot_df["error"].fillna("").astype(str)
    plot_df = plot_df[plot_df["modality"].isin(MODALITIES)].copy()
    plot_df = plot_df.drop_duplicates(subset=["scheme", "event", "modality"], keep="last")
    plot_df["is_valid"] = plot_df["error"].eq("") & plot_df["mean_c_index"].notna()
    plot_df["is_significant"] = plot_df["is_valid"] & (plot_df["mean_c_index"] > CINDEX_SIG_THRESHOLD)
    return plot_df


def build_metric_matrix(univar_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    matrix = (univar_df
              .pivot(index=["scheme", "event"], columns="modality", values=value_col)
              .reset_index())
    matrix.columns.name = None
    return matrix


def load_event_subset(prevalence_filter: str | None) -> set[tuple[str, str]] | None:
    if prevalence_filter in (None, ""):
        return None
    compiled_fp = get_compiled_metrics_path(prevalence_filter)
    compiled_df = pd.read_csv(compiled_fp, usecols=["scheme", "event"])
    return set(compiled_df.itertuples(index=False, name=None))


def subset_run_outputs(
    univar_df: pd.DataFrame,
    event_subset: set[tuple[str, str]] | None,
) -> pd.DataFrame:
    if event_subset is None:
        return univar_df

    univar_sub = univar_df[
        univar_df[["scheme", "event"]].apply(tuple, axis=1).isin(event_subset)
    ].copy()
    return univar_sub


# ---------------------------------------------------------------------------
# Panel A — stacked bar: significant endpoints per modality × scheme
# ---------------------------------------------------------------------------

def draw_panel_a(ax: plt.Axes, univar_df: pd.DataFrame) -> None:
    scheme_order = ["death_met", "icd3_post", "icd4_post", "phecode_post"]
    records = []
    for mod in MODALITIES:
        for scheme in scheme_order:
            sub = univar_df[(univar_df["scheme"] == scheme) & (univar_df["modality"] == mod)]
            n_sig = int(sub["is_significant"].sum())
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

def draw_panel_b(ax: plt.Axes, univar_df: pd.DataFrame) -> None:
    labels = [MODALITY_DISPLAY[m] for m in MODALITIES]
    n = len(labels)
    matrix = np.zeros((n, n))

    sig_sets = {}
    for modality in MODALITIES:
        sig_events = univar_df[
            (univar_df["modality"] == modality) & univar_df["is_significant"]
        ][["scheme", "event"]]
        sig_sets[MODALITY_DISPLAY[modality]] = set(sig_events.itertuples(index=False, name=None))

    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if i == j:
                matrix[i, j] = len(sig_sets[li])
            else:
                matrix[i, j] = len(sig_sets[li] & sig_sets[lj])

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

def draw_panel_c(ax: plt.Axes, univar_df: pd.DataFrame) -> None:
    cindex_df = build_metric_matrix(univar_df, "mean_c_index")
    sig_df = build_metric_matrix(univar_df, "is_significant")

    if "text" not in cindex_df.columns or "text" not in sig_df.columns:
        ax.text(0.5, 0.5, "No text modality results found",
                ha="center", va="center", transform=ax.transAxes)
        return

    competitor_cols = [m for m in MODALITIES if m != "text" and m in cindex_df.columns]
    if not competitor_cols:
        ax.text(0.5, 0.5, "No competing modality results found",
                ha="center", va="center", transform=ax.transAxes)
        return

    plot_df = cindex_df.merge(
        sig_df[["scheme", "event", "text"]].rename(columns={"text": "text_is_significant"}),
        on=["scheme", "event"],
        how="left",
    )
    plot_df = plot_df[plot_df["text_is_significant"].fillna(False)].copy()
    if plot_df.empty:
        ax.text(0.5, 0.5, "No significant text endpoints", ha="center", va="center",
                transform=ax.transAxes)
        return

    # Best competitor per endpoint
    comp_matrix = plot_df[competitor_cols].copy()
    plot_df["best_comp_cindex"] = comp_matrix.max(axis=1)
    plot_df["best_comp_modality"] = comp_matrix.idxmax(axis=1).map(MODALITY_DISPLAY)
    plot_df["text_cindex"] = plot_df["text"]

    plot_df = plot_df.dropna(subset=["text_cindex", "best_comp_cindex"])
    if plot_df.empty:
        ax.text(0.5, 0.5, "No endpoints with comparable modality results",
                ha="center", va="center", transform=ax.transAxes)
        return

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
# Panel D — joint-model log HRs
# ---------------------------------------------------------------------------

def draw_panel_d(ax: plt.Axes, beta_df: pd.DataFrame) -> None:
    if beta_df.empty:
        ax.text(0.5, 0.5, "Joint-model beta outputs not available",
                ha="center", va="center", transform=ax.transAxes)
        return

    plot_df = beta_df[
        beta_df["modality"].isin(MODALITIES)
        & beta_df["is_valid"]
    ].copy()
    if plot_df.empty:
        ax.text(0.5, 0.5, "No supported joint-model log HR values found",
                ha="center", va="center", transform=ax.transAxes)
        return
    plot_df["Modality"] = plot_df["modality"].map(MODALITY_DISPLAY)

    mod_display_order = [MODALITY_DISPLAY[m] for m in MODALITIES]
    available_mods = [m for m in mod_display_order if m in plot_df["Modality"].dropna().unique()]
    palette = {m: MODALITY_COLORS[m] for m in available_mods}

    sns.boxplot(data=plot_df, x="Modality", y="beta",
                order=available_mods, palette=palette,
                width=0.58, showfliers=False, linewidth=0.9, ax=ax)
    sns.stripplot(data=plot_df, x="Modality", y="beta",
                  order=available_mods, color="#333", alpha=0.12,
                  jitter=True, size=2.0, ax=ax)

    ax.axhline(0, color="#444", linestyle="--", lw=1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Joint-model log HR")
    ax.set_title("Joint CoxPH Log-HR by Modality")
    ax.set_xticklabels(available_mods, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.5)
    ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_style()

    univar_df = load_feature_risk_score_run_outputs()
    beta_df = load_joint_model_betas()
    generated = 0

    for prevalence_filter in PREVALENCE_FILTERS_TO_PLOT:
        try:
            event_subset = load_event_subset(prevalence_filter)
        except FileNotFoundError as exc:
            warnings.warn(str(exc), stacklevel=2)
            continue

        prevalence_label = prevalence_filter_to_label(prevalence_filter)
        suffix = prevalence_filter_to_suffix(prevalence_filter)
        univar_sub = subset_run_outputs(univar_df, event_subset)
        beta_sub = subset_run_outputs(beta_df, event_subset)

        fig = plt.figure(figsize=(15, 13))
        ax_a = fig.add_axes([0.07, 0.55, 0.38, 0.38])
        ax_b = fig.add_axes([0.56, 0.55, 0.38, 0.38])
        ax_c = fig.add_axes([0.07, 0.07, 0.38, 0.38])
        ax_d = fig.add_axes([0.56, 0.07, 0.38, 0.38])

        draw_panel_a(ax_a, univar_sub)
        draw_panel_b(ax_b, univar_sub)
        draw_panel_c(ax_c, univar_sub)
        draw_panel_d(ax_d, beta_sub)

        for label, ax in {"A": ax_a, "B": ax_b, "C": ax_c, "D": ax_d}.items():
            ax.text(-0.12, 1.04, label, transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="top", ha="right")

        fig.text(0.98, 0.975, f"Event set: {prevalence_label}", ha="right", va="top",
                 fontsize=8, color="#444")

        out_stem = os.path.join(OUTPUT_DIR, f"figure3_feature_comps_{suffix}")
        for ext in ("png", "pdf"):
            fig.savefig(f"{out_stem}.{ext}", facecolor="white", bbox_inches="tight")
        plt.close(fig)
        generated += 1
        print(f"Saved figure 3 ({prevalence_label}) → {out_stem}.png/.pdf")

    if generated == 0:
        raise FileNotFoundError("No figure3 outputs were generated because no prevalence-filter event sets were found.")


if __name__ == "__main__":
    main()
