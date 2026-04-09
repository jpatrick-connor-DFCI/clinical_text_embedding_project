"""Figure 5 — Biomarker discovery overview and robustness mockup.

Panels:
  A: Propensity-score ROC curves for predicting ICI receipt, with cancer-type AUC inset
  B: Bubble-grid summary of biomarker robustness across analysis specifications
  C: Three illustrative survival panels for representative biomarker patterns

Data sources (when available):
  - treatment_prediction/{cohort}/{ps_model}_propensity/w_30_day_buffer/predictions.csv
  - biomarker_analysis/matched_cohorts/matched_cohort_{cohort}.csv
  - biomarker_analysis/compiled_results/track1_all_significant_hits.csv
  - biomarker_analysis/compiled_results/track2_all_significant_hits.csv

When one or more sources are unavailable, the script falls back to an illustrative
mockup so the full five-figure manuscript workflow remains runnable.
"""

from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_generation_utils import DATA_PATH, OUTPUT_DIR, set_manuscript_style
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, roc_curve


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MARKER_PATH = os.path.join(DATA_PATH, "biomarker_analysis/")
COMPILED_DIR = os.path.join(MARKER_PATH, "compiled_results/")
TREATMENT_PATH = os.path.join(DATA_PATH, "treatment_prediction/")


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
ROC_COLORS = {
    "Covariates only": "#F28E2B",
    "Covariates + text": "#2E6F9E",
    "Text embeddings only": "#59A14F",
    "All covariates + text": "#E15759",
}
KM_COLORS = {
    "best": "#2E6F9E",
    "mid_hi": "#5DA5DA",
    "mid_lo": "#F28E2B",
    "worst": "#E15759",
}
ROBUSTNESS_BLUE = "#2E86C1"
ROBUSTNESS_RED = "#E76F51"
ROBUSTNESS_GRAY = "#D9D9D9"


def set_style() -> None:
    set_manuscript_style(legend_fontsize=7)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def pretty_marker(marker: str) -> str:
    suffixes = ("_SNV", "_SV", "_FUSION", "_DEL", "_AMP")
    for suffix in suffixes:
        if marker.endswith(suffix):
            return marker[: -len(suffix)] + suffix
    return marker


def _safe_read_csv(path: str, **kwargs) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, **kwargs)


# ---------------------------------------------------------------------------
# Panel A — propensity ROC + cancer-type inset
# ---------------------------------------------------------------------------
def _mock_panel_a_data() -> tuple[list[dict], pd.DataFrame]:
    fpr = np.linspace(0, 1, 300)
    curves = [
        {"label": "Text embeddings only", "auc": 0.66, "color": ROC_COLORS["Text embeddings only"]},
        {"label": "Covariates only", "auc": 0.69, "color": ROC_COLORS["Covariates only"]},
        {"label": "Covariates + text", "auc": 0.78, "color": ROC_COLORS["Covariates + text"]},
        {"label": "All covariates + text", "auc": 0.80, "color": ROC_COLORS["All covariates + text"]},
    ]
    curve_rows: list[dict] = []
    for curve in curves:
        gamma = (1.0 / curve["auc"]) - 1.0
        tpr = np.clip(1 - (1 - fpr) ** gamma, 0, 1)
        curve_rows.append({
            "label": curve["label"],
            "color": curve["color"],
            "auc": curve["auc"],
            "fpr": fpr,
            "tpr": tpr,
        })

    auc_df = pd.DataFrame({
        "cancer_type": ["Melanoma", "Prostate", "CRC", "Breast", "NSCLC"],
        "auc": [0.85, 0.91, 0.83, 0.80, 0.65],
    })
    return curve_rows, auc_df


def load_panel_a_data(cohort: str = "cohort2") -> tuple[list[dict], pd.DataFrame | None, bool]:
    """Return ROC curve data and inset AUC-by-cancer table.

    Returns (curves, auc_by_cancer, used_mockup).
    """
    model_specs = [
        ("covariates_only", "Covariates only"),
        ("covariates_plus_embeddings", "Covariates + text"),
    ]

    curves: list[dict] = []
    selected_model_df: pd.DataFrame | None = None

    for model_key, label in model_specs:
        pred_path = os.path.join(
            TREATMENT_PATH, cohort, f"{model_key}_propensity", "w_30_day_buffer", "predictions.csv"
        )
        pred_df = _safe_read_csv(pred_path)
        if pred_df is None:
            continue
        required = {"ground_truth", "model_probs"}
        if not required.issubset(pred_df.columns):
            warnings.warn(f"Skipping ROC input missing columns: {pred_path}", stacklevel=2)
            continue

        y_true = pred_df["ground_truth"].astype(int).to_numpy()
        y_score = pred_df["model_probs"].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        curves.append({
            "label": label,
            "color": ROC_COLORS[label],
            "auc": auc,
            "fpr": fpr,
            "tpr": tpr,
        })
        if model_key == "covariates_plus_embeddings":
            selected_model_df = pred_df.copy()

    if not curves:
        mock_curves, mock_auc = _mock_panel_a_data()
        return mock_curves, mock_auc, True

    auc_by_cancer = None
    if selected_model_df is not None and "DFCI_MRN" in selected_model_df.columns:
        cohort_path = os.path.join(MARKER_PATH, "matched_cohorts", f"matched_cohort_{cohort}.csv")
        cohort_df = _safe_read_csv(cohort_path)
        if cohort_df is not None:
            cancer_col = "cancer_type" if "cancer_type" in cohort_df.columns else "CANCER_TYPE"
            if cancer_col in cohort_df.columns:
                merged = selected_model_df.merge(
                    cohort_df[["DFCI_MRN", cancer_col]], on="DFCI_MRN", how="left"
                ).dropna(subset=[cancer_col])
                auc_rows = []
                for cancer_type, grp in merged.groupby(cancer_col):
                    if len(grp) < 15 or grp["ground_truth"].nunique() < 2:
                        continue
                    auc_rows.append({
                        "cancer_type": cancer_type,
                        "auc": roc_auc_score(grp["ground_truth"], grp["model_probs"]),
                        "n": len(grp),
                    })
                if auc_rows:
                    auc_by_cancer = (
                        pd.DataFrame(auc_rows)
                        .sort_values(["auc", "n"], ascending=[False, False])
                        .head(5)
                        .reset_index(drop=True)
                    )

    return curves, auc_by_cancer, False


def draw_panel_a(ax: plt.Axes, inset_ax: plt.Axes) -> bool:
    curves, auc_by_cancer, used_mockup = load_panel_a_data()

    ax.plot([0, 1], [0, 1], color="#999", linestyle="--", lw=1.0, zorder=0)
    for curve in curves:
        ax.plot(
            curve["fpr"], curve["tpr"], color=curve["color"], lw=1.8,
            label=f"{curve['label']} (AUC = {curve['auc']:.2f})",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Propensity Score Model: Predicting ICI Receipt")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#ddd")
    if used_mockup:
        ax.text(
            0.03, 0.05, "Illustrative fallback", transform=ax.transAxes,
            fontsize=7, color="#666",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ddd"),
        )

    inset_ax.set_title("AUC by Cancer Type\n(preferred model)", fontsize=7.5)
    if auc_by_cancer is None or auc_by_cancer.empty:
        inset_ax.text(0.5, 0.5, "No per-cancer AUC data",
                      ha="center", va="center", transform=inset_ax.transAxes, fontsize=7)
        inset_ax.set_axis_off()
        return used_mockup

    auc_plot = auc_by_cancer.sort_values("auc", ascending=True)
    colors = plt.cm.RdYlBu(np.linspace(0.15, 0.85, len(auc_plot)))
    inset_ax.barh(auc_plot["cancer_type"], auc_plot["auc"], color=colors, edgecolor="white")
    inset_ax.set_xlim(max(0.45, auc_plot["auc"].min() - 0.05), min(1.0, auc_plot["auc"].max() + 0.05))
    inset_ax.set_xlabel("AUC", fontsize=7)
    inset_ax.tick_params(axis="both", labelsize=6.5)
    for y, (_, row) in enumerate(auc_plot.iterrows()):
        inset_ax.text(row["auc"] + 0.005, y, f"{row['auc']:.2f}", va="center", fontsize=6.5)
    inset_ax.grid(axis="x", alpha=0.35)
    inset_ax.grid(axis="y", visible=False)
    return used_mockup


# ---------------------------------------------------------------------------
# Panel B — biomarker robustness grid
# ---------------------------------------------------------------------------
ROBUSTNESS_SPECS = [
    {"track": 2, "ps_model": "covariates_plus_embeddings", "weight_type": "ATE", "label": "ATE\n(embed PS)"},
    {"track": 2, "ps_model": "covariates_plus_embeddings", "weight_type": "noIPTW", "label": "Unweighted\n(embed PS)"},
    {"track": 1, "ps_model": "covariates_plus_embeddings", "weight_type": "ATE", "label": "ICI-only\n(embed PS)"},
    {"track": 2, "ps_model": "covariates_only", "weight_type": "ATE", "label": "ATE\n(covar PS)"},
    {"track": 2, "ps_model": "covariates_only", "weight_type": "noIPTW", "label": "Unweighted\n(covar PS)"},
    {"track": 1, "ps_model": "covariates_only", "weight_type": "ATE", "label": "ICI-only\n(covar PS)"},
]


def _mock_panel_b_matrix() -> tuple[pd.DataFrame, bool]:
    markers = [
        "KRAS_SNV", "MET_AMP", "BRCA2_DEL", "ALK_FUSION", "STK11_SNV", "PTEN_DEL",
        "EGFR_AMP", "TP53_SNV", "CDKN2A_DEL", "PIK3CA_SNV", "MDM2_AMP", "RB1_DEL",
    ]
    rows = []
    rng = np.random.default_rng(42)
    for marker_idx, marker in enumerate(markers):
        hr = [0.62, 1.45, 0.71, 0.85, 1.27, 0.90, 0.78, 1.05, 0.95, 0.88, 1.15, 0.92][marker_idx]
        for spec_idx, spec in enumerate(ROBUSTNESS_SPECS):
            is_hit = rng.random() < (0.52 - spec_idx * 0.03 + marker_idx * 0.005)
            if is_hit:
                fdr = float(rng.uniform(0.001, 0.08))
                rows.append({
                    "marker": marker,
                    "track": spec["track"],
                    "ps_model": spec["ps_model"],
                    "weight_type": spec["weight_type"],
                    "effect_hr": hr,
                    "fdr": fdr,
                    "support": -np.log10(fdr),
                })
    return pd.DataFrame(rows), True


def load_panel_b_data() -> tuple[pd.DataFrame, bool]:
    t1_path = os.path.join(COMPILED_DIR, "track1_all_significant_hits.csv")
    t2_path = os.path.join(COMPILED_DIR, "track2_all_significant_hits.csv")
    t1 = _safe_read_csv(t1_path)
    t2 = _safe_read_csv(t2_path)

    if t1 is None and t2 is None:
        return _mock_panel_b_matrix()

    frames = []
    if t1 is not None and not t1.empty:
        frames.append(pd.DataFrame({
            "marker": t1["marker"],
            "track": 1,
            "ps_model": t1["ps_model"],
            "weight_type": t1["weight_type"],
            "effect_hr": t1["HR_marker"],
            "fdr": t1.get("FDR_marker", t1.get("p_marker", pd.Series(np.nan, index=t1.index))),
        }))
    if t2 is not None and not t2.empty:
        frames.append(pd.DataFrame({
            "marker": t2["marker"],
            "track": 2,
            "ps_model": t2["ps_model"],
            "weight_type": t2["weight_type"],
            "effect_hr": t2["HR_markerxICI"],
            "fdr": t2.get("FDR_markerxICI", t2.get("p_markerxICI", pd.Series(np.nan, index=t2.index))),
        }))

    if not frames:
        return _mock_panel_b_matrix()

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.dropna(subset=["marker", "effect_hr"])
    if long_df.empty:
        return _mock_panel_b_matrix()

    long_df["fdr"] = pd.to_numeric(long_df["fdr"], errors="coerce").fillna(0.25)
    long_df["support"] = -np.log10(long_df["fdr"].clip(lower=1e-6))
    return long_df, False


def build_robustness_matrix(top_n: int = 12) -> tuple[pd.DataFrame, bool]:
    long_df, used_mockup = load_panel_b_data()
    if long_df.empty:
        return long_df, True

    spec_lookup = {(s["track"], s["ps_model"], s["weight_type"]): s["label"] for s in ROBUSTNESS_SPECS}
    long_df["spec_label"] = long_df.apply(
        lambda row: spec_lookup.get((row["track"], row["ps_model"], row["weight_type"])),
        axis=1,
    )
    long_df = long_df.dropna(subset=["spec_label"])
    if long_df.empty:
        return long_df, True

    marker_rank = (
        long_df.groupby("marker")
        .agg(
            n_specs=("spec_label", "nunique"),
            best_support=("support", "max"),
            median_hr=("effect_hr", "median"),
        )
        .sort_values(["n_specs", "best_support"], ascending=[False, False])
        .head(top_n)
        .reset_index()
    )
    top_markers = marker_rank["marker"].tolist()
    plot_df = long_df[long_df["marker"].isin(top_markers)].copy()
    plot_df["marker"] = pd.Categorical(plot_df["marker"], categories=top_markers[::-1], ordered=True)
    plot_df["spec_label"] = pd.Categorical(
        plot_df["spec_label"], categories=[s["label"] for s in ROBUSTNESS_SPECS], ordered=True
    )
    return plot_df, used_mockup


def draw_panel_b(ax: plt.Axes) -> tuple[pd.DataFrame, bool]:
    plot_df, used_mockup = build_robustness_matrix()

    if plot_df.empty:
        ax.text(0.5, 0.5, "No biomarker robustness data available",
                ha="center", va="center", transform=ax.transAxes)
        return plot_df, True

    markers = list(plot_df["marker"].cat.categories)
    specs = [s["label"] for s in ROBUSTNESS_SPECS]
    x_positions = np.arange(len(specs))
    y_positions = np.arange(len(markers))

    for x in x_positions:
        ax.axvline(x, color="#F2F2F2", lw=0.8, zorder=0)
    for y in y_positions:
        ax.axhline(y, color="#F2F2F2", lw=0.8, zorder=0)

    # draw empty circles first
    for marker_idx, marker in enumerate(markers):
        for spec_idx, spec in enumerate(specs):
            ax.scatter(spec_idx, marker_idx, s=14, facecolors="white",
                       edgecolors=ROBUSTNESS_GRAY, linewidths=0.7, zorder=1)

    for _, row in plot_df.iterrows():
        x = specs.index(row["spec_label"])
        y = markers.index(row["marker"])
        size = 22 + 26 * min(float(row["support"]), 4.0)
        color = ROBUSTNESS_BLUE if row["effect_hr"] < 1 else ROBUSTNESS_RED
        ax.scatter(x, y, s=size, color=color, edgecolors="white", linewidths=0.5, zorder=3)

    hr_lookup = plot_df.groupby("marker")["effect_hr"].median().to_dict()
    for marker_idx, marker in enumerate(markers):
        hr = hr_lookup.get(marker, np.nan)
        if np.isfinite(hr):
            ax.text(len(specs) + 0.45, marker_idx, f"HR={hr:.2f}", va="center", fontsize=7, color="#666")

    ax.set_xlim(-0.55, len(specs) + 1.0)
    ax.set_ylim(-0.6, len(markers) - 0.4)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(specs, fontsize=7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([pretty_marker(m) for m in markers], fontsize=7.5)
    ax.set_title("Biomarker Association Robustness")
    ax.invert_yaxis()
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ROBUSTNESS_BLUE,
               markeredgecolor="white", markersize=6.5, label="Favorable signal (HR < 1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ROBUSTNESS_RED,
               markeredgecolor="white", markersize=6.5, label="Adverse signal (HR > 1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=ROBUSTNESS_GRAY, markersize=5, label="Not significant / unavailable"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=False, fontsize=6.7)

    if used_mockup:
        ax.text(
            0.02, 0.03, "Illustrative fallback", transform=ax.transAxes,
            fontsize=7, color="#666",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ddd"),
        )

    return plot_df, used_mockup


# ---------------------------------------------------------------------------
# Panel C — illustrative survival panels
# ---------------------------------------------------------------------------
def _infer_example_rows(plot_df: pd.DataFrame) -> list[dict]:
    if plot_df.empty:
        return []

    raw = plot_df.copy()
    raw["direction"] = np.where(raw["effect_hr"] < 1, "benefit", "harm")
    track2 = raw[raw["track"] == 2].sort_values("fdr")
    track1 = raw[raw["track"] == 1].sort_values("fdr")

    examples = []
    if not track2[track2["direction"] == "benefit"].empty:
        row = track2[track2["direction"] == "benefit"].iloc[0]
        examples.append({"kind": "predictive_benefit", "row": row})
    if not track1[track1["direction"] == "benefit"].empty:
        row = track1[track1["direction"] == "benefit"].iloc[0]
        examples.append({"kind": "prognostic", "row": row})
    if not track2[track2["direction"] == "harm"].empty:
        row = track2[track2["direction"] == "harm"].iloc[0]
        examples.append({"kind": "predictive_harm", "row": row})

    return examples


def _mock_example_rows() -> list[dict]:
    return [
        {"kind": "predictive_benefit", "row": {"marker": "KRAS_SNV", "effect_hr": 0.62, "fdr": 0.003}},
        {"kind": "prognostic", "row": {"marker": "TMB_HIGH", "effect_hr": 0.72, "fdr": 0.006}},
        {"kind": "predictive_harm", "row": {"marker": "MET_AMP", "effect_hr": 1.45, "fdr": 0.012}},
    ]


def choose_panel_c_examples(plot_df: pd.DataFrame) -> tuple[list[dict], bool]:
    examples = _infer_example_rows(plot_df)
    if len(examples) == 3:
        return examples, False
    return _mock_example_rows(), True


def _exp_survival(t: np.ndarray, rate: float) -> np.ndarray:
    return np.exp(-rate * t)


def _build_survival_groups(kind: str, hr: float) -> list[dict]:
    hr = float(hr) if np.isfinite(hr) else 1.0
    interaction_scale = max(0.45, min(1.8, hr))

    if kind == "predictive_benefit":
        rates = [0.012 * interaction_scale, 0.024, 0.038, 0.070]
        labels = ["Marker+ / ICI+", "Marker- / ICI+", "Marker- / ICI-", "Marker+ / ICI-"]
        order = ["best", "mid_hi", "mid_lo", "worst"]
    elif kind == "predictive_harm":
        rates = [0.070 / interaction_scale, 0.022, 0.038, 0.010]
        labels = ["Marker+ / ICI+", "Marker+ / ICI-", "Marker- / ICI-", "Marker- / ICI+"]
        order = ["worst", "mid_lo", "mid_hi", "best"]
    else:
        favorable = max(0.65, min(1.5, hr))
        rates = [0.014 * favorable, 0.018 * favorable, 0.026, 0.032]
        labels = ["Marker+ / ICI+", "Marker+ / ICI-", "Marker- / ICI+", "Marker- / ICI-"]
        order = ["best", "mid_hi", "mid_lo", "worst"]

    return [
        {"label": label, "rate": rate, "color": KM_COLORS[color_key]}
        for label, rate, color_key in zip(labels, rates, order)
    ]


def _at_risk_counts(rate: float, n0: int, times: list[int]) -> list[int]:
    return [max(1, int(round(n0 * np.exp(-rate * t)))) for t in times]


def draw_survival_mock_panel(ax: plt.Axes, example: dict, show_ylabel: bool = True) -> None:
    row = example["row"]
    marker = pretty_marker(str(row["marker"]))
    hr = float(row.get("effect_hr", 1.0))
    p_val = float(row.get("fdr", np.nan))
    kind = example["kind"]
    groups = _build_survival_groups(kind, hr)
    time = np.linspace(0, 60, 240)
    risk_times = [0, 15, 30, 45, 60]

    if kind == "predictive_benefit":
        title = f"{marker} — ICI Benefit"
        annotation = f"interaction p = {p_val:.3f}" if np.isfinite(p_val) else "interaction p = n/a"
        base_ns = [250, 210, 190, 95]
    elif kind == "predictive_harm":
        title = f"{marker} — ICI Harm"
        annotation = f"interaction p = {p_val:.3f}" if np.isfinite(p_val) else "interaction p = n/a"
        base_ns = [85, 170, 160, 235]
    else:
        title = f"{marker} — Prognostic"
        annotation = f"marker p = {p_val:.3f}" if np.isfinite(p_val) else "marker p = n/a"
        base_ns = [220, 180, 205, 165]

    for group, n0 in zip(groups, base_ns):
        ax.plot(time, _exp_survival(time, group["rate"]), color=group["color"], lw=1.5, label=group["label"])

    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Months")
    if show_ylabel:
        ax.set_ylabel("Survival Probability")
    else:
        ax.set_ylabel("")
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    ax.text(0.03, 0.10, annotation, transform=ax.transAxes, fontsize=6.5, color="#666")
    ax.grid(alpha=0.35)

    # Risk table
    ax.text(-0.04, -0.22, "At risk", transform=ax.transAxes, fontsize=5.5, ha="right", va="top", color="#666")
    for idx, group in enumerate(groups):
        y_pos = -0.30 - 0.07 * idx
        ax.text(-0.04, y_pos, group["label"][:12], transform=ax.transAxes,
                fontsize=5.4, ha="right", va="top", color=group["color"])
        counts = _at_risk_counts(group["rate"], base_ns[idx], risk_times)
        for t, count in zip(risk_times, counts):
            ax.text(t / 60.0, y_pos, str(count), transform=ax.transAxes,
                    fontsize=5.0, ha="center", va="top", color=group["color"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    set_style()

    fig = plt.figure(figsize=(15.5, 11.8))

    ax_a = fig.add_axes([0.06, 0.57, 0.40, 0.36])
    inset_a = fig.add_axes([0.15, 0.62, 0.14, 0.12])
    ax_b = fig.add_axes([0.54, 0.57, 0.40, 0.36])

    ax_c1 = fig.add_axes([0.06, 0.11, 0.26, 0.30])
    ax_c2 = fig.add_axes([0.37, 0.11, 0.26, 0.30])
    ax_c3 = fig.add_axes([0.68, 0.11, 0.26, 0.30])

    used_mockup_a = draw_panel_a(ax_a, inset_a)
    panel_b_df, used_mockup_b = draw_panel_b(ax_b)
    examples, used_mockup_c = choose_panel_c_examples(panel_b_df)
    for idx, (ax, example) in enumerate(zip([ax_c1, ax_c2, ax_c3], examples)):
        draw_survival_mock_panel(ax, example, show_ylabel=(idx == 0))

    panel_labels = {"A": ax_a, "B": ax_b, "C": ax_c1}
    for label, ax in panel_labels.items():
        ax.text(-0.10, 1.04, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="right")

    fallback_bits = []
    if used_mockup_a:
        fallback_bits.append("Panel A")
    if used_mockup_b:
        fallback_bits.append("Panel B")
    if used_mockup_c:
        fallback_bits.append("Panel C")
    if fallback_bits:
        fig.text(
            0.98, 0.975, "Illustrative fallback: " + ", ".join(fallback_bits),
            ha="right", va="top", fontsize=8, color="#666",
        )

    caption = (
        "Figure 5. Biomarker discovery workflow summary. "
        "(A) Propensity-model discrimination for predicting ICI receipt, with inset AUC values by cancer type "
        "for the preferred model when available. "
        "(B) Robustness grid across biomarker-analysis specifications; point size scales with support "
        "(-log10 FDR), blue indicates favorable associations (HR < 1), and red indicates adverse associations (HR > 1). "
        "(C) Three representative biomarker patterns shown as illustrative survival panels: predictive ICI benefit, "
        "prognostic effect, and predictive ICI harm. Panel C is intentionally rendered as a publication-style mockup "
        "rather than a direct Kaplan-Meier export."
    )
    fig.text(
        0.5, 0.015, caption, ha="center", va="bottom", fontsize=7.4, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#ddd", alpha=0.85),
    )

    out_stem = os.path.join(OUTPUT_DIR, "figure5_biomarkers")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure 5 -> {out_stem}.png/.pdf")


if __name__ == "__main__":
    main()
