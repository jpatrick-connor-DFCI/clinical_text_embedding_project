"""Pre-compute inputs for Figure 5 (ICI biomarker discovery).

Writes to FIGURE_DATA_DIR:
- fig5_ps_predictions.csv         DFCI_MRN, ps_model, model_probs, ground_truth,
                                  cancer_type
- fig5_robust_hits.csv            marker, cancer_type, spec, HR_markerxICI,
                                  CI95_marker_ICI_low, CI95_marker_ICI_high,
                                  p_markerxICI, n_specs, mean_HR,
                                  direction_consistent
- fig5_km_top_hit.csv             DFCI_MRN, marker_value, PX_on_ICI, death, tt_death
- fig5_km_examples.csv            example_id, title, marker, cancer, marker_value, PX_on_ICI,
                                  death, tt_death — KMs for HEADLINE_KM_PANELS
- fig5_top_hit_meta.csv           marker, cancer, ps_model
- fig5_love_smd.csv               covariate, smd_unweighted, smd_weighted  (primary-spec balance)
- fig5_forest_headline.csv        gene, cancer, cohort, track, HR, CI95_low/high,
                                  p_value, n_pos, n_specs, validation_level — feeds
                                  the new fig5E synthesis forest panel (T1 prognostic-
                                  in-ICI side-by-side with T2 predictive-of-ICI)
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl

from config import BIOMARKER_PATH, DATA_PATH, FEATURE_PATH
from figures.io import save_figure_data

COHORT = "cohort2"
PS_BUFFER = 30
TRACK2_WEIGHT = "ATE"
PRIMARY_PS_MODEL = "covariates_plus_embeddings"
COMPILED_DIR = os.path.join(BIOMARKER_PATH, "compiled_results/")
PS_BASE = os.path.join(DATA_PATH, f"treatment_prediction/{COHORT}/")
PRIMARY_SPEC_DIR = os.path.join(BIOMARKER_PATH, f"IPTW_runs_{COHORT}_covariates_plus_embeddings/")

PS_PREDICTION_COLUMNS = ["DFCI_MRN", "ps_model", "model_probs", "ground_truth", "cancer_type", "cohort"]
ROBUST_COLUMNS = [
    "marker", "cancer_type", "spec", "HR_markerxICI",
    "CI95_marker_ICI_low", "CI95_marker_ICI_high", "p_markerxICI",
    "cohort", "ps_model", "weight_type", "n_specs", "mean_HR",
    "direction_consistent", "n_significant_markers",
]
KM_COLUMNS = ["DFCI_MRN", "marker_value", "PX_on_ICI", "death", "tt_death"]
KM_EXAMPLE_COLUMNS = [
    "DFCI_MRN", "example_id", "title", "marker", "cancer", "marker_value",
    "PX_on_ICI", "death", "tt_death", "hr", "hr_label", "track",
]
TOP_HIT_META_COLUMNS = ["marker", "cancer", "cohort", "ps_model", "weight_type"]
LOVE_SMD_COLUMNS = ["covariate", "smd_unweighted", "smd_weighted"]
LOVE_TOP_N = 15  # covariates shown beyond the always-kept demographics

# ---------------------------------------------------------------------------
# Headline gene set for the synthesis forest panel (fig5e) + KM panels (fig5d)
#
# Hand-curated from a manual literature-triage of the compiled IPTW hits
# (the one-off triage script that produced this shortlist has been removed;
# recover it from git history at commit 8774b85 if the selection needs revisiting):
#   - MET: only Tier-A gene (T1+T2 robust, both-benefit, Moderate validation)
#   - STK11, KEAP1, ARID1A, CD274, PDCD1LG2: Tier-B T1-robust with Strong+ prior
#     literature support (pipeline replicates known ICI biomarkers)
#   - PTPRD, CSF1R: T2-only predictive signal with Strong+ prior support
# ---------------------------------------------------------------------------
HEADLINE_FOREST = [
    # gene             cancer       cohort     mutation_type  pretty_label
    # Mutation types pulled from the primary-spec hit row in the compiled CSVs.
    # Note: STK11 alterations appear as structural variants (`_SV`) in the primary
    # spec; KEAP1 / CD274 / PDCD1LG2 / PTPRD as deletions; MET/ARID1A/CSF1R as SNV.
    ("MET",      "pan_cancer",  "cohort2",  "SNV",  "MET (SNV)"),
    ("STK11",    "LUNG",        "cohort1",  "SV",   "STK11 (SV)"),
    ("STK11",    "pan_cancer",  "cohort2",  "SV",   "STK11 (SV)"),
    ("KEAP1",    "pan_cancer",  "cohort2",  "DEL",  "KEAP1 (DEL)"),
    ("ARID1A",   "pan_cancer",  "cohort2",  "SNV",  "ARID1A (SNV)"),
    ("CD274",    "pan_cancer",  "cohort2",  "DEL",  "CD274 / PD-L1 (DEL)"),
    ("PDCD1LG2", "pan_cancer",  "cohort2",  "DEL",  "PDCD1LG2 / PD-L2 (DEL)"),
    ("PTPRD",    "BRAIN",       "cohort1",  "DEL",  "PTPRD (DEL)"),
    ("CSF1R",    "pan_cancer",  "cohort2",  "SNV",  "CSF1R (SNV)"),
]
# Subset used as the 3-panel KM strip in fig5D — protagonist + lung-NSCLC
# replication + brain-cancer extension.
HEADLINE_KM_PANELS = [
    ("MET",   "pan_cancer", "cohort2", "SNV"),
    ("STK11", "LUNG",       "cohort1", "SV"),
    ("PTPRD", "BRAIN",      "cohort1", "DEL"),
]

FOREST_COLUMNS = [
    "gene", "cancer", "cohort", "mutation_type", "label", "track",
    "HR", "CI95_low", "CI95_high", "p_value", "n_pos", "n_specs",
    "validation_level", "narrative_role",
]


def _patient_cancer_type() -> pl.DataFrame:
    fp = os.path.join(FEATURE_PATH, "cancer_type_df.csv.gz")
    if not os.path.exists(fp):
        return pl.DataFrame(schema={"DFCI_MRN": pl.Int64, "cancer_type": pl.Utf8})
    cancer = pl.read_csv(fp)
    if "DFCI_MRN" not in cancer.columns:
        return pl.DataFrame(schema={"DFCI_MRN": pl.Int64, "cancer_type": pl.Utf8})
    # Prefer the raw label column (added by generate_all_non_text_covariates.py).
    # Falling back to idxmax over drop_first dummies mislabels the dropped
    # reference category (all-zero rows), so only use it when the raw column is
    # absent (legacy files) — and warn so the result is regenerated.
    if "CANCER_TYPE" in cancer.columns:
        out = cancer.select(["DFCI_MRN", "CANCER_TYPE"]).with_columns(
            pl.col("CANCER_TYPE").cast(pl.Utf8).alias("cancer_type")
        )
        return out.select(["DFCI_MRN", "cancer_type"])
    type_cols = [c for c in cancer.columns if c.startswith("CANCER_TYPE_")]
    if not type_cols:
        return pl.DataFrame(schema={"DFCI_MRN": pl.Int64, "cancer_type": pl.Utf8})
    print("  WARN: cancer_type_df.csv.gz has no raw CANCER_TYPE column; "
          "reference-category patients may be mislabeled. Regenerate covariates.")
    type_matrix = cancer.select(type_cols).to_numpy()
    argmax_idx = np.argmax(type_matrix, axis=1)
    cancer_type_vals = [type_cols[i].replace("CANCER_TYPE_", "", 1) for i in argmax_idx]
    out = cancer.select(["DFCI_MRN"]).with_columns(
        pl.Series("cancer_type", cancer_type_vals, dtype=pl.Utf8)
    )
    return out.select(["DFCI_MRN", "cancer_type"])


def _ps_predictions() -> pl.DataFrame:
    cancer = _patient_cancer_type()
    rows = []
    for ps_model in ("covariates_only", "covariates_plus_embeddings"):
        fp = os.path.join(PS_BASE, f"{ps_model}_propensity/w_{PS_BUFFER}_day_buffer/predictions.csv.gz")
        if not os.path.exists(fp):
            print(f"  missing {fp}")
            continue
        d = pl.read_csv(fp)
        required = {"DFCI_MRN", "model_probs", "ground_truth"}
        if not required.issubset(d.columns):
            print(f"  skipping {fp}: missing {sorted(required - set(d.columns))}")
            continue
        d = d.select(["DFCI_MRN", "model_probs", "ground_truth"])
        d = d.with_columns([
            pl.lit(ps_model).alias("ps_model"),
            pl.lit(COHORT).alias("cohort"),
        ])
        if not cancer.is_empty():
            d = d.join(cancer, on="DFCI_MRN", how="left")
        else:
            d = d.with_columns(pl.lit(None, dtype=pl.Utf8).alias("cancer_type"))
        missing_cols = [c for c in PS_PREDICTION_COLUMNS if c not in d.columns]
        if missing_cols:
            d = d.with_columns([pl.lit(None).alias(c) for c in missing_cols])
        rows.append(d.select(PS_PREDICTION_COLUMNS))
    if not rows:
        return pl.DataFrame(schema={c: pl.Float64 for c in PS_PREDICTION_COLUMNS})
    return pl.concat(rows, how="diagonal_relaxed")


def _robust_hits() -> pl.DataFrame:
    fp = os.path.join(COMPILED_DIR, "track2_all_significant_hits.csv")
    if not os.path.exists(fp):
        print(f"  missing {fp}")
        return pl.DataFrame(schema={c: pl.Float64 for c in ROBUST_COLUMNS})
    try:
        hits = pl.read_csv(fp)
    except pl.exceptions.NoDataError:
        return pl.DataFrame(schema={c: pl.Float64 for c in ROBUST_COLUMNS})
    if hits.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in ROBUST_COLUMNS})
    hits = hits.with_columns(
        (pl.col("cohort").cast(pl.Utf8) + "|" + pl.col("ps_model").cast(pl.Utf8)
         + "|" + pl.col("weight_type").cast(pl.Utf8)).alias("spec")
    )

    # Geometric mean for HR aggregation: HRs are multiplicative, so the
    # arithmetic mean over-weights large HRs. Drop pathological HRs (<=0, inf,
    # NaN, or already-flagged extremes) before averaging so a single model-
    # separation row can't dominate the geometric mean.
    EXTREME = 50.0

    def _is_clean(arr: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return (arr > 0) & np.isfinite(arr) & (arr >= 1 / EXTREME) & (arr <= EXTREME)

    def _clean_geomean(arr: np.ndarray) -> float:
        clean = arr[_is_clean(arr)]
        if clean.size == 0:
            return float("nan")
        return float(np.exp(np.log(clean).mean()))

    def _clean_direction(arr: np.ndarray) -> bool:
        clean = arr[_is_clean(arr)]
        if clean.size == 0:
            return False
        return bool((clean > 1).all() or (clean < 1).all())

    def _n_clean_specs(arr: np.ndarray) -> int:
        # Count only specs whose HR survives the cleaning filter, so n_specs
        # matches what actually contributed to mean_HR / direction_consistent.
        return int(_is_clean(arr).sum())

    hr_vals = hits["HR_markerxICI"].cast(pl.Float64, strict=False).to_numpy()
    hits = hits.with_columns(pl.Series("_hr_clean", hr_vals))
    grp_rows = []
    for (marker, cancer_type), sub in hits.group_by(["marker", "cancer_type"], maintain_order=True):
        arr = sub["_hr_clean"].to_numpy()
        grp_rows.append({
            "marker": marker, "cancer_type": cancer_type,
            "n_specs": _n_clean_specs(arr), "mean_HR": _clean_geomean(arr),
            "direction_consistent": _clean_direction(arr),
        })
    grp = pl.DataFrame(grp_rows)
    robust = grp.filter((pl.col("n_specs") >= 2) & pl.col("direction_consistent"))
    cols = ["marker", "cancer_type", "spec", "HR_markerxICI",
            "CI95_marker_ICI_low", "CI95_marker_ICI_high", "p_markerxICI"]
    cols = [c for c in cols if c in hits.columns]
    out = hits.join(robust.select(["marker", "cancer_type"]), on=["marker", "cancer_type"], how="inner")
    out = out.select(cols + ["cohort", "ps_model", "weight_type"])
    out = out.join(robust, on=["marker", "cancer_type"], how="left")
    # Denominator for the survivorship-aware panel B annotation: markers significant
    # in >=1 spec (the universe these robust hits were filtered from).
    n_sig = hits.select(["marker", "cancer_type"]).unique().height
    out = out.with_columns(pl.lit(n_sig).alias("n_significant_markers"))
    missing_cols = [c for c in ROBUST_COLUMNS if c not in out.columns]
    if missing_cols:
        out = out.with_columns([pl.lit(None).alias(c) for c in missing_cols])
    return out.select(ROBUST_COLUMNS)


def _km_top_hit() -> tuple[pl.DataFrame, pl.DataFrame]:
    t2_fp = os.path.join(COMPILED_DIR, "track2_all_significant_hits.csv")
    if not os.path.exists(t2_fp):
        return (pl.DataFrame(schema={c: pl.Float64 for c in KM_COLUMNS}),
                pl.DataFrame(schema={c: pl.Float64 for c in TOP_HIT_META_COLUMNS}))
    try:
        hits = pl.read_csv(t2_fp)
    except pl.exceptions.NoDataError:
        return (pl.DataFrame(schema={c: pl.Float64 for c in KM_COLUMNS}),
                pl.DataFrame(schema={c: pl.Float64 for c in TOP_HIT_META_COLUMNS}))
    if hits.is_empty():
        return (pl.DataFrame(schema={c: pl.Float64 for c in KM_COLUMNS}),
                pl.DataFrame(schema={c: pl.Float64 for c in TOP_HIT_META_COLUMNS}))
    # Restrict "top hit" to the primary spec so panel D shows what the methods
    # section claims is the primary analysis, not the most-extreme p-value
    # across all sensitivity specifications.
    primary = hits.filter(
        (pl.col("ps_model") == PRIMARY_PS_MODEL)
        & (pl.col("weight_type").str.to_uppercase() == TRACK2_WEIGHT.upper())
        & (pl.col("cohort") == COHORT)
    )
    if primary.is_empty():
        print(f"  no hits in primary spec ({COHORT}, {PRIMARY_PS_MODEL}, {TRACK2_WEIGHT}); "
              "falling back to overall best p")
        primary = hits
    top = primary.sort("p_markerxICI").row(0, named=True)
    marker = top["marker"]
    cancer = top["cancer_type"]
    cohort = top["cohort"]
    ps_model = top["ps_model"]
    weight_type = top.get("weight_type", TRACK2_WEIGHT)
    print(f"  top hit: {marker} in {cancer} ({cohort}, {ps_model}, {weight_type})")
    iptw_fp = os.path.join(BIOMARKER_PATH, f"IPTW_df_{cohort}_{ps_model}.csv.gz")
    if not os.path.exists(iptw_fp):
        print(f"  missing {iptw_fp}")
        return (pl.DataFrame(schema={c: pl.Float64 for c in KM_COLUMNS}),
                pl.DataFrame(schema={c: pl.Float64 for c in TOP_HIT_META_COLUMNS}))
    iptw = pl.read_csv(iptw_fp)
    cancer_col = f"CANCER_TYPE_{cancer}"
    if cancer != "pan_cancer" and cancer_col in iptw.columns:
        iptw = iptw.filter(pl.col(cancer_col) == 1)
    iptw = iptw.drop_nulls(subset=[marker, "PX_on_ICI", "tt_death", "death"])
    iptw = iptw.filter(pl.col("tt_death") > 0)
    km = iptw.select(["DFCI_MRN", marker, "PX_on_ICI", "death", "tt_death"]).rename(
        {marker: "marker_value"}
    )
    meta = pl.DataFrame([{
        "marker": marker, "cancer": cancer, "cohort": cohort, "ps_model": ps_model,
        "weight_type": weight_type,
    }])
    return km, meta


def _select_km_example_hits(t1_hits: pl.DataFrame, t2_hits: pl.DataFrame,
                             max_examples: int = 3) -> pl.DataFrame:
    """Return one row per (marker, cancer, cohort) in HEADLINE_KM_PANELS, in order.

    Tries Track 2 first (so the KM title can carry the interaction HR), falls
    back to Track 1 if the gene has no T2 interaction hit but is significant
    in T1. The output is normalized to a track-agnostic schema:

        marker, cancer_type, cohort, ps_model, weight_type, track, hr, hr_label

    so `_km_examples()` can iterate uniformly.
    """
    chosen = []

    def _best(sub: pl.DataFrame, p_col: str) -> dict | None:
        primary = sub.filter((pl.col("ps_model") == PRIMARY_PS_MODEL)
                              & (pl.col("weight_type").str.to_uppercase() == TRACK2_WEIGHT.upper()))
        if primary.is_empty():
            primary = sub.sort(p_col)
        return primary.row(0, named=True) if not primary.is_empty() else None

    for gene, cancer, cohort, mut in HEADLINE_KM_PANELS:
        marker = f"{gene}_{mut}"
        # Try Track 2 first (interaction HR is the headline number for fig5D)
        t2_sub = t2_hits.filter((pl.col("marker") == marker)
                                 & (pl.col("cancer_type") == cancer)
                                 & (pl.col("cohort") == cohort))
        t2_row = _best(t2_sub, "p_markerxICI") if not t2_sub.is_empty() else None
        if t2_row is not None:
            chosen.append({
                "marker":     t2_row["marker"],
                "cancer_type": t2_row["cancer_type"],
                "cohort":     t2_row["cohort"],
                "ps_model":   t2_row["ps_model"],
                "weight_type": t2_row["weight_type"],
                "track":      2,
                "hr":         float(t2_row["HR_markerxICI"]),
                "hr_label":   "HR(marker×ICI)",
            })
            continue

        # Fall back to Track 1 (prognostic-within-ICI HR)
        t1_sub = t1_hits.filter((pl.col("marker") == marker)
                                 & (pl.col("cancer_type") == cancer)
                                 & (pl.col("cohort") == cohort))
        t1_row = _best(t1_sub, "p_marker") if not t1_sub.is_empty() else None
        if t1_row is not None:
            chosen.append({
                "marker":     t1_row["marker"],
                "cancer_type": t1_row["cancer_type"],
                "cohort":     t1_row["cohort"],
                "ps_model":   t1_row["ps_model"],
                "weight_type": t1_row["weight_type"],
                "track":      1,
                "hr":         float(t1_row["HR_marker"]),
                "hr_label":   "HR(marker | ICI)",
            })
            continue

        print(f"  KM headline {marker}/{cancer}/{cohort}: no compiled hit; skipping")

    if not chosen:
        return pl.DataFrame(schema={c: pl.Float64 for c in
                                     ["marker","cancer_type","cohort","ps_model",
                                      "weight_type","track","hr","hr_label"]})
    return pl.DataFrame(chosen)[:max_examples]


# ---------------------------------------------------------------------------
# Forest panel (fig5e): T1 (prognostic-in-ICI) vs T2 (predictive-of-ICI) for
# the headline gene set, with literature-validation strip.
# ---------------------------------------------------------------------------
_VALIDATION_FILENAMES = (
    "all_findings_with_validation.csv.gz",
    "all_findings_with_validation.csv",
)


def _load_validation_lookup() -> dict[tuple[str, str, str], tuple[str, str]]:
    """{(gene, cancer, cohort) → (validation_level, validation_notes)}.

    Tries .gz then plain .csv in COMPILED_DIR. Returns empty dict if absent.
    """
    for name in _VALIDATION_FILENAMES:
        fp = os.path.join(COMPILED_DIR, name)
        if os.path.exists(fp):
            try:
                d = pl.read_csv(fp)
            except (pl.exceptions.NoDataError, OSError) as e:
                print(f"  validation CSV present but unreadable ({e}); no annotation")
                return {}
            cols = {"gene", "cancer_type", "cohort", "validation_level"}
            if not cols.issubset(d.columns):
                return {}
            d = d.drop_nulls(subset=["validation_level"])
            d = d.unique(subset=["gene", "cancer_type", "cohort"], keep="first")
            return {
                (r["gene"], r["cancer_type"], r["cohort"]):
                    (r["validation_level"], r.get("validation_notes", ""))
                for r in d.iter_rows(named=True)
            }
    return {}


def _narrative_role(t1_robust: bool, t2_robust: bool, validation: str) -> str:
    strong = validation in ("Strong", "Very Strong", "Moderate")
    if t1_robust and t2_robust:
        return "Replication of known biology" if strong else "Novel discovery"
    if t1_robust or t2_robust:
        return "Supports known biology" if strong else "Hypothesis-generating"
    return "Lower-confidence"


def _track1_row(t1: pl.DataFrame, marker: str, cancer: str, cohort: str) -> dict | None:
    """Primary-spec Track-1 effect (HR + CI95 + p + n_pos + n_specs)."""
    sub = t1.filter((pl.col("marker") == marker) & (pl.col("cancer_type") == cancer)
                     & (pl.col("cohort") == cohort))
    if sub.is_empty():
        return None
    primary = sub.filter((pl.col("ps_model") == PRIMARY_PS_MODEL)
                          & (pl.col("weight_type").str.to_uppercase() == "ATE"))
    if primary.is_empty():
        primary = sub.sort("p_marker").head(1)
    r = primary.row(0, named=True)
    return {
        "HR":       float(r["HR_marker"]),
        "CI95_low": float(r["CI95_marker_low"]),
        "CI95_high": float(r["CI95_marker_high"]),
        "p_value":  float(r["p_marker"]),
        "n_pos":    int(r["n_marker_pos"]),
        "n_specs":  int(sub.height),
    }


def _track2_row(t2: pl.DataFrame, marker: str, cancer: str, cohort: str) -> dict | None:
    """Primary-spec Track-2 interaction estimate, with the real Wald CI95.

    `CI95_markerxICI_low`/`_high` are the Wald interval for the interaction term
    itself (run_IPTW_analysis._fit_track2_marker: se_mx = sqrt(V[mx, mx])), not
    the inter-spec min/max range this used to plot.
    """
    sub = t2.filter((pl.col("marker") == marker) & (pl.col("cancer_type") == cancer)
                     & (pl.col("cohort") == cohort))
    if sub.is_empty():
        return None
    primary = sub.filter((pl.col("ps_model") == PRIMARY_PS_MODEL)
                          & (pl.col("weight_type").str.to_uppercase() == TRACK2_WEIGHT.upper()))
    if primary.is_empty():
        primary = sub.sort("p_markerxICI").head(1)
    r = primary.row(0, named=True)
    return {
        "HR":       float(r["HR_markerxICI"]),
        "CI95_low": float(r["CI95_markerxICI_low"]),
        "CI95_high": float(r["CI95_markerxICI_high"]),
        "p_value":  float(r["p_markerxICI"]),
        "n_pos":    int(r.get("n_ICI_pos", 0) or 0),
        "n_specs":  int(sub.height),
    }


def _forest_headline() -> pl.DataFrame:
    t1_fp = os.path.join(COMPILED_DIR, "track1_all_significant_hits.csv")
    t2_fp = os.path.join(COMPILED_DIR, "track2_all_significant_hits.csv")
    if not (os.path.exists(t1_fp) and os.path.exists(t2_fp)):
        print(f"  missing T1 or T2 compiled hits at {COMPILED_DIR}; forest empty")
        return pl.DataFrame(schema={c: pl.Float64 for c in FOREST_COLUMNS})
    t1 = pl.read_csv(t1_fp)
    t2 = pl.read_csv(t2_fp)
    validation = _load_validation_lookup()

    rows = []
    for gene, cancer, cohort, mut, label in HEADLINE_FOREST:
        marker = f"{gene}_{mut}"
        t1_row = _track1_row(t1, marker, cancer, cohort)
        t2_row = _track2_row(t2, marker, cancer, cohort)
        val_level, _ = validation.get((gene, cancer, cohort), ("", ""))
        t1_robust = t1_row is not None and t1_row["n_specs"] >= 2
        t2_robust = t2_row is not None and t2_row["n_specs"] >= 2
        role = _narrative_role(t1_robust, t2_robust, val_level)
        base = dict(gene=gene, cancer=cancer, cohort=cohort,
                    mutation_type=mut, label=label,
                    validation_level=val_level, narrative_role=role)
        if t1_row is not None:
            rows.append({**base, "track": "T1_prognostic_in_ICI", **t1_row})
        if t2_row is not None:
            rows.append({**base, "track": "T2_predictive_of_ICI", **t2_row})
    if not rows:
        return pl.DataFrame(schema={c: pl.Float64 for c in FOREST_COLUMNS})
    return pl.DataFrame(rows).select(FOREST_COLUMNS)


def _km_examples() -> pl.DataFrame:
    t1_fp = os.path.join(COMPILED_DIR, "track1_all_significant_hits.csv")
    t2_fp = os.path.join(COMPILED_DIR, "track2_all_significant_hits.csv")
    if not os.path.exists(t2_fp):
        print(f"  missing {t2_fp}; cannot build KM examples")
        return pl.DataFrame(schema={c: pl.Float64 for c in KM_EXAMPLE_COLUMNS})
    try:
        t2_hits = pl.read_csv(t2_fp)
    except pl.exceptions.NoDataError:
        t2_hits = pl.DataFrame()
    try:
        t1_hits = pl.read_csv(t1_fp) if os.path.exists(t1_fp) else pl.DataFrame()
    except pl.exceptions.NoDataError:
        t1_hits = pl.DataFrame()
    if t2_hits.is_empty() and t1_hits.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in KM_EXAMPLE_COLUMNS})

    examples = _select_km_example_hits(t1_hits, t2_hits)
    if examples.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in KM_EXAMPLE_COLUMNS})

    frames = []
    for i, row in enumerate(examples.iter_rows(named=True), start=1):
        marker = row["marker"]
        cancer = row["cancer_type"]
        cohort = row["cohort"]
        ps_model = row["ps_model"]
        hr = float(row["hr"])
        hr_label = row["hr_label"]
        track = int(row["track"])
        iptw_fp = os.path.join(BIOMARKER_PATH, f"IPTW_df_{cohort}_{ps_model}.csv.gz")
        if not os.path.exists(iptw_fp):
            print(f"  missing {iptw_fp}")
            continue
        iptw = pl.read_csv(iptw_fp)
        cancer_col = f"CANCER_TYPE_{cancer}"
        if cancer != "pan_cancer" and cancer_col in iptw.columns:
            iptw = iptw.filter(pl.col(cancer_col) == 1)
        required = [marker, "PX_on_ICI", "tt_death", "death"]
        if not set(required).issubset(iptw.columns):
            print(f"  skipping KM example {marker}/{cancer}: missing required columns")
            continue
        cur = iptw.drop_nulls(subset=required)
        cur = cur.filter(pl.col("tt_death") > 0)
        if cur.is_empty():
            continue
        # Track 2 (interaction) → ICI Benefit / ICI Harm; Track 1 fallback is
        # a prognostic-within-ICI story, so title differently.
        if track == 2:
            direction = "ICI Benefit" if hr < 1 else "ICI Harm"
            title = f"{marker} — {direction}"
        else:
            direction = "Prognostic (lower risk)" if hr < 1 else "Prognostic (higher risk)"
            title = f"{marker} — {direction} within ICI"
        cur = cur.select(["DFCI_MRN", marker, "PX_on_ICI", "death", "tt_death"]).rename(
            {marker: "marker_value"}
        )
        cur = cur.with_columns([
            pl.lit(i).alias("example_id"),
            pl.lit(title).alias("title"),
            pl.lit(marker).alias("marker"),
            pl.lit(cancer).alias("cancer"),
            pl.lit(hr).alias("hr"),
            pl.lit(hr_label).alias("hr_label"),
            pl.lit(track).alias("track"),
        ])
        frames.append(cur)
    if not frames:
        return pl.DataFrame(schema={c: pl.Float64 for c in KM_EXAMPLE_COLUMNS})
    out = pl.concat(frames, how="diagonal_relaxed")
    # Preserve forward-compat schema (Track 1 rows carry extra columns)
    return out


def _smd(x: np.ndarray, treat: np.ndarray, w: np.ndarray | None = None) -> float:
    """Standardized mean difference between treated/control (optionally IPTW-weighted)."""
    t, c = treat == 1, treat == 0
    if x[t].size == 0 or x[c].size == 0:
        return np.nan
    if w is None:
        mt, mc = x[t].mean(), x[c].mean()
        vt, vc = x[t].var(ddof=1), x[c].var(ddof=1)
    else:
        wt, wc = w[t], w[c]
        mt, mc = np.average(x[t], weights=wt), np.average(x[c], weights=wc)
        vt = np.average((x[t] - mt) ** 2, weights=wt)
        vc = np.average((x[c] - mc) ** 2, weights=wc)
    denom = np.sqrt((vt + vc) / 2)
    return float((mt - mc) / denom) if denom > 0 else 0.0


def _love_smd() -> pl.DataFrame:
    """Covariate balance (SMD) before vs after IPTW for the primary spec, for the love plot.

    Recomputes stabilized ATE weights from the held-out propensity (ICI_prediction) so the panel
    is self-contained, mirroring run_IPTW_analysis.compute_smd + the ATE-weight formula.
    """
    fp = os.path.join(BIOMARKER_PATH, f"IPTW_df_{COHORT}_{PRIMARY_PS_MODEL}.csv.gz")
    if not os.path.exists(fp):
        print(f"  no IPTW df at {fp}; emitting empty love-plot data")
        return pl.DataFrame(schema={c: pl.Float64 for c in LOVE_SMD_COLUMNS})
    df = pl.read_csv(fp)
    if not {"PX_on_ICI", "ICI_prediction"}.issubset(df.columns):
        print("  IPTW df missing PX_on_ICI/ICI_prediction; emitting empty love-plot data")
        return pl.DataFrame(schema={c: pl.Float64 for c in LOVE_SMD_COLUMNS})

    # Rows without a propensity/treatment value can't be weighted — drop them so the
    # weighted SMDs don't collapse to NaN.
    df = df.drop_nulls(subset=["PX_on_ICI", "ICI_prediction"])
    if df.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in LOVE_SMD_COLUMNS})

    base = [c for c in ("GENDER", "AGE_AT_TREATMENTSTART") if c in df.columns]
    prefixed = [c for c in df.columns
                if c.startswith(("LINE_", "CANCER_TYPE_", "PANEL_VERSION_"))]
    covars = base + prefixed
    if not covars:
        return pl.DataFrame(schema={c: pl.Float64 for c in LOVE_SMD_COLUMNS})

    treat = df["PX_on_ICI"].to_numpy()
    ps = df["ICI_prediction"].clip(1e-6, 1 - 1e-6).to_numpy()
    p_treat = treat.mean()
    w = np.where(treat == 1, p_treat / ps, (1 - p_treat) / (1 - ps))
    lo, hi = np.percentile(w, [1, 99])
    w = np.clip(w, lo, hi)

    rows = []
    for cov in covars:
        x = df[cov].cast(pl.Float64, strict=False).to_numpy()
        if np.isnan(x).any():
            continue
        rows.append({
            "covariate": cov,
            "smd_unweighted": _smd(x, treat),
            "smd_weighted": _smd(x, treat, w),
        })
    if not rows:
        return pl.DataFrame(schema={c: pl.Float64 for c in LOVE_SMD_COLUMNS})
    out = pl.DataFrame(rows).select(LOVE_SMD_COLUMNS)
    # Keep demographics + the most-imbalanced covariates for a readable love plot
    out = out.with_columns(pl.col("smd_unweighted").abs().alias("_abs"))
    keep_base = out.filter(pl.col("covariate").is_in(base))
    keep_top = out.filter(~pl.col("covariate").is_in(base)).sort("_abs", descending=True).head(LOVE_TOP_N)
    return pl.concat([keep_base, keep_top], how="diagonal_relaxed").drop("_abs")


def main() -> None:
    save_figure_data(_ps_predictions(), "fig5_ps_predictions.csv")
    save_figure_data(_robust_hits(), "fig5_robust_hits.csv")
    km, meta = _km_top_hit()
    save_figure_data(km, "fig5_km_top_hit.csv")
    save_figure_data(_km_examples(), "fig5_km_examples.csv")
    save_figure_data(meta, "fig5_top_hit_meta.csv")
    save_figure_data(_love_smd(), "fig5_love_smd.csv")
    save_figure_data(_forest_headline(), "fig5_forest_headline.csv")


if __name__ == "__main__":
    main()
