"""Pre-compute inputs for Figure 1 (cohort & embedding overview).

Writes to FIGURE_DATA_DIR (defined in _figure_utils.py):
- fig1_cohort_counts.csv          step → n
- fig1_note_volume.csv            year_bin × note_type → count
- fig1_cancer_type_counts.csv     category, n
- fig1_stage_counts.csv           category, n
- fig1_treatment_counts.csv       category, n
- fig1_umap_coords.csv            DFCI_MRN, x, y, method, cancer_type
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sibling _figure_utils importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _figure_utils import (
    EMBEDDING_FILES, FEATURE_PATH, NOTES_PATH, SURV_PATH,
    save_figure_data,
)
from slurm_array_utils import _get_common_feature_mrns


SCHEME_FOR_EMBED = "icd3_post"  # widest cohort


def _cohort_counts(emb_df: pd.DataFrame, cancer_type_df: pd.DataFrame) -> pd.DataFrame:
    with_notes = emb_df["DFCI_MRN"].nunique()
    with_cancer = len(set(emb_df["DFCI_MRN"]) & set(cancer_type_df["DFCI_MRN"]))
    common = _get_common_feature_mrns()
    with_all = len(set(emb_df["DFCI_MRN"]) & common)
    return pd.DataFrame([
        {"step": "with_notes", "n": with_notes},
        {"step": "with_cancer_type", "n": with_cancer},
        {"step": "full_cohort_analysis_set", "n": with_cancer},
        {"step": "modality_comparison_set", "n": with_all},
    ])


def _note_volume(notes_meta: pd.DataFrame) -> pd.DataFrame:
    year_col = next((c for c in (
        "YEAR_RELATIVE_TO_FIRST_TREATMENT", "years_since_first_treatment",
        "days_since_first_treatment", "NOTE_DT_OFFSET_DAYS",
    ) if c in notes_meta.columns), None)
    type_col = next((c for c in ("NOTE_TYPE", "note_type", "NOTE_KIND")
                     if c in notes_meta.columns), None)
    if year_col is None or type_col is None:
        raise RuntimeError(
            f"Could not find year/type columns in notes_meta; have: {list(notes_meta.columns)[:25]}")
    nm = notes_meta[[year_col, type_col]].copy()
    nm["_years"] = nm[year_col] / 365.25 if "days" in year_col.lower() else nm[year_col]
    nm = nm[(nm["_years"] >= -5) & (nm["_years"] <= 0)].copy()
    nm["year_bin"] = nm["_years"].round().astype(int)
    out = (nm.groupby(["year_bin", type_col]).size()
             .reset_index(name="n_notes")
             .rename(columns={type_col: "note_type"}))
    return out


def _composition_counts(df: pd.DataFrame, cols: list[str], prefix: str, top_n: int) -> pd.DataFrame:
    s = df[cols].sum().sort_values(ascending=False)
    s.index = s.index.str.replace(prefix, "", regex=False)
    s = s.head(top_n)
    return pd.DataFrame({"category": s.index.astype(str), "n": s.values.astype(int)})


def _umap_coords(emb_df: pd.DataFrame, cancer_type_df: pd.DataFrame,
                 type_cols: list[str], embed_cols: list[str], n_sample: int = 15000) -> pd.DataFrame:
    merged = emb_df.merge(cancer_type_df[["DFCI_MRN"] + type_cols], on="DFCI_MRN", how="inner")
    if len(merged) == 0:
        print("  no overlap between embedding cohort and cancer-type cohort; "
              "emitting empty UMAP coords")
        return pd.DataFrame(columns=["DFCI_MRN", "x", "y", "method", "cancer_type"])

    X = merged[embed_cols].values
    type_arg = merged[type_cols].values.argmax(axis=1)
    type_labels = np.array([type_cols[i].replace("CANCER_TYPE_", "") for i in type_arg])

    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    X_s = X[idx]
    labels_s = type_labels[idx]
    mrns_s = merged["DFCI_MRN"].values[idx]

    try:
        import umap
        coords = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(X_s)
        method = "UMAP"
    except Exception as e:
        print(f"  UMAP unavailable ({e}); falling back to PCA")
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=0).fit_transform(X_s)
        method = "PCA"
    return pd.DataFrame({
        "DFCI_MRN": mrns_s,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "method": method,
        "cancer_type": labels_s,
    })


def main() -> None:
    emb_df = pd.read_csv(os.path.join(SURV_PATH, EMBEDDING_FILES[SCHEME_FOR_EMBED]))
    cancer_type_df = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_type_df.csv.gz"))
    stage_df = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_stage_df.csv.gz"))
    treatment_df = pd.read_csv(os.path.join(FEATURE_PATH, "categorical_treatment_data_by_line.csv.gz"))
    notes_meta = pd.read_csv(os.path.join(NOTES_PATH, "full_VTE_embeddings_metadata.csv.gz"))

    embed_cols = [c for c in emb_df.columns if "EMBEDDING" in c]
    type_cols = [c for c in cancer_type_df.columns if c.startswith("CANCER_TYPE_")]
    stage_cols = [c for c in stage_df.columns if c.startswith("CANCER_STAGE_")]
    tx_cols = [c for c in treatment_df.columns if c.startswith("PX_on_")]
    tx1 = treatment_df.loc[treatment_df["treatment_line"] == 1]

    save_figure_data(_cohort_counts(emb_df, cancer_type_df), "fig1_cohort_counts.csv")
    save_figure_data(_note_volume(notes_meta), "fig1_note_volume.csv")
    save_figure_data(_composition_counts(cancer_type_df, type_cols, "CANCER_TYPE_", 15),
                     "fig1_cancer_type_counts.csv")
    save_figure_data(_composition_counts(stage_df, stage_cols, "CANCER_STAGE_", 15),
                     "fig1_stage_counts.csv")
    save_figure_data(_composition_counts(tx1, tx_cols, "PX_on_", 15),
                     "fig1_treatment_counts.csv")
    save_figure_data(_umap_coords(emb_df, cancer_type_df, type_cols, embed_cols),
                     "fig1_umap_coords.csv")


if __name__ == "__main__":
    main()
