"""Pre-compute inputs for Figure 1 (cohort & embedding overview).

Writes to FIGURE_DATA_DIR (defined in _figure_utils.py):
- fig1_cohort_counts.csv          step → n
- fig1_endpoint_counts.csv        scheme → n_endpoints
- fig1_note_volume.csv            year_bin × note_type → count
- fig1_cancer_type_counts.csv     category, n
- fig1_stage_counts.csv           category, n
- fig1_treatment_counts.csv       category, n
- fig1_notes_per_patient.csv      DFCI_MRN, note_type, n_notes  (per patient x note type)
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sibling _figure_utils importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _figure_utils import (
    EMBEDDING_FILES, FEATURE_PATH, NOTES_PATH, SURV_PATH,
    SCHEME_RESULT_DIRS, list_trained_events,
    save_figure_data,
)
from slurm_array_utils import _get_common_feature_mrns


SCHEME_FOR_EMBED = "icd3_post"  # widest cohort
ENDPOINT_COUNT_COLUMNS = ["scheme", "n_endpoints"]

# Raw, complete stage labels — `cancer_stage_df.csv.gz` is produced with
# pd.get_dummies(drop_first=True) (generate_all_non_text_covariates.py:39) which
# silently drops Stage I from the one-hot. Read the pickle directly to recover it.
# Same pattern as prep_figure_2._major_stage_labels().
STAGE_PATH = (
    "/data/gusev/PROFILE/CLINICAL/OncDRS/DERIVED_FROM_CLINICAL_TEXTS_2024_03/"
    "derived_files/cancer_stage/dfci_cancer_mrn_to_derived_cancer_stage.pkl"
)
STAGE_ORDER = ["I", "II", "III", "IV"]
_STAGE_TOKEN = re.compile(r"^(IV|III|II|I|4|3|2|1)[A-D]?$")


def _normalize_stage(raw) -> str | None:
    if pd.isna(raw):
        return None
    s = str(raw).upper().strip().replace("STAGE", "").strip()
    s = re.sub(r"\.0+$", "", s)
    m = _STAGE_TOKEN.match(s)
    if not m:
        return None
    arabic_to_roman = {"1": "I", "2": "II", "3": "III", "4": "IV"}
    token = m.group(1)
    return arabic_to_roman.get(token, token)


def _stage_counts_from_pickle(cohort_mrns: set[int]) -> pd.DataFrame:
    """Major-stage breakdown (I/II/III/IV) for the analysis cohort."""
    try:
        with open(STAGE_PATH, "rb") as f:
            mrn_to_stage = pickle.load(f)
    except (FileNotFoundError, OSError, pickle.UnpicklingError) as e:
        print(f"  stage pickle unavailable ({type(e).__name__}); falling back to one-hot CSV")
        oh = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_stage_df.csv.gz"))
        oh = oh[oh["DFCI_MRN"].isin(cohort_mrns)]
        stage_cols = [c for c in oh.columns if c.startswith("CANCER_STAGE_")]
        present = [_normalize_stage(c.replace("CANCER_STAGE_", "")) for c in stage_cols]
        reference = next((s for s in STAGE_ORDER if s not in present), None)
        active = oh[stage_cols].to_numpy()
        labels = [reference if row.max() <= 0 else present[int(row.argmax())]
                  for row in active]
    else:
        labels = [_normalize_stage(v) for k, v in mrn_to_stage.items() if k in cohort_mrns]
    s = pd.Series([x for x in labels if x in STAGE_ORDER]).value_counts()
    s = s.reindex(STAGE_ORDER, fill_value=0)
    return pd.DataFrame({"category": s.index.astype(str), "n": s.values.astype(int)})


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
    # (column_name, unit) — first matching column wins; unit declared explicitly
    # because not all column names carry "days"/"years" in their text.
    TIME_CANDIDATES = [
        ("NOTE_TIME_REL_FIRST_TREATMENT_START", "days"),  # written by knit_longformer_embeddings.py
        ("YEAR_RELATIVE_TO_FIRST_TREATMENT", "years"),
        ("years_since_first_treatment", "years"),
        ("days_since_first_treatment", "days"),
        ("NOTE_DT_OFFSET_DAYS", "days"),
    ]
    time_col, unit = next(((c, u) for c, u in TIME_CANDIDATES if c in notes_meta.columns),
                          (None, None))
    type_col = next((c for c in ("NOTE_TYPE", "note_type", "NOTE_KIND")
                     if c in notes_meta.columns), None)
    if time_col is None or type_col is None:
        raise RuntimeError(
            f"Could not find time/type columns in notes_meta; have: {list(notes_meta.columns)[:25]}")
    nm = notes_meta[[time_col, type_col]].copy()
    nm["_years"] = nm[time_col] / 365.25 if unit == "days" else nm[time_col]
    nm = nm[(nm["_years"] >= -5) & (nm["_years"] <= 0)].copy()
    nm["year_bin"] = nm["_years"].round().astype(int)
    out = (nm.groupby(["year_bin", type_col]).size()
             .reset_index(name="n_notes")
             .rename(columns={type_col: "note_type"}))
    return out


def _notes_per_patient(notes_meta: pd.DataFrame) -> pd.DataFrame:
    """Per-(patient, note_type) note counts → distribution input for the Fig 1 box/violin."""
    type_col = next((c for c in ("NOTE_TYPE", "note_type", "NOTE_KIND")
                     if c in notes_meta.columns), None)
    if type_col is None or "DFCI_MRN" not in notes_meta.columns:
        raise RuntimeError(
            f"Could not find DFCI_MRN/note-type columns in notes_meta; have: {list(notes_meta.columns)[:25]}")
    out = (notes_meta.groupby(["DFCI_MRN", type_col]).size()
           .reset_index(name="n_notes")
           .rename(columns={type_col: "note_type"}))
    return out[["DFCI_MRN", "note_type", "n_notes"]]


def _composition_counts(df: pd.DataFrame, cols: list[str], prefix: str, top_n: int) -> pd.DataFrame:
    s = df[cols].sum().sort_values(ascending=False)
    s.index = s.index.str.replace(prefix, "", regex=False)
    s = s.head(top_n)
    return pd.DataFrame({"category": s.index.astype(str), "n": s.values.astype(int)})


def _endpoint_counts() -> pd.DataFrame:
    rows = [
        {"scheme": scheme, "n_endpoints": len(list_trained_events(scheme))}
        for scheme in SCHEME_RESULT_DIRS
    ]
    return pd.DataFrame(rows, columns=ENDPOINT_COUNT_COLUMNS)


def main() -> None:
    emb_df = pd.read_csv(os.path.join(SURV_PATH, EMBEDDING_FILES[SCHEME_FOR_EMBED]))
    cancer_type_df = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_type_df.csv.gz"))
    treatment_df = pd.read_csv(os.path.join(FEATURE_PATH, "categorical_treatment_data_by_line.csv.gz"))
    notes_meta = pd.read_csv(os.path.join(NOTES_PATH, "full_VTE_embeddings_metadata.csv.gz"))

    type_cols = [c for c in cancer_type_df.columns if c.startswith("CANCER_TYPE_")]
    tx_cols = [c for c in treatment_df.columns if c.startswith("PX_on_")]
    tx1 = treatment_df.loc[treatment_df["treatment_line"] == 1]
    cohort_mrns = set(emb_df["DFCI_MRN"])

    save_figure_data(_cohort_counts(emb_df, cancer_type_df), "fig1_cohort_counts.csv")
    save_figure_data(_endpoint_counts(), "fig1_endpoint_counts.csv")
    save_figure_data(_note_volume(notes_meta), "fig1_note_volume.csv")
    save_figure_data(_composition_counts(cancer_type_df, type_cols, "CANCER_TYPE_", 15),
                     "fig1_cancer_type_counts.csv")
    save_figure_data(_stage_counts_from_pickle(cohort_mrns), "fig1_stage_counts.csv")
    save_figure_data(_composition_counts(tx1, tx_cols, "PX_on_", 15),
                     "fig1_treatment_counts.csv")
    save_figure_data(_notes_per_patient(notes_meta), "fig1_notes_per_patient.csv")


if __name__ == "__main__":
    main()
