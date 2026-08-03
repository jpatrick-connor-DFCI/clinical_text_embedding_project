"""Pre-compute inputs for Figure 1 (cohort & embedding overview).

Writes to FIGURE_DATA_DIR:
- fig1_endpoint_counts.csv        scheme → n_endpoints
- fig1_cancer_type_counts.csv     category, n   (top-10 raw cancer types + pooled "Other";
                                  restricted to the analysis cohort so n sums to the cohort N)
- fig1_stage_counts.csv           category, n
- fig1_treatment_counts.csv       category, n
- fig1_notes_per_patient.csv      DFCI_MRN, note_type, n_notes  (per patient x note type)
"""

from __future__ import annotations

import os

import pandas as pd

from config import FEATURE_PATH, NOTES_PATH, SURV_PATH
from figures.io import save_figure_data
from schemes import SCHEMES, embedding_file, list_trained_events
from shared.stages import STAGE_ORDER, load_stage_map, normalize_stage

SCHEME_FOR_EMBED = "icd3_post"  # widest cohort
ENDPOINT_COUNT_COLUMNS = ["scheme", "n_endpoints"]


# Raw, complete stage labels — `cancer_stage_df.csv.gz` is produced with
# pd.get_dummies(drop_first=True) (generate_all_non_text_covariates.py:39) which
# silently drops Stage I from the one-hot. Read the pickle directly (via
# load_stage_map()) to recover it. Same pattern as prep_figure_2._major_stage_labels().
def _stage_counts_from_pickle(cohort_mrns: set[int]) -> pd.DataFrame:
    """Major-stage breakdown (I/II/III/IV) for the analysis cohort."""
    mrn_to_stage = load_stage_map()
    if mrn_to_stage is None:
        print("  falling back to one-hot CSV")
        oh = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_stage_df.csv.gz"))
        oh = oh[oh["DFCI_MRN"].isin(cohort_mrns)]
        stage_cols = [c for c in oh.columns if c.startswith("CANCER_STAGE_")]
        present = [normalize_stage(c.replace("CANCER_STAGE_", "")) for c in stage_cols]
        reference = next((s for s in STAGE_ORDER if s not in present), None)
        active = oh[stage_cols].to_numpy()
        labels = [reference if row.max() <= 0 else present[int(row.argmax())]
                  for row in active]
    else:
        labels = [normalize_stage(v) for k, v in mrn_to_stage.items() if k in cohort_mrns]
    s = pd.Series([x for x in labels if x in STAGE_ORDER]).value_counts()
    s = s.reindex(STAGE_ORDER, fill_value=0)
    return pd.DataFrame({"category": s.index.astype(str), "n": s.values.astype(int)})


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


def _cancer_type_counts(cancer_type_df: pd.DataFrame, cohort_mrns: set[int], top_n: int) -> pd.DataFrame:
    """Cancer-type composition of the cohort, from the raw CANCER_TYPE string column.

    Uses the raw label (preserved by generate_all_non_text_covariates.py precisely so
    the drop_first reference category isn't lost, unlike summing the one-hot
    CANCER_TYPE_* columns) and restricts to the analysis cohort so the reported total
    matches the cohort N shown elsewhere (Fig 0). Rows beyond top_n are pooled into
    "Other" so the counts still sum to the full cohort.
    """
    sub = cancer_type_df[cancer_type_df["DFCI_MRN"].isin(cohort_mrns)]
    if "CANCER_TYPE" not in sub.columns:
        # Fallback: reconstruct from one-hot (loses the drop_first reference class).
        type_cols = [c for c in sub.columns if c.startswith("CANCER_TYPE_")]
        return _composition_counts(sub, type_cols, "CANCER_TYPE_", top_n)
    vc = sub["CANCER_TYPE"].astype(str).value_counts()
    # Upstream preprocessing already collapses cancer types with <500 patients
    # into "OTHER". Exclude that bucket when choosing the top named types, then
    # combine it with any additional types falling outside the displayed top_n.
    # Otherwise the plot receives both "OTHER" and a newly pooled "Other" row,
    # which render as two identically labelled pie slices.
    is_other = vc.index.str.strip().str.upper() == "OTHER"
    named = vc.loc[~is_other]
    top = named.head(top_n)
    rows = [{"category": cat, "n": int(cnt)} for cat, cnt in top.items()]
    other = int(vc.loc[is_other].sum() + named.iloc[top_n:].sum())
    if other > 0:
        rows.append({"category": "Other", "n": other})
    return pd.DataFrame(rows, columns=["category", "n"])


def _endpoint_counts() -> pd.DataFrame:
    rows = [
        {"scheme": scheme, "n_endpoints": len(list_trained_events(scheme))}
        for scheme in SCHEMES
    ]
    return pd.DataFrame(rows, columns=ENDPOINT_COUNT_COLUMNS)


def main() -> None:
    emb_df = pd.read_csv(os.path.join(SURV_PATH, embedding_file(SCHEME_FOR_EMBED)))
    cancer_type_df = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_type_df.csv.gz"))
    treatment_df = pd.read_csv(os.path.join(FEATURE_PATH, "categorical_treatment_data_by_line.csv.gz"))
    notes_meta = pd.read_csv(os.path.join(NOTES_PATH, "full_VTE_embeddings_metadata.csv.gz"))

    tx_cols = [c for c in treatment_df.columns if c.startswith("PX_on_")]
    tx1 = treatment_df.loc[treatment_df["treatment_line"] == 1]
    cohort_mrns = set(emb_df["DFCI_MRN"])

    save_figure_data(_endpoint_counts(), "fig1_endpoint_counts.csv")
    save_figure_data(_cancer_type_counts(cancer_type_df, cohort_mrns, 10),
                     "fig1_cancer_type_counts.csv")
    save_figure_data(_stage_counts_from_pickle(cohort_mrns), "fig1_stage_counts.csv")
    save_figure_data(_composition_counts(tx1, tx_cols, "PX_on_", 15),
                     "fig1_treatment_counts.csv")
    save_figure_data(_notes_per_patient(notes_meta), "fig1_notes_per_patient.csv")


if __name__ == "__main__":
    main()
