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

import polars as pl

from config import FEATURE_PATH, NOTES_PATH, SURV_PATH
from figures.io import save_figure_data
from schemes import SCHEMES, embedding_file, list_trained_events
from shared.stages import STAGE_ORDER, load_stage_map, normalize_stage

SCHEME_FOR_EMBED = "icd3_post"  # widest cohort
ENDPOINT_COUNT_COLUMNS = ["scheme", "n_endpoints"]


# Raw, complete stage labels — cancer_stage_df.csv.gz now carries the raw
# CANCER_STAGE string column directly (see generate_all_non_text_covariates.py),
# read here via load_stage_map(). Same pattern as prep_figure_2._major_stage_labels().
def _stage_counts_from_pickle(cohort_mrns: set[int]) -> pl.DataFrame:
    """Major-stage breakdown (I/II/III/IV) for the analysis cohort."""
    mrn_to_stage = load_stage_map()
    if mrn_to_stage is None:
        print("  cancer_stage_df.csv.gz unavailable, no stage data")
        labels = []
    else:
        labels = [normalize_stage(v) for k, v in mrn_to_stage.items() if k in cohort_mrns]
    labels = [x for x in labels if x in STAGE_ORDER]
    counts = {stage: 0 for stage in STAGE_ORDER}
    for label in labels:
        counts[label] += 1
    return pl.DataFrame({"category": STAGE_ORDER, "n": [counts[s] for s in STAGE_ORDER]})


def _notes_per_patient(notes_meta: pl.DataFrame) -> pl.DataFrame:
    """Per-(patient, note_type) note counts → distribution input for the Fig 1 box/violin."""
    type_col = next((c for c in ("NOTE_TYPE", "note_type", "NOTE_KIND")
                     if c in notes_meta.columns), None)
    if type_col is None or "DFCI_MRN" not in notes_meta.columns:
        raise RuntimeError(
            f"Could not find DFCI_MRN/note-type columns in notes_meta; have: {list(notes_meta.columns)[:25]}")
    out = (notes_meta.group_by(["DFCI_MRN", type_col])
           .agg(pl.len().alias("n_notes"))
           .rename({type_col: "note_type"}))
    return out.select(["DFCI_MRN", "note_type", "n_notes"])


def _composition_counts(df: pl.DataFrame, cols: list[str], prefix: str, top_n: int) -> pl.DataFrame:
    sums = {c.replace(prefix, ""): int(df[c].sum()) for c in cols}
    ordered = sorted(sums.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return pl.DataFrame({"category": [k for k, _ in ordered], "n": [v for _, v in ordered]})


def _cancer_type_counts(cancer_type_df: pl.DataFrame, cohort_mrns: set[int], top_n: int) -> pl.DataFrame:
    """Cancer-type composition of the cohort, from the raw CANCER_TYPE string column.

    Uses the raw label (preserved by generate_all_non_text_covariates.py precisely so
    the drop_first reference category isn't lost, unlike summing the one-hot
    CANCER_TYPE_* columns) and restricts to the analysis cohort so the reported total
    matches the cohort N shown elsewhere (Fig 0). Rows beyond top_n are pooled into
    "Other" so the counts still sum to the full cohort.
    """
    sub = cancer_type_df.filter(pl.col("DFCI_MRN").is_in(cohort_mrns))
    vc = (sub.select(pl.col("CANCER_TYPE").cast(pl.Utf8))
          .group_by("CANCER_TYPE").agg(pl.len().alias("n"))
          .sort("n", descending=True))
    # Upstream preprocessing already collapses cancer types with <500 patients
    # into "OTHER". Exclude that bucket when choosing the top named types, then
    # combine it with any additional types falling outside the displayed top_n.
    # Otherwise the plot receives both "OTHER" and a newly pooled "Other" row,
    # which render as two identically labelled pie slices.
    is_other = vc["CANCER_TYPE"].str.strip_chars().str.to_uppercase() == "OTHER"
    named = vc.filter(~is_other)
    other_sum = vc.filter(is_other)["n"].sum()
    top = named.head(top_n)
    rows = [{"category": cat, "n": int(cnt)} for cat, cnt in zip(top["CANCER_TYPE"].to_list(), top["n"].to_list())]
    other = int(other_sum + sum(named["n"].to_list()[top_n:]))
    if other > 0:
        rows.append({"category": "Other", "n": other})
    return pl.DataFrame(rows, schema=["category", "n"]) if rows else pl.DataFrame(schema={"category": pl.Utf8, "n": pl.Int64})


def _endpoint_counts() -> pl.DataFrame:
    rows = [
        {"scheme": scheme, "n_endpoints": len(list_trained_events(scheme))}
        for scheme in SCHEMES
    ]
    return pl.DataFrame(rows, schema=ENDPOINT_COUNT_COLUMNS)


def main() -> None:
    emb_df = pl.read_parquet(os.path.join(SURV_PATH, embedding_file(SCHEME_FOR_EMBED)))
    cancer_type_df = pl.read_csv(os.path.join(FEATURE_PATH, "cancer_type_df.csv.gz"))
    treatment_df = pl.read_csv(os.path.join(FEATURE_PATH, "categorical_treatment_data_by_line.csv.gz"))
    notes_meta = pl.read_parquet(os.path.join(NOTES_PATH, "full_clinical_notes_embeddings_metadata.parquet"))

    tx_cols = [c for c in treatment_df.columns if c.startswith("PX_on_")]
    tx1 = treatment_df.filter(pl.col("treatment_line") == 1)
    cohort_mrns = set(emb_df["DFCI_MRN"].to_list())

    save_figure_data(_endpoint_counts(), "fig1_endpoint_counts.csv")
    save_figure_data(_cancer_type_counts(cancer_type_df, cohort_mrns, 10),
                     "fig1_cancer_type_counts.csv")
    save_figure_data(_stage_counts_from_pickle(cohort_mrns), "fig1_stage_counts.csv")
    save_figure_data(_composition_counts(tx1, tx_cols, "PX_on_", 15),
                     "fig1_treatment_counts.csv")
    save_figure_data(_notes_per_patient(notes_meta), "fig1_notes_per_patient.csv")


if __name__ == "__main__":
    main()
