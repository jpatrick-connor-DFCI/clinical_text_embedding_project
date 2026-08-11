"""Single source of truth for "which patients have modality X".

Lifted from `figures/prep/figure0.py`'s `_mrns_with_*` helpers (the
`treatment_line == 1` filter, `load_stage_map()` from `shared/stages.py`) so
`figure0.py` and `report_data_availability.py` can never disagree — figure0.py
now imports `modality_mrn_sets` from here rather than reimplementing it.

All modalities are sourced from the raw feature files under
clinical_and_genomic_features/ (FEATURE_PATH) — the same files
generate_all_non_text_covariates.py writes.
"""

from __future__ import annotations

import itertools
import os

import polars as pl

from config import FEATURE_PATH, NOTES_PATH
from shared.stages import load_stage_map, normalize_stage

MODALITY_ORDER = ["text", "stage", "treatment", "somatic", "prs", "metburden"]


def _mrns_with_stage(cohort_mrns: set[int]) -> set[int]:
    mrn_to_stage = load_stage_map()
    if mrn_to_stage is None:
        print("  cancer_stage_df.csv.gz unavailable, no stage data")
        return set()
    return {
        mrn for mrn, v in mrn_to_stage.items()
        if mrn in cohort_mrns and normalize_stage(v) is not None
    }


def _mrns_with_treatment(cohort_mrns: set[int]) -> set[int]:
    path = os.path.join(FEATURE_PATH, "categorical_treatment_data_by_line.csv.gz")
    if not os.path.exists(path):
        print(f"  missing {path}")
        return set()
    tx_df = pl.read_csv(path)
    return set(tx_df.filter(pl.col("treatment_line") == 1).get_column("DFCI_MRN")) & cohort_mrns


def _mrns_with_raw_feature(filename: str, cohort_mrns: set[int]) -> set[int]:
    fp = os.path.join(FEATURE_PATH, filename)
    if not os.path.exists(fp):
        print(f"  missing {fp}")
        return set()
    d = pl.read_csv(fp, columns=["DFCI_MRN"])
    return set(d.get_column("DFCI_MRN")) & cohort_mrns


def _mrns_with_text(cohort_mrns: set[int]) -> set[int]:
    """Estimate text-modality coverage directly from the knitted note metadata
    (one row per note), rather than a downstream *_embedding_prediction_df.parquet
    scheme output — those aren't built until after the embeddings pipeline runs, so
    depending on one would make this report's ordering depend on stages that run
    much later (see 01_run_preprocessing.ipynb)."""
    metadata_fp = os.path.join(NOTES_PATH, "full_clinical_notes_embeddings_metadata.parquet")
    if not os.path.exists(metadata_fp):
        print(f"  missing {metadata_fp}")
        return set()
    notes_meta = pl.read_parquet(metadata_fp, columns=["DFCI_MRN"])
    return set(notes_meta.get_column("DFCI_MRN")) & cohort_mrns


def modality_mrn_sets(cohort_mrns: set[int]) -> dict[str, set[int]]:
    """MRN set per modality, restricted to `cohort_mrns`. Keys match
    MODALITY_ORDER: text, stage, treatment, somatic, prs, metburden.

    metburden is zero-filled to every cohort patient (see build_met_burden_df),
    so it is expected to read ~100% here - that is not a bug, and does not
    move the "all thresholds" intersection row."""
    return {
        "text": _mrns_with_text(cohort_mrns),
        "stage": _mrns_with_stage(cohort_mrns),
        "treatment": _mrns_with_treatment(cohort_mrns),
        "somatic": _mrns_with_raw_feature("complete_somatic_data_df.csv.gz", cohort_mrns),
        "prs": _mrns_with_raw_feature("complete_germline_data_df.csv.gz", cohort_mrns),
        "metburden": _mrns_with_raw_feature("met_burden_df.csv.gz", cohort_mrns),
    }


def availability_matrix(cohort_mrns: set[int], modality_sets: dict[str, set[int]]) -> pl.DataFrame:
    """DFCI_MRN x boolean-per-modality, one row per cohort patient."""
    mrns = sorted(cohort_mrns)
    data = {"DFCI_MRN": mrns}
    for modality in MODALITY_ORDER:
        mrn_set = modality_sets.get(modality, set())
        data[modality] = [mrn in mrn_set for mrn in mrns]
    return pl.DataFrame(data)


def combination_counts(matrix: pl.DataFrame) -> pl.DataFrame:
    """One row per *observed* boolean pattern across MODALITY_ORDER + n_patients.
    Emits only observed patterns, not all 2^len(MODALITY_ORDER)."""
    modalities = [m for m in MODALITY_ORDER if m in matrix.columns]
    return matrix.group_by(modalities).len(name="n_patients").sort("n_patients", descending=True).select(modalities + ["n_patients"])


def pairwise_overlap(modality_sets: dict[str, set[int]]) -> pl.DataFrame:
    """k x k intersection matrix (modality_a, modality_b, n_overlap)."""
    modalities = [m for m in MODALITY_ORDER if m in modality_sets]
    rows = []
    for mod_a, mod_b in itertools.product(modalities, modalities):
        n_overlap = len(modality_sets[mod_a] & modality_sets[mod_b])
        rows.append({"modality_a": mod_a, "modality_b": mod_b, "n_overlap": n_overlap})
    return pl.DataFrame(rows, schema=["modality_a", "modality_b", "n_overlap"])
