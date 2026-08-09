"""Driver: write the combination-level data-availability report to SURV_PATH.

Denominator is `cohort_df.parquet` (the actual cohort-eligibility source of
truth), NOT `cancer_type_df.csv.gz` (which figure0.py uses as its historical
denominator). The summary JSON reconciles the two and records the delta
explicitly, resolving the mismatch flagged in V2_PIPELINE_PLAN.md's Context.

Writes to SURV_PATH:
  - data_availability_marginals.csv     modality, n_patients, n_total
  - data_availability_combinations.csv  upset-ready: one row per observed pattern
  - data_availability_pairwise.csv      k x k intersection matrix
  - data_availability_summary.json      cohort-vs-cancer_type_df reconciliation
"""

from __future__ import annotations

import json
import os

import pandas as pd

from config import FEATURE_PATH, SURV_PATH
from pipelines.preprocessing.data_availability import (
    MODALITY_ORDER,
    availability_matrix,
    combination_counts,
    modality_mrn_sets,
    pairwise_overlap,
)


def main() -> None:
    cohort_df = pd.read_parquet(os.path.join(SURV_PATH, "cohort_df.parquet"), columns=["DFCI_MRN"])
    cohort_mrns = set(cohort_df["DFCI_MRN"])

    cancer_type_fp = os.path.join(FEATURE_PATH, "cancer_type_df.csv.gz")
    cancer_type_mrns: set[int] = set()
    if os.path.exists(cancer_type_fp):
        cancer_type_mrns = set(pd.read_csv(cancer_type_fp, usecols=["DFCI_MRN"])["DFCI_MRN"])

    modality_sets = modality_mrn_sets(cohort_mrns)

    marginals_rows = [
        {"modality": "full_cohort", "n_patients": len(cohort_mrns), "n_total": len(cohort_mrns)}
    ]
    for modality in MODALITY_ORDER:
        marginals_rows.append({
            "modality": modality,
            "n_patients": len(modality_sets[modality]),
            "n_total": len(cohort_mrns),
        })
    all_thresholds = set.intersection(*modality_sets.values()) if modality_sets else set()
    marginals_rows.append({
        "modality": "all", "n_patients": len(all_thresholds), "n_total": len(cohort_mrns),
    })
    marginals_df = pd.DataFrame(marginals_rows, columns=["modality", "n_patients", "n_total"])

    matrix = availability_matrix(cohort_mrns, modality_sets)
    combinations_df = combination_counts(matrix)
    pairwise_df = pairwise_overlap(modality_sets)

    marginals_fp = os.path.join(SURV_PATH, "data_availability_marginals.csv")
    combinations_fp = os.path.join(SURV_PATH, "data_availability_combinations.csv")
    pairwise_fp = os.path.join(SURV_PATH, "data_availability_pairwise.csv")
    summary_fp = os.path.join(SURV_PATH, "data_availability_summary.json")

    marginals_df.to_csv(marginals_fp, index=False)
    combinations_df.to_csv(combinations_fp, index=False)
    pairwise_df.to_csv(pairwise_fp, index=False)

    delta_mrns = cohort_mrns.symmetric_difference(cancer_type_mrns)
    summary = {
        "cohort_df_n": len(cohort_mrns),
        "cancer_type_df_n": len(cancer_type_mrns),
        "cohort_minus_cancer_type_n": len(cohort_mrns - cancer_type_mrns),
        "cancer_type_minus_cohort_n": len(cancer_type_mrns - cohort_mrns),
        "symmetric_difference_n": len(delta_mrns),
        "all_five_modalities_n": len(all_thresholds),
        "n_observed_combinations": len(combinations_df),
    }
    with open(summary_fp, "w") as f:
        f.write(json.dumps(summary, indent=2))

    print(f"[wrote] {marginals_fp} ({len(marginals_df)} rows)")
    print(f"[wrote] {combinations_fp} ({len(combinations_df)} rows)")
    print(f"[wrote] {pairwise_fp} ({len(pairwise_df)} rows)")
    print(f"[wrote] {summary_fp}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
