"""Pre-compute inputs for Figure 0 (cohort data-availability cascade).

A CONSORT-style attrition panel run just ahead of Figure 1: starting from the
full cancer-type cohort, how many patients have each successive modality
available, down to how many pass every data-availability threshold at once
(the set actually usable for the full multi-modal comparison in Figures 3+).

All modalities are sourced from the raw feature files under
clinical_and_genomic_features/ (FEATURE_PATH) — the same files
generate_all_non_text_covariates.py writes — rather than any downstream,
event-specific held-out risk-score files:
- cohort/cancer type: cancer_type_df.csv.gz     (also the Fig 1 cohort source)
- stage:              cancer_stage_df.csv.gz (raw CANCER_STAGE column)
- treatment:          categorical_treatment_data_by_line.csv.gz
- somatic:            complete_somatic_data_df.csv.gz
- prs:                complete_germline_data_df.csv.gz

Writes to FIGURE_DATA_DIR:
- fig0_data_availability.csv           stage, label, n_patients, n_total  (in cascade order)
- fig0_availability_combinations.csv   one row per observed modality-availability pattern

Modality set-membership logic (text/stage/treatment/somatic/prs MRN sets) now
lives in `pipelines.preprocessing.data_availability`, imported here rather
than reimplemented, so this figure and `report_data_availability.py` can
never disagree.
"""

from __future__ import annotations

import os

import polars as pl

from config import FEATURE_PATH
from figures.io import save_figure_data
from pipelines.preprocessing.data_availability import (
    MODALITY_ORDER,
    availability_matrix,
    combination_counts,
    modality_mrn_sets,
)

DATA_AVAILABILITY_COLUMNS = ["stage", "label", "n_patients", "n_total"]

_MODALITY_LABELS = {
    "text": "With Text",
    "stage": "With Stage",
    "treatment": "With Treatment",
    "somatic": "With Somatic",
    "prs": "With PRS",
    "metburden": "With Met. Burden",
}


def _data_availability() -> tuple[pl.DataFrame, pl.DataFrame]:
    # Full cohort + cancer type, same source used as the Fig 1 cohort denominator.
    cancer_type_df = pl.read_csv(os.path.join(FEATURE_PATH, "cancer_type_df.csv.gz"),
                                 columns=["DFCI_MRN"])
    cohort_mrns = set(cancer_type_df["DFCI_MRN"])
    n_total = len(cohort_mrns)

    modality_sets = modality_mrn_sets(cohort_mrns)
    all_thresholds = set.intersection(*modality_sets.values()) if modality_sets else set()

    rows = [("full_cohort", "Full Cohort", cohort_mrns)]
    rows += [(m, _MODALITY_LABELS[m], modality_sets[m]) for m in MODALITY_ORDER]
    rows.append(("all", "Passes All Thresholds", all_thresholds))

    cascade_df = pl.DataFrame(
        [{"stage": s, "label": lbl, "n_patients": len(mrns), "n_total": n_total}
         for s, lbl, mrns in rows],
        schema=DATA_AVAILABILITY_COLUMNS,
    )

    matrix = availability_matrix(cohort_mrns, modality_sets)
    combinations_df = combination_counts(matrix)

    return cascade_df, combinations_df


def main() -> None:
    cascade_df, combinations_df = _data_availability()
    save_figure_data(cascade_df, "fig0_data_availability.csv")
    save_figure_data(combinations_df, "fig0_availability_combinations.csv")


if __name__ == "__main__":
    main()
