"""Pre-compute inputs for Figure 0 (cohort data-availability cascade).

A CONSORT-style attrition panel run just ahead of Figure 1: starting from the
eligible oncology cohort, how many patients have cancer type and each successive modality
available, down to how many pass every data-availability threshold at once
(the set actually usable for the full multi-modal comparison in Figures 3+).

All modalities are sourced from the raw feature files under
clinical_and_genomic_features/ (FEATURE_PATH) — the same files
generate_all_non_text_covariates.py writes — rather than any downstream,
event-specific held-out risk-score files:
- cohort denominator: SURV_PATH/cohort_df.parquet
- cancer type:        cancer_type_df.csv.gz
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

from config import FEATURE_PATH, SURV_PATH
from figures.io import save_figure_data
from pipelines.preprocessing.data_availability import (
    MODALITY_ORDER,
    availability_matrix,
    combination_counts,
    modality_mrn_sets,
)

DATA_AVAILABILITY_COLUMNS = ["stage", "label", "n_patients", "n_total"]

_MODALITY_LABELS = {
    "cancer_type": "Cancer type available",
    "text": "Cancer type + text available",
    "stage": "With Stage",
    "treatment": "With Treatment",
    "somatic": "With Somatic",
    "prs": "With PRS",
    "metburden": "With Met. Burden",
}


def _data_availability() -> tuple[pl.DataFrame, pl.DataFrame]:
    # Start before cancer-type restriction so cancer-type availability is an
    # explicit eligibility step rather than an invisible denominator choice.
    cohort_df = pl.read_parquet(os.path.join(SURV_PATH, "cohort_df.parquet"),
                                columns=["DFCI_MRN"])
    cohort_mrns = set(cohort_df["DFCI_MRN"])
    n_total = len(cohort_mrns)

    modality_sets = modality_mrn_sets(cohort_mrns)
    # True cumulative attrition: every box is a subset of the preceding box.
    rows = [("full_cohort", "Eligible oncology cohort", cohort_mrns)]
    running = set(cohort_mrns)
    for modality in MODALITY_ORDER:
        running &= modality_sets[modality]
        rows.append((modality, _MODALITY_LABELS[modality], set(running)))
    rows.append(("all", "Final complete-case cohort", set(running)))

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
