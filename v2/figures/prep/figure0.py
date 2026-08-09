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
- fig0_data_availability.csv      stage, label, n_patients, n_total  (in cascade order)
"""

from __future__ import annotations

import os

import pandas as pd

from config import FEATURE_PATH, SURV_PATH
from figures.io import save_figure_data
from schemes import embedding_file
from shared.stages import load_stage_map, normalize_stage

SCHEME_FOR_EMBED = "icd3_post"  # widest post-text-merge cohort (mirrors prep_figure_1.py)

DATA_AVAILABILITY_COLUMNS = ["stage", "label", "n_patients", "n_total"]


def _mrns_with_stage(cohort_mrns: set[int]) -> set[int]:
    mrn_to_stage = load_stage_map()
    if mrn_to_stage is None:
        print("  cancer_stage_df.csv.gz unavailable, no stage data")
        return set()
    mrns = {mrn for mrn, v in mrn_to_stage.items()
            if mrn in cohort_mrns and normalize_stage(v) is not None}
    return mrns


def _mrns_with_treatment(cohort_mrns: set[int]) -> set[int]:
    tx_df = pd.read_csv(os.path.join(FEATURE_PATH, "categorical_treatment_data_by_line.csv.gz"))
    tx1 = tx_df.loc[tx_df["treatment_line"] == 1]
    return set(tx1["DFCI_MRN"]) & cohort_mrns


def _mrns_with_raw_feature(filename: str, cohort_mrns: set[int]) -> set[int]:
    fp = os.path.join(FEATURE_PATH, filename)
    if not os.path.exists(fp):
        print(f"  missing {fp}")
        return set()
    d = pd.read_csv(fp, usecols=["DFCI_MRN"])
    return set(d["DFCI_MRN"]) & cohort_mrns


def _data_availability() -> pd.DataFrame:
    # Full cohort + cancer type, same source used as the Fig 1 cohort denominator.
    cancer_type_df = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_type_df.csv.gz"),
                                 usecols=["DFCI_MRN"])
    cohort_mrns = set(cancer_type_df["DFCI_MRN"])
    n_total = len(cohort_mrns)

    emb_df = pd.read_csv(os.path.join(SURV_PATH, embedding_file(SCHEME_FOR_EMBED)),
                         usecols=["DFCI_MRN"])
    text_mrns = set(emb_df["DFCI_MRN"]) & cohort_mrns
    stage_mrns = _mrns_with_stage(cohort_mrns)
    treatment_mrns = _mrns_with_treatment(cohort_mrns)
    somatic_mrns = _mrns_with_raw_feature("complete_somatic_data_df.csv.gz", cohort_mrns)
    prs_mrns = _mrns_with_raw_feature("complete_germline_data_df.csv.gz", cohort_mrns)

    all_thresholds = (text_mrns & stage_mrns & treatment_mrns
                      & somatic_mrns & prs_mrns)

    rows = [
        ("full_cohort", "Full Cohort",              cohort_mrns),
        ("text",       "With Text",                text_mrns),
        ("stage",      "With Stage",                stage_mrns),
        ("treatment",  "With Treatment",            treatment_mrns),
        ("somatic",    "With Somatic",              somatic_mrns),
        ("prs",        "With PRS",                  prs_mrns),
        ("all",        "Passes All Thresholds",     all_thresholds),
    ]
    return pd.DataFrame(
        [{"stage": s, "label": lbl, "n_patients": len(mrns), "n_total": n_total}
         for s, lbl, mrns in rows],
        columns=DATA_AVAILABILITY_COLUMNS,
    )


def main() -> None:
    save_figure_data(_data_availability(), "fig0_data_availability.csv")


if __name__ == "__main__":
    main()
