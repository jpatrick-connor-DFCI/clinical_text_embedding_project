"""Pre-compute inputs for Figure 0 (cohort data-availability cascade).

A CONSORT-style attrition panel run just ahead of Figure 1: starting from the
base VTE cohort, how many patients have each successive modality available,
down to how many pass every data-availability threshold at once (the set
actually usable for the full multi-modal comparison in Figures 3+).

Representative (scheme, event) = death_met / death, matching the convention
already used for the Fig 3B correlation heatmap (prep_figure_3.DEATH_SCHEME).

Writes to FIGURE_DATA_DIR:
- fig0_data_availability.csv      stage, label, n_patients, n_total  (in cascade order)
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _figure_utils import (
    EMBEDDING_FILES, FEATURE_PATH, SURV_PATH,
    feature_held_out_dir, save_figure_data,
)

# Base (pre-embedding, pre-filtering) VTE cohort — same source file as
# generate_embedding_prediction_datasets.py:_load_shared_inputs.
INTAE_DATA_PATH = "/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/"
BASE_COHORT_FILE = os.path.join(INTAE_DATA_PATH, "follow_up_vte_df_cohort.csv")

SCHEME_FOR_EMBED = "icd3_post"       # widest post-text-merge cohort (mirrors prep_figure_1.py)
REPRESENTATIVE_SCHEME = "death_met"  # mirrors prep_figure_3.DEATH_SCHEME
REPRESENTATIVE_EVENT = "death"

# Same stage pickle used by prep_figure_1._stage_counts_from_pickle.
STAGE_PATH = (
    "/data/gusev/PROFILE/CLINICAL/OncDRS/DERIVED_FROM_CLINICAL_TEXTS_2024_03/"
    "derived_files/cancer_stage/dfci_cancer_mrn_to_derived_cancer_stage.pkl"
)
_STAGE_TOKEN = re.compile(r"^(IV|III|II|I|4|3|2|1)[A-D]?$")

DATA_AVAILABILITY_COLUMNS = ["stage", "label", "n_patients", "n_total"]


def _normalize_stage(raw) -> str | None:
    if pd.isna(raw):
        return None
    s = str(raw).upper().strip().replace("STAGE", "").strip()
    s = re.sub(r"\.0+$", "", s)
    return "matched" if _STAGE_TOKEN.match(s) else None


def _mrns_with_stage(cohort_mrns: set[int]) -> set[int]:
    try:
        with open(STAGE_PATH, "rb") as f:
            mrn_to_stage = pickle.load(f)
    except (FileNotFoundError, OSError, pickle.UnpicklingError) as e:
        print(f"  stage pickle unavailable ({type(e).__name__}); falling back to one-hot CSV")
        oh = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_stage_df.csv.gz"))
        oh = oh[oh["DFCI_MRN"].isin(cohort_mrns)]
        stage_cols = [c for c in oh.columns if c.startswith("CANCER_STAGE_")]
        # One-hot rows are "has a stage" as soon as *any* stage indicator or the
        # (drop_first) reference class applies — every row in this table has a
        # resolvable stage by construction, so presence in `oh` is sufficient.
        return set(oh["DFCI_MRN"]) if stage_cols else set()
    mrns = {mrn for mrn, v in mrn_to_stage.items()
            if mrn in cohort_mrns and _normalize_stage(v) is not None}
    return mrns


def _mrns_with_treatment(cohort_mrns: set[int]) -> set[int]:
    tx_df = pd.read_csv(os.path.join(FEATURE_PATH, "categorical_treatment_data_by_line.csv.gz"))
    tx1 = tx_df.loc[tx_df["treatment_line"] == 1]
    return set(tx1["DFCI_MRN"]) & cohort_mrns


def _mrns_with_modality_risk_score(modality: str, cohort_mrns: set[int]) -> set[int]:
    fp = os.path.join(
        feature_held_out_dir(REPRESENTATIVE_SCHEME, REPRESENTATIVE_EVENT),
        f"{modality}_risk_scores.csv",
    )
    if not os.path.exists(fp):
        print(f"  missing {fp}")
        return set()
    d = pd.read_csv(fp, usecols=["DFCI_MRN"])
    return set(d["DFCI_MRN"]) & cohort_mrns


def _data_availability() -> pd.DataFrame:
    base = pd.read_csv(BASE_COHORT_FILE, usecols=["DFCI_MRN"])
    cohort_mrns = set(base["DFCI_MRN"])
    n_total = len(cohort_mrns)

    emb_df = pd.read_csv(os.path.join(SURV_PATH, EMBEDDING_FILES[SCHEME_FOR_EMBED]),
                         usecols=["DFCI_MRN"])
    text_mrns = set(emb_df["DFCI_MRN"]) & cohort_mrns
    stage_mrns = _mrns_with_stage(cohort_mrns)
    treatment_mrns = _mrns_with_treatment(cohort_mrns)
    somatic_mrns = _mrns_with_modality_risk_score("somatic", cohort_mrns)
    prs_mrns = _mrns_with_modality_risk_score("prs", cohort_mrns)

    all_thresholds = (text_mrns & stage_mrns & treatment_mrns
                      & somatic_mrns & prs_mrns)

    rows = [
        ("vte_cohort", "VTE Cohort",              cohort_mrns),
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
