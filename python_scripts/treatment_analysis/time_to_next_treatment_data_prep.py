import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from embed_surv_utils import generate_survival_embedding_df
from treatment_analysis_common import (
    DATA_PATH,
    load_note_embeddings,
    load_vte_cohort_treatments,
    add_treatment_line_columns,
)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
ICI_PRED_PATH = os.path.join(DATA_PATH, "treatment_prediction/line_ICI_prediction_data/")
LINE_PRED_PATH = os.path.join(DATA_PATH,'treatment_prediction/time-to-next-treatment/')
INTAE_DATA_PATH = "/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/"
METS_PROJECT = "/data/gusev/Recurrent_Mets_Project/"
PROCESSED_DATA_PATH = os.path.join(METS_PROJECT, "clinical_to_ag/")

# -------------------------------------------------------------------
# Load core datasets
# -------------------------------------------------------------------
cohort_treatment_df = (load_vte_cohort_treatments()
                       .rename(columns={'MRN': 'DFCI_MRN', 'LOT_start_date': 'treatment_start_date'}))
cohort_treatment_df = add_treatment_line_columns(
    cohort_treatment_df, mrn_col='DFCI_MRN', start_col='treatment_start_date'
)

# -------------------------------------------------------------------
# Load survival / follow-up data
# -------------------------------------------------------------------
vte_data = pd.read_csv(os.path.join(INTAE_DATA_PATH, "follow_up_vte_df_cohort.csv"))
vte_data_sub = vte_data[
    ["DFCI_MRN", "AGE_AT_FIRST_TREAT", "BIOLOGICAL_SEX", "first_treatment_date", 
     "death_date", "last_contact_date", "tt_death", "death", "tt_vte", "vte"]
].copy()
vte_data_sub["last_contact_date"] = pd.to_datetime(vte_data_sub["last_contact_date"])

# Merge in last contact date for censoring
cohort_treatment_df = cohort_treatment_df.merge(
    vte_data_sub[["DFCI_MRN", "last_contact_date"]],
    on="DFCI_MRN",
    how="left"
)

# -------------------------------------------------------------------
# Construct TTNT variables
# -------------------------------------------------------------------
cohort_treatment_df["next_line_start_date"] = cohort_treatment_df.groupby("DFCI_MRN")["treatment_start_date"].shift(-1)

cohort_treatment_df["time_on_treatment"] = np.where(
    cohort_treatment_df["next_line_start_date"].notnull(),
    (cohort_treatment_df["next_line_start_date"] - cohort_treatment_df["treatment_start_date"]).dt.days,
    (cohort_treatment_df["last_contact_date"] - cohort_treatment_df["treatment_start_date"]).dt.days,
)

cohort_treatment_df["event"] = np.where(cohort_treatment_df["next_line_start_date"].notnull(), 1, 0)

ttnt_df = cohort_treatment_df[
    ["DFCI_MRN", "treatment_line", "treatment_start_date", "last_contact_date", "time_on_treatment", "event"]
]

# -------------------------------------------------------------------
# Notes embeddings
# -------------------------------------------------------------------
notes_meta, embeddings_data = load_note_embeddings()

note_types = ["Clinician", "Imaging", "Pathology"]
pool_fx = {nt: "time_decay_mean" for nt in note_types}

# -------------------------------------------------------------------
# Generate line-by-line survival embedding dataframes
# -------------------------------------------------------------------
pred_dfs = []
for treatment_line in tqdm(ttnt_df["treatment_line"].unique(), desc="Processing treatment lines"):
    line_subset = ttnt_df.loc[ttnt_df["treatment_line"] == treatment_line]

    notes_meta_sub = (
        notes_meta[notes_meta["DFCI_MRN"].isin(line_subset["DFCI_MRN"])]
        .merge(line_subset, on="DFCI_MRN", how="left")
        .assign(
            NOTE_TIME_REL_PRED_START_DT=lambda df: (
                pd.to_datetime(df["NOTE_DATETIME"]) - pd.to_datetime(df["treatment_start_date"])
            ).dt.days
        )
    )

    line_embeds = generate_survival_embedding_df(
        notes_meta=notes_meta_sub,
        survival_df=None,
        embedding_array=embeddings_data,
        note_types=note_types,
        note_timing_col="NOTE_TIME_REL_PRED_START_DT",
        max_note_window=0,
        pool_fx=pool_fx,
        decay_param=0.01,
        continuous_window=True,
    )

    line_subset = line_subset.merge(line_embeds, on="DFCI_MRN")
    pred_dfs.append(line_subset)

# Combine results
full_ttnt_df = pd.concat(pred_dfs).sort_values(by=["DFCI_MRN", "treatment_line"])
full_ttnt_df.to_csv(os.path.join(LINE_PRED_PATH, 'full_tt_next_treatment_pred_df.csv'), index=False)
