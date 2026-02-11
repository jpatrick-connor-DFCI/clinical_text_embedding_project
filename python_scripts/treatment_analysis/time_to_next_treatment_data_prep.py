import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from embed_surv_utils import generate_survival_embedding_df

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
SURV_PATH = os.path.join(DATA_PATH, "survival_data/")
NOTES_PATH = os.path.join(DATA_PATH, "batched_datasets/VTE_data/processed_datasets/")
ICI_PRED_PATH = os.path.join(DATA_PATH, "treatment_prediction/line_ICI_prediction_data/")
LINE_PRED_PATH = os.path.join(DATA_PATH,'treatment_prediction/time-to-next-treatment/')
INTAE_DATA_PATH = "/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/"
METS_PROJECT = "/data/gusev/Recurrent_Mets_Project/"
PROCESSED_DATA_PATH = os.path.join(METS_PROJECT, "clinical_to_ag/")

# -------------------------------------------------------------------
# Load core datasets
# -------------------------------------------------------------------
treatment_df = pd.read_csv(
    "/data/gusev/USERS/mjsaleh/profile_lines_of_rx/profile_rxlines.csv"
)
tt_phecode_df = pd.read_csv(os.path.join(SURV_PATH, "time-to-phecode/tt_vte_plus_phecodes.csv"))

# Restrict to patients in the VTE cohort
cohort_treatment_df = (
    treatment_df.loc[treatment_df["MRN"].isin(tt_phecode_df["DFCI_MRN"].unique())]
    .copy()
    .rename(columns={"MRN": "DFCI_MRN", "LOT_start_date": "treatment_start_date"})
)

# Parse and sort treatment start dates
cohort_treatment_df["treatment_start_date"] = pd.to_datetime(cohort_treatment_df["treatment_start_date"])
cohort_treatment_df = cohort_treatment_df.sort_values(["DFCI_MRN", "treatment_start_date"])
cohort_treatment_df["treatment_line"] = cohort_treatment_df.groupby("DFCI_MRN").cumcount() + 1

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
notes_meta = pd.read_csv(os.path.join(NOTES_PATH, "full_VTE_embeddings_metadata.csv"))
embeddings_data = np.load(os.path.join(NOTES_PATH, "full_VTE_embeddings_as_array.npy"))

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