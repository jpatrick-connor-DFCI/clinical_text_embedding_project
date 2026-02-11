import os
import pandas as pd

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
NOTES_PATH = os.path.join(DATA_PATH, "batched_datasets/processed_datasets/")
TREATMENT_PRED_PATH = os.path.join(DATA_PATH,'treatment_prediction/first_line_treatment_prediction_data/')
buffer_path = os.path.join(TREATMENT_PRED_PATH, 'buffered_prediction_data/')

med_classes = pd.read_csv(os.path.join(DATA_PATH, 'GPT_generated_med_classes.csv'))

# Load treatment data
treatment_df = (pd.read_csv("/data/gusev/USERS/mjsaleh/profile_lines_of_rx/ALL_MEDICATION_LINES.csv")
                .rename(columns={"MRN": "DFCI_MRN", "MED_START_DT": "treatment_start_date"}))
treatment_df["treatment_start_date"] = pd.to_datetime(treatment_df["treatment_start_date"])
treatment_df = treatment_df.sort_values(["DFCI_MRN", "treatment_start_date"])
treatment_df["treatment_line"] = treatment_df.groupby("DFCI_MRN").cumcount() + 1

medications = (
    treatment_df['MED_NAME']
    .str.split(';')
    .explode()
    .str.strip())

med_class_dict = dict(zip(med_classes['MED_NAME'], med_classes['MOA_Category']))

# --- explode meds from the string column ---
treatments_long = (
    treatment_df["MED_NAME"]
      .fillna("")
      .str.split(";")
      .explode()
      .str.strip())

# drop empties
treatments_long = treatments_long[treatments_long != ""]

# --- map med -> class, unknown -> OTHER ---
classes_long = treatments_long.map(med_class_dict).fillna("OTHER")

# --- one-hot encode classes at the LOT row level ---
dummies = (pd.get_dummies(classes_long, prefix="PX_on", dtype=int)
    .groupby(level=0).max())

# ensure LOT rows with no meds still appear (all zeros)
dummies = dummies.reindex(treatment_df.index, fill_value=0)
treatment_df = pd.concat([treatment_df, dummies], axis=1)

line_by_line_treatment_data = treatment_df.drop(columns=['TPLAN_TYPE', 'MED_NAME', 'THERAPY_TYPE', 'HAS_ICI', 'LINE', 
                                                         'THERAPY_TYPES', 'ICI_SUBTYPES', 'type_of_rx', 'type_of_rx_sub', 'TPLAN_DX_NAME'])

line_by_line_treatment_data.to_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/categorical_treatment_data_by_line.csv'), index=False)