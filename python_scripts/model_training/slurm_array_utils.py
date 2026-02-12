"""Slurm Array Utils script for model training workflows."""

import os
from typing import Any

import numpy as np
import pandas as pd

DATA_PATH = "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/"
SURV_PATH = os.path.join(DATA_PATH, "time-to-event_analysis/")
FEATURE_PATH = os.path.join(DATA_PATH, "clinical_and_genomic_features/")

SCHEME_CONFIG: dict[str, dict[str, str]] = {
    "icd3": {
        "embedding_file": "level_3_ICD_embedding_prediction_df.csv",
        "results_dir": "level_3_ICD_results",
    },
    "icd4": {
        "embedding_file": "level_4_ICD_embedding_prediction_df.csv",
        "results_dir": "level_4_ICD_results",
    },
    "phecode": {
        "embedding_file": "phecode_embedding_prediction_df.csv",
        "results_dir": "phecode_results",
    },
    "death_met": {
        "embedding_file": "death_met_embedding_prediction_df.csv",
        "results_dir": "death_met_results",
    },
}

MET_EVENTS = {"brainM", "boneM", "adrenalM", "liverM", "lungM", "nodeM", "peritonealM"}
DEFAULT_ALPHAS = np.logspace(-5, 0, 25).tolist()
DEFAULT_L1_RATIOS = [0.5, 1.0]


def _ensure_scheme(scheme: str) -> str:
    if scheme not in SCHEME_CONFIG:
        valid = ", ".join(sorted(SCHEME_CONFIG))
        raise ValueError(f"Unsupported scheme '{scheme}'. Valid options: {valid}")
    return scheme


def _get_n_jobs(default: int | None = None) -> int:
    if default is not None:
        return int(default)
    return int(os.getenv("SLURM_CPUS_PER_TASK", "1"))


def load_embedding_prediction_df(scheme: str) -> pd.DataFrame:
    scheme = _ensure_scheme(scheme)
    fp = os.path.join(SURV_PATH, SCHEME_CONFIG[scheme]["embedding_file"])
    return pd.read_csv(fp)


def load_cancer_type_df() -> tuple[pd.DataFrame, list[str]]:
    cancer_type_df = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_type_df.csv"))
    type_cols = [col for col in cancer_type_df.columns if col.startswith("CANCER_TYPE_")]
    return cancer_type_df, type_cols


def get_output_dir(scheme: str, run_type: str) -> str:
    scheme = _ensure_scheme(scheme)
    if run_type not in {"full_cohort", "feature_comps"}:
        raise ValueError("run_type must be one of {'full_cohort', 'feature_comps'}")
    out = os.path.join(SURV_PATH, "results", SCHEME_CONFIG[scheme]["results_dir"], run_type)
    os.makedirs(out, exist_ok=True)
    return out


def get_events_from_df(df: pd.DataFrame) -> list[str]:
    return [col[3:] for col in df.columns if col.startswith("tt_")]


def build_full_prediction_df(scheme: str) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    emb_df = load_embedding_prediction_df(scheme)
    cancer_type_df, type_cols = load_cancer_type_df()
    full_prediction_df = emb_df.merge(cancer_type_df[["DFCI_MRN"] + type_cols], on="DFCI_MRN", how="left")
    embed_cols = [col for col in full_prediction_df.columns if ("EMBEDDING" in col or "2015" in col)]
    events = get_events_from_df(full_prediction_df)
    return full_prediction_df, type_cols, embed_cols, events


def filter_event_rows(full_prediction_df: pd.DataFrame, event: str) -> pd.DataFrame:
    tt_col = f"tt_{event}"
    if tt_col not in full_prediction_df.columns:
        raise ValueError(f"Missing column '{tt_col}' in prediction dataframe.")
    event_pred_df = full_prediction_df.loc[full_prediction_df[tt_col] > 0].copy()
    if event == "brainM" and "CANCER_TYPE_BRAIN" in event_pred_df.columns:
        event_pred_df = event_pred_df.loc[~event_pred_df["CANCER_TYPE_BRAIN"]].copy()
    return event_pred_df


def load_feature_modalities_df(
    scheme: str,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any], dict[str, list[str]]]:
    emb_df = load_embedding_prediction_df(scheme)

    mrn_stage_df = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_stage_df.csv"))
    cancer_type_df = pd.read_csv(os.path.join(FEATURE_PATH, "cancer_type_df.csv"))
    somatic_df = pd.read_csv(os.path.join(FEATURE_PATH, "complete_somatic_data_df.csv"))
    prs_df = pd.read_csv(os.path.join(FEATURE_PATH, "complete_germline_data_df.csv"))
    treatment_df = pd.read_csv(os.path.join(FEATURE_PATH, "categorical_treatment_data_by_line.csv"))
    labs_df = pd.read_csv(os.path.join(FEATURE_PATH, "mean_lab_vals_pre_first_treatment.csv"))

    stage_cols = [col for col in mrn_stage_df.columns if col.startswith("CANCER_STAGE_")]
    type_cols = [col for col in cancer_type_df.columns if col.startswith("CANCER_TYPE_")]
    somatic_cols = [col for col in somatic_df.columns if col.endswith(('_AMP', '_DEL', '_CNV', '_SNV'))]
    prs_cols = [col for col in prs_df.columns if "PGS" in col]
    treatment_cols = [col for col in treatment_df.columns if col.startswith("PX_on_")]
    embed_cols = [col for col in emb_df.columns if ("EMBEDDING" in col or "2015" in col)]
    labs_cols = [col for col in labs_df.columns if col != "DFCI_MRN"]

    full_prediction_df = (
        emb_df.merge(somatic_df[["DFCI_MRN"] + somatic_cols], on="DFCI_MRN")
        .merge(prs_df[["DFCI_MRN"] + prs_cols], on="DFCI_MRN")
        .merge(
            treatment_df.loc[treatment_df["treatment_line"] == 1, ["DFCI_MRN"] + treatment_cols],
            on="DFCI_MRN",
        )
        .merge(cancer_type_df[["DFCI_MRN"] + type_cols], on="DFCI_MRN")
        .merge(mrn_stage_df[["DFCI_MRN"] + stage_cols], on="DFCI_MRN")
        .merge(labs_df[["DFCI_MRN"] + labs_cols], on="DFCI_MRN")
    )

    modality_cfg: dict[str, dict[str, Any]] = {
        "stage": {
            "continuous_vars": ["AGE_AT_TREATMENTSTART"],
            "penalized_cols": stage_cols,
            "pca_config": None,
        },
        "treatment": {
            "continuous_vars": ["AGE_AT_TREATMENTSTART"],
            "penalized_cols": treatment_cols,
            "pca_config": None,
        },
        "labs": {
            "continuous_vars": ["AGE_AT_TREATMENTSTART"] + labs_cols,
            "penalized_cols": labs_cols,
            "pca_config": None,
        },
        "somatic": {
            "continuous_vars": ["AGE_AT_TREATMENTSTART"],
            "penalized_cols": somatic_cols,
            "pca_config": None,
        },
        "prs": {
            "continuous_vars": ["AGE_AT_TREATMENTSTART"],
            "penalized_cols": prs_cols,
            "pca_config": {"PGS": (prs_cols, 1500)},
        },
        "text": {
            "continuous_vars": ["AGE_AT_TREATMENTSTART"] + embed_cols,
            "penalized_cols": embed_cols,
            "pca_config": None,
        },
    }
    feature_cols = {
        "type_cols": type_cols,
        "embed_cols": embed_cols,
    }
    return full_prediction_df, type_cols, embed_cols, modality_cfg, feature_cols


def parse_manifest_line(line: str, expected_fields: int) -> list[str]:
    fields = line.rstrip("\n").split("\t")
    if len(fields) != expected_fields:
        raise ValueError(f"Expected {expected_fields} tab-separated fields, got {len(fields)}: {line!r}")
    return fields


__all__ = [
    "DATA_PATH",
    "SURV_PATH",
    "FEATURE_PATH",
    "SCHEME_CONFIG",
    "MET_EVENTS",
    "DEFAULT_ALPHAS",
    "DEFAULT_L1_RATIOS",
    "_get_n_jobs",
    "build_full_prediction_df",
    "filter_event_rows",
    "get_events_from_df",
    "get_output_dir",
    "load_embedding_prediction_df",
    "load_feature_modalities_df",
    "parse_manifest_line",
]
