"""Patient-level labels used by the semantic-search prediction arm.

Every loader returns exactly two columns, ``DFCI_MRN`` and ``label``, with at
most one row per patient.  Patients absent from a source remain unlabeled; in
particular, patients absent from the cohort-complete LLM prostate artifact are
never assumed to be conventional prostate cancer.
"""

from __future__ import annotations

import os

import polars as pl

from config import AVPC_NEPC_LABELS_PATH, MED_CLASSES_FILE, SURV_PATH
from semantic_search.clinical_data import load_cancer_type, load_stage
from semantic_search.common import PATIENT_KEY

TARGETS = ["cancer_type", "stage", "first_treatment", "prostate_subtype"]
TARGET_DISPLAY_NAMES = {
    "cancer_type": "Cancer type",
    "stage": "Cancer stage",
    "first_treatment": "First treatment type",
    "prostate_subtype": "Prostate phenotype",
}


def _finalize(frame: pl.DataFrame, source: str) -> pl.DataFrame:
    required = {PATIENT_KEY, "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")

    frame = (
        frame.select(
            pl.col(PATIENT_KEY).cast(pl.Int64, strict=False),
            pl.col("label").cast(pl.String, strict=False).str.strip_chars(),
        )
        .drop_nulls([PATIENT_KEY, "label"])
        .filter(pl.col("label") != "")
    )
    conflicts = (
        frame.group_by(PATIENT_KEY)
        .agg(pl.col("label").n_unique().alias("n_labels"))
        .filter(pl.col("n_labels") > 1)
    )
    if conflicts.height:
        examples = conflicts.get_column(PATIENT_KEY).head(5).to_list()
        raise ValueError(
            f"{source} has conflicting labels for {conflicts.height} patients; "
            f"example MRNs: {examples}"
        )
    return frame.unique(subset=PATIENT_KEY, keep="first").sort(PATIENT_KEY)


def load_cancer_type_target() -> pl.DataFrame:
    frame, _, _ = load_cancer_type()
    if "CANCER_TYPE" not in frame.columns:
        raise FileNotFoundError("Cancer-type labels are unavailable")
    return _finalize(
        frame.select(PATIENT_KEY, pl.col("CANCER_TYPE").alias("label")),
        "cancer_type_df.csv.gz",
    )


def load_stage_target() -> pl.DataFrame:
    frame, _, _ = load_stage()
    if "CANCER_STAGE" not in frame.columns:
        raise FileNotFoundError("Cancer-stage labels are unavailable")
    return _finalize(
        frame.select(PATIENT_KEY, pl.col("CANCER_STAGE").alias("label")),
        "cancer_stage_df.csv.gz",
    )


def load_first_treatment_target(
    *,
    cohort_path: str | None = None,
    granularity: str = "category",
    med_classes_path: str | None = None,
) -> pl.DataFrame:
    """Load the first medication slot's treatment class (or drug name).

    ``build_cohort._first_treatment_dates`` obtains both fields from the same
    chronologically first medication row, so this does not reconstruct an
    anchor differently from the rest of the project.

    At ``category`` granularity the label is the condensed GPT-generated
    ``MOA_Category``, joined onto the frozen ``ANCHOR_DRUG`` exactly as
    ``generate_all_non_text_covariates.build_treatment_by_line_df`` joins it
    onto ``MED_NAME``, including the ``OTHER`` fill for drugs the class table
    does not cover.  This keeps the prediction target in the same vocabulary as
    the ``PX_on_*`` treatment covariates; PROFILE's raw ``ANCHOR_DRUG_CATEG``
    (``MED_ANTINEO_DRUG_CATEG``) is a different, uncondensed vocabulary and is
    deliberately not used here.
    """
    if granularity not in {"category", "drug"}:
        raise ValueError("granularity must be 'category' or 'drug'")
    cohort_path = cohort_path or os.path.join(SURV_PATH, "cohort_df.parquet")
    if not os.path.exists(cohort_path):
        raise FileNotFoundError(f"First-treatment cohort artifact not found: {cohort_path}")
    frame = pl.read_parquet(cohort_path)
    if "ANCHOR_DRUG" not in frame.columns:
        raise ValueError(f"{cohort_path} is missing ANCHOR_DRUG")

    if granularity == "drug":
        label = pl.col("ANCHOR_DRUG")
    else:
        med_classes_path = med_classes_path or MED_CLASSES_FILE
        if not os.path.exists(med_classes_path):
            raise FileNotFoundError(
                f"GPT-generated medication classes not found: {med_classes_path}"
            )
        # `unique(keep="last")` and the OTHER fill mirror
        # build_treatment_by_line_df, so an anchor drug resolves to the same
        # class the PX_on_* covariates assign it.
        med_classes = pl.read_csv(med_classes_path)
        for column in ("MED_NAME", "MOA_Category"):
            if column not in med_classes.columns:
                raise ValueError(f"{med_classes_path} is missing {column}")
        med_classes = med_classes.unique("MED_NAME", keep="last")
        frame = frame.join(
            med_classes.select("MED_NAME", "MOA_Category"),
            left_on="ANCHOR_DRUG",
            right_on="MED_NAME",
            how="left",
        )
        label = pl.col("MOA_Category").fill_null("OTHER")

    return _finalize(
        frame.select(
            PATIENT_KEY,
            label.cast(pl.String, strict=False).str.to_uppercase().alias("label"),
        ),
        cohort_path,
    )


def _flag(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def load_prostate_subtype_target(*, labels_path: str | None = None) -> pl.DataFrame:
    """Derive mutually exclusive NEPC/AVPC/conventional labels.

    NEPC takes precedence when a patient also meets AVPC criteria, matching the
    upstream timeline labeler's NEPC-precedence rule.  Only rows present in the
    upstream, cohort-complete artifact are used.
    """
    labels_path = labels_path or AVPC_NEPC_LABELS_PATH
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            "LLM AVPC/NEPC labels not found. Set AVPC_NEPC_LABELS_PATH or pass "
            f"--avpc-nepc-labels. Looked for: {labels_path}"
        )
    frame = pl.read_parquet(labels_path)
    required = {PATIENT_KEY, "has_avpc", "has_nepc_timeline"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{labels_path} is missing columns: {sorted(missing)}")

    frame = frame.with_columns(
        pl.col("has_avpc")
        .map_elements(_flag, return_dtype=pl.Boolean, skip_nulls=False)
        .alias("_avpc"),
        pl.col("has_nepc_timeline")
        .map_elements(_flag, return_dtype=pl.Boolean, skip_nulls=False)
        .alias("_nepc"),
    ).with_columns(
        pl.when(pl.col("_nepc"))
        .then(pl.lit("NEPC"))
        .when(pl.col("_avpc"))
        .then(pl.lit("AVPC"))
        .otherwise(pl.lit("CONVENTIONAL"))
        .alias("label")
    )
    return _finalize(frame.select(PATIENT_KEY, "label"), labels_path)


def load_target(
    target: str,
    *,
    avpc_nepc_labels_path: str | None = None,
    treatment_granularity: str = "category",
    cohort_path: str | None = None,
    med_classes_path: str | None = None,
) -> pl.DataFrame:
    if target == "cancer_type":
        return load_cancer_type_target()
    if target == "stage":
        return load_stage_target()
    if target == "first_treatment":
        return load_first_treatment_target(
            cohort_path=cohort_path,
            granularity=treatment_granularity,
            med_classes_path=med_classes_path,
        )
    if target == "prostate_subtype":
        return load_prostate_subtype_target(labels_path=avpc_nepc_labels_path)
    raise ValueError(f"Unknown target {target!r}; choose from {TARGETS}")


def collapse_rare_treatment_labels(
    labels: pl.DataFrame, min_class_n: int
) -> tuple[pl.DataFrame, list[str]]:
    """Fold sparse first-treatment categories into ``OTHER``.

    This is deliberately treatment-only. Stage levels and the three prostate
    phenotypes retain their clinical meaning, while cancer type is already
    collapsed upstream by ``build_cancer_type_df``.
    """
    if min_class_n < 1:
        raise ValueError("min_class_n must be positive")
    counts = labels.group_by("label").len(name="n")
    rare = sorted(
        counts.filter(pl.col("n") < min_class_n).get_column("label").to_list()
    )
    if not rare:
        return labels, []
    collapsed = labels.with_columns(
        pl.when(pl.col("label").is_in(rare))
        .then(pl.lit("OTHER"))
        .otherwise(pl.col("label"))
        .alias("label")
    )
    return collapsed, rare
