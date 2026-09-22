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
from semantic_search.drug_classes import (
    DRUG_CLASS_LABELS,
    DRUG_CLASS_SEX,
    DRUG_CLASSES,
    classify_moa_category,
    load_med_classes,
)

# One binary target per drug class, named `treatment_<class>`.  These replace
# the multinomial `first_treatment` question with six one-vs-rest questions; see
# `semantic_search.drug_classes` for how membership is resolved and for the
# provenance caveat on the MOA vocabulary.
DRUG_CLASS_TARGETS = [f"treatment_{name}" for name in DRUG_CLASSES]

TARGETS = [
    "cancer_type",
    "stage",
    "first_treatment",
    "prostate_subtype",
    "n_lines",
    *DRUG_CLASS_TARGETS,
]
TARGET_DISPLAY_NAMES = {
    "cancer_type": "Cancer type",
    "stage": "Cancer stage",
    "first_treatment": "First treatment type",
    "prostate_subtype": "Prostate phenotype",
    "n_lines": "Total lines of therapy",
    "treatment_estrogen": "Estrogen therapy (female patients)",
    "treatment_androgen_axis": "Androgen-axis therapy (male patients)",
    "treatment_ici": "Immune checkpoint inhibitor",
    "treatment_tki": "Tyrosine kinase inhibitor",
    "treatment_monoclonal_antibody": "Monoclonal antibody",
    "treatment_adc": "Antibody-drug conjugate",
}

# Upper edges of the line-count bins; the final bin is open-ended ("4+ lines").
# Binned rather than regressed because the prediction arm is classifier-only
# (LabelEncoder -> predict_proba -> AUC), and because MEDICATIONS_SUMMARY caps a
# patient at 7 drug slots (profile_sources.MED_SLOTS), so the raw count is not
# trustworthy at its top end.  An open-ended top bin absorbs that ceiling: a
# 7-slot-truncated patient lands in "4+ lines", which is where they belong
# regardless of the true count.  Labels are worded so that lexicographic order
# (what LabelEncoder applies) matches clinical order.
N_LINES_BIN_EDGES = (1, 2, 3)


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


def _bin_line_count(count: int) -> str:
    """Map a line count to its ordered bin label."""
    for edge in N_LINES_BIN_EDGES:
        if count <= edge:
            return f"{edge} line" if edge == 1 else f"{edge} lines"
    return f"{N_LINES_BIN_EDGES[-1] + 1}+ lines"


def load_n_lines_target(*, cohort_path: str | None = None) -> pl.DataFrame:
    """Total lines of therapy per patient, binned into ordered classes.

    Lines come from ``profile_lines.derive_lines_of_therapy``, the same
    derivation the biomarker arm uses, so a "line" means the same thing in both
    places (a 28-day regimen window; see ``LINE_WINDOW_DAYS``).

    CENSORING: every patient in the cohort is labeled, including those still in
    follow-up, per the analysis decision for this arm.  The count is therefore
    "lines observed so far", not a completed lifetime total, and for a living
    patient it is a lower bound.  Because follow-up duration bounds the
    observable count, a short-follow-up patient is pushed toward the low bins
    for a reason unrelated to their disease, and a model can exploit that.
    ``load_n_lines_followup_stats`` reports follow-up by bin so this confound
    can be quantified alongside the model's accuracy; it is not adjusted for
    here.

    TRUNCATION: the 7 fixed MEDICATIONS_SUMMARY slots cap the derivable count,
    so the top bin is open-ended (see ``N_LINES_BIN_EDGES``).
    """
    from pipelines.biomarkers.profile_lines import derive_lines_of_therapy
    from pipelines.preprocessing import profile_sources as ps

    cohort_path = cohort_path or os.path.join(SURV_PATH, "cohort_df.parquet")
    if not os.path.exists(cohort_path):
        raise FileNotFoundError(f"Cohort artifact not found: {cohort_path}")
    cohort = pl.read_parquet(cohort_path)
    if PATIENT_KEY not in cohort.columns:
        raise ValueError(f"{cohort_path} is missing {PATIENT_KEY}")

    lines = derive_lines_of_therapy(ps.unpivot_medications_summary())
    counts = (
        lines.join(cohort.select(PATIENT_KEY).unique(), on=PATIENT_KEY, how="semi")
        .group_by(PATIENT_KEY)
        .agg(pl.col("LINE").max().alias("n_lines"))
    )
    # An inner join, not a zero-fill: a cohort patient with no derivable
    # medication row has an unknown line count, not a count of zero.  Every
    # patient here has at least one line by construction.
    return _finalize(
        counts.select(
            PATIENT_KEY,
            pl.col("n_lines")
            .map_elements(_bin_line_count, return_dtype=pl.String)
            .alias("label"),
        ),
        "derive_lines_of_therapy",
    )


def load_n_lines_followup_stats(*, cohort_path: str | None = None) -> pl.DataFrame:
    """Follow-up duration per line-count bin, for auditing the censoring confound.

    If the low bins show systematically shorter follow-up than the high bins,
    a classifier separating them may be reading follow-up length rather than
    disease course.  Reported in the run metadata; see ``load_n_lines_target``.
    """
    cohort_path = cohort_path or os.path.join(SURV_PATH, "cohort_df.parquet")
    labels = load_n_lines_target(cohort_path=cohort_path)
    cohort = pl.read_parquet(cohort_path)
    required = {"first_treatment_date", "death_date", "last_contact_date"}
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"{cohort_path} is missing columns: {sorted(missing)}")

    death_expr = (
        pl.col("death").cast(pl.Float64)
        if "death" in cohort.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    followed = (
        cohort.select(
            PATIENT_KEY,
            "first_treatment_date",
            pl.coalesce(["death_date", "last_contact_date"]).alias("_end"),
            death_expr.alias("_death"),
        )
        .with_columns(
            (pl.col("_end") - pl.col("first_treatment_date"))
            .dt.total_days()
            .alias("follow_up_days")
        )
        .join(labels, on=PATIENT_KEY, how="inner")
    )
    return (
        followed.group_by("label")
        .agg(
            pl.len().alias("n"),
            pl.col("follow_up_days").median().alias("median_follow_up_days"),
            pl.col("follow_up_days").quantile(0.25).alias("q25_follow_up_days"),
            pl.col("follow_up_days").quantile(0.75).alias("q75_follow_up_days"),
            pl.col("_death").mean().alias("death_fraction"),
        )
        .sort("label")
    )


def load_drug_class_target(
    drug_class: str,
    *,
    cohort_path: str | None = None,
    med_classes_path: str | None = None,
) -> pl.DataFrame:
    """Ever-exposure to ``drug_class`` as a binary label over the cohort.

    A patient is positive if ANY drug in ANY of their lines maps to a
    ``MOA_Category`` in the class, negative otherwise.  Drugs come from
    ``profile_sources.unpivot_medications_summary`` -- all 7 medication slots,
    not just the anchor -- so this is "ever treated", not "first treated".

    LEAKAGE: because exposure may occur at any line, a positive label can be
    caused by a drug started long after the notes a model reads.  That makes
    this target NOT leak-free against the ``pretreatment`` note window, unlike
    ``first_treatment``, which is anchored to the same first-treatment date the
    window is cut at.  A ``pretreatment`` run of these targets is therefore
    measuring "do the pre-treatment notes predict the patient's whole future
    treatment course", which is a legitimate question but not the same one; the
    ``alltime`` window has no such separation at all.  Recorded in run metadata
    as ``drug_class_exposure="ever_any_line"``.

    TRUNCATION: MEDICATIONS_SUMMARY carries 7 fixed drug slots
    (``profile_sources.MED_SLOTS``), so a patient on more than 7 antineoplastics
    has later drugs dropped.  That can only turn a true positive into a false
    negative, never the reverse, so the positive class is under-counted at the
    margin.

    SEX RESTRICTION: ``estrogen`` is asked of female patients only and
    ``androgen_axis`` of male patients only (``DRUG_CLASS_SEX``), using the
    cohort's ``GENDER`` column.  Patients with a null ``GENDER`` are dropped
    from those two targets rather than assigned a sex.
    """
    from pipelines.preprocessing import profile_sources as ps

    if drug_class not in DRUG_CLASS_LABELS:
        raise ValueError(
            f"Unknown drug class {drug_class!r}; choose from {list(DRUG_CLASSES)}"
        )
    cohort_path = cohort_path or os.path.join(SURV_PATH, "cohort_df.parquet")
    if not os.path.exists(cohort_path):
        raise FileNotFoundError(f"Cohort artifact not found: {cohort_path}")
    cohort = pl.read_parquet(cohort_path)
    if PATIENT_KEY not in cohort.columns:
        raise ValueError(f"{cohort_path} is missing {PATIENT_KEY}")

    required_sex = DRUG_CLASS_SEX[drug_class]
    if required_sex is not None:
        if "GENDER" not in cohort.columns:
            raise ValueError(
                f"{cohort_path} is missing GENDER, required by the {drug_class} target"
            )
        cohort = cohort.filter(pl.col("GENDER") == required_sex)
        if cohort.is_empty():
            raise ValueError(
                f"No cohort patients with GENDER == {required_sex} for {drug_class}"
            )

    med_classes = load_med_classes(med_classes_path)
    members = set(
        med_classes.filter(
            pl.col("MOA_Category").map_elements(
                lambda value: classify_moa_category(value, drug_class),
                return_dtype=pl.Boolean,
                skip_nulls=False,
            )
        )
        .get_column("MED_NAME")
        .to_list()
    )

    # A patient with no derivable medication row has an unknown course, not a
    # negative one, so the semi-join keeps only patients we actually observed
    # treated -- the same stance `load_n_lines_target` takes.
    exposure = (
        ps.unpivot_medications_summary()
        .join(cohort.select(PATIENT_KEY).unique(), on=PATIENT_KEY, how="semi")
        .group_by(PATIENT_KEY)
        .agg(pl.col("DRUG").is_in(members).any().alias("_exposed"))
    )
    positive, negative = DRUG_CLASS_LABELS[drug_class]
    return _finalize(
        exposure.select(
            PATIENT_KEY,
            pl.when(pl.col("_exposed").fill_null(False))
            .then(pl.lit(positive))
            .otherwise(pl.lit(negative))
            .alias("label"),
        ),
        f"drug_class:{drug_class}",
    )


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
    if target == "n_lines":
        return load_n_lines_target(cohort_path=cohort_path)
    if target in DRUG_CLASS_TARGETS:
        return load_drug_class_target(
            target.removeprefix("treatment_"),
            cohort_path=cohort_path,
            med_classes_path=med_classes_path,
        )
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
