"""Assemble the per-patient clinical table the clusters are compared against.

Reads the materialized covariate files under FEATURE_PATH (built by
`1_data/01_preprocessing`) rather than re-running the builders in-process: this
arm runs strictly downstream of that stage, and reading the files keeps stage 3
cheap and keeps every arm comparing against the same frozen covariates.

Variables are grouped into FAMILIES, and BH-FDR is applied within family (see
semantic_search.stats.add_fdr_within).  Each loader returns
(frame, continuous_vars, categorical_vars) and degrades to an empty frame with a
printed warning when its source file is absent, so a partial cohort build still
yields a partial characterization rather than a crash.

What this repo does NOT have, and so is absent here: labs, vitals,
race/ethnicity, smoking, ECOG/performance status, and PFS.  GENDER and age are
the only demographics; OS is the only time-to-event endpoint.
"""

from __future__ import annotations

import os

import polars as pl

from config import FEATURE_PATH, SURV_PATH
from semantic_search.common import NOTE_TYPES, PATIENT_KEY
from shared.icd10 import MET_SITE_GROUPS
from shared.stages import load_stage_map, normalize_stage

FAMILIES = [
    "demographics", "cancer_type", "stage", "met_burden",
    "treatment", "somatic", "prs", "note_volume",
]

SURV_FILE = "death_met_surv_df.parquet"
# Wide families are capped by prevalence: a somatic marker present in 5 patients
# cannot separate clusters, and thousands of such columns would dominate the
# family's FDR denominator and bury the markers that can.
MIN_SOMATIC_PREVALENCE = 25
MIN_TREATMENT_PREVALENCE = 25


def _empty(reason: str, name: str) -> tuple[pl.DataFrame, list[str], list[str]]:
    print(f"  {name} unavailable ({reason})", flush=True)
    return pl.DataFrame({PATIENT_KEY: []}, schema={PATIENT_KEY: pl.Int64}), [], []


def _read_csv_gz(filename: str, name: str):
    path = os.path.join(FEATURE_PATH, filename)
    if not os.path.exists(path):
        return None, _empty("file not found", name)
    try:
        return pl.read_csv(path), None
    except (OSError, ValueError, pl.exceptions.PolarsError) as e:
        return None, _empty(type(e).__name__, name)


def load_demographics() -> tuple[pl.DataFrame, list[str], list[str]]:
    path = os.path.join(SURV_PATH, SURV_FILE)
    if not os.path.exists(path):
        return _empty("file not found", "demographics")
    df = pl.read_parquet(path)
    cont = [c for c in ["AGE_AT_TREATMENTSTART"] if c in df.columns]
    cat = [c for c in ["GENDER"] if c in df.columns]
    return df.select([PATIENT_KEY] + cont + cat), cont, cat


def load_survival() -> pl.DataFrame:
    path = os.path.join(SURV_PATH, SURV_FILE)
    if not os.path.exists(path):
        print("  survival unavailable (file not found)", flush=True)
        return pl.DataFrame({PATIENT_KEY: []}, schema={PATIENT_KEY: pl.Int64})
    return pl.read_parquet(path).select([PATIENT_KEY, "tt_death", "death"])


def load_cancer_type() -> tuple[pl.DataFrame, list[str], list[str]]:
    df, failure = _read_csv_gz("cancer_type_df.csv.gz", "cancer_type")
    if failure is not None:
        return failure
    if "CANCER_TYPE" not in df.columns:
        return _empty("missing CANCER_TYPE column", "cancer_type")
    return df.select([PATIENT_KEY, "CANCER_TYPE"]), [], ["CANCER_TYPE"]


def load_stage() -> tuple[pl.DataFrame, list[str], list[str]]:
    """Major stage I-IV, via the shared normalizer rather than a local regex."""
    stage_map = load_stage_map()
    if stage_map is None:
        return _empty("load_stage_map returned None", "stage")
    rows = [(mrn, normalize_stage(raw)) for mrn, raw in stage_map.items()]
    rows = [(mrn, stg) for mrn, stg in rows if stg is not None]
    if not rows:
        return _empty("no stage value normalized", "stage")
    df = pl.DataFrame({
        PATIENT_KEY: [r[0] for r in rows],
        "CANCER_STAGE": [r[1] for r in rows],
    })
    return df, [], ["CANCER_STAGE"]


def load_met_burden() -> tuple[pl.DataFrame, list[str], list[str]]:
    df, failure = _read_csv_gz("met_burden_df.csv.gz", "met_burden")
    if failure is not None:
        return failure
    cont = [c for c in ["N_MET_SITES"] if c in df.columns]
    cat = [f"MET_SITE_{g}" for g in MET_SITE_GROUPS if f"MET_SITE_{g}" in df.columns]
    if not cont and not cat:
        return _empty("no met columns", "met_burden")
    return df.select([PATIENT_KEY] + cont + cat), cont, cat


def load_treatment() -> tuple[pl.DataFrame, list[str], list[str]]:
    df, failure = _read_csv_gz("categorical_treatment_data_by_line.csv.gz", "treatment")
    if failure is not None:
        return failure
    cat = [c for c in df.columns if c.startswith("PX_on_")]
    if not cat:
        return _empty("no PX_on_* columns", "treatment")
    df = df.unique(subset=PATIENT_KEY, keep="first").select([PATIENT_KEY] + cat)
    cat = _prevalent_binary_cols(df, cat, MIN_TREATMENT_PREVALENCE)
    return df.select([PATIENT_KEY] + cat), [], cat


def load_somatic() -> tuple[pl.DataFrame, list[str], list[str]]:
    df, failure = _read_csv_gz("complete_somatic_data_df.csv.gz", "somatic")
    if failure is not None:
        return failure
    suffixes = ("_SNV", "_AMP", "_DEL", "_SV", "_FUSION")
    cat = [c for c in df.columns if c.upper().endswith(suffixes)]
    if not cat:
        return _empty("no alteration columns", "somatic")
    df = df.unique(subset=PATIENT_KEY, keep="first").select([PATIENT_KEY] + cat)
    cat = _prevalent_binary_cols(df, cat, MIN_SOMATIC_PREVALENCE)
    if not cat:
        return _empty(f"no marker reached {MIN_SOMATIC_PREVALENCE} patients", "somatic")
    return df.select([PATIENT_KEY] + cat), [], cat


def load_prs() -> tuple[pl.DataFrame, list[str], list[str]]:
    df, failure = _read_csv_gz("complete_germline_data_df.csv.gz", "prs")
    if failure is not None:
        return failure
    cont = [c for c in df.columns if c.upper().startswith("PGS")]
    if not cont:
        return _empty("no PGS* columns", "prs")
    df = df.unique(subset=PATIENT_KEY, keep="first")
    return df.select([PATIENT_KEY] + cont), cont, []


def _prevalent_binary_cols(df: pl.DataFrame, cols: list[str], min_n: int) -> list[str]:
    """Keep binary columns with at least `min_n` positives and at least one
    negative -- anything rarer cannot separate clusters and only inflates the
    family's FDR denominator."""
    keep = []
    for c in cols:
        positives = df.get_column(c).cast(pl.Float64, strict=False).fill_null(0.0)
        n_pos = int((positives > 0).sum())
        if min_n <= n_pos < df.height:
            keep.append(c)
    return keep


def note_volume(notes_meta: pl.DataFrame) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Documentation-intensity covariates -- a confound check, not a finding.

    An unweighted mean over a patient's notes encodes how much was written about
    them as well as what was written.  If clusters separate mainly on these, the
    partition is a documentation artifact; this makes that visible rather than
    letting it masquerade as a clinical signal.
    """
    per_type = notes_meta.group_by([PATIENT_KEY, "NOTE_TYPE"]).len(name="n")
    wide = per_type.pivot(on="NOTE_TYPE", index=PATIENT_KEY, values="n")
    rename = {nt: f"N_NOTES_{nt.upper()}" for nt in NOTE_TYPES if nt in wide.columns}
    wide = wide.rename(rename).fill_null(0)

    span = notes_meta.group_by(PATIENT_KEY).agg([
        pl.len().alias("N_NOTES_TOTAL"),
        (pl.col("NOTE_DATETIME").max() - pl.col("NOTE_DATETIME").min())
        .dt.total_days().cast(pl.Float64).alias("NOTE_SPAN_DAYS"),
        pl.col("NOTE_DATETIME").dt.year().mean().cast(pl.Float64).alias("MEAN_NOTE_YEAR"),
    ])

    df = span.join(wide, on=PATIENT_KEY, how="left").fill_null(0)
    cont = [c for c in df.columns if c != PATIENT_KEY]
    return df, cont, []


def load_all(notes_meta: pl.DataFrame | None = None) -> dict[str, tuple]:
    """Every family as {name: (frame, continuous_vars, categorical_vars)}.

    `note_volume` is included only when `notes_meta` is supplied, since it is the
    one family derived from the note metadata rather than a covariate file.
    """
    print("Loading clinical families...", flush=True)
    families = {
        "demographics": load_demographics(),
        "cancer_type": load_cancer_type(),
        "stage": load_stage(),
        "met_burden": load_met_burden(),
        "treatment": load_treatment(),
        "somatic": load_somatic(),
        "prs": load_prs(),
    }
    if notes_meta is not None:
        families["note_volume"] = note_volume(notes_meta)

    for name, (df, cont, cat) in families.items():
        print(f"  {name}: {df.height:,} patients, "
              f"{len(cont)} continuous + {len(cat)} categorical", flush=True)
    return families
