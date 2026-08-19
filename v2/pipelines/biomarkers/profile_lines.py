"""Lines of therapy and ICI exposure derived from PROFILE_DATA.

Replaces the lab-owned ALL_MEDICATION_LINES.csv (`config.MED_LINES_FILE`) as the
source of the two variables the biomarker causal design turns on: the line-of-
therapy index with its start date, and whether a line contains an immune
checkpoint inhibitor.

Everything here is a pure function over Polars frames built from
`profile_sources.unpivot_medications_summary()`, so it is unit-testable without
cluster data (see tests/test_profile_lines.py).

Two limitations of the underlying source, both stated because they bound what
this module can claim:

1. MEDICATIONS_SUMMARY carries **7 fixed drug slots** per patient
   (`profile_sources.MED_SLOTS`). A patient on more than 7 antineoplastics has
   their late lines truncated. This is harmless for the first-line-only cohort
   design in build_line_matched_cohort.py -- line 1 is the earliest and is never
   the one lost -- but it is load-bearing if that design is ever extended to
   later lines.
2. A restart of the same drug after a long gap counts as a new line. That is the
   intended reading for a re-challenge, but it over-counts an interrupted
   maintenance course.
"""

import polars as pl

# --- ICI definition -------------------------------------------------------
#
# Exposure is defined by an explicit, auditable drug list matched against
# MED_NCI_PREFERRED_NM, not by MED_ANTINEO_DRUG_CATEG or the GPT-generated
# MOA_Category. Both of those are reported by `ici_concordance_report` as a
# cross-check, but neither defines the exposure: the PROFILE category vocabulary
# is not documented in this repo, and the MOA mapping is LLM-generated and
# unvalidated for exposure definition.
#
# Names are NCI preferred (generic) names, lowercase; matching is
# case-insensitive substring so combination formulations
# ("nivolumab and relatlimab-rmbw") and biosimilar/product suffixes still hit.
ICI_DRUGS: frozenset[str] = frozenset({
    # --- PD-1 ---
    "pembrolizumab",    # 2014
    "nivolumab",        # 2014
    "cemiplimab",       # 2018
    "dostarlimab",      # 2021
    "tislelizumab",     # 2024
    "toripalimab",      # 2023
    "retifanlimab",     # 2023
    "cosibelimab",      # 2024
    "penpulimab",       # 2025
    # --- PD-L1 ---
    "atezolizumab",     # 2016
    "avelumab",         # 2017
    "durvalumab",       # 2017
    # --- CTLA-4 ---
    "ipilimumab",       # 2011
    "tremelimumab",     # 2022
    # --- LAG-3 (combination partner; only ever co-administered with an anti-PD-1) ---
    "relatlimab",       # 2022
})

# Days from a line's start within which a further drug start is read as part of
# the same regimen rather than as the next line. 28 days collapses combination
# therapy (ipilimumab + nivolumab, pembrolizumab + a platinum doublet) into one
# line while keeping a genuine switch separate.
LINE_WINDOW_DAYS = 28


def is_ici_expr(name_col: str = "DRUG") -> pl.Expr:
    """Boolean expression: does `name_col` name an immune checkpoint inhibitor?

    Case-insensitive substring match against `ICI_DRUGS`, so combination
    formulations and product suffixes match. Null names are False, not null.
    """
    lowered = pl.col(name_col).fill_null("").str.to_lowercase()
    matches = [lowered.str.contains(drug, literal=True) for drug in sorted(ICI_DRUGS)]
    combined = matches[0]
    for expr in matches[1:]:
        combined = combined | expr
    return combined


def derive_lines_of_therapy(
    long: pl.DataFrame, window_days: int = LINE_WINDOW_DAYS
) -> pl.DataFrame:
    """Group a long medication frame into lines of therapy, one row per line.

    Args:
        long: `(DFCI_MRN, DRUG, START_DT, ...)` as returned by
            `profile_sources.unpivot_medications_summary()`. Extra columns are
            ignored. Rows with a null START_DT are dropped.
        window_days: a drug starting within this many days of the current line's
            start joins that line; the first start beyond it opens the next line.

    Returns:
        One row per `(DFCI_MRN, LINE)`, sorted, with columns:
          LINE                 1-based, ordered by line start date
          treatment_start_date min START_DT across the line's drugs
          HAS_ICI              1 if any drug in the line is an ICI, else 0
          n_drugs, drugs       line composition, diagnostics only
    """
    if window_days < 0:
        raise ValueError(f"window_days must be non-negative, got {window_days}")

    ordered = (
        long.select(["DFCI_MRN", "DRUG", "START_DT"])
        .drop_nulls(subset=["DFCI_MRN", "START_DT"])
        .sort(["DFCI_MRN", "START_DT", "DRUG"])
    )
    if ordered.is_empty():
        return pl.DataFrame(
            schema={
                "DFCI_MRN": long.schema.get("DFCI_MRN", pl.Int64),
                "LINE": pl.Int64,
                "treatment_start_date": long.schema.get("START_DT", pl.Date),
                "HAS_ICI": pl.Int64,
                "n_drugs": pl.UInt32,
                "drugs": pl.String,
            }
        )

    # A line is opened by the first drug start that is more than `window_days`
    # after the start of the line currently open. That reference start is itself
    # defined by the assignment, so it cannot be expressed as a fixed-lag diff;
    # the running comparison below carries it forward per patient.
    line_index = _assign_line_index(ordered, window_days)

    lines = (
        line_index.group_by(["DFCI_MRN", "LINE"])
        .agg(
            pl.col("START_DT").min().alias("treatment_start_date"),
            is_ici_expr("DRUG").any().cast(pl.Int64).alias("HAS_ICI"),
            pl.col("DRUG").n_unique().alias("n_drugs"),
            pl.col("DRUG").fill_null("").unique().sort().str.join("; ").alias("drugs"),
        )
        .sort(["DFCI_MRN", "LINE"])
    )
    return lines


def _assign_line_index(ordered: pl.DataFrame, window_days: int) -> pl.DataFrame:
    """Attach a 1-based LINE column to a per-patient date-sorted drug frame.

    Walks each patient's starts once, carrying the open line's reference start
    date forward. A vectorized `cum_sum` over a fixed-lag gap cannot express
    this: whether a start opens a new line depends on the line assignment made
    for the preceding rows, not on the immediately preceding row's date. With at
    most 7 rows per patient the walk is cheap.
    """
    mrns = ordered.get_column("DFCI_MRN").to_list()
    starts = ordered.get_column("START_DT").to_list()

    line_numbers: list[int] = []
    current_mrn = None
    current_line = 0
    line_start = None
    for mrn, start in zip(mrns, starts):
        if mrn != current_mrn:
            current_mrn = mrn
            current_line = 1
            line_start = start
        elif (start - line_start).days > window_days:
            current_line += 1
            line_start = start
        line_numbers.append(current_line)

    return ordered.with_columns(pl.Series("LINE", line_numbers, dtype=pl.Int64))


def ici_concordance_report(long: pl.DataFrame, med_classes: pl.DataFrame | None = None) -> None:
    """Print how the curated ICI list agrees with the two categorical vocabularies.

    Args:
        long: `unpivot_medications_summary()` output; needs DRUG and, for the
            first cross-tab, DRUG_CATEG (MED_ANTINEO_DRUG_CATEG).
        med_classes: optional `MED_CLASSES_FILE` frame (MED_NAME, MOA_Category).
    """
    flagged = long.with_columns(is_ici_expr("DRUG").alias("is_ici"))

    print(f"  Drug rows: {flagged.height}, flagged ICI by curated list: {int(flagged['is_ici'].sum())}")

    ici_names = (
        flagged.filter(pl.col("is_ici"))
        .get_column("DRUG")
        .value_counts(sort=True)
    )
    print(f"\n  Distinct drug names matched by the curated list ({ici_names.height}):")
    print(ici_names)

    if "DRUG_CATEG" in flagged.columns:
        print("\n  curated ICI flag x MED_ANTINEO_DRUG_CATEG:")
        print(
            flagged.group_by(["is_ici", "DRUG_CATEG"])
            .agg(pl.len().alias("n"))
            .sort(["is_ici", "n"], descending=[False, True])
        )

    if med_classes is not None:
        joined = flagged.join(
            med_classes.unique("MED_NAME", keep="last"),
            left_on="DRUG", right_on="MED_NAME", how="left",
        )
        print("\n  curated ICI flag x MOA_Category (GPT-generated):")
        print(
            joined.group_by(["is_ici", "MOA_Category"])
            .agg(pl.len().alias("n"))
            .sort(["is_ici", "n"], descending=[False, True])
        )
