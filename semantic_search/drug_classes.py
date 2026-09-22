"""Binary drug-class definitions for the semantic-search treatment targets.

The six treatment targets ask a one-vs-rest question ("was this patient ever
treated with an ICI?") rather than the multinomial "which of N MOA categories
was the first drug?" question ``first_treatment`` asks.  Class membership is
resolved from the GPT-generated ``MOA_Category`` vocabulary
(``config.MED_CLASSES_FILE``), the same table the ``PX_on_*`` covariates and the
``first_treatment`` target use.

PROVENANCE CAVEAT: ``pipelines/biomarkers/profile_lines`` deliberately does NOT
define ICI exposure from ``MOA_Category`` -- it uses an explicit generic-name
drug list, on the grounds that the MOA mapping is LLM-generated and unvalidated
for exposure definition.  These targets are a descriptive prediction arm, not a
causal exposure, so they accept the MOA vocabulary; but the two definitions are
not interchangeable, and ``ici_moa_concordance`` reports how far they diverge so
the gap is measurable rather than assumed.

MATCHING: the ``MOA_Category`` strings are free-form LLM output, so membership is
tested with normalized regular expressions (case-folded, punctuation and
whitespace collapsed to single spaces) rather than literal equality -- the same
tactic ``figures/prep/figure4._find_ici_column`` uses on the derived column
names.  ``audit_moa_coverage`` prints every category the patterns leave
unmatched, which is how the patterns get tightened against the real table; the
file is cluster-only, so it cannot be enumerated here.
"""

from __future__ import annotations

import os
import re

import polars as pl

from config import MED_CLASSES_FILE

# Targets restricted to one sex.  GENDER is 0=MALE / 1=FEMALE per
# `build_cohort` (`{"MALE": 0, "FEMALE": 1}`); patients with a null GENDER are
# dropped from these two targets rather than guessed at.
FEMALE = 1
MALE = 0

# Each class is a list of regexes tested against the normalized MOA_Category.
# Patterns are deliberately broader than a single canonical spelling because the
# vocabulary is LLM-generated and inconsistent ("ICI", "immune checkpoint
# inhibitor", "checkpoint inhibitors").
DRUG_CLASS_PATTERNS: dict[str, tuple[str, ...]] = {
    # Deliberately NOT a bare `hormon\w* therapy`: "Hormone Therapy" is
    # ambiguous (in a prostate patient it means ADT, not an estrogen agent), and
    # it also swept in "Thyroid Hormone Therapy" and "Growth Hormone Therapy",
    # which are supportive care rather than antineoplastic endocrine therapy.
    # A bare "Hormone Therapy" category therefore lands in neither the estrogen
    # nor the androgen class and shows up in `audit_moa_coverage` as unmatched,
    # which is the honest outcome: the category does not say which axis it is.
    "estrogen": (
        r"\bestrogen\b",
        r"\banti ?estrogen\b",
        r"\bselective estrogen receptor\b",
        r"\bserm\b",
        r"\bserd\b",
        r"\baromatase inhibitor",
        r"\bendocrine therapy\b",
    ),
    "androgen_axis": (
        r"\bandrogen\b",
        r"\banti ?androgen\b",
        r"\bandrogen receptor\b",
        r"\bar (inhibitor|antagonist|signaling)",
        r"\bcyp17\b",
        r"\bgnrh\b",
        r"\blhrh\b",
        r"\badt\b",
        r"\bandrogen deprivation\b",
    ),
    "ici": (
        r"\bici\b",
        r"\bimmune checkpoint\b",
        r"\bcheckpoint inhibitor",
        r"\bpd ?1\b",
        r"\bpd ?l1\b",
        r"\bctla ?4\b",
    ),
    # Deliberately NOT a bare `kinase inhibitor`: that would sweep in
    # serine/threonine kinase inhibitors (BRAF, MEK, mTOR, CDK4/6), which are
    # kinase inhibitors but not TYROSINE kinase inhibitors.  The named receptor
    # families below are all receptor tyrosine kinases, so they are TKIs even
    # when the category omits the word "tyrosine".
    "tki": (
        r"\btki\b",
        r"\btyrosine kinase\b",
        r"\bmulti kinase inhibitor",
        r"\b(egfr|vegfr|alk|ros1|ret|met|kit|pdgfr|fgfr|her2|ntrk|bcr abl|abl|flt3|jak|btk|src)\b"
        r".{0,20}\binhibitor",
    ),
    # A bispecific or trispecific antibody is not a monoclonal antibody, so
    # `_matches` is not enough on its own here -- `classify_moa_category`
    # excludes those (and ADCs) before testing these patterns.
    "monoclonal_antibody": (
        r"\bmonoclonal antibod",
        r"\bmab\b",
        r"\bantibod(y|ies)\b",
    ),
    "adc": (
        r"\badc\b",
        r"\bantibody ?drug conjugate",
        r"\bantibody conjugate",
    ),
}

# Class -> (positive label, negative label).  Wording is fixed here so the
# artifact class names stay stable across runs and so lexicographic order
# (what LabelEncoder applies) is deterministic.
DRUG_CLASS_LABELS: dict[str, tuple[str, str]] = {
    "estrogen": ("ESTROGEN", "NON_ESTROGEN"),
    "androgen_axis": ("ANDROGEN_AXIS", "NON_ANDROGEN_AXIS"),
    "ici": ("ICI", "NON_ICI"),
    "tki": ("TKI", "NON_TKI"),
    "monoclonal_antibody": ("MONOCLONAL_ANTIBODY", "NON_MONOCLONAL_ANTIBODY"),
    "adc": ("ADC", "NON_ADC"),
}

# Class -> the GENDER value the cohort is restricted to, or None for all
# patients.  Estrogen and androgen-axis therapy are asked of one sex only, per
# the analysis specification for this arm.
DRUG_CLASS_SEX: dict[str, int | None] = {
    "estrogen": FEMALE,
    "androgen_axis": MALE,
    "ici": None,
    "tki": None,
    "monoclonal_antibody": None,
    "adc": None,
}

DRUG_CLASSES = tuple(DRUG_CLASS_PATTERNS)

# An ADC is an antibody, so a patient on an ADC alone would otherwise count as a
# monoclonal-antibody positive.  The requested target is "monoclonal antibodies
# vs. all else", which reads as naked mAbs, so ADC categories are excluded from
# the mAb class.  Recorded here rather than inlined so the run metadata can
# state it.
MAB_EXCLUDES_ADC = True

# Likewise a bispecific/trispecific antibody is not a monoclonal antibody, and a
# radioimmunoconjugate is an antibody carrying a payload rather than a naked mAb.
MAB_EXCLUSION_PATTERNS: tuple[str, ...] = (
    r"\bbi ?specific\b",
    r"\btri ?specific\b",
    r"\bradio ?immuno",
    r"\bimmunoconjugate\b",
)


def _normalize(value: str) -> str:
    """Case-fold and collapse punctuation/whitespace to single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _matches(normalized: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, normalized) for pattern in patterns)


def classify_moa_category(category: str | None, drug_class: str) -> bool:
    """Is ``category`` a member of ``drug_class``?

    A null or empty category is False, not null: an unmapped drug is evidence of
    absence for this class, which is what a one-vs-rest label requires.
    """
    if drug_class not in DRUG_CLASS_PATTERNS:
        raise ValueError(
            f"Unknown drug class {drug_class!r}; choose from {list(DRUG_CLASSES)}"
        )
    if category is None:
        return False
    normalized = _normalize(str(category))
    if not normalized:
        return False
    if drug_class == "monoclonal_antibody":
        if MAB_EXCLUDES_ADC and _matches(normalized, DRUG_CLASS_PATTERNS["adc"]):
            return False
        if _matches(normalized, MAB_EXCLUSION_PATTERNS):
            return False
    return _matches(normalized, DRUG_CLASS_PATTERNS[drug_class])


def load_med_classes(med_classes_path: str | None = None) -> pl.DataFrame:
    """Load the GPT class table as ``(MED_NAME, MOA_Category)``.

    ``unique(keep="last")`` mirrors ``build_treatment_by_line_df`` so a drug
    resolves to the same class the ``PX_on_*`` covariates assign it.
    """
    med_classes_path = med_classes_path or MED_CLASSES_FILE
    if not os.path.exists(med_classes_path):
        raise FileNotFoundError(
            f"GPT-generated medication classes not found: {med_classes_path}"
        )
    med_classes = pl.read_csv(med_classes_path)
    for column in ("MED_NAME", "MOA_Category"):
        if column not in med_classes.columns:
            raise ValueError(f"{med_classes_path} is missing {column}")
    return med_classes.select("MED_NAME", "MOA_Category").unique(
        "MED_NAME", keep="last"
    )


def drug_class_members(
    drug_class: str, med_classes: pl.DataFrame | None = None
) -> pl.DataFrame:
    """The ``(MED_NAME, MOA_Category)`` rows belonging to ``drug_class``."""
    med_classes = med_classes if med_classes is not None else load_med_classes()
    return med_classes.filter(
        pl.col("MOA_Category").map_elements(
            lambda value: classify_moa_category(value, drug_class),
            return_dtype=pl.Boolean,
            skip_nulls=False,
        )
    )


def audit_moa_coverage(med_classes: pl.DataFrame | None = None) -> pl.DataFrame:
    """Every distinct ``MOA_Category`` with the classes it matched.

    The class table lives only on the cluster, so the patterns in
    ``DRUG_CLASS_PATTERNS`` cannot be validated against the real vocabulary from
    a checkout.  Run this once there: a category with ``matched_classes == ""``
    and a large ``n_drugs`` is a pattern gap worth closing, and a category
    matching two classes is an overlap worth reviewing.  ``n_drugs`` counts rows
    of the class table, not patients.
    """
    med_classes = med_classes if med_classes is not None else load_med_classes()

    def _labels(value: str | None) -> str:
        hits = [name for name in DRUG_CLASSES if classify_moa_category(value, name)]
        return ",".join(hits)

    return (
        med_classes.group_by("MOA_Category")
        .agg(pl.len().alias("n_drugs"))
        .with_columns(
            pl.col("MOA_Category")
            .map_elements(_labels, return_dtype=pl.String, skip_nulls=False)
            .alias("matched_classes")
        )
        .with_columns(
            (pl.col("matched_classes") == "").alias("unmatched"),
            pl.col("matched_classes")
            .str.split(",")
            .list.len()
            .alias("n_matched_classes"),
        )
        .with_columns(
            pl.when(pl.col("matched_classes") == "")
            .then(0)
            .otherwise(pl.col("n_matched_classes"))
            .alias("n_matched_classes")
        )
        .sort(["unmatched", "n_drugs"], descending=[True, True])
    )


def ici_moa_concordance(med_classes: pl.DataFrame | None = None) -> pl.DataFrame:
    """Cross-tab the MOA-derived ICI class against the curated ICI drug list.

    ``profile_lines.ICI_DRUGS`` is the project's auditable ICI definition. These
    targets use the MOA vocabulary instead, so this reports the disagreement
    rather than leaving it implicit: rows off the diagonal are drugs one
    definition calls an ICI and the other does not.
    """
    from pipelines.biomarkers.profile_lines import is_ici_expr

    med_classes = med_classes if med_classes is not None else load_med_classes()
    return (
        med_classes.with_columns(
            pl.col("MOA_Category")
            .map_elements(
                lambda value: classify_moa_category(value, "ici"),
                return_dtype=pl.Boolean,
                skip_nulls=False,
            )
            .alias("moa_is_ici"),
            is_ici_expr("MED_NAME").alias("curated_is_ici"),
        )
        .group_by(["curated_is_ici", "moa_is_ici"])
        .agg(
            pl.len().alias("n_drugs"),
            pl.col("MED_NAME").sort().head(10).alias("example_drugs"),
        )
        .sort(["curated_is_ici", "moa_is_ici"], descending=[True, True])
    )


def main() -> None:
    """Print the coverage audit and the ICI concordance cross-check.

    Run on the cluster, where MED_CLASSES_FILE exists:

        python -m semantic_search.drug_classes

    Read the unmatched rows first: a category with a large ``n_drugs`` and no
    matched class is a gap in ``DRUG_CLASS_PATTERNS``, and closing it changes
    who counts as positive, so it must be settled before the targets are
    trained.
    """
    med_classes = load_med_classes()
    print(f"{med_classes.height:,} drugs in the class table\n")

    audit = audit_moa_coverage(med_classes)
    matched = audit.filter(~pl.col("unmatched"))
    unmatched = audit.filter(pl.col("unmatched"))

    print("MATCHED categories:")
    with pl.Config(tbl_rows=-1, fmt_str_lengths=60):
        print(matched.select("MOA_Category", "n_drugs", "matched_classes"))

    overlapping = audit.filter(pl.col("n_matched_classes") > 1)
    if overlapping.height:
        print("\nWARNING: categories matching MORE THAN ONE class:")
        with pl.Config(tbl_rows=-1, fmt_str_lengths=60):
            print(overlapping.select("MOA_Category", "n_drugs", "matched_classes"))

    print(
        f"\nUNMATCHED categories ({unmatched.height} categories, "
        f"{int(unmatched.get_column('n_drugs').sum() or 0):,} drugs) -- "
        "every one of these is a negative for all six targets:"
    )
    with pl.Config(tbl_rows=-1, fmt_str_lengths=60):
        print(unmatched.select("MOA_Category", "n_drugs"))

    print("\nPer-class drug counts:")
    for name in DRUG_CLASSES:
        members = drug_class_members(name, med_classes)
        print(f"  {name:<22} {members.height:>5,} drugs")

    print("\nICI concordance (curated ICI_DRUGS list vs. MOA category):")
    with pl.Config(tbl_rows=-1, fmt_str_lengths=80):
        print(ici_moa_concordance(med_classes))


if __name__ == "__main__":
    main()
