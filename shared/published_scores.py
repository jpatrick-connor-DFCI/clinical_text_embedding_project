"""Published within-cancer-type prognostic-score catalog (MDCalc-style: IPI,
IMDC, mGPS, LIPI, ALBI, RMH, MSKCC, MELD, CAPRA).

Pure polars expressions and frozen dataclasses only, in the style of
`shared/icd10.py` - no file reading here. `pipelines/preprocessing/
build_published_scores.py` (step 4, not yet built) supplies the input
DataFrame; this module only says how to turn columns into points.

Scoring is complete-case: a score's `points` are null if any of its items
is null, and a `missing_items` list records which item(s) were missing (see
`score_expr`). ECOG/KPS is absent from every PROFILE_DATA and LLM-derived
source (confirmed by `audit_published_score_inputs.py`'s performance-status
check), so IPI, IMDC and MSKCC are only available as **ECOG-free modified**
variants via `ecog_free()` - each keeps a `performance_status` slot so the
exact original version can be added later if a performance-status source
ever becomes available.

Units are the PROFILE-testing lab harmonizer's canonical units (see
`consolidate_dfci_labs`):
  - LDH in U/L
  - albumin and hemoglobin in g/dL
  - CRP, bilirubin, creatinine and calcium in mg/dL
  - ANC, WBC and platelets in 10^3/uL

Derived inputs (each taken from a single specimen day, paired by the
builder):
  - dNLR = ANC / (WBC - ANC)
  - corrected_Ca = Ca + 0.8 * (4 - alb)

Excluded scores (considered and rejected, not merely omitted):
  - NPI: needs tumor grade, which this data doesn't have.
  - ISS: needs beta-2-microglobulin, which the lab harmonizer doesn't map.
  - Khorana: predicts VTE, not survival.
  - BCLC, GPA, ELN: need ECOG, KPS or karyotype.
  - GPS: collinear with mGPS (its ECOG-free sibling would be identical).

Reference limits (ULN/LLN) are provisional constants below, each cited to a
source; `build_published_scores.py` prefers a per-result reference-range
column from LABS if the audit finds one, which removes the LDH
assay-variation problem noted in the plan's risks section.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import polars as pl


class ItemSource(Enum):
    LAB = "lab"
    REGISTRY = "registry"
    ICD = "icd"
    LLM = "llm"
    COHORT = "cohort"
    PERFORMANCE_STATUS = "performance_status"


# --- Provisional reference limits (used unless the audit finds per-result
# reference ranges in LABS) ---
# Sources are the conventional adult ranges cited by each score's validation
# paper / MDCalc, not a single lab's assay-specific range; see risks.
REFERENCE_LIMITS = {
    "LDH_ULN": (245.0, "U/L; standard adult upper limit, e.g. mGPS/LIPI/RMH validation cohorts"),
    "HGB_LLN_MALE": (13.5, "g/dL; WHO adult male lower limit (IMDC/MSKCC)"),
    "HGB_LLN_FEMALE": (12.0, "g/dL; WHO adult female lower limit (IMDC/MSKCC)"),
    "CA_ULN": (10.2, "mg/dL; standard adult upper limit (IMDC)"),
    "ANC_ULN": (7.7, "10^3/uL; standard adult upper limit (IMDC)"),
    "PLT_ULN": (450.0, "10^3/uL; standard adult upper limit (IMDC)"),
}


@dataclass(frozen=True)
class ScoreItem:
    name: str
    inputs: tuple[str, ...]
    point_expr: pl.Expr
    source: ItemSource
    approximate: bool = False


@dataclass(frozen=True)
class RiskGroup:
    label: str
    low: float
    high: float

    def contains(self, points: pl.Expr) -> pl.Expr:
        return points.is_between(self.low, self.high)


@dataclass(frozen=True)
class PublishedScore:
    id: str
    display_name: str
    citation: str
    eligibility: pl.Expr
    items: tuple[ScoreItem, ...]
    risk_groups: tuple[RiskGroup, ...]
    variant: str = "exact"  # "exact" or "modified"
    modification_note: str | None = None
    continuous_formula: pl.Expr | None = None
    performance_status: ScoreItem | None = None
    direction: int = 1  # 1 if higher points/formula = worse prognosis, -1 otherwise

    def __post_init__(self) -> None:
        if self.variant not in ("exact", "modified"):
            raise ValueError(f"{self.id}: variant must be 'exact' or 'modified', got {self.variant!r}")
        if self.variant == "modified" and not self.modification_note:
            raise ValueError(f"{self.id}: modified variant requires a modification_note")


def score_expr(score: PublishedScore) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Return (points_expr, complete_expr, missing_items_expr) for `score`.

    points_expr is null unless every item is non-null (complete-case).
    missing_items_expr is a list of item names that were null, [] if none.
    Continuous-formula scores (ALBI, MELD) use `continuous_formula` directly
    in place of a points sum, but still gate on item completeness.
    """
    if not score.items:
        raise ValueError(f"{score.id}: has no items")

    item_values = [item.point_expr for item in score.items]
    any_missing = pl.any_horizontal([v.is_null() for v in item_values])
    missing_items = pl.concat_list([
        pl.when(v.is_null()).then(pl.lit(item.name)).otherwise(None)
        for item, v in zip(score.items, item_values)
    ]).list.drop_nulls().alias("missing_items")

    if score.continuous_formula is not None:
        raw_points = score.continuous_formula
    else:
        raw_points = pl.sum_horizontal(item_values)

    points = pl.when(any_missing).then(None).otherwise(raw_points).alias(f"{score.id}__points")
    complete = (~any_missing).alias(f"{score.id}__complete")
    return points, complete, missing_items.alias(f"{score.id}__missing_items")


def group_expr(score: PublishedScore, points_col: str) -> pl.Expr:
    """Map `points_col` (a Float/Int column, e.g. `{id}__points`) onto
    `score.risk_groups`; null if no group's [low, high] range contains it."""
    points = pl.col(points_col)
    expr = pl.lit(None, dtype=pl.String)
    for group in reversed(score.risk_groups):
        expr = pl.when(group.contains(points)).then(pl.lit(group.label)).otherwise(expr)
    return expr


def ecog_free(score: PublishedScore) -> PublishedScore:
    """Drop `score.performance_status` from the scored items, leaving the
    resulting points a lower bound relative to the exact score (each
    performance-status item only ever adds points in these scores' original
    formulations). Raises if `score` has no performance_status item to
    drop, since calling this on an already-ECOG-free score is a bug at the
    call site, not a no-op."""
    if score.performance_status is None:
        raise ValueError(f"{score.id}: has no performance_status item to drop")

    note = "ECOG-free: points are a lower bound; original cutpoints applied"
    if score.modification_note:
        note = f"{score.modification_note}; {note}"

    remaining_items = tuple(item for item in score.items if item is not score.performance_status)
    return PublishedScore(
        id=f"{score.id}_noecog" if not score.id.endswith("_noecog") else score.id,
        display_name=f"{score.display_name} (ECOG-free)",
        citation=score.citation,
        eligibility=score.eligibility,
        items=remaining_items,
        risk_groups=score.risk_groups,
        variant="modified",
        modification_note=note,
        continuous_formula=score.continuous_formula,
        performance_status=None,
        direction=score.direction,
    )


# ===========================================================================
# Derived-input expressions (shared across multiple scores)
# ===========================================================================

def dnlr_expr(anc_col: str, wbc_col: str) -> pl.Expr:
    """dNLR = ANC / (WBC - ANC), i.e. ANC / (non-neutrophil WBC)."""
    anc = pl.col(anc_col)
    wbc = pl.col(wbc_col)
    denom = wbc - anc
    return pl.when(denom > 0).then(anc / denom).otherwise(None)


def corrected_calcium_expr(ca_col: str, alb_col: str) -> pl.Expr:
    """corrected_Ca = Ca + 0.8 * (4 - alb), alb in g/dL."""
    return pl.col(ca_col) + 0.8 * (4.0 - pl.col(alb_col))


def hgb_lln_expr(hgb_col: str, gender_col: str) -> pl.Expr:
    """Sex-specific hemoglobin lower limit of normal (REFERENCE_LIMITS)."""
    male_lln, _ = REFERENCE_LIMITS["HGB_LLN_MALE"]
    female_lln, _ = REFERENCE_LIMITS["HGB_LLN_FEMALE"]
    return pl.when(pl.col(gender_col) == "Male").then(pl.lit(male_lln)).otherwise(pl.lit(female_lln))


# ===========================================================================
# mGPS: Glasgow Prognostic Score, modified
# ===========================================================================

def _mgps_points(crp_col: str, alb_col: str) -> pl.Expr:
    crp = pl.col(crp_col)
    alb = pl.col(alb_col)
    return (
        pl.when(crp <= 1.0).then(pl.lit(0))
        .when((crp > 1.0) & (alb >= 3.5)).then(pl.lit(1))
        .when((crp > 1.0) & (alb < 3.5)).then(pl.lit(2))
        .otherwise(None)
    )


def build_mgps(crp_col: str, alb_col: str, eligibility: pl.Expr) -> PublishedScore:
    return PublishedScore(
        id="mgps",
        display_name="Modified Glasgow Prognostic Score",
        citation="McMillan DC, Cancer Treat Rev 2013",
        eligibility=eligibility,
        items=(
            ScoreItem("mgps_crp_alb", (crp_col, alb_col), _mgps_points(crp_col, alb_col), ItemSource.LAB),
        ),
        risk_groups=(
            RiskGroup("0", 0, 0),
            RiskGroup("1", 1, 1),
            RiskGroup("2", 2, 2),
        ),
        variant="exact",
    )


# ===========================================================================
# RMH: Royal Marsden Hospital prognostic score (modified: ICD site count)
# ===========================================================================

def build_rmh(ldh_col: str, alb_col: str, n_met_sites_col: str, eligibility: pl.Expr) -> PublishedScore:
    ldh_uln, _ = REFERENCE_LIMITS["LDH_ULN"]
    items = (
        ScoreItem(
            "rmh_ldh", (ldh_col,),
            pl.when(pl.col(ldh_col) > ldh_uln).then(pl.lit(1)).when(pl.col(ldh_col) <= ldh_uln).then(pl.lit(0)).otherwise(None),
            ItemSource.LAB,
        ),
        ScoreItem(
            "rmh_albumin", (alb_col,),
            pl.when(pl.col(alb_col) < 3.5).then(pl.lit(1)).when(pl.col(alb_col) >= 3.5).then(pl.lit(0)).otherwise(None),
            ItemSource.LAB,
        ),
        ScoreItem(
            "rmh_met_sites", (n_met_sites_col,),
            pl.when(pl.col(n_met_sites_col) >= 3).then(pl.lit(1)).when(pl.col(n_met_sites_col) < 3).then(pl.lit(0)).otherwise(None),
            ItemSource.ICD,
            approximate=True,
        ),
    )
    return PublishedScore(
        id="rmh",
        display_name="Royal Marsden Hospital Prognostic Score",
        citation="Arkenau HT, Br J Cancer 2009",
        eligibility=eligibility,
        items=items,
        risk_groups=(
            RiskGroup("good (0-1)", 0, 1),
            RiskGroup("poor (2-3)", 2, 3),
        ),
        variant="modified",
        modification_note="metastatic-site count derived from ICD-10 C77-C79 codes, not imaging report review",
    )


# ===========================================================================
# LIPI: Lung Immune Prognostic Index
# ===========================================================================

def build_lipi(anc_col: str, wbc_col: str, ldh_col: str, eligibility: pl.Expr) -> PublishedScore:
    ldh_uln, _ = REFERENCE_LIMITS["LDH_ULN"]
    dnlr = dnlr_expr(anc_col, wbc_col)
    items = (
        ScoreItem(
            "lipi_dnlr", (anc_col, wbc_col),
            pl.when(dnlr > 3).then(pl.lit(1)).when(dnlr <= 3).then(pl.lit(0)).otherwise(None),
            ItemSource.LAB,
        ),
        ScoreItem(
            "lipi_ldh", (ldh_col,),
            pl.when(pl.col(ldh_col) > ldh_uln).then(pl.lit(1)).when(pl.col(ldh_col) <= ldh_uln).then(pl.lit(0)).otherwise(None),
            ItemSource.LAB,
        ),
    )
    return PublishedScore(
        id="lipi",
        display_name="Lung Immune Prognostic Index",
        citation="Mezquita L, JAMA Oncol 2018",
        eligibility=eligibility,
        items=items,
        risk_groups=(
            RiskGroup("0 (good)", 0, 0),
            RiskGroup("1 (intermediate)", 1, 1),
            RiskGroup("2 (poor)", 2, 2),
        ),
        variant="exact",
    )


# ===========================================================================
# ALBI: Albumin-Bilirubin grade (HCC)
# ===========================================================================

def build_albi(bili_col: str, alb_col: str, eligibility: pl.Expr) -> PublishedScore:
    # bili_col in mg/dL -> umol/L via *17.1; alb_col in g/dL -> g/L via *10.
    formula = (
        0.66 * (pl.col(bili_col) * 17.1).log10() - 0.085 * (pl.col(alb_col) * 10)
    ).alias("albi__continuous")
    items = (
        ScoreItem("albi_bilirubin", (bili_col,), pl.col(bili_col), ItemSource.LAB),
        ScoreItem("albi_albumin", (alb_col,), pl.col(alb_col), ItemSource.LAB),
    )
    return PublishedScore(
        id="albi",
        display_name="Albumin-Bilirubin Grade",
        citation="Johnson PJ, J Clin Oncol 2015",
        eligibility=eligibility,
        items=items,
        continuous_formula=formula,
        risk_groups=(
            RiskGroup("grade 1", -100, -2.60),
            RiskGroup("grade 2", -2.60, -1.39),
            RiskGroup("grade 3", -1.39, 100),
        ),
        variant="exact",
    )


# ===========================================================================
# MELD (original UNOS formula, no dialysis field) - only if the audit finds
# enough INR coverage.
# ===========================================================================

def build_meld(bili_col: str, inr_col: str, creat_col: str, eligibility: pl.Expr) -> PublishedScore:
    bili = pl.col(bili_col).cast(pl.Float64).clip(lower_bound=1.0)
    inr = pl.col(inr_col).cast(pl.Float64).clip(lower_bound=1.0)
    creat = pl.col(creat_col).cast(pl.Float64).clip(lower_bound=1.0, upper_bound=4.0)
    formula = (
        9.57 * creat.log() + 3.78 * bili.log() + 11.2 * inr.log() + 6.43
    ).alias("meld__continuous")
    items = (
        ScoreItem("meld_bilirubin", (bili_col,), pl.col(bili_col), ItemSource.LAB),
        ScoreItem("meld_inr", (inr_col,), pl.col(inr_col), ItemSource.LAB),
        ScoreItem("meld_creatinine", (creat_col,), pl.col(creat_col), ItemSource.LAB),
    )
    return PublishedScore(
        id="meld",
        display_name="Model for End-Stage Liver Disease (original UNOS)",
        citation="Kamath PS, Hepatology 2001",
        eligibility=eligibility,
        items=items,
        continuous_formula=formula,
        risk_groups=(
            RiskGroup("<10", -1000, 9.999),
            RiskGroup("10-19", 10, 19.999),
            RiskGroup(">=20", 20, 1000),
        ),
        variant="exact",
    )


# ===========================================================================
# CAPRA, modified (no % positive cores; exploratory, diagnosis-time)
# ===========================================================================

def _capra_psa_points(psa_col: str) -> pl.Expr:
    psa = pl.col(psa_col)
    return (
        pl.when(psa <= 6.0).then(pl.lit(0))
        .when(psa <= 10.0).then(pl.lit(1))
        .when(psa <= 20.0).then(pl.lit(2))
        .when(psa <= 30.0).then(pl.lit(3))
        .when(psa > 30.0).then(pl.lit(4))
        .otherwise(None)
    )


def _capra_gleason_points(primary_col: str, secondary_col: str) -> pl.Expr:
    primary = pl.col(primary_col)
    secondary = pl.col(secondary_col)
    return (
        pl.when((primary <= 3) & (secondary <= 3)).then(pl.lit(0))
        .when((primary <= 3) & (secondary >= 4)).then(pl.lit(1))
        .when(primary >= 4).then(pl.lit(3))
        .otherwise(None)
    )


def build_capra_mod(
    psa_col: str, gleason_primary_col: str, gleason_secondary_col: str,
    clinical_t_ge_t3a_col: str, age_col: str, eligibility: pl.Expr,
) -> PublishedScore:
    items = (
        ScoreItem("capra_psa", (psa_col,), _capra_psa_points(psa_col), ItemSource.LAB),
        ScoreItem(
            "capra_gleason", (gleason_primary_col, gleason_secondary_col),
            _capra_gleason_points(gleason_primary_col, gleason_secondary_col), ItemSource.LLM,
        ),
        ScoreItem(
            "capra_clinical_t", (clinical_t_ge_t3a_col,),
            pl.when(pl.col(clinical_t_ge_t3a_col)).then(pl.lit(1)).when(~pl.col(clinical_t_ge_t3a_col)).then(pl.lit(0)).otherwise(None),
            ItemSource.REGISTRY,
        ),
        ScoreItem(
            "capra_age", (age_col,),
            pl.when(pl.col(age_col) >= 50).then(pl.lit(1)).when(pl.col(age_col) < 50).then(pl.lit(0)).otherwise(None),
            ItemSource.COHORT,
        ),
    )
    return PublishedScore(
        id="capra_mod",
        display_name="CAPRA (modified, no % positive cores)",
        citation="Cooperberg MR, J Urol 2005",
        eligibility=eligibility,
        items=items,
        risk_groups=(
            RiskGroup("low (0-2)", 0, 2),
            RiskGroup("intermediate (3-5)", 3, 5),
            RiskGroup("high (6-9)", 6, 9),
        ),
        variant="modified",
        modification_note="percent-positive-cores item omitted (not captured); exploratory, scored at diagnosis not at anchor",
    )


# ===========================================================================
# IPI, ECOG-free modified (aggressive NHL)
# ===========================================================================

def build_ipi(
    age_col: str, ldh_col: str, ann_arbor_stage_ge_3_col: str,
    n_extranodal_sites_col: str, ecog_col: str | None, eligibility: pl.Expr,
) -> PublishedScore:
    ldh_uln, _ = REFERENCE_LIMITS["LDH_ULN"]
    items = [
        ScoreItem(
            "ipi_age", (age_col,),
            pl.when(pl.col(age_col) > 60).then(pl.lit(1)).when(pl.col(age_col) <= 60).then(pl.lit(0)).otherwise(None),
            ItemSource.COHORT,
        ),
        ScoreItem(
            "ipi_ldh", (ldh_col,),
            pl.when(pl.col(ldh_col) > ldh_uln).then(pl.lit(1)).when(pl.col(ldh_col) <= ldh_uln).then(pl.lit(0)).otherwise(None),
            ItemSource.LAB,
        ),
        ScoreItem(
            "ipi_ann_arbor", (ann_arbor_stage_ge_3_col,),
            pl.when(pl.col(ann_arbor_stage_ge_3_col)).then(pl.lit(1)).when(~pl.col(ann_arbor_stage_ge_3_col)).then(pl.lit(0)).otherwise(None),
            ItemSource.REGISTRY,
        ),
        ScoreItem(
            "ipi_extranodal", (n_extranodal_sites_col,),
            pl.when(pl.col(n_extranodal_sites_col) > 1).then(pl.lit(1)).when(pl.col(n_extranodal_sites_col) <= 1).then(pl.lit(0)).otherwise(None),
            ItemSource.ICD,
            approximate=True,
        ),
    ]
    performance_status = None
    if ecog_col is not None:
        performance_status = ScoreItem(
            "ipi_ecog", (ecog_col,),
            pl.when(pl.col(ecog_col) >= 2).then(pl.lit(1)).when(pl.col(ecog_col) < 2).then(pl.lit(0)).otherwise(None),
            ItemSource.PERFORMANCE_STATUS,
        )

    full = PublishedScore(
        id="ipi",
        display_name="International Prognostic Index",
        citation="International Non-Hodgkin's Lymphoma Prognostic Factors Project, NEJM 1993",
        eligibility=eligibility,
        items=tuple(items) + ((performance_status,) if performance_status else ()),
        risk_groups=(
            RiskGroup("0-1 (low)", 0, 1),
            RiskGroup("2 (low-intermediate)", 2, 2),
            RiskGroup("3 (high-intermediate)", 3, 3),
            RiskGroup("4-5 (high)", 4, 5),
        ),
        variant="exact" if performance_status else "modified",
        modification_note=None if performance_status else "no ECOG source available",
        performance_status=performance_status,
    )
    if performance_status is None:
        return PublishedScore(
            id="ipi_noecog",
            display_name=f"{full.display_name} (ECOG-free)",
            citation=full.citation,
            eligibility=full.eligibility,
            items=full.items,
            risk_groups=(
                RiskGroup("0-1 (low)", 0, 1),
                RiskGroup("2 (low-intermediate)", 2, 2),
                RiskGroup("3 (high-intermediate)", 3, 3),
                RiskGroup("4 (high)", 4, 4),
            ),
            variant="modified",
            modification_note="ECOG-free: points are a lower bound; original cutpoints applied",
            performance_status=None,
        )
    return full


# ===========================================================================
# IMDC, ECOG-free modified (advanced RCC)
# ===========================================================================

def build_imdc(
    diagnosis_to_anchor_days_col: str, hgb_col: str, gender_col: str,
    corrected_ca_col: str, anc_col: str, plt_col: str, ecog_col: str | None,
    eligibility: pl.Expr,
) -> PublishedScore:
    ca_uln, _ = REFERENCE_LIMITS["CA_ULN"]
    anc_uln, _ = REFERENCE_LIMITS["ANC_ULN"]
    plt_uln, _ = REFERENCE_LIMITS["PLT_ULN"]
    hgb_lln = hgb_lln_expr(hgb_col, gender_col)

    items = [
        ScoreItem(
            "imdc_time_to_treatment", (diagnosis_to_anchor_days_col,),
            pl.when(pl.col(diagnosis_to_anchor_days_col) < 365).then(pl.lit(1)).when(pl.col(diagnosis_to_anchor_days_col) >= 365).then(pl.lit(0)).otherwise(None),
            ItemSource.REGISTRY,
        ),
        ScoreItem(
            "imdc_hemoglobin", (hgb_col, gender_col),
            pl.when(pl.col(hgb_col) < hgb_lln).then(pl.lit(1)).when(pl.col(hgb_col) >= hgb_lln).then(pl.lit(0)).otherwise(None),
            ItemSource.LAB,
        ),
        ScoreItem(
            "imdc_calcium", (corrected_ca_col,),
            pl.when(pl.col(corrected_ca_col) > ca_uln).then(pl.lit(1)).when(pl.col(corrected_ca_col) <= ca_uln).then(pl.lit(0)).otherwise(None),
            ItemSource.LAB,
        ),
        ScoreItem(
            "imdc_neutrophils", (anc_col,),
            pl.when(pl.col(anc_col) > anc_uln).then(pl.lit(1)).when(pl.col(anc_col) <= anc_uln).then(pl.lit(0)).otherwise(None),
            ItemSource.LAB,
        ),
        ScoreItem(
            "imdc_platelets", (plt_col,),
            pl.when(pl.col(plt_col) > plt_uln).then(pl.lit(1)).when(pl.col(plt_col) <= plt_uln).then(pl.lit(0)).otherwise(None),
            ItemSource.LAB,
        ),
    ]
    performance_status = None
    if ecog_col is not None:
        performance_status = ScoreItem(
            "imdc_ecog", (ecog_col,),
            pl.when(pl.col(ecog_col) >= 1).then(pl.lit(1)).when(pl.col(ecog_col) < 1).then(pl.lit(0)).otherwise(None),
            ItemSource.PERFORMANCE_STATUS,
        )

    risk_groups_exact = (
        RiskGroup("favorable (0)", 0, 0),
        RiskGroup("intermediate (1-2)", 1, 2),
        RiskGroup("poor (>=3)", 3, 6),
    )
    if performance_status is not None:
        return PublishedScore(
            id="imdc",
            display_name="International Metastatic RCC Database Consortium score",
            citation="Heng DYC, Lancet Oncol 2013",
            eligibility=eligibility,
            items=tuple(items) + (performance_status,),
            risk_groups=risk_groups_exact,
            variant="exact",
            performance_status=performance_status,
        )
    return PublishedScore(
        id="imdc_noecog",
        display_name="International Metastatic RCC Database Consortium score (ECOG-free)",
        citation="Heng DYC, Lancet Oncol 2013",
        eligibility=eligibility,
        items=tuple(items),
        risk_groups=risk_groups_exact,
        variant="modified",
        modification_note="ECOG-free: points are a lower bound; original cutpoints applied",
        performance_status=None,
    )


# ===========================================================================
# MSKCC (Motzer), ECOG-free modified (advanced RCC)
# ===========================================================================

def build_mskcc(
    ldh_col: str, hgb_col: str, gender_col: str, corrected_ca_col: str,
    diagnosis_to_anchor_days_col: str, ecog_col: str | None, eligibility: pl.Expr,
) -> PublishedScore:
    ldh_uln, _ = REFERENCE_LIMITS["LDH_ULN"]
    hgb_lln = hgb_lln_expr(hgb_col, gender_col)

    items = [
        ScoreItem(
            "mskcc_ldh", (ldh_col,),
            pl.when(pl.col(ldh_col) > 1.5 * ldh_uln).then(pl.lit(1)).when(pl.col(ldh_col) <= 1.5 * ldh_uln).then(pl.lit(0)).otherwise(None),
            ItemSource.LAB,
        ),
        ScoreItem(
            "mskcc_hemoglobin", (hgb_col, gender_col),
            pl.when(pl.col(hgb_col) < hgb_lln).then(pl.lit(1)).when(pl.col(hgb_col) >= hgb_lln).then(pl.lit(0)).otherwise(None),
            ItemSource.LAB,
        ),
        ScoreItem(
            "mskcc_calcium", (corrected_ca_col,),
            pl.when(pl.col(corrected_ca_col) > 10.0).then(pl.lit(1)).when(pl.col(corrected_ca_col) <= 10.0).then(pl.lit(0)).otherwise(None),
            ItemSource.LAB,
        ),
        ScoreItem(
            "mskcc_time_to_treatment", (diagnosis_to_anchor_days_col,),
            pl.when(pl.col(diagnosis_to_anchor_days_col) < 365).then(pl.lit(1)).when(pl.col(diagnosis_to_anchor_days_col) >= 365).then(pl.lit(0)).otherwise(None),
            ItemSource.REGISTRY,
        ),
    ]
    performance_status = None
    if ecog_col is not None:
        performance_status = ScoreItem(
            "mskcc_ecog", (ecog_col,),
            pl.when(pl.col(ecog_col) >= 1).then(pl.lit(1)).when(pl.col(ecog_col) < 1).then(pl.lit(0)).otherwise(None),
            ItemSource.PERFORMANCE_STATUS,
        )

    risk_groups_exact = (
        RiskGroup("favorable (0)", 0, 0),
        RiskGroup("intermediate (1-2)", 1, 2),
        RiskGroup("poor (>=3)", 3, 5),
    )
    if performance_status is not None:
        return PublishedScore(
            id="mskcc",
            display_name="MSKCC (Motzer) risk model",
            citation="Motzer RJ, J Clin Oncol 1999/2002",
            eligibility=eligibility,
            items=tuple(items) + (performance_status,),
            risk_groups=risk_groups_exact,
            variant="exact",
            performance_status=performance_status,
        )
    return PublishedScore(
        id="mskcc_noecog",
        display_name="MSKCC (Motzer) risk model (ECOG-free)",
        citation="Motzer RJ, J Clin Oncol 1999/2002",
        eligibility=eligibility,
        items=tuple(items),
        risk_groups=risk_groups_exact,
        variant="modified",
        modification_note="ECOG-free: points are a lower bound; original cutpoints applied",
        performance_status=None,
    )


# ===========================================================================
# Fixed subtype-name lists (cite PROFILE_data_processing/cancer_annotations.py)
# ===========================================================================

# DLBCL sub-stratum of AGGR_NHL, verbatim from
# PROFILE_data_processing/cancer_annotations.py:get_lymphoid_subtype_lists (dlbcl_list).
DLBCL_SUBTYPE_NAMES = (
    "DLBCL Associated with Chronic Inflammation",
    "EBV Positive DLBCL, NOS",
    "Primary Cutaneous DLBCL, Leg Type",
    "Primary DLBCL of the central nervous system",
    "Diffuse Large B-Cell Lymphoma, NOS",
    "HHV8 Positive DLBCL, NOS",
    "Intravascular Large B-Cell Lymphoma",
    "Diffuse Large B-Cell Lymphoma",
    "Mediastinal Large B-Cell Lymphoma",
    "Primary CNS Lymphoma",
)

# SCLC and carcinoid exclusion for the LIPI "advanced NSCLC" population.
SCLC_CARCINOID_EXCLUSION_NAMES = (
    "Small Cell Lung Cancer",
    "Small Cell Carcinoma",
    "Carcinoid Tumor",
    "Lung Carcinoid",
)


REGISTRY: dict[str, PublishedScore] = {}


def register(score: PublishedScore) -> PublishedScore:
    if score.id in REGISTRY:
        raise ValueError(f"Duplicate score id: {score.id}")
    REGISTRY[score.id] = score
    return score


def build_catalog(columns: dict[str, str], eligibility: dict[str, pl.Expr]) -> dict[str, PublishedScore]:
    """Construct all 9 catalog scores against concrete builder column names,
    in one call, and return them as a fresh local dict (no shared mutable
    state, so this is safe to call more than once per process -- e.g. once
    at `figures/prep/published_scores.py` import time and again in tests).
    `columns` maps generic item-input names (see each `build_*` call below)
    to the actual column names present in the builder's wide frame;
    `eligibility` maps score id to that score's population filter (`pl.Expr`
    over the same frame).

    This is the single place `build_published_scores.py` (step 4) needs to
    call: the `build_*` functions above are pure and take concrete column
    names, so nothing is assembled until both columns and eligibility
    expressions exist, which only the builder has.
    """
    catalog: dict[str, PublishedScore] = {}

    def _add(score: PublishedScore) -> None:
        catalog[score.id] = score

    _add(build_mgps(columns["crp"], columns["albumin"], eligibility["mgps"]))
    _add(build_rmh(columns["ldh"], columns["albumin"], columns["n_met_sites"], eligibility["rmh"]))
    _add(build_lipi(columns["anc"], columns["wbc"], columns["ldh"], eligibility["lipi"]))
    _add(build_albi(columns["bilirubin"], columns["albumin"], eligibility["albi"]))
    _add(build_meld(columns["bilirubin"], columns["inr"], columns["creatinine"], eligibility["meld"]))
    _add(build_capra_mod(
        columns["psa"], columns["gleason_primary"], columns["gleason_secondary"],
        columns["clinical_t_ge_t3a"], columns["age"], eligibility["capra_mod"],
    ))
    _add(build_ipi(
        columns["age"], columns["ldh"], columns["ann_arbor_stage_ge_3"],
        columns["n_extranodal_sites"], None, eligibility["ipi_noecog"],
    ))
    _add(build_imdc(
        columns["diagnosis_to_anchor_days"], columns["hemoglobin"], columns["gender"],
        columns["corrected_calcium"], columns["anc"], columns["platelets"], None,
        eligibility["imdc_noecog"],
    ))
    _add(build_mskcc(
        columns["ldh"], columns["hemoglobin"], columns["gender"], columns["corrected_calcium"],
        columns["diagnosis_to_anchor_days"], None, eligibility["mskcc_noecog"],
    ))

    return catalog


# Canonical column-name map for the 9-score catalog: both
# build_published_scores.py (real columns on its wide frame) and
# figures/prep/published_scores.py (metadata only -- direction, risk_groups,
# variant, continuous_formula -- no real eligibility data needed there) build
# from this same map, so a single source of truth defines what a "score id"
# means across the two scripts.
CATALOG_COLUMNS = {
    "crp": "crp", "albumin": "albumin", "ldh": "ldh", "n_met_sites": "n_met_sites",
    "anc": "anc", "wbc": "wbc", "bilirubin": "bilirubin", "inr": "inr", "creatinine": "creatinine",
    "psa": "psa", "gleason_primary": "gleason_primary", "gleason_secondary": "gleason_secondary",
    "clinical_t_ge_t3a": "clinical_t_ge_t3a", "age": "age",
    "ann_arbor_stage_ge_3": "ann_arbor_stage_ge_3", "n_extranodal_sites": "n_extranodal_sites",
    "diagnosis_to_anchor_days": "diagnosis_to_anchor_days", "hemoglobin": "hemoglobin", "gender": "gender",
    "corrected_calcium": "corrected_calcium", "platelets": "platelets",
}
CATALOG_SCORE_IDS = (
    "mgps", "rmh", "lipi", "albi", "meld", "capra_mod",
    "ipi_noecog", "imdc_noecog", "mskcc_noecog",
)


def default_catalog() -> dict[str, PublishedScore]:
    """The 9-score catalog with placeholder (always-true) eligibility --
    for callers that only need score metadata (id, direction, risk_groups,
    variant, continuous_formula), such as figures/prep/published_scores.py,
    which reads real per-patient eligibility from the builder's CSV output
    rather than from these expressions."""
    eligibility = {score_id: pl.lit(True) for score_id in CATALOG_SCORE_IDS}
    return build_catalog(CATALOG_COLUMNS, eligibility)
