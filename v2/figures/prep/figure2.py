"""Pre-compute inputs for Figure 2 (text vs base full-cohort prediction).

Writes to FIGURE_DATA_DIR:
- fig2_full_cohort_metrics.csv      scheme, event, event_lbl, phecode_id, phecode_ids,
                                    top_hit_eligible, text_cindex, base_cindex,
                                    text_auc, base_auc  (event_lbl is a human-readable description:
                                    "Death" for death_met/death, "Mets: <site>" for other death_met
                                    events, the cohort diagnosis name for icd3_post/icd4_post codes
                                    (with survival.find_icd_code as fallback), and a cohort
                                    diagnosis name mapped to each phecode_post code (with
                                    code_data/phecode_descriptions.csv as fallback; raw code if both
                                    miss). phecode_id identifies each
                                    row's underlying condition — the event itself for phecode_post,
                                    a representative mapped phecode (via
                                    code_data/icd10_to_phecode_mapping.csv) for
                                    icd3_post/icd4_post, else NA. Deduplication uses the complete
                                    set of mapped phecodes serialized in phecode_ids, not only the
                                    representative display column, so duplicate conditions are not
                                    annotated under two schemes.)
- fig2_within_vs_pan_cancer.csv     stratum, auc_pan, auc_within, delta, cindex_pan, cindex_within,
                                    cindex_delta, n_heldout, is_overall
- fig2_within_vs_pan_treatment.csv  stratum, auc_pan, auc_within, delta, cindex_pan, cindex_within,
                                    cindex_delta, n_heldout, is_overall
- fig2_km_tertiles.csv              DFCI_MRN, text_risk_score, base_risk_score, death, tt_death,
                                    text_tertile, base_tertile
- fig2_km_stage_vs_risk.csv         DFCI_MRN, tt_death, death, text_risk_score, stage_group,
                                    stage_ordinal, risk_quartile   (known-stage cohort)
- fig2_stage_vs_risk_cindex.csv     predictor, cindex, n   (stage ordinal vs text risk score, OS)
- fig2_stage_vs_risk_cindex_by_stage.csv  stage_group, cindex, n   (within-stratum C-index of text
                                    risk score for OS, including pooled I-II; FigS2 annotation)
- fig2_stage_vs_risk_auc.csv        stage_group, mean_auc, n   (within-stratum AUC(t) of text
                                    risk score for OS, including pooled I-II; FigS2 annotation)
- fig2_scheme_delta_topk_{cindex,auc}.csv
                                    category, rank, scheme, event, event_lbl, metric,
                                    text_value, base_value, delta (top-3 events per category by
                                    the selected positive metric delta; category in
                                    {mets, ICD10, phecodes} — mets = death_met minus the literal
                                    "death" event, ICD10 = icd3_post + icd4_post pooled,
                                    phecodes = phecode_post. Categories with fewer than 3
                                    eligible positive-delta events yield fewer rows. Social-
                                    determinant outcomes mapping exclusively to ICD-10-CM Z55-Z65
                                    remain in the full metrics/scatter but are ineligible for top-hit
                                    selection. Cross-scheme dedup: an ICD10 event and a phecode event
                                    with any shared mapping never both appear — ICD10 is ranked first,
                                    so the phecode is skipped for its next-best-delta event.)
- fig2_scheme_event_km_{cindex,auc}.csv
                                    category, scheme, event, event_lbl, DFCI_MRN, text_risk_score,
                                    base_risk_score, event_flag, tt, text_tertile, base_tertile
                                    (held-out risk scores + survival for the events selected in
                                    matching metric-specific top-k CSV, merged from each scheme's
                                    full_cohort_risk_scores/<event>/ output; events missing
                                    risk-score files are skipped)
"""

from __future__ import annotations

import os
import re

import numpy as np
import polars as pl
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

from config import CODE_PATH, RESULTS_PATH, SURV_PATH
from figures.io import save_figure_data
from pipelines.training.slurm_array_utils import filter_event_rows
from schemes import full_cohort_event_dir, full_cohort_risk_dir, list_trained_events, load_embedding_prediction_df
from shared.polars_utils import filter_finite_rows
from shared.stages import STAGE_ORDER, load_stage_map, normalize_stage
from survival import find_icd_code

# code_data/ (CODE_PATH) holds the code lookups this module labels its panels from. They are
# built by notebooks/06a_generate_code_lookups.Rmd (which runs the R generators in
# pipelines/preprocessing/); this module only reads them. See _initialize_code_lookups().
CODE_LOOKUP_FILES = (
    "icd10_to_phecode_mapping.csv",
    "phecode_descriptions.csv",
)


SCHEMES = ["death_met", "icd3_post", "icd4_post", "phecode_post"]

# Major-stage ordering / ordinal encoding for the stage-vs-risk comparison
STAGE_ORDINAL = {"I": 1, "II": 2, "III": 3, "IV": 4}
RISK_QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]

FULL_COHORT_METRIC_COLUMNS = [
    "scheme", "event", "event_lbl", "phecode_id", "phecode_ids",
    "top_hit_eligible", "text_cindex", "base_cindex", "text_auc", "base_auc",
]
# Metastatic-site codes get a custom "Mets: <site>" label instead of an ICD lookup.
MET_SITES = {"brainM", "boneM", "adrenalM", "liverM", "lungM", "nodeM", "peritonealM"}

# Code lookups are generated and loaded once at the start of main().
def _normalize_phecode(code: object) -> str | None:
    """Canonicalize phecodes exactly as endpoint generation does."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    value = re.sub(r"[^0-9.]", "", str(code).strip())
    if not value:
        return None
    if value.count(".") > 1:
        left, right = value.split(".", 1)
        value = f"{left}.{right.replace('.', '')}"
    if "." in value:
        left, right = value.split(".", 1)
        left = left.lstrip("0") or "0"
        right = right.rstrip("0")
        value = left if not right else f"{left}.{right}"
    else:
        value = value.lstrip("0") or "0"
    return value


def _normalize_icd10(code: object) -> str | None:
    """Return the uppercase, undotted ICD-10-CM representation."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    value = re.sub(r"[^A-Z0-9]", "", str(code).strip().upper())
    return value or None


def _lookup_columns(df: pl.DataFrame, *required: str) -> list[str]:
    """Resolve lookup-table columns without depending on capitalization."""
    available = {str(col).strip().lower(): col for col in df.columns}
    missing = [col for col in required if col.lower() not in available]
    if missing:
        raise ValueError(
            f"Missing columns {missing}; available columns: {list(df.columns)}"
        )
    return [available[col.lower()] for col in required]


def _load_icd_descriptions() -> dict[str, str]:
    """Load ICD-10-CM names from the cohort used to build the endpoints.

    The timestamped cohort file carries the diagnosis name alongside every code,
    making it the primary authority for exactly the codes in these figures. This
    also covers newer codes such as R05.1 that older third-party catalogs miss.
    """
    path = os.path.join(SURV_PATH, "timestamped_icd_info.parquet")
    if not os.path.exists(path):
        print(f"  [ICD labels] {path} not found — package lookup only")
        return {}

    out: dict[str, str] = {}
    try:
        icd_df = pl.read_parquet(path)
        code_col, description_col = _lookup_columns(
            icd_df, "DIAGNOSIS_ICD10_CD", "DIAGNOSIS_ICD10_NM"
        )
        for raw_code, raw_description in zip(
            icd_df[code_col].to_list(), icd_df[description_col].to_list()
        ):
            code = _normalize_icd10(raw_code)
            description = (
                "" if raw_description is None else str(raw_description).strip()
            )
            if code and description:
                out.setdefault(code, description)
    except (OSError, ValueError) as exc:
        print(f"  [ICD labels] could not read {path}: {exc} — package lookup only")
        return {}
    return out


def _load_phecode_descriptions() -> dict[str, str]:
    path = os.path.join(CODE_PATH, "phecode_descriptions.csv")
    if not os.path.exists(path):
        print(f"  [phecode labels] {path} not found — phecode events will fall back to raw codes")
        return {}
    df = pl.read_csv(path, infer_schema=False)
    phecode_col, description_col = _lookup_columns(df, "phecode", "description")
    out: dict[str, str] = {}
    for raw_code, raw_description in zip(df[phecode_col].to_list(), df[description_col].to_list()):
        code = _normalize_phecode(raw_code)
        description = (
            "" if raw_description is None else str(raw_description).strip()
        )
        if code and description:
            out.setdefault(code, description)
    return out


def _load_icd10_to_phecode() -> tuple[
    dict[str, frozenset[str]],
    dict[str, str],
    dict[str, frozenset[str]],
]:
    """Load ICD-prefix mappings and phecode labels inferred from mapped ICD names.

    The second result supplies the primary figure label for a phecode directly
    from the original cohort's diagnosis names. The canonical Phecode definition
    table remains a fallback when none of its mapped ICD codes has a cohort name.
    """
    path = os.path.join(CODE_PATH, "icd10_to_phecode_mapping.csv")
    if not os.path.exists(path):
        print(f"  [icd->phecode map] {path} not found — cross-scheme event dedup disabled")
        return {}, {}, {}
    df = pl.read_csv(path, infer_schema=False)
    icd_col, phecode_col = _lookup_columns(df, "icd10_code", "phecode")
    # icd3_post/icd4_post events are level-3/4 ICD prefixes, not full codes; the mapping
    # file is keyed on full codes, so retain every phecode reachable from a mapped
    # code sharing each event prefix.
    out: dict[str, set[str]] = {}
    phecode_cohort_descriptions: dict[str, str] = {}
    phecode_to_icd10: dict[str, set[str]] = {}
    for raw_icd, raw_phecode in zip(df[icd_col].to_list(), df[phecode_col].to_list()):
        icd_code = _normalize_icd10(raw_icd)
        phecode = _normalize_phecode(raw_phecode)
        if not icd_code or not phecode:
            continue
        phecode_to_icd10.setdefault(phecode, set()).add(icd_code)
        icd_description = ICD_DESCRIPTIONS.get(icd_code)
        if icd_description:
            phecode_cohort_descriptions.setdefault(phecode, icd_description)
        for plen in (3, 4):
            prefix = icd_code[:plen]
            if len(prefix) == plen:
                out.setdefault(prefix, set()).add(phecode)
    return (
        {prefix: frozenset(phecodes) for prefix, phecodes in out.items()},
        phecode_cohort_descriptions,
        {phecode: frozenset(codes) for phecode, codes in phecode_to_icd10.items()},
    )


ICD_DESCRIPTIONS: dict[str, str] = {}
PHECODE_DESCRIPTIONS: dict[str, str] = {}
ICD10_TO_PHECODES: dict[str, frozenset[str]] = {}
PHECODE_COHORT_DESCRIPTIONS: dict[str, str] = {}
PHECODE_TO_ICD10: dict[str, frozenset[str]] = {}


def _initialize_code_lookups() -> None:
    """Load all code-description inputs used by Figure 2.

    The lookups themselves are built by the R scripts in pipelines/preprocessing/, driven by
    notebooks/06a_generate_code_lookups.Rmd. They are static reference data — regenerated only
    on a cohort rebuild or a Phecode package upgrade — so this module reads them and never
    invokes R, keeping the whole Python figure tier runnable without Rscript on PATH.

    Missing lookups are a degraded run, not a failure: labels fall back to raw codes and
    cross-scheme event dedup is disabled. _report_lookup_misses() surfaces the cost at the end
    of the run, so warn loudly here rather than failing.
    """
    missing = [name for name in CODE_LOOKUP_FILES
               if not os.path.exists(os.path.join(CODE_PATH, name))]
    if missing:
        print(
            "  [code lookups] WARNING: missing " + ", ".join(missing)
            + " — falling back to raw codes; run notebooks/06a_generate_code_lookups.Rmd"
        )

    global ICD_DESCRIPTIONS
    global PHECODE_DESCRIPTIONS
    global ICD10_TO_PHECODES
    global PHECODE_COHORT_DESCRIPTIONS
    global PHECODE_TO_ICD10
    ICD_DESCRIPTIONS = _load_icd_descriptions()
    PHECODE_DESCRIPTIONS = _load_phecode_descriptions()
    (
        ICD10_TO_PHECODES,
        PHECODE_COHORT_DESCRIPTIONS,
        PHECODE_TO_ICD10,
    ) = _load_icd10_to_phecode()

# Hit/miss counts for the two code-description lookups, so a silent raw-code
# fallback (missing CSV, package not installed, or a genuine normalization
# mismatch) is visible in the run log instead of only showing up as an
# unlabeled code in the figure. See _report_lookup_misses(), called at the
# end of this script.
_ICD_LOOKUP_STATS = {"hit": 0, "miss": 0, "misses": []}
_PHECODE_LOOKUP_STATS = {"hit": 0, "miss": 0, "misses": []}


def _event_label(scheme: str, event: str) -> str:
    """Human-readable label for a single (scheme, event) — mirrors plot_figure_2.R's
    prior R-side event_description(), but resolves real ICD-10 descriptions for
    icd3_post/icd4_post codes via survival.find_icd_code (unavailable in R),
    and real phecode descriptions for phecode_post codes via PHECODE_DESCRIPTIONS."""
    if scheme == "death_met" and event == "death":
        return "Death"
    if scheme == "death_met" and event in MET_SITES:
        return f"Mets: {event.replace('M', '').title()}"
    if scheme == "death_met":
        return event.replace("_", " ").title()
    if scheme in ("icd3_post", "icd4_post"):
        label = ICD_DESCRIPTIONS.get(_normalize_icd10(event), event)
        if label == event:
            label = find_icd_code(event)
        if label == event:
            _ICD_LOOKUP_STATS["miss"] += 1
            _ICD_LOOKUP_STATS["misses"].append(event)
        else:
            _ICD_LOOKUP_STATS["hit"] += 1
        return label
    if scheme == "phecode_post":
        normalized_event = _normalize_phecode(event)
        label = PHECODE_COHORT_DESCRIPTIONS.get(
            normalized_event,
            PHECODE_DESCRIPTIONS.get(normalized_event, event),
        )
        if label == event:
            _PHECODE_LOOKUP_STATS["miss"] += 1
            _PHECODE_LOOKUP_STATS["misses"].append(event)
        else:
            _PHECODE_LOOKUP_STATS["hit"] += 1
        return label
    return event.replace("_", " ").title()[:22]


def _report_lookup_misses() -> None:
    """Print a summary of ICD/phecode description lookup failures. Call once,
    after all _event_label() calls, so raw-code fallbacks (missing generator
    CSV, survival's icd10 package not installed, or a genuine
    normalization mismatch) are visible in the run log instead of only
    showing up silently as unlabeled codes in the figure panels."""
    for name, stats in (("ICD-10", _ICD_LOOKUP_STATS), ("phecode", _PHECODE_LOOKUP_STATS)):
        total = stats["hit"] + stats["miss"]
        if total == 0:
            continue
        if stats["miss"] == 0:
            print(f"  [{name} labels] {stats['hit']}/{total} resolved")
        else:
            examples = ", ".join(stats["misses"][:10])
            more = "" if stats["miss"] <= 10 else f" (+{stats['miss'] - 10} more)"
            print(f"  [{name} labels] {stats['hit']}/{total} resolved, "
                  f"{stats['miss']} fell back to raw code: {examples}{more}")


# Category grouping for the Fig 2 per-scheme Δ C-index barplots + event KM panels: mets =
# death_met minus the literal "death" event, ICD10 = icd3_post + icd4_post pooled,
# phecodes = phecode_post. death_met's "death" event itself has no category (excluded).
# Order matters for _scheme_delta_topk's cross-scheme dedup: ICD10 is ranked before
# phecodes so phecode events already represented by a selected ICD10 event are skipped.
CATEGORY_ORDER = ["mets", "ICD10", "phecodes"]
TOPK_PER_CATEGORY = 3
SCHEME_DELTA_TOPK_COLUMNS = [
    "category", "rank", "scheme", "event", "event_lbl", "metric",
    "text_value", "base_value", "delta",
]
SCHEME_EVENT_KM_COLUMNS = [
    "category", "scheme", "event", "event_lbl", "DFCI_MRN",
    "text_risk_score", "base_risk_score", "event_flag", "tt",
    "text_tertile", "base_tertile",
]


def _event_category(scheme: str, event: str) -> str | None:
    """Map a (scheme, event) row to one of CATEGORY_ORDER, or None to exclude it."""
    if scheme == "death_met":
        return None if event == "death" else "mets"
    if scheme in ("icd3_post", "icd4_post"):
        return "ICD10"
    if scheme == "phecode_post":
        return "phecodes"
    return None


def _event_phecode(scheme: str, event: str) -> str | None:
    """The phecode identifying a row's underlying condition, used to detect the same
    condition surfacing under two schemes (e.g. an ICD-10 code for nutritional
    marasmus and its mapped phecode): the event itself for phecode_post, or one
    deterministic representative from ICD10_TO_PHECODES for icd3_post/icd4_post.
    The complete set used for deduplication is returned by _event_phecodes()."""
    if scheme == "phecode_post":
        return _normalize_phecode(event)
    if scheme in ("icd3_post", "icd4_post"):
        phecodes = _event_phecodes(scheme, event)
        return min(phecodes) if phecodes else None
    return None


def _event_phecodes(scheme: str, event: str) -> frozenset[str]:
    """All phecodes reachable from an event, for cross-scheme deduplication.

    An ICD-10 code can map to multiple phecodes. Keeping only one arbitrary
    mapping allows the same condition to be selected again under phecode_post.
    """
    if scheme == "phecode_post":
        phecode = _normalize_phecode(event)
        return frozenset((phecode,)) if phecode else frozenset()
    if scheme in ("icd3_post", "icd4_post"):
        normalized_event = _normalize_icd10(event)
        return ICD10_TO_PHECODES.get(normalized_event, frozenset())
    return frozenset()


# Social determinants rather than medical diagnoses: education/literacy,
# employment, environment, housing/economic and psychosocial circumstances.
_SOCIAL_DETERMINANT_ICD_PREFIXES = frozenset(
    f"Z{category}" for category in range(55, 66)
)


def _is_social_determinant_icd(code: str) -> bool:
    normalized = _normalize_icd10(code)
    return bool(
        normalized
        and len(normalized) >= 3
        and normalized[:3] in _SOCIAL_DETERMINANT_ICD_PREFIXES
    )


def _top_hit_eligible(scheme: str, event: str) -> bool:
    """Whether an event may be highlighted as an antineoplastic-therapy top hit.

    This is intentionally a narrow, predeclared exclusion rather than a causal
    claim: social-determinant Z55-Z65 outcomes remain plotted but are not eligible
    for top-hit labels/bars. Unknown mappings are retained.
    """
    if scheme in ("icd3_post", "icd4_post"):
        return not _is_social_determinant_icd(event)
    if scheme == "phecode_post":
        phecode = _normalize_phecode(event)
        mapped_icds = PHECODE_TO_ICD10.get(phecode, frozenset())
        return not mapped_icds or any(
            not _is_social_determinant_icd(code) for code in mapped_icds
        )
    return True


WITHIN_VS_PAN_COLUMNS = [
    "stratum", "auc_pan", "auc_within", "delta",
    "cindex_pan", "cindex_within", "cindex_delta",
    "n_heldout", "is_overall",
]
# (subdir, filename, stratum-column) for each within-vs-pan comparison written by Pipeline 3.
_WITHIN_VS_PAN_SPEC = {
    "cancer":    ("pan_vs_within_cancer",    "metrics_by_cancer_type.csv", "CANCER_TYPE"),
    "treatment": ("pan_vs_within_treatment", "metrics_by_treatment.csv",   "TREATMENT"),
}
KM_TERTILE_COLUMNS = [
    "DFCI_MRN", "text_risk_score", "base_risk_score", "death", "tt_death",
    "text_tertile", "base_tertile",
]
STAGE_VS_RISK_COLUMNS = [
    "DFCI_MRN", "tt_death", "death", "text_risk_score",
    "outer_fold", "stage_group", "stage_ordinal", "risk_quartile",
]
STAGE_VS_RISK_CINDEX_COLUMNS = ["predictor", "cindex", "n"]
STAGE_VS_RISK_AUC_COLUMNS = ["stage_group", "mean_auc", "n"]
# Number of evaluation points on the 5th-95th percentile time grid, matching
# within_vs_pan_cancer_models.py's mean-AUC(t) convention.
AUC_TIME_GRID_POINTS = 50


def _full_cohort_metrics() -> pl.DataFrame:
    rows = []
    n_skipped = 0
    for scheme in SCHEMES:
        for ev in list_trained_events(scheme):
            d = full_cohort_event_dir(scheme, ev)
            try:
                text = pl.read_csv(os.path.join(d, "text_test.csv")).row(0, named=True)
                base = pl.read_csv(os.path.join(d, "base_test.csv")).row(0, named=True)
            except (FileNotFoundError, KeyError, IndexError, pl.exceptions.OutOfBoundsError) as e:
                print(f"  [{scheme}:{ev}] skipped — {type(e).__name__}: {e}")
                n_skipped += 1
                continue
            rows.append({
                "scheme": scheme, "event": ev, "event_lbl": _event_label(scheme, ev),
                "phecode_id": _event_phecode(scheme, ev),
                "phecode_ids": ";".join(sorted(_event_phecodes(scheme, ev))),
                "top_hit_eligible": _top_hit_eligible(scheme, ev),
                "text_cindex": text["mean_c_index"], "base_cindex": base["mean_c_index"],
                "text_auc": text["mean_auc(t)"], "base_auc": base["mean_auc(t)"],
            })
    if n_skipped:
        print(f"  total skipped: {n_skipped}")
    if not rows:
        return pl.DataFrame(schema={c: pl.Float64 for c in FULL_COHORT_METRIC_COLUMNS})
    return pl.DataFrame(rows).select(FULL_COHORT_METRIC_COLUMNS)


def _full_cohort_validation_metrics() -> pl.DataFrame:
    """Training/CV-only metrics used for endpoint selection.

    Test metrics remain the reported performance estimates, but must never
    choose which endpoints receive top-hit follow-up.  The text row is the
    AUC-selected hyperparameter row; base_val contains the aggregate CV row.
    """
    rows = []
    for scheme in SCHEMES:
        for ev in list_trained_events(scheme):
            d = full_cohort_event_dir(scheme, ev)
            try:
                text_val = pl.read_csv(os.path.join(d, "text_val.csv"))
                text = (
                    text_val.filter(pl.col("mean_auc(t)").is_finite())
                    .sort("mean_auc(t)", descending=True)
                    .row(0, named=True)
                )
                base = pl.read_csv(os.path.join(d, "base_val.csv")).row(0, named=True)
            except (FileNotFoundError, KeyError, IndexError, pl.exceptions.OutOfBoundsError):
                continue
            rows.append({
                "scheme": scheme, "event": ev, "event_lbl": _event_label(scheme, ev),
                "phecode_id": _event_phecode(scheme, ev),
                "phecode_ids": ";".join(sorted(_event_phecodes(scheme, ev))),
                "top_hit_eligible": _top_hit_eligible(scheme, ev),
                "text_cindex": text["mean_c_index"], "base_cindex": base["mean_c_index"],
                "text_auc": text["mean_auc(t)"], "base_auc": base["mean_auc(t)"],
            })
    return (
        pl.DataFrame(rows).select(FULL_COHORT_METRIC_COLUMNS)
        if rows else pl.DataFrame(schema={c: pl.Float64 for c in FULL_COHORT_METRIC_COLUMNS})
    )


def _merge_risk_with_surv(
    event: str, surv_df: pl.DataFrame, scheme: str = "death_met",
) -> pl.DataFrame | None:
    """Merge a scheme's per-event held-out text/base risk scores with survival labels.

    `surv_df` must already carry the `event`/`tt_{event}` columns for `scheme` (for
    death_met that's death_met_surv_df.parquet; for other schemes, the scheme's
    embedding_prediction_df via load_embedding_prediction_df)."""
    rd = full_cohort_risk_dir(scheme, event)
    tp = os.path.join(rd, "text_risk_scores.csv")
    bp = os.path.join(rd, "base_risk_scores.csv")
    if not (os.path.exists(tp) and os.path.exists(bp)):
        return None
    text_rs = pl.read_csv(tp)
    base_rs = pl.read_csv(bp)
    merged = (text_rs.join(base_rs, on="DFCI_MRN")
                     .join(surv_df.select(["DFCI_MRN", event, f"tt_{event}"]), on="DFCI_MRN"))
    merged = filter_finite_rows(
        merged, ["text_risk_score", "base_risk_score", event, f"tt_{event}"]
    ).filter(pl.col(f"tt_{event}") > 0)
    return merged


def _safe_quantiles(scores: pl.Series, n: int, labels: list[str], label: str) -> pl.Series:
    """qcut into n equal-frequency bins, falling back to rank-based bins when ties at
    boundaries would otherwise raise. Base-model risk often has many ties
    (e.g. age + sex + cancer-type alone collapses many patients onto identical
    linear predictors)."""
    try:
        out = scores.qcut(n, labels=labels, allow_duplicates=False)
        return out.cast(pl.Utf8)
    except pl.exceptions.DuplicateError:
        ranks = scores.rank(method="ordinal")
        out = ranks.qcut(n, labels=labels, allow_duplicates=False).cast(pl.Utf8)
        print(f"  [{label}] qcut hit duplicate edges; used rank-based bins (n={n})")
        return out


def _safe_tertiles(scores: pl.Series, label: str) -> pl.Series:
    """Backwards-compatible low/mid/high tertiles (used by the Fig 2D panel)."""
    return _safe_quantiles(scores, 3, ["low", "mid", "high"], label)


def _km_tertiles(surv_df: pl.DataFrame) -> pl.DataFrame:
    """Patient-level table for the Fig 2D text-vs-base tertile KM panel."""
    m = _merge_risk_with_surv("death", surv_df)
    if m is None:
        return pl.DataFrame(schema={c: pl.Float64 for c in KM_TERTILE_COLUMNS})
    m = m.with_columns([
        _safe_tertiles(m["text_risk_score"], "text").alias("text_tertile"),
        _safe_tertiles(m["base_risk_score"], "base").alias("base_tertile"),
    ])
    return m.select(KM_TERTILE_COLUMNS)


def _scheme_delta_topk(
    metrics: pl.DataFrame, metric: str = "cindex", k: int = TOPK_PER_CATEGORY
) -> pl.DataFrame:
    """Top-k events per category by largest positive text-minus-base metric delta.

    ``metric`` is ``cindex`` or ``auc``; both use the same eligibility and
    cross-scheme deduplication rules. Events with text-minus-base delta <= 0 are
    excluded — a "top" event is never a
    net-negative one. Social-determinant outcomes are not eligible for highlighting.
    Categories with fewer than k eligible positive-delta events yield fewer rows.

    Cross-scheme dedup: ICD10 and phecodes can both surface an event for the same
    underlying condition (e.g. an ICD-10 code for nutritional marasmus and its mapped
    phecode). CATEGORY_ORDER ranks ICD10 before phecodes, so once an ICD10 event is
    selected, any later phecode event mapping to the same phecode is skipped in favor
    of the next-best-delta phecode event."""
    if metrics.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in SCHEME_DELTA_TOPK_COLUMNS})
    if metric not in {"cindex", "auc"}:
        raise ValueError("metric must be 'cindex' or 'auc'")
    text_col, base_col = f"text_{metric}", f"base_{metric}"
    d = metrics
    d = d.with_columns(
        pl.Series("category",
                   [_event_category(s, e) for s, e in zip(d["scheme"].to_list(), d["event"].to_list())])
    )
    if "phecode_id" not in d.columns:
        d = d.with_columns(
            pl.Series("phecode_id",
                       [_event_phecode(s, e) for s, e in zip(d["scheme"].to_list(), d["event"].to_list())])
        )
    d = d.with_columns((pl.col(text_col) - pl.col(base_col)).alias("delta"))
    d = filter_finite_rows(d.drop_nulls(subset=["category"]), ["delta"])
    eligible = [_top_hit_eligible(scheme, event)
                for scheme, event in zip(d["scheme"].to_list(), d["event"].to_list())]
    d = d.filter(pl.Series(eligible))
    d = d.filter(pl.col("delta") > 0)
    rows = []
    seen_phecodes: set[str] = set()
    for category in CATEGORY_ORDER:
        sub = d.filter(pl.col("category") == category).sort("delta", descending=True)
        selected = []
        for r in sub.iter_rows(named=True):
            event_phecodes = _event_phecodes(r["scheme"], r["event"])
            if event_phecodes & seen_phecodes:
                continue
            selected.append(r)
            seen_phecodes.update(event_phecodes)
            if len(selected) == k:
                break
        for rank, r in enumerate(selected, start=1):
            rows.append({
                "category": category, "rank": rank, "scheme": r["scheme"], "event": r["event"],
                "event_lbl": r["event_lbl"], "metric": metric,
                "text_value": r[text_col], "base_value": r[base_col], "delta": r["delta"],
            })
    if not rows:
        return pl.DataFrame(schema={c: pl.Float64 for c in SCHEME_DELTA_TOPK_COLUMNS})
    return pl.DataFrame(rows).select(SCHEME_DELTA_TOPK_COLUMNS)


def _scheme_event_km(topk: pl.DataFrame) -> pl.DataFrame:
    """Patient-level held-out text/base risk scores + survival for the events selected in
    the metric-specific top-k CSV, one row per (event, patient). Events whose held-out
    risk-score files don't yet exist on disk (run_full_cohort_risk_scores.py not yet run
    for that scheme/event) are skipped with a printed note rather than raising."""
    if topk.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in SCHEME_EVENT_KM_COLUMNS})
    frames = []
    surv_cache: dict[str, pl.DataFrame] = {}
    for row in topk.iter_rows(named=True):
        scheme, event = row["scheme"], row["event"]
        if scheme not in surv_cache:
            surv_cache[scheme] = load_embedding_prediction_df(scheme)
        surv_df = filter_event_rows(surv_cache[scheme], event)
        m = _merge_risk_with_surv(event, surv_df, scheme=scheme)
        if m is None:
            print(f"  [{scheme}:{event}] skipped fig2_scheme_event_km — missing risk-score files")
            continue
        if m.is_empty():
            print(f"  [{scheme}:{event}] skipped fig2_scheme_event_km — no overlapping patients")
            continue
        m = m.with_columns([
            pl.lit(row["category"]).alias("category"),
            pl.lit(row["event_lbl"]).alias("event_lbl"),
            _safe_tertiles(m["text_risk_score"], f"{scheme}:{event} text").alias("text_tertile"),
            _safe_tertiles(m["base_risk_score"], f"{scheme}:{event} base").alias("base_tertile"),
        ])
        m = m.rename({event: "event_flag", f"tt_{event}": "tt"})
        m = m.with_columns([
            pl.lit(scheme).alias("scheme"),
            pl.lit(event).alias("event"),
        ])
        frames.append(m.select(SCHEME_EVENT_KM_COLUMNS))
    if not frames:
        return pl.DataFrame(schema={c: pl.Float64 for c in SCHEME_EVENT_KM_COLUMNS})
    return pl.concat(frames, how="diagonal_relaxed")


# Raw, complete stage labels from cancer_stage_df.csv.gz's raw CANCER_STAGE
# column. Mirrors generate_all_non_text_covariates.py:17.
def _major_stage_labels() -> pl.DataFrame:
    """DFCI_MRN -> stage_group (I/II/III/IV) from the raw derived-stage column."""
    mrn_to_stage = load_stage_map()
    if mrn_to_stage is None:
        print("  cancer_stage_df.csv.gz unavailable, no stage data")
        return pl.DataFrame(schema={"DFCI_MRN": pl.Int64, "stage_group": pl.Utf8})
    df = pl.DataFrame({"DFCI_MRN": list(mrn_to_stage.keys()),
                       "stage_group": [normalize_stage(v) for v in mrn_to_stage.values()]})

    df = df.drop_nulls(subset=["stage_group"])
    df = df.filter(pl.col("stage_group").is_in(STAGE_ORDER)).unique(subset=["DFCI_MRN"])
    counts = df["stage_group"].value_counts()
    count_map = dict(zip(counts["stage_group"].to_list(), counts["count"].to_list()))
    summary = "\n".join(f"{s}    {count_map.get(s, 0)}" for s in STAGE_ORDER)
    print("  normalized stage value_counts:\n" + summary)
    return df


def _stage_vs_risk(surv_df: pl.DataFrame) -> pl.DataFrame:
    """Patient-level table for the Fig 2E stage-vs-text-risk KM panel.

    Cohort = patients with a known major stage AND a death text risk score; this same
    set drives both KM subpanels and both c-indices for an apples-to-apples comparison."""
    m = _merge_risk_with_surv("death", surv_df)
    if m is None:
        return pl.DataFrame(schema={c: pl.Float64 for c in STAGE_VS_RISK_COLUMNS})
    stage_df = _major_stage_labels()
    m = m.join(stage_df, on="DFCI_MRN", how="inner")
    if m.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in STAGE_VS_RISK_COLUMNS})
    stage_ordinal_map = STAGE_ORDINAL
    m = m.with_columns([
        pl.col("stage_group").replace_strict(stage_ordinal_map, default=None).alias("stage_ordinal"),
        _safe_quantiles(m["text_risk_score"], 4, RISK_QUARTILE_LABELS, "text_risk").alias("risk_quartile"),
    ])
    return m.select(STAGE_VS_RISK_COLUMNS)


def _stage_vs_risk_cindex(df: pl.DataFrame) -> pl.DataFrame:
    """Concordance of clinical stage (ordinal) vs text risk score for predicting OS,
    on the shared cohort from _stage_vs_risk. Higher estimate = higher risk for both."""
    if df.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in STAGE_VS_RISK_CINDEX_COLUMNS})
    event = df["death"].cast(pl.Boolean, strict=False).to_numpy()
    time = df["tt_death"].cast(pl.Float64, strict=False).to_numpy()
    rows = []
    for predictor, estimate in [
        ("stage", df["stage_ordinal"].cast(pl.Float64, strict=False).to_numpy()),
        ("text_risk", df["text_risk_score"].cast(pl.Float64, strict=False).to_numpy()),
    ]:
        try:
            cidx = concordance_index_censored(event, time, estimate)[0]
        except (ValueError, ZeroDivisionError) as e:
            print(f"  c-index failed for {predictor}: {e}")
            cidx = float("nan")
        rows.append({"predictor": predictor, "cindex": cidx, "n": df.height})
    return pl.DataFrame(rows).select(STAGE_VS_RISK_CINDEX_COLUMNS)


def _stage_vs_risk_cindex_by_stage(df: pl.DataFrame) -> pl.DataFrame:
    """Concordance of the text risk score within each stage and pooled Stages I-II.

    Per-stage analogue of _stage_vs_risk_cindex (which reports one pooled
    cindex for the whole known-stage cohort), matching the per-stage grouping
    of _stage_vs_risk_auc so FigS2 can annotate either metric per stage panel.
    """
    out_cols = STAGE_VS_RISK_AUC_COLUMNS[:1] + ["cindex", "n"]
    if df.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in out_cols})
    rows = []
    groups = [(k, g) for (k,), g in df.group_by(["stage_group"], maintain_order=True)]
    early_stage = df.filter(pl.col("stage_group").is_in(["I", "II"]))
    if not early_stage.is_empty():
        groups.append(("I-II", early_stage))
    for stage_lbl, sub in groups:
        event = sub["death"].cast(pl.Boolean, strict=False).to_numpy()
        time = sub["tt_death"].cast(pl.Float64, strict=False).to_numpy()
        try:
            cidx = concordance_index_censored(
                event, time, sub["text_risk_score"].cast(pl.Float64, strict=False).to_numpy())[0]
        except (ValueError, ZeroDivisionError) as e:
            print(f"  c-index failed for stage {stage_lbl}: {e}")
            cidx = float("nan")
        rows.append({"stage_group": stage_lbl, "cindex": cidx, "n": sub.height})
    return pl.DataFrame(rows).select(out_cols)


def _stage_vs_risk_auc(df: pl.DataFrame) -> pl.DataFrame:
    """Mean time-dependent AUC within each stage and pooled Stages I-II.

    IPCW reference + eval-time grid (5th-95th percentile, 50 points) are fit on the
    pooled known-stage cohort (this table has no train/test split, unlike
    within_vs_pan_cancer_models.py's held-out data), then cumulative_dynamic_auc is
    evaluated per stage subgroup against that shared reference — mirroring the
    project-standard mean-AUC(t) definition used everywhere else in the pipeline.
    """
    if df.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in STAGE_VS_RISK_AUC_COLUMNS})
    if "outer_fold" not in df.columns:
        raise ValueError("Risk scores predate nested CV; regenerate files with outer_fold metadata")

    rows = []
    groups = [(k, g) for (k,), g in df.group_by(["stage_group"], maintain_order=True)]
    early_stage = df.filter(pl.col("stage_group").is_in(["I", "II"]))
    if not early_stage.is_empty():
        groups.append(("I-II", early_stage))
    for stage_lbl, sub in groups:
        fold_aucs, fold_weights = [], []
        for fold in sub["outer_fold"].unique().sort().to_list():
            fold_eval = sub.filter(pl.col("outer_fold") == fold)
            fold_ref = df.filter(pl.col("outer_fold") != fold)
            sub_tt = fold_eval["tt_death"].cast(pl.Float64, strict=False).to_numpy()
            ref_tt = fold_ref["tt_death"].cast(pl.Float64, strict=False).to_numpy()
            lo, hi = np.percentile(ref_tt, [5, 95])
            et = np.linspace(lo, hi, AUC_TIME_GRID_POINTS)
            et = et[(et > sub_tt.min()) & (et < sub_tt.max())]
            if len(et) == 0:
                continue
            try:
                sub_death = fold_eval["death"].cast(pl.Boolean, strict=False).to_numpy()
                y_test = Surv.from_arrays(sub_death, sub_tt)
                y_ref = Surv.from_arrays(
                    fold_ref["death"].cast(pl.Boolean, strict=False).to_numpy(), ref_tt
                )
                fold_aucs.append(float(cumulative_dynamic_auc(
                    y_ref, y_test, fold_eval["text_risk_score"].cast(pl.Float64, strict=False).to_numpy(), et,
                )[1]))
                fold_weights.append(fold_eval.height)
            except (ValueError, ZeroDivisionError) as e:
                print(f"  AUC(t) failed for stage {stage_lbl}: {e}")
        mean_auc = float(np.average(fold_aucs, weights=fold_weights)) if fold_aucs else float("nan")
        rows.append({"stage_group": stage_lbl, "mean_auc": mean_auc, "n": sub.height})
    return pl.DataFrame(rows).select(STAGE_VS_RISK_AUC_COLUMNS)


def _within_vs_pan(kind: str) -> pl.DataFrame:
    """Read a Pipeline-3 pan-vs-within metrics CSV (mean time-dependent AUC + C-index).

    Maps the upstream schema to the unified figure schema, keeps the `Overall`
    row, drops NaN-AUC strata, and applies an n>=30 floor to per-stratum rows
    (the treatment script writes all strata; the cancer script already filters).
    Returns a header-only frame if the upstream file is missing or pre-dates the
    AUC columns (re-run the Pipeline-3 script), so the R panel degrades gracefully.
    C-index columns are passed through when present so the R side can render a
    parallel C-index version of this panel (falls back to NaN otherwise).
    """
    subdir, fname, stratum_col = _WITHIN_VS_PAN_SPEC[kind]
    fp = os.path.join(RESULTS_PATH, subdir, fname)
    if not os.path.exists(fp):
        print(f"  missing {fp}; skipping within-vs-pan {kind}")
        return pl.DataFrame(schema={c: pl.Float64 for c in WITHIN_VS_PAN_COLUMNS})
    df = pl.read_csv(fp)
    if not {stratum_col, "AUC_PAN", "AUC_WITHIN", "N_HELDOUT"}.issubset(df.columns):
        print(f"  {fp} has no AUC columns — re-run the Pipeline-3 script; skipping {kind}")
        return pl.DataFrame(schema={c: pl.Float64 for c in WITHIN_VS_PAN_COLUMNS})
    has_cindex = {"CINDEX_PAN", "CINDEX_WITHIN"}.issubset(df.columns)
    if not has_cindex:
        print(f"  {fp} has no CINDEX columns — re-run the Pipeline-3 script for cindex panel")
    out = pl.DataFrame({
        "stratum": df[stratum_col].cast(pl.Utf8),
        "auc_pan": df["AUC_PAN"],
        "auc_within": df["AUC_WITHIN"],
        "delta": df["DELTA_AUC_WITHIN_MINUS_PAN"] if "DELTA_AUC_WITHIN_MINUS_PAN" in df.columns
                 else (df["AUC_WITHIN"] - df["AUC_PAN"]),
        "cindex_pan": df["CINDEX_PAN"] if has_cindex else pl.Series([None] * df.height, dtype=pl.Float64),
        "cindex_within": df["CINDEX_WITHIN"] if has_cindex else pl.Series([None] * df.height, dtype=pl.Float64),
        "n_heldout": df["N_HELDOUT"],
    })
    out = out.with_columns([
        (pl.col("cindex_within") - pl.col("cindex_pan")).alias("cindex_delta"),
        (pl.col("stratum") == "Overall").alias("is_overall"),
    ])
    out = filter_finite_rows(out, ["auc_pan", "auc_within"])
    out = out.filter(pl.col("is_overall") | (pl.col("n_heldout") >= 30))
    return out.select(WITHIN_VS_PAN_COLUMNS)


def main() -> None:
    _initialize_code_lookups()
    surv_df = pl.read_parquet(os.path.join(SURV_PATH, "death_met_surv_df.parquet"))

    full_cohort_metrics = _full_cohort_metrics()
    validation_metrics = _full_cohort_validation_metrics()
    _report_lookup_misses()
    save_figure_data(full_cohort_metrics, "fig2_full_cohort_metrics.csv")
    save_figure_data(_within_vs_pan("cancer"), "fig2_within_vs_pan_cancer.csv")
    save_figure_data(_within_vs_pan("treatment"), "fig2_within_vs_pan_treatment.csv")
    save_figure_data(_km_tertiles(surv_df), "fig2_km_tertiles.csv")

    stage_vs_risk_df = _stage_vs_risk(surv_df)
    save_figure_data(stage_vs_risk_df, "fig2_km_stage_vs_risk.csv")
    save_figure_data(_stage_vs_risk_cindex(stage_vs_risk_df), "fig2_stage_vs_risk_cindex.csv")
    save_figure_data(_stage_vs_risk_cindex_by_stage(stage_vs_risk_df),
                      "fig2_stage_vs_risk_cindex_by_stage.csv")
    save_figure_data(_stage_vs_risk_auc(stage_vs_risk_df), "fig2_stage_vs_risk_auc.csv")

    for metric in ("cindex", "auc"):
        scheme_delta_topk = _scheme_delta_topk(validation_metrics, metric=metric)
        save_figure_data(scheme_delta_topk, f"fig2_scheme_delta_topk_{metric}.csv")
        save_figure_data(
            _scheme_event_km(scheme_delta_topk),
            f"fig2_scheme_event_km_{metric}.csv",
        )


if __name__ == "__main__":
    main()
