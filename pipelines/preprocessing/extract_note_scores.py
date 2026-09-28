"""Regex-extract the published prognostic scores as clinicians documented them
in clinical notes, as a second score source next to the structured-data
scores from build_published_scores.py.

Command line: `--anchor {treatment,sequencing} --lookback-days 90
--note-types PROGRESS_NOTES [...]`.

Method
------
Notes for cohort patients are scanned lazily. A cheap keyword prefilter
(`KEYWORD_RE`) runs first. Each note then goes through `MENTION_PATTERNS`,
which covers IPI, MELD, ALBI, CAPRA, mGPS, LIPI, RMH, IMDC, MSKCC, ECOG and
KPS, with case-insensitive Rust-regex patterns. The Rust engine has no
look-arounds, so exclusions (ipilimumab doses, out-of-range values) are
applied after extraction. `parse_matches` turns each match into
(mention, measure, variant, value).

`NOTE_SCORES` then maps mentions onto the catalog score ids, each on a single
ordinal/continuous scale:
- mgps: mGPS points only. Plain "GPS" / "Glasgow prognostic score" is a
  different score and is dropped.
- rmh: points 0-3.
- lipi: points, or good/intermediate/poor mapped to 0/1/2.
- albi: grade 1-3. A documented continuous ALBI is graded at -2.60/-1.39.
- meld: any documented MELD variant (MELD, MELD-Na, MELD 3.0), as the
  clinician used it.
- capra_mod: CAPRA points 0-10. CAPRA-S (post-prostatectomy) is dropped.
- ipi: IPI / R-IPI points 0-5, which include ECOG. NCCN-IPI (0-8) is
  dropped.
- imdc / mskcc: risk group favorable/intermediate/poor as 0/1/2. Documented
  points map to groups at 0 / 1-2 / >=3.
Clinicians document the full IPI/IMDC/MSKCC, so these are keyed under the
full-score ids; the ECOG-free `*_noecog` variants have no note source.

Performance status (`performance_status_frame`) completes the calculated
full IPI/IMDC/MSKCC in build_published_scores.py. It is one ECOG value per
patient: a documented ECOG (the upper end of a range such as "ECOG 1-2"), or
a documented KPS converted on the ECOG-ACRIN scale (KPS 100 -> 0, 80-90 -> 1,
60-70 -> 2, 40-50 -> 3, 10-30 -> 4). The conversion keeps IMDC/MSKCC's
original KPS < 80% criterion exact at ECOG >= 2.

Performance status is also a predictor on its own, in two scores that are
not in the published catalog and have no calculated source
(`PERFORMANCE_STATUS_SCORES`, every cohort patient eligible):
- ecog_only: documented ECOG only (upper end of a range), no KPS conversion.
- kps_only: documented KPS only, on its native 10-100 scale (higher is
  better, so its direction is -1).

No-leakage window: only mentions in notes dated 0..`lookback_days` days
before the anchor count. The latest such note is used; ties on one date take
the highest (worst) value. The same rule picks the performance status.

Outputs
-------
- `FEATURE_PATH/published_scores_note_df{anchor_suffix}.csv.gz`: one row per
  cohort patient, with `note_lookback_days` and, per score,
  `{id}__note_value`, `{id}__note_group`, `{id}__note_days_before_anchor` and
  `{id}__note_n_mentions` (in-window mentions), plus `ecog__note_value`,
  `ecog__note_source` (ECOG or KPS), `ecog__note_days_before_anchor` and
  `ecog__note_n_mentions`, and the same four `__note_*` columns as the
  scores for `ecog_only` and `kps_only`. Values are null when nothing was documented. No note text is written; per-match snippets for review
  are in notebooks/1_data/01c_published_scores_note_regex.ipynb.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace

import polars as pl

from anchors import ANCHORS, DEFAULT_ANCHOR, anchor_suffix, date_col
from config import CLINICAL_NOTES_PATH, FEATURE_PATH, SURV_PATH
from pipelines.preprocessing import profile_sources as ps
from shared.published_scores import (
    CATALOG_SCORE_IDS, ECOG_FREE_OF, PublishedScore, RiskGroup, default_catalog, group_expr,
)

DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_NOTE_TYPES = ("PROGRESS_NOTES",)

_SEP = r"\s*(?:of|is|was|=|:)?\s*"
_GROUP = r"(?P<value>favou?rable|good|intermediate|poor)"

# (mention, measure, pattern). Every pattern has a named `value` group;
# `variant`, `value_hi` (upper end of an ECOG range) and `unit` (ipilimumab
# doses) are optional.
MENTION_PATTERNS: list[tuple[str, str, str]] = [
    ("ipi", "points",
     r"\b(?P<variant>R-?IPI|NCCN-?IPI|IPI)\b(?:\s+score)?" + _SEP + r"(?P<value>[0-8])\b(?P<unit>\s*mg)?"),
    ("meld", "continuous",
     r"\b(?P<variant>MELD(?:\s*3\.0)?(?:[- ]?Na)?)\b(?:\s+score)?" + _SEP + r"(?P<value>\d{1,2})\b"),
    ("albi", "grade",
     r"\bALBI\b(?:\s+score)?" + _SEP + r"(?:grade|gr\.?)\s*(?P<value>[123])\b"),
    ("albi", "continuous",
     r"\bALBI\b(?:\s+score)?" + _SEP + r"(?P<value>[-−–]\s?\d\.\d+)"),
    ("capra", "points",
     r"\b(?P<variant>CAPRA(?:-S)?)\b(?:\s+score)?" + _SEP + r"(?P<value>\d{1,2})\b"),
    ("mgps", "points",
     r"\b(?P<variant>m?GPS|(?:modified\s+)?Glasgow\s+prognostic\s+score)\b(?:\s+score)?" + _SEP + r"(?P<value>[0-2])\b"),
    ("lipi", "points",
     r"\bLIPI\b(?:\s+score)?" + _SEP + r"(?P<value>[0-2])\b"),
    ("lipi", "group",
     r"\bLIPI\b(?:\s+(?:score|group|category|risk))?\s*(?:of|is|was|=|:|-)?\s*" + _GROUP + r"\b"),
    ("lipi", "group",
     _GROUP + r"[- ](?:risk\s+)?(?:by\s+|per\s+)?LIPI\b"),
    ("rmh", "points",
     r"\b(?P<variant>RMH|Royal\s+Marsden(?:\s+Hospital)?)\s+(?:prognostic\s+)?score" + _SEP + r"(?P<value>[0-3])\b"),
    ("imdc", "group",
     r"\b(?P<variant>IMDC|Heng)\b(?:\s+(?:risk|prognostic))?(?:\s+(?:score|criteria|group|category|model|classification))?"
     r"\s*(?:of|is|was|=|:|-)?\s*" + _GROUP + r"\b"),
    ("imdc", "group",
     _GROUP + r"[- ]risk\b[^.\n]{0,25}?\b(?:by|per|according\s+to|based\s+on)\s+(?:the\s+)?(?P<variant>IMDC|Heng)\b"),
    ("imdc", "points",
     r"\b(?P<variant>IMDC)\b(?:\s+(?:risk\s+)?(?:score|criteria|risk\s+factors?))?" + _SEP + r"(?P<value>[0-6])\b"),
    ("imdc", "points",
     r"\b(?P<variant>Heng)\s+(?:risk\s+)?(?:score|criteria|risk\s+factors?)" + _SEP + r"(?P<value>[0-6])\b"),
    ("mskcc", "group",
     r"\b(?P<variant>MSKCC(?:/Motzer)?|Motzer)\b(?:\s+(?:risk|prognostic))?(?:\s+(?:score|criteria|group|category|model|classification))?"
     r"\s*(?:of|is|was|=|:|-)?\s*" + _GROUP + r"\b"),
    ("mskcc", "group",
     _GROUP + r"[- ]risk\b[^.\n]{0,25}?\b(?:by|per|according\s+to|based\s+on)\s+(?:the\s+)?(?P<variant>MSKCC|Motzer)\b"),
    ("mskcc", "points",
     r"\b(?P<variant>MSKCC(?:/Motzer)?|Motzer)\s+(?:risk\s+)?(?:score|criteria|risk\s+factors?)" + _SEP + r"(?P<value>[0-5])\b"),
    ("ecog", "points",
     r"\bECOG\b\)?(?:\s+(?:PS|performance\s+(?:status|score)))?\)?\s*(?:of|is|was|=|:|-)?\s*"
     r"(?P<value>[0-4])(?:\s*(?:-|to|/)\s*(?P<value_hi>[0-4]))?\b"),
    ("kps", "points",
     r"\b(?P<variant>KPS|Karnofsky(?:\s+performance\s+(?:status|scale|score))?)\b" + _SEP + r"(?P<value>100|[1-9]0)\b"),
]

VALID_RANGE = {
    ("ipi", "points"): (0, 8), ("meld", "continuous"): (6, 40), ("albi", "grade"): (1, 3),
    ("albi", "continuous"): (-5, 0), ("capra", "points"): (0, 12), ("mgps", "points"): (0, 2),
    ("lipi", "points"): (0, 2), ("rmh", "points"): (0, 3), ("imdc", "points"): (0, 6),
    ("mskcc", "points"): (0, 5), ("ecog", "points"): (0, 4), ("kps", "points"): (0, 100),
}

KEYWORD_RE = (
    r"(?i)\b(?:IPI|MELD|ALBI|CAPRA|m?GPS|Glasgow|LIPI|RMH|Marsden|IMDC|Heng|MSKCC|Motzer|ECOG|KPS|Karnofsky)\b"
)
_COMPILED = [f"(?i){pattern}" for _, _, pattern in MENTION_PATTERNS]
_SNIPPET = [f"(?i)([^\\n]{{0,80}}(?:{pattern})[^\\n]{{0,80}})" for _, _, pattern in MENTION_PATTERNS]
_OPTIONAL_GROUPS = ("variant", "value_hi", "unit")


def match_columns(text: pl.Expr, snippets: bool = False) -> list[pl.Expr]:
    """Per-note list of matches for every pattern (`_m{i}`), plus one context
    snippet per pattern (`_s{i}`) when `snippets`."""
    exprs = []
    for i in range(len(MENTION_PATTERNS)):
        exprs.append(text.str.extract_all(_COMPILED[i]).alias(f"_m{i}"))
        if snippets:
            exprs.append(text.str.extract(_SNIPPET[i], 1).alias(f"_s{i}"))
    return exprs


def parse_matches(wide: pl.DataFrame, id_cols: list[str]) -> pl.DataFrame:
    """Wide per-note match lists -> one row per (note, match) with a parsed,
    normalized, range-checked value. Ipilimumab doses (`ipi 1 mg`) are dropped."""
    parts = []
    for i, (mention, measure, _) in enumerate(MENTION_PATTERNS):
        snippet = pl.col(f"_s{i}") if f"_s{i}" in wide.columns else pl.lit(None, dtype=pl.String)
        long = (
            wide.select(*id_cols, pl.col(f"_m{i}").alias("match"), snippet.alias("snippet"))
            .explode("match", empty_as_null=True)
            .drop_nulls("match")
        )
        if long.is_empty():
            continue
        groups = long.with_columns(pl.col("match").str.extract_groups(_COMPILED[i]).alias("_g")).unnest("_g")
        parts.append(groups.select(
            *id_cols,
            pl.lit(mention).alias("mention"),
            pl.lit(measure).alias("measure"),
            pl.lit(i).alias("pattern_idx"),
            *[
                (pl.col(g) if g in groups.columns else pl.lit(None, dtype=pl.String)).alias(g)
                for g in _OPTIONAL_GROUPS
            ],
            pl.col("value"),
            "match",
            "snippet",
        ))
    if not parts:
        return _empty_mentions(wide.select(id_cols).schema)
    out = pl.concat(parts, how="diagonal_relaxed")
    value = (
        pl.col("value").str.to_lowercase().str.replace_all(r"[−–]", "-").str.replace_all(r"\s", "")
        .str.replace("favourable", "favorable")
    )
    out = out.with_columns(
        value.alias("value"),
        pl.col("variant").str.to_uppercase().str.replace_all(r"\s+", " "),
    ).with_columns(pl.col("value").cast(pl.Float64, strict=False).alias("value_num"))
    ranges = pl.DataFrame(
        [{"mention": k[0], "measure": k[1], "_lo": float(v[0]), "_hi": float(v[1])} for k, v in VALID_RANGE.items()]
    )
    out = out.join(ranges, on=["mention", "measure"], how="left")
    in_range = pl.col("value_num").is_null() | pl.col("value_num").is_between(pl.col("_lo"), pl.col("_hi"))
    is_dose = pl.col("unit").is_not_null()
    return out.filter(in_range & ~is_dose).drop("_lo", "_hi", "unit")


def _empty_mentions(id_schema: pl.Schema | dict) -> pl.DataFrame:
    return pl.DataFrame(schema={
        **dict(id_schema), "mention": pl.String, "measure": pl.String, "pattern_idx": pl.Int32,
        "variant": pl.String, "value_hi": pl.String, "value": pl.String, "match": pl.String,
        "snippet": pl.String, "value_num": pl.Float64,
    })


def load_anchor_cohort(anchor: str) -> pl.DataFrame:
    """DFCI_MRN (Int64) and `_anchor_date` (Date) for every cohort patient."""
    anchor_col = date_col(anchor)
    cohort = pl.read_parquet(os.path.join(SURV_PATH, "cohort_df.parquet"), columns=["DFCI_MRN", anchor_col])
    anchor_date = pl.col(anchor_col)
    if cohort.schema[anchor_col] == pl.String:
        anchor_date = anchor_date.str.to_datetime(strict=False)
    return cohort.select(
        pl.col("DFCI_MRN").cast(pl.Int64, strict=False),
        anchor_date.cast(pl.Date, strict=False).alias("_anchor_date"),
    )


def scan_note_mentions(
    cohort: pl.DataFrame, note_types: tuple[str, ...] = DEFAULT_NOTE_TYPES, *, snippets: bool = False,
    notes_path: str = CLINICAL_NOTES_PATH,
) -> pl.DataFrame:
    """Every parsed mention in the cohort's notes, one row per match, with
    `note_date`, `note_type` and `days_before_anchor` (anchor - note date;
    negative after the anchor). `cohort` is `load_anchor_cohort` output."""
    cohort_mrns = cohort["DFCI_MRN"].implode()
    id_cols = ["DFCI_MRN", "RPT_ID", "note_date", "note_type"]
    parts = []
    for note_type in note_types:
        lf = pl.scan_parquet(os.path.join(notes_path, f"{note_type}.parquet"))
        # EVENT_DATE is tz-aware UTC, or a string on some releases (as in build_cohort).
        event_dtype = lf.collect_schema()[ps.EVENT_DATE]
        event_dt = pl.col(ps.EVENT_DATE)
        if event_dtype == pl.String:
            event_dt = event_dt.str.to_datetime(time_zone="UTC", strict=False)
        if event_dtype == pl.String or isinstance(event_dtype, pl.Datetime):
            event_dt = event_dt.dt.replace_time_zone(None)
        event_dt = event_dt.cast(pl.Date, strict=False)
        wide = (
            lf.select(ps.MRN, ps.RPT_ID, ps.EVENT_DATE, ps.RPT_TEXT)
            .with_columns(pl.col(ps.MRN).cast(pl.Int64, strict=False))
            .filter(pl.col(ps.MRN).is_in(cohort_mrns) & pl.col(ps.RPT_TEXT).str.contains(KEYWORD_RE))
            .select(
                pl.col(ps.MRN).alias("DFCI_MRN"),
                pl.col(ps.RPT_ID).cast(pl.String).alias("RPT_ID"),
                event_dt.alias("note_date"),
                pl.lit(note_type).alias("note_type"),
                *match_columns(pl.col(ps.RPT_TEXT).cast(pl.String), snippets=snippets),
            )
            .collect(engine="streaming")
        )
        print(f"[note scores] {note_type}: {wide.height:,} keyword-matching cohort notes")
        parts.append(parse_matches(wide, id_cols))
    mentions = pl.concat(parts, how="diagonal_relaxed") if parts else _empty_mentions({})
    return mentions.join(cohort, on="DFCI_MRN", how="inner").with_columns(
        (pl.col("_anchor_date") - pl.col("note_date")).dt.total_days().alias("days_before_anchor")
    ).drop("_anchor_date")


_GROUP_ORDINAL = {"favorable": 0.0, "good": 0.0, "intermediate": 1.0, "poor": 2.0}


def _group_ordinal() -> pl.Expr:
    return pl.col("value").replace_strict(_GROUP_ORDINAL, default=None, return_dtype=pl.Float64)


def _points_to_rcc_group() -> pl.Expr:
    return pl.when(pl.col("value_num") == 0).then(0.0).when(pl.col("value_num") <= 2).then(1.0).otherwise(2.0)


_RCC_GROUPS = (RiskGroup("favorable", 0, 0), RiskGroup("intermediate", 1, 1), RiskGroup("poor", 2, 2))
_measure = pl.col("measure")
_variant = pl.col("variant")
_num = pl.col("value_num")
# A documented ECOG, taking the upper end of a range such as "ECOG 1-2".
_ecog_value = pl.coalesce(pl.col("value_hi").cast(pl.Float64, strict=False), _num)


@dataclass(frozen=True)
class NoteScore:
    """How one catalog score is read from parsed mentions: `value` maps a
    mention row to the note value (null to drop it), on the scale
    `risk_groups` cuts."""
    score_id: str
    mention: str
    value: pl.Expr
    risk_groups: tuple[RiskGroup, ...]
    note: str


_CATALOG = default_catalog()
NOTE_SCORES: dict[str, NoteScore] = {s.score_id: s for s in (
    NoteScore("mgps", "mgps",
              pl.when(_variant.is_in(["MGPS", "MODIFIED GLASGOW PROGNOSTIC SCORE"])).then(_num),
              _CATALOG["mgps"].risk_groups, "mGPS points; unmodified GPS dropped"),
    NoteScore("rmh", "rmh", _num, _CATALOG["rmh"].risk_groups, "RMH points"),
    NoteScore("lipi", "lipi",
              pl.when(_measure == "points").then(_num).when(_measure == "group").then(_group_ordinal()),
              _CATALOG["lipi"].risk_groups, "LIPI points, or good/intermediate/poor as 0/1/2"),
    NoteScore("albi", "albi",
              pl.when(_measure == "grade").then(_num)
              .when(_measure == "continuous").then(
                  pl.when(_num <= -2.60).then(1.0).when(_num <= -1.39).then(2.0).otherwise(3.0)),
              (RiskGroup("grade 1", 1, 1), RiskGroup("grade 2", 2, 2), RiskGroup("grade 3", 3, 3)),
              "ALBI grade; documented continuous ALBI graded at -2.60/-1.39"),
    NoteScore("meld", "meld", _num, _CATALOG["meld"].risk_groups, "any MELD variant (MELD, MELD-Na, MELD 3.0)"),
    NoteScore("capra_mod", "capra",
              pl.when((_variant == "CAPRA") & (_num <= 10)).then(_num),
              (RiskGroup("low (0-2)", 0, 2), RiskGroup("intermediate (3-5)", 3, 5), RiskGroup("high (6-10)", 6, 10)),
              "full CAPRA points 0-10; CAPRA-S dropped"),
    NoteScore("ipi", "ipi",
              pl.when(_variant.is_in(["IPI", "R-IPI", "RIPI"]) & (_num <= 5)).then(_num),
              (RiskGroup("0-1 (low)", 0, 1), RiskGroup("2 (low-intermediate)", 2, 2),
               RiskGroup("3 (high-intermediate)", 3, 3), RiskGroup("4-5 (high)", 4, 5)),
              "full IPI / R-IPI points 0-5 (with ECOG); NCCN-IPI dropped"),
    NoteScore("imdc", "imdc",
              pl.when(_measure == "group").then(_group_ordinal())
              .when(_measure == "points").then(_points_to_rcc_group()),
              _RCC_GROUPS, "full IMDC risk group (with KPS) as 0/1/2; points grouped 0 / 1-2 / >=3"),
    NoteScore("mskcc", "mskcc",
              pl.when(_measure == "group").then(_group_ordinal())
              .when(_measure == "points").then(_points_to_rcc_group()),
              _RCC_GROUPS, "full MSKCC risk group (with KPS) as 0/1/2; points grouped 0 / 1-2 / >=3"),
)}
assert set(NOTE_SCORES) == set(CATALOG_SCORE_IDS) - set(ECOG_FREE_OF)

# Performance status alone, as documented: not catalog scores (no calculated
# source, no eligibility restriction), so figures/prep/published_scores.py
# takes their metadata from here. Risk groups run from lowest to highest risk.
PERFORMANCE_STATUS_SCORES: dict[str, PublishedScore] = {s.id: s for s in (
    PublishedScore(
        "ecog_only", "ECOG performance status", "Oken et al., Am J Clin Oncol 1982", pl.lit(True), (),
        (RiskGroup("0", 0, 0), RiskGroup("1", 1, 1), RiskGroup("2", 2, 2), RiskGroup("3-4", 3, 4)),
    ),
    PublishedScore(
        "kps_only", "Karnofsky performance status", "Karnofsky & Burchenal 1949", pl.lit(True), (),
        (RiskGroup("80-100", 80, 100), RiskGroup("60-70", 60, 70), RiskGroup("10-50", 10, 50)),
        direction=-1,
    ),
)}
PERFORMANCE_STATUS_NOTE_SCORES: dict[str, NoteScore] = {s.score_id: s for s in (
    NoteScore("ecog_only", "ecog", _ecog_value, PERFORMANCE_STATUS_SCORES["ecog_only"].risk_groups,
              "documented ECOG only (upper end of a range); KPS not converted"),
    NoteScore("kps_only", "kps", _num, PERFORMANCE_STATUS_SCORES["kps_only"].risk_groups,
              "documented KPS only, 10-100"),
)}


# ECOG-ACRIN KPS -> ECOG equivalents, as (lowest KPS, ECOG), checked top-down.
_KPS_TO_ECOG = ((100, 0.0), (80, 1.0), (60, 2.0), (40, 3.0), (10, 4.0))


def _kps_to_ecog(kps: pl.Expr) -> pl.Expr:
    expr = pl.lit(None, dtype=pl.Float64)
    for low, ecog in reversed(_KPS_TO_ECOG):
        expr = pl.when(kps >= low).then(ecog).otherwise(expr)
    return expr


def _latest_per_patient(rows: pl.DataFrame, prefix: str, extra: tuple[pl.Expr, ...] = ()) -> pl.DataFrame:
    """Latest in-window `_value` per patient (ties on one date -> highest),
    its days before anchor and the number of in-window mentions."""
    latest_day = pl.col("days_before_anchor") == pl.col("days_before_anchor").min()
    return rows.group_by("DFCI_MRN").agg(
        pl.col("_value").filter(latest_day).max().alias(f"{prefix}__note_value"),
        *extra,
        pl.col("days_before_anchor").min().cast(pl.Int64).alias(f"{prefix}__note_days_before_anchor"),
        pl.len().cast(pl.Int64).alias(f"{prefix}__note_n_mentions"),
    )


def performance_status_frame(in_window: pl.DataFrame) -> pl.DataFrame:
    """One row per patient with an in-window ECOG or KPS: `ecog__note_value`
    on the ECOG scale, `ecog__note_source` (ECOG or KPS; ECOG wins a same-day
    tie at equal value), days before anchor and mention count."""
    rows = in_window.filter(pl.col("mention").is_in(["ecog", "kps"])).with_columns(
        pl.when(pl.col("mention") == "ecog").then(_ecog_value).otherwise(_kps_to_ecog(_num)).alias("_value"),
        pl.col("mention").str.to_uppercase().alias("_source"),
    ).drop_nulls("_value")
    latest_day = pl.col("days_before_anchor") == pl.col("days_before_anchor").min()
    source = (
        pl.col("_source").filter(latest_day & (pl.col("_value") == pl.col("_value").filter(latest_day).max()))
        .sort().first().alias("ecog__note_source")
    )
    return _latest_per_patient(rows, "ecog", (source,))


def note_score_frame(
    mentions: pl.DataFrame, cohort: pl.DataFrame, lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> pl.DataFrame:
    """One row per cohort patient: the latest in-window documented value of
    every note score (ties on one date -> highest value), its risk group,
    its days before anchor and the number of in-window mentions, plus the
    performance status (`performance_status_frame`) and ECOG and KPS alone
    (`PERFORMANCE_STATUS_NOTE_SCORES`)."""
    in_window = mentions.filter(pl.col("days_before_anchor").is_between(0, lookback_days))
    out = cohort.select("DFCI_MRN").unique().sort("DFCI_MRN").with_columns(
        pl.lit(lookback_days, dtype=pl.Int64).alias("note_lookback_days")
    )
    for spec in (*NOTE_SCORES.values(), *PERFORMANCE_STATUS_NOTE_SCORES.values()):
        value_col = f"{spec.score_id}__note_value"
        rows = (
            in_window.filter(pl.col("mention") == spec.mention)
            .with_columns(spec.value.cast(pl.Float64).alias("_value"))
            .drop_nulls("_value")
        )
        latest = _latest_per_patient(rows, spec.score_id)
        score = _CATALOG.get(spec.score_id) or PERFORMANCE_STATUS_SCORES[spec.score_id]
        group = group_expr(replace(score, risk_groups=spec.risk_groups), value_col)
        latest = latest.with_columns(group.alias(f"{spec.score_id}__note_group"))
        out = out.join(latest, on="DFCI_MRN", how="left")
    return out.join(performance_status_frame(in_window), on="DFCI_MRN", how="left")


def note_score_path(anchor: str) -> str:
    return os.path.join(FEATURE_PATH, f"published_scores_note_df{anchor_suffix(anchor)}.csv.gz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--anchor", choices=sorted(ANCHORS), default=DEFAULT_ANCHOR)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--note-types", nargs="+", default=list(DEFAULT_NOTE_TYPES))
    args = parser.parse_args()
    if args.lookback_days < 0:
        parser.error("--lookback-days must be >= 0")

    cohort = load_anchor_cohort(args.anchor)
    mentions = scan_note_mentions(cohort, tuple(args.note_types))
    frame = note_score_frame(mentions, cohort, args.lookback_days)
    os.makedirs(FEATURE_PATH, exist_ok=True)
    out_path = note_score_path(args.anchor)
    frame.write_csv(out_path, compression="gzip")
    counts = ", ".join(f"{s}={frame[f'{s}__note_value'].is_not_null().sum()}" for s in (*NOTE_SCORES, "ecog", *PERFORMANCE_STATUS_NOTE_SCORES))
    print(f"[note scores] wrote {out_path} ({frame.height:,} patients; documented: {counts})")


if __name__ == "__main__":
    main()
