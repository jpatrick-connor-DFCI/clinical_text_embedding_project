"""Published prognostic scores vs. the held-out text risk score (overall survival).

For each score in `shared.published_scores.default_catalog()` (mgps, rmh, lipi, albi,
meld, capra_mod, and ipi/imdc/mskcc each full and `_noecog`), evaluates up to
two score sources separately:

- `calculated`: the score computed by build_published_scores.py, within the
  score's eligible, lab-observable, complete-case population. The full
  IPI/IMDC/MSKCC take their performance-status item from note-documented
  ECOG/KPS, so their complete cases are patients with one in the note
  window; the `_noecog` variants score everyone else as a lower bound.
- `regex`: the score as clinicians documented it in notes, regex-extracted by
  pipelines/preprocessing/extract_note_scores.py (latest mention 0..90 days
  before the anchor), within the same eligible population. Only scores in
  `NOTE_SCORES` have this source (not the `_noecog` variants, which nobody
  documents). IMDC/MSKCC are documented as a risk group (0/1/2), so the
  `variant` is `documented`. The note score
  does not depend on the lab window, so it is evaluated once per anchor,
  under the 30-day lab-window key.

Performance status is also evaluated alone, as two note-only scores from
`PERFORMANCE_STATUS_SCORES` (`ecog_only`: documented ECOG; `kps_only`:
documented KPS on its 10-100 scale, direction -1). They have only the
`regex` source (variant `documented`), every cohort patient is eligible,
and like the other note scores they run once per anchor under the 30-day
lab-window key.

Each source is evaluated on its own cohort. Within one (score, source) there
are three models on one shared set of comparable pairs (`blocks = text_fold`,
as in figure3_combined):

- `published`: the raw points (or ALBI/MELD's continuous value) x direction,
  unfitted -- this is how the score is used clinically.
- `text`: the raw out-of-fold full-cohort OS text risk score
  (`full_cohort_risk_dir("death_met", "death")/text_risk_scores.csv`).
- `published+text`: `_cross_fitted_risk([points, text_z], ...)`, imported
  from figure3_combined, with `text_z` the text score z-scored within its
  own outer fold via `_standardize_within_folds(frame, modalities=("text",))`.

The 30-day-window, treatment-anchor run is primary; the 90-day-window and
sequencing-anchor runs are sensitivity analyses (same code, different
`published_scores_df{...}.csv.gz` input, driven by build_published_scores.py's
`--anchor`/`--lab-window-days`).

Writes to FIGURE_DATA_DIR:
Every output carries a `source` column next to `score`.

- pubscore_cindex.csv  anchor, lab_window_days, score, source, variant, stratum, model,
                       cindex, ci_lower, ci_upper, n_patients, n_events,
                       n_comparable_pairs, n_fold_blocks, n_boot, status
- pubscore_delta.csv   the three paired contrasts (published+text - published,
                       published+text - text, text - published), each with a
                       bootstrap CI
- pubscore_cox.csv     one lifelines CoxPHFitter fit per (score, stratum): text
                       HR/SD adjusted for published and published HR/point
                       adjusted for text, both with 95% CIs, plus LRT p-values
                       for adding each term (df=1). A fit failing the same
                       |coef|>5 guard as figure3._fit_joint_cox is dropped.
- pubscore_km.csv      patient-level time/event, published risk group, text
                       tertiles, and text groups cut to the published groups'
                       sizes
- pubscore_cohort.csv  filtering counts, fraction of pairs tied on the
                       published score, Spearman(published, text), the
                       unblocked published C (comparable to validation
                       papers), and the `underpowered` flag. For `regex`
                       rows, n_observable is null and n_complete counts
                       eligible patients with a documented score.
                       n_both_sources counts patients evaluated under both
                       sources, and note_lookback_days gives the note
                       window.

Statuses mirror figure3_combined: ok, missing_inputs, too_few_patients,
too_few_events, too_few_non_events, constant_published, fit_failed,
no_comparable_pairs, no_performance_status_source (reserved; the full
IPI/IMDC/MSKCC fall to too_few_patients when little ECOG is documented).

Command line: --n-boot (default 1000), --n-jobs.
"""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from scipy.stats import spearmanr, chi2

from config import FEATURE_PATH
from figures.io import save_figure_data
from figures.prep.figure2 import _safe_quantiles
from figures.prep.figure3_combined import (
    BOOT_SEED,
    DEFAULT_N_BOOT,
    MIN_EVENTS,
    MIN_PATIENTS,
    _bootstrap_weights,
    _cross_fitted_risk,
    _standardize_within_folds,
    pair_weighted_concordance,
)
from figures.prep.parallel import process_pool, resolve_workers
from figures.prep.within_cancer import _READ_ERRORS, _valid_fold
from pipelines.preprocessing.extract_note_scores import (
    NOTE_SCORES, PERFORMANCE_STATUS_SCORES, read_note_score_frame,
)
from schemes import full_cohort_risk_dir
from shared.published_scores import PublishedScore, default_catalog

ANCHORS_TO_RUN = ("treatment", "sequencing")
LAB_WINDOWS_TO_RUN = (30, 90)
PRIMARY_ANCHOR = "treatment"
PRIMARY_LAB_WINDOW = 30

SOURCES = ("calculated", "regex")
REGEX_VARIANT = "documented"

TEXT_FOLD = "text_fold"
PAIR_CHUNK = 512
UNDERPOWERED_PATIENTS = 100
UNDERPOWERED_EVENTS = 30

CINDEX_SCHEMA = {
    "anchor": pl.String, "lab_window_days": pl.Int64, "score": pl.String, "source": pl.String,
    "variant": pl.String, "stratum": pl.String, "model": pl.String, "cindex": pl.Float64, "ci_lower": pl.Float64,
    "ci_upper": pl.Float64, "n_patients": pl.Int64, "n_events": pl.Int64,
    "n_comparable_pairs": pl.Int64, "n_fold_blocks": pl.Int64, "n_boot": pl.Int64, "status": pl.String,
}
DELTA_SCHEMA = {
    "anchor": pl.String, "lab_window_days": pl.Int64, "score": pl.String, "source": pl.String,
    "stratum": pl.String,
    "model": pl.String, "reference": pl.String, "delta_cindex": pl.Float64,
    "ci_lower": pl.Float64, "ci_upper": pl.Float64, "n_boot": pl.Int64,
}
COX_SCHEMA = {
    "anchor": pl.String, "lab_window_days": pl.Int64, "score": pl.String, "source": pl.String,
    "stratum": pl.String,
    "term": pl.String, "hr": pl.Float64, "ci_lower": pl.Float64, "ci_upper": pl.Float64,
    "lrt_p": pl.Float64, "n": pl.Int64, "n_events": pl.Int64, "status": pl.String,
}
KM_SCHEMA = {
    "anchor": pl.String, "lab_window_days": pl.Int64, "score": pl.String, "source": pl.String,
    "stratum": pl.String,
    "DFCI_MRN": pl.String, "time": pl.Float64, "event_flag": pl.Float64,
    "published_group": pl.String, "text_tertile": pl.String, "text_group_matched": pl.String,
}
COHORT_SCHEMA = {
    "anchor": pl.String, "lab_window_days": pl.Int64, "score": pl.String, "source": pl.String,
    "stratum": pl.String,
    "n_eligible": pl.Int64, "n_observable": pl.Int64, "n_complete": pl.Int64,
    "n_with_text": pl.Int64, "n_with_outcome": pl.Int64, "n_patients": pl.Int64,
    "n_events": pl.Int64, "frac_tied_published": pl.Float64, "spearman_published_text": pl.Float64,
    "unblocked_published_cindex": pl.Float64, "underpowered": pl.Boolean,
    "n_both_sources": pl.Int64, "note_lookback_days": pl.Int64,
}

CATALOG = default_catalog()
NOTE_ONLY_SCORES = PERFORMANCE_STATUS_SCORES
MODELS = ("published", "text", "published+text")
CONTRASTS = (
    ("published+text", "published"),
    ("published+text", "text"),
    ("text", "published"),
)


def _score_df_path(anchor: str, lab_window_days: int) -> Path:
    suffix = "" if anchor == "treatment" else f"__{anchor}"
    lab_suffix = "" if lab_window_days == 30 else f"__lab{lab_window_days}d"
    return Path(FEATURE_PATH) / f"published_scores_df{suffix}{lab_suffix}.csv.gz"


def _note_df_path(anchor: str) -> Path:
    suffix = "" if anchor == "treatment" else f"__{anchor}"
    return Path(FEATURE_PATH) / f"published_scores_note_df{suffix}.csv.gz"


def _variant(score: PublishedScore, source: str) -> str:
    return score.variant if source == "calculated" else REGEX_VARIANT


def _regex_score(score: PublishedScore) -> PublishedScore:
    """`score` with the note-documented scale's risk groups (e.g. IMDC 0/1/2).
    Note-only scores are already on that scale."""
    if score.id not in NOTE_SCORES:
        return score
    return replace(score, risk_groups=NOTE_SCORES[score.id].risk_groups)


def _load_text_scores() -> pl.DataFrame:
    path = Path(full_cohort_risk_dir("death_met", "death")) / "text_risk_scores.csv"
    scores = pl.read_csv(path, schema_overrides={"DFCI_MRN": pl.String})
    score_col = "text_risk_score" if "text_risk_score" in scores.columns else "risk_score"
    return scores.select(
        pl.col("DFCI_MRN").cast(pl.String).str.strip_chars(),
        pl.col(score_col).cast(pl.Float64, strict=False).alias("text_score"),
        pl.col("outer_fold").cast(pl.Float64, strict=False).alias(TEXT_FOLD),
    )


def _load_outcomes() -> pl.DataFrame:
    from schemes import embedding_file
    from config import SURV_PATH
    source = pl.scan_parquet(os.path.join(SURV_PATH, embedding_file("death_met")))
    frame = source.select(
        "DFCI_MRN",
        pl.col("death").cast(pl.Float64, strict=False).alias("event_flag"),
        pl.col("tt_death").cast(pl.Float64, strict=False).alias("time"),
    ).collect()
    return frame.with_columns(pl.col("DFCI_MRN").cast(pl.String).str.strip_chars())


def _points_col(score: PublishedScore) -> str:
    return f"{score.id}__continuous" if score.continuous_formula is not None else f"{score.id}__points"


def _score_cohort(
    score_df: pl.DataFrame, text: pl.DataFrame, outcomes: pl.DataFrame, score: PublishedScore,
) -> pl.DataFrame:
    """Eligible, lab-observable, complete-case, scored, text-matched, outcome-matched."""
    points_col = _points_col(score)
    frame = score_df.select(
        "DFCI_MRN", f"{score.id}__eligible", f"{score.id}__complete", points_col,
        f"{score.id}__group", "labs_observable",
    ).with_columns(
        # An all-null column round-trips through CSV as String.
        pl.col(points_col).cast(pl.Float64, strict=False),
        pl.col(f"{score.id}__group").cast(pl.String),
    ).filter(
        pl.col(f"{score.id}__eligible").fill_null(False)
        & pl.col("labs_observable").fill_null(True)
        & pl.col(f"{score.id}__complete").fill_null(False)
        & pl.col(points_col).is_finite()
    ).rename({points_col: "published_score", f"{score.id}__group": "published_group"})
    frame = frame.join(text, on="DFCI_MRN", how="inner", validate="1:1")
    frame = frame.join(outcomes, on="DFCI_MRN", how="inner", validate="1:1")
    valid = (
        pl.col("time").is_finite() & (pl.col("time") > 0) & pl.col("event_flag").is_in([0.0, 1.0])
        & pl.col("text_score").is_finite() & _valid_fold(TEXT_FOLD)
        & (pl.col("published_score") * score.direction).is_finite()
    )
    return frame.filter(valid.fill_null(False)).sort("DFCI_MRN")


def _regex_cohort(
    score_df: pl.DataFrame, note_df: pl.DataFrame, text: pl.DataFrame, outcomes: pl.DataFrame,
    score: PublishedScore,
) -> pl.DataFrame:
    """Eligible (same population as `calculated`), with a documented note
    score in the window, text-matched, outcome-matched."""
    value_col, group_col = f"{score.id}__note_value", f"{score.id}__note_group"
    notes = note_df.select(
        pl.col("DFCI_MRN").cast(pl.String).str.strip_chars(),
        pl.col(value_col).cast(pl.Float64, strict=False).alias("published_score"),
        pl.col(group_col).cast(pl.String).alias("published_group"),
    ).filter(pl.col("published_score").is_finite())
    frame = score_df.filter(pl.col(f"{score.id}__eligible").fill_null(False)).select("DFCI_MRN")
    frame = frame.join(notes, on="DFCI_MRN", how="inner", validate="1:1")
    frame = frame.join(text, on="DFCI_MRN", how="inner", validate="1:1")
    frame = frame.join(outcomes, on="DFCI_MRN", how="inner", validate="1:1")
    valid = (
        pl.col("time").is_finite() & (pl.col("time") > 0) & pl.col("event_flag").is_in([0.0, 1.0])
        & pl.col("text_score").is_finite() & _valid_fold(TEXT_FOLD)
    )
    return frame.filter(valid.fill_null(False)).sort("DFCI_MRN")


def _status_cindex_rows(
    anchor: str, lab_window_days: int, score: PublishedScore, stratum: str, status: str,
    n: int | None = None, n_events: int | None = None, source: str = "calculated",
) -> pl.DataFrame:
    return pl.DataFrame([{
        "anchor": anchor, "lab_window_days": lab_window_days, "score": score.id, "source": source,
        "variant": _variant(score, source), "stratum": stratum, "model": model, "cindex": None,
        "ci_lower": None, "ci_upper": None, "n_patients": n, "n_events": n_events,
        "n_comparable_pairs": None, "n_fold_blocks": None, "n_boot": 0, "status": status,
    } for model in MODELS], schema=CINDEX_SCHEMA)


def _interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.nanquantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def evaluate_score(
    frame: pl.DataFrame, *, anchor: str, lab_window_days: int, score: PublishedScore,
    stratum: str = "all", n_boot: int = 0, source: str = "calculated",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """C-index of published/text/published+text on `frame` (one score's cohort)."""
    n = frame.height
    n_events = int(frame["event_flag"].sum()) if n else 0
    gate = (
        "too_few_patients" if n < MIN_PATIENTS
        else "too_few_events" if n_events < MIN_EVENTS
        else "too_few_non_events" if n - n_events < MIN_EVENTS
        else "constant_published" if frame["published_score"].n_unique() <= 1
        else None
    )
    if gate is not None:
        return (
            _status_cindex_rows(anchor, lab_window_days, score, stratum, gate, n, n_events, source),
            pl.DataFrame(schema=DELTA_SCHEMA),
        )

    frame = frame.with_columns((pl.col("published_score") * score.direction).alias("published_oriented"))
    frame = _standardize_within_folds(frame.rename({"text_score": "text_score_raw"}).with_columns(
        pl.col("text_score_raw").alias("text_score")
    ), modalities=("text",))

    time = frame["time"].to_numpy()
    status = frame["event_flag"].to_numpy() == 1.0
    folds = frame[TEXT_FOLD].to_numpy()
    blocks = folds.astype(np.int64)

    risks = {
        "published": frame["published_oriented"].to_numpy(),
        "text": frame["text_z"].to_numpy(),
        "published+text": _cross_fitted_risk(
            frame.select("published_oriented", "text_z").to_numpy(), status, time, folds
        ),
    }
    fitted = [m for m in MODELS if risks[m] is not None]
    stacked = np.vstack([risks[m] for m in fitted]) if fitted else np.empty((0, n))
    numerators, pairs, n_blocks = pair_weighted_concordance(time, status, stacked, np.ones((1, n)), blocks)
    n_pairs = int(round(pairs[0]))
    cindex = dict(zip(fitted, numerators[0] / n_pairs)) if n_pairs else {}

    rows = []
    for model in MODELS:
        model_status = ("fit_failed" if risks[model] is None else "ok" if n_pairs else "no_comparable_pairs")
        rows.append({
            "anchor": anchor, "lab_window_days": lab_window_days, "score": score.id, "source": source,
            "variant": _variant(score, source), "stratum": stratum, "model": model, "cindex": cindex.get(model), "ci_lower": None, "ci_upper": None,
            "n_patients": n, "n_events": n_events, "n_comparable_pairs": n_pairs, "n_fold_blocks": n_blocks,
            "n_boot": n_boot if n_boot and cindex else 0, "status": model_status,
        })

    if not n_boot or not cindex:
        return pl.DataFrame(rows, schema=CINDEX_SCHEMA), pl.DataFrame(schema=DELTA_SCHEMA)

    boot_num, boot_pairs, _ = pair_weighted_concordance(time, status, stacked, _bootstrap_weights(n, n_boot, BOOT_SEED), blocks)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = dict(zip(fitted, (boot_num / boot_pairs[:, None]).T))
    for row in rows:
        if row["model"] in boot:
            row["ci_lower"], row["ci_upper"] = _interval(boot[row["model"]])

    delta_rows = [{
        "anchor": anchor, "lab_window_days": lab_window_days, "score": score.id, "source": source,
        "stratum": stratum, "model": model, "reference": reference, "delta_cindex": cindex[model] - cindex[reference],
        **dict(zip(("ci_lower", "ci_upper"), _interval(boot[model] - boot[reference]))),
        "n_boot": n_boot,
    } for model, reference in CONTRASTS if model in cindex and reference in cindex]

    return pl.DataFrame(rows, schema=CINDEX_SCHEMA), pl.DataFrame(delta_rows, schema=DELTA_SCHEMA)


def evaluate_cox(
    frame: pl.DataFrame, *, anchor: str, lab_window_days: int, score: PublishedScore, stratum: str = "all",
    source: str = "calculated",
) -> pl.DataFrame:
    """lifelines CoxPHFitter with standardized published+text, both terms;
    LRT p-values against each single-term nested fit. Guards on |coef|>5,
    mirroring figure3._fit_joint_cox's pathological-fit drop."""
    n = frame.height
    n_events = int(frame["event_flag"].sum()) if n else 0
    if n < MIN_PATIENTS or n_events < MIN_EVENTS or n - n_events < MIN_EVENTS:
        return pl.DataFrame(schema=COX_SCHEMA)

    from sklearn.preprocessing import StandardScaler
    oriented = (frame["published_score"] * score.direction).to_numpy()
    X = StandardScaler().fit_transform(
        np.column_stack([oriented, frame["text_score"].to_numpy()])
    )
    if not np.isfinite(X).all():
        return pl.DataFrame(schema=COX_SCHEMA)

    fit_df = pd.DataFrame(X, columns=["published_z", "text_z"])
    fit_df["event_flag"] = frame["event_flag"].cast(pl.Int64).to_numpy()
    fit_df["time"] = frame["time"].to_numpy()

    def _fit(columns: list[str]) -> CoxPHFitter | None:
        cph = CoxPHFitter()
        try:
            cph.fit(fit_df[columns + ["event_flag", "time"]], duration_col="time", event_col="event_flag")
        except (ConvergenceError, TypeError, ValueError, np.linalg.LinAlgError):
            return None
        if (cph.summary["coef"].abs() > 5).any():
            return None
        return cph

    full = _fit(["published_z", "text_z"])
    pub_only = _fit(["published_z"])
    text_only = _fit(["text_z"])
    if full is None:
        return pl.DataFrame(schema=COX_SCHEMA)

    rows = []
    for term, nested in (("text_z", pub_only), ("published_z", text_only)):
        lrt_p = None
        if nested is not None:
            lrt = 2 * (full.log_likelihood_ - nested.log_likelihood_)
            lrt_p = float(chi2.sf(max(lrt, 0.0), df=1))
        srow = full.summary.loc[term]
        rows.append({
            "anchor": anchor, "lab_window_days": lab_window_days, "score": score.id, "source": source,
            "stratum": stratum, "term": term, "hr": float(srow["exp(coef)"]), "ci_lower": float(srow["exp(coef) lower 95%"]),
            "ci_upper": float(srow["exp(coef) upper 95%"]), "lrt_p": lrt_p, "n": n, "n_events": n_events,
            "status": "ok",
        })
    return pl.DataFrame(rows, schema=COX_SCHEMA)


def evaluate_km(
    frame: pl.DataFrame, *, anchor: str, lab_window_days: int, score: PublishedScore, stratum: str = "all",
    source: str = "calculated",
) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=KM_SCHEMA)
    n_groups = len(score.risk_groups)
    labels = [g.label for g in score.risk_groups]
    text_group = _safe_quantiles(frame["text_score"], n_groups, labels, f"{score.id}/text")
    text_tertile = _safe_quantiles(frame["text_score"], 3, ["low", "mid", "high"], f"{score.id}/text_tertile")
    return frame.select(
        pl.lit(anchor).alias("anchor"), pl.lit(lab_window_days, dtype=pl.Int64).alias("lab_window_days"),
        pl.lit(score.id).alias("score"), pl.lit(source).alias("source"), pl.lit(stratum).alias("stratum"),
        "DFCI_MRN", "time", "event_flag", "published_group",
    ).with_columns(
        text_tertile.alias("text_tertile"), text_group.alias("text_group_matched"),
    ).select(list(KM_SCHEMA))


def evaluate_cohort(
    frame: pl.DataFrame, score_df: pl.DataFrame, *, anchor: str, lab_window_days: int, score: PublishedScore,
    stratum: str = "all", source: str = "calculated", note_df: pl.DataFrame | None = None,
    n_both_sources: int | None = None,
) -> pl.DataFrame:
    eligible = score_df.filter(pl.col(f"{score.id}__eligible").fill_null(False))
    n_eligible = eligible.height
    note_lookback = None
    if source == "calculated":
        n_observable = int(eligible["labs_observable"].fill_null(True).sum())
        n_complete = int(eligible[f"{score.id}__complete"].fill_null(False).sum())
    else:
        n_observable = None
        documented = note_df.filter(
            pl.col(f"{score.id}__note_value").cast(pl.Float64, strict=False).is_finite()
        ).select(pl.col("DFCI_MRN").cast(pl.String).str.strip_chars())
        n_complete = eligible.join(documented, on="DFCI_MRN", how="semi").height
        if "note_lookback_days" in note_df.columns and note_df.height:
            note_lookback = int(note_df["note_lookback_days"].max())
    n = frame.height
    n_events = int(frame["event_flag"].sum()) if n else 0

    frac_tied = None
    spearman = None
    unblocked_c = None
    if n >= 2:
        points = frame["published_score"].to_numpy()
        frac_tied = float((pd.Series(points).duplicated(keep=False)).mean())
        if frame["published_score"].n_unique() > 1 and frame["text_score"].n_unique() > 1:
            rho = spearmanr(points, frame["text_score"].to_numpy()).correlation
            spearman = float(rho) if not np.isnan(rho) else None
        if n_events >= 1:
            time = frame["time"].to_numpy()
            status = frame["event_flag"].to_numpy() == 1.0
            oriented = (points * score.direction).reshape(1, -1)
            blocks = np.zeros(n, dtype=np.int64)
            num, pairs, _ = pair_weighted_concordance(time, status, oriented, np.ones((1, n)), blocks)
            if pairs[0] > 0:
                unblocked_c = float(num[0, 0] / pairs[0])

    underpowered = n < UNDERPOWERED_PATIENTS or n_events < UNDERPOWERED_EVENTS
    return pl.DataFrame([{
        "anchor": anchor, "lab_window_days": lab_window_days, "score": score.id, "source": source,
        "stratum": stratum, "n_eligible": n_eligible, "n_observable": n_observable, "n_complete": n_complete,
        "n_with_text": n, "n_with_outcome": n, "n_patients": n, "n_events": n_events,
        "frac_tied_published": frac_tied, "spearman_published_text": spearman,
        "unblocked_published_cindex": unblocked_c, "underpowered": underpowered,
        "n_both_sources": n_both_sources, "note_lookback_days": note_lookback,
    }], schema=COHORT_SCHEMA)


def _missing_outputs(
    anchor: str, lab_window_days: int, score: PublishedScore, source: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    empty_cohort = pl.DataFrame([{
        "anchor": anchor, "lab_window_days": lab_window_days, "score": score.id, "source": source,
        "stratum": "all",
    }], schema=COHORT_SCHEMA)
    return (
        _status_cindex_rows(anchor, lab_window_days, score, "all", "missing_inputs", source=source),
        pl.DataFrame(schema=DELTA_SCHEMA), pl.DataFrame(schema=COX_SCHEMA),
        pl.DataFrame(schema=KM_SCHEMA), empty_cohort,
    )


def _evaluate_source(
    frame: pl.DataFrame, score_df: pl.DataFrame, *, anchor: str, lab_window_days: int, score: PublishedScore,
    source: str, n_boot: int, note_df: pl.DataFrame | None, n_both_sources: int | None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    keys = {"anchor": anchor, "lab_window_days": lab_window_days, "source": source}
    source_score = score if source == "calculated" else _regex_score(score)
    cindex, delta = evaluate_score(frame, score=source_score, n_boot=n_boot, **keys)
    cox = evaluate_cox(frame, score=source_score, **keys)
    km = evaluate_km(frame, score=source_score, **keys)
    cohort = evaluate_cohort(
        frame, score_df, score=source_score, note_df=note_df, n_both_sources=n_both_sources, **keys,
    )
    return cindex, delta, cox, km, cohort


def _evaluate_task(task: tuple) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Both sources for one (anchor, lab window, score). The note score does
    not depend on the lab window, so `regex` only runs under the primary one,
    and only for scores clinicians document."""
    anchor, lab_window_days, score_id, n_boot = task
    if score_id in NOTE_ONLY_SCORES:
        return _evaluate_note_only_task(anchor, lab_window_days, NOTE_ONLY_SCORES[score_id], n_boot)
    score = CATALOG[score_id]
    regex = lab_window_days == PRIMARY_LAB_WINDOW and score_id in NOTE_SCORES
    sources = SOURCES if regex else ("calculated",)
    try:
        score_df = pl.read_csv(_score_df_path(anchor, lab_window_days), schema_overrides={"DFCI_MRN": pl.String})
        text = _load_text_scores()
        outcomes = _load_outcomes()
        frames = {"calculated": _score_cohort(score_df, text, outcomes, score)}
    except _READ_ERRORS:
        outputs = [_missing_outputs(anchor, lab_window_days, score, source) for source in sources]
        return tuple(pl.concat([out[i] for out in outputs]) for i in range(5))

    note_df = None
    if "regex" in sources:
        try:
            note_df = read_note_score_frame(_note_df_path(anchor), pl.String)
            frames["regex"] = _regex_cohort(score_df, note_df, text, outcomes, score)
        except _READ_ERRORS:
            note_df = None
    n_both = (
        frames["calculated"].join(frames["regex"], on="DFCI_MRN", how="semi").height
        if "regex" in frames else None
    )
    outputs = [
        _evaluate_source(
            frames[source], score_df, anchor=anchor, lab_window_days=lab_window_days, score=score,
            source=source, n_boot=n_boot, note_df=note_df, n_both_sources=n_both,
        ) if source in frames else _missing_outputs(anchor, lab_window_days, score, source)
        for source in sources
    ]
    schemas = (CINDEX_SCHEMA, DELTA_SCHEMA, COX_SCHEMA, KM_SCHEMA, COHORT_SCHEMA)
    return tuple(pl.concat([out[i].cast(schema) for out in outputs]) for i, schema in enumerate(schemas))


def _evaluate_note_only_task(
    anchor: str, lab_window_days: int, score: PublishedScore, n_boot: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """`regex` source only, with every cohort patient (every note_df row)
    eligible; no builder output is read."""
    try:
        note_df = read_note_score_frame(_note_df_path(anchor), pl.String)
        eligible = note_df.select(
            pl.col("DFCI_MRN").cast(pl.String).str.strip_chars(), pl.lit(True).alias(f"{score.id}__eligible"),
        )
        frame = _regex_cohort(eligible, note_df, _load_text_scores(), _load_outcomes(), score)
    except _READ_ERRORS:
        return _missing_outputs(anchor, lab_window_days, score, "regex")
    return _evaluate_source(
        frame, eligible, anchor=anchor, lab_window_days=lab_window_days, score=score, source="regex",
        n_boot=n_boot, note_df=note_df, n_both_sources=None,
    )


def _tasks(n_boot: int) -> list[tuple]:
    tasks = []
    for anchor in ANCHORS_TO_RUN:
        for lab_window_days in LAB_WINDOWS_TO_RUN:
            primary = anchor == PRIMARY_ANCHOR and lab_window_days == PRIMARY_LAB_WINDOW
            score_ids = [*CATALOG, *NOTE_ONLY_SCORES] if lab_window_days == PRIMARY_LAB_WINDOW else CATALOG
            for score_id in score_ids:
                tasks.append((anchor, lab_window_days, score_id, n_boot if primary else 0))
    return tasks


def prepare_published_scores(
    n_boot: int = DEFAULT_N_BOOT, n_jobs: int | None = None, show_progress: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    tasks = _tasks(n_boot)
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    def _run(tasklist):
        return [_evaluate_task(t) for t in tasklist]

    with process_pool(min(resolve_workers(n_jobs), max(len(tasks), 1)),
                      initializer=warnings.simplefilter, initargs=("ignore",)) as pool:
        if pool is None:
            outputs = _run(tasks)
        else:
            futures = [pool.submit(_evaluate_task, task) for task in tasks]
            if tqdm is not None:
                for _ in tqdm(as_completed(futures), total=len(futures), desc="Published scores", disable=not show_progress):
                    pass
            else:
                for _ in as_completed(futures):
                    pass
            outputs = [future.result() for future in futures]

    schemas = (CINDEX_SCHEMA, DELTA_SCHEMA, COX_SCHEMA, KM_SCHEMA, COHORT_SCHEMA)
    return tuple(
        pl.concat([out[i].cast(schema) for out in outputs]) if outputs else pl.DataFrame(schema=schema)
        for i, schema in enumerate(schemas)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    if args.n_boot < 0:
        parser.error("--n-boot must be >= 0")
    cindex, delta, cox, km, cohort = prepare_published_scores(n_boot=args.n_boot, n_jobs=args.n_jobs)
    save_figure_data(cindex, "pubscore_cindex.csv")
    save_figure_data(delta, "pubscore_delta.csv")
    save_figure_data(cox, "pubscore_cox.csv")
    save_figure_data(km, "pubscore_km.csv")
    save_figure_data(cohort, "pubscore_cohort.csv")


if __name__ == "__main__":
    main()
