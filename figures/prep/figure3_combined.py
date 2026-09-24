"""Stacked Cox models on the Figure 3 held-out modality risk scores (Figure 3 supplement).

For every endpoint with all six modalities' held-out risk scores, Cox models are
fitted on combinations of those scores (MODELS):

- each non-text modality alone and with the text score (`stage`, `stage+text`, ...)
- `text` alone, `all_minus_text` (the five non-text modalities) and `all`

Every model for an endpoint uses one cohort (patients with every modality's
finite held-out score and outer fold, and a valid outcome) and one set of
comparable pairs, so C-index differences between models are paired.

Out-of-sample protocol. The inputs are already out-of-fold predictions from each
modality's nested CV (held_out_risk_scores/, with outer_fold). Each modality's
score is standardized within its own outer fold, which removes location/scale
differences between that modality's fitted fold models without using outcomes.
The stacking Cox model (unpenalized, Breslow ties; sksurv) is cross-fitted over
the text model's outer folds: fold k is predicted by a fit on the other folds.
Harrell's C is computed within blocks sharing every modality's outer fold, where
each input and the stacking fit are single fitted models, and pooled by
comparable-pair counts, as in figures.prep.within_cancer's modality table.
Single-score models go through the same stacking, so a model and its +text
counterpart differ only in the text score.

For overall survival (OS_ENDPOINT) the models and the paired differences in
CONTRASTS also get 95% percentile intervals from a patient bootstrap (--n-boot
resamples, default 1000; fitted models are held fixed, so the intervals omit
refitting variability).

Writes to FIGURE_DATA_DIR:
- fig3_combined_cindex.csv     scheme, event, model, modalities, cindex, n_patients,
                               n_events, n_comparable_pairs, n_fold_blocks, status
- fig3_combined_os_cindex.csv  scheme, event, model, modalities, cindex, ci_lower,
                               ci_upper, n_patients, n_events, n_boot
- fig3_combined_os_delta.csv   scheme, event, model, reference, delta_cindex,
                               ci_lower, ci_upper, n_boot   (model minus reference)

Only status == "ok" rows carry a C-index. Other statuses: missing_inputs (a
modality's score file or the outcome is unavailable), too_few_patients (< 20),
too_few_events / too_few_non_events (< 5), fit_failed (a stacking fold fit
failed) and no_comparable_pairs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed
from pathlib import Path
import warnings

import numpy as np
import polars as pl
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from tqdm.auto import tqdm

from config import SURV_PATH
from figures.io import save_figure_data
from figures.prep.figure3 import SCHEMES
from figures.prep.parallel import process_pool, resolve_workers
from figures.prep.within_cancer import _READ_ERRORS, _load_scores, _read_outcomes, _valid_fold
from schemes import embedding_file, scheme_results_dir
from shared.palette import MODALITY_ORDER

OS_ENDPOINT = ("death_met", "death")
NON_TEXT = tuple(m for m in MODALITY_ORDER if m != "text")
# model -> modalities whose held-out scores enter its stacked Cox model
MODELS: dict[str, tuple[str, ...]] = {
    **{name: mods for m in NON_TEXT for name, mods in ((m, (m,)), (f"{m}+text", (m, "text")))},
    "text": ("text",),
    "all_minus_text": NON_TEXT,
    "all": tuple(MODALITY_ORDER),
}
# (model, reference): reported as model-minus-reference C-index differences
CONTRASTS: tuple[tuple[str, str], ...] = (
    *((f"{m}+text", m) for m in NON_TEXT),
    ("all", "all_minus_text"),
    ("all", "text"),
    ("text", "all_minus_text"),
)
STACKING_FOLD = "text_fold"
# Same eligibility as the Figure 3 joint Cox fits (figure3._fit_joint_cox).
MIN_PATIENTS = 20
MIN_EVENTS = 5
DEFAULT_N_BOOT = 1000
BOOT_SEED = 2026
# Event rows per pairwise block; bounds the (chunk x block size) comparison matrices.
PAIR_CHUNK = 512

CINDEX_SCHEMA = {
    "scheme": pl.String, "event": pl.String, "model": pl.String, "modalities": pl.String,
    "cindex": pl.Float64, "n_patients": pl.Int64, "n_events": pl.Int64,
    "n_comparable_pairs": pl.Int64, "n_fold_blocks": pl.Int64, "status": pl.String,
}
OS_CINDEX_SCHEMA = {
    "scheme": pl.String, "event": pl.String, "model": pl.String, "modalities": pl.String,
    "cindex": pl.Float64, "ci_lower": pl.Float64, "ci_upper": pl.Float64,
    "n_patients": pl.Int64, "n_events": pl.Int64, "n_boot": pl.Int64,
}
OS_DELTA_SCHEMA = {
    "scheme": pl.String, "event": pl.String, "model": pl.String, "reference": pl.String,
    "delta_cindex": pl.Float64, "ci_lower": pl.Float64, "ci_upper": pl.Float64,
    "n_boot": pl.Int64,
}


def _endpoint_cohort(directory: Path, outcomes: pl.DataFrame) -> pl.DataFrame:
    """Patients with every modality's finite score and fold and a valid outcome."""
    frame = outcomes
    for m in MODALITY_ORDER:
        scores = _load_scores(directory / f"{m}_risk_scores.csv", m, m)
        frame = frame.join(scores, on="DFCI_MRN", how="inner", validate="1:1")
    valid = (pl.col("time").is_finite() & (pl.col("time") > 0)
             & pl.col("event_flag").is_in([0.0, 1.0]))
    for m in MODALITY_ORDER:
        valid = valid & pl.col(f"{m}_score").is_finite() & _valid_fold(f"{m}_fold")
    return frame.filter(valid.fill_null(False)).with_columns(
        pl.col(f"{m}_fold").cast(pl.Int64) for m in MODALITY_ORDER
    ).sort("DFCI_MRN")


def _standardize_within_folds(frame: pl.DataFrame) -> pl.DataFrame:
    """Add `{m}_z`: each score z-scored within its own outer fold (0 where constant)."""
    exprs = []
    for m in MODALITY_ORDER:
        score, fold = pl.col(f"{m}_score"), f"{m}_fold"
        sd = score.std().over(fold)
        exprs.append(pl.when(sd > 0).then((score - score.mean().over(fold)) / sd)
                     .otherwise(0.0).alias(f"{m}_z"))
    return frame.with_columns(exprs)


def _cross_fitted_risk(x: np.ndarray, event: np.ndarray, time: np.ndarray,
                       folds: np.ndarray) -> np.ndarray | None:
    """Out-of-fold Cox linear predictors of `x`'s columns; None if any fold fit fails.

    Columns constant within a training split are left out of that fold's fit.
    """
    risk = np.empty(len(time))
    for fold in np.unique(folds):
        test = folds == fold
        train = ~test
        keep = x[train].std(axis=0) > 0
        if not keep.any() or not event[train].any():
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = CoxPHSurvivalAnalysis(ties="breslow").fit(
                    x[train][:, keep], Surv.from_arrays(event[train], time[train]))
        except (ArithmeticError, np.linalg.LinAlgError, ValueError):
            return None
        if not np.isfinite(model.coef_).all():
            return None
        risk[test] = x[test][:, keep] @ model.coef_
    return risk


def pair_weighted_concordance(
    time: np.ndarray, event: np.ndarray, risks: np.ndarray, weights: np.ndarray,
    blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Harrell's C numerators and comparable-pair totals within blocks, under patient weights.

    `risks` is (n_models, n); `weights` is (n_weightings, n) patient multiplicities
    (a row of ones gives the ordinary counts; bootstrap counts give a resample).
    Only pairs within one block are compared. As in sksurv, (i, j) is comparable
    when i has an event and T_j > T_i, or T_j == T_i and j is censored; it adds
    w_i * w_j to the pair total and to the numerator when risk_i > risk_j (half
    when tied). Returns numerators (n_weightings, n_models), pair totals
    (n_weightings,) and the number of blocks with any comparable pair.
    """
    numerators = np.zeros((weights.shape[0], risks.shape[0]))
    pairs = np.zeros(weights.shape[0])
    n_blocks = 0
    for block in np.unique(blocks):
        idx = np.flatnonzero(blocks == block)
        t, e, r, w = time[idx], event[idx], risks[:, idx], weights[:, idx]
        block_pairs = 0
        cases = np.flatnonzero(e)
        for start in range(0, len(cases), PAIR_CHUNK):
            i = cases[start:start + PAIR_CHUNK]
            comparable = (t[None, :] > t[i, None]) | ((t[None, :] == t[i, None]) & ~e[None, :])
            block_pairs += int(comparable.sum())
            pairs += (w[:, i] * (w @ comparable.T)).sum(axis=1)
            for k in range(r.shape[0]):
                ri, rj = r[k, i, None], r[k, None, :]
                score = comparable * ((ri > rj) + 0.5 * (ri == rj))
                numerators[:, k] += (w[:, i] * (w @ score.T)).sum(axis=1)
        n_blocks += int(block_pairs > 0)
    return numerators, pairs, n_blocks


def _bootstrap_weights(n: int, n_boot: int, seed: int = BOOT_SEED) -> np.ndarray:
    """Patient multiplicities for `n_boot` resamples of n patients with replacement."""
    rng = np.random.default_rng(seed)
    weights = np.empty((n_boot, n))
    for b in range(n_boot):
        weights[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return weights


def _status_frame(scheme: str, event: str, status: str, n: int | None = None,
                  n_events: int | None = None) -> pl.DataFrame:
    return pl.DataFrame([{
        "scheme": scheme, "event": event, "model": model, "modalities": "+".join(mods),
        "cindex": None, "n_patients": n, "n_events": n_events,
        "n_comparable_pairs": None, "n_fold_blocks": None, "status": status,
    } for model, mods in MODELS.items()], schema=CINDEX_SCHEMA)


def _empty_os() -> tuple[pl.DataFrame, pl.DataFrame]:
    return pl.DataFrame(schema=OS_CINDEX_SCHEMA), pl.DataFrame(schema=OS_DELTA_SCHEMA)


def evaluate_endpoint(
    frame: pl.DataFrame, *, scheme: str, event: str, n_boot: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """C-index of every MODELS entry on one endpoint's cohort (_endpoint_cohort).

    Returns (cindex rows, bootstrap model rows, bootstrap contrast rows); the
    bootstrap frames are empty unless n_boot > 0.
    """
    n = frame.height
    n_events = int(frame["event_flag"].sum()) if n else 0
    if n < MIN_PATIENTS:
        return _status_frame(scheme, event, "too_few_patients", n, n_events), *_empty_os()
    if n_events < MIN_EVENTS:
        return _status_frame(scheme, event, "too_few_events", n, n_events), *_empty_os()
    if n - n_events < MIN_EVENTS:
        return _status_frame(scheme, event, "too_few_non_events", n, n_events), *_empty_os()

    frame = _standardize_within_folds(frame)
    time = frame["time"].to_numpy()
    status = frame["event_flag"].to_numpy() == 1.0
    folds = frame[STACKING_FOLD].to_numpy()
    fold_ids = frame.select(f"{m}_fold" for m in MODALITY_ORDER).to_numpy()
    blocks = np.unique(fold_ids, axis=0, return_inverse=True)[1].reshape(-1)

    risks = {model: _cross_fitted_risk(frame.select(f"{m}_z" for m in mods).to_numpy(),
                                       status, time, folds)
             for model, mods in MODELS.items()}
    fitted = [model for model, risk in risks.items() if risk is not None]
    stacked = np.vstack([risks[model] for model in fitted]) if fitted else np.empty((0, n))
    numerators, pairs, n_blocks = pair_weighted_concordance(
        time, status, stacked, np.ones((1, n)), blocks)
    n_pairs = int(round(pairs[0]))
    cindex = dict(zip(fitted, numerators[0] / n_pairs)) if n_pairs else {}

    rows = []
    for model, mods in MODELS.items():
        model_status = ("fit_failed" if risks[model] is None
                        else "ok" if n_pairs else "no_comparable_pairs")
        rows.append({
            "scheme": scheme, "event": event, "model": model, "modalities": "+".join(mods),
            "cindex": cindex.get(model), "n_patients": n, "n_events": n_events,
            "n_comparable_pairs": n_pairs, "n_fold_blocks": n_blocks, "status": model_status,
        })
    results = pl.DataFrame(rows, schema=CINDEX_SCHEMA)
    if not n_boot or not cindex:
        return results, *_empty_os()

    boot_num, boot_pairs, _ = pair_weighted_concordance(
        time, status, stacked, _bootstrap_weights(n, n_boot), blocks)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = dict(zip(fitted, (boot_num / boot_pairs[:, None]).T))

    def interval(values: np.ndarray) -> tuple[float, float]:
        lower, upper = np.nanquantile(values, [0.025, 0.975])
        return float(lower), float(upper)

    os_rows = [{
        "scheme": scheme, "event": event, "model": model, "modalities": "+".join(MODELS[model]),
        "cindex": cindex[model], **dict(zip(("ci_lower", "ci_upper"), interval(boot[model]))),
        "n_patients": n, "n_events": n_events, "n_boot": n_boot,
    } for model in fitted]
    delta_rows = [{
        "scheme": scheme, "event": event, "model": model, "reference": reference,
        "delta_cindex": cindex[model] - cindex[reference],
        **dict(zip(("ci_lower", "ci_upper"), interval(boot[model] - boot[reference]))),
        "n_boot": n_boot,
    } for model, reference in CONTRASTS if model in cindex and reference in cindex]
    return (results, pl.DataFrame(os_rows, schema=OS_CINDEX_SCHEMA),
            pl.DataFrame(delta_rows, schema=OS_DELTA_SCHEMA))


def _evaluate_task(task: tuple) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load and evaluate one (scheme, event) endpoint; runs in a worker process."""
    scheme, event, directory, source, n_boot = task
    try:
        outcomes = _read_outcomes(pl.scan_parquet(source), event)
        frame = _endpoint_cohort(directory, outcomes)
    except _READ_ERRORS:
        return _status_frame(scheme, event, "missing_inputs"), *_empty_os()
    return evaluate_endpoint(frame, scheme=scheme, event=event, n_boot=n_boot)


def _endpoint_tasks(n_boot: int) -> list[tuple]:
    tasks = []
    for scheme in SCHEMES:
        root = Path(scheme_results_dir(scheme)) / "held_out_risk_scores"
        if not root.is_dir():
            print(f"  missing {root}; skipping {scheme}")
            continue
        source = Path(SURV_PATH) / embedding_file(scheme)
        for directory in sorted(p for p in root.iterdir() if p.is_dir()):
            boot = n_boot if (scheme, directory.name) == OS_ENDPOINT else 0
            tasks.append((scheme, directory.name, directory, source, boot))
    return tasks


def prepare_combined(
    n_boot: int = DEFAULT_N_BOOT, n_jobs: int | None = None, show_progress: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(all-endpoint C-indices, OS model intervals, OS contrast intervals).

    Endpoints run on worker processes; results are assembled in task order, so
    the outputs match a serial run.
    """
    tasks = _endpoint_tasks(n_boot)
    with tqdm(total=len(tasks), desc="Stacked models", unit="event",
              disable=not show_progress) as progress, \
            process_pool(min(resolve_workers(n_jobs), max(len(tasks), 1)),
                         initializer=warnings.simplefilter, initargs=("ignore",)) as pool:
        if pool is None:
            outputs = []
            for task in tasks:
                outputs.append(_evaluate_task(task))
                progress.update(1)
        else:
            futures = [pool.submit(_evaluate_task, task) for task in tasks]
            for _ in as_completed(futures):
                progress.update(1)
            outputs = [future.result() for future in futures]
    schemas = (CINDEX_SCHEMA, OS_CINDEX_SCHEMA, OS_DELTA_SCHEMA)
    return tuple(pl.concat([out[i] for out in outputs]) if outputs else pl.DataFrame(schema=schema)
                 for i, schema in enumerate(schemas))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT,
                        help="bootstrap resamples for the overall-survival intervals (0 disables)")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="worker processes (default: FIGURE_PREP_N_JOBS, else 16 capped at the CPU allocation)")
    args = parser.parse_args()
    if args.n_boot < 0:
        parser.error("--n-boot must be >= 0")
    cindex, os_cindex, os_delta = prepare_combined(n_boot=args.n_boot, n_jobs=args.n_jobs)
    save_figure_data(cindex, "fig3_combined_cindex.csv")
    save_figure_data(os_cindex, "fig3_combined_os_cindex.csv")
    save_figure_data(os_delta, "fig3_combined_os_delta.csv")


if __name__ == "__main__":
    main()
