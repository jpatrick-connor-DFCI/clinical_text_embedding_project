"""Elastic-net (Coxnet) grid search.

Split out of the original monolithic cox_models.py (Phase 4 of the refactor): the public
``run_grid_CoxPH_parallel`` still has its original signature and does the shared setup (filtering,
train/val/test split, CV folds, eval times, penalty vector, parallel backend), then dispatches to
one of two extracted, previously-nested execution paths — ``_run_grid_no_pca`` (in-RAM) or
``_run_grid_with_pca`` (precomputed + memmapped folds) — which are otherwise unchanged from the
original function body.
"""

import json
import os
import shutil
import threading
import time
import warnings

import joblib
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split, StratifiedKFold

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc

from ._common import (
    _SUPPRESSED_WARNINGS,
    _make_surv_array,
    evaluate_surv_model,
    apply_group_pca_np,
    _scale_continuous_train_test_np,
    _impute_train_test_np,
    _best_mmap_dir,
)

import logging

logger = logging.getLogger(__name__)


class _LineProgress:
    """Thread-safe, newline-delimited progress bar for piped subprocess output."""

    def __init__(self, total: int, display_every: int) -> None:
        self.total = max(1, int(total))
        self.display_every = max(1, int(display_every))
        self.completed = 0
        self._last_displayed = 0
        self._lock = threading.Lock()
        self._display()

    def _display(self) -> None:
        width = 30
        fraction = min(1.0, self.completed / self.total)
        filled = round(width * fraction)
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"[CV] [{bar}] {self.completed}/{self.total} "
            f"fold×hyperparameter fits ({fraction:.0%})",
            flush=True,
        )

    def update(self) -> None:
        with self._lock:
            self.completed = min(self.total, self.completed + 1)
            if (
                self.completed == self.total
                or self.completed - self._last_displayed >= self.display_every
            ):
                self._display()
                self._last_displayed = self.completed


def _json_array(values) -> str:
    """Compact, standards-compliant JSON for a numeric array of any shape."""
    array = np.asarray(values)
    payload = (
        np.where(np.isfinite(array), array, None).tolist()
        if np.issubdtype(array.dtype, np.number)
        else array.tolist()
    )
    return json.dumps(payload, separators=(",", ":"))


def _ipcw_reference_row(eval_data: str, fold: int | None, ids, y) -> dict:
    """One normalized audit row describing an IPCW reference population."""
    return {
        "eval_data": eval_data,
        "fold": fold,
        "reference_ids": json.dumps(np.asarray(ids).tolist(), separators=(",", ":")),
        "reference_events": _json_array(y["Status"].astype(int)),
        "reference_times": _json_array(y["Survival_in_days"]),
    }


# ==========================================
# Grid search (rewritten with your edits)
# ==========================================

def run_grid_CoxPH_parallel(
    df: pl.DataFrame,
    base_cols: list[str],
    continuous_vars: list[str],
    penalized_cols: list[str],
    l1_ratios: list[float],
    alphas_to_test: list[float],
    pca_config: dict[str, tuple[list[str], int]] | None = None,
    event_col: str = "event",
    tstop_col: str = "tstop",
    max_iter: int = 1000,
    n_splits: int = 5,
    time_evals: tuple[int, int] = (5, 95),
    n_jobs: int = -1,
    verbose: int = 0,
    ignore_warnings: bool = True,
    backend: str = "threading",     # "threading" or "loky"
    pca_iterated_power: int = 1,
    pre_dispatch: int | str = "2*n_jobs",
    batch_size: int | str = "auto",
    parallel_axis: str = "auto",    # "auto", "l1", or "fold"
    id_col: str = "DFCI_MRN",
    return_audit: bool = False,
    show_progress: bool = False,
    adaptive_low_alphas: list[float] | None = None,
):
    """
    Auto-switch behavior:
      - If pca_config is None or {}, runs the NO-memmap in-RAM path (fastest for dense text).
      - If pca_config has entries, runs the PRECOMPUTE+MEMMAP path (avoids recomputing PCA/scaling).

    Why:
      - No PCA => memmaps are mostly overhead (I/O, mmap setup).
      - PCA/scaling => precomputing per fold once is a big win for grid search.

    Parallelism:
      - parallel_axis="l1": parallelize over l1_ratios (best when many l1 values).
      - parallel_axis="fold": parallelize over CV folds inside each l1 (best when few l1 values).
      - parallel_axis="auto": picks "fold" when l1 grid is small, else "l1".
    """

    # Scoped to setup only: the per-fold fits in _run_grid_no_pca / _run_grid_with_pca already
    # suppress these warnings themselves (unconditionally, inside their own catch_warnings()),
    # so this covers only train_test_split/StratifiedKFold below without leaking into the
    # caller's process-global warning filters afterward.
    with warnings.catch_warnings():
        if ignore_warnings:
            for _cat, _msg in _SUPPRESSED_WARNINGS:
                warnings.filterwarnings("ignore", category=_cat, message=_msg)

        if pca_config is None:
            pca_config = {}
        use_memmap = len(pca_config) > 0  # <-- automatic switch

        # ---- Filter invalid (NaN/non-positive tstop, NaN event) ----
        n_before = len(df)
        df = df.filter(
            pl.col(tstop_col).cast(pl.Float64, strict=False).is_finite()
            & (pl.col(tstop_col) > 0)
            & pl.col(event_col).cast(pl.Float64, strict=False).is_finite()
        )
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info("run_grid_CoxPH_parallel: dropped %d/%d rows with invalid tstop/event", n_dropped, n_before)

        # Continuous variables may introduce additional unpenalized predictors
        # (for example N_MET_SITES). A variable explicitly listed as penalized
        # remains penalized even when it is also continuous.
        all_cols = list(dict.fromkeys(base_cols + continuous_vars + penalized_cols))
        base_col_set = set(base_cols) | (set(continuous_vars) - set(penalized_cols))

        # ---- X in RAM (float32); NaN in features is handled per-fold via _impute_train_test_np ----
        X_full = df.select(all_cols).to_numpy().astype(np.float32, copy=False)

        # ---- Structured survival array ----
        y_struct = _make_surv_array(df[event_col].to_numpy(), df[tstop_col].to_numpy())

        # ---- Train/val vs test split ----
        idx = np.arange(X_full.shape[0])
        idx_train_val, idx_test = train_test_split(
            idx,
            test_size=0.2,
            stratify=df[event_col].cast(pl.Int64).to_numpy(),
            random_state=1234,
        )

        X_train_val = X_full[idx_train_val]
        X_test = X_full[idx_test]
        y_train_val = y_struct[idx_train_val]
        y_test = y_struct[idx_test]
        ids = df[id_col].to_numpy() if id_col in df.columns else np.arange(df.height)
        ids_train_val = ids[idx_train_val]

        # ---- CV ----
        event_int_np = df[event_col].cast(pl.Int64).to_numpy()
        strat_labels = event_int_np[idx_train_val]
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)
        folds = list(cv.split(X_train_val, strat_labels))  # materialize once

    # ---- Evaluation time points ----
    lower, upper = np.percentile(y_train_val["Survival_in_days"], [time_evals[0], time_evals[1]])
    eval_times = np.linspace(lower, upper, 50) if lower != upper else np.array([lower], dtype=float)

    ipcw_reference_df = None
    if return_audit:
        ipcw_rows = [_ipcw_reference_row("test", None, ids_train_val, y_train_val)]
        ipcw_rows.extend(
            _ipcw_reference_row("cv", fold_i, ids_train_val[tr], y_train_val[tr])
            for fold_i, (tr, _va) in enumerate(folds)
        )
        ipcw_reference_df = pl.DataFrame(ipcw_rows)

    # ---- penalty once if no PCA; recomputed per fold if PCA changes columns ----
    penalty_no_pca = np.fromiter((0.0 if c in base_col_set else 1.0 for c in all_cols), dtype=np.float32)

    # ---- Parallel backend context ----
    # A factory, not a single context object. joblib.parallel_backend is NOT reentrant: its
    # __exit__ unregisters the backend globally, so re-entering the same instance silently falls
    # back to the default (loky). This function enters the context once per alpha grid, and nested
    # CV calls it once per outer fold, so a shared instance meant every use after the first ran on
    # loky regardless of `backend` -- pickling the closures to worker processes, which fails
    # outright once anything unpicklable (a progress bar's lock) is captured.
    def parallel_ctx():
        return (
            joblib.parallel_backend("loky", inner_max_num_threads=1)
            if backend == "loky"
            else joblib.parallel_backend("threading")
        )

    if parallel_axis not in {"auto", "l1", "fold"}:
        raise ValueError("parallel_axis must be one of {'auto', 'l1', 'fold'}")
    if parallel_axis == "auto":
        parallel_axis_eff = "fold" if len(l1_ratios) <= 2 else "l1"
    else:
        parallel_axis_eff = parallel_axis

    if show_progress and backend != "threading":
        raise ValueError("show_progress=True currently requires backend='threading'")

    def _run_alpha_grid(alpha_grid: list[float]):
        progress = None
        if show_progress:
            n_combinations = len(folds) * len(l1_ratios) * len(alpha_grid)
            progress = _LineProgress(
                total=n_combinations,
                display_every=len(alpha_grid),
            )
        if not use_memmap:
            return _run_grid_no_pca(
                X_train_val, X_test, y_train_val, y_test, folds,
                all_cols, continuous_vars, alpha_grid, l1_ratios,
                max_iter, eval_times, penalty_no_pca,
                parallel_ctx, parallel_axis_eff, n_jobs, verbose, pre_dispatch, batch_size,
                progress,
            )
        return _run_grid_with_pca(
            X_train_val, X_test, y_train_val, y_test, folds,
            all_cols, continuous_vars, base_col_set, pca_config, pca_iterated_power,
            alpha_grid, l1_ratios, max_iter, eval_times,
            parallel_ctx, parallel_axis_eff, n_jobs, verbose, pre_dispatch, batch_size,
            progress,
        )

    result = _run_alpha_grid(alphas_to_test)

    if adaptive_low_alphas:
        primary_best = (
            result[1].filter(pl.col("mean_auc(t)").is_finite())
            .sort("mean_auc(t)", descending=True)
            .row(0, named=True)
        )
        lower_boundary = float(np.min(alphas_to_test))
        if np.isclose(float(primary_best["alpha"]), lower_boundary, rtol=1e-12, atol=0.0):
            print(
                f"[CV] Best alpha {lower_boundary:.3e} reached the primary lower boundary; "
                f"evaluating {len(adaptive_low_alphas)} lower alphas.",
                flush=True,
            )
            try:
                low_result = _run_alpha_grid(adaptive_low_alphas)
            except RuntimeError as exc:
                if not str(exc).startswith("All CV evaluations failed"):
                    raise
                print(f"[CV] Low-alpha refinement failed; retaining primary grid: {exc}", flush=True)
            else:
                low_best = (
                    low_result[1].filter(pl.col("mean_auc(t)").is_finite())
                    .sort("mean_auc(t)", descending=True)
                    .row(0, named=True)
                )
                chosen = (
                    low_result
                    if float(low_best["mean_auc(t)"]) > float(primary_best["mean_auc(t)"])
                    else result
                )
                combined_val = pl.concat([result[1], low_result[1]], how="vertical")
                result = (chosen[0], combined_val, chosen[2])

    return (*result, ipcw_reference_df) if return_audit else result


# ==========================================================================================
# Path A: NO PCA => NO MEMMAP (fastest)
# ==========================================================================================

def _run_grid_no_pca(
    X_train_val, X_test, y_train_val, y_test, folds,
    all_cols, continuous_vars, alphas_to_test, l1_ratios,
    max_iter, eval_times, penalty_no_pca,
    parallel_ctx, parallel_axis_eff, n_jobs, verbose, pre_dispatch, batch_size,
    progress,
):
    alphas_desc = np.sort(alphas_to_test)[::-1].tolist()
    n_alphas = len(alphas_desc)
    # Collected across every l1 path so a total failure can report why (see the
    # RuntimeError below). Appended to under joblib's threading backend, which is
    # why this is a plain list mutated in the sequential result-gathering loop
    # rather than inside the parallel worker.
    grid_error_messages: list[str] = []

    def _evaluate_fold_no_pca(fi: int, tr: np.ndarray, va: np.ndarray, l1_ratio: float):
        fold_auc = np.full(n_alphas, np.nan)
        fold_cindex = np.full(n_alphas, np.nan)
        fold_auc_curves = np.full((n_alphas, len(eval_times)), np.nan)
        start = time.time()
        error_flag = False
        # First exception seen in this fold. Every alpha's failure is otherwise only
        # logger.debug'd, and run_feature_comp_task never configures logging, so a
        # fully-failed grid used to surface as a bare "all NaN" with no cause.
        first_error = None

        # Fancy indexing already copies, and X_train_val is float32 (built at grid_search.py:99),
        # so the previous np.array(..., copy=True) was a redundant second copy (Phase-A A4).
        # np.asarray is a no-op here since dtype already matches (unlike np.array, which
        # defaults to copy=True regardless).
        X_tr = np.asarray(X_train_val[tr], dtype=np.float32)
        X_va = np.asarray(X_train_val[va], dtype=np.float32)
        y_tr = y_train_val[tr]
        y_va = y_train_val[va]

        # Per-fold imputation then scaling
        X_tr, X_va = _impute_train_test_np(X_tr, X_va)
        X_tr, X_va = _scale_continuous_train_test_np(X_tr, X_va, all_cols, continuous_vars)

        with warnings.catch_warnings():
            for _cat, _msg in _SUPPRESSED_WARNINGS:
                warnings.filterwarnings("ignore", category=_cat, message=_msg)

            for ai, a in enumerate(alphas_desc):
                try:
                    m = CoxnetSurvivalAnalysis(
                        alphas=[a], l1_ratio=l1_ratio,
                        max_iter=max_iter, fit_baseline_model=False,
                        penalty_factor=penalty_no_pca,
                    )
                    m.fit(X_tr, y_tr)
                    predictions = m.predict(X_va)
                    fold_auc_curves[ai], fold_auc[ai] = cumulative_dynamic_auc(
                        y_tr, y_va, predictions, eval_times
                    )
                    fold_cindex[ai] = concordance_index_censored(
                        y_va["Status"], y_va["Survival_in_days"], predictions
                    )[0]
                except Exception as e:
                    logger.debug("Grid CV fold %d, alpha=%.2e, l1=%.2f failed: %s", fi, a, l1_ratio, e)
                    error_flag = True
                    if first_error is None:
                        first_error = f"fold {fi}, alpha={a:.2e}, l1={l1_ratio:.2f}: {type(e).__name__}: {e}"
                finally:
                    if progress is not None:
                        progress.update()

        return fi, fold_auc, fold_cindex, fold_auc_curves, time.time() - start, error_flag, first_error

    def evaluate_l1_path_no_pca(l1_ratio: float):
        fold_aucs = np.full((len(folds), n_alphas), np.nan)
        fold_cindices = np.full((len(folds), n_alphas), np.nan)
        fold_auc_curves = np.full((len(folds), n_alphas, len(eval_times)), np.nan)
        fold_times = np.zeros(len(folds))
        fold_errors = np.zeros(len(folds), dtype=bool)
        fold_error_messages = []

        if parallel_axis_eff == "fold":
            with parallel_ctx():
                fold_results = joblib.Parallel(
                    n_jobs=n_jobs,
                    verbose=verbose,
                    pre_dispatch=pre_dispatch,
                    batch_size=batch_size,
                )(
                    joblib.delayed(_evaluate_fold_no_pca)(fi, tr, va, l1_ratio)
                    for fi, (tr, va) in enumerate(folds)
                )
        else:
            fold_results = [
                _evaluate_fold_no_pca(fi, tr, va, l1_ratio)
                for fi, (tr, va) in enumerate(folds)
            ]

        for fi, fold_auc, fold_cindex, fold_curves, fold_time, error_flag, first_error in fold_results:
            fold_aucs[fi, :] = fold_auc
            fold_cindices[fi, :] = fold_cindex
            fold_auc_curves[fi, :, :] = fold_curves
            fold_times[fi] = fold_time
            fold_errors[fi] = error_flag
            if first_error is not None:
                fold_error_messages.append(first_error)
        grid_error_messages.extend(fold_error_messages)

        rows = []
        for ai, alpha in enumerate(alphas_desc):
            rows.append([
                float(l1_ratio),
                float(alpha),
                float(np.nanmean(fold_cindices[:, ai])),
                float(np.nanmean(fold_aucs[:, ai])),
                np.nan,  # mean_ibs: not computed in CV, same as mean_c_index above; always NaN here
                float(np.mean(fold_times)),
                json.dumps(fold_times.tolist()),
                json.dumps(fold_errors.tolist()),  # fold_error_flags: one bool per fold for this whole l1-path, identical across every alpha row -- a single alpha's failure marks the entire path, not just that row
                float(np.mean(fold_errors)),  # error_rate: same per-l1-path (not per-alpha) caveat as fold_error_flags above
                json.dumps([0] * len(folds)),
                int(np.sum(~np.isnan(fold_aucs[:, ai]))),  # n_folds_contributing: per-alpha count behind mean_auc(t); no minimum enforced, a 1-fold mean is a valid winner
                _json_array(eval_times),
                _json_array(fold_auc_curves[:, ai, :]),
                _json_array(fold_cindices[:, ai]),
            ])
        return rows

    if parallel_axis_eff == "l1":
        with parallel_ctx():
            nested = joblib.Parallel(
                n_jobs=n_jobs,
                verbose=verbose,
                pre_dispatch=pre_dispatch,
                batch_size=batch_size,
            )(
                joblib.delayed(evaluate_l1_path_no_pca)(l1)
                for l1 in l1_ratios
            )
    else:
        nested = [evaluate_l1_path_no_pca(l1) for l1 in l1_ratios]
    results = [row for batch in nested for row in batch]

    cv_results_df = pl.DataFrame(
        results,
        schema=[
            "l1_ratio",
            "alpha",
            "mean_c_index",
            "mean_auc(t)",
            "mean_ibs",
            "mean_train_time",
            "fold_times",
            "fold_error_flags",
            "error_rate",
            "fold_warning_counts",
            "n_folds_contributing",
            "auc_eval_times",
            "fold_auc_curves",
            "fold_c_indices",
        ],
        orient="row",
    )

    valid_cv = cv_results_df.filter(pl.col("mean_auc(t)").is_finite())
    if valid_cv.is_empty():
        detail = ""
        if grid_error_messages:
            unique_errors = list(dict.fromkeys(grid_error_messages))
            shown = "; ".join(unique_errors[:3])
            more = f" (+{len(unique_errors) - 3} more)" if len(unique_errors) > 3 else ""
            detail = f" Underlying fold errors: {shown}{more}"
        else:
            # No exception was raised anywhere: every fit succeeded but produced a
            # non-finite AUC, which points at eval_times / follow-up coverage rather
            # than at the solver.
            detail = (
                " No fold raised an exception, so every fit succeeded but scored non-finite "
                f"AUC — check follow-up coverage against the {len(eval_times)} eval time(s) "
                f"spanning [{eval_times[0]:.1f}, {eval_times[-1]:.1f}]."
            )
        raise RuntimeError(
            "All CV evaluations failed (all NaN mean_auc(t)). Check data and parameters." + detail
        )
    opt = valid_cv.sort("mean_auc(t)", descending=True).row(0, named=True)
    opt_l1, opt_alpha = float(opt["l1_ratio"]), float(opt["alpha"])
    logger.info(
        "Selected l1_ratio=%.3f alpha=%.3e with mean_auc(t)=%.4f from %d/%d fold(s) (no min_folds gate)",
        opt_l1, opt_alpha, opt["mean_auc(t)"], int(opt["n_folds_contributing"]), len(folds),
    )

    # Impute then scale continuous variables for final fit
    X_trval_scaled = np.array(X_train_val, dtype=np.float32, copy=True)
    X_test_scaled = np.array(X_test, dtype=np.float32, copy=True)
    X_trval_scaled, X_test_scaled = _impute_train_test_np(X_trval_scaled, X_test_scaled)
    X_trval_scaled, X_test_scaled = _scale_continuous_train_test_np(
        X_trval_scaled, X_test_scaled, all_cols, continuous_vars)

    try:
        final_model = CoxnetSurvivalAnalysis(
            alphas=[opt_alpha],
            l1_ratio=opt_l1,
            max_iter=max_iter,
            fit_baseline_model=True,
            penalty_factor=penalty_no_pca,
        )
        final_model.fit(X_trval_scaled, y_train_val)
        mean_auc, ibs, cidx, auc_curve = evaluate_surv_model(
            final_model, X_test_scaled, y_train_val, y_test, eval_times,
            return_auc_curve=True,
        )
    except Exception as e:
        logger.warning("Grid search final model (no PCA) failed: %s", e)
        final_model, mean_auc, ibs, cidx = None, np.nan, np.nan, np.nan
        auc_curve = np.full(len(eval_times), np.nan)

    test_df = pl.DataFrame({
        "mean_auc(t)": [mean_auc], "mean_ibs": [ibs], "mean_c_index": [cidx],
        "auc_eval_times": [_json_array(eval_times)], "auc_curve": [_json_array(auc_curve)],
    })
    return test_df, cv_results_df, final_model


# ==========================================================================================
# Path B: PCA present => PRECOMPUTE folds + MEMMAP transformed X
# ==========================================================================================

def _run_grid_with_pca(
    X_train_val, X_test, y_train_val, y_test, folds,
    all_cols, continuous_vars, base_col_set, pca_config, pca_iterated_power,
    alphas_to_test, l1_ratios, max_iter, eval_times,
    parallel_ctx, parallel_axis_eff, n_jobs, verbose, pre_dispatch, batch_size,
    progress,
):
    fold_dir = _best_mmap_dir(prefix="coxnet_folds_")
    fold_meta: list[dict] = []
    colnames0 = list(all_cols)

    try:
        for fold_i, (tr, va) in enumerate(folds):
            X_tr = np.array(X_train_val[tr], dtype=np.float32, copy=True)
            X_va = np.array(X_train_val[va], dtype=np.float32, copy=True)
            y_tr = y_train_val[tr]
            y_va = y_train_val[va]
            colnames = list(colnames0)

            # Per-fold imputation before PCA/scaling
            X_tr, X_va = _impute_train_test_np(X_tr, X_va)

            with warnings.catch_warnings():
                for _cat, _msg in _SUPPRESSED_WARNINGS:
                    warnings.filterwarnings("ignore", category=_cat, message=_msg)

                for gname, (cols, k) in pca_config.items():
                    X_tr, X_va, colnames, _ = apply_group_pca_np(
                        X_tr, X_va, colnames, cols, gname, k,
                        random_state=1234,
                        iterated_power=pca_iterated_power,
                    )

                X_tr, X_va = _scale_continuous_train_test_np(X_tr, X_va, colnames, continuous_vars)

            penalty = np.fromiter((0.0 if c in base_col_set else 1.0 for c in colnames), dtype=np.float32)

            tr_path = os.path.join(fold_dir, f"fold{fold_i}_Xtr.mmap")
            va_path = os.path.join(fold_dir, f"fold{fold_i}_Xva.mmap")

            Xtr_mm = np.memmap(tr_path, mode="w+", dtype=np.float32, shape=X_tr.shape)
            Xva_mm = np.memmap(va_path, mode="w+", dtype=np.float32, shape=X_va.shape)
            Xtr_mm[:] = X_tr
            Xva_mm[:] = X_va
            Xtr_mm.flush()
            Xva_mm.flush()

            fold_meta.append(
                {
                    "fold": fold_i,
                    "tr_path": tr_path,
                    "va_path": va_path,
                    "tr_shape": X_tr.shape,
                    "va_shape": X_va.shape,
                    "y_tr": y_tr,        # in-memory
                    "y_va": y_va,        # in-memory
                    "penalty": penalty,  # in-memory
                }
            )

        alphas_desc_b = np.sort(alphas_to_test)[::-1].tolist()
        n_alphas_b = len(alphas_desc_b)

        def _evaluate_fold_with_pca(fi: int, meta: dict, l1_ratio: float):
            fold_auc = np.full(n_alphas_b, np.nan)
            fold_cindex = np.full(n_alphas_b, np.nan)
            fold_auc_curves = np.full((n_alphas_b, len(eval_times)), np.nan)
            start = time.time()
            error_flag = False

            X_tr_np = np.memmap(meta["tr_path"], mode="r", dtype=np.float32, shape=meta["tr_shape"])
            X_va_np = np.memmap(meta["va_path"], mode="r", dtype=np.float32, shape=meta["va_shape"])
            y_tr = meta["y_tr"]
            y_va = meta["y_va"]
            penalty = meta["penalty"]

            with warnings.catch_warnings():
                for _cat, _msg in _SUPPRESSED_WARNINGS:
                    warnings.filterwarnings("ignore", category=_cat, message=_msg)

                for ai, a in enumerate(alphas_desc_b):
                    try:
                        m = CoxnetSurvivalAnalysis(
                            alphas=[a], l1_ratio=l1_ratio,
                            max_iter=max_iter, fit_baseline_model=False,
                            penalty_factor=penalty,
                        )
                        m.fit(X_tr_np, y_tr)
                        predictions = m.predict(X_va_np)
                        fold_auc_curves[ai], fold_auc[ai] = cumulative_dynamic_auc(
                            y_tr, y_va, predictions, eval_times
                        )
                        fold_cindex[ai] = concordance_index_censored(
                            y_va["Status"], y_va["Survival_in_days"], predictions
                        )[0]
                    except Exception as e:
                        logger.debug("Grid CV fold %d (PCA), alpha=%.2e, l1=%.2f failed: %s", fi, a, l1_ratio, e)
                        error_flag = True
                    finally:
                        if progress is not None:
                            progress.update()

            return fi, fold_auc, fold_cindex, fold_auc_curves, time.time() - start, error_flag

        def evaluate_l1_path_with_pca(l1_ratio: float):
            fold_aucs = np.full((len(fold_meta), n_alphas_b), np.nan)
            fold_cindices = np.full((len(fold_meta), n_alphas_b), np.nan)
            fold_auc_curves = np.full((len(fold_meta), n_alphas_b, len(eval_times)), np.nan)
            fold_times = np.zeros(len(fold_meta))
            fold_errors = np.zeros(len(fold_meta), dtype=bool)

            if parallel_axis_eff == "fold":
                with parallel_ctx():
                    fold_results = joblib.Parallel(
                        n_jobs=n_jobs,
                        verbose=verbose,
                        pre_dispatch=pre_dispatch,
                        batch_size=batch_size,
                    )(
                        joblib.delayed(_evaluate_fold_with_pca)(fi, meta, l1_ratio)
                        for fi, meta in enumerate(fold_meta)
                    )
            else:
                fold_results = [
                    _evaluate_fold_with_pca(fi, meta, l1_ratio)
                    for fi, meta in enumerate(fold_meta)
                ]

            for fi, fold_auc, fold_cindex, fold_curves, fold_time, error_flag in fold_results:
                fold_aucs[fi, :] = fold_auc
                fold_cindices[fi, :] = fold_cindex
                fold_auc_curves[fi, :, :] = fold_curves
                fold_times[fi] = fold_time
                fold_errors[fi] = error_flag

            rows = []
            for ai, alpha in enumerate(alphas_desc_b):
                rows.append([
                    float(l1_ratio),
                    float(alpha),
                    float(np.nanmean(fold_cindices[:, ai])),
                    float(np.nanmean(fold_aucs[:, ai])),
                    np.nan,  # mean_ibs: not computed in CV, always NaN here
                    float(np.mean(fold_times)),
                    json.dumps(fold_times.tolist()),
                    json.dumps(fold_errors.tolist()),  # fold_error_flags: per-l1-path, identical across every alpha row
                    float(np.mean(fold_errors)),  # error_rate: per-l1-path, not per-alpha
                    json.dumps([0] * len(fold_meta)),
                    int(np.sum(~np.isnan(fold_aucs[:, ai]))),  # n_folds_contributing: see no-PCA path's comment
                    _json_array(eval_times),
                    _json_array(fold_auc_curves[:, ai, :]),
                    _json_array(fold_cindices[:, ai]),
                ])
            return rows

        if parallel_axis_eff == "l1":
            with parallel_ctx():
                nested = joblib.Parallel(
                    n_jobs=n_jobs,
                    verbose=verbose,
                    pre_dispatch=pre_dispatch,
                    batch_size=batch_size,
                )(
                    joblib.delayed(evaluate_l1_path_with_pca)(l1)
                    for l1 in l1_ratios
                )
        else:
            nested = [evaluate_l1_path_with_pca(l1) for l1 in l1_ratios]
        results = [row for batch in nested for row in batch]

        cv_results_df = pl.DataFrame(
            results,
            schema=[
                "l1_ratio",
                "alpha",
                "mean_c_index",
                "mean_auc(t)",
                "mean_ibs",
                "mean_train_time",
                "fold_times",
                "fold_error_flags",
                "error_rate",
                "fold_warning_counts",
                "n_folds_contributing",
                "auc_eval_times",
                "fold_auc_curves",
                "fold_c_indices",
            ],
            orient="row",
        )

        # ---- final fit (apply same preprocessing to train_val/test) ----
        valid_cv = cv_results_df.filter(pl.col("mean_auc(t)").is_finite())
        if valid_cv.is_empty():
            raise RuntimeError("All CV evaluations failed (all NaN mean_auc(t)). Check data and parameters.")
        opt = valid_cv.sort("mean_auc(t)", descending=True).row(0, named=True)
        opt_l1, opt_alpha = float(opt["l1_ratio"]), float(opt["alpha"])
        logger.info(
            "Selected l1_ratio=%.3f alpha=%.3e with mean_auc(t)=%.4f from %d/%d fold(s) (no min_folds gate)",
            opt_l1, opt_alpha, opt["mean_auc(t)"], int(opt["n_folds_contributing"]), len(fold_meta),
        )

        X_trval = np.array(X_train_val, dtype=np.float32, copy=True)
        X_te = np.array(X_test, dtype=np.float32, copy=True)
        colnames = list(colnames0)

        # Per-fold imputation before PCA/scaling for final model
        X_trval, X_te = _impute_train_test_np(X_trval, X_te)

        with warnings.catch_warnings():
            for _cat, _msg in _SUPPRESSED_WARNINGS:
                warnings.filterwarnings("ignore", category=_cat, message=_msg)

            for gname, (cols, k) in pca_config.items():
                X_trval, X_te, colnames, _ = apply_group_pca_np(
                    X_trval, X_te, colnames, cols, gname, k,
                    random_state=1234,
                    iterated_power=pca_iterated_power,
                )

            X_trval, X_te = _scale_continuous_train_test_np(X_trval, X_te, colnames, continuous_vars)

        penalty_final = np.fromiter((0.0 if c in base_col_set else 1.0 for c in colnames), dtype=np.float32)

        try:
            final_model = CoxnetSurvivalAnalysis(
                alphas=[opt_alpha],
                l1_ratio=opt_l1,
                max_iter=max_iter,
                fit_baseline_model=True,
                penalty_factor=penalty_final,
            )
            final_model.fit(X_trval, y_train_val)
            mean_auc, ibs, cidx, auc_curve = evaluate_surv_model(
                final_model, X_te, y_train_val, y_test, eval_times,
                return_auc_curve=True,
            )
        except Exception as e:
            logger.warning("Grid search final model (PCA) failed: %s", e)
            final_model, mean_auc, ibs, cidx = None, np.nan, np.nan, np.nan
            auc_curve = np.full(len(eval_times), np.nan)

        test_df = pl.DataFrame({
            "mean_auc(t)": [mean_auc], "mean_ibs": [ibs], "mean_c_index": [cidx],
            "auc_eval_times": [_json_array(eval_times)], "auc_curve": [_json_array(auc_curve)],
        })
        return test_df, cv_results_df, final_model

    finally:
        try:
            shutil.rmtree(fold_dir)
        except Exception:
            pass
