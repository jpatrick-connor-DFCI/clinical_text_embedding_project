"""Held-out cross-validated risk scores from a (possibly penalized) CoxPH model.

Not split further (unlike run_grid_CoxPH_parallel in grid_search.py): despite having an analogous
no-PCA / PCA+memmap two-path structure, this function is not named in REFACTOR_PLAN.md's "Very
long functions" problem table, so it is transcribed here as one function.
"""

import logging
import os
import shutil
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis

from ._common import (
    _SUPPRESSED_WARNINGS,
    _make_surv_array,
    apply_group_pca_np,
    _scale_continuous_train_test_np,
    _impute_train_test_np,
    _best_mmap_dir,
)

logger = logging.getLogger(__name__)


def get_heldout_risk_scores_CoxPH(
    df: pd.DataFrame,
    base_cols: list[str],
    continuous_vars: list[str],
    penalized_cols: list[str],
    pca_config: dict[str, tuple[list[str], int]] | None = None,
    event_col: str = "event",
    tstop_col: str = "tstop",
    id_col: str = "DFCI_MRN",
    n_splits: int = 5,
    n_jobs: int = -1,
    max_iter: int = 10000,
    penalized: bool = False,
    l1_ratio: float = 0.5,
    alpha: float = 1.0,
    verbose: int = 0,
    ignore_warnings: bool = True,
    backend: str = "loky",          # "loky" or "threading"
    pca_iterated_power: int = 1,
) -> pd.DataFrame:
    """
    Auto-switch behavior:
      - If pca_config is None or {}, run NO-memmap in-RAM fold fitting (fastest for dense text).
      - If pca_config has entries, precompute fold-transformed matrices and MEMMAP them (avoids recomputing PCA/scaling).

    Other principles:
      - float32 X, minimal pandas.
      - y and penalty kept in memory.
    """

    if ignore_warnings:
        for _cat, _msg in _SUPPRESSED_WARNINGS:
            warnings.filterwarnings("ignore", category=_cat, message=_msg)

    if pca_config is None:
        pca_config = {}
    use_memmap = len(pca_config) > 0  # <-- automatic switch

    # ---- Filter invalid (NaN/non-positive tstop, NaN event) ----
    n_before = len(df)
    df = df[df[tstop_col].notna() & (df[tstop_col] > 0) & df[event_col].notna()].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info("get_heldout_risk_scores: dropped %d/%d rows with invalid tstop/event", n_dropped, n_before)

    all_cols = base_cols + penalized_cols
    base_col_set = set(base_cols)

    # ---- X in RAM (float32); NaN in features handled per-fold via _impute_train_test_np ----
    X = df[all_cols].to_numpy(dtype=np.float32, copy=False)

    # ---- Structured survival array ----
    y = _make_surv_array(df[event_col].to_numpy(), df[tstop_col].to_numpy())

    # ---- CV ----
    strat_labels = df[event_col].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)
    splits = list(cv.split(X, strat_labels))  # materialize once

    out_risk = np.full(X.shape[0], np.nan, dtype=np.float64)

    # ---- Parallel backend ----
    parallel_ctx = (
        joblib.parallel_backend("loky", inner_max_num_threads=1)
        if backend == "loky"
        else joblib.parallel_backend("threading")
    )

    # ==========================================================================================
    # Path A: NO PCA => NO MEMMAP (fastest)
    # ==========================================================================================
    if not use_memmap:
        penalty = np.fromiter((0.0 if c in base_col_set else 1.0 for c in all_cols), dtype=np.float32)

        def fit_predict_fold_no_pca(train_idx, test_idx):
            X_tr = np.array(X[train_idx], dtype=np.float32, copy=True)
            X_te = np.array(X[test_idx], dtype=np.float32, copy=True)
            y_tr = y[train_idx]

            # Per-fold imputation then scaling
            X_tr, X_te = _impute_train_test_np(X_tr, X_te)
            X_tr, X_te = _scale_continuous_train_test_np(X_tr, X_te, all_cols, continuous_vars)

            with warnings.catch_warnings():
                for _cat, _msg in _SUPPRESSED_WARNINGS:
                    warnings.filterwarnings("ignore", category=_cat, message=_msg)

                try:
                    if penalized:
                        model = CoxnetSurvivalAnalysis(
                            alphas=[alpha],
                            l1_ratio=l1_ratio,
                            max_iter=max_iter,
                            fit_baseline_model=True,
                            penalty_factor=penalty,
                        )
                        model.fit(X_tr, y_tr)
                        preds = model.predict(X_te)
                    else:
                        model = CoxPHSurvivalAnalysis(n_iter=max_iter)
                        model.fit(X_tr, y_tr)
                        preds = model.predict(X_te)
                except Exception as e:
                    if verbose:
                        print(f"[heldout] failure: {e}")
                    preds = np.full(len(test_idx), np.nan, dtype=np.float64)

            return test_idx, np.asarray(preds, dtype=np.float64)

        with parallel_ctx:
            outs = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
                joblib.delayed(fit_predict_fold_no_pca)(tr, te) for tr, te in splits
            )

        for test_idx, preds in outs:
            out_risk[test_idx] = preds

        if id_col in df.columns:
            return pd.DataFrame({id_col: df[id_col].to_numpy(), "risk_score": out_risk})
        return pd.DataFrame({"index": df.index.to_numpy(), "risk_score": out_risk})

    # ==========================================================================================
    # Path B: PCA present => PRECOMPUTE + MEMMAP transformed X
    # ==========================================================================================
    fold_dir = _best_mmap_dir(prefix="cox_heldout_folds_")
    fold_meta: list[dict] = []
    colnames0 = list(all_cols)

    try:
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_tr = np.array(X[train_idx], dtype=np.float32, copy=True)
            X_te = np.array(X[test_idx], dtype=np.float32, copy=True)
            y_tr = y[train_idx]
            colnames = list(colnames0)

            # Per-fold imputation before PCA/scaling
            X_tr, X_te = _impute_train_test_np(X_tr, X_te)

            with warnings.catch_warnings():
                for _cat, _msg in _SUPPRESSED_WARNINGS:
                    warnings.filterwarnings("ignore", category=_cat, message=_msg)

                for group_name, (cols, k) in pca_config.items():
                    X_tr, X_te, colnames, _ = apply_group_pca_np(
                        X_tr,
                        X_te,
                        colnames,
                        cols,
                        group_name,
                        k,
                        random_state=1234,
                        iterated_power=pca_iterated_power,
                    )

                X_tr, X_te = _scale_continuous_train_test_np(X_tr, X_te, colnames, continuous_vars)

            penalty = np.fromiter((0.0 if c in base_col_set else 1.0 for c in colnames), dtype=np.float32)

            tr_path = os.path.join(fold_dir, f"fold{fold_i}_Xtr.mmap")
            te_path = os.path.join(fold_dir, f"fold{fold_i}_Xte.mmap")

            Xtr_mm = np.memmap(tr_path, mode="w+", dtype=np.float32, shape=X_tr.shape)
            Xte_mm = np.memmap(te_path, mode="w+", dtype=np.float32, shape=X_te.shape)
            Xtr_mm[:] = X_tr
            Xte_mm[:] = X_te
            Xtr_mm.flush()
            Xte_mm.flush()

            fold_meta.append(
                {
                    "fold": fold_i,
                    "test_idx": test_idx,
                    "tr_path": tr_path,
                    "te_path": te_path,
                    "tr_shape": X_tr.shape,
                    "te_shape": X_te.shape,
                    "y_tr": y_tr,        # in-memory
                    "penalty": penalty,  # in-memory
                }
            )

        def fit_predict_fold_memmap(meta: dict):
            fold = meta["fold"]
            test_idx = meta["test_idx"]

            X_tr_np = np.memmap(meta["tr_path"], mode="r", dtype=np.float32, shape=meta["tr_shape"])
            X_te_np = np.memmap(meta["te_path"], mode="r", dtype=np.float32, shape=meta["te_shape"])
            y_tr = meta["y_tr"]
            penalty = meta["penalty"]

            with warnings.catch_warnings():
                for _cat, _msg in _SUPPRESSED_WARNINGS:
                    warnings.filterwarnings("ignore", category=_cat, message=_msg)

                try:
                    if penalized:
                        model = CoxnetSurvivalAnalysis(
                            alphas=[alpha],
                            l1_ratio=l1_ratio,
                            max_iter=max_iter,
                            fit_baseline_model=True,
                            penalty_factor=penalty,
                        )
                        model.fit(X_tr_np, y_tr)
                        preds = model.predict(X_te_np)
                    else:
                        model = CoxPHSurvivalAnalysis(n_iter=max_iter)
                        model.fit(X_tr_np, y_tr)
                        preds = model.predict(X_te_np)
                except Exception as e:
                    if verbose:
                        print(f"[fold {fold}] failure: {e}")
                    preds = np.full(len(test_idx), np.nan, dtype=np.float64)

            return test_idx, np.asarray(preds, dtype=np.float64)

        with parallel_ctx:
            outs = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
                joblib.delayed(fit_predict_fold_memmap)(m) for m in fold_meta
            )

        for test_idx, preds in outs:
            out_risk[test_idx] = preds

        if id_col in df.columns:
            return pd.DataFrame({id_col: df[id_col].to_numpy(), "risk_score": out_risk})
        return pd.DataFrame({"index": df.index.to_numpy(), "risk_score": out_risk})

    finally:
        try:
            shutil.rmtree(fold_dir)
        except Exception:
            pass
