import os
import shutil
import time
import warnings
import tempfile
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score

def scale_model_data(X_train: pd.DataFrame, X_test: pd.DataFrame, continuous_vars: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize continuous variables in training and test sets.

    Args:
        X_train (pd.DataFrame): Training feature DataFrame.
        X_test (pd.DataFrame): Test feature DataFrame.
        continuous_vars (list[str]): List of column names to standardize.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Scaled X_train and X_test.
    """
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    scaler = StandardScaler().fit(X_train_scaled[continuous_vars])
    X_train_scaled[continuous_vars] = scaler.transform(X_train_scaled[continuous_vars])
    X_test_scaled[continuous_vars] = scaler.transform(X_test_scaled[continuous_vars])
    return X_train_scaled, X_test_scaled

def evaluate_surv_model(surv_model, X_eval, y_train, y_eval, eval_times: np.ndarray) -> tuple[float, float, float]:
    """
    Evaluate a survival model on test/validation data.

    Computes:
        - Time-dependent AUC
        - Integrated Brier Score
        - Concordance index

    Args:
        surv_model: Fitted survival model (CoxPHSurvivalAnalysis or CoxnetSurvivalAnalysis).
        X_eval (pd.DataFrame): Evaluation features.
        y_train: Training structured survival array for dynamic AUC calculation.
        y_eval: Evaluation structured survival array.
        eval_times (np.ndarray): Times at which to compute AUC and Brier score.

    Returns:
        tuple[float, float, float]: mean_auc_t, ibs, c_index. Returns NaN on failure.
    """
    try:
        chf_funcs = surv_model.predict_cumulative_hazard_function(X_eval, return_array=False)
        risk_scores = np.vstack([chf(eval_times) for chf in chf_funcs])
        surv_probs = np.exp(-risk_scores)  # S(t) = exp(-H(t)); IBS requires survival probs in [0,1]
        _, mean_auc_t = cumulative_dynamic_auc(y_train, y_eval, risk_scores, eval_times)
        ibs = integrated_brier_score(y_train, y_eval, surv_probs, eval_times)
        c_index = surv_model.score(X_eval, y_eval)
    except Exception:
        mean_auc_t, ibs, c_index = np.nan, np.nan, np.nan
    return mean_auc_t, ibs, c_index

def _make_surv_array(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Build a sksurv-compatible structured survival array without list(zip(...)) overhead."""
    y = np.empty(len(event), dtype=[("Status", "?"), ("Survival_in_days", "<f8")])
    y["Status"] = np.asarray(event, dtype=bool)
    y["Survival_in_days"] = np.asarray(time, dtype=np.float64)
    return y


def run_base_CoxPH(df: pd.DataFrame, base_cols: list[str], continuous_vars: list[str],
                   event_col: str = 'event', tstop_col: str = 'tstop',
                   max_iter: int = 1000, n_splits: int = 5,
                   time_evals: tuple[int, int] = (5, 95),
                   test_size: float = 0.2,
                   ignore_warnings: bool = True) -> pd.DataFrame:
    """
    Fit a baseline CoxPH model using base covariates.
    - 20% of data is held out as a test set.
    - Cross-validation is performed on the remaining data.
    """
    if ignore_warnings:
        warnings.filterwarnings('ignore')

    df = df[df[tstop_col] > 0].copy()

    # Split into train+val and held-out test set
    strat_labels = df[event_col].astype(int)
    df_trainval, df_test = train_test_split(df, test_size=test_size, 
                                            stratify=strat_labels, random_state=1234)

    Xt_trainval = df_trainval[base_cols]
    y_trainval = _make_surv_array(df_trainval[event_col].to_numpy(), df_trainval[tstop_col].to_numpy())

    Xt_test = df_test[base_cols]
    y_test = _make_surv_array(df_test[event_col].to_numpy(), df_test[tstop_col].to_numpy())

    # Compute evaluation times
    lower, upper = np.percentile(df_trainval[tstop_col], [time_evals[0], time_evals[1]])
    eval_times = np.linspace(lower, upper, 50) if lower != upper else np.array([lower], dtype=float)

    # --- Cross-validation on train+val ---
    c_index_vals, mean_auc_t_vals, ibs_vals = [], [], []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)

    for train_idx, val_idx in cv.split(Xt_trainval, df_trainval[event_col]):
        X_train, X_val = Xt_trainval.iloc[train_idx], Xt_trainval.iloc[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        X_train, X_val = scale_model_data(X_train, X_val, continuous_vars)

        cox_model = CoxPHSurvivalAnalysis(n_iter=max_iter)
        try:
            cox_model.fit(X_train, y_train)
            mean_auc_t, ibs, c_index = evaluate_surv_model(cox_model, X_val, y_train, y_val, eval_times)
        except Exception:
            mean_auc_t, ibs, c_index = np.nan, np.nan, np.nan

        c_index_vals.append(c_index)
        mean_auc_t_vals.append(mean_auc_t)
        ibs_vals.append(ibs)

    # --- Evaluate on held-out test set ---
    # Fit CoxPH on full train+val
    Xt_trainval_scaled, Xt_test_scaled = scale_model_data(Xt_trainval, Xt_test, continuous_vars)
    cox_model_final = CoxPHSurvivalAnalysis(n_iter=max_iter)
    try:
        cox_model_final.fit(Xt_trainval_scaled, y_trainval)
        mean_auc_t_test, ibs_test, c_index_test = evaluate_surv_model(
            cox_model_final, Xt_test_scaled, y_trainval, y_test, eval_times)
    except Exception:
        mean_auc_t_test, ibs_test, c_index_test = np.nan, np.nan, np.nan

    # Return summary
    return pd.DataFrame([
        ['cv_data', np.nanmean(c_index_vals), np.nanmean(mean_auc_t_vals), np.nanmean(ibs_vals)],
        ['test_data', c_index_test, mean_auc_t_test, ibs_test]
    ], columns=['eval_data', 'mean_c_index', 'mean_auc(t)', 'mean_ibs'])

# =========================
# Fast NumPy preprocessing
# =========================

def _standardize_train_test(Xtr: np.ndarray, Xte: np.ndarray, eps: float = 1e-12):
    mu = Xtr.mean(axis=0, dtype=np.float32)
    sig = Xtr.std(axis=0, dtype=np.float32)
    sig = np.maximum(sig, eps).astype(np.float32, copy=False)
    return (Xtr - mu) / sig, (Xte - mu) / sig

def apply_group_pca_np(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    colnames: list[str],
    group_cols: list[str],
    group_name: str,
    k: int,
    random_state: int = 1234,
    iterated_power: int = 1,   # faster default at your scale
):
    if k is None or k <= 0:
        return X_tr, X_te, colnames, []

    name_to_idx = {c: i for i, c in enumerate(colnames)}
    idx = [name_to_idx[c] for c in group_cols if c in name_to_idx]
    if not idx:
        return X_tr, X_te, colnames, []

    n_train = X_tr.shape[0]
    k_eff = int(min(k, len(idx), n_train))
    if k_eff <= 0:
        return X_tr, X_te, colnames, []

    idx = np.asarray(idx, dtype=np.int32)

    G_tr = X_tr[:, idx]
    G_te = X_te[:, idx]
    G_trz, G_tez = _standardize_train_test(G_tr, G_te)

    pca = PCA(
        n_components=k_eff,
        svd_solver="randomized",
        random_state=random_state,
        iterated_power=iterated_power,
    )
    Z_tr = pca.fit_transform(G_trz).astype(np.float32, copy=False)
    Z_te = pca.transform(G_tez).astype(np.float32, copy=False)

    pc_names = [f"{group_name}_PC{i+1}" for i in range(k_eff)]

    keep_mask = np.ones(len(colnames), dtype=bool)
    keep_mask[idx] = False

    X_tr_new = np.concatenate([X_tr[:, keep_mask], Z_tr], axis=1)
    X_te_new = np.concatenate([X_te[:, keep_mask], Z_te], axis=1)
    new_colnames = [c for j, c in enumerate(colnames) if keep_mask[j]] + pc_names

    return X_tr_new, X_te_new, new_colnames, pc_names

def _scale_continuous_train_test_np(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    colnames: list[str],
    continuous_vars: list[str],
    eps: float = 1e-12,
):
    name_to_idx = {c: i for i, c in enumerate(colnames)}
    idx = [name_to_idx[c] for c in continuous_vars if c in name_to_idx]
    if not idx:
        return X_tr, X_te

    idx = np.asarray(idx, dtype=np.int32)
    mu = X_tr[:, idx].mean(axis=0, dtype=np.float32)
    sig = X_tr[:, idx].std(axis=0, dtype=np.float32)
    sig = np.maximum(sig, eps).astype(np.float32, copy=False)

    X_tr[:, idx] = (X_tr[:, idx] - mu) / sig
    X_te[:, idx] = (X_te[:, idx] - mu) / sig
    return X_tr, X_te

def _best_mmap_dir(prefix="coxnet_folds_"):
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return tempfile.mkdtemp(prefix=prefix, dir="/dev/shm")
    return tempfile.mkdtemp(prefix=prefix)


# ==========================================
# Grid search (rewritten with your edits)
# ==========================================

def run_grid_CoxPH_parallel(
    df: pd.DataFrame,
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
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
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

    if ignore_warnings:
        warnings.filterwarnings("ignore")

    if pca_config is None:
        pca_config = {}
    use_memmap = len(pca_config) > 0  # <-- automatic switch

    # ---- Filter invalid ----
    df = df[df[tstop_col] > 0].copy()

    all_cols = base_cols + penalized_cols
    base_col_set = set(base_cols)

    # ---- X in RAM (float32) ----
    X_full = df[all_cols].to_numpy(dtype=np.float32, copy=False)

    # ---- Structured survival array ----
    y_struct = _make_surv_array(df[event_col].to_numpy(), df[tstop_col].to_numpy())

    # ---- Train/val vs test split ----
    idx = np.arange(X_full.shape[0])
    idx_train_val, idx_test = train_test_split(
        idx,
        test_size=0.2,
        stratify=df[event_col].astype(int),
        random_state=1234,
    )

    X_train_val = X_full[idx_train_val]
    X_test = X_full[idx_test]
    y_train_val = y_struct[idx_train_val]
    y_test = y_struct[idx_test]

    # ---- CV ----
    strat_labels = df[event_col].astype(int).iloc[idx_train_val].to_numpy()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)
    folds = list(cv.split(X_train_val, strat_labels))  # materialize once

    # ---- Evaluation time points ----
    lower, upper = np.percentile(y_train_val["Survival_in_days"], [time_evals[0], time_evals[1]])
    eval_times = np.linspace(lower, upper, 50) if lower != upper else np.array([lower], dtype=float)

    # ---- penalty once if no PCA; recomputed per fold if PCA changes columns ----
    penalty_no_pca = np.fromiter((0.0 if c in base_col_set else 1.0 for c in all_cols), dtype=np.float32)

    # ---- Parallel backend context ----
    parallel_ctx = (
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

    # ==========================================================================================
    # Path A: NO PCA => NO MEMMAP (fastest)
    # ==========================================================================================
    if not use_memmap:
        alphas_desc = np.sort(alphas_to_test)[::-1].tolist()
        n_alphas = len(alphas_desc)

        def _evaluate_fold_no_pca(fi: int, tr: np.ndarray, va: np.ndarray, l1_ratio: float):
            fold_auc = np.full(n_alphas, np.nan)
            start = time.time()
            error_flag = False

            X_tr = X_train_val[tr]
            X_va = X_train_val[va]
            y_tr = y_train_val[tr]
            y_va = y_train_val[va]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*`trapz` is deprecated.*")

                try:
                    model = CoxnetSurvivalAnalysis(
                        alphas=alphas_desc,
                        l1_ratio=l1_ratio,
                        max_iter=max_iter,
                        fit_baseline_model=False,
                        penalty_factor=penalty_no_pca,
                    )
                    model.fit(X_tr, y_tr)

                    risk_all = model.predict(X_va)
                    if risk_all.ndim == 1:
                        risk_all = risk_all[:, np.newaxis]

                    if risk_all.shape[1] >= n_alphas:
                        for ai in range(n_alphas):
                            try:
                                _, fold_auc[ai] = cumulative_dynamic_auc(
                                    y_tr, y_va, risk_all[:, ai], eval_times
                                )
                            except Exception:
                                pass
                    else:
                        raise ValueError("path returned fewer alphas than requested")
                except Exception:
                    # Path failed or returned partial results; fall back to individual fits
                    error_flag = True
                    for ai, a in enumerate(alphas_desc):
                        try:
                            m = CoxnetSurvivalAnalysis(
                                alphas=[a], l1_ratio=l1_ratio,
                                max_iter=max_iter, fit_baseline_model=False,
                                penalty_factor=penalty_no_pca,
                            )
                            m.fit(X_tr, y_tr)
                            _, fold_auc[ai] = cumulative_dynamic_auc(
                                y_tr, y_va, m.predict(X_va), eval_times
                            )
                        except Exception:
                            pass

            return fi, fold_auc, time.time() - start, error_flag

        def evaluate_l1_path_no_pca(l1_ratio: float):
            fold_aucs = np.full((len(folds), n_alphas), np.nan)
            fold_times = np.zeros(len(folds))
            fold_errors = np.zeros(len(folds), dtype=bool)

            if parallel_axis_eff == "fold":
                with parallel_ctx:
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

            for fi, fold_auc, fold_time, error_flag in fold_results:
                fold_aucs[fi, :] = fold_auc
                fold_times[fi] = fold_time
                fold_errors[fi] = error_flag

            rows = []
            for ai, alpha in enumerate(alphas_desc):
                rows.append([
                    float(l1_ratio),
                    float(alpha),
                    np.nan,
                    float(np.nanmean(fold_aucs[:, ai])),
                    np.nan,
                    float(np.mean(fold_times)),
                    fold_times.tolist(),
                    fold_errors.tolist(),
                    float(np.mean(fold_errors)),
                    [0] * len(folds),
                ])
            return rows

        if parallel_axis_eff == "l1":
            with parallel_ctx:
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

        cv_results_df = pd.DataFrame(
            results,
            columns=[
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
            ],
        )

        valid_cv = cv_results_df.dropna(subset=["mean_auc(t)"])
        if valid_cv.empty:
            raise RuntimeError("All CV evaluations failed (all NaN mean_auc(t)). Check data and parameters.")
        opt = valid_cv.sort_values("mean_auc(t)", ascending=False).iloc[0]
        opt_l1, opt_alpha = float(opt.l1_ratio), float(opt.alpha)

        try:
            final_model = CoxnetSurvivalAnalysis(
                alphas=[opt_alpha],
                l1_ratio=opt_l1,
                max_iter=max_iter,
                fit_baseline_model=True,
                penalty_factor=penalty_no_pca,
            )
            final_model.fit(X_train_val, y_train_val)
            mean_auc, ibs, cidx = evaluate_surv_model(final_model, X_test, y_train_val, y_test, eval_times)
        except Exception:
            final_model, mean_auc, ibs, cidx = None, np.nan, np.nan, np.nan

        test_df = pd.DataFrame({"mean_auc(t)": [mean_auc], "mean_ibs": [ibs], "mean_c_index": [cidx]})
        return test_df, cv_results_df, final_model

    # ==========================================================================================
    # Path B: PCA present => PRECOMPUTE folds + MEMMAP transformed X
    # ==========================================================================================
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

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*`trapz` is deprecated.*")

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
            start = time.time()
            error_flag = False

            X_tr_np = np.memmap(meta["tr_path"], mode="r", dtype=np.float32, shape=meta["tr_shape"])
            X_va_np = np.memmap(meta["va_path"], mode="r", dtype=np.float32, shape=meta["va_shape"])
            y_tr = meta["y_tr"]
            y_va = meta["y_va"]
            penalty = meta["penalty"]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*`trapz` is deprecated.*")

                try:
                    model = CoxnetSurvivalAnalysis(
                        alphas=alphas_desc_b,
                        l1_ratio=l1_ratio,
                        max_iter=max_iter,
                        fit_baseline_model=False,
                        penalty_factor=penalty,
                    )
                    model.fit(X_tr_np, y_tr)

                    risk_all = model.predict(X_va_np)
                    if risk_all.ndim == 1:
                        risk_all = risk_all[:, np.newaxis]

                    if risk_all.shape[1] >= n_alphas_b:
                        for ai in range(n_alphas_b):
                            try:
                                _, fold_auc[ai] = cumulative_dynamic_auc(
                                    y_tr, y_va, risk_all[:, ai], eval_times
                                )
                            except Exception:
                                pass
                    else:
                        raise ValueError("path returned fewer alphas than requested")
                except Exception:
                    # Path failed or returned partial results; fall back to individual fits
                    error_flag = True
                    for ai, a in enumerate(alphas_desc_b):
                        try:
                            m_fb = CoxnetSurvivalAnalysis(
                                alphas=[a], l1_ratio=l1_ratio,
                                max_iter=max_iter, fit_baseline_model=False,
                                penalty_factor=penalty,
                            )
                            m_fb.fit(X_tr_np, y_tr)
                            _, fold_auc[ai] = cumulative_dynamic_auc(
                                y_tr, y_va, m_fb.predict(X_va_np), eval_times
                            )
                        except Exception:
                            pass

            return fi, fold_auc, time.time() - start, error_flag

        def evaluate_l1_path_with_pca(l1_ratio: float):
            fold_aucs = np.full((len(fold_meta), n_alphas_b), np.nan)
            fold_times = np.zeros(len(fold_meta))
            fold_errors = np.zeros(len(fold_meta), dtype=bool)

            if parallel_axis_eff == "fold":
                with parallel_ctx:
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

            for fi, fold_auc, fold_time, error_flag in fold_results:
                fold_aucs[fi, :] = fold_auc
                fold_times[fi] = fold_time
                fold_errors[fi] = error_flag

            rows = []
            for ai, alpha in enumerate(alphas_desc_b):
                rows.append([
                    float(l1_ratio),
                    float(alpha),
                    np.nan,
                    float(np.nanmean(fold_aucs[:, ai])),
                    np.nan,
                    float(np.mean(fold_times)),
                    fold_times.tolist(),
                    fold_errors.tolist(),
                    float(np.mean(fold_errors)),
                    [0] * len(fold_meta),
                ])
            return rows

        if parallel_axis_eff == "l1":
            with parallel_ctx:
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

        cv_results_df = pd.DataFrame(
            results,
            columns=[
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
            ],
        )

        # ---- final fit (apply same preprocessing to train_val/test) ----
        valid_cv = cv_results_df.dropna(subset=["mean_auc(t)"])
        if valid_cv.empty:
            raise RuntimeError("All CV evaluations failed (all NaN mean_auc(t)). Check data and parameters.")
        opt = valid_cv.sort_values("mean_auc(t)", ascending=False).iloc[0]
        opt_l1, opt_alpha = float(opt.l1_ratio), float(opt.alpha)

        X_trval = np.array(X_train_val, dtype=np.float32, copy=True)
        X_te = np.array(X_test, dtype=np.float32, copy=True)
        colnames = list(colnames0)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*`trapz` is deprecated.*")

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
            mean_auc, ibs, cidx = evaluate_surv_model(final_model, X_te, y_train_val, y_test, eval_times)
        except Exception:
            final_model, mean_auc, ibs, cidx = None, np.nan, np.nan, np.nan

        test_df = pd.DataFrame({"mean_auc(t)": [mean_auc], "mean_ibs": [ibs], "mean_c_index": [cidx]})
        return test_df, cv_results_df, final_model

    finally:
        try:
            shutil.rmtree(fold_dir)
        except Exception:
            pass

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
    max_iter: int = 1000,
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
        warnings.filterwarnings("ignore")

    if pca_config is None:
        pca_config = {}
    use_memmap = len(pca_config) > 0  # <-- automatic switch

    # ---- Filter invalid ----
    df = df[df[tstop_col] > 0].copy()

    all_cols = base_cols + penalized_cols
    base_col_set = set(base_cols)

    # ---- X in RAM (float32) ----
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
            X_tr = X[train_idx]
            X_te = X[test_idx]
            y_tr = y[train_idx]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*`trapz` is deprecated.*")

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
            outs = joblib.Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=1, batch_size=1)(
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

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*`trapz` is deprecated.*")

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
                warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*`trapz` is deprecated.*")

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
            outs = joblib.Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=1, batch_size=1)(
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
