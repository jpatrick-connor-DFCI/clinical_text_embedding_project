"""Shared low-level helpers for the cox_models package.

Split out of the original monolithic cox_models.py (Phase 4 of the refactor) so the
grid-search (grid_search.py) and held-out-risk-score (heldout.py) modules — which both
need fast per-fold numpy imputation/scaling/PCA plus the sksurv structured-array and
evaluation helpers — share one implementation instead of two copies.
"""

import logging
import tempfile
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning

from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score

logger = logging.getLogger(__name__)

_SUPPRESSED_WARNINGS = [
    (ConvergenceWarning, ""),
    (RuntimeWarning, ""),
    (DeprecationWarning, ".*`trapz` is deprecated.*"),
    # joblib/loky emits this when a parallel worker is respawned; noisy and benign,
    # and it floods the Jupyter IOPub channel during long runs (IOStream.flush timeouts).
    (UserWarning, "A worker stopped while some jobs were given to the executor"),
]


def _make_surv_array(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Build a sksurv-compatible structured survival array without list(zip(...)) overhead."""
    event = np.asarray(event)
    time = np.asarray(time, dtype=np.float64)
    n_nan_event = np.isnan(event.astype(float)).sum()
    n_nan_time = np.isnan(time).sum()
    n_neg_time = (time < 0).sum()
    if n_nan_event > 0 or n_nan_time > 0 or n_neg_time > 0:
        logger.warning(
            "_make_surv_array: %d NaN events, %d NaN times, %d negative times (out of %d rows)",
            n_nan_event, n_nan_time, n_neg_time, len(event),
        )
    y = np.empty(len(event), dtype=[("Status", "?"), ("Survival_in_days", "<f8")])
    y["Status"] = event.astype(bool)
    y["Survival_in_days"] = time
    return y


def evaluate_surv_model(surv_model, X_eval, y_train, y_eval, eval_times: np.ndarray) -> tuple[float, float, float]:
    """
    Evaluate a survival model on test/validation data.

    Computes:
        - Time-dependent AUC
        - Integrated Brier Score
        - Concordance index

    Args:
        surv_model: Fitted survival model (CoxPHSurvivalAnalysis or CoxnetSurvivalAnalysis).
        X_eval: Evaluation features (pd.DataFrame or np.ndarray, as required by surv_model.predict).
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
    except Exception as e:
        logger.warning("evaluate_surv_model failed: %s", e)
        mean_auc_t, ibs, c_index = np.nan, np.nan, np.nan
    return mean_auc_t, ibs, c_index


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

def _impute_train_test_np(
    X_tr: np.ndarray,
    X_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-fold mean imputation: fill NaN using training column means only."""
    col_means = np.nanmean(X_tr, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)  # all-NaN columns → 0
    X_tr = np.where(np.isnan(X_tr), col_means, X_tr)
    X_te = np.where(np.isnan(X_te), col_means, X_te)
    return X_tr, X_te


def _best_mmap_dir(prefix="coxnet_folds_"):
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return tempfile.mkdtemp(prefix=prefix, dir="/dev/shm")
    return tempfile.mkdtemp(prefix=prefix)
