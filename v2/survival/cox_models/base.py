"""Baseline (unpenalized) CoxPH model: scaling helper + CV/held-out-test fit."""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sksurv.linear_model import CoxPHSurvivalAnalysis

from ._common import _SUPPRESSED_WARNINGS, _make_surv_array, evaluate_surv_model

logger = logging.getLogger(__name__)


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


def run_base_CoxPH(df: pd.DataFrame, base_cols: list[str], continuous_vars: list[str],
                   event_col: str = 'event', tstop_col: str = 'tstop',
                   max_iter: int = 10000, n_splits: int = 5,
                   time_evals: tuple[int, int] = (5, 95),
                   test_size: float = 0.2,
                   ignore_warnings: bool = True) -> pd.DataFrame:
    """
    Fit a baseline CoxPH model using base covariates.
    - 20% of data is held out as a test set.
    - Cross-validation is performed on the remaining data.
    """
    if ignore_warnings:
        for _cat, _msg in _SUPPRESSED_WARNINGS:
            warnings.filterwarnings("ignore", category=_cat, message=_msg)

    # Drop rows with NaN/non-positive time-to-event or NaN event indicators
    n_before = len(df)
    df = df[df[tstop_col].notna() & (df[tstop_col] > 0) & df[event_col].notna()].copy()
    # Drop rows with NaN in any feature column
    df = df.dropna(subset=base_cols + continuous_vars)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info("run_base_CoxPH: dropped %d/%d rows with invalid tstop/event/features", n_dropped, n_before)

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
        except Exception as e:
            logger.warning("run_base_CoxPH CV fold failed: %s", e)
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
    except Exception as e:
        logger.warning("run_base_CoxPH final model failed: %s", e)
        mean_auc_t_test, ibs_test, c_index_test = np.nan, np.nan, np.nan

    # Return summary
    return pd.DataFrame([
        ['cv_data', np.nanmean(c_index_vals), np.nanmean(mean_auc_t_vals), np.nanmean(ibs_vals)],
        ['test_data', c_index_test, mean_auc_t_test, ibs_test]
    ], columns=['eval_data', 'mean_c_index', 'mean_auc(t)', 'mean_ibs'])
