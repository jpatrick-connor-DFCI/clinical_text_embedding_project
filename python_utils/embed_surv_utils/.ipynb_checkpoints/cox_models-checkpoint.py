import os
import warnings
import tempfile
import joblib
import numpy as np
import pandas as pd
from itertools import product

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score

warnings.filterwarnings('ignore')

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
    scaler = StandardScaler().fit(X_train[continuous_vars])
    X_train[continuous_vars] = scaler.transform(X_train[continuous_vars])
    X_test[continuous_vars] = scaler.transform(X_test[continuous_vars])
    return X_train, X_test

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
        auc_t, mean_auc_t = cumulative_dynamic_auc(y_train, y_eval, risk_scores, eval_times)
        ibs = integrated_brier_score(y_train, y_eval, risk_scores, eval_times)
        c_index = surv_model.score(X_eval, y_eval)
    except:
        mean_auc_t, ibs, c_index = np.nan, np.nan, np.nan
    return mean_auc_t, ibs, c_index

def apply_group_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    group_cols: list[str],
    group_name: str,
    k: int,
    random_state: int = 1234
):
    if k is None or k <= 0:
        return X_train, X_test, []

    group_cols_present = [c for c in group_cols if c in X_train.columns]
    if len(group_cols_present) == 0:
        return X_train, X_test, []

    # ---- Scale before PCA ----
    scaler = StandardScaler().fit(X_train[group_cols_present])
    X_train_scaled = scaler.transform(X_train[group_cols_present])
    X_test_scaled  = scaler.transform(X_test[group_cols_present])

    n_train = X_train_scaled.shape[0]
    k_eff = min(k, len(group_cols_present), n_train)
    if k_eff <= 0:
        return X_train, X_test, []

    # ---- PCA ----
    pca = PCA(n_components=k_eff, random_state=random_state)
    train_pcs = pca.fit_transform(X_train_scaled)
    test_pcs  = pca.transform(X_test_scaled)

    pc_names = [f"{group_name}_PC{i+1}" for i in range(k_eff)]

    # ---- Build PC DataFrames once (fixes fragmentation) ----
    pc_train_df = pd.DataFrame(train_pcs, columns=pc_names, index=X_train.index)
    pc_test_df  = pd.DataFrame(test_pcs,  columns=pc_names, index=X_test.index)

    # ---- Drop original group cols, concat in one shot ----
    X_train_new = pd.concat(
        [X_train.drop(columns=group_cols_present), pc_train_df],
        axis=1
    ).copy()

    X_test_new = pd.concat(
        [X_test.drop(columns=group_cols_present), pc_test_df],
        axis=1
    ).copy()

    return X_train_new, X_test_new, pc_names
    
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

    # Split into train+val and held-out test set
    strat_labels = df[event_col].astype(int)
    df_trainval, df_test = train_test_split(df, test_size=test_size, 
                                            stratify=strat_labels, random_state=1234)

    Xt_trainval = df_trainval[base_cols]
    y_trainval = np.asarray(list(zip(df_trainval[event_col], df_trainval[tstop_col])),
                            dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

    Xt_test = df_test[base_cols]
    y_test = np.asarray(list(zip(df_test[event_col], df_test[tstop_col])),
                        dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

    # Compute evaluation times
    lower, upper = np.percentile(df_trainval[tstop_col], [time_evals[0], time_evals[1]])
    eval_times = np.arange(lower, upper + 1)

    # --- Cross-validation on train+val ---
    c_index_vals, mean_auc_t_vals, ibs_vals = [], [], []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)

    for train_idx, val_idx in cv.split(Xt_trainval, df_trainval[event_col]):
        X_train, X_val = Xt_trainval.iloc[train_idx], Xt_trainval.iloc[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        X_train, X_val = scale_model_data(X_train, X_val, continuous_vars)

        cox_model = CoxPHSurvivalAnalysis()
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
    cox_model_final = CoxPHSurvivalAnalysis()
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

def run_grid_CoxPH_parallel(
    df: pd.DataFrame,
    base_cols: list[str],
    continuous_vars: list[str],
    penalized_cols: list[str],
    l1_ratios: list[float],
    alphas_to_test: list[float],
    pca_config: dict[str, tuple[list[str], int]] | None = None,
    event_col: str = 'event',
    tstop_col: str = 'tstop',
    max_iter: int = 1000,
    n_splits: int = 5,
    time_evals: tuple[int, int] = (5, 95),
    n_jobs: int = -1,
    verbose: int = 0,
    ignore_warnings: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, object]:

    if ignore_warnings:
        warnings.filterwarnings('ignore')

    if pca_config is None:
        pca_config = {}

    # ---- Filter invalid patients ----
    df = df[df[tstop_col] > 0].copy()

    all_cols = base_cols + penalized_cols
    X_df = df[all_cols].copy()

    # ---- Structured survival array ----
    y_struct = np.asarray(
        list(zip(df[event_col].astype(bool).to_numpy(),
                 df[tstop_col].to_numpy())),
        dtype=[('Status', '?'), ('Survival_in_days', '<f8')]
    )

    # ---- Train/val vs test split ----
    idx = np.arange(len(X_df))
    idx_train_val, idx_test = train_test_split(
        idx,
        test_size=0.2,
        stratify=df[event_col].astype(int),
        random_state=1234
    )

    X_train_val_np = X_df.iloc[idx_train_val].to_numpy()
    X_test_np = X_df.iloc[idx_test].to_numpy()
    y_train_val_struct = y_struct[idx_train_val]
    y_test_struct = y_struct[idx_test]

    # ---- Persist training data so parallel workers can mmap ----
    X_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    y_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    X_file, y_file = X_tmp.name, y_tmp.name
    X_tmp.close()
    y_tmp.close()
    joblib.dump(X_train_val_np, X_file)
    joblib.dump(y_train_val_struct, y_file)

    # ---- CV ----
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)
    strat_labels = df[event_col].astype(int).iloc[idx_train_val].to_numpy()

    # ---- Evaluation time points ----
    lower, upper = np.percentile(
        y_struct['Survival_in_days'], [time_evals[0], time_evals[1]]
    )
    eval_times = (
        np.linspace(lower, upper, 50) if lower != upper else np.array([lower])
    )

    # ==========================================================================================
    # Worker: evaluate one l1_ratio/alpha combination WITH TIMING + ERROR FLAGS
    # ==========================================================================================
    def evaluate_param(l1_ratio, alpha):
    
        X_shared = joblib.load(X_file, mmap_mode='r')
        y_shared = joblib.load(y_file, mmap_mode='r')
    
        auc_list, ibs_list, c_list = [], [], []
        fold_times = []
        fold_error_flags = []
    
        for tr, va in cv.split(X_shared, strat_labels):
    
            start = time.time()
            error_flag = False
    
            X_tr = pd.DataFrame(X_shared[tr], columns=all_cols)
            X_va = pd.DataFrame(X_shared[va], columns=all_cols)
            y_tr = y_shared[tr]
            y_va = y_shared[va]
    
            # =====================================================================
            # Catch BOTH warnings + exceptions inside fold
            # =====================================================================
            with warnings.catch_warnings():
                # Treat convergence/numerical warnings as errors
                warnings.filterwarnings("error", category=RuntimeWarning)
                warnings.filterwarnings("error", category=UserWarning)
    
                # Ignore benign pandas warnings
                warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

                try:
                    # ---- Apply PCA ----
                    for group_name, (cols, k) in pca_config.items():
                        X_tr, X_va, _ = apply_group_pca(
                            X_tr, X_va,
                            group_cols=cols,
                            group_name=group_name,
                            k=k,
                            random_state=1234
                        )
    
                    # ---- Scale continuous vars ----
                    cont_cols = [c for c in continuous_vars if c in X_tr.columns]
                    if cont_cols:
                        scaler = StandardScaler().fit(X_tr[cont_cols])
                        X_tr[cont_cols] = scaler.transform(X_tr[cont_cols])
                        X_va[cont_cols] = scaler.transform(X_va[cont_cols])
    
                    # ---- Penalty ----
                    penalty = [0 if c in base_cols else 1 for c in X_tr.columns]
    
                    # ---- Fit Coxnet ----
                    model = CoxnetSurvivalAnalysis(
                        alphas=[alpha],
                        l1_ratio=l1_ratio,
                        max_iter=max_iter,
                        fit_baseline_model=True,
                        penalty_factor=penalty
                    )
                    model.fit(X_tr, y_tr)
    
                    # ---- Evaluate ----
                    mean_auc, ibs, cidx = evaluate_surv_model(
                        model, X_va, y_tr, y_va, eval_times
                    )
    
                except Exception as e:
                    # ANY exception OR warning will end up here
                    error_flag = True
                    mean_auc, ibs, cidx = np.nan, np.nan, np.nan
    
            # store metrics
            auc_list.append(mean_auc)
            ibs_list.append(ibs)
            c_list.append(cidx)
            fold_times.append(time.time() - start)
            fold_error_flags.append(error_flag)
    
        # Summaries per hyperparameter combo
        avg_time = float(np.nanmean(fold_times))
        mean_error_rate = np.mean(fold_error_flags)
    
        return [
            float(l1_ratio),
            float(alpha),
            np.nanmean(c_list),
            np.nanmean(auc_list),
            np.nanmean(ibs_list),
            avg_time,
            fold_times,
            fold_error_flags,
            mean_error_rate
        ]

    # ---- Grid search ----
    with joblib.parallel_backend("loky", inner_max_num_threads=1):
        results = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
            joblib.delayed(evaluate_param)(l1, a)
            for l1, a in product(l1_ratios, alphas_to_test)
        )

    cv_results_df = pd.DataFrame(
        results,
        columns=[
            'l1_ratio',
            'alpha',
            'mean_c_index',
            'mean_auc(t)',
            'mean_ibs',
            'mean_train_time',
            'fold_times',
            'fold_error_flags',
            'error_rate'
        ]
    )

    # ==========================================================================================
    #            SELECT BEST PARAMETERS + FINAL TRAIN→TEST EVALUATION (unchanged)
    # ==========================================================================================

    opt = cv_results_df.sort_values("mean_auc(t)", ascending=False).iloc[0]
    opt_l1, opt_alpha = float(opt.l1_ratio), float(opt.alpha)

    X_trval = pd.DataFrame(X_train_val_np, columns=all_cols)
    X_te = pd.DataFrame(X_test_np, columns=all_cols)

    for group_name, (cols, k) in pca_config.items():
        X_trval, X_te, _ = apply_group_pca(
            X_trval, X_te,
            group_cols=cols,
            group_name=group_name,
            k=k,
            random_state=1234
        )

    cont_cols = [c for c in continuous_vars if c in X_trval.columns]
    if cont_cols:
        scaler_final = StandardScaler().fit(X_trval[cont_cols])
        X_trval[cont_cols] = scaler_final.transform(X_trval[cont_cols])
        X_te[cont_cols] = scaler_final.transform(X_te[cont_cols])

    penalty_final = [0 if c in base_cols else 1 for c in X_trval.columns]

    try:
        final_model = CoxnetSurvivalAnalysis(
            alphas=[opt_alpha],
            l1_ratio=opt_l1,
            max_iter=max_iter,
            fit_baseline_model=True,
            penalty_factor=penalty_final
        )
        final_model.fit(X_trval, y_train_val_struct)

        mean_auc, ibs, cidx = evaluate_surv_model(
            final_model, X_te, y_train_val_struct, y_test_struct, eval_times
        )
    except:
        final_model, mean_auc, ibs, cidx = None, np.nan, np.nan, np.nan

    test_df = pd.DataFrame({
        "mean_auc(t)": [mean_auc],
        "mean_ibs": [ibs],
        "mean_c_index": [cidx]
    })

    # ---- Clean up ----
    for f in (X_file, y_file):
        try: os.remove(f)
        except: pass

    return test_df, cv_results_df, final_model


def get_heldout_risk_scores_CoxPH(
    df: pd.DataFrame,
    base_cols: list[str],
    continuous_vars: list[str],
    penalized_cols: list[str],
    pca_config: dict[str, tuple[list[str], int]] | None = None,
    event_col: str = 'event',
    tstop_col: str = 'tstop',
    id_col: str = 'DFCI_MRN',
    n_splits: int = 5,
    n_jobs: int = -1,
    max_iter: int = 1000,
    penalized: bool = False,
    l1_ratio: float = 0.5,
    alpha: float = 1.0,
    verbose: int = 0,
    ignore_warnings: bool = True
) -> pd.DataFrame:

    if ignore_warnings:
        warnings.filterwarnings('ignore')

    if pca_config is None:
        pca_config = {}

    df = df[df[tstop_col] > 0].copy()
    all_cols = base_cols + penalized_cols
    X = df[all_cols].copy()

    y = np.array(
        list(zip(df[event_col].astype(bool), df[tstop_col])),
        dtype=[('event', bool), ('time', float)]
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)
    out_risk = np.full(len(df), np.nan)

    def fit_predict_fold(train_idx, test_idx):
        X_tr = X.iloc[train_idx].copy()
        X_te = X.iloc[test_idx].copy()
        y_tr = y[train_idx]
        y_te = y[test_idx]

        # --- group-based PCA ---
        for group_name, (cols, k) in pca_config.items():
            X_tr, X_te, _ = apply_group_pca(
                X_tr, X_te,
                group_cols=cols,
                group_name=group_name,
                k=k,
                random_state=1234
            )

        # --- scale continuous vars ---
        cont_cols = [c for c in continuous_vars if c in X_tr.columns]
        if cont_cols:
            scaler = StandardScaler().fit(X_tr[cont_cols])
            X_tr[cont_cols] = scaler.transform(X_tr[cont_cols])
            X_te[cont_cols] = scaler.transform(X_te[cont_cols])

        # --- choose model ---
        if penalized:
            penalty = [0 if c in base_cols else 1 for c in X_tr.columns]
            model = CoxnetSurvivalAnalysis(
                alphas=[alpha], l1_ratio=l1_ratio,
                max_iter=max_iter, fit_baseline_model=True,
                penalty_factor=penalty
            )
        else:
            model = CoxPHSurvivalAnalysis(n_iter=max_iter)

        try:
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
        except Exception as e:
            print(f"Fold failure: {e}")
            preds = np.full(len(test_idx), np.nan)

        return test_idx, preds

    with joblib.parallel_backend("loky", inner_max_num_threads=1):
        fold_outputs = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(fit_predict_fold)(tr, te)
            for tr, te in cv.split(X, df[event_col])
        )

    for test_idx, preds in fold_outputs:
        out_risk[test_idx] = preds

    if id_col in df.columns:
        return pd.DataFrame({id_col: df[id_col], "risk_score": out_risk})
    else:
        return pd.DataFrame({"index": df.index, "risk_score": out_risk})