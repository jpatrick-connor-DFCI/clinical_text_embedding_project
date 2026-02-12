""".Ipynb Checkpoints utilities for the embed surv utils package."""

def run_grid_CoxPH_parallel(
    df: pd.DataFrame,
    base_cols: list[str],
    continuous_vars: list[str],
    embed_cols: list[str],
    l1_ratios: list[float],
    alphas_to_test: list[float],
    event_col: str = 'event',
    tstop_col: str = 'tstop',
    max_iter: int = 1000,
    n_splits: int = 5,
    time_evals: tuple[int, int] = (5, 95),
    n_jobs: int = -1,
    verbose: int = 0,
    ignore_warnings: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Parallelized elastic-net CoxPH grid search with Stratified CV.

    """
    if ignore_warnings:
        warnings.filterwarnings('ignore')

    # Step 1: Filter invalid patients
    df = df[df[tstop_col] > 0].copy()

    # Step 2: Prediction matrix & structured survival array
    all_cols = list(base_cols) + list(embed_cols)
    Xt_df = df[all_cols].copy()
    y_struct = np.asarray(
        list(zip(df[event_col].to_numpy(), df[tstop_col].to_numpy())),
        dtype=[('Status', '?'), ('Survival_in_days', '<f8')]
    )

    # Train/validation split indices (held-out test set = 20%), stratify on event status
    idx = np.arange(len(Xt_df))
    idx_train_val, idx_test = train_test_split(
        idx,
        test_size=0.2,
        stratify=df[event_col].astype(int),
        random_state=1234
    )

    # Materialize train/val and test arrays
    X_train_val_np = Xt_df.iloc[idx_train_val].to_numpy(copy=True)
    y_train_val_struct = y_struct[idx_train_val].copy()
    X_test_np = Xt_df.iloc[idx_test].to_numpy(copy=True)
    y_test_struct = y_struct[idx_test].copy()

    # Step 3: Persist memory-mapped files for parallel workers
    X_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    y_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    X_file, y_file = X_tmp.name, y_tmp.name
    X_tmp.close()
    y_tmp.close()
    joblib.dump(X_train_val_np, X_file)
    joblib.dump(y_train_val_struct, y_file)

    # Step 4: Stratified CV and evaluation times
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)

    # === FIX: build strat_labels using positional indexing so labels align with X_train_val_np rows ===
    strat_labels = df[event_col].astype(int).to_numpy()[idx_train_val]
    # (alternatively: strat_labels = df.iloc[idx_train_val][event_col].astype(int).to_numpy())

    lower, upper = np.percentile(
        y_struct['Survival_in_days'], [time_evals[0], time_evals[1]]
    )
    eval_times = (
        np.linspace(lower, upper, num=50, dtype=float)
        if lower != upper else np.array([lower])
    )

    # Column indices for scaling
    col_to_idx = {c: i for i, c in enumerate(all_cols)}
    continuous_idx = np.array([col_to_idx[c] for c in continuous_vars], dtype=int)

    # Penalize only embeddings
    penalty_factor = [0] * len(base_cols) + [1] * len(embed_cols)

    # ------------------------------------------------------------------
    # Worker function for parallel evaluation
    # ------------------------------------------------------------------
    def evaluate_param(l1_ratio, alpha):
        """
        Evaluate one (l1_ratio, alpha) combination using Stratified CV.
        Returns averaged C-index, mean AUC(t), and IBS across folds.
        """
        X_shared = joblib.load(X_file, mmap_mode="r")
        y_shared = joblib.load(y_file, mmap_mode="r")

        c_index_vals, mean_auc_t_vals, ibs_vals = [], [], []

        # Use X_shared and strat_labels (both have the same row order/length)
        for tr, va in cv.split(X_shared, strat_labels):
            X_tr = pd.DataFrame(X_shared[tr].copy(), columns=all_cols)
            X_va = pd.DataFrame(X_shared[va].copy(), columns=all_cols)
            y_tr = y_shared[tr].copy()
            y_va = y_shared[va].copy()

            # Standardize continuous columns
            scaler = StandardScaler().fit(X_tr.iloc[:, continuous_idx])
            X_tr.iloc[:, continuous_idx] = scaler.transform(X_tr.iloc[:, continuous_idx])
            X_va.iloc[:, continuous_idx] = scaler.transform(X_va.iloc[:, continuous_idx])

            # Fit Coxnet for this fold
            try:
                cox_model = CoxnetSurvivalAnalysis(
                    alphas=[alpha],
                    l1_ratio=l1_ratio,
                    fit_baseline_model=True,
                    max_iter=max_iter,
                    penalty_factor=penalty_factor
                )
                cox_model.fit(X_tr, y_tr)
                mean_auc_t, ibs, c_index = evaluate_surv_model(
                    cox_model, X_va, y_tr, y_va, eval_times
                )
            except Exception:
                mean_auc_t, ibs, c_index = np.nan, np.nan, np.nan

            mean_auc_t_vals.append(mean_auc_t)
            ibs_vals.append(ibs)
            c_index_vals.append(c_index)

        # Average across folds
        mean_c_index = np.nanmean(c_index_vals)
        mean_auc_t = np.nanmean(mean_auc_t_vals)
        mean_ibs = np.nanmean(ibs_vals)

        return [float(l1_ratio), float(alpha), mean_c_index, mean_auc_t, mean_ibs]

    # Step 5: Run all hyperparameter combinations in parallel
    with joblib.parallel_backend("loky", inner_max_num_threads=1):
        results_list = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
            joblib.delayed(evaluate_param)(l1, a)
            for (l1, a) in product(l1_ratios, alphas_to_test)
        )

    cv_results_df = pd.DataFrame(
        results_list,
        columns=['l1_ratio', 'alpha', 'mean_c_index', 'mean_auc(t)', 'mean_ibs']
    )

    # Step 6: Pick best hyperparameters and evaluate on test set
    opt_row = cv_results_df.sort_values('mean_auc(t)', ascending=False).iloc[0]
    opt_l1, opt_alpha = float(opt_row['l1_ratio']), float(opt_row['alpha'])

    X_trval_df = pd.DataFrame(X_train_val_np, columns=all_cols)
    X_test_df = pd.DataFrame(X_test_np, columns=all_cols)

    scaler_final = StandardScaler().fit(X_trval_df.iloc[:, continuous_idx])
    X_trval_df.iloc[:, continuous_idx] = scaler_final.transform(X_trval_df.iloc[:, continuous_idx])
    X_test_df.iloc[:, continuous_idx] = scaler_final.transform(X_test_df.iloc[:, continuous_idx])

    try:
        final_cox = CoxnetSurvivalAnalysis(
            alphas=[opt_alpha],
            l1_ratio=opt_l1,
            fit_baseline_model=True,
            max_iter=max_iter,
            penalty_factor=penalty_factor
        )
        final_cox.fit(X_trval_df, y_train_val_struct)
        mean_auc_t, ibs, c_index = evaluate_surv_model(
            final_cox, X_test_df, y_train_val_struct, y_test_struct, eval_times
        )
    except Exception:
        mean_auc_t, ibs, c_index = np.nan, np.nan, np.nan

    test_results_df = pd.DataFrame({
        'mean_auc(t)': [mean_auc_t],
        'mean_ibs': [ibs],
        'mean_c_index': [c_index]
    })

    # Step 7: Clean up temp files
    for path in (X_file, y_file):
        try:
            os.remove(path)
        except OSError:
            pass

    return test_results_df, cv_results_df, final_cox

def get_heldout_risk_scores_CoxPH(df: pd.DataFrame, base_cols: list[str], continuous_vars: list[str], embed_cols: list[str], 
                                  event_col: str = 'event', tstop_col: str = 'tstop', id_col: str = 'DFCI_MRN', n_splits: int = 5, 
                                  n_jobs: int = -1, max_iter: int = 1000, penalized: bool = False, l1_ratio: float = 0.5, alpha: float = 1.0, 
                                  verbose: int = 0, ignore_warnings: bool = True) -> pd.DataFrame:
    """
    Compute held-out risk scores using CoxPH or penalized Coxnet models with stratified cross-validation.

    Steps:
        1. Filter patients with positive survival time.
        2. Construct feature matrix and structured survival array.
        3. Standardize continuous variables per fold.
        4. Fit CoxPH or Coxnet on train folds, predict risk on test fold.
        5. Parallelize across folds for speed.
        6. Collect all predictions and return a DataFrame with risk scores.

    Args:
        df (pd.DataFrame): Input dataset containing features and survival outcome.
        base_cols (list[str]): Base features (demographics, labs, etc.).
        continuous_vars (list[str]): Continuous columns to standardize.
        embed_cols (list[str]): Embedding columns.
        event_col (str): Column indicating event occurrence.
        tstop_col (str): Column indicating survival time.
        id_col (str): Patient identifier column to include in output.
        n_splits (int): Number of CV splits.
        n_jobs (int): Number of parallel workers.
        max_iter (int): Maximum iterations for solver.
        penalized (bool): Use penalized Coxnet (True) or standard CoxPH (False).
        l1_ratio (float): Elastic net L1 ratio (if penalized=True).
        alpha (float): Regularization strength (if penalized=True).
        verbose (int): Verbosity for parallel execution.
        ignore_warnings (bool): Suppress warnings if True.

    Returns:
        pd.DataFrame: DataFrame with columns [id_col, risk_score] containing predictions for all patients.
    """
    # Optionally ignore warnings globally
    if ignore_warnings:
        warnings.filterwarnings('ignore')

    # Step 1: Filter out patients with non-positive survival times
    df = df[df[tstop_col] > 0].copy()
    all_cols = base_cols + embed_cols
    X = df[all_cols].copy()

    # Step 2: Structured survival array required by scikit-survival
    y = np.array(list(zip(df[event_col].astype(bool), df[tstop_col])),
                 dtype=[('event', bool), ('time', float)])

    # Step 3: Stratified cross-validation (balance by event status)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)

    # Identify indices of continuous variables for scaling
    continuous_idx = [X.columns.get_loc(c) for c in continuous_vars]

    # Prepare array to store risk scores for all patients
    results = np.full(len(df), np.nan)

    # Step 4: Define function to fit & predict on a single CV fold
    def fit_predict_fold(train_idx, test_idx):
        """
        Fit Cox model on training fold, predict risk scores on test fold.
        Standardizes continuous variables per fold to avoid data leakage.
        """
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize continuous variables
        scaler = StandardScaler().fit(X_train.iloc[:, continuous_idx])
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        # Ensure columns are floats before scaling
        for col_idx in continuous_idx:
            col_name = X_train_scaled.columns[col_idx]
            X_train_scaled[col_name] = X_train_scaled[col_name].astype(float)
            X_test_scaled[col_name] = X_test_scaled[col_name].astype(float)

        # Apply scaling
        X_train_scaled.iloc[:, continuous_idx] = scaler.transform(X_train.iloc[:, continuous_idx])
        X_test_scaled.iloc[:, continuous_idx] = scaler.transform(X_test.iloc[:, continuous_idx])

        # Fit Cox model
        try:
            if penalized:
                model = CoxnetSurvivalAnalysis(
                    alphas=[alpha],
                    l1_ratio=l1_ratio,
                    fit_baseline_model=True,
                    max_iter=max_iter,
                    penalty_factor=[0] * len(base_cols) + [1] * len(embed_cols)
                )
            else:
                model = CoxPHSurvivalAnalysis(n_iter=max_iter)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_scaled, y_train)

            # Predict risk scores on test fold
            risk_scores = model.predict(X_test_scaled)

        except Exception as e:
            risk_scores = np.full(len(test_idx), np.nan)
            print(f"Warning: model failed on fold with error: {e}")

        return test_idx, risk_scores

    # Step 5: Run stratified CV folds in parallel
    with joblib.parallel_backend("loky", inner_max_num_threads=1):
        fold_preds = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
            joblib.delayed(fit_predict_fold)(train_idx, test_idx)
            for train_idx, test_idx in cv.split(X, df[event_col])
        )

    # Step 6: Collect fold predictions into results array
    for test_idx, preds in fold_preds:
        results[test_idx] = preds

    # Step 7: Build final output DataFrame
    if id_col and id_col in df.columns:
        output_df = pd.DataFrame({id_col: df[id_col].values, 'risk_score': results})
    else:
        output_df = pd.DataFrame({'index': df.index, 'risk_score': results})

    return output_df

def run_grid_CoxPH_w_seq_psa_parallel(df: pd.DataFrame, base_cols: list[str], psa_cols: list[str], continuous_vars: list[str], 
                                      embed_cols: list[str], l1_ratios: list[float], alphas_to_test: list[float], event_col: str = 'event', 
                                      tstop_col: str = 'tstop', max_iter: int = 1000, n_splits: int = 5, time_evals: tuple[int, int] = (25, 75), 
                                      n_jobs: int = -1, verbose: int = 0, ignore_warnings: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, CoxnetSurvivalAnalysis]:
    """
    Run a grid search for penalized Cox models (Coxnet) on base + embedding + sequential PSA features
    using K-Fold cross-validation, including PSA-specific standardization.

    Steps:
        1. Prepare survival outcome and feature matrix.
        2. Split dataset into train+validation and held-out test sets.
        3. For each hyperparameter combo (l1_ratio, alpha):
            - Scale continuous variables per fold.
            - Standardize PSA columns across all folds.
            - Fit Coxnet on train fold, predict on validation fold.
            - Compute performance metrics: C-index, mean AUC over eval times, Integrated Brier Score.
        4. Run hyperparameter grid in parallel.
        5. Select best hyperparameters based on mean AUC.
        6. Refit final Coxnet model on full train+validation set, evaluate on held-out test set.

    Args:
        df (pd.DataFrame): Input dataset containing features and survival outcome.
        base_cols (list[str]): Base features (demographics, labs, etc.).
        psa_cols (list[str]): Sequential PSA columns to standardize separately.
        continuous_vars (list[str]): Continuous features to standardize.
        embed_cols (list[str]): Embedding features to include in model.
        l1_ratios (list[float]): L1 ratios for elastic net regularization.
        alphas_to_test (list[float]): Alpha values (regularization strengths) to test.
        event_col (str): Column name for event indicator.
        tstop_col (str): Column name for survival time.
        max_iter (int): Maximum iterations for Coxnet solver.
        n_splits (int): Number of K-Fold CV splits.
        time_evals (tuple[int, int]): Percentiles to define evaluation time range.
        n_jobs (int): Number of parallel workers.
        verbose (int): Verbosity for parallel execution.
        ignore_warnings (bool): Suppress warnings if True.

    Returns:
        tuple:
            - test_results_df (pd.DataFrame): Performance on held-out test set.
            - cv_results_df (pd.DataFrame): CV results for all hyperparameter combinations.
            - final_cox (CoxnetSurvivalAnalysis): Fitted final Coxnet model on train+val.
    """
    if ignore_warnings:
        warnings.filterwarnings('ignore')
        
    # Step 1: Prepare feature matrix and survival outcome
    Xt = df[base_cols + embed_cols]
    y = np.asarray(list(zip(df[event_col], df[tstop_col])),
                   dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

    # Step 2: Split into training+validation and test sets
    X_train_plus_val, X_test, y_train_plus_val, y_test = train_test_split(Xt, y, test_size=0.2, random_state=1234)

    # Step 3: Set up K-Fold cross-validation
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=1234)

    # Define evaluation times (from percentiles of survival distribution)
    lower, upper = np.percentile(y['Survival_in_days'], [time_evals[0], time_evals[1]])
    eval_times = np.arange(lower, upper + 1)

    # Define penalty factors: 0 for base features, 1 for embedding/PSA
    penalty_factor = [0] * len(base_cols) + [1] * len(embed_cols)

    # Step 4: Define function to evaluate a single hyperparameter combo
    def evaluate_param(l1_ratio, alpha_to_test):
        """
        Evaluate a given (l1_ratio, alpha) using CV on train+val set.
        Performs:
            - Scaling of continuous variables
            - Standardization of PSA columns
            - Coxnet fitting
            - Performance evaluation (C-index, mean AUC, IBS)
        """
        c_index_vals, mean_auc_t_vals, ibs_vals = [], [], []

        for train_idx, val_idx in cv.split(X_train_plus_val, y_train_plus_val):
            # Extract train/val fold
            X_train, y_train = Xt.iloc[train_idx].copy(), y[train_idx]
            X_val, y_val = Xt.iloc[val_idx].copy(), y[val_idx]

            # Standardize continuous variables per fold
            scaler = StandardScaler().fit(X_train[continuous_vars])
            X_train[continuous_vars] = scaler.transform(X_train[continuous_vars])
            X_val[continuous_vars] = scaler.transform(X_val[continuous_vars])

            # Standardize PSA features across all PSA columns
            psa_mu = np.mean([X_train[psa_col].values for psa_col in psa_cols])
            psa_std = np.std([X_train[psa_col].values for psa_col in psa_cols])
            for psa_col in psa_cols:
                X_train[psa_col] = (X_train[psa_col] - psa_mu) / psa_std
                X_val[psa_col] = (X_val[psa_col] - psa_mu) / psa_std

            # Fit Coxnet and evaluate metrics
            try:
                cox_model = CoxnetSurvivalAnalysis(
                    alphas=[alpha_to_test], l1_ratio=l1_ratio,
                    fit_baseline_model=True, max_iter=max_iter,
                    penalty_factor=penalty_factor).fit(X_train, y_train)

                mean_auc_t, ibs, c_index = evaluate_surv_model(cox_model, X_val, y_train, y_val, eval_times)
            except:
                mean_auc_t = ibs = c_index = np.nan

            c_index_vals.append(c_index)
            mean_auc_t_vals.append(mean_auc_t)
            ibs_vals.append(ibs)

        # Return mean performance across folds
        return [l1_ratio, alpha_to_test, np.nanmean(c_index_vals), np.nanmean(mean_auc_t_vals), np.nanmean(ibs_vals)]

    # Step 5: Run grid search in parallel
    with joblib.parallel_backend("loky", inner_max_num_threads=1):
        results_list = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
            joblib.delayed(evaluate_param)(l1, a) for l1, a in product(l1_ratios, alphas_to_test))

    cv_results_df = pd.DataFrame(results_list, columns=['l1_ratio', 'alpha', 'mean_c_index', 'mean_auc(t)', 'mean_ibs'])
    
    # Step 6: Select best hyperparameters based on mean AUC
    opt_l1, opt_alpha = cv_results_df.sort_values(by='mean_auc(t)', ascending=False).iloc[0][['l1_ratio', 'alpha']]

    # Step 7: Refit final model on train+val and evaluate on held-out test set
    X_train = X_train_plus_val.copy()
    X_test_final = X_test.copy()

    # Standardize continuous vars on full train+val
    scaler = StandardScaler().fit(X_train[continuous_vars])
    X_train[continuous_vars] = scaler.transform(X_train[continuous_vars])
    X_test_final[continuous_vars] = scaler.transform(X_test_final[continuous_vars])

    # Standardize PSA features
    psa_mu = np.mean([X_train[psa_col].values for psa_col in psa_cols])
    psa_std = np.std([X_train[psa_col].values for psa_col in psa_cols])
    for psa_col in psa_cols:
        X_train[psa_col] = (X_train[psa_col] - psa_mu) / psa_std
        X_test_final[psa_col] = (X_test_final[psa_col] - psa_mu) / psa_std

    # Fit final Coxnet model
    final_cox = CoxnetSurvivalAnalysis(
        alphas=[opt_alpha], l1_ratio=opt_l1,
        fit_baseline_model=True, max_iter=max_iter,
        penalty_factor=penalty_factor).fit(X_train, y_train_plus_val)

    # Evaluate on test set
    mean_auc_t, ibs, c_index = evaluate_surv_model(final_cox, X_test_final, y_train_plus_val, y_test, eval_times)
    test_results_df = pd.DataFrame({'mean_auc(t)': [mean_auc_t],
                                    'mean_ibs': [ibs], 
                                    'mean_c_index': [c_index]})

    return test_results_df, cv_results_df, final_cox
    