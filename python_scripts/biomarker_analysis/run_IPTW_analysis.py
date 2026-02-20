"""Run Iptw Analysis script for biomarker analysis workflows."""

import os
import logging
import random
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import LinAlgWarning
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from biomarker_common import get_mutation_type
random.seed(42)  # set seed for reproducibility

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

N_JOBS = int(os.getenv("SLURM_CPUS_PER_TASK", "-1"))

# ---------------------------------------
# Summary classifier
# ---------------------------------------
def classify(row):
    # Predictive biomarker: interaction significant AND direction
    if row['significant_predictive']:
        if row['HR_markerxICI'] < 1:
            return "predictive_ICI_benefit"
        else:
            return "predictive_ICI_harm"

    # Not predictive, but has ICI-specific effect
    if row['significant_in_ICI'] and not row['significant_prognostic_nonICI']:
        return "ICI_specific_effect"

    # Prognostic only (acts in controls but not ICI)
    if row['significant_prognostic_nonICI'] and not row['significant_in_ICI']:
        return "prognostic_nonICI"

    # no clear signal
    return "no_signal"


def merge_rare_cancer_types_into_other(
    df: pd.DataFrame,
    min_total: int = 30,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Collapse rare CANCER_TYPE_* dummies into CANCER_TYPE_OTHER for pan-cancer fitting."""
    out = df.copy()
    cancer_cols = [col for col in out.columns if col.startswith('CANCER_TYPE_')]
    if not cancer_cols:
        return out, [], []

    cancer_matrix = out[cancer_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    cancer_matrix = (cancer_matrix > 0).astype(int)

    total_counts = cancer_matrix.sum(axis=0)

    keep_cols = [
        col for col in cancer_cols
        if (total_counts[col] >= min_total)
        and (col != 'CANCER_TYPE_OTHER')
    ]
    rare_cols = [col for col in cancer_cols if col not in keep_cols and col != 'CANCER_TYPE_OTHER']

    existing_other = (
        cancer_matrix['CANCER_TYPE_OTHER'].copy()
        if 'CANCER_TYPE_OTHER' in cancer_matrix.columns
        else pd.Series(0, index=out.index, dtype=int)
    )
    merged_other = existing_other.copy()
    if rare_cols:
        merged_other = ((merged_other + cancer_matrix[rare_cols].sum(axis=1)) > 0).astype(int)

    out = out.drop(columns=cancer_cols)
    for col in keep_cols:
        out[col] = cancer_matrix[col].astype(int)
    out['CANCER_TYPE_OTHER'] = merged_other.astype(int)

    kept_for_model = [col for col in (keep_cols + ['CANCER_TYPE_OTHER']) if out[col].sum() > 0]
    return out, kept_for_model, rare_cols

def marker_has_within_arm_support(
    df: pd.DataFrame,
    marker: str,
    treat_col: str = 'PX_on_ICI',
    min_pos_per_arm: int = 5,
    min_neg_per_arm: int = 5,
) -> bool:
    marker_bin = (pd.to_numeric(df[marker], errors='coerce').fillna(0) > 0).astype(int)
    treatment = pd.to_numeric(df[treat_col], errors='coerce').fillna(0).astype(int)

    for arm in (0, 1):
        arm_mask = treatment == arm
        if arm_mask.sum() == 0:
            return False
        arm_pos = int(marker_bin.loc[arm_mask].sum())
        arm_neg = int(arm_mask.sum() - arm_pos)
        if arm_pos < min_pos_per_arm or arm_neg < min_neg_per_arm:
            return False
    return True


def compute_smd(df, covariates, treat_col='PX_on_ICI', weights=None):
    """Compute standardized mean differences (unweighted and optionally weighted)."""
    t_mask = df[treat_col] == 1
    rows = []
    for cov in covariates:
        x = pd.to_numeric(df[cov], errors='coerce').fillna(0).values
        x_t, x_c = x[t_mask], x[~t_mask]

        # Unweighted SMD
        pooled_sd = np.sqrt((x_t.var() + x_c.var()) / 2)
        smd_raw = (x_t.mean() - x_c.mean()) / pooled_sd if pooled_sd > 0 else 0.0

        # Weighted SMD
        smd_w = np.nan
        if weights is not None:
            w_t, w_c = weights[t_mask], weights[~t_mask]
            wm_t = np.average(x_t, weights=w_t) if w_t.sum() > 0 else x_t.mean()
            wm_c = np.average(x_c, weights=w_c) if w_c.sum() > 0 else x_c.mean()
            smd_w = (wm_t - wm_c) / pooled_sd if pooled_sd > 0 else 0.0

        rows.append({'covariate': cov, 'SMD_unweighted': smd_raw, 'SMD_weighted': smd_w})
    return pd.DataFrame(rows)


def recalibrate_propensity_within_subset(df, ps_col='ICI_prediction', treat_col='PX_on_ICI'):
    """Recalibrate pan-cancer propensity scores within a cancer-type subset via Platt scaling.

    Fits a logistic regression of treatment status on the pan-cancer propensity score
    within the subset, producing recalibrated probabilities that reflect within-type
    covariate balance rather than the pan-cancer distribution.
    """
    ps = df[[ps_col]].values
    y = df[treat_col].values.astype(int)
    lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    lr.fit(ps, y)
    return lr.predict_proba(ps)[:, 1]


# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')
IPTW_RUN_PATH = os.path.join(MARKER_PATH, 'IPTW_runs/')
os.makedirs(IPTW_RUN_PATH, exist_ok=True)

interaction_ICI_df = pd.read_csv(os.path.join(MARKER_PATH, 'IPTW_ICI_interaction_runs_df.csv'))

required_vars = ['DFCI_MRN', 'tt_death', 'death']
base_covars = ['GENDER', 'AGE_AT_TREATMENTSTART']
panel_cols = [col for col in interaction_ICI_df.columns if col.upper().startswith('PANEL_VERSION_')]
cancer_type_cols = [col for col in interaction_ICI_df.columns if col.startswith('CANCER_TYPE_')]
excluded_cols = required_vars + base_covars + panel_cols + cancer_type_cols + ['PX_on_ICI', 'ICI_prediction', 'text_risk_score']
mutation_tags = ('_SNV', '_SV', '_FUSION', '_DEL', '_AMP', '_CNV')
biomarker_cols = [
    col for col in interaction_ICI_df.columns
    if (col not in excluded_cols) and any(tag in col.upper() for tag in mutation_tags)
]
MIN_CANCER_TYPE_TOTAL = 30
COMMON_SUPPORT_PCT = (1, 99)
IPTW_TRUNC_PCT = (5, 95)
MAX_IPTW = 20.0
MIN_MARKER_PREVALENCE = 0.05
MIN_MARKER_POS_PER_ARM = 5
MIN_MARKER_NEG_PER_ARM = 5

result_cols = [
    'marker', 'beta_markerxICI', 'HR_markerxICI', 'p_markerxICI',
    'beta_marker_ICI', 'HR_marker_ICI', 'CI95_marker_ICI_low',
    'CI95_marker_ICI_high', 'p_marker_ICI',
    'beta_marker_nonICI', 'HR_marker_nonICI', 'CI95_marker_nonICI_low',
    'CI95_marker_nonICI_high', 'p_marker_nonICI',
    'beta_ICI_at_marker0', 'p_ICI_at_marker0'
]


def _fdr_within_mutation_type(results_df, p_col, fdr_col, sig_col):
    """Apply FDR correction within each mutation type."""
    results_df[fdr_col] = np.nan
    results_df[sig_col] = False
    for mut_type in results_df['mutation_type'].unique():
        mask = results_df['mutation_type'] == mut_type
        pvals = results_df.loc[mask, p_col]
        if pvals.empty:
            continue
        rej, fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        results_df.loc[mask, fdr_col] = fdr
        results_df.loc[mask, sig_col] = rej


def add_fdr_and_labels(results_df, classifier_fn):
    if results_df.empty:
        for col in ['mutation_type', 'classifier']:
            results_df[col] = pd.Series(dtype=str)
        for col in ['FDR_markerxICI', 'FDR_marker_ICI', 'FDR_marker_nonICI']:
            results_df[col] = pd.Series(dtype=float)
        for col in ['significant_predictive', 'significant_in_ICI', 'significant_prognostic_nonICI']:
            results_df[col] = pd.Series(dtype=bool)
        return results_df

    results_df['mutation_type'] = results_df['marker'].apply(get_mutation_type)

    _fdr_within_mutation_type(results_df, 'p_markerxICI', 'FDR_markerxICI', 'significant_predictive')
    _fdr_within_mutation_type(results_df, 'p_marker_ICI', 'FDR_marker_ICI', 'significant_in_ICI')
    _fdr_within_mutation_type(results_df, 'p_marker_nonICI', 'FDR_marker_nonICI', 'significant_prognostic_nonICI')

    results_df['classifier'] = results_df.apply(classifier_fn, axis=1)
    return results_df

def fit_cph_log_warnings(
    cph: CoxPHFitter,
    df_fit: pd.DataFrame,
    duration_col: str,
    event_col: str,
    weights_col: str | None = None,
    robust: bool = True,
    marker_name: str = "",
) -> CoxPHFitter:
    """Fit CoxPH, logging convergence/LinAlg warnings instead of silencing them."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if weights_col is not None:
            cph.fit(
                df_fit,
                duration_col=duration_col,
                event_col=event_col,
                weights_col=weights_col,
                robust=robust,
            )
        else:
            cph.fit(
                df_fit,
                duration_col=duration_col,
                event_col=event_col,
                robust=robust,
            )
    for w in caught:
        if issubclass(w.category, (ConvergenceWarning, LinAlgWarning)):
            logger.warning("marker=%s: %s: %s", marker_name, w.category.__name__, w.message)
    return cph


def _fit_one_marker(
    df: pd.DataFrame,
    marker: str,
    base_vars: list[str],
    weights_col: str | None,
) -> dict | None:
    """Fit a single marker interaction Cox model. Returns result dict or None on failure."""
    cols = ['tt_death', 'death', 'PX_on_ICI'] + base_vars + [marker]
    if weights_col is not None:
        cols.append(weights_col)
    df_fit = df[cols].dropna().copy()

    mx = f"{marker}_x_ICI"
    df_fit[mx] = df_fit['PX_on_ICI'] * df_fit[marker]

    cph = CoxPHFitter()
    cph = fit_cph_log_warnings(
        cph, df_fit,
        duration_col='tt_death', event_col='death',
        weights_col=weights_col, robust=True,
        marker_name=marker,
    )

    summ = cph.summary.reset_index()
    V = cph.variance_matrix_
    b = cph.params_

    beta_m = float(b[marker])
    se_m = float(np.sqrt(V.loc[marker, marker]))
    p_m = float(summ.loc[summ['covariate'] == marker, 'p'].values[0])

    beta_mx = float(b[mx])
    se_mx = float(np.sqrt(V.loc[mx, mx]))
    p_mx = float(summ.loc[summ['covariate'] == mx, 'p'].values[0])

    hr_nonici = np.exp(beta_m)
    ci_nonici = (np.exp(beta_m - 1.96 * se_m), np.exp(beta_m + 1.96 * se_m))

    cov_m_mx = float(V.loc[marker, mx])
    se_ici = np.sqrt(se_m**2 + se_mx**2 + 2 * cov_m_mx)
    beta_ici = beta_m + beta_mx
    hr_ici = np.exp(beta_ici)
    ci_ici = (np.exp(beta_ici - 1.96 * se_ici), np.exp(beta_ici + 1.96 * se_ici))

    z_ici = beta_ici / se_ici
    p_ici = 2 * (1 - stats.norm.cdf(abs(z_ici)))

    beta_IO0 = float(b['PX_on_ICI']) if 'PX_on_ICI' in b.index else np.nan
    p_IO0 = (float(summ.loc[summ['covariate'] == 'PX_on_ICI', 'p'].values[0])
             if 'PX_on_ICI' in summ['covariate'].values else np.nan)

    return {
        "marker": marker,
        "beta_markerxICI": beta_mx, "HR_markerxICI": np.exp(beta_mx), "p_markerxICI": p_mx,
        "beta_marker_ICI": beta_ici, "HR_marker_ICI": hr_ici,
        "CI95_marker_ICI_low": ci_ici[0], "CI95_marker_ICI_high": ci_ici[1], "p_marker_ICI": p_ici,
        "beta_marker_nonICI": beta_m, "HR_marker_nonICI": hr_nonici,
        "CI95_marker_nonICI_low": ci_nonici[0], "CI95_marker_nonICI_high": ci_nonici[1], "p_marker_nonICI": p_m,
        "beta_ICI_at_marker0": beta_IO0, "p_ICI_at_marker0": p_IO0,
    }


def _run_marker_screen(df, markers, base_vars, weights_col, n_jobs, label=""):
    """Run parallel marker screening. Returns (results_list, failed_list)."""
    def _safe_fit(marker):
        try:
            return _fit_one_marker(df, marker, base_vars, weights_col), None
        except Exception as e:
            return None, (marker, str(e))

    raw = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_safe_fit)(m) for m in tqdm(markers, desc=label)
    )
    results = [r for r, _ in raw if r is not None]
    failed = [f for _, f in raw if f is not None]
    return results, failed

types_to_test = ['pan_cancer', 'SKIN', 'LUNG']

for cancer_type in types_to_test:
    if cancer_type == 'pan_cancer':
        type_specific_interaction_ICI_df = interaction_ICI_df.copy()
        (
            type_specific_interaction_ICI_df,
            pan_cancer_type_cols,
            merged_rare_type_cols,
        ) = merge_rare_cancer_types_into_other(
            type_specific_interaction_ICI_df,
            min_total=MIN_CANCER_TYPE_TOTAL,
        )
        if merged_rare_type_cols:
            print(
                f"[pan_cancer] Merged {len(merged_rare_type_cols)} rare cancer types into CANCER_TYPE_OTHER: "
                + ", ".join(sorted(merged_rare_type_cols))
            )
        panel_cols_for_fit = [col for col in type_specific_interaction_ICI_df if 'PANEL' in col]
        cancer_type_cols_for_fit = [col for col in type_specific_interaction_ICI_df if 'CANCER_TYPE' in col]
        base_vars = base_covars + panel_cols_for_fit + cancer_type_cols_for_fit
    else:
        type_specific_interaction_ICI_df = interaction_ICI_df.loc[interaction_ICI_df[f'CANCER_TYPE_{cancer_type}']].copy()
        # Recalibrate pan-cancer propensity scores within this cancer-type subset
        if type_specific_interaction_ICI_df['PX_on_ICI'].nunique() >= 2 and len(type_specific_interaction_ICI_df) > 10:
            ps_before = type_specific_interaction_ICI_df['ICI_prediction'].copy()
            type_specific_interaction_ICI_df['ICI_prediction'] = recalibrate_propensity_within_subset(
                type_specific_interaction_ICI_df)
            print(f"[{cancer_type}] Recalibrated propensity scores within subset "
                  f"(mean {ps_before.mean():.3f} -> {type_specific_interaction_ICI_df['ICI_prediction'].mean():.3f})")
        panel_cols_for_fit = [col for col in type_specific_interaction_ICI_df if 'PANEL' in col]
        base_vars = base_covars + panel_cols_for_fit

    if type_specific_interaction_ICI_df.empty:
        print(f"[{cancer_type}] Skipping: no rows available.")
        continue

    if type_specific_interaction_ICI_df['PX_on_ICI'].nunique() < 2:
        print(f"[{cancer_type}] Skipping: only one treatment group present before trimming.")
        continue

    # ---------------------------------------
    # Common support trimming
    # ---------------------------------------
    eps = 1e-6
    ps_raw = type_specific_interaction_ICI_df['ICI_prediction'].clip(eps, 1 - eps)
    
    ps_t = ps_raw[type_specific_interaction_ICI_df['PX_on_ICI'] == 1]
    ps_c = ps_raw[type_specific_interaction_ICI_df['PX_on_ICI'] == 0]

    if ps_t.empty or ps_c.empty:
        print(f"[{cancer_type}] Skipping: missing treated or control patients for common support.")
        continue

    lower = max(np.percentile(ps_t, COMMON_SUPPORT_PCT[0]),
                np.percentile(ps_c, COMMON_SUPPORT_PCT[0]))
    upper = min(np.percentile(ps_t, COMMON_SUPPORT_PCT[1]),
                np.percentile(ps_c, COMMON_SUPPORT_PCT[1]))

    if pd.isna(lower) or pd.isna(upper) or lower >= upper:
        print(f"[{cancer_type}] Skipping: no propensity overlap after common-support checks.")
        continue
    
    type_specific_interaction_ICI_df = type_specific_interaction_ICI_df[(ps_raw >= lower) & (ps_raw <= upper)].copy()

    if type_specific_interaction_ICI_df.empty:
        print(f"[{cancer_type}] Skipping: no rows left after common-support trimming.")
        continue

    if type_specific_interaction_ICI_df['PX_on_ICI'].nunique() < 2:
        print(f"[{cancer_type}] Skipping: only one treatment group left after trimming.")
        continue

    ps = type_specific_interaction_ICI_df['ICI_prediction'].clip(eps, 1 - eps)
    
    # ---------------------------------------
    # ATT (Average Treatment effect on the Treated) IPTW with truncation
    # Treated patients receive weight 1; controls receive ps/(1-ps)
    # so the pseudo-population reflects the covariate distribution
    # of the treated group.
    # ---------------------------------------
    p_treated = type_specific_interaction_ICI_df['PX_on_ICI'].mean()

    if p_treated <= 0 or p_treated >= 1:
        print(f"[{cancer_type}] Skipping: invalid treated proportion ({p_treated:.4f}).")
        continue

    w = np.where(type_specific_interaction_ICI_df['PX_on_ICI']==1, 1.0, ps/(1-ps))

    if len(w) == 0:
        print(f"[{cancer_type}] Skipping: no IPTW weights computed.")
        continue

    low, high = np.percentile(w, IPTW_TRUNC_PCT)
    if not np.isfinite(low) or not np.isfinite(high):
        print(f"[{cancer_type}] Skipping: non-finite IPTW truncation bounds.")
        continue

    w_trunc = np.clip(w, low, high)
    w_trunc = np.clip(w_trunc, 0, MAX_IPTW)
    type_specific_interaction_ICI_df['IPTW'] = w_trunc

    # Effective sample size per arm
    treat_mask = type_specific_interaction_ICI_df['PX_on_ICI'] == 1
    w_t, w_c = w_trunc[treat_mask], w_trunc[~treat_mask]
    ess_t = w_t.sum() ** 2 / (w_t ** 2).sum()
    ess_c = w_c.sum() ** 2 / (w_c ** 2).sum()
    print(f"[{cancer_type}] N treated={treat_mask.sum()}, N control={(~treat_mask).sum()} | "
          f"ESS treated={ess_t:.0f}, ESS control={ess_c:.0f}")

    # ---------------------------------------
    # Overlap diagnostics
    # ---------------------------------------
    diag_path = os.path.join(IPTW_RUN_PATH, f'{cancer_type}_diagnostics/')
    os.makedirs(diag_path, exist_ok=True)

    # Propensity score summary by arm
    ps_diag = type_specific_interaction_ICI_df[['PX_on_ICI', 'ICI_prediction']].copy()
    ps_diag['IPTW'] = w_trunc
    ps_summary = ps_diag.groupby('PX_on_ICI')['ICI_prediction'].describe()
    ps_summary.to_csv(os.path.join(diag_path, 'propensity_score_summary.csv'))

    # ESS
    pd.DataFrame([{
        'N_treated': int(treat_mask.sum()), 'N_control': int((~treat_mask).sum()),
        'ESS_treated': ess_t, 'ESS_control': ess_c,
    }]).to_csv(os.path.join(diag_path, 'effective_sample_sizes.csv'), index=False)

    # Covariate balance (SMD before and after weighting)
    balance_covars = base_covars + [c for c in type_specific_interaction_ICI_df.columns
                                    if c.startswith('CANCER_TYPE_') or c.upper().startswith('PANEL_VERSION_')]
    smd_df = compute_smd(type_specific_interaction_ICI_df, balance_covars, weights=w_trunc)
    smd_df.to_csv(os.path.join(diag_path, 'covariate_balance_smd.csv'), index=False)

    # ---------------------------------------
    # Cox IPTW marker screening
    # ---------------------------------------
    markers_to_test = []
    for marker in biomarker_cols:
        prevalence = pd.to_numeric(type_specific_interaction_ICI_df[marker], errors='coerce').sum(skipna=True) / len(type_specific_interaction_ICI_df)
        if prevalence < MIN_MARKER_PREVALENCE:
            continue
        if marker_has_within_arm_support(
            type_specific_interaction_ICI_df,
            marker,
            treat_col='PX_on_ICI',
            min_pos_per_arm=MIN_MARKER_POS_PER_ARM,
            min_neg_per_arm=MIN_MARKER_NEG_PER_ARM,
        ):
            markers_to_test.append(marker)
    
    # --- 4 sensitivity specifications: ±IPTW × ±text_risk_score ---
    has_risk = 'text_risk_score' in type_specific_interaction_ICI_df.columns and \
               type_specific_interaction_ICI_df['text_risk_score'].notna().any()
    base_vars_no_risk = list(base_vars)
    base_vars_with_risk = base_vars_no_risk + (['text_risk_score'] if has_risk else [])

    specs = [
        ('IPTW_risk',     base_vars_with_risk, 'IPTW'),
        ('IPTW_norisk',   base_vars_no_risk,   'IPTW'),
        ('noIPTW_risk',   base_vars_with_risk, None),
        ('noIPTW_norisk', base_vars_no_risk,   None),
    ]
    if not has_risk:
        print(f"[{cancer_type}] text_risk_score not available; running 2 specs (±IPTW only).")
        specs = [s for s in specs if 'norisk' in s[0]]

    for spec_name, spec_base_vars, spec_weights in specs:
        results, failed = _run_marker_screen(
            type_specific_interaction_ICI_df, markers_to_test, spec_base_vars,
            weights_col=spec_weights, n_jobs=N_JOBS, label=f"{cancer_type} {spec_name}",
        )
        if failed:
            print(f"[{cancer_type}] {spec_name} failures: {len(failed)}. First: {failed[0][0]} -> {failed[0][1]}")

        spec_df = pd.DataFrame(results, columns=result_cols)
        spec_df = add_fdr_and_labels(spec_df, classify)
        spec_df.to_csv(os.path.join(IPTW_RUN_PATH, f'{cancer_type}_{spec_name}_ICI_predictive_markers.csv'), index=False)
