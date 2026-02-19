"""Run Iptw Analysis script for biomarker analysis workflows."""

import os
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
random.seed(42)  # set seed for reproducibility

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


def add_fdr_and_labels(results_df, classifier_fn):
    if results_df.empty:
        results_df['FDR_markerxICI'] = pd.Series(dtype=float)
        results_df['significant_predictive'] = pd.Series(dtype=bool)
        results_df['FDR_marker_ICI'] = pd.Series(dtype=float)
        results_df['significant_in_ICI'] = pd.Series(dtype=bool)
        results_df['FDR_marker_nonICI'] = pd.Series(dtype=float)
        results_df['significant_prognostic_nonICI'] = pd.Series(dtype=bool)
        results_df['classifier'] = pd.Series(dtype=str)
        return results_df

    rej_int, fdr_int, _, _ = multipletests(results_df['p_markerxICI'], alpha=0.05, method='fdr_bh')
    results_df['FDR_markerxICI'] = fdr_int
    results_df['significant_predictive'] = rej_int

    rej_ici, fdr_ici, _, _ = multipletests(results_df['p_marker_ICI'], alpha=0.05, method='fdr_bh')
    results_df['FDR_marker_ICI'] = fdr_ici
    results_df['significant_in_ICI'] = rej_ici

    rej_nonici, fdr_nonici, _, _ = multipletests(results_df['p_marker_nonICI'], alpha=0.05, method='fdr_bh')
    results_df['FDR_marker_nonICI'] = fdr_nonici
    results_df['significant_prognostic_nonICI'] = rej_nonici

    results_df['classifier'] = results_df.apply(classifier_fn, axis=1)
    return results_df

def fit_cph_suppress_warnings(
    cph: CoxPHFitter,
    df_fit: pd.DataFrame,
    duration_col: str,
    event_col: str,
    weights_col: str | None = None,
    robust: bool = True,
) -> CoxPHFitter:
    """Fit CoxPH while suppressing lifelines convergence warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=LinAlgWarning)
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
    cph = fit_cph_suppress_warnings(
        cph, df_fit,
        duration_col='tt_death', event_col='death',
        weights_col=weights_col, robust=True,
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

    lower, upper = max(ps_t.min(), ps_c.min()), min(ps_t.max(), ps_c.max())

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
    # ATT IPTW with truncation
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
