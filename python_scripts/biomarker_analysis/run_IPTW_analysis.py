"""Run IPTW biomarker analysis with two tracks.

Track 1 (ICI-only, generalizability-weighted):
  S(t) ~ base_vars + line_dummies + marker
  Effect: marker coefficient

Track 2 (full cohort, IPTW-weighted):
  S(t) ~ base_vars + line_dummies + marker + PX_on_ICI + marker x ICI
  Effect: interaction coefficient

Usage:
  python run_IPTW_analysis.py --matching 1to1 --ps_model embeddings_only
  python run_IPTW_analysis.py --matching 1tok --ps_model all_covariates
"""

import os
import argparse
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

random.seed(42)

parser = argparse.ArgumentParser(description='Run IPTW biomarker analysis.')
parser.add_argument('--matching', choices=['1to1', '1tok'], required=True)
parser.add_argument('--ps_model', choices=['embeddings_only', 'all_covariates'], required=True)
args = parser.parse_args()

MATCHING = args.matching
PS_MODEL = args.ps_model
SPEC_LABEL = f'{MATCHING}_{PS_MODEL}'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

N_JOBS = int(os.getenv("SLURM_CPUS_PER_TASK", "-1"))

# === Paths ===
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')
RUN_PATH = os.path.join(MARKER_PATH, f'IPTW_runs_{SPEC_LABEL}/')
os.makedirs(RUN_PATH, exist_ok=True)

print(f"[run_IPTW_analysis] Spec: {SPEC_LABEL}")
print(f"[run_IPTW_analysis] Output: {RUN_PATH}")

# === Load data ===
input_file = os.path.join(MARKER_PATH, f'IPTW_df_{SPEC_LABEL}.csv')
full_df = pd.read_csv(input_file)

# === Identify column groups ===
required_vars = ['DFCI_MRN', 'tt_death', 'death']
base_covars = ['GENDER', 'AGE_AT_TREATMENTSTART']
line_cols = sorted([col for col in full_df.columns if col.startswith('LINE_')])
panel_cols = [col for col in full_df.columns if col.upper().startswith('PANEL_VERSION_')]
cancer_type_cols = [col for col in full_df.columns if col.startswith('CANCER_TYPE_')]
excluded_cols = (required_vars + base_covars + line_cols + panel_cols +
                 cancer_type_cols + ['PX_on_ICI', 'ICI_prediction'])
mutation_tags = ('_SNV', '_SV', '_FUSION', '_DEL', '_AMP', '_CNV')
biomarker_cols = [
    col for col in full_df.columns
    if (col not in excluded_cols) and any(tag in col.upper() for tag in mutation_tags)
]

# === Constants ===
MIN_CANCER_TYPE_TOTAL = 30
COMMON_SUPPORT_PCT = (0.5, 99.5)
IPTW_TRUNC_PCT = (1, 99)
MIN_MARKER_POS_PER_ARM = 3
MIN_MARKER_NEG_PER_ARM = 3
MIN_MARKER_POS_ICI_ONLY = 5


# === Utility functions ===

def classify(row):
    if row['significant_predictive']:
        if row['HR_markerxICI'] < 1:
            return "predictive_ICI_benefit"
        else:
            return "predictive_ICI_harm"
    if row['significant_in_ICI'] and not row['significant_prognostic_nonICI']:
        return "ICI_specific_effect"
    if row['significant_prognostic_nonICI'] and not row['significant_in_ICI']:
        return "prognostic_nonICI"
    return "no_signal"


def merge_rare_cancer_types_into_other(df, min_total=30):
    out = df.copy()
    c_cols = [col for col in out.columns if col.startswith('CANCER_TYPE_')]
    if not c_cols:
        return out, [], []

    cancer_matrix = out[c_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    cancer_matrix = (cancer_matrix > 0).astype(int)
    total_counts = cancer_matrix.sum(axis=0)

    keep_cols = [c for c in c_cols if total_counts[c] >= min_total and c != 'CANCER_TYPE_OTHER']
    rare_cols = [c for c in c_cols if c not in keep_cols and c != 'CANCER_TYPE_OTHER']

    existing_other = (cancer_matrix['CANCER_TYPE_OTHER'].copy()
                      if 'CANCER_TYPE_OTHER' in cancer_matrix.columns
                      else pd.Series(0, index=out.index, dtype=int))
    merged_other = existing_other.copy()
    if rare_cols:
        merged_other = ((merged_other + cancer_matrix[rare_cols].sum(axis=1)) > 0).astype(int)

    out = out.drop(columns=c_cols)
    for c in keep_cols:
        out[c] = cancer_matrix[c].astype(int)
    out['CANCER_TYPE_OTHER'] = merged_other.astype(int)

    kept = [c for c in (keep_cols + ['CANCER_TYPE_OTHER']) if out[c].sum() > 0]
    return out, kept, rare_cols


def marker_has_within_arm_support(df, marker, treat_col='PX_on_ICI',
                                  min_pos_per_arm=5, min_neg_per_arm=5):
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


def marker_has_ici_only_support(df, marker, min_pos=5):
    """Check if marker has enough positive cases within ICI patients only."""
    ici_df = df.loc[df['PX_on_ICI'] == 1]
    marker_pos = (pd.to_numeric(ici_df[marker], errors='coerce').fillna(0) > 0).sum()
    marker_neg = len(ici_df) - marker_pos
    return marker_pos >= min_pos and marker_neg >= min_pos


def compute_smd(df, covariates, treat_col='PX_on_ICI', weights=None):
    t_mask = df[treat_col] == 1
    rows = []
    for cov in covariates:
        x = pd.to_numeric(df[cov], errors='coerce').fillna(0).values
        x_t, x_c = x[t_mask], x[~t_mask]
        pooled_sd = np.sqrt((x_t.var() + x_c.var()) / 2)
        smd_raw = (x_t.mean() - x_c.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        smd_w = np.nan
        if weights is not None:
            w_t, w_c = weights[t_mask], weights[~t_mask]
            wm_t = np.average(x_t, weights=w_t) if w_t.sum() > 0 else x_t.mean()
            wm_c = np.average(x_c, weights=w_c) if w_c.sum() > 0 else x_c.mean()
            smd_w = (wm_t - wm_c) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append({'covariate': cov, 'SMD_unweighted': smd_raw, 'SMD_weighted': smd_w})
    return pd.DataFrame(rows)


def recalibrate_propensity_within_subset(df, ps_col='ICI_prediction', treat_col='PX_on_ICI'):
    ps = df[[ps_col]].values
    y = df[treat_col].values.astype(int)
    lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    lr.fit(ps, y)
    return lr.predict_proba(ps)[:, 1]


def fit_cph_log_warnings(cph, df_fit, duration_col, event_col,
                          weights_col=None, robust=True, marker_name=""):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit_kwargs = dict(duration_col=duration_col, event_col=event_col, robust=robust)
        if weights_col is not None:
            fit_kwargs['weights_col'] = weights_col
        cph.fit(df_fit, **fit_kwargs)
    for w in caught:
        if issubclass(w.category, (ConvergenceWarning, LinAlgWarning)):
            logger.warning("marker=%s: %s: %s", marker_name, w.category.__name__, w.message)
    return cph


def _fdr_within_mutation_type(results_df, p_col, fdr_col, sig_col):
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


# =============================================
# Track 2: Full-cohort interaction model
# =============================================

TRACK2_RESULT_COLS = [
    'marker', 'beta_markerxICI', 'HR_markerxICI', 'p_markerxICI',
    'beta_marker_ICI', 'HR_marker_ICI', 'CI95_marker_ICI_low',
    'CI95_marker_ICI_high', 'p_marker_ICI',
    'beta_marker_nonICI', 'HR_marker_nonICI', 'CI95_marker_nonICI_low',
    'CI95_marker_nonICI_high', 'p_marker_nonICI',
    'beta_ICI_at_marker0', 'p_ICI_at_marker0',
]


def _fit_track2_marker(df, marker, base_vars, weights_col):
    """Track 2: interaction model on full cohort."""
    cols = ['tt_death', 'death', 'PX_on_ICI'] + base_vars + [marker]
    if weights_col is not None:
        cols.append(weights_col)
    df_fit = df[cols].dropna().copy()

    mx = f"{marker}_x_ICI"
    df_fit[mx] = df_fit['PX_on_ICI'] * df_fit[marker]

    cph = CoxPHFitter(penalizer=0.01)
    cph = fit_cph_log_warnings(cph, df_fit, 'tt_death', 'death',
                                weights_col=weights_col, robust=True, marker_name=marker)

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
        "CI95_marker_nonICI_low": ci_nonici[0], "CI95_marker_nonICI_high": ci_nonici[1],
        "p_marker_nonICI": p_m,
        "beta_ICI_at_marker0": beta_IO0, "p_ICI_at_marker0": p_IO0,
    }


def add_track2_fdr_and_labels(results_df):
    if results_df.empty:
        for col in ['mutation_type', 'classifier']:
            results_df[col] = pd.Series(dtype=str)
        return results_df

    results_df['mutation_type'] = results_df['marker'].apply(get_mutation_type)
    _fdr_within_mutation_type(results_df, 'p_markerxICI', 'FDR_markerxICI', 'significant_predictive')
    _fdr_within_mutation_type(results_df, 'p_marker_ICI', 'FDR_marker_ICI', 'significant_in_ICI')
    _fdr_within_mutation_type(results_df, 'p_marker_nonICI', 'FDR_marker_nonICI',
                              'significant_prognostic_nonICI')
    results_df['classifier'] = results_df.apply(classify, axis=1)
    return results_df


# =============================================
# Track 1: ICI-only generalizability-weighted
# =============================================

TRACK1_RESULT_COLS = [
    'marker', 'beta_marker', 'HR_marker', 'CI95_marker_low', 'CI95_marker_high', 'p_marker',
]


def _fit_track1_marker(df, marker, base_vars, weights_col):
    """Track 1: marker-only model on ICI patients."""
    cols = ['tt_death', 'death'] + base_vars + [marker]
    if weights_col is not None:
        cols.append(weights_col)
    df_fit = df[cols].dropna().copy()

    cph = CoxPHFitter(penalizer=0.01)
    cph = fit_cph_log_warnings(cph, df_fit, 'tt_death', 'death',
                                weights_col=weights_col, robust=True, marker_name=marker)

    summ = cph.summary.reset_index()
    b = cph.params_
    V = cph.variance_matrix_

    beta_m = float(b[marker])
    se_m = float(np.sqrt(V.loc[marker, marker]))
    p_m = float(summ.loc[summ['covariate'] == marker, 'p'].values[0])
    hr_m = np.exp(beta_m)
    ci_m = (np.exp(beta_m - 1.96 * se_m), np.exp(beta_m + 1.96 * se_m))

    return {
        "marker": marker,
        "beta_marker": beta_m, "HR_marker": hr_m,
        "CI95_marker_low": ci_m[0], "CI95_marker_high": ci_m[1],
        "p_marker": p_m,
    }


def add_track1_fdr(results_df):
    if results_df.empty:
        results_df['mutation_type'] = pd.Series(dtype=str)
        return results_df
    results_df['mutation_type'] = results_df['marker'].apply(get_mutation_type)
    _fdr_within_mutation_type(results_df, 'p_marker', 'FDR_marker', 'significant_marker')
    return results_df


# =============================================
# Parallel runner
# =============================================

def _run_marker_screen(df, markers, base_vars, weights_col, fit_fn, n_jobs, label=""):
    def _safe_fit(marker):
        try:
            return fit_fn(df, marker, base_vars, weights_col), None
        except Exception as e:
            return None, (marker, str(e))

    raw = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_safe_fit)(m) for m in tqdm(markers, desc=label)
    )
    results = [r for r, _ in raw if r is not None]
    failed = [f for _, f in raw if f is not None]
    return results, failed


# =============================================
# Main loop over cancer types
# =============================================

types_to_test = ['pan_cancer', 'SKIN', 'LUNG']

for cancer_type in types_to_test:
    print(f"\n{'='*60}")
    print(f"Cancer type: {cancer_type}")
    print(f"{'='*60}")

    if cancer_type == 'pan_cancer':
        type_df = full_df.copy()
        type_df, pan_ct_cols, merged_rare = merge_rare_cancer_types_into_other(
            type_df, min_total=MIN_CANCER_TYPE_TOTAL)
        if merged_rare:
            print(f"  Merged {len(merged_rare)} rare cancer types into OTHER: "
                  + ", ".join(sorted(merged_rare)))
        panel_cols_fit = [c for c in type_df if 'PANEL' in c]
        ct_cols_fit = [c for c in type_df if 'CANCER_TYPE' in c]
        base_vars = base_covars + line_cols + panel_cols_fit + ct_cols_fit
    else:
        ct_col = f'CANCER_TYPE_{cancer_type}'
        if ct_col not in full_df.columns:
            print(f"  Skipping: column {ct_col} not found.")
            continue
        type_df = full_df.loc[full_df[ct_col].astype(bool)].copy()
        # Recalibrate propensity within subset
        if type_df['PX_on_ICI'].nunique() >= 2 and len(type_df) > 10:
            ps_before = type_df['ICI_prediction'].copy()
            type_df['ICI_prediction'] = recalibrate_propensity_within_subset(type_df)
            print(f"  Recalibrated PS (mean {ps_before.mean():.3f} -> "
                  f"{type_df['ICI_prediction'].mean():.3f})")
        panel_cols_fit = [c for c in type_df if 'PANEL' in c]
        base_vars = base_covars + line_cols + panel_cols_fit

    if type_df.empty:
        print(f"  Skipping: no rows.")
        continue
    if type_df['PX_on_ICI'].nunique() < 2:
        print(f"  Skipping: only one treatment group.")
        continue

    # --- Common support trimming ---
    eps = 1e-6
    ps_raw = type_df['ICI_prediction'].clip(eps, 1 - eps)
    ps_t = ps_raw[type_df['PX_on_ICI'] == 1]
    ps_c = ps_raw[type_df['PX_on_ICI'] == 0]

    if ps_t.empty or ps_c.empty:
        print(f"  Skipping: missing treated or control for common support.")
        continue

    lower = max(np.percentile(ps_t, COMMON_SUPPORT_PCT[0]),
                np.percentile(ps_c, COMMON_SUPPORT_PCT[0]))
    upper = min(np.percentile(ps_t, COMMON_SUPPORT_PCT[1]),
                np.percentile(ps_c, COMMON_SUPPORT_PCT[1]))

    if pd.isna(lower) or pd.isna(upper) or lower >= upper:
        print(f"  Skipping: no propensity overlap.")
        continue

    type_df = type_df[(ps_raw >= lower) & (ps_raw <= upper)].copy()
    if type_df.empty or type_df['PX_on_ICI'].nunique() < 2:
        print(f"  Skipping: no rows or one group after trimming.")
        continue

    ps = type_df['ICI_prediction'].clip(eps, 1 - eps)
    treat_mask = type_df['PX_on_ICI'] == 1
    p_treated = type_df['PX_on_ICI'].mean()

    if p_treated <= 0 or p_treated >= 1:
        print(f"  Skipping: invalid treated proportion ({p_treated:.4f}).")
        continue

    # --- Stabilized ATE weights ---
    w_ate = np.where(treat_mask, p_treated / ps, (1 - p_treated) / (1 - ps))
    low, high = np.percentile(w_ate, IPTW_TRUNC_PCT)
    if not np.isfinite(low) or not np.isfinite(high):
        print(f"  Skipping: non-finite ATE truncation bounds.")
        continue
    w_ate_trunc = np.clip(w_ate, low, high)
    type_df['IPTW_ATE'] = w_ate_trunc

    # --- Stabilized ATT weights ---
    w_att = np.where(treat_mask, 1.0, ((1 - p_treated) / p_treated) * (ps / (1 - ps)))
    low_att, high_att = np.percentile(w_att, IPTW_TRUNC_PCT)
    if not np.isfinite(low_att) or not np.isfinite(high_att):
        print(f"  Skipping: non-finite ATT truncation bounds.")
        continue
    w_att_trunc = np.clip(w_att, low_att, high_att)
    type_df['IPTW_ATT'] = w_att_trunc

    # --- ESS ---
    w_t, w_c = w_ate_trunc[treat_mask], w_ate_trunc[~treat_mask]
    ess_t = w_t.sum() ** 2 / (w_t ** 2).sum()
    ess_c = w_c.sum() ** 2 / (w_c ** 2).sum()
    print(f"  ATE: N treated={treat_mask.sum()}, N control={(~treat_mask).sum()} | "
          f"ESS treated={ess_t:.0f}, ESS control={ess_c:.0f}")

    w_t_att, w_c_att = w_att_trunc[treat_mask], w_att_trunc[~treat_mask]
    ess_t_att = w_t_att.sum() ** 2 / (w_t_att ** 2).sum()
    ess_c_att = w_c_att.sum() ** 2 / (w_c_att ** 2).sum()
    print(f"  ATT: N treated={treat_mask.sum()}, N control={(~treat_mask).sum()} | "
          f"ESS treated={ess_t_att:.0f}, ESS control={ess_c_att:.0f}")

    # --- Diagnostics ---
    diag_path = os.path.join(RUN_PATH, f'{cancer_type}_diagnostics/')
    os.makedirs(diag_path, exist_ok=True)

    ps_diag = type_df[['PX_on_ICI', 'ICI_prediction']].copy()
    ps_diag['IPTW_ATE'] = w_ate_trunc
    ps_diag['IPTW_ATT'] = w_att_trunc
    ps_diag.groupby('PX_on_ICI')['ICI_prediction'].describe().to_csv(
        os.path.join(diag_path, 'propensity_score_summary.csv'))

    pd.DataFrame([{
        'N_treated': int(treat_mask.sum()), 'N_control': int((~treat_mask).sum()),
        'ESS_ATE_treated': ess_t, 'ESS_ATE_control': ess_c,
        'ESS_ATT_treated': ess_t_att, 'ESS_ATT_control': ess_c_att,
    }]).to_csv(os.path.join(diag_path, 'effective_sample_sizes.csv'), index=False)

    balance_covars = base_covars + line_cols + [
        c for c in type_df.columns
        if c.startswith('CANCER_TYPE_') or c.upper().startswith('PANEL_VERSION_')]
    smd_ate = compute_smd(type_df, balance_covars, weights=w_ate_trunc)
    smd_ate.to_csv(os.path.join(diag_path, 'covariate_balance_smd_ATE.csv'), index=False)
    smd_att = compute_smd(type_df, balance_covars, weights=w_att_trunc)
    smd_att.to_csv(os.path.join(diag_path, 'covariate_balance_smd_ATT.csv'), index=False)

    # === Track 2: full-cohort interaction ===
    track2_markers = [
        m for m in biomarker_cols
        if marker_has_within_arm_support(type_df, m, min_pos_per_arm=MIN_MARKER_POS_PER_ARM,
                                         min_neg_per_arm=MIN_MARKER_NEG_PER_ARM)
    ]
    print(f"  Track 2 markers to test: {len(track2_markers)}")

    for spec_name, spec_weights in [('ATE', 'IPTW_ATE'), ('ATT', 'IPTW_ATT'), ('noIPTW', None)]:
        results, failed = _run_marker_screen(
            type_df, track2_markers, base_vars, spec_weights,
            _fit_track2_marker, N_JOBS, label=f"T2 {cancer_type} {spec_name}")
        if failed:
            print(f"  Track 2 {spec_name} failures: {len(failed)}. "
                  f"First: {failed[0][0]} -> {failed[0][1]}")
        spec_df = pd.DataFrame(results, columns=TRACK2_RESULT_COLS)
        spec_df = add_track2_fdr_and_labels(spec_df)
        spec_df.to_csv(
            os.path.join(RUN_PATH, f'{cancer_type}_track2_{spec_name}_interaction.csv'),
            index=False)

    # === Track 1: ICI-only generalizability-weighted ===
    ici_only_df = type_df.loc[type_df['PX_on_ICI'] == 1].copy()

    # Generalizability weights for ICI subset: weight = 1 / ps
    # (reweight ICI patients to look like the full eligible population)
    ici_ps = ici_only_df['ICI_prediction'].clip(eps, 1 - eps)
    w_gen = 1.0 / ici_ps
    low_gen, high_gen = np.percentile(w_gen, IPTW_TRUNC_PCT)
    if np.isfinite(low_gen) and np.isfinite(high_gen):
        w_gen_trunc = np.clip(w_gen, low_gen, high_gen)
    else:
        w_gen_trunc = np.ones(len(ici_only_df))
    ici_only_df['IPTW_GEN'] = w_gen_trunc.values

    track1_markers = [
        m for m in biomarker_cols
        if marker_has_ici_only_support(type_df, m, min_pos=MIN_MARKER_POS_ICI_ONLY)
    ]
    print(f"  Track 1 markers to test: {len(track1_markers)}")

    # ICI-only base vars (no cancer type interaction needed for type-specific,
    # include for pan-cancer)
    ici_base_vars = list(base_vars)

    for spec_name, spec_weights in [('weighted', 'IPTW_GEN'), ('unweighted', None)]:
        results, failed = _run_marker_screen(
            ici_only_df, track1_markers, ici_base_vars, spec_weights,
            _fit_track1_marker, N_JOBS, label=f"T1 {cancer_type} {spec_name}")
        if failed:
            print(f"  Track 1 {spec_name} failures: {len(failed)}. "
                  f"First: {failed[0][0]} -> {failed[0][1]}")
        spec_df = pd.DataFrame(results, columns=TRACK1_RESULT_COLS)
        spec_df = add_track1_fdr(spec_df)
        spec_df.to_csv(
            os.path.join(RUN_PATH, f'{cancer_type}_track1_{spec_name}_ICI_only.csv'),
            index=False)

print(f"\n[run_IPTW_analysis] Done. Results in {RUN_PATH}")
