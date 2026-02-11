import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter
random.seed(42)  # set seed for reproducibility

# ---------------------------------------
# Summary classifier
# ---------------------------------------
def classify(row):
    # Predictive biomarker: interaction significant AND direction
    if row['significant_predictive']:
        if row['HR_markerxIO'] < 1:
            return "predictive_IO_benefit"
        else:
            return "predictive_IO_harm"

    # Not predictive, but has IO-specific effect
    if row['significant_in_IO'] and not row['significant_prognostic_nonIO']:
        return "IO_specific_effect"

    # Prognostic only (acts in controls but not IO)
    if row['significant_prognostic_nonIO'] and not row['significant_in_IO']:
        return "prognostic_nonIO"

    # no clear signal
    return "no_signal"

# ---------------------------------------
# Summary classifier (same logic)
# ---------------------------------------
def classify_noiptw(row):
    if row['significant_predictive']:
        return "predictive_IO_benefit" if row['HR_markerxIO'] < 1 else "predictive_IO_harm"
    if row['significant_in_IO'] and not row['significant_prognostic_nonIO']:
        return "IO_specific_effect"
    if row['significant_prognostic_nonIO'] and not row['significant_in_IO']:
        return "prognostic_nonIO"
    return "no_signal"


# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/processed_datasets/')
MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')

interaction_IO_df = pd.read_csv(os.path.join(MARKER_PATH, 'IPTW_IO_interaction_runs_df.csv'))

required_vars = ['DFCI_MRN', 'tt_death', 'death']
panel_cols = [col for col in interaction_IO_df.columns if 'PANEL_VERSION' in col]
cancer_type_cols = [col for col in interaction_IO_df.columns if col.startswith('CANCER_TYPE_')]
biomarker_cols = [col for col in interaction_IO_df.columns if col not in (required_vars + panel_cols + cancer_type_cols + ['PX_on_IO', 'IO_prediction'])]

cols_to_test = ['pan_cancer'] + [col.replace('CANCER_TYPE_', '') for col in interaction_IO_df.columns if (col.startswith('CANCER_TYPE_')) and ('OTHER' not in col)]

for cancer_type in cols_to_test:

    if cancer_type == 'pan_cancer':
        type_specific_interaction_IO_df = interaction_IO_df.copy()
        base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART'] + panel_cols + cancer_type_cols
    else:
        type_specific_interaction_IO_df = interaction_IO_df.loc[interaction_IO_df[f'CANCER_TYPE_{cancer_type}']].copy()
        base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART'] + panel_cols

    # ---------------------------------------
    # Common support trimming
    # ---------------------------------------
    eps = 1e-6
    ps_raw = type_specific_interaction_IO_df['IO_prediction'].clip(eps, 1 - eps)
    
    ps_t = ps_raw[type_specific_interaction_IO_df['PX_on_IO'] == 1]
    ps_c = ps_raw[type_specific_interaction_IO_df['PX_on_IO'] == 0]
    lower, upper = max(ps_t.min(), ps_c.min()), min(ps_t.max(), ps_c.max())
    
    type_specific_interaction_IO_df = type_specific_interaction_IO_df[(ps_raw >= lower) & (ps_raw <= upper)].copy()
    ps = type_specific_interaction_IO_df['IO_prediction'].clip(eps, 1 - eps)
    
    # ---------------------------------------
    # Stabilized ATE IPTW with truncation
    # ---------------------------------------
    p_treated = type_specific_interaction_IO_df['PX_on_IO'].mean()
    p_control = 1 - p_treated
    
    w = np.where(type_specific_interaction_IO_df['PX_on_IO']==1, p_treated/ps, p_control/(1-ps))
    low, high = np.percentile(w, [1,99])
    w_trunc = np.clip(w, low, high)
    type_specific_interaction_IO_df['IPTW'] = w_trunc
    
    # ---------------------------------------
    # Cox IPTW marker screening
    # ---------------------------------------
    markers_to_test = [m for m in biomarker_cols if (type_specific_interaction_IO_df[m].sum()/len(type_specific_interaction_IO_df)) >= 0.01]
    
    results = []
    for marker in tqdm(markers_to_test):
        try:
            df_fit = type_specific_interaction_IO_df[['tt_death','death','PX_on_IO'] + base_vars + [marker,'IPTW']].copy()
            df_fit = df_fit.dropna().copy()
    
            mx = f"{marker}_x_IO"
            df_fit[mx] = df_fit['PX_on_IO'] * df_fit[marker]
    
            cph = CoxPHFitter()
            cph.fit(df_fit, duration_col='tt_death', event_col='death',
                    weights_col='IPTW', robust=True)
    
            summ = cph.summary.reset_index()
            V = cph.variance_matrix_
            b = cph.params_
    
            # main + interaction
            beta_m = float(b[marker])
            se_m = float(np.sqrt(V.loc[marker, marker]))
            p_m = float(summ.loc[summ['covariate']==marker, 'p'].values[0])
    
            beta_mx = float(b[mx])
            se_mx = float(np.sqrt(V.loc[mx, mx]))
            p_mx = float(summ.loc[summ['covariate']==mx,'p'].values[0])
    
            # marker effect in NON-IO
            hr_nonio = np.exp(beta_m)
            ci_nonio = (np.exp(beta_m - 1.96*se_m), np.exp(beta_m + 1.96*se_m))
    
            # marker effect in IO = beta_m + beta_mx
            cov_m_mx = float(V.loc[marker, mx])
            se_io = np.sqrt(se_m**2 + se_mx**2 + 2*cov_m_mx)
            beta_io = beta_m + beta_mx
            hr_io = np.exp(beta_io)
            ci_io = (np.exp(beta_io - 1.96*se_io), np.exp(beta_io + 1.96*se_io))
    
            # p-value for marker effect IN IO patients
            z_io = beta_io / se_io
            p_io = 2 * (1 - stats.norm.cdf(abs(z_io)))
    
            # optional: treatment main effect at marker=0
            beta_IO0 = float(b['PX_on_IO']) if 'PX_on_IO' in b.index else np.nan
            p_IO0 = float(summ.loc[summ['covariate']=='PX_on_IO','p'].values[0]) if 'PX_on_IO' in summ['covariate'].values else np.nan
    
            results.append({
                "marker": marker,
    
                # Predictive (IO-specific modulation)
                "beta_markerxIO": beta_mx,
                "HR_markerxIO": np.exp(beta_mx),
                "p_markerxIO": p_mx,
    
                # Effect in IO
                "beta_marker_IO": beta_io,
                "HR_marker_IO": hr_io,
                "CI95_marker_IO_low": ci_io[0],
                "CI95_marker_IO_high": ci_io[1],
                "p_marker_IO": p_io,
    
                # Effect in non-IO
                "beta_marker_nonIO": beta_m,
                "HR_marker_nonIO": hr_nonio,
                "CI95_marker_nonIO_low": ci_nonio[0],
                "CI95_marker_nonIO_high": ci_nonio[1],
                "p_marker_nonIO": p_m,
    
                # IO effect at marker-negative baseline
                "beta_IO_at_marker0": beta_IO0,
                "p_IO_at_marker0": p_IO0
            })
    
        except Exception as e:
            continue
    
    results_df = pd.DataFrame(results)
    
    # ---------------------------------------
    # FDR corrections
    # ---------------------------------------
    # Predictive biomarkers = significant interaction
    rej_int, fdr_int, _, _ = multipletests(results_df['p_markerxIO'], alpha=0.05, method='fdr_bh')
    results_df['FDR_markerxIO'] = fdr_int
    results_df['significant_predictive'] = rej_int
    
    # Marker effect in IO patients
    rej_io, fdr_io, _, _ = multipletests(results_df['p_marker_IO'], alpha=0.05, method='fdr_bh')
    results_df['FDR_marker_IO'] = fdr_io
    results_df['significant_in_IO'] = rej_io
    
    # Marker effect in NON-IO
    rej_nonio, fdr_nonio, _, _ = multipletests(results_df['p_marker_nonIO'], alpha=0.05, method='fdr_bh')
    results_df['FDR_marker_nonIO'] = fdr_nonio
    results_df['significant_prognostic_nonIO'] = rej_nonio
    
    results_df['classifier'] = results_df.apply(classify, axis=1)
    results_df.to_csv(os.path.join(MARKER_PATH, f'IPTW_runs/{cancer_type}_IPTW_IO_predictive_markers.csv'), index=False)

    # ===============================================
    # ==  NO IPTW VERSION OF MARKER SCREENING  ==
    # ===============================================
    results_noiptw = []
    for marker in tqdm(markers_to_test):
        try:
            # same columns as before, but no IPTW
            df_fit = type_specific_interaction_IO_df[['tt_death','death','PX_on_IO'] 
                                       + base_vars + [marker]].dropna().copy()
    
            # interaction term
            mx = f"{marker}_x_IO"
            df_fit[mx] = df_fit['PX_on_IO'] * df_fit[marker]
    
            cph = CoxPHFitter()
            cph.fit(df_fit,
                    duration_col='tt_death',
                    event_col='death',
                    robust=True)  # ✅ no weights_col
    
            summ = cph.summary.reset_index()
            V = cph.variance_matrix_
            b = cph.params_
    
            # main + interaction
            beta_m = float(b[marker])
            se_m = float(np.sqrt(V.loc[marker, marker]))
            p_m = float(summ.loc[summ['covariate']==marker, 'p'].values[0])
    
            beta_mx = float(b[mx])
            se_mx = float(np.sqrt(V.loc[mx, mx]))
            p_mx = float(summ.loc[summ['covariate']==mx,'p'].values[0])
    
            # effect in non-IO
            hr_nonio = np.exp(beta_m)
            ci_nonio = (np.exp(beta_m - 1.96*se_m), np.exp(beta_m + 1.96*se_m))
    
            # effect in IO
            cov_m_mx = float(V.loc[marker, mx])
            se_io = np.sqrt(se_m**2 + se_mx**2 + 2*cov_m_mx)
            beta_io = beta_m + beta_mx
            hr_io = np.exp(beta_io)
            ci_io = (np.exp(beta_io - 1.96*se_io), np.exp(beta_io + 1.96*se_io))
    
            z_io = beta_io / se_io
            p_io = 2 * (1 - stats.norm.cdf(abs(z_io)))
    
            # baseline IO effect at marker = 0
            beta_IO0 = float(b['PX_on_IO']) if 'PX_on_IO' in b.index else np.nan
            p_IO0 = float(summ.loc[summ['covariate']=='PX_on_IO','p'].values[0]) if 'PX_on_IO' in summ['covariate'].values else np.nan
    
            results_noiptw.append({
                "marker": marker,
                "beta_markerxIO": beta_mx,
                "HR_markerxIO": np.exp(beta_mx),
                "p_markerxIO": p_mx,
    
                "beta_marker_IO": beta_io,
                "HR_marker_IO": hr_io,
                "CI95_marker_IO_low": ci_io[0],
                "CI95_marker_IO_high": ci_io[1],
                "p_marker_IO": p_io,
    
                "beta_marker_nonIO": beta_m,
                "HR_marker_nonIO": hr_nonio,
                "CI95_marker_nonIO_low": ci_nonio[0],
                "CI95_marker_nonIO_high": ci_nonio[1],
                "p_marker_nonIO": p_m,
    
                "beta_IO_at_marker0": beta_IO0,
                "p_IO_at_marker0": p_IO0
            })
    
        except Exception:
            continue
    
    results_noiptw_df = pd.DataFrame(results_noiptw)
    
    # ---------------------------------------
    # FDR corrections (same as IPTW version)
    # ---------------------------------------
    rej_int, fdr_int, _, _ = multipletests(results_noiptw_df['p_markerxIO'], alpha=0.05, method='fdr_bh')
    results_noiptw_df['FDR_markerxIO'] = fdr_int
    results_noiptw_df['significant_predictive'] = rej_int
    
    rej_io, fdr_io, _, _ = multipletests(results_noiptw_df['p_marker_IO'], alpha=0.05, method='fdr_bh')
    results_noiptw_df['FDR_marker_IO'] = fdr_io
    results_noiptw_df['significant_in_IO'] = rej_io
    
    rej_nonio, fdr_nonio, _, _ = multipletests(results_noiptw_df['p_marker_nonIO'], alpha=0.05, method='fdr_bh')
    results_noiptw_df['FDR_marker_nonIO'] = fdr_nonio
    results_noiptw_df['significant_prognostic_nonIO'] = rej_nonio
    
    results_noiptw_df['classifier'] = results_noiptw_df.apply(classify_noiptw, axis=1)
    results_noiptw_df.to_csv(os.path.join(MARKER_PATH, f'IPTW_runs/{cancer_type}_noIPTW_IO_predictive_markers.csv'), index=False)
    print("Saved no-IPTW marker results ✅")