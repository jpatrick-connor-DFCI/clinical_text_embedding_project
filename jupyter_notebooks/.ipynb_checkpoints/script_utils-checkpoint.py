import os 
import re
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from itertools import product
from tqdm import tqdm
import time

from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored, integrated_brier_score

from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scipy.stats import sem

def find_continuous_records_to_analyze(notes_meta, survival_df, gap_threshold=3*365, min_ubound=-365, max_note_window=None):

    # consider only notes up to and including treatment window or up to max_note_window
    if max_note_window is not None:
        notes_within_window = notes_meta.loc[notes_meta['NOTE_TIME_REL_IO_START'] < max_note_window]
    else:
        indices_to_include = list()
        for mrn_val in notes_meta['DFCI_MRN'].unique():
            tstop = surv_df.loc[surv_df['PATIENT_ID'] == mrn_val, 'tstop'].values[0]
            indices_to_include += notes_meta.loc[(notes_meta['DFCI_MRN'] == mrn_val) & 
                                                 (notes_meta['NOTE_TIME_REL_IO_START'] <= tstop)].index.tolist()
        notes_within_window = notes_meta.loc[list(indices_to_include)]

    # find "continuous records" (e.g. sets such that no consecutive record has a gap greater
    # than a given threshold) and keep only a continuous record that is wihtin one year
    # of treatment start
    mrn_window_dict = dict()
    indices_to_include = list()
    for mrn_idx in notes_within_window['DFCI_MRN'].unique().tolist():

        mrn_notes = notes_within_window.loc[notes_within_window['DFCI_MRN'] == mrn_idx].sort_values(by='NOTE_TIME_REL_IO_START')
        note_times = mrn_notes['NOTE_TIME_REL_IO_START'].values.tolist()

        idx_series = []
        lbound = note_times[0]
        for i in range(1, len(note_times)):
            if (note_times[i] - note_times[i-1]) >= gap_threshold:
                idx_series.append((lbound, note_times[i-1]))
                lbound = note_times[i]

        if len(idx_series) == 0:
            idx_series.append((lbound, note_times[len(note_times)-1]))

        for lbound, ubound in idx_series:
            if (ubound >= -365):
                cur_mrn_window = (lbound, ubound)

        indices_to_include += mrn_notes.loc[(mrn_notes['NOTE_TIME_REL_IO_START'] >= lbound) & 
                                            (mrn_notes['NOTE_TIME_REL_IO_START'] <= ubound)].index.tolist()

    notes_within_window = notes_within_window.loc[list(indices_to_include)]
    
    return notes_within_window

def pool_embedding_series(meta_df, embedding_array, note_types, pool_fx=None, decay_param=None, year_adj_cols=['Imaging', 'Pathology']):
    _, embed_size = embedding_array.shape
    unique_mrns = meta_df['DFCI_MRN'].unique()
    embedding_results = [np.zeros((len(unique_mrns), embed_size)) for _ in range(len(note_types))]

    meta_df['NOTE_YEAR'] = meta_df['NOTE_DATETIME'].apply(lambda x : int(re.split('-', x)[0]))
    year_adjustment_dict = {note_type : np.zeros((len(unique_mrns), 1)) for note_type in note_types if note_type in year_adj_cols}

    df_columns = ['PATIENT_ID'] + ['PERCENT_' + note_type.upper() + '_NOTES_PRE_2015' for note_type in note_types if note_type in year_adj_cols]

    for note_type_idx, note_type in enumerate(note_types):
        note_type_meta = meta_df.loc[meta_df['NOTE_TYPE'] == note_type]
        df_columns += [note_type.upper() + f'_EMBEDDING_{i}' for i in range(embed_size)]

        for mrn_idx, mrn_val in enumerate(unique_mrns):

            meta_in_param = note_type_meta.loc[note_type_meta['DFCI_MRN'] == mrn_val]

            if len(meta_in_param) > 0:
                if pool_fx[note_type] == 'recent':
                    embed_idx = meta_in_param.iloc[np.argmax(meta_in_param['NOTE_TIME_REL_IO_START'].values)]['EMBEDDING_INDEX']
                    embedding_results[note_type_idx][mrn_idx,:] = embedding_array[embed_idx,:]
                elif pool_fx[note_type] == 'time_decay_mean':
                    t_vals, embed_idx = -meta_in_param['NOTE_TIME_REL_IO_START'], meta_in_param['EMBEDDING_INDEX']
                    embed_vals = embedding_array[embed_idx.values.tolist(),:]
                    embedding_results[note_type_idx][mrn_idx,:] = np.mean(np.exp(-decay_param * t_vals.values.reshape(-1,1)) * embed_vals, axis=0)
                else:
                    embedding_results[note_type_idx][mrn_idx,:] = np.mean(embedding_array[meta_in_param['EMBEDDING_INDEX'].values.tolist(),:], axis=0)
            else:
                embedding_results[note_type_idx][mrn_idx,:] = np.asarray([np.nan for _ in range(embed_size)])

            if note_type in year_adj_cols:
                if len(meta_in_param) > 0:
                    num_pre_2015 = len(meta_in_param.loc[meta_in_param['NOTE_YEAR'] < 2015])
                    year_adjustment_dict[note_type][mrn_idx] = (num_pre_2015 / len(meta_in_param))
                else:
                    year_adjustment_dict[note_type][mrn_idx] = np.nan

    return pd.DataFrame(np.concat([unique_mrns.reshape(-1,1)] + [np.asarray(year_adjustment_dict[key]) for key in year_adjustment_dict.keys()] + 
                                  embedding_results, axis=1), columns=df_columns)
    
    
def generate_survival_embedding_df(notes_meta, survival_df, embedding_array, note_type, pool_fx=None, decay_param=None):
    
    notes_to_include = find_continuous_records_to_analyze(notes_meta, survival_df, max_note_window=0)
    pooled_embedding_df = pool_embedding_series(notes_meta, embedding_array, note_type, pool_fx, decay_param)

    return survival_df.merge(pooled_embedding_df, on='PATIENT_ID')
    
def run_grid_CoxPH(df, event_map, base_cols, continuous_vars, embed_cols, l1_ratios, alphas_to_test, max_iter=1000, n_splits=5):
    
    # specify covariates and event data
    Xt = df[base_cols + embed_cols]
    y = np.asarray(list(zip(df['event'].map(event_map), df['tstop'])),
               dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    
    # define the cross-validation scheme
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=1234)
    
    # define prediction window to compute the time-dependent metrics
    lower, upper = np.percentile(y['Survival_in_days'], [10,90])
    eval_times = np.arange(lower, upper + 1)

    results_list = []
    
    for (l1_ratio, alpha_to_test) in tqdm(list(product(l1_ratios, alphas_to_test))):

        c_index_vals = []
        mean_auc_t_vals = []
        ibs_vals=[]

        for train_idx, test_idx in cv.split(Xt, y):
            X_train, y_train = Xt.iloc[train_idx], y[train_idx]
            X_test, y_test = Xt.iloc[test_idx], y[test_idx]

            # scale datasets
            scaler = StandardScaler().fit(X_train[continuous_vars])
            X_train[continuous_vars] = scaler.transform(X_train[continuous_vars])
            X_test[continuous_vars] = scaler.transform(X_test[continuous_vars])

            # train 
            cox_model = CoxnetSurvivalAnalysis(alphas=[alpha_to_test], l1_ratio=l1_ratio, fit_baseline_model=True, max_iter=max_iter, 
                                               penalty_factor=[0 for _ in range(len(base_cols))] + [1 for _ in range(len(embed_cols))])
            cox_model.fit(X_train, y_train)

            chf_funcs = cox_model.predict_cumulative_hazard_function(X_test, return_array=False)
            risk_scores = np.vstack([chf(eval_times) for chf in chf_funcs])

            auc_t, mean_auc_t = cumulative_dynamic_auc(y_train, y_test, risk_scores, eval_times)
            ibs = integrated_brier_score(y_train, y_test, risk_scores, eval_times)
            c_index = cox_model.score(X_test, y_test)

            c_index_vals.append(c_index)
            mean_auc_t_vals.append(mean_auc_t)
            ibs_vals.append(ibs)

        results_list.append([l1_ratio, alpha_to_test, np.mean(c_index_vals), sem(c_index_vals), 
                             np.mean(mean_auc_t_vals), sem(mean_auc_t_vals), np.mean(ibs_vals), sem(ibs_vals)])

    return pd.DataFrame(results_list, columns=['l1_ratio', 'alpha', 'mean_c_index', 'sem_c_index', 'mean_auc(t)', 'sem_auc(t)', 'mean_ibs', 'sem_ibs'])

def run_grid_RSF(df, event_map, base_cols, continuous_vars, embed_cols, n_estimators, max_depths, n_jobs=-1, n_splits=5):

    # specify covariates and event data/
    Xt = df[base_cols + embed_cols]
    y = np.asarray(list(zip(df['event'].map(event_map), df['tstop'])),
               dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    
    # define the cross-validation scheme
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=1234)
    
    # define prediction window to compute the time-dependent metrics
    lower, upper = np.percentile(y['Survival_in_days'], [10,90])
    eval_times = np.arange(lower, upper + 1)

    results_list = []
    
    for (n_estimator, max_depth) in tqdm(list(product(n_estimators, max_depths))):

        c_index_vals = []
        mean_auc_t_vals = []
        ibs_vals=[]

        for train_idx, test_idx in cv.split(Xt, y):
            X_train, y_train = Xt.iloc[train_idx], y[train_idx]
            X_test, y_test = Xt.iloc[test_idx], y[test_idx]

            # scale datasets
            scaler = StandardScaler().fit(X_train[continuous_vars])
            X_train[continuous_vars] = scaler.transform(X_train[continuous_vars])
            X_test[continuous_vars] = scaler.transform(X_test[continuous_vars])

            # train 
            rsf_model = RandomSurvivalForest(n_estimators=n_estimator, max_depth=max_depth, n_jobs=n_jobs, random_state=1234)
            rsf_model.fit(X_train, y_train)

            chf_funcs = rsf_model.predict_cumulative_hazard_function(X_test, return_array=False)
            risk_scores = np.vstack([chf(eval_times) for chf in chf_funcs])

            auc_t, mean_auc_t = cumulative_dynamic_auc(y_train, y_test, risk_scores, eval_times)
            ibs = integrated_brier_score(y_train, y_test, risk_scores, eval_times)
            c_index = rsf_model.score(X_test, y_test)

            c_index_vals.append(c_index)
            mean_auc_t_vals.append(mean_auc_t)
            ibs_vals.append(ibs)

        results_list.append([n_estimator, max_depth, np.mean(c_index_vals), sem(c_index_vals), 
                             np.mean(mean_auc_t_vals), sem(mean_auc_t_vals), np.mean(ibs_vals), sem(ibs_vals)])

    return pd.DataFrame(results_list, columns=['n_estimators', 'max_depth', 'mean_c_index', 'sem_c_index', 'mean_auc(t)', 'sem_auc(t)', 'mean_ibs', 'sem_ibs'])

class EarlyStoppingMonitor:
    def __init__(self, window_size, max_iter_without_improvement):
        self.window_size = window_size
        self.max_iter_without_improvement = max_iter_without_improvement
        self._best_step = -1

    def __call__(self, iteration, estimator, args):
        # continue training for first self.window_size iterations
        if iteration < self.window_size:
            return False

        # compute average improvement in last self.window_size iterations.
        # oob_improvement_ is the different in negative log partial likelihood
        # between the previous and current iteration.
        start = iteration - self.window_size + 1
        end = iteration + 1
        improvement = np.mean(estimator.oob_improvement_[start:end])

        if improvement > 1e-6:
            self._best_step = iteration
            return False  # continue fitting

        # stop fitting if there was no improvement
        # in last max_iter_without_improvement iterations
        diff = iteration - self._best_step
        return diff >= self.max_iter_without_improvement

def run_grid_GBM(df, event_map, base_cols, continuous_vars, embed_cols, n_estimators, max_depths, learning_rates, subsamples, dropout_rates, n_splits=5):

    # specify covariates and event data/
    Xt = df[base_cols + embed_cols]
    y = np.asarray(list(zip(df['event'].map(event_map), df['tstop'])),
               dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    
    # define the cross-validation scheme
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=1234)
    
    # setup early stopping monitor
    monitor = EarlyStoppingMonitor(10,5)
    
    # define prediction window to compute the time-dependent metrics
    lower, upper = np.percentile(y['Survival_in_days'], [10,90])
    eval_times = np.arange(lower, upper + 1)

    results_list = []
    
    for (n_estimator, max_depth, learning_rate, subsample, dropout_rate) in tqdm(list(product(n_estimators, max_depths, learning_rates, subsamples, dropout_rates))):

        c_index_vals = []
        mean_auc_t_vals = []
        ibs_vals=[]

        for train_idx, test_idx in cv.split(Xt, y):
            X_train, y_train = Xt.iloc[train_idx], y[train_idx]
            X_test, y_test = Xt.iloc[test_idx], y[test_idx]

            # scale datasets
            scaler = StandardScaler().fit(X_train[continuous_vars])
            X_train[continuous_vars] = scaler.transform(X_train[continuous_vars])
            X_test[continuous_vars] = scaler.transform(X_test[continuous_vars])

            # train 
            gbm_model = GradientBoostingSurvivalAnalysis(n_estimators=n_estimator, learning_rate=learning_rate, subsample=subsample,
                                                         dropout_rate=dropout_rate, random_state=1234)
            gbm_model.fit(X_train, y_train, monitor=monitor)

            chf_funcs = gbm_model.predict_cumulative_hazard_function(X_test, return_array=False)
            risk_scores = np.vstack([chf(eval_times) for chf in chf_funcs])

            auc_t, mean_auc_t = cumulative_dynamic_auc(y_train, y_test, risk_scores, eval_times)
            ibs = integrated_brier_score(y_train, y_test, risk_scores, eval_times)
            c_index = gbm_model.score(X_test, y_test)

            c_index_vals.append(c_index)
            mean_auc_t_vals.append(mean_auc_t)
            ibs_vals.append(ibs)
        
        results_list.append([n_estimator, max_depth, learning_rate, subsample, dropout_rate, np.mean(c_index_vals), sem(c_index_vals), 
                             np.mean(mean_auc_t_vals), sem(mean_auc_t_vals), np.mean(ibs_vals), sem(ibs_vals)])

    return pd.DataFrame(results_list, columns=['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'dropout_rate', 
                                               'mean_c_index', 'sem_c_index', 'mean_auc(t)', 'sem_auc(t)', 'mean_ibs', 'sem_ibs'])


def train_CoxPH_model(df, event_map, base_cols, continuous_vars, embed_cols, l1_ratio, alpha, model_path=None, max_iter=1000):
    
    # specify covariates and event data
    Xt = df[base_cols + embed_cols]
    y = np.asarray(list(zip(df['event'].map(event_map), df['tstop'])),
               dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    
    # define prediction window to compute the time-dependent metrics
    lower, upper = np.percentile(y['Survival_in_days'], [10,90])
    eval_times = np.arange(lower, upper + 1)

    X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2, random_state=1234)

    # scale datasets
    scaler = StandardScaler().fit(X_train[continuous_vars])
    X_train[continuous_vars] = scaler.transform(X_train[continuous_vars])
    X_test[continuous_vars] = scaler.transform(X_test[continuous_vars])

    # train 
    cox_model = CoxnetSurvivalAnalysis(alphas=[alpha], l1_ratio=l1_ratio, fit_baseline_model=True, max_iter=max_iter, 
                                       penalty_factor=[0 for _ in range(len(base_cols))] + [1 for _ in range(len(embed_cols))])
    cox_model.fit(X_train, y_train)
    
    if model_path:
        pickle.dump(cox_model, open(model_path, 'wb'))

    return cox_model


def train_RSF_model(df, event_map, base_cols, continuous_vars, embed_cols, n_estimator, max_depth, model_path=None, n_jobs=-1):

    # specify covariates and event data
    Xt = df[base_cols + embed_cols]
    y = np.asarray(list(zip(df['event'].map(event_map), df['tstop'])),
               dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    
    # define prediction window to compute the time-dependent metrics
    lower, upper = np.percentile(y['Survival_in_days'], [10,90])
    eval_times = np.arange(lower, upper + 1)

    X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2, random_state=1234)

    # scale datasets
    scaler = StandardScaler().fit(X_train[continuous_vars])
    X_train[continuous_vars] = scaler.transform(X_train[continuous_vars])
    X_test[continuous_vars] = scaler.transform(X_test[continuous_vars])

    # train 
    rsf_model = RandomSurvivalForest(n_estimators=n_estimator, max_depth=max_depth, n_jobs=n_jobs, random_state=1234)
    rsf_model.fit(X_train, y_train)
    
    if model_path:
        pickle.dump(rsf_model, open(model_path, 'wb'))
    
    return rsf_model

def train_GBM_model(df, event_map, base_cols, continuous_vars, embed_cols, n_estimator, max_depth, learning_rate, subsample, dropout_rate, model_path=None):

    # specify covariates and event data
    Xt = df[base_cols + embed_cols]
    y = np.asarray(list(zip(df['event'].map(event_map), df['tstop'])),
               dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    
    # define prediction window to compute the time-dependent metrics
    lower, upper = np.percentile(y['Survival_in_days'], [10,90])
    eval_times = np.arange(lower, upper + 1)
    
    # setup early stopping monitor
    monitor = EarlyStoppingMonitor(10,5)
    
    X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2, random_state=1234)

    # scale datasets
    scaler = StandardScaler().fit(X_train[continuous_vars])
    X_train[continuous_vars] = scaler.transform(X_train[continuous_vars])
    X_test[continuous_vars] = scaler.transform(X_test[continuous_vars])

    # train 
    gbm_model = GradientBoostingSurvivalAnalysis(n_estimators=n_estimator, learning_rate=learning_rate, subsample=subsample,
                                                 dropout_rate=dropout_rate, random_state=1234)
    gbm_model.fit(X_train, y_train, monitor=monitor)

    if model_path:
        pickle.dump(gbm_model, open(model_path, 'wb'))
    
    return gbm_model