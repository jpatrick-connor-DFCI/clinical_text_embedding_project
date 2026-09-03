"""Stratum-level worker for the within-vs-pan model scripts.

Both ``within_vs_pan_cancer_models`` and ``within_treatment_vs_pan_treatment_models`` fit the
same two models per stratum -- a within-stratum model and a size-matched pan comparator -- and
differ only in which column defines a stratum and what the score columns are called. That
shared fitting work lives here so it can run in a process pool.

Why a module-level function rather than the closures the scripts used before: workers are
spawned, so every task callable and its arguments are pickled. A closure over ``train_df`` is
not picklable, and would ship the full cohort per task even if it were. ``fit_stratum`` takes
only what one stratum needs and is imported by name in the child.

Nothing here touches the checkpoint store. ``RunCheckpoint`` keeps in-memory manifest state and
appends to a single shared CSV, so concurrent writers would interleave rows and diverge from
each other's view of the manifest. Workers therefore return plain frames and the parent does
every ``save_stratum``/``mark_skipped`` call, in completion order, exactly as before.
"""

from __future__ import annotations

import hashlib
import time

import polars as pl

from survival import (fit_predict_external_CoxPH, get_nested_heldout_risk_scores_CoxPH,
                      run_grid_CoxPH_parallel)
from shared.polars_utils import filter_finite_rows


def matched_pan_seed(stratum: str) -> int:
    """Per-stratum sampling seed: base seed offset by a stable hash of the stratum name.

    md5, not the builtin ``hash``, because the builtin is salted per process -- under a worker
    pool that would draw a different matched-pan subsample on every run.
    """
    return 1234 + int(hashlib.md5(stratum.encode('utf-8')).hexdigest()[:8], 16) % 10_000


def _best_hyperparams(val_df):
    """Best (l1_ratio, alpha) row by mean AUC(t), ignoring non-finite scores."""
    return filter_finite_rows(val_df, ['mean_auc(t)']).sort(
        'mean_auc(t)', descending=True
    ).row(0, named=True)


def fit_stratum(stratum, spec, train_df, held_df, need_within, need_matched_pan):
    """Fit one stratum's within model and its size-matched pan comparator.

    Returns a result dict the parent checkpoints. ``need_within``/``need_matched_pan`` let the
    parent ask for only the half that is missing from a resumed checkpoint, so a partially
    completed stratum does not refit the half already on disk.

    Exceptions are caught and returned rather than raised: one diverging stratum should cost
    that stratum, not the pool. The parent records it and carries on.
    """
    result = {'stratum': stratum, 'within': None, 'matched_pan': None,
              'within_skip': None, 'matched_pan_skip': None, 'error': None}
    started = time.time()
    try:
        stratum_col = spec['stratum_col']
        within_score_col = spec['within_score_col']
        pan_score_col = spec['pan_score_col']
        event = spec['event']
        tstop = f"tt_{event}"
        base_cols = spec['base_cols']
        # The matched-pan arm may use a wider unpenalized set than the within arm: a pan model
        # spans strata, so cancer-type dummies carry signal there, while inside a single
        # stratum they are constant. The treatment script passes the same list for both.
        pan_base_cols = spec.get('pan_base_cols', base_cols)
        continuous_vars = spec['continuous_vars']
        embed_cols = spec['embed_cols']
        l1_ratios = spec['l1_ratios']
        alphas_to_test = spec['alphas_to_test']
        max_iter = spec['max_iter']

        sub_df = train_df.filter(pl.col(stratum_col) == stratum)

        if need_within:
            _t0 = time.time()
            _, cur_val, cur_model = run_grid_CoxPH_parallel(
                sub_df, base_cols, continuous_vars, embed_cols,
                l1_ratios, alphas_to_test, event_col=event, tstop_col=tstop, max_iter=max_iter,
            )
            if cur_model is None:
                result['within_skip'] = ('no_converge', int(len(sub_df)))
                return result
            best = _best_hyperparams(cur_val)
            best_l1, best_alpha = float(best['l1_ratio']), float(best['alpha'])

            trained_sub = get_nested_heldout_risk_scores_CoxPH(
                sub_df, base_cols, continuous_vars, embed_cols,
                l1_ratios, alphas_to_test, event_col=event, tstop_col=tstop,
                max_iter=max_iter, backend="threading",
            ).rename({'risk_score': within_score_col}).with_columns(
                pl.lit(stratum).alias('STRATUM'))

            sub_held = held_df.filter(pl.col(stratum_col) == stratum)
            held_sub = pl.DataFrame({
                'DFCI_MRN': sub_held['DFCI_MRN'],
                within_score_col: fit_predict_external_CoxPH(
                    sub_df, sub_held, base_cols, continuous_vars, embed_cols,
                    event_col=event, tstop_col=tstop, l1_ratio=best_l1,
                    alpha=best_alpha, max_iter=max_iter,
                ),
            }).with_columns(pl.lit(stratum).alias('STRATUM'))

            result['within'] = {
                'train': trained_sub, 'held': held_sub,
                'meta': {'n_train': int(len(sub_df)), 'n_held': int(len(held_sub)),
                         'l1': best_l1, 'alpha': best_alpha,
                         'elapsed_s': round(time.time() - _t0, 1)},
            }

        if need_matched_pan:
            _t0 = time.time()
            pan_pool = train_df.filter(pl.col(stratum_col) != stratum)
            n_match = min(len(sub_df), len(pan_pool))
            matched_pan_train = pan_pool.sample(n=n_match, seed=matched_pan_seed(stratum))

            _, matched_val, matched_model = run_grid_CoxPH_parallel(
                matched_pan_train, pan_base_cols, continuous_vars, embed_cols,
                l1_ratios, alphas_to_test, event_col=event, tstop_col=tstop, max_iter=max_iter,
            )
            if matched_model is None:
                result['matched_pan_skip'] = ('no_converge', int(len(sub_df)))
                return result
            matched_best = _best_hyperparams(matched_val)
            matched_l1 = float(matched_best['l1_ratio'])
            matched_alpha = float(matched_best['alpha'])

            trained_matched_pan = get_nested_heldout_risk_scores_CoxPH(
                matched_pan_train, pan_base_cols, continuous_vars, embed_cols,
                l1_ratios, alphas_to_test, event_col=event, tstop_col=tstop,
                max_iter=max_iter, backend="threading",
            ).rename({'risk_score': pan_score_col}).with_columns(
                pl.lit(stratum).alias('STRATUM'))

            matched_pan_held = pl.DataFrame({
                'DFCI_MRN': held_df['DFCI_MRN'],
                pan_score_col: fit_predict_external_CoxPH(
                    matched_pan_train, held_df, pan_base_cols, continuous_vars, embed_cols,
                    event_col=event, tstop_col=tstop, l1_ratio=matched_l1,
                    alpha=matched_alpha, max_iter=max_iter,
                ),
            }).with_columns(pl.lit(stratum).alias('STRATUM'))

            result['matched_pan'] = {
                'train': trained_matched_pan, 'held': matched_pan_held,
                'meta': {'n_train': int(len(sub_df)),
                         'l1': matched_l1, 'alpha': matched_alpha,
                         'elapsed_s': round(time.time() - _t0, 1)},
            }
    except Exception as exc:  # noqa: BLE001 - reported to the parent, which records and continues
        result['error'] = f"{type(exc).__name__}: {exc}"
    result['elapsed_s'] = round(time.time() - started, 1)
    return result
