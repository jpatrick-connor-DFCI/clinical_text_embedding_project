"""Render Figure 2 text vs base manuscript panels.

It loads prepared CSVs via
`_figure_utils.load_figure_data` and writes panel PNGs via `save_panel`.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

# %% imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from _figure_utils import apply_style, load_figure_data, save_panel, MODEL_COLORS

apply_style()

# %% fig2a
metrics = load_figure_data('fig2_full_cohort_metrics.csv')
if metrics.empty:
    print('fig2_full_cohort_metrics.csv is empty; skipping panel A')
else:
    scheme_markers = {'death_met': 'o', 'icd3_post': 's', 'icd4_post': '^', 'phecode_post': 'D'}
    scheme_colors = {'death_met': '#D62728', 'icd3_post': '#1F77B4',
                     'icd4_post': '#2CA02C', 'phecode_post': '#9467BD'}
    fig, ax = plt.subplots(figsize=(5, 5))
    for scheme, g in metrics.groupby('scheme'):
        ax.scatter(g['base_cindex'], g['text_cindex'],
                   marker=scheme_markers.get(scheme, 'o'),
                   c=scheme_colors.get(scheme, '#555'),
                   alpha=0.6, s=22, edgecolors='none', label=f'{scheme} (n={len(g)})')
    lim = [metrics[['base_cindex', 'text_cindex']].min().min() - 0.02,
           metrics[['base_cindex', 'text_cindex']].max().max() + 0.02]
    ax.plot(lim, lim, ls='--', c='#888', lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Base model C-index')
    ax.set_ylabel('Text model C-index')
    ax.set_title('Text vs base on held-out test (all events)')
    ax.legend(loc='lower right', fontsize=8)
    save_panel(fig, 'fig2a')
    plt.close(fig)

# %% fig2b
forest = load_figure_data('fig2_bootstrap_deltas.csv')
if forest.empty:
    print('fig2_bootstrap_deltas.csv is empty; skipping panel B')
else:
    forest = forest.sort_values('delta').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    y = np.arange(len(forest))
    # Percentile bootstrap CI can produce lo > delta or hi < delta for skewed
    # distributions; clip to keep matplotlib's errorbar from raising.
    xerr_lo = np.maximum(0, forest['delta'] - forest['lo'])
    xerr_hi = np.maximum(0, forest['hi'] - forest['delta'])
    ax.errorbar(forest['delta'], y, xerr=[xerr_lo, xerr_hi],
                fmt='o', color='#D62728', ecolor='#888', capsize=3)
    ax.axvline(0, color='#333', lw=0.8, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels([f'{r.event}  (n={int(r.n):,})' for _, r in forest.iterrows()])
    ax.set_xlabel('Δ C-index (text − base)')
    ax.set_title('Mortality + metastasis endpoints')
    save_panel(fig, 'fig2b')
    plt.close(fig)

# %% fig2c
km = load_figure_data('fig2_km_death_data.csv')
if km.empty:
    print('fig2_km_death_data.csv is empty; skipping panel C')
else:
    tertile_colors = {'low': '#2CA02C', 'mid': '#FF7F0E', 'high': '#D62728'}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    kmf = KaplanMeierFitter()
    for t in ['low', 'mid', 'high']:
        sub = km[km['text_tertile'] == t]
        if sub.empty:
            continue
        kmf.fit(sub['tt_death'] / 30.44, sub['death'], label=f'text {t} (n={len(sub):,})')
        kmf.plot_survival_function(ax=ax, ci_show=False, color=tertile_colors[t], lw=2)
    for t in ['low', 'mid', 'high']:
        sub = km[km['base_tertile'] == t]
        if sub.empty:
            continue
        kmf.fit(sub['tt_death'] / 30.44, sub['death'], label=f'base {t} (n={len(sub):,})')
        kmf.plot_survival_function(ax=ax, ci_show=False, color=tertile_colors[t], lw=1.2, linestyle='--')
    try:
        lr_text = multivariate_logrank_test(km['tt_death'], km['text_tertile'], km['death'])
        lr_base = multivariate_logrank_test(km['tt_death'], km['base_tertile'], km['death'])
        ax.text(0.02, 0.05,
                f'text logrank p={lr_text.p_value:.1e}\nbase logrank p={lr_base.p_value:.1e}',
                transform=ax.transAxes, fontsize=8)
    except Exception as e:
        print(f'logrank test failed: {e}')
    ax.set_xlim(0, 60)
    ax.set_xlabel('Months from first treatment')
    ax.set_ylabel('Overall survival')
    ax.set_title('Mortality by risk-score tertile (text solid, base dashed)')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    save_panel(fig, 'fig2c')
    plt.close(fig)

# %% fig2d
td = load_figure_data('fig2_td_auc.csv')
if td.empty:
    print('fig2_td_auc.csv is empty; skipping panel D')
else:
    events_to_plot = list(td['event'].unique())
    fig, axes = plt.subplots(1, len(events_to_plot),
                             figsize=(4 * len(events_to_plot), 4), sharey=True, squeeze=False)
    for ax, ev in zip(axes[0], events_to_plot):
        sub = td[td['event'] == ev].sort_values('time_months')
        ax.plot(sub['time_months'], sub['auc_text'], '-o', color=MODEL_COLORS['text'], label='text')
        ax.plot(sub['time_months'], sub['auc_base'], '-o', color=MODEL_COLORS['base'], label='base')
        ax.set_xlabel('Months from first treatment')
        ax.set_title(ev)
        ax.set_ylim(0.4, 0.95)
    axes[0, 0].set_ylabel('Time-dependent AUC')
    axes[0, -1].legend(loc='lower left')
    fig.tight_layout()
    save_panel(fig, 'fig2d')
    plt.close(fig)
