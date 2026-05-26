"""Render Figure 4 mortality trajectories manuscript panels.

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
from matplotlib.colors import to_rgba_array

from _figure_utils import (
    apply_style, load_figure_data, save_panel, CLUSTER_COLORS,
)

apply_style()

# %% fig4a
traj = load_figure_data('fig4_trajectories_heatmap.csv')
if traj.empty:
    print('fig4_trajectories_heatmap.csv is empty; skipping panel A')
else:
    month_cols = [c for c in traj.columns if c not in ('DFCI_MRN', 'cluster')]
    if not month_cols:
        print('No month columns in trajectory CSV; skipping panel A')
    else:
        # Rows are already sorted by (cluster, within-cluster mean risk) in prep.
        X = traj[month_cols].values
        cluster_sidebar = traj['cluster'].values
        fig, (ax_side, ax_main) = plt.subplots(
            1, 2, figsize=(8, 5),
            gridspec_kw={'width_ratios': [0.04, 1], 'wspace': 0.02})
        sidebar = to_rgba_array([CLUSTER_COLORS[int(c)] for c in cluster_sidebar])[:, None, :]
        ax_side.imshow(sidebar, aspect='auto')
        ax_side.axis('off')
        im = ax_main.imshow(X, aspect='auto', cmap='magma',
                            vmin=np.nanpercentile(X, 2), vmax=np.nanpercentile(X, 98))
        ax_main.set_xticks(np.arange(0, len(month_cols), max(1, len(month_cols) // 8)))
        ax_main.set_xticklabels([month_cols[i] for i in ax_main.get_xticks()], fontsize=8)
        ax_main.set_xlabel('Months post-treatment')
        ax_main.set_ylabel('Patients (downsampled per cluster, sorted by within-cluster mean risk)')
        n_clusters = traj['cluster'].nunique()
        ax_main.set_title(f'Per-patient mortality risk trajectories '
                          f'(n={len(traj):,} shown, k={n_clusters})')
        fig.colorbar(im, ax=ax_main, label='Risk score', shrink=0.7)
        save_panel(fig, 'fig4a')
        plt.close(fig)

# %% fig4b
means = load_figure_data('fig4_cluster_means.csv')
if means.empty:
    print('fig4_cluster_means.csv is empty; skipping panel B')
else:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k, g in means.groupby('cluster'):
        g = g.sort_values('month')
        x = np.arange(len(g))
        n = int(g['n_patients'].iloc[0])
        ax.plot(x, g['mean'], color=CLUSTER_COLORS[int(k)], lw=2,
                label=f'Cluster {int(k)} (n={n:,})')
        ax.fill_between(x, g['mean'] - 1.96 * g['sem'], g['mean'] + 1.96 * g['sem'],
                        color=CLUSTER_COLORS[int(k)], alpha=0.2)
    month_labels = means[means['cluster'] == means['cluster'].iloc[0]]['month'].values
    ax.set_xticks(np.arange(0, len(month_labels), max(1, len(month_labels) // 8)))
    ax.set_xticklabels([month_labels[i] for i in ax.get_xticks()], fontsize=8)
    ax.set_xlabel('Months post-treatment')
    ax.set_ylabel('Mean risk score')
    ax.set_title('Cluster mean trajectories (lower cluster id = lower mean risk)')
    ax.legend(loc='best', fontsize=8)
    save_panel(fig, 'fig4b')
    plt.close(fig)

# %% fig4c
comps = {
    'Cancer type':           load_figure_data('fig4_cluster_composition_cancer.csv'),
    'Stage':                 load_figure_data('fig4_cluster_composition_stage.csv'),
    'First-line treatment':  load_figure_data('fig4_cluster_composition_treatment.csv'),
}
if all(c.empty for c in comps.values()):
    print('All composition CSVs empty; skipping panel C')
else:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (title, comp) in zip(axes, comps.items()):
        if comp.empty:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, color='#888')
            ax.set_title(title)
            ax.axis('off')
            continue
        comp = comp.set_index('cluster')
        bottom = np.zeros(len(comp))
        palette = plt.cm.tab20(np.linspace(0, 1, len(comp.columns)))
        for i, col in enumerate(comp.columns):
            ax.bar(comp.index.astype(str), comp[col], bottom=bottom,
                   color=palette[i], label=col, edgecolor='white', lw=0.3)
            bottom += comp[col].values
        ax.set_title(title)
        ax.set_xlabel('Cluster')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=7)
    axes[0].set_ylabel('Proportion')
    fig.tight_layout()
    save_panel(fig, 'fig4c')
    plt.close(fig)

# %% fig4d
km = load_figure_data('fig4_km_data.csv')
if km.empty:
    print('fig4_km_data.csv is empty; skipping panel D')
else:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    kmf = KaplanMeierFitter()
    for k in sorted(km['cluster'].unique()):
        sub = km[km['cluster'] == k]
        if sub.empty:
            continue
        kmf.fit(sub['tt_death'] / 30.44, sub['death'], label=f'Cluster {int(k)} (n={len(sub):,})')
        kmf.plot_survival_function(ax=ax, ci_show=False, color=CLUSTER_COLORS[int(k)], lw=2)
    try:
        lr = multivariate_logrank_test(km['tt_death'], km['cluster'], km['death'])
        ax.text(0.02, 0.05, f'logrank p = {lr.p_value:.1e}', transform=ax.transAxes, fontsize=9)
    except Exception as e:
        print(f'logrank test failed: {e}')
    ax.set_xlim(0, 60)
    ax.set_xlabel('Months from first treatment')
    ax.set_ylabel('Overall survival')
    ax.set_title('Survival by trajectory cluster')
    ax.legend(loc='upper right', fontsize=8)
    save_panel(fig, 'fig4d')
    plt.close(fig)
