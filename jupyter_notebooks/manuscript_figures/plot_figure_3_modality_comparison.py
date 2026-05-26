"""Render Figure 3 modality comparison manuscript panels.

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

from _figure_utils import (
    apply_style, load_figure_data, save_panel,
    MODALITY_ORDER, MODALITY_COLORS,
)

apply_style()

# %% fig3a
cindex_df = load_figure_data('fig3_modality_cindex.csv')
if cindex_df.empty:
    print('fig3_modality_cindex.csv is empty; skipping panel A')
else:
    print(f'{len(cindex_df)} (modality, event) rows, {cindex_df["event"].nunique()} events')
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, mod in enumerate(MODALITY_ORDER):
        vals = cindex_df.loc[cindex_df['modality'] == mod, 'cindex'].dropna().values
        if len(vals) == 0:
            continue
        ax.boxplot(vals, positions=[i], widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor=MODALITY_COLORS[mod], alpha=0.6, edgecolor='#333'),
                   medianprops=dict(color='#000', lw=1.5), showfliers=False)
        jit = (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.3
        ax.scatter(np.full_like(vals, i) + jit, vals, s=6, alpha=0.35,
                   color=MODALITY_COLORS[mod], edgecolors='none')
    ax.set_xticks(range(len(MODALITY_ORDER)))
    ax.set_xticklabels(MODALITY_ORDER)
    ax.set_ylabel('Test C-index')
    ax.set_title(f'C-index by modality across {cindex_df["event"].nunique()} events')
    ax.axhline(0.5, ls='--', color='#888', lw=0.8)
    save_panel(fig, 'fig3a')
    plt.close(fig)

# %% fig3b
betas = load_figure_data('fig3_joint_betas.csv')
if betas.empty:
    print('fig3_joint_betas.csv is empty; skipping panel B')
else:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, mod in enumerate(MODALITY_ORDER):
        vals = betas.loc[betas['modality'] == mod, 'beta'].dropna().values
        if len(vals) == 0:
            continue
        ax.boxplot(vals, positions=[i], widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor=MODALITY_COLORS[mod], alpha=0.6, edgecolor='#333'),
                   medianprops=dict(color='#000', lw=1.5), showfliers=False)
        jit = (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.3
        ax.scatter(np.full_like(vals, i) + jit, vals, s=6, alpha=0.35,
                   color=MODALITY_COLORS[mod], edgecolors='none')
    ax.axhline(0, ls='--', color='#888', lw=0.8)
    ax.set_xticks(range(len(MODALITY_ORDER)))
    ax.set_xticklabels(MODALITY_ORDER)
    ax.set_ylabel('Joint-model β (standardized risk score)')
    ax.set_title('Modality contribution in joint Cox model')
    save_panel(fig, 'fig3b')
    plt.close(fig)

# %% fig3c
corr_long = load_figure_data('fig3_risk_score_corr.csv')
if corr_long.empty:
    print('fig3_risk_score_corr.csv is empty; skipping panel C')
else:
    n_patients = int(corr_long['n_patients'].iloc[0]) if 'n_patients' in corr_long.columns else 0
    corr_mat = corr_long.drop(columns=[c for c in ('n_patients',) if c in corr_long.columns]).set_index('modality')
    present = [m for m in MODALITY_ORDER if m in corr_mat.index and m in corr_mat.columns]
    corr_mat = corr_mat.loc[present, present]
    if corr_mat.empty:
        print('No modalities present in correlation matrix; skipping panel C')
    else:
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        im = ax.imshow(corr_mat.values, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(len(corr_mat.columns)))
        ax.set_yticks(range(len(corr_mat.index)))
        ax.set_xticklabels(corr_mat.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_mat.index)
        for i in range(len(corr_mat)):
            for j in range(len(corr_mat)):
                v = corr_mat.values[i, j]
                if pd.isna(v):
                    continue
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        color='white' if abs(v) > 0.5 else '#222', fontsize=8)
        ax.set_title(f'Risk-score correlation (mortality, n={n_patients:,})')
        fig.colorbar(im, ax=ax, shrink=0.7, label='Pearson r')
        save_panel(fig, 'fig3c')
        plt.close(fig)

# %% fig3d
uvj = load_figure_data('fig3_univariate_vs_joint.csv')
if uvj.empty:
    print('fig3_univariate_vs_joint.csv is empty; skipping panel D')
else:
    uvj = uvj.set_index('modality')
    joint_mean = uvj['joint_auc'].dropna().iloc[0] if uvj['joint_auc'].notna().any() else np.nan
    if pd.isna(joint_mean):
        print('joint_auc all NaN; skipping panel D')
    else:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        x_left, x_right = 0, 1
        for mod in MODALITY_ORDER:
            if mod not in uvj.index or pd.isna(uvj.loc[mod, 'univariate_auc']):
                continue
            y_u = uvj.loc[mod, 'univariate_auc']
            ax.plot([x_left, x_right], [y_u, joint_mean],
                    color=MODALITY_COLORS[mod], lw=1.5, marker='o')
            ax.text(x_left - 0.05, y_u, mod, ha='right', va='center', fontsize=8,
                    color=MODALITY_COLORS[mod])
        ax.text(x_right + 0.05, joint_mean, 'joint', ha='left', va='center', fontsize=9, color='#000')
        ax.set_xticks([x_left, x_right])
        ax.set_xticklabels(['Univariate', 'Joint (all modalities)'])
        ax.set_ylabel('Mean time-dependent AUC (across events)')
        ax.set_title('Marginal lift from joint modeling')
        ax.set_xlim(-0.4, 1.4)
        save_panel(fig, 'fig3d')
        plt.close(fig)
