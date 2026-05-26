"""Render Figure 1 cohort overview manuscript panels.

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
from matplotlib.patches import FancyBboxPatch

from _figure_utils import apply_style, load_figure_data, save_panel

apply_style()

# %% fig1a
counts_df = load_figure_data('fig1_cohort_counts.csv')
if counts_df.empty:
    print('fig1_cohort_counts.csv is empty; skipping panel A')
else:
    counts = counts_df.set_index('step')['n']
    n_notes = counts.get('with_notes', 0)
    n_cancer = counts.get('with_cancer_type', 0)
    n_full = counts.get('full_cohort_analysis_set', 0)
    n_modality = counts.get('modality_comparison_set', 0)
    steps = [
        (f'Patients with pre-treatment notes\n(n = {n_notes:,})', None),
        (f'+ cancer type annotation\n(n = {n_cancer:,})', f'dropped n = {n_notes - n_cancer:,}'),
        (f'Full-cohort analysis set\n(n = {n_full:,})', None),
        (f'Modality-comparison set\n(intersection of all feature files)\n(n = {n_modality:,})',
         f'dropped n = {n_full - n_modality:,}'),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, (label, drop) in enumerate(steps):
        y = 1 - i * 0.25
        box = FancyBboxPatch((0.15, y - 0.08), 0.7, 0.12,
                              boxstyle='round,pad=0.02', linewidth=1.2,
                              edgecolor='#333', facecolor='#F2F2F2')
        ax.add_patch(box)
        ax.text(0.5, y - 0.02, label, ha='center', va='center', fontsize=9)
        if i < len(steps) - 1:
            ax.annotate('', xy=(0.5, y - 0.18), xytext=(0.5, y - 0.08),
                        arrowprops=dict(arrowstyle='->', lw=1.2))
        if drop:
            ax.text(0.88, y - 0.13, drop, ha='left', va='center', fontsize=8, color='#888')
    ax.set_xlim(0, 1.25); ax.set_ylim(0, 1.1); ax.axis('off')
    save_panel(fig, 'fig1a')
    plt.close(fig)

# %% fig1b
vol = load_figure_data('fig1_note_volume.csv')
if vol.empty:
    print('fig1_note_volume.csv is empty; skipping panel B')
else:
    pivot = vol.pivot_table(index='year_bin', columns='note_type', values='n_notes', fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    pivot.plot(kind='bar', stacked=True, ax=ax, width=0.85,
               color=['#5DA5DA', '#60BD68', '#E8B72E'][:pivot.shape[1]])
    ax.set_xlabel('Years before first treatment')
    ax.set_ylabel('Notes')
    ax.set_title('Pre-treatment note volume by type')
    ax.legend(title='Note type', loc='upper left', bbox_to_anchor=(1.02, 1.0))
    ax.tick_params(axis='x', rotation=0)
    save_panel(fig, 'fig1b')
    plt.close(fig)

# %% fig1c
ct = load_figure_data('fig1_cancer_type_counts.csv')
st = load_figure_data('fig1_stage_counts.csv')
tx = load_figure_data('fig1_treatment_counts.csv')
if ct.empty and st.empty and tx.empty:
    print('All composition CSVs empty; skipping panel C')
else:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, data, title in zip(
        axes, [ct, st, tx],
        ['Cancer type', 'Cancer stage', 'First-line treatment class'],
    ):
        if data.empty:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, color='#888')
            ax.set_title(title)
            ax.axis('off')
            continue
        ax.barh(range(len(data)), data['n'].values[::-1], color='#4A4A4A')
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['category'].values[::-1], fontsize=8)
        ax.set_xlabel('Patients')
        ax.set_title(title)
    fig.tight_layout()
    save_panel(fig, 'fig1c')
    plt.close(fig)

# %% fig1d
coords = load_figure_data('fig1_umap_coords.csv')
if coords.empty:
    print('fig1_umap_coords.csv is empty; skipping panel D')
else:
    method = coords['method'].iloc[0]
    top_types = coords['cancer_type'].value_counts().head(10).index.tolist()
    plot_label = np.where(coords['cancer_type'].isin(top_types), coords['cancer_type'], 'OTHER')
    palette = plt.cm.tab10(np.linspace(0, 1, len(top_types)))
    color_map = {t: palette[i] for i, t in enumerate(top_types)}
    color_map['OTHER'] = (0.7, 0.7, 0.7, 0.4)
    fig, ax = plt.subplots(figsize=(6, 5))
    for label in top_types + ['OTHER']:
        mask = plot_label == label
        ax.scatter(coords.loc[mask, 'x'], coords.loc[mask, 'y'], s=3, alpha=0.6,
                   c=[color_map[label]], label=label, linewidths=0)
    ax.set_xlabel(f'{method} 1')
    ax.set_ylabel(f'{method} 2')
    ax.set_title(f'Pooled per-patient embeddings ({method})')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7, markerscale=2)
    save_panel(fig, 'fig1d')
    plt.close(fig)
