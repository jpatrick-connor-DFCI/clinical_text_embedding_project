"""Render Figure 5 biomarkers manuscript panels.

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

from _figure_utils import apply_style, load_figure_data, save_panel

apply_style()

# %% fig5a
ps = load_figure_data('fig5_ps_predictions.csv')
if ps.empty:
    print('fig5_ps_predictions.csv is empty; skipping panel A')
else:
    ps_models = ['covariates_only', 'covariates_plus_embeddings']
    fig, axes = plt.subplots(1, len(ps_models), figsize=(10, 4), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, model in zip(axes, ps_models):
        sub = ps[ps['ps_model'] == model]
        if sub.empty:
            ax.text(0.5, 0.5, f'no {model} predictions', ha='center', va='center',
                    transform=ax.transAxes, color='#888')
            ax.set_title(model.replace('_', ' '))
            ax.axis('off')
            continue
        treated = sub.loc[sub['ground_truth'] == 1, 'model_probs']
        control = sub.loc[sub['ground_truth'] == 0, 'model_probs']
        ax.hist(treated, bins=bins, alpha=0.55, color='#D62728',
                label=f'ICI (n={len(treated):,})', density=True)
        ax.hist(control, bins=bins, alpha=0.55, color='#4A4A4A',
                label=f'no-ICI (n={len(control):,})', density=True)
        ax.set_xlabel('Estimated P(ICI)')
        ax.set_title(model.replace('_', ' '))
        ax.legend(loc='upper center', fontsize=8)
    axes[0].set_ylabel('Density')
    fig.suptitle('Propensity score overlap by specification', y=1.02)
    fig.tight_layout()
    save_panel(fig, 'fig5a')
    plt.close(fig)

# %% fig5b
vol = load_figure_data('fig5_volcano_track2.csv')
if vol.empty:
    print('fig5_volcano_track2.csv is empty; skipping panel B')
else:
    sig = vol['significant'].astype(bool)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.scatter(vol.loc[~sig, 'log_hr'], vol.loc[~sig, 'neglog10_p'],
               s=8, color='#BBBBBB', alpha=0.6, label='not sig.', edgecolors='none')
    ax.scatter(vol.loc[sig, 'log_hr'], vol.loc[sig, 'neglog10_p'],
               s=18, color='#D62728', alpha=0.85,
               label=f'significant (n={sig.sum()})', edgecolors='black', linewidths=0.3)
    top = vol.sort_values('p_markerxICI').head(8)
    for _, r in top.iterrows():
        ax.annotate(f"{r['marker']} ({r['cancer']})",
                    (r['log_hr'], r['neglog10_p']),
                    fontsize=7, ha='left', va='bottom', alpha=0.9)
    ax.axvline(0, color='#888', lw=0.8, ls='--')
    ax.set_xlabel('log HR (marker × ICI)')
    ax.set_ylabel('−log10 p')
    ax.set_title('Track 2 interaction volcano')
    ax.legend(loc='upper left', fontsize=8)
    save_panel(fig, 'fig5b')
    plt.close(fig)

# %% fig5c
robust = load_figure_data('fig5_robust_hits.csv')
if robust.empty:
    print('fig5_robust_hits.csv is empty; skipping panel C')
else:
    # Top 10 by abs(log geometric-mean HR) — mean_HR is already geometric-mean from prep
    summary = (robust[['marker', 'cancer_type', 'mean_HR']]
               .drop_duplicates()
               .dropna(subset=['mean_HR'])
               .assign(abs_log=lambda d: np.abs(np.log(d['mean_HR'])))
               .sort_values('abs_log', ascending=False)
               .head(10))
    if summary.empty:
        print('No robust markers with non-null mean_HR; skipping panel C')
    else:
        spec_order = sorted(robust['spec'].dropna().unique())
        palette = plt.cm.tab10(np.linspace(0, 1, len(spec_order)))
        spec_colors = {s: palette[i] for i, s in enumerate(spec_order)}
        fig, ax = plt.subplots(figsize=(7.5, max(3, 0.45 * len(summary))))
        y_labels = []
        for y_i, (_, r) in enumerate(summary.iterrows()):
            marker, cancer = r['marker'], r['cancer_type']
            sub = robust[(robust['marker'] == marker) & (robust['cancer_type'] == cancer)]
            for j, (_, row) in enumerate(sub.iterrows()):
                offset = (j - (len(sub) - 1) / 2) * 0.15
                lo, hi = row.get('CI95_marker_ICI_low'), row.get('CI95_marker_ICI_high')
                if pd.notna(lo) and pd.notna(hi):
                    ax.errorbar(row['HR_markerxICI'], y_i + offset,
                                xerr=[[max(1e-3, row['HR_markerxICI'] - lo)],
                                      [max(1e-3, hi - row['HR_markerxICI'])]],
                                fmt='o', color=spec_colors[row['spec']],
                                capsize=2, markersize=4)
                else:
                    ax.plot(row['HR_markerxICI'], y_i + offset, 'o',
                            color=spec_colors[row['spec']], markersize=4)
            y_labels.append(f'{marker} ({cancer})')
        ax.axvline(1.0, color='#333', ls='--', lw=0.8)
        ax.set_xscale('log')
        ax.set_yticks(range(len(summary)))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('HR (marker × ICI)')
        ax.set_title('Robust predictive markers (≥2 specs, consistent direction; ranked by geo-mean HR)')
        handles = [plt.Line2D([0], [0], marker='o', color=c, lw=0, label=s)
                   for s, c in spec_colors.items()]
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)
        save_panel(fig, 'fig5c')
        plt.close(fig)

# %% fig5d
km = load_figure_data('fig5_km_top_hit.csv')
meta = load_figure_data('fig5_top_hit_meta.csv')
if km.empty or meta.empty:
    print('No top-hit KM data; skipping panel D')
else:
    m = meta.iloc[0]
    marker, cancer, ps_model = m['marker'], m['cancer'], m['ps_model']
    cohort = m.get('cohort', '')
    weight_type = m.get('weight_type', '')
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    kmf = KaplanMeierFitter()
    strata = [
        ('ICI / marker−',    (km['PX_on_ICI'] == 1) & (km['marker_value'] == 0), '#D62728', '-'),
        ('ICI / marker+',    (km['PX_on_ICI'] == 1) & (km['marker_value'] == 1), '#D62728', '--'),
        ('no-ICI / marker−', (km['PX_on_ICI'] == 0) & (km['marker_value'] == 0), '#4A4A4A', '-'),
        ('no-ICI / marker+', (km['PX_on_ICI'] == 0) & (km['marker_value'] == 1), '#4A4A4A', '--'),
    ]
    for label, mask, color, ls in strata:
        sub = km[mask]
        if len(sub) < 5:
            continue
        kmf.fit(sub['tt_death'] / 30.44, sub['death'], label=f'{label} (n={len(sub)})')
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color, linestyle=ls, lw=1.8)
    ax.set_xlim(0, 60)
    ax.set_xlabel('Months from first treatment')
    ax.set_ylabel('Overall survival')
    spec_desc = ' / '.join(s for s in (cohort, ps_model, weight_type) if s)
    ax.set_title(f'{marker} × ICI ({cancer})\n[{spec_desc}]')
    ax.legend(loc='upper right', fontsize=8)
    save_panel(fig, 'fig5d')
    plt.close(fig)
