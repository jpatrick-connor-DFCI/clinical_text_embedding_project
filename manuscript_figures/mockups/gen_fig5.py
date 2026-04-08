import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

rng = np.random.default_rng(42)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

TEAL   = '#2a9d8f'; BLUE2 = '#457b9d'; AMBER = '#e9c46a'
CORAL  = '#e76f51'; BLUE1 = '#264653'; DKGRAY = '#495057'; LGRAY = '#dee2e6'
RED    = '#e63946'; PURPLE = '#6a4c93'
GREEN  = '#2dc653'

fig = plt.figure(figsize=(15, 13), constrained_layout=False)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.42,
                       left=0.08, right=0.97, top=0.96, bottom=0.06)

# Panel A: ROC + cancer-type AUC bars
gs_A = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0],
                                         wspace=0.45, width_ratios=[1.6, 1])
ax_A_roc  = fig.add_subplot(gs_A[0])
ax_A_bar  = fig.add_subplot(gs_A[1])

# Panel B: SMD dot plot + mini KM
gs_B = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1],
                                         wspace=0.45, width_ratios=[1.7, 1])
ax_B_smd  = fig.add_subplot(gs_B[0])
gs_B_km   = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_B[1], hspace=0.55)
ax_B_km1  = fig.add_subplot(gs_B_km[0])
ax_B_km2  = fig.add_subplot(gs_B_km[1])

ax_C = fig.add_subplot(gs[1, 0])

# Panel D: 3 KM plots
gs_D = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 1], wspace=0.50)

def add_label(ax, txt, x=-0.14, y=1.07):
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — ROC curves
# ══════════════════════════════════════════════════════════════════════════════
add_label(ax_A_roc, 'A', x=-0.16)

def make_roc(auc, n=200):
    fpr = np.linspace(0, 1, n)
    # parametric curve that achieves approx given AUC
    tpr = 1 - (1 - fpr)**((1-auc)/auc * 2)
    noise = rng.normal(0, 0.008, n)
    tpr = np.clip(tpr + noise, 0, 1)
    tpr[0] = 0; tpr[-1] = 1
    return np.sort(fpr), np.sort(tpr)

roc_specs = [
    ('Text embeddings only',         0.82, TEAL),
    ('Demographics + cancer type',   0.71, BLUE2),
    ('All covariates (no text)',      0.68, AMBER),
    ('Text + all covariates',         0.84, CORAL),
]
for lbl, auc, clr in roc_specs:
    fpr, tpr = make_roc(auc)
    ax_A_roc.plot(fpr, tpr, color=clr, lw=2,
                  label=f'{lbl}\n(AUC={auc:.2f})')

ax_A_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax_A_roc.set_xlabel('False Positive Rate', fontsize=9)
ax_A_roc.set_ylabel('True Positive Rate', fontsize=9)
ax_A_roc.set_title('ROC: ICI receipt\npropensity model', fontsize=9, fontweight='bold')
ax_A_roc.legend(fontsize=6.5, loc='lower right', frameon=True, framealpha=0.9)
ax_A_roc.set_xlim(0, 1); ax_A_roc.set_ylim(0, 1.02)

# AUC by cancer type
cancer_types = ['NSCLC', 'Breast', 'CRC', 'Prostate', 'Melanoma']
cancer_aucs  = [0.87, 0.82, 0.79, 0.75, 0.91]
bar_colors   = plt.cm.tab10(np.linspace(0, 0.5, len(cancer_types)))
ax_A_bar.barh(range(len(cancer_types)), cancer_aucs,
              color=bar_colors, height=0.55, edgecolor='white')
ax_A_bar.set_yticks(range(len(cancer_types)))
ax_A_bar.set_yticklabels(cancer_types, fontsize=8)
ax_A_bar.set_xlabel('AUC', fontsize=8.5)
ax_A_bar.set_title('AUC by cancer\n(text model)', fontsize=8.5, fontweight='bold')
ax_A_bar.set_xlim(0.60, 0.95)
ax_A_bar.axvline(0.82, color=TEAL, lw=1, ls='--', alpha=0.7, label='Mean')
ax_A_bar.tick_params(axis='y', length=0)
ax_A_bar.spines['left'].set_visible(False)
for i, v in enumerate(cancer_aucs):
    ax_A_bar.text(v + 0.004, i, f'{v:.2f}', va='center', fontsize=7.5)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — SMD dot plot + mini KMs
# ══════════════════════════════════════════════════════════════════════════════
add_label(ax_B_smd, 'B', x=-0.14)

covariates = ['Age', 'Sex (male)', 'Stage IV', 'ECOG score',
              'Cancer: NSCLC', 'Cancer: Breast', 'Prior chemo',
              'Lab: WBC', 'Lab: Albumin', 'TMB']
n_cov = len(covariates)
smd_before = rng.uniform(0.08, 0.38, n_cov)
smd_after  = rng.uniform(0.005, 0.09, n_cov)
# ensure some are nicely small after weighting
smd_after  = np.clip(smd_after, 0.003, 0.08)

y_pos = np.arange(n_cov)
ax_B_smd.scatter(smd_before, y_pos, color=CORAL, s=55, zorder=4,
                  label='Before weighting')
ax_B_smd.scatter(smd_after,  y_pos, color=TEAL, s=55, marker='o',
                  facecolors='none', edgecolors=TEAL, linewidths=1.8,
                  zorder=4, label='After IPTW')
for i in range(n_cov):
    ax_B_smd.plot([smd_before[i], smd_after[i]], [i, i],
                  color=LGRAY, lw=0.8, zorder=2)

ax_B_smd.axvline(0.1, color=DKGRAY, lw=1, ls='--', alpha=0.7, label='SMD=0.1')
ax_B_smd.set_yticks(y_pos)
ax_B_smd.set_yticklabels(covariates, fontsize=8)
ax_B_smd.set_xlabel('Standardized Mean Difference', fontsize=8.5)
ax_B_smd.set_title('Covariate balance\n(IPTW)', fontsize=9, fontweight='bold')
ax_B_smd.legend(fontsize=7, loc='lower right', frameon=True, framealpha=0.9)
ax_B_smd.set_xlim(0, 0.45)
ax_B_smd.spines['left'].set_visible(False)
ax_B_smd.tick_params(axis='y', length=0)

# mini KM panels
x_km = np.linspace(0, 60, 200)
for ax_km, title, sep in [(ax_B_km1, 'Unweighted', 0.18),
                           (ax_B_km2, 'IPTW-weighted', 0.05)]:
    ici_y = 0.72 * np.exp(-0.9 * x_km / 60) + rng.normal(0, 0.008, 200)
    non_y = (0.72 - sep) * np.exp(-1.4 * x_km / 60) + rng.normal(0, 0.008, 200)
    ax_km.plot(x_km, np.clip(ici_y, 0, 1), color=TEAL, lw=1.5, label='ICI')
    ax_km.plot(x_km, np.clip(non_y, 0, 1), color=CORAL, lw=1.5, ls='--', label='No ICI')
    ax_km.set_title(title, fontsize=7.5, fontweight='bold', pad=2)
    ax_km.set_xlim(0, 60); ax_km.set_ylim(0, 1.05)
    ax_km.set_xlabel('Months', fontsize=7)
    ax_km.set_ylabel('Surv.', fontsize=7)
    ax_km.tick_params(labelsize=6.5)
    ax_km.spines['top'].set_visible(False)
    ax_km.spines['right'].set_visible(False)
    if title == 'Unweighted':
        ax_km.legend(fontsize=6, loc='lower left', frameon=False)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — biomarker association dot grid
# ══════════════════════════════════════════════════════════════════════════════
add_label(ax_C, 'C', x=-0.10)

markers_list = [
    'KRAS G12C', 'TMB-High', 'MET Amp.', 'PD-L1 TPS≥50%',
    'STK11 mut.', 'KEAP1 mut.', 'BRAF V600E', 'RET fusion',
    'EGFR exon 20', 'ALK fusion', 'MSI-High', 'CDKN2A del.',
    'TP53 mut.', 'PTEN loss', 'BRCA1/2 mut.',
]
specs = ['ATE\n(embed PS)', 'ATT\n(embed PS)', 'Unweighted',
         'ATE\n(full PS)', 'ATT\n(full PS)', 'ICI-only']
n_m = len(markers_list); n_s = len(specs)

# simulate p-values and effect directions
log_p = rng.exponential(1.5, (n_m, n_s))
direction = rng.choice([-1, 1], (n_m, n_s), p=[0.35, 0.65])
sig = log_p > 1.3  # FDR<0.1 proxy

# some markers consistently significant
for mi in [0, 1, 2, 3]:
    sig[mi, :] = True
    log_p[mi, :] = rng.uniform(2.5, 4.5, n_s)
# STK11 harmful
direction[4, :] = -1

for mi, marker in enumerate(markers_list):
    for si, spec in enumerate(specs):
        size = np.clip(log_p[mi, si] * 15, 10, 100)
        if sig[mi, si]:
            clr = '#1d6fa4' if direction[mi, si] > 0 else '#e63946'
            ax_C.scatter(si, mi, s=size, color=clr, alpha=0.85,
                          zorder=4, edgecolors='none')
        else:
            ax_C.scatter(si, mi, s=size * 0.5, facecolors='none',
                          edgecolors=LGRAY, linewidths=1, alpha=0.6, zorder=3)

# HR annotation column
hr_vals = [0.71, 0.68, 0.78, 0.65, 1.42, 1.38, 0.72, 0.69,
           1.25, 0.74, 0.63, 1.31, 1.18, 1.29, 0.81]
for mi, hr in enumerate(hr_vals):
    clr = '#1d6fa4' if hr < 1 else '#e63946'
    ax_C.text(n_s + 0.3, mi, f'HR={hr:.2f}', va='center', fontsize=6.5,
              color=clr, fontweight='bold' if abs(hr-1) > 0.25 else 'normal')

# bold markers sig in ≥4 specs
n_sig_by_marker = sig.sum(axis=1)
ytick_labels = []
for mi, (m, nsig) in enumerate(zip(markers_list, n_sig_by_marker)):
    if nsig >= 4:
        ytick_labels.append(f'{m}*')
    else:
        ytick_labels.append(m)

ax_C.set_yticks(range(n_m)); ax_C.set_yticklabels(ytick_labels, fontsize=8)
# bold the labels for markers sig in >=4 specs
for tick, nsig in zip(ax_C.get_yticklabels(), n_sig_by_marker):
    if nsig >= 4:
        tick.set_fontweight('bold')
ax_C.set_xticks(range(n_s)); ax_C.set_xticklabels(specs, fontsize=7.5)
ax_C.set_xlim(-0.6, n_s + 1.6); ax_C.set_ylim(-0.8, n_m - 0.2)
ax_C.invert_yaxis()
ax_C.set_title('Biomarker associations\nacross analysis specifications',
               fontsize=10, fontweight='bold')
ax_C.spines['left'].set_visible(False); ax_C.tick_params(axis='y', length=0)

# legend
leg_els = [
    mpatches.Patch(color='#1d6fa4', label='Benefit (FDR<0.1)'),
    mpatches.Patch(color='#e63946', label='Harm (FDR<0.1)'),
    Line2D([0],[0], marker='o', color='w', markeredgecolor=LGRAY,
           markersize=7, label='Not significant'),
]
ax_C.legend(handles=leg_els, fontsize=7, loc='lower right',
            frameon=True, framealpha=0.9)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL D — 3 KM interaction plots
# ══════════════════════════════════════════════════════════════════════════════
pos_D = gs[1, 1].get_position(fig)
fig.text(pos_D.x0 - 0.04, pos_D.y1 + 0.025, 'D',
         fontsize=14, fontweight='bold', va='top', ha='left')

km_panels = [
    {
        'title': 'KRAS G12C\n(Predictive: ICI benefit)',
        'curves': [
            ('Mut+/ICI+',  0.80, 0.70, '#2dc653', '-'),
            ('Mut-/ICI+',  0.76, 1.10, TEAL, '-'),
            ('Mut-/ICI-',  0.72, 1.50, BLUE2, '--'),
            ('Mut+/ICI-',  0.65, 2.20, CORAL, '--'),
        ],
        'annot': 'Interaction\np=0.003',
    },
    {
        'title': 'TMB-High\n(Prognostic)',
        'curves': [
            ('TMB-Hi/ICI+', 0.82, 0.75, '#2dc653', '-'),
            ('TMB-Hi/ICI-', 0.78, 1.20, TEAL, '--'),
            ('TMB-Lo/ICI+', 0.70, 1.55, CORAL, '-'),
            ('TMB-Lo/ICI-', 0.65, 2.00, RED, '--'),
        ],
        'annot': 'Interaction\np=0.21',
    },
    {
        'title': 'MET Amp.\n(Predictive: ICI harm)',
        'curves': [
            ('MET+/ICI-',  0.72, 1.10, TEAL, '--'),
            ('MET-/ICI+',  0.74, 1.00, BLUE2, '-'),
            ('MET-/ICI-',  0.70, 1.40, AMBER, '--'),
            ('MET+/ICI+',  0.68, 2.10, CORAL, '-'),
        ],
        'annot': 'Interaction\np=0.011',
    },
]
x_km = np.linspace(0, 60, 300)

for idx, panel in enumerate(km_panels):
    ax_d = fig.add_subplot(gs_D[idx])
    n_curves = len(panel['curves'])
    ns = [55, 68, 120, 40]
    for ci, (lbl, start, rate, clr, ls) in enumerate(panel['curves']):
        y = start * np.exp(-rate * x_km / 60)
        y += rng.normal(0, 0.008, len(y))
        y = np.clip(y, 0, 1)
        ax_d.plot(x_km, y, color=clr, lw=1.5, ls=ls, label=lbl)
    ax_d.set_xlim(0, 60); ax_d.set_ylim(0, 1.05)
    ax_d.set_title(panel['title'], fontsize=7.5, fontweight='bold', pad=3)
    ax_d.set_xlabel('Months', fontsize=8)
    ax_d.set_ylabel('Survival' if idx == 0 else '', fontsize=8)
    ax_d.tick_params(labelsize=7)
    ax_d.text(0.97, 0.97, panel['annot'],
              transform=ax_d.transAxes, fontsize=6.5,
              ha='right', va='top', color=DKGRAY,
              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))
    ax_d.spines['top'].set_visible(False); ax_d.spines['right'].set_visible(False)
    ax_d.legend(fontsize=5.5, loc='lower left', frameon=False, handlelength=1.2)
    # at-risk table
    ar_t = [0, 20, 40, 60]
    for ci, (lbl, start, rate, clr, ls) in enumerate(panel['curves']):
        n0 = ns[ci]
        for ti, t in enumerate(ar_t):
            surv = start * np.exp(-rate * t / 60)
            n_left = int(n0 * np.clip(surv / start, 0, 1))
            ax_d.text(t, -0.09 - ci*0.06, str(n_left),
                      transform=ax_d.get_xaxis_transform(),
                      fontsize=5.5, ha='center', va='top', color=clr)

out = '/Users/connorpa/Documents/BIG PhD/Gusev Lab/Projects/Clinical Embeddings/clinical_text_embedding_project/manuscript_figures/mockups/figure5_biomarkers_v2.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved {out}')
