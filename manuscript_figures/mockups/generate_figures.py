"""
Generate 5 publication-quality mockup figures for clinical NLP paper.
Clinical-Longformer embeddings from pre-treatment EHR notes predicting
survival outcomes in oncology patients.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linewidth': 0.5,
})

RNG = np.random.default_rng(42)
OUT = "/Users/connorpa/Documents/BIG PhD/Gusev Lab/Projects/Clinical Embeddings/clinical_text_embedding_project/manuscript_figures/mockups"

PANEL_KW = dict(fontsize=14, fontweight='bold', va='top', ha='left')

def add_panel_label(ax, label, x=-0.10, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, **PANEL_KW)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — pipeline schematic
# ─────────────────────────────────────────────────────────────────────────────
def make_figure1():
    fig = plt.figure(figsize=(16, 10), constrained_layout=False)
    fig.patch.set_facecolor('white')

    # Manual layout: top row wide (A + B), bottom row (C + D)
    gs_top = gridspec.GridSpec(1, 2, figure=fig, left=0.03, right=0.97,
                                top=0.97, bottom=0.52, wspace=0.12)
    gs_bot = gridspec.GridSpec(1, 2, figure=fig, left=0.03, right=0.97,
                                top=0.47, bottom=0.04, wspace=0.12)

    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_c = fig.add_subplot(gs_bot[0, 0])
    ax_d = fig.add_subplot(gs_bot[0, 1])

    # ── Panel A: pipeline schematic ──────────────────────────────────────────
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 5)
    ax_a.axis('off')
    add_panel_label(ax_a, 'A', x=0.0, y=1.02)
    ax_a.set_title('Model Pipeline', fontsize=11, pad=4)

    box_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']
    box_labels = ['Pre-treatment\nEHR Notes', 'Clinical-\nLongformer', '768-dim\nEmbeddings',
                  'Penalized Cox\nModel', 'Risk\nScores']
    sub_labels  = [['Clinician Notes', 'Imaging Reports', 'Pathology Reports'],
                   [], [], ['Elastic Net\nRegularization'], []]
    box_x = [0.5, 2.5, 4.5, 6.5, 8.5]
    box_w, box_h = 1.6, 1.1

    for i, (bx, lab, col) in enumerate(zip(box_x, box_labels, box_colors)):
        rect = FancyBboxPatch((bx - box_w/2, 2.6 - box_h/2), box_w, box_h,
                               boxstyle='round,pad=0.07', linewidth=1.5,
                               edgecolor='white', facecolor=col, zorder=3)
        ax_a.add_patch(rect)
        ax_a.text(bx, 2.6, lab, ha='center', va='center', fontsize=8.5,
                  color='white', fontweight='bold', zorder=4)
        if sub_labels[i]:
            for j, sl in enumerate(sub_labels[i]):
                ax_a.text(bx, 1.55 - j*0.38, sl, ha='center', va='center',
                          fontsize=7, color='#444', style='italic')
        if i < 3:
            sub_note = sub_labels[i+1]
            if sub_note:
                ax_a.text(box_x[i+1], 1.0, sub_note[0], ha='center',
                          fontsize=7, color='#666', style='italic')

    # Arrows
    for i in range(len(box_x)-1):
        x0 = box_x[i] + box_w/2
        x1 = box_x[i+1] - box_w/2
        ax_a.annotate('', xy=(x1, 2.6), xytext=(x0, 2.6),
                      arrowprops=dict(arrowstyle='->', color='#333', lw=1.8))

    # Sub-label annotations
    sub_items = [('Clinician Notes', box_x[0]-0.55, 1.65),
                 ('Imaging Reports', box_x[0]-0.55, 1.27),
                 ('Pathology Reports', box_x[0]-0.55, 0.89),
                 ('Elastic Net\nRegularization', box_x[3], 1.4)]
    for txt, tx, ty in sub_items:
        ax_a.text(tx, ty, txt, ha='center', va='center', fontsize=7.5,
                  color='#333', style='italic',
                  bbox=dict(boxstyle='round,pad=0.2', fc='#F0F4FF', ec='#AABBDD', lw=0.7))

    # Dashed lines from EHR box to note types
    for y_sub in [1.65, 1.27, 0.89]:
        ax_a.plot([box_x[0], box_x[0]-0.55], [2.05, y_sub+0.12],
                  '--', color='#999', lw=0.8)
    ax_a.plot([box_x[3], box_x[3]], [2.05, 1.65],
              '--', color='#999', lw=0.8)

    # ── Panel B: endpoint counts ─────────────────────────────────────────────
    schemes = ['death_met', 'ICD-10\nLevel-3', 'ICD-10\nLevel-4', 'PhecodeX']
    counts  = [8, 243, 412, 218]
    colors  = ['#E15759', '#4E79A7', '#F28E2B', '#76B7B2']
    bars = ax_b.barh(schemes, counts, color=colors, edgecolor='white', height=0.55)
    for bar, cnt in zip(bars, counts):
        ax_b.text(cnt + 5, bar.get_y() + bar.get_height()/2, str(cnt),
                  va='center', fontsize=9, color='#333')
    ax_b.set_xlabel('Number of Predicted Endpoints', fontsize=10)
    ax_b.set_title('Predicted Endpoint Schemes', fontsize=11)
    ax_b.set_xlim(0, 480)
    ax_b.tick_params(axis='y', labelsize=9)
    add_panel_label(ax_b, 'B', x=-0.08, y=1.02)

    # ── Panel C: patient timeline schematic ──────────────────────────────────
    ax_c.set_xlim(-36, 60)
    ax_c.set_ylim(-0.5, 4.5)
    ax_c.set_xlabel('Months Relative to Treatment Start', fontsize=10)
    ax_c.set_title('Patient Timeline Schematic', fontsize=11)
    ax_c.set_yticks([])
    ax_c.spines['left'].set_visible(False)
    add_panel_label(ax_c, 'C', x=-0.06, y=1.02)

    # Main timeline bar
    ax_c.barh(2, 90, left=-36, height=0.25, color='#CCCCCC', zorder=2)

    # Pre-treatment window (shaded)
    ax_c.axvspan(-36, 0, alpha=0.12, color='#4E79A7', zorder=1)
    ax_c.text(-18, 4.1, 'Note Collection\nWindow (Pre-treatment)', ha='center',
              fontsize=8, color='#4E79A7', fontweight='bold')

    # Post-treatment window
    ax_c.axvspan(0, 60, alpha=0.08, color='#59A14F', zorder=1)
    ax_c.text(30, 4.1, 'Outcome Window\n(Post-treatment)', ha='center',
              fontsize=8, color='#59A14F', fontweight='bold')

    # Treatment start line
    ax_c.axvline(0, color='#E15759', lw=2.5, zorder=5, linestyle='--')
    ax_c.text(1, 4.25, 'Treatment\nStart', fontsize=8, color='#E15759',
              fontweight='bold', va='top')

    # Note type markers
    note_colors = ['#4E79A7', '#F28E2B', '#E15759']
    note_labels = ['Clinician Notes', 'Imaging Reports', 'Pathology Reports']
    note_freqs  = [[-30, -24, -18, -12, -6], [-28, -14, -2], [-20, -8]]
    note_ys     = [1.2, 0.6, 0.0]

    for nc, nl, nf, ny in zip(note_colors, note_labels, note_freqs, note_ys):
        ax_c.scatter(nf, [ny]*len(nf), color=nc, s=40, zorder=6, marker='D')
        ax_c.text(-36, ny, nl, fontsize=7.5, color=nc, va='center',
                  ha='right', fontweight='bold')

    # End-of-follow-up marker
    ax_c.axvline(54, color='#999', lw=1.5, linestyle=':')
    ax_c.text(55, 3.9, 'End of\nFollow-up', fontsize=7.5, color='#666', va='top')

    ax_c.set_xticks([-36, -24, -12, 0, 12, 24, 36, 48, 60])

    # ── Panel D: cancer type cohort ───────────────────────────────────────────
    cancer_types = ['NSCLC', 'Breast', 'CRC', 'Prostate', 'Melanoma',
                    'Bladder', 'Pancreatic', 'Ovarian', 'Head & Neck']
    counts_d = [1214, 924, 698, 583, 412, 378, 301, 247, 189]
    colors_d = sns.color_palette('tab10', len(cancer_types))

    wedges, texts, autotexts = ax_d.pie(
        counts_d, labels=None, colors=colors_d,
        autopct='%1.1f%%', startangle=90, pctdistance=0.80,
        wedgeprops=dict(edgecolor='white', linewidth=1.2))
    for at in autotexts:
        at.set_fontsize(7)
    ax_d.legend(wedges, [f'{ct} (n={c:,})' for ct, c in zip(cancer_types, counts_d)],
                loc='center left', bbox_to_anchor=(0.92, 0.5), fontsize=7.5,
                frameon=False)
    ax_d.set_title(f'Cohort Composition  (N={sum(counts_d):,})', fontsize=11)
    add_panel_label(ax_d, 'D', x=-0.02, y=1.02)

    fig.savefig(f'{OUT}/figure1_schematic.png', dpi=150)
    plt.close(fig)
    print('Figure 1 saved.')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — text model results
# ─────────────────────────────────────────────────────────────────────────────
def make_figure2():
    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ── Panel A: scatter text vs base C-index ─────────────────────────────────
    n_pts = 243
    cat_names = ['Infections', 'Metabolic', 'Cardiovascular', 'Endocrine',
                 'Musculoskeletal', 'Other']
    cat_colors = ['#E15759', '#F28E2B', '#4E79A7', '#76B7B2', '#59A14F', '#B07AA1']
    cat_n = [35, 42, 48, 38, 30, 50]

    base_c = RNG.uniform(0.5, 0.75, n_pts)
    # text is generally better, ~30% clearly above diagonal
    improvement = RNG.exponential(0.04, n_pts)
    text_c = np.clip(base_c + improvement, 0.5, 0.95)

    cats = np.repeat(np.arange(len(cat_names)), cat_n)
    RNG.shuffle(cats)

    for ci, (cn, cc) in enumerate(zip(cat_names, cat_colors)):
        mask = cats == ci
        ax_a.scatter(base_c[mask], text_c[mask], c=cc, s=22, alpha=0.75,
                     label=cn, linewidths=0.3, edgecolors='white', zorder=3)

    lim = [0.48, 0.97]
    ax_a.plot(lim, lim, 'k--', lw=1, zorder=2, alpha=0.6, label='y = x')
    ax_a.set_xlim(*lim); ax_a.set_ylim(*lim)
    ax_a.set_xlabel('Base Model C-index', fontsize=10)
    ax_a.set_ylabel('Text Model C-index', fontsize=10)
    ax_a.set_title('Text vs. Base Model Performance\n(ICD-10 Endpoints, n=243)', fontsize=11)
    ax_a.legend(fontsize=7.5, frameon=True, loc='upper left',
                framealpha=0.9, edgecolor='#ccc')

    # Annotate top points
    top_labels = {
        (0.53, 0.91): 'Cytomegaloviral\ndisease',
        (0.59, 0.87): 'Ovarian\ndysfunction',
        (0.62, 0.84): 'Septicemia',
    }
    for (bx, tx), lbl in top_labels.items():
        idx = np.argmin((base_c - bx)**2 + (text_c - tx)**2)
        ax_a.annotate(lbl, (base_c[idx], text_c[idx]),
                      xytext=(base_c[idx]+0.05, text_c[idx]-0.04),
                      fontsize=7, arrowprops=dict(arrowstyle='->', lw=0.8, color='#555'),
                      color='#333')
    add_panel_label(ax_a, 'A', x=-0.12, y=1.04)

    # ── Panel B: heatmap top20 endpoints × cancer types ────────────────────────
    ep_names = [
        'Cytomegaloviral disease', 'Ovarian dysfunction', 'Septicemia',
        'Hypothyroidism', 'Deep vein thrombosis', 'Pneumocystosis',
        'Hypomagnesemia', 'Adrenal insufficiency', 'Neutropenic fever',
        'Atrial fibrillation', 'Peripheral neuropathy', 'Mucositis',
        'Hepatotoxicity', 'Acute kidney injury', 'Hypocalcemia',
        'Anemia', 'Thrombocytopenia', 'Hyponatremia',
        'Colitis (immune)', 'Pulmonary embolism']
    cancer_cols = ['NSCLC', 'Breast', 'CRC', 'Prostate', 'Melanoma']
    hm_data = RNG.uniform(0.55, 0.88, (20, 5))
    # Insert some NaN (not enough events)
    nan_idx = RNG.choice(100, size=18, replace=False)
    flat = hm_data.flatten()
    flat[nan_idx] = np.nan
    hm_data = flat.reshape(20, 5)

    mask_nan = np.isnan(hm_data)
    hm_fill  = np.where(mask_nan, 0.5, hm_data)

    cmap = plt.cm.Blues.copy()
    cmap.set_bad('#CCCCCC')

    im = ax_b.imshow(hm_data, aspect='auto', cmap=cmap, vmin=0.5, vmax=0.92,
                     interpolation='nearest')
    ax_b.set_xticks(np.arange(5)); ax_b.set_xticklabels(cancer_cols, rotation=30, ha='right', fontsize=8)
    ax_b.set_yticks(np.arange(20)); ax_b.set_yticklabels(ep_names, fontsize=7)
    ax_b.set_title('Text Model C-index by Endpoint × Cancer Type\n(gray = insufficient events)', fontsize=10)
    plt.colorbar(im, ax=ax_b, shrink=0.6, label='C-index', pad=0.02)

    # Gray out NaN cells explicitly
    for r in range(20):
        for c in range(5):
            if mask_nan[r, c]:
                ax_b.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='#CCCCCC', zorder=2))

    ax_b.grid(False)
    add_panel_label(ax_b, 'B', x=-0.22, y=1.04)

    # ── Panel C: violin/box — delta AUC by category ──────────────────────────
    cat_names_c = ['Infectious', 'Metabolic', 'Endocrine', 'Cardiovascular',
                   'Autoimmune', 'Hematologic']
    cat_colors_c = ['#E15759', '#F28E2B', '#76B7B2', '#4E79A7', '#B07AA1', '#59A14F']
    data_c = [RNG.normal(m, 0.04, 30) for m in [0.08, 0.05, 0.04, 0.03, 0.06, 0.04]]

    vp = ax_c.violinplot(data_c, positions=np.arange(len(cat_names_c)),
                          showmedians=True, showextrema=False)
    for i, (body, col) in enumerate(zip(vp['bodies'], cat_colors_c)):
        body.set_facecolor(col); body.set_alpha(0.7)
    vp['cmedians'].set_color('white'); vp['cmedians'].set_linewidth(2)

    # Overlay box
    bp = ax_c.boxplot(data_c, positions=np.arange(len(cat_names_c)),
                       widths=0.12, patch_artist=True, notch=False,
                       medianprops=dict(color='white', lw=2),
                       whiskerprops=dict(lw=0.8, color='#555'),
                       capprops=dict(lw=0.8, color='#555'),
                       flierprops=dict(marker='o', ms=3, alpha=0.5))
    for patch, col in zip(bp['boxes'], cat_colors_c):
        patch.set_facecolor(col); patch.set_alpha(0.9)

    ax_c.axhline(0, color='#333', lw=1.5, linestyle='--', zorder=5)
    ax_c.set_xticks(np.arange(len(cat_names_c)))
    ax_c.set_xticklabels(cat_names_c, rotation=25, ha='right', fontsize=8.5)
    ax_c.set_ylabel('Δ C-index (Text − Base)', fontsize=10)
    ax_c.set_title('C-index Improvement by Endpoint Category', fontsize=11)
    add_panel_label(ax_c, 'C', x=-0.12, y=1.04)

    # ── Panel D: Kaplan-Meier curves ─────────────────────────────────────────
    t = np.linspace(0, 60, 300)
    risk_groups = ['Low Risk', 'Mid Risk', 'High Risk']
    lambdas     = [0.008, 0.022, 0.048]
    colors_km   = ['#59A14F', '#F28E2B', '#E15759']
    n_per_group = [320, 310, 295]

    for label, lam, col, n in zip(risk_groups, lambdas, colors_km, n_per_group):
        S = np.exp(-lam * t)
        # Add some noise / CI
        se = np.sqrt(S * (1 - S) / n)
        ax_d.plot(t, S, color=col, lw=2, label=f'{label} (n={n})')
        ax_d.fill_between(t, S - 1.96*se, S + 1.96*se, color=col, alpha=0.15)

    ax_d.set_xlim(0, 60); ax_d.set_ylim(0, 1.02)
    ax_d.set_xlabel('Months Since Treatment Start', fontsize=10)
    ax_d.set_ylabel('Overall Survival Probability', fontsize=10)
    ax_d.set_title('Kaplan-Meier Curves by Embedding Risk Group\n(Mortality Endpoint)', fontsize=10)
    ax_d.legend(fontsize=8.5, frameon=True, loc='upper right', framealpha=0.9)

    # At-risk table below
    at_risk_t = [0, 12, 24, 36, 48, 60]
    at_risk_vals = {
        'Low':  [320, 295, 270, 248, 220, 198],
        'Mid':  [310, 275, 240, 208, 178, 145],
        'High': [295, 240, 185, 140, 102, 72],
    }
    y_offsets = [-0.09, -0.15, -0.21]
    for (gname, vals), col, yo in zip(at_risk_vals.items(), colors_km, y_offsets):
        for xi, (t_val, v) in enumerate(zip(at_risk_t, vals)):
            ax_d.text(t_val, yo, str(v), ha='center', fontsize=6.5,
                      color=col, transform=ax_d.get_xaxis_transform())
    ax_d.text(-5, -0.09, 'At risk:', ha='right', fontsize=6.5, color='#333',
              transform=ax_d.get_xaxis_transform(), style='italic')
    ax_d.text(30, 0.78, 'Log-rank p < 0.001', fontsize=9, color='#333',
              ha='center',
              bbox=dict(boxstyle='round,pad=0.3', fc='#FFFBE6', ec='#DDCB77', lw=0.8))
    add_panel_label(ax_d, 'D', x=-0.12, y=1.04)

    fig.savefig(f'{OUT}/figure2_text_results.png', dpi=150)
    plt.close(fig)
    print('Figure 2 saved.')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — feature modality comparisons
# ─────────────────────────────────────────────────────────────────────────────
def make_figure3():
    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    modalities   = ['Text', 'Stage', 'Treatment', 'Labs', 'Somatic', 'PRS']
    sig_counts   = [116, 71, 59, 80, 44, 54]
    mod_colors   = ['#4E79A7', '#E15759', '#F28E2B', '#59A14F', '#B07AA1', '#76B7B2']

    # ── Panel A: bar chart sig endpoints ─────────────────────────────────────
    bars = ax_a.bar(modalities, sig_counts, color=mod_colors, edgecolor='white',
                    linewidth=1.2, width=0.6)
    for bar, cnt in zip(bars, sig_counts):
        ax_a.text(bar.get_x() + bar.get_width()/2, cnt + 1.5, str(cnt),
                  ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#333')
    ax_a.set_ylabel('Significant Endpoints (FDR < 0.05)', fontsize=10)
    ax_a.set_title('Significant Endpoints per Feature Modality', fontsize=11)
    ax_a.set_ylim(0, 140)
    add_panel_label(ax_a, 'A', x=-0.12, y=1.04)

    # ── Panel B: pairwise overlap heatmap ────────────────────────────────────
    n_mod = 6
    overlap = np.array([
        [116,  52,  48,  60,  30,  38],
        [ 52,  71,  40,  45,  24,  30],
        [ 48,  40,  59,  38,  19,  25],
        [ 60,  45,  38,  80,  28,  32],
        [ 30,  24,  19,  28,  44,  18],
        [ 38,  30,  25,  32,  18,  54],
    ], dtype=float)

    cmap_ov = sns.color_palette('Blues', as_cmap=True)
    im_b = ax_b.imshow(overlap, cmap=cmap_ov, aspect='auto', vmin=0, vmax=120)
    ax_b.set_xticks(np.arange(n_mod)); ax_b.set_xticklabels(modalities, rotation=30, ha='right', fontsize=9)
    ax_b.set_yticks(np.arange(n_mod)); ax_b.set_yticklabels(modalities, fontsize=9)
    for i in range(n_mod):
        for j in range(n_mod):
            val = int(overlap[i, j])
            color = 'white' if overlap[i, j] > 70 else '#333'
            ax_b.text(j, i, str(val), ha='center', va='center', fontsize=9,
                      color=color, fontweight='bold' if i == j else 'normal')
    plt.colorbar(im_b, ax=ax_b, shrink=0.7, label='Shared significant endpoints')
    ax_b.set_title('Pairwise Endpoint Overlap Between Modalities\n(diagonal = total significant)', fontsize=10)
    ax_b.grid(False)
    add_panel_label(ax_b, 'B', x=-0.12, y=1.04)

    # ── Panel C: scatter text vs best competitor ──────────────────────────────
    n_sig_text = 116
    # For each text-significant endpoint, who is the best competitor?
    winner_dist = {'Stage': 25, 'Labs': 30, 'Treatment': 18, 'Somatic': 12, 'PRS': 11, 'Text': 20}
    winner_colors_c = {'Stage': '#E15759', 'Labs': '#59A14F', 'Treatment': '#F28E2B',
                       'Somatic': '#B07AA1', 'PRS': '#76B7B2', 'Text': '#4E79A7'}

    text_cindex = RNG.uniform(0.55, 0.90, n_sig_text)
    best_other  = np.clip(text_cindex - RNG.normal(0.04, 0.06, n_sig_text), 0.5, 0.9)
    winner_labels = []
    for wname, wcnt in winner_dist.items():
        winner_labels.extend([wname]*wcnt)
    RNG.shuffle(winner_labels)

    for wname in winner_dist.keys():
        mask = [w == wname for w in winner_labels]
        ax_c.scatter(best_other[mask], text_cindex[mask],
                     c=winner_colors_c[wname], s=30, alpha=0.75, label=wname,
                     edgecolors='white', linewidths=0.3)

    lim_c = [0.48, 0.93]
    ax_c.plot(lim_c, lim_c, 'k--', lw=1.2, alpha=0.6)
    ax_c.set_xlim(*lim_c); ax_c.set_ylim(*lim_c)
    ax_c.set_xlabel('Best Competing Modality C-index', fontsize=10)
    ax_c.set_ylabel('Text Model C-index', fontsize=10)
    ax_c.set_title('Text vs. Best Competing Modality\n(text-significant endpoints)', fontsize=10)
    ax_c.legend(title='Best modality', fontsize=8, title_fontsize=8,
                frameon=True, framealpha=0.9, edgecolor='#ccc')
    add_panel_label(ax_c, 'C', x=-0.12, y=1.04)

    # ── Panel D: C-index heatmap modalities × top15 endpoints ────────────────
    top15_eps = [
        'Cytomegaloviral disease', 'Ovarian dysfunction', 'Septicemia',
        'Hypothyroidism', 'Deep vein thrombosis', 'Pneumocystosis',
        'Hypomagnesemia', 'Adrenal insufficiency', 'Neutropenic fever',
        'Atrial fibrillation', 'Peripheral neuropathy', 'Mucositis',
        'Hepatotoxicity', 'Acute kidney injury', 'Hypocalcemia']
    hm_d = RNG.uniform(0.52, 0.88, (15, 6))
    # Not significant = gray (set to NaN)
    ns_mask = RNG.random((15, 6)) < 0.28
    ns_mask[:, 0] = False  # Text always significant in top-15
    hm_d_plot = hm_d.copy().astype(float)
    hm_d_plot[ns_mask] = np.nan

    cmap_d = plt.cm.RdYlBu_r.copy()
    cmap_d.set_bad('#D0D0D0')
    im_d = ax_d.imshow(hm_d_plot, cmap=cmap_d, aspect='auto', vmin=0.5, vmax=0.92)
    ax_d.set_xticks(np.arange(6)); ax_d.set_xticklabels(modalities, rotation=30, ha='right', fontsize=9)
    ax_d.set_yticks(np.arange(15)); ax_d.set_yticklabels(top15_eps, fontsize=7.5)
    plt.colorbar(im_d, ax=ax_d, shrink=0.6, label='C-index', pad=0.02)

    for r in range(15):
        for c in range(6):
            if ns_mask[r, c]:
                ax_d.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='#D0D0D0', zorder=2))

    ax_d.set_title('C-index Across Modalities × Top Endpoints\n(gray = not significant)', fontsize=10)
    ax_d.grid(False)
    add_panel_label(ax_d, 'D', x=-0.22, y=1.04)

    fig.savefig(f'{OUT}/figure3_feature_comps.png', dpi=150)
    plt.close(fig)
    print('Figure 3 saved.')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — risk score trajectories
# ─────────────────────────────────────────────────────────────────────────────
def make_figure4():
    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    t_months = np.array([3, 6, 9, 12, 18, 24, 30, 36, 42, 48, 54, 60])
    cluster_defs = {
        'Rapidly Increasing':       (0.20, 0.85, 'concave',   '#E15759', 124, '--'),
        'Stable High':              (0.72, 0.78, 'flat_high',  '#F28E2B', 98,  '-'),
        'Stable Low':               (0.12, 0.16, 'flat_low',   '#59A14F', 210, ':'),
        'Decreasing then\nRebounding': (0.50, 0.46, 'dip',    '#4E79A7', 156, '-.'),
    }

    def cluster_curve(ctype, start, end, t):
        T = t / 60.0
        if ctype == 'concave':
            return start + (end - start) * (T / 0.6)**1.5
        elif ctype == 'flat_high':
            return np.linspace(start, end, len(t)) + 0.03*np.sin(T*3*np.pi)
        elif ctype == 'flat_low':
            return np.linspace(start, end, len(t))
        elif ctype == 'dip':
            return start + (end - start)*T - 0.25*np.sin(T*np.pi)
        return np.linspace(start, end, len(t))

    # ── Panel A: mean trajectories ────────────────────────────────────────────
    means = {}
    for name, (s, e, ct, col, n, ls) in cluster_defs.items():
        mu = cluster_curve(ct, s, e, t_months)
        se = RNG.uniform(0.02, 0.045, len(t_months))
        means[name] = (mu, col, n, ls)
        ax_a.plot(t_months, mu, color=col, lw=2.5, linestyle=ls,
                  label=f'{name} (n={n})')
        ax_a.fill_between(t_months, mu - 1.96*se, mu + 1.96*se, color=col, alpha=0.15)

    ax_a.set_xlabel('Months Since Treatment Start', fontsize=10)
    ax_a.set_ylabel('Predicted Mortality Risk Score', fontsize=10)
    ax_a.set_title('Risk Score Trajectory Clusters', fontsize=11)
    ax_a.set_xlim(0, 63); ax_a.set_ylim(-0.05, 1.05)
    ax_a.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor='#ccc')
    add_panel_label(ax_a, 'A', x=-0.12, y=1.04)

    # ── Panel B: KM curves per cluster ────────────────────────────────────────
    t_km = np.linspace(0, 120, 400)
    km_lambdas = [0.040, 0.030, 0.010, 0.022]
    km_names   = ['Rapidly Increasing', 'Stable High', 'Stable Low',
                  'Decreasing then\nRebounding']
    km_colors  = ['#E15759', '#F28E2B', '#59A14F', '#4E79A7']
    km_ns      = [124, 98, 210, 156]
    km_ls      = ['--', '-', ':', '-.']

    for name, lam, col, n, ls in zip(km_names, km_lambdas, km_colors, km_ns, km_ls):
        S = np.exp(-lam * t_km)
        se = np.sqrt(S*(1-S)/n)
        ax_b.plot(t_km, S, color=col, lw=2.2, linestyle=ls, label=f'{name} (n={n})')
        ax_b.fill_between(t_km, S - 1.96*se, S + 1.96*se, color=col, alpha=0.12)

    ax_b.set_xlabel('Months Since Treatment Start', fontsize=10)
    ax_b.set_ylabel('Overall Survival Probability', fontsize=10)
    ax_b.set_title('Overall Survival by Risk Trajectory Cluster', fontsize=11)
    ax_b.set_xlim(0, 120); ax_b.set_ylim(0, 1.02)
    ax_b.legend(fontsize=7.5, frameon=True, framealpha=0.9, loc='upper right')
    ax_b.text(60, 0.78, 'Log-rank p < 0.001', fontsize=9,
              bbox=dict(boxstyle='round,pad=0.3', fc='#FFFBE6', ec='#DDCB77', lw=0.8))

    # At-risk table (simplified)
    at_risk_t = [0, 24, 48, 72, 96, 120]
    at_risk_data = {
        'Rapid↑': [124, 88, 54, 30, 14, 5],
        'High':   [98,  72, 48, 28, 14, 6],
        'Low':    [210, 198, 182, 163, 140, 108],
        'Dip':    [156, 138, 118, 92, 65, 40],
    }
    for yi, (gname, vals) in enumerate(at_risk_data.items()):
        col = km_colors[yi]
        for xi, (tv, v) in enumerate(zip(at_risk_t, vals)):
            ax_b.text(tv, -0.07 - yi*0.06, str(v), ha='center', fontsize=6,
                      color=col, transform=ax_b.get_xaxis_transform())
    add_panel_label(ax_b, 'B', x=-0.12, y=1.04)

    # ── Panel C: individual patient trajectories (small multiples) ───────────
    cluster_list = list(cluster_defs.items())
    n_cols = 2; n_rows = 2
    inner_gs = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs[1, 0],
                                                 hspace=0.5, wspace=0.4)
    ax_c.remove()
    axes_c = [[fig.add_subplot(inner_gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]
    for idx, (name, (s, e, ct, col, n, ls)) in enumerate(cluster_list):
        r, c = divmod(idx, 2)
        axi = axes_c[r][c]
        mu = cluster_curve(ct, s, e, t_months)
        # Individual patient curves
        for p in range(8):
            noise = RNG.normal(0, 0.06, len(t_months))
            ind = np.clip(mu + noise, 0, 1)
            axi.plot(t_months, ind, color=col, lw=0.7, alpha=0.35)
        # Mean
        axi.plot(t_months, mu, color=col, lw=2.2, linestyle=ls)
        axi.set_title(name.replace('\n', ' '), fontsize=7.5, color=col, fontweight='bold')
        axi.set_xlim(0, 63); axi.set_ylim(-0.05, 1.05)
        axi.tick_params(labelsize=7)
        if r == 1: axi.set_xlabel('Months', fontsize=8)
        if c == 0: axi.set_ylabel('Risk Score', fontsize=8)

    axes_c[0][0].text(-0.18, 1.12, 'C', transform=axes_c[0][0].transAxes, **PANEL_KW)

    # ── Panel D: clinical characteristics dot/bar plot ────────────────────────
    cluster_names_d = ['Rapid\nIncreasing', 'Stable\nHigh', 'Stable\nLow', 'Dip &\nRebound']
    features_d = ['% Male', 'Median Age', '% Met. at Dx', '% ICI Treated']
    feat_data  = np.array([
        [62, 58, 45, 51],   # % Male
        [65, 61, 57, 63],   # Median Age
        [72, 55, 18, 42],   # % Met at Dx
        [38, 29, 52, 44],   # % ICI
    ])
    feat_colors = ['#4E79A7', '#F28E2B', '#59A14F', '#B07AA1']
    x = np.arange(len(cluster_names_d))
    width = 0.18
    for fi, (flab, fdat, fcol) in enumerate(zip(features_d, feat_data, feat_colors)):
        ax_d.bar(x + (fi - 1.5)*width, fdat, width=width, color=fcol,
                 label=flab, edgecolor='white', linewidth=0.8)

    ax_d.set_xticks(x); ax_d.set_xticklabels(cluster_names_d, fontsize=9)
    ax_d.set_ylabel('Value (% or Median)', fontsize=10)
    ax_d.set_title('Clinical Characteristics by Trajectory Cluster', fontsize=11)
    ax_d.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor='#ccc', ncol=2)

    # p-value annotations
    for xi, plab in enumerate(['***', '***', 'ns', '**']):
        ax_d.text(xi, max(feat_data[:, xi]) + 3, plab, ha='center', fontsize=10,
                  color='#333', fontweight='bold')

    add_panel_label(ax_d, 'D', x=-0.12, y=1.04)

    fig.savefig(f'{OUT}/figure4_trajectories.png', dpi=150)
    plt.close(fig)
    print('Figure 4 saved.')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — ICI biomarker discovery
# ─────────────────────────────────────────────────────────────────────────────
def make_figure5():
    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ── Panel A: Volcano plot ─────────────────────────────────────────────────
    n_vol = 280
    mut_types = ['SNV', 'AMP', 'DEL', 'Fusion']
    mt_colors = {'SNV': '#4E79A7', 'AMP': '#F28E2B', 'DEL': '#59A14F', 'Fusion': '#B07AA1'}
    mt_ns     = [140, 70, 50, 20]

    log2hr  = RNG.normal(0, 0.8, n_vol)
    neg_log_p = RNG.exponential(1.2, n_vol)
    mut_type_labels = np.repeat(mut_types, mt_ns)
    event_counts = RNG.integers(10, 120, n_vol)

    sig_thresh = 2.5  # -log10(0.05/corrections)
    sig_mask   = neg_log_p > sig_thresh

    for mt in mut_types:
        mask = mut_type_labels == mt
        ax_a.scatter(log2hr[mask], neg_log_p[mask],
                     c=mt_colors[mt], s=event_counts[mask]*0.35 + 8,
                     alpha=0.65, label=mt, edgecolors='white', linewidths=0.3)

    ax_a.axhline(sig_thresh, color='#999', lw=1.2, linestyle='--', alpha=0.8)
    ax_a.axvline(-1, color='#AAA', lw=1, linestyle='--', alpha=0.7)
    ax_a.axvline(1, color='#AAA', lw=1, linestyle='--', alpha=0.7)
    ax_a.text(1.02, sig_thresh + 0.1, f'FDR = 0.05', fontsize=7.5, color='#888')
    ax_a.text(-1.05, ax_a.get_ylim()[1] * 0.9, 'HR < 0.5\n(benefit)', fontsize=7.5,
              color='#555', ha='right')
    ax_a.text(1.05, ax_a.get_ylim()[1] * 0.9, 'HR > 2\n(harm)', fontsize=7.5, color='#555')

    # Annotate top hits
    top_hits = [('KRAS_SNV', -1.6, 6.8), ('BRCA2_DEL', 1.9, 5.4),
                ('ALK_FUSION', -2.1, 4.8), ('MET_AMP', 1.5, 4.2),
                ('STK11_SNV', 1.8, 3.8), ('TMB_high_SNV', -1.9, 5.1)]
    for gene, lhr, nlp in top_hits:
        # find nearest point
        dist = (log2hr - lhr)**2 + (neg_log_p - nlp)**2
        idx  = np.argmin(dist)
        ax_a.annotate(gene, (log2hr[idx], neg_log_p[idx]),
                      xytext=(log2hr[idx] + 0.25, neg_log_p[idx] + 0.25),
                      fontsize=7, arrowprops=dict(arrowstyle='->', lw=0.7, color='#555'),
                      color='#222', fontweight='bold')

    ax_a.set_xlabel('log₂(HR) — ICI interaction term', fontsize=10)
    ax_a.set_ylabel('−log₁₀(FDR p-value)', fontsize=10)
    ax_a.set_title('ICI Predictive Biomarker Volcano Plot', fontsize=11)
    legend_handles = [mpatches.Patch(color=mt_colors[mt], label=mt) for mt in mut_types]
    ax_a.legend(handles=legend_handles, title='Mutation type', fontsize=8,
                title_fontsize=8, frameon=True, framealpha=0.9)
    # Size legend
    for sz, lbl in [(20, '~50 events'), (45, '~100 events')]:
        ax_a.scatter([], [], s=sz, color='#888', alpha=0.7, label=lbl)
    ax_a.legend(handles=legend_handles +
                [plt.scatter([], [], s=20, color='#888', alpha=0.7, label='~50 events'),
                 plt.scatter([], [], s=45, color='#888', alpha=0.7, label='~100 events')],
                title='Mutation type / event count', fontsize=7.5, title_fontsize=8,
                frameon=True, framealpha=0.9, ncol=1)
    add_panel_label(ax_a, 'A', x=-0.12, y=1.04)

    # ── Panel B: Forest plot top 10 ────────────────────────────────────────────
    forest_genes = ['KRAS_SNV', 'ALK_FUSION', 'TMB_high_SNV', 'EGFR_SNV',
                    'MET_AMP', 'BRCA2_DEL', 'STK11_SNV', 'CDKN2A_DEL',
                    'MDM2_AMP', 'PIK3CA_SNV']
    hrs   = [0.38, 0.45, 0.42, 0.51, 2.1, 1.85, 1.72, 1.60, 1.48, 0.68]
    ci_lo = [v * RNG.uniform(0.6, 0.8) for v in hrs]
    ci_hi = [v * RNG.uniform(1.2, 1.6) for v in hrs]
    pvals = [0.001, 0.003, 0.002, 0.008, 0.004, 0.006, 0.01, 0.02, 0.03, 0.04]
    events = [48, 32, 55, 41, 28, 36, 44, 29, 33, 52]
    benefit = [h < 1 for h in hrs]

    ax_b.axvline(1, color='#333', lw=1.5, zorder=2)
    y_pos = np.arange(len(forest_genes))[::-1]
    for i, (gene, hr, lo, hi, pv, ev, ben, yp) in enumerate(
            zip(forest_genes, hrs, ci_lo, ci_hi, pvals, events, benefit, y_pos)):
        col = '#4E79A7' if ben else '#E15759'
        ax_b.plot([lo, hi], [yp, yp], color=col, lw=2, solid_capstyle='round', zorder=3)
        ax_b.scatter([hr], [yp], color=col, s=55, zorder=4, edgecolors='white', linewidths=0.8)
        # P-value and n
        ax_b.text(12, yp, f'p={pv:.3f}', va='center', fontsize=7.5, color='#444')
        ax_b.text(22, yp, f'n={ev}', va='center', fontsize=7.5, color='#666')

    ax_b.set_xscale('log')
    ax_b.set_xlim(0.1, 30)
    ax_b.set_xlabel('Hazard Ratio (log scale)', fontsize=10)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(forest_genes, fontsize=8.5)
    ax_b.set_title('Forest Plot — Top ICI Predictive Biomarkers', fontsize=11)
    benefit_patch = mpatches.Patch(color='#4E79A7', label='ICI benefit')
    harm_patch    = mpatches.Patch(color='#E15759', label='ICI harm')
    ax_b.legend(handles=[benefit_patch, harm_patch], fontsize=8, frameon=True,
                framealpha=0.9, loc='lower right')
    ax_b.text(12, len(forest_genes) - 0.3, 'p-value', fontsize=8, color='#444', fontweight='bold')
    ax_b.text(22, len(forest_genes) - 0.3, 'Events', fontsize=8, color='#666', fontweight='bold')
    add_panel_label(ax_b, 'B', x=-0.18, y=1.04)

    # ── Panel C: IPTW KM curves (2 sub-panels) ────────────────────────────────
    ax_c.remove()
    inner_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], wspace=0.4)
    ax_c1 = fig.add_subplot(inner_c[0])
    ax_c2 = fig.add_subplot(inner_c[1])

    t_km = np.linspace(0, 60, 300)
    # Left: predictive benefit marker
    curves_left = [
        ('Marker+/ICI+', 0.010, '#4E79A7', '-',  'ICI benefit in marker+'),
        ('Marker+/ICI−', 0.030, '#F28E2B', '--', ''),
        ('Marker−/ICI+', 0.025, '#59A14F', ':',  ''),
        ('Marker−/ICI−', 0.028, '#E15759', '-.', ''),
    ]
    for name, lam, col, ls, annot in curves_left:
        S = np.exp(-lam * t_km)
        ax_c1.plot(t_km, S, color=col, lw=2, linestyle=ls, label=name)
    ax_c1.set_xlabel('Months', fontsize=9); ax_c1.set_ylabel('Survival Probability', fontsize=9)
    ax_c1.set_title('Predictive Biomarker\n(KRAS_SNV)', fontsize=9, fontweight='bold')
    ax_c1.legend(fontsize=6.5, frameon=True, loc='lower left')
    ax_c1.text(30, 0.82, 'Interaction\np = 0.004', fontsize=7.5, ha='center',
               bbox=dict(boxstyle='round,pad=0.2', fc='#EEF4FF', ec='#AABBDD', lw=0.7))
    ax_c1.set_xlim(0, 60); ax_c1.set_ylim(0, 1.02)

    # Right: prognostic marker
    curves_right = [
        ('Marker+/ICI+', 0.040, '#4E79A7', '-'),
        ('Marker+/ICI−', 0.042, '#F28E2B', '--'),
        ('Marker−/ICI+', 0.018, '#59A14F', ':'),
        ('Marker−/ICI−', 0.020, '#E15759', '-.'),
    ]
    for name, lam, col, ls in curves_right:
        S = np.exp(-lam * t_km)
        ax_c2.plot(t_km, S, color=col, lw=2, linestyle=ls, label=name)
    ax_c2.set_xlabel('Months', fontsize=9)
    ax_c2.set_title('Prognostic Biomarker\n(STK11_SNV)', fontsize=9, fontweight='bold')
    ax_c2.legend(fontsize=6.5, frameon=True, loc='lower left')
    ax_c2.text(30, 0.82, 'Interaction\np = 0.62', fontsize=7.5, ha='center',
               bbox=dict(boxstyle='round,pad=0.2', fc='#FFF4EE', ec='#DDBBAA', lw=0.7))
    ax_c2.set_xlim(0, 60); ax_c2.set_ylim(0, 1.02)

    ax_c1.text(-0.18, 1.08, 'C', transform=ax_c1.transAxes, **PANEL_KW)

    # ── Panel D: robustness summary dot plot ──────────────────────────────────
    top15_markers = ['KRAS_SNV', 'ALK_FUSION', 'TMB_high_SNV', 'EGFR_SNV',
                     'MET_AMP', 'BRCA2_DEL', 'STK11_SNV', 'CDKN2A_DEL',
                     'MDM2_AMP', 'PIK3CA_SNV', 'PTEN_DEL', 'BRAF_SNV',
                     'FGFR3_AMP', 'RET_FUSION', 'NF1_SNV']
    specs = ['ATE/Emb.PS', 'ATT/Emb.PS', 'Unwtd/Emb.PS',
             'ATE/Full.PS', 'ATT/Full.PS', 'Unwtd/Full.PS']
    n_markers = len(top15_markers); n_specs = len(specs)
    # Generate significance + direction data
    sig_mat = RNG.random((n_markers, n_specs)) < 0.65
    # Top markers more consistently significant
    sig_mat[:5, :] = RNG.random((5, n_specs)) < 0.88
    direction = RNG.choice([-1, 1], size=(n_markers, n_specs))

    for mi, marker in enumerate(top15_markers):
        y = n_markers - 1 - mi
        n_sig = sig_mat[mi].sum()
        for si, spec in enumerate(specs):
            if sig_mat[mi, si]:
                col = '#4E79A7' if direction[mi, si] > 0 else '#E15759'
                ax_d.scatter(si, y, s=70, color=col, zorder=4,
                             edgecolors='white', linewidths=0.5)
                if n_sig >= 4:
                    ax_d.scatter(si, y, s=130, color='none', zorder=5,
                                 edgecolors='#FFD700', linewidths=1.5)
            else:
                ax_d.scatter(si, y, s=50, color='none', edgecolors='#BBBBBB',
                             linewidths=1.2, zorder=3)

    ax_d.set_xticks(np.arange(n_specs))
    ax_d.set_xticklabels(specs, rotation=35, ha='right', fontsize=8)
    ax_d.set_yticks(np.arange(n_markers))
    ax_d.set_yticklabels(top15_markers[::-1], fontsize=8)
    ax_d.set_xlim(-0.5, n_specs - 0.5)
    ax_d.set_ylim(-0.5, n_markers - 0.5)
    ax_d.set_title('Cross-Specification Robustness\n(gold ring = significant in ≥4 specs)', fontsize=10)
    ax_d.grid(True, alpha=0.3, linewidth=0.5)

    benefit_patch_d = mpatches.Patch(color='#4E79A7', label='ICI benefit (significant)')
    harm_patch_d    = mpatches.Patch(color='#E15759', label='ICI harm (significant)')
    ns_patch        = mpatches.Patch(facecolor='none', edgecolor='#BBBBBB', lw=1.2, label='Not significant')
    robust_patch    = Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                              markeredgecolor='#FFD700', markeredgewidth=1.5, markersize=8,
                              label='Robust (≥4 specs)')
    ax_d.legend(handles=[benefit_patch_d, harm_patch_d, ns_patch, robust_patch],
                fontsize=7.5, frameon=True, framealpha=0.9, loc='lower right',
                edgecolor='#ccc')
    add_panel_label(ax_d, 'D', x=-0.18, y=1.04)

    fig.savefig(f'{OUT}/figure5_biomarkers.png', dpi=150)
    plt.close(fig)
    print('Figure 5 saved.')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    make_figure5()
    print('All figures saved to:', OUT)
