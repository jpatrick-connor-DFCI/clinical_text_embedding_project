import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

rng = np.random.default_rng(42)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

TEAL  = '#2a9d8f'; BLUE2 = '#457b9d'; AMBER = '#e9c46a'
CORAL = '#e76f51'; BLUE1 = '#264653'; DKGRAY = '#495057'
LGRAY = '#dee2e6'; PURPLE = '#6a4c93'

modalities = ['Text', 'Stage', 'Treatment', 'Labs', 'Somatic', 'PRS']
mod_colors  = [TEAL, BLUE2, AMBER, CORAL, PURPLE, '#20b2aa']
scheme_colors = [TEAL, BLUE2, AMBER, CORAL]
scheme_names  = ['ICD3', 'ICD4', 'PhecodeX', 'death_met']

fig = plt.figure(figsize=(15, 13), constrained_layout=False)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38,
                       left=0.08, right=0.97, top=0.96, bottom=0.06)

ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[1, 0])
ax_D = fig.add_subplot(gs[1, 1])

def add_label(ax, txt, x=-0.12, y=1.05):
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — stacked bar: significant endpoints per modality × scheme
# ══════════════════════════════════════════════════════════════════════════════
add_label(ax_A, 'A')
# totals: Text=116, Stage=71, Treatment=59, Labs=80, Somatic=44, PRS=54
totals = [116, 71, 59, 80, 44, 54]
# distribute across schemes (ICD3, ICD4, PhecodeX, death_met)
scheme_fracs = np.array([[0.35, 0.40, 0.20, 0.05],
                          [0.30, 0.42, 0.22, 0.06],
                          [0.28, 0.44, 0.22, 0.06],
                          [0.32, 0.40, 0.22, 0.06],
                          [0.27, 0.45, 0.22, 0.06],
                          [0.29, 0.42, 0.23, 0.06]])
seg_counts = (scheme_fracs * np.array(totals)[:,None]).astype(int)

x = np.arange(len(modalities))
bottoms = np.zeros(len(modalities))
for si, (sc, sn) in enumerate(zip(scheme_colors, scheme_names)):
    vals = seg_counts[:, si]
    bars = ax_A.bar(x, vals, bottom=bottoms, color=sc, label=sn,
                    width=0.55, edgecolor='white', linewidth=0.5)
    bottoms += vals

ax_A.set_xticks(x)
ax_A.set_xticklabels(modalities, fontsize=9)
ax_A.set_ylabel('Significant endpoints', fontsize=10)
ax_A.set_title('Significant endpoints per\nmodality (by scheme)', fontsize=10, fontweight='bold')
ax_A.legend(title='Scheme', fontsize=8, title_fontsize=8.5, loc='upper right',
            frameon=True, framealpha=0.8)
# total count on top of each bar
for xi, tot in zip(x, totals):
    ax_A.text(xi, tot + 2, str(tot), ha='center', va='bottom', fontsize=8.5,
              fontweight='bold')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — pairwise overlap heatmap
# ══════════════════════════════════════════════════════════════════════════════
add_label(ax_B, 'B', x=-0.10)
totals_b = [116, 71, 59, 80, 44, 54]
# symmetric overlap matrix
overlap = np.array([
    [116,  48,  39,  52,  26,  31],
    [ 48,  71,  33,  41,  21,  26],
    [ 39,  33,  59,  35,  18,  22],
    [ 52,  41,  35,  80,  24,  29],
    [ 26,  21,  18,  24,  44,  19],
    [ 31,  26,  22,  29,  19,  54],
], dtype=float)

im = ax_B.imshow(overlap, cmap='Blues', aspect='auto')
ax_B.set_xticks(range(6)); ax_B.set_yticks(range(6))
ax_B.set_xticklabels(modalities, fontsize=8.5, rotation=30, ha='right')
ax_B.set_yticklabels(modalities, fontsize=8.5)
ax_B.set_title('Pairwise endpoint overlap\n(co-significant endpoints)',
               fontsize=10, fontweight='bold')
for i in range(6):
    for j in range(6):
        clr = 'white' if overlap[i,j] > 60 else 'black'
        ax_B.text(j, i, str(int(overlap[i,j])), ha='center', va='center',
                  fontsize=8, color=clr, fontweight='bold' if i==j else 'normal')
plt.colorbar(im, ax=ax_B, shrink=0.8, label='# co-significant endpoints')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — text vs best-competitor scatter
# ══════════════════════════════════════════════════════════════════════════════
add_label(ax_C, 'C')
n_pts = 120
text_ci = rng.uniform(0.52, 0.92, n_pts)
# best competitor is within -0.1 to +0.05 of text
best_ci = text_ci + rng.normal(-0.04, 0.07, n_pts)
best_ci = np.clip(best_ci, 0.50, 0.90)
winning_mod = rng.choice(['Stage','Treatment','Labs','Somatic','PRS'],
                          n_pts, p=[0.30,0.22,0.25,0.13,0.10])
for mod, clr in zip(['Stage','Treatment','Labs','Somatic','PRS'],
                    [BLUE2, AMBER, CORAL, PURPLE, '#20b2aa']):
    mask = winning_mod == mod
    ax_C.scatter(text_ci[mask], best_ci[mask], color=clr,
                 s=28, alpha=0.6, label=mod, edgecolors='none')

diag = np.linspace(0.50, 0.92, 100)
ax_C.plot(diag, diag, 'k--', lw=1, alpha=0.5)
ax_C.set_xlabel('Text model C-index', fontsize=10)
ax_C.set_ylabel('Best competitor C-index', fontsize=10)
ax_C.set_title('Text vs. best competitor\nC-index', fontsize=10, fontweight='bold')
ax_C.legend(title='Best competitor', fontsize=8, title_fontsize=8.5,
            loc='upper left', frameon=True, framealpha=0.8)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL D — modality rank strip chart (bump-style)
# ══════════════════════════════════════════════════════════════════════════════
add_label(ax_D, 'D', x=-0.12)

endpoint_names = [
    'CMV disease (B25)', 'Ovarian dysfn', 'BPH (N40)',
    'Iron def. anemia', 'Septicemia (A41)', 'HIV disease',
    'Lymphedema', 'Viral hepatitis C', 'Drug-induced liver',
    'Malnutrition (E44)', 'Hypothyroidism', 'Insomnia (G47)',
    'Neuropathy (G62)', 'Pleural effusion', 'VTE (I82)',
    'Acute kidney inj.', 'Hyponatremia', 'Leukopenia (D72)',
    'GI hemorrhage', 'Hypercalcemia',
]
n_ep = len(endpoint_names)

# generate plausible ranks (1–6) for each modality per endpoint
rank_data = np.zeros((n_ep, 6), dtype=int)
for i in range(n_ep):
    rank_data[i] = rng.permutation(6) + 1

# Text is usually rank 1-2
rank_data[:, 0] = rng.choice([1, 2, 3], n_ep, p=[0.65, 0.25, 0.10])

y_pos = np.arange(n_ep)
jitter_strength = 0.06

for mod_idx, (mod, clr) in enumerate(zip(modalities, mod_colors)):
    ranks = rank_data[:, mod_idx]
    ax_D.scatter(ranks + rng.uniform(-jitter_strength, jitter_strength, n_ep),
                 y_pos, color=clr, s=30, zorder=4, alpha=0.85,
                 edgecolors='white', linewidths=0.4)

# connect dots with thin lines per endpoint
for i in range(n_ep):
    sorted_ranks = np.sort(rank_data[i])
    ax_D.plot([sorted_ranks[0], sorted_ranks[-1]], [i, i],
              color=LGRAY, lw=0.8, zorder=2)

ax_D.set_yticks(y_pos)
ax_D.set_yticklabels(endpoint_names, fontsize=7.5)
ax_D.set_xticks([1, 2, 3, 4, 5, 6])
ax_D.set_xticklabels(['Rank 1\n(best)', '2', '3', '4', '5', 'Rank 6\n(worst)'], fontsize=8)
ax_D.set_xlabel('Modality rank (by C-index)', fontsize=10)
ax_D.set_title('Modality ranks: top 20 endpoints\n(by text C-index)', fontsize=10, fontweight='bold')
ax_D.set_ylim(-1, n_ep)
ax_D.invert_yaxis()
ax_D.spines['left'].set_visible(False)
ax_D.tick_params(axis='y', length=0)

leg_handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=clr,
                       markersize=8, label=mod)
               for mod, clr in zip(modalities, mod_colors)]
ax_D.legend(handles=leg_handles, title='Modality', fontsize=8,
            title_fontsize=8.5, loc='lower right', frameon=True, framealpha=0.9)

out = '/Users/connorpa/Documents/BIG PhD/Gusev Lab/Projects/Clinical Embeddings/clinical_text_embedding_project/manuscript_figures/mockups/figure3_feature_comps_v2.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved {out}')
