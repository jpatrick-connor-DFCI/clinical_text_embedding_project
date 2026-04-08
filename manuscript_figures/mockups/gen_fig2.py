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
LGRAY = '#dee2e6'

scheme_colors = {'death_met': CORAL, 'ICD3': TEAL, 'ICD4': BLUE2, 'PhecodeX': AMBER}
scheme_markers = {'death_met': 'D', 'ICD3': 'o', 'ICD4': '^', 'PhecodeX': 's'}

cat_colors = {
    'Infectious':     '#e63946',
    'Metabolic':      '#2a9d8f',
    'Cardiovascular': '#457b9d',
    'Endocrine':      '#e9c46a',
    'Neoplasm':       '#6a4c93',
    'Other':          '#adb5bd',
}
cat_names = list(cat_colors.keys())

fig = plt.figure(figsize=(15, 13), constrained_layout=False)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38,
                       left=0.07, right=0.97, top=0.96, bottom=0.07)

ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[1, 0])
# Panel D: 3 sub-KM panels
gs_D = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 1], wspace=0.45)

def add_label(ax, txt, x=-0.12, y=1.05):
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — multi-scheme scatter
# ══════════════════════════════════════════════════════════════════════════════
add_label(ax_A, 'A')
n_pts = {'ICD3': 200, 'ICD4': 120, 'PhecodeX': 70, 'death_met': 8}
all_x, all_y, all_scheme, all_cat = [], [], [], []
for scheme, n in n_pts.items():
    base_x = rng.uniform(0.49, 0.75, n)
    delta   = rng.beta(2, 3, n) * 0.22 + rng.normal(0.03, 0.04, n)
    y_vals  = np.clip(base_x + delta, 0.49, 0.95)
    cats    = rng.choice(cat_names, n,
                         p=[0.12, 0.18, 0.14, 0.12, 0.20, 0.24])
    all_x.extend(base_x); all_y.extend(y_vals)
    all_scheme.extend([scheme]*n); all_cat.extend(cats)

all_x = np.array(all_x); all_y = np.array(all_y)

for scheme in ['ICD3', 'ICD4', 'PhecodeX', 'death_met']:
    mask = np.array(all_scheme) == scheme
    cats_s = np.array(all_cat)[mask]
    for cat in cat_names:
        cmask = cats_s == cat
        if cmask.sum() == 0: continue
        ax_A.scatter(all_x[mask][cmask], all_y[mask][cmask],
                     marker=scheme_markers[scheme], s=22,
                     color=cat_colors[cat], alpha=0.45, linewidths=0.3,
                     edgecolors='none')

diag = np.linspace(0.48, 0.96, 100)
ax_A.plot(diag, diag, 'k--', lw=1, alpha=0.5, label='y=x')
ax_A.set_xlabel('Base model C-index', fontsize=10)
ax_A.set_ylabel('Text model C-index', fontsize=10)
ax_A.set_title('Text vs. Base model C-index\n(all schemes)', fontsize=10, fontweight='bold')
ax_A.set_xlim(0.47, 0.78); ax_A.set_ylim(0.47, 0.98)

# annotate top points
top_pts = [
    (0.62, 0.94, 'Mortality\n(death_met)', 'D'),
    (0.57, 0.91, 'CMV (ICD3-B25)', 'o'),
    (0.56, 0.89, 'CMV (ICD4-B25.9)', '^'),
    (0.51, 0.85, 'Ovarian dysfn\n(ICD3)', 'o'),
    (0.53, 0.82, 'BPH (ICD4)', '^'),
]
for xi, yi, lbl, mk in top_pts:
    ax_A.scatter([xi], [yi], marker=mk, s=55, color='#e63946',
                 zorder=10, edgecolors='black', linewidths=0.6)
    ax_A.annotate(lbl, (xi, yi), fontsize=6.5,
                  xytext=(xi+0.035, yi-0.02),
                  arrowprops=dict(arrowstyle='-', color=DKGRAY, lw=0.7),
                  ha='left', va='top')

# scheme legend
scheme_leg = [Line2D([0],[0], marker=scheme_markers[s], color='gray',
                     markersize=6, linestyle='none', label=s)
              for s in ['ICD3','ICD4','PhecodeX','death_met']]
cat_leg = [mpatches.Patch(color=cat_colors[c], label=c, alpha=0.7)
           for c in cat_names]
leg1 = ax_A.legend(handles=scheme_leg, title='Scheme', fontsize=7,
                   title_fontsize=7.5, loc='upper left',
                   frameon=True, framealpha=0.8)
ax_A.add_artist(leg1)
ax_A.legend(handles=cat_leg, title='Category', fontsize=7,
            title_fontsize=7.5, loc='lower right',
            frameon=True, framealpha=0.8)

# inset: % text > base by scheme
ax_ins = ax_A.inset_axes([0.60, 0.05, 0.38, 0.30])
pct_keys = list(scheme_colors.keys())  # ['death_met','ICD3','ICD4','PhecodeX']
pct_vals_list = [88, 71, 68, 73]
x_ins = np.arange(len(pct_keys))
bars_ins = ax_ins.bar(x_ins, pct_vals_list,
                       color=[scheme_colors[k] for k in pct_keys],
                       width=0.6, edgecolor='white')
ax_ins.set_xticks(x_ins)
ax_ins.set_xticklabels(pct_keys, fontsize=5.5, rotation=30, ha='right')
ax_ins.axhline(50, color='black', lw=0.8, ls='--')
ax_ins.set_ylim(0, 100)
ax_ins.set_ylabel('%', fontsize=6.5)
ax_ins.set_title('% Text > Base', fontsize=6.5, fontweight='bold')
ax_ins.tick_params(labelsize=6); ax_ins.spines['top'].set_visible(False)
ax_ins.spines['right'].set_visible(False)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — delta C-index violins by scheme
# ══════════════════════════════════════════════════════════════════════════════
add_label(ax_B, 'B')
scheme_list = ['death_met', 'ICD3', 'ICD4', 'PhecodeX']
scheme_n    = [8, 243, 412, 198]
vdata = []
for n in scheme_n:
    d = rng.normal(0.06, 0.08, n)
    d = np.clip(d, -0.15, 0.35)
    vdata.append(d)

parts = ax_B.violinplot(vdata, positions=range(len(scheme_list)),
                         showmedians=True, showextrema=False,
                         widths=0.6)
for pc, clr in zip(parts['bodies'], [scheme_colors[s] for s in scheme_list]):
    pc.set_facecolor(clr); pc.set_alpha(0.55)
parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(1.5)

for i, (d, clr) in enumerate(zip(vdata, [scheme_colors[s] for s in scheme_list])):
    jit = rng.uniform(-0.18, 0.18, len(d))
    ax_B.scatter(i + jit, d, s=8, color=clr, alpha=0.3, linewidths=0)

ax_B.axhline(0, color='black', lw=1, ls='--', alpha=0.7)
ax_B.set_xticks(range(len(scheme_list)))
ax_B.set_xticklabels(scheme_list, fontsize=9)
ax_B.set_ylabel('Delta C-index (Text − Base)', fontsize=10)
ax_B.set_title('C-index improvement by scheme', fontsize=10, fontweight='bold')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — category × scheme heatmap
# ══════════════════════════════════════════════════════════════════════════════
add_label(ax_C, 'C')
row_cats = ['Infectious', 'Metabolic', 'Cardiovascular',
            'Endocrine', 'Musculoskeletal', 'Hematologic']
col_schemes = ['ICD3', 'ICD4', 'PhecodeX']
hdata = np.array([
    [14,  22,  9],
    [18,  31, 12],
    [11,  19,  8],
    [13,  24, 10],
    [ 8,  17,  6],
    [10,  18,  7],
], dtype=float)

im = ax_C.imshow(hdata, cmap='YlOrRd', aspect='auto', vmin=0, vmax=35)
ax_C.set_xticks(range(len(col_schemes)))
ax_C.set_xticklabels(col_schemes, fontsize=9)
ax_C.set_yticks(range(len(row_cats)))
ax_C.set_yticklabels(row_cats, fontsize=9)
ax_C.set_title('Significant text endpoints\nby category × scheme',
               fontsize=10, fontweight='bold')
for i in range(len(row_cats)):
    for j in range(len(col_schemes)):
        ax_C.text(j, i, str(int(hdata[i,j])), ha='center', va='center',
                  fontsize=9, color='black' if hdata[i,j] < 20 else 'white',
                  fontweight='bold')
plt.colorbar(im, ax=ax_C, shrink=0.8, label='# significant endpoints')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL D — 3 example KM plots
# ══════════════════════════════════════════════════════════════════════════════
pos_D = gs[1, 1].get_position(fig)
fig.text(pos_D.x0 - 0.04, pos_D.y1 + 0.025, 'D',
         fontsize=14, fontweight='bold', va='top', ha='left')

km_specs = [
    ('ICD-3: CMV\nDisease',          48, [0.92, 0.75, 0.60]),
    ('ICD-4: T2DM w/\nRenal Compl.', 48, [0.88, 0.74, 0.62]),
    ('PhecodeX: Iron Def.\nAnemia',   48, [0.90, 0.78, 0.68]),
]
km_rates = [[1.5, 0.9, 0.5], [1.2, 0.8, 0.45], [1.0, 0.65, 0.35]]
km_clrs  = ['#e63946', '#6c757d', '#1d6fa4']
risk_lbls = ['High', 'Mid', 'Low']
risk_ns   = [[40, 85, 110], [35, 75, 95], [45, 90, 120]]

for idx in range(3):
    ax_d = fig.add_subplot(gs_D[idx])
    title, xmax, starts = km_specs[idx]
    x = np.linspace(0, xmax, 300)
    for s, r, clr, lbl in zip(starts, km_rates[idx], km_clrs, risk_lbls):
        y = s * np.exp(-r * x / xmax)
        y += rng.normal(0, 0.009, len(y))
        y = np.clip(y, 0, 1)
        ax_d.plot(x, y, lw=1.5, color=clr, label=lbl)
    ax_d.set_xlim(0, xmax); ax_d.set_ylim(0, 1.05)
    ax_d.set_title(title, fontsize=8, fontweight='bold', pad=3)
    ax_d.set_xlabel('Months', fontsize=8)
    ax_d.set_ylabel('Survival prob.' if idx == 0 else '', fontsize=8)
    ax_d.tick_params(labelsize=7.5)
    ax_d.text(0.97, 0.97, 'log-rank\np<0.001',
              transform=ax_d.transAxes, fontsize=6.5,
              ha='right', va='top', color=DKGRAY,
              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))
    ax_d.spines['top'].set_visible(False); ax_d.spines['right'].set_visible(False)
    if idx == 0:
        ax_d.legend(fontsize=6.5, loc='lower left', frameon=False)
    # mini at-risk table
    at_risk_t = [0, 12, 24, 36, 48]
    for ti, t in enumerate(at_risk_t):
        for ri, (n0, clr) in enumerate(zip(risk_ns[idx], km_clrs)):
            surv_frac = np.exp(-km_rates[idx][ri] * t / xmax)
            n_left = int(n0 * surv_frac)
            ax_d.text(t, -0.08 - ri*0.06, str(n_left),
                      transform=ax_d.get_xaxis_transform(),
                      fontsize=5.5, ha='center', va='top', color=clr)

out = '/Users/connorpa/Documents/BIG PhD/Gusev Lab/Projects/Clinical Embeddings/clinical_text_embedding_project/manuscript_figures/mockups/figure2_text_results_v2.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved {out}')
