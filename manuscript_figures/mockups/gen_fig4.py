import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

rng = np.random.default_rng(42)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

TEAL   = '#2a9d8f'; BLUE2 = '#457b9d'; AMBER = '#e9c46a'
CORAL  = '#e76f51'; BLUE1 = '#264653'; DKGRAY = '#495057'; LGRAY = '#dee2e6'

cluster_colors = [CORAL, BLUE2, TEAL, AMBER]
cluster_names  = ['Rapidly Increasing\n(n=312)',
                  'Stable High\n(n=247)',
                  'Stable Low\n(n=489)',
                  'Decreasing→Rebounding\n(n=184)']
cluster_ns = [312, 247, 489, 184]

fig = plt.figure(figsize=(14, 12), constrained_layout=False)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.42,
                       left=0.09, right=0.97, top=0.96, bottom=0.06)

ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
gs_C = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1, 0], wspace=0.50, hspace=0.60)
ax_D = fig.add_subplot(gs[1, 1])

def add_label(ax, txt, x=-0.14, y=1.05):
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — trajectory cluster mean lines with CI
# ══════════════════════════════════════════════════════════════════════════════
add_label(ax_A, 'A')
months = np.linspace(3, 60, 100)

def traj_1(t):   # rapidly increasing
    return 0.20 + 0.65 * (t - 3) / 57 + rng.normal(0, 0.008, len(t))
def traj_2(t):   # stable high
    return 0.75 + rng.normal(0, 0.012, len(t))
def traj_3(t):   # stable low
    return 0.15 + rng.normal(0, 0.008, len(t))
def traj_4(t):   # decreasing then rebounding
    mid = (t - 3) / 57
    base = 0.50 - 0.30 * np.sin(np.pi * mid)
    base[mid > 0.5] = 0.20 + 0.25 * (mid[mid > 0.5] - 0.5) * 2
    return base + rng.normal(0, 0.010, len(t))

trajs = [traj_1(months), traj_2(months), traj_3(months), traj_4(months)]
ci_width = [0.06, 0.04, 0.04, 0.07]

for mean_t, ci, clr, lbl in zip(trajs, ci_width, cluster_colors, cluster_names):
    ax_A.plot(months, mean_t, color=clr, lw=2.2, label=lbl)
    ax_A.fill_between(months, mean_t - ci, mean_t + ci, color=clr, alpha=0.18)

ax_A.set_xlabel('Months from treatment', fontsize=10)
ax_A.set_ylabel('Standardized risk score', fontsize=10)
ax_A.set_title('Risk score trajectories\nby cluster', fontsize=10, fontweight='bold')
ax_A.set_xlim(3, 60); ax_A.set_ylim(-0.05, 1.05)
ax_A.legend(fontsize=7.5, loc='upper left', frameon=True, framealpha=0.85,
            handlelength=1.5)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — KM by cluster
# ══════════════════════════════════════════════════════════════════════════════
add_label(ax_B, 'B', x=-0.10)
km_starts = [0.42, 0.58, 0.85, 0.72]
km_rates  = [2.2, 1.6, 0.55, 1.0]
x_km = np.linspace(0, 60, 300)

for start, rate, clr, lbl in zip(km_starts, km_rates, cluster_colors, cluster_names):
    y = start * np.exp(-rate * x_km / 60)
    y += rng.normal(0, 0.009, len(y))
    y = np.clip(y, 0, 1)
    ax_B.plot(x_km, y, color=clr, lw=2, label=lbl.split('\n')[0])

ax_B.set_xlabel('Months', fontsize=10)
ax_B.set_ylabel('Survival probability', fontsize=10)
ax_B.set_title('Kaplan-Meier by\ntrajectory cluster', fontsize=10, fontweight='bold')
ax_B.set_xlim(0, 60); ax_B.set_ylim(0, 1.05)
ax_B.text(0.97, 0.97, 'log-rank\np<0.001',
          transform=ax_B.transAxes, fontsize=8.5, ha='right', va='top',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=LGRAY, alpha=0.9))
ax_B.legend(fontsize=7.5, loc='upper right', frameon=True)

# at-risk table
ar_t = [0, 12, 24, 36, 48, 60]
for ri, (n0, rate, clr) in enumerate(zip(cluster_ns, km_rates, cluster_colors)):
    for ti, t in enumerate(ar_t):
        surv = np.exp(-rate * t / 60)
        n_left = int(n0 * surv)
        ax_B.text(t, -0.08 - ri*0.055, str(n_left),
                  transform=ax_B.get_xaxis_transform(),
                  fontsize=5.5, ha='center', va='top', color=clr)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — trajectory feature distributions by cluster (violin)
# ══════════════════════════════════════════════════════════════════════════════
pos_C = gs[1, 0].get_position(fig)
fig.text(pos_C.x0 - 0.04, pos_C.y1 + 0.025, 'C',
         fontsize=14, fontweight='bold', va='top', ha='left')

feat_names = ['Baseline\nrisk (m3)', 'Final\nrisk (m60)',
              'Overall\nslope', 'AUC under\ntrajectory',
              'Minimum\nrisk', 'Rebound\nmagnitude']
feat_params = [
    # (means_per_cluster, sd)
    ([0.22, 0.75, 0.15, 0.50], 0.06),
    ([0.85, 0.74, 0.15, 0.45], 0.07),
    ([0.40, 0.00, 0.00, -0.05], 0.06),
    ([0.52, 0.74, 0.15, 0.34], 0.07),
    ([0.18, 0.72, 0.13, 0.19], 0.05),
    ([0.04, 0.03, 0.02, 0.26], 0.04),
]

for fi, (feat, (means, sd)) in enumerate(zip(feat_names, feat_params)):
    ax_c = fig.add_subplot(gs_C[fi//3, fi%3])
    vdata = [rng.normal(m, sd, n) for m, n in zip(means, cluster_ns)]
    # truncate for display
    vdata_show = [d[:min(len(d), 200)] for d in vdata]
    parts = ax_c.violinplot(vdata_show, positions=range(4),
                             showmedians=True, showextrema=False, widths=0.6)
    for pc, clr in zip(parts['bodies'], cluster_colors):
        pc.set_facecolor(clr); pc.set_alpha(0.55)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(1.2)
    ax_c.set_xticks(range(4))
    ax_c.set_xticklabels(['C1', 'C2', 'C3', 'C4'], fontsize=7)
    ax_c.set_title(feat, fontsize=7.5, fontweight='bold', pad=2)
    ax_c.tick_params(labelsize=6.5)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL D — clinical characteristics by cluster
# ══════════════════════════════════════════════════════════════════════════════
add_label(ax_D, 'D', x=-0.12)

char_labels = ['% Male', 'Median age\n(/100)', '% Stage IV', '% ICI treated', '% Alive 5y']
char_vals = np.array([
    [55, 62, 48, 71, 21],   # C1
    [58, 65, 52, 68, 28],   # C2
    [45, 60, 31, 45, 58],   # C3
    [51, 63, 44, 60, 35],   # C4
], dtype=float)
char_vals[:, 1] /= 100  # scale age to 0-1

x = np.arange(len(char_labels))
w = 0.18
for ci, (clr, lbl_short) in enumerate(zip(cluster_colors, ['C1','C2','C3','C4'])):
    bars = ax_D.bar(x + (ci-1.5)*w, char_vals[ci], w,
                    color=clr, label=lbl_short, edgecolor='white', linewidth=0.5)

# asterisks for "significant" differences (just annotation)
sig_idx = [0, 2, 4]
for si in sig_idx:
    ymax = char_vals[:, si].max()
    ax_D.text(si, ymax + 2.5, '***', ha='center', va='bottom', fontsize=9,
              color=DKGRAY)

ax_D.set_xticks(x)
ax_D.set_xticklabels(char_labels, fontsize=8.5)
ax_D.set_ylabel('Value (%)', fontsize=10)
ax_D.set_title('Clinical characteristics\nby cluster', fontsize=10, fontweight='bold')
ax_D.legend(title='Cluster', fontsize=8, title_fontsize=8.5,
            loc='upper right', frameon=True, framealpha=0.8)
ax_D.set_ylim(0, 80)

out = '/Users/connorpa/Documents/BIG PhD/Gusev Lab/Projects/Clinical Embeddings/clinical_text_embedding_project/manuscript_figures/mockups/figure4_trajectories_v2.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved {out}')
