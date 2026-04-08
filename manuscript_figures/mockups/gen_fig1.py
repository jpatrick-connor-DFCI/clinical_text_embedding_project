import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

rng = np.random.default_rng(42)

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

fig = plt.figure(figsize=(16, 11), constrained_layout=False)

# ── layout: Panel A occupies top 2/3, bottom third split B/C/D ──────────────
gs_main = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.6, 1],
                             hspace=0.38, left=0.05, right=0.97,
                             top=0.96, bottom=0.04)
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[1],
                                          wspace=0.42)

ax_A = fig.add_subplot(gs_main[0])
ax_B = fig.add_subplot(gs_bot[0])
ax_C_outer = gs_bot[1]   # will become 2×2 sub-grid
ax_D = fig.add_subplot(gs_bot[2])

# ─── helpers ──────────────────────────────────────────────────────────────────
TEAL   = '#2a9d8f'
BLUE1  = '#264653'
BLUE2  = '#457b9d'
AMBER  = '#e9c46a'
CORAL  = '#e76f51'
LGRAY  = '#dee2e6'
DKGRAY = '#495057'

def add_label(ax, txt, x=-0.08, y=1.06, fontsize=14):
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — pipeline schematic
# ══════════════════════════════════════════════════════════════════════════════
ax_A.set_xlim(0, 10)
ax_A.set_ylim(0, 6)
ax_A.axis('off')
add_label(ax_A, 'A', x=0.0, y=1.0)

# ── document stack (row 1 left) ──────────────────────────────────────────────
doc_labels = ['Clinician\nNotes', 'Imaging\nReports', 'Pathology\nReports']
doc_colors = ['#a8dadc', '#457b9d', '#1d3557']
for i, (lbl, clr) in enumerate(zip(doc_labels, doc_colors)):
    y0 = 4.8 - i * 0.65
    # shadow
    ax_A.add_patch(Rectangle((0.18, y0-0.04), 1.28, 0.5,
                              linewidth=0, facecolor='#ced4da', zorder=1))
    ax_A.add_patch(FancyBboxPatch((0.12, y0), 1.28, 0.5,
                                  boxstyle='round,pad=0.04',
                                  linewidth=1, edgecolor=clr,
                                  facecolor='white', zorder=2))
    # text lines icon
    for j in range(3):
        ax_A.add_patch(Rectangle((0.22, y0+0.32-j*0.08), 0.55, 0.04,
                                  facecolor='#ced4da', linewidth=0, zorder=3))
    ax_A.text(0.82, y0+0.25, lbl, ha='left', va='center', fontsize=7.5,
              color=BLUE1, zorder=4)

# arrow doc → transformer
ax_A.annotate('', xy=(2.05, 3.5), xytext=(1.42, 3.5),
              arrowprops=dict(arrowstyle='->', color=TEAL, lw=2))

# ── transformer block ─────────────────────────────────────────────────────────
tx0, ty0 = 2.05, 2.65
tw, th = 2.0, 1.7
ax_A.add_patch(FancyBboxPatch((tx0, ty0), tw, th,
                               boxstyle='round,pad=0.06',
                               facecolor='#e8f4f8', edgecolor=TEAL,
                               linewidth=1.8, zorder=2))
ax_A.text(tx0 + tw/2, ty0 + th + 0.14, 'Clinical-Longformer',
          ha='center', va='bottom', fontsize=9, fontweight='bold',
          color=BLUE1, zorder=5)
# colored attention grid
cell_colors = [['#2a9d8f','#457b9d','#e9c46a','#e76f51'],
               ['#457b9d','#2a9d8f','#2a9d8f','#e9c46a'],
               ['#e76f51','#2a9d8f','#457b9d','#2a9d8f'],
               ['#e9c46a','#e76f51','#2a9d8f','#457b9d']]
cw = 0.26; ch = 0.26; cx0 = tx0+0.28; cy0 = ty0+0.28
for ri in range(4):
    for ci in range(4):
        ax_A.add_patch(Rectangle((cx0+ci*(cw+0.04), cy0+ri*(ch+0.04)),
                                   cw, ch,
                                   facecolor=cell_colors[ri][ci],
                                   alpha=0.7, linewidth=0, zorder=3))
ax_A.text(tx0+tw/2, ty0+0.12, 'Attention layers', ha='center',
          fontsize=7, color=DKGRAY, zorder=5)

# arrow transformer → embedding
ax_A.annotate('', xy=(5.05, 3.5), xytext=(4.07, 3.5),
              arrowprops=dict(arrowstyle='->', color=TEAL, lw=2))

# ── embedding cloud ───────────────────────────────────────────────────────────
n_pts = 200
colors3 = [TEAL, BLUE2, CORAL]
clabels = ['Clinician Notes', 'Imaging', 'Pathology']
for k, (clr, lbl) in enumerate(zip(colors3, clabels)):
    cx = rng.normal(5.7 + k*0.12, 0.32, n_pts//3)
    cy = rng.normal(3.5 + (k-1)*0.18, 0.38, n_pts//3)
    ax_A.scatter(cx, cy, c=clr, s=14, alpha=0.55, zorder=4, label=lbl)

ax_A.text(5.72, 4.55, '768-dim Patient Embeddings',
          ha='center', fontsize=9, fontweight='bold', color=BLUE1, zorder=6)
ax_A.text(5.72, 4.27,
          'Time-decay weighted pooling  |  Pre-treatment notes only',
          ha='center', fontsize=7.5, color=DKGRAY, style='italic', zorder=6)

# legend for embedding cloud
leg_handles = [mpatches.Patch(color=c, label=l, alpha=0.7)
               for c, l in zip(colors3, clabels)]
ax_A.legend(handles=leg_handles, loc='lower center',
            bbox_to_anchor=(0.58, 0.0), ncol=3,
            fontsize=7, frameon=False)

# arrows embedding → downstream
ax_A.annotate('', xy=(7.55, 4.4), xytext=(6.38, 4.0),
              arrowprops=dict(arrowstyle='->', color=DKGRAY, lw=1.5))
ax_A.annotate('', xy=(7.55, 2.6), xytext=(6.38, 3.1),
              arrowprops=dict(arrowstyle='->', color=DKGRAY, lw=1.5))

# ── downstream boxes ──────────────────────────────────────────────────────────
def draw_downstream_box(ax, x0, y0, w, h, title, color):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                 boxstyle='round,pad=0.06',
                                 facecolor='#f8f9fa', edgecolor=color,
                                 linewidth=1.8, zorder=2))
    ax.text(x0+w/2, y0+h-0.12, title,
            ha='center', va='top', fontsize=8.5, fontweight='bold',
            color=color, zorder=5)

# Survival box + mini KM
draw_downstream_box(ax_A, 7.55, 3.85, 2.0, 0.95, 'Survival Prediction', TEAL)
km_x = np.linspace(0, 1, 30)
for base, clr in zip([0.9, 0.7, 0.5], [TEAL, AMBER, CORAL]):
    km_y = base * np.exp(-km_x * (2 - base))
    ax_A.plot(7.72 + km_x*1.6, 3.92 + km_y*0.55, lw=1.2, color=clr, zorder=5)

# ICI box + mini volcano
draw_downstream_box(ax_A, 7.55, 2.2, 2.0, 0.95,
                    'ICI Biomarker\nDiscovery', CORAL)
vx = rng.normal(0, 1.2, 80); vy = rng.exponential(1.5, 80)
sig = np.abs(vx) > 1.5
ax_A.scatter(7.72 + (vx[~sig]+3)*0.27, 2.27 + vy[~sig]*0.14,
             s=5, color=LGRAY, alpha=0.6, zorder=4)
ax_A.scatter(7.72 + (vx[sig]+3)*0.27, 2.27 + vy[sig]*0.14,
             s=9, color=CORAL, alpha=0.8, zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — endpoint counts
# ══════════════════════════════════════════════════════════════════════════════
add_label(ax_B, 'B', x=-0.22, y=1.06)
schemes  = ['death_met', 'ICD-10\nLevel-3', 'ICD-10\nLevel-4', 'PhecodeX']
counts   = [8, 243, 412, 198]
bcolors  = [CORAL, TEAL, BLUE2, AMBER]
y_pos    = np.arange(len(schemes))
bars = ax_B.barh(y_pos, counts, color=bcolors, height=0.55, edgecolor='white',
                  linewidth=0.5)
for bar, cnt in zip(bars, counts):
    ax_B.text(cnt + 6, bar.get_y() + bar.get_height()/2,
              str(cnt), va='center', fontsize=8.5)
ax_B.set_yticks(y_pos)
ax_B.set_yticklabels(schemes, fontsize=9)
ax_B.set_xlabel('Number of endpoints', fontsize=9)
ax_B.set_title('Endpoints per scheme', fontsize=10, fontweight='bold', pad=6)
ax_B.set_xlim(0, 480)
ax_B.spines['left'].set_visible(False)
ax_B.tick_params(axis='y', length=0)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — 4 mini KM plots
# ══════════════════════════════════════════════════════════════════════════════
gs_C = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax_C_outer,
                                         wspace=0.45, hspace=0.65)
km_events = [
    ('All-cause\nMortality',   60, [0.82, 0.60, 0.35], [2.2, 1.4, 0.7],  0.78),
    ('Brain\nMetastasis',      60, [0.80, 0.65, 0.52], [1.4, 0.9, 0.5],  0.72),
    ('CMV Disease\n(B25)',     24, [0.75, 0.58, 0.45], [3.0, 1.8, 0.9],  0.81),
    ('Type 2 Diabetes\n(E11)', 48, [0.90, 0.82, 0.76], [0.7, 0.4, 0.2],  0.68),
]
km_colors = ['#e63946', '#6c757d', '#1d6fa4']
km_labels = ['High', 'Mid', 'Low']

for idx, (title, xmax, starts, rates, cind) in enumerate(km_events):
    ax_c = fig.add_subplot(gs_C[idx//2, idx%2])
    x = np.linspace(0, xmax, 200)
    for s, r, clr, lbl in zip(starts, rates, km_colors, km_labels):
        y = s * np.exp(-r * x / xmax)
        # add step-like jitter
        y += rng.normal(0, 0.012, len(y))
        y = np.clip(y, 0, 1)
        ax_c.plot(x, y, lw=1.5, color=clr, label=lbl)
    ax_c.set_xlim(0, xmax)
    ax_c.set_ylim(0, 1.02)
    ax_c.set_title(title, fontsize=7.5, fontweight='bold', pad=3)
    ax_c.set_xlabel('Months', fontsize=7)
    ax_c.set_ylabel('Survival', fontsize=7)
    ax_c.tick_params(labelsize=6.5)
    ax_c.text(0.97, 0.97, f'C-index={cind:.2f}',
              transform=ax_c.transAxes, fontsize=6.5,
              ha='right', va='top', color=DKGRAY,
              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    if idx == 0:
        ax_c.legend(fontsize=6, loc='lower left', frameon=False)

# panel C label — attach to top-left subplot
fig.add_subplot(gs_C[0, 0]).set_visible(False)  # dummy; label via text
# get position of gs_C parent
pos_C = gs_bot[1].get_position(fig)
fig.text(pos_C.x0 - 0.01, pos_C.y1 + 0.02, 'C',
         fontsize=14, fontweight='bold', va='top', ha='left')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL D — cohort composition pie
# ══════════════════════════════════════════════════════════════════════════════
add_label(ax_D, 'D', x=-0.18, y=1.06)
cancers = ['NSCLC', 'Breast', 'CRC', 'Prostate', 'Melanoma',
           'Lymphoma', 'Pancreatic', 'Ovarian', 'Bladder', 'Other']
counts_d = [1180, 890, 720, 580, 440, 390, 310, 280, 220, 650]
pie_colors = plt.cm.tab10(np.linspace(0, 1, len(cancers)))
wedges, texts, autotexts = ax_D.pie(
    counts_d, labels=cancers, autopct='%1.0f%%',
    colors=pie_colors, startangle=140,
    pctdistance=0.78, labeldistance=1.08,
    wedgeprops=dict(linewidth=0.5, edgecolor='white'))
for t in texts:
    t.set_fontsize(7.5)
for at in autotexts:
    at.set_fontsize(6.5)
ax_D.set_title('Cohort cancer type\ncomposition (N=5,660)',
               fontsize=10, fontweight='bold', pad=6)

out = '/Users/connorpa/Documents/BIG PhD/Gusev Lab/Projects/Clinical Embeddings/clinical_text_embedding_project/manuscript_figures/mockups/figure1_schematic_v2.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved {out}')
