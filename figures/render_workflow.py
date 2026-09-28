"""Render a data-free, editable manuscript workflow schematic.

Run from the repository root: python3 figures/render_workflow.py
Only matplotlib and numpy are required. All mini-plots are conceptual drawings.
"""
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ctep-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.colors import to_rgb
import numpy as np

OUT = Path(__file__).resolve().parent / "workflow"
W, H = 1200, 1110
S = (180 / 25.4 * 72) / W
C = dict(ink="#243448", muted="#617083", line="#D8E0E7", pale="#F5F8FA",
         clinical="#267DAB", imaging="#269388", pathology="#9370AE",
         red="#D62728", gold="#C59137", blue="#427BB0")
TYPES = [C["clinical"], C["imaging"], C["pathology"]]
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.unicode_minus": False})
fig = plt.figure(figsize=(180 / 25.4, 180 / 25.4 * H / W), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0))
ax.set_axis_off()
texts = []


def tint(color, strength=.12):
    return tuple(1 - strength * (1 - v) for v in to_rgb(color))


def txt(x, y, text, size=16, color=None, weight="normal", ha="left", **kwargs):
    t = ax.text(x, y, text, fontsize=size*S, color=color or C["ink"],
                fontweight=weight, ha=ha, va="center", linespacing=1.45, **kwargs)
    texts.append(t)
    return t


def line(x1, y1, x2, y2, color=None, lw=1.3, **kwargs):
    ax.plot([x1, x2], [y1, y2], color=color or C["line"], lw=lw*S,
            solid_capstyle="round", **kwargs)


def box(x, y, w, h, fill="white", edge=None, radius=8, lw=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                      facecolor=fill, edgecolor=edge or C["line"], linewidth=lw*S)
    ax.add_patch(p)
    return p


def arrow(x1, y1, x2, y2, color=None, lw=1.6, **kwargs):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                mutation_scale=9*S, linewidth=lw*S,
                                color=color or C["muted"], shrinkA=0, shrinkB=0,
                                **kwargs))


def section(letter, title, y):
    txt(30, y, letter, 25, weight="bold")
    txt(59, y, title, 21, weight="bold")


def vector(x, y, w, h, color, n=12, phase=0):
    for k in range(n):
        strength = .22 + .68 * ((k * 7 + phase * 3) % 13) / 12
        ax.add_patch(Rectangle((x+k*w/n, y), w/n-1.2, h,
                               facecolor=tint(color, strength), edgecolor="none"))


def document(x, y, color, label):
    box(x+5, y-4, 174, 57, fill=tint(color, .05), edge=tint(color, .30), radius=4)
    box(x, y, 174, 57, fill="white", edge=tint(color, .40), radius=4)
    line(x+10, y+10, x+10, y+47, color, 3)
    txt(x+22, y+17, label, 14.5, color=color, weight="bold")
    for j, width in enumerate([130, 111, 120]):
        line(x+22, y+31+j*7, x+width, y+31+j*7, tint(color, .33), 1.5)


# a: note-level encoding and patient-level pooling.
txt(30, 32, "From clinical narratives to outcome prediction", 27, weight="bold")
section("a", "Construct a patient representation from pretreatment notes", 85)
for x, num, title in [(30, "1", "Unstructured notes"), (302, "2", "Encode each note"),
                      (590, "3", "Pool within note type"), (926, "4", "Concatenate")]:
    txt(x, 134, num, 16, color=C["muted"], weight="bold")
    txt(x+23, 134, title, 18, weight="bold")

for y, color, label in zip([171, 249, 327], TYPES, ["Clinician notes", "Imaging reports", "Pathology reports"]):
    document(47, y, color, label)
line(237, 277, 269, 277, C["muted"], 1.6)
line(269, 277, 269, 201, C["muted"], 1.6)
arrow(269, 201, 293, 201)

box(300, 171, 240, 61, fill=C["pale"], edge="#B9C7D3")
txt(420, 192, "Clinical ModernBERT", 18, weight="bold", ha="center")
txt(420, 216, "Pretrained text encoder", 15, color=C["muted"], ha="center")
arrow(420, 233, 420, 250)
for row in range(4):
    vector(335, 259+row*13, 170, 10, C["clinical"], n=15, phase=row)
txt(420, 322, "Token representations", 15, color=C["muted"], ha="center")
arrow(420, 335, 420, 350)
txt(420, 367, "Mean over tokens", 16, weight="bold", ha="center")
vector(349, 386, 142, 17, C["clinical"], phase=4)
txt(420, 426, "768 values per note", 16, color=C["muted"], ha="center")
line(502, 395, 562, 395, C["muted"], 1.6)
line(562, 395, 562, 277, C["muted"], 1.6)
arrow(562, 277, 592, 277)

txt(729, 173, "Older", 15, color=C["muted"], ha="center")
txt(820, 173, "Recent", 15, color=C["muted"], ha="center")
arrow(749, 173, 777, 173, color=C["line"])
for row, color in enumerate(TYPES):
    y = 200 + row*68
    for k in range(3):
        vector(606+k*67, y, 51, 16, color, n=6, phase=k+row)
        ax.add_patch(Circle((631+k*67, y+29), 2.5+1.8*k,
                            facecolor=tint(color, .35+.25*k), edgecolor="none"))
    arrow(808, y+8, 838, y+8, color=color)
    vector(850, y, 38, 16, color, n=5, phase=row+2)
txt(744, 397, "Recency-weighted mean", 17, weight="bold", ha="center")
txt(744, 426, "One 768-value vector per type", 16, color=C["muted"], ha="center")
arrow(899, 277, 919, 277)

box(934, 189, 232, 156, fill=C["pale"], radius=9)
txt(1050, 216, "Patient text vector", 18, weight="bold", ha="center")
for k, color in enumerate(TYPES):
    vector(952+k*67, 247, 62, 43, color, n=6, phase=k+1)
    txt(983+k*67, 313, "768", 15, color=color, ha="center")
txt(1050, 371, "2,304 text features", 18, weight="bold", ha="center")
txt(1050, 413, "All three note types\nrequired per patient", 16, color=C["muted"], ha="center")

# Temporal boundary belongs to the baseline representation only.
box(30, 454, 1136, 58, fill=C["pale"], edge=C["pale"], radius=5)
txt(48, 483, "Baseline timing", 16, weight="bold")
arrow(219, 483, 1145, 483, color="#A5B4C0", lw=1.4)
for x in [265, 314, 353, 403, 450, 481, 507]:
    ax.add_patch(Circle((x, 483), 3.6, facecolor=C["clinical"], edgecolor="none"))
line(571, 463, 571, 503, C["ink"], 1.6)
txt(365, 467, "Notes before treatment", 15, color=C["muted"], ha="center")
txt(581, 466, "t = 0", 15, weight="bold")
txt(581, 499, "First treatment", 15, weight="bold")
txt(941, 467, "Outcome follow-up", 15, color=C["muted"], ha="center")

# b: one patient representation is reused across separate endpoint models.
section("b", "Predict time to each clinical endpoint", 556)
box(30, 592, 253, 161, fill=C["pale"])
txt(48, 616, "Patient-level inputs", 18, weight="bold")
for k, color in enumerate(TYPES):
    vector(48+k*69, 639, 65, 19, color, n=7, phase=k+2)
txt(48, 681, "+ Age, sex and cancer type", 16)
txt(48, 708, "+ Note-era covariates", 16)
txt(48, 736, "Reuse across endpoints", 15, color=C["muted"])
arrow(285, 671, 315, 671)

box(319, 592, 282, 161, fill="white", edge="#A8B8C7")
txt(460, 619, "Endpoint-specific Cox models", 16, weight="bold", ha="center")
txt(460, 648, "Elastic-net text coefficients", 16, color=C["muted"], ha="center")
txt(460, 679, "Training / tuning", 15, ha="center")
for k in range(5):
    box(361+k*40, 701, 33, 16,
        fill=C["red"] if k == 4 else "#DCE5EE", edge="white", radius=2)
txt(460, 738, "Held-out evaluation", 16, weight="bold", ha="center")
arrow(605, 671, 630, 671)
line(632, 610, 632, 736, C["muted"])
for y, label in [(610, "Mortality / metastasis"), (652, "ICD-10 · 3-character"),
                 (694, "ICD-10 · 4-character"), (736, "PhecodeX")]:
    arrow(632, y, 654, y)
    box(656, y-16, 230, 32, fill=C["pale"], edge=C["pale"], radius=4)
    txt(671, y, label, 16)
    line(888, y, 908, y, C["muted"])
line(909, 610, 909, 736, C["muted"])
arrow(909, 671, 935, 671)
txt(1054, 606, "Endpoint risk scores", 18, weight="bold", ha="center")
for row in range(6):
    vector(971, 632+row*15, 180, 12, C["red"], n=9, phase=row+1)
txt(953, 679, "Patients", 14, color=C["muted"], rotation=90, ha="center")
txt(1061, 739, "Endpoints", 15, color=C["muted"], ha="center")
txt(30, 782, "Time-to-event labels include censoring; patient splits separate model fitting from evaluation.",
    16, color=C["muted"])
line(30, 805, 1166, 805)

# c: main manuscript analysis families, illustrated without implying results.
section("c", "Main manuscript analyses", 842)
CARD_Y, CARD_H = 876, 191
for x in [30, 416, 802]:
    box(x, CARD_Y, 364, CARD_H, fill="white", radius=7)
    line(x+17, CARD_Y+1, x+347, CARD_Y+1, C["red"], 2.3)

txt(48, 902, "Predictive performance", 18, weight="bold")
txt(434, 902, "Complementary information", 18, weight="bold")
txt(820, 902, "Longitudinal risk", 18, weight="bold")

# Conceptual C-index scatter and survival curves.
x, y = 60, 925
line(x, y+66, x+96, y+66, C["muted"])
line(x, y+66, x, y+3, C["muted"])
line(x+4, y+63, x+86, y+6, C["line"], ls="--")
for px, py in [(12, 48), (21, 43), (29, 48), (33, 29), (45, 31), (56, 19), (62, 23), (76, 8), (69, 15)]:
    ax.add_patch(Circle((x+px, y+py), 2.8, facecolor=C["red"], edgecolor="none"))
txt(107, 1006, "C-index", 14, color=C["muted"], ha="center")
x, y = 220, 929
line(x, y+62, x+139, y+62, C["muted"])
line(x, y+62, x, y, C["muted"])
for vals, color in [([0, 6, 9, 16, 20, 23, 27], C["blue"]),
                    ([0, 12, 22, 35, 43, 49, 55], C["red"])]:
    ax.step(x+np.array([0, 18, 38, 60, 83, 107, 132]), y+np.array(vals),
            where="post", color=color, lw=1.6*S)
txt(290, 1006, "Survival", 14, color=C["muted"], ha="center")
txt(48, 1031, "Text vs. base; stage / risk stratification", 15.5)
txt(48, 1052, "Pan-cancer and within-cancer evaluation", 15.5, color=C["muted"])

# Conceptual modality comparisons; colors match shared/palette.json.
mods = [("Stage", "#8C8C8C"), ("Treatment", "#E8B72E"), ("Somatic", "#60BD68"),
        ("PRS", "#B276B2"), ("Met. burden", "#1F77B4"), ("Text", C["red"])]
line(687, 925, 687, 1011, C["line"], ls="--")
for k, (label, color) in enumerate(mods):
    yy = 930+k*14.8
    txt(438, yy, label, 14)
    xx = 650+[0, 18, 28, 5, 36, 44][k]
    line(xx-20, yy, xx+22, yy, color, 2)
    ax.add_patch(Circle((xx, yy), 3.8, facecolor=color, edgecolor="none"))
txt(434, 1031, "Modality ranks and joint Cox associations", 15.5)
txt(434, 1052, "Added value of text in combined models", 15.5, color=C["muted"])

# Repeated scoring is a separate extension using updated note windows.
x, y = 827, 928
line(x, y+67, x+142, y+67, C["muted"])
line(x, y+67, x, y, C["muted"])
for points, color in [([53, 48, 40, 31, 25, 11], C["red"]),
                       ([34, 32, 36, 34, 31, 32], "#8493A1"),
                       ([13, 25, 34, 44, 46, 56], C["blue"])]:
    ax.plot(x+np.array([3, 28, 54, 80, 105, 137]), y+np.array(points),
            color=color, lw=1.8*S, marker="o", markersize=2*S)
txt(984, 940, "Rising", 14, color=C["red"])
txt(984, 964, "Stable", 14, color=C["muted"])
txt(984, 989, "Falling", 14, color=C["blue"])
txt(899, 1009, "Time", 14, color=C["muted"], ha="center")
txt(820, 1031, "Re-pool notes; fixed mortality model", 15.5)
txt(820, 1052, "Risk dynamics and landmark survival", 15.5, color=C["muted"])

txt(30, 1091, "Schematic: note marks, embedding values and mini-plots are illustrative, not study results.",
    14, color=C["muted"])

OUT.mkdir(parents=True, exist_ok=True)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
canvas_bounds = fig.bbox
for t in texts:
    bounds = t.get_window_extent(renderer)
    if not (canvas_bounds.contains(bounds.x0, bounds.y0) and
            canvas_bounds.contains(bounds.x1, bounds.y1)):
        raise RuntimeError(f"Text outside canvas: {t.get_text()}")
for ext in ("svg", "pdf"):
    fig.savefig(OUT / f"clinical_text_workflow.{ext}", facecolor="white")
fig.savefig(OUT / "clinical_text_workflow.png", dpi=600, facecolor="white")
fig.savefig(OUT / "clinical_text_workflow_preview.png", dpi=180, facecolor="white")
plt.close(fig)
print(f"Wrote SVG, PDF, 600-dpi PNG and preview to {OUT}")
