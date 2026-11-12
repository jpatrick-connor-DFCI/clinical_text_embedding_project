"""Render compact Figure 1a model schematic without panel lettering.

Run from the repository root: python3 figures/render_model_schematic.py
Only matplotlib is required. Embedding cells are conceptual, not study data.
The physical canvas is 180 x 78 mm; SVG and PDF retain editable vector text.
"""
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ctep-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "workflow"
W, H = 180, 78
C = dict(ink="#243448", muted="#617083", line="#D8E0E7", pale="#F5F8FA",
         clinical="#267DAB", imaging="#269388", pathology="#9370AE")
TYPES = [C["clinical"], C["imaging"], C["pathology"]]
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.unicode_minus": False})
fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0))
ax.set_axis_off()
texts = []


def tint(color, strength=.12):
    return tuple(1 - strength * (1 - v) for v in to_rgb(color))


def txt(x, y, text, size=8, color=None, weight="normal", ha="center",
        within=None, **kwargs):
    artist = ax.text(x, y, text, fontsize=size, color=color or C["ink"],
                     fontweight=weight, ha=ha, va="center", linespacing=1.25,
                     **kwargs)
    texts.append((artist, within))
    return artist


def line(x1, y1, x2, y2, color=None, lw=.65, **kwargs):
    ax.plot([x1, x2], [y1, y2], color=color or C["line"], lw=lw,
            solid_capstyle="round", **kwargs)


def box(x, y, w, h, fill="white", edge=None, radius=1.2, lw=.6):
    patch = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=fill, edgecolor=edge or C["line"], lw=lw)
    ax.add_patch(patch)
    return patch


def arrow(x1, y1, x2, y2, color=None, lw=.8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                mutation_scale=4, linewidth=lw,
                                color=color or C["muted"],
                                shrinkA=0, shrinkB=0))


def vector(x, y, w, h, color, n=7, phase=0):
    for k in range(n):
        strength = .22 + .68 * ((k * 7 + phase * 3) % 13) / 12
        ax.add_patch(Rectangle((x + k * w / n, y), w / n - .15, h,
                               facecolor=tint(color, strength), edgecolor="none"))


def document(x, y, color, label):
    box(x + .7, y - .7, 28.4, 12, fill=tint(color, .05),
        edge=tint(color, .35), radius=.6)
    box(x, y, 28.4, 12, edge=tint(color, .45), radius=.6)
    line(x + 1.8, y + 1.6, x + 1.8, y + 10.4, color, 1.5)
    txt(x + 15.5, y + 5.7, label, size=8, color=color, weight="bold",
        within=(x + 3, y + .4, 25, 11.2))


# Stages read left to right; all drawing coordinates are millimetres.
columns = [(1, 30, "Pretreatment\nnotes"), (35, 31, "Encode each note"),
           (70, 37, "Pool + concatenate"), (111, 40, "Fit endpoint models"),
           (155, 24, "Predict risk")]
for x, width, title in columns:
    txt(x + width / 2, 5, title, size=8, weight="bold", within=(x, 1, width, 8))
for x1, x2 in [(31.6, 34.3), (66.6, 69.3), (107.6, 110.3), (151.6, 154.3)]:
    arrow(x1, 38.5, x2, 38.5)

# All three source types are available before the first treatment.
for y, color, label in zip([15, 32, 49], TYPES,
                           ["Clinician\nnotes", "Imaging\nreports", "Pathology\nreports"]):
    document(1.4, y, color, label)

# A fixed encoder produces one mean-pooled representation for each note.
box(35, 12, 31, 54, fill=C["pale"], edge="#B9C7D3")
txt(50.5, 20, "Clinical\nModernBERT", size=8.7, weight="bold", within=(36, 13, 29, 13))
txt(50.5, 28, "Frozen encoder", size=8, color=C["muted"], within=(36, 25, 29, 6))
for row in range(3):
    vector(39.8, 33 + 2.2 * row, 21.4, 1.6, C["clinical"], n=10, phase=row)
arrow(50.5, 40.1, 50.5, 42.2)
txt(50.5, 47, "Mean token\npooling", within=(36, 43, 29, 9))
vector(41.8, 54, 17.4, 2.4, C["clinical"], n=8, phase=4)
txt(50.5, 62, "768 features/note", size=7.8, color=C["muted"], within=(36, 58, 29, 7))

# Pool independently by note type, then concatenate the three vectors.
box(70, 12, 37, 54)
txt(88.5, 19, "Recency-weighted\nmean within type", size=8.2, weight="bold",
    within=(71, 13, 35, 13))
txt(88.5, 28, "λ = 0.01/day", size=8, color=C["muted"], within=(71, 25, 35, 6))
for row, color in enumerate(TYPES):
    y = 33.6 + row * 5.2
    for k in range(3):
        vector(72.7 + k * 6.2, y, 5.4, 2.6, color, n=4, phase=row + k)
    arrow(91.4, y + 1.3, 96, y + 1.3, color=color)
    vector(97.2, y, 7.2, 2.6, color, n=5, phase=row + 2)
txt(88.5, 52.5, "Concatenate", size=8, weight="bold", within=(71, 49, 35, 7))
for k, color in enumerate(TYPES):
    vector(74.5 + 9.4 * k, 56.5, 9.1, 2.8, color, n=5, phase=k + 1)
txt(88.5, 63, "2,304 text features", size=8, color=C["muted"], within=(71, 60, 35, 6))

# Shared clinical covariates accompany text in each endpoint-specific fit.
box(111, 12, 40, 54, fill=C["pale"], edge="#B9C7D3")
txt(131, 19, "Elastic-net Cox", size=8.5, weight="bold", within=(112, 14, 38, 9))
txt(131, 28, "One model per endpoint", size=8, within=(112, 24, 38, 8))
line(115, 35, 147, 35)
txt(131, 40, "Clinical covariates", size=8, weight="bold", within=(112, 37, 38, 6))
txt(131, 52, "Age, sex, cancer type\nand 2 note-era covariates", size=8,
    within=(112, 44, 38, 16))

# Endpoint families are labels only: no schematic cells imply measured risk.
box(155, 12, 24, 54)
txt(167, 20, "Held-out\nrisk scores", size=8.2, weight="bold", within=(156, 13, 22, 14))
line(158, 29, 176, 29)
txt(167, 36, "Death /\nmetastasis", size=8, within=(156, 30, 22, 12))
txt(167, 49, "ICD-10\n3 / 4 character", size=7.8, within=(156, 43, 22, 12))
txt(167, 61, "PhecodeX", size=8, within=(156, 57, 22, 8))

# A compact timing strip keeps baseline eligibility visible at native size.
box(1, 69, 178, 8, fill=C["pale"], edge=C["pale"], radius=.8)
txt(90, 73, "Baseline: notes before first treatment  •  All three note types required", size=8,
    color=C["muted"], within=(2, 69, 176, 8))

OUT.mkdir(parents=True, exist_ok=True)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
canvas_bounds = fig.bbox
for artist, within in texts:
    bounds = artist.get_window_extent(renderer)
    if not (canvas_bounds.contains(bounds.x0, bounds.y0) and
            canvas_bounds.contains(bounds.x1, bounds.y1)):
        raise RuntimeError(f"Text outside canvas: {artist.get_text()}")
    if within is not None:
        x, y, width, height = within
        (left, bottom), (right, top) = ax.transData.transform(
            [(x, y + height), (x + width, y)])
        if (bounds.x0 < left or bounds.x1 > right or
                bounds.y0 < bottom or bounds.y1 > top):
            raise RuntimeError(f"Text outside allocated bounds: {artist.get_text()}")
for ext in ("svg", "pdf"):
    fig.savefig(OUT / f"model_schematic.{ext}", facecolor="white")
fig.savefig(OUT / "model_schematic.png", dpi=600, facecolor="white")
fig.savefig(OUT / "model_schematic_preview.png", dpi=180, facecolor="white")
plt.close(fig)
print(f"Wrote SVG, PDF, 600-dpi PNG and preview to {OUT}")
