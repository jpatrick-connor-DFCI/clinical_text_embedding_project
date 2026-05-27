"""Compose manuscript target figures from current standalone panel PNGs.

This keeps the current pipeline contract intact:
data_generation/prep_figure_*.py writes CSV inputs, plot_figure_*.py renders
one PNG per panel, and this script only assembles those panel PNGs into the
older manuscript-style full-figure targets.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


THIS_DIR = Path(__file__).resolve().parent
# Make sibling _figure_utils importable when this script is run directly so
# we share its (env-var-overridable) FIGURE_OUT_DIR rather than defaulting to
# the in-tree `jupyter_notebooks/.../figures` directory.
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from _figure_utils import FIGURE_OUT_DIR  # noqa: E402

TARGET_OUT_DIR = FIGURE_OUT_DIR / "target_figures"


def _load_plotting() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    return mpimg, plt


@dataclass(frozen=True)
class PanelSlot:
    label: str
    panel: str
    rect: tuple[float, float, float, float]


@dataclass(frozen=True)
class FigureSpec:
    key: str
    stem: str
    size: tuple[float, float]
    panels: tuple[PanelSlot, ...]
    note: str | None = None


FIGURE_SPECS: dict[str, FigureSpec] = {
    "1": FigureSpec(
        key="1",
        stem="figure1_schematic",
        size=(16.0, 10.2),
        panels=(
            PanelSlot("A", "fig1a", (0.04, 0.55, 0.62, 0.38)),
            PanelSlot("B", "fig1b", (0.72, 0.55, 0.24, 0.38)),
            PanelSlot("C", "fig1c", (0.04, 0.08, 0.46, 0.38)),
            PanelSlot("D", "fig1d", (0.58, 0.08, 0.38, 0.38)),
        ),
    ),
    "2": FigureSpec(
        key="2",
        stem="figure2_text_results",
        size=(15.5, 12.8),
        panels=(
            PanelSlot("A", "fig2a", (0.05, 0.72, 0.40, 0.24)),
            PanelSlot("B", "fig2b", (0.56, 0.72, 0.38, 0.24)),
            PanelSlot("C", "fig2c", (0.08, 0.42, 0.84, 0.22)),
            PanelSlot("D", "fig2d", (0.05, 0.08, 0.90, 0.24)),
        ),
    ),
    "3": FigureSpec(
        key="3",
        stem="figure3_feature_comps",
        size=(15.0, 13.0),
        panels=(
            PanelSlot("A", "fig3a", (0.06, 0.56, 0.40, 0.36)),
            PanelSlot("B", "fig3b", (0.56, 0.56, 0.40, 0.36)),
            PanelSlot("C", "fig3c", (0.06, 0.08, 0.40, 0.36)),
            PanelSlot("D", "fig3d", (0.56, 0.08, 0.40, 0.36)),
        ),
    ),
    "4": FigureSpec(
        key="4",
        stem="figure4_trajectories",
        size=(14.0, 12.0),
        panels=(
            PanelSlot("A", "fig4a", (0.07, 0.54, 0.40, 0.38)),
            PanelSlot("B", "fig4b", (0.57, 0.54, 0.38, 0.38)),
            PanelSlot("C", "fig4c", (0.10, 0.07, 0.80, 0.34)),
        ),
    ),
    "5": FigureSpec(
        key="5",
        stem="figure5_biomarkers",
        size=(15.5, 13.0),
        panels=(
            PanelSlot("A", "fig5a", (0.05, 0.56, 0.41, 0.38)),
            PanelSlot("B", "fig5b", (0.55, 0.56, 0.40, 0.38)),
            PanelSlot("C", "fig5c", (0.05, 0.08, 0.90, 0.34)),
        ),
    ),
}


def _panel_path(panel: str) -> Path:
    return FIGURE_OUT_DIR / f"{panel}.png"


def _draw_missing(ax: Any, path: Path) -> None:
    ax.set_facecolor("#f7f7f7")
    ax.text(
        0.5,
        0.5,
        f"Missing panel\n{path.name}",
        ha="center",
        va="center",
        fontsize=10,
        color="#777777",
        transform=ax.transAxes,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#cccccc")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_panel(fig: Any, slot: PanelSlot, strict: bool, mpimg: Any) -> bool:
    path = _panel_path(slot.panel)
    ax = fig.add_axes(slot.rect)
    ax.set_axis_off()

    if not path.exists():
        if strict:
            raise FileNotFoundError(f"Missing panel PNG: {path}")
        _draw_missing(ax, path)
        found = False
    else:
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_axis_off()
        found = True

    if slot.label:
        ax.text(
            -0.035,
            1.035,
            slot.label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="right",
            color="#111111",
        )
    return found


def compose_figure(spec: FigureSpec, out_dir: Path, strict: bool = False) -> Path:
    mpimg, plt = _load_plotting()
    fig = plt.figure(figsize=spec.size)
    found_count = 0
    for slot in spec.panels:
        found_count += int(_draw_panel(fig, slot, strict=strict, mpimg=mpimg))

    if found_count == 0 and strict:
        raise FileNotFoundError(f"No panels found for figure {spec.key}")

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{spec.stem}.png"
    pdf_path = out_dir / f"{spec.stem}.pdf"
    fig.savefig(png_path, facecolor="white", bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[composed] {png_path}")
    print(f"[composed] {pdf_path}")
    return png_path


def _parse_figures(values: Iterable[str]) -> list[str]:
    keys: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip().lower().removeprefix("figure").removeprefix("fig")
            if not part:
                continue
            if part not in FIGURE_SPECS:
                choices = ", ".join(FIGURE_SPECS)
                raise argparse.ArgumentTypeError(f"Unknown figure {part!r}; choose from {choices}")
            keys.append(part)
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose old-style manuscript target figures from current panel PNGs."
    )
    parser.add_argument(
        "--figures",
        nargs="*",
        default=list(FIGURE_SPECS),
        help="Figure numbers to compose, e.g. --figures 1 2 4 or --figures 1,2,4.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TARGET_OUT_DIR,
        help=f"Composite output directory (default: {TARGET_OUT_DIR}).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected panel PNG is missing.",
    )
    args = parser.parse_args()

    _load_plotting()

    from _figure_utils import apply_style

    apply_style()
    figure_keys = _parse_figures(args.figures)
    for key in figure_keys:
        compose_figure(FIGURE_SPECS[key], args.output_dir, strict=args.strict)


if __name__ == "__main__":
    main()
