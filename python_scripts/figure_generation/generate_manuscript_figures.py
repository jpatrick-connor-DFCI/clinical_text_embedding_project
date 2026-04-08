"""Run the full manuscript-figure generation workflow.

This script calls the standalone figure generators for figures 1-5 in sequence.
It is a thin orchestration layer so the full mockup set can be regenerated with

    python python_scripts/figure_generation/generate_manuscript_figures.py
"""

from __future__ import annotations

import traceback

import figure1_schematic
import figure2_text_results
import figure3_feature_comps
import figure4_trajectories
import figure5_biomarkers


FIGURE_STEPS = [
    ("Figure 1", figure1_schematic.main),
    ("Figure 2", figure2_text_results.main),
    ("Figure 3", figure3_feature_comps.main),
    ("Figure 4", figure4_trajectories.main),
    ("Figure 5", figure5_biomarkers.main),
]


def main() -> None:
    failures: list[tuple[str, str]] = []

    for label, fn in FIGURE_STEPS:
        print(f"\n{'=' * 72}")
        print(f"Running {label}")
        print(f"{'=' * 72}")
        try:
            fn()
        except Exception as exc:
            failures.append((label, str(exc)))
            print(f"[error] {label} failed: {exc}")
            traceback.print_exc()

    if failures:
        lines = "\n".join(f"- {label}: {msg}" for label, msg in failures)
        raise SystemExit(f"One or more figure-generation steps failed:\n{lines}")

    print("\nAll manuscript figure scripts completed successfully.")


if __name__ == "__main__":
    main()
