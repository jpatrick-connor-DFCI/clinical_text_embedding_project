"""Render Figure 2e from saved, aggregate published-score C-index estimates.

This supports local manuscript exports when only the report table (rather than
the full preparation outputs/R environment) is available. Score selection uses
the same catalog as R/published_score_utils.R; no estimates are fitted here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
SCORES = json.loads((REPO_ROOT / "shared/figure2_scores.json").read_text())["scores"]
MODELS = {
    "published": "Published score",
    "text": "Text",
    "published+text": "Published + text",
}
SOURCES = {"calculated": "calculated", "regex": "notes (regex)"}
MIN_PATIENTS = 50


def select_estimates(records):
    """Keep primary, evaluable estimates in the requested catalog/source order."""
    selected = []
    keys = set()
    for record in records:
        row = dict(record)
        row.setdefault("source", "calculated")
        if row["score"] not in SCORES or row["source"] not in SOURCES:
            continue
        if row["model"] not in MODELS or row.get("status", "ok") != "ok":
            continue
        if row.get("anchor", "treatment") != "treatment":
            continue
        if float(row.get("lab_window_days", 30)) != 30 or row.get("stratum", "all") != "all":
            continue
        row["n_patients"] = int(row["n_patients"])
        row["n_events"] = int(row["n_events"])
        if row["n_patients"] < MIN_PATIENTS:
            continue
        row["cindex"] = float(row["cindex"])
        if not math.isfinite(row["cindex"]):
            continue
        for column in ("ci_lower", "ci_upper"):
            value = row.get(column, "")
            row[column] = float(value) if value not in (None, "", "NA") else math.nan
        key = (row["score"], row["source"], row["model"])
        if key in keys:
            raise ValueError(f"Duplicate score/source/model estimate: {key}")
        keys.add(key)
        selected.append(row)
    return sorted(selected, key=lambda row: (
        list(SCORES).index(row["score"]), list(SOURCES).index(row["source"]),
        list(MODELS).index(row["model"]),
    ))


def score_rows(estimates):
    return list(dict.fromkeys((row["score"], row["source"]) for row in estimates))


def unavailable_estimates(estimates):
    available = {(row["score"], row["source"], row["model"]) for row in estimates}
    return [(score, source, model) for score, source in score_rows(estimates)
            for model in MODELS if (score, source, model) not in available]


def render_panel(table, output_stem, width=10, dpi=300):
    with Path(table).open(newline="") as handle:
        estimates = select_estimates(csv.DictReader(handle))
    rows = score_rows(estimates)
    if not rows:
        raise ValueError("No eligible published-score estimates in the input table")

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ctep-mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = json.loads((REPO_ROOT / "shared/palette.json").read_text())["MODEL_COLORS"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    height = 1.6 + 0.45 * len(rows)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.32, right=0.975, bottom=0.19, top=0.975)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)
    ax.axvline(0.5, color="#B3B3B3", linestyle="--", linewidth=0.8)
    offsets = dict(zip(MODELS, (-0.22, 0, 0.22)))
    for row in estimates:
        y = rows.index((row["score"], row["source"])) + offsets[row["model"]]
        if math.isfinite(row["ci_lower"]) and math.isfinite(row["ci_upper"]):
            ax.hlines(y, row["ci_lower"], row["ci_upper"], color=colors[row["model"]],
                      linewidth=1.3)
        ax.plot(row["cindex"], y, "o", color=colors[row["model"]], markersize=4.4)
    labels = []
    for score, source in rows:
        row = next(row for row in estimates if (row["score"], row["source"]) == (score, source))
        labels.append(f"{SCORES[score]}, {SOURCES[source]}\n"
                      f"n = {row['n_patients']:,}, {row['n_events']:,} deaths")
    ax.set_yticks(range(len(rows)), labels, fontsize=10)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel("Overall survival C-index", fontsize=12)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], marker="o", markersize=4.4, linewidth=1.3,
                      color=colors[model], label=label) for model, label in MODELS.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.57, 0.048),
               ncol=3, frameon=False, fontsize=10, handlelength=1.5, columnspacing=2)
    missing = unavailable_estimates(estimates)
    if missing:
        note = "Unavailable estimates: " + "; ".join(
            f"{SCORES[score]} ({SOURCES[source]}), {MODELS[model]}"
            for score, source, model in missing) + "."
        fig.text(0.035, 0.019, note, fontsize=9, ha="left", va="bottom", color="#444444")
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".pdf", ".png"):
        output = output_stem.with_suffix(suffix)
        fig.savefig(output, dpi=dpi, facecolor="white")
        outputs.append(output)
    plt.close(fig)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    for output in render_panel(args.table, args.output_stem, dpi=args.dpi):
        print(output)


if __name__ == "__main__":
    main()
