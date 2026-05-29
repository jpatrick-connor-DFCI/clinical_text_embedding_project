"""Triage biomarker hits for Figure 5 literature-verification story.

Reads compiled_results/all_findings_with_validation.csv (or wherever
BIOMARKER_COMPILED_DIR points) and produces a tiered, lit-verified shortlist
that joins:
  Track 1 (prognostic within ICI-treated patients) — marker main effect
  Track 2 (predictive of differential ICI benefit) — marker × ICI interaction

per (gene, cancer, cohort) so you can see the prognostic AND predictive
signal for the same gene side by side.

Writes (alongside the input):
  biomarker_shortlist.csv        — full tiered table (one row per gene×cancer×cohort)
  biomarker_review_template.md   — review sheet w/ PubMed + OncoKB + CIViC URLs
  biomarker_triage_summary.txt   — console summary of the headline candidates

Run:
    python -m manuscript_figures.biomarker_triage
or
    BIOMARKER_COMPILED_DIR=~/Desktop/compiled_results \\
      python jupyter_notebooks/manuscript_figures/biomarker_triage.py
"""

from __future__ import annotations

import os
import sys
import textwrap
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COMPILED_DIR = Path(
    os.environ.get("BIOMARKER_COMPILED_DIR",
                   Path.home() / "Desktop" / "compiled_results")
).expanduser()
SRC = COMPILED_DIR / "all_findings_with_validation.csv"

# Tier-A inclusion thresholds (purely data-driven, before any lit lookup)
N_MIN_TIER_A    = 20   # max marker-positive n across tracks
SPECS_MIN_ROBUST = 2   # n_specs >= this to count a track as "robust"
EXTREME_HR      = 50.0 # |log HR| > log(EXTREME_HR) flagged as model separation

# PubMed query template — change to taste
ICI_TERMS = (
    '"immune checkpoint" OR "PD-1" OR "PD-L1" OR "anti-PD-1" OR '
    '"anti-PD-L1" OR pembrolizumab OR nivolumab OR atezolizumab OR '
    'ipilimumab OR durvalumab'
)

# Ordinal map for validation_level (used for sorting + narrative role)
VAL_ORDER = {
    "Very Strong": 6, "Strong": 5, "Moderate": 4, "Partial": 3,
    "Weak": 2, "Indirect": 1, "No Evidence": 0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _row_hr(row: pd.Series) -> float:
    return row["HR_markerxICI"] if row["track"] == 2 else row["HR_marker"]


def _row_p(row: pd.Series) -> float:
    return row["p_markerxICI"] if row["track"] == 2 else row["p_marker"]


def _row_n_pos(row: pd.Series) -> float:
    # Marker-positive count: Track 2 uses ICI arm by convention; Track 1
    # is ICI-only so n_marker_pos already IS the ICI-arm marker+ count.
    if row["track"] == 2:
        return row.get("n_ICI_pos") or row.get("n_marker_pos") or 0
    return row.get("n_marker_pos") or 0


def _direction(hrs: pd.Series) -> str:
    s = hrs.dropna()
    s = s[s > 0]
    if s.empty:
        return "none"
    if (s < 1).all():
        return "benefit"
    if (s > 1).all():
        return "harm"
    return "mixed"


def _pubmed_url(gene: str, cancer: str) -> str:
    cancer_clause = f' AND "{cancer}"[Title/Abstract]' if cancer != "pan_cancer" else ""
    q = f'"{gene}"[Title/Abstract] AND ({ICI_TERMS}){cancer_clause}'
    return "https://pubmed.ncbi.nlm.nih.gov/?term=" + urllib.parse.quote(q)


def _oncokb_url(gene: str) -> str:
    return f"https://www.oncokb.org/gene/{urllib.parse.quote(gene)}"


def _civic_url(gene: str) -> str:
    return "https://civicdb.org/search?query=" + urllib.parse.quote(gene)


def _fmt(x, spec=".2f") -> str:
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))):
            return "—"
        return format(x, spec)
    except (TypeError, ValueError):
        return str(x)


def _int_or(x, default: int = 0) -> int:
    """`NaN or 0` returns NaN in Python (NaN is truthy), so we need an explicit
    is-null check before falling back to the default."""
    if x is None:
        return default
    try:
        if isinstance(x, float) and np.isnan(x):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _str_or(x, default: str = "—") -> str:
    if x is None:
        return default
    try:
        if isinstance(x, float) and np.isnan(x):
            return default
    except (TypeError, ValueError):
        pass
    s = str(x)
    return s if s else default


# ---------------------------------------------------------------------------
# Tier + narrative-role assignment
# ---------------------------------------------------------------------------
def _assign_tier(row: pd.Series) -> str:
    t1_specs = _int_or(row.get("n_specs_T1"))
    t2_specs = _int_or(row.get("n_specs_T2"))
    n_pos = max(_int_or(row.get("max_n_pos_T1")), _int_or(row.get("max_n_pos_T2")))
    hrs = [v for v in (row.get("median_HR_T1"), row.get("median_HR_T2"))
           if pd.notna(v) and v > 0]
    extreme = any(v > EXTREME_HR or v < 1 / EXTREME_HR for v in hrs)
    if extreme:
        return "C_extreme_HR"
    if n_pos < 10:
        return "C_small_n"
    t1_robust = t1_specs >= SPECS_MIN_ROBUST
    t2_robust = t2_specs >= SPECS_MIN_ROBUST
    if t1_robust and t2_robust and n_pos >= N_MIN_TIER_A:
        return "A_both_tracks_robust"
    if (t1_robust or t2_robust) and n_pos >= N_MIN_TIER_A:
        return "B_single_track_robust"
    return "B_low_robustness"


def _narrative_role(row: pd.Series) -> str:
    tier = row["tier"]
    val = VAL_ORDER.get(row.get("validation_level") or "", 0)
    if tier.startswith("C"):
        return "Caveat — small n or extreme HR; mention with reservation"
    if tier == "A_both_tracks_robust":
        return ("Novel discovery — robust both-tracks signal, no prior lit"
                if val <= 2 else
                "Replication of known biology — robust both-tracks signal w/ prior support")
    if tier == "B_single_track_robust":
        return ("Hypothesis-generating — single-track robust, no prior lit"
                if val <= 2 else
                "Supports known biology — single-track robust w/ prior support")
    return "Lower-confidence — single-spec only or weak signal"


def _cross_track_consistency(row: pd.Series) -> str:
    d1 = row.get("direction_T1")
    d2 = row.get("direction_T2")
    if d1 in (None, "none", float("nan")) or pd.isna(d1):
        return "T2_only" if d2 not in (None, "none") and pd.notna(d2) else "neither"
    if d2 in (None, "none", float("nan")) or pd.isna(d2):
        return "T1_only"
    if d1 == d2 and d1 != "mixed":
        return f"both_{d1}"
    if d1 != d2 and "mixed" not in (d1, d2):
        return "opposite_directions"
    return "mixed_within_track"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not SRC.exists():
        sys.exit(f"Missing {SRC}; set BIOMARKER_COMPILED_DIR or pull data first")
    df = pd.read_csv(SRC)
    print(f"Loaded {len(df)} hit-rows from {SRC.name}")
    print(f"  unique genes: {df['gene'].nunique()}")
    print(f"  tracks:       {sorted(df['track'].unique())}")
    print(f"  cohorts:      {sorted(df['cohort'].unique())}\n")

    # Normalize HR/p/n across tracks
    df["eff_HR"] = df.apply(_row_hr, axis=1)
    df["eff_p"]  = df.apply(_row_p, axis=1)
    df["eff_n_pos"] = df.apply(_row_n_pos, axis=1)
    df["track_tag"] = df["track"].map({1: "T1", 2: "T2"})

    # Per-track aggregation: n_specs, median HR, min p, max n_marker+, direction
    per_track = (df.groupby(["gene", "cancer_type", "cohort", "track_tag"])
                   .agg(n_specs   =("marker", "size"),
                        median_HR =("eff_HR", "median"),
                        min_p     =("eff_p",  "min"),
                        max_n_pos =("eff_n_pos", "max"),
                        direction =("eff_HR", _direction),
                        validation_level=("validation_level", "first"),
                        validation_notes=("validation_notes", "first"),
                        mutation_type=("mutation_type", "first"))
                   .reset_index())

    # Pivot T1 and T2 side-by-side per (gene, cancer, cohort).
    # Carry validation columns via a separate first-non-null reduction to avoid
    # losing the value when one track is absent for a gene-cancer pair.
    keys = ["gene", "cancer_type", "cohort"]
    val_meta = (per_track.dropna(subset=["validation_level"])
                          .drop_duplicates(keys, keep="first")
                          [keys + ["validation_level", "validation_notes", "mutation_type"]])
    notes_meta = (per_track.dropna(subset=["validation_notes"])
                            .drop_duplicates(keys, keep="first")
                            [keys + ["validation_notes"]])

    metric_cols = ["n_specs", "median_HR", "min_p", "max_n_pos", "direction"]
    wide = per_track.pivot_table(index=keys, columns="track_tag",
                                  values=metric_cols, aggfunc="first")
    wide.columns = [f"{m}_{t}" for m, t in wide.columns]
    wide = wide.reset_index().merge(val_meta, on=keys, how="left")
    # Prefer non-null validation_notes from either track
    wide = (wide.drop(columns=["validation_notes"], errors="ignore")
                .merge(notes_meta, on=keys, how="left"))

    # Tier + narrative + cross-track consistency
    wide["tier"] = wide.apply(_assign_tier, axis=1)
    wide["cross_track_consistency"] = wide.apply(_cross_track_consistency, axis=1)
    wide["narrative_role"] = wide.apply(_narrative_role, axis=1)

    # Lookup URLs (one per gene; same for every gene-cancer-cohort row)
    wide["pubmed_url"] = wide.apply(lambda r: _pubmed_url(r["gene"], r["cancer_type"]), axis=1)
    wide["oncokb_url"] = wide["gene"].apply(_oncokb_url)
    wide["civic_url"]  = wide["gene"].apply(_civic_url)

    # Sort: tier A first, then by validation strength, then by larger of T1/T2 n_pos
    tier_rank = {"A_both_tracks_robust": 0, "B_single_track_robust": 1,
                 "B_low_robustness": 2, "C_small_n": 3, "C_extreme_HR": 4}
    wide["_tier_rank"] = wide["tier"].map(tier_rank)
    wide["_val_rank"] = -wide["validation_level"].map(VAL_ORDER).fillna(-1)
    wide["_n_rank"] = -wide[["max_n_pos_T1", "max_n_pos_T2"]].max(axis=1).fillna(0)
    wide = (wide.sort_values(["_tier_rank", "_val_rank", "_n_rank"])
                .drop(columns=["_tier_rank", "_val_rank", "_n_rank"]))

    # Reorder columns for readability
    front = (["gene", "cancer_type", "cohort", "tier", "narrative_role",
              "cross_track_consistency", "validation_level",
              "n_specs_T1", "median_HR_T1", "min_p_T1", "max_n_pos_T1", "direction_T1",
              "n_specs_T2", "median_HR_T2", "min_p_T2", "max_n_pos_T2", "direction_T2",
              "validation_notes", "mutation_type",
              "pubmed_url", "oncokb_url", "civic_url"])
    wide = wide[[c for c in front if c in wide.columns]]

    # ---- write artifacts ----
    out_csv = COMPILED_DIR / "biomarker_shortlist.csv"
    wide.to_csv(out_csv, index=False)

    _write_markdown_template(wide, COMPILED_DIR / "biomarker_review_template.md")
    _write_summary(wide, COMPILED_DIR / "biomarker_triage_summary.txt")

    # ---- console summary ----
    print(f"Wrote {out_csv.name}  ({len(wide)} rows)")
    print(f"Wrote biomarker_review_template.md  (paste into Notion / fill in / decide highlights)")
    print(f"Wrote biomarker_triage_summary.txt  (cat for the punch list)\n")
    print("Tier counts:")
    print(wide["tier"].value_counts().to_string())
    print()
    _print_tier_a(wide)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _print_tier_a(wide: pd.DataFrame) -> None:
    tier_a = wide[wide["tier"] == "A_both_tracks_robust"]
    if tier_a.empty:
        print("No Tier-A candidates — relax SPECS_MIN_ROBUST or inspect Tier B")
        return
    print(f"=== Tier A — both tracks robust, n+ >= {N_MIN_TIER_A} ({len(tier_a)}) ===")
    for _, r in tier_a.iterrows():
        print(
            f"  {r['gene']:8s}  {r['cancer_type']:18s} {r['cohort']:8s} | "
            f"T1: HR={_fmt(r.get('median_HR_T1'))} ({_int_or(r.get('n_specs_T1'))} specs, "
            f"{_str_or(r.get('direction_T1'))})  "
            f"T2: HR={_fmt(r.get('median_HR_T2'))} ({_int_or(r.get('n_specs_T2'))} specs, "
            f"{_str_or(r.get('direction_T2'))})  "
            f"| {_str_or(r.get('validation_level'))}  | {r['cross_track_consistency']}"
        )


def _write_markdown_template(wide: pd.DataFrame, path: Path) -> None:
    """Markdown review sheet — one section per tier."""
    sections = [
        ("A_both_tracks_robust", "Tier A — both tracks robust"),
        ("B_single_track_robust", "Tier B — single-track robust"),
        ("B_low_robustness", "Tier B — single-spec / lower robustness"),
        ("C_small_n", "Tier C — small n (caveat)"),
        ("C_extreme_HR", "Tier C — extreme HR (likely model separation)"),
    ]
    lines = [
        "# Figure 5 biomarker shortlist — literature review template",
        "",
        f"Generated from `{SRC.name}` ({len(wide)} gene × cancer × cohort rows).",
        "Pre-filled with pipeline `validation_level` and `validation_notes`.",
        "Mark `headline?` Y/N per row to feed into the figure narrative.",
        "",
        "**Workflow per row:** open PubMed link → skim 5 most-recent abstracts → ",
        "update `validation_notes` if the existing note misses something → flip `headline?`.",
        "",
    ]
    for tier, label in sections:
        sub = wide[wide["tier"] == tier]
        if sub.empty:
            continue
        lines += [
            f"## {label} ({len(sub)} rows)",
            "",
            ("| Gene | Cancer | Cohort | T1 HR (n_specs, dir) | T2 HR (n_specs, dir) | "
             "max n+ | Validation | Cross-track | Notes | PubMed | OncoKB | CIViC | headline? |"),
            ("|------|--------|--------|---------------------|---------------------|"
             "--------|------------|-------------|-------|--------|--------|-------|-----------|"),
        ]
        for _, r in sub.iterrows():
            n_pos = max(_int_or(r.get("max_n_pos_T1")), _int_or(r.get("max_n_pos_T2")))
            t1 = (f"{_fmt(r.get('median_HR_T1'))} "
                  f"({_int_or(r.get('n_specs_T1'))}, {_str_or(r.get('direction_T1'))})")
            t2 = (f"{_fmt(r.get('median_HR_T2'))} "
                  f"({_int_or(r.get('n_specs_T2'))}, {_str_or(r.get('direction_T2'))})")
            notes = _str_or(r.get("validation_notes"), default="")
            notes = notes.replace("|", "/").replace("\n", " ")
            if len(notes) > 160:
                notes = notes[:157] + "…"
            lines.append(
                f"| {r['gene']} | {r['cancer_type']} | {r['cohort']} | {t1} | {t2} | "
                f"{n_pos} | {_str_or(r.get('validation_level'))} | "
                f"{r['cross_track_consistency']} | {notes} | "
                f"[link]({r['pubmed_url']}) | [link]({r['oncokb_url']}) | "
                f"[link]({r['civic_url']}) | TBD |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def _write_summary(wide: pd.DataFrame, path: Path) -> None:
    """Short punch list of headline-eligible candidates for fig 5."""
    lines = [
        "Biomarker triage summary — Figure 5 headline candidates",
        "=" * 60,
        "",
        f"Source: {SRC}",
        f"Rows:   {len(wide)} unique (gene, cancer, cohort)",
        "",
        "Tier counts:",
    ]
    for tier, n in wide["tier"].value_counts().items():
        lines.append(f"  {tier:30s} {n}")
    lines += ["", "Cross-track consistency counts:", ""]
    for cat, n in wide["cross_track_consistency"].value_counts().items():
        lines.append(f"  {cat:30s} {n}")

    for tier in ("A_both_tracks_robust", "B_single_track_robust"):
        sub = wide[wide["tier"] == tier]
        if sub.empty:
            continue
        lines += ["", "-" * 60, f"{tier}  ({len(sub)} candidates)", "-" * 60]
        for _, r in sub.iterrows():
            n_max = max(_int_or(r.get('max_n_pos_T1')), _int_or(r.get('max_n_pos_T2')))
            note  = _str_or(r.get('validation_notes'))[:200]
            lines.append(
                textwrap.dedent(f"""
                  {r['gene']}  in  {r['cancer_type']}  ({r['cohort']})
                    T1 (prognostic in ICI):  HR={_fmt(r.get('median_HR_T1'))}  n_specs={_int_or(r.get('n_specs_T1'))}  dir={_str_or(r.get('direction_T1'))}
                    T2 (predictive of ICI):  HR={_fmt(r.get('median_HR_T2'))}  n_specs={_int_or(r.get('n_specs_T2'))}  dir={_str_or(r.get('direction_T2'))}
                    Max n+:        {n_max}
                    Validation:    {_str_or(r.get('validation_level'))}
                    Cross-track:   {r['cross_track_consistency']}
                    Narrative:     {r['narrative_role']}
                    PubMed:        {r['pubmed_url']}
                    Existing note: {note}
                """).rstrip()
            )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
