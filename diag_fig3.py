"""Diagnose why fig3_modality_ranks_long_* has only the death endpoint.

Read-only. Run from the repo root on the cluster:
    python diag_fig3.py
"""
import os
import time

import polars as pl

from config import FIGURE_DATA_DIR
from shared.palette import MODALITY_ORDER

print(f"MODALITY_ORDER: {MODALITY_ORDER}\n")

cindex_fp = os.path.join(FIGURE_DATA_DIR, "fig3_modality_cindex.csv")
m = pl.read_csv(cindex_fp)
print(f"fig3_modality_cindex.csv: {m.height} rows, "
      f"{m.select(['scheme','event']).unique().height} endpoints")

# --- 1. staleness: is the rank file older than its own input? ---
print("\n=== 1. file mtimes (is the rank file stale?) ===")
for name in ("fig3_modality_cindex.csv",
             "fig3_modality_ranks_long_cindex.csv",
             "fig3_modality_avg_rank_cindex.csv"):
    fp = os.path.join(FIGURE_DATA_DIR, name)
    if os.path.exists(fp):
        print(f"  {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(fp)))}  {name}")
    else:
        print(f"  {'MISSING':<16}  {name}")

# --- 2. coverage: which modalities does each endpoint actually have? ---
print("\n=== 2. modality coverage per endpoint ===")
cov = (m.group_by(["scheme", "event"])
        .agg(pl.col("modality").sort().alias("mods"))
        .with_columns(pl.col("mods").list.len().alias("n"))
        .sort(["n", "scheme", "event"]))
for r in cov.iter_rows(named=True):
    missing = [x for x in MODALITY_ORDER if x not in r["mods"]]
    print(f"  {r['scheme']:<14} {r['event']:<24} n={r['n']}  missing={missing}")

# --- 3. what the fixed complete-case rule would now keep ---
print("\n=== 3. what the CURRENT code would retain ===")
present = [x for x in MODALITY_ORDER if x in set(m["modality"])]
absent = [x for x in MODALITY_ORDER if x not in present]
print(f"  present anywhere : {present}")
print(f"  absent everywhere: {absent}  (dropped from the completeness rule)")
complete = cov.filter(
    pl.col("mods").list.set_intersection(present).list.len() == len(present))
print(f"  endpoints retained: {complete.height}/{cov.height}")
for r in complete.iter_rows(named=True):
    print(f"    KEEP {r['scheme']:<14} {r['event']}")

# blame: how many endpoints each present modality costs
print("\n  endpoints lost per modality (of the ones present somewhere):")
for mod in present:
    n_missing = cov.filter(~pl.col("mods").list.contains(mod)).height
    if n_missing:
        print(f"    {mod:<12} missing from {n_missing}/{cov.height} endpoints")

# --- 4. what actually landed in the rank file ---
print("\n=== 4. actual fig3_modality_ranks_long_cindex.csv ===")
rl_fp = os.path.join(FIGURE_DATA_DIR, "fig3_modality_ranks_long_cindex.csv")
if os.path.exists(rl_fp):
    rl = pl.read_csv(rl_fp)
    print(f"  {rl.height} rows; events: {sorted(set(rl['event'])) if rl.height else '(none)'}")
    print(f"  modalities: {sorted(set(rl['modality'])) if rl.height else '(none)'}")
    if rl.height and complete.height > rl.select(['scheme','event']).unique().height:
        print("\n  >>> STALE: the code would now retain more endpoints than this file has.")
        print("  >>> Re-run figure3 with FORCE -- the notebook skipped it (outputs present).")
else:
    print("  MISSING")
