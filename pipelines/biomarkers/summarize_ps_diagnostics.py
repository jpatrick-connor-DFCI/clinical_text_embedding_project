"""Read-only diagnostic summary: the two gates for Figure 5.

Two independent questions, each of which can sink the main-figure framing on
its own:

  1. Does text improve confounding control? (the AUC / ESS / balance table)
  2. Is the discovery framing powered at all? (`summarize_hit_support`, which
     re-scores already-compiled hits against the raised per-cell event floor)

Gate 2 needs no re-run and can answer even when the diagnostics tree is absent:
if most hits sit near the old floor of 5 events per marker x arm cell, raising
that floor to 20 empties the hit list, and the genomic screen belongs in a
permutation-calibrated supplement regardless of what the propensity diagnostics
say.

Gate 1 is the gate for the Figure 5 main panels. The biomarker arm's thesis-
advancing claim is not the gene screen -- it is that embeddings of clinical
notes capture confounding structure the structured covariates miss, which is
what licenses everything downstream. That claim is testable directly from the
diagnostics the screen already writes, without re-running anything.

Reads `IPTW_runs_{cohort}_{ps_model}/{cancer_type}_diagnostics.parquet` through
`read_diagnostic_section` and emits one tidy table per cohort x cancer type,
pivoted covariates_only vs covariates_plus_embeddings:

  PS_AUC, ESS_ATE_treated/control, N_treated/control, events_treated/control
  max|SMD| weighted and unweighted, and the count over the 0.1 threshold,
  broken out by covariate family (structured vs embedding)
  analyzable, and why not when it isn't

How to read the output (see `interpret()` below, which prints this verdict):

  SUCCESS    -- AUC rises materially, post-weighting balance improves, ESS stays
                usable. Text captures confounding the structured covariates
                miss. Build the Figure 5 main panels as designed.
  FAILURE    -- AUC essentially unchanged, or up while balance does not move.
                Embeddings add nothing here; the arm should not be a main
                figure.
  AMBIGUOUS  -- AUC up sharply but ESS collapses. A sharper propensity model
                means more extreme weights and a smaller effective sample:
                better balance bought with variance. This is why AUC and ESS
                are always reported as a pair -- a HIGHER PS AUC is not
                automatically better, since perfect treatment prediction means
                no overlap and no identifiable effect at all.

Writes nothing outside OUTPUT_DIR and runs in seconds. Notebook-ready: no
argparse.
"""

import os

import polars as pl

from config import BIOMARKER_PATH
from pipelines.biomarkers.run_IPTW_analysis import read_diagnostic_section

OUTPUT_DIR = os.path.join(BIOMARKER_PATH, 'compiled_results/')

COHORTS = ['cohort1', 'cohort2']
PS_MODELS = ['covariates_only', 'covariates_plus_embeddings']
REFERENCE_PS_MODEL = 'covariates_only'
COMPARISON_PS_MODEL = 'covariates_plus_embeddings'

# Thresholds for the verdict. AUC_GAIN_MIN is deliberately modest: the question
# is whether text adds *anything* beyond the structured covariates, not whether
# it predicts treatment well on its own.
AUC_GAIN_MIN = 0.02
# Below this, a balance improvement bought with variance is not worth taking.
ESS_FRACTION_MIN = 0.30
# An ESS drop this large outweighs an AUC gain -- the ambiguous/dangerous case.
ESS_RATIO_WARN = 0.75

COHORT_METRICS = [
    'PS_AUC', 'N_treated', 'N_control', 'events_treated', 'events_control',
    'ESS_ATE_treated', 'ESS_ATE_control',
]

# The second gate. Every hit records deaths in all four marker x arm cells, and
# the interaction is identified by the contrast across them, so the binding
# quantity is the SMALLEST of the four -- a hit whose scarcest cell holds three
# deaths is determined by those three regardless of how large the other cells
# are. The screen's floor is now 20 (MIN_EVENTS_PER_MARKER_GROUP); hits compiled
# under the old floor of 5 are re-scored against it here, without re-running.
EVENT_CELL_COLUMNS = [
    'events_ICI_pos', 'events_ICI_neg', 'events_nonICI_pos', 'events_nonICI_neg',
]
LEGACY_EVENT_FLOOR = 5


def _run_dir(cohort, ps_model):
    return os.path.join(BIOMARKER_PATH, f'IPTW_runs_{cohort}_{ps_model}/')


def discover_cancer_types():
    """Cancer types with at least one diagnostics file anywhere in the grid.

    Discovered from diagnostics rather than from results filenames: a spec that
    fails the analyzability gate writes diagnostics but no results, and those
    are exactly the specs this summary must not hide.
    """
    found = set()
    for cohort in COHORTS:
        for ps_model in PS_MODELS:
            run = _run_dir(cohort, ps_model)
            if not os.path.isdir(run):
                continue
            for fname in os.listdir(run):
                if fname.endswith('_diagnostics.parquet'):
                    found.add(fname[: -len('_diagnostics.parquet')])
    return sorted(found)


def _balance_summary(path):
    """max|SMD| and imbalance counts per covariate family, weighted and not.

    Files written before the balance diagnostic carried a covariate family have
    no `covariate_family` column; those are reported as a single 'structured'
    family, which is what they contained.
    """
    bal = read_diagnostic_section(path, 'balance_ATE')
    if bal.is_empty():
        return {}
    if 'covariate_family' not in bal.columns:
        bal = bal.with_columns(pl.lit('structured').alias('covariate_family'))

    out = {}
    for family in ('structured', 'embedding', 'all'):
        fam = bal if family == 'all' else bal.filter(pl.col('covariate_family') == family)
        if fam.is_empty():
            continue
        for src, tag in (('SMD_weighted', 'weighted'), ('SMD_unweighted', 'unweighted')):
            if src not in fam.columns:
                continue
            out[f'max_smd_{tag}_{family}'] = float(fam[src].abs().max())
            out[f'n_imbalanced_{tag}_{family}'] = float((fam[src].abs() > 0.1).sum())
        out[f'n_covariates_{family}'] = float(len(fam))
    return out


def collect():
    """One row per cohort x ps_model x cancer_type, long over metrics."""
    rows = []
    for cohort in COHORTS:
        for ps_model in PS_MODELS:
            run = _run_dir(cohort, ps_model)
            if not os.path.isdir(run):
                print(f"  Skipping {cohort}/{ps_model}: directory not found")
                continue
            for cancer_type in discover_cancer_types():
                path = os.path.join(run, f'{cancer_type}_diagnostics.parquet')
                if not os.path.isfile(path):
                    continue

                row = {'cohort': cohort, 'ps_model': ps_model,
                       'cancer_type': cancer_type}

                cohort_sec = read_diagnostic_section(path, 'cohort')
                if not cohort_sec.is_empty():
                    rec = cohort_sec.row(0, named=True)
                    for m in COHORT_METRICS:
                        if m in rec:
                            row[m] = float(rec[m])

                # Written only by runs that include the analyzability gate;
                # absent in older diagnostics, which is reported as unknown
                # rather than silently as "analyzable".
                gate = read_diagnostic_section(path, 'analyzability')
                if not gate.is_empty():
                    rec = gate.row(0, named=True)
                    for m in ('analyzable', 'max_smd_ate', 'n_imbalanced_ate',
                              'ESS_fraction'):
                        if m in rec:
                            row[m] = float(rec[m])

                row.update(_balance_summary(path))

                n_tot = row.get('N_treated', 0) + row.get('N_control', 0)
                ess_tot = row.get('ESS_ATE_treated', 0) + row.get('ESS_ATE_control', 0)
                if n_tot > 0:
                    row.setdefault('ESS_fraction', ess_tot / n_tot)
                row['results_written'] = float(
                    os.path.isfile(os.path.join(run, f'{cancer_type}_results.parquet')))
                rows.append(row)

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        ['cohort', 'cancer_type', 'ps_model'])


def pivot_by_ps_model(tidy):
    """Reshape to one row per cohort x cancer type, reference vs comparison.

    The comparison is the whole point, so it is made structural rather than
    left to whoever reads the table.
    """
    if tidy.is_empty():
        return tidy
    metric_cols = [c for c in tidy.columns
                   if c not in ('cohort', 'ps_model', 'cancer_type')]
    ref = tidy.filter(pl.col('ps_model') == REFERENCE_PS_MODEL).drop('ps_model')
    cmp_ = tidy.filter(pl.col('ps_model') == COMPARISON_PS_MODEL).drop('ps_model')
    ref = ref.rename({c: f'{c}__covars' for c in metric_cols if c in ref.columns})
    cmp_ = cmp_.rename({c: f'{c}__embed' for c in metric_cols if c in cmp_.columns})
    wide = ref.join(cmp_, on=['cohort', 'cancer_type'], how='full', coalesce=True)

    deltas = []
    if 'PS_AUC__covars' in wide.columns and 'PS_AUC__embed' in wide.columns:
        deltas.append((pl.col('PS_AUC__embed') - pl.col('PS_AUC__covars')).alias('delta_AUC'))
    if 'ESS_fraction__covars' in wide.columns and 'ESS_fraction__embed' in wide.columns:
        deltas.append(
            (pl.col('ESS_fraction__embed') / pl.col('ESS_fraction__covars')).alias('ESS_ratio'))
    for fam in ('structured', 'all'):
        a, b = f'max_smd_weighted_{fam}__covars', f'max_smd_weighted_{fam}__embed'
        if a in wide.columns and b in wide.columns:
            deltas.append((pl.col(b) - pl.col(a)).alias(f'delta_max_smd_{fam}'))
    return wide.with_columns(deltas) if deltas else wide


def interpret(wide):
    """Print the success/failure/ambiguous verdict, per cohort x cancer type.

    Returns the verdict frame. This does not decide anything on its own -- it
    states which of the three readings each stratum supports, so the figure
    decision is made against a written criterion rather than by eyeballing.
    """
    if wide.is_empty() or 'delta_AUC' not in wide.columns:
        print("No comparable specs found: need both ps_models for a cohort x cancer type.")
        return pl.DataFrame()

    verdicts = []
    for row in wide.iter_rows(named=True):
        d_auc = row.get('delta_AUC')
        ess_ratio = row.get('ESS_ratio')
        ess_embed = row.get('ESS_fraction__embed')
        d_smd = row.get('delta_max_smd_structured')

        if d_auc is None:
            verdict, why = 'INCOMPLETE', 'missing one of the two PS models'
        elif ess_embed is not None and ess_embed < ESS_FRACTION_MIN:
            verdict, why = 'AMBIGUOUS', (
                f'ESS fraction {ess_embed:.2f} below {ESS_FRACTION_MIN} with embeddings: '
                'balance bought with variance')
        elif ess_ratio is not None and ess_ratio < ESS_RATIO_WARN:
            verdict, why = 'AMBIGUOUS', (
                f'ESS falls to {ess_ratio:.2f}x of covariates-only; read AUC and ESS together')
        elif d_auc < AUC_GAIN_MIN:
            verdict, why = 'FAILURE', (
                f'AUC gain {d_auc:+.3f} below {AUC_GAIN_MIN}: embeddings add little')
        elif d_smd is not None and d_smd > 0:
            verdict, why = 'FAILURE', (
                f'AUC up {d_auc:+.3f} but max|SMD| worsens by {d_smd:+.3f}')
        else:
            verdict, why = 'SUCCESS', f'AUC {d_auc:+.3f} with balance held or improved'

        verdicts.append({'cohort': row['cohort'], 'cancer_type': row['cancer_type'],
                         'verdict': verdict, 'reason': why})

    vdf = pl.DataFrame(verdicts)
    print("\n" + "=" * 72)
    print("VERDICT -- does text improve confounding control?")
    print("=" * 72)
    for row in vdf.iter_rows(named=True):
        print(f"  {row['cohort']:<9} {row['cancer_type']:<12} {row['verdict']:<10} {row['reason']}")
    counts = vdf.group_by('verdict').len().sort('len', descending=True)
    print("\n  " + ", ".join(f"{r['verdict']}: {r['len']}" for r in counts.iter_rows(named=True)))
    print("\n  SUCCESS in the primary strata -> build the Figure 5 main panels.")
    print("  Otherwise -> the arm belongs in the supplement; do not lead with it.")
    return vdf


def summarize_hit_support():
    """Re-score already-compiled hits against the raised per-cell event floor.

    Answers the question the AUC/ESS table cannot: even if the weighting works,
    is the discovery framing powered? If most hits sit near the old floor of 5
    events, raising it to 20 empties the hit list, and the genomic screen belongs
    in a permutation-calibrated supplement rather than a discovery figure.

    Returns (per_hit, summary) or (None, None) when no compiled hits exist.
    """
    fp = os.path.join(OUTPUT_DIR, 'track2_all_significant_hits.csv')
    if not os.path.isfile(fp):
        print(f"No compiled hits at {fp}; skipping the hit-support gate.")
        return None, None

    try:
        hits = pl.read_csv(fp)
    except pl.exceptions.NoDataError:
        # A screen that found nothing writes a file with no rows at all -- not
        # even a header -- which is a legitimate result, not a failure.
        print("Compiled hits file has no rows; nothing to re-score.")
        return None, None
    if hits.is_empty():
        print("Compiled hits table is empty; nothing to re-score.")
        return None, None

    present = [c for c in EVENT_CELL_COLUMNS if c in hits.columns]
    if not present:
        print("Compiled hits carry no per-cell event counts "
              f"(expected any of {EVENT_CELL_COLUMNS}); skipping the hit-support gate.")
        return None, None
    if len(present) < len(EVENT_CELL_COLUMNS):
        missing = sorted(set(EVENT_CELL_COLUMNS) - set(present))
        print(f"  note: {missing} absent; min taken over {present} only, "
              "so the floor below is optimistic.")

    from pipelines.biomarkers.run_IPTW_analysis import MIN_EVENTS_PER_MARKER_GROUP

    per_hit = hits.with_columns(
        pl.min_horizontal([pl.col(c) for c in present]).alias('min_cell_events')
    ).with_columns(
        (pl.col('min_cell_events') >= MIN_EVENTS_PER_MARKER_GROUP).alias('survives_floor')
    )

    n = per_hit.height
    n_survive = int(per_hit['survives_floor'].sum())
    n_at_legacy = int((per_hit['min_cell_events'] <= LEGACY_EVENT_FLOOR).sum())
    q = per_hit['min_cell_events'].quantile

    print("\n" + "=" * 72)
    print("HIT SUPPORT -- is the discovery framing powered?")
    print("=" * 72)
    print(f"  {n} compiled hit(s); smallest of the four marker x arm event cells:")
    print(f"    min {int(per_hit['min_cell_events'].min())}, "
          f"p25 {q(0.25):.0f}, median {q(0.5):.0f}, p75 {q(0.75):.0f}, "
          f"max {int(per_hit['min_cell_events'].max())}")
    print(f"  {n_at_legacy}/{n} ({n_at_legacy / n:.0%}) sit at or below the old floor "
          f"of {LEGACY_EVENT_FLOOR} events")
    print(f"  {n_survive}/{n} ({n_survive / n:.0%}) clear the current floor "
          f"of {MIN_EVENTS_PER_MARKER_GROUP}")

    if n_survive == 0:
        print("\n  VERDICT: no compiled hit clears the raised floor. The discovery "
              "framing is not supported;\n           the genomic screen belongs in the "
              "permutation-calibrated supplement.")
    elif n_survive / n < 0.25:
        print(f"\n  VERDICT: only {n_survive / n:.0%} of hits survive. Report the screen "
              "against a permutation null;\n           do not lead with a curated gene list.")
    else:
        print(f"\n  VERDICT: {n_survive}/{n} hits are adequately supported. A discovery "
              "panel is defensible,\n           still reported against the permutation null.")

    group_cols = [c for c in ('cohort', 'ps_model', 'weight_type', 'cancer_type')
                  if c in per_hit.columns]
    summary = (per_hit.group_by(group_cols)
               .agg(pl.len().alias('n_hits'),
                    pl.col('survives_floor').sum().alias('n_survive_floor'),
                    pl.col('min_cell_events').min().alias('min_cell_events'),
                    pl.col('min_cell_events').median().alias('median_min_cell_events'))
               .sort(group_cols)) if group_cols else pl.DataFrame()
    return per_hit, summary


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tidy = collect()
    if tidy.is_empty():
        print(f"No diagnostics found under {BIOMARKER_PATH}. "
              "Run the IPTW screen first (stages 1-5).")
        # The hit-support gate reads the compiled hits, not the diagnostics, so
        # it can still answer even when the diagnostics tree is absent.
        summarize_hit_support()
        return

    wide = pivot_by_ps_model(tidy)
    vdf = interpret(wide)

    tidy.write_parquet(os.path.join(OUTPUT_DIR, 'ps_diagnostics_tidy.parquet'))
    wide.write_parquet(os.path.join(OUTPUT_DIR, 'ps_diagnostics_by_ps_model.parquet'))
    wide.write_csv(os.path.join(OUTPUT_DIR, 'ps_diagnostics_by_ps_model.csv'))
    if not vdf.is_empty():
        vdf.write_csv(os.path.join(OUTPUT_DIR, 'ps_diagnostics_verdict.csv'))

    not_analyzable = tidy.filter(pl.col('analyzable') == 0.0) \
        if 'analyzable' in tidy.columns else pl.DataFrame()
    if not not_analyzable.is_empty():
        print(f"\n{len(not_analyzable)} spec(s) failed the analyzability gate:")
        for row in not_analyzable.iter_rows(named=True):
            print(f"  {row['cohort']}/{row['ps_model']}/{row['cancer_type']}: "
                  f"max|SMD|={row.get('max_smd_ate', float('nan')):.3f}, "
                  f"ESS fraction={row.get('ESS_fraction', float('nan')):.3f}")

    per_hit, hit_summary = summarize_hit_support()
    if per_hit is not None:
        per_hit.write_csv(os.path.join(OUTPUT_DIR, 'hit_support_per_hit.csv'))
        if hit_summary is not None and not hit_summary.is_empty():
            hit_summary.write_csv(os.path.join(OUTPUT_DIR, 'hit_support_by_spec.csv'))

    print(f"\nOutputs saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
