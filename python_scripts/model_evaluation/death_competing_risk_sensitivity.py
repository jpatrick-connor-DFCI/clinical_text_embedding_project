"""Death-as-competing-risk sensitivity check.

For each ICD3-post and phecode-post endpoint, compares:
  - Standard Kaplan-Meier estimate of event probability (current pipeline:
    death treated as independent censoring)
  - Aalen-Johansen cumulative incidence function (CIF), treating death as a
    competing event

A large discrepancy between the two indicates that the Cox model results for
that endpoint may be biased upward (KM > CIF when death is positively
correlated with the event).

Outputs a CSV with per-endpoint discrepancy metrics to help prioritise which
endpoints warrant a formal competing-risk analysis.

Usage
-----
    python death_competing_risk_sensitivity.py [--schemes icd3_post phecode_post ...]
"""

import argparse
import os

import numpy as np
import pandas as pd

DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
INTAE_DATA_PATH = '/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/'

SCHEME_SURV_FILES = {
    'icd3_post':    'level_3_ICD_post_surv_df.csv',
    'icd4_post':    'level_4_ICD_post_surv_df.csv',
    'phecode_post': 'phecode_post_surv_df.csv',
}

LANDMARK_YEARS = [1, 3, 5]
LANDMARK_DAYS  = [y * 365 for y in LANDMARK_YEARS]


# ---------------------------------------------------------------------------
# Aalen-Johansen CIF (no external dependencies)
# ---------------------------------------------------------------------------

def compute_km_and_cif(
    time: np.ndarray,
    status: np.ndarray,   # 0=censored, 1=event of interest, 2=competing event
    eval_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (km_prob, cif) evaluated at eval_times.

    km_prob  : 1 - KM(t) treating all non-events as censored.
    cif      : Aalen-Johansen CIF for cause 1, with cause 2 as competing risk.
    """
    order = np.argsort(time)
    t_ord = time[order]
    s_ord = status[order]

    unique_times = np.unique(t_ord)
    n_total = len(t_ord)

    km_surv = 1.0        # S(t) for KM
    aj_surv = 1.0        # S(t) overall for AJ (both causes as events)
    cif_val = 0.0        # cumulative incidence for cause 1

    km_at_t = {}
    cif_at_t = {}

    at_risk = n_total
    i = 0
    for ut in unique_times:
        # collect events at this time
        d1 = 0   # cause-1 events
        d2 = 0   # cause-2 events
        while i < len(t_ord) and t_ord[i] == ut:
            if s_ord[i] == 1:
                d1 += 1
            elif s_ord[i] == 2:
                d2 += 1
            i += 1

        if at_risk > 0:
            # KM update (only cause-1 as event)
            km_surv *= (1.0 - d1 / at_risk)
            # AJ CIF update: CIF_1 += S(t-) * d1/n
            cif_val += aj_surv * d1 / at_risk
            # Overall survival update for AJ
            aj_surv *= (1.0 - (d1 + d2) / at_risk)

        km_at_t[ut]  = km_surv
        cif_at_t[ut] = cif_val

        # censored at this time don't reduce at_risk until after processing
        n_at_time = np.sum(t_ord == ut)
        at_risk -= n_at_time

    # Evaluate at requested times using step-function interpolation
    km_times  = np.array(sorted(km_at_t.keys()))
    km_vals   = np.array([km_at_t[t] for t in km_times])
    cif_times = km_times
    cif_vals  = np.array([cif_at_t[t] for t in cif_times])

    def step_eval(times_grid, vals_grid, query):
        out = np.zeros(len(query))
        for qi, q in enumerate(query):
            idx = np.searchsorted(times_grid, q, side='right') - 1
            if idx < 0:
                out[qi] = 0.0
            else:
                out[qi] = vals_grid[idx]
        return out

    km_prob  = 1.0 - step_eval(km_times,  km_vals,  eval_times)
    cif_prob = step_eval(cif_times, cif_vals, eval_times)
    return km_prob, cif_prob


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sensitivity(schemes: list[str]) -> pd.DataFrame:
    # Load death data (ground-truth competing event)
    death_df = pd.read_csv(
        os.path.join(INTAE_DATA_PATH, 'follow_up_vte_df_cohort.csv'),
        usecols=['DFCI_MRN', 'tt_death', 'death'],
    )

    rows = []
    for scheme in schemes:
        surv_file = os.path.join(SURV_PATH, SCHEME_SURV_FILES[scheme])
        if not os.path.exists(surv_file):
            print(f"[skip] {surv_file} not found")
            continue

        surv_df = pd.read_csv(surv_file)
        surv_df = surv_df.merge(death_df, on='DFCI_MRN', how='left')

        events = [col for col in surv_df.columns
                  if not col.startswith('tt_') and col not in
                  ('DFCI_MRN', 'first_treatment_date', 'last_contact_date',
                   'AGE_AT_TREATMENTSTART', 'GENDER', 'death', 'tt_death')]

        print(f"\n{scheme}: {len(events)} endpoints")

        for event in events:
            tt_col = f'tt_{event}'
            if tt_col not in surv_df.columns:
                continue

            sub = surv_df[[tt_col, event, 'tt_death', 'death']].dropna()
            if len(sub) < 50:
                continue

            time    = sub[tt_col].to_numpy(dtype=float)
            ev      = sub[event].to_numpy(dtype=int)
            t_death = sub['tt_death'].to_numpy(dtype=float)
            died    = sub['death'].to_numpy(dtype=int)

            # Construct competing-risk status:
            # 1 = event of interest occurred first (or same day)
            # 2 = death occurred first without the event
            # 0 = censored (neither)
            status = np.zeros(len(ev), dtype=int)
            for j in range(len(ev)):
                if ev[j] == 1:
                    status[j] = 1          # event occurred (time is already time-to-event)
                elif died[j] == 1 and t_death[j] < time[j]:
                    # patient died before the event would have occurred;
                    # in current pipeline they are censored at death time
                    status[j] = 2
                    time[j]   = t_death[j]
                # else: censored as-is

            n_events     = int((status == 1).sum())
            n_competing  = int((status == 2).sum())
            eval_days    = np.array([d for d in LANDMARK_DAYS if d <= time.max()])
            if len(eval_days) == 0:
                continue

            km_prob, cif_prob = compute_km_and_cif(time, status, eval_days)

            for di, days in enumerate(eval_days):
                rows.append({
                    'scheme':        scheme,
                    'event':         event,
                    'n':             len(sub),
                    'n_events':      n_events,
                    'n_competing':   n_competing,
                    'landmark_year': days / 365,
                    'km_prob':       round(km_prob[di],  4),
                    'cif_prob':      round(cif_prob[di], 4),
                    'discrepancy':   round(km_prob[di] - cif_prob[di], 4),
                })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Death-as-competing-risk sensitivity check.")
    parser.add_argument(
        '--schemes', nargs='+',
        default=list(SCHEME_SURV_FILES.keys()),
        choices=list(SCHEME_SURV_FILES.keys()),
    )
    parser.add_argument('--out', default=os.path.join(SURV_PATH, 'death_competing_risk_sensitivity.csv'))
    args = parser.parse_args()

    results = run_sensitivity(args.schemes)
    results.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")

    # Summary: flag endpoints with large discrepancy at any landmark
    flagged = results[results['discrepancy'].abs() > 0.05]
    if flagged.empty:
        print("No endpoints with |KM - CIF| > 5% at any landmark time.")
    else:
        summary = (
            flagged.groupby(['scheme', 'event'])['discrepancy']
            .max()
            .reset_index()
            .sort_values('discrepancy', ascending=False)
        )
        print(f"\nFlagged endpoints (|KM - CIF| > 5% at ≥1 landmark): {len(summary)}")
        print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
